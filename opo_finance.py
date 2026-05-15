import os
import argparse
import pandas as pd

import copy
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from permutation_weight import sinkhorn_log_domain_with_row_masses


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# set_seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ucb", type=str, default='bootstrap')
parser.add_argument("--ucb_alpha", type=float, default=0.1)
parser.add_argument("--out_dir", type=str, default="/scratch/aliu74/yihong/conformal/risk_control_result/debug")
args = parser.parse_args()

set_seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Load breast cancer dataset
#
# sklearn original:
#   0 = malignant
#   1 = benign
#
# relabeled here:
#   1 = malignant
#   0 = benign
# ============================================================
X = pd.read_csv('/home/yihongguo/scratchaliu74/yihong/conformal/risk_data/context.csv').drop(columns=['Date'])
reward = pd.read_csv("/home/yihongguo/scratchaliu74/yihong/conformal/risk_data/reward_matrix.csv").drop(columns=['Date'])
risk = pd.read_csv("/home/yihongguo/scratchaliu74/yihong/conformal/risk_data/risk_matrix.csv").drop(columns=['Date'])
X = np.array(X)
reward = np.array(reward) * 1000
risk = np.array(risk)
action = np.arange(6)

# X = data.data.astype(np.float32)
# y_raw = data.target.astype(np.int64)
# y = 1 - y_raw

num_actions = 6
input_dim = X.shape[1]
hidden_dim = 64

# print("Original target names:", data.target_names)
# print("After relabeling:")
# print("  1 = malignant")
# print("  0 = benign")
# print("Class counts:", {i: int((y == i).sum()) for i in range(num_actions)})

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.3,
    random_state=42,
)

X_train, reward_train, risk_train = X[train_idx], reward[train_idx], risk[train_idx]
X_test, reward_test, risk_test = X[test_idx], reward[test_idx], risk[test_idx]

# ============================================================
# 3. Replay buffer dataset
# ============================================================
class LoggedBanditDataset(Dataset):
    def __init__(self, logged_dict):
        self.x = torch.tensor(logged_dict["x"], dtype=torch.float32)
        self.a = torch.tensor(logged_dict["a"], dtype=torch.long)
        self.r = torch.tensor(logged_dict["r"], dtype=torch.float32)
        self.c = torch.tensor(logged_dict["c"], dtype=torch.float32)
        self.mu = torch.tensor(logged_dict["mu"], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.a[idx], self.r[idx], self.c[idx], self.mu[idx]


# ============================================================
# 4. Policy network
#
# pos_bias is NOT a learnable parameter.
# It is passed in externally each time.
# ============================================================
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int = 6,
        hidden_dim: int = 64,
        action_emb_dim: int = 16,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim

        # Encode context x
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)

        # Reward head: input = [state_repr, action_emb]
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Risk head: input = [state_repr, action_emb]
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def predict_reward_risk(self, x: torch.Tensor):
        """
        x: [B, input_dim]

        Returns:
            reward: [B, num_actions]
            risk:   [B, num_actions]
        """
        batch_size = x.size(0)
        h = self.backbone(x)  # [B, hidden_dim]

        # action ids: [num_actions]
        action_ids = torch.arange(self.num_actions, device=x.device)
        action_emb = self.action_embedding(action_ids)  # [A, action_emb_dim]

        # expand to all (x, a) pairs
        h_expanded = h.unsqueeze(1).expand(batch_size, self.num_actions, h.size(-1))   # [B, A, H]
        a_expanded = action_emb.unsqueeze(0).expand(batch_size, self.num_actions, self.action_emb_dim)  # [B, A, E]

        joint = torch.cat([h_expanded, a_expanded], dim=-1)  # [B, A, H+E]

        reward = self.reward_head(joint).squeeze(-1)  # [B, A]
        risk = self.risk_head(joint).squeeze(-1)      # [B, A]

        return reward, risk


    def forward(self, x: torch.Tensor, pos_bias: float = 0.0):
        reward, risk = self.predict_reward_risk(x)
        return reward - pos_bias * risk

    def action_probs(
        self,
        x: torch.Tensor,
        softmax_temperature: float = 1.0,
        pos_bias: float = 0.0,
    ):
        reward, risk = self.predict_reward_risk(x)
        logits = (reward - pos_bias * risk) / softmax_temperature
        return torch.softmax(logits, dim=-1)

    
policy = PolicyNet(
    input_dim=input_dim,
    num_actions=num_actions,
    hidden_dim=hidden_dim,
).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)


# ============================================================
# 5. Logging policy for data collection
#
# mu(a|x) = (1-eps) * pi(a|x; pos_bias) + eps / K
# ============================================================
@torch.no_grad()
def get_logging_probs(
    policy_model,
    x_batch,
    epsilon: float = 0.1,
    softmax_temperature: float = 1.0,
    pos_bias: float = 0.0,
):
    policy_model.eval()
    x_t = torch.tensor(x_batch, dtype=torch.float32).to(device)
    pi = policy_model.action_probs(
        x_t,
        softmax_temperature=softmax_temperature,
        pos_bias=pos_bias,
    ).cpu().numpy()
    behavior_probs = (1.0 - epsilon) * pi + epsilon / pi.shape[1]
    return behavior_probs.astype(np.float32)


def collect_logged_batch_from_current_policy(
    policy_model,
    X_pool,
    reward_pool,
    risk_pool,
    pool_indices,
    batch_collect_size: int = 1,
    epsilon: float = 0.1,
    softmax_temperature: float = 1.0,
    pos_bias: float = 0.0,
):
    assert len(pool_indices) > 0, "No more unused training samples left in the pool."

    actual_size = min(batch_collect_size, len(pool_indices))
    chosen_positions = np.random.choice(len(pool_indices), size=actual_size, replace=False)
    chosen_indices = pool_indices[chosen_positions]

    x_batch = X_pool[chosen_indices]
    reward_batch = reward_pool[chosen_indices]
    risk_batch = risk_pool[chosen_indices]

    behavior_probs = get_logging_probs(
        policy_model,
        x_batch,
        epsilon=epsilon,
        softmax_temperature=softmax_temperature,
        pos_bias=pos_bias,
    )

    actions = np.array(
        [np.random.choice(num_actions, p=behavior_probs[i]) for i in range(actual_size)],
        dtype=np.int64,
    )

    propensities = behavior_probs[np.arange(actual_size), actions].astype(np.float32)
    rewards = reward_batch[np.arange(len(reward_batch)), actions]
    risks = risk_batch[np.arange(len(reward_batch)), actions]
    
    keep_mask = np.ones(len(pool_indices), dtype=bool)
    keep_mask[chosen_positions] = False
    remaining_pool_indices = pool_indices[keep_mask]

    logged_batch = {
        "x": x_batch.astype(np.float32),
        "a": actions,
        "r": rewards,
        "c": risks,
        "mu": propensities,
        "idx": chosen_indices.astype(np.int64),
    }
    return logged_batch, remaining_pool_indices


# ============================================================
# 6. Helper to append new logged data to replay buffer
# ============================================================
def append_logged_data(buffer_dict, new_batch_dict):
    if buffer_dict is None:
        return {k: v.copy() for k, v in new_batch_dict.items()}

    out = {}
    for k in buffer_dict.keys():
        out[k] = np.concatenate([buffer_dict[k], new_batch_dict[k]], axis=0)
    return out


# ============================================================
# 7. Offline policy gradient training on replay buffer
#
# Pure IPS objective only.
# No soft_FNR penalty.
# Training uses pos_bias = 0.0.
# ============================================================
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

@torch.no_grad()
def evaluate_one_epoch(policy_model, loader, device):
    policy_model.eval()

    total_loss = 0.0
    total_reward_loss = 0.0
    total_risk_loss = 0.0
    total_samples = 0

    mse = nn.MSELoss()

    for x, a, r, c, mu in loader:
        x = x.to(device)
        a = a.to(device)
        r = r.to(device)
        c = c.to(device)

        pred_reward, pred_risk = policy_model.predict_reward_risk(x)

        pred_reward_a = pred_reward.gather(1, a.unsqueeze(1)).squeeze(1)
        pred_risk_a = pred_risk.gather(1, a.unsqueeze(1)).squeeze(1)

        reward_loss = mse(pred_reward_a, r)
        risk_loss = mse(pred_risk_a, c)
        loss = reward_loss + risk_loss

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_reward_loss += reward_loss.item() * bs
        total_risk_loss += risk_loss.item() * bs
        total_samples += bs

    return {
        "val_total_loss": total_loss / total_samples,
        "val_reward_mse": total_reward_loss / total_samples,
        "val_risk_mse": total_risk_loss / total_samples,
    }


def train_one_epoch(
    policy_model,
    dataset,
    optimizer_obj,
    device,
    train_batch_size: int = 64,
    train_epochs: int = 40,
    val_ratio: float = 0.3,
    early_stopping_patience: int = 5,
    clip_grad: float = 5.0,
    seed: int = 42,
):
    """
    Train outcome model with an internal random train/validation split.

    Args:
        policy_model: PolicyNet
        dataset: LoggedBanditDataset
        optimizer_obj: torch optimizer
        device: torch.device
        train_batch_size: batch size
        train_epochs: max training epochs
        val_ratio: fraction of dataset used for validation
        early_stopping_patience: stop if val loss does not improve for this many epochs
        clip_grad: gradient clipping
        seed: random seed for split

    Returns:
        dict with train/val metrics and number of epochs trained
    """
    n = len(dataset)
    if n < 2:
        return {
            "train_total_loss": np.nan,
            "train_reward_mse": np.nan,
            "train_risk_mse": np.nan,
            "val_total_loss": np.nan,
            "val_reward_mse": np.nan,
            "val_risk_mse": np.nan,
            "best_epoch": 0,
            "epochs_trained": 0,
        }

    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    n_val = max(1, int(val_ratio * n))
    n_train = n - n_val
    if n_train <= 0:
        n_train = n - 1
        n_val = 1

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=min(train_batch_size, len(train_subset)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=min(train_batch_size, len(val_subset)),
        shuffle=False,
    )

    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0

    last_train_stats = {
        "train_total_loss": np.nan,
        "train_reward_mse": np.nan,
        "train_risk_mse": np.nan,
    }
    best_val_stats = {
        "val_total_loss": np.nan,
        "val_reward_mse": np.nan,
        "val_risk_mse": np.nan,
    }

    epochs_trained = 0

    for epoch in range(train_epochs):
        policy_model.train()

        total_loss = 0.0
        total_reward_loss = 0.0
        total_risk_loss = 0.0
        total_samples = 0

        for x, a, r, c, mu in train_loader:
            x = x.to(device)
            a = a.to(device)
            r = r.to(device)
            c = c.to(device)

            pred_reward, pred_risk = policy_model.predict_reward_risk(x)

            pred_reward_a = pred_reward.gather(1, a.unsqueeze(1)).squeeze(1)
            pred_risk_a = pred_risk.gather(1, a.unsqueeze(1)).squeeze(1)

            reward_loss = mse(pred_reward_a, r)
            risk_loss = mse(pred_risk_a, c)
            loss = reward_loss + risk_loss

            optimizer_obj.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_model.parameters(), clip_grad)
            optimizer_obj.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_reward_loss += reward_loss.item() * bs
            total_risk_loss += risk_loss.item() * bs
            total_samples += bs

        last_train_stats = {
            "train_total_loss": total_loss / total_samples,
            "train_reward_mse": total_reward_loss / total_samples,
            "train_risk_mse": total_risk_loss / total_samples,
        }

        current_val_stats = evaluate_one_epoch(policy_model, val_loader, device)
        epochs_trained += 1

        if current_val_stats["val_total_loss"] < best_val_loss:
            best_val_loss = current_val_stats["val_total_loss"]
            best_state_dict = deepcopy(policy_model.state_dict())
            best_val_stats = current_val_stats.copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    if best_state_dict is not None:
        policy_model.load_state_dict(best_state_dict)

    return {
        **last_train_stats,
        **best_val_stats,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
    }
# ============================================================
# 8. Logged validation set for OPE
# ============================================================
# def collect_random_logged_bandit_data(X_arr, reward_arr, risk_arr, num_actions_: int = 6):
#     n = len(X_arr)
#     logged_actions = np.random.randint(low=0, high=num_actions_, size=n)
#     rewards = reward_arr[np.arange(len(logged_actions)),logged_actions]
#     risks = risk_arr[np.arange(len(logged_actions)),logged_actions]
#     propensities = np.ones(n, dtype=np.float32) / num_actions_

#     return {
#         "x": X_arr.astype(np.float32),
#         "a": logged_actions.astype(np.int64),
#         "r": rewards.astype(np.float32),
#         "c": risks,
#         "mu": propensities.astype(np.float32),
#     }





# ============================================================
# 9. Search pos_bias to satisfy FNR constraint
# ============================================================


# ============================================================
# 10. Additional metrics under chosen pos_bias
# ============================================================
@torch.no_grad()
@torch.no_grad()
def evaluate_policy_metrics(
    policy_model,
    X_arr,
    reward_arr,
    risk_arr,
    softmax_temperature: float = 1.0,
    pos_bias: float = 0.0,
):
    policy_model.eval()

    X_t = torch.tensor(X_arr, dtype=torch.float32).to(device)
    probs = policy_model.action_probs(
        X_t,
        softmax_temperature=softmax_temperature,
        pos_bias=pos_bias,
    )  # [N, A]

    sampled_action = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()
    greedy_action = probs.argmax(dim=1).cpu().numpy()

    sampled_reward = reward_arr[np.arange(len(reward_arr)), sampled_action]
    sampled_risk = risk_arr[np.arange(len(risk_arr)), sampled_action]

    greedy_reward = reward_arr[np.arange(len(reward_arr)), greedy_action]
    greedy_risk = risk_arr[np.arange(len(risk_arr)), greedy_action]

    probs_np = probs.cpu().numpy()
    expected_reward = np.sum(probs_np * reward_arr, axis=1)
    expected_risk = np.sum(probs_np * risk_arr, axis=1)

    return {
        "test_sampled_reward_mean": float(np.mean(sampled_reward)),
        "test_sampled_risk_mean": float(np.mean(sampled_risk)),
        "test_greedy_reward_mean": float(np.mean(greedy_reward)),
        "test_greedy_risk_mean": float(np.mean(greedy_risk)),
        "test_expected_reward": float(np.mean(expected_reward)),
        "test_expected_risk": float(np.mean(expected_risk)),
    }

# ============================================================
# 11. Matrix helpers
# ============================================================
@torch.no_grad()
def compute_actions_and_probs_for_model_state(
    model_state,
    buffer_x,
    pos_bias,
    input_dim_,
    num_actions_: int = 2,
    hidden_dim_: int = 64,
    sampled_actions=None,
):
    model = PolicyNet(
        input_dim=input_dim_,
        num_actions=num_actions_,
        hidden_dim=hidden_dim_,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    x_t = torch.tensor(buffer_x, dtype=torch.float32).to(device)
    probs = model.action_probs(x_t, pos_bias=pos_bias)

    if sampled_actions is None:
        sampled_actions = torch.multinomial(probs, num_samples=1).squeeze(1)

    sampled_action_probs = probs.gather(1, sampled_actions.unsqueeze(1)).squeeze(1)

    return (
        sampled_actions.cpu().numpy().astype(np.int64),
        sampled_action_probs.cpu().numpy().astype(np.float32),
    )




@torch.no_grad()
def rebuild_action_and_prob_matrix_one_point(
    saved_models,
    saved_pos_biases,
    row_masses,
    buffer_x,
    buffer_a,
    buffer_reward,
    buffer_risk,
    input_dim_,
    pool_indices,
    collect_batch_size,
    logging_epsilon,
    logging_softmax_temperature,
    policy_model,
    X_pool,
    reward_pool,
    risk_pool,
    num_actions_: int = 2,
    hidden_dim_: int = 64,
    fnr_target: float = 0.04,
    args=None,
):
    """
    Search external pos_bias by:
      1. sampling one new point,
      2. building policy-by-sample probability matrix,
      3. running Sinkhorn,
      4. checking weighted FNR.
    """
    pos_bias_grid = np.linspace(0.0, 40.0, 20)

    if len(saved_models) == 0:
        return pos_bias_grid[-1]

    buffer_x_list = [row.copy() for row in buffer_x]
    buffer_a_list = [int(v) for v in buffer_a]
    buffer_reward_list = [float(v) for v in buffer_reward]
    buffer_risk_list = [float(v) for v in buffer_risk]


    dummy_batch, _ = collect_logged_batch_from_current_policy(
        policy_model=policy_model,
        X_pool=X_pool,
        reward_pool = reward_pool,
        risk_pool = risk_pool,
        pool_indices=pool_indices,
        batch_collect_size=collect_batch_size,
        epsilon=logging_epsilon,
        softmax_temperature=logging_softmax_temperature,
        pos_bias=0.0,
    )
    buffer_x_list.append(dummy_batch["x"][0].copy())
    buffer_a_list.append(int(dummy_batch["a"][0]))
    buffer_reward_list.append(0)
    buffer_risk_list.append(max(buffer_risk_list))

    chosen_bias = pos_bias_grid[-1]
    row_masses_local = list(row_masses) + [1.0]

    weights_for_bootstrap = np.zeros((len(pos_bias_grid), len(buffer_x_list)))

    for bias_idx, pos_bias in enumerate(pos_bias_grid):
        saved_pos_biases_local = deepcopy(list(saved_pos_biases))
        if len(saved_pos_biases_local) > 0:
            saved_pos_biases_local[-1] = pos_bias

        new_logged_batch, _ = collect_logged_batch_from_current_policy(
            policy_model=policy_model,
            X_pool=X_pool,
            reward_pool = reward_pool,
            risk_pool = risk_pool,
            pool_indices=pool_indices,
            batch_collect_size=collect_batch_size,
            epsilon=logging_epsilon,
            softmax_temperature=logging_softmax_temperature,
            pos_bias=pos_bias,
        )

        buffer_x_list[-1] = new_logged_batch["x"][0].copy()
        buffer_a_list[-1] = int(new_logged_batch["a"][0])
        
        prob_cols = []
        for model_state, b in zip(saved_models, saved_pos_biases_local):
            _, action_probs = compute_actions_and_probs_for_model_state(
                model_state=model_state,
                buffer_x=np.asarray(buffer_x_list, dtype=np.float32),
                pos_bias=b,
                input_dim_=input_dim_,
                num_actions_=num_actions_,
                hidden_dim_=hidden_dim_,
                sampled_actions=torch.tensor(buffer_a_list, dtype=torch.long).to(device),
            )
            prob_cols.append(action_probs.reshape(-1, 1))

        action_prob_matrix = np.concatenate(prob_cols, axis=1)   # [N_data, N_policies]
        sinkhorn_input = action_prob_matrix.transpose(1, 0)      # [N_policies, N_data]

        col_masses = np.ones(sinkhorn_input.shape[1], dtype=np.float64)
        weights = sinkhorn_log_domain_with_row_masses(
            A=sinkhorn_input,
            row_masses=np.asarray(row_masses_local, dtype=np.float64),
            col_masses=col_masses,
            eps=1e-6,
            max_iter=1000,
            target_row=-1,
        )

        weights_for_bootstrap[bias_idx] = weights

        
        weights = weights / (weights.sum() + 1e-12)
        assert np.array(buffer_risk_list).shape == weights.shape
        fnr = np.sum(np.array(buffer_risk_list) * weights)
        
    if args.ucb == 'bootstrap':
        _, _, risk_ucb = gaussian_multiplier_max_ucb_from_weight_matrix(
            weight_matrix=weights_for_bootstrap,
            risk=np.array(buffer_risk_list),
            n_boot=300,
            alpha=args.ucb_alpha,
            seed=np.random.randint(0, 10**9),
        )
        feasible_idx = risk_ucb<fnr_target
        if sum(feasible_idx) == 0:
            return pos_bias_grid[-1], risk_ucb[-1]
        else:
            return pos_bias_grid[feasible_idx][0], risk_ucb[feasible_idx][0]
    elif args.ucb == 'tail_safe':
        weights_for_bootstrap = weights_for_bootstrap / (
            np.sum(weights_for_bootstrap, axis=1, keepdims=True) + 1e-12
        )        
        risk_hat = weights_for_bootstrap @ np.array(buffer_risk_list)
        pos_grid_selected_id = first_idx_all_future_true(risk_hat, fnr_target)
        return pos_bias_grid[pos_grid_selected_id], risk_hat[pos_grid_selected_id]

    else:
        _, _, risk_ucb = bernstein_ucb_from_weight_matrix(
            weight_matrix=weights_for_bootstrap,
            risk=np.array(buffer_risk_list),
            alpha=args.ucb_alpha,
        )
    
        feasible_idx = risk_ucb<fnr_target
        if sum(feasible_idx) == 0:
            return pos_bias_grid[-1], risk_ucb[-1]
        else:
            return pos_bias_grid[feasible_idx][0], risk_ucb[feasible_idx][0]

import numpy as np

def first_idx_all_future_true(risk_hat, fnr_threshold):
    risk_hat = np.asarray(risk_hat)
    ok = (risk_hat < fnr_threshold)   # bool array

    # suffix AND: future (including itself) are all True
    future_all_true = np.logical_and.accumulate(ok[::-1])[::-1]

    idx = np.where(future_all_true)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def bernstein_ucb_from_weight_matrix(
    weight_matrix,
    risk,
    alpha=0.1,
    eps=1e-12,
):
    """
    Compute simultaneous empirical-Bernstein-style UCBs from a precomputed weight matrix.

    Args:
        weight_matrix: np.ndarray, shape [L, n]
            Row l contains the sample weights for candidate l.
            They do not need to already sum to 1; we normalize row-wise.
        risk: np.ndarray, shape [n]
            Observed risk values.
        alpha: float
            Failure probability level. Uses simultaneous correction over rows.
        eps: float
            Numerical stability constant.

    Returns:
        risk_hat: np.ndarray, shape [L]
        bonus:    np.ndarray, shape [L]
        risk_ucb: np.ndarray, shape [L]
    """
    W = np.asarray(weight_matrix, dtype=np.float64).copy()
    c = np.asarray(risk, dtype=np.float64)

    assert W.ndim == 2
    assert c.ndim == 1
    L, n = W.shape
    assert c.shape[0] == n

    # Normalize each row to sum to 1
    W = W / (W.sum(axis=1, keepdims=True) + eps)

    # Point estimate
    risk_hat = W @ c   # [L]

    # For normalized weighted average, a convenient variance proxy is:
    # Var_hat ≈ sum_i w_i^2 (c_i - hat)^2
    centered = c[None, :] - risk_hat[:, None]
    var_hat = np.sum((W ** 2) * (centered ** 2), axis=1)  # [L]

    # Need a bound on each weighted summand.
    # Since summand is w_i * c_i and often c_i in [0,1],
    # we can use max_i w_i * max(c).
    c_max = np.max(c)
    if c_max < 0:
        raise ValueError("Risk values should usually be nonnegative for this UCB.")
    B = np.max(W, axis=1) * c_max   # [L]

    # Simultaneous correction across L rows
    log_term = np.log(L / alpha)

    bonus = np.sqrt(2.0 * var_hat * log_term) + 3.0 * B * log_term
    risk_ucb = risk_hat + bonus

    return risk_hat, bonus, risk_ucb
def bootstrap_max_ucb_from_weight_matrix(
    weight_matrix: np.ndarray,
    risk: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
):
    """
    Simultaneous bootstrap-max UCB over lambda grid.

    weight_matrix: [L, n], each row already normalized to sum to 1
    risk: [n]
    Returns:
        base_risk: [L]
        correction: scalar added to all lambdas
        ucb_risk: [L]
    """
    rng = np.random.default_rng(seed)
    L, n = weight_matrix.shape

    base_risk = weight_matrix @ risk
    max_devs = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        w_boot = weight_matrix[:, idx].copy()
        w_boot = w_boot / (np.sum(w_boot, axis=1, keepdims=True) + 1e-12)
        r_boot = risk[idx]
        risk_boot = np.sum(w_boot * r_boot[None, :], axis=1)
        print(risk_boot - base_risk)
        max_devs[b] = np.max(risk_boot - base_risk)

    correction = np.quantile(max_devs, 1.0 - alpha)
    ucb_risk = base_risk + correction
    return base_risk, np.full(L, correction), ucb_risk


import numpy as np

def gaussian_multiplier_max_ucb_from_weight_matrix(
    weight_matrix: np.ndarray,
    risk: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
):
    """
    Gaussian multiplier bootstrap for simultaneous max-UCB.

    weight_matrix: [L, n], each row sums to 1
    risk: [n]

    Returns:
        base_risk: [L]
        correction: scalar
        ucb_risk: [L]
    """
    rng = np.random.default_rng(seed)
    L, n = weight_matrix.shape

    base_risk = weight_matrix @ risk  # [L]

    # centered per-sample contributions: [L, n]
    centered = weight_matrix * (risk[None, :] - base_risk[:, None])
    max_devs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        xi = rng.standard_normal(n)   # one Gaussian multiplier draw
        # print( (centered @ xi - base_risk).shape ,'hh')
        # print(base_risk[:, None] )
        g_boot = centered @ xi     # [L], same as sum_i xi_i * centered[l, i]
        # print(g_boot)
        max_devs[b] = np.max(g_boot)

    correction = np.quantile(max_devs, 1.0 - alpha)
    ucb_risk = base_risk + correction
    return base_risk, np.full(L, correction), ucb_risk
# ============================================================
# 12. Sequential collect-then-optimize loop
# ============================================================
collect_batch_size = 100

rounds = 60
train_epochs_per_round = 10
train_batch_size = 64

logging_epsilon = 0.0
logging_softmax_temperature = 1.0
logging_pos_bias = 0.0

pos_bias_grid = np.linspace(0.0, 8.0, 81)
fnr_target = 0.1

pool_indices = np.arange(len(X_train))
replay_buffer = None

saved_models = []
saved_pos_biases = []
saved_metrics = []

action_matrix = None
action_prob_matrix = None
row_masses = []
logging_pos_bias = 0
for rd in range(1, rounds + 1):
    if len(pool_indices) == 0:
        print("Training pool exhausted.")
        break

    # --------------------------------------------------------
    # A. Train on current replay buffer
    # --------------------------------------------------------

    
    if replay_buffer is not None and len(replay_buffer["x"]) > 0:
        train_id = np.arange(0,len(replay_buffer["x"]),2)
        cal_id = np.arange(1,len(replay_buffer["x"]),2)
        
        train_dataset = LoggedBanditDataset({
            "x": replay_buffer["x"][train_id],
            "a": replay_buffer["a"][train_id],
            "r": replay_buffer["r"][train_id],
            "c": replay_buffer["c"][train_id],
            "mu": replay_buffer["mu"][train_id],
        })

        last_train_stats = train_one_epoch(
            policy_model=policy,
            dataset=train_dataset,
            optimizer_obj=optimizer,
            device=device,
            train_batch_size=train_batch_size,
            train_epochs=train_epochs_per_round,
            val_ratio=0.3,
            early_stopping_patience=5,
            clip_grad=5.0,
            seed=args.seed + rd,
        )
        
        saved_models.append(copy.deepcopy(policy.state_dict()))
        saved_pos_biases.append(0.0)
    else:
        saved_models.append(copy.deepcopy(policy.state_dict()))
        saved_pos_biases.append(0.0)
        last_train_stats = {
            "train_total_loss": np.nan,
            "train_reward_mse": np.nan,
            "train_risk_mse": np.nan,
        }

    # --------------------------------------------------------
    # B. Search pos_bias
    # --------------------------------------------------------
    if replay_buffer is not None and len(replay_buffer["x"]) > 0 and len(saved_models) > 0:
        chosen_pos_bias, fnr_estimated = rebuild_action_and_prob_matrix_one_point(
            saved_models=saved_models,
            saved_pos_biases=saved_pos_biases,
            row_masses=row_masses,
            buffer_x=replay_buffer["x"][cal_id],
            buffer_a=replay_buffer["a"][cal_id],
            buffer_reward=replay_buffer["r"][cal_id],
            buffer_risk=replay_buffer["c"][cal_id],
            input_dim_=input_dim,
            pool_indices=pool_indices,
            collect_batch_size=collect_batch_size,
            logging_epsilon=logging_epsilon,
            logging_softmax_temperature=logging_softmax_temperature,
            policy_model=policy,
            X_pool=X_train,
            reward_pool=reward_train,
            risk_pool=risk_train,
            num_actions_=num_actions,
            hidden_dim_=hidden_dim,
            fnr_target=fnr_target,
            args=args
        )
    else:
        chosen_pos_bias = 0.0
        fnr_estimated = 0.0

    logging_pos_bias = chosen_pos_bias
    saved_pos_biases[-1] = chosen_pos_bias
    print(saved_pos_biases)
    # --------------------------------------------------------
    # C. Sample new data point(s)
    # --------------------------------------------------------
    new_logged_batch, pool_indices_after = collect_logged_batch_from_current_policy(
        policy_model=policy,
        X_pool=X_train,
        reward_pool=reward_train,
        risk_pool=risk_train,
        pool_indices=pool_indices,
        batch_collect_size=collect_batch_size,
        epsilon=logging_epsilon,
        softmax_temperature=logging_softmax_temperature,
        pos_bias=logging_pos_bias,
    )

    # --------------------------------------------------------
    # C.1 Compute collected-batch FNR
    # If there is no positive sample in this batch, output np.nan
    # --------------------------------------------------------
    batch_reward = new_logged_batch["r"]
    batch_risk = new_logged_batch["c"]
    # print(batch_reward)
    # print(batch_risk)
    # print(d)
    collect_reward_mean = float(np.mean(batch_reward))
    collect_risk_mean = float(np.mean(batch_risk))
    # --------------------------------------------------------
    # D. Append new data into replay buffer
    # --------------------------------------------------------
    replay_buffer = append_logged_data(replay_buffer, new_logged_batch)
    row_masses.append(float(collect_batch_size/2))
    pool_indices = pool_indices_after
    buffer_size = len(replay_buffer["x"])

    # --------------------------------------------------------
    # F. Evaluate under chosen pos_bias
    # --------------------------------------------------------
    # ope = evaluate_ope(
    #     policy,
    #     val_loader,
    #     softmax_temperature=1.0,
    #     pos_bias=chosen_pos_bias,
    # )
    metrics = evaluate_policy_metrics(
        policy,
        X_test,
        reward_test,
        risk_test,
        softmax_temperature=1.0,
        pos_bias=chosen_pos_bias,
    )



    saved_metrics.append({
        "round": rd,
        "buffer_size": buffer_size,
        "lambda_value": float(chosen_pos_bias),

        "train_total_loss": float(last_train_stats["train_total_loss"]),
        "train_reward_mse": float(last_train_stats["train_reward_mse"]),
        "train_risk_mse": float(last_train_stats["train_risk_mse"]),

        "collect_reward_mean": collect_reward_mean,
        "collect_risk_mean": collect_risk_mean,

        "test_sampled_reward_mean": float(metrics["test_sampled_reward_mean"]),
        "test_sampled_risk_mean": float(metrics["test_sampled_risk_mean"]),
        "test_greedy_reward_mean": float(metrics["test_greedy_reward_mean"]),
        "test_greedy_risk_mean": float(metrics["test_greedy_risk_mean"]),
        "test_expected_reward": float(metrics["test_expected_reward"]),
        "test_expected_risk": float(metrics["test_expected_risk"]),
        "fnr_estimated": fnr_estimated,
        "fnr_estimated_bounded": fnr_estimated >= metrics["test_expected_risk"],
    })


    print(
        f"Round {rd:03d} | "
        f"buffer_size={buffer_size} | "
        f"lambda={chosen_pos_bias:.4f} | "
        f"train_reward_mse={last_train_stats['train_reward_mse']:.6f} | "
        f"train_risk_mse={last_train_stats['train_risk_mse']:.6f} | "
        f"collect_reward={collect_reward_mean:.4f} | "
        f"collect_risk={collect_risk_mean:.4f} | "
        f"test_sampled_reward={metrics['test_sampled_reward_mean']:.4f} | "
        f"test_sampled_risk={metrics['test_sampled_risk_mean']:.4f} | "
        f"test_greedy_reward={metrics['test_greedy_reward_mean']:.4f} | "
        f"test_greedy_risk={metrics['test_greedy_risk_mean']:.4f} | "
        f"test_expected_reward={metrics['test_expected_reward']:.4f} | "
        f"test_expected_risk={metrics['test_expected_risk']:.4f}"
    )


# ============================================================
# 13. Final evaluation
# ============================================================
# ============================================================
# 13. Final evaluation
# ============================================================
final_lambda = float(saved_pos_biases[-1]) if len(saved_pos_biases) > 0 else 0.0

final_metrics = evaluate_policy_metrics(
    policy,
    X_test,
    reward_test,
    risk_test,
    softmax_temperature=1.0,
    pos_bias=final_lambda,
)

print("\nFinal lambda:")
print(final_lambda)

print("\nFinal policy metrics:")
print(final_metrics)

print("\nSaved objects summary:")
print(f"number of saved models     = {len(saved_models)}")
print(f"number of saved lambdas    = {len(saved_pos_biases)}")
print(f"collected data indices     = {None if replay_buffer is None else replay_buffer['idx'].shape}")

for metric_row in saved_metrics:
    print(metric_row)


# ============================================================
# 14. Save outputs to CSV
# ============================================================
df_rounds = pd.DataFrame(saved_metrics)

final_summary = {
    "seed": args.seed,
    "final_lambda": final_lambda,
    "final_buffer_size": int(0 if replay_buffer is None else len(replay_buffer["x"])),

    "final_test_sampled_reward_mean": float(final_metrics["test_sampled_reward_mean"]),
    "final_test_sampled_risk_mean": float(final_metrics["test_sampled_risk_mean"]),
    "final_test_greedy_reward_mean": float(final_metrics["test_greedy_reward_mean"]),
    "final_test_greedy_risk_mean": float(final_metrics["test_greedy_risk_mean"]),
    "final_test_expected_reward": float(final_metrics["test_expected_reward"]),
    "final_test_expected_risk": float(final_metrics["test_expected_risk"]),
}
df_final = pd.DataFrame([final_summary])

rounds_csv_path = os.path.join(args.out_dir, f"seed_{args.seed:04d}_all_rounds.csv")
final_csv_path = os.path.join(args.out_dir, f"seed_{args.seed:04d}_final.csv")

df_rounds.to_csv(rounds_csv_path, index=False)
df_final.to_csv(final_csv_path, index=False)

print(f"\nSaved round-by-round CSV to: {rounds_csv_path}")
print(f"Saved final summary CSV to: {final_csv_path}")