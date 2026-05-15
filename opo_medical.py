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
parser.add_argument("--ucb", type=str, default="bootstrap", choices=["bootstrap", "bernstein", "tail_safe", "none"])
parser.add_argument("--ucb_alpha", type=float, default=0.1)
parser.add_argument("--out_dir", type=str, default="")
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
data = load_breast_cancer()
X = data.data.astype(np.float32)
y_raw = data.target.astype(np.int64)
y = 1 - y_raw

num_actions = 2
input_dim = X.shape[1]
hidden_dim = 64

print("Original target names:", data.target_names)
print("After relabeling:")
print("  1 = malignant")
print("  0 = benign")
print("Class counts:", {i: int((y == i).sum()) for i in range(num_actions)})

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)


# ============================================================
# 3. Replay buffer dataset
# ============================================================
class LoggedBanditDataset(Dataset):
    def __init__(self, logged_dict):
        self.x = torch.tensor(logged_dict["x"], dtype=torch.float32)
        self.a = torch.tensor(logged_dict["a"], dtype=torch.long)
        self.r = torch.tensor(logged_dict["r"], dtype=torch.float32)
        self.mu = torch.tensor(logged_dict["mu"], dtype=torch.float32)
        self.y = torch.tensor(logged_dict["y"], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.a[idx], self.r[idx], self.mu[idx], self.y[idx]


# ============================================================
# 4. Policy network
#
# pos_bias is NOT a learnable parameter.
# It is passed in externally each time.
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x, pos_bias: float = 0.0):
        logits = self.backbone(x)
        logits = logits.clone()
        logits[:, 1] = logits[:, 1] + pos_bias
        return logits

    def action_probs(self, x, softmax_temperature: float = 1.0, pos_bias: float = 0.0):
        logits = self.forward(x, pos_bias=pos_bias) / softmax_temperature
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
    y_pool,
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
    y_batch = y_pool[chosen_indices]

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
    rewards = (actions == y_batch).astype(np.float32)

    keep_mask = np.ones(len(pool_indices), dtype=bool)
    keep_mask[chosen_positions] = False
    remaining_pool_indices = pool_indices[keep_mask]

    logged_batch = {
        "x": x_batch.astype(np.float32),
        "a": actions,
        "r": rewards,
        "mu": propensities,
        "y": y_batch.astype(np.int64),
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
def train_one_epoch(policy_model, loader, optimizer_obj, pos_bias = 0, clip_grad: float = 5.0):
    policy_model.train()
    total_loss = 0.0
    total_obj = 0.0
    total_samples = 0

    for x, a, r, mu, _ in loader:
        x = x.to(device)
        a = a.to(device)
        r = r.to(device)
        mu = mu.to(device)

        probs = policy_model.action_probs(x, pos_bias=pos_bias)
        chosen_prob = probs.gather(1, a.unsqueeze(1)).squeeze(1)
        ratio = chosen_prob / mu

        ips_obj = (r * ratio).mean()
        loss = -ips_obj

        optimizer_obj.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_model.parameters(), clip_grad)
        optimizer_obj.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_obj += ips_obj.item() * bs
        total_samples += bs

    return total_loss / total_samples, total_obj / total_samples


# ============================================================
# 8. Logged validation set for OPE
# ============================================================
def collect_random_logged_bandit_data(X_arr, y_arr, num_actions_: int = 2):
    n = len(X_arr)
    logged_actions = np.random.randint(low=0, high=num_actions_, size=n)
    rewards = (logged_actions == y_arr).astype(np.float32)
    propensities = np.ones(n, dtype=np.float32) / num_actions_

    return {
        "x": X_arr.astype(np.float32),
        "a": logged_actions.astype(np.int64),
        "r": rewards.astype(np.float32),
        "mu": propensities.astype(np.float32),
        "y": y_arr.astype(np.int64),
    }


val_logged = collect_random_logged_bandit_data(X_test, y_test, num_actions)
val_dataset = LoggedBanditDataset(val_logged)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


@torch.no_grad()
def evaluate_ope(
    policy_model,
    loader,
    softmax_temperature: float = 1.0,
    pos_bias: float = 0.0,
):
    policy_model.eval()

    all_r = []
    all_w = []

    for x, a, r, mu, _ in loader:
        x = x.to(device)
        a = a.to(device)
        r = r.to(device)
        mu = mu.to(device)

        probs = policy_model.action_probs(
            x,
            softmax_temperature=softmax_temperature,
            pos_bias=pos_bias,
        )
        chosen_prob = probs.gather(1, a.unsqueeze(1)).squeeze(1)
        w = chosen_prob / mu

        all_r.append(r.cpu().numpy())
        all_w.append(w.cpu().numpy())

    r = np.concatenate(all_r)
    w = np.concatenate(all_w)

    ips = np.mean(w * r)
    snips = np.sum(w * r) / (np.sum(w) + 1e-12)
    ess = (np.sum(w) ** 2) / (np.sum(w ** 2) + 1e-12)

    return {
        "IPS": float(ips),
        "SNIPS": float(snips),
        "ESS": float(ess),
    }


# ============================================================
# 9. Search pos_bias to satisfy FNR constraint
# ============================================================
@torch.no_grad()
def search_pos_bias_for_fnr(
    policy_model,
    X_arr,
    y_arr,
    pos_bias_grid,
    softmax_temperature: float = 1.0,
    use_sampled: bool = False,
    fnr_target: float = 0.04,
):
    policy_model.eval()

    X_t = torch.tensor(X_arr, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_arr, dtype=torch.long).to(device)
    positive_mask = (y_t == 1)

    chosen_bias = pos_bias_grid[-1]
    chosen_fnr = None
    chosen_acc = None
    chosen_tp = None
    chosen_fn = None

    for b in pos_bias_grid:
        probs = policy_model.action_probs(
            X_t,
            softmax_temperature=softmax_temperature,
            pos_bias=b,
        )

        if use_sampled:
            pred = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            pred = probs.argmax(dim=1)

        tp = ((pred == 1) & positive_mask).sum().item()
        fn = ((pred == 0) & positive_mask).sum().item()
        fnr = fn / (tp + fn + 1e-12)
        acc = (pred == y_t).float().mean().item()

        chosen_bias = b
        chosen_fnr = fnr
        chosen_acc = acc
        chosen_tp = tp
        chosen_fn = fn

        if fnr < fnr_target:
            break

    return {
        "pos_bias": float(chosen_bias),
        "FNR": float(chosen_fnr),
        "ACC": float(chosen_acc),
        "TP": int(chosen_tp),
        "FN": int(chosen_fn),
    }


# ============================================================
# 10. Additional metrics under chosen pos_bias
# ============================================================
@torch.no_grad()
def evaluate_policy_metrics(
    policy_model,
    X_arr,
    y_arr,
    softmax_temperature: float = 1.0,
    pos_bias: float = 0.0,
):
    policy_model.eval()

    X_t = torch.tensor(X_arr, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_arr, dtype=torch.long).to(device)
    probs = policy_model.action_probs(
        X_t,
        softmax_temperature=softmax_temperature,
        pos_bias=pos_bias,
    )

    sampled_pred = torch.multinomial(probs, num_samples=1).squeeze(1)
    greedy_pred = probs.argmax(dim=1)
    positive_mask = (y_t == 1)

    tp_sample = ((sampled_pred == 1) & positive_mask).sum().item()
    fn_sample = ((sampled_pred == 0) & positive_mask).sum().item()
    sampled_fnr = fn_sample / (tp_sample + fn_sample + 1e-12)

    tp_greedy = ((greedy_pred == 1) & positive_mask).sum().item()
    fn_greedy = ((greedy_pred == 0) & positive_mask).sum().item()
    greedy_fnr = fn_greedy / (tp_greedy + fn_greedy + 1e-12)

    sampled_acc = (sampled_pred == y_t).float().mean().item()
    greedy_acc = (greedy_pred == y_t).float().mean().item()
    expected_reward = probs.gather(1, y_t.unsqueeze(1)).mean().item()
    soft_fnr = probs[positive_mask, 0].mean().item()

    return {
        "soft_FNR": float(soft_fnr),
        "sampled_FNR": float(sampled_fnr),
        "greedy_FNR": float(greedy_fnr),
        "sampled_ACC": float(sampled_acc),
        "greedy_ACC": float(greedy_acc),
        "expected_reward": float(expected_reward),
        "sampled_TP": int(tp_sample),
        "sampled_FN": int(fn_sample),
        "greedy_TP": int(tp_greedy),
        "greedy_FN": int(fn_greedy),
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




def first_idx_all_future_true(risk_hat, fnr_threshold):
    """Return the first grid index such that this and all later risks are feasible."""
    risk_hat = np.asarray(risk_hat, dtype=np.float64)
    ok = risk_hat < fnr_threshold
    future_all_true = np.logical_and.accumulate(ok[::-1])[::-1]
    idx = np.where(future_all_true)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def bernstein_ucb_from_weight_matrix(
    weight_matrix: np.ndarray,
    risk: np.ndarray,
    alpha: float = 0.1,
    eps: float = 1e-12,
):
    """
    Simultaneous empirical-Bernstein-style UCBs for a grid of candidates.

    weight_matrix: [L, n], one row per candidate; rows are normalized internally.
    risk: [n], observed risk/loss values.
    """
    W = np.asarray(weight_matrix, dtype=np.float64).copy()
    c = np.asarray(risk, dtype=np.float64)
    if W.ndim != 2 or c.ndim != 1 or W.shape[1] != c.shape[0]:
        raise ValueError(f"Shape mismatch: weight_matrix={W.shape}, risk={c.shape}")

    L, n = W.shape
    W = W / (W.sum(axis=1, keepdims=True) + eps)
    risk_hat = W @ c

    centered = c[None, :] - risk_hat[:, None]
    var_hat = np.sum((W ** 2) * (centered ** 2), axis=1)

    c_max = float(np.max(c)) if c.size else 0.0
    B = np.max(W, axis=1) * max(c_max, 0.0)
    log_term = np.log(max(L, 1) / alpha)

    bonus = np.sqrt(2.0 * var_hat * log_term) + 3.0 * B * log_term
    risk_ucb = risk_hat + bonus
    return risk_hat, bonus, risk_ucb


def gaussian_multiplier_max_ucb_from_weight_matrix(
    weight_matrix: np.ndarray,
    risk: np.ndarray,
    n_boot: int,
    alpha: float,
    seed: int,
    eps: float = 1e-12,
):
    """
    Gaussian multiplier bootstrap for simultaneous max-UCB.

    The correction is the empirical (1-alpha)-quantile of
        max_l sum_i xi_i w_{l,i} (risk_i - risk_hat_l),
    and is added to every candidate's base risk estimate.
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(weight_matrix, dtype=np.float64).copy()
    c = np.asarray(risk, dtype=np.float64)
    if W.ndim != 2 or c.ndim != 1 or W.shape[1] != c.shape[0]:
        raise ValueError(f"Shape mismatch: weight_matrix={W.shape}, risk={c.shape}")

    W = W / (W.sum(axis=1, keepdims=True) + eps)
    L, n = W.shape
    base_risk = W @ c

    centered = W * (c[None, :] - base_risk[:, None])
    max_devs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        xi = rng.standard_normal(n)
        max_devs[b] = np.max(centered @ xi)

    correction = float(np.quantile(max_devs, 1.0 - alpha))
    correction = max(correction, 0.0)
    ucb_risk = base_risk + correction
    return base_risk, np.full(L, correction, dtype=np.float64), ucb_risk


@torch.no_grad()
def rebuild_action_and_prob_matrix_one_point(
    saved_models,
    saved_pos_biases,
    row_masses,
    buffer_x,
    buffer_a,
    buffer_y,
    input_dim_,
    pool_indices,
    collect_batch_size,
    logging_epsilon,
    logging_softmax_temperature,
    policy_model,
    X_pool,
    y_pool,
    num_actions_: int = 2,
    hidden_dim_: int = 64,
    fnr_target: float = 0.04,
    args=None,
):
    """
    Search external pos_bias using the same UCB options as opo_finance.py.

    For each candidate pos_bias:
      1. sample a provisional next point,
      2. build the policy-by-sample probability matrix,
      3. compute the Sinkhorn last-row marginal,
      4. restrict/renormalize weights to positive-label samples to estimate FNR,
      5. optionally add a simultaneous UCB correction across the pos_bias grid.
    """
    pos_bias_grid = np.linspace(0.0, 10.0, 162)

    if len(saved_models) == 0:
        return float(pos_bias_grid[-1]), 0.0, 0.0

    buffer_x_list = [row.copy() for row in buffer_x]
    buffer_a_list = [int(v) for v in buffer_a]
    buffer_y_list = [int(v) for v in buffer_y]

    # Add one provisional next-round sample. It will be overwritten for each candidate bias.
    dummy_batch, _ = collect_logged_batch_from_current_policy(
        policy_model=policy_model,
        X_pool=X_pool,
        y_pool=y_pool,
        pool_indices=pool_indices,
        batch_collect_size=collect_batch_size,
        epsilon=logging_epsilon,
        softmax_temperature=logging_softmax_temperature,
        pos_bias=0.0,
    )
    buffer_x_list.append(dummy_batch["x"][0].copy())
    buffer_a_list.append(int(dummy_batch["a"][0]))
    buffer_y_list.append(1)

    row_masses_local = list(row_masses) + [1.0]

    # FNR is conditional on true positive samples.
    pos_id_full = (np.asarray(buffer_y_list, dtype=np.int64) == 1)
    if not np.any(pos_id_full):
        return float(pos_bias_grid[-1]), 0.0, 0.0

    cost_full = (
        np.asarray(buffer_a_list, dtype=np.int64)[pos_id_full]
        != np.asarray(buffer_y_list, dtype=np.int64)[pos_id_full]
    ).astype(np.float64)

    weights_for_ucb = np.zeros((len(pos_bias_grid), int(pos_id_full.sum())), dtype=np.float64)
    risk_hat_perm = np.zeros(len(pos_bias_grid), dtype=np.float64)
    risk_hat_snips = np.zeros(len(pos_bias_grid), dtype=np.float64)

    for bias_idx, pos_bias in enumerate(pos_bias_grid):
        saved_pos_biases_local = deepcopy(list(saved_pos_biases))
        if len(saved_pos_biases_local) > 0:
            saved_pos_biases_local[-1] = pos_bias

        new_logged_batch, _ = collect_logged_batch_from_current_policy(
            policy_model=policy_model,
            X_pool=X_pool,
            y_pool=y_pool,
            pool_indices=pool_indices,
            batch_collect_size=collect_batch_size,
            epsilon=logging_epsilon,
            softmax_temperature=logging_softmax_temperature,
            pos_bias=pos_bias,
        )

        buffer_x_list[-1] = new_logged_batch["x"][0].copy()
        buffer_a_list[-1] = int(new_logged_batch["a"][0])
        buffer_y_list[-1] = 1

        # FNR is evaluated on positive-label samples; the provisional point is treated as positive
        # as in the original opo_medical implementation.
        pos_id = (np.asarray(buffer_y_list, dtype=np.int64) == 1)
        cost = (
            np.asarray(buffer_a_list, dtype=np.int64)[pos_id]
            != np.asarray(buffer_y_list, dtype=np.int64)[pos_id]
        ).astype(np.float64)

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

        # Permutation FNR: normalize only among positives.
        w_pos = weights[pos_id]
        w_pos = w_pos / (w_pos.sum() + 1e-12)

        # If the provisional point changes the number of positives, align to the current length.
        # In practice y can change across provisional draws; use current row only for estimate.
        if w_pos.shape[0] != weights_for_ucb.shape[1]:
            # Fall back to the fixed original positive set for the UCB matrix shape.
            pos_id_fixed = pos_id_full
            cost_fixed = (
                np.asarray(buffer_a_list, dtype=np.int64)[pos_id_fixed]
                != np.asarray(buffer_y_list, dtype=np.int64)[pos_id_fixed]
            ).astype(np.float64)
            w_fixed = weights[pos_id_fixed]
            w_fixed = w_fixed / (w_fixed.sum() + 1e-12)
            weights_for_ucb[bias_idx] = w_fixed
            risk_hat_perm[bias_idx] = float(np.sum(w_fixed * cost_fixed))
        else:
            weights_for_ucb[bias_idx] = w_pos
            risk_hat_perm[bias_idx] = float(np.sum(w_pos * cost))

        # SNIPS diagnostic for the same candidate.
        behavior_like = np.zeros(sinkhorn_input.shape[1], dtype=np.float64)
        h = 0
        for j, row_mass in enumerate(row_masses_local):
            m = int(row_mass)
            if m > 0:
                behavior_like[h:h + m] = sinkhorn_input[j, h:h + m]
                h += m
        snips_weights = sinkhorn_input[-1] / (behavior_like + 1e-12)
        snips_weights = snips_weights[pos_id]
        snips_weights = snips_weights / (snips_weights.sum() + 1e-12)
        risk_hat_snips[bias_idx] = float(np.sum(snips_weights * cost))

    ucb_method = "none" if args is None else getattr(args, "ucb", "none")
    alpha = 0.1 if args is None else getattr(args, "ucb_alpha", 0.1)

    if ucb_method == "bootstrap":
        _, _, risk_score = gaussian_multiplier_max_ucb_from_weight_matrix(
            weight_matrix=weights_for_ucb,
            risk=cost_full,
            n_boot=300,
            alpha=alpha,
            seed=np.random.randint(0, 10**9),
        )
    elif ucb_method == "bernstein":
        _, _, risk_score = bernstein_ucb_from_weight_matrix(
            weight_matrix=weights_for_ucb,
            risk=cost_full,
            alpha=alpha,
        )
    elif ucb_method == "tail_safe":
        risk_score = risk_hat_perm
        selected_id = first_idx_all_future_true(risk_score, fnr_target)
        if selected_id < 0:
            selected_id = len(pos_bias_grid) - 1
        return float(pos_bias_grid[selected_id]), float(risk_hat_perm[selected_id]), float(risk_hat_snips[selected_id])
    else:
        risk_score = risk_hat_perm

    feasible_idx = np.where(risk_score < fnr_target)[0]
    selected_id = int(feasible_idx[0]) if len(feasible_idx) > 0 else len(pos_bias_grid) - 1
    return (
        float(pos_bias_grid[selected_id]),
        float(risk_score[selected_id]),
        float(risk_hat_snips[selected_id]),
    )


# ============================================================
# 12. Sequential collect-then-optimize loop
# ============================================================
collect_batch_size = 20

rounds = 30
train_epochs_per_round = 40
train_batch_size = 64

logging_epsilon = 0.0
logging_softmax_temperature = 1.0
logging_pos_bias = 0.0

pos_bias_grid = np.linspace(0.0, 8.0, 81)
fnr_target = 0.04

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
            "mu": replay_buffer["mu"][train_id],
            "y": replay_buffer["y"][train_id],
        })

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(train_batch_size, len(train_dataset)),
            shuffle=True,
        )

        last_loss, last_obj = None, None
        for _ in range(train_epochs_per_round):
            last_loss, last_obj = train_one_epoch(policy, train_loader, optimizer, pos_bias=0)

        saved_models.append(copy.deepcopy(policy.state_dict()))
        saved_pos_biases.append(0.0)
    else:
        saved_models.append(copy.deepcopy(policy.state_dict()))
        saved_pos_biases.append(0.0)
        last_loss, last_obj = np.nan, np.nan

    # --------------------------------------------------------
    # B. Search pos_bias
    # --------------------------------------------------------
    if replay_buffer is not None and len(replay_buffer["x"]) > 0 and len(saved_models) > 0:
        chosen_pos_bias,fnr_estimated,snips_fnr_estimated = rebuild_action_and_prob_matrix_one_point(
            saved_models=saved_models,
            saved_pos_biases=saved_pos_biases,
            row_masses=row_masses,
            buffer_x=replay_buffer["x"][cal_id],
            buffer_a=replay_buffer["a"][cal_id],
            buffer_y=replay_buffer["y"][cal_id],
            input_dim_=input_dim,
            pool_indices=pool_indices,
            collect_batch_size=collect_batch_size,
            logging_epsilon=logging_epsilon,
            logging_softmax_temperature=logging_softmax_temperature,
            policy_model=policy,
            X_pool=X_train,
            y_pool=y_train,
            num_actions_=num_actions,
            hidden_dim_=hidden_dim,
            fnr_target=fnr_target,
            args=args,
        )
    else:
        chosen_pos_bias = 0.0
        fnr_estimated = 0.0
        snips_fnr_estimated = 0.0

    logging_pos_bias = chosen_pos_bias
    saved_pos_biases[-1] = chosen_pos_bias
    print(saved_pos_biases)
    # --------------------------------------------------------
    # C. Sample new data point(s)
    # --------------------------------------------------------
    new_logged_batch, pool_indices_after = collect_logged_batch_from_current_policy(
        policy_model=policy,
        X_pool=X_train,
        y_pool=y_train,
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
    batch_a = new_logged_batch["a"]
    batch_y = new_logged_batch["y"]

    positive_mask = (batch_y == 1)
    collect_num_pos = int(positive_mask.sum())
    collect_num_fn = int(((batch_a == 0) & positive_mask).sum())

    if collect_num_pos == 0:
        collect_fnr = np.nan
    else:
        collect_fnr = collect_num_fn / collect_num_pos

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
    ope = evaluate_ope(
        policy,
        val_loader,
        softmax_temperature=1.0,
        pos_bias=chosen_pos_bias,
    )


    metrics = evaluate_policy_metrics(
        policy,
        X_test,
        y_test,
        softmax_temperature=1.0,
        pos_bias=chosen_pos_bias,
    )

    if 'cal_id' in locals():
        metrics_buffer = evaluate_policy_metrics(
            policy,
            replay_buffer["x"][cal_id],
            replay_buffer["y"][cal_id],
            softmax_temperature=1.0,
            pos_bias=chosen_pos_bias,
        )
    else:
        metrics_buffer = metrics


    saved_metrics.append({
        "round": rd,
        "buffer_size": buffer_size,
        "last_train_loss": float(last_loss) if not np.isnan(last_loss) else np.nan,
        "last_train_IPS_obj": float(last_obj) if not np.isnan(last_obj) else np.nan,
        "chosen_pos_bias": float(chosen_pos_bias),
        "collect_num_pos": collect_num_pos,
        "collect_num_fn": collect_num_fn,
        "collect_FNR": float(collect_fnr) if not np.isnan(collect_fnr) else np.nan,
        "val_IPS": float(ope["IPS"]),
        "val_SNIPS": float(ope["SNIPS"]),
        "ESS": float(ope["ESS"]),
        "soft_FNR": float(metrics["soft_FNR"]),
        "sampled_FNR": float(metrics["sampled_FNR"]),
        "greedy_FNR": float(metrics["greedy_FNR"]),
        "sampled_ACC": float(metrics["sampled_ACC"]),
        "greedy_ACC": float(metrics["greedy_ACC"]),
        "expected_reward": float(metrics["expected_reward"]),
        "sampled_TP": int(metrics["sampled_TP"]),
        "sampled_FN": int(metrics["sampled_FN"]),
        "greedy_TP": int(metrics["greedy_TP"]),
        "greedy_FN": int(metrics["greedy_FN"]),
        "mse_fnr": (fnr_estimated - float(metrics_buffer["sampled_FNR"]))**2,
        "fnr_estimated": fnr_estimated,
        "fnr_estimated_conservative": fnr_estimated >= float(metrics_buffer["sampled_FNR"]),
        "fnr_estimate_above_sampled": max(fnr_estimated - metrics_buffer["sampled_FNR"], 0),
        "fnr_estimate_below_sampled": max(metrics_buffer["sampled_FNR"] - fnr_estimated, 0),
        "bias_fnr": abs(fnr_estimated - float(metrics_buffer["sampled_FNR"])),
        "signed_bias_fnr": fnr_estimated - float(metrics_buffer["sampled_FNR"]),
        "snips_fnr_estimated": snips_fnr_estimated
    })

    action_matrix_shape = None if action_matrix is None else action_matrix.shape
    action_prob_matrix_shape = None if action_prob_matrix is None else action_prob_matrix.shape

    collect_fnr_str = "nan" if np.isnan(collect_fnr) else f"{collect_fnr:.4f}"

    print(
        f"Round {rd:03d} | "
        f"buffer_size={buffer_size} | "
        f"chosen_pos_bias={chosen_pos_bias:.3f} | "
        f"collect_num_pos={collect_num_pos} | "
        f"collect_num_fn={collect_num_fn} | "
        f"collect_FNR={collect_fnr_str} | "
        f"val_IPS={ope['IPS']:.4f} | "
        f"val_SNIPS={ope['SNIPS']:.4f} | "
        f"ESS={ope['ESS']:.2f} | "
        f"sampled_FNR={metrics['sampled_FNR']:.4f} | "
        f"greedy_FNR={metrics['greedy_FNR']:.4f} | "
        f"sampled_ACC={metrics['sampled_ACC']:.4f} | "
        f"greedy_ACC={metrics['greedy_ACC']:.4f} | "
        f"action_matrix_shape={action_matrix_shape} | "
        f"action_prob_matrix_shape={action_prob_matrix_shape}"
    )


# ============================================================
# 13. Final evaluation
# ============================================================
final_bias_search = search_pos_bias_for_fnr(
    policy,
    X_test,
    y_test,
    pos_bias_grid=pos_bias_grid,
    softmax_temperature=1.0,
    use_sampled=False,
    fnr_target=fnr_target,
)
final_pos_bias = final_bias_search["pos_bias"]

final_ope = evaluate_ope(
    policy,
    val_loader,
    softmax_temperature=1.0,
    pos_bias=final_pos_bias,
)
final_metrics = evaluate_policy_metrics(
    policy,
    X_test,
    y_test,
    softmax_temperature=1.0,
    pos_bias=final_pos_bias,
)

print("\nFinal chosen pos_bias:")
print(final_bias_search)

print("\nFinal OPE:")
print(final_ope)

print("\nFinal policy metrics:")
print(final_metrics)

print("\nSaved objects summary:")
print(f"number of saved models          = {len(saved_models)}")
print(f"number of saved pos_biases      = {len(saved_pos_biases)}")
print(f"action_matrix shape             = {None if action_matrix is None else action_matrix.shape}")
print(f"action_prob_matrix shape        = {None if action_prob_matrix is None else action_prob_matrix.shape}")
print(f"collected data indices shape    = {None if replay_buffer is None else replay_buffer['idx'].shape}")

for metric_row in saved_metrics:
    print(metric_row)


# ============================================================
# 14. Save outputs to CSV
# ============================================================
df_rounds = pd.DataFrame(saved_metrics)

final_summary = {
    "seed": args.seed,
    "final_pos_bias": float(final_pos_bias),
    "final_search_FNR": float(final_bias_search["FNR"]),
    "final_search_ACC": float(final_bias_search["ACC"]),
    "final_search_TP": int(final_bias_search["TP"]),
    "final_search_FN": int(final_bias_search["FN"]),
    "final_IPS": float(final_ope["IPS"]),
    "final_SNIPS": float(final_ope["SNIPS"]),
    "final_ESS": float(final_ope["ESS"]),
    "final_soft_FNR": float(final_metrics["soft_FNR"]),
    "final_sampled_FNR": float(final_metrics["sampled_FNR"]),
    "final_greedy_FNR": float(final_metrics["greedy_FNR"]),
    "final_sampled_ACC": float(final_metrics["sampled_ACC"]),
    "final_greedy_ACC": float(final_metrics["greedy_ACC"]),
    "final_expected_reward": float(final_metrics["expected_reward"]),
    "final_sampled_TP": int(final_metrics["sampled_TP"]),
    "final_sampled_FN": int(final_metrics["sampled_FN"]),
    "final_greedy_TP": int(final_metrics["greedy_TP"]),
    "final_greedy_FN": int(final_metrics["greedy_FN"]),
    "num_saved_models": int(len(saved_models)),
    "num_saved_pos_biases": int(len(saved_pos_biases)),
    "final_buffer_size": int(0 if replay_buffer is None else len(replay_buffer["x"])),
}

df_final = pd.DataFrame([final_summary])

rounds_csv_path = os.path.join(args.out_dir, f"seed_{args.seed:04d}_all_rounds.csv")
final_csv_path = os.path.join(args.out_dir, f"seed_{args.seed:04d}_final.csv")

df_rounds.to_csv(rounds_csv_path, index=False)
df_final.to_csv(final_csv_path, index=False)

print(f"\nSaved round-by-round CSV to: {rounds_csv_path}")
print(f"Saved final summary CSV to: {final_csv_path}")