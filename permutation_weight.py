import numpy as np


def logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    """Stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def sinkhorn_log_domain_with_row_masses(
    A: np.ndarray,
    row_masses: np.ndarray,
    col_masses: np.ndarray | None = None,
    target_row: int = -1,
    max_iter: int = 5000,
    tol: float = 1e-10,
    eps: float = 1e-300,
    return_full: bool = False,
):
    """
    Log-domain Sinkhorn for a nonnegative matrix A with prescribed row/column sums.

    Solves for
        Q = diag(u) A diag(v)
    such that
        Q 1 = row_masses
        Q^T 1 = col_masses

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Nonnegative matrix. Positive entries are best; zeros are clipped to eps.
        Rows can represent policies, columns can represent samples.
    row_masses : np.ndarray, shape (m,)
        Desired row sums, e.g. [k1, k2, ..., km].
    col_masses : np.ndarray, shape (n,), optional
        Desired column sums. If None, defaults to all ones.
    target_row : int
        Row index whose marginal weights you want returned.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on row/column residuals.
    eps : float
        Tiny floor for numerical stability.
    return_full : bool
        If True, also return Q, u, v, log_u, log_v.

    Returns
    -------
    p : np.ndarray, shape (n,)
        The designated target row of Q.
        If target_row corresponds to your evaluation policy row, this is the
        Sinkhorn marginal weight over columns.
    If return_full=True, additionally returns:
        Q, u, v, log_u, log_v
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")

    m, n = A.shape

    row_masses = np.asarray(row_masses, dtype=np.float64)
    if row_masses.shape != (m,):
        raise ValueError(f"row_masses must have shape ({m},).")

    if col_masses is None:
        col_masses = np.ones(n, dtype=np.float64)
    else:
        col_masses = np.asarray(col_masses, dtype=np.float64)
        if col_masses.shape != (n,):
            raise ValueError(f"col_masses must have shape ({n},).")

    if np.any(row_masses <= 0):
        raise ValueError("All row masses must be positive.")
    if np.any(col_masses <= 0):
        raise ValueError("All column masses must be positive.")
    if not np.isclose(np.sum(row_masses), np.sum(col_masses), rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"Total mass mismatch: sum(row_masses)={np.sum(row_masses)}, "
            f"sum(col_masses)={np.sum(col_masses)}"
        )

    A = np.maximum(A, eps)
    logA = np.log(A)

    log_r = np.log(row_masses)
    log_c = np.log(col_masses)

    log_u = np.zeros(m, dtype=np.float64)
    log_v = np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        log_u_old = log_u.copy()
        log_v_old = log_v.copy()

        # Update log_u so that row sums match row_masses
        # row i: sum_j exp(log_u[i] + logA[i,j] + log_v[j]) = row_masses[i]
        row_lse = logsumexp(logA + log_v[None, :], axis=1)
        log_u = log_r - row_lse

        # Update log_v so that col sums match col_masses
        col_lse = logsumexp(logA + log_u[:, None], axis=0)
        log_v = log_c - col_lse

        # Convergence check in primal residuals
        if np.max(np.abs(log_u - log_u_old)) < tol and np.max(np.abs(log_v - log_v_old)) < tol:
            break

    logQ = log_u[:, None] + logA + log_v[None, :]
    Q = np.exp(logQ)

    row_err = np.max(np.abs(Q.sum(axis=1) - row_masses))
    col_err = np.max(np.abs(Q.sum(axis=0) - col_masses))
    if row_err > 1e-6 or col_err > 1e-6:
        print(
            f"Warning: may not have fully converged. "
            f"max row err={row_err:.3e}, max col err={col_err:.3e}"
        )

    p = Q[target_row, :]

    if return_full:
        u = np.exp(log_u)
        v = np.exp(log_v)
        return p, Q, u, v, log_u, log_v
    return p


# # -------------------------
# # Example 1:
# # 3 behavior policies with k1, k2, k3 samples
# # rows = [policy1, policy2, policy3]
# # columns = all samples
# # -------------------------
# if __name__ == "__main__":
#     np.random.seed(0)

#     k1, k2, k3 = 3, 4, 5
#     n_samples = k1 + k2 + k3

#     # Suppose 3 policy rows, n_samples columns
#     A = np.random.rand(3, n_samples) + 0.1

#     row_masses = np.array([k1, k2, k3], dtype=float)
#     col_masses = np.ones(n_samples, dtype=float)

#     # return row 0 marginal as example
#     p, Q, u, v, log_u, log_v = sinkhorn_log_domain_with_row_masses(
#         A=A,
#         row_masses=row_masses,
#         col_masses=col_masses,
#         target_row=0,
#         return_full=True,
#     )

#     print("row sums:", Q.sum(axis=1))
#     print("col sums:", Q.sum(axis=0))
#     print("target row p:", p)
#     print("sum(target row p):", p.sum())










from copy import deepcopy


def compute_w_ptest_split_active_replacement(cal_test_vals_mat, depth_max):
    '''
        Computes the estimated MFCS Split CP weights for calibration and test points 
        (i.e., numerator in Eq. (9) in main paper or Eq. (16) in Appendix B.2)
        
        @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
                ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
                ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1
                
        @param : depth_max          : (int) indicating the maximum recursion depth
        
        :return: Unnormalized weights on calibration and test points, computed for recursion depth depth_max
    '''
    if (depth_max < 1):
        raise ValueError('Error: depth_max should be an integer >= 1. Currently, depth_max=' + str(depth_max))
      
    if (depth_max == 1):
        ## 
        return cal_test_vals_mat[-1]
        
    n_cal_test = np.shape(cal_test_vals_mat)[1]
    adjusted_vals = deepcopy(cal_test_vals_mat[-1])
    idx_include = np.repeat(True, n_cal_test)
    
    for i in range(n_cal_test):
        idx_include[i] = False
        idx_include[i-1] = True
        summation = compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], depth_max-1)
        adjusted_vals[i] = adjusted_vals[i] * summation
    return adjusted_vals
            
        
def compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat, depth_max):
    '''
        Helper function for "compute_w_ptest_split_active_replacement". Computes a summation such as the two sums in the numerator in equation (7) in paper
        
        @param : cal_test_vals_mat    : (float) matrix of weights with dim (depth_max, n_cal + 1).
                ## For t \in {1, ..., depth_max} : cal_test_vals_mat[t-1, j-1] = w_{n+t}(X_j) = exp(\lambda * \hat{\sigma^2}(X_j))
                ## where X_j is a calibration point for j \in {1, ..., n_cal} and the test point for j=n_cal + 1
                
        @param : depth_max          : (int) indicating the maximum recursion depth
        
        :return: Summation such as the two sums in the numerator in equation (7) in paper
    '''
    if depth_max == 1 or len(cal_test_vals_mat) == 1:
        return np.sum(cal_test_vals_mat)
    
    else:
        summation = 0
        n_cal_test = np.shape(cal_test_vals_mat)[1]
        idx_include = np.repeat(True, n_cal_test)
        for i in range(n_cal_test):
            idx_include[i] = False
            idx_include[i-1] = True
            summation += cal_test_vals_mat[-1,i]*compute_w_ptest_split_active_replacement_helper(cal_test_vals_mat[:-1,idx_include], depth_max - 1) 
        return summation
    
