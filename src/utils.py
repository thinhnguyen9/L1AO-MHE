import numpy as np
from scipy.linalg import block_diag

def build_mhe_qp_multiple_shooting(A_seq, B_seq, G_seq, C_seq, Q_inv, R_inv, x_prior, P_prior_inv, u_seq, y_seq):
    """
    Optimization variable: z = [x(0),..., x(N), w(0),..., w(N-1)]

    Cost:
        J = 0.5*(x_0 - x_prior)' P_prior^{-1} (x_0 - x_prior)
            + 0.5*sum_{i=0}^{N-1} w_i' Q^{-1} w_i
            + 0.5*sum_{i=0}^{N}   (y_i - C_i x_i)' R^{-1} (y_i - C_i x_i)
    Dynamics constraints: A_eq z = b_eq
    J = 0.5 z'H z + f'z + v'(A_eq z - b_eq)  (Lagrange multipliers v)
    """
    N = len(A_seq)          # window steps N (transitions), states are N+1
    # n = A_seq[0].shape[0]
    # m = u_seq[0].shape[0]
    # p = y_seq[0].shape[0]
    n = x_prior.shape[0]

    # Quadratic terms initialization
    z_len = (2*N+1)*n   # z = [x(0),..., x(N), w(0),..., w(N-1)]
    H = np.zeros((z_len, z_len))
    f = np.zeros((z_len,))
    A_eq = np.zeros((N*n, z_len))
    b_eq = np.zeros((N*n,))

    # Prior term on x_0
    H[0:n, 0:n] += P_prior_inv
    f[0:n]      += -P_prior_inv @ x_prior

    for i in range(N):
        Ai, Bi, Gi, Ci, ui, yi = A_seq[i], B_seq[i], G_seq[i], C_seq[i], u_seq[i], y_seq[i]
        idx_xi  = slice(i*n, (i+1)*n)
        idx_xi1 = slice((i+1)*n, (i+2)*n)
        idx_wi  = slice(n*(N+1) + i*n, n*(N+1) + (i+1)*n)

        # Process noise cost
        H[idx_wi, idx_wi] += Q_inv

        # Measurement cost
        H[idx_xi, idx_xi] += Ci.T @ R_inv @ Ci
        f[idx_xi]         += -Ci.T @ R_inv @ yi

        # Dynamics constraint: x(i+1) - A(i)x(i) - G(i)w(i) = B(i)u(i)
        A_eq[i*n:(i+1)*n, idx_xi1] = np.eye(n)
        A_eq[i*n:(i+1)*n, idx_xi] = -Ai
        A_eq[i*n:(i+1)*n, idx_wi] = -Gi
        b_eq[i*n:(i+1)*n] = Bi @ ui

    idx_xN = slice(N*n, (N+1)*n)
    C = C_seq[N]
    H[idx_xN, idx_xN] += C.T @ R_inv @ C
    f[idx_xN] += -C.T @ R_inv @ y_seq[N]

    return H, f, A_eq, b_eq


def build_mhe_qp_equality_constraints_lagrangian(H, f, A_eq, b_eq):

    # With Lagrange multiplier:
    # V = 0.5 z'Hz + f'z + v'(A_eq z - b_eq)
    #   = 0.5 [z;v]' [H, A_eq'; A_eq, 0] [z;v] + [f; -b_eq]' [z;v]
    v_len = A_eq.shape[0]
    H = np.block([[H, A_eq.T],
                  [A_eq, np.zeros((v_len, v_len))]])
    f = np.hstack([f, -b_eq])
    return H, f

def build_mhe_qp_single_shooting(
        A_seq, B_seq, G_seq, C_seq, Qinv_seq, Rinv_seq,
        x_prior, P_prior_inv, u_seq, y_seq,
        smoothing_adjustment=False, Q_seq=None, R_seq=None
    ):
    """
    Linear QP with dynamics incorporated into the objective (no constraints).
    Optimization variable: z = [x(0), w(0), ..., w(N-1)]
    Objective: V = 0.5 z'Hz + f'z

    Arguments:
        A_seq = [A(0), ..., A(N-1)]
        B_seq = [B(0), ..., B(N-1)]
        G_seq = [G(0), ..., G(N-1)]
        C_seq = [C(0), ..., C(N)]
        Qinv_seq = [Q(0)^{-1}, ..., Q(N-1)^{-1}]
        Rinv_seq = [R(0)^{-1}, ..., R(N)^{-1}]
        u_seq = [u(0), ..., u(N-1)]
        y_seq = [y(0), ..., y(N)]
        Q_seq = [Q(0), ..., Q(N)]   (only used by smoothing MHE)
        R_seq = [R(0), ..., R(N)]   (only used by smoothing MHE)
    """
    N = len(A_seq)
    nx = x_prior.shape[0]
    ny = y_seq[0].shape[0]
    if N>0: nu = u_seq[0].shape[0]

    A1 = np.eye((N+1)*nx)
    for i in range(N):
        A1[(i+1)*nx:(i+2)*nx, 0:(i+1)*nx] = A_seq[i] @ A1[i*nx:(i+1)*nx, 0:(i+1)*nx]
    
    G = block_diag(np.eye(nx), *G_seq)
    B = block_diag(*B_seq)
    matA = A1 @ G
    if N>0:
        matb = A1[:, nx:] @ B @ u_seq.flatten()
    else:
        matb = np.zeros((nx, ))
    matC = block_diag(*C_seq)
    matR = block_diag(*Rinv_seq)
    matQ = block_diag(np.zeros((nx,nx)), *Qinv_seq)

    # QP matrices
    AT_CT_R = matA.T @ matC.T @ matR
    H = matQ + AT_CT_R @ matC @ matA
    f = AT_CT_R @ (matC @ matb - y_seq.flatten())
    H[0:nx, 0:nx] += P_prior_inv
    f[0:nx]       += -P_prior_inv @ x_prior

    if smoothing_adjustment and N>0:
        Ax = np.concatenate([matA[:, :nx], np.zeros(((N+1)*nx, N*nx))], axis=1)
        Aw = np.concatenate([np.zeros(((N+1)*nx, nx)), matA[:, nx:]], axis=1)
        Q = block_diag(*Q_seq)  # last element can be 0 or Q(N), doesn't matter
        R = block_diag(*R_seq[:N])
        ybar = y_seq[:N]
        Cbar = np.concatenate([block_diag(*C_seq[:N]), np.zeros((N*ny, nx))], axis=1)
        W = Cbar @ Aw @ Q @ Aw.T @ Cbar.T + R
        W_inv = np.linalg.inv(W)
        AT_CT_W = Ax.T @ Cbar.T @ W_inv
        H += - AT_CT_W @ Cbar @ Ax
        f += - AT_CT_W @ (Cbar @ matb - ybar.flatten())

    # [x0...xN]' = matA @ z + matb
    return H, f, matA

def rmse(x_true, x_est):
    err = x_true - x_est    # N x nx
    rmse_val = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse_val
