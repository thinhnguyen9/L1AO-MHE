import numpy as np
import scipy
from pypardiso import spsolve, factorized

class LinearSolver:
    """
    Implement gradient methods to solve a linear system Ax=b.
    """
    def __init__(self, tol=1e-6, max_iters=100):
        # Iterative linear solver settings (for A x = b)
        self.cg_max_iters = max_iters
        self.cg_tol = tol
        # self.cg_min_denom = min_denom

        self.bfgs_max_iters = max_iters
        self.bfgs_tol = tol     # tol for Ax - b
    
    def solve(self, A, b, method='direct-lu', x0=None, A_inv_guess=None):
        """
        Solve Ax=b.

        Methods:
        - direct-lu       : LU factorization (numpy.linalg.solve)
        - direct-sym      : for symmetric A
        - direct-spd      : for symmetric positive definite A
        - direct-sparse   : for large sparse A
        - direct-cholesky : for symmetric positive definite A
        - iterative-cg1   : Conjugate Gradient (self.CG_linsol), for symmetric positive definite A
        - iterative-cg2   : Conjugate Gradient (scipy.sparse.linalg.cg)
        - iterative-gmres : GMRES, for general A

        Single-shooting:
        - Fastest: iterative-cg1, iterative-cg2, direct-sym, direct-spd
        - Slowest: direct-cholesky, direct-lu, iterative-gmres, direct-sparse

        Multiple-shooting:
        - Fastest: direct-sparse
        - Slowest: direct-cholesky, direct-lu, direct-sym
        - Failed/too slow: direct-spd, iterative-cg1, iterative-cg2, iterative-gmres
        """
        if method == 'direct-lu':
            x = np.linalg.solve(A, b)
        elif method == 'direct-sym':
            x = scipy.linalg.solve(A, b, assume_a='sym', check_finite=False)
        elif method == 'direct-spd':
            x = scipy.linalg.solve(A, b, assume_a='pos', check_finite=False)
        elif method == 'direct-sparse':
            x = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(A), b)
            # x = spsolve(scipy.sparse.csr_matrix(A), b)
            # try:
            #     x = self.pypardiso_solver(b)
            # except:
            #     self.pypardiso_solver = factorized(scipy.sparse.csr_matrix(A))
            #     x = self.pypardiso_solver(b)
        elif method == 'direct-cholesky':
            try:
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, b)
                x = np.linalg.solve(L.T, y)
            except:
                x = np.linalg.solve(A, b)
            # L, lower = cho_factor(A, lower=True)
            # x = cho_solve((L, lower), b)
        elif method == 'iterative-cg1':
            x = self.CG_linsol(A, b, x0=x0)
        elif method == 'iterative-cg2':
            # x = self.CG_linsol(A, b, x0=x0)
            x, exit_code = scipy.sparse.linalg.cg(A, b, x0=x0, atol=self.cg_tol, maxiter=self.cg_max_iters)
        elif method == 'iterative-gmres':
            x, exit_code = scipy.sparse.linalg.gmres(A, b, x0=x0, atol=self.cg_tol, maxiter=self.cg_max_iters)
        # elif method == 'BFGS':
        #     H, x = self.BFGS_linsol(A, b, last_Ainv=A_inv_guess, x0=x0)
        else:
            raise ValueError(f"Unsupported linear solver method: {method}. Choose from direct-{{lu,sym,spd,sparse,cholesky}} / iterative-{{cg1,cg2,gmres}}.")
        return x
    
    def CG_linsol(self, A, b, x0=None):
        """
        Solve A x = b approximately with Conjugate Gradient (CG).
        Assumes A is symmetric positive definite or close to it.
        """
        # initialize the solution
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        r = b - A @ x               # initial residual
        p = r.copy()                # initial search direction
        rs_old = float(r @ r)       # squared norm of residual

        if np.sqrt(rs_old) <= self.cg_tol:
            return x

        for _ in range(self.cg_max_iters):
            Ap = A @ p              # search direction
            denom = float(p @ Ap)
            # if abs(denom) <= self.cg_min_denom or not np.isfinite(denom):
            #     break

            alpha = rs_old / denom  # step size
            x = x + alpha * p       # update solution
            r = r - alpha * Ap      # update residual (save one A@p computation)

            rs_new = float(r @ r)
            if np.sqrt(rs_new) <= self.cg_tol:
                break

            # beta = rs_new / max(rs_old, self.cg_min_denom)
            # p = r + beta * p
            p = r + (rs_new / rs_old) * p   # update search direction
            rs_old = rs_new                 # update squared residual norm

        return x

    def BFGS_linsol(self, A, b, last_Ainv, x0=None):
        """
        Solve A x = b approximately with BFGS.
        min f(x) = 0.5 x^T A x - b^T x
        grad f(x) = A x - b
        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        H = last_Ainv

        d = len(b) # dimension of problem
        nabla = A @ x - b # initial gradient 
        # H = np.eye(d) # initial inverse A^{-1}

        for _ in range(self.bfgs_max_iters):
            if np.linalg.norm(nabla) <= self.bfgs_tol:
                break

            p = -H @ nabla # search direction (Newton Method)

            # ------------------- line search -------------------
            a = 1   # fixed a=1 also works!
            # c1 = 1e-4 
            # c2 = 0.9 
            # fx = 0.5 * x.T @ A @ x - b.T @ x
            # x_new = x + a * p 
            # nabla_new = A @ x_new - b
            # fx_new = 0.5 * x_new.T @ A @ x_new - b.T @ x_new
            # while fx_new >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
            #     a *= 0.5
            #     x_new = x + a * p 
            #     nabla_new = A @ x_new - b
            #     fx_new = 0.5 * x_new.T @ A @ x_new - b.T @ x_new
            #     if a < 1e-6:
            #         break
            # ---------------------------------------------------

            s = a * p 
            x_new = x + s 
            nabla_new = A @ x_new - b
            y = nabla_new - nabla 
            y = np.array([y])
            s = np.array([s])
            y = np.reshape(y,(d,1))
            s = np.reshape(s,(d,1))
            r = 1/(y.T@s)
            li = (np.eye(d)-(r*((s@(y.T)))))
            ri = (np.eye(d)-(r*((y@(s.T)))))
            hess_inter = li@H@ri
            H = hess_inter + (r*((s@(s.T)))) # BFGS Update
            nabla = nabla_new[:] 
            x = x_new[:]
        x = H @ b
        return H, x
    
