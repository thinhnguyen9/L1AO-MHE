import numpy as np
# from scipy.linalg import cho_factor, cho_solve
import time
from src.linear_solvers import LinearSolver
from scipy import sparse
from src.utils import remove_outliers_iqr, summarize_statistics

def saturate(val, lower_bound, upper_bound):
    return max(lower_bound, min(upper_bound, val))

class PCIPQP:
    """
    Prediction-Correction Interior-Point solver for the Quadratic Program:
        minimize    0.5 z'Hz + f'z
        subject to  Az = b
                    Gz <= h
    If shooting_method=="multiple", assume all matrices are sparse CSC/CSR!
    """
    def __init__(
            self, alpha, ts,
            model, shooting_method,
            interior_point_barrier=None, interior_point_slack=None,
            enable_prediction=False,
            linear_solver='direct-lu',
            linear_solver_tol=1e-6, linear_solver_max_iters=100,
            l1ao_augmentation=None  # dict with keys 'a' and 'lpf_omega'
        ):
        self.alpha   = alpha       # correction gain
        self.ts      = ts          # Euler step
        self.enable_prediction = enable_prediction    # False: reduced to continuous-time Newton's method
        self.Nx      = model.Nx
        self.shooting_method = shooting_method
        if self.shooting_method not in ["single", "multiple"]:
            raise ValueError(f"Invalid MHE/MPC shooting method: {self.shooting_method}. Must be 'single' or 'multiple'.")

        # Barrier parameter: c(t) = c0*exp(gamma_c*t) \to \infty
        if interior_point_barrier is None:
            self.c0 = 100.0
            self.gamma_c = 0.0
        else:
            self.c0, self.gamma_c = interior_point_barrier
        self.cmax = max(self.c0, 100.0)     # if c=\infty, constraints are not enforced

        # Slack variable: s(t) = s0*exp(-gamma_s*t) \to 0
        if interior_point_slack is None:
            self.s0 = 0.0
            self.gamma_s = 0.0
        else:
            self.s0, self.gamma_s = interior_point_slack

        # Try to save some computation time
        self.fixed_barrier_parameter = False
        self.fixed_slack_variable = False
        if self.gamma_c < 1e-12:     self.fixed_barrier_parameter = True
        if self.gamma_s < 1e-12:     self.fixed_slack_variable = True

        # Linear solver
        self.linear_solver_method = linear_solver
        self.linear_solver = LinearSolver(tol=linear_solver_tol, max_iters=linear_solver_max_iters)
        self._linear_solver_times = []
        self._gradient_times = []
        self._hessian_times = []
        self._total_solver_times = []
        self._last_zdot = None
        self._last_Hess_inv = None

        # -------------------- L1AO augmentation -------------------- #
        self.enable_l1ao = False
        if l1ao_augmentation is not None:
            self.enable_l1ao = True
            self._last_sigma_hat = None

            # As: diagonal Hurwitz matrix (assume As = diag([a, a, ..., a]), a<0)
            self.As = l1ao_augmentation['a']    # no need to form diag matrix! Save computation time
            self.mu = self.As / (np.exp(-self.As*self.ts) - 1.)

            # Low-pass filter
            self.lpf_omega = l1ao_augmentation['lpf_omega']
        # ----------------------------------------------------------- #
    
    def set_QP(self, H, f, A=None, b=None, G=None, h=None, t=None):
        if A is not None and b is not None:
            self.has_equality_constraints = True
        else:
            self.has_equality_constraints = False
        if G is not None and h is not None:
            self.has_inequality_constraints = True
        else:
            self.has_inequality_constraints = False
        # ---------------- debug ---------------- #
        # self.has_inequality_constraints = False
        # self.G = None
        # self.h = None
        # self.G0 = None
        # self.h0 = None
        # --------------------------------------- #
        if hasattr(self, 'H') and hasattr(self, 'f'):
            self.H0 = self.H
            self.f0 = self.f
        else:
            self.H0 = H
            self.f0 = f
        self.H = H
        self.f = f

        if self.has_equality_constraints:
            if hasattr(self, 'A') and hasattr(self, 'b'):
                self.A0 = self.A
                self.b0 = self.b
            else:
                self.A0 = A
                self.b0 = b
            self.A = A
            self.b = b
        else:
            self.A0 = None
            self.b0 = None
            self.A = None
            self.b = None

        if self.has_inequality_constraints:
            if hasattr(self, 'G') and hasattr(self, 'h'):
                self.G0 = self.G
                self.h0 = self.h
            else:
                self.G0 = G
                self.h0 = h
            self.G = G
            self.h = h
        else:
            self.G0 = None
            self.h0 = None
            self.G = None
            self.h = None
    
    def get_params(self, t):
        # c = saturate(self.c0 * np.exp(self.gamma_c*t), self.c0, self.cmax)
        c = self.c0 * np.exp(self.gamma_c*t)    # continuous c gives better performance (but too large, gives overflow warning)
        s = self.s0 * np.exp(-self.gamma_s*t)
        cdot = self.gamma_c * c
        sdot = -self.gamma_s * s
        return c, s, cdot, sdot
    
    def _phi(self, H, f, z, A, b, G, h, t):
        """
        Objective function phi(t) with:
            - Lagrange multiplier term E for equality constraints: l'(Ax - b),
            - log barrier function B for inequality constraints: B = - (1/c)*sum(log(s - (Gx - h))).
        """
        Nx = H.shape[0]
        Nl = A.shape[0] if A is not None else 0
        x = z[:Nx]      # Decision variables
        l = z[Nx:]      # Lagrange multipliers
        phi = 0.5*x.T@H@x + f.T@x
        if self.has_inequality_constraints:
            c, s, _, _ = self.get_params(t)
            slack = s - (G@x - h)
            if np.any(slack <= 0):
                return np.inf
            B = -(1./c)*np.sum(np.log(slack))
            phi += B
        if self.has_equality_constraints:
            phi += l.T@(A@x - b)
        return phi

    def _nabla_z_phi(self, H, f, z, A, b, G, h, t):
        """
        Objective Jacobian: nabla_z_phi(t)
        """
        t0 = time.perf_counter()                # uncomment to record computation time
        Nx = H.shape[0]
        Nl = A.shape[0] if A is not None else 0
        x = z[:Nx]      # Decision variables
        l = z[Nx:]      # Lagrange multipliers
        nabla_z_phi = H@x + f
        if self.has_inequality_constraints:
            c, s, _, _ = self.get_params(t)
            slack = s - (G@x - h)
            inv_slack = 1./slack
            nabla_z_B = (1./c) * (G.T @ inv_slack)
            nabla_z_phi = nabla_z_phi + nabla_z_B
        if self.has_equality_constraints:
            nabla_z_phi = np.hstack((nabla_z_phi + A.T@l, A@x - b))
        t1 = time.perf_counter()                # uncomment to record computation time
        self._gradient_times.append(t1 - t0)    # uncomment to record computation time
        return nabla_z_phi

    def _nabla_zz_phi(self, H, f, z, A, b, G, h, t):
        """
        Objective Hessian: nabla_zz_phi(t)
        """
        t0 = time.perf_counter()                # uncomment to record computation time
        Nx = H.shape[0]
        Nl = A.shape[0] if A is not None else 0
        x = z[:Nx]      # Decision variables
        l = z[Nx:]      # Lagrange multipliers
        nabla_zz_phi = H
        if self.has_inequality_constraints:
            c, s, _, _ = self.get_params(t)
            slack = s - (G@x - h)
            inv_slack_sq = 1.0 / (slack**2)
            weighted_G = G.multiply(inv_slack_sq[:,None]) if hasattr(G, "multiply") else inv_slack_sq[:,None]*G
            nabla_zz_B = (1./c) * (G.T @ weighted_G)
            nabla_zz_phi = nabla_zz_phi + nabla_zz_B
            # if self.shooting_method == "single":
            #     # ----------- Method 1 ----------- #
            #     # nabla_zz_B = (1./c) * G.T @ np.diag(1./(slack**2)) @ G    # 50% slower than method 2
            #     # ----------- Method 2 ----------- #
            #     weighted_G = inv_slack_sq[:, None] * G
            # elif self.shooting_method == "multiple":
            #     # ----------- Method 3 ----------- #
            #     # Multiple-shooting w/ very sparse G = (-I 0; I 0). 2-4x faster than method 2
            #     # p = int(slack.shape[0] / 2)
            #     # temp = (1./c) * (inv_slack_sq[:p] + inv_slack_sq[p:])
            #     # # nabla_zz_B = np.zeros((H.shape[0], H.shape[0]))
            #     # for i, val in enumerate(temp):
            #     #     # nabla_zz_B[i, i] = val
            #     #     nabla_zz_phi[i, i] += val
        if self.has_equality_constraints:
            nabla_zz_phi = sparse.bmat([
                [nabla_zz_phi,  A.T],
                [A,             None]
            ], format="csc")
        t1 = time.perf_counter()                # uncomment to record computation time
        self._hessian_times.append(t1 - t0)     # uncomment to record computation time
        return nabla_zz_phi

    def dynamics(
            self,
            z0,
            z0_unextended,
            zdot0,
            t,
            l1ao_zdot0=None,
            l1ao_sigma_hat0=None,
            l1ao_nabla_z_phi_hat0=None
        ):
        """
        Dimension of z0:
            - with equality constraints    : H.shape[0] + A.shape[0]
            - without equality constraints : H.shape[0]
        In MHE context:
            - with equality constraints (multiple-shooting)  : (3N+1)*Nx
            - without equality constraints (single-shooting) : (N+1)*Nx
        """
        t0 = time.perf_counter()                # uncomment to record computation time
        
        # %% Compute current gradient and Hessian
        nabla_z_phi = self._nabla_z_phi(
            H=self.H,
            f=self.f,
            z=z0,
            A=self.A,
            b=self.b,
            G=self.G,
            h=self.h,
            t=t
        )
        nabla_zz_phi = self._nabla_zz_phi(
            H=self.H,
            f=self.f,
            z=z0,
            A=self.A,
            b=self.b,
            G=self.G,
            h=self.h,
            t=t
        )
        
        if self.enable_prediction or self.enable_l1ao:
            nabla_z_phi0 = self._nabla_z_phi(
                H=self.H0,
                f=self.f0,
                z=z0_unextended,
                A=self.A0,
                b=self.b0,
                G=self.G0,
                h=self.h0,
                t=t-self.ts
            )
            if nabla_z_phi0.shape[0] < nabla_z_phi.shape[0]:
                nabla_z_phi0 = self.growing_horizon_extend_variables(nabla_z_phi0, nabla_z_phi)

        # %% Prediction term
        prediction = np.zeros_like(z0)
        if self.enable_prediction and t > 0.0:

            # Estimate nabla_zt_phi by finite differences
            prediction += (nabla_z_phi - nabla_z_phi0)/self.ts

            # nabla_zc_phi*cdot + nabla_zs_phi*sdot
            if self.has_inequality_constraints:
                c, s, cdot, sdot = self.get_params(t)
                Nx = self.H.shape[0]
                Nl = self.A.shape[0] if self.A is not None else 0
                slack = s - (self.G@z0[:Nx] - self.h)
                if not self.fixed_barrier_parameter:
                    nabla_zc_phi = (-1./(c**2)) * self.G.T @ (1./slack)
                    prediction = prediction + np.hstack((
                        nabla_zc_phi*cdot,
                        np.zeros((Nl,))
                    ))
                if not self.fixed_slack_variable:
                    nabla_zs_phi = (-1./c) * self.G.T @ (1./(slack**2))
                    prediction = prediction + np.hstack((
                        nabla_zs_phi*sdot,
                        np.zeros((Nl,))
                    ))
        # prediction = 0.0    # debug

        # %% PCIP step
        correction = self.alpha * nabla_z_phi

        t1 = time.perf_counter()                # uncomment to record computation time
        zdot = self.linear_solver.solve(
            A      = nabla_zz_phi,
            b      = - prediction - correction,
            method = self.linear_solver_method,
            x0     = zdot0
        )
        # # --------- BFGS --------- #
        # if t > 0.0 and diff == 0:
        #     last_Hess_inv = self._last_Hess_inv
        # else:
        #     last_Hess_inv = np.linalg.inv(nabla_zz_phi)
        # self._last_Hess_inv, zdot = self.linear_solver.BFGS_linsol(A, b, last_Hess_inv, x0=x0)
        t2 = time.perf_counter()                # uncomment to record computation time
        total_linsol_time = t2 - t1             # uncomment to record computation time

        # %% L1AO augmentation
        l1ao_zdot = None
        l1ao_sigma_hat = None
        l1ao_nabla_z_phi_hat = None
        if self.enable_l1ao:
            e = l1ao_nabla_z_phi_hat0 - nabla_z_phi0    # e(T). (self._l1ao_nabla_z_phi_hat0 - grad_phi) gives worse result
            h = self.mu * e                             # h(T)
            t3 = time.perf_counter()            # uncomment to record computation time
            l1ao_sigma_hat = self.linear_solver.solve(  # sigma_hat(T)
                A      = nabla_zz_phi,
                b      = h,
                method = self.linear_solver_method,
                x0     = l1ao_sigma_hat0,
                reuse_factorization = True
            )
            t4 = time.perf_counter()            # uncomment to record computation time
            l1ao_zdot = self._l1ao_lpf(l1ao_zdot0, -l1ao_sigma_hat)  # za_dot(T)
            # za_dot = np.zeros(Nz)  # debug

            # z(T+1)
            zdot += l1ao_zdot

            # Gradient prediction: grad_phi_hat(T+1)
            l1ao_nabla_z_phi_hat = l1ao_nabla_z_phi_hat0 + (self.As*e + prediction + nabla_zz_phi@zdot + h)*self.ts

            total_linsol_time += t4 - t3        # uncomment to record computation time
        # ----------------------------------------------------------- #

        # %% Finishing
        self._linear_solver_times.append(total_linsol_time)     # uncomment to record computation time

        self._last_zdot = zdot.copy()
        z = z0 + self.ts*zdot  # z(T+1)!!

        t5 = time.perf_counter()                    # uncomment to record computation time
        self._total_solver_times.append(t5 - t0)    # uncomment to record computation time
        return zdot, z, l1ao_zdot, l1ao_sigma_hat, l1ao_nabla_z_phi_hat
    
    def growing_horizon_extend_variables(self, z_old, z_new):
        """
        Extend z_old with new variables from z_new.
        """
        if self.shooting_method == "single":
            return np.hstack((
                z_old,
                z_new[-self.Nx:]
            ))
        elif self.shooting_method == "multiple":
            N1 = int((z_old.shape[0]/self.Nx - 1) / 3)   # old horizon length
            N2 = int((z_new.shape[0]/self.Nx - 1) / 3)   # new horizon length
            return np.hstack((
                z_old[               0 : (N1+1)*self.Nx  ],   # x(0)...x(N)
                z_new[      N2*self.Nx : (N2+1)*self.Nx  ],   # x(N)
                z_old[  (N1+1)*self.Nx : (2*N1+1)*self.Nx],   # w(0)...w(N-1)
                z_new[    2*N2*self.Nx : (2*N2+1)*self.Nx],   # w(N-1)
                z_old[(2*N1+1)*self.Nx :                 ],   # l(0)...l(N-1)
                z_new[        -self.Nx :                 ]    # l(N-1)
            ))
    
    def print_computation_times(self, start_idx=0):
        gradient_times = np.array(self._gradient_times[start_idx:]) * 1000
        hessian_times = np.array(self._hessian_times[start_idx:]) * 1000
        linsol_times = np.array(self._linear_solver_times[start_idx:]) * 1000
        total_times = np.array(self._total_solver_times[start_idx:]) * 1000

        gradient_times = remove_outliers_iqr(gradient_times)
        hessian_times = remove_outliers_iqr(hessian_times)
        linsol_times = remove_outliers_iqr(linsol_times)
        total_times = remove_outliers_iqr(total_times)

        gradient_stats = summarize_statistics(gradient_times)
        hessian_stats = summarize_statistics(hessian_times)
        linsol_stats = summarize_statistics(linsol_times)
        total_stats = summarize_statistics(total_times)

        if self.enable_l1ao: print("------------------------------- PCIP+L1AO computation time breakdown -------------------------------")
        else:                print("--------------------------------- PCIP computation time breakdown ----------------------------------")
        print(f"{'Metric':<22} {'Mean [ms]':>12} {'% Total':>12} {'Median [ms]':>12} {'Std [ms]':>12} {'Min [ms]':>12} {'Max [ms]':>12}")
        print("-" * 100)
        for label, stats in [
            ('Gradient time', gradient_stats),
            ('Hessian time', hessian_stats),
            ('Linear solver time', linsol_stats),
            ('Total solver time', total_stats),
        ]:
            print(
                f"{label:<22}"
                f" {stats['mean']:12.4f}"
                f" {stats['mean'] / total_stats['mean'] * 100.0:10.1f}%"
                f" {stats['median']:12.4f}"
                f" {stats['std']:12.4f}"
                f" {stats['min']:12.4f}"
                f" {stats['max']:12.4f}"
            )

    # %% L1AO augmentation methods %%
    def _l1ao_lpf(self, x0, u):
        """
        C(s) = omega / (s + omega)
        xdot = -omega*x + omega*u
        """
        return x0 + self.lpf_omega*(u - x0)*self.ts
