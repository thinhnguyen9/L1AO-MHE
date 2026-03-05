import numpy as np
# from scipy.linalg import cho_factor, cho_solve
import time
from src.linear_solvers import LinearSolver

def saturate(val, lower_bound, upper_bound):
    return max(lower_bound, min(upper_bound, val))

class PCIPQP:
    """
    Prediction-Correction Interior-Point solver for the Quadratic Program:
        minimize 0.5 z'Hz + f'z
        s.t. Gz <= h
    """
    def __init__(
            self, alpha, ts,
            interior_point_barrier=None, interior_point_slack=None,
            enable_prediction=False,
            linear_solver='np',     # 'np' (numpy - LU decomposition), 'cg' (conjugate gradient), 'bfgs' (BFGS quasi-Newton)
            linear_solver_tol=1e-6, linear_solver_max_iters=100,
            l1ao_augmentation=None  # dict with keys 'a' and 'lpf_omega'
        ):
        self.alpha   = alpha       # correction gain
        self.ts      = ts          # Euler step
        self.enable_prediction = enable_prediction    # False: reduced to continuous-time Newton's method

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
        if linear_solver not in ['np', 'cg', 'bfgs']:
            raise ValueError(f"Invalid linear solver: {linear_solver}. Must be 'np', 'cg', or 'bfgs'.")
        self.linear_solver_method = linear_solver
        self.linear_solver = LinearSolver(tol=linear_solver_tol, max_iters=linear_solver_max_iters)
        self._linear_solver_times = []
        self._last_zdot = None
        self._last_Hess_inv = None

        # -------------------- L1AO augmentation -------------------- #
        self.enable_l1ao = False
        if l1ao_augmentation is not None:
            self.enable_l1ao = True
            self._last_sigma_hat = None

            # As: diagonal Hurwitz matrix (assume As = diag([a, a, ..., a]), a<0)
            self.a = l1ao_augmentation['a']
            self.u = self.a / (np.exp(-self.a*self.ts) - 1.)
            self.dim = 0

            # Low-pass filter
            self.lpf_omega = l1ao_augmentation['lpf_omega']
        # ----------------------------------------------------------- #
    
    def set_QP(self, H, f, G=None, h=None, t=None):
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
        if self.enable_prediction:
            if hasattr(self, 'H') and hasattr(self, 'f'):
                self.H0 = self.H
                self.f0 = self.f
            else:
                self.H0 = H
                self.f0 = f
            if self.has_inequality_constraints:
                if hasattr(self, 'G') and hasattr(self, 'h'):
                    self.G0 = self.G
                    self.h0 = self.h
                else:
                    self.G0 = G
                    self.h0 = h
            else:
                self.G0 = None
                self.h0 = None
        self.H = H
        self.f = f
        if self.has_inequality_constraints:
            self.G = G
            self.h = h
        else:
            self.G = None
            self.h = None
    
    def get_params(self, t):
        c = saturate(self.c0 * np.exp(self.gamma_c*t), self.c0, self.cmax)
        s = self.s0 * np.exp(-self.gamma_s*t)
        cdot = self.gamma_c * c
        sdot = -self.gamma_s * s
        return c, s, cdot, sdot
    
    def _phi(self, H, f, z, G, h, t):
        """
        Objective function phi(t) with log barrier function B.
        """
        if self.has_inequality_constraints:
            c, s, _, _ = self.get_params(t)
            slack = s - (G@z - h)
            if np.any(slack <= 0):
                return np.inf
            B = -(1./c)*np.sum(np.log(slack))
        else:
            B = 0.0
        return 0.5*z.T@H@z + f.T@z + B
    
    def _nabla_z_phi(self, H, f, z, G, h, t):
        """
        Objective Jacobian: nabla_z_phi(t)
        """
        if self.has_inequality_constraints:
            c, s, _, _ = self.get_params(t)
            slack = s - (G@z - h)
            inv_slack = 1./slack
            nabla_z_B = (1./c) * (G.T @ inv_slack)
        else:
            nabla_z_B = 0.0
        return H@z + f + nabla_z_B
    
    def _nabla_zz_phi(self, H, f, z, G, h, t):
        """
        Objective Hessian: nabla_zz_phi(t)
        """
        if self.has_inequality_constraints:
            c, s, _, _ = self.get_params(t)
            slack = s - (G@z - h)
            # nabla_zz_B = (1./c) * G.T @ np.diag(1./(slack**2)) @ G    # 50% slower than the following
            inv_slack_sq = 1.0 / (slack**2)
            weighted_G = inv_slack_sq[:, None] * G
            nabla_zz_B = (1./c) * (G.T @ weighted_G)
        else:
            nabla_zz_B = 0.0
        return H + nabla_zz_B

    def dynamics(self, z0, t):
        # t0 = time.perf_counter()
        nabla_z_phi = self._nabla_z_phi(
            H=self.H,
            f=self.f,
            z=z0,
            G=self.G,
            h=self.h,
            t=t
        )
        # t1 = time.perf_counter()
        nabla_zz_phi = self._nabla_zz_phi(
            H=self.H,
            f=self.f,
            z=z0,
            G=self.G,
            h=self.h,
            t=t
        )
        # t2 = time.perf_counter()

        if self.enable_prediction or self.enable_l1ao:
            diff = self.H.shape[0] - self.H0.shape[0]
            nabla_z_phi0 = self._nabla_z_phi(
                H=self.H0,
                f=self.f0,
                z=z0[:-diff] if diff>0 else z0,
                G=self.G0,
                h=self.h0,
                t=t-self.ts
            )
            if diff > 0:
                nabla_z_phi0 = np.hstack((nabla_z_phi0, nabla_z_phi[-diff:]))

        # Prediction term
        prediction = np.zeros_like(z0)
        if self.enable_prediction and t > 0.0:

            # Estimate nabla_zt_phi by finite differences
            # diff = self.H.shape[0] - self.H0.shape[0]
            # nabla_z_phi0 = self._nabla_z_phi(
            #     H=self.H0,
            #     f=self.f0,
            #     z=z0[:-diff] if diff>0 else z0,
            #     G=self.G0,
            #     h=self.h0,
            #     t=t-self.ts
            # )
            # if diff > 0:
            #     nabla_z_phi0 = np.hstack((nabla_z_phi0, nabla_z_phi[-diff:]))
            prediction += (nabla_z_phi - nabla_z_phi0)/self.ts

            # nabla_zc_phi*cdot + nabla_zs_phi*sdot
            if self.has_inequality_constraints:
                c, s, cdot, sdot = self.get_params(t)
                slack = s - (self.G@z0 - self.h)
                if not self.fixed_barrier_parameter:
                    nabla_zc_phi = (-1./(c**2)) * self.G.T @ (1./slack)
                    prediction += nabla_zc_phi*cdot
                if not self.fixed_slack_variable:
                    nabla_zs_phi = (-1./c) * self.G.T @ (1./(slack**2))
                    prediction += nabla_zs_phi*sdot
        # t3 = time.perf_counter()

        # ================================= PCIP step ================================= #
        correction = self.alpha * nabla_z_phi
        A, b = nabla_zz_phi, - prediction - correction

        t4 = time.perf_counter()
        if self.linear_solver_method == 'np':
            # --------- Direct solve using numpy (LU decomposition) --------- #
            zdot = np.linalg.solve(A, b)
        else:
            # --------- Quasi-Newton methods --------- #
            x0 = None
            if self._last_zdot is not None and self._last_zdot.shape == b.shape:
                # warm_start = self._last_zdot
                x0 = np.hstack((self._last_zdot[12:], self._last_zdot[-12:]))
            if self.linear_solver_method == 'cg':
                # --------- Conjugate gradient --------- #
                zdot = self.linear_solver.CG_linsol(A, b, x0=x0)
            elif self.linear_solver_method == 'bfgs':
                # --------- BFGS --------- #
                if t > 0.0 and diff == 0:
                    last_Hess_inv = self._last_Hess_inv
                else:
                    last_Hess_inv = np.linalg.inv(nabla_zz_phi)
                self._last_Hess_inv, zdot = self.linear_solver.BFGS_linsol(A, b, last_Hess_inv, x0=x0)
        t5 = time.perf_counter()

        # ================================= L1AO augmentation ================================= #
        if self.enable_l1ao:
            Nz = z0.shape[0]
            if Nz != self.dim:
                self._l1ao_dimension_update(Nz)   # Update As, mu, self._l1ao_nabla_z_phi_hat0, self._l1ao_za_dot0

            e = self._l1ao_nabla_z_phi_hat0 - nabla_z_phi0    # e(T). (self._l1ao_nabla_z_phi_hat0 - grad_phi) gives worse result
            h = self.mu @ e                 # h(T)
            t6 = time.perf_counter()
            # sigma_hat = np.linalg.solve(nabla_zz_phi, h)   # sigma_hat(T)
            # --------- Quasi-Newton methods --------- #
            x0 = None
            if self._last_sigma_hat is not None and self._last_sigma_hat.shape == h.shape:
                x0 = np.hstack((self._last_sigma_hat[12:], self._last_sigma_hat[-12:]))
            sigma_hat = self.linear_solver.CG_linsol(nabla_zz_phi, h, x0=x0)
            self._last_sigma_hat = sigma_hat.copy()
            # ---------------------------------------- #
            t7 = time.perf_counter()
            za_dot = self._l1ao_lpf(self._l1ao_za_dot0, -sigma_hat)  # za_dot(T)
            # za_dot = np.zeros(Nz)  # debug

            # z(T+1)
            zdot += za_dot

            # Gradient prediction: grad_phi_hat(T+1)
            grad_phi_hat = self._l1ao_nabla_z_phi_hat0 + (self.As@e + prediction + nabla_zz_phi@zdot + h)*self.ts

            # Save for next step
            self._l1ao_za_dot0 = za_dot
            self._l1ao_nabla_z_phi_hat0 = grad_phi_hat

            self._linear_solver_times.append(t5 - t4 + t7 - t6)
        # ----------------------------------------------------------- #
        else:
            self._linear_solver_times.append(t5 - t4)

        # print("-----------------------------------")
        # print(f"Gradient time:      {(t1-t0)*1000:.2f} ms")
        # print(f"Hessian time:       {(t2-t1)*1000:.2f} ms")
        # print(f"Prediction time:    {(t3-t2)*1000:.2f} ms")
        # print(f"Linear solver time: {(t5-t4)*1000:.2f} ms")
        # print(f"Total time:         {(t5-t0)*1000:.2f} ms")

        self._last_zdot = zdot.copy()
        z = z0 + self.ts*zdot  # z(T+1)!!
        return zdot, z
    
    def get_mean_linsol_time(self, start_idx=0):
        if len(self._linear_solver_times) == 0:
            return 0.0
        else:
            return np.mean(self._linear_solver_times[start_idx:])
    
    # %% L1AO augmentation methods %%
    def _l1ao_dimension_update(self, dim):
        """
        mu = inv(inv(As)*(I - expm(As*Ts)))*expm(As*Ts)
        Below implementation is only true for diagonal As
        For the MHE problem: dim is continuously growing until it reaches the horizon length
        """
        self.As = np.diag([self.a]*dim)
        self.mu = np.diag([self.u]*dim)
        self.dim = dim

        # T=0: initialize
        if not hasattr(self, '_l1ao_nabla_z_phi_hat0') or not hasattr(self, '_l1ao_za_dot0'):
            self._l1ao_nabla_z_phi_hat0 = np.zeros(dim)
            self._l1ao_za_dot0 = np.zeros(dim)
        
        # T=1,...,N-1: growing horizon
        else:
            self._l1ao_nabla_z_phi_hat0 = np.hstack((
                self._l1ao_nabla_z_phi_hat0,
                self._l1ao_nabla_z_phi_hat0[-12:]
            ))
            self._l1ao_za_dot0 = np.hstack((
                self._l1ao_za_dot0,
                self._l1ao_za_dot0[-12:]
            ))

    def _l1ao_lpf(self, x0, u):
        """
        C(s) = omega / (s + omega)
        xdot = -omega*x + omega*u
        """
        return x0 + self.lpf_omega*(u - x0)*self.ts
