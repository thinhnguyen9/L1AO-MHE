import numpy as np
# from scipy.linalg import cho_factor, cho_solve

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
            enable_prediction=False
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
            nabla_z_B = (1./c) * G.T @ (1./slack)
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
            nabla_zz_B = (1./c) * G.T @ np.diag(1./(slack**2)) @ G
        else:
            nabla_zz_B = 0.0
        return H + nabla_zz_B

    def dynamics(self, z0, t):
        nabla_z_phi = self._nabla_z_phi(
            H=self.H,
            f=self.f,
            z=z0,
            G=self.G,
            h=self.h,
            t=t
        )
        nabla_zz_phi = self._nabla_zz_phi(
            H=self.H,
            f=self.f,
            z=z0,
            G=self.G,
            h=self.h,
            t=t
        )

        # Prediction term
        prediction = np.zeros_like(z0)
        if self.enable_prediction and t > 0.0:

            # Estimate nabla_zt_phi by finite differences
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

        # Solution
        correction = self.alpha * nabla_z_phi
        zdot = np.linalg.solve(nabla_zz_phi, - prediction - correction)
        z = z0 + self.ts*zdot  # z(T+1)!!
        return zdot, z
