import numpy as np

def saturate(val, lower_bound, upper_bound):
    return max(lower_bound, min(upper_bound, val))

class L1AOQP():
    """
    L1 Adaptive Optimization solver for the Quadratic Program:
        minimize 0.5 z'Hz + f'z
        s.t. Gz <= h
    """
    def __init__(
            self, ts, a, lpf_omega,
            interior_point_barrier=None, interior_point_slack=None,
            enable_prediction=False, clip_zdot=False):
        self.ts = ts
        self.enable_prediction = enable_prediction
        self.clip_zdot = clip_zdot
        self.zdot_min, self.zdot_max = -10., 10.

        # As: diagonal Hurwitz matrix (assume As = diag([a, a, ..., a]), a<0)
        self.a = a
        self.u = self.a / (np.exp(-self.a*self.ts) - 1.)
        self.dim = 0

        # Low-pass filter
        self.lpf_omega = lpf_omega

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

    def dimension_update(self, dim):
        """
        mu = inv(inv(As)*(I - expm(As*Ts)))*expm(As*Ts)
        Below implementation is only true for diagonal As
        For the MHE problem: dim is continuously growing until it reaches the horizon length
        """
        self.As = np.diag([self.a]*dim)
        self.mu = np.diag([self.u]*dim)
        self.dim = dim

    def lpf(self, x0, u):
        """
        C(s) = omega / (s + omega)
        xdot = -omega*x + omega*u
        """
        return x0 + self.lpf_omega*(u - x0)*self.ts

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

    def dynamics(self, z0, zdot0, za_dot0, grad_phi_hat0, zb_dot, t):
        """
        Assumes all vectors are of correct dimension (w.r.t. self.H, self.f). We cannot save these last values
        in this class because as the dimension grows, the previous values must be updated depending upon MHE
        formulation. This is handled in "mhe.py".

        Args:
            z0      := z(T): last solution
            zdot0   := zdot(T-1)
            za_dot0 := za_dot(T-1): L1AO derivative from last time step
            grad_phi_hat0 := grad_phi_hat(T): gradient prediction from last time step
            zb_dot(T): from baseline TV optimizer, e.g. PCIP
        Returns:
            z       := z(T+1): current solution
            zdot    := zdot(T)
        """
        Nz = z0.shape[0]
        if Nz != self.dim:
            self.dimension_update(Nz)   # Update As, mu
        """
        grad_phi = self.H @ z0 + self.f
        hess_phi = self.H

        diff = self.H.shape[0] - self.H0.shape[0]
        if diff == 0:   # QP size fixed
            grad_phi0 = self.H0 @ z0 + self.f0
            # hess_phi0 = self.H0
        else:   # QP size grew
            grad_phi0 = np.hstack([
                self.H0 @ z0[:-diff] + self.f0,
                grad_phi[-diff:]
            ])
            # hess_phi0 = self.H.copy()
            # hess_phi0[:-diff, :-diff] = self.H0.copy()

        # Estimate grad_zt_phi by finite difference
        if self.enable_prediction:
            grad_zt = (grad_phi - grad_phi0)/self.ts
        else:
            grad_zt = np.zeros_like(z0)
        """
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

        # L1AO
        e = grad_phi_hat0 - nabla_z_phi0    # e(T). (grad_phi_hat0 - grad_phi) gives worse result
        h = self.mu @ e                 # h(T)
        sigma_hat = np.linalg.solve(nabla_zz_phi, h)   # sigma_hat(T)
        za_dot = self.lpf(za_dot0, -sigma_hat)  # za_dot(T)
        if self.clip_zdot:
            for i in range(Nz):
                za_dot[i] = max(self.zdot_min, min(self.zdot_max, za_dot[i]))
        # za_dot = np.zeros(Nz)  # debug

        # Solution
        zdot = za_dot + zb_dot  # zdot(T)
        z = z0 + zdot*self.ts   # z(T+1)

        # Gradient prediction: grad_phi_hat(T+1)
        grad_phi_hat = grad_phi_hat0 + (self.As@e + prediction + nabla_zz_phi@zdot + h)*self.ts

        return za_dot, grad_phi_hat, z, zdot

