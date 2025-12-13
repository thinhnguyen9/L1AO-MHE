import numpy as np

class L1AOQP():
    """
    L1 Adaptive Optimization solver for (unconstrained) Quadratic Program:
        minimize 0.5 z^T H z + f^T z
    """
    def __init__(self, ts, a, lpf_omega, enable_prediction=False, clip_zdot=False):
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

    def set_QP(self, H, f):
        if hasattr(self, 'H') and hasattr(self, 'f'):
            self.H0 = self.H
            self.f0 = self.f
        else:
            self.H0 = H
            self.f0 = f
        self.H = H
        self.f = f
    
    def dynamics(self, z0, zdot0, za_dot0, grad_phi_hat0, zb_dot):
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

        # L1AO
        e = grad_phi_hat0 - grad_phi0    # e(T). (grad_phi_hat0 - grad_phi) gives worse result
        h = self.mu @ e                 # h(T)
        sigma_hat = np.linalg.solve(hess_phi, h)   # sigma_hat(T)
        za_dot = self.lpf(za_dot0, -sigma_hat)  # za_dot(T)
        if self.clip_zdot:
            for i in range(Nz):
                za_dot[i] = max(self.zdot_min, min(self.zdot_max, za_dot[i]))
        # za_dot = np.zeros(Nz)  # debug

        # Solution
        zdot = za_dot + zb_dot  # zdot(T)
        z = z0 + zdot*self.ts   # z(T+1)

        # Gradient prediction: grad_phi_hat(T+1)
        grad_phi_hat = grad_phi_hat0 + (self.As @ e + grad_zt + hess_phi @ zdot + h)*self.ts

        return za_dot, grad_phi_hat, z, zdot

