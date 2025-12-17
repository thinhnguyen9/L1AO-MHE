import numpy as np
from math import sin, cos   
import cvxpy as cp
import osqp
from scipy import sparse
from cvxopt import matrix, solvers
# from scipy.optimize import minimize, LinearConstraint
import copy
from src.utils import build_mhe_qp_with_dyn_constraints, build_mhe_qp_with_dyn_constraints_lagrangian, build_mhe_qp


class MHE():

    def __init__(
            self, model, ts, N, X0, P0, xs, us,
            mhe_type="linearized_once", mhe_update="filtering", prior_method="zero",
            xmin=None, xmax=None,
            solver="cvxpy", pcip_obj=None, l1ao_obj=None):
        """
        Args:
            model: dynamical model object
            ts: sampling time (for discretization)
            N: prediction horizon
            X0: mean of initial state (shape: (Nx,))
            P0: covariance of initial state (shape: (Nx, Nx))
            xs, us: linearization point
            mhe_type: "linearized_once" to linearize at xs, us,
                      "linearized_every" to linearize at each step,
                      "nonlinear" to use the nonlinear dynamics (very slow)
            mhe_update: "filtering" use x(T-N|T-N), i.e. do not override xvec
                        "smoothing" use x(T-N|T) and adjust arrival cost (Rawlings2017 chap 4.3.4)
                        "smoothing_naive" use x(T-N|T) but do not adjust arrival cost (like most papers)
            prior_method: "zero" to use zero prior weighting,
                          "ekf" to use the EKF covariance update,
                          "uniform" to use a fixed prior weighting P0
            xmin, xmax: state constraints (shape: (Nx,)), optional
            solver: "cvxpy" to use CVXPY parser with OSQP (default),
                    "osqp" to use OSQP directly without parser (faster) (sparse QP),
                    "cvxopt" to use cvxopt directly (dense QP),
                    "pcip" to use PCIPQP solver,
                    "pcip_l1ao" to use PCIPQP + L1AOQP
        """
        self.model = model
        self.Nx = model.Nx  # states
        self.Nu = model.Nu  # inputs
        self.Ny = model.Ny  # outputs
        self.ts = ts        # sampling time
        self.N = N      # prediction horizon
        
        self.xs = xs
        self.us = us
        if mhe_type not in ["linearized_once", "linearized_every", "nonlinear"]:
            raise ValueError("mhe_type must be 'linearized_once', 'linearized_every', or 'nonlinear'.")
        if mhe_update not in ["filtering", "smoothing", "smoothing_naive"]:
            raise ValueError("mhe_update must be 'filtering', 'smoothing', or 'smoothing_naive'.")
        if prior_method not in ["zero", "ekf", "uniform"]:
            raise ValueError("prior_method must be 'zero', 'ekf', or 'uniform'.")
        if solver not in ["cvxpy", "osqp", "cvxopt", "pcip", "pcip_l1ao"]:
            raise ValueError("solver must be one of cvxpy/osqp/cvxopt/pcip/pcip_l1ao.")
        self.mhe_type = mhe_type
        self.mhe_update = mhe_update
        self.prior_method = prior_method
        if xmin is not None and xmax is not None:
            self.has_inequality_constraints = True
            self.xmin, self.xmax = xmin, xmax
        else:
            self.has_inequality_constraints = False
        self.solver = solver
        A, B, G, C = self.model.linearize(xs, us)
        self.updateModel(A, B, G, C)
        
        self.xvec = np.zeros((1, self.Nx))              # estimates  x(T-N)...x(T) - len: N+1
                                                        # filtering scheme: x(T-N|T-N),...,x(T-1|T-1),x(T|T)
                                                        # smoothing scheme: x(T-N|T),...,x(T-1|T),x(T|T)
        self.Pvec = np.zeros((1, self.Nx, self.Nx))     # covariance P(k|k-1):  P(T-N)...P(T) - len: N+1
        self.Pvec1 = np.zeros((1, self.Nx, self.Nx))    # covariance P(k|k):    P(T-N)...P(T) - len: N+1
        self.xvec[0] = X0
        self.Pvec[0] = P0
        self.Pvec1[0] = P0
        self.P0 = P0
        self.X0 = X0

        if self.solver in ["pcip", "pcip_l1ao"]:
            if pcip_obj is None:
                raise ValueError("Missing 'pcip_obj'.")
            self.pcip = pcip_obj
        if self.solver == "pcip_l1ao":
            if l1ao_obj is None:
                raise ValueError("Missing 'l1ao_obj'.")
            self.l1ao = l1ao_obj

    def updateModel(self, A, B, G, C):
        # Discretize
        self.A = np.eye(self.Nx) + A*self.ts
        self.B = B*self.ts
        self.G = G*self.ts
        self.C = C

    def doEstimation(self, yvec, uvec, Qinv_seq, Rinv_seq, Q_seq, R_seq):
        """
        Run MHE to estimate state trajectory over the horizon.
        
        Args:
            yvec: sequence of outputs y(0)...y(T)
            uvec: sequence of inputs u(0)...u(T-1)
            Qinv_seq: Q(0)^{-1}...Q(T)^{-1}
            Rinv_seq: R(0)^{-1}...R(T)^{-1}
            Q_seq: Q(0)...Q(T) (used by smoothing scheme)
            R_seq: R(0)...R(T) (used by smoothing scheme)
        Returns:
            Estimated current state x(T)
        """
        # %% ========================================================================================================= #
        #                                               DEFINE HORIZON
        # ============================================================================================================ #
        T = np.size(yvec, 0) - 1
        N = min(self.N, T)
        if np.size(uvec, 0) != T:
            raise ValueError("yvec and uvec did not agree in size (yvec must have N+1 rows, uvec must have N rows)!")
        tvec = [(T-N+k)*self.ts for k in range(N+1)]    # T-N...T
        
        if T <= self.N:
            # Do Full Information Estimation (FIE) if T <= N
            # self.xvec  : x(0)...x(T-1)
            # self.Pvec  : P(0)...P(T|T-1)
            # self.Pvec1 : P(0)...P(T-1|T-1)
            if self.mhe_update == "filtering":
                X0 = self.X0    # self.xvec[0] was overriden, for filtering MHE we need to use fixed X0
            elif self.mhe_update in ["smoothing", "smoothing_naive"]:
                X0 = self.xvec[0]
            P0 = self.P0
            # X0 = self.xvec[0]   # x(0)
            # if self.prior_method == "zero":         pass
            # elif self.prior_method == "uniform":    P0 = self.P0
            # elif self.prior_method == "ekf":        P0 = self.Pvec[0]   # P(0)
            yseq_raw = yvec    # y(0)...y(T)
            useq_raw = uvec    # u(0)...u(T-1)
        
        else:
            # self.xvec  : x(T-N-1)...x(T-1)
            # self.Pvec  : P(T-N|T-N-1)...P(T|T-1)
            # self.Pvec1 : P(T-N-1|T-1)...P(T-1|T-1)
            X0 = self.xvec[1]   # x(T-N)
            if self.prior_method == "zero":         pass
            elif self.prior_method == "uniform":    P0 = self.P0
            elif self.prior_method == "ekf":        P0 = self.Pvec[0]   # P(T-N|T-N-1)
            yseq_raw = yvec[-self.N-1 :]   # y(T-N)...y(T)
            useq_raw = uvec[-self.N :]     # u(T-N)...u(T-1)
            Qinv_seq = Qinv_seq[-self.N-1 :] # (T-N)...(T)
            Rinv_seq = Rinv_seq[-self.N-1 :] # (T-N)...(T)
            Q_seq = Q_seq[-self.N-1 :]  # (T-N)...(T)
            R_seq = R_seq[-self.N-1 :]  # (T-N)...(T)

        # %% ========================================================================================================= #
        #                                               LINEARIZATION
        # ============================================================================================================ #
        if self.mhe_type == "linearized_once":
            X0 = X0 - self.xs
            y = yseq_raw - self.C @ self.xs
            u = useq_raw - self.us

        elif self.mhe_type == "linearized_every":
            if self.mhe_update == "filtering":
                # Use nonlinear model to get nominal trajectory
                xnom = np.zeros((N+1, self.Nx))     # x(T-N)...x(T)
                xnom[0] = X0
                for k in range(N):
                    xnom[k+1] = xnom[k] + self.model.dx(xnom[k], useq_raw[k], t=tvec[k])*self.ts

            elif self.mhe_update in ["smoothing", "smoothing_naive"]:
                # Use nonlinear model to get nominal trajectory
                # xnom = np.zeros((N+1, self.Nx))     # x(T-N)...x(T)
                # xnom[0] = X0
                # for k in range(N):
                #     xnom[k+1] = xnom[k] + self.model.dx(xnom[k], useq_raw[k], t=tvec[k])*self.ts
                    
                # # Use self.xvec as nominal trajectory
                if T == 0:
                    # self.xvec  : x(0)prior
                    xnom = self.xvec
                else:
                    # self.xvec  : x(0)...x(T-1), OR
                    # self.xvec  : x(T-N-1)...x(T-1)
                    xTnom = self.xvec[-1] + self.model.dx(self.xvec[-1], useq_raw[-1], t=tvec[-2])*self.ts
                    xnom = np.concatenate((self.xvec[-self.N:], xTnom.reshape((1,self.Nx))), axis=0)

            X0 = np.zeros(self.Nx)
            y = yseq_raw - xnom @ self.C.T
            u = np.zeros((N, self.Nu))

        elif self.mhe_type == "nonlinear":
            # X0 = X0
            y = yseq_raw
            u = useq_raw

        # %% ========================================================================================================= #
        #           Backward interation to find P(T-1|T-1)...P(T-N|T-1) for smoothing scheme
        #           RAUCH, TUNG and STRIEBEL, 1965
        # ============================================================================================================ #
        if self.mhe_update == "smoothing" and N > 1:
            # self.xvec  : x(T-N-1)...x(T-1)            (len: N+1)
            # self.Pvec  : P(T-N|T-N-1)...P(T|T-1)      (len: N+1)
            # self.Pvec1 : P(T-N-1|T-1)...P(T-1|T-1)    (len: N)
            # useq_raw   : u(T-N)...u(T-1)              (len: N)

            # given P(T-1|T-1), iterate from P(T-2|T-1) till P(T-N|T-1) (N-1 steps)
            P_temp = self.Pvec1[-1]    # P(T-1|T-1)
            for i in range(N-1):    # k=T-2,...,T-N
                A, _, _, _ = self.model.linearize(self.xvec[-i-2], useq_raw[-i-2], t=tvec[-i-3])  # A(T-2)
                A = np.eye(self.Nx) + A*self.ts

                # C(k) = P(k|k) * A'(k) * inv(P(k+1|k))
                try:    C = self.Pvec1[-i-2] @ A.T @ np.linalg.inv(self.Pvec[-i-2]) # C(T-2) (NOT output matrix)
                except: C = self.Pvec1[-i-2] @ A.T @ np.linalg.pinv(self.Pvec[-i-2])

                # P(k|T-1) = P(k|k) + C(k)(P(k+1|T-1) - P(k+1|k))C'(k)
                # start: k=T-2, end: k=T-N
                P_temp = self.Pvec1[-i-2] + C @ (P_temp - self.Pvec[-i-2]) @ C.T    # P(T-2|T-1)
            P0 = P_temp

        # %% ========================================================================================================= #
        #                                       OPTIMIZATION - LINEAR MHE (CVXPY/PCIP)
        # ============================================================================================================ #
        if self.mhe_type in ["linearized_once", "linearized_every"]:

            # Time-varying model
            A_seq = np.zeros((N, self.Nx, self.Nx))
            B_seq = np.zeros((N, self.Nx, self.Nu))
            G_seq = np.zeros((N, self.Nx, self.Nx))
            C_seq = np.zeros((N+1, self.Ny, self.Nx))
            for k in range(N):
                if self.mhe_type == "linearized_every":   # Linearize around nominal trajectory
                    A, B, G, C = self.model.linearize(xnom[k], useq_raw[k], t=tvec[k])
                    self.updateModel(A, B, G, C)
                A_seq[k], B_seq[k], G_seq[k], C_seq[k] = self.A, self.B, self.G, self.C
            C_seq[N] = self.C    # TODO: relinearize??
            
            # Build QP
            if self.prior_method=="zero":
                P0_inv = np.zeros((self.Nx, self.Nx))
            else:
                P0_inv = np.linalg.inv(P0)
            H, f, matA = build_mhe_qp(
                A_seq, B_seq, G_seq, C_seq, Qinv_seq[:-1], Rinv_seq, X0, P0_inv, u, y,
                smoothing_adjustment=(self.mhe_update=="smoothing"),
                Q_seq=Q_seq, R_seq=R_seq
            )

            # State constraints
            if self.has_inequality_constraints:
                # [dx0...dxN] = [x0...xN] - xnom = matA @ z
                # [x0...xN] in [xmin, xmax]  <=>  matA @ z = [dx0...dxN] in [xmin, xmax] - xnom
                zmin = np.kron(np.ones((N+1,)), self.xmin) - xnom.flatten()
                zmax = np.kron(np.ones((N+1,)), self.xmax) - xnom.flatten()     # zmin <= matA @ z <= zmax
            
            if self.solver == "cvxpy":
                """ # Dynamics as equality constraints
                H, f, A_eq, b_eq = build_mhe_qp_with_dyn_constraints(A_seq, B_seq, G_seq, C_seq, self.Q_inv, self.R_inv,
                                                                     X0, P0_inv, u, y)

                # Variable z = [x0...xN, w0...w(N-1)]
                z = cp.Variable(((2*N+1)*self.Nx,))
                constraints = []
                cost = 0.5 * cp.quad_form(z, cp.psd_wrap(H)) + f @ z
                if N>0: constraints.append(A_eq @ z == b_eq)
                """
                # Variable z = [x0, w0...w(N-1)]
                z = cp.Variable(((N+1)*self.Nx,))
                constraints = []
                # constraints.append(z[0] >= 0.)
                # constraints.append(z[1] >= 0.)
                # constraints.append(z[2] <= 0.)
                # constraints.append(z[8] >= 0.)
                # constraints.append(z[8] >= -np.pi/36)
                # constraints.append(z[8] <= np.pi/36)
                cost = 0.5 * cp.quad_form(z, cp.psd_wrap(H)) + f @ z

                prob = cp.Problem(cp.Minimize(cost), constraints)
                # prob.solve(solver=cp.OSQP, warm_start=True)
                # prob.solve(solver=cp.ECOS, feastol=1e-04, reltol=1e-6, abstol=1e-3, verbose=True)
                # prob.solve(solver=cp.CLARABEL)
                try:
                    # prob.solve()
                    prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=100)
                except:
                    prob.solve(solver=cp.ECOS, feastol=1e-03, reltol=1e-3, abstol=1e-3, verbose=True)

                # Result
                # xvec = z.value[0:(N+1)*self.Nx].reshape((N+1, self.Nx))
                z = z.value
            
            elif self.solver == "osqp": # TODO: same tolerances for all solvers?
                prob = osqp.OSQP()
                prob.setup(
                    P=sparse.csc_matrix(H),
                    q=f,
                    A=sparse.csc_matrix(matA) if self.has_inequality_constraints else None,
                    l=zmin if self.has_inequality_constraints else None,
                    u=zmax if self.has_inequality_constraints else None,
                    warm_start=True, verbose=False
                )
                res = prob.solve()
                if res.info.status != 'solved':
                    raise ValueError('OSQP did not solve the problem! Time step: ' + str(T))
                z = res.x

            elif self.solver == "cvxopt":
                solvers.options['show_progress'] = False
                sol = solvers.qp(
                    P=matrix(H), q=matrix(f),
                    G=matrix(np.vstack((-matA, matA))) if self.has_inequality_constraints else None,
                    h=matrix(np.hstack((-zmin, zmax))) if self.has_inequality_constraints else None
                )
                z = np.array(sol['x']).flatten()
            
            elif self.solver in ["pcip", "pcip_l1ao"]: # only dynamics constraints!!
                """ # Dynamics as equality constraints
                # Lagrange multiplier v, length: N*Nx
                # z = [x(0),..., x(N), w(0),..., w(N-1), v], length: (3N+1)*Nx
                H, f, A_eq, b_eq = build_mhe_qp_with_dyn_constraints(A_seq, B_seq, G_seq, C_seq, self.Q_inv, self.R_inv,
                                                                     X0, P0_inv, u, y)
                H, f = build_mhe_qp_with_dyn_constraints_lagrangian(H, f, A_eq, b_eq)
                
                # Initialize z0
                if not hasattr(self, 'pcip_z0'):    # T=0: initialize z=0
                    z0 = np.zeros((self.Nx,))
                elif self.pcip_z0.shape[0] < (3*N+1)*self.Nx: # horizon still growing
                    # z0 = np.zeros(((3*N+1)*self.Nx,))
                    z0 = np.hstack((self.pcip_z0[ : N*self.Nx],                      # x(0)...x(N-1)
                                    self.pcip_z0[(N-1)*self.Nx : N*self.Nx],         # x(N-1)
                                    self.pcip_z0[N*self.Nx : (2*N-1)*self.Nx],       # w(0)...w(N-2)
                                    self.pcip_z0[(2*N-2)*self.Nx : (2*N-1)*self.Nx], # w(N-2)
                                    self.pcip_z0[(2*N-1)*self.Nx : ],                # v(0)...v(N-2)
                                    self.pcip_z0[-self.Nx : ]))                      # v(N-2)
                else:   # full horizon reached - size of z fixed
                    z0 = self.pcip_z0

                # solve QP with PCIP
                self.pcip.set_QP(H, f)
                _, z_hat = self.pcip.dynamics(z0, H, f)
                self.pcip_z0 = z_hat
                xvec = z_hat[:(N+1)*self.Nx].reshape((N+1, self.Nx)) + xnom
                """
                # z = [x(0), w(0), ..., w(N-1)]
                # Initialize z0
                if not hasattr(self, 'tvopt_z0'):    # T=0: initialize z=0
                    # z0    = np.zeros((self.Nx,))
                    z0 = X0
                    zdot0 = np.zeros((self.Nx,))
                    if self.solver == "pcip_l1ao":
                        za_dot0       = np.zeros((self.Nx,))
                        grad_phi_hat0 = np.zeros((self.Nx,))

                elif self.tvopt_z0.shape[0] < (N+1)*self.Nx: # horizon still growing
                    z0    = np.hstack((self.tvopt_z0, self.tvopt_z0[-self.Nx : ]))
                    zdot0 = np.hstack((self.tvopt_zdot0, self.tvopt_zdot0[-self.Nx : ]))
                    if self.solver == "pcip_l1ao":
                        za_dot0       = np.hstack((self.l1ao_za_dot0, self.l1ao_za_dot0[-self.Nx : ]))
                        grad_phi_hat0 = np.hstack((self.l1ao_grad_phi_hat0, self.l1ao_grad_phi_hat0[-self.Nx : ]))

                else:   # full horizon reached - size of z fixed
                    z0    = self.tvopt_z0
                    zdot0 = self.tvopt_zdot0
                    if self.solver == "pcip_l1ao":
                        za_dot0       = self.l1ao_za_dot0
                        grad_phi_hat0 = self.l1ao_grad_phi_hat0

                # Solve QP with PCIP / PCIP+L1AO
                self.pcip.set_QP(
                    H=H,
                    f=f,
                    G=np.vstack((-matA, matA)) if self.has_inequality_constraints else None,
                    h=np.hstack((-zmin, zmax)) if self.has_inequality_constraints else None,
                    t=tvec[-1]
                )
                zb_dot, zb = self.pcip.dynamics(z0, tvec[-1])

                if self.solver == "pcip_l1ao":
                    self.l1ao.set_QP(
                        H=H,
                        f=f,
                        G=np.vstack((-matA, matA)) if self.has_inequality_constraints else None,
                        h=np.hstack((-zmin, zmax)) if self.has_inequality_constraints else None,
                        t=tvec[-1]
                    )
                    za_dot, grad_phi_hat, z, zdot = self.l1ao.dynamics(z0, zdot0, za_dot0, grad_phi_hat0, zb_dot, tvec[-1])

                    # Save for next time step (only L1AO): za_dot(T), grad_phi_hat(T+1)
                    self.l1ao_za_dot0 = za_dot
                    self.l1ao_grad_phi_hat0 = grad_phi_hat
                else:
                    z, zdot = zb, zb_dot
                
                # Save for next time step: z(T+1), zdot(T)
                self.tvopt_z0 = z
                self.tvopt_zdot0 = zdot
            
            # Result for linear MHE
            # if z is None:   # CVXPY failed
            #     xvec = xnom.copy()
            # else:
            xvec = self.construct_X_from_X0(z[:self.Nx], A_seq, B_seq, G_seq,
                                            z[self.Nx:].reshape((N,self.Nx)), u)
            if self.mhe_type == "linearized_once":
                xvec = xvec + self.xs   # x(T-N)...x(T)
            elif self.mhe_type == "linearized_every":
                xvec = xvec + xnom      # x(T-N)...x(T)

        # %% ========================================================================================================= #
        #                                   OPTIMIZATION - NONLINEAR MHE (scipy.optimize)
        # ============================================================================================================ #
        # Too many changes - need to update nonlinear MHE!
        # elif self.mhe_type in ["nonlinear"]:
        #     def cost_fun(z):    # for nonlinear MHE using scipy.optimize.minimze
        #         x0 = z[ : self.Nx]
        #         w = z[self.Nx : ].reshape((N, self.Nx))

        #         # Arrival cost - adjusted for smoothing scheme
        #         if self.prior_method == "zero":
        #             cost = 0.0
        #         else:
        #             cost = .5 * (x0 - X0).T @ np.linalg.inv(P0) @ (x0 - X0)
        #         """
        #         if self.mhe_update == "smoothing" and N > 1:
        #             # a_random_matrix = np.zeros((self.Ny*N, self.Nu*N))
        #             # for r in range(N):
        #             #     for c in range(r):
        #             #         a_random_matrix[r*self.Ny:(r+1)*self.Ny, c*self.Nx:(c+1)*self.Nx] = self.C @ np.linalg.matrix_power(self.A, r-c-1) @ self.B
        #             # uflat = u.flatten()         # u(T-N)...u(T-1)
        #             # yflat = y[:-1].flatten()    # y(T-N)...y(T-1)
        #             # temp = yflat - O@x0 - a_random_matrix@uflat
        #             # cost -= .5 * temp.T @ W_inv @ temp

        #             yflat = y[:-1].flatten()    # y(T-N)...y(T-1)
        #             temp = yflat - O@x0
        #             cost -= .5 * temp.T @ W_inv @ temp
        #         """
        #         # Running cost
        #         for k in range(N):
        #             y_pred = self.model.getOutput(x0)
        #             cost += .5*w[k].T @ self.Q_inv @ w[k] + .5*(y[k] - y_pred).T @ self.R_inv @ (y[k] - y_pred)
        #             x0 = x0 + self.model.dx(x0, u[k], w[k]) * self.ts   # x(k+1)
        #         y_pred = self.model.getOutput(x0)
        #         cost += .5*(y[N] - y_pred).T @ self.R_inv @ (y[N] - y_pred)
        #         return cost
            
        #     state_constraint = []
        #     # A = np.zeros([2, self.Nx + N*self.Nx])
        #     # A[0,0] = 1.
        #     # A[1,1] = 1.
        #     # state_constraint = LinearConstraint(A, 0., np.inf)
        
        #     w_init = np.zeros(self.Nx*N)
        #     z_init = np.concatenate([X0, w_init])
        #     res = minimize(cost_fun, z_init, constraints=state_constraint, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
        #     x0 = res.x[ : self.Nx]
        #     w = res.x[self.Nx : ].reshape((N, self.Nx))
        #     # Rescontruct trajectory
        #     xvec = np.zeros((N+1, self.Nx))
        #     xvec[0] = x0
        #     for k in range(N):
        #         xvec[k+1] = xvec[k] + self.model.dx(xvec[k], u[k], w[k]) * self.ts
        #     # print("Done 1 loop of nonlinear MHE!")

        # %% ========================================================================================================= #
        #                                               UPDATE self.xvec
        # ============================================================================================================ #
        # self.xvec:    x(T-N-1)...x(T-1)
        #      xvec:    x(T-N)...x(T)
        if self.mhe_update == "filtering":
            if T > 0:
                # Only save the latest estimate x(T|T)
                self.xvec = np.concatenate((self.xvec, xvec[-1].reshape(1, self.Nx)), axis=0)
                self.xvec = self.xvec[-self.N-1:]
            else:
                self.xvec = xvec    # only 1 value, override initial guess X0
        elif self.mhe_update in ["smoothing", "smoothing_naive"]:    # always override even at T=0,1 - trust me bro
            # Save the entire horizon of latest estimate x(T-N|T)...x(T|T)
            self.xvec = xvec

        # %% ========================================================================================================= #
        #                                               UPDATE COVARIANCE
        # ============================================================================================================ #
        if self.prior_method == "ekf":

            # Calculate P(T|T) from P(T|T-1)
            P0 = self.Pvec[-1]  # P(T|T-1)
            if self.mhe_type in ["linearized_every", "nonlinear"]:  # Linearize around xhat(T|T)
                A, B, G, C = self.model.linearize(self.xvec[-1], useq_raw[-1] if N>0 else self.us, t=tvec[-1]) # TODO: need correct u?
                self.updateModel(A, B, G, C)
            # R_k = np.linalg.inv(Rinv_seq[-1])
            # Q_k = np.linalg.inv(Qinv_seq[-1])
            L = P0 @ self.C.T @ np.linalg.inv(R_seq[-1] + self.C @ P0 @ self.C.T)
            P = P0 - L @ self.C @ P0    # P(T|T)
            if T > 0:
                self.Pvec1 = np.concatenate((self.Pvec1, P.reshape((1, self.Nx, self.Nx))), axis=0)
                self.Pvec1 = self.Pvec1[-self.N-1:]
            else:
                self.Pvec1[0] = P

            # Calculate P(T+1|T) from P(T|T)
            P = self.G @ Q_seq[-1] @ self.G.T + self.A @ P @ self.A.T  # P(T+1|T)
            self.Pvec = np.concatenate((self.Pvec, P.reshape((1, self.Nx, self.Nx))), axis=0)
            self.Pvec = self.Pvec[-self.N-1:]

        # %% ========================================================================================================= #
        #                                                    DONE
        # ============================================================================================================ #
        # self.uvec = u
        # return self.xvec
        return self.xvec[-1]      # x(T)
    
    def construct_X_from_X0(self, x0, A_seq, B_seq, G_seq, w_seq, u_seq):
        N = len(A_seq)
        xvec = np.zeros((N+1, self.Nx))
        xvec[0] = x0
        for k in range(N):
            xvec[k+1] = A_seq[k] @ xvec[k] + B_seq[k] @ u_seq[k] + G_seq[k] @ w_seq[k]
        return xvec
