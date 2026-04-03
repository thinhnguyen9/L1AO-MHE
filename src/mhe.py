import numpy as np
from math import sin, cos   
import cvxpy as cp
import osqp
from scipy import sparse
import cvxopt
# from scipy.optimize import minimize, LinearConstraint
import copy
from src.utils import build_mhe_qp_multiple_shooting, build_mhe_qp_equality_constraints_lagrangian, build_mhe_qp_single_shooting
import time


class MHE():

    def __init__(
            self, model, ts, N, X0, P0, xs, us,
            mhe_type="linearized_once", mhe_update="filtering", prior_method="zero", mhe_shooting="single",
            xmin=None, xmax=None,
            solver="osqp", pcip_obj=None):
        """
        Args:
            model: dynamical model object
            ts: sampling time (for discretization)
            N: prediction horizon
            X0: mean of initial state (shape: (Nx,))
            P0: covariance of initial state (shape: (Nx, Nx))
            xs, us: linearization point
            mhe_type: "linearized_once" to linearize at xs, us,
                      "linearized_every" to linearize at each step
            mhe_update: "filtering" use x(T-N|T-N), i.e. do not override xvec
                        "smoothing" use x(T-N|T) and adjust arrival cost (Rawlings2017 chap 4.3.4)
                        "smoothing_naive" use x(T-N|T) but do not adjust arrival cost (like most papers)
            prior_method: "zero" to use zero prior weighting,
                          "ekf" to use the EKF covariance update,
                          "uniform" to use a fixed prior weighting P0
            mhe_shooting: "single" gives small & dense QP, z=[x0, w0...w(N-1)],
                          "multiple" gives large & sparse QP, z=[x0...xN, w0...w(N-1)]
            xmin, xmax: state constraints (shape: (Nx,)), optional
            solver: "cvxpy" to use CVXPY parser with OSQP,
                    "osqp" to use OSQP directly without parser (faster) (sparse QP),
                    "cvxopt" to use cvxopt directly (dense QP),
                    "pcip" to use PCIPQP solver (possibly with L1AO)
        """
        self.model = model
        self.Nx = model.Nx  # states
        self.Nu = model.Nu  # inputs
        self.Ny = model.Ny  # outputs
        self.ts = ts        # sampling time
        self.N = N      # prediction horizon
        
        self.xs = xs
        self.us = us
        if mhe_type not in ["linearized_once", "linearized_every"]:
            raise ValueError("mhe_type must be 'linearized_once' or 'linearized_every'.")
        if mhe_update not in ["filtering", "smoothing", "smoothing_naive"]:
            raise ValueError("mhe_update must be 'filtering', 'smoothing', or 'smoothing_naive'.")
        if prior_method not in ["zero", "ekf", "uniform"]:
            raise ValueError("prior_method must be 'zero', 'ekf', or 'uniform'.")
        if solver not in ["cvxpy", "osqp", "cvxopt", "pcip"]:
            raise ValueError("solver must be one of cvxpy/osqp/cvxopt/pcip.")
        if mhe_shooting not in ["single", "multiple"]:
            raise ValueError("mhe_shooting must be 'single' or 'multiple'.")
        self.mhe_type = mhe_type
        self.mhe_update = mhe_update
        self.prior_method = prior_method
        self.mhe_shooting = mhe_shooting
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

        if self.solver == "cvxopt":
            cvxopt.solvers.options['abstol'] = 1e-6
            cvxopt.solvers.options['reltol'] = 1e-6
            cvxopt.solvers.options['feastol'] = 1e-6
            cvxopt.solvers.options['maxiters'] = 100
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['refinement'] = 0

        if self.solver == "pcip":
            if pcip_obj is None:
                raise ValueError("Missing 'pcip_obj'.")
            self.pcip = pcip_obj
        
        self._solver_times = []

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
            
            # ------------------------ Single shooting ------------------------ #
            if self.mhe_shooting == "single":
                H, f, matA = build_mhe_qp_single_shooting(
                    A_seq, B_seq, G_seq, C_seq, Qinv_seq[:-1], Rinv_seq, X0, P0_inv, u, y,
                    smoothing_adjustment=(self.mhe_update=="smoothing"),
                    Q_seq=Q_seq, R_seq=R_seq
                )

                # State constraints
                if self.has_inequality_constraints:
                    # [dx0...dxN] = [x0...xN] - xnom = matA @ z
                    # [x0...xN] in [xmin, xmax]  <=>  matA @ z = [dx0...dxN] in [xmin, xmax] - xnom
                    A_ineq = matA.copy()
                    zmin = np.kron(np.ones((N+1,)), self.xmin) - xnom.flatten()
                    zmax = np.kron(np.ones((N+1,)), self.xmax) - xnom.flatten()     # zmin <= A_ineq @ z <= zmax
            
            # ------------------------ Multiple shooting ------------------------ #
            elif self.mhe_shooting == "multiple":
                H, f, A_eq, b_eq = build_mhe_qp_multiple_shooting(
                    A_seq, B_seq, G_seq, C_seq, Qinv_seq[:-1], Rinv_seq, X0, P0_inv, u, y,
                    smoothing_adjustment=(self.mhe_update=="smoothing"),
                    Q_seq=Q_seq, R_seq=R_seq
                )

                # State constraints
                if self.has_inequality_constraints:
                    A_ineq = np.hstack((
                        np.eye((N+1)*self.Nx),
                        np.zeros(((N+1)*self.Nx, N*self.Nx))
                    ))  # extract x(0)...x(N) from z
                    zmin = np.kron(np.ones((N+1,)), self.xmin) - xnom.flatten()
                    zmax = np.kron(np.ones((N+1,)), self.xmax) - xnom.flatten()     # zmin <= A_ineq @ z <= zmax
            
            # ------------------------ Initialize z0 ------------------------ #
            # T=0: initialize z=0 (applicable to both single & multiple shooting)
            if T == 0:
                # z0    = np.zeros((self.Nx,))
                z0 = X0
                if self.solver == "pcip":
                    zdot0 = np.zeros((self.Nx,))
                    if self.pcip.enable_l1ao:
                        l1ao_zdot0            = np.zeros((self.Nx,))
                        l1ao_sigma_hat0       = np.zeros((self.Nx,))
                        l1ao_nabla_z_phi_hat0 = np.zeros((self.Nx,))

            # horizon still growing
            elif T <= self.N:
                z0 = self.growing_horizon_extend_variables(self.z0)
                if self.solver == "pcip":
                    zdot0 = self.growing_horizon_extend_variables(self.pcip_zdot0)
                    if self.pcip.enable_l1ao:
                        l1ao_zdot0            = self.growing_horizon_extend_variables(self.l1ao_zdot0)
                        l1ao_sigma_hat0       = self.growing_horizon_extend_variables(self.l1ao_sigma_hat0)
                        l1ao_nabla_z_phi_hat0 = self.growing_horizon_extend_variables(self.l1ao_nabla_z_phi_hat0)
            
            # full horizon reached - size of z fixed
            else:
                z0 = self.z0
                if self.solver == "pcip":
                    zdot0 = self.pcip_zdot0
                    if self.pcip.enable_l1ao:
                        l1ao_zdot0            = self.l1ao_zdot0
                        l1ao_sigma_hat0       = self.l1ao_sigma_hat0
                        l1ao_nabla_z_phi_hat0 = self.l1ao_nabla_z_phi_hat0
            
            # ------------------------------------------------------- #
            # Solver time: clock starts at problem setup, ends after 
            #              solution is assigned to z.
            # ------------------------ CVXPY ------------------------ #
            if self.solver == "cvxpy":
                constraints = []
                if self.mhe_shooting == "single":
                    z = cp.Variable(((N+1)*self.Nx,))       # z = [x(0), w(0)...w(T-1)]
                elif self.mhe_shooting == "multiple":
                    z = cp.Variable(((2*N+1)*self.Nx,))     # z = [x(0)...x(T), w(0)...w(T-1)]
                    if N > 0:
                        constraints.append(A_eq @ z == b_eq)
                cost = 0.5 * cp.quad_form(z, cp.psd_wrap(H)) + f @ z
                if self.has_inequality_constraints:
                    constraints.append(np.vstack((-A_ineq, A_ineq)) @ z <= np.hstack((-zmin, zmax)))

                t0 = time.perf_counter()
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
                t1 = time.perf_counter()
            
            # ------------------------ OSQP ------------------------ #
            elif self.solver == "osqp":
                QP = {}
                QP['P'] = sparse.csc_matrix(H)
                QP['q'] = f
                if self.mhe_shooting == "single":
                    QP['A'] = sparse.csc_matrix(A_ineq) if self.has_inequality_constraints else None
                    QP['l'] = zmin if self.has_inequality_constraints else None
                    QP['u'] = zmax if self.has_inequality_constraints else None
                elif self.mhe_shooting == "multiple":
                    QP['A'] = sparse.csc_matrix(np.vstack((A_eq, A_ineq))) if self.has_inequality_constraints else sparse.csc_matrix(A_eq)
                    QP['l'] = np.hstack((b_eq, zmin)) if self.has_inequality_constraints else b_eq
                    QP['u'] = np.hstack((b_eq, zmax)) if self.has_inequality_constraints else b_eq

                t0 = time.perf_counter()
                prob = osqp.OSQP()
                prob.setup(
                    P=QP['P'], q=QP['q'], A=QP['A'], l=QP['l'], u=QP['u'],
                    eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polish=False, warm_start=True, verbose=False
                )
                if self.mhe_shooting == "single":       prob.warm_start(x=z0)
                elif self.mhe_shooting == "multiple":   prob.warm_start(x=z0[:(2*N+1)*self.Nx])  # exclude Lagrange multipliers
                res = prob.solve()
                if res.info.status != 'solved':
                    raise ValueError('OSQP did not solve the problem! Time step: ' + str(T))
                z = res.x
                t1 = time.perf_counter()

            # ------------------------ CVXOPT ------------------------ #
            elif self.solver == "cvxopt":
                QP = {
                    'P': cvxopt.matrix(H),
                    'q': cvxopt.matrix(f),
                    'G': cvxopt.matrix(np.vstack((-A_ineq, A_ineq))) if self.has_inequality_constraints else None,
                    'h': cvxopt.matrix(np.hstack((-zmin, zmax))) if self.has_inequality_constraints else None,
                    'A': cvxopt.matrix(A_eq) if self.mhe_shooting=="multiple" else None,
                    'b': cvxopt.matrix(b_eq) if self.mhe_shooting=="multiple" else None
                }
                t0 = time.perf_counter()
                sol = cvxopt.solvers.qp(P=QP['P'], q=QP['q'], G=QP['G'], h=QP['h'], A=QP['A'], b=QP['b'])
                # TODO: warm starting only 'x' is even slower. Try all variables.
                # initvals={'x': cvxopt.matrix(z0[:(2*N+1)*self.Nx])} if self.mhe_shooting=="multiple" else {'x': cvxopt.matrix(z0)}
                z = np.array(sol['x']).flatten()
                t1 = time.perf_counter()
            
            # ------------------------ PCIP ------------------------ #
            elif self.solver == "pcip":
                QP = {}
                if self.mhe_shooting == "single":
                    QP['P'] = H
                    QP['q'] = f
                    QP['A'] = None
                    QP['b'] = None
                    QP['G'] = np.vstack((-A_ineq, A_ineq)) if self.has_inequality_constraints else None
                    QP['h'] = np.hstack((-zmin, zmax)) if self.has_inequality_constraints else None
                elif self.mhe_shooting == "multiple":
                    QP['P'] = sparse.csc_matrix(H)
                    QP['q'] = f
                    QP['A'] = sparse.csc_matrix(A_eq)
                    QP['b'] = b_eq
                    QP['G'] = sparse.csc_matrix(np.vstack((-A_ineq, A_ineq))) if self.has_inequality_constraints else None
                    QP['h'] = np.hstack((-zmin, zmax)) if self.has_inequality_constraints else None

                t0 = time.perf_counter()
                self.pcip.set_QP(H=QP['P'], f=QP['q'], A=QP['A'], b=QP['b'], G=QP['G'], h=QP['h'], t=tvec[-1])
                zdot, z, l1ao_zdot, l1ao_sigma_hat, l1ao_nabla_z_phi_hat = self.pcip.dynamics(
                    z0                    = z0,
                    z0_unextended         = self.z0 if T>0 else z0,
                    zdot0                 = zdot0,
                    t                     = tvec[-1],
                    l1ao_zdot0            = l1ao_zdot0 if self.pcip.enable_l1ao else None,
                    l1ao_sigma_hat0       = l1ao_sigma_hat0 if self.pcip.enable_l1ao else None,
                    l1ao_nabla_z_phi_hat0 = l1ao_nabla_z_phi_hat0 if self.pcip.enable_l1ao else None
                )
                t1 = time.perf_counter()
                
                # Save for next time step: z(T+1), zdot(T)
                self.pcip_zdot0 = zdot
                self.l1ao_zdot0 = l1ao_zdot
                self.l1ao_sigma_hat0 = l1ao_sigma_hat
                self.l1ao_nabla_z_phi_hat0 = l1ao_nabla_z_phi_hat

            # ------------------------ Linear MHE result ------------------------ #
            self._solver_times.append(t1 - t0)
            self.z0 = z
            if self.mhe_shooting == "single":
                xvec = self.construct_X_from_X0(z[:self.Nx], A_seq, B_seq, G_seq,
                                                z[self.Nx:].reshape((N,self.Nx)), u)
            elif self.mhe_shooting == "multiple":
                xvec = z[:(N+1)*self.Nx].reshape((N+1, self.Nx))
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
    
    def growing_horizon_extend_variables(self, z):
        """
        MHE at time k=0...N experiences growing problem size.
            - Single shooting   : z(T) = [x(0), w(0)...w(T-1)] -> Extend z by simply repeating w(T-1)
            - Multiple shooting : z(T) = [x(0)...x(T), w(0)...w(T-1), l(0)...l(T-1)] -> Repeat x(T), w(T-1), l(T-1)
        """
        if self.mhe_shooting == "single":
            return np.hstack((z, z[-self.Nx:]))
        elif self.mhe_shooting == "multiple":
            N = int((z.shape[0]/self.Nx - 1) / 3)   # old horizon length
            return np.hstack((
                z[              0 : (N+1)*self.Nx  ],   # x(0)...x(N)
                z[      N*self.Nx : (N+1)*self.Nx  ],   # x(N)
                z[  (N+1)*self.Nx : (2*N+1)*self.Nx],   # w(0)...w(N-1)
                z[    2*N*self.Nx : (2*N+1)*self.Nx],   # w(N-1)
                z[(2*N+1)*self.Nx :                ],   # l(0)...l(N-1)
                z[       -self.Nx :                ]    # l(N-1)
            ))
    
    def construct_X_from_X0(self, x0, A_seq, B_seq, G_seq, w_seq, u_seq):
        N = len(A_seq)
        xvec = np.zeros((N+1, self.Nx))
        xvec[0] = x0
        for k in range(N):
            xvec[k+1] = A_seq[k] @ xvec[k] + B_seq[k] @ u_seq[k] + G_seq[k] @ w_seq[k]
        return xvec
    
    def get_mean_solver_time(self, start_idx=0):
        if len(self._solver_times) == 0:
            return 0.0
        else:
            return np.mean(self._solver_times[start_idx:])
    
    def get_solver_times(self, start_idx=0):
        return self._solver_times[start_idx:]