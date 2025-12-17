import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import pathlib
from math import sin, cos
import cvxpy as cp
import copy
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src.controllers import LQR
# from src.estimators import KF, MHE
from models.quadrotors import Quadrotor1



class Simulator():

    def __init__(
            self,
            mode,
            sys,
            w_means,
            w_stds,
            v_means,
            v_stds,
            x0_stds,
            T=5.0,
            ts=0.01,
            noise_distribution="gaussian",
            time_varying_measurement_noise=False,
            use_QR_guess=False,
            Q_guess=None,
            R_guess=None
        ):
        if mode not in ['quadrotor', 'reactor']:
            raise ValueError("Invalid mode. Supported modes: 'quadrotor', 'reactor'.")
        if noise_distribution not in ['gaussian', 'uniform']:
            raise ValueError("Invalid noise distribution. Supported distributions: 'gaussian', 'uniform'.")
        self.mode   = mode
        self.sys    = sys
        self.T      = T
        self.ts     = ts
        self.tvec   = np.arange(0.0, T+ts, ts)
        self.N      = len(self.tvec)
        self.Nx     = sys.Nx
        self.Nu     = sys.Nu
        self.Ny     = sys.Ny
        self.w_means    = w_means
        self.w_stds     = w_stds
        self.v_means    = v_means
        self.v_stds     = v_stds
        self.x0_stds    = x0_stds
        self.noise_distribution = noise_distribution
        self.time_varying_measurement_noise = time_varying_measurement_noise
        self.use_QR_guess   = use_QR_guess

        # Set equilibrium point for linearization
        if self.mode == 'quadrotor':
            if type(sys).__name__ == 'Quadrotor1':
                xhover = np.zeros(sys.Nx)
                xhover[12:16] = sys.f_h
                uhover = np.array([sys.u_h]*sys.Nu)
            elif type(sys).__name__ == 'Quadrotor2':
                xhover = np.zeros(sys.Nx)
                uhover = np.array([sys.m*sys.g, 0, 0, 0])
            self.x_eq = xhover
            self.u_eq = uhover
        else:
            self.x_eq = np.zeros(self.Nx)
            self.u_eq = np.zeros(self.Nu)
        
        # Time-varying noise covariance
        self.w_stds_fixed = self.w_stds.copy()
        self.v_stds_fixed = self.v_stds.copy()
        if self.time_varying_measurement_noise:
            # w_omega = np.pi / 2.
            v_omega = 15.
            epsilon = 1e-3  # prevent infinite Q^{-1}, R^{-1}
            # w_scale = np.sin(w_omega*self.tvec.reshape(self.N,1) + np.pi/4)**2 + epsilon
            v_scale = np.sin(v_omega*self.tvec.reshape(self.N,1))**2 + epsilon
        else:
            # w_scale = np.ones((self.N, 1))
            v_scale = np.ones((self.N, 1))
        w_scale = np.ones((self.N, 1))
        self.w_stds = np.kron(w_scale, self.w_stds)
        self.v_stds = np.kron(v_scale, self.v_stds)

        # For estimator ONLY
        if self.use_QR_guess:   # use FIXED guesses of Q, R matrices
            if Q_guess is None or R_guess is None:
                raise ValueError("Q_guess and R_guess must be provided when use_QR_guess=True.")
            temp = np.ones((self.N, 1, 1))
            self.Q     = np.kron(temp, Q_guess)
            self.R     = np.kron(temp, R_guess)
            self.Q_inv = np.kron(temp, np.linalg.inv(Q_guess))
            self.R_inv = np.kron(temp, np.linalg.inv(R_guess))
        else:   # use true values of Q (fixed) and R (time-varying)
            self.Q     = np.zeros((self.N, self.Nx, self.Nx))
            self.R     = np.zeros((self.N, self.Ny, self.Ny))
            self.Q_inv = np.zeros((self.N, self.Nx, self.Nx))
            self.R_inv = np.zeros((self.N, self.Ny, self.Ny))
            for k in range(self.N):
                self.Q[k]     = np.diag(self.w_stds[k]**2)
                self.R[k]     = np.diag(self.v_stds[k]**2)
                self.Q_inv[k] = np.diag(1./(self.w_stds[k]**2))
                self.R_inv[k] = np.diag(1./(self.v_stds[k]**2))


    @staticmethod
    def saturate(val, lower_bound, upper_bound):
        return max(lower_bound, min(upper_bound, val))


    def simulate_quadrotor_lqr_control(
            self,   # for reproducible noise generation
            seed,
            traj_mode="hover",
            x0=None,
            xref=None,
            zero_disturbance=False,
            zero_noise=False,
            measurement_delay=0
        ):
        """
        traj_mode: "hover" (stay still at x0),
                   "p2p" (point-to-point from x0 to xref),
                   "circle" (circular trajectory),
                   "triangle" (aggressive with abrubt turns)
        """
        rng = np.random.default_rng(seed)
        # ----------------------- Initial & ref states -----------------------
        if self.mode != 'quadrotor':
            return
        else:
            if traj_mode == "hover":
                if type(self.sys).__name__ == 'Quadrotor2':
                    if x0 is None:
                        x0 = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                    if xref is None:
                        xref = x0.copy()
            elif traj_mode == "p2p":
                if type(self.sys).__name__ == 'Quadrotor1':
                    if x0 is None:
                        x0 = np.array([ 0., 0., 0., 0., -1., 0.,
                                        0., 0., 0.,
                                        0., 0., 0.,
                                        self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
                    if xref is None:
                        xref = np.array([ .25, 0., .5, 0., -1., 0.,
                                        0., 0., 0.,
                                        0., 0., 0.,
                                        self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
                elif type(self.sys).__name__ == 'Quadrotor2':
                    if x0 is None:
                        x0 = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                    if xref is None:
                        xref = np.array([1., -2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            elif traj_mode == "circle":
                if type(self.sys).__name__ == 'Quadrotor1':
                    radius = 0.5    # m
                    omega = 1.75    # rad/s
                    vz = -.1        # m/s
                    z0 = -1.
                    x0 = np.array([ radius, 0., 0., 0., z0, 0.,
                                    0., 0., 0.,
                                    0., 0., 0.,
                                    self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
                    xref = np.array([ 0., 0., 0., 0., z0, 0.,
                                    0., 0., 0.,
                                    0., 0., 0.,
                                    self.sys.f_h, self.sys.f_h, self.sys.f_h, self.sys.f_h ])
                elif type(self.sys).__name__ == 'Quadrotor2':
                    radius = 0.3    # m
                    omega = 5.    # rad/s
                    # vz = .1        # m/s
                    z0 = 1.
                    dz = .1
                    x0 = np.array([radius, 0., z0, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                    xref = np.array([0., 0., z0, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            elif traj_mode == "triangle":
                if type(self.sys).__name__ == 'Quadrotor1':
                    pass
                elif type(self.sys).__name__ == 'Quadrotor2':
                    a = .75  # triangle side length
                    z0 = 1.
                    time_per_side = 1.  # second
                    vel = a / time_per_side
                    waypoints = np.array([
                        [0., .5*(3**.5)*a, z0,  .5*vel, -.5*(3**.5)*vel, 0.],
                        [.5*a, 0., z0,          -vel, 0., 0.],
                        [-.5*a, 0., z0,         .5*vel, .5*(3**.5)*vel, 0.]
                    ])
                    x0 = np.array([0., .5*(3**.5)*a, z0, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                    xref = np.zeros(12)

        # ----------------------- Controller -----------------------
        StateFeedback = LQR(
            type = 'continuous',
            n = self.Nx,
            m = self.Nu
        )
        """
        Tune these:
            Q: 1/(maxError^2)
            R: 1/(umax^2)
        """
        if self.mode == 'quadrotor':
            if type(self.sys).__name__ == 'Quadrotor1':
                Q = np.diag([1/(0.05**2), 1/(0.2**2),               # x, xdot
                            1/(0.05**2), 1/(0.2**2),               # y, ydot
                            1/(0.05**2), 1/(0.2**2),               # z, zdot
                            1/(0.1**2), 1/(0.1**2), 1/(0.01**2),   # roll, pitch, yaw
                            1/(0.5**2), 1/(0.5**2), 1/(0.5**2),    # angular rates
                            0, 0, 0, 0 ])                          # motor thrusts - allow large errors
                R = np.diag([1/(self.sys.umax**2)]*4)
            elif type(self.sys).__name__ == 'Quadrotor2':
                Q = np.diag([1/(0.005**2), 1/(0.005**2), 1/(0.005**2), # x, y, z
                             1/(0.2**2), 1/(0.2**2), 1/(0.2**2),    # xd, yd, zd
                             1/(0.2**2), 1/(0.2**2), 1/(0.01**2),    # roll, pitch, yaw
                             1/(0.5**2), 1/(0.5**2), 1/(0.5**2)])   # angular rates
                R = np.diag([1e0, 1e1, 1e1, 1e1])   # cheap control for more aggressive trajectories
        else:
            Q = np.eye(self.Nx)
            R = np.eye(self.Nu)
        StateFeedback.setWeight(Q, R)

        # Linearize around hover position
        A, B, G, C = self.sys.linearize(self.x_eq, self.u_eq)
        StateFeedback.setModel(A, B)
        StateFeedback.calculateGain()
        dmax = .2   # bound on x,y,z errors for stability

        # ----------------------- Simulation -----------------------
        xvec = np.zeros((self.N, self.Nx))
        yvec = np.zeros((self.N, self.Ny))
        uvec = np.zeros((self.N, self.Nu))

        # Process noise (disturbance) - sinusoidal, low-frequency
        if zero_disturbance:
            wvec = np.zeros((self.N, self.Nx))
        else:
            # if self.noise_distribution=="gaussian":
            #     # wvec = np.random.normal(loc=self.w_means, scale=self.w_stds)   # size=(self.N, self.Nx)
            #     wvec = rng.normal(loc=self.w_means, scale=self.w_stds)
            # elif self.noise_distribution=="uniform":
            #     # wvec = np.random.uniform(low=-self.w_stds*3, high=self.w_stds*3)   # size=(self.N, self.Nx)
            #     wvec = rng.uniform(low=-self.w_stds*3, high=self.w_stds*3)
            if type(self.sys).__name__ == 'Quadrotor1':
                w_omega = np.array([ 0., np.pi, 0., 1.7*np.pi, 0., 2.4*np.pi,     # external forces acting on Xdd, Ydd, Zdd
                                    0., 0., 0.,
                                    1.5*np.pi, .9*np.pi, .4*np.pi,  # external moments acting on pd, qd, rd
                                    0., 0., 0., 0. ])
                w_phase = np.array([ 0., np.pi/4, 0., np.pi/3, 0., np.pi/5,
                                    0., 0., 0.,
                                    0., np.pi/8, np.pi/7,
                                    0., 0., 0., 0. ])
            elif type(self.sys).__name__ == 'Quadrotor2':
                # w_omega = np.array([ 0., 0., 0., np.pi, 1.7*np.pi, 2.4*np.pi,     # external forces acting on Xdd, Ydd, Zdd
                #                      0., 0., 0., 1.5*np.pi, .9*np.pi, .4*np.pi])  # external moments acting on pd, qd, rd
                # w_phase = np.array([ 0., 0., 0., np.pi/4, np.pi/3, np.pi/5,
                #                      0., 0., 0., 0., np.pi/8, np.pi/7])
                rand_omega1 = rng.uniform(low=.5*np.pi, high=3.*np.pi, size=(6,))
                rand_omega2 = rng.uniform(low=.5*np.pi, high=3.*np.pi, size=(6,))
                rand_omega3 = rng.uniform(low=.5*np.pi, high=3.*np.pi, size=(6,))
                rand_phase1 = rng.uniform(low=0.,       high=2.*np.pi, size=(6,))
                rand_phase2 = rng.uniform(low=0.,       high=2.*np.pi, size=(6,))
                rand_phase3 = rng.uniform(low=0.,       high=2.*np.pi, size=(6,))
                w_omega1 = np.zeros((12,))
                w_omega2 = np.zeros((12,))
                w_omega3 = np.zeros((12,))
                w_phase1 = np.zeros((12,))
                w_phase2 = np.zeros((12,))
                w_phase3 = np.zeros((12,))
                w_omega1[3:6],  w_omega2[3:6],  w_omega3[3:6]  = rand_omega1[0:3], rand_omega2[0:3], rand_omega3[0:3]
                w_omega1[9:12], w_omega2[9:12], w_omega3[9:12] = rand_omega1[3:6], rand_omega2[3:6], rand_omega3[3:6]
                w_phase1[3:6],  w_phase2[3:6],  w_phase3[3:6]  = rand_phase1[0:3], rand_phase2[0:3], rand_phase3[0:3]
                w_phase1[9:12], w_phase2[9:12], w_phase3[9:12] = rand_phase1[3:6], rand_phase2[3:6], rand_phase3[3:6]
            wvec = self.w_means + self.w_stds_fixed/3*(
                      np.sin(w_omega1*self.tvec.reshape(self.N,1) + w_phase1)
                    + np.sin(w_omega2*self.tvec.reshape(self.N,1) + w_phase2)
                    + np.sin(w_omega3*self.tvec.reshape(self.N,1) + w_phase3)
                )
        
        # Measurement noise - Gaussian/uniform, high-frequency
        if zero_noise:
            vvec = np.zeros((self.N, self.Ny))
        else:
            if self.noise_distribution=="gaussian":
                # vvec = np.random.normal(loc=self.v_means, scale=self.v_stds)   # size=(self.N, self.Ny)
                vvec = rng.normal(loc=self.v_means, scale=self.v_stds)
            elif self.noise_distribution=="uniform":
                # vvec = np.random.uniform(low=-self.v_stds*3, high=self.v_stds*3)   # size=(self.N, self.Ny)
                vvec = rng.uniform(low=-self.v_stds*3, high=self.v_stds*3)

        wp_idx = 0
        timer = 0.
        X0 = x0.copy()
        yvec[:measurement_delay] = self.sys.getOutput(x0, vvec[0,:])
        for i in range(self.N):
            # if i % 126 == 0:
            #     vvec[i] = vvec[i]*10.   # introduce outliers
            
            if traj_mode=="circle":
                if type(self.sys).__name__ == 'Quadrotor1':
                    xref[0] = radius * cos(omega*self.tvec[i])
                    xref[2] = radius * sin(omega*self.tvec[i])
                    xref[4] = z0 + vz * self.tvec[i]
                    xref[1] = -radius * omega * sin(omega*self.tvec[i])
                    xref[3] = radius * omega * cos(omega*self.tvec[i])
                    xref[5] = vz
                elif type(self.sys).__name__ == 'Quadrotor2':
                    # omega += .001
                    # omega = min(omega, 5.)
                    xref[0] = radius * cos(omega*self.tvec[i])
                    xref[1] = radius * sin(omega*self.tvec[i])
                    # xref[2] = z0 + vz * self.tvec[i]
                    xref[2] = z0 + dz * sin(omega*self.tvec[i])
                    xref[3] = -radius * omega * sin(omega*self.tvec[i])
                    xref[4] = radius * omega * cos(omega*self.tvec[i])
                    # xref[5] = vz
                    xref[5] = dz * omega * cos(omega*self.tvec[i])
            elif traj_mode=="triangle":
                if type(self.sys).__name__ == 'Quadrotor1':
                    pass
                elif type(self.sys).__name__ == 'Quadrotor2':
                    wp_curr = waypoints[wp_idx, 0:3]
                    wp_next = waypoints[(wp_idx + 1) % waypoints.shape[0], 0:3]
                    s = (self.tvec[i] - timer) / time_per_side  # s \in [0,1]
                    xref[0:3] = (1 - s)*wp_curr + s*wp_next     # position
                    xref[3:6] = waypoints[wp_idx, 3:6]          # velocity
                    if self.tvec[i] - timer >= time_per_side:
                        timer += time_per_side
                        wp_idx = (wp_idx + 1) % waypoints.shape[0]

            xvec[i,:] = x0
            # yvec[i,:] = self.sys.getOutput(x0, vvec[i,:])
            i1 = i + measurement_delay
            if i1 < self.N:
                yvec[i1,:] = self.sys.getOutput(x0, vvec[i1,:])

            # Bound x,y,z errors for stability
            err = xref - x0
            if type(self.sys).__name__ == 'Quadrotor1':
                err[0] = self.saturate(err[0], -dmax, dmax)
                err[2] = self.saturate(err[2], -dmax, dmax)
                err[4] = self.saturate(err[4], -dmax, dmax)
            elif type(self.sys).__name__ == 'Quadrotor2':
                err[0] = self.saturate(err[0], -dmax, dmax)
                err[1] = self.saturate(err[1], -dmax, dmax)
                err[2] = self.saturate(err[2], -dmax, dmax)

            # Calculate control input
            u = StateFeedback.getGain()@err + self.u_eq
            u = self.sys.saturateControl(u)
            uvec[i,:] = u

            # Propagate dynamics
            dx = self.sys.dx(x0, u, wvec[i,:], t=self.tvec[i])
            x0 += dx*self.ts

        self.xvec = xvec
        self.uvec = uvec
        self.yvec = yvec
        return self.tvec, X0, xvec, uvec, yvec
    

    def simulate_free_response(self, x0, u=None, zero_disturbance=False, zero_noise=False):
        xvec = np.zeros((self.N, self.Nx))
        yvec = np.zeros((self.N, self.Ny))
        if u is None:
            uvec = np.zeros((self.N, self.Nu))
        else:
            uvec = u.reshape((self.N, self.Nu))
        if zero_disturbance:
            wvec = np.zeros((self.N, self.Nx))
        else:
            if self.noise_distribution=="gaussian":  wvec = np.random.normal(loc=self.w_means, scale=self.w_stds, size=(self.N, self.Nx))
            elif self.noise_distribution=="uniform": wvec = np.random.uniform(low=-self.w_stds*3, high=self.w_stds*3, size=(self.N, self.Nx))
        if zero_noise:
            vvec = np.zeros((self.N, self.Ny))
        else:
            if self.noise_distribution=="gaussian":  vvec = np.random.normal(loc=self.v_means, scale=self.v_stds, size=(self.N, self.Ny))
            elif self.noise_distribution=="uniform": vvec = np.random.uniform(low=-self.v_stds*3, high=self.v_stds*3, size=(self.N, self.Ny))

        for i in range(self.N):
            xvec[i,:] = x0
            yvec[i,:] = self.sys.getOutput(x0, vvec[i,:])
            dx = self.sys.dx(x0, uvec[i,:], wvec[i,:])
            x0 += dx*self.ts

        self.xvec = xvec
        self.uvec = uvec
        self.yvec = yvec
        return self.tvec, xvec, uvec, yvec

    
    def run_estimation(self, estimator, x0_est):
        x0hat = copy.deepcopy(x0_est)       # Initial estimate
        xhat = np.zeros((self.N, self.Nx))  # All estimates
        estimator_class = type(estimator).__name__
        if estimator_class not in ['KF', 'MHE']:
            raise ValueError("Invalid estimator class. Supported classes: KF, MHE.")

        t0 = time.perf_counter()
        try:
            for i in range(self.N):
                # -------------- Kalman filter --------------
                if estimator_class == 'KF':
                    estimator.update_covariance(
                        Q = self.Q[i],
                        R = self.R[i]
                    )
                    x0hat = estimator.correction(x0hat, self.yvec[i], t=self.tvec[i])
                    xhat[i] = x0hat
                    x0hat = estimator.prediction(x0hat, self.uvec[i], t=self.tvec[i])

                # -------------- Linear MHE --------------
                elif estimator_class == 'MHE':
                    x0hat = estimator.doEstimation(
                        yvec = self.yvec[: i+1],
                        uvec = self.uvec[: i],
                        Qinv_seq = self.Q_inv[: i+1],
                        Rinv_seq = self.R_inv[: i+1],
                        Q_seq = self.Q[: i+1],  # only used by smoothing MHE
                        R_seq = self.R[: i+1]   # only used by smoothing MHE
                    )
                    xhat[i] = x0hat
                    # estimator.updateCovariance(xhat[i], self.uvec[i])
        except:
            print(estimator_class + " failed! Returning NaN...")
            xhat.fill(np.nan)
        elapsed_time = time.perf_counter() - t0
        return xhat, elapsed_time


    def get_time_step(self):
        return self.ts
