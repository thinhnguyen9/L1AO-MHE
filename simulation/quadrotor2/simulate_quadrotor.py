import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import pathlib
from math import sin, cos
import cvxpy as cp
import copy
import time
import csv
from datetime import datetime
import os
import uuid

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from src.kf import KF
from src.mhe import MHE
from models.quadrotors import Quadrotor2
from src.simulator import Simulator
from src.pcip import PCIPQP
from src.l1ao import L1AOQP
from src.utils import rmse, remove_outliers_iqr, summarize_statistics

def main(
        enabled_estimators,
        v_means,    # Measurement noise (gaussian)
        v_stds,     # max error ~ 3 std.dev.
        w_means,    # Process noise (gaussian)
        w_stds,     # max error ~ 3 std.dev.
        x0_stds,    # Initial state ~ uniform distribution
        P0,         # initial covariance penalty
        trajectory_shape="hover",
        Q=None,     # weighting matrix, override if needed
        R=None,     # weighting matrix, override if needed
        T=1.0,
        t0=0.,
        ts=0.01,
        loops=1,
        mhe_horizon=10,
        mhe_update="filtering",
        prior_method="ekf",
        mhe_shooting="single",
        zero_measurement_noise=False,
        zero_process_noise=False,
        time_varying_measurement_noise=False,
        bad_model_knowledge=False,
        time_varying_dynamics=False,
        keep_initial_guess=False,
        save_csv_simulation_instance=False,
        save_csv_estimation_error=False,
        save_csv_raw_data=False,
        enable_plot=False,
        measurement_delay=0,
        lmhe1_solver="osqp",
        lmhe2_pcip_alpha=1./.01,
        lmhe2_pcip_prediction=True,
        lmhe2_interior_point_barrier=None,
        lmhe2_interior_point_slack=None,
        lmhe3_pcip_alpha=1./.01,
        lmhe3_pcip_prediction=True,
        lmhe3_l1ao_As=-.1,
        lmhe3_l1ao_omega=50.,
        lmhe3_interior_point_barrier=None,
        lmhe3_interior_point_slack=None,
        xmin=None,
        xmax=None,
        linear_solver="direct-lu"
    ):
    
    # ----------------------- Quadrotor ----------------------- #
    drone = Quadrotor2(
        m                           = 1.0,
        g                           = 9.81,
        J                           = np.diag([0.005, 0.005, 0.009]),
        time_varying_mass           = time_varying_dynamics,
        mass_scale_rate_of_change   = -0.25   # per second
    )
    if bad_model_knowledge:
        drone_est = Quadrotor2(     # used for estimation
            m                           = 1.5,
            g                           = 9.81,
            J                           = np.diag([0.005, 0.005, 0.009])*1.5,
            time_varying_mass           = time_varying_dynamics,
            mass_scale_rate_of_change   = -0.   # 0 means no change
        )
    else:
        drone_est = copy.deepcopy(drone)
    
    xhover_est = np.zeros(drone_est.Nx)
    uhover_est = np.array([drone_est.m*drone_est.g, 0, 0, 0])

    # ----------------------- Simulation ----------------------- #
    if Q is not None and R is not None:
        use_QR_guess = True
    else:
        use_QR_guess = False
        if Q is None:   Q = np.diag(w_stds**2)
        if R is None:   R = np.diag(v_stds**2)
    print("\n")
    print("=================== Simulation settings ===================")
    print(f"Simulation time    : {T:.1f} s")
    print(f"Sampling period    : {ts/1e-3:.1f} ms")
    print(f"Loops              : {loops:.0f}")
    print(f"Time-varying noise : " + str(time_varying_measurement_noise))
    print(f"Use Q,R guesses    : " + str(use_QR_guess))
    print("======================= MHE settings ======================")
    print(f"Horizon            : {mhe_horizon:.0f}")
    print("MHE formulation    : " + mhe_update + ", " + mhe_shooting + "-shooting")
    print("State constraints  : " + ("yes" if (xmin is not None or xmax is not None) else "no"))
    print("Prior weighting    : " + prior_method)
    print("Linear solver      : " + linear_solver)

    sim = Simulator(
        mode    = 'quadrotor',
        sys     = drone,
        w_means = w_means,
        w_stds  = w_stds,
        v_means = v_means,
        v_stds  = v_stds,
        x0_stds = x0_stds,
        T       = T,
        ts      = ts,
        # noise_distribution = 'uniform',
        time_varying_measurement_noise = time_varying_measurement_noise,
        use_QR_guess = use_QR_guess,
        Q_guess = Q,
        R_guess = R
    )
    t0 = int(t0/ts)
    rng = np.random.default_rng(88)
    run_id = uuid.uuid4().hex[:8]

    # ----------------------- Run estimation ----------------------- #
    for loop in range(loops):
        print("================ Simulation instance " + str(loop+1) + " of " + str(loops) + " ================")

        # Random initial state
        x0 = rng.uniform(low=x0_stds[0], high=x0_stds[1])

        # Run simulation
        tvec, xvec, uvec, yvec = sim.simulate_quadrotor_lqr_control(
            seed                = loop,    # for deterministic measurement noise generation
            traj_mode           = trajectory_shape,
            x0                  = x0,
            xref                = None,
            zero_disturbance    = zero_process_noise,
            zero_noise          = zero_measurement_noise,
            measurement_delay   = measurement_delay
        )
        # # Initial estimate variation
        # x0norm = 10.
        # if np.linalg.norm(x0_stds) > 1e-6:
        #     x0var = rng.uniform(low=x0_stds[0], high=x0_stds[1])
        #     norm_x0var = np.linalg.norm(x0var)
        #     if norm_x0var == 0.:
        #         x0var = np.zeros(drone.Nx)
        #         x0var[0] = x0norm
        #         x0 = x0 + x0var
        #     else:
        #         x0 = x0 + x0norm*x0var/norm_x0var  # normalize so that norm(e0)=1
        # # x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Initialize estimators - must be done every loop
        enabled_estimators = [est for est in enabled_estimators if est in ['KF', 'EKF', 'LMHE1', 'LMHE2', 'LMHE3']]
        estimator_objs = {}
        estimator_labels = {}
        if 'KF' in enabled_estimators:
            SKF = KF(
                model   = drone_est,
                ts      = sim.get_time_step(),
                P0      = P0,
                type    = "standard",
                xs      = xhover_est,
                us      = uhover_est
            )
            estimator_objs['KF'] = SKF
            estimator_labels['KF'] = "KF"
        if 'EKF' in enabled_estimators:
            EKF = KF(
                model   = drone_est,
                ts      = sim.get_time_step(),
                P0      = P0,
                type    = "extended"
            )
            estimator_objs['EKF'] = EKF
            estimator_labels['EKF'] = "EKF"
        if 'LMHE1' in enabled_estimators:
            lmhe1_obj = MHE(
                model           = drone_est,
                ts              = sim.get_time_step(),
                N               = mhe_horizon,
                X0              = xhover_est,
                P0              = P0,
                mhe_type        = "linearized_every",
                mhe_update      = mhe_update,
                mhe_shooting    = mhe_shooting,
                prior_method    = prior_method,
                solver          = lmhe1_solver,
                xs              = xhover_est,
                us              = uhover_est,
                xmin            = xmin,
                xmax            = xmax
            )
            estimator_objs['LMHE1'] = lmhe1_obj
            estimator_labels['LMHE1'] = "MHE (" + lmhe1_solver.upper() + ")"
        if 'LMHE2' in enabled_estimators:
            lmhe2_pcip_obj = PCIPQP(
                alpha               = lmhe2_pcip_alpha,
                ts                  = ts,
                model               = drone_est,
                shooting_method     = mhe_shooting,
                enable_prediction   = lmhe2_pcip_prediction,
                interior_point_barrier = lmhe2_interior_point_barrier,
                interior_point_slack   = lmhe2_interior_point_slack,
                linear_solver          = linear_solver
            )
            lmhe2_obj = MHE(
                model           = drone_est,
                ts              = sim.get_time_step(),
                N               = mhe_horizon,
                X0              = xhover_est,
                P0              = P0,
                mhe_type        = "linearized_every",
                mhe_update      = mhe_update,
                mhe_shooting    = mhe_shooting,
                prior_method    = prior_method,
                solver          = "pcip",
                xs              = xhover_est,
                us              = uhover_est,
                pcip_obj        = lmhe2_pcip_obj,
                xmin            = xmin,
                xmax            = xmax
            )
            estimator_objs['LMHE2'] = lmhe2_obj
            estimator_labels['LMHE2'] = "MHE (PCIP)"
        if 'LMHE3' in enabled_estimators:
            lmhe3_pcip_obj = PCIPQP(
                alpha               = lmhe3_pcip_alpha,
                ts                  = ts,
                model               = drone_est,
                shooting_method     = mhe_shooting,
                enable_prediction   = lmhe3_pcip_prediction,
                interior_point_barrier = lmhe3_interior_point_barrier,
                interior_point_slack   = lmhe3_interior_point_slack,
                linear_solver          = linear_solver,
                l1ao_augmentation      = {"a": lmhe3_l1ao_As, "lpf_omega": lmhe3_l1ao_omega}
            )
            lmhe3_obj = MHE(
                model           = drone_est,
                ts              = sim.get_time_step(),
                N               = mhe_horizon,
                X0              = xhover_est,
                P0              = P0,
                mhe_type        = "linearized_every",
                mhe_update      = mhe_update,
                mhe_shooting    = mhe_shooting,
                prior_method    = prior_method,
                solver          = "pcip",
                xs              = xhover_est,
                us              = uhover_est,
                pcip_obj        = lmhe3_pcip_obj,
                xmin            = xmin,
                xmax            = xmax
            )
            estimator_objs['LMHE3'] = lmhe3_obj
            estimator_labels['LMHE3'] = "MHE (PCIP+L1AO)"

        # Start estimation
        def run_estimator(estimator):

            # Obtain estimation
            xhat, total_time = sim.run_estimation(estimator, xhover_est)
            estimation_rmse = rmse(xvec[t0:], xhat[t0:])
            if keep_initial_guess:
                xhat[0] = xhover_est.copy()
            
            # Compute norm(estimation error) for each time step
            norm_err = np.linalg.norm(xvec-xhat, axis=1)

            # Compute solver times statistics: from t0 onwards, and remove outliers with IQR method (only for MHE)
            if type(estimator).__name__ == "MHE":
                solver_times = np.array(estimator.get_solver_times(t0)) * 1000
                solver_times = remove_outliers_iqr(solver_times)
            else:
                solver_times = total_time*1000./N * np.ones(N)
            solver_time_stats = summarize_statistics(solver_times)
            
            return {
                "xhat": xhat,
                "rmse": estimation_rmse,
                "norm_err": norm_err,
                "total_time": total_time,
                "solver_times": solver_times,
                "solver_time_stats": solver_time_stats
            }
        
        N = len(tvec)
        results = {}
        for estimator_name in enabled_estimators:
            results[estimator_name] = run_estimator(estimator_objs[estimator_name])

        # Print RMSE & MHE times
        print(f"(k={t0:.0f} onwards)".ljust(15) + f" {'RMSE':>15} {'MHE step [ms]':>15} {'Solver [ms]':>15}")
        print("--------------------------------------------------------------")
        for estimator_name in enabled_estimators:
            print(
                f"{estimator_labels[estimator_name]:<15}"
                f" {results[estimator_name]['rmse']:15.4f}"
                f" {results[estimator_name]['total_time']*1000./N:15.4f}"
                f" {results[estimator_name]['solver_time_stats']['mean']:15.4f}"
            )

        # # Print MHE solver times
        # print("------------------------------- MHE solver times -------------------------------")
        # print(f"{'MHE':<15} {'Mean [ms]':>12} {'Median [ms]':>12} {'Std [ms]':>12} {'Min [ms]':>12} {'Max [ms]':>12}")
        # print("--------------------------------------------------------------------------------")
        # for estimator_name in enabled_estimators:
        #     print(
        #         f"{estimator_labels[estimator_name]:<15}"
        #         f" {results[estimator_name]['solver_time_stats']['mean']:12.4f}"
        #         f" {results[estimator_name]['solver_time_stats']['median']:12.4f}"
        #         f" {results[estimator_name]['solver_time_stats']['std']:12.4f}"
        #         f" {results[estimator_name]['solver_time_stats']['min']:12.4f}"
        #         f" {results[estimator_name]['solver_time_stats']['max']:12.4f}"
        #     )

        # # Print solver times breakdown for PCIP/L1AO
        # if 'LMHE2' in enabled_estimators:   lmhe2_pcip_obj.print_computation_times(t0)
        # if 'LMHE3' in enabled_estimators:   lmhe3_pcip_obj.print_computation_times(t0)

        # Save results of this instance
        if save_csv_simulation_instance:
            data = []
            for estimator_name in enabled_estimators:
                data.append([
                    run_id,
                    loop + 1,
                    T,
                    ts,
                    t0,
                    estimator_labels[estimator_name],
                    linear_solver if estimator_name in ['LMHE2', 'LMHE3'] else (lmhe1_solver if estimator_name=='LMHE1' else None),
                    results[estimator_name]['rmse'],
                    np.max(results[estimator_name]['norm_err'][t0:]),
                    mhe_shooting + "-shooting, " + mhe_update if estimator_name.startswith('LMHE') else None,
                    mhe_horizon if estimator_name.startswith('LMHE') else None,
                    results[estimator_name]['solver_time_stats']['mean'],
                    results[estimator_name]['solver_time_stats']['median'],
                    results[estimator_name]['solver_time_stats']['std'],
                    results[estimator_name]['solver_time_stats']['min'],
                    results[estimator_name]['solver_time_stats']['max']
                ])
            file_path = os.path.join(os.path.dirname(__file__), 'data/simulation_instances.csv')
            write_header = not os.path.exists(file_path)
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow([
                        'run_id', 'loop', 'T', 'ts', 'stats_start_step', 'estimator', 'solver',
                        'RMSE', 'max_norm_err', 'MHE_scheme', 'MHE_horizon', 'mean_solver_time',
                        'median_solver_time', 'std_solver_time', 'min_solver_time', 'max_solver_time'
                    ])
                writer.writerows(data)
        
        # Write raw data (norm(x-xhat)) to csv
        if save_csv_estimation_error:
            file_path = os.path.join(os.path.dirname(__file__), 'data/estimation_error.csv')
            write_header = not os.path.exists(file_path)
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(['run_id','loop','estimator','T','time','estimation_error_norm'])
                for i in range(N):
                    t = float(tvec[i])
                    for estimator_name in enabled_estimators:
                        writer.writerow([
                            run_id,
                            loop + 1,
                            estimator_labels[estimator_name],
                            T,
                            f"{t:.2f}",
                            f"{results[estimator_name]['norm_err'][i]:.6f}"
                        ])
            print("Data written to " + file_path)
    print("============================================================")
    with open('simulation/quadrotor2/sim_data.npy', 'wb') as f:
        np.save(f, tvec)
        np.save(f, xvec)
        np.save(f, yvec)
        np.save(f, uvec)
        if 'EKF'   in enabled_estimators:   np.save(f, results['EKF']['xhat'])
        if 'LMHE1' in enabled_estimators:   np.save(f, results['LMHE1']['xhat'])
        if 'LMHE2' in enabled_estimators:   np.save(f, results['LMHE2']['xhat'])
        if 'LMHE3' in enabled_estimators:   np.save(f, results['LMHE3']['xhat'])
    print("Simulation data saved to 'sim_data.npy'")
    print("\n")

    # ----------------------- Plot results -----------------------
    if enable_plot:
        
        if 'KF' in enabled_estimators:      rmse_kf     = np.sqrt(np.mean((xvec - results['KF']['xhat'])**2,    axis=0))
        if 'EKF' in enabled_estimators:     rmse_ekf    = np.sqrt(np.mean((xvec - results['EKF']['xhat'])**2,   axis=0))
        if 'LMHE1' in enabled_estimators:   rmse_lmhe1  = np.sqrt(np.mean((xvec - results['LMHE1']['xhat'])**2, axis=0))
        if 'LMHE2' in enabled_estimators:   rmse_lmhe2  = np.sqrt(np.mean((xvec - results['LMHE2']['xhat'])**2, axis=0))
        if 'LMHE3' in enabled_estimators:   rmse_lmhe3  = np.sqrt(np.mean((xvec - results['LMHE3']['xhat'])**2, axis=0))

        def plot_state(idx, ylabel=None, invert_y=False, rad2deg=False, title_prefix=''):
            plt.plot(tvec, xvec[:,idx]*(180/np.pi if rad2deg else 1), 'k-', lw=3., label=title_prefix+'_true')
            if idx in [0, 1, 2]:
                plt.plot(tvec, yvec[:,idx]*(180/np.pi if rad2deg else 1), 'k:', lw=0.5, label=title_prefix+'_meas')
            if idx in [9, 10, 11]:
                plt.plot(tvec, yvec[:,idx-6]*(180/np.pi if rad2deg else 1), 'k:', lw=0.5, label=title_prefix+'_meas')
            colors = {'KF': 'r', 'EKF': 'tab:red', 'LMHE1': 'tab:blue', 'LMHE2': 'tab:orange', 'LMHE3': 'tab:green'}
            for estimator_name in enabled_estimators:
                plt.plot(
                    tvec,
                    results[estimator_name]['xhat'][:,idx]*(180/np.pi if rad2deg else 1),
                    color=colors[estimator_name],
                    ls='-',
                    lw=1.5,
                    label=estimator_labels[estimator_name]
                )
            plt.grid()
            # plt.ylim((-2,2))
            leg = plt.legend()
            leg.set_draggable(True)
            if ylabel: plt.ylabel(ylabel)
            if invert_y: plt.gca().invert_yaxis()
            # Compose RMSE string only for enabled estimators
            rmse_str = []
            if 'KF' in enabled_estimators:      rmse_str.append(f"KF={rmse_kf[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'EKF' in enabled_estimators:     rmse_str.append(f"EKF={rmse_ekf[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'LMHE1' in enabled_estimators:   rmse_str.append(f"LMHE1={rmse_lmhe1[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'LMHE2' in enabled_estimators:   rmse_str.append(f"LMHE2={rmse_lmhe2[idx]*(180/np.pi if rad2deg else 1):.4f}")
            if 'LMHE3' in enabled_estimators:   rmse_str.append(f"LMHE3={rmse_lmhe3[idx]*(180/np.pi if rad2deg else 1):.4f}")
            plt.title(f"{title_prefix} RMSE: {', '.join(rmse_str)}", fontsize=10)

        plt.figure(1)
        plt.suptitle('Estimators comparison')

        # -------------- X, Y, Z --------------
        plt.subplot(4,3,1)
        plot_state(0, ylabel='Position - world (m)', title_prefix='x')
        plt.subplot(4,3,2)
        plot_state(1, title_prefix='y')
        plt.subplot(4,3,3)
        plot_state(2, title_prefix='z')
        # -------------- Xd, Yd, Zd --------------
        plt.subplot(4,3,4)
        plot_state(3, ylabel='Velocity - world (m/s)', title_prefix='xd')
        plt.subplot(4,3,5)
        plot_state(4, title_prefix='yd')
        plt.subplot(4,3,6)
        plot_state(5, title_prefix='zd')
        # -------------- roll, pitch, yaw --------------
        plt.subplot(4,3,7)
        plot_state(6, ylabel='Angles - world (deg)', rad2deg=True, title_prefix='roll')
        plt.subplot(4,3,8)
        plot_state(7, rad2deg=True, title_prefix='pitch')
        plt.subplot(4,3,9)
        plot_state(8, rad2deg=True, title_prefix='yaw')
        # -------------- p, q, r --------------
        plt.subplot(4,3,10)
        plot_state(9, ylabel='Angular rates - body (rad/s)', title_prefix='p')
        plt.xlabel('Time (s)')
        plt.subplot(4,3,11)
        plot_state(10, title_prefix='q')
        plt.xlabel('Time (s)')
        plt.subplot(4,3,12)
        plot_state(11, title_prefix='r')
        plt.xlabel('Time (s)')

        plt.figure(2)
        plt.suptitle('Control inputs')
        
        plt.subplot(2,1,1)
        plt.plot(tvec, uvec[:,0], 'k-', lw=1.)
        plt.ylabel('Total thrust (N)')
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(tvec, uvec[:,1], 'r-', lw=1., label=r'$\tau_x$')
        plt.plot(tvec, uvec[:,2], 'g-', lw=1., label=r'$\tau_y$')
        plt.plot(tvec, uvec[:,3], 'b-', lw=1., label=r'$\tau_z$')
        plt.ylabel('Total torque (Nm)')
        plt.xlabel('Time (s)')
        leg = plt.legend()
        leg.set_draggable(True)
        plt.grid()

        def plot_error(idx, ylabel=None, title_prefix=''):
            if 'KF' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-results['KF']['xhat'][:,idx], 'r-', lw=1., label='KF')
            if 'EKF' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-results['EKF']['xhat'][:,idx], 'b-', lw=1., label='EKF')
            if 'LMHE1' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-results['LMHE1']['xhat'][:,idx], 'm-', lw=1., label='LMHE1')
            if 'LMHE2' in enabled_estimators:
                plt.plot(tvec, xvec[:,idx]-results['LMHE2']['xhat'][:,idx], 'c-', lw=1., label='LMHE2')
            if 'EKF' in enabled_estimators and 'LMHE2' in enabled_estimators:
                plt.plot(tvec, results['LMHE2']['xhat'][:,idx]-results['EKF']['xhat'][:,idx], 'k--', lw=1., label='LMHE2-EKF')
            plt.grid()
            leg = plt.legend()
            leg.set_draggable(True)
            if ylabel: plt.ylabel(ylabel)
            plt.title(title_prefix, fontsize=10)

        # plt.figure(3)
        # plt.suptitle('Estimation error (by state)')
        # for idx in range(drone.Nx):
        #     plt.subplot(4, 4, idx+1)
        #     plot_error(idx)

        if 'EKF' in enabled_estimators:
            plt.figure(4)
            # plt.suptitle('Estimation error')
            err_ekf   = np.linalg.norm(xvec - results['EKF']['xhat'], axis=1)
            if 'LMHE1' in enabled_estimators:   err_lmhe1 = np.linalg.norm(xvec - results['LMHE1']['xhat'], axis=1)
            if 'LMHE2' in enabled_estimators:   err_lmhe2 = np.linalg.norm(xvec - results['LMHE2']['xhat'], axis=1)
            if 'LMHE3' in enabled_estimators:   err_lmhe3 = np.linalg.norm(xvec - results['LMHE3']['xhat'], axis=1)

            plt.subplot(211)
            plt.title('Estimation error')
            plt.plot(tvec, err_ekf, color='tab:red', ls='-', lw=1.5, label='EKF')
            if 'LMHE1' in enabled_estimators:   plt.plot(tvec, err_lmhe1, color='tab:blue', ls='-', lw=1.5, label=f"MHE ({lmhe1_solver.upper()})")
            if 'LMHE2' in enabled_estimators:   plt.plot(tvec, err_lmhe2, color='tab:orange', ls='-', lw=1.5, label="MHE (PCIP)")
            if 'LMHE3' in enabled_estimators:   plt.plot(tvec, err_lmhe3, color='tab:green', ls='-', lw=1.5, label="MHE (PCIP+L1AO)")
            # plt.yscale('log')
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel(r'$\|x - \hat{x}\|$', fontsize=10)
            leg = plt.legend()
            leg.set_draggable(True)

            plt.subplot(212)
            plt.title('Comparison with EKF')
            plt.axhline(0.0, color='tab:red', linestyle='--', linewidth=2.)
            if 'LMHE1' in enabled_estimators:   plt.plot(tvec, err_lmhe1-err_ekf, color='tab:blue', ls='-', lw=1.5, label='LMHE1-EKF')
            if 'LMHE2' in enabled_estimators:   plt.plot(tvec, err_lmhe2-err_ekf, color='tab:orange', ls='-', lw=1.5, label='LMHE2-EKF')
            if 'LMHE3' in enabled_estimators:   plt.plot(tvec, err_lmhe3-err_ekf, color='tab:green', ls='-', lw=1.5, label='LMHE3-EKF')
            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel('LMHE - EKF')
            leg = plt.legend()
            leg.set_draggable(True)

        plt.show()


if __name__ == "__main__":
    """
    Working estimators: "KF", "EKF",
                        "LMHE1" (CVXPY OSQP/ECOS),
                        "LMHE2" (PCIP),
                        "LMHE3" (PCIP+L1AO)
    Working MHE update schemes: "filtering" (equivalent to the EKF),
                                "smoothing_naive" (unstable for class Quadrotor2),
                                "smoothing" (works best for class Quadrotor2)
    Might need to retune (Q, R, PCIP, L1AO) for different schemes.
    """
    main(
        enabled_estimators=['EKF', 'LMHE1', 'LMHE2', 'LMHE3'],
        # trajectory_shape='circle',  # "hover" (default), "p2p", "circle, "triangle"
        T=.5,
        # t0=1.,  # time to start RMSE calculation (to skip transient phase)
        # ts=0.001,
        # loops=5,
        keep_initial_guess=True,  # keep initial guess at T=0 (so all estimators look like they start at the same x0)
                                  # purely for plotting purpose
        # save_csv_simulation_instance=True,
        # save_csv_estimation_error=True,
        # save_csv_raw_data=True,
        enable_plot=True,

        # ---------------- Actual noise characteristics ----------------
        v_means=np.zeros(6),
        w_means=np.zeros(12),
        v_stds=np.array([.09, .09, .09, .3, .3, .3])/3,     # Measurement noise ~ Gaussian (max ~ 3 std.dev.)
        w_stds=np.array([1e-6, 1e-6, 1e-6, .5, .5, .5,      # Disturbance: sinunoidal in linear and angular accelerations
                         1e-6, 1e-6, 1e-6, .2, .2, .2]),
        x0_stds=np.kron([[-1],[1]], [1., 1., 1., .1, .1, .1, .1, .1, .1, 1., 1., 1.]), # Random initial guess ~ uniform distribution
        # x0_stds=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]*2),
        # zero_measurement_noise=True,
        zero_process_noise=True,

        # ---------------- Covariance matrices for EKF/MHE ----------------
        #       Override both Q,R to use these (bad) values for estimation
        #       Lower Q = trust model, lower R = trust measurements
        #       PCIP/L1AO might work better when trusting model (high R, low Q)
        #       High Q might make L1AO converges very slowly
        #       Tuning Q: if a state converges slowly, increase Q for that state
        Q=np.eye(12) * 0.01,
        R=np.eye(6) * 0.01,
        P0=np.eye(12) * 1e-2,

        # ---------------- MHE settings ----------------
        # mhe_horizon  = 18,              # longer horizon: lower P0 so PCIP/L1AO doesn't blow up
        mhe_update   = "smoothing",     # "filtering" (default), "smoothing", or "smoothing_naive"
        # prior_method = "uniform",     # "zero", "uniform", "ekf" (default)
        # xmin = np.array([-1., -1.,  .5, -5., -5., -1., -np.pi/3, -np.pi/3, -np.pi/9, -50., -50., -50.]), # state constraints for circle traj
        # xmax = np.array([ 1.,  1., 1.5,  5.,  5.,  1.,  np.pi/3,  np.pi/3,  np.pi/9,  50.,  50.,  50.]),

        # ---------------- MHE solvers ----------------
        lmhe1_solver            = "osqp",   # cvxpy/osqp/cvxopt/pcip/pcip_l1ao
        lmhe2_pcip_alpha        = 1./.01,
        lmhe2_pcip_prediction   = True,     # False: reduce to Newton method
        lmhe3_pcip_alpha        = .5/.01,
        lmhe3_pcip_prediction   = True,     # False: reduce to Newton method
        lmhe3_l1ao_As           = -.1,
        lmhe3_l1ao_omega        = 150.,
        # interior_point_barrier  = [0.001, 10.0],    # c(t)=c0*exp(gamma_c*t): c0, gamma_c
        # interior_point_slack    = [10.0, 10.0],     # s(t)=s0*exp(-gamma_s*t): s0, gamma_s

        # ---------------- Corner cases ----------------
        # time_varying_measurement_noise = True,
        # bad_model_knowledge=True,
        # time_varying_dynamics=True,
        # measurement_delay=15,    # measurement delayed by how many time steps
    )
