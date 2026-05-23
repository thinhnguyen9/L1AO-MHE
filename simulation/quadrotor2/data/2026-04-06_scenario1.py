import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Use LaTeX-style mathtext and serif fonts for prettier rendering (no external LaTeX required)
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "serif",
#     "mathtext.fontset": "stix",                # STIX math fonts (good for publications)
#     "font.serif": ["Times New Roman", "STIXGeneral"],
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 12,
#     "legend.fontsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "figure.dpi": 150
# })
plt.rcParams.update({
    "text.usetex": True,                       # require TeX, dvipng/ghostscript installed
    "font.family": "serif",
    "font.serif": ["Computer Modern"],   # LaTeX default (Computer Modern)
    "mathtext.fontset": "cm",                  # math rendering to Computer Modern
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150
})

def load_and_aggregate_errors(csv_file, estimator=None, manual_exclusions=None):
    """
    Load estimation_error.csv and aggregate errors by time step.
    
    Args:
        csv_file: path to estimation_error.csv
        estimator: filter by estimator name (e.g. 'EKF'). If None, process all.
        manual_exclusions: list of (run_id, estimator, loop) to exclude.
            Example: [('51c9ba33', 'EKF', 10)]
    
    Returns:
        t_unique: sorted unique time points
        mean_err: mean error at each time point
        min_err: minimum error at each time point
        max_err: maximum error at each time point
        n_runs: number of simulation instances
    """
    df = pd.read_csv(csv_file)

    # Manually exclude specific trajectories before any aggregation.
    if manual_exclusions:
        exclude_mask = np.zeros(len(df), dtype=bool)
        for run_id, est_name, loop_id in manual_exclusions:
            exclude_mask |= (
                (df['run_id'].astype(str) == str(run_id))
                & (df['estimator'] == est_name)
                & (df['loop'] == loop_id)
            )
        df = df[~exclude_mask].copy()
    
    # Filter by estimator if specified
    if estimator:
        df = df[df['estimator'] == estimator]
    
    # Remove any (run_id, loop, estimator) trajectory containing NaN errors.
    # If one estimator has NaN in a loop, only that estimator's trajectory is dropped.
    loop_has_nan = (
        df.groupby(['run_id', 'loop', 'estimator'])['estimation_error_norm']
        .transform(lambda s: s.isna().any())
    )
    df = df[~loop_has_nan].copy()

    if df.empty:
        raise ValueError(
            f"No valid data left after filtering NaN loops"
            + (f" for estimator '{estimator}'" if estimator else "")
        )

    # Group by time and calculate mean/min/max across all valid loops
    grouped = df.groupby('time')['estimation_error_norm'].agg(['mean', 'min', 'max', 'count'])
    
    t_unique = grouped.index.values
    mean_err = grouped['mean'].values
    min_err = grouped['min'].values
    max_err = grouped['max'].values
    n_runs = df[['run_id', 'loop']].drop_duplicates().shape[0]
    
    return t_unique, mean_err, min_err, max_err, n_runs



if __name__ == "__main__":
    
    # csv file name
    csv_path = os.path.join(os.path.dirname(__file__), "2026-04-06_scenario1_estimation_error.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Manually exclude known bad trajectories: (run_id, estimator, loop)
    manual_exclusions = [
        ('51c9ba33', 'EKF', 10),
        ('51c9ba33', 'EKF', 52),
        ('51c9ba33', 'EKF', 87),
    ]
    
    # Estimators to plot
    estimators = ['EKF', 'MHE (OSQP)', 'MHE (PCIP)', 'MHE (PCIP+L1AO)'] # sort highest performing last for visibility
    
    # ===================================================================================================== #
    colors = {
        'EKF': 'tab:red',
        'MHE (OSQP)': 'tab:blue', 
        'MHE (PCIP)': 'tab:orange',
        'MHE (PCIP+L1AO)': 'tab:green',
    }
    markers = {
        'EKF': 'o',
        'MHE (OSQP)': '^', 
        'MHE (PCIP)': 's',
        'MHE (PCIP+L1AO)': 'd',
    }
    plot_labels = {
        'KF': 'KF',
        'EKF': 'EKF',
        'MHE (OSQP)': 'MHE (OSQP)', 
        'MHE (PCIP)': 'MHE (PCIP)',
        'MHE (PCIP+L1AO)': 'MHE (PCIP+$\mathcal{L}_1$-AO)',
        'NMHE': 'NMHE'
    }
    plt.figure(figsize=(9,5))
    for est in estimators:
        try:
            t, mean_err, min_err, max_err, n_runs = load_and_aggregate_errors(
                csv_path,
                estimator=est,
                manual_exclusions=manual_exclusions,
            )
            plt.plot(
                t, mean_err, color=colors.get(est, 'k'), lw=2.0,
                label=plot_labels.get(est, ''),
                marker=markers.get(est, None), markersize=6, markevery=20
            )
            plt.fill_between(t, min_err, max_err,
                           color=colors.get(est, 'k'), alpha=0.25)
        except Exception as e:
            print(f"Skipping {est}: {e}")
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(r'$\|x-\hat{x}\|$', fontsize=14)
    # plt.title(rf'Mean ± Stddev estimation error over {n_runs} runs', fontsize=14)

    # reorder legend by custom order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        labels.index(plot_labels.get('EKF')),
        labels.index(plot_labels.get('MHE (OSQP)')),
        labels.index(plot_labels.get('MHE (PCIP)')),
        labels.index(plot_labels.get('MHE (PCIP+L1AO)'))
    ]
    plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=10)

    # plt.grid(True, which='both', color='k', ls='--', lw=0.5, alpha=0.4)
    plt.grid(axis="y", which="major", color="black", linestyle="-", linewidth=0.6, alpha=0.4, zorder=0)
    plt.grid(axis="y", which="minor", color="black", linestyle="-", linewidth=0.4, alpha=0.2, zorder=0)
    plt.grid(axis="x", which="major", color="black", linestyle="-", linewidth=0.6, alpha=0.2, zorder=0)
    # plt.axis('square')
    # plt.xlim((t[0]-.005, .2))
    # plt.ylim((0., 10.5))

    plt.xlim((t[0]-.05, t[-1]))

    # plt.xlim((0.0, 0.2))
    # plt.ylim((0.18, 8.0))

    plt.yscale('log')

    # -------------------- inset: zoom first 0.2 s --------------------
    # ax = plt.gca()
    # # Use t from the previous loop (fallback if not defined)
    # try:
    #     t0 = float(t[0])
    # except Exception:
    #     t0 = 0.0
    # zoom_width = 0.5
    # zoom_t0 = t0
    # zoom_t1 = zoom_t0 + zoom_width

    # # Create inset axes (x, y, width, height in axes fraction coords)
    # axins = ax.inset_axes([0.3, 0.52, 0.4, 0.4])
    # for est in estimators:
    #     try:
    #         t_est, mean_err, min_err, max_err, _ = load_and_aggregate_errors(
    #             csv_path,
    #             estimator=est,
    #             manual_exclusions=manual_exclusions,
    #         )
    #     except Exception:
    #         continue
    #     # select indices in zoom window
    #     mask = (t_est >= zoom_t0) & (t_est <= zoom_t1)
    #     if not np.any(mask):
    #         continue
    #     c = colors.get(est, 'k')
    #     axins.plot(t_est[mask], mean_err[mask], color=c, lw=1.6, label=plot_labels.get(est, est))
    #     axins.fill_between(t_est[mask], min_err[mask], max_err[mask],
    #                        color=c, alpha=0.25)
    # axins.set_xlim(zoom_t0, zoom_t1)
    # # y-limits with small padding
    # yvals_masked = []
    # for est in estimators:
    #     try:
    #         t_est, mean_err, _, _, _ = load_and_aggregate_errors(
    #             csv_path,
    #             estimator=est,
    #             manual_exclusions=manual_exclusions,
    #         )
    #     except Exception:
    #         continue
    #     mask = (t_est >= zoom_t0) & (t_est <= zoom_t1)
    #     if np.any(mask):
    #         yvals_masked.append(mean_err[mask])
    # if yvals_masked:
    #     yall = np.concatenate(yvals_masked)
    #     ymin, ymax = yall.min(), yall.max()
    #     pad = max(1e-6, 0.05*(ymax - ymin) if (ymax - ymin) > 0 else 0.1*ymax)
    #     axins.set_ylim(max(0.0, ymin - pad), ymax + pad)
    # axins.grid(True, linestyle='--', linewidth=0.4)
    # axins.tick_params(axis='both', which='major', labelsize=8)
    # # indicate zoom rectangle on main axes (best-effort)
    # try:
    #     ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6)
    # except Exception:
    #     # fallback: draw simple rectangle
    #     rect = plt.Rectangle((zoom_t0, ax.get_ylim()[0]), zoom_width, ax.get_ylim()[1]-ax.get_ylim()[0],
    #                          linewidth=0.8, edgecolor='black', facecolor='none', linestyle='--', alpha=0.4)
    #     ax.add_patch(rect)
    # axins.set_yscale('log')
    # ------------------ end inset --------------------

    plt.show()
