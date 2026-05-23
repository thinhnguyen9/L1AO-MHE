import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


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

plot_labels = {
        'MHE (OSQP)': 'OSQP',
        'MHE (CVXOPT)': 'CVXOPT',
        'MHE (PCIP)': 'PCIP',
        'MHE (PCIP+L1AO)': 'PCIP+$\mathcal{L}_1$-AO',
    }

plot_colors = {
        'MHE (OSQP)': 'tab:blue',
        'MHE (CVXOPT)': 'm',
        'MHE (PCIP)': 'tab:orange',
        'MHE (PCIP+L1AO)': 'tab:green',
    }

def prepare_plot_data(csv_path: str, estimator_filter=None) -> pd.DataFrame:
    """Load and aggregate solver timing data for grouped bar plotting."""
    df = pd.read_csv(csv_path)

    required_cols = [
        "estimator",
        "MHE_horizon",
        "mean_solver_time",
        "min_solver_time",
        "max_solver_time",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only rows with numeric timing and horizon data.
    for c in ["MHE_horizon", "mean_solver_time", "min_solver_time", "max_solver_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["estimator", "MHE_horizon", "mean_solver_time", "min_solver_time", "max_solver_time"]).copy()

    if estimator_filter is not None:
        if isinstance(estimator_filter, str):
            estimator_filter = [estimator_filter]
        df = df[df["estimator"].isin(estimator_filter)].copy()

    if df.empty:
        raise ValueError("No valid rows found after applying estimator filter.")

    # Remove invalid rows where bounds are inconsistent.
    df = df[(df["min_solver_time"] <= df["mean_solver_time"]) & (df["mean_solver_time"] <= df["max_solver_time"])].copy()

    if df.empty:
        raise ValueError("No valid rows found after filtering invalid timing data.")

    # Aggregate in case there are multiple runs for one (horizon, estimator).
    agg = (
        df.groupby(["MHE_horizon", "estimator"], as_index=False)
        .agg(
            mean_time=("mean_solver_time", "mean"),
            min_time=("min_solver_time", "min"),
            max_time=("max_solver_time", "max"),
        )
    )

    # Keep only valid whisker geometry after aggregation.
    agg = agg[(agg["min_time"] <= agg["mean_time"]) & (agg["mean_time"] <= agg["max_time"])].copy()

    if agg.empty:
        raise ValueError("No valid aggregated rows found after filtering.")

    return agg


def plot_grouped_dots_with_whiskers(agg: pd.DataFrame) -> None:
    """Plot grouped points: x=horizon, dot=mean time, whiskers=[min,max]."""
    horizons = np.sort(agg["MHE_horizon"].unique())
    estimators = sorted(agg["estimator"].unique())

    x = np.arange(len(horizons), dtype=float)
    n_est = len(estimators)
    # Tighter clustering for estimators within each horizon.
    group_span = 0.4
    offset_step = group_span / max(n_est - 1, 1)

    # cmap = plt.get_cmap("tab10")
    # colors = {est: cmap(i % 10) for i, est in enumerate(estimators)}

    # Specify plot-area (axes) size in inches, independent of outer margins.
    plot_width_in = 5   # long: 5., short: 3.5
    plot_height_in = 2.6
    left_margin_in = 0.8
    right_margin_in = 0.15
    bottom_margin_in = 0.55
    top_margin_in = 0.12

    fig_width_in = left_margin_in + plot_width_in + right_margin_in
    fig_height_in = bottom_margin_in + plot_height_in + top_margin_in
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    ax = fig.add_axes([
        left_margin_in / fig_width_in,
        bottom_margin_in / fig_height_in,
        plot_width_in / fig_width_in,
        plot_height_in / fig_height_in,
    ])

    for i, est in enumerate(estimators):
        est_df = agg[agg["estimator"] == est]

        means = []
        mins = []
        maxs = []
        for h in horizons:
            row = est_df[est_df["MHE_horizon"] == h]
            if row.empty:
                means.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)
            else:
                means.append(float(row["mean_time"].iloc[0]))
                mins.append(float(row["min_time"].iloc[0]))
                maxs.append(float(row["max_time"].iloc[0]))

        means = np.asarray(means, dtype=float)
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)

        offset = (i - (n_est - 1) / 2.0) * offset_step
        xpos = x + offset

        valid = np.isfinite(means) & np.isfinite(mins) & np.isfinite(maxs)
        if not np.any(valid):
            continue

        lower_err = np.clip(means[valid] - mins[valid], a_min=0.0, a_max=None)
        upper_err = np.clip(maxs[valid] - means[valid], a_min=0.0, a_max=None)

        ax.errorbar(
            xpos[valid],
            means[valid],
            yerr=np.vstack([lower_err, upper_err]),
            fmt="o",
            color=plot_colors.get(est, "black"),
            markerfacecolor=plot_colors.get(est, "black"),
            markeredgecolor=plot_colors.get(est, "black"),
            markeredgewidth=.8,
            markersize=5,
            ecolor=plot_colors.get(est, "black"),
            elinewidth=.8,
            capsize=8,
            capthick=.8,
            label=plot_labels.get(est, est),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(h)) if float(h).is_integer() else str(h) for h in horizons])
    ax.set_xlabel("MHE horizon")
    ax.set_ylabel("Time per step (ms)")
    # ax.set_title("Test Time by Horizon (mean dots with min/max whiskers)")
    ax.set_yscale("log", base=10)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.6, alpha=0.8, zorder=1)
    ax.grid(axis="y", which="minor", linestyle="-", linewidth=0.4, alpha=0.4, zorder=1)
    ax.grid(axis="x", which="major", linestyle="-", linewidth=0.6, alpha=0.4, zorder=1)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "2026-04-04_compare_solvers_MS.csv")
    # csv_path = os.path.join(os.path.dirname(__file__), "2026-04-04_compare_solvers_SS_unconstrainedMHE.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Set this to a string or list of estimator names to plot only a subset.
    estimator_filter = [
        "MHE (OSQP)",
        "MHE (CVXOPT)",
        "MHE (PCIP)",
        "MHE (PCIP+L1AO)",
    ]

    data = prepare_plot_data(csv_path, estimator_filter=estimator_filter)
    # Log scale requires strictly positive values.
    data = data[(data["mean_time"] > 0) & (data["min_time"] > 0) & (data["max_time"] > 0)].copy()
    plot_grouped_dots_with_whiskers(data)
