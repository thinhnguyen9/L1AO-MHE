import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
})


def build_rmse_grid(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"Q", "time", "estimator", "estimation_error_norm"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {sorted(required_cols)}")

    df["Q"] = df["Q"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["estimation_error_norm"] = pd.to_numeric(df["estimation_error_norm"], errors="coerce")

    # Normalize estimator names, e.g. "MHE (baseline N=10)" -> "MHE (baseline)".
    def normalize_estimator_name(name):
        name = str(name)
        return re.sub(r"\s+N\s*=\s*\d+", "", name)

    # Parse N from estimator label if needed, e.g. "MHE (PCIP N=40)" -> 40.
    def parse_n_from_estimator(name):
        m = re.search(r"N\s*=\s*(\d+)", str(name))
        return int(m.group(1)) if m else np.nan

    df["estimator_norm"] = df["estimator"].map(normalize_estimator_name)

    if "N" in df.columns:
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
    else:
        df["N"] = np.nan

    # Backfill N from estimator text when not provided in a separate column.
    missing_n = df["N"].isna()
    if missing_n.any():
        df.loc[missing_n, "N"] = df.loc[missing_n, "estimator"].map(parse_n_from_estimator)

    # Keep only rows with valid estimator and raw error norm, then apply time filter.
    df = df.dropna(subset=["time", "estimator_norm", "estimation_error_norm"]).copy()
    df = df[df["time"] >= 1.0].copy()

    if df.empty:
        raise ValueError("No rows found with valid Q, estimator, and estimation_error_norm")

    # Compute RMSE by (Q, N, normalized estimator) from raw estimation_error_norm.
    grouped = (
        df.groupby(["Q", "N", "estimator_norm"], dropna=False, as_index=False)["estimation_error_norm"]
        .apply(lambda s: float(np.sqrt(np.mean(np.square(s.values)))))
        .rename(columns={"estimation_error_norm": "RMSE"})
    )

    # Split EKF rows (no horizon) from MHE rows (with horizon).
    ekf_df = grouped[grouped["estimator_norm"] == "EKF"].copy()
    mhe_df = grouped[grouped["estimator_norm"] != "EKF"].dropna(subset=["N"]).copy()

    if mhe_df.empty:
        raise ValueError("No MHE rows with valid N were found")

    # The user expects unique run for each (Q, N, estimator). Do not summarize.
    dup_mask = mhe_df.duplicated(subset=["Q", "estimator_norm", "N"], keep=False)
    if dup_mask.any():
        dup_rows = mhe_df.loc[dup_mask, ["Q", "estimator_norm", "N"]]
        raise ValueError(
            "Found multiple rows for same (Q, estimator, N), expected unique runs. "
            f"Examples:\n{dup_rows.head(8).to_string(index=False)}"
        )

    # EKF should be unique per Q and then repeated across all N rows in that Q group.
    ekf_dup_mask = ekf_df.duplicated(subset=["Q", "estimator_norm"], keep=False)
    if ekf_dup_mask.any():
        dup_rows = ekf_df.loc[ekf_dup_mask, ["Q", "estimator_norm"]]
        raise ValueError(
            "Found multiple EKF rows for same Q, expected unique runs. "
            f"Examples:\n{dup_rows.head(8).to_string(index=False)}"
        )

    pivot = (
        mhe_df.pivot(index=["Q", "N"], columns="estimator_norm", values="RMSE")
        .sort_index(axis=1)
    )

    # Broadcast EKF RMSE within each Q group for all N rows.
    if not ekf_df.empty:
        ekf_by_q = ekf_df.set_index("Q")["RMSE"].to_dict()
        pivot["EKF"] = [ekf_by_q.get(q, np.nan) for q, _ in pivot.index]

    # Sort rows by Q index (Q1, Q2, ...) then by N.
    def q_sort_key(q):
        if isinstance(q, str) and len(q) >= 2 and q[0] == "Q" and q[1:].isdigit():
            return int(q[1:])
        return q

    ordered_row_index = sorted(pivot.index, key=lambda x: (q_sort_key(x[0]), x[1]))
    pivot = pivot.loc[ordered_row_index]

    return pivot


def row_normalize_for_coloring(values_2d):
    """Map each row independently to [0, 1] for row-wise color scaling."""
    normalized = np.full(values_2d.shape, np.nan, dtype=float)

    for i in range(values_2d.shape[0]):
        row = values_2d[i, :]
        valid = np.isfinite(row)
        if not np.any(valid):
            continue

        rmin = np.min(row[valid])
        rmax = np.max(row[valid])

        if np.isclose(rmin, rmax):
            # All equal in this row -> neutral midpoint color.
            normalized[i, valid] = 0.5
        else:
            normalized[i, valid] = (row[valid] - rmin) / (rmax - rmin)

    return normalized


def plot_rmse_grid(pivot, estimator_label_map=None):
    if estimator_label_map is None:
        estimator_label_map = {}

    # Display as x=estimator, y=(Q, N) rows.
    display = pivot.copy()
    estimator_order = ["MHE (baseline)", "MHE (PCIP)", "MHE (PCIP+L1AO)", "EKF"]
    ordered_cols = [c for c in estimator_order if c in display.columns]
    remaining_cols = [c for c in display.columns if c not in ordered_cols]
    display = display[ordered_cols + remaining_cols]

    rmse = display.values.astype(float)

    # Exclude EKF from the color normalization scheme.
    ekf_col = None
    if "EKF" in display.columns:
        ekf_col = display.columns.get_loc("EKF")
    color_input = rmse.copy()
    if ekf_col is not None:
        color_input[:, ekf_col] = np.nan
    color_vals = row_normalize_for_coloring(color_input)

    cmap = plt.get_cmap("RdYlGn_r").copy()  # 0->green, 1->red
    cmap.set_bad(color="darkgray")

    fig, ax = plt.subplots(figsize=(7.8, 6.))
    ax.set_box_aspect(1)
    im = ax.imshow(color_vals, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", alpha=0.8)

    # Axis ticks and labels
    ax.set_xticks(np.arange(display.shape[1]))
    y_positions = np.arange(display.shape[0])
    ax.set_yticks(y_positions)
    ax.set_xticklabels([estimator_label_map.get(c, c) for c in display.columns], rotation=0, ha="center")
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("center")
        lbl.set_multialignment("center")
        lbl.set_rotation_mode("anchor")
    ax.set_yticklabels([
        f"$Q=Q_{int(q[1:])}$, $N={int(n) if float(n).is_integer() else n}$"
        for q, n in display.index
    ])

    # ax.set_xlabel("Estimator")
    # ax.set_ylabel("(Q, N)")
    # ax.set_title("RMSE Grid across Q and N, row-wise min=green, max=red")

    # Show x labels on top and hide the tick marks on both axes.
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=6)
    ax.tick_params(axis="y", left=False, right=False, length=0)

    # Cell borders
    # ax.set_xticks(np.arange(-0.5, display.shape[1], 1), minor=True)
    # ax.set_yticks(np.arange(-0.5, display.shape[0], 1), minor=True)
    # ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    # ax.tick_params(which="minor", bottom=False, left=False)

    # Add separators between Q groups.
    q_vals = [q for q, _ in display.index]
    for i in range(1, len(q_vals)):
        if q_vals[i] != q_vals[i - 1]:
            ax.axhline(i - 0.5, color="black", linewidth=1.2)

    # Annotate each non-EKF cell with RMSE value.
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if ekf_col is not None and j == ekf_col:
                continue
            v = rmse[i, j]
            if np.isfinite(v):
                norm_v = color_vals[i, j]
                txt_color = "white" if np.isfinite(norm_v) and (norm_v>0.8 or norm_v<0.15) else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", color=txt_color, fontsize=11)
            else:
                ax.text(j, i, "-", ha="center", va="center", color="black", fontsize=11)

    # For EKF, show one value spanning each Q group (e.g., 3 N-rows per Q group).
    if ekf_col is not None:
        q_vals = [q for q, _ in display.index]
        group_start = 0
        for i in range(1, len(q_vals) + 1):
            end_group = (i == len(q_vals)) or (q_vals[i] != q_vals[i - 1])
            if not end_group:
                continue

            group_end = i - 1
            group_vals = rmse[group_start:group_end + 1, ekf_col]
            finite_group_vals = group_vals[np.isfinite(group_vals)]
            if finite_group_vals.size:
                ekf_val = float(finite_group_vals[0])
                y_center = 0.5 * (group_start + group_end)
                ax.text(
                    ekf_col,
                    y_center,
                    f"{ekf_val:.4f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                )
            else:
                y_center = 0.5 * (group_start + group_end)
                ax.text(ekf_col, y_center, "-", ha="center", va="center", color="black", fontsize=11)

            group_start = i

    # Colorbar explains row-wise relative scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative RMSE within row")

    fig.tight_layout()
    plt.show()


def main():
    csv_path = os.path.join(os.path.dirname(__file__), "2026-04-10_scenario2_estimation_error.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    estimator_label_map = {
        # Customize estimator labels here.
        "EKF": rf"\textbf{{EKF}}",
        "MHE (baseline)": r"\shortstack{\textbf{MHE}\\\textbf{(CVXOPT)}}",
        "MHE (PCIP)": r"\shortstack{\textbf{MHE}\\\textbf{(PCIP)}}",
        "MHE (PCIP+L1AO)": r"\shortstack{\textbf{MHE}\\\textbf{(PCIP+$\mathcal{L}_1$-AO)}}",
    }

    pivot = build_rmse_grid(csv_path)
    plot_rmse_grid(pivot, estimator_label_map=estimator_label_map)


if __name__ == "__main__":
    main()
