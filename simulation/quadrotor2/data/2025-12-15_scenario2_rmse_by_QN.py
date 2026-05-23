import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

colors = {
    'EKF': 'tab:red',
    'MHE (CVXOPT)': 'tab:blue', 
    'MHE (PCIP)': 'tab:orange',
    'MHE (PCIP+L1AO)': 'tab:green',
}

plot_labels = {
    'EKF': 'EKF',
    'MHE (CVXOPT)': 'MHE (CVXOPT)', 
    'MHE (PCIP)': 'MHE (PCIP)',
    'MHE (PCIP+L1AO)': 'MHE (PCIP+$\mathcal{L}_1$-AO)',
}


def format_xtick_label(v):
	if isinstance(v, str) and len(v) >= 2 and v[0] == "Q" and v[1:].isdigit():
		return rf"$Q_{{{v[1:]}}}$"
	if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
		return str(int(v))
	return str(v)

def grouped_bar(ax, df, x_col, y_col, estimator_col, title, xlabel, ylabel, color_map, label_map):
	"""Draw grouped bar chart with one unique color per estimator."""
	x_vals = sorted(df[x_col].dropna().unique())
	estimators = sorted(df[estimator_col].dropna().unique())

	if len(x_vals) == 0 or len(estimators) == 0:
		raise ValueError(f"No data available for plot: {title}")

	x = np.arange(len(x_vals), dtype=float)
	n_est = len(estimators)
	width = 0.5 / max(n_est, 1)

	# Aggregate in case multiple rows exist for a given (x, estimator).
	agg = (
		df.groupby([x_col, estimator_col], as_index=False)[y_col]
		.mean()
	)

	for i, est in enumerate(estimators):
		vals = []
		est_df = agg[agg[estimator_col] == est]
		for xv in x_vals:
			row = est_df[est_df[x_col] == xv]
			vals.append(float(row[y_col].iloc[0]) if not row.empty else np.nan)

		vals = np.asarray(vals, dtype=float)
		xpos = x + (i - (n_est - 1) / 2.0) * width
		mask = np.isfinite(vals)
		if np.any(mask):
			ax.bar(
				xpos[mask],
				vals[mask],
				width=width*.8,
				color=color_map.get(est, None),
				edgecolor="none",
				linewidth=0.0,
				label=label_map.get(est, est),
				alpha=1.0,
				zorder=1,
			)

	ax.set_xticks(x)
	ax.set_xticklabels([format_xtick_label(v) for v in x_vals])
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_axisbelow(True)
	ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)


def main():
	csv_path = os.path.join(os.path.dirname(__file__), "2025-12-15_scenario2_summary.csv")
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	required = {"matrix_Q", "estimator", "MHE_horizon", "RMSE"}
	if not required.issubset(df.columns):
		raise ValueError(f"Missing required columns. Need: {sorted(required)}")

	df["MHE_horizon"] = pd.to_numeric(df["MHE_horizon"], errors="coerce")
	df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
	df = df.dropna(subset=["estimator", "matrix_Q", "RMSE"]).copy()

	# Use user-defined estimator colors and plot labels.
	color_map = colors
	label_map = plot_labels

	# (i) x-axis = MHE_horizon, filter matrix_Q == Q1.
	df_q1 = df[(df["matrix_Q"] == "Q1") & df["MHE_horizon"].notna()].copy()
	# (ii) x-axis = matrix_Q, filter MHE_horizon == 10.
	df_n10 = df[df["MHE_horizon"] == 10].copy()

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	grouped_bar(
		ax=ax1,
		df=df_q1,
		x_col="MHE_horizon",
		y_col="RMSE",
		estimator_col="estimator",
		# title="RMSE vs MHE horizon (matrix_Q = Q1)",
		title="$Q = 10^{-2}\mathbf{I}_n$",
		xlabel="MHE horizon $N$",
		ylabel="RMSE",
		color_map=color_map,
		label_map=label_map,
	)
	# ax1.legend()

	grouped_bar(
		ax=ax2,
		df=df_n10,
		x_col="matrix_Q",
		y_col="RMSE",
		estimator_col="estimator",
		# title="RMSE vs matrix_Q (MHE_horizon = 10)",
		title="$N = 10$",
		xlabel="Weighting matrix $Q$",
		ylabel=None,
		color_map=color_map,
		label_map=label_map,
	)
	ax2.legend()
	fig.tight_layout()

	plt.show()


if __name__ == "__main__":
	main()
