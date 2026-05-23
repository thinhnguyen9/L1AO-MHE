import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,                       # require TeX, dvipng/ghostscript installed
    "font.family": "serif",
    "font.serif": ["Computer Modern"],   # LaTeX default (Computer Modern)
    "mathtext.fontset": "cm",                  # math rendering to Computer Modern
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150
})

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

# =============================================================== #
file_path = os.path.join(os.path.dirname(__file__), '2026-04-10_scenario1.npy')
enabled_estimators = ["EKF", "MHE (OSQP)", "MHE (PCIP)", "MHE (PCIP+L1AO)"]
# =============================================================== #


def load_saved_arrays(npy_path, estimators):
	"""Load arrays saved sequentially with np.save in simulate_quadrotor.py order."""
	with open(npy_path, "rb") as f:
		tvec = np.load(f)
		xvec = np.load(f)
		yvec = np.load(f)
		uvec = np.load(f)

		xhat_map = {}
		for est in estimators:
			xhat_map[est] = np.load(f)

	return tvec, xvec, yvec, uvec, xhat_map


def plot_state_idx_vs_estimates(tvec, xvec, xhat_map, state_idx=6):
	plt.figure(figsize=(4, 4))
	plt.axhspan(
		np.rad2deg(1.5 * np.pi),
		np.rad2deg(2.5 * np.pi),
		color="gray",
		alpha=0.6,
		zorder=0,
	)
	# plt.axhline(np.rad2deg(1.5 * np.pi), color="gray", linestyle="-", linewidth=2.0, alpha=0.75, zorder=1)
	# plt.axhline(np.rad2deg(2.5 * np.pi), color="gray", linestyle="-", linewidth=2.0, alpha=0.75, zorder=1)
	plt.plot(tvec, np.rad2deg(xvec[:, state_idx]), "k--", lw=1.5, label="true")

	for est_name, xhat in xhat_map.items():
		plt.plot(
			tvec, np.rad2deg(xhat[:, state_idx]), 
			lw=2.0,
			label=plot_labels.get(est_name, est_name), 
			color=colors.get(est_name, None),
			marker=markers.get(est_name, None), 
			markersize=6, markevery=10
		)

	plt.xlabel("Time (s)")
	plt.ylabel("Roll angle $\phi$ (deg)")
	plt.xlim((-.0, .5))
	plt.grid(True, linestyle="-", color="black",linewidth=0.5, alpha=0.2)
	legend = plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0.0)
	legend.set_draggable(True)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"File not found: {file_path}")

	tvec, xvec, yvec, uvec, xhat_map = load_saved_arrays(file_path, enabled_estimators)
	plot_state_idx_vs_estimates(tvec, xvec, xhat_map, state_idx=6)

