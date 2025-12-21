import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================== #
file_path = os.path.join(os.path.dirname(__file__), '2025-12-06_scenario2_Q1_L1AO_33e0cf8b.npy')
enabled_estimators = ["EKF", "LMHE1", "LMHE4"]
tmin, tmax = 2000, 3000     # array indices
# =============================================================== #

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "figure.constrained_layout.use": True,
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150
})

LABEL_MAP = {
    "EKF": "EKF",
    "LMHE1": "MHE ($N=10$)",
    "LMHE2": "MHE ($N=20$)",
    "LMHE3": "MHE ($N=50$)",
    "LMHE4": "MHE ($N=100$)",
}

COLORS = {
    "EKF": "tab:red",
    "LMHE1": "tab:olive", 
    "LMHE2": "tab:green",
    "LMHE3": "tab:orange",
    "LMHE4": "tab:blue"
}

LINES = {
    "EKF": "solid",
    "LMHE1": (0, (5, 10)), 
    # "LMHE1": "dashed",
    "LMHE2": "solid",
    "LMHE3": "solid",
    "LMHE4": "solid"
}

STATE_NAMES = {
    0: r"$p_x$",
    1: r"$p_y$",
    2: r"$p_z$",
    3: r"$q_x$",
    4: r"$q_y$",
    5: r"$q_z$",
    6: r"$\phi$",
    7: r"$\theta$",
    8: r"$\psi$",
    9: r"$\Omega_x$",
    10: r"$\Omega_y$",
    11: r"$\Omega_z$",
}

XLABELS = {
    3: "Time (s)",
    4: "Time (s)",
    5: "Time (s)",
    9: "Time (s)",
    10: "Time (s)",
    11: "Time (s)",
}

YLABELS = {
    0: r"$p_x$ (m)",
    1: r"$p_y$ (m)",
    2: r"$p_z$ (m)",
    3: r"$q_x$ (m/s)",
    4: r"$q_y$ (m/s)",
    5: r"$q_z$ (m/s)",
    6: r"$\phi$ (rad)",
    7: r"$\theta$ (rad)",
    8: r"$\psi$ (rad)",
    9: r"$\Omega_x$ (rad/s)",
    10: r"$\Omega_y$ (rad/s)",
    11: r"$\Omega_z$ (rad/s)",
}

# %% Load data - only support .npy files with exactly 9 numpy arrays inside
def load_all_arrays(path, allow_pickle=False):
    arrays = []
    with open(path, 'rb') as f:
        while True:
            try:
                arr = np.load(f, allow_pickle=allow_pickle)
            except EOFError:
                break
            arrays.append(arr)
    return arrays

arrays = load_all_arrays(file_path, allow_pickle=False)
print(f'Found {len(arrays)} arrays')
for i, a in enumerate(arrays):
    print(i+1, a.shape, a.dtype)

tvec, xvec, yvec, uvec = arrays[:4]
if len(arrays)==9:
    xhat_ekf, xhat_lmhe1, xhat_lmhe2, xhat_lmhe3, xhat_lmhe4 = arrays[4:9]
else:
    raise Exception("Check what data is in this .npy file!")

# N = tvec.shape[0]
N = tmax - tmin + 1
norms = np.zeros((N,3))     # v_norm, theta_norm, omega_norm
for i in range(N):
    norms[i, 0] = np.linalg.norm(xvec[tmin+i, 3:6])
    norms[i, 1] = np.linalg.norm(xvec[tmin+i, 6:9])
    norms[i, 2] = np.linalg.norm(xvec[tmin+i, 9:12])
print(f"Max linear velocity   : {max(norms[:,0]):.4f} m/s")
print(f"Max Euler angles      : {np.rad2deg(max(norms[:,1])):.4f} deg")
print(f"Max angular velocity  : {max(norms[:,2]):.4f} rad/s")

# %% Plot data
def plot_state(
        idx, ylim=None,
        invert_y=False, rad2deg=False
    ):
    plt.plot(tvec[tmin:tmax], xvec[tmin:tmax,idx]*(180/np.pi if rad2deg else 1), 'k--', lw=1., label="true")
    if idx in [0, 1, 2, 9, 10, 11]:
        plt.plot(
            tvec[tmin:tmax], yvec[tmin:tmax, idx if idx<3 else idx-6]*(180/np.pi if rad2deg else 1),
            ls="", marker=".", markersize=3, markevery=1,
            c="black", alpha=.3,
            label="measured"
        )
    if 'EKF' in enabled_estimators:
        plt.plot(
            tvec[tmin:tmax], xhat_ekf[tmin:tmax,idx]*(180/np.pi if rad2deg else 1),
            color=COLORS.get("EKF"), ls=LINES.get("EKF"), lw=2., label=LABEL_MAP.get("EKF")
        )
    if 'LMHE1' in enabled_estimators:
        plt.plot(
            tvec[tmin:tmax], xhat_lmhe1[tmin:tmax,idx]*(180/np.pi if rad2deg else 1),
            color=COLORS.get("LMHE1"), ls=LINES.get("LMHE1"), lw=2., label=LABEL_MAP.get("LMHE1")
        )
    if 'LMHE2' in enabled_estimators:
        plt.plot(
            tvec[tmin:tmax], xhat_lmhe2[tmin:tmax,idx]*(180/np.pi if rad2deg else 1),
            color=COLORS.get("LMHE2"), ls=LINES.get("LMHE2"), lw=2., label=LABEL_MAP.get("LMHE2")
        )
    if 'LMHE3' in enabled_estimators:
        plt.plot(
            tvec[tmin:tmax], xhat_lmhe3[tmin:tmax,idx]*(180/np.pi if rad2deg else 1),
            color=COLORS.get("LMHE3"), ls=LINES.get("LMHE3"), lw=2., label=LABEL_MAP.get("LMHE3")
        )
    if 'LMHE4' in enabled_estimators:
        plt.plot(
            tvec[tmin:tmax], xhat_lmhe4[tmin:tmax,idx]*(180/np.pi if rad2deg else 1),
            color=COLORS.get("LMHE4"), ls=LINES.get("LMHE4"), lw=2., label=LABEL_MAP.get("LMHE4")
        )
    plt.grid()
    leg = plt.legend(loc='upper right')
    leg.get_frame().set_alpha(0.9)
    leg.set_draggable(True)
    plt.xlim(tvec[tmin], tvec[tmax])
    if ylim:   plt.ylim(ylim)
    plt.xlabel(XLABELS.get(idx))
    plt.ylabel(YLABELS.get(idx))
    if invert_y: plt.gca().invert_yaxis()
    # plt.title(state_name, fontsize=10)

ylims = [(min(xvec[tmin:tmax,i])-.1, max(xvec[tmin:tmax,i])+.1) for i in range(12)]

plt.figure(1)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plot_state(i)

plt.figure(2)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plot_state(i+6)

plt.show()

# %%
