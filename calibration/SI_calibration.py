##### 绘制补充材料中calibration simultaion部分结果

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
#data_folder = "D:\\MyDownload\\Code\\OD-COVID\\outputs\\baseline-5states\\results"
#state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']

data_folder = "D:\\MyDownload\\Code\\OD-COVID\\outputs\\baseline-us\\results"
state_list = ['utah', 'arkansas', 'iowa']

# df = pd.read_csv(f"{data_folder}\\{state}_results.csv")
gt_cols = ['Q_gt', 'D_gt']
base_cols = ['Q_pred', 'D_pred']

out_dir = "figures/5 states/SI_calibrate//"
os.makedirs(out_dir, exist_ok=True)

gt_cols = ['Q_gt', 'D_gt']
base_cols = ['Q_pred', 'D_pred']

# -----------------------------
# Nature-like plotting style
# -----------------------------
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "Times New Roman",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
})

# restrained, publication-friendly palette
COLOR_GT = "#222222"      # near-black
COLOR_BASE = "#1f77b4"    # blue (readable, not too saturated)

def _clean_pair(x, y):
    """Align and drop NaNs for metric computation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _get_x(df):
    """
    Use date column if exists; otherwise use sequential index.
    """
    for c in ["date", "Date", "day", "Day", "time", "Time"]:
        if c in df.columns:
            x = pd.to_datetime(df[c], errors="coerce")
            if x.notna().any():
                df = df.copy()
                df["_x_"] = x
                df = df.sort_values("_x_")
                return df, df["_x_"].values
    return df, np.arange(len(df))

# -----------------------------
# Main loop: one figure per state
# -----------------------------
for state in state_list:
    fp = os.path.join(data_folder, f"{state}_results.csv")
    df = pd.read_csv(fp)

    # Ensure needed columns exist
    needed = gt_cols + base_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"[{state}] Missing columns in CSV: {missing}")

    df, x = _get_x(df)

    # Prepare y
    Q_gt, Q_pred = df["Q_gt"].values, df["Q_pred"].values
    D_gt, D_pred = df["D_gt"].values, df["D_pred"].values

    # R^2
    qx, qy = _clean_pair(Q_gt, Q_pred)
    dx, dy = _clean_pair(D_gt, D_pred)
    r2_q = r2_score(qx, qy) if len(qx) >= 2 else np.nan
    r2_d = r2_score(dx, dy) if len(dx) >= 2 else np.nan

    # Figure
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 10), constrained_layout=True)

    # ---- Left: Q ----
    ax = axes[0]
    ax.plot(x, Q_gt, color=COLOR_GT, lw=1.6, label="GT")
    ax.plot(x, Q_pred, color=COLOR_BASE, lw=1.6, ls="--", label="Simulation")
    #ax.set_title("Quarantined (Q)")
    ax.set_ylabel("Quarantined Count", fontsize=30)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(20)  # Set fontsize for scientific notation (e.g., 1e6)
    ax.text(
        0.02, 0.96,
        rf"$R^2={r2_q:.3f}$" if np.isfinite(r2_q) else r"$R^2=$ NA",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=22,
        # bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", lw=0.8)
    )

    # ---- Right: D ----
    ax = axes[1]
    ax.plot(x, D_gt, color=COLOR_GT, lw=1.6, label="GT")
    ax.plot(x, D_pred, color=COLOR_BASE, lw=1.6, ls="--", label="Simulation")
    # ax.set_title("Deaths (D)")
    ax.set_ylabel("Deaths Count", fontsize=30)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(20)  # Set fontsize for scientific notation (e.g., 1e6)
    ax.text(
        0.02, 0.96,
        rf"$R^2={r2_d:.3f}$" if np.isfinite(r2_d) else r"$R^2=$ NA",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=22,
        # bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", lw=0.8)
    )

    # Shared styling
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)

    # One shared legend (clean)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10), fontsize=24)

    # State title
    # pretty_state = state.title()
    # fig.suptitle(pretty_state, y=1.18, fontsize=11)

    # Save
    out_png = os.path.join(out_dir, f"{state}_gt_vs_base_QD.png")
    #out_pdf = os.path.join(out_dir, f"{state}_gt_vs_base_QD.pdf")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

print(f"Done. Figures saved to: {out_dir}")