### 单独20个州各自未来演化 (4 weeks vs 8 weeks, 折线图带预测)

# -*- coding: utf-8 -*-
"""
Per-state plots (one figure per state) with 180-day forecast:
- Top panel: Cumulative Infected Cases (Q+R+D)
- Bottom panel: Cumulative Death Cases (D)
Policies:
- TIR (4 weeks)
- TIR (8 weeks)
- SIS (ori_restriction)
- TIS (detection)
- Ground Truth (rule)

This script mirrors your aggregate plotting style, but computes curves per state.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------
# Global style (match your setup)
# -----------------------------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})

mpl.rcParams.update({
    "axes.labelsize": 26,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

# -----------------------------
# Paths & states
# -----------------------------
state_list = [
    'alabama','arizona','arkansas','idaho','indiana','iowa','kentucky','michigan',
    'minnesota','mississippi','nebraska','new mexico','ohio','oklahoma',
    'south carolina','tennessee','texas','utah','virginia','wisconsin'
]

base_dir_4w  = 'outputs//US//4 weeks'
base_dir_8w  = 'outputs//US//8 weeks'
base_dir_sis = 'outputs//US//ori_restriction'
base_dir_tis = 'outputs//US//detection'

agent_dir_4w  = os.path.join(base_dir_4w,  'agent')
agent_dir_8w  = os.path.join(base_dir_8w,  'agent')
agent_dir_sis = os.path.join(base_dir_sis, 'agent')
agent_dir_tis = os.path.join(base_dir_tis, 'agent')

gt_dir = os.path.join(base_dir_4w, 'rule')

output_dir = 'figures//US//SI'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Helpers: read experiments per state
# -----------------------------
def read_multiple_experiments_one_state(agent_dir: str, state: str):
    """
    Read multiple experiment folders under agent_dir and compute:
    - mean/std of cumulative confirmed (Q_pred+R_pred+D_pred)
    - mean/std of cumulative deaths (D_pred)
    Returns: (conf_mean, conf_std, death_mean, death_std) each shape (T,)
    """
    sub_exp_dirs = [f.path for f in os.scandir(agent_dir) if f.is_dir()]
    sub_exp_dirs = [d for d in sub_exp_dirs if 'metrics' not in os.path.basename(d).lower()]

    confirmed_all = []
    death_all = []

    for exp_dir in sub_exp_dirs:
        results_path = os.path.join(exp_dir, 'results', f"{state}_results.csv")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Missing file: {results_path}")

        df = pd.read_csv(results_path)
        q = df['Q_pred'].to_numpy(dtype=float)
        r = df['R_pred'].to_numpy(dtype=float)
        d = df['D_pred'].to_numpy(dtype=float)

        confirmed_all.append(q + r + d)
        death_all.append(d)

    confirmed_all = np.vstack(confirmed_all)  # (E, T)
    death_all = np.vstack(death_all)

    conf_mean = confirmed_all.mean(axis=0)
    conf_std  = confirmed_all.std(axis=0, ddof=1) if confirmed_all.shape[0] > 1 else np.zeros_like(conf_mean)

    death_mean = death_all.mean(axis=0)
    death_std  = death_all.std(axis=0, ddof=1) if death_all.shape[0] > 1 else np.zeros_like(death_mean)

    return conf_mean, conf_std, death_mean, death_std


def read_ground_truth_one_state(gt_dir: str, state: str):
    """
    Read GT (rule) results for one state:
    - cumulative confirmed: Q_gt + R_gt + D_gt
    - cumulative deaths: D_gt
    Returns: (gt_confirmed, gt_death) each shape (T,)
    """
    results_path = os.path.join(gt_dir, 'results', f"{state}_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing GT file: {results_path}")

    df = pd.read_csv(results_path)
    q_gt = df['Q_gt'].to_numpy(dtype=float)
    r_gt = df['R_gt'].to_numpy(dtype=float)
    d_gt = df['D_gt'].to_numpy(dtype=float)

    gt_confirmed = q_gt + r_gt + d_gt
    gt_death = d_gt
    return gt_confirmed, gt_death


# -----------------------------
# Forecast & plotting utilities (keep logic consistent with your aggregate plot)
# -----------------------------
def forecast_cumulative(data: np.ndarray, horizon: int = 180):
    """
    Linear extrapolation using mean daily increase over the last 14 days.
    Enforces nonnegative slope for cumulative series.
    """
    if len(data) < 15:
        daily_increase = np.diff(data).mean() if len(data) > 1 else 0.0
    else:
        daily_increase = np.diff(data[-14:]).mean()

    slope = max(daily_increase, 0.0)
    last_val = data[-1]
    steps = np.arange(1, horizon + 1)
    forecast_values = last_val + slope * steps
    return np.concatenate([data, forecast_values])


def plot_styled_errorbar_with_forecast(ax, x_all, mean_all, std_hist, color, label,
                                      history_len, mark_every=7, lw_hist=1.0, lw_fore=2.0):
    """
    Plot history (solid + errorbars) and forecast (dashed + sparse errorbars).
    std_hist is the historical std array with length == history_len.
    """
    x_hist = x_all[:history_len]
    x_fore = x_all[history_len-1:]  # include last history point for continuity

    m_hist = mean_all[:history_len]
    m_fore = mean_all[history_len-1:]

    s_hist = std_hist[:history_len]
    s_fore_val = float(s_hist[-1]) if len(s_hist) else 0.0

    # A) History error bars (sampled)
    ax.errorbar(
        x_hist[::mark_every], m_hist[::mark_every], yerr=s_hist[::mark_every],
        fmt='none', ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7, zorder=2
    )
    # History solid line
    ax.plot(x_hist, m_hist, color=color, linewidth=lw_hist, label=label, zorder=4)

    # B) Forecast dashed line
    ax.plot(x_fore, m_fore, color=color, linewidth=lw_fore,
            linestyle=(0, (3, 2)), alpha=0.7, zorder=3)

    # C) Forecast error bars (sparser)
    fore_idx = np.arange(0, len(x_fore), mark_every)
    ax.errorbar(
        x_fore[fore_idx], m_fore[fore_idx], yerr=s_fore_val,
        fmt='none', ecolor=color, elinewidth=0.8, capsize=2, alpha=0.2, zorder=1
    )


def format_axes(ax, ylabel, show_legend=False):
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(18)
    if show_legend:
        ax.legend(frameon=False, loc="upper left", fontsize=18)


# -----------------------------
# Main plotting loop (one figure per state)
# -----------------------------
def main():
    warm_day = 10
    start_date = pd.Timestamp('2020-04-12') + pd.Timedelta(days=warm_day)

    horizon = 180
    mark_every = 7

    # Colors (use your palette)
    color_4w  = "#C08A3E"
    color_8w  = "#1f77b4"
    color_sis = "#3C8D57"
    color_tis = "#B3A3CD"
    gt_color  = "#333333"

    y_labels = ['Cumulative Infected Cases', 'Cumulative Death Cases']

    infection_df = {}
    death_df = {}
    for state in state_list:
        # --- load per-state series (history) ---
        conf_4w_m, conf_4w_s, death_4w_m, death_4w_s = read_multiple_experiments_one_state(agent_dir_4w, state)
        conf_8w_m, conf_8w_s, death_8w_m, death_8w_s = read_multiple_experiments_one_state(agent_dir_8w, state)
        conf_sis_m, conf_sis_s, death_sis_m, death_sis_s = read_multiple_experiments_one_state(agent_dir_sis, state)
        conf_tis_m, conf_tis_s, death_tis_m, death_tis_s = read_multiple_experiments_one_state(agent_dir_tis, state)

        gt_conf, gt_death = read_ground_truth_one_state(gt_dir, state)

        # --- build x axis ---
        T = len(gt_conf)
        x = pd.date_range(start=start_date, periods=T, freq='D')
        x_forecast = pd.date_range(start=x[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
        x_all = x.append(x_forecast)
        history_len = T

        # --- store history data ---
        infection_df[state] = (gt_conf[-1] - conf_8w_m[-1])/gt_conf[-1]
        death_df[state] = (gt_death[-1] - death_8w_m[-1])/gt_death[-1]
        if state == state_list[-1]:
            infection_df = pd.DataFrame(infection_df, index = [0])
            death_df = pd.DataFrame(death_df, index = [0])
            death_df.to_csv(os.path.join(output_dir, 'US_death_history_8w.csv'))
            infection_df.to_csv(os.path.join(output_dir, 'US_infection_history_8w.csv'))

        # --- extend mean curves ---
        conf_4w_ext  = forecast_cumulative(conf_4w_m,  horizon)
        conf_8w_ext  = forecast_cumulative(conf_8w_m,  horizon)
        conf_sis_ext = forecast_cumulative(conf_sis_m, horizon)
        conf_tis_ext = forecast_cumulative(conf_tis_m, horizon)
        gt_conf_ext  = forecast_cumulative(gt_conf,   horizon)

        death_4w_ext  = forecast_cumulative(death_4w_m,  horizon)
        death_8w_ext  = forecast_cumulative(death_8w_m,  horizon)
        death_sis_ext = forecast_cumulative(death_sis_m, horizon)
        death_tis_ext = forecast_cumulative(death_tis_m, horizon)
        gt_death_ext  = forecast_cumulative(gt_death,    horizon)

        # --- figure: 2 panels (one figure per state) ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ax0, ax1 = axes

        # ===== Panel 1: confirmed =====
        plot_styled_errorbar_with_forecast(ax0, x_all, conf_4w_ext,  conf_4w_s,             color_4w,  "TIR (4 weeks)", history_len, mark_every)
        plot_styled_errorbar_with_forecast(ax0, x_all, conf_8w_ext,  conf_8w_s,             color_8w,  "TIR (8 weeks)", history_len, mark_every)
        plot_styled_errorbar_with_forecast(ax0, x_all, conf_sis_ext, conf_sis_s * 5.0,      color_sis, "SIS",          history_len, mark_every)
        plot_styled_errorbar_with_forecast(ax0, x_all, conf_tis_ext, conf_tis_s * 5.0,      color_tis, "TIS",          history_len, mark_every)

        ax0.plot(x_all[:history_len],      gt_conf_ext[:history_len],      color=gt_color, linewidth=2.6, label="Ground Truth", zorder=5)
        ax0.plot(x_all[history_len-1:],    gt_conf_ext[history_len-1:],    color=gt_color, linewidth=2.6,
                 linestyle=(0, (5, 5)), alpha=0.4, zorder=1)

        ax0.axvspan(x[history_len-1], x_all[-1], color='gray', alpha=0.05, label='Forecast (180d)')
        ax0.axvline(x[history_len-1], color='#333333', linestyle=':', linewidth=1.5, alpha=0.6)
        ax0.text(x[history_len-1], ax0.get_ylim()[1]*0.98, 'Simulation End ', ha='right', va='top', fontsize=16, alpha=0.6)

        format_axes(ax0, y_labels[0], show_legend=True)

        # ===== Panel 2: death =====
        plot_styled_errorbar_with_forecast(ax1, x_all, death_4w_ext,  death_4w_s,            color_4w,  "TIR (4 weeks)", history_len, mark_every)
        plot_styled_errorbar_with_forecast(ax1, x_all, death_8w_ext,  death_8w_s,            color_8w,  "TIR (8 weeks)", history_len, mark_every)
        plot_styled_errorbar_with_forecast(ax1, x_all, death_sis_ext, death_sis_s * 5.0,    color_sis, "SIS",          history_len, mark_every)
        plot_styled_errorbar_with_forecast(ax1, x_all, death_tis_ext, death_tis_s * 5.0,    color_tis, "TIS",          history_len, mark_every)

        ax1.plot(x_all[:history_len],      gt_death_ext[:history_len],     color=gt_color, linewidth=2.6, label="Ground Truth", zorder=5)
        ax1.plot(x_all[history_len-1:],    gt_death_ext[history_len-1:],   color=gt_color, linewidth=2.6,
                 linestyle=(0, (5, 5)), alpha=0.4, zorder=1)

        ax1.axvspan(x[history_len-1], x_all[-1], color='gray', alpha=0.05)
        ax1.axvline(x[history_len-1], color='#333333', linestyle=':', linewidth=1.5, alpha=0.6)

        format_axes(ax1, y_labels[1], show_legend=False)
        ax1.set_xlabel("Date", fontsize=24)

        # Title (state-level)
        #fig.suptitle(state.title(), fontsize=22, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = os.path.join(output_dir, f"{state.replace(' ', '_')}_forecast_180d.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    print(f"Done. Saved per-state figures to: {output_dir}")


if __name__ == "__main__":
    main()
