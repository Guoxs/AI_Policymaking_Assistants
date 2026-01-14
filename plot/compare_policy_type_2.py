#### 对比所提出三类policy 累计死亡人数和感染人数total对比（柱状图—）

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- 全局字体 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})


base_dir = 'outputs//5 states//6 weeks'  # 三个子目录都在这个目录下
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
agent_dir = os.path.join(base_dir, 'gpt-3.5//')
output_dir = 'figures//5 states//policy type compare'
agent_dir_2 = 'outputs//5 states//ori_restriction//gpt-3.5//'
agent_dir_3 = 'outputs//5 states//detection//gpt-3.5//'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def read_statewise_mean_std(agent_dir, state_list):
    sub_exp_dirs = [
        f.path for f in os.scandir(agent_dir)
        if f.is_dir() and 'metrics' not in f.name
    ]

    # 收集：state -> list of (T,) arrays
    total_by_state = {s: [] for s in state_list}   # total = Q+R+D
    death_by_state = {s: [] for s in state_list}   # death = D

    for exp_dir in sub_exp_dirs:
        for state in state_list:
            fp = os.path.join(exp_dir, "results", f"{state}_results.csv")
            df = pd.read_csv(fp)

            q = df["Q_pred"].to_numpy(dtype=float)
            r = df["R_pred"].to_numpy(dtype=float)
            d = df["D_pred"].to_numpy(dtype=float)

            total_by_state[state].append(q + r + d)
            death_by_state[state].append(d)
    # 统计：state -> mean/std (T,)
    total_stats = {}
    death_stats = {}
    for state in state_list:
        total_mat = np.vstack(total_by_state[state])   # (E, T)
        death_mat = np.vstack(death_by_state[state])   # (E, T)

        total_stats[state] = {
            "mean": total_mat.mean(axis=0),
            "std":  total_mat.std(axis=0, ddof=1),
            "final_mean": total_mat[:, -1].mean(),
            "final_std":  total_mat[:, -1].std(ddof=1),
        }
        death_stats[state] = {
            "mean": death_mat.mean(axis=0),
            "std":  death_mat.std(axis=0, ddof=1),
            "final_mean": death_mat[:, -1].mean(),
            "final_std":  death_mat[:, -1].std(ddof=1),
        }
    return total_stats, death_stats

type_name = ['TIR', 'SIS', 'TIS']
total_stats, death_stats = read_statewise_mean_std(agent_dir, state_list)
total_stats_2, death_stats_2 = read_statewise_mean_std(agent_dir_2, state_list)
total_stats_3, death_stats_3 = read_statewise_mean_std(agent_dir_3, state_list)

agent_total_list = [total_stats, total_stats_2, total_stats_3]
agent_death_list = [death_stats, death_stats_2, death_stats_3]

gt_confirmed = {}
gt_death = {}
gt_dir = os.path.join(base_dir, 'no_action')
for state in state_list:
    results_path = os.path.join(gt_dir, 'results', f"{state}_results.csv")
    df = pd.read_csv(results_path)
    q_gt = df['Q_gt'].to_numpy(dtype=float)
    r_gt = df['R_pred'].to_numpy(dtype=float)
    d_gt = df['D_gt'].to_numpy(dtype=float)
    if state not in gt_confirmed:
        gt_confirmed[state] = q_gt + r_gt + d_gt
        gt_death[state] = d_gt
    else:
        gt_confirmed[state] += q_gt + r_gt + d_gt
        gt_death[state] += d_gt

total_stats_gt = {}
death_stats_gt = {}
for state in state_list:
    total_stats_gt[state] = {
        "final_mean": gt_confirmed[state][-1],
    }
    death_stats_gt[state] = {
        "final_mean": gt_death[state][-1],
    }



mpl.rcParams.update({
    "axes.labelsize": 30,      # x label, y label
    # 坐标轴刻度标签
    "xtick.labelsize": 22,
    "ytick.labelsize": 25,
    # 图例
    "legend.fontsize": 22,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

abbr = {
    "arizona": "AZ",
    "mississippi": "MS",
    "new mexico": "NM",
    "texas": "TX",
    "virginia": "VA"
}
x_labels = [abbr.get(s, s.upper()) for s in state_list]
x = np.arange(len(state_list))

def plot_final_bar_compare_3policies_with_gtline_pct(
    value_name: str,
    agent_list: list,    # 长度=3, 分别对应三种 policy 的 stats dict
    gt_stats: dict,      # gt stats dict
    ylabel: str,
    filename: str,
    state_list: list,
    x_labels: list,
    output_dir: str,
    type_name = ['TIR', 'SIS', 'TIS'],
):
    # =========================
    # 1) 数据提取
    # =========================
    assert len(agent_list) == 3, "agent_list 必须包含 3 个 policy 的统计结果（TIR/SIS/TIS）"

    means = []
    stds  = []
    for p in range(3):
        means.append(np.array([agent_list[p][s]["final_mean"] for s in state_list], dtype=float))
        stds.append( np.array([agent_list[p][s]["final_std"]  for s in state_list], dtype=float))

    gt_mean = np.array([gt_stats[s]["final_mean"] for s in state_list], dtype=float)

    means = np.vstack(means)  # (3, n_states)
    stds  = np.vstack(stds)   # (3, n_states)

    n_states = len(state_list)
    x = np.arange(n_states)

    # =========================
    # 2) 图形参数（Nature-like）
    # =========================
    fig_w, fig_h = 16.0, 6
    width = 0.22
    gap   = 0.06
    d = width + gap
    offsets = np.array([-d, 0.0, +d])

    colors = ["#1f77b4", "#B3D7BD", "#B3A3CD"]
    c_gt   = "#4D4D4D"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # =========================
    # 3) 三种 policy 柱状图
    # =========================
    bar_positions = []  # 记录每个 policy 的 x 位置，后面用于标注
    for p in range(3):
        pos = x + offsets[p]
        bar_positions.append(pos)
        ax.bar(
            pos, means[p], width=width,
            color=colors[p], alpha=0.90,
            edgecolor="black", linewidth=0.8,
            yerr=stds[p], capsize=3,
            label=type_name[p],
            zorder=3
        )

    # =========================
    # 4) GT 横线段（覆盖每组 3 根柱的宽度）
    # =========================
    group_left  = x + offsets.min() - width/2
    group_right = x + offsets.max() + width/2
    for i in range(n_states):
        if not np.isfinite(gt_mean[i]):
            continue
        ax.hlines(
            y=gt_mean[i],
            xmin=group_left[i],
            xmax=group_right[i],
            colors=c_gt,
            linewidth=2.2,
            linestyles='--',
            zorder=4,
            color=c_gt,
            label="Ground Truth" if i == 0 else None
        )
    # ax.plot([], [], color=c_gt, linestyles='--', label="Ground Truth")

    # =========================
    # 5) 百分比标注：每个州的三根柱各标一个 (relative to GT)
    # =========================
    eps = 1e-12
    gt_abs = np.maximum(np.abs(gt_mean), eps)

    # 统一的 y 方向 padding：基于全图尺度，避免贴着柱顶
    # 用“所有柱顶(含std)”和 GT 的最大值估计一个全局尺度
    y_scale = np.nanmax(np.concatenate([
        (means + np.nan_to_num(stds, nan=0.0)).ravel(),
        gt_mean.ravel()
    ]))
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0
    y_pad = 0.005 * y_scale  # 可调：0.015~0.05

    for i in range(n_states):
        g = gt_mean[i]
        # gt 接近 0：百分比无意义，跳过该州
        if not np.isfinite(g) or np.abs(g) < 1e-9:
            continue

        for p in range(3):
            a = means[p, i]
            s = stds[p, i] if np.isfinite(stds[p, i]) else 0.0
            pct = (a - g) / gt_abs[i] * 100.0

            # 标注位置：该柱上方（含误差线）
            x_pos = bar_positions[p][i]
            if i == 1:
                y_top = gt_mean[i] + y_pad * 4
            else:
                y_top = gt_mean[i] + y_pad
            if ylabel == r"Total Infected" :
                if i == 0:
                    y_top = gt_mean[i] + y_pad * 18
                elif i == 4:
                    y_top = gt_mean[i-1] + y_pad

            ax.text(
                x_pos, y_top,
                f"{pct:+.0f}%",
                ha="center", va="bottom",
                fontsize=20,
                color="black",
                zorder=6
            )

    # =========================
    # 6) 轴、网格与排版
    # =========================
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle="--", alpha=0.22, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if ylabel == r"Total Infected":
        ax.legend(frameon=False, loc="upper left", ncol=4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")

# -----------------------
# 图1：Total infected final
# -----------------------
plot_final_bar_compare_3policies_with_gtline_pct(
    value_name="total_infected_final",
    agent_list=agent_total_list,
    gt_stats=total_stats_gt,
    ylabel=r"Total Infected",
    filename="final_total_infected_agent_3_policy.png",
    state_list=state_list,
    x_labels=x_labels,
    output_dir=output_dir
)

# -----------------------
# 图2：Death final
# -----------------------
plot_final_bar_compare_3policies_with_gtline_pct(
    value_name="death_final",
    agent_list=agent_death_list,
    gt_stats=death_stats_gt,
    ylabel=r"Total Deaths",
    filename="final_deaths_agent_3_policy.png",
    state_list=state_list,
    x_labels=x_labels,
    output_dir=output_dir
)





    # plt.tight_layout()
    # plt.savefig(f'{output_dir}/compare_rule{"confirmed" if i==0 else "death"}.png', dpi=300)
    # plt.show()