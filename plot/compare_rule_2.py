#### 对比所提出的policy的rule based 规则 （绘制逐州的死亡人数，感染人数total对比）

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
output_dir = 'figures//5 states//rule compare final'
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

total_stats, death_stats = read_statewise_mean_std(agent_dir, state_list)

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

# --- 画图函数：一个 value 一张图 ---
def plot_final_bar_compare(
    value_name: str,
    agent_stats: dict,   # total_stats 或 death_stats
    gt_stats: dict,      # total_stats_gt 或 death_stats_gt
    ylabel: str,
    filename: str
):
    # 提取 final mean/std
    agent_mean = np.array([agent_stats[s]["final_mean"] for s in state_list], dtype=float)
    agent_std  = np.array([agent_stats[s]["final_std"]  for s in state_list], dtype=float)
    gt_mean    = np.array([gt_stats[s]["final_mean"]    for s in state_list], dtype=float)

    # --- 图形参数（Nature 风格：细柱、留白）---
    fig_w = 16.0   # “长图”
    fig_h = 4
    width = 0.26   # 细一点的柱子
    gap = 0.12     # 组内间距（视觉更清爽）

    # 两组柱的位置
    pos_agent = x - (width/2 + gap/2)
    pos_gt    = x + (width/2 + gap/2)

    # 颜色（克制）
    c_agent = "#1f77b4"  # blue
    c_gt    = "#4D4D4D"  # dark gray（GT用黑也可以，但灰更柔和）

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- Agent 柱 + 误差线 ---
    ax.bar(
        pos_agent, agent_mean, width=width,
        color=c_agent, alpha=0.90,
        edgecolor="black", linewidth=0.8,
        yerr=agent_std, capsize=3,
        label="LLM Agent",
        zorder=3
    )

    # --- GT 柱（无误差线）---
    ax.bar(
        pos_gt, gt_mean, width=width,
        color=c_gt, alpha=0.85,
        edgecolor="black", linewidth=0.8,
        label="Ground Truth",
        zorder=2
    )
    # --- 轴与网格（Nature-like）---
    eps = 1e-12
    pct_change = (agent_mean - gt_mean) / np.maximum(np.abs(gt_mean), eps) * 100.0
    # 用于控制标注高度（避免贴着柱顶）
    y_max = np.nanmax([agent_mean + agent_std, gt_mean])
    y_pad = 0.03 * y_max  # 上方留白（可调：0.02~0.06）
    for i_state in range(len(state_list)):
        a = agent_mean[i_state]
        g = gt_mean[i_state]
        aerr = agent_std[i_state] if np.isfinite(agent_std[i_state]) else 0.0

        # 如果 gt=0（或极小），百分比没有意义，跳过或写 N/A
        if np.abs(g) < 1e-9:
            continue
        pct = pct_change[i_state]
        # 文本：下降显示 “−xx.x%”，上升显示 “+xx.x%”
        txt = f"{pct:+.1f}%"
        # 文字颜色：下降用更醒目（rule of thumb：深色），上升也可用同色
        # 这里用黑色最稳妥（印刷友好）
        text_color = "#0B3C8C"
        # 标注位置：放在两根柱子的中间上方
        x_mid = pos_agent[i_state] 
        # 标注高度：取两根柱子的更高者，再加一点 padding
        y_top = a + agent_std[i_state] + y_pad

        ax.text(
            x_mid, y_top,
            txt,
            ha="center", va="bottom",
            fontsize=25, color=text_color,
            zorder=5
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.22, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 科学计数法（大数更清晰）
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


    # Legend（无边框，位置建议右上/左上）
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


# -----------------------
# 图1：Total infected final
# -----------------------
plot_final_bar_compare(
    value_name="total_infected_final",
    agent_stats=total_stats,
    gt_stats=total_stats_gt,
    ylabel=r"Total Infected",
    filename="final_total_infected_agent_vs_gt.png"
)

# -----------------------
# 图2：Death final
# -----------------------
plot_final_bar_compare(
    value_name="death_final",
    agent_stats=death_stats,
    gt_stats=death_stats_gt,
    ylabel=r"Total Deaths",
    filename="final_deaths_agent_vs_gt.png"
)





    # plt.tight_layout()
    # plt.savefig(f'{output_dir}/compare_rule{"confirmed" if i==0 else "death"}.png', dpi=300)
    # plt.show()