#### 20个州 感染和死亡人数演化对比

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
# print("CWD =", os.getcwd())
# ---------- 全局字体 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})


base_dir = 'outputs//US//4 weeks'  # 三个子目录都在这个目录下
state_list = ['alabama','arizona', 'arkansas', 'idaho','indiana', 'iowa', 'kentucky', 'michigan','minnesota', 'mississippi', 'nebraska','new mexico','ohio','oklahoma','south carolina', 'tennessee', 'texas', 'utah','virginia','wisconsin']
agent_dir = os.path.join(base_dir, 'agent//')
output_dir = 'figures//US//SI'
def read_multiple_experiments(agent_dir, state_list):
    sub_exp_dirs = [f.path for f in os.scandir(agent_dir) if f.is_dir()]
    sub_exp_dirs = [f for f in sub_exp_dirs if 'metrics' not in f]
    confirmed_all = []  # list of (T,) arrays
    death_all = []
    for exp_dir in sub_exp_dirs:
        confirmed_sum = None
        death_sum = None
        for state in state_list:
            results_path = os.path.join(exp_dir, 'results', f"{state}_results.csv")
            df = pd.read_csv(results_path)
            q = df['Q_pred'].to_numpy(dtype=float)
            r = df['R_pred'].to_numpy(dtype=float)
            d = df['D_pred'].to_numpy(dtype=float)
            if confirmed_sum is None:
                confirmed_sum = q.copy() + r.copy() + d.copy()
                death_sum = d.copy()
            else:
                confirmed_sum += q + r + d
                death_sum += d
        confirmed_all.append(confirmed_sum)
        death_all.append(death_sum)
    # (E, T)
    confirmed_all = np.vstack(confirmed_all)
    death_all = np.vstack(death_all)
    # ---- experiment 维度统计 ----
    agent_confirmed_mean = confirmed_all.mean(axis=0)
    agent_confirmed_std  = confirmed_all.std(axis=0, ddof=1)
    agent_death_mean = death_all.mean(axis=0)
    agent_death_std  = death_all.std(axis=0, ddof=1)
    return agent_confirmed_mean, agent_confirmed_std, agent_death_mean, agent_death_std

agent_confirmed_mean, agent_confirmed_std, agent_death_mean, agent_death_std = read_multiple_experiments(agent_dir, state_list)

base_dir_8w = 'outputs//US//8 weeks'  # 三个子目录都在这个目录下
### read rule based results
agent_dir_8w = os.path.join(base_dir_8w, 'agent')
agent_confirmed_mean_8w, agent_confirmed_std_8w, agent_death_mean_8w, agent_death_std_8w = read_multiple_experiments(agent_dir_8w, state_list)


gt_dir = 'outputs//US//4 weeks//rule' 
gt_confirmed = None
gt_death = None
for state in state_list:
    results_path = os.path.join(gt_dir, 'results', f"{state}_results.csv")
    df = pd.read_csv(results_path)
    q_gt = df['Q_gt'].to_numpy(dtype=float)
    d_gt = df['D_gt'].to_numpy(dtype=float)
    r_gt = df['R_gt'].to_numpy(dtype=float)
    if gt_confirmed is None:
        gt_confirmed = q_gt.copy() + r_gt.copy() + d_gt.copy()
        gt_death = d_gt.copy()
    else:
        gt_confirmed += q_gt + r_gt + d_gt
        gt_death += d_gt


# #### read random results
# random_dir = os.path.join(base_dir, 'random')
# random_confirmed_mean, random_confirmed_std, random_death_mean, random_death_std = read_multiple_experiments(random_dir, state_list)


# gt_dir = os.path.join(base_dir, 'no_action')
# for state in state_list:
#     results_path = os.path.join(gt_dir, 'results', f"{state}_results.csv")
#     df = pd.read_csv(results_path)
#     q_gt = df['Q_gt'].to_numpy(dtype=float)
#     r_gt = df['R_pred'].to_numpy(dtype=float)
#     d_gt = df['D_gt'].to_numpy(dtype=float)
#     if gt_confirmed is None:
#         gt_confirmed = q_gt.copy() + r_gt.copy() + d_gt.copy()
#     else:
#         gt_confirmed += q_gt + r_gt + d_gt

death_data = {
    'agent_mean_4w': agent_death_mean,
    'agent_std_4w': agent_death_std,
    'agent_mean_8w': agent_death_mean_8w,
    'agent_std_8w': agent_death_std_8w,
    'gt_mean': gt_death,
    # 'random_mean': random_death_mean,
    # 'random_std': random_death_std
}

confirmed_data = {
    'agent_mean_4w': agent_confirmed_mean,
    'agent_std_4w': agent_confirmed_std,
    'agent_mean_8w': agent_confirmed_mean_8w,
    'agent_std_8w': agent_confirmed_std_8w,
    'gt_mean': gt_confirmed,
    'gt_std': gt_confirmed,
    # 'random_mean': random_confirmed_mean,
    # 'random_std': random_confirmed_std
}

mpl.rcParams.update({
    "axes.labelsize": 26,      # x label, y label
    # 坐标轴刻度标签
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    # 图例
    "legend.fontsize": 20,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

# x 轴（如果你有 dates，用 dates；否则用时间步）
warm_day = 10
start_date = pd.Timestamp('2020-04-12') + pd.Timedelta(days=warm_day)
T = len(agent_confirmed_mean)
x = pd.date_range(start=start_date, periods=T, freq='D')
y_labels = ['Cumulative Infected Cases', 'Cumulative Death Cases']


mark_every = max(T // 12, 1)

# 统一 dash patterns（比 '--' 更 Nature）
DASH_RULE   = (0, (4, 2))   # 短虚线：精致
DASH_RANDOM = (0, (1, 2))   # 点线：弱化且清晰
mark_every = 3       # 采样频率，根据你的数据量调整，避免太拥挤

def plot_styled_errorbar(ax, x, mean, std, color, label):
    # 1. 绘制带帽误差棒 (Error Bars with Caps)
    # 使用 thin line 和 small caps 增加精致感
    ax.errorbar(
        x[::mark_every], 
        mean[::mark_every], 
        yerr=std[::mark_every],
        fmt='none',                # 不在 errorbar 函数里连线
        ecolor=color,              # 误差棒颜色
        elinewidth=1.0,           # 误差棒线条粗细
        capsize=3,                # 上下短横线（帽）的大小
        capthick=1.0,             # 帽的厚度
        alpha=0.4,                # 较低透明度，使背景收敛趋势柔和
        markevery=mark_every*5,
        zorder=2
    )
    
    # 2. 绘制均值主线与实心圆点
    # 参考图中是实心点，如果需要空心，可设置 markerfacecolor='white'
    ax.plot(
        x, mean,
        color=color, 
        linewidth=1.8,
        linestyle='-',
        marker='o', 
        markersize=5, 
        markevery=mark_every,
        markerfacecolor=color,    # 实心点
        markeredgecolor='white',  # 给点加一圈白边，使其在重叠时更清晰
        markeredgewidth=0.5,
        label=label,
        zorder=4
    )

for i in range(2):
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    y_data = confirmed_data if i == 0 else death_data
    color_4w = "#3C8D57" 
    color_8w = "#C08A3E"  
    # mark_every = mark_every
    # color_4w = "#3C8D57"  # 绿色系
    # color_8w = "#704E85"  # 紫色系
    plot_styled_errorbar(ax, x, y_data['agent_mean_4w'], y_data['agent_std_4w'], color_4w, "Agent (4 Weeks)")
    plot_styled_errorbar(ax, x, y_data['agent_mean_8w'], y_data['agent_std_8w'], color_8w, "Agent (8 Weeks)")

    # 修饰 ax
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.2)
    # --- LLM Agent（主方法）---
    # agent_color = "#1f77b4"
    # ax.plot(
    #     x, y_data['agent_mean_4w'],
    #     color=agent_color, linewidth=2.4,
    #     linestyle='-',
    #     marker='o', markersize=8, markevery=mark_every,
    #     markerfacecolor='white', markeredgecolor=agent_color, markeredgewidth=1.0,
    #     label="Agent(4 Weeks)",
    #     zorder=4
    # )
    # ax.fill_between(
    #     x,
    #     np.clip(y_data['agent_mean_4w'] - y_data['agent_std_4w'], 0, None),
    #     y_data['agent_mean_4w'] + y_data['agent_std_4w'],
    #     alpha=0.14,
    #     zorder=1
    # )
    # agent_color = "#1f77b4"
    # ax.plot(
    #     x, y_data['agent_mean_8w'],
    #     color=agent_color, linewidth=2.4,
    #     linestyle='-',
    #     marker='o', markersize=8, markevery=mark_every,
    #     markerfacecolor='white', markeredgecolor=agent_color, markeredgewidth=1.0,
    #     label="Agent(8 Weeks)",
    #     zorder=4
    # )
    # ax.fill_between(
    #     x,
    #     np.clip(y_data['agent_mean_8w'] - y_data['agent_std_8w'], 0, None),
    #     y_data['agent_mean_8w'] + y_data['agent_std_8w'],
    #     alpha=0.14,
    #     zorder=1
    # )

    # --- Ground Truth（参考真值）---
    gt_color = "#4F6A8A" 
    ax.plot(
        x, y_data['gt_mean'],
        color=gt_color, linewidth=2.6,
        linestyle='-',
        label="Ground Truth",
        zorder=5
    )

    # # --- Rule-based（基线对照）---
    # rule_color = "#D55E00"
    # ax.plot(
    #     x, y_data['rule_mean'],
    #     color=rule_color, linewidth=2.0,
    #     linestyle=DASH_RULE,
    #     marker='^', markersize=8, markevery=mark_every,
    #     markerfacecolor='white', markeredgecolor=rule_color, markeredgewidth=1.0,
    #     label="Expert-guided Policy",
    #     zorder=3
    # )

    # # --- Random（弱基线）---
    # random_color = "#808080"
    # ax.plot(
    #     x, y_data['random_mean'],
    #     color=random_color, linewidth=1.8,
    #     linestyle=DASH_RANDOM,
    #     marker='D', markersize=6, markevery=mark_every,
    #     markerfacecolor='white', markeredgecolor=random_color, markeredgewidth=0.9,
    #     label="Random",
    #     zorder=2
    # )
    # ax.fill_between(
    #     x,
    #     np.clip(y_data['random_mean'] - y_data['random_std'], 0, None),
    #     y_data['random_mean'] + y_data['random_std'],
    #     alpha=0.10,
    #     zorder=0
    # )

    ax.set_xlabel("Date")
    ax.set_ylabel(y_labels[i])

    ax.grid(axis='y', linestyle='--', alpha=0.22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(18)
    ax.tick_params(axis='y', labelsize=20)
    # ax.tick_params(axis='both', labelsize=12)

    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/compare_rule{"confirmed" if i==0 else "death"}.png', dpi=300)
    plt.show()