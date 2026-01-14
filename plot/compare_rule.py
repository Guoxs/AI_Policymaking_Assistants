#### 对比所提出的policy的rule based 规则 （死亡人数和 total inflected 人数）

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
output_dir = 'figures//5 states//rule compare'
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
### read rule based results
rule_dir = os.path.join(base_dir, 'rule_based')
rule_confirmed = None
rule_death = None
gt_confirmed = None
gt_death = None
for state in state_list:
    results_path = os.path.join(rule_dir, 'results', f"{state}_results.csv")
    df = pd.read_csv(results_path)
    q = df['Q_pred'].to_numpy(dtype=float)
    d = df['D_pred'].to_numpy(dtype=float)
    r = df['R_pred'].to_numpy(dtype=float)
    q_gt = df['Q_gt'].to_numpy(dtype=float)
    d_gt = df['D_gt'].to_numpy(dtype=float)
    if rule_confirmed is None:
        rule_confirmed = q.copy() + r.copy() + d.copy()
        rule_death = d.copy()
        gt_death = d_gt.copy()
    else:
        rule_confirmed += q + r + d
        rule_death += d
        gt_death += d_gt


#### read random results
random_dir = os.path.join(base_dir, 'random')
random_confirmed_mean, random_confirmed_std, random_death_mean, random_death_std = read_multiple_experiments(random_dir, state_list)


gt_dir = os.path.join(base_dir, 'no_action')
for state in state_list:
    results_path = os.path.join(gt_dir, 'results', f"{state}_results.csv")
    df = pd.read_csv(results_path)
    q_gt = df['Q_gt'].to_numpy(dtype=float)
    r_gt = df['R_pred'].to_numpy(dtype=float)
    d_gt = df['D_gt'].to_numpy(dtype=float)
    if gt_confirmed is None:
        gt_confirmed = q_gt.copy() + r_gt.copy() + d_gt.copy()
    else:
        gt_confirmed += q_gt + r_gt + d_gt

death_data = {
    'agent_mean': agent_death_mean,
    'agent_std': agent_death_std,
    'rule_mean': rule_death,
    'rule_std': rule_death,
    'gt_mean': gt_death,
    'gt_std': gt_death,
    'random_mean': random_death_mean,
    'random_std': random_death_std
}

confirmed_data = {
    'agent_mean': agent_confirmed_mean,
    'agent_std': agent_confirmed_std,
    'rule_mean': rule_confirmed,
    'rule_std': rule_confirmed,
    'gt_mean': gt_confirmed,
    'gt_std': gt_confirmed,
    'random_mean': random_confirmed_mean,
    'random_std': random_confirmed_std
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

for i in range(2):
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    y_data = confirmed_data if i == 0 else death_data

    # --- LLM Agent（主方法）---
    agent_color = "#1f77b4"
    ax.plot(
        x, y_data['agent_mean'],
        color=agent_color, linewidth=2.4,
        linestyle='-',
        marker='o', markersize=8, markevery=mark_every,
        markerfacecolor='white', markeredgecolor=agent_color, markeredgewidth=1.0,
        label="LLM Agent",
        zorder=4
    )
    ax.fill_between(
        x,
        np.clip(y_data['agent_mean'] - y_data['agent_std'], 0, None),
        y_data['agent_mean'] + y_data['agent_std'],
        alpha=0.14,
        zorder=1
    )

    # --- Ground Truth（参考真值）---
    gt_color = "black"
    ax.plot(
        x, y_data['gt_mean'],
        color=gt_color, linewidth=2.6,
        linestyle='-',
        label="Ground Truth",
        zorder=5
    )

    # --- Rule-based（基线对照）---
    rule_color = "#D55E00"
    ax.plot(
        x, y_data['rule_mean'],
        color=rule_color, linewidth=2.0,
        linestyle=DASH_RULE,
        marker='^', markersize=8, markevery=mark_every,
        markerfacecolor='white', markeredgecolor=rule_color, markeredgewidth=1.0,
        label="Expert-guided Policy",
        zorder=3
    )

    # --- Random（弱基线）---
    random_color = "#808080"
    ax.plot(
        x, y_data['random_mean'],
        color=random_color, linewidth=1.8,
        linestyle=DASH_RANDOM,
        marker='D', markersize=6, markevery=mark_every,
        markerfacecolor='white', markeredgecolor=random_color, markeredgewidth=0.9,
        label="Random",
        zorder=2
    )
    ax.fill_between(
        x,
        np.clip(y_data['random_mean'] - y_data['random_std'], 0, None),
        y_data['random_mean'] + y_data['random_std'],
        alpha=0.10,
        zorder=0
    )

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