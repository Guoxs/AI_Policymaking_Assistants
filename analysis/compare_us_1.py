####柱状图对比不同policy下的指标表现

import numpy as np
import os
import random
import pandas as pd
import os
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",      
    "axes.unicode_minus": False      
})

state_list = ['alabama','arizona', 'arkansas','idaho','indiana', 'iowa', 'kentucky', 'michigan','minnesota', 'mississippi', 'nebraska','new mexico','ohio','oklahoma','south carolina', 'tennessee', 'texas', 'utah','virginia','wisconsin']
state_abbr = ['AL','AZ','AR','ID','IN','IA','KY','MI','MN','MS','NE','NM','OH','OK','SC','TN','TX','UT','VA','WI']
data_folder = 'datasets/US/epedimic/'
suffix = 'cases.csv'
data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}


week_freqs = ["4 weeks", "8 weeks", 'ori_restriction', 'detection']
label_name = ['TIR (4 weeks)', 'TIR (8 weeks)', 'SIS', 'TIS']
results_root = "outputs\\US"
# 重新排序：ACR 放在第一个 (最上方)
metric_cols = ['active_case_ratio', 'incidence_rate_7d', 'death_incidence_7d']
y_labels = [
    'ACR (per 100K)',
    'IR (per 100k)',
    'DR (per 100K)'
]

# 配色方案
color_gt     = "#4F6A8A"   # Muted slate blue
color_4weeks = "#4F6A8A"  # Green-ish
color_8weeks = "#1f77b4"  # Warm ochre
color_ori    = "#B3D7BD"   # Bright orange
color_det    = "#B3A3CD"   # Muted purple
err_color    = "#333333"   # Dark grey for error bars

# 创建 3x1 的子图布局
fig, axes = plt.subplots(3, 1, figsize=(40, 16), sharex=True)

for ax_idx, metric_name in enumerate(metric_cols):
    ax = axes[ax_idx]
    ylabel = y_labels[ax_idx]

    gt_values, agent_values_4, agent_values_8, agent_ori, agent_dec = [], [], [], [], []
    agent_stds_4, agent_stds_8, agent_stds_ori, agent_stds_dec = [], [], [], []

    # 数据读取逻辑 (保持你原有的逻辑)
    for state in state_list:
        vals_gt_state = None
        vals_agent = {}
        for wf in week_freqs:
            results_folder = os.path.join(results_root, wf, "agent", 'metrics')
            metric_gt = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_gt_{wf}.csv"))
            metric_mean = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_mean_{wf}.csv"))
            metric_std = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_std_{wf}.csv"))
            
            if vals_gt_state is None:
                vals_gt_state = metric_gt[metric_cols].mean(numeric_only=True)
            
            vals_agent[wf] = (metric_mean[metric_cols].mean(numeric_only=True), 
                              metric_std[metric_cols].mean(numeric_only=True))

        gt_values.append(vals_gt_state[metric_name])
        agent_values_4.append(vals_agent['4 weeks'][0][metric_name])
        agent_stds_4.append(vals_agent['4 weeks'][1][metric_name])
        agent_values_8.append(vals_agent['8 weeks'][0][metric_name])
        agent_stds_8.append(vals_agent['8 weeks'][1][metric_name])
        agent_ori.append(vals_agent['ori_restriction'][0][metric_name])
        agent_stds_ori.append(vals_agent['ori_restriction'][1][metric_name])
        agent_dec.append(vals_agent['detection'][0][metric_name])
        agent_stds_dec.append(vals_agent['detection'][1][metric_name])

    # 绘制柱状图
    n_states = len(state_list)
    x = np.arange(n_states)
    width = 0.2

    # rects1 = ax.bar(x - width, gt_values, width, label='Ground Truth', color=color_gt, edgecolor='white', linewidth=0.5)
    # rects2 = ax.bar(x, agent_values_4, width, yerr=agent_stds_4, capsize=3, label=label_name[0], 
    #                 color=color_4weeks, edgecolor='white', linewidth=0.5, error_kw={'ecolor': err_color, 'elinewidth': 0.8})
    # rects3 = ax.bar(x + width, agent_values_8, width, yerr=agent_stds_8, capsize=3, label=label_name[1], 
    #                 color=color_8weeks, edgecolor='white', linewidth=0.5, error_kw={'ecolor': err_color, 'elinewidth': 0.8})
    # rects4 = ax.bar(x + 2*width, agent_ori, width, yerr=agent_stds_ori, capsize=3, label=label_name[2], 
    #                 color=color_ori, edgecolor='white', linewidth=0.5, error_kw={'ecolor': err_color, 'elinewidth': 0.8})
    # rects5 = ax.bar(x + 3*width, agent_dec, width, yerr=agent_stds_dec, capsize=3, label=label_name[3], 
    #                 color=color_det, edgecolor='white', linewidth=0.5, error_kw={'ecolor': err_color, 'elinewidth': 0.8})
    group_gap = 1.0
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width * group_gap

    err_kw = dict(ecolor=err_color, elinewidth=0.8, capsize=2, capthick=0.8)

    # GT：用黑色点 + 细线（或水平短线）
    gt_half_len = width * 0.55   # 短线长度（≈ 一个柱宽，可微调 0.45–0.65）

    for xi, yi in zip(x, gt_values):
        ax.hlines(
            y=yi,
            xmin=xi - gt_half_len*3,
            xmax=xi + gt_half_len*3,
            colors = color_gt,          # 深灰/近黑
            linestyles='--',
            linewidth=1.2,
            zorder=4
        )

    # 给 legend 一个“代理句柄”
    from matplotlib.lines import Line2D
    gt_handle = Line2D(
        [0], [0],
        color='#1A1A1A',
        linestyle='--',
        linewidth=1.2,
        label='Ground Truth'
    )

    # 或者画成水平短线（更克制）
    # ax.hlines(gt_values, x-0.18, x+0.18, colors='#1A1A1A', linewidth=1.0, label='Ground Truth', zorder=4)

    rects_4  = ax.bar(x + offsets[0], agent_values_4, width, yerr=agent_stds_4, label=label_name[0],
                    color=color_4weeks, edgecolor='#4D4D4D', linewidth=0.4, error_kw=err_kw, zorder=3)
    rects_8  = ax.bar(x + offsets[1], agent_values_8, width, yerr=agent_stds_8, label=label_name[1],
                    color=color_8weeks, edgecolor='#4D4D4D', linewidth=0.4, error_kw=err_kw, zorder=3)
    rects_ori= ax.bar(x + offsets[2], agent_ori,      width, yerr=agent_stds_ori, label=label_name[2],
                    color=color_ori, edgecolor='#4D4D4D', linewidth=0.4, error_kw=err_kw, zorder=3)
    rects_dec= ax.bar(x + offsets[3], agent_dec,      width, yerr=agent_stds_dec, label=label_name[3],
                    color=color_det, edgecolor='#4D4D4D', linewidth=0.4, error_kw=err_kw, zorder=3)
    # 细节修饰
    ax.set_ylabel(ylabel, fontsize=40, fontweight='bold')
    ax.tick_params(axis='y', labelsize=36)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0),useMathText=True)
    ax.yaxis.offsetText.set_fontsize(30)
    # 如果是最上方的子图，设置图例
    if ax_idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        handles = [gt_handle] + handles
        labels  = ['Ground Truth'] + labels
        ax.legend(handles, labels,
              loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=5,
              fontsize=55, frameon=False, columnspacing=1)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=4, 
        #           fontsize=50, frameon=False, columnspacing=2)

# 设置 x 轴
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(state_abbr, fontsize=36)
# axes[-1].set_xlabel('States', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.25) # 调整子图垂直间距

# 保存
out_fig_path = 'figures/US/US_states_combined_metrics_2.jpg'
plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
plt.show()