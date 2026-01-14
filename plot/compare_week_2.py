# import os
# import pandas as pd
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# # ---------- 全局字体 ----------
# mpl.rcParams.update({
#     "font.family": "Times New Roman",
#     "mathtext.fontset": "stix",
#     "axes.unicode_minus": False
# })

# base_dir = 'outputs//5 states'  # 三个子目录都在这个目录下
# state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
# state = state_list[1]  # 选择一个州
# warm_day = 10
# for state in state_list:

#     # 频率配置： (子目录名, 文件名后缀, 线条颜色, 标签)
#     cadences = [
#         # ('2 weeks',  '2 weeks',  '#E69F00', 'Agent (2 weeks)'),   # orange
#         ('4 weeks//gpt-3.5',  '4 weeks',  '#009E73', 'Agent (4 weeks)'),   # bluish green
#         ('6 weeks//gpt-3.5',  '6 weeks',  '#D55E00', 'Agent (6 weeks)'),   # vermillion
#         ('8 weeks//gpt-3.5',  '8 weeks',  '#CC79A7', 'Agent (8 weeks)'),   # reddish purple
#         ('10 weeks//gpt-3.5', '10 weeks', '#1f77b4', 'Agent (10 weeks)') ,  # yellow (Okabe–Ito)
#     ]

#     for i in range(cadences.__len__()):
#         subdir, suffix, color, label = cadences[i]
#         folder = os.path.join(base_dir, subdir)
#         subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
#         subfolders = [f for f in subfolders if 'metrics' not in f]
#         confirmed_list = []
#         death_list = []
#         for subfolder in subfolders:
#             temp_file = pd.read_csv(os.path.join(subfolder, 'results', f"{state}_results.csv"))
#             final_condition = temp_file.iloc[-1]
#             confirmed_list.append(final_condition['Q_pred'])
#             death_list.append(final_condition['D_pred'])
#         mean_confirmed = np.mean(confirmed_list)
#         std_confirmed = np.std(confirmed_list, ddof=1)
#         mean_death = np.mean(death_list)
#         std_death = np.std(death_list, ddof=1)





import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Nature-like 全局样式 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

base_dir = 'outputs//5 states'
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']

cadences = [
        # ('2 weeks',  '2 weeks',  '#E69F00', 'Agent (2 weeks)'),   # orange
        ('4 weeks//gpt-3.5',  '4 weeks',  '#009E73', 'Agent (4 weeks)'),   # bluish green
        ('6 weeks//gpt-3.5',  '6 weeks',  '#D55E00', 'Agent (6 weeks)'),   # vermillion
        ('8 weeks//gpt-3.5',  '8 weeks',  '#CC79A7', 'Agent (8 weeks)'),   # reddish purple
        ('10 weeks//gpt-3.5', '10 weeks', '#1f77b4', 'Agent (10 weeks)') ,  # yellow (Okabe–Ito)
    ]

#save_dir = 'figures//5 states//weeks compare final//'
save_dir = "figures/5 states/SI_week//"
os.makedirs(save_dir, exist_ok=True)
death = True
# 细柱宽度（你要更细）
bar_width = 0.35  # 可再小：0.35 / 0.30

for state in state_list:
    means, stds, xticklabels, colors = [], [], [], []

    for subdir, week_label, color, legend_label in cadences:
        folder = os.path.join(base_dir, subdir)
        if not os.path.exists(folder):
            means.append(np.nan); stds.append(np.nan)
            xticklabels.append(week_label); colors.append(color)
            continue

        # 你的 metrics 目录下再枚举子文件夹（每次运行一个子文件夹）
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

        total_list = []
        for subfolder in subfolders:
            fp = os.path.join(subfolder, 'results', f"{state}_results.csv")
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            final_row = df.iloc[-1]
            if death:
                total = float(final_row['D_pred'])
            else:
                total = float(final_row['Q_pred']) + float(final_row['D_pred']) + float(final_row['R_pred'])
            total_list.append(total)

        xticklabels.append(week_label)
        colors.append(color)

        if len(total_list) == 0:
            means.append(np.nan); stds.append(np.nan)
        else:
            means.append(np.mean(total_list))
            stds.append(np.std(total_list, ddof=1) if len(total_list) > 1 else 0.0)

    means = np.array(means, dtype=float)
    stds  = np.array(stds, dtype=float)
    x = np.arange(len(cadences))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    # --- 细柱：每个 cadence 一个颜色 ---
    bars = ax.bar(
        x, means,
        width=bar_width,
        yerr=stds,
        capsize=3,
        linewidth=0.8,
        edgecolor='black',
        alpha=0.90,
        zorder=2
    )
    for b, c in zip(bars, colors):
        b.set_facecolor(c)

    # --- 趋势线（克制）：黑色细线 + 空心圆点（不抢 cadence 颜色）---
    ax.plot(
        x, means,
        linewidth=1.4,
        color='black',
        marker='o',
        markersize=4.0,
        markerfacecolor='white',
        markeredgecolor='black',
        markeredgewidth=1.0,
        zorder=3
    )

    # --- 坐标与标题 ---
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    # ax.set_title(state.capitalize(), fontsize=13, pad=8)

    # --- 网格与边框（Nature-like） ---
    ax.grid(axis='y', linestyle='--', alpha=0.25, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 科学计数法（大数更清晰）
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(20)  # Set fontsize for scientific notation (e.g., 1e6)

    # --- Legend（显示每个 cadence 的颜色含义）---
    from matplotlib.patches import Patch
    # if state == 'arizona':
    #     print("debug")
    #     ax.set_ylim(0.4 * 10**6, 1.3 * 10**6)
#     legend_handles = [
#     Patch(facecolor=cadences[i][2], edgecolor='black', label=cadences[i][3])
#     for i in range(len(cadences))
# ]
#     ax.legend(
#         handles=legend_handles,
#         loc='upper left',
#         frameon=False,
#         fontsize=17
#     )
    ax.set_xlabel('Agent Policy update cadence (weeks)', fontsize=25)
    if death:
        ax.set_ylabel(r'Total deaths', fontsize=25)
    else:
        ax.set_ylabel(r'Total infected', fontsize=25)
    # elif state == 'virginia':
    #     ax.set_ylim(1 * 10**6, 2.5 * 10**6)
    plt.tight_layout()
    if death:
        out_path = os.path.join(save_dir, f"{state}_total_deaths_weekcolor.jpg")
    else:
        out_path = os.path.join(save_dir, f"{state}_total_infected_weekcolor.jpg")
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
print(f"Saved figures to: {save_dir}")
