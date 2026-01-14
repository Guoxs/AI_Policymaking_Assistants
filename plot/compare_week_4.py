### 绘制各个州总感染人数差异的柱状图

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
    ('4 weeks//gpt-3.5',  '4 weeks',  '#009E73', 'Agent (4 weeks)'),
    ('6 weeks//gpt-3.5',  '6 weeks',  '#D55E00', 'Agent (6 weeks)'),
    ('8 weeks//gpt-3.5',  '8 weeks',  '#CC79A7', 'Agent (8 weeks)'),
    ('10 weeks//gpt-3.5', '10 weeks', '#1f77b4', 'Agent (10 weeks)'),
]

save_dir = 'figures//5 states//weeks compare final//'
os.makedirs(save_dir, exist_ok=True)

# 细柱宽度（保持你原设置）
bar_width = 0.35

means, stds, xticklabels, colors = [], [], [], []

for subdir, week_label, color, legend_label in cadences:
    folder = os.path.join(base_dir, subdir)
    xticklabels.append(week_label)
    colors.append(color)

    if not os.path.exists(folder):
        means.append(np.nan); stds.append(np.nan)
        continue

    # cadence 下的多个实验子文件夹（每次运行一个 subfolder）
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    # 每个 run 的 “all-states total infected”
    total_list = []

    for subfolder in subfolders:
        run_total = 0.0
        ok = True

        # 对这个 run，把 5 个州的 final total 相加
        for state in state_list:
            fp = os.path.join(subfolder, 'results', f"{state}_results.csv")
            if not os.path.exists(fp):
                ok = False
                break

            df = pd.read_csv(fp)
            final_row = df.iloc[-1]
            run_total += float(final_row['Q_pred']) + float(final_row['D_pred']) + float(final_row['R_pred'])

        if ok:
            total_list.append(run_total)

    if len(total_list) == 0:
        means.append(np.nan); stds.append(np.nan)
    else:
        means.append(np.mean(total_list))
        stds.append(np.std(total_list, ddof=1) if len(total_list) > 1 else 0.0)

means = np.array(means, dtype=float)
stds  = np.array(stds, dtype=float)
x = np.arange(len(cadences)) * 0.75 

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

# --- 趋势线（克制）：黑色细线 + 空心圆点 ---
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

# --- 坐标与标题（保持你原风格，默认不加 title） ---
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=28)
ax.tick_params(axis='y', labelsize=26)

# --- 网格与边框（Nature-like） ---
ax.grid(axis='y', linestyle='--', alpha=0.25, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel(r'Total infected', fontsize=35)
# 科学计数法
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.yaxis.offsetText.set_fontsize(20)

plt.tight_layout()
out_path = os.path.join(save_dir, "ALL_STATES_SUM_total_infected_weekcolor.jpg")
plt.savefig(out_path, dpi=300)
plt.show()
plt.close(fig)

print(f"Saved figure to: {out_path}")
