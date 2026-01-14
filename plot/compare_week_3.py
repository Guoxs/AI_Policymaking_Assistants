### 相比之前，此处绘制了所有state之和的表现（折线图）

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

# ---------- 配置 ----------
base_dir   = 'outputs//5 states'
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
warm_day   = 10

data_folder = 'datasets/5 states/'
suffix = 'cases_0412_1231.csv'
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}
pop_ratios = {state: pop_list[state] / sum(pop_list.values()) for state in state_list}
print("Population ratios:", pop_ratios)

# 频率配置： (子目录名, 文件名后缀, 线条颜色, 标签)
cadences = [
    ('4 weeks//gpt-3.5//metrics',  '4 weeks',  '#009E73', 'TIR (4 weeks)'),
    ('6 weeks//gpt-3.5//metrics',  '6 weeks',  '#D55E00', 'TIR (6 weeks)'),
    ('8 weeks//gpt-3.5//metrics',  '8 weeks',  '#CC79A7', 'TIR (8 weeks)'),
    ('10 weeks//gpt-3.5//metrics', '10 weeks', '#1f77b4', 'TIR (10 weeks)'),
]

# 指标配置
metric_cols = ['incidence_rate_7d', 'active_case_ratio', 'death_incidence_7d']
y_labels    = ['IR (per 100k)', 'ACR (per 100k)', 'DR (per 100k)']

# 输出目录
out_dir = "figures/5 states/weeks compare/all_states_sum"
os.makedirs(out_dir, exist_ok=True)

# 起始日期（基于 warm_day）
start_date = pd.Timestamp('2020-04-12') + pd.Timedelta(days=warm_day)

def _read_df(folder, fname):
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path).reset_index(drop=True)

def aggregate_sum_mean_std(folder, suffix, metric_col):
    """
    将 5 个州的 mean/std 按天聚合：
      mean_total(t) = sum_i mean_i(t)
      std_total(t)  = sqrt( sum_i std_i(t)^2 )  # 独立假设下的方差可加
    返回: mean_total(np.array), std_total(np.array), n(对齐后长度)
    """
    means = []
    stds  = []
    n_list = []

    for st in state_list:
        mean_df = _read_df(folder, f"{st}_metrics_mean_{suffix}.csv")
        std_df  = _read_df(folder, f"{st}_metrics_std_{suffix}.csv")
        n_list.append(min(len(mean_df), len(std_df)))
        means.append(mean_df[metric_col].to_numpy() * pop_ratios[st])  # 按人口比例加权
        stds.append(std_df[metric_col].to_numpy() * pop_ratios[st])    # 按人口比例加权

    n = min(n_list)
    means = np.vstack([m[:n] for m in means])              # shape: (S, n)
    stds  = np.vstack([np.clip(s[:n], 0, None) for s in stds])

    mean_total = means.sum(axis=0)
    std_total  = np.sqrt((stds ** 2).sum(axis=0))
    return mean_total, std_total, n

def aggregate_sum_gt(gt_folder, gt_suffix, metric_col):
    """
    将 5 个州的 GT 按天聚合：
      gt_total(t) = sum_i gt_i(t)
    返回: gt_total(np.array), dates(pd.Series), n(对齐后长度)
    """
    gts = []
    n_list = []

    for st in state_list:
        gt_df = _read_df(gt_folder, f"{st}_metrics_gt_{gt_suffix}.csv")
        n_list.append(len(gt_df))
        gts.append(gt_df[metric_col].to_numpy()* pop_ratios[st])  # 按人口比例加权

    n = min(n_list)
    gts = np.vstack([g[:n] for g in gts])
    gt_total = gts.sum(axis=0)

    dates = pd.date_range(start=start_date, periods=n, freq='D')
    return gt_total, pd.to_datetime(dates), n

# ---------- 读取 GT（用第一个 cadence 的 GT 作为日期基准） ----------
gt_dir, gt_suffix, *_ = cadences[0]
gt_folder = os.path.join(base_dir, gt_dir)

for idx, col in enumerate(metric_cols):
    # 聚合 GT
    gt_total, dates, n_gt = aggregate_sum_gt(gt_folder, gt_suffix, col)

    # ---------- 绘图 ----------
    plt.figure(figsize=(12, 8))

    # 画各频率的 mean ± std（sum版）
    for subdir, suffix, color, label in cadences:
        folder = os.path.join(base_dir, subdir)

        mean_total, std_total, n_ms = aggregate_sum_mean_std(folder, suffix, col)
        n = min(n_ms, n_gt)

        mean_s = mean_total[:n]
        std_s  = std_total[:n]

        # 平滑（保持你原来的 std 平滑逻辑）
        std_s = pd.Series(std_s).ewm(span=7, adjust=False).mean().clip(lower=0).to_numpy()

        # 画线
        plt.plot(dates[:n], mean_s, label=label, color=color, linewidth=2.0, marker='o',
                 markersize=10,
                 markerfacecolor='white',
                 markeredgecolor=color,
                 markeredgewidth=1.0,
                 markevery=7)

        # 阴影（延续你原来的裁剪规则）
        std_s_clipped = np.minimum(std_s, mean_s * 0.20)
        plt.fill_between(
            dates[:n],
            np.clip(mean_s - std_s_clipped, a_min=0, a_max=None),
            mean_s + std_s_clipped,
            color=color,
            alpha=0.18
        )

    # 画 GT
    plt.plot(dates[:n], gt_total[:n], label='Ground Truth', color='#4D4D4D', linewidth=2.0)

    # 轴与图例
    if col == 'incidence_rate_7d':
        plt.legend(frameon=False, fontsize=30, loc='upper left')
    plt.xticks(fontsize=28, rotation=15)
    plt.yticks(fontsize=32)
    plt.ylabel(y_labels[idx], fontsize=45)
    # ax.yaxis.get_offset_text().set_color(line_color)
    #plt.xlabel('Date', fontsize=45)
    ax = plt.gca()
    # 科学计数法（可选；如果不想用就删掉三行）
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存
    out_path = os.path.join(out_dir, f"all_states_sum_{y_labels[idx]}.jpg")
    plt.savefig(out_path, dpi=300)
    plt.close()
