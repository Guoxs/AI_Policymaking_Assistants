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
base_dir = 'outputs//5 states'  # 三个子目录都在这个目录下
state_list = [ 'mississippi']
warm_day = 10
for state in state_list:

    # 频率配置： (子目录名, 文件名后缀, 线条颜色, 标签)
    cadences = [
        # ('2 weeks',  '2 weeks',  '#E69F00', 'Agent (2 weeks)'),   # orange
        # ('4 weeks//gpt-3.5//metrics',  '4 weeks',  '#009E73', 'Agent (4 weeks)'),   # bluish green
        ('6 weeks//gpt-3.5//metrics',  '6 weeks',  '#D55E00', 'LLM Agent'),   # vermillion
        # ('8 weeks//gpt-3.5//metrics',  '8 weeks',  '#CC79A7', 'Agent (8 weeks)'),   # reddish purple
        # ('10 weeks//gpt-3.5//metrics', '10 weeks', '#1f77b4', 'Agent (10 weeks)') ,  # yellow (Okabe–Ito)
    ]

    # 指标配置
    metric_cols = ['incidence_rate_7d'] #'incidence_rate_7d', 'active_case_ratio', 'death_incidence_7d'
    y_labels    = ['IR (per 100k)']   #IR (per 100k)， ACR (per 100k)， DR (per 100k)
    
    for index in range(len(y_labels)):
        # 起始日期（基于 warm_day）
        start_date = pd.Timestamp('2020-04-12') + pd.Timedelta(days=warm_day)

        # ---------- 读取 Ground Truth（用第一个频率目录的 GT，保证日期长度一致） ----------
        gt_dir, gt_suffix, *_ = cadences[0]
        gt_folder = os.path.join(base_dir, gt_dir)
        metric_gt = pd.read_csv(os.path.join(gt_folder, f"{state}_metrics_gt_{gt_suffix}.csv"))
        metric_gt = metric_gt.reset_index(drop=True)
        metric_gt['date'] = pd.date_range(start=start_date, periods=len(metric_gt), freq='D')
        metric_gt['date'] = pd.to_datetime(metric_gt['date'])

        # ---------- 绘图 ----------
        plt.figure(figsize=(18, 6))

        # 画 GT
        col = metric_cols[index]
        # 画各频率的 mean ± std
        for subdir, suffix, color, label in cadences:
            folder = os.path.join(base_dir, subdir)

            # 读取本频率的 mean/std
            mean_df = pd.read_csv(os.path.join(folder, f"{state}_metrics_mean_{suffix}.csv")).reset_index(drop=True)
            std_df  = pd.read_csv(os.path.join(folder, f"{state}_metrics_std_{suffix}.csv")).reset_index(drop=True)

            # 对齐日期长度（防止不同长度导致越界）
            n = min(len(mean_df), len(std_df), len(metric_gt))
            mean_df = mean_df.iloc[:n].copy()
            std_df  = std_df.iloc[:n].copy()
            dates   = metric_gt['date'].iloc[:n]  # 用同一套 dates 对齐

            # 取列
            mean_s = mean_df[col].to_numpy()
            std_s  = np.clip(std_df[col].to_numpy(), a_min=0, a_max=None)
            # mean_s = pd.Series(mean_s).ewm(span=7, adjust=False).mean().to_numpy()
            std_s  = pd.Series(std_s).ewm(span=7, adjust=False).mean().clip(lower=0).to_numpy()
            # 画线 + 阴影
            plt.plot(dates, mean_s, label=label, color=color, linewidth=2.0, marker='*',
                     markersize=15,
                     markerfacecolor='white',
                     markeredgecolor=color,
                     markeredgewidth=1.0,
                     markevery=(32, 42))

            std_s_clipped = np.minimum(std_s, mean_s * 0.20)

            plt.fill_between(dates, np.clip(mean_s - std_s_clipped, a_min=0, a_max=None), mean_s + std_s_clipped,
                            color=color, alpha=0.18)
        interval = 42
        x_vals = dates[32::interval].values
        x_vals = np.insert(x_vals, 0, dates[0])  # 插入第一个标记点
        x_vals = np.append(x_vals, np.array([dates.iloc[-1]], dtype=x_vals.dtype))  # 插入最后一个标记点，保持数据类型一致
        for i in range(len(x_vals)):
            plt.axvline(x=x_vals[i], color='gray', linestyle='--', alpha=0.5)
            if i < len(x_vals) - 1:
                plt.axvspan(x_vals[i], x_vals[i+1], color=('lightblue' if i % 2 == 0 else 'lightyellow'), alpha=0.2)  
        plt.plot(metric_gt['date'].iloc[:n], metric_gt[col].iloc[:n], label='Ground Truth', color='#4D4D4D', linewidth=2.0)
        # 轴与图例
        plt.legend(frameon=False, fontsize=22, loc='upper left')
        plt.ylabel(y_labels[index], fontsize=25)
        plt.xlabel('Date', fontsize=25)
        plt.xticks(fontsize=22,rotation=15)
        plt.yticks(fontsize=22)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/5 states/mississippi/{state}_{y_labels[index]}.png")
        plt.close()


