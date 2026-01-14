### 预测20个州未来演化 (4 weeks vs 8 weeks, 折线图带预测)


import numpy as np
from scipy.ndimage import gaussian_filter1d
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
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
agent_dir = os.path.join(base_dir, 'agent//')
output_dir = 'figures//US'
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 定义外推预测函数 ---
def forecast_cumulative(data, horizon=180):
    """
    针对累积数据进行外推。
    均值：取最后14天的平均日增量进行线性外推。
    """
    # 计算最近两周的日增量均值作为未来斜率
    daily_increase = np.diff(data[-14:]).mean()
    # 确保斜率不为负（累积量必须递增）
    slope = max(daily_increase, 0)
    
    last_val = data[-1]
    forecast_steps = np.arange(1, horizon + 1)
    forecast_values = last_val + slope * forecast_steps
    
    return np.concatenate([data, forecast_values])

# --- 2. 增强版的绘图函数 ---
def plot_styled_errorbar_with_forecast(ax, x_all, mean_all, std_all, color, label, history_len, mark_every):
    # 拆分历史与预测
    x_hist, x_fore = x_all[:history_len], x_all[history_len-1:]
    m_hist, m_fore = mean_all[:history_len], mean_all[history_len-1:]
    # std 在预测期保持为历史最后的水平（如你之前要求）
    s_hist = std_all[:history_len]
    s_fore_val = std_all[-1] 

    # A. 绘制历史部分（实线 + 带帽误差棒）
    ax.errorbar(
        x_hist[::mark_every], m_hist[::mark_every], yerr=s_hist[::mark_every],
        fmt='none', ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7, zorder=2
    )
    ax.plot(x_hist, m_hist, color=color, linewidth=2.0, label=label, zorder=4)

    # B. 绘制预测部分（虚线）
    ax.plot(x_fore, m_fore, color=color, linewidth=2.0, linestyle=(0, (3, 2)), alpha=0.7, zorder=3)
    
    # C. 绘制预测期误差棒（更稀疏以示区别）
    fore_idx = np.arange(0, len(x_fore), mark_every * 3)
    ax.errorbar(
        x_fore[fore_idx], m_fore[fore_idx], yerr=s_fore_val,
        fmt='none', ecolor=color, elinewidth=0.8, capsize=2, alpha=0.2, zorder=1
    )

# --- 3. 主循环绘制 ---
horizon = 180
history_len = len(x)
# 扩展日期轴
x_forecast = pd.date_range(start=x[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
x_all = x.append(x_forecast)

for i in range(2):
    fig, ax = plt.subplots(figsize=(10, 7))
    y_data = confirmed_data if i == 0 else death_data
    
    color_4w = "#3C8D57" 
    color_8w = "#C08A3E"  
    gt_color = "#4F6A8A"

    # 执行外推
    m4w_ext = forecast_cumulative(y_data['agent_mean_4w'], horizon)
    m8w_ext = forecast_cumulative(y_data['agent_mean_8w'], horizon)
    gt_ext  = forecast_cumulative(y_data['gt_mean'], horizon)
    
    # 绘制 Agent 曲线
    plot_styled_errorbar_with_forecast(ax, x_all, m4w_ext, y_data['agent_std_4w'], color_4w, "Agent (4 Weeks)", history_len, mark_every)
    plot_styled_errorbar_with_forecast(ax, x_all, m8w_ext, y_data['agent_std_8w'], color_8w, "Agent (8 Weeks)", history_len, mark_every)

    # 绘制 Ground Truth 曲线
    ax.plot(x_all[:history_len], gt_ext[:history_len], color=gt_color, linewidth=2.6, label="Ground Truth", zorder=5)
    ax.plot(x_all[history_len-1:], gt_ext[history_len-1:], color=gt_color, linewidth=2.6, linestyle=(0, (5, 5)), alpha=0.4, zorder=1)

    # --- Nature 风格修饰 ---
    # 阴影区标记预测范围
    ax.axvspan(x[history_len-1], x_all[-1], color='gray', alpha=0.05, label='Forecast (180d)')
    ax.axvline(x[history_len-1], color='#333333', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # 文本标注
    ax.text(x[history_len-1], ax.get_ylim()[1]*0.98, 'Simulation End ', ha='right', va='top', fontsize=18, alpha=0.6)

    # 轴格式化
    ax.set_ylabel(y_labels[i], fontsize=24, fontweight='bold')
    ax.set_xlabel("Date", fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 处理 Y 轴科学计数法
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(20)

    ax.legend(frameon=False, loc="upper left", fontsize=20)

    plt.tight_layout()
    # 自动保存
    save_path = f'{output_dir}/forecast_180d_{"confirmed" if i==0 else "death"}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()