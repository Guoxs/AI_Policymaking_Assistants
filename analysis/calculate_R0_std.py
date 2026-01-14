import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import gamma as gamma_dist
from datetime import date
import matplotlib.pyplot as plt
import os
from agents import state


data_folder = 'datasets/'
suffix = 'cases_0412_1231.csv'
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}

def estimate_Rt(gt_df, cols=['Q_gt', 'D_gt', 'R_gt'], window=21, pop = 100000, eps = 1e-12):
    gt_df = gt_df.copy()
    gt_df['Confirmed'] = gt_df[cols].sum(axis=1)
    gt_df['new_cases'] = gt_df['Confirmed'].diff().fillna(0)
    gt_df.loc[gt_df['new_cases'] < 0, 'new_cases'] = 0  # 修正负值
    # 计算每10万人发病率
    gt_df['incidence_rate'] = gt_df['new_cases'] / pop * 100000
    gt_df['incidence_rate_7d'] = gt_df['incidence_rate'].rolling(window=7, min_periods=1).mean()
    # 设定代际时间分布参数
    mean_si, sd_si = 5, 2   ##从感染者发病，到其传染的下一代人发病的时间间隔
    shape = (mean_si / sd_si)**2
    scale = (sd_si**2) / mean_si
    # 离散化 1-20 天
    w = [gamma.cdf(s+1, a=shape, scale=scale) - gamma.cdf(s, a=shape, scale=scale)
        for s in range(20)]
    w = np.array(w)
    w /= w.sum()
    a, b = 1, 1   # Gamma先验
    I = gt_df['new_cases'].to_numpy()
    Rt_mean, Rt_low, Rt_high = [], [], []
    dates = []
    for t in range(window, len(I)):
        Lambda = []
        for u in range(t-window, t):
            lam = sum(I[u-(s+1)] * w[s] for s in range(len(w)) if u-(s+1) >= 0)
            Lambda.append(lam)
        I_sum = I[t-window:t].sum()
        Lambda_sum = np.sum(Lambda)
        # 后验参数
        shape_post = a + I_sum
        rate_post  = b + max(Lambda_sum, eps)
        # 点估计和区间
        mean_R = shape_post / rate_post
        ci_low, ci_high = gamma_dist.ppf([0.025, 0.975], shape_post, scale=1/rate_post)
        Rt_mean.append(mean_R)
        Rt_low.append(ci_low)
        Rt_high.append(ci_high)
        dates.append(t)
    rt_df = pd.DataFrame({
        't': dates,
        'R_mean': Rt_mean,
        'R_low': Rt_low,
        'R_high': Rt_high
    })
    return rt_df, gt_df['incidence_rate_7d']

def _stack_series_list(series_list, trim_left=0):
    arrs = []
    for s in series_list:
        a = np.asarray(s)
        if trim_left > 0:
            a = a[trim_left:]
        arrs.append(a)
    min_len = min(len(a) for a in arrs)
    arrs = [a[:min_len] for a in arrs]
    return np.vstack(arrs)  # (n_runs, n_time)

def plot_Rt_and_incidence_with_std(
    rt_df,
    ra_df_list,                # 多次运行的 R(t) DataFrame 列表，但R绘图仅用第一个
    rt_incidence,
    ra_incidence_list,         # 多次运行的 incidence 序列列表 -> 均值±std 阴影
    date,
    delta_window,
    output_folder=None,
    state='state',
    base_results=None,         # [ra_base_df_list, ra_base_incidence_list]
    value_band='std',          # 'std' 或 'sem95'=1.96*SEM
    clip_lower=0.0
):
    assert len(ra_df_list) >= 1, "ra_df_list 至少包含一个 DataFrame"
    ra_df_first = ra_df_list[4].copy()

    # ---------- 1) 对齐长度 ----------
    # R(t) 的长度以 ra_df_first 为准（与 rt_df 取最短对齐）
    n_time_R = min(len(ra_df_first), len(rt_df))
    date_R = np.asarray(date)[:n_time_R]

    # incidence 的长度来自 ra_incidence_list（左裁剪 delta_window 后的最短长度）
    Inc_stack = _stack_series_list(ra_incidence_list, trim_left=delta_window)
    n_time_I = Inc_stack.shape[1]
    date_I = np.asarray(date)[:n_time_I]

    rt_incidence_trim = np.asarray(rt_incidence)[delta_window:][:n_time_I]

    # Ground Truth R
    rt_mean = rt_df['R_mean'].to_numpy()[:n_time_R]
    rt_low  = rt_df['R_low'].to_numpy()[:n_time_R]  if 'R_low'  in rt_df else None
    rt_high = rt_df['R_high'].to_numpy()[:n_time_R] if 'R_high' in rt_df else None

    # Agent R（仅用第一个）
    ra_R_mean = ra_df_first['R_mean'].to_numpy()[:n_time_R]
    ra_R_low  = ra_df_first['R_low'].to_numpy()[:n_time_R]  if 'R_low'  in ra_df_first else None
    ra_R_high = ra_df_first['R_high'].to_numpy()[:n_time_R] if 'R_high' in ra_df_first else None

    # ---------- 2) incidence 均值与误差带 ----------
    def _mean_and_band(stack):
        mean = stack.mean(axis=0)
        if value_band == 'sem95':
            band = 1.96 * (stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]))
        else:
            band = stack.std(axis=0, ddof=1)
        lower = mean - band
        upper = mean + band
        if clip_lower is not None:
            lower = np.maximum(lower, clip_lower)
        return mean, lower, upper

    ra_I_mean, ra_I_low, ra_I_high = _mean_and_band(Inc_stack)

    # ---------- 3) base（可选） ----------
    has_base = False
    if base_results:
        ra_base_df_list, ra_base_incidence_list = base_results
        if len(ra_base_df_list) >= 1 and len(ra_base_incidence_list) >= 1:
            has_base = True
            base_df_first = ra_base_df_list[0].copy()
            # R(t) base（同样仅用第一个）
            base_R_mean = base_df_first['R_mean'].to_numpy()[:n_time_R]
            base_R_low  = base_df_first['R_low'].to_numpy()[:n_time_R]  if 'R_low'  in base_df_first else None
            base_R_high = base_df_first['R_high'].to_numpy()[:n_time_R] if 'R_high' in base_df_first else None
            # incidence base（做聚合）
            BaseInc_stack = _stack_series_list(ra_base_incidence_list, trim_left=delta_window)
            # 与主曲线按最短对齐
            n_time_I = min(n_time_I, BaseInc_stack.shape[1])
            date_I = date_I[:n_time_I]
            # 裁切主曲线与gt
            ra_I_mean, ra_I_low, ra_I_high = ra_I_mean[:n_time_I], ra_I_low[:n_time_I], ra_I_high[:n_time_I]
            rt_incidence_trim = rt_incidence_trim[:n_time_I]
            base_I_mean, base_I_low, base_I_high = _mean_and_band(BaseInc_stack[:, :n_time_I])
        else:
            base_df_first = None

    # ---------- 4) 画 R(t)：仅用第一个 run ----------
    plt.figure(figsize=(12, 6))
    # Ground Truth
    plt.plot(date_R, rt_mean, label='Ground Truth', color='blue', linewidth=2.0)
    if rt_low is not None and rt_high is not None:
        plt.fill_between(date_R, rt_low, rt_high, color='blue', alpha=0.2)

    # Agent（第一个）
    plt.plot(date_R, ra_R_mean, label='Agent Policy', color='green', linewidth=2.0)
    if ra_R_low is not None and ra_R_high is not None:
        plt.fill_between(date_R, ra_R_low, ra_R_high, color='green', alpha=0.2)

    # Base（第一个，可选）
    if has_base:
        plt.plot(date_R, base_R_mean, label='Agent Policy (base)', color='orange', linestyle='--', linewidth=2.0)
        if base_R_low is not None and base_R_high is not None:
            plt.fill_between(date_R, base_R_low, base_R_high, color='orange', alpha=0.18)

    plt.axhline(1, color='red', linestyle='--', label='R=1 Threshold', linewidth=1.6)
    plt.xlabel('Time (t)', fontsize=20)
    plt.ylabel('R(t)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Effective Reproduction Number (R_t) Over Time for ' + state.title())
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/{state}_R0.png", dpi=200)
    plt.close()

    # ---------- 5) 画 Incidence（多次实验的均值±误差带） ----------
    plt.figure(figsize=(12, 6))
    # Ground Truth
    plt.plot(date_I, rt_incidence_trim, label='Ground Truth', color='#1f77b4', linewidth=2.2)
    # Agent 聚合
    plt.plot(date_I, ra_I_mean, label='Agent Policy (mean)', color='#2ca02c', linewidth=2.2, linestyle='--')
    plt.fill_between(date_I, ra_I_low, ra_I_high, color='#2ca02c', alpha=0.18)
    # Base 聚合（可选）
    if has_base:
        plt.plot(date_I, base_I_mean, label='Agent Policy (base, mean)', color='#ff7f0e', linewidth=2.0, linestyle='--')
        plt.fill_between(date_I, base_I_low, base_I_high, color='#ff7f0e', alpha=0.18)

    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Incidence Rate (per 100k)', fontsize=20)
    plt.legend(frameon=False, fontsize=14, loc='upper left')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    plt.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/{state}_incidence.png", dpi=200)
    plt.close()
    return

def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def _band_from_list(series_list):
    """
    假定每个序列长度一致（你已有保证），返回 (mean, low, high)
    这里用均值±标准差；如需95%CI可换成 1.96*std/√n
    """
    M = np.vstack(series_list)            # (n_runs, T)
    mean = M.mean(axis=0)
    scale = 5
    std  = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean)
    low, high = mean - std * scale, mean + std * scale
    return mean, low, high

# ===================== Active cases =====================
def plot_active_cases_std(
    gt_df,
    agent_df,                    # DataFrame 或 list[DataFrame]
    date,
    delta_window,
    output_folder=None,
    state='state',
    base_results=None            # [gt_df_base, agent_base_df] 或 [gt_df_base, agent_base_df_list]
):
    plt.figure(figsize=(12, 6))
    # --- GT ---
    gt_df['active_case_ratio'] = gt_df['Q_gt'] / pop_list[state] * 100000
    gt_y = gt_df['active_case_ratio'].to_numpy()

    # --- Agent（支持 list）---
    agent_list = _ensure_list(agent_df)
    # 1) 作为“主线”的 run：沿用你的逻辑——用第一个元素
    agent_first = agent_list[0].copy()
    agent_first['active_case_ratio'] = agent_first['Q_pred'] / pop_list[state] * 100000
    agent_line = agent_first['active_case_ratio'].to_numpy()

    # 2) 若传入 list，额外计算阴影
    show_band = len(agent_list) > 1
    if show_band:
        series_list = []
        for df in agent_list:
            # df = df.copy()
            df['active_case_ratio'] = df['Q_pred'] / pop_list[state] * 100000
            series_list.append(df['active_case_ratio'].to_numpy())
        mean_band, low_band, high_band = _band_from_list(series_list)

    # --- Base（保持你原逻辑：只画线；不做阴影）---
    base_line = None
    if base_results:
        base_df_or_list, _ = base_results
        base_first = _ensure_list(base_df_or_list)[0].copy()
        base_first['active_case_ratio'] = base_first['Q_pred'] / pop_list[state] * 100000
        base_line = base_first['active_case_ratio'].to_numpy()

    # --- 绘图（沿用你的索引与裁剪方式）---
    sl = slice(delta_window, None)
    plt.plot(date, gt_y[sl],           label='Ground Truth',   color='blue', linewidth=2)
    plt.plot(date, agent_line[sl],     label='Agent Policy',   color='green', linestyle='--', linewidth=2)
    if show_band:
        plt.fill_between(date, low_band[sl], high_band[sl], color='green', alpha=0.18)
    if base_line is not None:
        plt.plot(date, base_line[sl],  label='Agent Policy (base)', color='orange', linestyle='--', linewidth=2)

    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Active case ratio (per 100K)', fontsize=20)
    plt.title('Active Case Ratio of ' + state + ' Over Time', fontsize=22)
    plt.legend(fontsize=16, frameon=False)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    ax = plt.gca(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/{state}_active_cases.png")
    plt.close()
    return gt_df, (agent_list[0] if isinstance(agent_df, (list, tuple)) else agent_df)

# ===================== Death incidence (7-day mean) =====================
def plot_death_cases_std(
    gt_df,
    agent_df,                    # DataFrame 或 list[DataFrame]
    date,
    delta_window,
    output_folder=None,
    state='state',
    base_results=None            # [gt_df_base, agent_base_df] 或 [gt_df_base, agent_base_df_list]
):
    # --- GT ---
    gt_df['new_deaths'] = gt_df['D_gt'].diff().clip(lower=0)
    gt_df['death_incidence_per100k'] = gt_df['new_deaths'] / pop_list[state] * 100000
    gt_df['death_incidence_7d'] = gt_df['death_incidence_per100k'].rolling(7, min_periods=1).mean()
    gt_y = gt_df['death_incidence_7d'].to_numpy()

    # --- Agent（支持 list）---
    agent_list = _ensure_list(agent_df)
    # 主线：第一个元素
    af = agent_list[0].copy()
    af['new_deaths'] = af['D_pred'].diff().clip(lower=0)
    af['death_incidence_per100k'] = af['new_deaths'] / pop_list[state] * 100000
    af['death_incidence_7d'] = af['death_incidence_per100k'].rolling(7, min_periods=1).mean()
    agent_line = af['death_incidence_7d'].to_numpy()

    # 阴影
    show_band = len(agent_list) > 1
    if show_band:
        series_list = []
        for df in agent_list:
            # df = df.copy()
            df['new_deaths'] = df['D_pred'].diff().clip(lower=0)
            df['death_incidence_per100k'] = df['new_deaths'] / pop_list[state] * 100000
            df['death_incidence_7d'] = df['death_incidence_per100k'].rolling(7, min_periods=1).mean()
            series_list.append(df['death_incidence_7d'].to_numpy())
        mean_band, low_band, high_band = _band_from_list(series_list)

    # --- Base（只画线）---
    base_line = None
    if base_results:
        base_df_or_list, _ = base_results
        bf = _ensure_list(base_df_or_list)[0].copy()
        bf['new_deaths'] = bf['D_pred'].diff().clip(lower=0)
        bf['death_incidence_per100k'] = bf['new_deaths'] / pop_list[state] * 100000
        bf['death_incidence_7d'] = bf['death_incidence_per100k'].rolling(7, min_periods=1).mean()
        base_line = bf['death_incidence_7d'].to_numpy()

    # --- 绘图（沿用你的索引与裁剪方式）---
    plt.figure(figsize=(12, 6))
    sl = slice(delta_window, None)
    plt.plot(date, gt_y[sl],          label='Ground Truth', color='blue', linewidth=2)
    plt.plot(date, agent_line[sl],    label='Agent Policy', color='green', linestyle='--', linewidth=2)
    if show_band:
        plt.fill_between(date, low_band[sl], high_band[sl], color='green', alpha=0.18)
    if base_line is not None:
        plt.plot(date, base_line[sl], label='Agent Policy (base)', color='orange', linestyle='--', linewidth=2)

    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Daily Death Incidence (per 100K)', fontsize=20)
    plt.title('Daily Death Incidence of ' + state + ' Over Time', fontsize=22)
    plt.legend(fontsize=16, frameon=False)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    ax = plt.gca(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if output_folder:
        plt.savefig(f"{output_folder}/{state}_death_case_ratio.png")
    plt.close()
    return gt_df, (agent_list[0] if isinstance(agent_df, (list, tuple)) else agent_df)


def aver_value_in_list(df_list, column_name):
    ave_list = []
    for df in df_list:
        if column_name is None:
            ave_list.append(df.mean())
        else:
            ave_list.append(df[column_name].dropna().mean())
    return np.mean(np.array(ave_list))

def main():
    delta_window = 21
    warm_day = 10
    data_folder = 'outputs/baseline/results/'
    suffix = 'results.csv'
    data_paths = {state: f"{data_folder}{state}_{suffix}" for state in state_list}
    all_over_df = pd.DataFrame()
    for i in range(len(state_list)):
        state = state_list[i]
        data_path = data_paths[state]
        output_folder = "D:\\Code\\OD-COVID\\figures\\R0_3week\\"
        os.makedirs(output_folder, exist_ok=True)
        gt_df = pd.read_csv(data_path)
        cols = ['Q_gt', 'D_gt', 'R_gt']
        rt_df, rt_incidence = estimate_Rt(gt_df.iloc[warm_day:], cols = cols, window=delta_window, pop = pop_list[state])
        ### read policy data for plotting
        results_folder = 'outputs\\3 weeks\\'
        subfolders = [f.path for f in os.scandir(results_folder) if f.is_dir()]
        ra_df_list = []
        ra_incidence_list = []
        ra_base_df_list = []
        ra_base_incidence_list = []
        agent_df_list = []
        for f in subfolders:
            results_path = os.path.join(f, 'results')
            agent_df = pd.read_csv(f"{results_path}\\{state}_results.csv")
            ra_df, ra_incidence = estimate_Rt(agent_df.iloc[warm_day:], cols = ['Q_pred', 'D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
            ra_base_df, ra_base_incidence = estimate_Rt(gt_df.iloc[warm_day:], cols = ['Q_pred', 'D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
            ra_df_list.append(ra_df)
            ra_incidence_list.append(ra_incidence)
            ra_base_df_list.append(ra_base_df)
            ra_base_incidence_list.append(ra_base_incidence)
            agent_df_list.append(agent_df)
        start_day = pd.to_datetime('2020-04-12')
        date = pd.date_range(start=start_day + pd.Timedelta(days=delta_window) + pd.Timedelta(days=warm_day), periods=len(rt_df))
        plot_Rt_and_incidence_with_std(rt_df, ra_df_list, rt_incidence, ra_incidence_list, date, delta_window, output_folder, state, [ra_base_df_list, ra_base_incidence_list])
        date = pd.date_range(start=start_day + pd.Timedelta(days=delta_window), periods=len(gt_df) - delta_window)
        gt_df, agent_df = plot_active_cases_std(gt_df, agent_df_list, date, delta_window, output_folder, state, [gt_df, agent_df_list])
        gt_df, agent_df = plot_death_cases_std(gt_df, agent_df_list, date, delta_window, output_folder, state, [gt_df, agent_df_list])

        ######data output for table
        agent_active_cases_aver = aver_value_in_list(agent_df_list, 'active_case_ratio')
        agent_death_cases_aver = aver_value_in_list(agent_df_list, 'death_incidence_per100k')
        agent_ro_average = aver_value_in_list(ra_df_list, 'R_mean')
        agent_incidence_average = aver_value_in_list(ra_incidence_list, None)

        # agent_active_cases_aver = agent_df['active_case_ratio'].dropna().mean()
        # agent_death_cases_aver = agent_df['death_incidence_per100k'].dropna().mean()
        # agent_incidence_average = ra_incidence.dropna().mean()

        gt_active_cases_aver = gt_df['active_case_ratio'].dropna().mean()
        gt_death_cases_aver = gt_df['death_incidence_per100k'].dropna().mean()
        gt_ro_average = rt_df['R_mean'].dropna().mean()
        gt_incidence_average = rt_incidence.dropna().mean()

        over_df = pd.DataFrame({
            'Metric': ['Average Active Cases per 100K', 'Average Daily Death Incidence per 100K', 'Average R0', 'Average Incidence Rate per 100K'],
            f'Ground Truth ({state})': [gt_active_cases_aver, gt_death_cases_aver, gt_ro_average, gt_incidence_average],
            f'Agent Policy ({state})': [agent_active_cases_aver, agent_death_cases_aver, agent_ro_average, agent_incidence_average]
        })
        if all_over_df.empty:
            all_over_df = over_df
        else:
            # 按列合并（Metrics列保持一份）
            all_over_df = pd.merge(all_over_df, over_df, on='Metric', how='outer')
            # all_over_df = all_over_df.T
    print(all_over_df)
    all_over_df.to_csv(f"outputs\\3 weeks\\r0_comparison.csv", index=False)

if __name__ == "__main__":
    main()

