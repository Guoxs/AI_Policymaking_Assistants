import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import gamma as gamma_dist
from datetime import date
import matplotlib.pyplot as plt
import os
from agents import state
import matplotlib as mpl
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
abbr = {
    "arizona": "AZ",
    "mississippi": "MS",
    "new mexico": "NM",
    "texas": "TX",
    "virginia": "VA"
}

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

def plot_Rt_and_incidence(rt_df, ra_df, rt_incidence, ra_incidence,date, delta_window, output_folder = None, state = 'state', base_results = None):
    '''tr_df: ground truth R_t dataframe
        ra_df: agent R_t dataframe
        rt_incidence: ground truth incidence rate series
        ra_incidence: agent incidence rate series'''
    plt.figure(figsize=(8, 6))
    plt.plot(date, rt_df['R_mean'], label='Ground Truth', color="#4D4D4D")
    plt.fill_between(date, rt_df['R_low'], rt_df['R_high'], color="#4D4D4D", alpha=0.2)
    plt.plot(date, ra_df['R_mean'], label='LLM Agent', color="#1f77b4")
    plt.fill_between(date, ra_df['R_low'], ra_df['R_high'], color="#1f77b4", alpha=0.2)
    if base_results:
        base_df, _ = base_results
        plt.plot(date, base_df['R_mean'], label='Agent Policy (base)', color='orange', linestyle='--')
        plt.fill_between(date, base_df['R_low'], base_df['R_high'], color='orange', alpha=0.2)
    plt.axhline(1, color='red', linestyle='--', label='R=1 Threshold')
    plt.xticks(fontsize=20, rotation=30)
    plt.yticks(fontsize=32)
    plt.title(abbr.get(state, state.upper()), fontsize=30)
    if state == 'arizona':
        plt.xlabel('Date', fontsize=40)
        plt.ylabel('R(t)', fontsize=40)
        plt.legend(fontsize=25)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if output_folder:
        if os.path.exists(output_folder) == False:
            os.makedirs(output_folder)
        plt.savefig(f"{output_folder}/{state.replace(' ', '_')}_R0.png")
    plt.close()

#    plt.figure(figsize=(12,6))
#    # date = pd.date_range(start=start_day, periods=len(gt_df))
#    plt.plot(date, rt_incidence[delta_window:], label='Ground Truth', color='#1f77b4', linewidth=2.2)
#    plt.plot(date, ra_incidence[delta_window:], label='LLM Agent', color='#2ca02c', linewidth=2.2, linestyle='--')
#    if base_results:
#       _, base_incidence = base_results
#       plt.plot(date, base_incidence[delta_window:], label='Agent Policy (base)', color='#ff7f0e', linewidth=2.2, linestyle='--')
#    # 标签
#    plt.xlabel('Time (days)', fontsize=20)
#    plt.ylabel('Incidence Rate (per 100k)', fontsize=20)
#    # 图例
#    plt.legend(frameon=False, fontsize=16, loc='upper left')
#    plt.xticks(fontsize=16)
#    plt.yticks(fontsize=16)
#    ax = plt.gca()
#    plt.grid(alpha=0.3)
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    plt.tight_layout()
#    if output_folder:
#     if os.path.exists(output_folder) == False:
#         os.makedirs(output_folder)
#     plt.savefig(f"{output_folder}/{state.replace(' ', '_')}_incidence.png")
#    plt.close()  


def main():
    data_folder = 'datasets/5 states/'
    suffix = 'cases_0412_1231.csv'
    state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
    data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
    pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}
    delta_window = 21
    warm_day = 10
    data_folder = 'outputs/5 states/6 weeks/no_action/results/'
    suffix = 'results.csv'
    data_paths = {state: f"{data_folder}{state}_{suffix}" for state in state_list}
    all_over_df = pd.DataFrame()
    for i in range(len(state_list)):
        state = state_list[i]
        data_path = data_paths[state]
        output_folder = "figures\\5 states\\R0\\"
        gt_df = pd.read_csv(data_path)
        cols = ['Q_gt', 'D_gt', 'R_gt']
        rt_df, rt_incidence = estimate_Rt(gt_df.iloc[warm_day:], cols =['Q_gt', 'D_gt', 'R_pred'], window=delta_window, pop = pop_list[state])
        ### read policy data for plotting
        results_path='outputs\\5 states\\6 weeks\\gpt-3.5\\gpt-3.5-turbo-0125_2025-12-16-13-39-47\\results'
        agent_df = pd.read_csv(f"{results_path}\\{state}_results.csv")
        ra_df, ra_incidence = estimate_Rt(agent_df.iloc[warm_day:], cols = ['Q_pred', 'D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
        start_day = pd.to_datetime('2020-04-12')
        date = pd.date_range(start=start_day + pd.Timedelta(days=delta_window) + pd.Timedelta(days=warm_day), periods=len(rt_df))
        ra_base_df, ra_base_incidence = estimate_Rt(gt_df.iloc[warm_day:], cols = ['Q_pred', 'D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
        #plot_Rt_and_incidence(rt_df, ra_df, rt_incidence, ra_incidence, date, delta_window, output_folder, state, [ra_base_df, ra_base_incidence])
        plot_Rt_and_incidence(rt_df, ra_df, rt_incidence, ra_incidence, date, delta_window, output_folder, state, None)
        date = pd.date_range(start=start_day + pd.Timedelta(days=delta_window), periods=len(gt_df) - delta_window)
        
        ######data output for table
        agent_ro_average = ra_df['R_mean'].dropna().mean()
        agent_incidence_average = ra_incidence.dropna().mean()

        gt_ro_average = rt_df['R_mean'].dropna().mean()
        gt_incidence_average = rt_incidence.dropna().mean()

        over_df = pd.DataFrame({
            'Metric': [ 'Average R0', 'Average Incidence Rate per 100K'],
            f'Ground Truth ({state})': [gt_ro_average, gt_incidence_average],
            f'Agent Policy ({state})': [agent_ro_average, agent_incidence_average]
        })
        if all_over_df.empty:
            all_over_df = over_df
        else:
            # 按列合并（Metrics列保持一份）
            all_over_df = pd.merge(all_over_df, over_df, on='Metric', how='outer')
            # all_over_df = all_over_df.T
    print(all_over_df)
    all_over_df.to_csv(f"outputs\\5 states\\6 weeks\\gpt-3.5\\r0_comparison.csv", index=False)

if __name__ == "__main__":
    main()

