import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import gamma as gamma_dist
from datetime import date
import matplotlib.pyplot as plt

from agents import state

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
   plt.figure(figsize=(12, 6))
   plt.plot(date, rt_df['R_mean'], label='Ground Truth', color='blue')
   plt.fill_between(date, rt_df['R_low'], rt_df['R_high'], color='blue', alpha=0.2)
   plt.plot(date, ra_df['R_mean'], label='Agent Policy', color='green')
   plt.fill_between(date, ra_df['R_low'], ra_df['R_high'], color='green', alpha=0.2)
   if base_results:
      base_df, _ = base_results
      plt.plot(date, base_df['R_mean'], label='Agent Policy (base)', color='orange', linestyle='--')
      plt.fill_between(date, base_df['R_low'], base_df['R_high'], color='orange', alpha=0.2)
   plt.axhline(1, color='red', linestyle='--', label='R=1 Threshold')
   plt.xlabel('Time (t)', fontsize=20)
   plt.ylabel('R(t)', fontsize=20)
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)
   plt.title('Effective Reproduction Number (R_t) Over Time for ' + state.title())
   plt.legend(fontsize=16)
   plt.grid()
   plt.tight_layout()
   if output_folder:
      plt.savefig(f"{output_folder}/{state.replace(' ', '_')}_R0.png")
   plt.close()

   plt.figure(figsize=(12,6))
   # date = pd.date_range(start=start_day, periods=len(gt_df))
   plt.plot(date, rt_incidence[delta_window:], label='Ground Truth', color='#1f77b4', linewidth=2.2)
   plt.plot(date, ra_incidence[delta_window:], label='Agent Policy', color='#2ca02c', linewidth=2.2, linestyle='--')
   if base_results:
      _, base_incidence = base_results
      plt.plot(date, base_incidence[delta_window:], label='Agent Policy (base)', color='#ff7f0e', linewidth=2.2, linestyle='--')
   # 标签
   plt.xlabel('Time (days)', fontsize=20)
   plt.ylabel('Incidence Rate (per 100k)', fontsize=20)
   # 图例
   plt.legend(frameon=False, fontsize=16, loc='upper left')
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)
   ax = plt.gca()
   plt.grid(alpha=0.3)
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   plt.tight_layout()
   if output_folder:
      plt.savefig(f"{output_folder}/{state.replace(' ', '_')}_incidence.png")
   plt.close()  


def plot_active_cases(gt_df, agent_df, date, delta_window, pop_list, output_folder = None, state = 'state', base_results = None):
   plt.figure(figsize=(12, 6))
   gt_df['active_case_ratio'] = gt_df['Q_gt'] / pop_list[state] *100000
   agent_df['active_case_ratio'] = agent_df['Q_pred'] / pop_list[state] * 100000
   # 绘制 gt_df 的 active_case_ratio
   plt.plot(date, gt_df['active_case_ratio'][delta_window:], label='Ground Truth', color='blue', linewidth=2)
   # 绘制 agent_df 的 active_case_ratio
   plt.plot(date, agent_df['active_case_ratio'][delta_window:], label='Agent Policy', color='green', linestyle='--', linewidth=2)
   if base_results:
      base_df, _ = base_results
      base_df['active_case_ratio'] = base_df['Q_pred'] / pop_list[state] * 100000
      plt.plot(date, base_df['active_case_ratio'][delta_window:], label='Agent Policy (base)', color='orange', linestyle='--', linewidth=2)
   # 添加图例、标题和标签
   plt.xlabel('Time (days)', fontsize=20)
   plt.ylabel('Active case ratio (per 100K)', fontsize=20)
   plt.title('Active Case Ratio of ' + state + ' Over Time', fontsize=22)
   plt.legend(fontsize=16, frameon=False)
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)
   # 美化图表
   plt.grid(alpha=0.3)
   ax = plt.gca()
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   # 显示图表
   plt.tight_layout()
   if output_folder:
      plt.savefig(f"{output_folder}/{state.replace(' ', '_')}_active_cases.png")
   plt.close()
   return gt_df, agent_df

def plot_death_cases(gt_df, agent_df, date, delta_window, pop_list, output_folder = None, state = 'state', base_results = None):
   gt_df['new_deaths'] = gt_df['D_gt'].diff().clip(lower=0)
   gt_df['death_incidence_per100k'] = gt_df['new_deaths'] / pop_list[state] * 100_000
   gt_df['death_incidence_7d'] = gt_df['death_incidence_per100k'].rolling(7, min_periods=1).mean()

   agent_df['new_deaths'] = agent_df['D_pred'].diff().clip(lower=0)
   agent_df['death_incidence_per100k'] = agent_df['new_deaths'] / pop_list[state] * 100000
   agent_df['death_incidence_7d'] = agent_df['death_incidence_per100k'].rolling(7, min_periods=1).mean()
   # 绘制 gt_df 的 death_case_ratio
   plt.figure(figsize=(12, 6))
   plt.plot(date, gt_df['death_incidence_7d'][delta_window:], label='Ground Truth', color='blue', linewidth=2)
   # 绘制 agent_df 的 death_case_ratio
   plt.plot(date, agent_df['death_incidence_7d'][delta_window:], label='Agent Policy', color='green', linestyle='--', linewidth=2)
   if base_results:  
      base_df, _ = base_results
      base_df['new_deaths'] = base_df['D_pred'].diff().clip(lower=0)
      base_df['death_incidence_per100k'] = base_df['new_deaths'] / pop_list[state] * 100000
      base_df['death_incidence_7d'] = base_df['death_incidence_per100k'].rolling(7, min_periods=1).mean()
      plt.plot(date, base_df['death_incidence_7d'][delta_window:], label='Agent Policy (base)', color='orange', linestyle='--', linewidth=2)
   # 添加图例、标题和标签
   plt.xlabel('Time (days)', fontsize=20)
   plt.ylabel('Daily Death Incidence (per 100K)', fontsize=20)
   plt.title('Daily Death Incidence of ' + state + ' Over Time', fontsize=22)
   plt.legend(fontsize=16, frameon=False)
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)
   # 美化图表
   plt.grid(alpha=0.3)
   ax = plt.gca()
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   # 显示图表
   plt.tight_layout()
   if output_folder:
      plt.savefig(f"{output_folder}/{state.replace(' ', '_')}_death_case_ratio.png")
#    plt.show()
   plt.close()
   return gt_df, agent_df

def main():
    data_folder = 'datasets/'
    suffix = 'cases_0412_1231.csv'
    state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
    data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
    pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}
    delta_window = 21
    warm_day = 10
    data_folder = 'outputs/baseline/results/'
    suffix = 'results.csv'
    data_paths = {state: f"{data_folder}{state}_{suffix}" for state in state_list}
    all_over_df = pd.DataFrame()
    for i in range(len(state_list)):
        state = state_list[i]
        data_path = data_paths[state]
        output_folder = "D:\\Code\\OD-COVID\\figures\\R0\\"
        gt_df = pd.read_csv(data_path)
        cols = ['Q_gt', 'D_gt', 'R_gt']
        rt_df, rt_incidence = estimate_Rt(gt_df.iloc[warm_day:], cols = cols, window=delta_window, pop = pop_list[state])
        ### read policy data for plotting
        results_path='outputs\\gpt-3.5-turbo-0125_2025-10-29-19-28-51\\results'
        agent_df = pd.read_csv(f"{results_path}\\{state}_results.csv")
        ra_df, ra_incidence = estimate_Rt(agent_df.iloc[warm_day:], cols = ['Q_pred', 'D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
        start_day = pd.to_datetime('2020-04-12')
        date = pd.date_range(start=start_day + pd.Timedelta(days=delta_window) + pd.Timedelta(days=warm_day), periods=len(rt_df))
        ra_base_df, ra_base_incidence = estimate_Rt(gt_df.iloc[warm_day:], cols = ['Q_pred', 'D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
        plot_Rt_and_incidence(rt_df, ra_df, rt_incidence, ra_incidence, date, delta_window, output_folder, state, [ra_base_df, ra_base_incidence])
        date = pd.date_range(start=start_day + pd.Timedelta(days=delta_window), periods=len(gt_df) - delta_window)
        gt_df, agent_df = plot_active_cases(gt_df, agent_df, date, delta_window, pop_list, output_folder, state, [gt_df, agent_df])
        gt_df, agent_df = plot_death_cases(gt_df, agent_df, date, delta_window, pop_list, output_folder, state, [gt_df, agent_df])

        ######data output for table
        agent_active_cases_aver = agent_df['active_case_ratio'].dropna().mean()
        agent_death_cases_aver = agent_df['death_incidence_per100k'].dropna().mean()
        agent_ro_average = ra_df['R_mean'].dropna().mean()
        agent_incidence_average = ra_incidence.dropna().mean()

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
    all_over_df.to_csv(f"outputs\\gpt-3.5-turbo-0125_2025-10-29-19-28-51\\r0_comparison.csv", index=False)
if __name__ == "__main__":
    main()

