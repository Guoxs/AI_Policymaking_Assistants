###### 对各州的各个指标进行计算

import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import gamma as gamma_dist
from datetime import date
import matplotlib.pyplot as plt
import os
from agents import state

# data_folder = 'datasets/5 states/'
# suffix = 'cases_0412_1231.csv'
# state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']

data_folder = 'datasets/US/epedimic/'
suffix = 'cases.csv'
state_list = ['alabama','arizona', 'arkansas', 'idaho','indiana', 'iowa', 'kentucky', 'michigan','minnesota', 'mississippi', 'nebraska','new mexico','ohio','oklahoma','south carolina', 'tennessee', 'texas', 'utah','virginia','wisconsin']
data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}

def obtain_metrics(gt_df, cols=['Q_gt', 'D_gt', 'R_gt'], pop=100000):
    gt_df['Confirmed'] = gt_df[cols].sum(axis=1)
    gt_df['new_cases'] = gt_df['Confirmed'].diff().fillna(0)
    gt_df.loc[gt_df['new_cases'] < 0, 'new_cases'] = 0  # 修正负值
    gt_df['incidence_rate'] = gt_df['new_cases'] / pop * 100000
    gt_df['incidence_rate_7d'] = gt_df['incidence_rate'].rolling(window=7, min_periods=1).mean()
    gt_df['active_case_ratio'] = gt_df[cols[0]] / pop * 100000
    gt_df['new_deaths'] = gt_df[cols[1]].diff().clip(lower=0)
    gt_df['death_incidence_per100k'] = gt_df['new_deaths'] / pop * 100000
    gt_df['death_incidence_7d'] = gt_df['death_incidence_per100k'].rolling(7, min_periods=1).mean()
    return gt_df

def estimate_Rt(gt_df, cols=['Q_gt', 'D_gt', 'R_gt'], window=21, pop = 100000, eps = 1e-12):
    gt_df = gt_df.copy()
    gt_df = obtain_metrics(gt_df, cols, pop)
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
    return rt_df, gt_df


def main():
    delta_window = 21
    warm_day = 10
    #data_folder = 'outputs/baseline-5states/results/'
    data_folder = 'outputs/baseline-us/results/'
    suffix = 'results.csv'
    #weeks = ['4 weeks', '6 weeks', '8 weeks', '10 weeks']
    weeks = ['detection', 'ori_restriction']
    for week_freq in weeks:
        results_folder = 'outputs\\US\\'+week_freq+'\\' +'agent\\'
        output_folder = results_folder + 'metrics\\'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data_paths = {state: f"{data_folder}{state}_{suffix}" for state in state_list}
        # output_folder = "D:\\Code\\OD-COVID\\figures\\R0_3week\\"
        # os.makedirs(output_folder, exist_ok=True)
        # all_over_df = pd.DataFrame()
        for i in range(len(state_list)):
            state = state_list[i]
            data_path = data_paths[state]
            gt_df = pd.read_csv(data_path)
            cols = ['Q_gt', 'D_gt', 'R_pred']
            rt_df, gt_df = estimate_Rt(gt_df.iloc[warm_day:], cols=cols, window=delta_window, pop=pop_list[state])
            subfolders = [f.path for f in os.scandir(results_folder) if f.is_dir()]
            subfolders = [f for f in subfolders if 'metric' not in f]
            #subfolders = [subfolders[0]]     ###attention: 只取了第一个做实验
            print(f"Processing state: {state}, found {subfolders} experiment folders.")
            policy_dfs = []
            metric_cols = ['incidence_rate_7d', 'active_case_ratio', 'death_incidence_7d']
            gt_df = gt_df[metric_cols]
            for f in subfolders:
                results_path = os.path.join(f, 'results')
                agent_df = pd.read_csv(f"{results_path}\\{state}_results.csv")
                ra_df, policy_df = estimate_Rt(agent_df.iloc[warm_day:], cols=['Q_pred','D_pred', 'R_pred'], window=delta_window, pop=pop_list[state])
                policy_dfs.append(policy_df[metric_cols])
            names = [f'df{i+1}' for i in range(len(policy_dfs))]
            big = pd.concat(policy_dfs, axis=1, keys=names)
            row_mean = big.groupby(level=1, axis=1).mean()  
            row_std  = big.groupby(level=1, axis=1).std(ddof=1) 
            print(f"State: {state}")
            # print(row_mean.tail(5))
            # print(row_std.tail(5))
            # print(gt_df.tail(5))
            # print(rt_df.tail(5))
            rt_df.to_csv(f"{output_folder}{state}_R0_gt_{week_freq}.csv", index=False)
            ra_df.to_csv(f"{output_folder}{state}_R0_agent_{week_freq}.csv", index=False)
            row_mean.to_csv(f"{output_folder}{state}_metrics_mean_{week_freq}.csv", index=False)
            row_std.to_csv(f"{output_folder}{state}_metrics_std_{week_freq}.csv", index=False)
            gt_df.to_csv(f"{output_folder}{state}_metrics_gt_{week_freq}.csv", index=False)


if __name__ == "__main__":
    main()