import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import gamma as gamma_dist
from datetime import date
import matplotlib.pyplot as plt
import os
from agents import state
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",      
    "axes.unicode_minus": False      
})

def main():
    # output_folder = "D:\\Code\\OD-COVID\\figures\\R0_3week\\"
    week_freq = '6 weeks'
    results_folder = 'outputs\\5 states\\' + week_freq + '\\gpt-3.5\\metrics\\'
    state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    state = state_list[4]  # Example for one state
    metric_gt = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_gt_{week_freq}.csv"))
    metric_mean = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_mean_{week_freq}.csv"))
    metric_std = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_std_{week_freq}.csv"))
    print(metric_gt.tail())
    print(metric_mean.tail())
    warm_day = 10
    start_date = date.fromisoformat('2020-04-12') + pd.DateOffset(days=warm_day)
    metric_gt['date'] = pd.date_range(start=start_date, periods=len(metric_gt), freq='D')
    metric_gt['date'] = pd.to_datetime(metric_gt['date'])
    metric_mean['date'] = metric_gt['date']
    metric_std['date'] = metric_gt['date']
    metric_cols = ['incidence_rate_7d', 'active_case_ratio', 'death_incidence_7d']
    y_labels = ['Incidence Rate (per 100k)','Active case ratio (per 100K)', 'Daily Death Incidence (per 100K)']
    plt.figure(figsize=(12, 6))
    index = 0
    print(metric_gt[metric_cols[index]].tail())
    print(metric_mean[metric_cols[index]].tail())
    plt.plot(metric_gt['date'], metric_gt[metric_cols[index]], label='Ground Truth', color='blue')
    plt.plot(metric_mean['date'], metric_mean[metric_cols[index]], label='Agent', color='orange')
    plt.fill_between(metric_mean['date'], metric_mean[metric_cols[index]] - metric_std[metric_cols[index]], metric_mean[metric_cols[index]] + metric_std[metric_cols[index]], color='orange', alpha=0.2)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel(y_labels[index], fontsize=20)
    plt.legend(frameon=False, fontsize=14, loc='upper left')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()