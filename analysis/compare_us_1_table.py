import os
import numpy as np
import pandas as pd

state_list = ['alabama','arizona', 'arkansas','idaho','indiana', 'iowa', 'kentucky', 'michigan','minnesota', 'mississippi',
              'nebraska','new mexico','ohio','oklahoma','south carolina', 'tennessee', 'texas', 'utah','virginia','wisconsin']
state_abbr = ['AL','AZ','AR','ID','IN','IA','KY','MI','MN','MS','NE','NM','OH','OK','SC','TN','TX','UT','VA','WI']

week_freqs  = ["4 weeks", "8 weeks", 'ori_restriction', 'detection']
label_name  = ['TIR (4 weeks)', 'TIR (8 weeks)', 'SIS', 'TIS']
results_root = r"outputs\US"

metric_cols = ['active_case_ratio', 'incidence_rate_7d', 'death_incidence_7d']

# 输出目录
out_dir = os.path.join("figures", "US",'SI')
os.makedirs(out_dir, exist_ok=True)

def read_metric_means_stds(state, wf, metric_cols):
    """
    读取某州某策略下：
    - GT: metric_gt[metric_cols].mean()
    - Agent mean: metric_mean[metric_cols].mean()
    - Agent std: metric_std[metric_cols].mean()
    """
    results_folder = os.path.join(results_root, wf, "agent", "metrics")
    fp_gt   = os.path.join(results_folder, f"{state}_metrics_gt_{wf}.csv")
    fp_mean = os.path.join(results_folder, f"{state}_metrics_mean_{wf}.csv")
    fp_std  = os.path.join(results_folder, f"{state}_metrics_std_{wf}.csv")

    metric_gt   = pd.read_csv(fp_gt)
    metric_mean = pd.read_csv(fp_mean)
    metric_std  = pd.read_csv(fp_std)

    gt_vals   = metric_gt[metric_cols].mean(numeric_only=True)
    mean_vals = metric_mean[metric_cols].mean(numeric_only=True)
    std_vals  = metric_std[metric_cols].mean(numeric_only=True)

    return gt_vals, mean_vals, std_vals

# 为每个 metric 单独建表并导出
for metric in metric_cols:
    rows = []

    for i, state in enumerate(state_list):
        abbr = state_abbr[i]

        # 逐策略读取
        gt_value = None
        agent_mean = {}
        agent_std  = {}

        for wf in week_freqs:
            gt_vals, mean_vals, std_vals = read_metric_means_stds(state, wf, metric_cols)

            # GT 对同一州在不同 wf 下通常相同，你原代码也是取第一次即可
            if gt_value is None:
                gt_value = float(gt_vals[metric])

            agent_mean[wf] = float(mean_vals[metric])
            agent_std[wf]  = float(std_vals[metric])

        rows.append({
            "State": state,
            "Abbr": abbr,
            "Ground Truth": gt_value,

            # 4 weeks
            f"{label_name[0]}": agent_mean["4 weeks"],
            #f"{label_name[0]}_std":  agent_std["4 weeks"],

            # 8 weeks
            f"{label_name[1]}": agent_mean["8 weeks"],
           # f"{label_name[1]}_std":  agent_std["8 weeks"],

            # ori_restriction
            f"{label_name[2]}": agent_mean["ori_restriction"],
            #f"{label_name[2]}_std":  agent_std["ori_restriction"],

            # detection
            f"{label_name[3]}": agent_mean["detection"],
            #f"{label_name[3]}_std":  agent_std["detection"],
        })

    df_metric = pd.DataFrame(rows)

    # 导出文件名
    out_csv = os.path.join(out_dir, f"US_metric_{metric}_mean_std.csv")
    df_metric.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
