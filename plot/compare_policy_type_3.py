####绘制不同政策在效率和公平性上的trade off 气泡图

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# ---------- 全局字体 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})



def shannon_entropy_norm(x, eps=1e-12):
    """
    x: (n_states,) nonnegative vector
    return: normalized entropy in [0,1]
    """
    x = np.asarray(x, dtype=float)
    # x = x * (-1)
    x = np.clip(x, 0, None)
    s = x.sum()
    if not np.isfinite(s) or s <= eps:
        return np.nan
    p = x / (s + eps)
    H = -(p * np.log(p + eps)).sum()
    return float(H / np.log(len(p) + eps))

def gini(x, eps=1e-12):
    """
    Gini coefficient for a nonnegative vector x.
    Returns np.nan if x has no mass (sum ~ 0).
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)               # ensure nonnegative
    s = x.sum()
    if not np.isfinite(s) or s <= eps:
        return np.nan
    xs = np.sort(x)                       # ascending
    n = xs.size
    i = np.arange(1, n + 1)               # 1..n
    G = np.sum((2 * i - n - 1) * xs) / (n * s)
    return float(G)

def try_read_execution_cost(exp_dir):
    """
    可执行成本（可选）：你需要根据自己的文件结构改这里。
    目前实现为：
      1) 若存在 exp_dir/metrics/policy_cost.csv 且含 'cost' 列 -> 取最后/均值
      2) 若不存在 -> 返回 np.nan（图里用统一点大小）
    """
    cost_csv = os.path.join(exp_dir, "metrics", "policy_cost.csv")
    if os.path.exists(cost_csv):
        dfc = pd.read_csv(cost_csv)
        if "cost" in dfc.columns and len(dfc) > 0:
            return float(dfc["cost"].to_numpy(dtype=float).mean())
    return np.nan

def collect_tradeoff_points(agent_dir, state_list, gt_total_final, gt_death_final):
    sub_exp_dirs = [
        f.path for f in os.scandir(agent_dir)
        if f.is_dir() and 'metrics' not in f.name
    ]

    # GT 各州 final 向量 + 总量（常数，可提前算）
    gt_tot = np.array([gt_total_final[s] for s in state_list], dtype=float)
    gt_dea = np.array([gt_death_final[s] for s in state_list], dtype=float)
    GT_total_sum = float(np.nansum(gt_tot))
    GT_death_sum = float(np.nansum(gt_dea))

    rows = []
    for exp_dir in sub_exp_dirs:
        agent_total_final = []
        agent_death_final = []

        ok = True
        for s in state_list:
            fp = os.path.join(exp_dir, "results", f"{s}_results.csv")
            if not os.path.exists(fp):
                ok = False
                break

            df = pd.read_csv(fp)
            q = float(df["Q_pred"].to_numpy(dtype=float)[-1])
            r = float(df["R_pred"].to_numpy(dtype=float)[-1])
            d = float(df["D_pred"].to_numpy(dtype=float)[-1])

            agent_total_final.append(q + r + d)
            agent_death_final.append(d)

        if not ok:
            continue

        agent_total_final = np.array(agent_total_final, dtype=float)
        agent_death_final = np.array(agent_death_final, dtype=float)

        # ===== X轴：总量减少（越大越好）=====
        total_sum = float(np.nansum(agent_total_final))
        death_sum = float(np.nansum(agent_death_final))

        reduction_total = GT_total_sum - total_sum
        reduction_death = GT_death_sum - death_sum

        # 若出现“变差”（reduction<0），可截断为0，保持“越大越好”的语义
        reduction_total = float(np.clip(reduction_total, 0, None))
        reduction_death = float(np.clip(reduction_death, 0, None))

        # ===== 各州改进率（用于公平性分配）=====
        eps = 1e-12
        imp_total = (gt_tot - agent_total_final) / np.maximum(np.abs(gt_tot), eps)
        imp_death = (gt_dea - agent_death_final) / np.maximum(np.abs(gt_dea), eps)

        # imp_total = gt_tot - agent_total_final
        # imp_death = gt_dea - agent_death_final

        # 只用“正向改进”来衡量收益分配（否则负改进会破坏分配解释）
        imp_total_pos = np.clip(imp_total, 0, None)
        imp_death_pos = np.clip(imp_death, 0, None)
        # ===== Y轴：公平性（越大越好）=====
        # Gini 越小越公平 -> fairness = 1 - Gini
        g_total = gini(imp_total_pos)
        g_death = gini(imp_death_pos)
        fairness_total = np.nan if not np.isfinite(g_total) else float(1.0 - g_total)
        fairness_death = np.nan if not np.isfinite(g_death) else float(1.0 - g_death)
        exec_cost = try_read_execution_cost(exp_dir)
        rows.append({
            "exp_dir": exp_dir,
            "reduction_total": reduction_total,
            "reduction_death": reduction_death,
            "fairness_total": fairness_total,
            "fairness_death": fairness_death,
            "exec_cost": exec_cost,
        })

    return pd.DataFrame(rows)

def plot_tradeoff_scatter(df_by_policy, x_col, y_col, xlabel, ylabel, filename,
                          output_dir, type_name=('TIR','SIS','TIS')):
    # Nature-like：简洁、留白、弱网格
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    colors  = ["#1f77b4", "#B3D7BD", "#B3A3CD"]
    markers = ["o", "^", "s"]

    # 点大小：若 exec_cost 全是 NaN，就固定大小；否则按 cost 缩放
    # 你可根据 cost 数量级调这个映射
    def size_map(cost_series):
        if np.all(~np.isfinite(cost_series.to_numpy(dtype=float))):
            return None
        c = cost_series.to_numpy(dtype=float)
        c = np.where(np.isfinite(c), c, np.nan)
        cmin, cmax = np.nanmin(c), np.nanmax(c)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            return np.full_like(c, 70.0)
        # 映射到 [50, 180]
        return 50.0 + (c - cmin) / (cmax - cmin) * (180.0 - 50.0)

    any_cost = any(np.any(np.isfinite(df["exec_cost"].to_numpy(dtype=float))) for df in df_by_policy)

    for i, df in enumerate(df_by_policy):
        dfp = df.copy()
        x = dfp[x_col].to_numpy(dtype=float)
        y = dfp[y_col].to_numpy(dtype=float)

        if any_cost:
            s = size_map(dfp["exec_cost"])
            if s is None:
                s = 70.0
        else:
            s = 270.0

        ax.scatter(
            x, y,
            s=s,
            c=colors[i],
            marker=markers[i],
            edgecolors="black",
            linewidths=0.7,
            alpha=0.90,
            label=type_name[i],
            zorder=3
        )

    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="both", labelsize=24)
    ax.ticklabel_format(
    axis="x",
    style="sci",
    scilimits=(0, 0),   # 强制始终用科学计数法
    # useMathText=True   # 使用 ×10^k 形式（Nature 更喜欢）
)
    ax.xaxis.get_offset_text().set_fontsize(24)   # 建议 20–26
    ax.grid(axis="both", linestyle="--", alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 熵归一化后一般在 [0,1]，可以固定范围更像论文图
    # ax.set_ylim(0, 1)
    if xlabel == "Reduction in total infections":
        ax.legend(frameon=False, fontsize=22)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved: {out_path}")


base_dir = 'outputs//5 states//6 weeks'  # 三个子目录都在这个目录下
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
agent_dir = os.path.join(base_dir, 'gpt-3.5//')
output_dir = 'figures//5 states//policy type compare'
agent_dir_2 = 'outputs//5 states//ori_restriction//gpt-3.5//'
agent_dir_3 = 'outputs//5 states//detection//gpt-3.5//'
type_name = ('TIR', 'SIS', 'TIS')

gt_confirmed = {}
gt_death = {}
gt_dir = os.path.join(base_dir, 'no_action')
for state in state_list:
    results_path = os.path.join(gt_dir, 'results', f"{state}_results.csv")
    df = pd.read_csv(results_path)
    q_gt = df['Q_gt'].to_numpy(dtype=float)
    r_gt = df['R_pred'].to_numpy(dtype=float)
    d_gt = df['D_gt'].to_numpy(dtype=float)
    if state not in gt_confirmed:
        gt_confirmed[state] = q_gt + r_gt + d_gt
        gt_death[state] = d_gt
    else:
        gt_confirmed[state] += q_gt + r_gt + d_gt
        gt_death[state] += d_gt

total_stats_gt = {}
death_stats_gt = {}
for state in state_list:
    total_stats_gt[state] = {
        "final_mean": gt_confirmed[state][-1],
    }
    death_stats_gt[state] = {
        "final_mean": gt_death[state][-1],
    }
gt_total_final = {s: float(total_stats_gt[s]["final_mean"]) for s in state_list}
gt_death_final = {s: float(death_stats_gt[s]["final_mean"]) for s in state_list}

# 分别为三种 policy 收集“每次实验一个点”
df_TIR = collect_tradeoff_points(agent_dir,   state_list, gt_total_final, gt_death_final)
df_SIS = collect_tradeoff_points(agent_dir_2, state_list, gt_total_final, gt_death_final)
df_TIS = collect_tradeoff_points(agent_dir_3, state_list, gt_total_final, gt_death_final)

df_by_policy = [df_TIR, df_SIS, df_TIS]

# 图A：感染 trade-off
plot_tradeoff_scatter(
    df_by_policy=df_by_policy,
    x_col="reduction_total",
    y_col="fairness_total",
    xlabel="Reduction in total infections",
    ylabel="Equity coefficient",
    filename="tradeoff_infections_entropy.png",
    output_dir=output_dir,
    type_name=type_name
)

# 图B：死亡 trade-off
plot_tradeoff_scatter(
    df_by_policy=df_by_policy,
    x_col="reduction_death",
    y_col="fairness_death",
    xlabel="Reduction in total deaths",
    ylabel="Equity coefficient",
    filename="tradeoff_deaths_entropy.png",
    output_dir=output_dir,
    type_name=type_name
)