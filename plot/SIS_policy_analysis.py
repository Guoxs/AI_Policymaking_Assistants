#### 绘制竞争策略在不同决策轮次下的选择概率堆叠图，以及感染率变化趋势图
import os
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def extract_policy_states_from_textfile(txt_path, keep_duplicates=True):
    # 1) 抓取 policy={...} 这一段（尽量不贪婪）
    policy_block_pat = re.compile(r"Parsed PolicyResponse:\s*policy\s*=\s*(\{.*?\})")
    # 2) 在 { ... } 中提取 'state name': 这种 key（支持空格）
    state_key_pat = re.compile(r"'([^']+)'\s*:")
    out = []
    seen = set()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = policy_block_pat.search(line)
            if not m:
                continue
            policy_str = m.group(1)  # "{'new mexico': [0.5, 0.5], ...}"
            states = state_key_pat.findall(policy_str)
            for s in states:
                if keep_duplicates:
                    out.append(s)
                else:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
    return out
data_folder = 'datasets/5 states/'
suffix = 'cases_0412_1231.csv'
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
data_paths = {state: data_folder + state + '_' + suffix for state in state_list}
data_dict = {state: pd.read_csv(path) for state, path in data_paths.items()}
pop_list = {state: data_dict[state]['Population'].iloc[0] for state in state_list}
pop_ratios = {state: pop_list[state] / sum(pop_list.values()) for state in state_list}
print("Population ratios:", pop_ratios)

#policy = 'ori_restriction'
policys = ['ori_restriction', 'detection']
policy_label = ['SIS', 'TIS']
state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
metric_cols = 'incidence_rate_7d'

abbr = {
    "arizona": "AZ", "mississippi": "MS", "new mexico": "NM",
    "texas": "TX", "virginia": "VA"
}
palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
line_color = "#2F2F2F"

output_dir = 'figures//5 states//policy type compare//'
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# 画布：1x2，保持单图比例
# 单图 12x5.5 -> 两图并排 24x5.5
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(24, 5.5), dpi=300)
ax2_list = []  # 存放每个子图的右轴，便于后续统一处理（可选）

# 用于全局 legend：先收集一个子图的 handles
legend_handles = None
legend_labels = None

for idx, policy in enumerate(policys):
    ax = axes[idx]

    agent_dir = f'outputs//5 states//{policy}//gpt-3.5//'
    sub_folders = os.listdir(agent_dir)

    metric_dirs = [f for f in sub_folders if 'metric' in f]
    metric_dir = os.path.join(agent_dir, metric_dirs[0])

    sub_folders2 = [f for f in sub_folders if 'metric' not in f]
    result_dir = os.path.join(agent_dir, sub_folders2[0])

    # -------------------------
    # 1) 读取 policy 序列 + incidence
    # -------------------------
    all_policy = {state: [] for state in state_list}
    daily_incidence_rate = 0.0

    for state in state_list:
        log_path = os.path.join(result_dir, state + '.log')
        control_states = extract_policy_states_from_textfile(log_path, keep_duplicates=True)

        mean_df = pd.read_csv(os.path.join(metric_dir, f"{state}_metrics_mean_{policy}.csv")).reset_index(drop=True)
        daily_incidence_rate += mean_df[metric_cols].to_numpy(dtype=float) * pop_ratios[state]

        all_policy[state] = control_states

    # 每14天平均（有余数则单独平均）
    arr = np.asarray(daily_incidence_rate, dtype=float)
    block = 14
    means = [arr[i:i+block].mean() for i in range(0, len(arr), block)]
    daily_incidence_rate_14d = np.array(means, dtype=float)

    # -------------------------
    # 2) 计算每个 round 被选中州的概率（叠状柱）
    # -------------------------
    df_shares = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_policy.items()]))
    freq_df = df_shares.apply(lambda row: row.value_counts(), axis=1).fillna(0).astype(int)
    freq_df = freq_df / freq_df.sum(axis=1).values[:, None]

    for state in state_list:
        if state not in freq_df.columns:
            freq_df[state] = 0.0

    weeks = freq_df.index.to_numpy() + 1
    bottom = np.zeros(len(weeks), dtype=float)

    for i, s in enumerate(state_list):
        vals = freq_df[s].to_numpy(dtype=float)
        ax.bar(
            weeks, vals,
            bottom=bottom,
            width=0.82,
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.8,
            label=abbr.get(s, s.capitalize()),
            zorder=3
        )
        bottom += vals

    # 左轴样式
    ax.set_xlabel("Decision Round", fontsize=30, labelpad=10)
    if idx == 0:
        ax.set_ylabel("Selection Probability", fontsize=30, labelpad=10)
    else:
        ax.set_ylabel("")  # 右图不重复写左轴 ylabel，版面更干净

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=25)

    ax.set_xticks(weeks)
    ax.set_xticklabels(weeks, fontsize=25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    # 子图标题（可按 Nature 风格简洁命名）
    ax.set_title(policy_label[idx].upper(), fontsize=24, pad=12)

    # -------------------------
    # 3) 右轴折线：14-day avg incidence
    # -------------------------
    ax2 = ax.twinx()
    ax2_list.append(ax2)

    y2 = np.asarray(daily_incidence_rate_14d, dtype=float)
    m = min(len(weeks), len(y2))
    weeks2 = weeks[:m]
    y2 = y2[:m]

    ax2.plot(
        weeks2, y2,
        color=line_color,
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=1.0,
        zorder=5,
        label="14-day avg IR"
    )

    # 右轴 ylabel：只在最右侧子图显示（避免重复）
    if idx == len(policys) - 1:
        ax2.set_ylabel("IR (per 100k)", fontsize=30, labelpad=12, color=line_color)
    else:
        ax2.set_ylabel("")

    ax2.tick_params(axis="y", labelsize=25, width=1.2, colors=line_color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["right"].set_color(line_color)

    # 科学计数法（可选；如果不想用就删掉三行）
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax2.yaxis.get_offset_text().set_fontsize(20)
    ax2.yaxis.get_offset_text().set_color(line_color)

    # -------------------------
    # 4) 收集全局 legend（只收集一次）
    # -------------------------
    if legend_handles is None:
        h1, l1 = ax.get_legend_handles_labels()   # 各州堆叠柱
        h2, l2 = ax2.get_legend_handles_labels()  # 折线
        # 合并：州 + incidence
        legend_handles = h1 + h2
        legend_labels  = l1 + l2
# -------------------------
# 全局 legend（放在大图上方，字号更大）
# -------------------------
fig.legend(
    legend_handles, legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=len(state_list) + 1,     # 5个州 + 1条折线
    frameon=False,
    fontsize=25,
    columnspacing=1.6,
    handlelength=1.8
)
# 给 legend 留空间
plt.tight_layout(rect=[0, 0, 1, 0.98])
out_path = os.path.join(output_dir, "policy_selection_share_two_panels.jpg")
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.show()
print("Saved:", out_path)