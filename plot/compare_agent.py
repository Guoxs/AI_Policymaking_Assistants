##### 对不不同agent的策略
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",      
    "axes.unicode_minus": False      
})


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_long_bar_circular_chart(
    df,
    metric_cols=("IR", "ACR", "DI"),
    metric_labels=("IR", "ACR", "DI"),
    agent_name="GPT-3.5",
    state_order=None,
    hole_radius=3,         # 1. 极小内环，给柱子腾出空间
    visual_scale=1,       # 2. 视觉放大系数：原始值 0.05 * 40 = 2 个单位长度，让柱子变长
    reference_vals=[0.05, 0.10, 0.15, 0.20] # 真实的原始值刻度
):
    if agent_name == 'gpt-3.5':
        agent_name = 'GPT-3.5'
    elif agent_name == 'gpt-4.1-mini':
        agent_name = 'GPT-4.1-Mini'
    elif agent_name == 'gemini-2.5':
        agent_name = 'Gemini-2.5'
    elif agent_name == 'llama-8b':
        agent_name = 'LLaMA-3-8B'
    elif agent_name == 'qwen-7b':
        agent_name = 'Qwen2.5-7B'
    elif agent_name == 'qwen-72b':
        agent_name = 'Qwen2.5-72B'
  
    # --- 1. 数据准备 ---
    if state_order:
        df = df.set_index('abbr').reindex(state_order).reset_index()
    abbrs = df['abbr'].values
    n_states = len(abbrs)
    n_metrics = len(metric_cols)
    # --- 2. 角度布局 ---
    full_circle = 2 * np.pi
    sector_width = full_circle / n_metrics
    sector_gap = 0.3 
    usable_sector = sector_width - sector_gap
    slot_width = usable_sector / n_states
    bar_width = slot_width * 0.85

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.subplot(111, polar=True)
    ax.set_theta_direction(-1) 
    ax.set_theta_offset(np.pi/2 + sector_gap/2)
    ax.set_axis_off()
    
    # 动态计算最大视觉高度，确保标签不超出
    max_raw_val = df[list(metric_cols)].max().max()
    max_h = max(max_raw_val, max(reference_vals)) * visual_scale
    ax.set_ylim(0, hole_radius + max_h + 3)

    colors = ["#236085", "#3E9671", "#49A8AD"] 

    # --- 3. 绘制背景刻度环 (基于原始值标注) ---
    for val in reference_vals:
        r_pos = hole_radius + (val * visual_scale)
        if val == reference_vals[-1]:
            ax.plot(np.linspace(0, 2*np.pi, 200), [r_pos]*200, color="black", lw=1)
        else:
            ax.plot(np.linspace(0, 2*np.pi, 200), [r_pos]*200, color="gray", linestyle=(0, (5, 8)), lw=1, alpha=0.8)
            ax.text(0, r_pos, f"{val}", color="black", fontsize=18, ha="center", va="center")

    # --- 4. 循环绘制 ---
    for j, m in enumerate(metric_cols):
        sector_start = j * sector_width
        # 3. 内部圆弧：紧贴柱子底部
        arc_r = hole_radius - 0.5
        arc_angles = np.linspace(sector_start, sector_start + usable_sector, 100)
        ax.plot(arc_angles, [arc_r]*100, color=colors[j], lw=2, solid_capstyle='round')

        # 4. 内部 Label：放在圆弧与圆心之间，调小字号防止遮挡
        metric_center_theta = sector_start + usable_sector/2
        ax.text(metric_center_theta, hole_radius * 0.8, metric_labels[j],
                ha="center", va="center", fontsize=20, fontweight="black", color=colors[j])
        
        for i, ab in enumerate(abbrs):
            v = float(df.iloc[i][m])
            h = v * visual_scale # 应用视觉放大
            if v<=0:
                h= 0
            theta = sector_start + (i + 0.5) * slot_width
            ax.bar(theta, h, width=bar_width, bottom=hole_radius,
                   color=colors[j], edgecolor="white", linewidth=0.7, alpha=0.9, zorder=2)

            # 5. 外部州名：统一对齐
            label_r = hole_radius + max_h + 3.5
            ax.text(theta, label_r, ab, ha="center", va="center", fontsize=24, fontweight='bold')

    # --- 5. 圆心 Agent 装饰 ---
    ax.text(0, 0, agent_name, ha="center", va="center", 
            fontsize=27, fontweight="black", color="#6E6E6E")

    plt.tight_layout()
    plt.savefig('figures/5 states/agent compare/'+agent_name+'_long_bar_circular_chart.png', dpi=300)
    return fig

week_freq = '6 weeks'
agent_list = ['gpt-3.5', 'gpt-4.1-mini', 'gemini-2.5', 'llama-8b','qwen-7b','qwen-72b']
for i in range(len(agent_list)):
    agent_name = agent_list[i]
    results_folder = f'outputs\\5 states\\{week_freq}\\{agent_name}\\metrics\\'
    state_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    metric_cols = ['incidence_rate_7d', 'active_case_ratio', 'death_incidence_7d']
    # state = state_list[4]  # Example for one state
    # metric_gt = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_gt_{week_freq}.csv"))
    # metric_mean = pd.read_csv(os.path.join(results_folder, f"{state}_metrics_mean_{week_freq}.csv"))
    # state_IR = (metric_gt[metric_cols[0]].mean() - metric_mean[metric_cols[0]].mean())/metric_gt[metric_cols[0]].mean()
    # state_ACR = (metric_gt[metric_cols[1]].mean() - metric_mean[metric_cols[1]].mean())/metric_gt[metric_cols[1]].mean()
    # state_DI = (metric_gt[metric_cols[2]].mean() - metric_mean[metric_cols[2]].mean())/metric_gt[metric_cols[2]].mean()
    # print(f"Agent: {agent_name}, State: {state}")
    # print(f"Average Incidence Rate: {state_IR}")
    # print(f"Average Active Case Ratio: {state_ACR}")
    # print(f"Average Death Incidence: {state_DI}")
    metric_labels = ['Incidence\nRate', 'Active\nCase Ratio', 'Death\nIncidence']  # 雷达图轴标签（更紧凑）

    STATE_ABBR_5 = {
        "arizona": "AZ",
        "mississippi": "MS",
        "new mexico": "NM",
        "texas": "TX",
        "virginia": "VA",
    }

    rows = []
    for state in state_list:
        gt_path = os.path.join(results_folder, f"{state}_metrics_gt_{week_freq}.csv")
        mean_path = os.path.join(results_folder, f"{state}_metrics_mean_{week_freq}.csv")

        metric_gt = pd.read_csv(gt_path)
        metric_mean = pd.read_csv(mean_path)

        vals = []
        for col in metric_cols:
            gt_m = metric_gt[col].mean()
            ag_m = metric_mean[col].mean()
            v = (gt_m - ag_m) / gt_m if gt_m != 0 else np.nan
            vals.append(v)

        rows.append({
            "state": state,
            "abbr": STATE_ABBR_5[state],
            "IR": vals[0],
            "ACR": vals[1],
            "DI": vals[2],
        })
    df_raw = pd.DataFrame(rows)

    # --- 执行绘图 ---
    fig = plot_long_bar_circular_chart(
        df_raw, 
        metric_labels=["IR", "ACR", "DR"],
        agent_name=agent_name,
        state_order=["AZ","MS","NM","TX","VA"],
        hole_radius=20,         # 缩小内环
        visual_scale=50,       # 增加这个值，柱子会变得更长、更修长
        reference_vals=[0.25,  0.5,  0.75] # 根据你的数据范围调整刻度线,
    )
    plt.close()