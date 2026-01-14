#### 绘制不同政策类别的统计

import sys
import pandas as pd
import numpy as np
import os
import re
import ast
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def aggregate_to_3_periods(eight_weeks):
    if len(eight_weeks) != 6:
        raise ValueError(f"expect 6 weeks, got {len(eight_weeks)}")
    return [
        eight_weeks[0] + eight_weeks[1],
        eight_weeks[2] + eight_weeks[3],
        eight_weeks[4] + eight_weeks[5],
    ]

def extract_policy_responses(log_path: str):
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    # 匹配整行的 Parsed PolicyResponse 段
    line_pattern = re.compile(r"Parsed PolicyResponse:\s*(.*)")
    results = []
    with log_path.open("r", encoding="utf-8",errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = line_pattern.search(line)
            if not m:
                continue
            # 拿到 'policy=... explanation=...' 这一整段
            payload = m.group(1).strip()
            # 再用一个正则，把 policy 和 explanation 拆开
            # 注意这里 policy 用贪婪匹配，到最后一个 " explanation=" 为止
            m2 = re.match(r"policy=(\{.*\})\s+explanation=(.*)", payload)
            if not m2:
                # 如果格式跟预期不符，可以打印出来调试
                print(f"[WARN] Cannot parse payload: {payload}", file=sys.stderr)
                continue
            policy_str = m2.group(1)
            explanation_str = m2.group(2)
            # 组装成一个完整的 Python 字面量，再用 ast.literal_eval 解析
            dict_source = "{'policy': " + policy_str + ", 'explanation': " + explanation_str + "}"
            try:
                obj = ast.literal_eval(dict_source)
                results.append(obj)
            except Exception as e:
                print(f"[ERROR] Failed to eval: {e}", file=sys.stderr)
                print(f"         source: {dict_source[:200]}...", file=sys.stderr)

    return results

STATE_ABBR_5 = {
    "arizona": "AZ",
    "mississippi": "MS",
    "new mexico": "NM",
    "texas": "TX",
    "virginia": "VA",
}

def main():
    # Get the last folder in the specified root directory
    results_root = "outputs\\5 states\\6 weeks\\gpt-4.1-mini"

    states_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
    for i in range(len(states_list)):
        log_file = f"{states_list[i]}.log"
        states = [ s for s in states_list if s != states_list[i] ]
        type_count = {s: {'strict_first': 0, 'relaxed_first': 0, 'balanced': 0} for s in states}
        folders = [f for f in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, f))]
        for last_folder in folders:
            print("Processing folder:", last_folder)
            agent_folder = os.path.join(results_root, last_folder)
            log_path = os.path.join(agent_folder, log_file)
            print("Processing log file:", log_path)
            responses = extract_policy_responses(log_path)
            for s in states:
                for resp in responses:
                    six_weeks = resp['policy'][s]
                    three_periods = aggregate_to_3_periods(six_weeks)
                    if three_periods[0] <= 0.3 and three_periods[2] >= 0.4:
                        type_count[s]['strict_first'] += 1
                    elif three_periods[0] >= 0.4 and three_periods[2] <= 0.3:
                        type_count[s]['relaxed_first'] += 1
                    else:
                        type_count[s]['balanced'] += 1
        print(type_count)
        cats = ["strict_first", "balanced", "relaxed_first"]   # 推荐把 balanced 放中间，语义像“基线”
        cat_labels = {
            "strict_first": "Strict-first",
            "balanced": "Balanced",
            "relaxed_first": "Relaxed-first"
        }
        # 组装矩阵
        counts = np.array([[type_count[s][c] for c in cats] for s in states], dtype=float)
        totals = counts.sum(axis=1, keepdims=True)
        props = np.divide(counts, totals, out=np.zeros_like(counts), where=totals>0)
        order = np.argsort(-props[:, 0])
        states = [states[i] for i in order]
        props = props[order, :]
        # --- 画图参数（Nature-like）
        plt.rcParams.update({
            "font.family": "Times New Roman",   # 若你本地有 Arial/Helvetica 可替换
            "font.size": 14,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        })

        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=300)  # 单栏/双栏可按版式改

        x = np.arange(len(states))
        bottom = np.zeros(len(states))

        # 颜色：不显式指定也可，但你要求“高端”，建议用灰阶+一处强调色
        # 这里用：深灰(Strict) / 浅灰(Balanced) / 中灰(Relaxed)，保证印刷友好
        colors = {
            "strict_first":  '#5d8eb9',  # Blue
            "balanced":      '#add9b8',  # Orange
            "relaxed_first": '#f2eda0',  # Bluish green
        }

        for j, c in enumerate(cats):
            ax.bar(
                x, props[:, j],
                bottom=bottom,
                width=0.42,
                edgecolor="white",
                linewidth=0.6,
                label=cat_labels[c],
                color=colors[c],
            )
            bottom += props[:, j]

        # 坐标轴与网格（极简）
        ax.set_ylim(0, 1.0)
        #ax.set_ylabel(f"Policy Type for {STATE_ABBR_5[states_list[i]]}", labelpad=6, fontsize=19)
        ax.set_title(
            f"Policy Type for {STATE_ABBR_5[states_list[i]]}",
            pad=6,
            fontsize=22
        )
        ax.set_xticks(x)
        ax.set_xticklabels([STATE_ABBR_5[s] for s in states], fontsize=20)
        ax.yaxis.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.grid(axis="y", linewidth=0.6, alpha=0.25)
        ax.set_axisbelow(True)
        # 去掉上/右边框（Nature 常见）
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 图例：上方横排，省空间
        if i == 0:
            leg = ax.legend(
                ncol=3, frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, 1.20),
                handlelength=1.2, columnspacing=1.5, fontsize=18
            )
        #ax.set_title("Policy sequencing types across states", pad=18)
        fig.tight_layout()
        plt.savefig(f"D:/MyDownload/Code/OD-COVID/figures/5 states/mississippi/policy_{log_file[:5]}.png", dpi=300)
        #plt.show()

if __name__ == "__main__":
    main()