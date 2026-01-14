### 绘制不同政策类别的回归分析

import os
import pandas as pd
from plot_type_classify import extract_policy_responses, aggregate_to_3_periods
import matplotlib.pyplot as plt
import re
import re
from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl
# ---------- 全局字体 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False
})

STATE_BLOCK_RE = re.compile(
    r"^\s*\d+\.\s*(?P<state>[a-z ]+)\s*\n"                 # 1. mississippi
    r"\s*-\s*Avg past 21-day population composition:\s*"   # population line prefix
    r"S=(?P<S>\d+),\s*E=(?P<E>\d+),\s*I=(?P<I>\d+),\s*Q=(?P<Q>\d+),\s*R=(?P<R>\d+),\s*D=(?P<D>\d+)\s*$",
    flags=re.MULTILINE
)

def parse_other_origin_states_pop(log_path) -> pd.DataFrame:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()
    anchor = log_text.find("Other origin states:")
    if anchor != -1:
        log_text = log_text[anchor:]
    rows = []
    for m in STATE_BLOCK_RE.finditer(log_text):
        d = m.groupdict()
        d["state"] = d["state"].strip()
        for k in ["S", "E", "I", "Q", "R", "D"]:
            d[k] = int(d[k])
        rows.append(d)

    return pd.DataFrame(rows, columns=["state", "S", "E", "I", "Q", "R", "D"])


HEADER_RE = re.compile(r"^\s*Current population composition:\s*$", re.MULTILINE)
KV_LINE_RE = re.compile(r"^\s*-\s*([SEIQRD])\s*=\s*(\d+)\s*$", re.MULTILINE)
def parse_current_population_composition(log_path: str) -> Optional[Dict[str, int]]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()
    rows  = []
    for m in HEADER_RE.finditer(log_text):
    # Take a window after the header to avoid matching unrelated later sections
        start = m.end()
        window = log_text[start:start + 500]  # 500 chars is enough for 6 lines
        values = {}
        for k, v in KV_LINE_RE.findall(window):
            values[k] = int(v)
        # Require all 6 keys to consider it valid; otherwise return partial if you prefer
        required = ["S", "E", "I", "Q", "R", "D"]
        if not all(k in values for k in required):
            return None
        rows.append(values)
    return pd.DataFrame(rows, columns=required)


ANCHOR = "Ground truth daily average for next 42-day inbound by origin states"
LINE_RE = re.compile(
    r"^\s*From\s+(?P<state>[a-z ]+):\s*Inflow\s*=\s*(?P<inflow>\d+(?:\.\d+)?)\s*;\s*$",
    flags=re.MULTILINE | re.IGNORECASE
)
REGION_RE = re.compile(r"Region\s+Agent\s+(?P<region>[a-z ]+)", re.IGNORECASE)
def parse_gt_42d_inbound_inflow(log_path: str) -> pd.DataFrame:
    # 尽量从锚点之后截取，避免误匹配其它“From ... Inflow ...”段落
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()
    anchors = [m.start() for m in re.finditer(re.escape(ANCHOR), log_text)]
    if not anchors:
        return pd.DataFrame(columns=["block_idx", "region", "origin_state", "inflow"])
    rows = []
    for i, start in enumerate(anchors):
        # 改动点 2：切块（start 到下一个 anchor 或 EOF），替代原来的 “idx:idx+2000”
        end = anchors[i + 1] if i + 1 < len(anchors) else len(log_text)
        block = log_text[start:end]
        # 可选：从锚点前 800 字符的上下文抓 region
        ctx = log_text[max(0, start - 800):start]
        mreg = REGION_RE.search(ctx)
        region = mreg.group("region").strip().lower() if mreg else None
        # 原逻辑不变：逐行匹配 inflow
        for m in LINE_RE.finditer(block):
            rows.append({
                "block_idx": i,
                "region": region,
                "origin_state": m.group("state").strip().lower(),
                "inflow": float(m.group("inflow")),
            })
    return pd.DataFrame(rows, columns=["block_idx", "region", "origin_state", "inflow"])




results_root = "outputs\\5 states\\6 weeks\\gpt-4.1-mini"
states_list = ['arizona', 'mississippi', 'new mexico', 'texas', 'virginia']
calculate_df = {'ori_state': [], 'des_state': [], 'ori_E':[], 'ori_I':[], 'ori_R':[], 'ori_D':[], 'des_E':[], 'des_I':[], 'des_R':[], 'des_D':[], 'inflow': [], 'policy_type': []}
for i in range(len(states_list)):
    log_file = f"{states_list[i]}.log"
    states = [ s for s in states_list if s != states_list[i] ]
    folders = [f for f in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, f))]
    for last_folder in folders:
        print("Processing folder:", last_folder)
        agent_folder = os.path.join(results_root, last_folder)
        log_path = os.path.join(agent_folder, log_file)
        print("Processing log file:", log_path)
        responses = extract_policy_responses(log_path)
        pandemic_info = parse_other_origin_states_pop(log_path)
        own_pop = parse_current_population_composition(log_path)
        origin_inflow = parse_gt_42d_inbound_inflow(log_path)
        for s in states:
            pandemic_state = pandemic_info[pandemic_info['state'] == s]
            inflow_value = origin_inflow[origin_inflow['origin_state'] == s]
            for j, resp in enumerate(responses):
                six_weeks = resp['policy'][s]
                three_periods = aggregate_to_3_periods(six_weeks)
                if three_periods[0] <= 0.3 and three_periods[2] >= 0.4:
                    calculate_df['policy_type'].append(1)  # strict_first
                elif three_periods[0] >= 0.4 and three_periods[2] <= 0.3:
                    calculate_df['policy_type'].append(-1)  # relaxed_first
                else:
                    calculate_df['policy_type'].append(0)  # balanced
                calculate_df['ori_state'].append(states_list[i])
                calculate_df['des_state'].append(s)
                calculate_df['ori_E'].append(pandemic_state['E'].values[j])
                calculate_df['ori_I'].append(pandemic_state['I'].values[j] + pandemic_state['Q'].values[j])
                calculate_df['ori_R'].append(pandemic_state['R'].values[j])
                calculate_df['ori_D'].append(pandemic_state['D'].values[j])
                calculate_df['des_E'].append(own_pop['E'].values[j])
                calculate_df['des_I'].append(own_pop['I'].values[j] + own_pop['Q'].values[j])
                calculate_df['des_R'].append(own_pop['R'].values[j])
                calculate_df['des_D'].append(own_pop['D'].values[j])
                calculate_df['inflow'].append(inflow_value['inflow'].values[j])
calculate_df = pd.DataFrame(calculate_df)
print(calculate_df.head())
print(len(calculate_df))




target = "policy_type"
# 特征列：保留你给出的全部解释变量
feature_cols = [
    "ori_state", "des_state",
    "ori_E", "ori_I", "ori_R", "ori_D",
    "des_E", "des_I", "des_R", "des_D",
    "inflow"
]
df = calculate_df[feature_cols + [target]].copy()
choice_state = 'texas'
df = df[df['des_state']==choice_state ]  # 只分析德克萨斯州作为目的地的流入

STATE_ABBR_5 = {
    "arizona": "AZ",
    "mississippi": "MS",
    "new mexico": "NM",
    "texas": "TX",
    "virginia": "VA",
}
df["ori_state"] = (
    df["ori_state"]
    .str.strip()
    .str.lower()
    .map(STATE_ABBR_5)
)
df["des_state"] = (
    df["des_state"]
    .str.strip()
    .str.lower()
    .map(STATE_ABBR_5)
)
X = df[feature_cols]
y = df[target].astype(float)   # 回归：转 float 更一致
# ========== 2) 类别/数值列划分 ==========
cat_cols = [ "ori_state", "des_state"]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

# ========== 3) 模型 ==========
rf = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", rf),
])

# ========== 4) 切分与训练 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model.fit(X_train, y_train)

# ========== 5) 简单评估 ==========
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)
print(f"Test MAE : {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R^2 : {r2:.4f}")
# ========== 6) SHAP 归因 ==========
# 树模型推荐用 TreeExplainer；关键点：对“预处理后的矩阵”做 SHAP
import shap
# 取出预处理后的特征矩阵（稀疏/稠密都可）
X_train_enc = model.named_steps["prep"].transform(X_train)
X_test_enc  = model.named_steps["prep"].transform(X_test)
# 获取 one-hot 后的特征名
ohe = model.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_feature_names = cat_feature_names + num_cols
# 若是稀疏矩阵，shap 有时更稳定用稠密（数据很大就别 toarray）
if hasattr(X_test_enc, "toarray"):
    X_test_for_shap = X_test_enc.toarray()
    X_train_for_shap = X_train_enc.toarray()
else:
    X_test_for_shap = X_test_enc
    X_train_for_shap = X_train_enc
explainer = shap.TreeExplainer(model.named_steps["rf"])
shap_values = explainer.shap_values(X_test_for_shap)
# ========== 7) 可视化 ==========
# # 7.1 全局重要性（beeswarm）
# shap.summary_plot(
#     shap_values,
#     X_test_for_shap,
#     feature_names=all_feature_names,
#     show=True
# )
# 7.2 全局重要性（bar）
def plot_signed_shap_bar(shap_values, feature_names, top_k=12, sort_by="abs"):
    sv = np.array(shap_values)
    mean_signed = sv.mean(axis=0)                 # 保留正负方向
    mean_abs = np.abs(sv).mean(axis=0)            # 用于排序（可选）
    df_bar = pd.DataFrame({
        "feature": feature_names,
        "mean_shap": mean_signed,
        "mean_abs_shap": mean_abs
    })
    if sort_by == "signed":
        df_bar = df_bar.sort_values("mean_shap", ascending=False)
    else:
        df_bar = df_bar.sort_values("mean_abs_shap", ascending=False)
    df_bar = df_bar.head(top_k).iloc[::-1]  # 反转便于水平bar从下往上显示
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    colors = []
    for v in df_bar["mean_shap"].values:
        if v > 0.02:
            colors.append("#5d8eb9")
        elif v < 0:
            colors.append("#f2eda0")
        else:
            colors.append("#add9b8")
    bars = ax.barh(
        df_bar["feature"],
        df_bar["mean_shap"],
        height=0.45,
        color=colors,
        edgecolor="black"
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Shapley Values", fontsize=20)
    #ax.set_xticklabels([-0.02, 0,0.02,0.04,0.06,0.08], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("")
    # ---- 新增：标注数值到柱子边上（自动判定正负放置位置）----
    xmax = np.max(np.abs(df_bar["mean_shap"].values)) if len(df_bar) else 1.0
    pad = 0.02 * xmax  # 与柱子末端的间距，可调大/调小
    for bar, v in zip(bars, df_bar["mean_shap"].values):
        y = bar.get_y() + bar.get_height() / 2
        if v >= 0:
            x = v + pad
            ha = "left"
        else:
            x = v - pad
            ha = "right"
        ax.text(
            x, y,
            f"{v:.3f}",       # 保留三位小数；你也可改成 :.2f 或科学计数法
            va="center",
            ha=ha,
            fontsize=11
        )
    # 为了避免文字被裁切，给 x 轴留一点边距
    ax.set_xlim(
        df_bar["mean_shap"].min() - 0.15 * xmax,
        df_bar["mean_shap"].max() + 0.15 * xmax
    )
    plt.tight_layout()
    plt.savefig(f"D:/MyDownload/Code/OD-COVID/figures/5 states/{choice_state}/shap_policy_regression_{choice_state}.png", dpi=300)
    plt.show()

plot_signed_shap_bar(
    shap_values=shap_values,
    feature_names=all_feature_names,
    top_k=12,
    sort_by="abs"   # 推荐：按重要性(绝对值)排序，但显示方向
)
# shap.summary_plot(
#     shap_values,
#     X_test_for_shap,
#     feature_names=all_feature_names,
#     plot_type="bar",
#     show=True
# )
# # 7.3 你最关心的单个特征依赖图：例如 inflow
# shap.dependence_plot(
#     "inflow",
#     shap_values,
#     X_test_for_shap,
#     feature_names=all_feature_names,
#     show=True
# )
