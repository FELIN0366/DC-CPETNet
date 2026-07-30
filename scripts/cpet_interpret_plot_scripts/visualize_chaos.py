import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# ================= 路径适配 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

try:
    from feature_mapping import IDX
    from data_preprocess_new import get_data_new, resample_to_fixed_length
    from label_extractor import load_labels, get_label_for_file
except ImportError as e:
    st.error(f"无法导入 src 模块: {e}")
    st.stop()

# ================= 配置 =================
DATA_ROOT = r"xx_path"
LABEL_FILE = r"xx_path"
CACHE_DIR = os.path.join(os.path.dirname(DATA_ROOT), "npy_cache")

CLASS_NAMES = {
    0: "Class 0: 去适应/高危 (Normal-ish)",
    1: "Class 1: 心律失常 (Arrhythmia)",
    3: "Class 3: 缺血性心脏病 (Ischemia)" # 对照组
}

# ================= 数据加载 =================
@st.cache_data
def load_metadata():
    if not os.path.exists(LABEL_FILE): return None, None
    label_dict, label_mapping = load_labels(LABEL_FILE)
    
    # 构建分类索引
    class_files = {i: [] for i in range(6)}
    for f in os.listdir(DATA_ROOT):
        if not f.endswith('.xlsx'): continue
        lbl = get_label_for_file(f, label_dict)
        if lbl and lbl in label_mapping:
            class_files[label_mapping[lbl]].append(os.path.join(DATA_ROOT, f))
    return class_files

def get_hr_data(filepath, use_resample=False):
    """
    获取 HR 数据。
    如果 use_resample=True，则模拟训练时的 162 点重采样。
    """
    # 1. 强制获取原始数据 (target_length=None)
    raw_data = get_data_new(filepath, target_length=None, cache_dir=CACHE_DIR)
    
    # 提取 HR (IDX['HR'] = 3)
    hr_raw = raw_data[:, IDX['HR']]
    
    # 2. 如果需要，进行重采样对比
    if use_resample:
        # 必须先对整体数据重采样，模拟 Dataset 的行为
        resampled_all = resample_to_fixed_length(raw_data, target_length=162)
        hr_resampled = resampled_all[:, IDX['HR']]
        return hr_resampled
    else:
        return hr_raw

# ================= 绘图逻辑 =================

def plot_chaos_analysis(file_a, file_b, label_a, label_b, use_resample):
    """
    绘制 1x3 对比图：时域 -> 差分域 -> 庞加莱图
    """
    # 加载数据
    hr_a = get_hr_data(file_a, use_resample)
    hr_b = get_hr_data(file_b, use_resample)
    
    # 1. 计算差分 (Differential)
    # diff[t] = HR[t] - HR[t-1]
    diff_a = np.diff(hr_a)
    diff_b = np.diff(hr_b)
    
    # 2. 准备庞加莱图数据 (HR_t vs HR_t+1)
    x_poincare_a, y_poincare_a = hr_a[:-1], hr_a[1:]
    x_poincare_b, y_poincare_b = hr_b[:-1], hr_b[1:]

    # 创建画布
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"时域 HR (Time Domain)<br>{label_a}", 
            f"差分波动 (Chaos/Diff)<br>{label_a}", 
            f"庞加莱图 (Poincaré)<br>{label_a}",
            f"时域 HR (Time Domain)<br>{label_b}", 
            f"差分波动 (Chaos/Diff)<br>{label_b}", 
            f"庞加莱图 (Poincaré)<br>{label_b}"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # --- Row 1: 病人 A (例如 Arrhythmia) ---
    # 1.1 时域
    fig.add_trace(go.Scatter(y=hr_a, mode='lines', name=f'{label_a} HR', line=dict(color='#ff7f0e')), row=1, col=1)
    
    # 1.2 差分 (关键！)
    fig.add_trace(go.Scatter(y=diff_a, mode='lines', name=f'{label_a} Diff', line=dict(color='#ff7f0e', width=1)), row=1, col=2)
    
    # 1.3 庞加莱
    fig.add_trace(go.Scatter(x=x_poincare_a, y=y_poincare_a, mode='markers', marker=dict(size=4, color='#ff7f0e', opacity=0.6), name=f'{label_a} Poincaré'), row=1, col=3)
    # 添加对角线
    min_val, max_val = min(hr_a.min(), hr_b.min()), max(hr_a.max(), hr_b.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"), row=1, col=3)

    # --- Row 2: 病人 B (例如 Normal) ---
    # 2.1 时域
    fig.add_trace(go.Scatter(y=hr_b, mode='lines', name=f'{label_b} HR', line=dict(color='#1f77b4')), row=2, col=1)
    
    # 2.2 差分
    fig.add_trace(go.Scatter(y=diff_b, mode='lines', name=f'{label_b} Diff', line=dict(color='#1f77b4', width=1)), row=2, col=2)
    
    # 2.3 庞加莱
    fig.add_trace(go.Scatter(x=x_poincare_b, y=y_poincare_b, mode='markers', marker=dict(size=4, color='#1f77b4', opacity=0.6), name=f'{label_b} Poincaré'), row=2, col=3)
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"), row=2, col=3)

    # 布局
    fig.update_layout(height=700, title_text=f"混沌特征诊断: {label_a} vs {label_b} ({'重采样后' if use_resample else '原始数据'})")
    return fig

# ================= Streamlit UI =================
st.set_page_config(layout="wide", page_title="HR 混沌诊断")
st.title("🫀 心律失常(Chaos)特征可视诊断")
st.markdown("""
**诊断逻辑**：
1. **差分图**：心律失常应表现为剧烈的上下震荡（幅度大），正常人应在 0 附近微小波动。
2. **庞加莱图**：正常人应沿对角线聚集（Cigar shape）；心律失常应发散（Scattered）。
**如果在这里看不出区别，神经网络也学不到。**
""")

class_files = load_metadata()
if not class_files: st.stop()

# Sidebar Controls
st.sidebar.header("数据选择")

# 对比组 A (目标：心律失常)
idx_a = 1 
files_a = class_files[idx_a]
if not files_a: st.error("没有 Class 1 样本"); st.stop()

# 对比组 B (对照：去适应/正常)
idx_b = 0 
files_b = class_files[idx_b]

# 随机抽样
if st.sidebar.button("🔄 随机抽取样本"):
    st.session_state['file_a'] = np.random.choice(files_a)
    st.session_state['file_b'] = np.random.choice(files_b)

if 'file_a' not in st.session_state:
    st.session_state['file_a'] = files_a[0]
    st.session_state['file_b'] = files_b[0]

# 核心开关：是否重采样
st.sidebar.markdown("---")
use_resample = st.sidebar.checkbox("模拟训练时的重采样 (162点)", value=False, 
                                   help="勾选后，展示模型实际『看到』的数据。如果不勾选，展示原始高频数据。")

st.sidebar.markdown("---")
st.sidebar.info(f"当前样本:\nA: {os.path.basename(st.session_state['file_a'])}\nB: {os.path.basename(st.session_state['file_b'])}")

# 绘图
fig = plot_chaos_analysis(
    st.session_state['file_a'], 
    st.session_state['file_b'], 
    "Class 1 (Arrhythmia)", 
    "Class 0 (Normal)", 
    use_resample
)
st.plotly_chart(fig, use_container_width=True)

# 统计指标计算
def calc_metrics(hr):
    diff = np.diff(hr)
    return {
        "Mean": np.mean(hr),
        "Std (HRV proxy)": np.std(hr),
        "Diff Std (Chaos proxy)": np.std(diff), # 你的核心指标
        "Max Jump": np.max(np.abs(diff))
    }

metrics_a = calc_metrics(get_hr_data(st.session_state['file_a'], use_resample))
metrics_b = calc_metrics(get_hr_data(st.session_state['file_b'], use_resample))

st.table(pd.DataFrame([metrics_a, metrics_b], index=["Class 1 (Arrhythmia)", "Class 0 (Normal)"]))

if use_resample:
    st.warning("⚠️ **注意观察**：重采样后，'Diff Std' (波动率) 是否显著下降？如果下降很多，说明预处理把心律失常的特征当做噪声过滤掉了。")
