import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import sys

# ================= 路径适配 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

try:
    from feature_mapping import IDX
    from data_preprocess_new import get_data_new
    from label_extractor import load_labels, get_label_for_file
except ImportError as e:
    st.error(f"无法导入 src 模块: {e}")
    st.stop()

# ================= 配置 =================
DATA_ROOT = r"xx_path"
LABEL_FILE = r"xx_path"
CACHE_DIR = os.path.join(os.path.dirname(DATA_ROOT), "npy_cache")

CLASS_NAMES = {
    0: "Class 0: 正常/去适应",
    1: "Class 1: 心律失常",
    3: "Class 3: 缺血性心脏病"
}

# ================= 数据加载 =================
@st.cache_data
def load_metadata():
    if not os.path.exists(LABEL_FILE): return None
    label_dict, label_mapping = load_labels(LABEL_FILE)
    class_files = {i: [] for i in range(6)}
    for f in os.listdir(DATA_ROOT):
        if not f.endswith('.xlsx'): continue
        lbl = get_label_for_file(f, label_dict)
        if lbl and lbl in label_mapping:
            class_files[label_mapping[lbl]].append(os.path.join(DATA_ROOT, f))
    return class_files

def get_macro_features(filepath):
    """提取 HR vs Load 的线性特征"""
    # 强制使用原始长度，避免重采样干扰回归分析
    data = get_data_new(filepath, target_length=None, cache_dir=CACHE_DIR)
    
    # 提取 HR 和 Load
    hr = data[:, IDX['HR']]
    load = data[:, IDX['Load']]
    
    # 过滤掉静息期 (Load <= 0)，只分析运动期
    mask = load > 0
    if np.sum(mask) < 10: # 如果运动数据太少
        return None
        
    hr_ex = hr[mask]
    load_ex = load[mask]
    
    # 线性回归: HR = a * Load + b
    X = load_ex.reshape(-1, 1)
    y = hr_ex
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 计算指标
    r2 = r2_score(y, y_pred) # 线性度 (越接近1越线性)
    residuals = y - y_pred   # 残差
    max_residual = np.max(np.abs(residuals)) # 最大偏离度
    
    return {
        "load": load_ex, "hr": hr_ex, "hr_pred": y_pred,
        "r2": r2, "slope": model.coef_[0], "intercept": model.intercept_,
        "max_res": max_residual, "residuals": residuals
    }

# ================= 绘图 =================
def plot_macro_comparison(file_a, file_b, label_a, label_b):
    feat_a = get_macro_features(file_a)
    feat_b = get_macro_features(file_b)
    
    if not feat_a or not feat_b:
        st.error("数据点不足，无法计算回归")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"A: HR vs Load (线性度检查)<br>{label_a}", 
            f"B: HR vs Load (线性度检查)<br>{label_b}",
            f"A: 残差图 (偏离线性的程度)", 
            f"B: 残差图 (偏离线性的程度)"
        ],
        vertical_spacing=0.15
    )

    # --- Plot A (Arrhythmia) ---
    # 散点
    fig.add_trace(go.Scatter(x=feat_a['load'], y=feat_a['hr'], mode='markers', name=f'{label_a} Real', marker=dict(color='#ff7f0e', size=4, opacity=0.6)), row=1, col=1)
    # 拟合线
    fig.add_trace(go.Scatter(x=feat_a['load'], y=feat_a['hr_pred'], mode='lines', name=f'{label_a} Fit (R²={feat_a["r2"]:.3f})', line=dict(color='black', dash='dash')), row=1, col=1)
    # 残差
    fig.add_trace(go.Scatter(x=feat_a['load'], y=feat_a['residuals'], mode='lines', name=f'{label_a} Residuals', line=dict(color='#ff7f0e')), row=2, col=1)

    # --- Plot B (Normal) ---
    fig.add_trace(go.Scatter(x=feat_b['load'], y=feat_b['hr'], mode='markers', name=f'{label_b} Real', marker=dict(color='#1f77b4', size=4, opacity=0.6)), row=1, col=2)
    fig.add_trace(go.Scatter(x=feat_b['load'], y=feat_b['hr_pred'], mode='lines', name=f'{label_b} Fit (R²={feat_b["r2"]:.3f})', line=dict(color='black', dash='dash')), row=1, col=2)
    fig.add_trace(go.Scatter(x=feat_b['load'], y=feat_b['residuals'], mode='lines', name=f'{label_b} Residuals', line=dict(color='#1f77b4')), row=2, col=2)

    fig.update_layout(height=700, title_text="宏观非线性特征诊断 (Macro Non-linearity)")
    return fig, feat_a, feat_b

# ================= UI =================
st.set_page_config(layout="wide", page_title="宏观特征诊断")
st.title("📈 宏观非线性 (Macro-Nonlinearity) 诊断")
st.markdown("""
**诊断假设**：
由于数据被平滑，微观波动丢失。我们转而检测**HR随Load变化的线性度**。
* **正常人 (Class 0)**: HR 随 Load 线性增加，**R² 应该很高 (>0.95)**。
* **心律失常 (Class 1)**: 可能表现为心率突然跳变、平台期或应答滞后，导致 **R² 降低** 或 **最大残差(Max Residual) 增大**。
""")

class_files = load_metadata()
if not class_files: st.stop()

if st.button("🔄 随机抽取样本"):
    st.session_state['f_a'] = np.random.choice(class_files[1]) # Class 1
    st.session_state['f_b'] = np.random.choice(class_files[0]) # Class 0

if 'f_a' not in st.session_state:
    st.session_state['f_a'] = class_files[1][0]
    st.session_state['f_b'] = class_files[0][0]

# 绘图
res = plot_macro_comparison(st.session_state['f_a'], st.session_state['f_b'], "Class 1 (Arrhythmia)", "Class 0 (Normal)")

if res:
    fig, stats_a, stats_b = res
    st.plotly_chart(fig, use_container_width=True)
    
    # 统计对比表
    st.subheader("关键特征对比")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Class 1: R² (线性度)", f"{stats_a['r2']:.4f}", 
                delta=f"{stats_a['r2'] - stats_b['r2']:.4f}")
    
    col2.metric("Class 1: Max Residual (最大突变)", f"{stats_a['max_res']:.2f}",
                delta=f"{stats_a['max_res'] - stats_b['max_res']:.2f}", delta_color="inverse")
    
    st.info("如果 Class 1 的 R² 显著低于 Class 0，或者 Max Residual 显著更高，这才是模型能学到的特征！")
