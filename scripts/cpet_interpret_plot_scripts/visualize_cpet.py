import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import colorsys
# 运行命令：streamlit run scripts/visualize_cpet.py
# ================= 路径适配 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)

try:
    from feature_mapping import NEW_FEATURES, IDX
    from data_preprocess_new import get_data_new
    from label_extractor import load_labels, get_label_for_file
except ImportError as e:
    st.error(f"无法导入 src 模块。请检查路径结构。\n当前路径: {current_dir}\n尝试导入路径: {src_dir}\n错误信息: {e}")
    st.stop()

# ================= 配置区 =================
PROJECT_ROOT = os.path.join(current_dir, '..', '..')
DATA_ROOT = r"xx_path"
LABEL_FILE = r"xx_path"
CACHE_DIR = os.path.join(os.path.dirname(DATA_ROOT), "npy_cache")

# 22个原始特征索引 (不含医学先验衍生的4个)
ORIGINAL_FEATURES = NEW_FEATURES[:22]  # 索引 0-21
ORIGINAL_FEATURE_IDX = list(range(22))

# 动态生成类别颜色 (支持任意数量的类别)
def generate_class_colors(n_classes):
    """生成 n_classes 个视觉区分度高的颜色"""
    colors = []
    for i in range(n_classes):
        hue = i / n_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

# 9-Panel 配置
# 修改点：Panel 1 开启 dual_y (用于显示 Load)
PANEL_CONFIG = {
    1: {"name": "1. VO2 & VCO2 & Load vs Time", "dual_y": True}, 
    2: {"name": "2. HR & VO2/HR vs Time", "dual_y": True},
    3: {"name": "3. HR & VCO2 vs VO2", "dual_y": True},
    4: {"name": "4. VE/VO2 & VE/VCO2 vs Time", "dual_y": False},
    5: {"name": "5. VE vs Time", "dual_y": False},
    6: {"name": "6. VE vs VCO2", "dual_y": False},
    7: {"name": "7. PetO2 & PetCO2 vs Time", "dual_y": False},
    8: {"name": "8. RER vs Time", "dual_y": False},
    9: {"name": "9. VT vs VE", "dual_y": False}
}

# ================= 工具函数 =================
def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3: hex_color = hex_color * 2
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

# ================= 数据加载逻辑 =================
@st.cache_data
def load_metadata():
    """
    加载元数据，自适应获取当前人群分类
    Returns:
        class_files_map: {class_idx: [file_paths]}
        label_mapping: {label_name: class_idx}
        class_names: {class_idx: label_name}
        class_colors: {class_idx: hex_color}
    """
    if not os.path.exists(LABEL_FILE) or not os.path.exists(DATA_ROOT):
        return None, None, None, None

    label_dict, label_mapping = load_labels(LABEL_FILE)

    # 动态构建 class_names (反转 label_mapping)
    class_names = {idx: name for name, idx in label_mapping.items()}

    # 动态生成类别颜色
    n_classes = len(label_mapping)
    colors = generate_class_colors(n_classes)
    class_colors = {idx: colors[idx] for idx in range(n_classes)}

    # 按类别组织文件
    files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.xlsx')]
    class_files = {i: [] for i in range(n_classes)}

    for f in files:
        label_name = get_label_for_file(f, label_dict)
        if label_name and label_name in label_mapping:
            idx = label_mapping[label_name]
            class_files[idx].append(os.path.join(DATA_ROOT, f))

    return class_files, label_mapping, class_names, class_colors

def load_patient_data(filepath):
    # 保留原始数据长度
    return get_data_new(filepath,cache_dir=CACHE_DIR)

# ================= 绘图核心 =================
def add_panel_trace(fig, row, col, panel_id, d, p_name, color, line_width, legend_group, show_legend):
    """
    通用 Panel 绘制器
    """
    time = np.arange(len(d))
    
    # 特征提取
    vo2 = d[:, IDX["V'O2"]]
    rer = d[:, IDX['RER']]
    vco2 = rer * vo2
    hr = d[:, IDX['HR']]
    o2_pulse = d[:, IDX['SVc']] # 即 VO2/HR
    ve = d[:, IDX["V'E"]]
    vt = d[:, IDX['VTex']]
    eq_o2 = d[:, IDX['EqO2']]
    eq_co2 = d[:, IDX['EqCO2']]
    pet_o2 = d[:, IDX['PETO2']]
    pet_co2 = d[:, IDX['PETCO2']]
    load = d[:, IDX['Load']] # 新增 Load

    def props(suffix="", dash=None, force_show=False, sec_y=False, override_color=None):
        # 只有当传入的 show_legend 为 True，且是该 trace 的主线条时才显示图例
        is_visible = show_legend and force_show
        # 图例显示的名称简化为类别名，鼠标悬停显示具体病人名
        hover_text = f"{p_name} {suffix}"
        
        line_color = override_color if override_color else color
        
        return dict(
            x=None, y=None, mode='lines',
            line=dict(color=line_color, width=line_width, dash=dash),
            name=legend_group,      # 图例显示类别名
            legendgroup=legend_group, # 分组
            showlegend=is_visible,
            text=hover_text,
            hoverinfo="text+x+y"
        ), sec_y

    traces = []
    
    if panel_id == 1:
        # 修改点：新增 Load (黑色实线, 右轴)
        # VO2/VCO2 在左轴，Load 在右轴
        t1, sy1 = props("(VO2)", force_show=True, sec_y=False); t1.update(x=time, y=vo2)
        t2, sy2 = props("(VCO2)", dash='dot', sec_y=False); t2.update(x=time, y=vco2)
        # Load 使用黑色 ('#000000')，不随类别变色，设为右轴
        t3, sy3 = props("(Load)", force_show=False, sec_y=True, override_color='#000000'); t3.update(x=time, y=load)
        # 注意：Load 不参与 legendgroup 的颜色显示，但它是背景参考
        
        traces = [(t1, sy1), (t2, sy2), (t3, sy3)]
        
    elif panel_id == 2:
        # 修改点：标签从 O2Pulse 改为 VO2/HR
        t1, sy1 = props("(HR)", force_show=True, sec_y=False); t1.update(x=time, y=hr)
        t2, sy2 = props("(VO2/HR)", dash='dot', sec_y=True); t2.update(x=time, y=o2_pulse)
        traces = [(t1, sy1), (t2, sy2)]
        
    elif panel_id == 3:
        t1, sy1 = props("(HR vs VO2)", force_show=True, sec_y=False); t1.update(x=vo2, y=hr)
        t2, sy2 = props("(VCO2 vs VO2)", dash='dot', sec_y=True); t2.update(x=vo2, y=vco2)
        traces = [(t1, sy1), (t2, sy2)]
        
    elif panel_id == 4:
        t1, _ = props("(VE/VO2)", force_show=True); t1.update(x=time, y=eq_o2)
        t2, _ = props("(VE/VCO2)", dash='dot'); t2.update(x=time, y=eq_co2)
        traces = [t1, t2]
        
    elif panel_id == 5:
        t1, _ = props("(VE)", force_show=True); t1.update(x=time, y=ve)
        traces = [t1]
        
    elif panel_id == 6:
        t1, _ = props("(VE vs VCO2)", force_show=True); t1.update(x=vco2, y=ve)
        traces = [t1]
        
    elif panel_id == 7:
        t1, _ = props("(PetO2)", force_show=True); t1.update(x=time, y=pet_o2)
        t2, _ = props("(PetCO2)", dash='dot'); t2.update(x=time, y=pet_co2)
        traces = [t1, t2]
        
    elif panel_id == 8:
        t1, _ = props("(RER)", force_show=True); t1.update(x=time, y=rer)
        traces = [t1]
        
    elif panel_id == 9:
        t1, _ = props("(VT vs VE)", force_show=True); t1.update(x=ve, y=vt)
        traces = [t1]

    for item in traces:
        if isinstance(item, tuple):
            tr, sec_y = item
            fig.add_trace(go.Scatter(**tr), row=row, col=col, secondary_y=sec_y)
        else:
            fig.add_trace(go.Scatter(**item), row=row, col=col)

def update_axes_labels(fig, row, col, panel_id):
    labels = {
        1: ("Time", "VO2 / VCO2 (L/min)", "Load (Watts)"), # Panel 1 增加右轴标签
        2: ("Time", "HR (bpm)", "VO2/HR (mL/beat)"),       # Panel 2 修改标签
        3: ("VO2 (L/min)", "HR (bpm)", "VCO2 (L/min)"),
        4: ("Time", "Equivalents", None),
        5: ("Time", "VE (L/min)", None),
        6: ("VCO2 (L/min)", "VE (L/min)", None),
        7: ("Time", "mmHg", None),
        8: ("Time", "RER", None),
        9: ("VE (L/min)", "VT (L)", None)
    }
    x_tit, y_tit, sec_y_tit = labels.get(panel_id, ("", "", ""))
    fig.update_xaxes(title_text=x_tit, row=row, col=col)
    fig.update_yaxes(title_text=y_tit, row=row, col=col, secondary_y=False)
    if sec_y_tit:
        fig.update_yaxes(title_text=sec_y_tit, row=row, col=col, secondary_y=True)

# ================= 多类别绘图逻辑 =================
def plot_multi_class_focus(data_list_with_class, selected_panel_ids, class_names, class_colors, title_suffix=""):
    """
    data_list_with_class: [(filename, data, class_idx), ...]
    class_names: {class_idx: label_name}
    class_colors: {class_idx: hex_color}
    """
    num_panels = len(selected_panel_ids)
    if num_panels == 0: return None

    specs = [{"secondary_y": PANEL_CONFIG[pid]['dual_y']} for pid in selected_panel_ids]

    fig = make_subplots(
        rows=1, cols=num_panels,
        subplot_titles=[PANEL_CONFIG[pid]['name'] for pid in selected_panel_ids],
        specs=[specs], horizontal_spacing=0.08
    )

    legend_added = set()

    for idx, (p_name, d, cls_idx) in enumerate(data_list_with_class):
        base_hex = class_colors.get(cls_idx, '#7f7f7f')
        c = hex_to_rgba(base_hex, alpha=0.6)
        grp_name = class_names[cls_idx]
        need_legend = (cls_idx not in legend_added)

        for col_idx, pid in enumerate(selected_panel_ids):
            is_first_panel = (col_idx == 0)
            show = need_legend and is_first_panel
            add_panel_trace(fig, 1, col_idx+1, pid, d, p_name, c, 1.5, grp_name, show)

        if need_legend:
            legend_added.add(cls_idx)

    for col_idx, pid in enumerate(selected_panel_ids):
        update_axes_labels(fig, 1, col_idx+1, pid)

    fig.update_layout(
        height=600,
        title_text=f"多类别病理特征对比 {title_suffix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return fig

# ================= 完整九图法逻辑 =================
def plot_full_9_panel(data_list, class_colors, title_suffix=""):
    # 修改点：Panel 1 (row 1, col 1) 现在的 secondary_y 必须为 True
    specs = [
        [{"secondary_y": True},  {"secondary_y": True},  {"secondary_y": True}], # Row 1: 全双轴
        [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
    ]
    fig = make_subplots(rows=3, cols=3, subplot_titles=[PANEL_CONFIG[i]['name'] for i in range(1, 10)],
                        specs=specs, vertical_spacing=0.10, horizontal_spacing=0.08)

    colors = list(class_colors.values())
    for idx, (p_name, d) in enumerate(data_list):
        c = colors[idx % len(colors)]
        for pid in range(1, 10):
            row, col = (pid-1)//3 + 1, (pid-1)%3 + 1
            add_panel_trace(fig, row, col, pid, d, p_name, c, 2.0, p_name, (pid==1))

    for pid in range(1, 10):
        update_axes_labels(fig, (pid-1)//3 + 1, (pid-1)%3 + 1, pid)

    fig.update_layout(height=1200, width=1400, title_text=f"Wasserman 9-Panel 分析 {title_suffix}")
    return fig

# ================= 新增：单类别特征总览 =================
def plot_class_feature_overview(data_list, class_name, feature_names=None):
    """
    展示某一类多个患者的所有特征随时间变化

    Args:
        data_list: [(patient_name, data), ...] 患者数据列表
        class_name: 类别名称
        feature_names: 要展示的特征名列表，默认为22个原始特征

    Returns:
        fig: Plotly 图表对象
    """
    if feature_names is None:
        feature_names = ORIGINAL_FEATURES

    n_features = len(feature_names)
    # 计算子图布局 (尽量接近正方形)
    n_cols = min(6, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=feature_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.06
    )

    # 为每个患者生成不同颜色/透明度
    n_patients = len(data_list)
    colors = generate_class_colors(n_patients) if n_patients > 1 else ['#1f77b4']

    for p_idx, (p_name, d) in enumerate(data_list):
        time = np.arange(len(d))
        color = colors[p_idx % len(colors)]
        alpha = 0.4 + 0.4 * (p_idx / max(n_patients, 1))  # 渐变透明度

        for f_idx, feat_name in enumerate(feature_names):
            row = f_idx // n_cols + 1
            col = f_idx % n_cols + 1

            feat_idx = IDX[feat_name]
            y_data = d[:, feat_idx]

            # 只在第一个子图显示图例
            show_legend = (f_idx == 0 and p_idx == 0)

            fig.add_trace(
                go.Scatter(
                    x=time, y=y_data,
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=alpha,
                    name=p_name if show_legend else None,
                    showlegend=show_legend,
                    hovertext=f"{p_name} - {feat_name}",
                    hoverinfo="text+x+y"
                ),
                row=row, col=col
            )

    # 设置统一的 Y 轴标签
    for f_idx in range(n_features):
        row = f_idx // n_cols + 1
        col = f_idx % n_cols + 1
        fig.update_xaxes(title_text="Time", row=row, col=col, title_font=dict(size=9))
        fig.update_yaxes(title_text="Value", row=row, col=col, title_font=dict(size=9))

    fig.update_layout(
        height=200 * n_rows,
        width=1200,
        title_text=f"【{class_name}】患者特征时序总览 (共 {n_patients} 名患者, {n_features} 个特征)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# ================= 新增：单类别特征网格热力图 =================
def plot_feature_heatmap_grid(data_list, class_name, feature_names=None):
    """
    以热力图形式展示某一类多个患者的特征时序变化
    每个患者一列，时间点为行，颜色代表特征值

    Args:
        data_list: [(patient_name, data), ...]
        class_name: 类别名称
        feature_names: 要展示的特征名列表

    Returns:
        fig: Plotly 图表对象
    """
    if feature_names is None:
        # 选择关键特征展示
        feature_names = ["V'O2", 'HR', "V'E", 'RER', 'EqO2', 'EqCO2',
                        'PETO2', 'PETCO2', 'Load', 'MET', 'SVc', 'VTex']

    n_features = len(feature_names)
    n_patients = len(data_list)

    # 计算布局
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=feature_names,
        vertical_spacing=0.10,
        horizontal_spacing=0.08
    )

    for f_idx, feat_name in enumerate(feature_names):
        row = f_idx // n_cols + 1
        col = f_idx % n_cols + 1
        feat_idx = IDX[feat_name]

        # 构建热力图矩阵: [时间点 x 患者]
        # 对齐所有患者到最短长度
        min_len = min(len(d) for _, d in data_list)
        heatmap_data = np.zeros((min_len, n_patients))

        for p_idx, (_, d) in enumerate(data_list):
            heatmap_data[:, p_idx] = d[:min_len, feat_idx]

        # 标准化 (按特征)
        feat_mean = np.mean(heatmap_data)
        feat_std = np.std(heatmap_data)
        if feat_std > 0:
            heatmap_data = (heatmap_data - feat_mean) / feat_std

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                colorscale='RdBu_r',
                showscale=(f_idx == 0),
                colorbar=dict(title="Z-score") if f_idx == 0 else None,
                hovertemplate='时间: %{x}<br>患者: %{y}<br>值: %{z:.2f}<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="患者", row=row, col=col, title_font=dict(size=9))
        fig.update_yaxes(title_text="时间", row=row, col=col, title_font=dict(size=9))

    fig.update_layout(
        height=220 * n_rows,
        width=1200,
        title_text=f"【{class_name}】特征热力图网格 (Z-score标准化)"
    )

    return fig

# ================= Streamlit UI =================
st.set_page_config(layout="wide", page_title="CPET 诊断台")
st.sidebar.title("🫀 CPET 诊断台")

# 加载元数据 (自适应分类)
result = load_metadata()
if result[0] is None:
    st.error("数据加载失败，请检查数据路径和标签文件")
    st.stop()

class_files_map, label_mapping, class_names, class_colors = result
n_classes = len(label_mapping)

# 显示当前分类信息
st.sidebar.markdown(f"**当前分类体系**: {n_classes} 类")
with st.sidebar.expander("📋 分类详情"):
    for idx in sorted(class_names.keys()):
        count = len(class_files_map.get(idx, []))
        st.markdown(f"- **{idx}**: {class_names[idx]} ({count} 样本)")

view_mode = st.sidebar.radio("选择模式", [
    "多类别 - 重点图表对比 (Multi-Class Focus)",
    "单类别 - 完整九图法 (Full 9-Panel)",
    "单类别 - 特征时序总览 (Feature Overview)",  # 新增
    "单类别 - 特征热力图网格 (Heatmap Grid)",    # 新增
    "全类别 - 单特征总览 (Overview)"
])

# ----------------- 模式 1: 多类别对比 -----------------
if view_mode == "多类别 - 重点图表对比 (Multi-Class Focus)":
    st.title("多类别病理特征对比")
    st.markdown("选择多个疾病类别，在同一坐标系下对比其关键生理曲线。")

    # 动态生成选项
    class_options = {idx: f"Class {idx}: {class_names[idx]}" for idx in sorted(class_names.keys())}

    selected_classes_names = st.sidebar.multiselect(
        "选择要对比的疾病类别 (建议2-3种)",
        options=list(class_options.values()),
        default=list(class_options.values())[:2] if len(class_options) >= 2 else list(class_options.values())
    )

    # 从选中名称中提取类别索引
    selected_indices = []
    for sel_name in selected_classes_names:
        for idx, full_name in class_options.items():
            if full_name == sel_name:
                selected_indices.append(idx)
                break

    panel_options = {k: v['name'] for k, v in PANEL_CONFIG.items()}
    selected_panels = st.sidebar.multiselect(
        "选择要显示的图表 (最多3张)",
        options=list(panel_options.keys()),
        format_func=lambda x: panel_options[x],
        default=[1, 2, 6] # 默认选 Panel 1, 2, 6
    )

    if len(selected_panels) > 3: st.error("最多选择3张图表"); st.stop()

    num_patients = st.sidebar.slider("每类抽取人数", 1, 15, 5)
    if st.sidebar.button("🔄 换一批"): st.cache_data.clear()

    if selected_panels and selected_indices:
        data_bundle = []
        for c_idx in selected_indices:
            files = class_files_map.get(c_idx, [])
            if not files: continue
            sample_size = min(len(files), num_patients)
            sel_files = np.random.choice(files, sample_size, replace=False)
            for f in sel_files:
                d = load_patient_data(f)
                data_bundle.append((os.path.basename(f), d, c_idx))

        if data_bundle:
            fig = plot_multi_class_focus(data_bundle, selected_panels, class_names, class_colors)
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **提示**: Panel 1 中包含黑色实线 Load (右轴)，可用于参考运动强度。")
        else:
            st.warning("所选类别无有效样本数据")
    else:
        st.info("请在左侧选择疾病类别和图表。")

# ----------------- 模式 2: 单类别九图 -----------------
elif view_mode == "单类别 - 完整九图法 (Full 9-Panel)":
    st.title("完整 Wasserman 9-Panel 分析")

    # 动态类别选择
    class_options = {idx: f"Class {idx}: {class_names[idx]}" for idx in sorted(class_names.keys())}
    c_idx = st.sidebar.selectbox("选择类别", list(class_options.keys()),
                                  format_func=lambda x: class_options[x])
    files = class_files_map.get(c_idx, [])

    if not files:
        st.warning(f"类别 {class_names[c_idx]} 无有效样本")
        st.stop()

    if st.sidebar.button("🔄 换一批"): st.cache_data.clear()

    sel = np.random.choice(files, min(len(files), 5), replace=False)
    data = [(os.path.basename(f), load_patient_data(f)) for f in sel]

    fig = plot_full_9_panel(data, class_colors, f"- {class_names[c_idx]}")
    st.plotly_chart(fig, use_container_width=True)

# ----------------- 模式 3: 单类别特征时序总览 (新增) -----------------
elif view_mode == "单类别 - 特征时序总览 (Feature Overview)":
    st.title("单类别特征时序总览")
    st.markdown("展示某一类多个患者的所有特征随时间变化趋势。")

    # 类别选择
    class_options = {idx: f"Class {idx}: {class_names[idx]}" for idx in sorted(class_names.keys())}
    c_idx = st.sidebar.selectbox("选择类别", list(class_options.keys()),
                                  format_func=lambda x: class_options[x], key="overview_class")

    # 特征选择
    feature_select_mode = st.sidebar.radio("特征选择", ["全部22个原始特征", "自定义选择"])
    if feature_select_mode == "自定义选择":
        selected_features = st.sidebar.multiselect(
            "选择特征 (建议6-12个)",
            options=ORIGINAL_FEATURES,
            default=["V'O2", 'HR', "V'E", 'RER', 'EqO2', 'EqCO2']
        )
    else:
        selected_features = ORIGINAL_FEATURES

    # 患者数量
    files = class_files_map.get(c_idx, [])
    if not files:
        st.warning(f"类别 {class_names[c_idx]} 无有效样本")
        st.stop()

    max_patients = min(len(files), 20)
    num_patients = st.sidebar.slider("患者数量", 1, max_patients, min(8, max_patients))

    if st.sidebar.button("🔄 换一批", key="refresh_overview"): st.cache_data.clear()

    # 随机抽取患者
    sel_files = np.random.choice(files, num_patients, replace=False)
    data = [(os.path.basename(f).replace('.xlsx', ''), load_patient_data(f)) for f in sel_files]

    # 绘图
    fig = plot_class_feature_overview(data, class_names[c_idx], selected_features)
    st.plotly_chart(fig, use_container_width=True)

    # 显示患者列表
    with st.expander("📋 当前显示的患者"):
        st.markdown("\n".join([f"- {name}" for name, _ in data]))

# ----------------- 模式 4: 单类别特征热力图网格 (新增) -----------------
elif view_mode == "单类别 - 特征热力图网格 (Heatmap Grid)":
    st.title("单类别特征热力图网格")
    st.markdown("以热力图形式展示某一类多个患者的特征时序变化（Z-score标准化）。")

    # 类别选择
    class_options = {idx: f"Class {idx}: {class_names[idx]}" for idx in sorted(class_names.keys())}
    c_idx = st.sidebar.selectbox("选择类别", list(class_options.keys()),
                                  format_func=lambda x: class_options[x], key="heatmap_class")

    # 特征选择
    default_features = ["V'O2", 'HR', "V'E", 'RER', 'EqO2', 'EqCO2',
                       'PETO2', 'PETCO2', 'Load', 'MET', 'SVc', 'VTex']
    selected_features = st.sidebar.multiselect(
        "选择特征 (建议8-12个)",
        options=ORIGINAL_FEATURES,
        default=default_features
    )

    # 患者数量
    files = class_files_map.get(c_idx, [])
    if not files:
        st.warning(f"类别 {class_names[c_idx]} 无有效样本")
        st.stop()

    max_patients = min(len(files), 30)
    num_patients = st.sidebar.slider("患者数量", 2, max_patients, min(10, max_patients))

    if st.sidebar.button("🔄 换一批", key="refresh_heatmap"): st.cache_data.clear()

    # 随机抽取患者
    sel_files = np.random.choice(files, num_patients, replace=False)
    data = [(os.path.basename(f).replace('.xlsx', ''), load_patient_data(f)) for f in sel_files]

    # 绘图
    fig = plot_feature_heatmap_grid(data, class_names[c_idx], selected_features)
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **提示**: 热力图颜色表示 Z-score 标准化后的特征值。红色=高于平均，蓝色=低于平均。")

# ----------------- 模式 5: 全类别总览 -----------------
else:
    st.title(f"{n_classes}类病患特征总览")

    # 特征选择
    feat = st.sidebar.selectbox("特征", NEW_FEATURES[:22], index=10)  # 默认选 VO2
    f_idx = IDX[feat]

    max_samp = max(len(files) for files in class_files_map.values()) if class_files_map else 10
    n_samp = st.sidebar.slider("每类人数", 1, min(max_samp, 20), 10)

    if st.sidebar.button("刷新"): st.cache_data.clear()

    # 动态计算子图布局
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Class {idx}: {class_names[idx]}" for idx in range(n_classes)]
    )

    colors = list(class_colors.values())

    for i in range(n_classes):
        files = class_files_map.get(i, [])
        if not files: continue

        row = i // n_cols + 1
        col = i % n_cols + 1

        sel = np.random.choice(files, min(len(files), n_samp), replace=False)
        for f in sel:
            d = load_patient_data(f)
            fig.add_trace(
                go.Scatter(
                    y=d[:, f_idx],
                    line=dict(color=colors[i], width=1),
                    showlegend=False,
                    opacity=0.5
                ),
                row=row, col=col
            )

    fig.update_layout(height=200 * n_rows, title_text=f"{feat} 特征总览")
    st.plotly_chart(fig, use_container_width=True)
