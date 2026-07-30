"""
CPET 30个动态生理特征的严谨分组统计分析脚本

核心特性：
1. 运动期切片 (Exercise Period Slicing) - 只提取运动期间数据
2. 异常值屏蔽 - 将 0.0 替换为 np.nan
3. 三种混合降维策略：
   - Peak 组（运动期最大值）：容量、负荷、绝对上限指标
   - Min 组（运动期最小值）：谷值、脱饱和、异常平台期下探指标
   - Mean 组（运动期均值）：全局效率斜率指标
4. 按疾病分组统计 Mean ± SD
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_preprocess_new import get_data_variable_length, find_exercise_periods
from label_extractor import load_labels


# =============================================================================
# 配置
# =============================================================================

# 数据路径
DATA_DIR = r'xx_path'
LABEL_FILE = r"xx_path"
CACHE_DIR = os.path.join(DATA_DIR, 'npy_cache')

# 输出路径
OUTPUT_DIR = r"xx_path"

# EHR / demographics 配置
GROUP_COL_NAME = "匹配的第一大类"
EHR_CONTINUOUS_SPECS = [
    ("Age (years)", ["年龄", "Age", "age"]),
    ("Weight (kg)", ["体重（kg）", "体重(kg)", "体重", "Weight (kg)", "Weight", "weight"]),
    ("Height (cm)", ["身高（cm）", "身高(cm)", "身高", "Height (cm)", "Height", "height"]),
    ("BMI (kg/m^2)", ["BMI", "bmi", "BMI（kg/m^2）", "BMI(kg/m^2)", "BMI (kg/m^2)", "BMI（kg/㎡）", "BMI(kg/㎡)"]),
]
EHR_SEX_SPEC = ("Sex (Male, n [%])", ["性别", "Sex", "sex", "Gender", "gender"])

# t1-t5 功能标签分布配置
FUNCTIONAL_TASK_SPECS = [
    ("t1 运动心功能分级", "运动心功能分级"),
    ("t2 运动耐量", "运动耐量"),
    ("t3 标准心电运动负荷试验", "标准心电运动负荷试验"),
    ("t4 运动中换气肺功能", "运动中换气肺功能"),
    ("t5 心率储备", "心率储备"),
]

# 如果实际 Excel 中出现额外标签，会自动追加到预设顺序之后。
FUNCTIONAL_LABEL_ORDERS = {
    "运动心功能分级": ["A", "B", "C", "D"],
    "运动耐量": ["中度下降", "正常/大致正常", "轻度下降", "重度/极重度下降"],
    "标准心电运动负荷试验": ["阳", "阴"],
    "运动中换气肺功能": ["下降", "正常"],
    "心率储备": ["未用尽", "用尽"],
}


# =============================================================================
# 特征定义（用户指定的输出顺序）
# =============================================================================

FEATURE_NAMES = [
    "Load", "V'O2", "HR", "O2Pulse", "dO2/dW", "MET", "d(O2P)/dt", "VO2/kg", "dH/dO2", "SVc", "OUES",
    "V'CO2", "V'E", "VTex", "BF", "RER", "BR", "EqO2", "EqCO2", "PETO2", "PETCO2", "SpO2", "EqO2_COP",
    "VDc/VT", "Psys", "Pdia", "PP", "HRR", "HR_diff", "d2(O2P)/dt2"
]

# 1. 求 Peak (np.nanmax) 的索引集合 (容量、负荷、绝对上限指标)
PEAK_INDICES = [0, 1, 2, 3, 5, 7, 9, 11, 12, 13, 14, 15, 19, 20, 24, 25, 26, 27]

# 2. 求 Min (np.nanmin) 的索引集合 (谷值、脱饱和、异常平台期下探指标)
MIN_INDICES = [6, 16, 17, 18, 21, 22, 28, 29]

# 3. 求 Mean (np.nanmean) 的索引集合 (全局效率斜率指标)
MEAN_INDICES = [4, 8, 10, 23]

# =============================================================================
# 数据索引映射
# get_data_variable_length 返回的特征顺序：
# section1 (0-11): MET, Load, RER, HR, HRR, dH/dO2, SVc, Psys, Pdia, SpO2, V'O2, VO2/kg
# section2 (12-21): dO2/dW, BF, V'E, BR, EqO2, EqCO2, PETO2, PETCO2, VDc/VT, VTex
# derived (22-25): PP, OUES, EqO2_COP, HR_diff
# o2pulse (26-28): O2Pulse, d(O2P)/dt, d2(O2P)/dt2
# vco2 (29): V'CO2
# =============================================================================

# 输出索引 -> 数据索引映射
OUTPUT_TO_DATA_INDEX = {
    0: 1,   # Load -> 数据索引 1
    1: 10,  # V'O2 -> 数据索引 10
    2: 3,   # HR -> 数据索引 3
    3: 26,  # O2Pulse -> 数据索引 26
    4: 12,  # dO2/dW -> 数据索引 12
    5: 0,   # MET -> 数据索引 0
    6: 27,  # d(O2P)/dt -> 数据索引 27
    7: 11,  # VO2/kg -> 数据索引 11
    8: 5,   # dH/dO2 -> 数据索引 5
    9: 6,   # SVc -> 数据索引 6
    10: 23, # OUES -> 数据索引 23
    11: 29, # V'CO2 -> 数据索引 29
    12: 14, # V'E -> 数据索引 14
    13: 21, # VTex -> 数据索引 21
    14: 13, # BF -> 数据索引 13
    15: 2,  # RER -> 数据索引 2
    16: 15, # BR -> 数据索引 15
    17: 16, # EqO2 -> 数据索引 16
    18: 17, # EqCO2 -> 数据索引 17
    19: 18, # PETO2 -> 数据索引 18
    20: 19, # PETCO2 -> 数据索引 19
    21: 9,  # SpO2 -> 数据索引 9
    22: 24, # EqO2_COP -> 数据索引 24
    23: 20, # VDc/VT -> 数据索引 20
    24: 7,  # Psys -> 数据索引 7
    25: 8,  # Pdia -> 数据索引 8
    26: 22, # PP -> 数据索引 22
    27: 4,  # HRR -> 数据索引 4
    28: 25, # HR_diff -> 数据索引 25
    29: 28, # d2(O2P)/dt2 -> 数据索引 28
}

# Load 在数据中的索引
LOAD_DATA_INDEX = 1


def mask_invalid_values(data):
    """
    将 0.0 替换为 np.nan，屏蔽无效值
    """
    data = data.copy()
    data[data == 0.0] = np.nan
    return data


def extract_exercise_period(data):
    """
    提取运动期数据

    Args:
        data: [T, 30] numpy 数组

    Returns:
        exercise_data: 运动期数据切片，如果无运动期则返回 None
    """
    # 使用 Load 列（数据索引 1）识别运动期
    load_col = data[:, LOAD_DATA_INDEX]

    # 调用 find_exercise_periods
    periods = find_exercise_periods(load_col)

    start = periods['exercise_start']
    end = periods['exercise_end']

    if start >= end:
        return None

    return data[start:end, :]


def compute_exercise_statistics(data):
    """
    计算运动期统计量（三种混合降维策略）

    Args:
        data: [T, 30] 原始数据

    Returns:
        stats: [30] 每个特征的统计值（Peak / Min / Mean）
    """
    # 1. 屏蔽无效值
    data = mask_invalid_values(data)

    # 2. 提取运动期
    exercise_data = extract_exercise_period(data)

    if exercise_data is None or len(exercise_data) == 0:
        return np.full(30, np.nan)

    # 3. 计算统计量
    stats = np.zeros(30)

    for out_idx in range(30):
        data_idx = OUTPUT_TO_DATA_INDEX[out_idx]

        # 提取该特征列
        col = exercise_data[:, data_idx]

        # 判断统计类型
        if out_idx in PEAK_INDICES:
            # Peak: 运动期最大值
            stats[out_idx] = np.nanmax(col)
        elif out_idx in MIN_INDICES:
            # Min: 运动期最小值
            stats[out_idx] = np.nanmin(col)
        else:
            # Mean: 运动期均值
            stats[out_idx] = np.nanmean(col)

    return stats


def load_patient_data(filename, cache_dir=None):
    """
    加载单个患者的 CPET 数据
    """
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None

    try:
        data = get_data_variable_length(
            filepath,
            cache_dir=cache_dir,
            adapt_mode='nine_graph',
            o2pulse_enabled=True,
            vco2_enabled=True
        )
        return data
    except Exception as e:
        print(f"加载文件 {filename} 时出错: {e}")
        return None


def get_statistic_type(out_idx):
    """获取统计类型"""
    if out_idx in PEAK_INDICES:
        return "Peak"
    elif out_idx in MIN_INDICES:
        return "Min"
    else:
        return "Mean"


def compute_group_statistics(patient_stats, group_name):
    """
    计算一组患者的统计量
    """
    stats_list = []

    for out_idx in range(30):
        col_data = patient_stats[:, out_idx]

        # 过滤 NaN
        valid_values = col_data[~np.isnan(col_data)]

        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            n_valid = len(valid_values)
        else:
            mean_val = np.nan
            std_val = np.nan
            n_valid = 0

        # 确定统计类型
        stat_type = get_statistic_type(out_idx)
        feat_name = FEATURE_NAMES[out_idx]

        stats_list.append({
            'Feature': f"{stat_type} {feat_name}",
            'Mean': mean_val,
            'SD': std_val,
            'N': n_valid,
            'Mean ± SD': f"{mean_val:.2f} ± {std_val:.2f}" if not np.isnan(mean_val) else "N/A"
        })

    return pd.DataFrame(stats_list)


def format_p_value(p):
    """
    论文表格中的 P value 格式化。
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fdr_bh(p_values):
    """
    Benjamini-Hochberg FDR correction.
    不依赖 statsmodels，仅对传入的有效 P values 进行校正。
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values)
    ranked_p = p_values[order]

    adjusted = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked_p[i] * n / rank
        value = min(value, prev)
        adjusted[i] = min(value, 1.0)
        prev = value

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def epsilon_squared_kruskal(H, n, k):
    """
    Kruskal-Wallis 的效应量 epsilon-squared。
    适用于多个独立组之间的连续变量/非正态变量比较。

    H: Kruskal-Wallis H statistic
    n: 总有效样本量
    k: 有效分组数
    """
    if pd.isna(H) or n <= k:
        return np.nan

    eps = (H - k + 1) / (n - k)
    return max(0.0, min(1.0, eps))


def compute_feature_group_tests(category_stats):
    """
    基于 t6 疾病类别分组，对 30 个 CPET features 进行组间比较。

    当前表格是按“匹配的第一大类”进行 t6 分组，因此这里使用：
    - P value: Kruskal-Wallis test
    - FDR-adjusted P value: Benjamini-Hochberg correction across 30 CPET features
    - Effect size: epsilon-squared

    Returns:
        tests_df: 每个 feature 一行的统计检验结果
    """
    rows = []
    labels = sorted(category_stats.keys())

    for out_idx in range(30):
        stat_type = get_statistic_type(out_idx)
        feat_name = FEATURE_NAMES[out_idx]
        feature_label = f"{stat_type} {feat_name}"

        groups = []
        n_by_label = {}

        for label in labels:
            stats_array = np.asarray(category_stats[label])

            if stats_array.size == 0:
                valid_values = np.array([], dtype=float)
            else:
                col_data = stats_array[:, out_idx]
                valid_values = col_data[~np.isnan(col_data)].astype(float)

            n_by_label[label] = len(valid_values)

            if len(valid_values) > 0:
                groups.append(valid_values)

        k = len(groups)
        n_total = sum(len(g) for g in groups)

        if k < 2:
            H = np.nan
            p_value = np.nan
            effect_size = np.nan
        else:
            try:
                H, p_value = stats.kruskal(*groups)
                effect_size = epsilon_squared_kruskal(H, n=n_total, k=k)
            except ValueError as e:
                # scipy 在所有数值完全相同时可能报错。
                # 这种情况下可视为组间无可检测差异：H=0, P=1, effect_size=0。
                if "All numbers are identical" in str(e):
                    H = 0.0
                    p_value = 1.0
                    effect_size = 0.0
                else:
                    H = np.nan
                    p_value = np.nan
                    effect_size = np.nan

        row = {
            'Feature': feature_label,
            'Test': 'Kruskal-Wallis',
            'Statistic': H,
            'P value_raw': p_value,
            'FDR-adjusted P value_raw': np.nan,
            'P value': format_p_value(p_value),
            'FDR-adjusted P value': '',
            'Effect size': effect_size,
            'Effect size type': 'epsilon-squared',
            'N_total_valid': n_total,
            'N_groups_valid': k
        }

        # 额外记录每个 t6 类别的有效样本量，方便排查缺失值。
        for label in labels:
            safe_label = str(label).replace('/', '_').replace('\\', '_')[:40]
            row[f'N_{safe_label}'] = n_by_label[label]

        rows.append(row)

    tests_df = pd.DataFrame(rows)

    valid_mask = tests_df['P value_raw'].notna()
    if valid_mask.any():
        adjusted = fdr_bh(tests_df.loc[valid_mask, 'P value_raw'].values)
        tests_df.loc[valid_mask, 'FDR-adjusted P value_raw'] = adjusted
        tests_df.loc[valid_mask, 'FDR-adjusted P value'] = [
            format_p_value(p) for p in adjusted
        ]

    return tests_df


def normalize_column_name(name):
    """
    统一列名格式，用于兼容中英文括号、空格和大小写。
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    replacements = {
        " ": "",
        "\u3000": "",
        "（": "(",
        "）": ")",
        "㎡": "m^2",
        "²": "^2",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def find_column(df, aliases, required=False):
    """
    根据候选别名在 df.columns 中查找真实列名。
    """
    norm_to_col = {normalize_column_name(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in norm_to_col:
            return norm_to_col[key]

    # 兜底：允许包含式匹配，例如 "体重（kg）" 与 "体重"。
    alias_keys = [normalize_column_name(a) for a in aliases]
    for col in df.columns:
        col_key = normalize_column_name(col)
        for key in alias_keys:
            if key and (key in col_key or col_key in key):
                return col

    if required:
        raise KeyError(f"无法在 Excel 中找到列: {aliases}; 当前 df.columns={df.columns.tolist()}")
    return None


def read_label_dataframe(label_file, group_col=GROUP_COL_NAME, max_header_rows=10):
    """
    读取 filtered_data_1149.xlsx，并自动尝试识别 header 行。
    如果你的 Excel 第一行就是列名，则 header_row=0 会直接命中；
    如果前面有说明行，也会在前 10 行内自动查找包含 group_col 的表头。
    """
    last_df = None
    for header_row in range(max_header_rows):
        try:
            df = pd.read_excel(label_file, engine='openpyxl', header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
            last_df = df
            if find_column(df, [group_col], required=False) is not None:
                print(f"EHR 表头识别成功: header_row={header_row}, Excel 行号={header_row + 1}")
                return df
        except Exception:
            continue

    if last_df is None:
        raise ValueError(f"无法读取 EHR Excel: {label_file}")
    raise KeyError(f"无法识别分组列 {group_col}; 当前最后一次 df.columns={last_df.columns.tolist()}")


def format_mean_sd(values):
    """
    连续变量 mean ± SD 格式化。与 CPET 部分保持一致，使用 np.std 默认 ddof=0。
    """
    values = pd.to_numeric(pd.Series(values), errors='coerce').dropna().astype(float)
    if len(values) == 0:
        return "N/A"
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def normalize_sex_value(value):
    """
    将性别字段统一为 Male / Female。
    当前兼容中文“男/女”和常见英文写法。
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"男", "男性", "male", "m", "man", "1", "1.0"}:
        return "Male"
    if s in {"女", "女性", "female", "f", "woman", "0", "0.0"}:
        return "Female"
    return np.nan


def format_male_n_percent(values):
    """
    分类变量 Sex 的显示格式：male n (%)。
    """
    sex = pd.Series(values).apply(normalize_sex_value).dropna()
    if len(sex) == 0:
        return "N/A"
    n_male = int((sex == "Male").sum())
    pct = n_male / len(sex) * 100
    return f"{n_male} ({pct:.1f}%)"


def cramers_v_from_table(contingency_table):
    """
    Cramér's V，用于分类变量组间关联效应量。
    """
    contingency_table = np.asarray(contingency_table)
    if contingency_table.size == 0:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    r, c = contingency_table.shape
    if n == 0 or min(r - 1, c - 1) == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min(r - 1, c - 1)))


def compute_continuous_ehr_test(df, value_col, group_col, labels):
    """
    EHR 连续变量：Kruskal-Wallis + epsilon-squared。
    """
    groups = []
    n_by_label = {}
    for label in labels:
        vals = pd.to_numeric(df.loc[df[group_col] == label, value_col], errors='coerce').dropna().astype(float).values
        n_by_label[label] = len(vals)
        if len(vals) > 0:
            groups.append(vals)

    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if k < 2:
        return np.nan, np.nan, np.nan, n_total, k, n_by_label

    try:
        H, p_value = stats.kruskal(*groups)
        effect_size = epsilon_squared_kruskal(H, n=n_total, k=k)
    except ValueError as e:
        if "All numbers are identical" in str(e):
            H, p_value, effect_size = 0.0, 1.0, 0.0
        else:
            H, p_value, effect_size = np.nan, np.nan, np.nan
    return H, p_value, effect_size, n_total, k, n_by_label


def compute_categorical_ehr_test(df, category_col, group_col, labels):
    """
    EHR 分类变量：Chi-square + Cramér's V。
    """
    sub = df[[group_col, category_col]].copy()
    sub[category_col] = sub[category_col].apply(normalize_sex_value)
    sub = sub.dropna(subset=[group_col, category_col])
    sub = sub[sub[group_col].isin(labels)]

    table = pd.crosstab(sub[group_col], sub[category_col])
    # 保证所有 label 都出现在表中，即便某组缺失也补 0。
    table = table.reindex(index=labels, fill_value=0)

    n_by_label = table.sum(axis=1).to_dict()
    n_total = int(table.values.sum())
    k = int((table.sum(axis=1) > 0).sum())

    if table.shape[0] < 2 or table.shape[1] < 2 or n_total == 0:
        return np.nan, np.nan, np.nan, n_total, k, n_by_label

    try:
        chi2, p_value, _, _ = stats.chi2_contingency(table.values, correction=False)
        effect_size = cramers_v_from_table(table.values)
    except ValueError:
        chi2, p_value, effect_size = np.nan, np.nan, np.nan
    return chi2, p_value, effect_size, n_total, k, n_by_label


def compute_ehr_summary(label_file, category_labels, group_col=GROUP_COL_NAME):
    """
    生成 EHR 统计表，格式对齐：
    Variable | Overall (N=1149) | IHD (n=...) | HTx (n=...) | ... | P value | FDR-adjusted P value | Effect size

    连续变量：Age / Weight / Height / BMI，显示 mean ± SD，检验用 Kruskal-Wallis，效应量 epsilon-squared。
    分类变量：Sex，显示 Male n (%)，检验用 Chi-square，效应量 Cramér's V。
    """
    df = read_label_dataframe(label_file, group_col=group_col)
    group_col_real = find_column(df, [group_col], required=True)
    df[group_col_real] = df[group_col_real].astype(str).str.strip()

    labels = [label for label in category_labels if label in set(df[group_col_real].dropna().astype(str).str.strip())]
    if not labels:
        labels = sorted(df[group_col_real].dropna().astype(str).unique().tolist())

    df = df[df[group_col_real].isin(labels)].copy()

    overall_header = f"Overall (N={len(df)})"
    group_headers = {label: f"{label} (n={int((df[group_col_real] == label).sum())})" for label in labels}

    summary_rows = []
    test_rows = []

    # 连续变量
    for display_name, aliases in EHR_CONTINUOUS_SPECS:
        col = find_column(df, aliases, required=False)
        row = {"Variable": display_name}
        test_row = {
            "Variable": display_name,
            "Variable type": "continuous",
            "Source column": col if col is not None else "",
            "Test": "Kruskal-Wallis",
            "Statistic": np.nan,
            "P value_raw": np.nan,
            "FDR-adjusted P value_raw": np.nan,
            "P value": "",
            "FDR-adjusted P value": "",
            "Effect size": np.nan,
            "Effect size type": "epsilon-squared",
            "N_total_valid": 0,
            "N_groups_valid": 0,
        }

        if col is None:
            row[overall_header] = "N/A"
            for label in labels:
                row[group_headers[label]] = "N/A"
        else:
            row[overall_header] = format_mean_sd(df[col])
            for label in labels:
                row[group_headers[label]] = format_mean_sd(df.loc[df[group_col_real] == label, col])

            stat, p_value, effect_size, n_total, k, n_by_label = compute_continuous_ehr_test(
                df=df,
                value_col=col,
                group_col=group_col_real,
                labels=labels
            )
            test_row.update({
                "Statistic": stat,
                "P value_raw": p_value,
                "P value": format_p_value(p_value),
                "Effect size": effect_size,
                "N_total_valid": n_total,
                "N_groups_valid": k,
            })
            for label in labels:
                test_row[f"N_{label}"] = n_by_label.get(label, 0)

        summary_rows.append(row)
        test_rows.append(test_row)

    # 分类变量：Sex
    sex_display_name, sex_aliases = EHR_SEX_SPEC
    sex_col = find_column(df, sex_aliases, required=False)
    row = {"Variable": sex_display_name}
    test_row = {
        "Variable": sex_display_name,
        "Variable type": "categorical",
        "Source column": sex_col if sex_col is not None else "",
        "Test": "Chi-square",
        "Statistic": np.nan,
        "P value_raw": np.nan,
        "FDR-adjusted P value_raw": np.nan,
        "P value": "",
        "FDR-adjusted P value": "",
        "Effect size": np.nan,
        "Effect size type": "Cramer's V",
        "N_total_valid": 0,
        "N_groups_valid": 0,
    }

    if sex_col is None:
        row[overall_header] = "N/A"
        for label in labels:
            row[group_headers[label]] = "N/A"
    else:
        row[overall_header] = format_male_n_percent(df[sex_col])
        for label in labels:
            row[group_headers[label]] = format_male_n_percent(df.loc[df[group_col_real] == label, sex_col])

        stat, p_value, effect_size, n_total, k, n_by_label = compute_categorical_ehr_test(
            df=df,
            category_col=sex_col,
            group_col=group_col_real,
            labels=labels
        )
        test_row.update({
            "Statistic": stat,
            "P value_raw": p_value,
            "P value": format_p_value(p_value),
            "Effect size": effect_size,
            "N_total_valid": n_total,
            "N_groups_valid": k,
        })
        for label in labels:
            test_row[f"N_{label}"] = n_by_label.get(label, 0)

    summary_rows.append(row)
    test_rows.append(test_row)

    ehr_summary_df = pd.DataFrame(summary_rows)
    ehr_tests_df = pd.DataFrame(test_rows)

    # EHR block 内部进行 FDR 校正。
    valid_mask = ehr_tests_df["P value_raw"].notna()
    if valid_mask.any():
        adjusted = fdr_bh(ehr_tests_df.loc[valid_mask, "P value_raw"].values)
        ehr_tests_df.loc[valid_mask, "FDR-adjusted P value_raw"] = adjusted
        ehr_tests_df.loc[valid_mask, "FDR-adjusted P value"] = [format_p_value(p) for p in adjusted]

    # 将检验结果回填到 summary。
    ehr_summary_df["P value"] = ehr_tests_df["P value"].tolist()
    ehr_summary_df["FDR-adjusted P value"] = ehr_tests_df["FDR-adjusted P value"].tolist()
    ehr_summary_df["Effect size"] = ehr_tests_df["Effect size"].apply(
        lambda x: f"{x:.3f}" if not pd.isna(x) else ""
    ).tolist()
    ehr_summary_df["Effect size type"] = ehr_tests_df["Effect size type"].tolist()
    ehr_summary_df["Test"] = ehr_tests_df["Test"].tolist()

    return ehr_summary_df, ehr_tests_df


def add_section_rows_for_table1(ehr_summary_df, cpet_summary_df):
    """
    额外生成一个可直接整理成论文 Table 1 的合并表。
    注意：EHR 与 CPET 的 FDR 分别在各自 block 内校正，不在此处重新混合校正。
    """
    ehr = ehr_summary_df.copy()
    ehr.insert(0, "Section", "Demographics")

    cpet = cpet_summary_df.copy()
    if "Feature" in cpet.columns:
        cpet = cpet.rename(columns={"Feature": "Variable"})

    # 将 CPET 的 Total / IHD / HTx ... 对齐到 EHR 的 Overall (N=...) / IHD (n=...) 表头。
    rename_map = {}
    overall_cols = [c for c in ehr.columns if str(c).startswith("Overall (N=")]
    if overall_cols and "Total" in cpet.columns:
        rename_map["Total"] = overall_cols[0]

    ehr_group_headers = {
        str(c).split(" (n=")[0]: c
        for c in ehr.columns
        if " (n=" in str(c)
    }
    for c in cpet.columns:
        c_str = str(c)
        if c_str in ehr_group_headers:
            rename_map[c] = ehr_group_headers[c_str]
        else:
            # 兼容 CPET Summary 中 label[:20] 的截断列名。
            for label_prefix, ehr_header in ehr_group_headers.items():
                if label_prefix.startswith(c_str) or c_str.startswith(label_prefix[:20]):
                    rename_map[c] = ehr_header
                    break
    cpet = cpet.rename(columns=rename_map)

    cpet.insert(0, "Section", cpet["Variable"].apply(lambda x: str(x).split(" ", 1)[0] + " Exercise(RAW)"))

    # 让列尽量按 EHR 表和 CPET 表共同字段排序。
    preferred = [
        "Section", "Variable",
        *[c for c in ehr.columns if str(c).startswith("Overall (N=")],
        *[c for c in ehr.columns if " (n=" in str(c)],
        "P value", "FDR-adjusted P value", "Effect size", "Effect size type", "Test"
    ]
    common_cols = []
    for col in preferred + list(ehr.columns) + list(cpet.columns):
        if col not in common_cols:
            common_cols.append(col)
    return pd.concat([ehr.reindex(columns=common_cols), cpet.reindex(columns=common_cols)], ignore_index=True)



# =============================================================================
# t1-t5 功能标签分布与关联统计模块
# =============================================================================

def normalize_functional_label_value(value):
    """
    将 t1-t5 标签标准化为可展示字符串。
    - 保留中文标签和 A/B/C/D；
    - 将 1.0 这类 Excel 数值标签转成 1；
    - 空字符串 / NaN 视为缺失。
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        v = float(s)
        if v.is_integer():
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return s


def unique_nonmissing_keep_order(values):
    """去重且保留首次出现顺序。"""
    seen = set()
    out = []
    for value in values:
        value = normalize_functional_label_value(value)
        if pd.isna(value):
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def get_functional_label_order(series, source_col):
    """
    优先使用预设临床标签顺序；若存在预设外标签，则按数据中首次出现顺序追加。
    """
    observed = unique_nonmissing_keep_order(series)
    preset = FUNCTIONAL_LABEL_ORDERS.get(source_col, [])
    ordered = [x for x in preset if x in observed]
    ordered.extend([x for x in observed if x not in ordered])
    return ordered


def format_count_percent(count, denominator):
    """格式化 n (%)。"""
    count = int(count)
    denominator = int(denominator) if denominator is not None else 0
    if denominator <= 0:
        return f"{count} (0.0%)"
    return f"{count} ({count / denominator * 100:.1f}%)"


def mutual_information_bits_from_table(contingency_table):
    """
    根据列联表计算 mutual information，单位 bits。
    行为组别，列为 t1-t5 标签类别。
    """
    table = np.asarray(contingency_table, dtype=float)
    total = table.sum()
    if total <= 0:
        return np.nan

    pxy = table / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py

    mask = (pxy > 0) & (expected > 0)
    return float(np.sum(pxy[mask] * np.log2(pxy[mask] / expected[mask])))


def association_strength_from_cramers_v(v):
    """
    基于 Cramér's V 给出描述性关联强度。
    阈值用于表格解读提示，不作为严格统计结论。
    """
    if pd.isna(v):
        return ""
    if v < 0.10:
        return "weak"
    if v < 0.30:
        return "moderate"
    if v < 0.50:
        return "strong"
    return "very strong"


def compute_functional_label_tables_from_df(
    df,
    group_col,
    group_labels,
    task_specs=FUNCTIONAL_TASK_SPECS,
    overall_name="Overall",
):
    """
    基于输入 dataframe 统计 t1-t5 在指定分组变量下的标签分布和关联统计。

    输出：
    1. distribution_df:
       Task | Label | Overall (N=...) | Group1 (n=...) | Group2 (n=...) | ...
    2. association_df:
       Task | Chi-square | df | P value | FDR-adjusted P value | Cramér's V | Mutual information (bits) | Association strength
    """
    df = df.copy()
    group_col_real = find_column(df, [group_col], required=True) if group_col != "__Set__" else group_col
    if group_col_real not in df.columns:
        raise KeyError(f"无法找到分组列: {group_col}; 当前 df.columns={df.columns.tolist()}")

    df[group_col_real] = df[group_col_real].astype(str).str.strip()
    group_labels = [str(g).strip() for g in group_labels]
    group_labels = [g for g in group_labels if g in set(df[group_col_real].dropna().astype(str).str.strip())]
    if not group_labels:
        group_labels = sorted(df[group_col_real].dropna().astype(str).unique().tolist())

    df = df[df[group_col_real].isin(group_labels)].copy()
    overall_header = f"{overall_name} (N={len(df)})"
    group_headers = {g: f"{g} (n={int((df[group_col_real] == g).sum())})" for g in group_labels}

    distribution_rows = []
    association_rows = []

    for task_display, source_col_name in task_specs:
        source_col = find_column(df, [source_col_name], required=False)
        if source_col is None:
            distribution_rows.append({
                "Task": task_display,
                "Label": f"缺失列: {source_col_name}",
                overall_header: "N/A",
                **{group_headers[g]: "N/A" for g in group_labels}
            })
            association_rows.append({
                "Task": task_display,
                "Source column": "",
                "Chi-square": np.nan,
                "df": np.nan,
                "P value_raw": np.nan,
                "FDR-adjusted P value_raw": np.nan,
                "P value": "",
                "FDR-adjusted P value": "",
                "Cramér's V": np.nan,
                "Mutual information (bits)": np.nan,
                "Association strength": "",
            })
            continue

        work = df[[group_col_real, source_col]].copy()
        work[source_col] = work[source_col].apply(normalize_functional_label_value)
        work = work.dropna(subset=[group_col_real, source_col])
        labels = get_functional_label_order(work[source_col], source_col_name)

        overall_denom = len(work)
        group_denoms = {g: int((work[group_col_real] == g).sum()) for g in group_labels}

        if not labels:
            distribution_rows.append({
                "Task": task_display,
                "Label": "N/A",
                overall_header: "N/A",
                **{group_headers[g]: "N/A" for g in group_labels}
            })
        else:
            for label in labels:
                row = {
                    "Task": task_display,
                    "Label": label,
                    overall_header: format_count_percent((work[source_col] == label).sum(), overall_denom),
                }
                for g in group_labels:
                    count = ((work[group_col_real] == g) & (work[source_col] == label)).sum()
                    row[group_headers[g]] = format_count_percent(count, group_denoms[g])
                distribution_rows.append(row)

        if len(labels) < 2 or len(group_labels) < 2:
            chi2 = np.nan
            p_value = np.nan
            dof = np.nan
            cramers_v = np.nan
            mi_bits = np.nan
        else:
            table = pd.crosstab(work[group_col_real], work[source_col])
            table = table.reindex(index=group_labels, columns=labels, fill_value=0)
            if table.values.sum() == 0 or table.shape[0] < 2 or table.shape[1] < 2:
                chi2 = np.nan
                p_value = np.nan
                dof = np.nan
                cramers_v = np.nan
                mi_bits = np.nan
            else:
                try:
                    chi2, p_value, dof, _ = stats.chi2_contingency(table.values, correction=False)
                    cramers_v = cramers_v_from_table(table.values)
                    mi_bits = mutual_information_bits_from_table(table.values)
                except ValueError:
                    chi2 = np.nan
                    p_value = np.nan
                    dof = np.nan
                    cramers_v = np.nan
                    mi_bits = np.nan

        association_rows.append({
            "Task": task_display,
            "Source column": source_col,
            "Chi-square": chi2,
            "df": dof,
            "P value_raw": p_value,
            "FDR-adjusted P value_raw": np.nan,
            "P value": format_p_value(p_value),
            "FDR-adjusted P value": "",
            "Cramér's V": cramers_v,
            "Mutual information (bits)": mi_bits,
            "Association strength": association_strength_from_cramers_v(cramers_v),
        })

    distribution_df = pd.DataFrame(distribution_rows)
    association_df = pd.DataFrame(association_rows)

    valid_mask = association_df["P value_raw"].notna()
    if valid_mask.any():
        adjusted = fdr_bh(association_df.loc[valid_mask, "P value_raw"].values)
        association_df.loc[valid_mask, "FDR-adjusted P value_raw"] = adjusted
        association_df.loc[valid_mask, "FDR-adjusted P value"] = [format_p_value(p) for p in adjusted]

    # 展示列保留格式化后的统计值；raw 列用于核查。
    display_assoc = association_df.copy()
    display_assoc["Chi-square"] = display_assoc["Chi-square"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "")
    display_assoc["df"] = display_assoc["df"].apply(lambda x: int(x) if not pd.isna(x) else "")
    display_assoc["Cramér's V"] = display_assoc["Cramér's V"].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "")
    display_assoc["Mutual information (bits)"] = display_assoc["Mutual information (bits)"].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "")

    return distribution_df, display_assoc, association_df


def combine_functional_label_distribution_and_association(distribution_df, association_display_df):
    """
    生成一个类似论文 Supplementary Table 的合并草稿：Section A 分布 + Section B 关联统计。
    """
    section_a_title = pd.DataFrame([{
        "Task": "Section A. Complete functional label distributions across groups"
    }])
    blank = pd.DataFrame([{}])
    section_b_title = pd.DataFrame([{
        "Task": "Section B. Association statistics between groups and t1–t5 labels"
    }])
    return pd.concat(
        [section_a_title, distribution_df, blank, section_b_title, association_display_df],
        ignore_index=True,
        sort=False
    )

def main():
    """主函数"""
    print("=" * 70)
    print("CPET 30个动态生理特征分组统计分析")
    print("(运动期切片 + 三种混合降维策略)")
    print("=" * 70)

    # 打印分组信息
    print("\n【降维策略】")
    print("Peak 组 (运动期最大值) - 容量、负荷、绝对上限指标:")
    peak_names = [FEATURE_NAMES[i] for i in PEAK_INDICES]
    print(f"  {', '.join(peak_names)}")

    print("\nMin 组 (运动期最小值) - 谷值、脱饱和、异常平台期下探指标:")
    min_names = [FEATURE_NAMES[i] for i in MIN_INDICES]
    print(f"  {', '.join(min_names)}")

    print("\nMean 组 (运动期均值) - 全局效率斜率指标:")
    mean_names = [FEATURE_NAMES[i] for i in MEAN_INDICES]
    print(f"  {', '.join(mean_names)}")

    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 加载标签
    print("\n[1/4] 加载标签数据...")
    label_dict, label_mapping = load_labels(
        label_file=LABEL_FILE,
        target_col_name=GROUP_COL_NAME
    )

    if not label_dict:
        print("错误: 无法加载标签数据")
        return

    print(f"成功加载 {len(label_dict)} 条标签记录")
    print(f"类别数: {len(label_mapping)}")
    for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = sum(1 for l in label_dict.values() if l == label)
        print(f"  {idx}: {label} ({count} 样本)")

    category_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

    # 3. 按类别分组
    print("\n[2/4] 按类别分组...")
    category_patients = defaultdict(list)
    for filename, label in label_dict.items():
        if not filename.endswith('.xlsx'):
            filename = filename + '.xlsx'
        category_patients[label].append(filename)

    # 4. 加载数据并计算统计量
    print("\n[3/4] 加载数据并计算运动期统计量...")

    # 存储每个类别的统计值
    category_stats = defaultdict(list)
    all_stats = []

    # 使用进度显示
    all_files = [(f, label) for label, files in category_patients.items() for f in files]
    total_files = len(all_files)

    for idx, (filename, label) in enumerate(all_files):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  处理进度: {idx + 1}/{total_files}")

        data = load_patient_data(filename, cache_dir=CACHE_DIR)
        if data is not None and data.shape[0] > 0 and data.shape[1] == 30:
            stats = compute_exercise_statistics(data)
            category_stats[label].append(stats)
            all_stats.append(stats)

    # 转换为 numpy 数组
    for label in category_stats:
        category_stats[label] = np.array(category_stats[label])

    all_stats = np.array(all_stats)

    print(f"\n成功处理 {len(all_stats)} 个患者")

    # 5. 计算统计量并输出到 Excel
    print("\n[4/4] 计算统计量并输出到 Excel...")

    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'cpet_features_statistics_by_category_{timestamp}.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 总体统计
        print("\n计算总体统计...")
        total_stats = compute_group_statistics(all_stats, "Total")
        total_stats.to_excel(writer, sheet_name='Total', index=False)
        print(f"  总体: {len(all_stats)} 个患者")

        # 各类别统计
        for label in category_labels:
            print(f"\n计算 {label} 统计...")
            group_stats = compute_group_statistics(category_stats[label], label)
            # 清理 sheet 名称（Excel 限制 31 字符，不能包含特殊字符）
            sheet_name = label.replace('/', '_').replace('\\', '_')[:31]
            group_stats.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {label}: {len(category_stats[label])} 个患者")

        # EHR / demographics 统计表
        print("\n生成 EHR/Demographics 统计表...")
        try:
            ehr_summary_df, ehr_tests_df = compute_ehr_summary(
                label_file=LABEL_FILE,
                category_labels=category_labels,
                group_col=GROUP_COL_NAME
            )
            ehr_summary_df.to_excel(writer, sheet_name='EHR_Summary', index=False)
            ehr_tests_df.to_excel(writer, sheet_name='EHR_Statistical_Tests', index=False)
            print(f"  EHR 统计完成: {len(ehr_summary_df)} 个变量")
        except Exception as e:
            print(f"警告: EHR 统计生成失败: {e}")
            ehr_summary_df = pd.DataFrame()
            ehr_tests_df = pd.DataFrame()

        # t1-t5 功能标签在 t6 疾病机制分组下的分布与关联统计
        print("\n生成 t1-t5 功能标签分布与 t6 关联统计表...")
        try:
            label_df_for_tasks = read_label_dataframe(LABEL_FILE, group_col=GROUP_COL_NAME)
            functional_dist_df, functional_assoc_display_df, functional_assoc_raw_df = compute_functional_label_tables_from_df(
                df=label_df_for_tasks,
                group_col=GROUP_COL_NAME,
                group_labels=category_labels,
                overall_name="Overall"
            )
            functional_table_df = combine_functional_label_distribution_and_association(
                functional_dist_df,
                functional_assoc_display_df
            )
            functional_dist_df.to_excel(writer, sheet_name='Functional_Label_Distribution', index=False)
            functional_assoc_display_df.to_excel(writer, sheet_name='Functional_Label_Association', index=False)
            functional_assoc_raw_df.to_excel(writer, sheet_name='Functional_Label_Assoc_Raw', index=False)
            functional_table_df.to_excel(writer, sheet_name='Functional_Label_Table', index=False)
            print(f"  t1-t5 功能标签统计完成: {len(functional_dist_df)} 行分布, {len(functional_assoc_display_df)} 个关联检验")
        except Exception as e:
            print(f"警告: t1-t5 功能标签统计生成失败: {e}")

        # 汇总表（所有类别并排）
        print("\n生成汇总表...")

        # 构建汇总 DataFrame
        summary_data = {'Feature': []}
        for out_idx in range(30):
            stat_type = get_statistic_type(out_idx)
            summary_data['Feature'].append(f"{stat_type} {FEATURE_NAMES[out_idx]}")

        # 总体
        for out_idx in range(30):
            col_data = all_stats[:, out_idx]
            valid_values = col_data[~np.isnan(col_data)]
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                summary_data.setdefault('Total', []).append(f"{mean_val:.2f} ± {std_val:.2f}")
            else:
                summary_data.setdefault('Total', []).append("N/A")

        # 各类别
        for label in category_labels:
            col_name = label[:20]  # 限制列名长度
            stats = category_stats[label]
            for out_idx in range(30):
                col_data = stats[:, out_idx]
                valid_values = col_data[~np.isnan(col_data)]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    summary_data.setdefault(col_name, []).append(f"{mean_val:.2f} ± {std_val:.2f}")
                else:
                    summary_data.setdefault(col_name, []).append("N/A")

        # t6 多类别组间统计检验：Kruskal-Wallis + FDR + epsilon-squared
        stat_tests_df = compute_feature_group_tests(category_stats)

        summary_data['P value'] = stat_tests_df['P value'].tolist()
        summary_data['FDR-adjusted P value'] = stat_tests_df['FDR-adjusted P value'].tolist()
        summary_data['Effect size'] = stat_tests_df['Effect size'].apply(
            lambda x: f"{x:.3f}" if not pd.isna(x) else ""
        ).tolist()
        summary_data['Effect size type'] = stat_tests_df['Effect size type'].tolist()
        summary_data['Test'] = stat_tests_df['Test'].tolist()

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 合并 EHR + CPET 的 Table 1 草稿，方便后续直接排版
        if not ehr_summary_df.empty:
            table1_combined_df = add_section_rows_for_table1(ehr_summary_df, summary_df)
            table1_combined_df.to_excel(writer, sheet_name='Table1_Combined', index=False)

        # 单独输出完整统计检验结果，保留原始 P value 与各组有效样本量
        stat_tests_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)

        # 样本量统计
        sample_counts = {
            'Category': ['Total'] + category_labels,
            'N': [len(all_stats)] + [len(category_stats[l]) for l in category_labels]
        }
        sample_df = pd.DataFrame(sample_counts)
        sample_df.to_excel(writer, sheet_name='Sample_Size', index=False)

        # 方法说明
        method_info = {
            'Feature': FEATURE_NAMES,
            'Statistic': [get_statistic_type(i) for i in range(30)],
            'Description': [
                '运动期最大值' if i in PEAK_INDICES else
                '运动期最小值' if i in MIN_INDICES else
                '运动期均值'
                for i in range(30)
            ]
        }
        method_df = pd.DataFrame(method_info)
        method_df.to_excel(writer, sheet_name='Method', index=False)

    print(f"\n统计结果已保存到: {output_file}")
    print("=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()
