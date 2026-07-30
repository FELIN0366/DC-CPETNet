"""
CPET 30个动态生理特征的严谨分组统计分析脚本

核心特性：
1. 运动期切片 (Exercise Period Slicing) - 只提取运动期间数据
2. 异常值屏蔽 - 将 0.0 替换为 np.nan
3. 三种混合降维策略：
   - Peak 组（运动期最大值）：容量、负荷、绝对上限指标
   - Min 组（运动期最小值）：谷值、脱饱和、异常平台期下探指标
   - Mean 组（运动期均值）：全局效率斜率指标
4. 按 Development / Validation(Holdout) 分组统计 Mean ± SD
5. 在 Summary 表中添加 P-value、FDR-adjusted P、effect size
6. 同步输出 EHR/Demographics 的 Development / Validation 统计
"""

import os
import sys
import json
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

from collections import defaultdict
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..','src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_preprocess_new import get_data_variable_length, find_exercise_periods
from label_extractor import load_labels


# =============================================================================
# 配置
# =============================================================================

# 数据路径
DATA_DIR = r'xx_path'
LABEL_FILE = r"xx_path"
SPLIT_INFO_FILE = r"xx_path"
CACHE_DIR = os.path.join(DATA_DIR, 'npy_cache')
OUTPUT_DIR = r"xx_path"


# =============================================================================
# EHR / demographics 配置
# =============================================================================

# 这些列来自 filtered_data_1149.xlsx；find_column 会自动兼容中英文、全角/半角括号和空格差异。
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


# 用于把 filtered_data_1149.xlsx 的 EHR 行对齐到 Development / Validation。
# 优先使用文件名列；如果没有文件名列，则使用“编号”与 label_dict / split json 对齐。
EHR_FILENAME_COL_CANDIDATES = [
    "匹配的Excel文件", "匹配的 Excel 文件", "匹配文件", "Excel文件", "Excel文件名",
    "文件名", "filename", "Filename", "file", "File"
]
EHR_ID_COL_CANDIDATES = ["编号", "ID", "id", "Index", "index", "样本编号", "病例编号"]

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
    patient_stats = np.asarray(patient_stats)
    if patient_stats.size == 0:
        patient_stats = np.empty((0, 30))

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
    """论文表格中的 P-value 格式化。"""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_effect_size(effect_size):
    """效应量格式化。"""
    if pd.isna(effect_size):
        return ""
    return f"{effect_size:.3f}"


def fdr_bh(p_values):
    """
    Benjamini-Hochberg FDR correction.
    不依赖 statsmodels；仅对非 NaN 的 p-value 调用。
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    if n == 0:
        return np.asarray([], dtype=float)

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


def mannwhitney_dev_validation_test(dev_values, val_values):
    """
    Development vs Validation 的连续变量两组比较。

    使用 Mann-Whitney U test：
    - 适用于不强制假定正态分布的连续 CPET 特征；
    - effect size 使用 rank-biserial correlation；
    - 正值表示 Development 组整体取值倾向高于 Validation 组，负值反之。
    """
    if stats is None:
        raise ImportError(
            "缺少 scipy，无法计算 Mann-Whitney U test。请先安装 scipy：pip install scipy"
        )

    dev_values = np.asarray(dev_values, dtype=float)
    val_values = np.asarray(val_values, dtype=float)

    dev_values = dev_values[~np.isnan(dev_values)]
    val_values = val_values[~np.isnan(val_values)]

    n_dev = len(dev_values)
    n_val = len(val_values)

    if n_dev == 0 or n_val == 0:
        return {
            'test': 'Mann-Whitney U',
            'statistic': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'effect_size_name': 'rank_biserial_correlation',
            'n_development': n_dev,
            'n_validation': n_val
        }

    # scipy 旧版本可能不支持 method='auto'，因此做兼容处理。
    try:
        res = stats.mannwhitneyu(
            dev_values,
            val_values,
            alternative='two-sided',
            method='auto'
        )
    except TypeError:
        res = stats.mannwhitneyu(
            dev_values,
            val_values,
            alternative='two-sided'
        )

    u_stat = float(res.statistic)
    p_value = float(res.pvalue)

    # rank-biserial correlation: [-1, 1]
    # 正值表示 Development 值整体更高，负值表示 Validation 值整体更高。
    effect_size = (2.0 * u_stat / (n_dev * n_val)) - 1.0

    return {
        'test': 'Mann-Whitney U',
        'statistic': u_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': 'rank_biserial_correlation',
        'n_development': n_dev,
        'n_validation': n_val
    }


def compute_dev_validation_feature_tests(dev_stats, val_stats):
    """
    对 30 个 CPET 运动期统计特征计算：
    - P-value
    - FDR-adjusted P
    - effect size

    FDR 在 30 个特征的有效 p-value 范围内统一进行 Benjamini-Hochberg 校正。
    """
    dev_stats = np.asarray(dev_stats)
    val_stats = np.asarray(val_stats)

    if dev_stats.size == 0:
        dev_stats = np.empty((0, 30))
    if val_stats.size == 0:
        val_stats = np.empty((0, 30))

    rows = []
    for out_idx in range(30):
        stat_type = get_statistic_type(out_idx)
        feature_name = f"{stat_type} {FEATURE_NAMES[out_idx]}"

        res = mannwhitney_dev_validation_test(
            dev_stats[:, out_idx],
            val_stats[:, out_idx]
        )

        rows.append({
            'Feature': feature_name,
            'Test': res['test'],
            'Statistic': res['statistic'],
            'P-value_raw': res['p_value'],
            'Effect size': res['effect_size'],
            'Effect size type': res['effect_size_name'],
            'N_Development': res['n_development'],
            'N_Validation': res['n_validation']
        })

    test_df = pd.DataFrame(rows)

    valid_mask = test_df['P-value_raw'].notna()
    test_df['FDR-adjusted P_raw'] = np.nan
    test_df.loc[valid_mask, 'FDR-adjusted P_raw'] = fdr_bh(
        test_df.loc[valid_mask, 'P-value_raw'].values
    )

    test_df['P-value'] = test_df['P-value_raw'].apply(format_p_value)
    test_df['FDR-adjusted P'] = test_df['FDR-adjusted P_raw'].apply(format_p_value)
    test_df['Effect size'] = test_df['Effect size'].apply(format_effect_size)

    # Summary 中展示的列顺序；raw 值保留在 Statistical_Tests sheet。
    summary_cols = [
        'Feature',
        'P-value',
        'FDR-adjusted P',
        'Effect size',
        'Effect size type',
        'Test'
    ]

    detailed_cols = [
        'Feature',
        'Test',
        'Statistic',
        'P-value_raw',
        'FDR-adjusted P_raw',
        'P-value',
        'FDR-adjusted P',
        'Effect size',
        'Effect size type',
        'N_Development',
        'N_Validation'
    ]

    return test_df[summary_cols], test_df[detailed_cols]



# =============================================================================
# EHR / demographics 统计模块
# =============================================================================

def normalize_column_name(name):
    """
    标准化列名，增强对中文括号、空格、大小写的兼容性。
    """
    s = str(name).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace(" ", "").replace("\t", "").replace("\n", "")
    s = s.replace("㎡", "m^2")
    return s.lower()


def find_column(df, aliases, required=False):
    """
    根据候选别名在 df.columns 中查找真实列名。
    """
    norm_to_col = {normalize_column_name(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in norm_to_col:
            return norm_to_col[key]

    # 兜底：允许包含式匹配，例如“体重（kg）”与“体重”。
    alias_keys = [normalize_column_name(a) for a in aliases]
    for col in df.columns:
        col_key = normalize_column_name(col)
        for key in alias_keys:
            if key and (key in col_key or col_key in key):
                return col

    if required:
        raise KeyError(f"无法在 Excel 中找到列: {aliases}; 当前 df.columns={df.columns.tolist()}")
    return None


def read_ehr_dataframe(label_file, max_header_rows=10):
    """
    读取 filtered_data_1149.xlsx，并自动尝试识别 header 行。
    命中条件：至少识别到一个 EHR 变量列，并且能识别到文件名列或编号列之一。
    """
    last_df = None
    for header_row in range(max_header_rows):
        try:
            df = pd.read_excel(label_file, engine='openpyxl', header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
            last_df = df

            has_ehr_col = any(
                find_column(df, aliases, required=False) is not None
                for _, aliases in EHR_CONTINUOUS_SPECS + [EHR_SEX_SPEC]
            )
            has_key_col = (
                find_column(df, EHR_FILENAME_COL_CANDIDATES, required=False) is not None
                or find_column(df, EHR_ID_COL_CANDIDATES, required=False) is not None
            )
            if has_ehr_col and has_key_col:
                print(f"EHR 表头识别成功: header_row={header_row}, Excel 行号={header_row + 1}")
                return df
        except Exception:
            continue

    if last_df is None:
        raise ValueError(f"无法读取 EHR Excel: {label_file}")
    raise KeyError(
        "无法识别 EHR 表头：需要至少一个 EHR 列，并且需要文件名列或编号列。"
        f"当前最后一次 df.columns={last_df.columns.tolist()}"
    )


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
    兼容中文“男/女”和常见英文/数值写法。
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
    if stats is None:
        raise ImportError("缺少 scipy，无法计算 Chi-square / Cramér's V。请先安装 scipy：pip install scipy")
    contingency_table = np.asarray(contingency_table)
    if contingency_table.size == 0:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    r, c = contingency_table.shape
    if n == 0 or min(r - 1, c - 1) == 0:
        return np.nan
    return np.sqrt(chi2 / (n * min(r - 1, c - 1)))


def build_id_to_split_from_label_dict(label_dict, split_patients):
    """
    根据 label_dict(filename -> 编号) 和 split_patients(filename -> set) 构建 编号 -> Development/Validation 的映射。
    """
    filename_to_split = {}
    for split_name, files in split_patients.items():
        for filename in files:
            filename_to_split[normalize_filename(filename)] = split_name

    id_to_split = {}
    for filename, sample_id in label_dict.items():
        norm_filename = normalize_filename(filename)
        split_name = filename_to_split.get(norm_filename)
        norm_id = normalize_sample_id(sample_id)
        if split_name is not None and norm_id is not None:
            id_to_split[norm_id] = split_name
    return id_to_split


def add_dev_validation_split_to_ehr_df(df, split_patients, label_dict=None, split_info=None):
    """
    给 EHR dataframe 添加 __Set__ 列，取值为 Development / Validation。

    对齐优先级：
    1. EHR 表中的文件名列 vs split_patients 中的文件名；
    2. EHR 表中的编号列 vs label_dict(filename -> 编号) + split_patients；
    3. EHR 表中的编号列 vs split_info 的 dev_indices/test_indices；
    4. 行号兜底：如果 split_info 的 indices 是 0-based 行号，则按 EHR 行顺序映射。
    """
    df = df.copy()

    filename_to_split = {}
    for split_name, files in split_patients.items():
        for filename in files:
            filename_to_split[normalize_filename(filename)] = split_name

    filename_col = find_column(df, EHR_FILENAME_COL_CANDIDATES, required=False)
    id_col = find_column(df, EHR_ID_COL_CANDIDATES, required=False)

    df['__Set__'] = np.nan

    # 方案 A：文件名列直接对齐。
    if filename_col is not None:
        norm_filenames = df[filename_col].apply(normalize_filename)
        df['__Set__'] = norm_filenames.map(filename_to_split)

    # 方案 B：编号列对齐 label_dict 映射。
    if df['__Set__'].isna().all() and id_col is not None and label_dict is not None:
        id_to_split = build_id_to_split_from_label_dict(label_dict, split_patients)
        norm_ids = df[id_col].apply(normalize_sample_id)
        df['__Set__'] = norm_ids.map(id_to_split)

    # 方案 C：编号列直接对齐 split_info 中的 dev_indices/test_indices。
    if df['__Set__'].isna().all() and id_col is not None and split_info is not None:
        id_to_split = {}
        for sample_id in split_info.get('dev_indices', []):
            id_to_split[normalize_sample_id(sample_id)] = 'Development'
        for sample_id in split_info.get('test_indices', []):
            id_to_split[normalize_sample_id(sample_id)] = 'Validation'
        norm_ids = df[id_col].apply(normalize_sample_id)
        df['__Set__'] = norm_ids.map(id_to_split)

    # 方案 D：行号兜底。如果 json 里的 indices 是 0-based 行号，可按 EHR 当前顺序映射。
    if df['__Set__'].isna().all() and split_info is not None:
        row_to_split = {}
        for pos in split_info.get('dev_indices', []):
            try:
                row_to_split[int(float(pos))] = 'Development'
            except (TypeError, ValueError):
                pass
        for pos in split_info.get('test_indices', []):
            try:
                row_to_split[int(float(pos))] = 'Validation'
            except (TypeError, ValueError):
                pass
        df['__Set__'] = [row_to_split.get(i, np.nan) for i in range(len(df))]

    df = df[df['__Set__'].isin(['Development', 'Validation'])].copy()
    if df.empty:
        raise ValueError(
            "EHR 数据无法对齐到 Development / Validation。请检查 filtered_data_1149.xlsx 中是否存在文件名列或编号列，"
            "以及 holdout_split_info_mtl.json 的 dev/test 字段是否与其一致。"
        )
    return df


def compute_continuous_ehr_dev_validation_test(df, value_col):
    """
    EHR 连续变量：Development vs Validation 的 Mann-Whitney U test + rank-biserial correlation。
    """
    dev_values = pd.to_numeric(
        df.loc[df['__Set__'] == 'Development', value_col], errors='coerce'
    ).dropna().astype(float).values
    val_values = pd.to_numeric(
        df.loc[df['__Set__'] == 'Validation', value_col], errors='coerce'
    ).dropna().astype(float).values

    res = mannwhitney_dev_validation_test(dev_values, val_values)
    return {
        'Statistic': res['statistic'],
        'P value_raw': res['p_value'],
        'Effect size': res['effect_size'],
        'Effect size type': res['effect_size_name'],
        'Test': res['test'],
        'N_Development': res['n_development'],
        'N_Validation': res['n_validation'],
    }


def compute_categorical_ehr_dev_validation_test(df, category_col):
    """
    EHR 分类变量：Sex 的 Chi-square test + Cramér's V。
    """
    if stats is None:
        raise ImportError("缺少 scipy，无法计算 Chi-square test。请先安装 scipy：pip install scipy")

    sub = df[['__Set__', category_col]].copy()
    sub[category_col] = sub[category_col].apply(normalize_sex_value)
    sub = sub.dropna(subset=['__Set__', category_col])

    table = pd.crosstab(sub['__Set__'], sub[category_col])
    table = table.reindex(index=['Development', 'Validation'], fill_value=0)

    n_dev = int(table.loc['Development'].sum()) if 'Development' in table.index else 0
    n_val = int(table.loc['Validation'].sum()) if 'Validation' in table.index else 0

    if table.shape[0] < 2 or table.shape[1] < 2 or table.values.sum() == 0:
        chi2, p_value, effect_size = np.nan, np.nan, np.nan
    else:
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(table.values, correction=False)
            effect_size = cramers_v_from_table(table.values)
        except ValueError:
            chi2, p_value, effect_size = np.nan, np.nan, np.nan

    return {
        'Statistic': chi2,
        'P value_raw': p_value,
        'Effect size': effect_size,
        'Effect size type': "Cramer's V",
        'Test': 'Chi-square',
        'N_Development': n_dev,
        'N_Validation': n_val,
    }


def compute_ehr_summary_dev_validation(label_file, split_patients, label_dict=None, split_info=None):
    """
    生成 EHR/Demographics 统计表，格式对齐论文 Table 1：
    Variable | Overall (N=...) | Development (n=...) | Validation (n=...) | P value | FDR-adjusted P value | Effect size

    连续变量：Age / Weight / Height / BMI，显示 mean ± SD，检验用 Mann-Whitney U，效应量 rank-biserial correlation。
    分类变量：Sex，显示 Male n (%)，检验用 Chi-square，效应量 Cramér's V。
    """
    df_raw = read_ehr_dataframe(label_file)
    df = add_dev_validation_split_to_ehr_df(
        df_raw,
        split_patients=split_patients,
        label_dict=label_dict,
        split_info=split_info
    )

    n_overall = len(df)
    n_dev = int((df['__Set__'] == 'Development').sum())
    n_val = int((df['__Set__'] == 'Validation').sum())

    overall_header = f"Overall (N={n_overall})"
    dev_header = f"Development (n={n_dev})"
    val_header = f"Validation (n={n_val})"

    summary_rows = []
    test_rows = []

    # 连续变量
    for display_name, aliases in EHR_CONTINUOUS_SPECS:
        value_col = find_column(df, aliases, required=False)

        row = {
            'Variable': display_name,
            overall_header: "N/A",
            dev_header: "N/A",
            val_header: "N/A",
        }
        test_row = {
            'Variable': display_name,
            'Variable type': 'continuous',
            'Source column': value_col if value_col is not None else '',
            'Test': 'Mann-Whitney U',
            'Statistic': np.nan,
            'P value_raw': np.nan,
            'FDR-adjusted P value_raw': np.nan,
            'P value': '',
            'FDR-adjusted P value': '',
            'Effect size': np.nan,
            'Effect size type': 'rank_biserial_correlation',
            'N_Development': 0,
            'N_Validation': 0,
        }

        if value_col is not None:
            row[overall_header] = format_mean_sd(df[value_col])
            row[dev_header] = format_mean_sd(df.loc[df['__Set__'] == 'Development', value_col])
            row[val_header] = format_mean_sd(df.loc[df['__Set__'] == 'Validation', value_col])

            test_res = compute_continuous_ehr_dev_validation_test(df, value_col)
            test_row.update({
                'Statistic': test_res['Statistic'],
                'P value_raw': test_res['P value_raw'],
                'P value': format_p_value(test_res['P value_raw']),
                'Effect size': test_res['Effect size'],
                'Effect size type': test_res['Effect size type'],
                'Test': test_res['Test'],
                'N_Development': test_res['N_Development'],
                'N_Validation': test_res['N_Validation'],
            })

        summary_rows.append(row)
        test_rows.append(test_row)

    # 性别变量
    sex_display_name, sex_aliases = EHR_SEX_SPEC
    sex_col = find_column(df, sex_aliases, required=False)
    row = {
        'Variable': sex_display_name,
        overall_header: "N/A",
        dev_header: "N/A",
        val_header: "N/A",
    }
    test_row = {
        'Variable': sex_display_name,
        'Variable type': 'categorical',
        'Source column': sex_col if sex_col is not None else '',
        'Test': 'Chi-square',
        'Statistic': np.nan,
        'P value_raw': np.nan,
        'FDR-adjusted P value_raw': np.nan,
        'P value': '',
        'FDR-adjusted P value': '',
        'Effect size': np.nan,
        'Effect size type': "Cramer's V",
        'N_Development': 0,
        'N_Validation': 0,
    }

    if sex_col is not None:
        row[overall_header] = format_male_n_percent(df[sex_col])
        row[dev_header] = format_male_n_percent(df.loc[df['__Set__'] == 'Development', sex_col])
        row[val_header] = format_male_n_percent(df.loc[df['__Set__'] == 'Validation', sex_col])

        test_res = compute_categorical_ehr_dev_validation_test(df, sex_col)
        test_row.update({
            'Statistic': test_res['Statistic'],
            'P value_raw': test_res['P value_raw'],
            'P value': format_p_value(test_res['P value_raw']),
            'Effect size': test_res['Effect size'],
            'Effect size type': test_res['Effect size type'],
            'Test': test_res['Test'],
            'N_Development': test_res['N_Development'],
            'N_Validation': test_res['N_Validation'],
        })

    summary_rows.append(row)
    test_rows.append(test_row)

    ehr_summary_df = pd.DataFrame(summary_rows)
    ehr_tests_df = pd.DataFrame(test_rows)

    # EHR block 内部进行 FDR 校正，不与 30 个 CPET features 混合校正。
    valid_mask = ehr_tests_df['P value_raw'].notna()
    if valid_mask.any():
        adjusted = fdr_bh(ehr_tests_df.loc[valid_mask, 'P value_raw'].values)
        ehr_tests_df.loc[valid_mask, 'FDR-adjusted P value_raw'] = adjusted
        ehr_tests_df.loc[valid_mask, 'FDR-adjusted P value'] = [format_p_value(p) for p in adjusted]

    # 将检验结果回填到 summary。
    ehr_summary_df['P value'] = ehr_tests_df['P value'].tolist()
    ehr_summary_df['FDR-adjusted P value'] = ehr_tests_df['FDR-adjusted P value'].tolist()
    ehr_summary_df['Effect size'] = ehr_tests_df['Effect size'].apply(format_effect_size).tolist()
    ehr_summary_df['Effect size type'] = ehr_tests_df['Effect size type'].tolist()
    ehr_summary_df['Test'] = ehr_tests_df['Test'].tolist()

    return ehr_summary_df, ehr_tests_df


def add_section_rows_for_table1_dev_validation(ehr_summary_df, cpet_summary_df):
    """
    生成一个合并版 Table 1 草稿：Demographics + Mean/Min/Peak Exercise(RAW)。
    注意：EHR 与 CPET 的 FDR 分别在各自 block 内校正，不在此处重新混合校正。
    """
    ehr = ehr_summary_df.copy()
    ehr.insert(0, 'Section', 'Demographics')

    cpet = cpet_summary_df.copy()
    if 'Feature' in cpet.columns:
        cpet = cpet.rename(columns={'Feature': 'Variable'})

    # 将 CPET 的 Total / Development / Validation 对齐到 EHR 的 Overall (N=...) / Development (n=...) / Validation (n=...) 表头。
    rename_map = {}
    overall_cols = [c for c in ehr.columns if str(c).startswith('Overall (N=')]
    dev_cols = [c for c in ehr.columns if str(c).startswith('Development (n=')]
    val_cols = [c for c in ehr.columns if str(c).startswith('Validation (n=')]
    if overall_cols and 'Total' in cpet.columns:
        rename_map['Total'] = overall_cols[0]
    if dev_cols and 'Development' in cpet.columns:
        rename_map['Development'] = dev_cols[0]
    if val_cols and 'Validation' in cpet.columns:
        rename_map['Validation'] = val_cols[0]
    cpet = cpet.rename(columns=rename_map)

    def section_name(variable):
        variable = str(variable)
        if variable.startswith('Mean '):
            return 'Mean Exercise(RAW)'
        if variable.startswith('Min '):
            return 'Min Exercise(RAW)'
        if variable.startswith('Peak '):
            return 'Peak Exercise(RAW)'
        return 'CPET Exercise(RAW)'

    cpet.insert(0, 'Section', cpet['Variable'].apply(section_name))

    preferred = [
        'Section', 'Variable',
        *[c for c in ehr.columns if str(c).startswith('Overall (N=')],
        *[c for c in ehr.columns if str(c).startswith('Development (n=')],
        *[c for c in ehr.columns if str(c).startswith('Validation (n=')],
        'P value', 'FDR-adjusted P value', 'Effect size', 'Effect size type', 'Test'
    ]
    common_cols = []
    for col in preferred + list(ehr.columns) + list(cpet.columns):
        if col not in common_cols:
            common_cols.append(col)

    return pd.concat(
        [ehr.reindex(columns=common_cols), cpet.reindex(columns=common_cols)],
        ignore_index=True
    )


def normalize_filename(filename):
    """
    统一文件名格式，便于 Excel 标签表、split json 与 DATA_DIR 下文件对齐。
    - 支持完整路径或仅文件名
    - 自动补全 .xlsx 后缀
    """
    if filename is None or (isinstance(filename, float) and np.isnan(filename)):
        return None

    filename = os.path.basename(str(filename).strip())
    if not filename:
        return None

    if not filename.lower().endswith('.xlsx'):
        filename = filename + '.xlsx'
    return filename


def normalize_sample_id(sample_id):
    """
    将“编号”或 json 中的 index 统一为可比较字符串。
    例如 12.0、"12"、12 都会转为 "12"。
    """
    if sample_id is None or (isinstance(sample_id, float) and np.isnan(sample_id)):
        return None

    try:
        value = float(sample_id)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass

    return str(sample_id).strip()


def unique_keep_order(items):
    """去重且保持原始顺序。"""
    seen = set()
    output = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def build_dev_validation_groups(label_dict, split_info_file):
    """
    根据 holdout_split_info_mtl.json 构建 Development / Validation 分组。

    优先使用 json 内的 dev_filenames/test_filenames 与原始 CPET Excel 文件对齐；
    如果 json 缺少文件名字段，则使用 dev_indices/test_indices，并尝试与
    label_dict 中读取到的“编号”对齐。这样既满足按“编号”读取标签，
    又避免把每个编号错误当成一个单独类别。
    """
    if not os.path.exists(split_info_file):
        raise FileNotFoundError(f"找不到划分文件: {split_info_file}")

    with open(split_info_file, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    groups = {
        'Development': [],
        'Validation': []
    }

    # 方案 A：优先直接使用 json 中的文件名，更稳健。
    if 'dev_filenames' in split_info and 'test_filenames' in split_info:
        groups['Development'] = unique_keep_order(
            normalize_filename(f) for f in split_info.get('dev_filenames', [])
        )
        groups['Validation'] = unique_keep_order(
            normalize_filename(f) for f in split_info.get('test_filenames', [])
        )
        return groups, split_info, 'filename'

    # 方案 B：如果没有 dev_filenames/test_filenames，则根据“编号”对齐。
    id_to_filename = {}
    ordered_filenames = []
    for filename, sample_id in label_dict.items():
        norm_filename = normalize_filename(filename)
        norm_id = normalize_sample_id(sample_id)
        if norm_filename is not None:
            ordered_filenames.append(norm_filename)
        if norm_id is not None and norm_filename is not None:
            id_to_filename[norm_id] = norm_filename

    def indices_to_filenames(indices):
        filenames = []
        missing = []
        for sample_idx in indices:
            norm_id = normalize_sample_id(sample_idx)
            filename = id_to_filename.get(norm_id)

            # 兜底：如果 json 里的 dev_indices/test_indices 是 0-based 行号，
            # 而不是“编号”本身，则按 label_dict 的读取顺序映射。
            if filename is None:
                try:
                    pos = int(float(sample_idx))
                    if 0 <= pos < len(ordered_filenames):
                        filename = ordered_filenames[pos]
                except (TypeError, ValueError):
                    filename = None

            if filename is None:
                missing.append(sample_idx)
            else:
                filenames.append(filename)

        return unique_keep_order(filenames), missing

    groups['Development'], missing_dev = indices_to_filenames(split_info.get('dev_indices', []))
    groups['Validation'], missing_val = indices_to_filenames(split_info.get('test_indices', []))

    if missing_dev or missing_val:
        print("警告: 部分 dev_indices/test_indices 未能映射到文件名")
        print(f"  missing_dev: {len(missing_dev)}")
        print(f"  missing_validation: {len(missing_val)}")

    return groups, split_info, 'index_or_id'



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

    # 2. 加载编号标签
    print("\n[1/4] 加载编号数据...")
    label_dict, label_mapping = load_labels(
        label_file=LABEL_FILE,
        target_col_name="编号",
        min_label_freq=1
    )

    if not label_dict:
        print("错误: 无法加载编号数据")
        return

    print(f"成功加载 {len(label_dict)} 条编号记录")

    # 3. 根据 holdout_split_info_mtl.json 构建 Development / Validation 分组
    print("\n[2/4] 根据 holdout split 构建 Development / Validation 分组...")
    try:
        split_patients, split_info, split_source = build_dev_validation_groups(
            label_dict=label_dict,
            split_info_file=SPLIT_INFO_FILE
        )
    except Exception as e:
        print(f"错误: 无法读取或解析 holdout split 文件: {e}")
        return

    n_dev_expected = split_info.get('n_dev', len(split_patients['Development']))
    n_val_expected = split_info.get('n_test', len(split_patients['Validation']))

    print(f"划分来源: {split_source}")
    print(f"  Development: {len(split_patients['Development'])} / expected {n_dev_expected}")
    print(f"  Validation:  {len(split_patients['Validation'])} / expected {n_val_expected}")

    overlap = set(split_patients['Development']).intersection(set(split_patients['Validation']))
    if overlap:
        print(f"警告: Development 与 Validation 存在 {len(overlap)} 个重复文件，请检查 split json")

    # 4. 加载数据并计算统计量
    print("\n[3/4] 加载数据并计算运动期统计量...")

    # 存储每个 split 的统计值
    split_stats = defaultdict(list)
    all_stats = []

    # 使用进度显示
    all_files = [(f, split_name) for split_name, files in split_patients.items() for f in files]
    total_files = len(all_files)

    for idx, (filename, split_name) in enumerate(all_files):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  处理进度: {idx + 1}/{total_files}")

        data = load_patient_data(filename, cache_dir=CACHE_DIR)
        if data is not None and data.shape[0] > 0 and data.shape[1] == 30:
            stats = compute_exercise_statistics(data)
            split_stats[split_name].append(stats)
            all_stats.append(stats)

    # 转换为 numpy 数组
    for split_name in split_stats:
        split_stats[split_name] = np.array(split_stats[split_name])

    all_stats = np.array(all_stats)
    if all_stats.size == 0:
        all_stats = np.empty((0, 30))

    for split_name in ['Development', 'Validation']:
        if split_name not in split_stats or len(split_stats[split_name]) == 0:
            split_stats[split_name] = np.empty((0, 30))

    print(f"\n成功处理 {len(all_stats)} 个患者")
    for split_name in ['Development', 'Validation']:
        print(f"  {split_name}: {len(split_stats.get(split_name, []))} 个患者")

    # 5. 计算统计量并输出到 Excel
    print("\n[4/4] 计算统计量并输出到 Excel...")

    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'cpet_features_statistics_by_dev_validation_{timestamp}.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 总体统计
        print("\n计算总体统计...")
        total_stats = compute_group_statistics(all_stats, "Total")
        total_stats.to_excel(writer, sheet_name='Total', index=False)
        print(f"  总体: {len(all_stats)} 个患者")

        # Development / Validation 统计
        for split_name in ['Development', 'Validation']:
            print(f"\n计算 {split_name} 统计...")
            group_stats = compute_group_statistics(split_stats[split_name], split_name)
            sheet_name = split_name[:31]
            group_stats.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {split_name}: {len(split_stats[split_name])} 个患者")

        # EHR / demographics 统计表
        print("\n生成 EHR/Demographics 统计表...")
        try:
            ehr_summary_df, ehr_tests_df = compute_ehr_summary_dev_validation(
                label_file=LABEL_FILE,
                split_patients=split_patients,
                label_dict=label_dict,
                split_info=split_info
            )
            ehr_summary_df.to_excel(writer, sheet_name='EHR_Summary', index=False)
            ehr_tests_df.to_excel(writer, sheet_name='EHR_Statistical_Tests', index=False)
            print(f"  EHR 统计完成: {len(ehr_summary_df)} 个变量")
        except Exception as e:
            print(f"警告: EHR 统计生成失败: {e}")
            ehr_summary_df = pd.DataFrame()
            ehr_tests_df = pd.DataFrame()

        # t1-t5 功能标签在 Development / Validation 下的分布与关联统计
        print("\n生成 t1-t5 功能标签 Development / Validation 分布与关联统计表...")
        try:
            label_df_for_tasks_raw = read_ehr_dataframe(LABEL_FILE)
            label_df_for_tasks = add_dev_validation_split_to_ehr_df(
                label_df_for_tasks_raw,
                split_patients=split_patients,
                label_dict=label_dict,
                split_info=split_info
            )
            functional_dist_df, functional_assoc_display_df, functional_assoc_raw_df = compute_functional_label_tables_from_df(
                df=label_df_for_tasks,
                group_col="__Set__",
                group_labels=['Development', 'Validation'],
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

        # 汇总表（Total / Development / Validation 并排，可直接用于绘图 x 轴）
        print("\n生成汇总表...")

        # 构建汇总 DataFrame
        summary_data = {'Feature': []}
        for out_idx in range(30):
            stat_type = get_statistic_type(out_idx)
            summary_data['Feature'].append(f"{stat_type} {FEATURE_NAMES[out_idx]}")

        def append_mean_sd_column(col_name, stats_array):
            for out_idx in range(30):
                col_data = stats_array[:, out_idx]
                valid_values = col_data[~np.isnan(col_data)]
                if len(valid_values) > 0:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    summary_data.setdefault(col_name, []).append(f"{mean_val:.2f} ± {std_val:.2f}")
                else:
                    summary_data.setdefault(col_name, []).append("N/A")

        append_mean_sd_column('Total', all_stats)
        append_mean_sd_column('Development', split_stats['Development'])
        append_mean_sd_column('Validation', split_stats['Validation'])

        # Development vs Validation 的统计检验：
        # P-value、FDR-adjusted P、effect size 会合并进 Summary。
        test_summary_df, test_detailed_df = compute_dev_validation_feature_tests(
            split_stats['Development'],
            split_stats['Validation']
        )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.merge(test_summary_df, on='Feature', how='left')
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 合并 EHR + CPET 的 Table 1 草稿，方便后续直接排版。
        if not ehr_summary_df.empty:
            table1_combined_df = add_section_rows_for_table1_dev_validation(ehr_summary_df, summary_df)
            table1_combined_df.to_excel(writer, sheet_name='Table1_Combined', index=False)

        # 额外输出原始数值版统计检验结果，方便后续画图或核查。
        test_detailed_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)

        # 样本量统计
        sample_counts = {
            'Set': ['Total', 'Development', 'Validation'],
            'N': [
                len(all_stats),
                len(split_stats['Development']),
                len(split_stats['Validation'])
            ],
            'Expected_N_From_Split_JSON': [
                split_info.get('n_samples', ''),
                split_info.get('n_dev', ''),
                split_info.get('n_test', '')
            ]
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
            ],
            'Group comparison': ['Development vs Validation'] * 30,
            'P-value test': ['Mann-Whitney U test, two-sided'] * 30,
            'FDR correction': ['Benjamini-Hochberg across 30 CPET features'] * 30,
            'Effect size': [
                'rank-biserial correlation; positive means Development > Validation'
            ] * 30
        }
        method_df = pd.DataFrame(method_info)
        method_df.to_excel(writer, sheet_name='Method', index=False)

    print(f"\n统计结果已保存到: {output_file}")
    print("=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()
