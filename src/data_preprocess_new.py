"""
新数据预处理模块
用于处理用户配置的 CPET 数据目录
"""

import os
import pandas as pd
import numpy as np
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")


def find_data_start_rows(df, start_search=8, max_search=2500):
    """
    自动检测两个数据段的起始行
    第一段: 包含'Time', 'MET', 'Load'等13个特征
    第二段: 包含'dO2/dW', 'BF', "V'E"等9个特征

    Returns:
        tuple: (section1_header_row, section2_header_row)
    """
    section1_header = None
    section2_header = None

    # 查找第一段表头（包含MET, Load, HR等）
    for i in range(start_search, min(max_search, len(df))):
        row_str = ' '.join([str(x) for x in df.iloc[i].tolist()])
        if 'Time' in row_str and 'MET' in row_str and 'Load' in row_str and 'HR' in row_str:
            section1_header = i
            break

    # 查找第二段表头（包含dO2/dW, BF, V'E等）
    for i in range(10, min(max_search, len(df))):
        row_str = ' '.join([str(x) for x in df.iloc[i].tolist()])
        if 'dO2/dW' in row_str or ('BF' in row_str and "V'E" in row_str):
            section2_header = i
            break

    # 默认值
    if section1_header is None:
        section1_header = 12
    if section2_header is None:
        section2_header = 156

    return section1_header, section2_header


def find_exercise_periods(load_column):
    """
    识别静息期、运动期和恢复期的边界
    通过检测Load列的变化（0 -> >0 -> 0）

    Args:
        load_column: Load列的数据（list或Series）

    Returns:
        periods: dict {
            'rest_start': 静息期起始索引,
            'rest_end': 静息期结束索引,
            'exercise_start': 运动期起始索引,
            'exercise_end': 运动期结束索引,
            'recovery_start': 恢复期起始索引,
            'recovery_end': 恢复期结束索引
        }
    """
    # 转换为Series以使用pandas方法
    if not isinstance(load_column, pd.Series):
        load_column = pd.Series(load_column)

    load_values = pd.to_numeric(load_column, errors='coerce')
    load_values = load_values.fillna(0).values

    # 找到所有Load > 0的位置
    exercise_mask = load_values > 0
    exercise_indices = np.where(exercise_mask)[0]

    if len(exercise_indices) == 0:
        # 没有运动期，全部视为静息期
        return {
            'rest_start': 0,
            'rest_end': len(load_values),
            'exercise_start': len(load_values),
            'exercise_end': len(load_values),
            'recovery_start': len(load_values),
            'recovery_end': len(load_values)
        }

    # 找到运动期的起始和结束
    exercise_start = exercise_indices[0]
    exercise_end = exercise_indices[-1] + 1

    # 静息期: 0 到 exercise_start
    rest_start = 0
    rest_end = exercise_start

    # 运动期: exercise_start 到 exercise_end
    # (已定义)

    # 恢复期: exercise_end 到结束
    recovery_start = exercise_end
    recovery_end = len(load_values)

    return {
        'rest_start': rest_start,
        'rest_end': rest_end,
        'exercise_start': exercise_start,
        'exercise_end': exercise_end,
        'recovery_start': recovery_start,
        'recovery_end': recovery_end
    }


def resample_to_fixed_length(data, target_length=162):
    """
    将数据重采样到固定长度

    Args:
        data: numpy array [n_samples, n_features]
        target_length: 目标长度（默认162，基于第一段数据统计）

    Returns:
        resampled_data: [target_length, n_features]
    """
    n_samples, n_features = data.shape

    if n_samples == target_length:
        return data

    # 使用线性插值
    x_old = np.linspace(0, 1, n_samples)
    x_new = np.linspace(0, 1, target_length)

    resampled_data = np.zeros((target_length, n_features))
    for i in range(n_features):
        f = interpolate.interp1d(x_old, data[:, i], kind='linear',
                                fill_value='extrapolate')
        resampled_data[:, i] = f(x_new)

    return resampled_data

# 定义处理函数：仅对前后非空的中间数值进行线性插值
def interpolate_internal(df, features):
    for feat in features:
        if feat in df.columns:
            # limit_area='inside' 确保只插值中间缺失值，不填充两端的 NaN (不外推)
            df[feat] = df[feat].interpolate(method='linear', limit_area='inside')
    return df

def clean_data(data_str):
    """
    清理数据中的异常值
    将 '-', 'nan', '' 等转换为NaN
    """
    if pd.isna(data_str):
        return np.nan
    if isinstance(data_str, str):
        data_str = data_str.strip()
        if data_str in ['-', '', 'nan', 'NaN']:
            return np.nan
    try:
        return float(data_str)
    except:
        return np.nan

def middle_5_of_7_filter(series):
    """
    实现经典的 'Middle-5-of-7' 平滑算法 (Wasserman et al.)
    对每个滑动窗口（7点），去除最大值和最小值，取剩余5点的平均值。
    能有效去除单次呼吸的噪声（咳嗽、吞咽等）。
    """
    # 滚动窗口计算，center=True 保证相位不偏移
    return series.rolling(window=7, center=True, min_periods=1).apply(
        lambda x: np.mean(np.sort(x)[1:-1]) if len(x) >= 3 else np.mean(x),
        raw=True
    )

def advanced_processing(df, base_enabled=True, o2pulse_enabled=False, vco2_enabled=False):
    """
    执行高级预处理流程：
    1. 异常值掩膜 (Masking)
    2. 缺失值插值 (Imputation)
    3. 信号平滑 (Smoothing)
    4. 气体延迟对齐 (Time Alignment)
    5. 衍生指标重算 (Recalculation)
    6. 医学先验衍生指标 (Medical Prior Features) [可插拔模块]
    7. 氧脉搏导数特征 (O2Pulse Derivatives) [可插拔模块]
    8. V'CO2 独立特征 (九图模式) [可插拔模块]

    Args:
        df: 原始 DataFrame
        base_enabled: 是否启用基础衍生特征 (PP, OUES, EqO2_COP, HR_diff) [新增]
        o2pulse_enabled: 是否启用氧脉搏导数特征 (O2Pulse, d(O2P)/dt, d2(O2P)/dt2)
        vco2_enabled: 是否输出 V'CO2 作为独立特征 (九图模式)

    Returns:
        处理后的 DataFrame
    """
    # ---------------------------------------------------------
    # 1. 异常值清洗 (Global Artifact Removal)
    # ---------------------------------------------------------
    # HR: 生理范围 30-240 bpm
    if 'HR' in df.columns:
        df.loc[(df['HR'] < 30) | (df['HR'] > 240), 'HR'] = np.nan

    # SpO2: 生理范围 50-100%
    if 'SpO2' in df.columns:
        df.loc[(df['SpO2'] < 50) | (df['SpO2'] > 100), 'SpO2'] = np.nan

    # 气体数据: 必须 > 0
    gas_cols = ["V'O2", "V'E", "PETO2", "PETCO2", "VO2/kg"]
    for col in gas_cols:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    # ---------------------------------------------------------
    # 2. 基础插值 (Basic Imputation)
    # ---------------------------------------------------------
    # 对清洗产生的 NaN 进行线性插值
    df = df.interpolate(method='linear', limit_direction='both', axis=0)
    # 边缘填充（如果开头/结尾是 NaN）
    df = df.ffill().bfill()

    # ---------------------------------------------------------
    # 3. 恢复隐含变量 V'CO2
    # ---------------------------------------------------------
    # 原始数据通常只给 RER，不给 VCO2。我们需要 VCO2 来做平滑。
    # V'CO2 = RER * V'O2
    if 'RER' in df.columns and "V'O2" in df.columns:
        df["V'CO2_raw"] = df['RER'] * df["V'O2"]
    else:
        df["V'CO2_raw"] = df["V'O2"] * 0.8 # 兜底逻辑

    # ---------------------------------------------------------
    # 4. 信号平滑 (Smoothing - Middle 5 of 7)
    # ---------------------------------------------------------
    # 需要应用强平滑的呼吸代谢指标
    smooth_targets = ["V'O2", "V'CO2_raw", "V'E", "VO2/kg", "BF", "VTex", "PETO2", "PETCO2", "HR"]

    for col in smooth_targets:
        if col in df.columns:
            # 创建平滑后的新列，覆盖原列或存为中间变量
            # 这里直接覆盖原列，因为我们需要的是处理后的干净数据
            df[col] = middle_5_of_7_filter(df[col])

    # 4.2 中值滤波：仅针对容易受运动伪影干扰产生瞬间"椒盐噪声"的指标
    if 'SpO2' in df.columns:
        # 完美剔除瞬间掉线的一两个点，保留生理下降趋势
        df['SpO2'] = df['SpO2'].rolling(window=5, center=True, min_periods=1).median()

    # 4.3 阶跃信号处理：血压这种低频定点测量指标，保持其阶跃状态 (前向填充)
    # 移除了 Load，因为 Load 本身就是干净的程序控制变量，不需要滤波
    step_targets = ['Psys', 'Pdia']
    for col in step_targets:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ---------------------------------------------------------
    # 5. 时间对齐 (Time Alignment)
    # ---------------------------------------------------------
    # 气体分析仪通常比 HR/Load 滞后约 15-20秒。
    # 假设数据采样率为 Breath-by-Breath (约3-4秒/点)，则滞后约 4-6 个点。
    # 我们将所有气体数据"向上平移"（向左，时间提前）5个单位。
    gas_shift_rows = -5
    gas_vars = ["V'O2", "V'CO2_raw", "V'E", "VO2/kg", "PETO2", "PETCO2", "BF", "VTex"]

    for col in gas_vars:
        if col in df.columns:
            df[col] = df[col].shift(gas_shift_rows)

    # 平移后末尾会有 NaN，用最后的值填充
    df = df.ffill()

    # ---------------------------------------------------------
    # 6. 衍生指标重算 (Recalculation)
    # ---------------------------------------------------------
    # 使用平滑且对齐后的 V'O2, V'CO2, HR 重新计算比率，消除噪声放大效应

    # RER = VCO2 / VO2
    if "V'O2" in df.columns and "V'CO2_raw" in df.columns:
        df['RER'] = (df["V'CO2_raw"] / df["V'O2"]).replace([np.inf, -np.inf], np.nan)
        # 限制 RER 在生理范围内
        df['RER'] = df['RER'].clip(0.6, 1.6)

    # EqO2 = VE / VO2
    if "V'E" in df.columns and "V'O2" in df.columns:
        df['EqO2'] = (df["V'E"] / (df["V'O2"] / 1000.0)).replace([np.inf, -np.inf], np.nan) # 注意单位，假设 VO2是mL, VE是L
        # 如果原始单位已经是匹配的（通常都是 L/min 或 mL/min 的比值），则不需要 /1000
        # 观察原始数据 EqO2 范围通常在 20-40 左右。
        # 如果 VE ~ 50 L/min, VO2 ~ 2000 mL/min = 2 L/min -> Ratio = 25.
        # 所以必须把 VO2 转为 L，即 / 1000。

    # EqCO2 = VE / VCO2
    if "V'E" in df.columns and "V'CO2_raw" in df.columns:
        df['EqCO2'] = (df["V'E"] / (df["V'CO2_raw"] / 1000.0)).replace([np.inf, -np.inf], np.nan)

    # O2 Pulse = VO2 / HR
    # (CPET标准中通常是 mL/beat，无需单位转换)
    # df['O2Pulse'] = df["V'O2"] / df['HR']

    # 处理斜率指标 dO2/dW, dH/dO2
    # 这些指标是导数，对噪声极其敏感。我们不对其重算（需要准确的Load），
    # 而是对现有的斜率列进行强平滑。
    slope_cols = ['dO2/dW', 'dH/dO2']
    for col in slope_cols:
        if col in df.columns:
            # 使用更长的窗口进行平滑
            df[col] = df[col].rolling(window=9, center=True, min_periods=1).mean()

    # ---------------------------------------------------------
    # 7. 医学先验衍生指标 (Medical Prior Features) [可插拔模块]
    # ---------------------------------------------------------
    # 目的：显式注入具有强物理意义的跨变量非线性关系，
    # 降低图网络对长尾疾病分类的拟合难度

    if base_enabled:  # [修改] 仅当启用时计算基础衍生特征
        # 7.1 脉压差 (Pulse Pressure, PP)
        # 医学动机：PP 是每搏输出量（Stroke Volume）的极佳代理指标
        # 当运动中 HR 持续上升但 PP 陷入平台或下降时，是缺血性心脏病的强特征
        # 数学定义：PP = Psys - Pdia
        if 'Psys' in df.columns and 'Pdia' in df.columns:
            df['PP'] = df['Psys'] - df['Pdia']
            df['PP'] = df['PP'].clip(lower=0)  # 防止异常负值
        else:
            df['PP'] = 0.0

        # 7.2 局部氧摄取效率斜率代理 (Local OUES Proxy)
        # 医学动机：VO2 与对数化 VE 的关系（OUES）对心衰极具预后价值
        # 数学定义：OUES_local = VO2 / log10(VE)
        if "V'O2" in df.columns and "V'E" in df.columns:
            # 防止 log(0) 或负数
            safe_ve = df["V'E"].clip(lower=1.0)
            df['OUES'] = df["V'O2"] / np.log10(safe_ve)
            df['OUES'] = df['OUES'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            df['OUES'] = 0.0

        # 7.3 心肺最佳点/通气谷值 (Cardiorespiratory Optimal Point, COP)
        # 医学动机：EqO2 曲线的最低点（Nadir）通常对应无氧阈（AT）
        # 让模型自动找谷值很难，用滑动窗口最小值突显这个特征
        # 数学定义：COP_t = min_{τ ∈ [t-w, t+w]} EqO2^(τ)
        if 'EqO2' in df.columns:
            # 使用 15 个时间步的滑动窗口求局部最小值
            df['EqO2_COP'] = df['EqO2'].rolling(window=15, center=True, min_periods=1).min()
        else:
            df['EqO2_COP'] = 0.0

        # 7.4 心率瞬态变化率 (HR Velocity)
        # 医学动机：捕捉心律失常、早搏或运动恢复期心率下降异常（HRR 迟缓）
        # 数学定义：ΔHR_t = HR_t - HR_{t-1}
        if 'HR' in df.columns:
            df['HR_diff'] = df['HR'].diff().fillna(0)
            # 平滑处理，防止纯噪声
            df['HR_diff'] = df['HR_diff'].rolling(window=3, center=True, min_periods=1).mean().fillna(0.0)
        else:
            df['HR_diff'] = 0.0
    else:
        # [新增] base_enabled=False 时，初始化为 0.0 (占位，避免 KeyError)
        df['PP'] = 0.0
        df['OUES'] = 0.0
        df['EqO2_COP'] = 0.0
        df['HR_diff'] = 0.0

    # 7.5 氧脉搏及其导数特征 (鉴别泵衰竭与缺血) [可插拔模块]
    # 医学动机：O2 Pulse 平台期的提前出现是心肌收缩力绝对受限的标志
    # 缺血性心脏病和泵衰竭在 CPET 中的核心鉴别点在于每搏输出量（SV）的受限轨迹
    # 泵功能极差的患者，O2 Pulse 会在运动极早期出现"平台期"甚至下降
    # 而单纯缺血患者通常在缺血阈值到达后才出现轨迹改变
    #
    # 配置: features.derived_features.o2pulse_derivatives.enabled
    if o2pulse_enabled:
        if "V'O2" in df.columns and 'HR' in df.columns:
            # 计算基础 O2 Pulse (mL/beat)
            # 将 HR 中的 0 替换为 NaN 防止除零错误
            safe_hr = df['HR'].replace(0, np.nan)
            df['O2Pulse'] = df["V'O2"] / safe_hr
            df['O2Pulse'] = df['O2Pulse'].interpolate(method='linear', limit_direction='both').fillna(0.0)

            # 强平滑处理，避免求导时噪声被二次放大
            df['O2Pulse_smooth'] = df['O2Pulse'].rolling(window=9, center=True, min_periods=1).mean()

            # O2 Pulse 一阶导数 (变化速率)
            df['d(O2P)/dt'] = df['O2Pulse_smooth'].diff().fillna(0.0)
            df['d(O2P)/dt'] = df['d(O2P)/dt'].rolling(window=5, center=True, min_periods=1).mean()

            # O2 Pulse 二阶导数 (加速度，对平台期转折点极度敏感)
            df['d2(O2P)/dt2'] = df['d(O2P)/dt'].diff().fillna(0.0)
            df['d2(O2P)/dt2'] = df['d2(O2P)/dt2'].rolling(window=5, center=True, min_periods=1).mean()

            # 清理中间平滑列，仅保留最终特征
            df = df.drop(columns=['O2Pulse_smooth'])
        else:
            df['O2Pulse'] = 0.0
            df['d(O2P)/dt'] = 0.0
            df['d2(O2P)/dt2'] = 0.0

    # ---------------------------------------------------------
    # 8. V'CO2 独立特征 (九图模式) [可插拔模块]
    # ---------------------------------------------------------
    # 医学动机：V'CO2 是通气驱动控制的核心指标 (九图 P4, P5)
    # 用于计算 V'E-V'CO2 斜率，评估通气效率
    # V'CO2 已经在步骤 3 中计算并平滑为 V'CO2_raw
    if vco2_enabled:
        if "V'CO2_raw" in df.columns:
            # 输出平滑后的 V'CO2 作为独立特征
            df["V'CO2"] = df["V'CO2_raw"]
        else:
            df["V'CO2"] = 0.0

    return df


def get_data_new(filename, target_length=162, cache_dir=None, adapt_mode=None, base_enabled=True, o2pulse_enabled=False, vco2_enabled=False):
    """
    从新数据格式中提取CPET数据
    ** 新增功能: 支持 .npy 缓存读取与保存 **

    从新数据格式中提取CPET数据
    Excel文件包含两个独立的数据段:
    - 第一段（行12-156）: Time + 12个特征
    - 第二段（行156-结尾）: Time + 9个特征

    自动识别静息期-运动期-恢复期，并重采样到固定长度

    Args:
        filename: Excel文件路径
        target_length: 目标时间点数量（默认162，基于第一段数据统计）
        cache_dir: 缓存目录
        adapt_mode: 特征适配模式
        base_enabled: 是否启用基础衍生特征 (PP, OUES, EqO2_COP, HR_diff) [新增]
        o2pulse_enabled: 是否启用氧脉搏导数特征 (可插拔模块)
        vco2_enabled: 是否输出 V'CO2 作为独立特征 (九图模式)

    Returns:
        merged_data: numpy array [target_length, num_features]
                    特征数量: 22 (无衍生) 或 26 (基础衍生) 或 29 (启用 O2Pulse) 或 30 (启用 V'CO2)
    """
    # ================= 缓存逻辑开始 =================
    # 动态计算特征版本号 - 当特征数量变化时自动使旧缓存失效
    # 22原始 + 4基础衍生(PP, OUES, EqO2_COP, HR_diff) + 3氧脉搏导数(可选) + 1 V'CO2(可选)
    FEATURE_VERSION = 22 + (4 if base_enabled else 0) + (3 if o2pulse_enabled else 0) + (1 if vco2_enabled else 0)

    cache_path = None
    if cache_dir is not None:
        # 构造缓存文件名: 原文件名.npy
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        # 添加后缀以区分使用了新的高级预处理逻辑
        # [新增] 包含特征数后缀，避免不同特征配置的缓存混淆
        feature_suffix = f"_f{FEATURE_VERSION}"
        if target_length is None:
            cache_filename = name_without_ext + '_raw' + feature_suffix
        else:
            suffix=""
            if adapt_mode=='select':
                suffix="_select"
            elif adapt_mode=='medical':
                suffix="_medical"
            cache_filename = name_without_ext + suffix + feature_suffix

        cache_path = os.path.join(cache_dir, cache_filename + ".npy")

        # 如果缓存文件存在，直接读取
        if os.path.exists(cache_path):
            try:
                # print(f"加载缓存: {base_name}") # 可选：打印日志
                data = np.load(cache_path)
                # [关键修复] 检查特征维度是否匹配当前版本
                # 如果特征数量不匹配，强制重新处理
                if data.shape[1] != FEATURE_VERSION:
                    print(f"缓存特征维度不匹配 ({data.shape[1]} vs {FEATURE_VERSION})，重新处理...")
                elif target_length is not None:
                    if data.shape[0] == target_length:
                        return data
                    else:
                        return resample_to_fixed_length(data, target_length)
                else:
                    return data
            except Exception as e:
                print(f"读取缓存失败 {cache_path}: {e}，将重新处理Excel")
    # ================= 缓存逻辑结束 =================
    try:
        # 1. 读取Excel文件
        df = pd.read_excel(filename, sheet_name=0, engine='openpyxl', header=None)
        # 2. 查找两个数据段的表头行
        section1_header, section2_header = find_data_start_rows(df)
        # 3. 定义第一段的特征列（13个特征，不含Time）
        section1_features = [
            'MET', 'Load', 'RER', 'HR', 'HRR', 'dH/dO2', 'SVc',
            'Psys', 'Pdia', 'SpO2', "V'O2", 'VO2/kg'
        ]
        # 4. 定义第二段的特征列（9个特征，不含Time）
        section2_features = [
            'dO2/dW', 'BF', "V'E", 'BR', 'EqO2', 'EqCO2',
            'PETO2', 'PETCO2', 'VDc/VT', 'VTex'
        ]
        # 4.1 定义医学先验衍生特征（可插拔模块）
        # 基础衍生特征 (可插拔): PP, OUES, EqO2_COP, HR_diff
        # 氧脉搏导数特征 (可插拔): O2Pulse, d(O2P)/dt, d2(O2P)/dt2
        # V'CO2 独立特征 (九图模式): V'CO2
        derived_features = []
        if base_enabled:  # [修改] 仅当启用时添加基础衍生特征
            derived_features.extend(['PP', 'OUES', 'EqO2_COP', 'HR_diff'])
        if o2pulse_enabled:
            derived_features.extend(['O2Pulse', 'd(O2P)/dt', 'd2(O2P)/dt2'])
        if vco2_enabled:
            derived_features.append("V'CO2")
        # 5. 读取第一段数据（数据范围: section1_header+3 到 section2_header-1）
        section1_data_start = section1_header + 3
        section1_data_end = section2_header - 1
        section1_nrows = section1_data_end - section1_data_start + 1

        df_section1 = pd.read_excel(filename, sheet_name=0, engine='openpyxl',
                                     header=section1_header,
                                     skiprows=[section1_header+1, section1_header+2],
                                     nrows=section1_nrows)

        # 6. 读取第二段数据（数据范围: section2_header+3 到 结尾）
        section2_data_start = section2_header + 3
        df_section2 = pd.read_excel(filename, sheet_name=0, engine='openpyxl',
                                     header=section2_header,
                                     skiprows=[section2_header+1, section2_header+2])

        # 7. 提取第一段特征（通过模糊匹配）
        data1_dict = {}
        for feat in section1_features:
            matching_cols = [c for c in df_section1.columns if feat.lower() in str(c).lower()]
            if matching_cols:
                col_data = df_section1[matching_cols[0]].apply(clean_data)
                data1_dict[feat] = col_data
            else:
                print(f"警告: 在第一段未找到特征 {feat}")
                data1_dict[feat] = pd.Series([np.nan] * len(df_section1))

        # 8. 提取第二段特征（通过模糊匹配）
        data2_dict = {}
        for feat in section2_features:
            matching_cols = [c for c in df_section2.columns if feat.lower() in str(c).lower()]
            if matching_cols:
                col_data = df_section2[matching_cols[0]].apply(clean_data)
                data2_dict[feat] = col_data
            else:
                print(f"警告: 在第二段未找到特征 {feat}")
                data2_dict[feat] = pd.Series([np.nan] * len(df_section2))

        # 0. 清理列名（去除原始数据中可能存在的空格）
        df_section1.columns = df_section1.columns.str.strip()
        df_section2.columns = df_section2.columns.str.strip()

        # 9. 对齐两段数据的长度（谁长截断谁）
        target_rows = min(len(df_section1), len(df_section2))

        # 统一物理截断
        df_section1 = df_section1.iloc[:target_rows].copy().reset_index(drop=True)
        df_section2 = df_section2.iloc[:target_rows].copy().reset_index(drop=True)

        # 【核心修正】强制转换为数值类型（float）
        # errors='coerce' 会把无法转换的字符串（如 "--", "n/a"）自动转为 np.nan
        data1_dict = {}
        for feat in section1_features:
            if feat in df_section1.columns:
                data1_dict[feat] = pd.to_numeric(df_section1[feat], errors='coerce')
        
        data2_dict = {}
        for feat in section2_features:
            if feat in df_section2.columns:
                data2_dict[feat] = pd.to_numeric(df_section2[feat], errors='coerce')

        # =========================================================================
        # [新增/修改] 在这里插入高级预处理逻辑 (在合并成numpy数组之前)
        # =========================================================================
        
        # 9a. 临时构建 DataFrame
        temp_data = {}
        temp_data.update(data1_dict)
        temp_data.update(data2_dict)
        df_processing = pd.DataFrame(temp_data)
        
        # 9b. 执行高级预处理 (清洗、滤波、重算)
        df_processing = advanced_processing(df_processing, base_enabled=base_enabled, o2pulse_enabled=o2pulse_enabled, vco2_enabled=vco2_enabled)
        
        # 10. 按照 NEW_FEATURES 顺序合并所有特征 (修改为从处理后的df_processing读取)
        all_features = []

        # 处理 section1 特征
        for feat in section1_features:
            if feat in df_processing.columns:
                val = df_processing[feat].values
            else:
                val = np.zeros(target_rows)
            all_features.append(val)

        # 处理 section2 特征
        for feat in section2_features:
            if feat in df_processing.columns:
                val = df_processing[feat].values
            else:
                val = np.zeros(target_rows)
            all_features.append(val)

        # 处理医学先验衍生特征
        for feat in derived_features:
            if feat in df_processing.columns:
                val = df_processing[feat].values
            else:
                val = np.zeros(target_rows)
            all_features.append(val)

        # 转换为 numpy 数组
        merged_array = np.column_stack(all_features).astype(float)

        # 11. 处理缺失值（保留原有逻辑作为兜底）
        merged_array = np.nan_to_num(merged_array, nan=0.0)
        # 替换 Inf 为 0 (或者一个大数，建议 0 以避免均值偏移)
        merged_array = np.nan_to_num(merged_array, nan=0.0, posinf=0.0, neginf=0.0)
        # 增加生理数值截断 (Clip)，防止个别巨大噪声拉偏标准化
        # 例如：限制值在 [-1e5, 1e5] 之间，虽然粗暴但能保命
        merged_array = np.clip(merged_array, -1e6, 1e6)

        # 12. 识别周期（基于 Load）
        if 'Load' in df_processing.columns:
            load_values = df_processing['Load'].values
            periods = find_exercise_periods(load_values)
        else:
             periods = {'rest_start': 0, 'rest_end': 0, 'exercise_start': 0, 'exercise_end': 0, 'recovery_start': 0, 'recovery_end': 0}

        # 重采样逻辑：仅当 target_length 不为 None 且长度不一致时执行
        if target_length is not None and merged_array.shape[0] != target_length:
            merged_array = resample_to_fixed_length(merged_array, target_length)

        # ================= 保存缓存 =================
        if cache_path is not None:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, merged_array)
            except Exception as e:
                print(f"保存缓存失败 {cache_path}: {e}")
        # ===========================================
        return merged_array

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回全零数组
        return np.zeros((target_length, FEATURE_VERSION))


def get_data_variable_length(filename, cache_dir=None, adapt_mode=None, base_enabled=True, o2pulse_enabled=False, vco2_enabled=False):
    """
    加载原始数据，保留原始长度（变长序列模式）

    与 get_data_new 不同，此函数不进行固定长度插值，
    而是保留数据的原始时间维度长度，用于变长序列训练。

    Args:
        filename: Excel文件路径
        cache_dir: 缓存目录（可选）
        adapt_mode: 适配模式（用于缓存文件名区分）
        base_enabled: 是否启用基础衍生特征 (PP, OUES, EqO2_COP, HR_diff) [新增]
        o2pulse_enabled: 是否启用氧脉搏导数特征 (可插拔模块)
        vco2_enabled: 是否输出 V'CO2 作为独立特征 (九图模式)

    Returns:
        merged_data: numpy array [原始长度, num_features]
                    特征数量: 22 (无衍生) 或 26 (基础衍生) 或 29 (启用 O2Pulse) 或 30 (启用 V'CO2)
    """
    # ================= 缓存逻辑开始 =================
    # 动态计算特征版本号 - 当特征数量变化时自动使旧缓存失效
    # 22原始 + 4基础衍生(PP, OUES, EqO2_COP, HR_diff) + 3氧脉搏导数(可选) + 1 V'CO2(可选)
    FEATURE_VERSION = 22 + (4 if base_enabled else 0) + (3 if o2pulse_enabled else 0) + (1 if vco2_enabled else 0)

    cache_path = None
    if cache_dir is not None:
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]

        # 变长序列使用 '_variable' 后缀
        # [新增] 包含特征数后缀，避免不同特征配置的缓存混淆
        feature_suffix = f"_f{FEATURE_VERSION}"
        suffix = "_variable"
        if adapt_mode == 'select':
            suffix = "_select_variable"
        elif adapt_mode == 'medical':
            suffix = "_medical_variable"

        cache_filename = name_without_ext + suffix + feature_suffix
        cache_path = os.path.join(cache_dir, cache_filename + ".npy")

        # 如果缓存文件存在，直接读取
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                # [关键修复] 检查特征维度是否匹配当前版本
                if data.shape[1] != FEATURE_VERSION:
                    print(f"变长缓存特征维度不匹配 ({data.shape[1]} vs {FEATURE_VERSION})，重新处理...")
                else:
                    return data
            except Exception as e:
                print(f"读取变长缓存失败 {cache_path}: {e}，将重新处理Excel")
    # ================= 缓存逻辑结束 =================

    try:
        # 1. 读取Excel文件
        df = pd.read_excel(filename, sheet_name=0, engine='openpyxl', header=None)

        # 2. 查找两个数据段的表头行
        section1_header, section2_header = find_data_start_rows(df)

        # 3. 定义第一段的特征列（13个特征，不含Time）
        section1_features = [
            'MET', 'Load', 'RER', 'HR', 'HRR', 'dH/dO2', 'SVc',
            'Psys', 'Pdia', 'SpO2', "V'O2", 'VO2/kg'
        ]

        # 4. 定义第二段的特征列（9个特征，不含Time）
        section2_features = [
            'dO2/dW', 'BF', "V'E", 'BR', 'EqO2', 'EqCO2',
            'PETO2', 'PETCO2', 'VDc/VT', 'VTex'
        ]

        # 4.1 定义医学先验衍生特征（可插拔模块）
        # 基础衍生特征 (可插拔): PP, OUES, EqO2_COP, HR_diff
        # 氧脉搏导数特征 (可插拔): O2Pulse, d(O2P)/dt, d2(O2P)/dt2
        # V'CO2 独立特征 (九图模式): V'CO2
        derived_features = []
        if base_enabled:  # [修改] 仅当启用时添加基础衍生特征
            derived_features.extend(['PP', 'OUES', 'EqO2_COP', 'HR_diff'])
        if o2pulse_enabled:
            derived_features.extend(['O2Pulse', 'd(O2P)/dt', 'd2(O2P)/dt2'])
        if vco2_enabled:
            derived_features.append("V'CO2")

        # 5. 读取第一段数据
        section1_data_start = section1_header + 3
        section1_data_end = section2_header - 1
        section1_nrows = section1_data_end - section1_data_start + 1

        df_section1 = pd.read_excel(filename, sheet_name=0, engine='openpyxl',
                                     header=section1_header,
                                     skiprows=[section1_header+1, section1_header+2],
                                     nrows=section1_nrows)

        # 6. 读取第二段数据
        section2_data_start = section2_header + 3
        df_section2 = pd.read_excel(filename, sheet_name=0, engine='openpyxl',
                                     header=section2_header,
                                     skiprows=[section2_header+1, section2_header+2])

        # 7. 清理列名
        df_section1.columns = df_section1.columns.str.strip()
        df_section2.columns = df_section2.columns.str.strip()

        # 8. 对齐两段数据的长度
        target_rows = min(len(df_section1), len(df_section2))
        df_section1 = df_section1.iloc[:target_rows].copy().reset_index(drop=True)
        df_section2 = df_section2.iloc[:target_rows].copy().reset_index(drop=True)

        # 9. 提取特征并转换为数值
        data1_dict = {}
        for feat in section1_features:
            if feat in df_section1.columns:
                data1_dict[feat] = pd.to_numeric(df_section1[feat], errors='coerce')
            else:
                data1_dict[feat] = pd.Series([np.nan] * target_rows)

        data2_dict = {}
        for feat in section2_features:
            if feat in df_section2.columns:
                data2_dict[feat] = pd.to_numeric(df_section2[feat], errors='coerce')
            else:
                data2_dict[feat] = pd.Series([np.nan] * target_rows)

        # 10. 构建 DataFrame 并执行高级预处理
        temp_data = {}
        temp_data.update(data1_dict)
        temp_data.update(data2_dict)
        df_processing = pd.DataFrame(temp_data)
        df_processing = advanced_processing(df_processing, base_enabled=base_enabled, o2pulse_enabled=o2pulse_enabled, vco2_enabled=vco2_enabled)

        # 11. 合并所有特征
        all_features = []

        for feat in section1_features:
            if feat in df_processing.columns:
                val = df_processing[feat].values
            else:
                val = np.zeros(target_rows)
            all_features.append(val)

        for feat in section2_features:
            if feat in df_processing.columns:
                val = df_processing[feat].values
            else:
                val = np.zeros(target_rows)
            all_features.append(val)

        # 处理医学先验衍生特征
        for feat in derived_features:
            if feat in df_processing.columns:
                val = df_processing[feat].values
            else:
                val = np.zeros(target_rows)
            all_features.append(val)

        # 转换为 numpy 数组
        merged_array = np.column_stack(all_features).astype(float)

        # 12. 处理缺失值和异常值
        merged_array = np.nan_to_num(merged_array, nan=0.0, posinf=0.0, neginf=0.0)
        merged_array = np.clip(merged_array, -1e6, 1e6)

        # ================= 保存缓存 =================
        if cache_path is not None:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, merged_array)
            except Exception as e:
                print(f"保存变长缓存失败 {cache_path}: {e}")
        # ===========================================

        return merged_array

    except Exception as e:
        print(f"处理变长数据文件 {filename} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回最小有效形状 (动态特征数)
        return np.zeros((1, FEATURE_VERSION))


def batch_process_data(data_dir, output_file=None, target_length=162):
    """
    批量处理数据目录中的所有Excel文件

    Args:
        data_dir: 数据目录路径
        output_file: 输出文件路径（可选，用于保存处理后的数据）
        target_length: 目标时间点数量（默认162）

    Returns:
        data_list: 列表，包含所有处理后的数据
        file_list: 文件名列表
    """
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    file_list.sort()

    # 可以在这里指定一个临时缓存目录用于测试
    cache_dir = os.path.join(data_dir, "npy_cache")

    data_list = []
    valid_files = []

    print(f"开始处理 {len(file_list)} 个文件...")

    for i, filename in enumerate(file_list):
        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{len(file_list)} 个文件")

        filepath = os.path.join(data_dir, filename)
        # 调用时传入 cache_dir
        data = get_data_new(filepath, target_length=target_length, cache_dir=cache_dir)

        # 检查数据是否有效（不是全零）
        if not np.all(data == 0):
            data_list.append(data)
            valid_files.append(filename)

    print(f"处理完成! 共 {len(valid_files)}/{len(file_list)} 个有效文件")

    if output_file:
        np.save(output_file, np.array(data_list))
        print(f"数据已保存到 {output_file}")

    return data_list, valid_files


if __name__ == "__main__":
    # 测试代码
    test_dir = "xx_path"
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.xlsx')][:3]
        for f in test_files:
            print(f"Testing {f}...")
            d = get_data_new(os.path.join(test_dir, f), target_length=162)
            print(f"Shape: {d.shape}, Non-zero Mean: {d[d!=0].mean():.2f}")
