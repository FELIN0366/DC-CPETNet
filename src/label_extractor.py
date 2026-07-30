"""
标签提取模块
从 final_summary_report.xlsx 提取疾病分类标签
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings

# 定义可选的分类列，方便在调用时参考
AVAILABLE_LABEL_COLUMNS = [
    "匹配的第一大类",
    "心率储备",
    "呼吸储备",
    "运动耐量",
    "运动心功能分级",
    "标准心电运动负荷试验",
    "运动中换气肺功能",
    "心脏神经调控",
    "小气道功能_程度",
    "小气道功能_类型"
]

import pandas as pd
from collections import Counter

# =============================================================================
# 默认标签文件路径 - 从配置文件读取（软编码）
# =============================================================================
def _get_default_label_file():
    """
    从配置文件读取默认标签文件路径（软编码，无硬编码备用）

    Returns:
        str: 标签文件路径

    Raises:
        RuntimeError: 如果无法从配置文件读取
    """
    try:
        from config import Config
        config = Config.load()
        if hasattr(config, 'data') and hasattr(config.data, 'label_file'):
            return config.data.label_file
        else:
            raise RuntimeError(
                "配置文件中未找到 data.label_file，请检查 configs/config.yaml"
            )
    except ImportError as e:
        raise RuntimeError(
            f"无法导入 Config 模块，请确保 src/config.py 存在: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"加载配置文件失败: {e}"
        )


def load_labels(label_file=None,
                header_row=1,
                target_col_name="匹配的第一大类",
                min_label_freq=50):
    """
    从标签文件中加载所有样本的标签

    Args:
        label_file: 标签文件路径，None 时从配置文件读取
        header_row: 表头行号
        target_col_name: 目标标签列名
        min_label_freq: 最小标签频次阈值
    """
    # 软编码：从配置文件读取默认路径
    if label_file is None:
        label_file = _get_default_label_file()

    try:
        # 读取Excel文件
        df = pd.read_excel(label_file, engine='openpyxl', header=header_row)

        print(f"标签文件列数: {len(df.columns)}")
        print(f"数据行数: {len(df)}")
        print(f"正在查找目标标签列: {target_col_name}")

        # 1. 精准匹配目标标签列
        target_col = None
        clean_target = target_col_name.replace('\n', '').replace('\r', '').replace(' ', '')
        
        for col in df.columns:
            col_str = str(col).strip()
            clean_col_str = col_str.replace('\n', '').replace('\r', '').replace(' ', '')
            
            # 使用严格相等判断
            if clean_target == clean_col_str:
                target_col = col
                break

        if target_col is None:
            print(f"错误: 未在Excel中找到精准匹配 '{target_col_name}' 的列。")
            print(f"现有列名: {[str(c) for c in df.columns]}")
            return {}, {}

        print(f">>> 成功定位标签列: {target_col}")

        # 2. 查找匹配的文件名列
        filename_col = None
        for col in df.columns:
            col_str = str(col)
            if '匹配' in col_str and 'Excel' in col_str and '文件' in col_str:
                filename_col = col
                break

        if filename_col is None:
            for col in df.columns:
                col_str = str(col).lower()
                if 'excel' in col_str or '文件' in col_str or 'file' in col_str:
                    filename_col = col
                    break

        if filename_col is None:
            print("警告: 未找到文件名列，将使用行索引作为标识")
        else:
            print(f"使用文件名列: {filename_col}")

        # 3. 提取标签和文件名（初步清洗）
        raw_labels = df[target_col].tolist()
        temp_labels = []
        temp_filenames = []
        skipped_count = 0

        for i, label in enumerate(raw_labels):
            if pd.isna(label):
                skipped_count += 1
                continue
                
            label_str = str(label).strip()
            if label_str == '' or label_str == '-':
                skipped_count += 1
                continue
            
            # 特殊标签替换处理
            if label_str == "患者因疲劳终止测试，未及AT，心肺功能分级不能评估":
                label_str = "无法评估"
                print(f"注意: 第{i}行特殊标签已重命名为'无法评估'")

            # 获取文件名
            fname_str = f"sample_{i}"
            if filename_col is not None:
                fname = df.iloc[i][filename_col]
                if not pd.isna(fname):
                    fname_str = str(fname).strip()
                    if fname_str.endswith('.xlsx'):
                        fname_str = fname_str[:-5]
                else:
                    fname_str = f"unknown_{i}"

            temp_labels.append(label_str)
            temp_filenames.append(fname_str)

        # 4. 统计频次并剔除样本数 < 50 的标签
        counts = Counter(temp_labels)
        # 找出频次 >= min_label_freq 的合法标签
        valid_labels_set = {label for label, count in counts.items() if count >= min_label_freq}
        
        labels_clean = []
        filenames = []
        for f, l in zip(temp_filenames, temp_labels):
            if l in valid_labels_set:
                labels_clean.append(l)
                filenames.append(f)
            else:
                skipped_count += 1 # 记录因样本过少而被剔除的样本

        # 5. 创建映射和字典
        unique_labels = sorted(list(set(labels_clean)))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        label_dict = {fname: label for fname, label in zip(filenames, labels_clean)}

        # --- 打印统计报告 ---
        print(f"\n清洗报告:")
        print(f" - 总样本数: {len(df)}")
        print(f" - 有效样本数: {len(labels_clean)}")
        print(f" - 剔除样本数: {skipped_count} (含空值、横杠及少于{min_label_freq}个样本的标签)")
        print(f" - 剩余唯一标签数: {len(unique_labels)}")
        print(f"最终标签分布:")
        for label in unique_labels:
            print(f"   {label}: {labels_clean.count(label)} 个样本")

        return label_dict, label_mapping

    except Exception as e:
        print(f"加载标签文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {}


def get_label_for_file(filename, label_dict):
    """
    根据文件名获取标签

    Args:
        filename: Excel文件名 (例如: 1000_20241129113219.xlsx)
        label_dict: 标签字典

    Returns:
        label: 标签名称，如果未找到返回None
    """
    # 移除.xlsx后缀
    if filename.endswith('.xlsx'):
        filename = filename[:-5]

    # 精确匹配
    if filename in label_dict:
        return label_dict[filename]

    # 尝试提取ID部分匹配
    # 尝试匹配ID_timestamp格式
    match = re.search(r'^(\d+)_', filename)
    if match:
        file_id = match.group(1)
        # 查找以这个ID开头的标签
        for key in label_dict.keys():
            if key.startswith(file_id + '_'):
                return label_dict[key]

    # 如果还是没找到，尝试部分匹配
    for key in label_dict.keys():
        if filename in key or key in filename:
            return label_dict[key]

    # 没有匹配
    return None


def create_label_encoder(label_mapping):
    """
    创建标签编码器

    Args:
        label_mapping: {标签名: 索引} 字典

    Returns:
        encoder: 函数，将标签名转换为one-hot编码
    """
    n_classes = len(label_mapping)

    def encode(label_name):
        if label_name not in label_mapping:
            return None

        one_hot = np.zeros(n_classes, dtype=np.float32)
        one_hot[label_mapping[label_name]] = 1
        return one_hot

    return encode


def load_static_features(label_file="xx_path",
                         header_row=1):
    """
    从标签文件提取静态 EHR 特征

    从 final_summary_report.xlsx 提取年龄、性别、体重、身高、BMI 等静态特征。
    相比从每个 Excel 头部解析，此方法更可靠且性能更好。

    Args:
        label_file: 标签文件路径
        header_row: 表头行号 (默认为 1)

    Returns:
        static_dict: {filename: {"age": float, "gender": int, "weight": float, "height": float, "bmi": float}}
        feature_names: ["age", "gender", "weight", "height", "bmi"]
    """
    try:
        df = pd.read_excel(label_file, header=header_row, engine='openpyxl')

        # 列索引映射 (基于数据验证结果)
        # 这些索引对应 final_summary_report.xlsx 中的列位置
        COL_AGE = 4      # 年龄
        COL_GENDER = 5   # 性别
        COL_WEIGHT = 7   # 体重
        COL_HEIGHT = 8   # 身高
        COL_BMI = 9      # BMI
        COL_FILENAME = 81  # '匹配的Excel文件' 列

        static_dict = {}
        feature_names = ["age", "gender", "weight", "height", "bmi"]

        for i, row in df.iterrows():
            # 获取文件名
            fname = str(row.iloc[COL_FILENAME]).strip() if not pd.isna(row.iloc[COL_FILENAME]) else ""
            if not fname:
                continue

            # 移除 .xlsx 后缀
            if fname.endswith('.xlsx'):
                fname = fname[:-5]

            # --- 新增逻辑：检查重复 ---
            if fname in static_dict:
                print(f"警告：发现重复的文件名 '{fname}'，行索引为: {i}。原有数据将被覆盖。")
            # -----------------------

            # 性别编码: 男=1, 女=0
            gender_raw = str(row.iloc[COL_GENDER]).strip() if not pd.isna(row.iloc[COL_GENDER]) else ""
            gender = 1 if gender_raw == '男' else 0

            # 提取静态特征 (处理可能的缺失值)
            try:
                age = float(row.iloc[COL_AGE]) if not pd.isna(row.iloc[COL_AGE]) else 0.0
                weight = float(row.iloc[COL_WEIGHT]) if not pd.isna(row.iloc[COL_WEIGHT]) else 0.0
                height = float(row.iloc[COL_HEIGHT]) if not pd.isna(row.iloc[COL_HEIGHT]) else 0.0
                bmi = float(row.iloc[COL_BMI]) if not pd.isna(row.iloc[COL_BMI]) else 0.0
            except (ValueError, TypeError):
                # 转换失败时使用默认值
                age, weight, height, bmi = 0.0, 0.0, 0.0, 0.0

            static_dict[fname] = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'bmi': bmi
            }

        print(f"[静态特征] 成功加载 {len(static_dict)} 条记录")
        return static_dict, feature_names

    except Exception as e:
        print(f"加载静态特征时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, ["age", "gender", "weight", "height", "bmi"]


# =============================================================================
# PFT (肺通气功能) 静态特征加载
# =============================================================================

def _get_pft_feature_columns(df):
    """
    动态识别 PFT 特征列

    特征选择规则:
    1. 列名包含 FVC, FEV1, PEF, PIF, FEV1/FVC, MEF75, MEF50, MEF25, FET100
    2. 填充率 > 90%
    3. 排除 ID、时间、状态等元数据列

    Args:
        df: DataFrame

    Returns:
        feature_cols: 有效特征列名列表
    """
    # PFT 特征关键词
    PFT_KEYWORDS = ['FVC', 'FEV1', 'PEF', 'PIF', 'FEV1/FVC', 'MEF75', 'MEF50', 'MEF25', 'FET100']

    # 排除的关键词 (元数据列)
    EXCLUDE_KEYWORDS = ['ID', '编号', '时间', '状态', '性别', '年龄', '身高', '体重', '设备', '科室']

    feature_cols = []
    for col in df.columns:
        col_str = str(col)

        # 检查是否包含 PFT 特征关键词
        has_pft_keyword = any(kw in col_str for kw in PFT_KEYWORDS)

        # 检查是否应排除
        should_exclude = any(kw in col_str for kw in EXCLUDE_KEYWORDS)

        # 检查填充率
        non_null_ratio = df[col].notna().sum() / len(df)

        if has_pft_keyword and not should_exclude and non_null_ratio > 0.9:
            feature_cols.append(col)

    return feature_cols


def load_pft_features(pft_file="xx_path"):
    """
    从通气功能数据文件提取 PFT 特征

    数据清洗规则:
    - 动态识别特征列 (基于关键词和填充率)
    - 填充率 > 90%
    - 缺失值用 0 填充

    Args:
        pft_file: PFT 数据文件路径

    Returns:
        pft_dict: {final_编号: [feature_values]}
        pft_feature_names: 有效特征名列表
    """
    try:
        df = pd.read_excel(pft_file, engine='openpyxl')

        print(f"[PFT特征] 原始数据: {len(df)} 行, {len(df.columns)} 列")

        # 查找 final_编号 列 (第一列)
        id_col = df.columns[0]
        print(f"[PFT特征] ID列: {id_col}")

        # 动态识别特征列
        feature_cols = _get_pft_feature_columns(df)

        if not feature_cols:
            print(f"[PFT特征] 错误: 未找到任何有效 PFT 特征")
            return {}, []

        print(f"[PFT特征] 有效特征数: {len(feature_cols)}")
        print(f"[PFT特征] 特征列表: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"[PFT特征] 特征列表: {feature_cols}")

        # 构建字典
        pft_dict = {}
        for i, row in df.iterrows():
            patient_id = str(row[id_col]).strip() if not pd.isna(row[id_col]) else ""
            if not patient_id:
                continue

            # 提取特征值
            feature_values = []
            for col in feature_cols:
                val = row.get(col, np.nan)
                if pd.isna(val):
                    feature_values.append(0.0)  # 缺失值用 0 填充
                else:
                    try:
                        feature_values.append(float(val))
                    except (ValueError, TypeError):
                        feature_values.append(0.0)

            pft_dict[patient_id] = feature_values

        print(f"[PFT特征] 成功加载 {len(pft_dict)} 条记录")
        return pft_dict, feature_cols

    except Exception as e:
        print(f"[PFT特征] 加载出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, []


def _build_filename_to_final_id_mapping(label_file="xx_path",
                                         header_row=1):
    """
    建立 filename -> final_编号 的映射

    映射规则: 从 filename 中提取编号部分
    例如: '4_20250527091158.xlsx' -> '4'

    Args:
        label_file: 标签文件路径
        header_row: 表头行号

    Returns:
        mapping: {filename: final_编号}
    """
    try:
        df = pd.read_excel(label_file, header=header_row, engine='openpyxl')

        # 查找文件名列
        filename_col = None
        for col in df.columns:
            col_str = str(col)
            if '匹配' in col_str and 'Excel' in col_str and '文件' in col_str:
                filename_col = col
                break

        if filename_col is None:
            print(f"[映射] 警告: 未找到文件名列")
            return {}

        mapping = {}
        for i, row in df.iterrows():
            fname = str(row[filename_col]).strip() if not pd.isna(row[filename_col]) else ""
            if not fname:
                continue

            # 移除 .xlsx 后缀
            if fname.endswith('.xlsx'):
                fname = fname[:-5]

            # 从 filename 提取编号 (例如: '4_20250527091158' -> '4')
            match = re.search(r'^(\d+)_', fname)
            if match:
                final_id = match.group(1)
                mapping[fname] = final_id

        print(f"[映射] 建立 {len(mapping)} 条 filename -> final_编号 映射")
        return mapping

    except Exception as e:
        print(f"[映射] 构建出错: {str(e)}")
        return {}


def _merge_static_features(ehr_dict, pft_dict, filename_to_final_id,
                           skip_missing_pft=True):
    """
    合并 EHR 和 PFT 静态特征

    Args:
        ehr_dict: {filename: {"age": float, "gender": int, "weight": float, "height": float, "bmi": float}}
        pft_dict: {final_编号: [feature_values]}
        filename_to_final_id: {filename: final_编号}
        skip_missing_pft: True=跳过缺失患者, False=零填充

    Returns:
        merged_dict: {filename: [ehr_features + pft_features]}
        num_ehr_features: 5
        num_pft_features: 动态计算
    """
    # EHR 特征顺序
    ehr_feature_names = ["age", "gender", "weight", "height", "bmi"]
    num_ehr_features = 5

    # 获取 PFT 特征数量
    if pft_dict:
        num_pft_features = len(next(iter(pft_dict.values())))
    else:
        num_pft_features = 0

    merged_dict = {}
    skipped_count = 0

    for fname, ehr_data in ehr_dict.items():
        # 提取 EHR 特征 (按顺序)
        ehr_features = [
            ehr_data['age'],
            ehr_data['gender'],
            ehr_data['weight'],
            ehr_data['height'],
            ehr_data['bmi']
        ]

        # 查找对应的 PFT 数据
        final_id = filename_to_final_id.get(fname, "")
        pft_features = pft_dict.get(final_id)

        if pft_features is None:
            if skip_missing_pft:
                skipped_count += 1
                continue  # 跳过缺失 PFT 数据的患者
            else:
                # 零填充
                pft_features = [0.0] * num_pft_features

        # 合并特征
        merged_dict[fname] = ehr_features + pft_features

    print(f"[特征合并] EHR: {num_ehr_features} 维, PFT: {num_pft_features} 维")
    print(f"[特征合并] 合并后: {len(merged_dict)} 条记录 (跳过 {skipped_count} 条缺失 PFT)")

    return merged_dict, num_ehr_features, num_pft_features


# =============================================================================
# 多标签分类支持 (新增)
# =============================================================================

def compute_co_occurrence_matrix(label_dict, label_mapping):
    """
    计算归一化共现矩阵

    归一化规则: co_matrix = M / (max(M) + ε)
    - M 为原始共现计数矩阵
    - 归一化到 [0, 1] 范围，保持全局相对关系

    Args:
        label_dict: {filename: [label1, label2, ...]} 多标签字典
        label_mapping: {label_name: index}

    Returns:
        co_matrix: [n_labels, n_labels] numpy数组, 归一化到 [0, 1]
    """
    n_labels = len(label_mapping)
    co_count = np.zeros((n_labels, n_labels), dtype=np.float32)

    # 统计共现次数
    for labels in label_dict.values():
        indices = [label_mapping[l] for l in labels if l in label_mapping]
        for i in indices:
            for j in indices:
                if i != j:  # 不计算自身
                    co_count[i, j] += 1

    # 全局最大值归一化: co_matrix = M / (max(M) + ε)
    # 保持全局相对关系，避免行归一化导致的信息损失
    max_val = co_count.max()
    epsilon = 1e-6
    if max_val > 0:
        co_matrix = co_count / (max_val + epsilon)
    else:
        co_matrix = co_count

    return co_matrix


def create_multilabel_encoder(label_mapping, n_labels):
    """
    创建多标签编码器

    Args:
        label_mapping: {label_name: index}
        n_labels: 标签数量

    Returns:
        encoder: 函数，将标签列表转换为 multi-hot 向量
    """
    def encode(label_list):
        """
        将标签列表转换为 multi-hot 向量

        Args:
            label_list: 标签名称列表

        Returns:
            multi_hot: [n_labels] numpy数组
        """
        multi_hot = np.zeros(n_labels, dtype=np.float32)
        for label in label_list:
            if label in label_mapping:
                multi_hot[label_mapping[label]] = 1.0
        return multi_hot

    return encode


def load_multilabel_labels(label_file=None,
                           header_row=1,
                           target_col_name="匹配的大类",
                           label_separator=";",
                           min_label_freq=50):
    """
    多标签解析函数

    目标列: "匹配的大类" (与单标签的 "匹配的第一大类" 不同)
    格式: 分号分隔的标签字符串，如 "慢性心力衰竭;冠心病"

    Args:
        label_file: 标签文件路径，None 时从配置文件读取
        header_row: 表头行号
        target_col_name: 目标列名 (默认 "匹配的大类")
        label_separator: 标签分隔符 (默认 ";")
        min_label_freq: 最小标签频次阈值

    Returns:
        label_dict: {filename: [label1, label2, ...]}
        label_mapping: {label_name: index}
        co_occurrence: np.ndarray 归一化共现矩阵
    """
    # 软编码：从配置文件读取默认路径
    if label_file is None:
        label_file = _get_default_label_file()

    try:
        df = pd.read_excel(label_file, engine='openpyxl', header=header_row)

        print(f"[多标签] 标签文件列数: {len(df.columns)}")
        print(f"[多标签] 数据行数: {len(df)}")
        print(f"[多标签] 正在查找目标标签列: {target_col_name}")

        # 1. 精准匹配目标标签列
        target_col = None
        clean_target = target_col_name.replace('\n', '').replace('\r', '').replace(' ', '')

        for col in df.columns:
            col_str = str(col).strip()
            clean_col_str = col_str.replace('\n', '').replace('\r', '').replace(' ', '')

            if clean_target == clean_col_str:
                target_col = col
                break

        if target_col is None:
            print(f"[多标签] 错误: 未在Excel中找到精准匹配 '{target_col_name}' 的列。")
            print(f"[多标签] 现有列名: {[str(c) for c in df.columns]}")
            return {}, {}, None

        print(f"[多标签] 成功定位标签列: {target_col}")

        # 2. 查找匹配的文件名列
        filename_col = None
        for col in df.columns:
            col_str = str(col)
            if '匹配' in col_str and 'Excel' in col_str and '文件' in col_str:
                filename_col = col
                break

        if filename_col is None:
            for col in df.columns:
                col_str = str(col).lower()
                if 'excel' in col_str or '文件' in col_str or 'file' in col_str:
                    filename_col = col
                    break

        if filename_col is None:
            print("[多标签] 警告: 未找到文件名列，将使用行索引作为标识")
        else:
            print(f"[多标签] 使用文件名列: {filename_col}")

        # 3. 提取多标签
        raw_labels = df[target_col].tolist()
        temp_labels = []  # List[List[str]]
        temp_filenames = []
        skipped_count = 0

        # 统计所有标签频次
        all_labels_counter = Counter()

        for i, label_value in enumerate(raw_labels):
            if pd.isna(label_value):
                skipped_count += 1
                continue

            label_str = str(label_value).strip()
            if label_str == '' or label_str == '-':
                skipped_count += 1
                continue

            # 拆分多标签
            labels = [l.strip() for l in label_str.split(label_separator) if l.strip()]
            if not labels:
                skipped_count += 1
                continue

            # 获取文件名
            fname_str = f"sample_{i}"
            if filename_col is not None:
                fname = df.iloc[i][filename_col]
                if not pd.isna(fname):
                    fname_str = str(fname).strip()
                    if fname_str.endswith('.xlsx'):
                        fname_str = fname_str[:-5]
                else:
                    fname_str = f"unknown_{i}"

            temp_labels.append(labels)
            temp_filenames.append(fname_str)

            # 统计标签频次
            all_labels_counter.update(labels)

        # 4. 过滤低频标签
        valid_labels_set = {label for label, count in all_labels_counter.items() if count >= min_label_freq}
        print(f"[多标签] 原始标签数: {len(all_labels_counter)}, 过滤后: {len(valid_labels_set)} (阈值: {min_label_freq})")

        # 5. 构建标签字典和映射
        labels_clean = []
        filenames = []
        for fname, labels in zip(temp_filenames, temp_labels):
            # 过滤低频标签
            filtered = [l for l in labels if l in valid_labels_set]
            if filtered:
                labels_clean.append(filtered)
                filenames.append(fname)
            else:
                skipped_count += 1

        # 创建映射 (按字母排序保证一致性)
        unique_labels = sorted(list(set(l for labels in labels_clean for l in labels)))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        label_dict = {fname: labels for fname, labels in zip(filenames, labels_clean)}

        # 6. 计算共现矩阵
        co_occurrence = compute_co_occurrence_matrix(label_dict, label_mapping)

        # --- 打印统计报告 ---
        print(f"\n[多标签] 清洗报告:")
        print(f" - 总样本数: {len(df)}")
        print(f" - 有效样本数: {len(labels_clean)}")
        print(f" - 剔除样本数: {skipped_count}")
        print(f" - 唯一标签数: {len(unique_labels)}")
        print(f"[多标签] 标签分布:")
        for label in unique_labels:
            count = sum(1 for labels in labels_clean if label in labels)
            print(f"   {label}: {count} 个样本")

        print(f"\n[多标签] 共现矩阵形状: {co_occurrence.shape}")

        return label_dict, label_mapping, co_occurrence

    except Exception as e:
        print(f"[多标签] 加载标签文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {}, None


def load_labels_unified(label_file=None,
                        header_row=1,
                        task_mode="single_label",
                        target_col_name="匹配的第一大类",
                        label_separator=";",
                        min_label_freq=50):
    """
    统一标签加载接口 - 根据 task_mode 自动选择单标签或多标签加载

    Args:
        label_file: 标签文件路径，None 时从配置文件读取
        header_row: 表头行号
        task_mode: "single_label" 或 "multi_label"
        target_col_name: 目标列名
        label_separator: 多标签分隔符
        min_label_freq: 最小标签频次阈值

    Returns:
        单标签模式: (label_dict, label_mapping, None)
        多标签模式: (label_dict, label_mapping, co_occurrence_matrix)
    """
    # 软编码：从配置文件读取默认路径
    if label_file is None:
        label_file = _get_default_label_file()

    if task_mode == "multi_label":
        # 多标签模式使用 "匹配的大类" 列
        return load_multilabel_labels(
            label_file=label_file,
            header_row=header_row,
            target_col_name="匹配的大类",  # 固定使用多标签列
            label_separator=label_separator,
            min_label_freq=min_label_freq
        )
    else:
        # 单标签模式
        label_dict, label_mapping = load_labels(
            label_file=label_file,
            header_row=header_row,
            target_col_name=target_col_name,
            min_label_freq=min_label_freq
        )
        return label_dict, label_mapping, None


if __name__ == "__main__":
    print("="*80)
    print("测试标签提取")
    print("="*80)

    # 示例1：使用默认的 "匹配的第一大类"
    print("\n>>> 测试 1: 默认列 (匹配的第一大类)")
    label_dict, label_mapping = load_labels() # 默认参数
    
    # 示例2：切换到 "心率储备" (如果在你的Excel里有这列的话)
    # 你可以在这里修改 target_col_name 来测试其他列
    print("\n>>> 测试 2: 切换到 '心率储备'")
    # 注意：如果你的Excel没有这一列，这里会报错或提示找不到，这只是演示用法
    label_dict_hr, label_mapping_hr = load_labels(target_col_name="心率储备")

    if label_dict_hr:
        print(f"\n'心率储备' 前5个样本:")
        for i, (fname, label) in enumerate(list(label_dict_hr.items())[:5]):
            print(f"  {fname}: {label}")
