"""
特征映射和选择模块
定义特征列表、分组索引以及基于生理拓扑的邻接矩阵构建

【重构版】所有硬编码集中在 FEATURE_CONFIG，其他函数自动派生
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# 核心配置：唯一硬编码位置
# =============================================================================

FEATURE_CONFIG = {
    # =========================================================================
    # 1. 特征定义
    # =========================================================================
    'features': {
        # 原始特征 (22个，固定不变，按数据读取顺序)
        'raw': [
            'MET',      # 0: 代谢当量
            'Load',     # 1: 运动负荷（W）
            'RER',      # 2: 呼吸交换率
            'HR',       # 3: 心率
            'HRR',      # 4: 心率储备
            'dH/dO2',   # 5: 心率与摄氧量斜率
            'SVc',      # 6: 每搏输出量
            'Psys',     # 7: 收缩压
            'Pdia',     # 8: 舒张压
            'SpO2',     # 9: 血氧饱和度
            "V'O2",     # 10: 摄氧量
            'VO2/kg',   # 11: 相对摄氧量
            'dO2/dW',   # 12: 摄氧效率斜率
            'BF',       # 13: 呼吸频率
            "V'E",      # 14: 每分通气量
            'BR',       # 15: 呼吸储备
            'EqO2',     # 16: 氧通气当量
            'EqCO2',    # 17: 二氧化碳通气当量
            'PETO2',    # 18: 呼气末氧分压
            'PETCO2',   # 19: 呼气末二氧化碳分压
            'VDc/VT',   # 20: 死腔量/潮气量比
            'VTex',     # 21: 呼气潮气量
        ],
        # 基础衍生特征 (4个，始终启用)
        'base_derived': [
            'PP',       # 脉压差 = Psys - Pdia
            'OUES',     # 氧摄取效率 = VO2 / log10(VE)
            'EqO2_COP', # 心肺最佳点 = EqO2滑动窗口最小值
            'HR_diff',  # 心率变化率 = d(HR)/dt
        ],
        # 可选衍生特征 (按功能模块分组)
        'optional': {
            'o2pulse': {
                'enabled_by_default': False,
                'features': [
                    'O2Pulse',      # 氧脉搏 = VO2 / HR (SV代理指标)
                    'd(O2P)/dt',    # 氧脉搏一阶导数 (变化速率)
                    'd2(O2P)/dt2',  # 氧脉搏二阶导数 (加速度)
                ],
                'description': '氧脉搏导数特征，鉴别泵衰竭与缺血',
            },
            'vco2': {
                'enabled_by_default': False,
                'features': ["V'CO2"],
                'description': "V'CO2 用于九图通气驱动分析 (P4, P5)",
            },
        },
    },

    # =========================================================================
    # 2. 分组定义 (按模式分类)
    # =========================================================================
    'groups': {
        # 'medical' 模式：按医学系统划分 (推荐)
        'medical': {
            'description': '按医学系统划分 (运动负荷、心血管、肺通气、气体交换)',
            'subgroups': {
                'G0_运动负荷与能量代谢': [
                    'Load', 'MET', "V'O2", 'VO2/kg', 'dO2/dW', 'OUES',
                ],
                'G1_心血管系统': [
                    'HR', 'HRR', 'Psys', 'Pdia', 'SVc', 'PP', 'HR_diff',
                    # 氧脉搏导数特征 (可选，启用时自动追加)
                    {'optional': 'o2pulse', 'features': ['O2Pulse', 'd(O2P)/dt', 'd2(O2P)/dt2']},
                ],
                'G2_肺通气': [
                    "V'E", 'BF', 'VTex', 'BR',
                ],
                'G3_气体交换': [
                    'RER', 'SpO2', 'EqO2', 'EqCO2', 'EqO2_COP',
                ],
            },
        },

        # 'full' 模式：包含所有特征的全量分组
        'full': {
            'description': '全量特征分组，包含PETCO2等更多指标',
            'subgroups': {
                'G0_运动负荷与能量代谢': [
                    'MET', 'Load', "V'O2", 'VO2/kg', 'dO2/dW', 'OUES',
                ],
                'G1_循环系统': [
                    'HR', 'HRR', 'dH/dO2', 'SVc', 'Psys', 'Pdia', 'PP', 'HR_diff',
                    {'optional': 'o2pulse', 'features': ['O2Pulse', 'd(O2P)/dt', 'd2(O2P)/dt2']},
                ],
                'G2_呼吸动力学': [
                    'BF', "V'E", 'BR', 'VTex',
                ],
                'G3_气体交换与效率': [
                    'RER', 'SpO2', 'EqO2', 'EqCO2', 'PETO2', 'PETCO2', 'VDc/VT', 'EqO2_COP',
                ],
            },
        },

        # 'select' 模式：精简特征选择
        'select': {
            'description': '精简特征分组，去除冗余指标',
            'subgroups': {
                'G0_能量代谢': [
                    'MET', "V'O2", 'VO2/kg', 'Load', 'OUES',
                ],
                'G1_心肺驱动': [
                    'HR', "V'E", 'BF', 'VTex', 'PP', 'HR_diff',
                    {'optional': 'o2pulse', 'features': ['O2Pulse', 'd(O2P)/dt', 'd2(O2P)/dt2']},
                ],
                'G2_换气效率': [
                    'RER', 'EqO2', 'EqCO2', 'SpO2', 'EqO2_COP',
                ],
            },
        },

        # 'nine_graph' 模式：基于 Wasserman 九图的生理耦合模式
        'nine_graph': {
            'description': '基于 Wasserman 九图的生理耦合模式',
            'requires_vco2': True,
            'subgroups': {
                'S0_oxygen_delivery': [  # 氧输送链 (P1, P2, P3)
                    'Load', "V'O2", 'HR', 'O2Pulse', 'dO2/dW', 'MET', 'd(O2P)/dt', 'VO2/kg',
                    'dH/dO2', 'SVc', 'OUES',  # 补充: 心率-摄氧量斜率、每搏输出量、氧摄取效率
                ],
                'S1_ventilation_drive': [  # 通气驱动控制 (P4, P5)
                    "V'CO2", "V'E", 'VTex', 'BF', 'RER', 'BR',
                ],
                'S2_vq_matching': [  # 换气效率与气血耦合 (P6, P7)
                    'EqO2', 'EqCO2', 'PETO2', 'PETCO2', 'SpO2', 'EqO2_COP',
                    'VDc/VT',  # 补充: 死腔量/潮气量比
                ],
                'S3_stability_reserve': [  # 储备与稳定性 (P8, P9)
                    'Psys', 'Pdia', 'PP', 'HRR', 'HR_diff', 'd2(O2P)/dt2',
                ],
            },
        },
    },

    # =========================================================================
    # 3. 边连接定义 (邻接矩阵拓扑)
    # =========================================================================
    'edges': {
        # 基础边连接 (始终存在)
        'base': [
            # --- G0: 代谢系统内部 ---
            ('Load', "V'O2"),
            ("V'O2", 'MET'),
            ("V'O2", 'VO2/kg'),
            ("V'O2", 'dO2/dW'),
            ("V'O2", 'OUES'),

            # --- G1: 循环系统内部 ---
            ('HR', 'HRR'),
            ('HR', 'dH/dO2'),
            ('HR', 'SVc'),
            ('Psys', 'Pdia'),
            ('Psys', 'PP'),
            ('Pdia', 'PP'),
            ('HR', 'HR_diff'),

            # --- G2: 通气系统内部 ---
            ("V'E", 'BF'),
            ("V'E", 'VTex'),
            ("V'E", 'BR'),
            ("V'E", 'OUES'),

            # --- G3: 气体交换系统内部 ---
            ('EqO2', 'SpO2'),
            ('EqCO2', 'PETCO2'),
            ('EqO2', "V'E"),
            ('VTex', 'VDc/VT'),
            ('EqO2', 'EqO2_COP'),

            # --- 跨系统耦合 ---
            ("V'O2", 'HR'),        # 代谢 <-> 循环
            ("V'O2", "V'E"),       # 代谢 <-> 通气
            ('RER', "V'O2"),       # 呼吸交换率 <-> 代谢
            ('HR', 'SpO2'),        # 循环 <-> 气体交换
            ("V'E", 'EqO2'),       # 通气 <-> 气体交换
            ("V'E", 'EqCO2'),      # 通气 <-> 气体交换
        ],

        # 可选边连接 (按功能模块分组)
        'optional': {
            'o2pulse': [
                # 氧脉搏导数链
                ("V'O2", 'O2Pulse'),
                ('HR', 'O2Pulse'),
                ('O2Pulse', 'd(O2P)/dt'),
                ('d(O2P)/dt', 'd2(O2P)/dt2'),
                ('O2Pulse', 'SVc'),
                ('d2(O2P)/dt2', 'PP'),
            ],
            'nine_graph': [
                # S0: 氧输送链内部
                ('Load', "V'O2"),           # 代谢转换起点
                ("V'O2", 'HR'),             # 心血管响应 (九图 P2)
                ("V'O2", 'O2Pulse'),        # 核心循环连接 (九图 P3)
                ('O2Pulse', 'd(O2P)/dt'),   # 平台期捕捉
                # S1: 通气驱动控制内部
                ("V'E", "V'CO2"),           # 通气效率斜率 (九图 P5)
                ("V'E", 'VTex'),            # 呼吸模式
                ("V'CO2", 'RER'),           # 呼吸交换率
                # S2: 换气效率内部
                ('EqCO2', 'PETCO2'),        # 死腔通气检测 (九图 P6)
                ('EqO2', 'SpO2'),           # 氧合状态
                # S3: 储备与稳定性内部
                ('Psys', 'PP'), ('Pdia', 'PP'),
                ('HR', 'HRR'), ('HR', 'HR_diff'),
                # 核心耦合 (跨子系统)
                ("V'O2", "V'E"),            # 代谢-通气耦合
                ("V'O2", "V'CO2"),          # 代谢-代谢产物耦合
                ("V'E", 'EqO2'), ("V'E", 'EqCO2'),
                ('PP', 'O2Pulse'),          # SV 双代理
            ],
        },
    },

    # =========================================================================
    # 4. 注意力权重配置 (用于 PriorMaskedGlobalTransformer)
    # =========================================================================
    'attention_weights': {
        'nine_graph': {
            'core_slopes': {
                ("V'O2", 'HR'): 2.0,          # V'O2-HR 斜率
                ("V'O2", 'O2Pulse'): 2.5,     # V'O2-O2Pulse 斜率
                ("V'E", "V'CO2"): 2.0,        # V'E-V'CO2 斜率
                ('EqCO2', 'PETCO2'): 1.8,     # EqCO2-PETCO2 关系
            },
            'default': 1.0,
        },
    },
}


# =============================================================================
# 派生函数：从 FEATURE_CONFIG 自动生成
# =============================================================================

def get_feature_list(optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> Tuple[List[str], int]:
    """
    根据配置生成完整特征列表

    Args:
        optional_keys: 启用的可选特征模块列表，如 ['o2pulse']
                       None 表示使用默认配置
        base_enabled: 是否启用基础衍生特征 (PP, OUES, EqO2_COP, HR_diff)
                      默认 True (保持向后兼容)

    Returns:
        features: 特征名称列表
        n_features: 特征数量
    """
    if optional_keys is None:
        # 使用默认配置
        optional_keys = [
            k for k, v in FEATURE_CONFIG['features']['optional'].items()
            if v.get('enabled_by_default', False)
        ]

    features = FEATURE_CONFIG['features']['raw'].copy()
    if base_enabled:  # [修改] 仅当启用时添加基础衍生特征
        features.extend(FEATURE_CONFIG['features']['base_derived'])

    for key in optional_keys:
        if key in FEATURE_CONFIG['features']['optional']:
            features.extend(FEATURE_CONFIG['features']['optional'][key]['features'])

    return features, len(features)


def get_feature_indices(optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> Dict[str, int]:
    """
    根据配置生成特征索引映射

    Args:
        optional_keys: 启用的可选特征模块列表
        base_enabled: 是否启用基础衍生特征

    Returns:
        IDX: {特征名: 索引} 字典
    """
    features, _ = get_feature_list(optional_keys, base_enabled=base_enabled)
    return {name: i for i, name in enumerate(features)}


def get_channel_groups(mode: str = 'medical', optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> Dict[str, List[List[int]]]:
    """
    根据配置返回特征分组索引

    Args:
        mode: 分组模式 ('medical', 'full', 'select', 'nine_graph')
        optional_keys: 启用的可选特征模块列表
        base_enabled: 是否启用基础衍生特征

    Returns:
        channel_groups: {模式名: [[索引列表], ...]} 分组索引字典
    """
    if mode not in FEATURE_CONFIG['groups']:
        raise ValueError(f"Unknown mode: {mode}. Supported: {list(FEATURE_CONFIG['groups'].keys())}")

    IDX = get_feature_indices(optional_keys, base_enabled=base_enabled)
    group_def = FEATURE_CONFIG['groups'][mode]['subgroups']

    groups = []
    for group_name, feature_list in group_def.items():
        indices = []
        for item in feature_list:
            if isinstance(item, dict):
                # 可选特征块
                opt_key = item['optional']
                if optional_keys and opt_key in optional_keys:
                    for f in item['features']:
                        if f in IDX:
                            indices.append(IDX[f])
            else:
                # 普通特征
                if item in IDX:
                    indices.append(IDX[item])
        groups.append(indices)

    # 返回所有模式的分组（用于向后兼容）
    result = {}
    for m in FEATURE_CONFIG['groups'].keys():
        if m == mode:
            result[m] = groups
        else:
            # 为其他模式也计算分组
            result[m] = _compute_groups_for_mode(m, optional_keys, base_enabled=base_enabled)

    return result


def _compute_groups_for_mode(mode: str, optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> List[List[int]]:
    """计算指定模式的分组索引"""
    IDX = get_feature_indices(optional_keys, base_enabled=base_enabled)
    group_def = FEATURE_CONFIG['groups'][mode]['subgroups']

    groups = []
    for group_name, feature_list in group_def.items():
        indices = []
        for item in feature_list:
            if isinstance(item, dict):
                opt_key = item['optional']
                if optional_keys and opt_key in optional_keys:
                    for f in item['features']:
                        if f in IDX:
                            indices.append(IDX[f])
            else:
                if item in IDX:
                    indices.append(IDX[item])
        groups.append(indices)

    return groups


def create_adjacency_matrix(mode: str = 'medical', optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> np.ndarray:
    """
    创建基于生理机制的二值化稀疏邻接矩阵

    Args:
        mode: 分组模式 ('medical', 'full', 'select')
        optional_keys: 启用的可选特征模块列表
        base_enabled: 是否启用基础衍生特征

    Returns:
        adj: [n_features, n_features] 邻接矩阵 (0.0 或 1.0)
    """
    features, n_total = get_feature_list(optional_keys, base_enabled=base_enabled)
    IDX = get_feature_indices(optional_keys, base_enabled=base_enabled)

    # 初始化矩阵 (自连接)
    adj = np.eye(n_total)

    # 添加基础边
    for f1, f2 in FEATURE_CONFIG['edges']['base']:
        if f1 in IDX and f2 in IDX:
            i, j = IDX[f1], IDX[f2]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # 添加可选边
    if optional_keys:
        for key in optional_keys:
            if key in FEATURE_CONFIG['edges']['optional']:
                for f1, f2 in FEATURE_CONFIG['edges']['optional'][key]:
                    if f1 in IDX and f2 in IDX:
                        i, j = IDX[f1], IDX[f2]
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

    # 根据模式切片
    if mode == 'full' and len(features) == n_total:
        # full 模式且使用全部特征，直接返回
        all_modes = get_channel_groups(mode, optional_keys, base_enabled=base_enabled)
        if 'full' in all_modes:
            indices = [idx for group in all_modes['full'] for idx in group]
            if len(indices) == n_total:
                return adj

    # 切片到指定模式的特征子集
    channel_groups = get_channel_groups(mode, optional_keys, base_enabled=base_enabled)
    indices = [idx for group in channel_groups[mode] for idx in group]
    sub_adj = adj[np.ix_(indices, indices)]

    return sub_adj


def get_semantic_adj(mode: str = 'medical', optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> np.ndarray:
    """
    获取语义邻接矩阵 (用于模型初始化)

    Args:
        mode: 分组模式
        optional_keys: 启用的可选特征模块列表
        base_enabled: 是否启用基础衍生特征

    Returns:
        semantic_adj: 标准化后的邻接矩阵
    """
    adj = create_adjacency_matrix(mode, optional_keys, base_enabled=base_enabled)

    # 对称归一化: D^{-1/2} A D^{-1/2}
    degree = np.sum(adj, axis=1)
    degree[degree == 0] = 1.0
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return normalized_adj.astype(np.float32)


def get_group_info(mode: str = 'medical', optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> Dict[str, Any]:
    """
    获取分组详细信息 (用于日志和调试)

    Args:
        mode: 分组模式
        optional_keys: 启用的可选特征模块列表
        base_enabled: 是否启用基础衍生特征

    Returns:
        info: 包含分组名称、特征列表、特征数量的字典
    """
    IDX = get_feature_indices(optional_keys, base_enabled=base_enabled)
    group_def = FEATURE_CONFIG['groups'][mode]['subgroups']

    info = {
        'mode': mode,
        'optional_keys': optional_keys or [],
        'base_enabled': base_enabled,  # [新增]
        'description': FEATURE_CONFIG['groups'][mode]['description'],
        'groups': [],
        'total_features': len(IDX),
    }

    for group_name, feature_list in group_def.items():
        group_features = []
        for item in feature_list:
            if isinstance(item, dict):
                opt_key = item['optional']
                if optional_keys and opt_key in optional_keys:
                    group_features.extend(item['features'])
            else:
                group_features.append(item)

        info['groups'].append({
            'name': group_name,
            'features': group_features,
            'n_features': len(group_features),
        })

    return info


def get_nine_graph_config(optional_keys: Optional[List[str]] = None, base_enabled: bool = True) -> Dict[str, Any]:
    """
    获取九图模式的完整配置，自动启用 V'CO2 和 o2pulse

    Args:
        optional_keys: 额外的可选特征模块列表
        base_enabled: 是否启用基础衍生特征

    Returns:
        dict: {
            'groups': 分组索引字典,
            'adjacency': 邻接矩阵,
            'attention_weights': 注意力权重矩阵 [N, N],
            'feature_names': 特征名称列表,
            'num_features': 特征数量,
            'optional_keys': 实际使用的可选特征列表,
            'base_enabled': 基础衍生特征开关,
        }
    """
    if optional_keys is None:
        optional_keys = []

    # 自动启用 vco2 和 o2pulse
    if 'vco2' not in optional_keys:
        optional_keys = optional_keys + ['vco2']
    if 'o2pulse' not in optional_keys:
        optional_keys = optional_keys + ['o2pulse']

    # 获取特征列表和索引
    feature_names, num_features = get_feature_list(optional_keys, base_enabled=base_enabled)
    IDX = get_feature_indices(optional_keys, base_enabled=base_enabled)

    # 获取九图分组
    channel_groups = get_channel_groups('nine_graph', optional_keys, base_enabled=base_enabled)

    # 构建邻接矩阵
    adj = create_adjacency_matrix('nine_graph', optional_keys, base_enabled=base_enabled)

    # 构建注意力权重矩阵
    attention_weights = np.ones((num_features, num_features), dtype=np.float32)
    if 'nine_graph' in FEATURE_CONFIG.get('attention_weights', {}):
        core_slopes = FEATURE_CONFIG['attention_weights']['nine_graph'].get('core_slopes', {})
        default = FEATURE_CONFIG['attention_weights']['nine_graph'].get('default', 1.0)

        for (f1, f2), weight in core_slopes.items():
            if f1 in IDX and f2 in IDX:
                i, j = IDX[f1], IDX[f2]
                attention_weights[i, j] = weight
                attention_weights[j, i] = weight  # 对称

    return {
        'groups': channel_groups,
        'adjacency': adj,
        'attention_weights': attention_weights,
        'feature_names': feature_names,
        'num_features': num_features,
        'optional_keys': optional_keys,
        'base_enabled': base_enabled,  # [新增]
    }


def get_nine_graph_adjacency(
    optional_keys: Optional[List[str]] = None,
    base_enabled: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    获取九图模式的邻接矩阵 (用于 MTL 模型)

    自动启用 V'CO2 和氧脉搏导数特征。

    Args:
        optional_keys: 额外的可选特征模块列表
        base_enabled: 是否启用基础衍生特征
        normalize: 是否进行对称归一化 (D^{-1/2} A D^{-1/2})

    Returns:
        adj: [num_features, num_features] 邻接矩阵
    """
    if optional_keys is None:
        optional_keys = []

    # 自动启用 vco2 和 o2pulse
    if 'vco2' not in optional_keys:
        optional_keys = optional_keys + ['vco2']
    if 'o2pulse' not in optional_keys:
        optional_keys = optional_keys + ['o2pulse']

    if normalize:
        return get_semantic_adj('nine_graph', optional_keys, base_enabled=base_enabled)
    else:
        return create_adjacency_matrix('nine_graph', optional_keys, base_enabled=base_enabled)


# =============================================================================
# 向后兼容：默认导出 (不启用可选特征)
# =============================================================================

NEW_FEATURES, NUM_FEATURES = get_feature_list(optional_keys=None)
IDX = get_feature_indices(optional_keys=None)
MODE_INDICES = {k: [idx for group in v for idx in group]
                for k, v in get_channel_groups(mode='medical', optional_keys=None).items()}


# =============================================================================
# 向后兼容：旧版函数签名 (o2pulse_enabled 参数)
# =============================================================================

def _o2pulse_to_optional_keys(o2pulse_enabled: bool) -> Optional[List[str]]:
    """将旧的 o2pulse_enabled 参数转换为新的 optional_keys 格式"""
    return ['o2pulse'] if o2pulse_enabled else None


# 旧版函数签名兼容
def get_feature_list_legacy(o2pulse_enabled: bool = False) -> Tuple[List[str], int]:
    """向后兼容: 旧版 get_feature_list(o2pulse_enabled)"""
    return get_feature_list(_o2pulse_to_optional_keys(o2pulse_enabled))


def get_feature_indices_legacy(o2pulse_enabled: bool = False) -> Dict[str, int]:
    """向后兼容: 旧版 get_feature_indices(o2pulse_enabled)"""
    return get_feature_indices(_o2pulse_to_optional_keys(o2pulse_enabled))


def get_channel_groups_legacy(o2pulse_enabled: bool = False) -> Dict[str, List[List[int]]]:
    """向后兼容: 旧版 get_channel_groups(o2pulse_enabled)，默认使用 medical 模式"""
    return get_channel_groups(mode='medical', optional_keys=_o2pulse_to_optional_keys(o2pulse_enabled))


def create_adjacency_matrix_legacy(mode: str = 'full', o2pulse_enabled: bool = False) -> np.ndarray:
    """向后兼容: 旧版 create_adjacency_matrix(mode, o2pulse_enabled)"""
    return create_adjacency_matrix(mode, _o2pulse_to_optional_keys(o2pulse_enabled))


# =============================================================================
# 标准化函数
# =============================================================================

def normalize_features(data, method='robust', feature_stats=None):
    """
    特征标准化 (修改版: 增加 Robust Scaler 和 强制 Clip)
    """
    original_shape = data.shape
    data = data.reshape(-1, original_shape[-1])

    if method == 'z-score':
        if feature_stats is None:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1
            stats = {'mean': mean, 'std': std}
        else:
            mean = feature_stats['mean']
            std = feature_stats['std']
            stats = feature_stats
        normalized = (data - mean) / std

    elif method == 'robust':
        if feature_stats is None:
            median = np.median(data, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            stats = {'median': median, 'q75': q75, 'q25': q25, 'iqr': iqr}
        else:
            median = feature_stats['median']
            q75 = feature_stats['q75']
            q25 = feature_stats['q25']
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            stats = feature_stats

        normalized = (data - median) / iqr

    elif method == 'min-max':
        if feature_stats is None:
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            stats = {'min': min_val, 'max': max_val, 'range': range_val}
        else:
            min_val = feature_stats['min']
            range_val = feature_stats['range']
            stats = feature_stats
        normalized = (data - min_val) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # 强制数值截断
    normalized = np.clip(normalized, -5.0, 5.0)

    return normalized.reshape(original_shape), stats


def get_feature_statistics(data_list):
    """
    计算特征的统计量（用于标准化）
    """
    all_data = np.concatenate([d.reshape(-1, d.shape[-1]) for d in data_list], axis=0)
    all_data = all_data[np.isfinite(all_data).all(axis=1)]

    stats = {
        'mean': np.mean(all_data, axis=0),
        'std': np.std(all_data, axis=0),
        'min': np.min(all_data, axis=0),
        'max': np.max(all_data, axis=0),
        'median': np.median(all_data, axis=0),
        'q25': np.percentile(all_data, 25, axis=0),
        'q75': np.percentile(all_data, 75, axis=0)
    }
    stats['std'] = np.nan_to_num(stats['std'], nan=1.0)
    stats['std'][stats['std'] == 0] = 1.0

    return stats


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试重构后的特征配置系统")
    print("=" * 70)

    # 测试1: 不同配置的特征列表
    print("\n【测试1】特征列表生成")
    for opt in [None, ['o2pulse']]:
        features, n = get_feature_list(opt)
        print(f"  可选特征: {opt} → 特征数: {n}")

    # 测试2: 不同模式的分组
    print("\n【测试2】分组模式")
    for mode in ['medical', 'full', 'select']:
        groups = get_channel_groups(mode, ['o2pulse'])
        total = sum(len(g) for g in groups[mode])
        print(f"  {mode}: {len(groups[mode])} 个子组, 共 {total} 个特征")

    # 测试3: 邻接矩阵
    print("\n【测试3】邻接矩阵")
    for mode in ['medical', 'full']:
        adj = create_adjacency_matrix(mode, ['o2pulse'])
        n_edges = (np.sum(adj) - adj.shape[0]) // 2
        print(f"  {mode}: 形状 {adj.shape}, 边数 {n_edges}")

    # 测试4: 分组信息
    print("\n【测试4】分组详情 (medical模式)")
    info = get_group_info('medical', ['o2pulse'])
    print(f"  描述: {info['description']}")
    for g in info['groups']:
        print(f"  - {g['name']}: {g['n_features']} 个特征")