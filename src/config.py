"""
统一配置管理模块
================

设计原则:
1. 所有参数均从 config.yaml 加载，不接受任何硬编码
2. 命令行参数可覆盖配置文件
3. 提供类型安全的配置访问接口

使用方法:
    from config import Config
    config = Config.load()
    print(config.model.name)  # 访问模型配置
"""

import os
import sys
import yaml
import argparse
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path

# 添加src到路径
sys.path.insert(0, os.path.dirname(__file__))

from feature_mapping import NEW_FEATURES, MODE_INDICES, get_feature_list, get_feature_indices, get_channel_groups, _o2pulse_to_optional_keys
from label_extractor import AVAILABLE_LABEL_COLUMNS


# =============================================================================
# 配置数据类 (类型安全)
# =============================================================================

@dataclass
class HoldoutConfig:
    """Holdout 测试集配置 (用于 K-Fold 前的独立测试集划分)"""
    enabled: bool = False           # 是否启用独立测试集划分
    test_ratio: float = 0.2         # 独立测试集比例
    random_seed: int = 42           # 全局切分随机种子
    save_split: bool = True         # 是否保存划分结果


@dataclass
class ExperimentConfig:
    """实验配置 (用于消融实验文件命名)"""
    suffix: str = ""                # 实验后缀，如: "v1", "baseline", "ablation_a"


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str
    label_file: str
    output_root: str
    use_mini: bool
    test_ratio: float
    L_win: int
    max_length: int
    stride: int
    # [新增] Holdout 测试集配置
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)


@dataclass
class O2PulseDerivativesConfig:
    """氧脉搏导数特征配置 (可插拔)"""
    enabled: bool = False
    features: List[str] = field(default_factory=lambda: ["O2Pulse", "d(O2P)/dt", "d2(O2P)/dt2"])


@dataclass
class Vco2Config:
    """V'CO2 特征配置 (九图模式)"""
    enabled: bool = False


@dataclass
class DerivedFeaturesConfig:
    """衍生特征配置"""
    base_enabled: bool = True  # [新增] 基础衍生特征开关
    base: List[str] = field(default_factory=lambda: ["PP", "OUES", "EqO2_COP", "HR_diff"])
    o2pulse_derivatives: O2PulseDerivativesConfig = field(default_factory=O2PulseDerivativesConfig)
    vco2: Vco2Config = field(default_factory=Vco2Config)  # [新增] V'CO2 配置


@dataclass
class FeaturesConfig:
    """特征配置"""
    adapt_mode: Literal["full", "select", "medical", "nine_graph"]
    target_col_name: str
    num_channels: int
    channel_groups: List[List[int]]
    channels: List[int] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    # [新增] 可插拔衍生特征配置
    derived_features: DerivedFeaturesConfig = field(default_factory=DerivedFeaturesConfig)
    base_enabled: bool = True        # [新增] 基础衍生特征开关 (便捷访问属性)
    o2pulse_enabled: bool = False    # 氧脉搏导数特征开关 (便捷访问属性)
    vco2_enabled: bool = False       # [新增] V'CO2 特征开关 (九图模式)


@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    type: str
    factor: float
    patience: int
    min_lr: float


@dataclass
class EarlyStoppingConfig:
    """早停配置"""
    patience: int
    min_delta: float


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    gradient_clip: float
    random_seed: int
    save_best_metric: str = "macro_f1"  # 最佳模型保存指标: macro_f1, auprc, loss
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None
    early_stopping: EarlyStoppingConfig = None


@dataclass
@dataclass
class BinaryLossConfig:
    """二分类损失配置 (BCEWithLogitsLoss)"""
    enabled: bool = False           # 强制启用二分类模式 (false=自动检测)
    auto_detect: bool = True        # 自动检测二分类场景 (n_classes==2)
    pos_weight: Any = "auto"        # 正类权重: auto(自动计算), 或手动数值如 5.0


@dataclass
class LossConfig:
    """损失函数配置"""
    type: str
    gamma: float
    alpha: Any  # auto, balanced, 或列表
    # 对比学习损失参数
    supcon_weight: float = 0.3
    temperature: float = 0.07
    # [新增] LCRLoss 参数 (多标签分类)
    lcr_enabled: bool = False
    lcr_lambda_co: float = 0.1
    lcr_epsilon: float = 1e-6
    # [新增] 二分类配置
    binary: BinaryLossConfig = field(default_factory=BinaryLossConfig)
    # [新增] Dice Loss 参数
    dice_smooth: float = 1.0         # Dice 平滑系数
    # [新增] LDAM Loss 参数
    ldam_max_m: float = 0.5          # 最大 margin
    ldam_scale_s: float = 30         # scale 参数


@dataclass
class TaskConfig:
    """任务模式配置 (单标签/多标签)"""
    mode: Literal["single_label", "multi_label"] = "single_label"
    label_separator: str = ";"              # 多标签分隔符
    min_label_freq: int = 50                # 最小标签频次阈值
    target_col_name: str = "匹配的第一大类"  # 单标签模式下的默认列


@dataclass
class StaticFeatureConfig:
    """静态特征配置 (EHR + PFT 多模态融合)"""
    enabled: bool = False
    static_dim: int = 16
    num_features: int = 5  # 动态计算
    ablation: str = "full"  # full, static_only, cpet_only
    features: List[str] = field(default_factory=lambda: ["age", "gender", "weight", "height", "bmi"])

    # PFT 配置 (新增)
    pft_enabled: bool = False
    pft_file: str = "xx_path"
    skip_missing_pft: bool = True
    pft_fill_method: str = "none"  # none=跳过, zero=零填充
    pft_features: List[str] = field(default_factory=list)


@dataclass
class TemporalEncoderConfig:
    """时序编码器配置"""
    type: str = "gru"  # gru (现有) 或 cnn (新增)
    T_mid: int = 24    # CNN 中间时序维度

    # === 可插拔模块配置 ===
    use_multiscale: bool = False       # 是否启用多尺度卷积
    use_residual: bool = False         # 是否启用残差连接
    multiscale_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    block1_kernel: int = 7             # Stage 1 卷积核大小
    block2_kernel: int = 5             # Stage 2 卷积核大小

    # === [新增] Masked Convolution 消融配置 ===
    use_masked_conv: bool = True       # 默认使用掩膜 (true=屏蔽padding)

    # === [新增] 时序提纯消融配置 ===
    # ablation 模式说明:
    #   - "full":        Baseline (Stage1+Stage2+Stage3+TimeAttn) 渐进降维 + 时间注意力
    #   - "merged_conv_global_pool": 合并 Conv + 全局池化 (移除渐进降维和时间注意力)
    ablation: str = "full"             # full | merged_conv_global_pool


@dataclass
class PriorGateConfig:
    """先验门控配置 (PriorWarmStart)"""
    gamma_init: float = 1.0    # 热启动初始值 (vs 0.0 冷启动)
    gamma_min: float = 0.1     # 先验保护下限
    gamma_lr_scale: float = 0.3  # gamma 学习率缩放


@dataclass
class ChannelAttentionConfig:
    """通道注意力配置"""
    enabled: bool = False           # 是否启用通道注意力
    init_value: float = 1.0         # 初始权重值


@dataclass
class FeatureAblationConfig:
    """特征选择消融配置"""
    enabled: bool = False           # 是否启用特征选择消融
    mode: str = "full"              # full, remove_weak, strong_only, remove_group
    weak_channels: List[int] = field(default_factory=list)      # 弱特征索引
    strong_channels: List[int] = field(default_factory=list)    # 强特征索引
    remove_group: str = ""          # 要移除的组名 (G0, G1, G2, G3)


@dataclass
class FlattenMLPConfig:
    """
    Flatten MLP 配置 (仅 flatten_only 模式生效)

    用于消融实验: w/o Stage2 跨变量交互
    移除 PriorMaskedGlobalTransformer，直接展平时序编码输出

    Args:
        hidden_dim: 中间层维度，None 表示直接 480→48 (单层 MLP)
        use_two_layer: 是否使用两层 MLP (480→hidden_dim→48)
        dropout: Dropout 率
        use_layer_norm: 是否使用 LayerNorm (与现有架构一致)
    """
    hidden_dim: Optional[int] = None     # 中间层维度，None 表示直接 480→48
    use_two_layer: bool = False          # 是否使用两层 MLP
    dropout: float = 0.3                 # Dropout 率
    use_layer_norm: bool = True          # 是否使用 LayerNorm


@dataclass
class PoolingOnlyConfig:
    """
    Pooling_only 模式配置 (仅 pooling_only 模式生效)

    用于消融实验: 用全局池化替代 Flatten，验证通道级聚合是否足够

    Args:
        pooling_type: 池化类型 ("avg" 或 "max")
        mlp_layers: MLP 层数 (1 或 2)
        hidden_dim: 中间层维度 (仅 mlp_layers=2 时生效)
        dropout: Dropout 率
        use_layer_norm: 是否使用 LayerNorm
    """
    pooling_type: str = "avg"           # "avg" 或 "max"
    mlp_layers: int = 1                 # 1 或 2
    hidden_dim: Optional[int] = None    # 仅 mlp_layers=2 时生效
    dropout: float = 0.3
    use_layer_norm: bool = True


@dataclass
class ModelConfig:
    """单个模型配置"""
    name: str
    hidden_dim: Optional[int]
    dropout: float
    # HDSTGCN 专用
    D_time: Optional[int] = None
    use_variable_length: Optional[bool] = None
    graph_ablation: Optional[str] = None
    # 静态特征融合
    static_features: Optional[StaticFeatureConfig] = None
    # 时序编码器配置
    temporal_encoder: Optional[TemporalEncoderConfig] = None
    # 先验门控配置
    prior_gate: Optional[PriorGateConfig] = None
    # 通道注意力配置
    channel_attention: Optional[ChannelAttentionConfig] = None
    # 特征选择消融配置
    feature_ablation: Optional[FeatureAblationConfig] = None
    # [新增] Flatten MLP 配置 (仅 flatten_only 模式生效)
    flatten_mlp: Optional[FlattenMLPConfig] = None
    # [新增] Pooling_only 配置 (仅 pooling_only 模式生效)
    pooling_only: Optional[PoolingOnlyConfig] = None
    # STFinalNet 专用
    ablation: Optional[str] = None
    use_var_embedding: Optional[bool] = None
    use_dynamic_graph: Optional[bool] = None
    var_embed_dim: Optional[int] = None
    # LSTM 专用
    num_layers: Optional[int] = None


@dataclass
class RuntimeConfig:
    """运行时配置"""
    mode: Literal["train", "inference", "kfold"]
    gpu: int
    n_folds: int
    disable_swanlab: bool
    model_path: Optional[str]
    skip_kfold: bool = False          # 跳过 K-Fold 训练，直接评估测试集
    eval_only: bool = False           # skip_kfold 的别名


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    enabled: bool
    intensity_scaling: bool
    intensity_range: List[float]
    time_shift: bool
    time_shift_range: List[int]
    gaussian_noise: bool
    noise_std: float


@dataclass
class AttentionWeightsConfig:
    """核心斜率注意力权重预设配置 (仅 prior_masked 模式生效)"""
    enabled: bool = True                    # 是否启用核心斜率权重预设


@dataclass
class NineGraphConfig:
    """九图模式配置 (基于 Wasserman 九图)"""
    attention_preset: str = "core_slopes"    # 核心斜率注意力预设
    coupling_strength: float = 1.5           # 跨子系统耦合强度
    # 核心斜率注意力权重预设配置 (仅 prior_masked 模式生效)
    attention_weights: AttentionWeightsConfig = field(default_factory=AttentionWeightsConfig)


@dataclass
class SamplerConfig:
    """WeightedRandomSampler 配置 (长尾分布采样策略)"""
    enabled: bool = False           # 是否启用 WeightedRandomSampler
    replacement: bool = True        # 有放回抽样 (必须为 True)
    drop_last: bool = True          # 丢弃不完整的 batch


@dataclass
class KnownT6ContextConfig:
    """Known-T6 Context 实验配置 (在 classifier 前拼接已知 t6 one-hot)

    实验目的: 测试已知疾病大类信息对当前分类任务的上下文增强效果
    使用场景: 当 target_col_name 设置为其他任务（如"运动耐量"、"心率储备"）时，
              启用 known_t6_context 提供疾病大类信息作为上下文特征

    Args:
        enabled: 是否启用 Known-T6 Context (默认关闭，保持向后兼容)
        source_column: t6 标签来源列名 (默认 "匹配的第一大类")
    """
    enabled: bool = False                    # 默认关闭
    source_column: str = "匹配的第一大类"    # t6 标签来源列


@dataclass
class DebugConfig:
    """调试配置"""
    verbose: bool
    log_interval: int
    check_logits: bool
    save_debug_log: bool


@dataclass
class Config:
    """统一配置类"""
    data: DataConfig
    features: FeaturesConfig
    training: TrainingConfig
    loss: LossConfig
    model: ModelConfig
    runtime: RuntimeConfig
    augmentation: AugmentationConfig
    debug: DebugConfig
    sampler: SamplerConfig = field(default_factory=SamplerConfig)  # [新增] 采样器配置
    # [新增] 任务配置
    task: TaskConfig = field(default_factory=TaskConfig)
    # [新增] 实验配置 (消融实验文件命名)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    # [新增] 九图模式配置
    nine_graph: NineGraphConfig = field(default_factory=NineGraphConfig)
    # [新增] Known-T6 Context 实验配置
    known_t6_context: KnownT6ContextConfig = field(default_factory=KnownT6ContextConfig)

    # 动态计算的属性
    n_class: int = 6
    part_actions: List[str] = field(default_factory=list)
    dataset: str = "CPET_New"
    # [新增] 多标签运行时属性
    is_multilabel: bool = False
    co_occurrence_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def load(cls, config_path: Optional[str] = None, cmd_args: Optional[argparse.Namespace] = None) -> 'Config':
        """
        加载配置

        Args:
            config_path: 配置文件路径 (默认为 configs/config.yaml)
            cmd_args: 命令行参数，用于覆盖配置文件

        Returns:
            Config 对象
        """
        # 确定配置文件路径
        if config_path is None:
            # 默认查找顺序: configs/config.yaml -> src/config.yaml
            script_dir = os.path.dirname(__file__)
            possible_paths = [
                os.path.join(script_dir, '..', 'configs', 'config.yaml'),
                os.path.join(script_dir, 'config.yaml'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

            if config_path is None:
                raise FileNotFoundError(
                    f"配置文件未找到。请确保在以下位置存在 config.yaml:\n"
                    f"  - {possible_paths[0]}\n"
                    f"  - {possible_paths[1]}"
                )

        # 加载 YAML 配置
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        # 解析数据配置 (需要特殊处理 holdout 嵌套配置)
        data_raw = raw_config['data']
        holdout_raw = data_raw.pop('holdout', {})  # 提取 holdout 配置
        holdout_cfg = HoldoutConfig(
            enabled=holdout_raw.get('enabled', False),
            test_ratio=holdout_raw.get('test_ratio', 0.2),
            random_seed=holdout_raw.get('random_seed', 42),
            save_split=holdout_raw.get('save_split', True)
        )
        data_cfg = DataConfig(**data_raw)
        data_cfg.holdout = holdout_cfg  # 设置解析后的 holdout 配置

        # 解析特征配置 (需要特殊处理 channel_groups 和 derived_features)
        features_raw = raw_config['features']
        adapt_mode = features_raw['adapt_mode']

        # 解析衍生特征配置 (可插拔模块)
        derived_features_cfg = None
        base_enabled = True  # [新增] 默认启用基础衍生特征
        o2pulse_enabled = False
        vco2_enabled = False
        if 'derived_features' in features_raw:
            df_raw = features_raw['derived_features']

            # [新增] 解析 base_enabled
            base_enabled = df_raw.get('base_enabled', True)

            # 解析 o2pulse_derivatives
            o2pulse_raw = df_raw.get('o2pulse_derivatives', {})
            o2pulse_derivatives_cfg = O2PulseDerivativesConfig(
                enabled=o2pulse_raw.get('enabled', False),
                features=o2pulse_raw.get('features', ["O2Pulse", "d(O2P)/dt", "d2(O2P)/dt2"])
            )
            o2pulse_enabled = o2pulse_derivatives_cfg.enabled

            # [新增] 解析 vco2 配置对象
            vco2_raw = df_raw.get('vco2', {})
            vco2_cfg = Vco2Config(enabled=vco2_raw.get('enabled', False))
            vco2_enabled = vco2_cfg.enabled

            # 构建 DerivedFeaturesConfig
            derived_features_cfg = DerivedFeaturesConfig(
                base_enabled=base_enabled,
                base=df_raw.get('base', ["PP", "OUES", "EqO2_COP", "HR_diff"]),
                o2pulse_derivatives=o2pulse_derivatives_cfg,
                vco2=vco2_cfg
            )
        else:
            o2pulse_enabled = False

        # [新增] 检查 nine_graph.disable_auto_enabling
        nine_graph_raw = raw_config.get('nine_graph', {})
        disable_auto_enabling = nine_graph_raw.get('disable_auto_enabling', False)

        # [修改] nine_graph 模式自动启用 vco2 和 o2pulse (受 disable_auto_enabling 控制)
        if adapt_mode == 'nine_graph' and not disable_auto_enabling:
            if not o2pulse_enabled:
                o2pulse_enabled = True
                print("[Config] nine_graph 模式自动启用 o2pulse_derivatives")
            if not vco2_enabled:
                vco2_enabled = True
                print("[Config] nine_graph 模式自动启用 vco2")

        # [修改] 根据 base_enabled、o2pulse_enabled 和 vco2_enabled 动态计算特征列表
        optional_keys = _o2pulse_to_optional_keys(o2pulse_enabled)
        if vco2_enabled:
            optional_keys = (optional_keys or []) + ['vco2']
        # [新增] 传递 base_enabled 到 get_feature_list
        feature_names, num_features = get_feature_list(optional_keys, base_enabled=base_enabled)
        channel_groups_dict = get_channel_groups(adapt_mode, optional_keys, base_enabled=base_enabled)
        channel_groups = channel_groups_dict.get(adapt_mode, channel_groups_dict.get('medical', list(channel_groups_dict.values())[0]))

        # 获取索引映射
        IDX = get_feature_indices(optional_keys, base_enabled=base_enabled)
        mode_indices = {k: [idx for group in v for idx in group]
                        for k, v in channel_groups_dict.items()}
        channels = mode_indices.get(adapt_mode, list(range(num_features)))

        features_cfg = FeaturesConfig(
            adapt_mode=adapt_mode,
            target_col_name=features_raw['target_col_name'],
            num_channels=len(channels),
            channel_groups=channel_groups,
            channels=channels,
            feature_names=[feature_names[i] for i in channels],
            derived_features=derived_features_cfg,
            base_enabled=base_enabled,  # [新增]
            o2pulse_enabled=o2pulse_enabled,
            vco2_enabled=vco2_enabled
        )

        # 解析训练配置
        training_raw = raw_config['training']
        training_cfg = TrainingConfig(
            epochs=training_raw['epochs'],
            batch_size=training_raw['batch_size'],
            lr=training_raw['lr'],
            weight_decay=training_raw['weight_decay'],
            gradient_clip=training_raw['gradient_clip'],
            random_seed=training_raw['random_seed'],
            save_best_metric=training_raw.get('save_best_metric', 'macro_f1'),
            optimizer=OptimizerConfig(**training_raw['optimizer']),
            scheduler=SchedulerConfig(**training_raw['scheduler']),
            early_stopping=EarlyStoppingConfig(**training_raw['early_stopping'])
        )

        # 解析损失配置
        loss_raw = raw_config['loss']

        # [新增] 解析二分类配置
        binary_raw = loss_raw.get('binary', {})
        binary_cfg = BinaryLossConfig(
            enabled=binary_raw.get('enabled', False),
            auto_detect=binary_raw.get('auto_detect', True),
            pos_weight=binary_raw.get('pos_weight', 'auto')
        )

        loss_cfg = LossConfig(
            type=loss_raw['type'],
            gamma=loss_raw['gamma'],
            alpha=loss_raw['alpha'],
            supcon_weight=loss_raw.get('supcon_weight', 0.3),
            temperature=loss_raw.get('temperature', 0.07),
            # [新增] LCRLoss 参数
            lcr_enabled=loss_raw.get('lcr_enabled', False),
            lcr_lambda_co=loss_raw.get('lcr_lambda_co', 0.1),
            lcr_epsilon=loss_raw.get('lcr_epsilon', 1e-6),
            # [新增] 二分类配置
            binary=binary_cfg,
            # [新增] Dice 和 LDAM 参数
            dice_smooth=loss_raw.get('dice_smooth', 1.0),
            ldam_max_m=loss_raw.get('ldam_max_m', 0.5),
            ldam_scale_s=loss_raw.get('ldam_scale_s', 30)
        )

        # 解析运行时配置
        runtime_raw = raw_config['runtime']
        runtime_cfg = RuntimeConfig(
            mode=runtime_raw['mode'],
            gpu=runtime_raw['gpu'],
            n_folds=runtime_raw['n_folds'],
            disable_swanlab=runtime_raw['disable_swanlab'],
            model_path=runtime_raw['model_path'],
            skip_kfold=runtime_raw.get('skip_kfold', False),
            eval_only=runtime_raw.get('eval_only', False)
        )

        # 解析模型配置
        models_raw = raw_config['models']
        model_name = models_raw['default']
        model_raw = models_raw.get(model_name, {})

        # 解析静态特征配置 (如果存在)
        static_features_cfg = None
        if 'static_features' in model_raw:
            sf_raw = model_raw['static_features']
            static_features_cfg = StaticFeatureConfig(
                enabled=sf_raw.get('enabled', False),
                static_dim=sf_raw.get('static_dim', 16),
                num_features=sf_raw.get('num_features', 5),
                ablation=sf_raw.get('ablation', 'full'),
                features=sf_raw.get('features', ["age", "gender", "weight", "height", "bmi"]),
                # PFT 配置 (新增)
                pft_enabled=sf_raw.get('pft_enabled', False),
                pft_file=sf_raw.get('pft_file', "xx_path"),
                skip_missing_pft=sf_raw.get('skip_missing_pft', True),
                pft_fill_method=sf_raw.get('pft_fill_method', "none"),
                pft_features=sf_raw.get('pft_features', [])
            )

        # 解析时序编码器配置 (如果存在)
        temporal_encoder_cfg = None
        if 'temporal_encoder' in model_raw:
            te_raw = model_raw['temporal_encoder']
            temporal_encoder_cfg = TemporalEncoderConfig(
                type=te_raw.get('type', 'gru'),
                T_mid=te_raw.get('T_mid', 24),
                use_multiscale=te_raw.get('use_multiscale', False),
                use_residual=te_raw.get('use_residual', False),
                multiscale_kernels=te_raw.get('multiscale_kernels', [3, 5, 7]),
                block1_kernel=te_raw.get('block1_kernel', 7),
                block2_kernel=te_raw.get('block2_kernel', 5),
                use_masked_conv=te_raw.get('use_masked_conv', True),  # [新增]
                ablation=te_raw.get('ablation', 'full'),  # [新增] 时序提纯消融配置
            )

        # 解析先验门控配置 (如果存在)
        prior_gate_cfg = None
        if 'prior_gate' in model_raw:
            pg_raw = model_raw['prior_gate']
            prior_gate_cfg = PriorGateConfig(
                gamma_init=pg_raw.get('gamma_init', 1.0),
                gamma_min=pg_raw.get('gamma_min', 0.1),
                gamma_lr_scale=pg_raw.get('gamma_lr_scale', 0.3)
            )

        # 解析通道注意力配置 (如果存在)
        channel_attention_cfg = None
        if 'channel_attention' in model_raw:
            ca_raw = model_raw['channel_attention']
            channel_attention_cfg = ChannelAttentionConfig(
                enabled=ca_raw.get('enabled', False),
                init_value=ca_raw.get('init_value', 1.0)
            )

        # 解析特征选择消融配置 (如果存在)
        feature_ablation_cfg = None
        if 'feature_ablation' in model_raw:
            fa_raw = model_raw['feature_ablation']
            feature_ablation_cfg = FeatureAblationConfig(
                enabled=fa_raw.get('enabled', False),
                mode=fa_raw.get('mode', 'full'),
                weak_channels=fa_raw.get('weak_channels', []),
                strong_channels=fa_raw.get('strong_channels', []),
                remove_group=fa_raw.get('remove_group', '')
            )

        # [新增] 解析 Flatten MLP 配置 (仅 flatten_only 模式生效)
        flatten_mlp_cfg = None
        if 'flatten_mlp' in model_raw:
            fm_raw = model_raw['flatten_mlp']
            flatten_mlp_cfg = FlattenMLPConfig(
                hidden_dim=fm_raw.get('hidden_dim', None),
                use_two_layer=fm_raw.get('use_two_layer', False),
                dropout=fm_raw.get('dropout', 0.3),
                use_layer_norm=fm_raw.get('use_layer_norm', True)
            )

        # [新增] 解析 Pooling_only 配置 (仅 pooling_only 模式生效)
        pooling_only_cfg = None
        if 'pooling_only' in model_raw:
            po_raw = model_raw['pooling_only']
            pooling_only_cfg = PoolingOnlyConfig(
                pooling_type=po_raw.get('pooling_type', 'avg'),
                mlp_layers=po_raw.get('mlp_layers', 1),
                hidden_dim=po_raw.get('hidden_dim', None),
                dropout=po_raw.get('dropout', 0.3),
                use_layer_norm=po_raw.get('use_layer_norm', True)
            )

        model_cfg = ModelConfig(
            name=model_name,
            hidden_dim=model_raw.get('hidden_dim'),
            dropout=model_raw.get('dropout', 0.3),
            D_time=model_raw.get('D_time'),
            use_variable_length=model_raw.get('use_variable_length'),
            static_features=static_features_cfg,
            temporal_encoder=temporal_encoder_cfg,
            prior_gate=prior_gate_cfg,
            channel_attention=channel_attention_cfg,
            feature_ablation=feature_ablation_cfg,
            flatten_mlp=flatten_mlp_cfg,  # [新增]
            pooling_only=pooling_only_cfg,  # [新增] Pooling_only 配置
            ablation=model_raw.get('ablation'),
            use_var_embedding=model_raw.get('use_var_embedding'),
            use_dynamic_graph=model_raw.get('use_dynamic_graph'),
            var_embed_dim=model_raw.get('var_embed_dim'),
            num_layers=model_raw.get('num_layers'),
            graph_ablation=model_raw.get('graph_ablation')
        )

        # 解析其他配置
        augmentation_cfg = AugmentationConfig(**raw_config['augmentation'])
        debug_cfg = DebugConfig(**raw_config['debug'])

        # [新增] 解析 sampler 配置
        sampler_raw = raw_config.get('sampler', {})
        sampler_cfg = SamplerConfig(
            enabled=sampler_raw.get('enabled', False),
            replacement=sampler_raw.get('replacement', True),
            drop_last=sampler_raw.get('drop_last', True)
        )

        # [新增] 解析任务配置
        task_raw = raw_config.get('task', {})
        task_cfg = TaskConfig(
            mode=task_raw.get('mode', 'single_label'),
            label_separator=task_raw.get('label_separator', ';'),
            min_label_freq=task_raw.get('min_label_freq', 50),
            target_col_name=task_raw.get('target_col_name', '匹配的第一大类')
        )

        # [新增] 解析实验配置 (消融实验文件命名)
        experiment_raw = raw_config.get('experiment', {})
        experiment_cfg = ExperimentConfig(
            suffix=experiment_raw.get('suffix', '')
        )

        # [新增] 解析九图模式配置
        nine_graph_raw = raw_config.get('nine_graph', {})
        attn_weights_raw = nine_graph_raw.get('attention_weights', {})
        attention_weights_cfg = AttentionWeightsConfig(
            enabled=attn_weights_raw.get('enabled', True)
        )
        nine_graph_cfg = NineGraphConfig(
            attention_preset=nine_graph_raw.get('attention_preset', 'core_slopes'),
            coupling_strength=nine_graph_raw.get('coupling_strength', 1.5),
            attention_weights=attention_weights_cfg
        )

        # [新增] 解析 Known-T6 Context 配置
        known_t6_context_raw = raw_config.get('known_t6_context', {})
        known_t6_context_cfg = KnownT6ContextConfig(
            enabled=known_t6_context_raw.get('enabled', False),
            source_column=known_t6_context_raw.get('source_column', '匹配的第一大类')
        )

        # 创建配置对象
        config = cls(
            data=data_cfg,
            features=features_cfg,
            training=training_cfg,
            loss=loss_cfg,
            model=model_cfg,
            runtime=runtime_cfg,
            augmentation=augmentation_cfg,
            debug=debug_cfg,
            sampler=sampler_cfg,  # [新增]
            task=task_cfg,
            experiment=experiment_cfg,  # [新增]
            nine_graph=nine_graph_cfg,  # [新增]
            known_t6_context=known_t6_context_cfg  # [新增] Known-T6 Context
        )

        # 应用命令行参数覆盖
        if cmd_args is not None:
            config = config._apply_cmd_args(cmd_args)

        return config

    def _apply_cmd_args(self, args: argparse.Namespace) -> 'Config':
        """应用命令行参数覆盖配置"""
        # 数据配置
        if hasattr(args, 'data_root') and args.data_root:
            self.data.data_root = args.data_root
        if hasattr(args, 'label_file') and args.label_file:
            self.data.label_file = args.label_file
        if hasattr(args, 'use_mini') and args.use_mini:
            self.data.use_mini = True
            self.data.label_file = self.data.label_file.replace(
                'final_summary_report.xlsx',
                'mini_final_summary_report.xlsx'
            )

        # 特征配置
        if hasattr(args, 'adapt_mode') and args.adapt_mode:
            self.features.adapt_mode = args.adapt_mode
            # 更新 channel_groups
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, '..', 'configs', 'config.yaml')
            if not os.path.exists(config_path):
                config_path = os.path.join(script_dir, 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            self.features.channel_groups = raw_config['features']['channel_groups'].get(
                args.adapt_mode,
                raw_config['features']['channel_groups']['full']
            )
            self.features.channels = MODE_INDICES.get(args.adapt_mode, list(range(22)))
            self.features.num_channels = len(self.features.channels)
            self.features.feature_names = [NEW_FEATURES[i] for i in self.features.channels]

        if hasattr(args, 'target_col_name') and args.target_col_name:
            self.features.target_col_name = args.target_col_name

        # 训练配置 (只有非 None 时才覆盖)
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.training.epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            self.training.lr = args.lr
            self.training.optimizer.lr = args.lr
        if hasattr(args, 'random_seed') and args.random_seed is not None:
            self.training.random_seed = args.random_seed

        # 模型配置
        if hasattr(args, 'model') and args.model:
            self.model.name = args.model
            # 重新加载模型特定配置
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, '..', 'configs', 'config.yaml')
            if not os.path.exists(config_path):
                config_path = os.path.join(script_dir, 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            model_raw = raw_config['models'].get(args.model, {})
            self.model.hidden_dim = model_raw.get('hidden_dim')
            self.model.dropout = model_raw.get('dropout', 0.3)
            self.model.D_time = model_raw.get('D_time')
            self.model.use_variable_length = model_raw.get('use_variable_length')
            self.model.ablation = model_raw.get('ablation')
            self.model.use_var_embedding = model_raw.get('use_var_embedding')
            self.model.use_dynamic_graph = model_raw.get('use_dynamic_graph')
            self.model.var_embed_dim = model_raw.get('var_embed_dim')

        if hasattr(args, 'ablation') and args.ablation is not None:
            self.model.ablation = args.ablation
        if hasattr(args, 'use_var_embedding') and args.use_var_embedding is not None:
            self.model.use_var_embedding = args.use_var_embedding
        if hasattr(args, 'use_dynamic_graph') and args.use_dynamic_graph is not None:
            self.model.use_dynamic_graph = args.use_dynamic_graph
        if hasattr(args, 'var_embed_dim') and args.var_embed_dim is not None:
            self.model.var_embed_dim = args.var_embed_dim
        if hasattr(args, 'D_time') and args.D_time is not None:
            self.model.D_time = args.D_time
        if hasattr(args, 'max_length') and args.max_length is not None:
            self.data.max_length = args.max_length
        if hasattr(args, 'use_variable_length') and args.use_variable_length is not None:
            self.model.use_variable_length = args.use_variable_length

        # 运行时配置
        if hasattr(args, 'mode') and args.mode is not None:
            self.runtime.mode = args.mode
        if hasattr(args, 'gpu') and args.gpu is not None:
            self.runtime.gpu = args.gpu
        if hasattr(args, 'n_folds') and args.n_folds is not None:
            self.runtime.n_folds = args.n_folds
        if hasattr(args, 'disable_swanlab') and args.disable_swanlab:
            self.runtime.disable_swanlab = True
        if hasattr(args, 'model_path') and args.model_path:
            self.runtime.model_path = args.model_path

        return self

    def update_with_dataset(self, dataset) -> 'Config':
        """根据数据集更新配置"""
        self.n_class = dataset.n_classes
        self.part_actions = list(dataset.label_mapping.keys())
        return self

    @property
    def exp_suffix(self) -> str:
        """获取实验后缀 (带下划线前缀，如: "_v1", "_baseline")"""
        suffix = self.experiment.suffix if hasattr(self, 'experiment') and self.experiment else ""
        return f"_{suffix}" if suffix else ""

    def print_config(self):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("配置信息")
        print("=" * 80)
        print(f"模型: {self.model.name}")
        print(f"任务模式: {self.task.mode}" + (" (多标签)" if self.task.mode == "multi_label" else " (单标签)"))
        print(f"特征模式: {self.features.adapt_mode}")
        print(f"特征数: {self.features.num_channels}")
        print(f"基础衍生特征: {'启用' if self.features.base_enabled else '禁用'} (PP, OUES, EqO2_COP, HR_diff)")
        print(f"氧脉搏导数特征: {'启用' if self.features.o2pulse_enabled else '禁用'}")
        print(f"V'CO2 特征: {'启用' if self.features.vco2_enabled else '禁用'}")
        print(f"特征列表: {self.features.feature_names}")
        print(f"时间窗口: {self.data.L_win} (固定) / {self.data.max_length} (变长)")
        print(f"批次大小: {self.training.batch_size}")
        print(f"学习率: {self.training.lr}")
        print(f"训练轮数: {self.training.epochs}")
        print(f"运行模式: {self.runtime.mode}")
        print(f"数据集模式: {'Mini' if self.data.use_mini else 'Full'}")
        if self.experiment.suffix:
            print(f"实验后缀: {self.experiment.suffix}")
        if self.sampler.enabled:
            print(f"WeightedRandomSampler: 已启用 (replacement={self.sampler.replacement}, drop_last={self.sampler.drop_last})")
        # [新增] Known-T6 Context 配置显示
        if hasattr(self, 'known_t6_context') and self.known_t6_context.enabled:
            print(f"Known-T6 Context: 已启用 (source_column={self.known_t6_context.source_column})")
        # 显示损失函数权重配置
        alpha_display = self.loss.alpha if self.loss.alpha else "None (均衡权重)"
        print(f"Loss alpha: {alpha_display}")
        if self.loss.lcr_enabled:
            print(f"LCRLoss: 已启用 (lambda_co={self.loss.lcr_lambda_co})")
        print("=" * 80 + "\n")

    def to_dict(self, exclude_paths: bool = True) -> Dict[str, Any]:
        """
        将配置转换为字典格式 (用于 SwanLab 记录)

        Args:
            exclude_paths: 是否排除文件路径 (避免泄露敏感信息)

        Returns:
            嵌套字典格式的配置
        """
        import copy

        def _dataclass_to_dict(obj):
            """递归将 dataclass 转换为字典"""
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for key, value in asdict(obj).items():
                    result[key] = _dataclass_to_dict(value)
                return result
            elif isinstance(obj, list):
                return [_dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        config_dict = _dataclass_to_dict(self)

        # 排除敏感路径信息
        if exclude_paths:
            path_keys_to_remove = [
                ('data', 'data_root'),
                ('data', 'label_file'),
                ('data', 'output_root'),
                ('model', 'static_features', 'pft_file'),
                ('runtime', 'model_path'),
            ]
            for key_path in path_keys_to_remove:
                d = config_dict
                for key in key_path[:-1]:
                    if key in d and isinstance(d[key], dict):
                        d = d[key]
                    else:
                        break
                else:
                    if key_path[-1] in d:
                        # 保留文件名，移除完整路径
                        full_path = d[key_path[-1]]
                        if full_path and isinstance(full_path, str):
                            d[key_path[-1]] = f".../{full_path.split('/')[-1]}"
                        elif full_path is None:
                            d[key_path[-1]] = None

        # 移除大型数组 (co_occurrence_matrix)
        if 'co_occurrence_matrix' in config_dict:
            config_dict['co_occurrence_matrix'] = f"<{type(self.co_occurrence_matrix).__name__} shape={self.co_occurrence_matrix.shape if self.co_occurrence_matrix is not None else None}>"

        return config_dict


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_cmd_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CPET疾病分类训练")

    # 模型配置
    parser.add_argument("--model", type=str, default=None,
                        choices=["HDSTGCN", "STFinalNet", "lstm", "resnet", "mednet"],
                        help="模型名称 (覆盖配置文件)")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (默认: src/config.yaml)")

    # 运行模式
    parser.add_argument("--mode", type=str, default=None,
                        choices=["train", "inference", "kfold"],
                        help="运行模式")
    parser.add_argument("--adapt_mode", type=str, default=None,
                        choices=["full", "select", "medical", "nine_graph"],
                        help="特征适配模式")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--gpu", type=int, default=None, help="GPU设备号")
    parser.add_argument("--random_seed", type=int, default=None, help="随机种子")

    # 数据参数
    parser.add_argument("--data_root", type=str, default=None, help="数据目录")
    parser.add_argument("--label_file", type=str, default=None, help="标签文件")
    parser.add_argument("--use_mini", action="store_true", help="使用mini数据集")
    parser.add_argument("--target_col_name", type=str, default=None,
                        choices=AVAILABLE_LABEL_COLUMNS,
                        help="目标分类列")

    # 模型特定参数
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["both", "tfe_only", "sfe_only"],
                        help="消融实验模式")
    parser.add_argument("--use_var_embedding", action="store_true", default=None,
                        help="使用变量身份嵌入")
    parser.add_argument("--no_var_embedding", dest="use_var_embedding",
                        action="store_false", help="禁用变量身份嵌入")
    parser.add_argument("--use_dynamic_graph", action="store_true", default=None,
                        help="使用动态图拓扑")
    parser.add_argument("--no_dynamic_graph", dest="use_dynamic_graph",
                        action="store_false", help="禁用动态图拓扑")
    parser.add_argument("--var_embed_dim", type=int, default=None, help="变量嵌入维度")
    parser.add_argument("--D_time", type=int, default=None, help="时序编码维度")
    parser.add_argument("--max_length", type=int, default=None, help="最大序列长度")
    parser.add_argument("--use_variable_length", action="store_true", default=None,
                        help="使用变长序列模式")
    parser.add_argument("--no_variable_length", dest="use_variable_length",
                        action="store_false", help="禁用变长序列模式")

    # 其他参数
    parser.add_argument("--n_folds", type=int, default=None, help="K折数量")
    parser.add_argument("--disable_swanlab", action="store_true", help="禁用SwanLab")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径 (推理模式)")

    return parser.parse_args()


# =============================================================================
# 兼容性接口 (与旧代码兼容)
# =============================================================================

def build_args_new(model_name: str = None, adapt_mode: str = None) -> Config:
    """
    兼容性接口：构建配置对象

    Args:
        model_name: 模型名称 (覆盖配置文件)
        adapt_mode: 特征适配模式 (覆盖配置文件)

    Returns:
        Config 对象
    """
    args = argparse.Namespace()

    if model_name:
        args.model = model_name
    if adapt_mode:
        args.adapt_mode = adapt_mode

    return Config.load(cmd_args=args if (model_name or adapt_mode) else None)


def update_args_with_dataset(config: Config, dataset) -> Config:
    """
    兼容性接口：根据数据集更新配置
    """
    return config.update_with_dataset(dataset)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("配置模块测试")
    print("=" * 80)

    # 测试加载配置
    config = Config.load()
    config.print_config()

    # 测试命令行参数覆盖
    print("\n测试命令行参数覆盖:")
    args = argparse.Namespace(
        model="HDSTGCN",
        epochs=50,
        lr=0.001,
        use_mini=True
    )
    config = Config.load(cmd_args=args)
    print(f"模型: {config.model.name}")
    print(f"轮数: {config.training.epochs}")
    print(f"学习率: {config.training.lr}")
    print(f"Mini模式: {config.data.use_mini}")
