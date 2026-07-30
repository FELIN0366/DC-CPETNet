"""
新任务的模型配置
支持22特征和多分类
"""

import argparse
import numpy as np
from feature_mapping import NEW_FEATURES, MODE_INDICES
from label_extractor import AVAILABLE_LABEL_COLUMNS


def build_args_new(model_name="STFinalNet", num_features=22, adapt_mode="full"):
    """
    构建新任务的配置参数

    Args:
        model_name: 模型名称
        num_features: 特征数量 (12或22)
        adapt_mode: 特征适配模式
            - "full": 使用全部22个特征
            - "select": 只使用与原始12特征对应的特征
            - "medical": 基于医学分组选择关键特征
    """
    parser = argparse.ArgumentParser()

    # ========== 数据路径配置 ==========
    parser.add_argument("--data_root", type=str,
                       default="xx_path",
                       help="数据目录")
    parser.add_argument("--label_file", type=str,
                       default="xx_path",
                       help="标签文件路径")
    parser.add_argument("--use_mini", action="store_true",
                       help="使用mini数据集（100样本）进行快速训练")
    parser.add_argument("--output_root", type=str,
                       default="xx_path",
                       help="输出目录")

    # ========== 模型配置 ==========
    parser.add_argument("--model_name", type=str, default=model_name,
                       help="模型名称")
    parser.add_argument("--hidden_dim", type=int, default=16,
                       help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout率")

    # ========== 数据配置 ==========
    parser.add_argument("--L_win", type=int, default=162,
                       help="时间窗口长度（基于第一段数据统计）")
    parser.add_argument("--stride", type=int, default=1,
                       help="滑动窗口步长")

    # ========== 训练配置 ==========
    parser.add_argument("--epochs", type=int, default=200,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="学习率")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                       help="测试集比例")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU设备号")

    # ========== 模式配置 ==========
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "inference"],
                       help="运行模式")
    parser.add_argument("--use_kfold", action="store_true",
                       help="是否使用K折交叉验证")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="K折交叉验证的折数")
    parser.add_argument("--random_seed", type=int, default=3407,
                       help="随机种子")

    # ========== 消融实验配置 ==========
    parser.add_argument("--ablation", type=str, default="both",
                       choices=["both", "tfe_only", "sfe_only"],
                       help="消融实验模式: both=双分支, tfe_only=仅TFE, sfe_only=仅SFE")

    # ========== SFE 分支 v3 配置 ==========
    parser.add_argument("--use_var_embedding", action="store_true", default=True,
                       help="是否使用变量身份嵌入 (Variable Embedding)")
    parser.add_argument("--use_dynamic_graph", action="store_true", default=True,
                       help="是否使用动态图拓扑融合")
    parser.add_argument("--var_embed_dim", type=int, default=8,
                       help="变量嵌入维度")

    # ========== HDSTGCN 配置 ==========
    parser.add_argument("--use_hd_stgcn", action="store_true", default=True,
                       help="使用 HDSTGCN 模型替代 STFinalNet")
    parser.add_argument("--use_variable_length", action="store_true", default=True,
                       help="使用变长序列模式")
    parser.add_argument("--no_variable_length", dest="use_variable_length", action="store_false",
                       help="禁用变长序列模式")
    parser.add_argument("--max_length", type=int, default=330,
                       help="变长模式下的最大序列长度")
    parser.add_argument("--D_time", type=int, default=16,
                       help="HDSTGCN 时序编码维度")

    # ========== 特征配置 ==========
    parser.add_argument("--adapt_mode", type=str, default=adapt_mode,
                       choices=["full", "select", "medical"],
                       help="特征适配模式")
    parser.add_argument("--target_col_name", type=str, default="匹配的第一大类",
                       choices=AVAILABLE_LABEL_COLUMNS,
                       help="标签文件中的目标分类列名称")

    args = parser.parse_args([])  # 使用空列表避免命令行参数冲突

    # 原始索引映射参考
    # 0:MET, 1:Load, 2:RER, 3:HR, 4:HRR, 5:dH/dO2, 6:SVc, 7:Psys, 8:Pdia, 9:SpO2, 
    # 10:VO2, 11:VO2/kg, 12:dO2/dw, 13:BF, 14:VE, 15:BR, 16:EqO2, 17:EqCO2, 
    # 18:PETO2, 19:PETCO2, 20:VDc/VT, 21:VTex

    # ========== 根据适配模式自动设置特征通道 ==========
    if adapt_mode not in MODE_INDICES:
        raise ValueError(f"Unknown adapt_mode: {adapt_mode}")
    
    # 从 feature_mapping 获取唯一可信的索引列表
    args.channels = MODE_INDICES[adapt_mode]
    args.num_channels = len(args.channels)
    args.feature_names = [NEW_FEATURES[i] for i in args.channels]
    
    # 动态生成 channel_groups (用于 StarTransformer 分组)
    # 这里的逻辑是简单的均匀切分或基于索引聚类，
    # 但由于 StarTransformer 的分组主要用于 Multi-head 注意力或子结构，
    # 我们这里保留一个简化的分组逻辑，或者如果模型不需要强分组，可以设为单一组。
    # 为了兼容旧代码逻辑，我们按数量粗略分 3-4 组

    if adapt_mode == "full":
        # 分组逻辑：
        # G0: 运动负荷与能量代谢 (MET, Load, VO2, VO2/kg, dO2/dw) 涵盖摄氧量水平及氧合效率，反映整体运动耐量
        # G1: 循环系统/血流动力学 (HR, HRR, SVc, Psys, Pdia, dH/dO2) 涵盖心率响应、血压、每搏输出量及心率/摄氧量耦合
        # G2: 呼吸动力学 (BF, VE, BR, VTex) 涵盖呼吸频率、通气量以及通气储备深度
        # G3: 气体交换与效率 (RER, SpO2, EqO2, EqCO2, PETO2, PETCO2, VDc/VT) 涵盖呼吸交换率、血氧、通气当量及呼气末分压（死腔率反映通气/血流灌注）
        # 定义切分点
        c = 0
        g0_len, g1_len, g2_len, g3_len = 5, 6, 4, 7
        
        args.channel_groups = [
            list(range(c, c + g0_len)),           # [0, 1, 2, 3, 4]
            list(range(c + g0_len, c + g0_len + g1_len)),
            list(range(c + g0_len + g1_len, c + g0_len + g1_len + g2_len)),
            list(range(c + g0_len + g1_len + g2_len, 22))
        ]

    elif adapt_mode == "select":
        # Wasserman 的“齿轮耦合模型”，将生理信号简化为“代谢、驱动、效率”三维度
        # 1. 定义原始特征分组 (Raw Indices)
        # 这里的数字是原始 Excel/Features 里的列号
        # Group 0: 能量代谢 (MET, VO2, VO2/kg, Load) 反映机体对运动任务的整体能量消耗与功率输出
        # Group 1: 心肺驱动 (HR, VE, BF, VTex) 反映心脏（HR）与肺泵（VE, BF, VTex）的机械动力输出
        # Group 2: 换气效率 (RER, EqO2, EqCO2, SpO2) 反映气体交换的质量、代谢状态(RER)及血氧稳定性(SpO2)

        args.channel_groups = [
            list(range(0, 4)),    # Group 0 的相对下标
            list(range(4, 8)),    # Group 1 的相对下标
            list(range(8, 12))    # Group 2 的相对下标
        ]
        

    elif adapt_mode == "medical":
        # 基于医学分组选择每组的核心特征（平衡各系统）
        # 运动负荷与能量代谢: Load, MET, V'O2, VO2/kg, dO2/dW
        # 心血管: HR, HRR, Psys, Pdia, SVc
        # 肺通气: V'E, BF, VTex, BR
        # 气体交换: RER, SpO2, EqO2, EqCO2
        args.channel_groups = [
            list(range(0, 5)),    # Group 0
            list(range(5, 10)),   # Group 1
            list(range(10, 14)),  # Group 2
            list(range(14, 18))   # Group 3
        ]

    # ========== 设置类别信息 ==========
    # 注意: n_class将在数据集加载后动态更新
    args.n_class = 6  # 默认值，实际会被数据集的类别数覆盖
    args.part_actions = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5"]  # 将被实际类别名替换

    # ========== 其他配置 ==========
    args.dataset = "CPET_New"
    args.subindex = 0

    # ========== Mini数据集支持 ==========
    # 在命令行解析后会自动设置，这里提供默认逻辑
    if hasattr(args, 'use_mini') and args.use_mini:
        args.label_file = args.label_file.replace('final_summary_report.xlsx',
                                                  'mini_final_summary_report.xlsx')
        print(f"\n>>> 使用Mini数据集模式")
        print(f">>> 标签文件: {args.label_file}")

    print("\n" + "="*80)
    print(f"模型配置: {model_name}")
    print("="*80)
    print(f"特征适配模式: {adapt_mode}")
    print(f"使用特征数: {args.num_channels}")
    print(f"特征列表: {args.feature_names}")
    print(f"时间窗口长度: {args.L_win}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    if hasattr(args, 'use_mini') and args.use_mini:
        print(f"数据集模式: Mini (约100样本)")
    else:
        print(f"数据集模式: Full (约3010样本)")
    print("="*80 + "\n")

    return args


def update_args_with_dataset(args, dataset):
    """
    根据数据集更新配置参数

    Args:
        args: 配置参数
        dataset: 数据集对象
    """
    # 更新类别信息
    args.n_class = dataset.n_classes
    args.part_actions = list(dataset.label_mapping.keys())

    print(f"更新配置 - 类别数: {args.n_class}")
    print(f"类别名称: {args.part_actions}")

    return args


if __name__ == "__main__":
    print("="*80)
    print("测试模型配置")
    print("="*80)

    # 测试不同的适配模式
    modes = ["full", "select", "medical"]

    for mode in modes:
        print(f"\n{'='*80}")
        print(f"测试模式: {mode}")
        print(f"{'='*80}")

        args = build_args_new(model_name="STFinalNet", adapt_mode=mode)

    print("\n" + "="*80)
    print("配置测试完成!")
    print("="*80)
