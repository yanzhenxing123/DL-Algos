"""
@Author: yanzx
@Time: 2025/1/6
@Description: RoPE 外推能力详解 - 为什么训练 2048 可以推理 4096+
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_rope_extrapolation():
    """演示 RoPE 的外推能力"""
    print("="*60)
    print("RoPE 外推能力详解")
    print("="*60)
    
    print("\n1. 核心原理：旋转矩阵的连续性")
    print("-" * 60)
    print("""
    RoPE 使用旋转矩阵编码位置信息：
    
    位置 m 的旋转角度: θ_m = m / (10000^(2i/d))
    位置 n 的旋转角度: θ_n = n / (10000^(2i/d))
    
    关键点：
    - 角度是位置的连续函数
    - 即使 m, n 超出训练范围，角度仍然有效
    - 旋转矩阵的数学性质保证连续性
    """)
    
    print("\n2. 为什么 Sin 位置编码不能外推？")
    print("-" * 60)
    print("""
    Sin 位置编码：
    - 位置编码是预计算的，存储在 embedding 表中
    - 训练时只计算了 max_seq_len 个位置的编码
    - 超出范围的位置编码是"全新的"，模型没见过
    
    示例：
    - 训练时：位置 0-511 的编码是预计算的
    - 推理时：位置 512-1023 的编码也是预计算的，但模型没见过
    - 模型无法理解这些新位置的含义
    """)
    
    print("\n3. 为什么 RoPE 可以外推？")
    print("-" * 60)
    print("""
    RoPE 位置编码：
    - 位置编码是动态计算的，不是预存储的
    - 旋转角度是位置的连续函数
    - 即使位置超出训练范围，角度仍然遵循相同的数学规律
    
    示例：
    - 训练时：位置 0-2047，角度范围 [0, θ_2047]
    - 推理时：位置 2048-4095，角度范围 [θ_2048, θ_4095]
    - 角度函数是连续的，模型可以理解这些新位置
    """)
    
    print("\n4. 数学证明：相对位置的保持")
    print("-" * 60)
    print("""
    对于位置 m 和 n，相对位置是 (m - n)
    
    RoPE 的注意力分数：
    Q_m^T K_n = (R_m Q)^T (R_n K)
              = Q^T R_m^T R_n K
              = Q^T R_{m-n} K
    
    关键点：
    - 相对位置 (m-n) 直接编码在旋转矩阵中
    - 即使 m, n 超出训练范围，相对位置关系仍然保持
    - 模型在训练时学到的相对位置模式可以外推到新位置
    """)


def visualize_rope_extrapolation():
    """可视化 RoPE 的外推能力"""
    print("\n" + "="*60)
    print("可视化对比：Sin vs RoPE")
    print("="*60)
    
    # 模拟训练和推理的长度
    train_len = 2048
    inference_len = 4096
    
    # Sin 位置编码（预计算）
    d_model = 128
    max_train_len = train_len
    
    print(f"\n训练长度: {train_len}")
    print(f"推理长度: {inference_len}")
    print(f"超出训练长度: {inference_len - train_len} 个位置")
    
    print("\nSin 位置编码:")
    print(f"  • 预计算位置编码: [0, {max_train_len-1}]")
    print(f"  • 推理时位置 {train_len}-{inference_len-1}: 模型没见过")
    print(f"  • 这些位置的编码是全新的，模型无法理解")
    print(f"  ❌ 外推失败")
    
    print("\nRoPE 位置编码:")
    print(f"  • 训练时位置范围: [0, {train_len-1}]")
    print(f"  • 推理时位置范围: [0, {inference_len-1}]")
    print(f"  • 旋转角度函数: θ(pos) = pos / (10000^(2i/d))")
    print(f"  • 角度函数是连续的，位置 {train_len}-{inference_len-1} 的角度仍然有效")
    print(f"  ✅ 外推成功")
    
    # 计算角度示例
    base = 10000
    dim = 64
    i = 0  # 第一个维度对
    
    print(f"\n角度计算示例（维度对 {i}）:")
    print(f"  训练位置 0:   θ = 0 / (10000^(0/64)) = 0.0")
    print(f"  训练位置 2047: θ = 2047 / (10000^(0/64)) = 2047.0")
    print(f"  推理位置 2048: θ = 2048 / (10000^(0/64)) = 2048.0")
    print(f"  推理位置 4095: θ = 4095 / (10000^(0/64)) = 4095.0")
    print(f"  ✅ 角度连续，模型可以理解")


def demonstrate_relative_position_preservation():
    """演示相对位置的保持"""
    print("\n" + "="*60)
    print("相对位置保持示例")
    print("="*60)
    
    print("\n场景：模型在训练时学习了相邻位置的模式")
    print("     推理时，即使位置超出训练范围，相邻关系仍然保持")
    
    print("\n训练时:")
    print("  位置 100 和 101: 相对位置 = 1")
    print("  位置 200 和 201: 相对位置 = 1")
    print("  模型学习到：相对位置 1 的模式")
    
    print("\n推理时（超出训练范围）:")
    print("  位置 3000 和 3001: 相对位置 = 1")
    print("  位置 4000 和 4001: 相对位置 = 1")
    print("  ✅ RoPE 保证相对位置 1 的关系仍然成立")
    print("  ✅ 模型可以应用训练时学到的模式")
    
    print("\n数学证明:")
    print("  Q_3000^T K_3001 = Q^T R_3000^T R_3001 K")
    print("                   = Q^T R_{3000-3001} K")
    print("                   = Q^T R_{-1} K")
    print("  Q_100^T K_101 = Q^T R_100^T R_101 K")
    print("                = Q^T R_{-1} K")
    print("  ✅ 相对位置相同，旋转矩阵相同")


def practical_example():
    """实际应用示例"""
    print("\n" + "="*60)
    print("实际应用：LLaMA 案例")
    print("="*60)
    
    print("\nLLaMA 的配置:")
    print("  • 训练时最大序列长度: 2048 tokens")
    print("  • 推理时可以处理: 4096+ tokens")
    print("  • 使用 RoPE 位置编码")
    
    print("\n为什么可以这样做？")
    print("  1. RoPE 的角度函数是连续的")
    print("  2. 相对位置关系在训练和推理时保持一致")
    print("  3. 模型学到的相对位置模式可以外推")
    
    print("\n实际效果:")
    print("  ✅ 可以处理更长的文档")
    print("  ✅ 可以生成更长的文本")
    print("  ✅ 无需重新训练或微调")
    print("  ✅ 性能基本保持（可能略有下降）")
    
    print("\n注意事项:")
    print("  ⚠️ 超出训练长度太多时，性能可能下降")
    print("  ⚠️ 建议不要超出训练长度太多（如 2-4 倍）")
    print("  ⚠️ 如果需要处理更长的序列，建议微调")


def compare_extrapolation_mechanisms():
    """对比外推机制"""
    print("\n" + "="*60)
    print("外推机制对比")
    print("="*60)
    
    print("\nSin 位置编码的外推问题:")
    print("""
    问题 1: 位置编码是离散的
    - 每个位置有唯一的编码向量
    - 超出训练范围的位置编码是"新"的
    - 模型无法理解这些新编码的含义
    
    问题 2: 没有连续性保证
    - 位置 511 和 512 的编码之间没有连续性
    - 模型无法从训练位置推导推理位置
    
    问题 3: 相对位置需要学习
    - 模型需要学习如何从绝对位置推导相对位置
    - 超出训练范围时，这种推导可能失效
    """)
    
    print("\nRoPE 位置编码的外推优势:")
    print("""
    优势 1: 角度函数是连续的
    - θ(pos) = pos / (10000^(2i/d))
    - 位置 2047 和 2048 的角度是连续的
    - 模型可以理解新位置
    
    优势 2: 相对位置直接编码
    - 相对位置 (m-n) 直接编码在旋转矩阵中
    - 不需要从绝对位置推导
    - 相对位置关系在训练和推理时保持一致
    
    优势 3: 数学性质保证
    - 旋转矩阵的数学性质保证连续性
    - 相对位置关系自动保持
    - 外推是数学上自然的
    """)


if __name__ == "__main__":
    demonstrate_rope_extrapolation()
    visualize_rope_extrapolation()
    demonstrate_relative_position_preservation()
    practical_example()
    compare_extrapolation_mechanisms()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("""
    RoPE 可以外推的核心原因：
    
    1. ✅ 角度函数连续
       - θ(pos) 是位置的连续函数
       - 即使位置超出训练范围，角度仍然有效
    
    2. ✅ 相对位置直接编码
       - 相对位置 (m-n) 直接编码在旋转矩阵中
       - 训练时学到的相对位置模式可以外推
    
    3. ✅ 数学性质保证
       - 旋转矩阵的数学性质保证连续性
       - 外推是数学上自然的，不需要额外学习
    
    因此，LLaMA 训练 2048 可以推理 4096+ 是 RoPE 的天然优势！
    """)

