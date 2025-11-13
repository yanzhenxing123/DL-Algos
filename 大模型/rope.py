"""
@Author: yanzx
@Time: 2025/1/6
@Description: 旋转位置编码 (Rotary Position Embedding, RoPE) 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    应用旋转位置编码到 Q 和 K
    
    Args:
        q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len, head_dim]
        cos: 余弦值 [seq_len, head_dim/2]
        sin: 正弦值 [seq_len, head_dim/2]
        position_ids: 位置索引 [batch_size, seq_len]
    
    Returns:
        q_rot: 旋转后的 Query
        k_rot: 旋转后的 Key
    """
    # 根据 position_ids 选择对应的 cos 和 sin
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, head_dim/2]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, head_dim/2]
    
    # 将 q 和 k 分成两半
    q1, q2 = q.chunk(2, dim=-1)  # 每部分 [batch_size, num_heads, seq_len, head_dim/2]
    k1, k2 = k.chunk(2, dim=-1)
    
    # 应用旋转矩阵
    # [cos(θ)  -sin(θ)] [x1]   [x1*cos(θ) - x2*sin(θ)]
    # [sin(θ)   cos(θ)] [x2] = [x1*sin(θ) + x2*cos(θ)]
    q_rot = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)
    
    k_rot = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)
    
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码"""
    
    def __init__(self, dim, max_seq_len=512, base=10000):
        """
        Args:
            dim: 每个头的维度（head_dim）
            max_seq_len: 最大序列长度
            base: 基础频率（默认 10000）
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算旋转角度
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos 和 sin（在 forward 中动态计算）
    
    def forward(self, seq_len, device):
        """
        计算旋转位置编码的 cos 和 sin 值
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            cos: [seq_len, head_dim/2]
            sin: [seq_len, head_dim/2]
        """
        # 位置索引 [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device).float()
        
        # 计算角度: θ_i = t / (base^(2i/d))
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        
        # 计算 cos 和 sin
        cos = freqs.cos()  # [seq_len, dim/2]
        sin = freqs.sin()  # [seq_len, dim/2]
        
        return cos, sin


class MultiHeadAttentionWithRoPE(nn.Module):
    """带 RoPE 的多头注意力"""
    
    def __init__(self, d_model, num_heads, max_seq_len=512, base=10000):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # RoPE（每个头的维度）
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len, base)
    
    def forward(self, x, position_ids=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            position_ids: [batch_size, seq_len] - 位置索引（可选）
        """
        batch_size, seq_len, d_model = x.shape
        
        # 如果没有提供 position_ids，使用默认的 [0, 1, 2, ...]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 计算 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE
        cos, sin = self.rope(seq_len, x.device)
        Q_rot, K_rot = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        
        # 计算注意力分数
        scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        
        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        
        return output


class TransformerBlockWithRoPE(nn.Module):
    """带 RoPE 的 Transformer Block"""
    
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, position_ids=None):
        # Self-attention with RoPE
        attn_output = self.self_attn(x, position_ids)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


def demo_rope():
    """演示 RoPE"""
    print("旋转位置编码 (RoPE) 演示")
    print("="*50)
    
    # 创建模型
    d_model = 128
    num_heads = 4
    seq_len = 10
    batch_size = 2
    
    model = MultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len=512)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    
    # 演示位置编码
    print("\n" + "="*50)
    print("RoPE 位置编码示例")
    print("="*50)
    
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=10)
    cos, sin = rope(seq_len=10, device=torch.device('cpu'))
    
    print(f"Cos 形状: {cos.shape}")
    print(f"Sin 形状: {sin.shape}")
    print(f"\n前3个位置的 Cos 值（前4个维度）:")
    print(cos[:3, :4])
    print(f"\n前3个位置的 Sin 值（前4个维度）:")
    print(sin[:3, :4])


def compare_rope_vs_absolute():
    """对比 RoPE 和绝对位置编码"""
    print("\n" + "="*50)
    print("RoPE vs 绝对位置编码")
    print("="*50)
    
    print("\n1. 绝对位置编码（传统方法）:")
    print("   - 将位置信息直接加到 embedding 中")
    print("   - 位置编码是固定的或可学习的")
    print("   - 只能编码绝对位置")
    print("   - 示例: BERT, GPT-2")
    
    print("\n2. 旋转位置编码（RoPE）:")
    print("   - 通过旋转矩阵编码位置信息")
    print("   - 位置信息编码在 Q 和 K 中")
    print("   - 天然支持相对位置")
    print("   - 可以外推到更长的序列")
    print("   - 示例: LLaMA, ChatGLM, PaLM")
    
    print("\n3. RoPE 的优势:")
    print("   ✅ 相对位置编码：注意力分数天然包含相对位置信息")
    print("   ✅ 外推能力：可以处理比训练时更长的序列")
    print("   ✅ 计算效率：不需要额外的位置 embedding")
    print("   ✅ 旋转不变性：保持向量的模长不变")


if __name__ == "__main__":
    demo_rope()
    compare_rope_vs_absolute()

