import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    MLA (Multi-head Latent Attention): 多头潜在注意力机制
    
    核心思想：
    1. 将Q、K、V映射到低维潜在空间（latent space）
    2. 在潜在空间中计算注意力，降低计算复杂度
    3. 将注意力结果映射回原始空间
    
    优势：
    - 计算复杂度从 O(n²·d) 降低到 O(n²·d_latent)，其中 d_latent << d
    - 对于长序列特别有效
    - 保持多头注意力的表达能力
    """
    
    def __init__(self, d_model, num_heads, d_latent=None, dropout=0.1):
        """
        Args:
            d_model: 输入/输出维度
            num_heads: 注意力头数
            d_latent: 潜在空间维度，如果为None则设为 d_model // 4
            dropout: dropout比率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # 潜在空间维度，通常设为原始维度的1/4或1/2
        if d_latent is None:
            d_latent = max(d_model // 4, num_heads)  # 至少要有num_heads个维度
        self.d_latent = d_latent
        
        # 将输入映射到潜在空间的投影层
        # Q, K, V 先映射到潜在空间，再映射到多头格式
        self.wq_latent = nn.Linear(d_model, d_latent)  # Q的潜在空间投影
        self.wk_latent = nn.Linear(d_model, d_latent)  # K的潜在空间投影
        self.wv_latent = nn.Linear(d_model, d_latent)  # V的潜在空间投影
        
        # 从潜在空间映射到多头格式
        self.wq_heads = nn.Linear(d_latent, d_model)  # 潜在空间 -> 多头Q
        self.wk_heads = nn.Linear(d_latent, d_model)  # 潜在空间 -> 多头K
        self.wv_heads = nn.Linear(d_latent, d_model)  # 潜在空间 -> 多头V
        
        # 输出投影
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, q_seq_len, d_model)
            key: (batch_size, k_seq_len, d_model)
            value: (batch_size, v_seq_len, d_model)
            mask: (batch_size, q_seq_len, k_seq_len) 可选，0表示mask
        Returns:
            output: (batch_size, q_seq_len, d_model)
        """
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        
        # Step 1: 映射到潜在空间
        # (batch_size, seq_len, d_latent)
        Q_latent = self.wq_latent(query)  # (batch_size, q_seq_len, d_latent)
        K_latent = self.wk_latent(key)    # (batch_size, k_seq_len, d_latent)
        V_latent = self.wv_latent(value)  # (batch_size, v_seq_len, d_latent)
        
        # Step 2: 从潜在空间映射到多头格式
        # (batch_size, seq_len, num_heads, head_dim)
        Q = self.wq_heads(Q_latent).view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        K = self.wk_heads(K_latent).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        V = self.wv_heads(V_latent).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        
        # Step 3: 转换维度以便计算注意力
        # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Step 4: 在潜在空间计算注意力分数（可选优化）
        # 也可以直接在潜在空间计算注意力，进一步降低复杂度
        # 这里我们使用标准的多头注意力计算
        
        # 计算注意力分数: (batch_size, num_heads, q_seq_len, k_seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, q_seq_len, k_seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Step 5: 合并多头
        # (batch_size, q_seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 输出投影
        output = self.wo(attn_output)
        
        return output
    
    def get_complexity_reduction(self):
        """计算复杂度降低比例"""
        # 标准注意力: O(n²·d)
        # MLA注意力: O(n²·d_latent) + O(n·d·d_latent) (投影开销)
        standard_cost = self.d_model
        latent_cost = self.d_latent + (self.d_model * self.d_latent) / (self.d_model * self.d_model)
        reduction = (standard_cost - latent_cost) / standard_cost
        return reduction


class OptimizedMLA(nn.Module):
    """
    优化版本的MLA：直接在潜在空间计算注意力
    进一步降低计算复杂度
    """
    
    def __init__(self, d_model, num_heads, d_latent=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        if d_latent is None:
            d_latent = max(d_model // 4, num_heads)
        self.d_latent = d_latent
        self.latent_head_dim = d_latent // num_heads
        
        assert d_latent % num_heads == 0, "d_latent must be divisible by num_heads"
        
        # 映射到潜在空间
        self.wq_latent = nn.Linear(d_model, d_latent)
        self.wk_latent = nn.Linear(d_model, d_latent)
        self.wv_latent = nn.Linear(d_model, d_latent)
        
        # 从潜在空间映射回原始空间
        self.wo_latent = nn.Linear(d_latent, d_model)
        
        # 输出投影
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.latent_head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        
        # 映射到潜在空间
        Q_latent = self.wq_latent(query)  # (batch_size, q_seq_len, d_latent)
        K_latent = self.wk_latent(key)    # (batch_size, k_seq_len, d_latent)
        V_latent = self.wv_latent(value)  # (batch_size, v_seq_len, d_latent)
        
        # 重塑为多头格式
        Q_latent = Q_latent.view(batch_size, q_seq_len, self.num_heads, self.latent_head_dim)
        K_latent = K_latent.view(batch_size, k_seq_len, self.num_heads, self.latent_head_dim)
        V_latent = V_latent.view(batch_size, k_seq_len, self.num_heads, self.latent_head_dim)
        
        # 转换维度
        Q_latent = Q_latent.transpose(1, 2)  # (batch_size, num_heads, q_seq_len, latent_head_dim)
        K_latent = K_latent.transpose(1, 2)
        V_latent = V_latent.transpose(1, 2)
        
        # 在潜在空间计算注意力
        attn_scores = torch.matmul(Q_latent, K_latent.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和（仍在潜在空间）
        attn_output_latent = torch.matmul(attn_weights, V_latent)
        # (batch_size, num_heads, q_seq_len, latent_head_dim)
        
        # 合并多头并映射回原始空间
        attn_output_latent = attn_output_latent.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_latent
        )
        
        # 从潜在空间映射回原始空间
        attn_output = self.wo_latent(attn_output_latent)
        
        # 输出投影
        output = self.wo(attn_output)
        
        return output


# 示例用法
if __name__ == "__main__":
    # 创建模型
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 128
    
    # 标准MLA
    mla = MultiHeadLatentAttention(d_model, num_heads, d_latent=128)
    
    # 优化版MLA
    optimized_mla = OptimizedMLA(d_model, num_heads, d_latent=128)
    
    # 创建输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output1 = mla(query, key, value)
    output2 = optimized_mla(query, key, value)
    
    print(f"输入形状: {query.shape}")
    print(f"标准MLA输出形状: {output1.shape}")
    print(f"优化MLA输出形状: {output2.shape}")
    print(f"\n复杂度降低比例: {mla.get_complexity_reduction():.2%}")
    
    # 计算参数量对比
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n标准MLA参数量: {count_parameters(mla):,}")
    print(f"优化MLA参数量: {count_parameters(optimized_mla):,}")
    
    # 计算FLOPs（简化版）
    # 标准注意力: O(n²·d)
    standard_flops = seq_len * seq_len * d_model
    # MLA: O(n²·d_latent) + O(n·d·d_latent)
    mla_flops = seq_len * seq_len * 128 + seq_len * d_model * 128
    reduction = (standard_flops - mla_flops) / standard_flops
    print(f"\n计算量降低比例: {reduction:.2%}")

