import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """
    MQA: 多头查询注意力，但只有单头键值注意力
    所有查询头共享同一套键和值投影
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # MQA 关键区别：K 和 V 的投影输出维度小得多
        self.wq = nn.Linear(d_model, d_model)  # 输出: d_model = num_heads * head_dim
        self.wk = nn.Linear(d_model, self.head_dim)  # 输出: head_dim (只有1个头!)
        self.wv = nn.Linear(d_model, self.head_dim)  # 输出: head_dim (只有1个头!)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, past_kv=None, use_cache=False):
        """
        Args:
            query: (batch_size, q_seq_len, d_model)
            key: (batch_size, k_seq_len, d_model)  
            value: (batch_size, v_seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) 可选
            past_kv: 推理时传入的过去KV缓存
            use_cache: 是否返回KV缓存用于推理
        """
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)

        # 1. 线性投影
        # Q: (batch_size, q_seq_len, num_heads, head_dim)
        Q = self.wq(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        
        # K, V: (batch_size, k_seq_len, head_dim) - 注意这里没有num_heads维度！
        K = self.wk(key)  # 形状: (batch_size, k_seq_len, head_dim)
        V = self.wv(value)  # 形状: (batch_size, v_seq_len, head_dim)

        # 2. 处理KV缓存（推理优化）
        if past_kv is not None:
            # 拼接过去的KV
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=1)
            V = torch.cat([past_v, V], dim=1)
            k_seq_len = K.size(1)

        # 保存当前的KV用于下次推理
        present_kv = (K, V) if use_cache else None

        # 3. 转换维度并扩展K,V以匹配Q的头数
        # Q: (batch_size, num_heads, q_seq_len, head_dim)
        Q = Q.transpose(1, 2)
        
        # K, V: 增加num_heads维度并扩展
        # 从 (batch_size, k_seq_len, head_dim) 
        # 到 (batch_size, num_heads, k_seq_len, head_dim)
        K = K.unsqueeze(1).expand(batch_size, self.num_heads, k_seq_len, self.head_dim)
        V = V.unsqueeze(1).expand(batch_size, self.num_heads, k_seq_len, self.head_dim)

        # 4. 计算缩放点积注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # 确保mask形状正确: (batch_size, 1, q_seq_len, k_seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 输出形状: (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # 5. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )

        # 输出投影
        output = self.wo(attn_output)
        
        return (output, present_kv) if use_cache else output

    def get_kv_cache_size(self, seq_len, batch_size=1, dtype_size=4):
        """计算KV缓存的内存占用（字节）"""
        # 每个KV的形状: (batch_size, seq_len, head_dim)
        per_tensor_size = batch_size * seq_len * self.head_dim * dtype_size
        return 2 * per_tensor_size  # K和V