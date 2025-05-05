"""
@Time: 2025/5/5 15:22
@Author: yanzx
@Description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义线性层生成Q、K、V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """将输入张量分割为多头"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # 生成Q、K、V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 分割多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重到V
        output = torch.matmul(attn_weights, v)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 最终线性层
        output = self.output(output)

        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, mask)
        # 1. add & norm
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        # 2. add & norm
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        # 嵌入层
        x = self.embedding(x)

        # 位置编码
        x = self.pos_encoding(x)

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return x


# 使用示例
if __name__ == "__main__":
    # 参数设置
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048

    # 创建模型
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff)

    # 模拟输入 (batch_size, seq_len)
    input_tensor = torch.randint(0, vocab_size, (32, 100))

    # 前向传播
    output = encoder(input_tensor)
    print("Output shape:", output.shape)  # 应该输出: torch.Size([32, 100, 512])