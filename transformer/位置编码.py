import torch
import torch.nn as nn
import math

# 假设输入嵌入
x = torch.rand(5, 10)  # (序列长度, 特征维度)

# 定义多头自注意力的参数
num_heads = 2  # 头的数量
d_model = 10  # 输入特征维度
d_k = d_model // num_heads  # 每个头的特征维度


def get_positional_encoding(seq_len, d_model):
    """
    生成位置编码
    """
    positional_encoding = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    return positional_encoding


# 添加位置编码
positional_encoding = get_positional_encoding(x.size(0), d_model)  # (5, 10)
x += positional_encoding  # 将位置编码加到输入嵌入上

# 线性层生成Q、K、V
query_layer = nn.Linear(d_model, d_model)
key_layer = nn.Linear(d_model, d_model)
value_layer = nn.Linear(d_model, d_model)

# 计算 Q、K、V
Q = query_layer(x)  # (5, 10)
K = key_layer(x)  # (5, 10)
V = value_layer(x)  # (5, 10)

# 将 Q、K、V 重塑为多头格式
Q = Q.view(5, num_heads, d_k)  # (5, 2, 5)
K = K.view(5, num_heads, d_k)  # (5, 2, 5)
V = V.view(5, num_heads, d_k)  # (5, 2, 5)

# 计算注意力分数
attention_scores = torch.matmul(Q, K.transpose(1, 2))  # (5, 2, 5) * (5, 5, 2) = (5, 2, 2)

# 计算注意力权重
attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (5, 2, 2)

# 计算加权值
output = torch.matmul(attention_weights, V)  # (5, 2, 2) * (5, 2, 5) = (5, 2, 5)

# 将多个头的输出合并
output = output.view(5, -1)  # (5, 10)

# 通过线性层映射回原始维度
final_output = nn.Linear(d_model, d_model)(output)  # (5, 10)

print(final_output)
