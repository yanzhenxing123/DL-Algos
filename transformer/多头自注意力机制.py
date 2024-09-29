import torch
import torch.nn as nn

# 假设输入嵌入
x = torch.rand(5, 10)  # (序列长度, 特征维度)

# 定义多头自注意力的参数
num_heads = 2  # 头的数量
d_model = 10    # 输入特征维度
d_k = d_model // num_heads  # 每个头的特征维度

# 线性层生成Q、K、V，注意这里的线性层需要考虑头的数量
query_layer = nn.Linear(d_model, d_model)
key_layer = nn.Linear(d_model, d_model)
value_layer = nn.Linear(d_model, d_model)

# 计算 Q、K、V
Q = query_layer(x)  # (5, 10)
K = key_layer(x)    # (5, 10)
V = value_layer(x)  # (5, 10)

# 将 Q、K、V 重塑为多头格式
Q = Q.view(5, num_heads, d_k)  # (5, 2, 5)
K = K.view(5, num_heads, d_k)  # (5, 2, 5)
V = V.view(5, num_heads, d_k)  # (5, 2, 5)

# 计算注意力分数
attention_scores = torch.matmul(Q, K.transpose(1, 2))  # (5, 2, 2)

# 计算注意力权重
attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (5, 2, 2)

# 计算加权值
output = torch.matmul(attention_weights, V)  # (5, 2, 5)

# 将多个头的输出合并
output = output.view(5, -1)  # (5, 10)

# 通过线性层映射回原始维度
final_output = nn.Linear(d_model, d_model)(output)  # (5, 10)

print(final_output)
