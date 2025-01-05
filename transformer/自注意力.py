import torch
import torch.nn as nn

# 假设输入嵌入
x = torch.rand(5, 10)  # (序列长度, 特征维度)

# 线性层生成Q、K、V
query_layer = nn.Linear(10, 10)
key_layer = nn.Linear(10, 10)
value_layer = nn.Linear(10, 10)

Q = query_layer(x)  # (5, 10)
K = key_layer(x)  # (5, 10)
V = value_layer(x)  # (5, 10)

# 计算注意力分数和加权值
attention_scores = torch.matmul(Q, K.T)  # (5, 5) a(i, i) 表示qi和ki的相似度
attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # (5, 5)
output = torch.matmul(attention_weights, V)  # (5, 5) * (5, 10) # 理解为推荐系统中 u-u 矩阵 × user-embedding矩阵
