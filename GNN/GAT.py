"""
@Author: yanzx
@Date: 2025/3/9 15:44
@Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # 线性变换
        self.attention = nn.Linear(2 * out_features, 1)  # 注意力机制

    def forward(self, x, adj):
        # x: 节点特征矩阵 (num_nodes, in_features)
        # adj: 邻接矩阵 (num_nodes, num_nodes)
        x = self.linear(x)  # 线性变换
        num_nodes = x.size(0)

        # 计算注意力系数
        attention_scores = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] == 1:  # 只计算有边的节点对
                    concat_features = torch.cat([x[i], x[j]], dim=-1)  # 拼接特征
                    score = self.attention(concat_features)  # 计算注意力分数
                    attention_scores.append(score)
        attention_scores = torch.stack(attention_scores)
        attention_scores = F.softmax(attention_scores, dim=0)  # 归一化

        # 聚合邻居信息
        output = torch.zeros_like(x)
        for i in range(num_nodes):
            neighbors = torch.where(adj[i] == 1)[0]  # 找到邻居节点
            for j in neighbors:
                print(j)
                print(attention_scores.shape)
                output[i] += attention_scores[j] * x[j]  # 加权聚合
        return F.relu(output)  # 激活函数


if __name__ == '__main__':
    # 假设有 3 个节点，每个节点有 2 维特征
    x = torch.tensor([[1.0, 2.0],
                      [2.0, 3.0],
                      [3.0, 4.0]])  # 节点特征
    adj = torch.tensor([[1, 1, 0],
                        [1, 1, 0],
                        [0, 1, 1]])  # 邻接矩阵

    gat_layer = GATLayer(in_features=2, out_features=2)
    output = gat_layer(x, adj)
    print(output)
