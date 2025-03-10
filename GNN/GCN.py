"""
@Author: yanzx
@Date: 2025/3/9 15:44
@Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # 线性变换

    def forward(self, x, adj):
        # x: 节点特征矩阵 (num_nodes, in_features)
        # adj: 归一化的邻接矩阵 (num_nodes, num_nodes)
        x = self.linear(x)  # 线性变换
        x = torch.matmul(adj, x)  # 邻接矩阵聚合邻居信息
        return F.relu(x)  # 激活函数


if __name__ == '__main__':
    # 假设有 3 个节点，每个节点有 2 维特征
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # 节点特征
    adj = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])  # 归一化邻接矩阵

    gcn_layer = GCNLayer(in_features=2, out_features=2)
    output = gcn_layer(x, adj)
    print(output)
