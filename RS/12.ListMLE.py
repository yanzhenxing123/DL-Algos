"""
@Author: yanzx
@Date: 2025/3/7 11:18
@Description:
"""

import torch
import torch.nn as nn
import torch.optim as optim


class ListMLE(nn.Module):
    def __init__(self, input_dim):
        super(ListMLE, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # 线性层用于预测得分

    def forward(self, x):
        return self.fc(x).squeeze()  # 返回物品得分


def plackett_luce(scores, true_ranking):
    # 计算Plackett-Luce似然概率
    log_likelihood = 0.0
    for i in range(len(true_ranking)):
        log_likelihood += scores[true_ranking[i]] - torch.logsumexp(scores[true_ranking[i:]], dim=0)
    return -log_likelihood  # 返回负对数似然


# 示例数据
query_features = torch.randn(10, 5)  # 10个物品，每个物品5维特征
true_ranking = torch.tensor([2, 0, 1, 3, 4, 5, 6, 7, 8, 9])  # 真实排序

# 模型训练
model = ListMLE(input_dim=5)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    pred_scores = model(query_features)
    loss = plackett_luce(pred_scores, true_ranking)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
