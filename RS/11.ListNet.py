"""
@Author: yanzx
@Date: 2025/3/7 11:09
@Description: 使用的方法是Listnet训练
"""

import torch
import torch.nn as nn
import torch.optim as optim


class ListNet(nn.Module):
    def __init__(self, input_dim):
        super(ListNet, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # 线性层用于预测得分

    def forward(self, x):
        return self.fc(x).squeeze()  # 返回物品得分


def plackett_luce(scores):
    # 计算Plackett-Luce排序概率
    return torch.softmax(scores, dim=-1)


def listnet_loss(pred_scores, true_scores):
    # 计算ListNet损失
    pred_probs = plackett_luce(pred_scores)
    true_probs = plackett_luce(true_scores)
    return -torch.sum(true_probs * torch.log(pred_probs))  # 交叉熵损失


# 示例数据
query_features = torch.randn(10, 5)  # 10个物品，每个物品5维特征
true_labels = torch.rand(10)  # 真实标签

# 模型训练
model = ListNet(input_dim=5)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    pred_scores = model(query_features)
    loss = listnet_loss(pred_scores, true_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
