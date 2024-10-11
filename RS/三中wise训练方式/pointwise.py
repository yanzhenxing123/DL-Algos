

import torch
import torch.nn as nn
import torch.optim as optim


# 定义 Pointwise 模型
class PointwiseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(PointwiseModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)  # 输出评分

    def forward(self, user_id, item_id):
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)
        x = torch.cat([user_embed, item_embed], dim=1)
        return self.fc(x)  # 返回评分


# 定义 Pointwise 训练过程
def train_pointwise(model, criterion, optimizer, user_ids, item_ids, ratings):
    model.train()
    optimizer.zero_grad()
    output = model(user_ids, item_ids).squeeze()  # 预测评分
    loss = criterion(output, ratings)  # 计算MSE损失
    loss.backward()
    optimizer.step()
    return loss.item()


# 定义 Pointwise 推理过程
def predict_pointwise(model, user_id, all_item_ids):
    model.eval()
    with torch.no_grad():
        # 预测所有物品的评分
        scores = model(user_id.expand_as(all_item_ids), all_item_ids)
        sorted_scores, indices = torch.sort(scores, descending=True)
    return indices


# 初始化数据和模型
num_users, num_items, embedding_dim = 100, 100, 8
model = PointwiseModel(num_users, num_items, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 模拟训练数据
user_ids = torch.randint(0, num_users, (64,))
item_ids = torch.randint(0, num_items, (64,))
ratings = torch.rand(64) * 5

# 训练模型
for epoch in range(100):
    loss = train_pointwise(model, criterion, optimizer, user_ids, item_ids, ratings)
    print(f"Epoch {epoch}, Loss: {loss}")

# 推理：给用户推荐物品
user_id = torch.tensor([10])  # 假设要为用户10推荐物品
all_item_ids = torch.arange(num_items)  # 所有物品的ID
recommended_items = predict_pointwise(model, user_id, all_item_ids)
print(f"推荐物品ID顺序: {recommended_items[:5].tolist()}")
