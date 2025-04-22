import torch
import torch.nn as nn
import torch.optim as optim

# 定义超参数
embedding_dim = 64  # 用户和物品的embedding维度
hidden_units = [128, 64]  # MLP隐藏层的维度
num_users = 1000  # 假设有1000个用户
num_items = 500  # 假设有500个物品
learning_rate = 0.001  # 学习率
epochs = 5  # 训练轮数
batch_size = 128  # 批大小


# 用户塔
class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim, hidden_units):
        super(UserTower, self).__init__()
        # 用户embedding
        self.embedding = nn.Embedding(num_users, embedding_dim)
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU()
        )

    def forward(self, user_input):
        user_emb = self.embedding(user_input).squeeze(1)  # (batch_size, embedding_dim)
        user_vector = self.mlp(user_emb)  # (batch_size, hidden_units[-1])
        return user_vector


# 物品塔
class ItemTower(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_units):
        super(ItemTower, self).__init__()
        # 物品embedding
        self.embedding = nn.Embedding(num_items, embedding_dim)
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU()
        )

    def forward(self, item_input=None):
        # 生成物品的embedding矩阵
        item_ids = torch.arange(num_items).unsqueeze(1)  # (num_items, 1)
        item_emb = self.embedding(item_ids).squeeze(1)  # (num_items, embedding_dim)
        item_vector = self.mlp(item_emb)  # (num_items, hidden_units[-1])
        return item_vector


# 双塔模型
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_units):
        super(TwoTowerModel, self).__init__()
        # 用户塔和物品塔
        self.user_tower = UserTower(num_users, embedding_dim, hidden_units)
        self.item_tower = ItemTower(num_items, embedding_dim, hidden_units)

    def forward(self, user_input):
        # 获取用户向量
        user_vector = self.user_tower(user_input)  # (batch_size, hidden_units[-1])
        # 获取所有物品向量
        item_vectors = self.item_tower()  # (num_items, hidden_units[-1])
        # 计算用户与所有物品的相似性（点积）
        scores = torch.matmul(user_vector, item_vectors.T)  # (batch_size, num_items)
        return scores  # 返回每个用户对所有物品的分数


# 创建模型
model = TwoTowerModel(num_users, num_items, embedding_dim, hidden_units)

# 生成模拟数据（训练）
user_ids = torch.randint(0, num_users, (batch_size, 1))
item_ids = torch.randint(0, num_items, (batch_size, 1))
labels = torch.randint(0, 2, (batch_size, 1)).float()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练部分略过，因为重点在推荐部分，模型已经有了

# 模拟用户ID输入进行推荐
model.eval()
test_user_ids = torch.tensor([[10], [20]])  # 假设有两个用户

# 对每个用户计算与所有物品的相似度得分
with torch.no_grad():
    scores = model(test_user_ids)  # (batch_size, num_items)
    # 对每个用户，找到得分最高的物品ID
    top_k = 5  # 推荐前5个物品
    recommended_items = torch.topk(scores, top_k, dim=1).indices  # 返回前K个物品ID

print("为每个用户推荐的物品ID：", recommended_items)
