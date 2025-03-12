"""
@Author: yanzx
@Date: 2025/3/12 10:29
@Description:
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(GCN, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, edge_index):
        # 初始化用户和物品的嵌入
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)

        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 分离用户和物品的嵌入
        user_emb_final, item_emb_final = torch.split(x, [self.user_embedding.num_embeddings, self.item_embedding.num_embeddings])

        return user_emb_final, item_emb_final


# 构建用户-物品交互图
def build_graph(num_users, num_items, interactions):
    # 交互数据：用户和物品的索引
    user_indices = torch.tensor(interactions[:, 0], dtype=torch.long)
    item_indices = torch.tensor(interactions[:, 1], dtype=torch.long) + num_users  # 偏移物品索引

    # 构建边
    edge_index = torch.stack([user_indices, item_indices], dim=0)

    # 构建图数据
    data = Data(edge_index=edge_index)
    return data


# 推荐分数计算
def recommend(user_emb, item_emb, user_id):
    # 计算用户与所有物品的内积
    scores = torch.matmul(user_emb[user_id], item_emb.t())
    return scores


# 示例数据
num_users = 10
num_items = 20
embedding_dim = 16
hidden_dim = 32
interactions = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])  # 用户-物品交互

# 构建图
graph = build_graph(num_users, num_items, interactions)

# 初始化模型
model = GCN(num_users, num_items, embedding_dim, hidden_dim)

# 前向传播
user_emb, item_emb = model(graph.edge_index)

# 为用户 0 推荐物品
user_id = 0
scores = recommend(user_emb, item_emb, user_id)
print("推荐分数:", scores)
