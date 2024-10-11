import torch
import torch.nn as nn
import torch.optim as optim


# 定义 BPR 模型
class BPRModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(BPRModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_i_id, item_j_id):
        user_embed = self.user_embedding(user_id)
        item_i_embed = self.item_embedding(item_i_id)
        item_j_embed = self.item_embedding(item_j_id)

        # 计算偏好分数
        pred_i = torch.sum(user_embed * item_i_embed, dim=1)
        pred_j = torch.sum(user_embed * item_j_embed, dim=1)
        return pred_i, pred_j

# 定义 BPR 损失
def bpr_loss(pred_i, pred_j):
    return -torch.mean(torch.log(torch.sigmoid(pred_i - pred_j)))

# BPR 训练过程
def train_bpr(model, optimizer, user_ids, item_i_ids, item_j_ids):
    model.train()
    optimizer.zero_grad()
    pred_i, pred_j = model(user_ids, item_i_ids, item_j_ids)
    loss = bpr_loss(pred_i, pred_j)
    loss.backward()
    optimizer.step()
    return loss.item()

# BPR 推理过程
def predict_bpr(model, user_id, all_item_ids):
    model.eval()
    with torch.no_grad():
        user_embed = model.user_embedding(user_id).unsqueeze(1)  # 用户嵌入
        item_embed = model.item_embedding(all_item_ids)  # 所有物品嵌入
        scores = torch.sum(user_embed * item_embed, dim=2)  # 计算偏好分数
        sorted_scores, indices = torch.sort(scores.squeeze(), descending=True)
    return indices

# 初始化数据和模型
num_users, num_items, embedding_dim = 100, 100, 8
model = BPRModel(num_users, num_items, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟训练数据
user_ids = torch.randint(0, num_users, (64,))
item_i_ids = torch.randint(0, num_items, (64,))
item_j_ids = torch.randint(0, num_items, (64,))

# 训练模型
for epoch in range(5):
    loss = train_bpr(model, optimizer, user_ids, item_i_ids, item_j_ids)
    print(f"Epoch {epoch}, BPR Loss: {loss}")

# 推理：给用户推荐物品
user_id = torch.tensor([10])  # 假设要为用户10推荐物品
all_item_ids = torch.arange(num_items)  # 所有物品ID
recommended_items = predict_bpr(model, user_id, all_item_ids)
print(f"推荐物品ID顺序: {recommended_items[:5].tolist()}")
