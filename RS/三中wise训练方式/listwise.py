import torch
import torch.nn as nn
import torch.optim as optim

# 定义 Listwise 模型
class ListwiseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(ListwiseModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_ids):
        user_embed = self.user_embedding(user_id).unsqueeze(1)
        item_embed = self.item_embedding(item_ids)
        scores = torch.sum(user_embed * item_embed, # 64, 10, 8
                           dim=2)
        return scores  # 返回物品得分 # [64, 10]


# 定义 NDCG 损失函数
def ndcg_loss(predicted_scores, true_scores):
    sorted_true_scores, indices = torch.sort(true_scores, descending=True)
    sorted_predicted_scores = torch.gather(predicted_scores, 1, indices)

    rank = torch.arange(1, sorted_true_scores.size(1) + 1).float()
    dcg = torch.sum((2 ** sorted_predicted_scores - 1) / torch.log2(1 + rank), dim=1)
    ideal_dcg = torch.sum((2 ** sorted_true_scores - 1) / torch.log2(1 + rank), dim=1)

    ndcg = dcg / ideal_dcg
    return 1 - torch.mean(ndcg)  # 返回NDCG损失


# Listwise 训练过程
def train_listwise(model, optimizer, user_ids, item_lists, true_ranks):
    model.train()
    optimizer.zero_grad()
    predicted_scores = model(user_ids, item_lists)
    loss = ndcg_loss(predicted_scores, true_ranks)
    loss.backward()
    optimizer.step()
    return loss.item()


# Listwise 推理过程
def predict_listwise(model, user_id, all_item_ids):
    model.eval()
    with torch.no_grad():
        scores = model(user_id.expand_as(all_item_ids), all_item_ids)
        sorted_scores, indices = torch.sort(scores.squeeze(), descending=True)
    return indices


# 初始化数据和模型
num_users, num_items, embedding_dim = 100, 100, 8
model = ListwiseModel(num_users, num_items, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟训练数据
user_ids = torch.randint(0, num_users, (64,))
item_lists = torch.randint(0, num_items, (64, 10))
true_ranks = torch.rand(64, 10) * 5  # 模拟真实物品排序

# 训练模型
for epoch in range(5):
    loss = train_listwise(model, optimizer, user_ids, item_lists, true_ranks)
    print(f"Epoch {epoch}, NDCG Loss: {loss}")

# 推理：给用户推荐物品
user_id = torch.tensor([10])
all_item_ids = torch.arange(num_items)
recommended_items = predict_listwise(model, user_id, all_item_ids)
print(f"推荐物品ID顺序: {recommended_items[:5].tolist()}")
