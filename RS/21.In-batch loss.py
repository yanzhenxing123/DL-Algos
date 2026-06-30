import torch
import torch.nn.functional as F

torch.manual_seed(42)

# =========================
# 1. 假设 batch size = 4，embedding dim = 3
# =========================
B = 4
D = 3

# 模拟 user tower 输出的 user embedding: [B, D]
user_emb = torch.randn(B, D)

# 模拟 item tower 输出的 item embedding: [B, D]
# 这里默认 item_emb[i] 是 user_emb[i] 的正样本
item_emb = torch.randn(B, D)

print("user_emb shape:", user_emb.shape)
print("item_emb shape:", item_emb.shape)

# =========================
# 2. 归一化，变成 cosine similarity
# =========================
user_emb = F.normalize(user_emb, dim=-1)
item_emb = F.normalize(item_emb, dim=-1)

import pdb
pdb.set_trace()  # 断点，看看 user_emb 和 item_emb 的值

# =========================
# 3. 计算 B × B 的相似度矩阵
# =========================
temperature = 0.05

logits = user_emb @ item_emb.T / temperature

print("\nlogits shape:", logits.shape)
print("logits:")
print(logits)

# logits[i, j] 表示第 i 个 user 和第 j 个 item 的匹配分数
# 对第 i 行来说：
# logits[i, i] 是正样本分数
# logits[i, j], j != i 是负样本分数

# =========================
# 4. 构造 label
# =========================
labels = torch.arange(B)

print("\nlabels:")
print(labels)

# labels = [0, 1, 2, 3]
# 表示：
# 第 0 行的正确类别是第 0 列
# 第 1 行的正确类别是第 1 列
# 第 2 行的正确类别是第 2 列
# 第 3 行的正确类别是第 3 列

# =========================
# 5. cross entropy loss
# =========================
loss = F.cross_entropy(logits, labels)

print("\nloss:", loss.item())

# =========================
# 6. 看每个 user 对 batch 内 item 的 softmax 概率
# =========================
probs = F.softmax(logits, dim=1)

print("\nsoftmax probs:")
print(probs)

print("\npositive probabilities:")
print(probs[torch.arange(B), labels])