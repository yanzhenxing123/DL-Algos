import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 1. 构造原始曝光数据
# =========================
np.random.seed(2026)
torch.manual_seed(2026)

n_pos = 100
n_neg = 9900

# 正样本
pos_df = pd.DataFrame({
    "feature": np.random.normal(loc=1.0, scale=1.0, size=n_pos),
    "click": 1
})

# 负样本
neg_df = pd.DataFrame({
    "feature": np.random.normal(loc=0.0, scale=1.0, size=n_neg),
    "click": 0
})

full_df = pd.concat([pos_df, neg_df], ignore_index=True)

print("原始数据：")
print(full_df["click"].value_counts())
print("真实 CTR =", full_df["click"].mean())


# =========================
# 2. 负采样：构造 1:4 训练集
# =========================
neg_sample_df = neg_df.sample(n=n_pos * 4, random_state=2026) # 选出来400个负样本

train_df = pd.concat([pos_df, neg_sample_df], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=2026).reset_index(drop=True)

print("\n负采样后的训练数据：")
print(train_df["click"].value_counts())
print("采样后 CTR =", train_df["click"].mean())


# =========================
# 3. 计算 sample_weight
# =========================
neg_sample_rate = len(neg_sample_df) / len(neg_df)

print("\n负样本采样率 =", neg_sample_rate)

train_df["sample_weight"] = 1.0
# 给负样本设置更大的权重
train_df.loc[train_df["click"] == 0, "sample_weight"] = 1.0 / neg_sample_rate

print("负样本权重 =", 1.0 / neg_sample_rate)


# =========================
# 4. 准备训练数据
# =========================
x = torch.tensor(train_df[["feature"]].values, dtype=torch.float32)
y = torch.tensor(train_df["click"].values, dtype=torch.float32).view(-1, 1)
w = torch.tensor(train_df["sample_weight"].values, dtype=torch.float32).view(-1, 1)


# =========================
# 5. 定义一个最简单的 LR 模型
# =========================
class LRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(use_weight=False, epochs=1000):
    model = LRModel()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(epochs):
        logits = model(x)

        loss_each = nn.functional.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none"
        )

        if use_weight:
            loss = (loss_each * w).sum() / w.sum()
        else:
            loss = loss_each.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# =========================
# 6. 分别训练：不加权 vs 加权
# =========================
model_no_weight = train_model(use_weight=False)
model_with_weight = train_model(use_weight=True)


# =========================
# 7. 在原始全量曝光数据上评估平均预测 CTR
# =========================
full_x = torch.tensor(full_df[["feature"]].values, dtype=torch.float32)

with torch.no_grad():
    pred_no_weight = torch.sigmoid(model_no_weight(full_x)).numpy()
    pred_with_weight = torch.sigmoid(model_with_weight(full_x)).numpy()

print("\n真实 CTR：", full_df["click"].mean())
print("不加 sample_weight 的平均预测 CTR：", pred_no_weight.mean())
print("加 sample_weight 的平均预测 CTR：", pred_with_weight.mean())


# =========================
# 8. 看一下模型参数
# =========================
print("\n不加 sample_weight 参数：")
print("weight =", model_no_weight.linear.weight.item())
print("bias   =", model_no_weight.linear.bias.item())

print("\n加 sample_weight 参数：")
print("weight =", model_with_weight.linear.weight.item())
print("bias   =", model_with_weight.linear.bias.item())