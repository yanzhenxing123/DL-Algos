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
neg_sample_df = neg_df.sample(n=n_pos * 4, random_state=2026)  # 选出来400个负样本

train_df = pd.concat([pos_df, neg_sample_df], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=2026).reset_index(drop=True)

print("\n负采样后的训练数据：")
print(train_df["click"].value_counts())
print("采样后 CTR =", train_df["click"].mean())


# =========================
# 3. 计算负样本采样率
# =========================
neg_sample_rate = len(neg_sample_df) / len(neg_df)

print("\n负样本采样率 =", neg_sample_rate)
print("logit correction =", -np.log(neg_sample_rate))


# =========================
# 4. 准备训练数据
# =========================
x = torch.tensor(train_df[["feature"]].values, dtype=torch.float32)
y = torch.tensor(train_df["click"].values, dtype=torch.float32).view(-1, 1)

full_x = torch.tensor(full_df[["feature"]].values, dtype=torch.float32)


# =========================
# 5. 定义一个最简单的 LR 模型
# =========================
class LRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train_model(mode="no_correction", epochs=1000):
    """
    mode:
    - no_correction: 不做任何校正，直接用采样后的数据训练
    - logit_correction: 训练时做 logit correction，线上预测用 raw logit
    """
    model = LRModel()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    sample_rate_tensor = torch.tensor(neg_sample_rate, dtype=torch.float32)

    for epoch in range(epochs):
        raw_logits = model(x)

        if mode == "logit_correction":
            # 关键：训练 loss 使用校正后的 logit
            # q = neg_sample_rate
            # z_sample = z_true - log(q)
            logits_for_loss = raw_logits - torch.log(sample_rate_tensor)
        else:
            # 不校正，直接拿 raw_logits 去拟合采样后的训练分布
            logits_for_loss = raw_logits

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_for_loss,
            y,
            reduction="mean"
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# =========================
# 6. 分别训练：不校正 vs logit correction
# =========================
model_no_correction = train_model(mode="no_correction")
model_logit_correction = train_model(mode="logit_correction")


# =========================
# 7. 在原始全量曝光数据上评估平均预测 CTR
# =========================
with torch.no_grad():
    # 不校正模型：raw logit 直接预测
    pred_no_correction = torch.sigmoid(model_no_correction(full_x)).numpy()

    # logit correction 模型：
    # 注意：线上预测用 raw logit，不再减 log(q)
    raw_logits = model_logit_correction(full_x)
    pred_logit_correction_raw = torch.sigmoid(raw_logits).numpy()

    # 这个只是为了观察：如果你错误地用训练时校正后的 logit 做预测，会偏高
    pred_logit_correction_for_loss = torch.sigmoid(
        raw_logits - torch.log(torch.tensor(neg_sample_rate, dtype=torch.float32))
    ).numpy()

print("\n真实 CTR：", full_df["click"].mean())
print("不做校正的平均预测 CTR：", pred_no_correction.mean())
print("logit correction 后，线上 raw pred 平均预测 CTR：", pred_logit_correction_raw.mean())
print("logit correction 后，如果错误使用训练 logit 的预测 CTR：", pred_logit_correction_for_loss.mean())


# =========================
# 8. 看一下模型参数
# =========================
print("\n不做校正参数：")
print("weight =", model_no_correction.linear.weight.item())
print("bias   =", model_no_correction.linear.bias.item())

print("\nlogit correction 参数：")
print("weight =", model_logit_correction.linear.weight.item())
print("bias   =", model_logit_correction.linear.bias.item())