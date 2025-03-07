import torch
import torch.nn.functional as F

# 假设有三个类别：书籍、电影、电子产品
num_classes = 3
batch_size = 2  # 假设有两个样本

# 模拟模型输出的logits（未经过softmax处理）
# 每个样本对应的三个类别的原始logits
logits = torch.tensor([[1.0, 2.0, 1.5],  # 样本1的logits [书籍, 电影, 电子产品]
                       [0.5, 1.5, 3.0]])  # 样本2的logits [书籍, 电影, 电子产品]

# 真实标签（类别索引）
# 样本1的真实类别是 "电影"， 样本2的真实类别是 "电子产品"
labels = torch.tensor([1, 2])  # 1 = 电影, 2 = 电子产品

# 1. 手动计算softmax
# softmax函数： e^(logit) / sum(e^(logit))，axis=1 表示按行计算
probabilities = F.softmax(logits, dim=1)
print("Softmax输出的概率：")
print(probabilities)

# 2. 选择每个样本的真实类别的概率
# 使用gather函数，按照标签的索引选出真实类别对应的概率
# .unsqueeze(1) 是为了让 labels 形状和 probabilities 匹配，便于gather操作
correct_class_probabilities = probabilities.gather(1, labels.unsqueeze(1))  # 索引的选择，选择真实label为1的，也就是第一个样本为电影，第二个样本为电子产品
print("每个样本真实类别对应的概率：")
print(correct_class_probabilities)

# 3. 计算交叉熵损失
# 损失函数： L = -log(p)，我们取概率的对数，再求平均
log_probabilities = torch.log(correct_class_probabilities)
print(f"log_probabilities: {log_probabilities}")
loss = -log_probabilities.mean()
print(f'手动计算的交叉熵损失: {loss.item()}')
