import torch
import torch.nn.functional as F

"""
# 多分类问题
"""
teacher_logits = torch.tensor([[2.0, 1.0, 0.1]])  # 教师模型的输出
student_logits = torch.tensor([[1.5, 0.8, 0.3]])  # 学生模型的输出

# 计算 KL 散度
# 1. 对教师模型和学生模型的输出进行 softmax，得到概率分布
teacher_probs = F.softmax(teacher_logits, dim=-1)
student_log_probs = F.log_softmax(student_logits, dim=-1)

# 2. 计算 KL 散度
kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

print("KL Divergence:", kl_div.item())

"""
# 二分类问题
"""
import torch
import torch.nn.functional as F

# 假设教师模型和学生模型的输出是一维的
teacher_output = torch.tensor([0.8])  # 教师模型的输出
student_output = torch.tensor([0.6])  # 学生模型的输出

# 将一维输出转换为概率分布
# 使用 sigmoid 将输出映射到 [0, 1]
teacher_prob = torch.sigmoid(teacher_output)  # P
student_prob = torch.sigmoid(student_output)  # Q

# 对于二分类问题，KL 散度的公式为：
# KL(P || Q) = P * log(P / Q) + (1 - P) * log((1 - P) / (1 - Q))
kl_div = teacher_prob * (torch.log(teacher_prob) - torch.log(student_prob)) + \
         (1 - teacher_prob) * (torch.log(1 - teacher_prob) - torch.log(1 - student_prob))

print("KL Divergence:", kl_div.item())


"""
# 多维不调用包
"""
import torch

# 假设教师模型和学生模型的输出是多维的
teacher_logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.5]])  # 教师模型的输出
student_logits = torch.tensor([[1.5, 0.8, 0.3], [0.8, 1.5, 0.4]])  # 学生模型的输出


# 将 logits 转换为概率分布
def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)  # 数值稳定性
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


teacher_probs = softmax(teacher_logits)  # P
student_probs = softmax(student_logits)  # Q

# 手动计算 KL 散度
kl_div = teacher_probs * (torch.log(teacher_probs) - torch.log(student_probs))
kl_div = torch.sum(kl_div, dim=-1)  # 对每个样本的 KL 散度求和
kl_div_mean = torch.mean(kl_div)  # 对 batch 求平均

print("KL Divergence (Mean):", kl_div_mean.item())
