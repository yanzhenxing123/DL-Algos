import torch

# 创建一个二维张量
x = torch.tensor([[3, 1, 4],
                  [1, 5, 9],
                  [2, 6, 5]])

# 在每行中找出最大的 2 个元素
values, indices = torch.topk(x, k=2, dim=1)

print("输入张量:", x)
print("每行最大的 2 个元素:", values)
print("对应的索引:", indices)