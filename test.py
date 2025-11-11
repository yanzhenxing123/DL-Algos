import torch
import torch.nn as nn

# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(in_features=10, out_features=5) # 权重: (5, 10), 偏置: (5,)
        self.linear2 = nn.Linear(in_features=5, out_features=2)  # 权重: (2, 5),  偏置: (2,)
        self.linear3 = nn.Linear(in_features=100, out_features=100)  # 权重: (2, 5),  偏置: (2,)
        self.k = 10


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

model = SimpleModel()

# 2. 使用 model.parameters()
print("Type of model.parameters():", type(model.parameters()))
# 输出: <class 'generator'>

# 3. 遍历它，看看里面有什么
print("\n--- Iterating through parameters ---")
for param in model.parameters():
    print(type(param), param.shape, param.requires_grad)
    # requires_grad 为 True 表示该参数需要计算梯度，即可训练。

# 输出会类似于：
# <class 'torch.Tensor'> torch.Size([5, 10]) True  # linear1.weight
# <class 'torch.Tensor'> torch.Size([5]) True      # linear1.bias
# <class 'torch.Tensor'> torch.Size([2, 5]) True   # linear2.weight
# <class 'torch.Tensor'> torch.Size([2]) True      # linear2.bias

# 4. 将其转换为列表以查看所有参数（对于小模型可以这样做）
params_list = list(model.parameters())
print(f"\nTotal number of parameter tensors: {len(params_list)}")
# 输出: Total number of parameter tensors: 4

# 5. 计算模型的总参数量（一个非常常见的用法）
total_parameters = sum(p.numel() for p in model.parameters())
print(f"\nTotal number of parameters: {total_parameters}")
# 计算: (5*10 + 5) + (2*5 + 2) = (50+5) + (10+2) = 55 + 12 = 67
# 输出: Total number of parameters: 67

# 6. 只计算需要梯度的参数量（通常和总参数量一样）
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_parameters}")
# 输出: Total trainable parameters: 67