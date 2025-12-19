import torch
import torch.nn as nn

# 创建一个简单网络
net = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# 输入需要梯度
x = torch.randn(2, 3, requires_grad=True)
print("=== 初始状态 ===")
print(f"初始 x:\n{x}")
print(f"x.requires_grad: {x.requires_grad}")

# 前向传播
output = net(x)
loss = output.sum()

# 反向传播前，梯度是None
print(f"\n反向传播前 x.grad: {x.grad}")

# 反向传播
loss.backward()
print(f"反向传播后 x.grad:\n{x.grad}")  # ✅ 现在有梯度了！

# 创建优化器，把 x 也加入优化
optimizer = torch.optim.SGD([
    {'params': net.parameters()},
    {'params': [x]}  # ⭐ 把 x 也加入优化！
], lr=0.1)

# 保存 x 的原始值
x_before = x.clone().detach()

# 更新参数（包括 x！）
optimizer.step()

print(f"\n=== 更新后 ===")
print(f"更新前 x:\n{x_before}")
print(f"更新后 x:\n{x}")
print(f"x 变化量:\n{x - x_before}")
print(f"✅ x 确实被改变了！")