import os
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 1. 构造一个小模型
# =========================
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = SmallModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# =========================
# 2. 假数据，训练一步
# =========================
x = torch.randn(64, 10)
y = torch.randn(64, 1)

pred = model(x)
loss = ((pred - y) ** 2).mean()

loss.backward()        # 生成梯度，存在 p.grad
optimizer.step()       # AdamW 使用梯度更新权重，并更新 m/v
scheduler.step()

# 注意：这里先不要 optimizer.zero_grad()
# 因为如果 zero_grad 了，p.grad 就被清掉了，后面就看不到梯度了

step = 1


# =========================
# 3. 保存方式一：只保存模型权重
# =========================
torch.save(
    model.state_dict(),
    "only_weights.pt"
)


# =========================
# 4. 保存方式二：常规续训 checkpoint
#    保存：权重 + optimizer状态(m/v) + scheduler + step
#    通常不保存 gradient
# =========================
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
}, "train_checkpoint.pt")


# =========================
# 5. 保存方式三：强行保存“四个”
#    权重 + 梯度 + Adam一阶矩 + Adam二阶矩
# =========================

# 5.1 保存权重
weights = {
    name: param.detach().cpu().clone()
    for name, param in model.named_parameters()
}

# 5.2 保存梯度
grads = {
    name: param.grad.detach().cpu().clone()
    for name, param in model.named_parameters()
    if param.grad is not None
}

# 5.3 从 optimizer.state_dict() 里取 AdamW 的一阶矩和二阶矩
opt_state = optimizer.state_dict()

adam_m = {}
adam_v = {}

# optimizer.state_dict()["state"] 的 key 是参数编号，不是参数名
# 所以这里先建立 参数编号 -> 参数名 的映射
param_id_to_name = {}

param_groups = opt_state["param_groups"]
named_params = list(model.named_parameters())

# 对于这个简单模型，param_groups[0]["params"] 的顺序和 model.named_parameters() 顺序一致
for pid, (name, param) in zip(param_groups[0]["params"], named_params):
    param_id_to_name[pid] = name

for pid, state in opt_state["state"].items():
    name = param_id_to_name.get(pid, f"param_{pid}")

    if "exp_avg" in state:
        adam_m[name] = state["exp_avg"].detach().cpu().clone()

    if "exp_avg_sq" in state:
        adam_v[name] = state["exp_avg_sq"].detach().cpu().clone()

torch.save({
    "weights": weights,   # 权重 W
    "grads": grads,       # 梯度 grad
    "adam_m": adam_m,     # 一阶矩 m
    "adam_v": adam_v,     # 二阶矩 v
    "step": step,
}, "four_states.pt")


# =========================
# 6. 对比三个文件大小
# =========================
def file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


print("only_weights.pt      :", file_size_mb("only_weights.pt"), "MB")
print("train_checkpoint.pt  :", file_size_mb("train_checkpoint.pt"), "MB")
print("four_states.pt       :", file_size_mb("four_states.pt"), "MB")


# =========================
# 7. 看一下 optimizer 里到底有什么
# =========================
print("\nOptimizer state 示例：")
for pid, state in opt_state["state"].items():
    print("param id:", pid)
    print("state keys:", state.keys())
    break