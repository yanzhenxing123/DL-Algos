"""
DL-Algos.LLM.1.1.gate_mlp 的 Docstring


不同于 sparse moe，
- gate mlp每个token的所有神经元都进行计算
- moe是每个token经过router选择路由后只激活部分专家的神经元进行计算


token hidden state
→ router 计算 expert 分数 （hidden_size × num_experts）
→ top-k 选 expert
→ token 只送进被选中的 expert
→ expert 输出按 router 分数加权合并


"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMLP(nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=8):
        super().__init__()

        # 生成门控信号：决定哪些维度打开/关闭
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # 生成候选内容：真正要被处理的信息
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # 把中间维度降回 hidden_size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))   # [batch, intermediate_size]
        up = self.up_proj(x)               # [batch, intermediate_size]

        hidden = gate * up                 # 门控：逐元素相乘
        out = self.down_proj(hidden)       # [batch, hidden_size]

        return out, gate, up, hidden


# 假设一个 token 的 hidden 表示是 4 维
x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])

mlp = GatedMLP(hidden_size=4, intermediate_size=8)

out, gate, up, hidden = mlp(x)

print("输入 x:")
print(x)

print("\ngate = silu(gate_proj(x)):")
print(gate)

print("\nup = up_proj(x):")
print(up)

print("\nhidden = gate * up:")
print(hidden)

print("\nout = down_proj(hidden):")
print(out)