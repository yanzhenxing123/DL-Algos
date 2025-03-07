"""
@Author: yanzx
@Date: 2025/3/2 19:30
@Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMoELayer(nn.Module):
    def __init__(self, feature_size, expert_num, expert_size, tower_size, gate_num):
        super(MMoELayer, self).__init__()
        """
        feature_size: 499
        """

        self.expert_num = expert_num  # 8
        self.expert_size = expert_size  # 16
        self.tower_size = tower_size  # 8
        self.gate_num = gate_num  # 2

        # 定义专家网络
        self._param_expert = nn.ModuleList([
            nn.Linear(feature_size, expert_size, bias=True) for _ in range(self.expert_num)
        ])
        # 初始化专家网络的权重和偏置
        for linear in self._param_expert:
            nn.init.constant_(linear.weight, 0.1)
            nn.init.constant_(linear.bias, 0.1)

        # 定义门控网络、塔网络和输出层
        self._param_gate = nn.ModuleList([
            nn.Linear(feature_size, expert_num, bias=True) for _ in range(self.gate_num)
        ])
        self._param_tower = nn.ModuleList([
            nn.Linear(expert_size, tower_size, bias=True) for _ in range(self.gate_num)
        ])
        self._param_tower_out = nn.ModuleList([
            nn.Linear(tower_size, 2, bias=True) for _ in range(self.gate_num)
        ])

        # 初始化门控网络、塔网络和输出层的权重和偏置
        for linear in self._param_gate + self._param_tower + self._param_tower_out:
            nn.init.constant_(linear.weight, 0.1)
            nn.init.constant_(linear.bias, 0.1)

    def forward(self, input_data):
        """
        input_data: Tensor(shape=[batch_size, feature_size]), e.g., shape=[2, 499]
        """
        expert_outputs = []
        # 计算每个专家的输出
        for expert in self._param_expert:
            linear_out = expert(input_data)  # Tensor(shape=[batch_size, expert_size])
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)

        # 拼接所有专家的输出
        expert_concat = torch.cat(expert_outputs, dim=1)  # Tensor(shape=[batch_size, expert_num * expert_size])
        expert_concat = expert_concat.view(-1, self.expert_num, self.expert_size)  # Tensor(shape=[batch_size, expert_num, expert_size])

        output_layers = []
        # 计算每个门控网络的输出
        for i in range(self.gate_num):  # gate_num gate_num = 2
            cur_gate_linear = self._param_gate[i](input_data)  # Tensor(shape=[batch_size, expert_num]) # [2, 3]
            cur_gate = F.softmax(cur_gate_linear, dim=1)  # Tensor(shape=[batch_size, expert_num])
            cur_gate = cur_gate.unsqueeze(2)  # Tensor(shape=[batch_size, expert_num, 1]) torch.Size([2, 3, 1])

            # 计算加权后的专家输出
            cur_gate_expert = expert_concat * cur_gate  # Tensor(shape=[batch_size, expert_num, expert_size])  torch.Size([2, 3, 16])
            cur_gate_expert = torch.sum(cur_gate_expert, dim=1)  # Tensor(shape=[batch_size, expert_size])  torch.Size([2, 16])
            # 通过塔网络
            cur_tower = self._param_tower[i](cur_gate_expert)  # Tensor(shape=[batch_size, tower_size])
            cur_tower = F.relu(cur_tower)

            # 通过输出层
            out = self._param_tower_out[i](cur_tower)  # Tensor(shape=[batch_size, 2])
            out = F.softmax(out, dim=1)

            # 防止数值下溢
            out = torch.clamp(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers


# 测试代码
if __name__ == "__main__":
    # 定义输入参数
    feature_size = 499
    expert_num = 4
    expert_size = 16
    tower_size = 8
    gate_num = 3
    batch_size = 32

    # 创建模型实例
    model = MMoELayer(feature_size, expert_num, expert_size, tower_size, gate_num)

    # 创建随机输入数据 (batch_size=2, feature_size=499)
    input_data = torch.randn(batch_size, feature_size)

    # 前向传播
    outputs = model(input_data)

    # 打印输出形状
    for i, out in enumerate(outputs):
        print(f"Output {i + 1} shape: {out.shape}")  # 预期输出形状: [2, 2]
