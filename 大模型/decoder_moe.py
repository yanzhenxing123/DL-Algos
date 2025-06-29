import torch
import torch.nn as nn
import torch.nn.functional as F


# 专家网络，假设每个专家是一个简单的全连接层
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))


# 门控网络，选择激活的专家
class GateNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)  # 使用softmax产生权重


# Transformer Decoder Layer with Mixture of Experts (MOE)
class DecoderLayerWithMOE(nn.Module):
    def __init__(self, d_model, num_experts, num_heads=8):
        super(DecoderLayerWithMOE, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        
        # 多头自注意力机制
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        
        # 专家网络 - 保持输入输出维度一致
        self.experts = nn.ModuleList([Expert(d_model, d_model) for _ in range(num_experts)])
        self.gate_network = GateNetwork(d_model, num_experts)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 输入格式: [seq_len, batch_size, d_model]
        # 自注意力计算
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)  # 残差连接 + Layer Norm

        # 门控网络选择专家
        gate_weights = self.gate_network(x)  # [seq_len, batch_size, num_experts]

        # 计算专家的输出并加权
        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=0)  # [num_experts, seq_len, batch_size, d_model]

        # 根据门控网络的权重选择专家
        gate_weights = gate_weights.permute(2, 0, 1).unsqueeze(-1)  # [num_experts, seq_len, batch_size, 1]
        expert_output = torch.sum(gate_weights * expert_outputs, dim=0)  # [seq_len, batch_size, d_model]
        
        x = self.norm2(x + expert_output)  # 残差连接 + Layer Norm

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)  # 残差连接 + Layer Norm
        
        return x


# 组合模型
class MOEDecoderModel(nn.Module):
    def __init__(self, d_model, num_experts, num_layers, num_heads=8):
        super(MOEDecoderModel, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayerWithMOE(d_model, num_experts, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 模型参数
d_model = 256  # 模型维度
num_experts = 4  # 专家数量
num_layers = 6   # 解码器层数
num_heads = 8    # 注意力头数

# 创建 MOE-Decoder 模型
model = MOEDecoderModel(d_model, num_experts, num_layers, num_heads)

# 输入数据: [seq_len, batch_size, d_model]
seq_len = 32
batch_size = 10
x = torch.randn(seq_len, batch_size, d_model)

print(f"Input shape: {x.shape}")
output = model(x)
print(f"Output shape: {output.shape}")

# 测试模型
print("\nModel test successful!")
print(f"Input: {x.shape}")
print(f"Output: {output.shape}")
