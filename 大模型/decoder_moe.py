import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, k=1, capacity_factor=1.0, aux_loss_weight=0.01):
        """
        Sparse Mixture of Experts Layer.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Output dimension of each expert.
            num_experts (int): Number of experts.
            k (int): Number of experts to activate per input (top-k).
            capacity_factor (float): Factor to control expert capacity (not used in simplified version).
            aux_loss_weight (float): Weight for auxiliary load balancing loss.
        """
        super(SparseMoELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        # 专家网络: 每个专家是一个简单的FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ) for _ in range(num_experts)
        ])

        # 门控网络 (路由)
        self.gate = nn.Linear(input_dim, num_experts)

        # 辅助损失：用于负载均衡
        self.aux_loss = 0.0

    def forward(self, x):
        """
        Forward pass of Sparse MoE.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        batch_size = x.shape[0]
        # 1. 通过门控网络计算每个专家的权重
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # 2. Top-k 路由: 选择权重最高的k个专家
        topk_weights, topk_indices = torch.topk(gate_probs, k=self.k, dim=-1)  # [batch_size, k]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # 重新归一化

        # 3. 初始化输出张量
        output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # 4. 计算辅助损失（负载均衡）
        self._calculate_aux_loss(gate_probs)

        # 5. 对于每个输入，将其发送给top-k个专家，并加权求和
        for i in range(batch_size):
            for j in range(self.k):
                expert_idx = topk_indices[i, j].item()
                expert_output = self.experts[expert_idx](x[i].unsqueeze(0))  # [1, output_dim]
                output[i] += topk_weights[i, j] * expert_output.squeeze(0)

        return output

    def _calculate_aux_loss(self, gate_probs):
        """
        计算辅助损失以鼓励负载均衡。
        这里使用一个简单的损失：鼓励每个专家被选择的概率均值接近均匀分布。
        """
        # 计算每个专家在所有输入上的平均选择概率
        prob_mean_per_expert = gate_probs.mean(dim=0)  # [num_experts]
        # 理想情况：每个专家被选择的概率都是 1/num_experts
        ideal_prob = 1.0 / self.num_experts
        # 辅助损失：均方差损失
        self.aux_loss = self.aux_loss_weight * torch.sum((prob_mean_per_expert - ideal_prob) ** 2)

    def get_aux_loss(self):
        """返回辅助损失"""
        return self.aux_loss


# 示例用法
if __name__ == "__main__":
    # 设置随机种子以便重现
    torch.manual_seed(42)

    # 定义超参数
    input_dim = 10
    output_dim = 5
    num_experts = 4
    k = 2  # 每个输入使用2个专家
    batch_size = 3

    # 创建Sparse MoE层
    moe_layer = SparseMoELayer(input_dim, output_dim, num_experts, k=k)

    # 创建随机输入数据
    x = torch.randn(batch_size, input_dim)
    print("输入数据 x:")
    print(x)
    print(f"输入形状: {x.shape}\n")

    # 前向传播
    output = moe_layer(x)
    aux_loss = moe_layer.get_aux_loss()

    print("MoE输出:")
    print(output)
    print(f"输出形状: {output.shape}")
    print(f"辅助损失 (负载均衡): {aux_loss.item():.4f}")

    # 打印门控权重和选择情况
    with torch.no_grad():
        gate_logits = moe_layer.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_probs, k=k, dim=-1)

        print("\n门控网络详情:")
        for i in range(batch_size):
            print(f"输入 {i}:")
            print(f"  所有专家概率: {gate_probs[i]}")
            print(f"  选择的top-{k}专家索引: {topk_indices[i]}")
            print(f"  对应的权重: {topk_weights[i]}")
