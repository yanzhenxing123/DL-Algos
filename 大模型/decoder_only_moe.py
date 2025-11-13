"""
@Author: yanzx
@Time: 2025/1/6
@Description: Decoder-Only 模型结构，集成 MoE (Mixture of Experts)
类似 GPT 架构，但使用 MoE 替换 FFN 层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts Layer"""
    
    def __init__(self, d_model, num_experts=4, k=2, expert_dim=64, aux_loss_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.aux_loss_weight = aux_loss_weight
        
        # 专家网络: 每个专家是一个 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(num_experts)
        ]) # (512 -> 64 -> 512)
        
        # 门控网络 (路由)
        self.gate = nn.Linear(d_model, num_experts)
        
        # 辅助损失
        self.aux_loss = 0.0
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape # torch.Size([3, 10, 512])
        
        # 重塑为 (batch_size * seq_len, d_model)
        x_flat = x.view(-1, d_model)   # torch.Size([30, 512])
        
        # 1. 计算门控权重
        gate_logits = self.gate(x_flat)  # [batch * seq_len, num_experts] # torch.Size([30, 4])
        gate_probs = F.softmax(gate_logits, dim=-1) # torch.Size([30, 4])
        
        # 2. Top-k 路由
        topk_weights, topk_indices = torch.topk(gate_probs, k=self.k, dim=-1)  # k = 2, topk_weights.shape = torch.Size([30, 2])
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 3. 计算辅助损失
        self._calculate_aux_loss(gate_probs)
        
        # 4. 初始化输出
        output_flat = torch.zeros_like(x_flat) # torch.Size([30, 512])
        
        # 5. 对每个token，使用top-k专家
        for i in range(batch_size * seq_len): # 30
            for j in range(self.k): # k = 2
                expert_idx = topk_indices[i, j].item() 
                expert_output = self.experts[expert_idx](x_flat[i:i+1]) # torch.Size([1, 512])
                output_flat[i] += topk_weights[i, j] * expert_output.squeeze(0)
        
        # 重塑回原始形状
        output = output_flat.view(batch_size, seq_len, d_model)
        
        return output
    
    def _calculate_aux_loss(self, gate_probs):
        """计算负载均衡损失"""
        prob_mean_per_expert = gate_probs.mean(dim=0) # torch.Size([4])
        ideal_prob = 1.0 / self.num_experts # 1.0 / 4 = 0.25
        self.aux_loss = self.aux_loss_weight * torch.sum((prob_mean_per_expert - ideal_prob) ** 2) # 0.01 * torch.sum((prob_mean_per_expert - 0.25) ** 2)
    
    def get_aux_loss(self):
        return self.aux_loss


class MultiHeadAttention(nn.Module):
    """多头自注意力机制（带因果掩码）"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model) # 默认包含bias
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - 因果掩码
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # 计算 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        """
        方差变大会导致向量之间元素的差值变大, softmax后的结果 会将大的值趋近于1, 小的值趋近于0
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用因果掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        
        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        """
        1. 头间交互：没有 W_o，各头输出仅拼接，彼此独立；W_o 学习到不同头的信息融合与重加权。
        2. 表达增强：让注意力后还能做可学习的线性组合，提升表示灵活性。
        """
        output = self.W_o(attn_output) 
        
        
        return output


class DecoderBlock(nn.Module):
    """Decoder Block: Self-Attention + MoE"""
    
    def __init__(self, d_model, num_heads, num_experts=4, k=2, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # MoE 层（替换 FFN）
        self.moe = SparseMoELayer(d_model, num_experts, k, aux_loss_weight=0.01)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: 因果掩码
            
        过层 att + dropout + 残差 + norm + ffn(moe) + dropout + 残差 + norm  (postnorm)
        """
        # Self-attention with residual
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output)) 
        
        # MoE with residual
        moe_output = self.moe(x)
        x = self.norm2(x + self.dropout(moe_output))
        
        return x


class DecoderOnlyMoE(nn.Module):
    """Decoder-Only 模型，使用 MoE"""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 num_experts=4, k=2, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, num_experts, k, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len, device):
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)  # [1, seq_len, seq_len]
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) - token ids
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape # 3, 10
        device = input_ids.device # cpu
        
        # 创建位置编码
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        
        # Token embedding + Positional embedding
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # 通过 decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, causal_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def get_aux_loss(self):
        """获取所有 MoE 层的辅助损失"""
        total_aux_loss = 0.0
        for block in self.decoder_blocks:
            total_aux_loss += block.moe.get_aux_loss()
        return total_aux_loss


def main():
    """主函数演示"""
    print("Decoder-Only MoE 模型演示")
    print("="*50)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模型参数
    vocab_size = 30522
    d_model = 768
    num_heads = 4
    num_layers = 12
    num_experts = 4
    k = 2
    batch_size = 3
    seq_len = 10
    
    # 创建模型
    model = DecoderOnlyMoE(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_experts=num_experts,
        k=k
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    
    
    # 3. 遍历它，看看里面有什么
    # print("=== 显示每一层的名字和参数信息 ===")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {type(param)}, {param.shape}, {param.requires_grad}")
        
    # 创建随机输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n输入形状: {input_ids.shape}")
    print(f"输入示例:\n{input_ids}")
    
    # 前向传播
    with torch.no_grad():
        logits = model(input_ids)
        aux_loss = model.get_aux_loss()
    
    print(f"\n输出形状: {logits.shape}")
    print(f"辅助损失 (负载均衡): {aux_loss.item():.4f}")
    
    # 计算预测
    predictions = torch.argmax(logits, dim=-1)
    print(f"\n预测结果形状: {predictions.shape}")
    print(f"预测示例:\n{predictions}")
    
    print("\n" + "="*50)
    print("模型特点:")
    print("1. Decoder-Only 架构（类似 GPT）")
    print("2. 使用 MoE 替换 FFN 层")
    print("3. 支持因果掩码（自回归生成）")
    print("4. 包含负载均衡辅助损失")


if __name__ == "__main__":
    main()

