"""
@Author: yanzx
@Time: 2025/1/6
@Description: DIN (Deep Interest Network) 简化实现
@Paper: Deep Interest Network for Click-Through Rate Prediction (KDD 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    """简化的注意力层"""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )
    
    def forward(self, behavior_emb, target_emb):
        """
        Args:
            behavior_emb: (batch_size, seq_len, embedding_dim) - 用户行为序列
            target_emb: (batch_size, embedding_dim) - 目标item
        Returns:
            attention_weights: (batch_size, seq_len) - 注意力权重
        """
        batch_size, seq_len, embedding_dim = behavior_emb.size()
        
        # 扩展target_emb到序列长度
        target_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 特征交互
        product = behavior_emb * target_expanded
        subtract = behavior_emb - target_expanded
        
        # 拼接特征
        concat_features = torch.cat([
            behavior_emb, target_expanded, product, subtract
        ], dim=-1)
        
        # 计算注意力权重
        attention_scores = self.attention_net(concat_features).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        return attention_weights


class SimpleDIN(nn.Module):
    """简化的DIN模型"""
    
    def __init__(self, num_users, num_items, embedding_dim=16):
        super().__init__()
        
        # Embedding层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 注意力层
        self.attention = SimpleAttention(embedding_dim)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),  # user + behavior_sequence
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids, behavior_sequences):
        """
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size,)
            behavior_sequences: (batch_size, seq_len)
        Returns:
            predictions: (batch_size, 1)
        """
        # 获取embedding
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        behavior_emb = self.item_embedding(behavior_sequences)
        
        # 计算注意力权重
        attention_weights = self.attention(behavior_emb, item_emb)
        
        # 加权平均得到行为序列表示
        behavior_sequence_emb = torch.sum(
            behavior_emb * attention_weights.unsqueeze(-1), dim=1
        )
        
        # 拼接用户embedding和行为序列embedding
        concat_features = torch.cat([user_emb, behavior_sequence_emb], dim=1)
        
        # 预测
        predictions = self.predictor(concat_features)
        
        return predictions


def main():
    """主函数演示"""
    print("简化版DIN模型演示")
    print("="*40)
    
    # 创建示例数据
    batch_size, seq_len = 4, 5
    num_users, num_items = 1000, 5000
    
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    behavior_sequences = torch.randint(0, num_items, (batch_size, seq_len))
    
    print(f"用户ID: {user_ids.shape}")
    print(f"商品ID: {item_ids.shape}")
    print(f"行为序列: {behavior_sequences.shape}")
    
    # 创建模型
    model = SimpleDIN(num_users, num_items, embedding_dim=16)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 前向传播
    with torch.no_grad():
        predictions = model(user_ids, item_ids, behavior_sequences)
    
    print(f"预测结果: {predictions.squeeze()}")
    
    print("\n" + "="*40)
    print("简化版DIN特点:")
    print("1. 注意力机制: 动态计算用户兴趣权重")
    print("2. 特征交互: 通过乘法和减法增强表达")
    print("3. 简洁架构: 保留核心思想，易于理解")


if __name__ == "__main__":
    main()
