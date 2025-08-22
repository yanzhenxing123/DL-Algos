"""
@Author: yanzx
@Time: 2025/1/6
@Description: 推荐系统中序列建模的平均池化和最大池化示例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SequencePooling(nn.Module):
    """序列池化模块"""
    
    def __init__(self, pooling_type='mean'):
        super(SequencePooling, self).__init__()
        self.pooling_type = pooling_type
        
    def forward(self, sequence_embeddings, mask=None):
        """
        Args:
            sequence_embeddings: (batch_size, seq_len, embedding_dim)
            mask: (batch_size, seq_len) - 1表示有效位置，0表示padding位置
        Returns:
            pooled_embeddings: (batch_size, embedding_dim)
        """
        if mask is None:
            mask = torch.ones(sequence_embeddings.size(0), sequence_embeddings.size(1))
        
        if self.pooling_type == 'mean':
            return self.mean_pooling(sequence_embeddings, mask)
        elif self.pooling_type == 'max':
            return self.max_pooling(sequence_embeddings, mask)
        elif self.pooling_type == 'attention':
            return self.attention_pooling(sequence_embeddings, mask)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
    
    def mean_pooling(self, sequence_embeddings, mask):
        """平均池化"""
        # 将mask扩展到embedding维度
        mask_expanded = mask.unsqueeze(-1).expand_as(sequence_embeddings)
        
        # 应用mask并计算平均值
        masked_embeddings = sequence_embeddings * mask_expanded
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_lengths = mask.sum(dim=1, keepdim=True)
        
        # 避免除零
        seq_lengths = torch.clamp(seq_lengths, min=1)
        mean_embeddings = sum_embeddings / seq_lengths
        
        return mean_embeddings
    
    def max_pooling(self, sequence_embeddings, mask):
        """最大池化"""
        # 将mask扩展到embedding维度
        mask_expanded = mask.unsqueeze(-1).expand_as(sequence_embeddings)
        
        # 将padding位置的embedding设为很小的值
        masked_embeddings = sequence_embeddings * mask_expanded + (1 - mask_expanded) * -1e9
        
        # 取每个维度的最大值
        max_embeddings = torch.max(masked_embeddings, dim=1)[0]
        
        return max_embeddings
    
    def attention_pooling(self, sequence_embeddings, mask):
        """注意力池化（作为对比）"""
        # 简单的注意力机制
        attention_weights = torch.softmax(
            torch.sum(sequence_embeddings, dim=-1), dim=-1
        )
        
        # 应用mask
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / (torch.sum(attention_weights, dim=1, keepdim=True) + 1e-9)
        
        # 加权平均
        weighted_embeddings = sequence_embeddings * attention_weights.unsqueeze(-1)
        pooled_embeddings = torch.sum(weighted_embeddings, dim=1)
        
        return pooled_embeddings


class UserBehaviorModel(nn.Module):
    """用户行为序列建模示例"""
    
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128):
        super(UserBehaviorModel, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.sequence_pooling = SequencePooling(pooling_type='mean')
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_sequences, mask=None):
        """
        Args:
            user_sequences: (batch_size, seq_len) - 用户行为序列
            mask: (batch_size, seq_len) - 序列mask
        """
        # 获取item embeddings
        sequence_embeddings = self.item_embedding(user_sequences)
        
        # 序列池化
        pooled_embeddings = self.sequence_pooling(sequence_embeddings, mask)
        
        # 预测
        prediction = self.predictor(pooled_embeddings)
        
        return prediction


def compare_pooling_methods():
    """比较不同池化方法的效果"""
    
    # 模拟数据
    batch_size, seq_len, embedding_dim = 4, 6, 8
    sequence_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    # 创建mask（模拟不同长度的序列）
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],  # 长度为4
        [1, 1, 1, 1, 1, 1],  # 长度为6
        [1, 1, 0, 0, 0, 0],  # 长度为2
        [1, 1, 1, 1, 1, 0]   # 长度为5
    ])
    
    print("原始序列embeddings形状:", sequence_embeddings.shape)
    print("Mask形状:", mask.shape)
    print("\n" + "="*50)
    
    # 测试不同池化方法
    pooling_methods = ['mean', 'max', 'attention']
    
    for method in pooling_methods:
        print(f"\n{method.upper()} 池化:")
        pooling_layer = SequencePooling(pooling_type=method)
        pooled_result = pooling_layer(sequence_embeddings, mask)
        
        print(f"输出形状: {pooled_result.shape}")
        print(f"输出示例:\n{pooled_result[0][:5]}")  # 显示第一个样本的前5个维度
        
        # 分析池化效果
        if method == 'mean':
            print("平均池化特点: 保留整体分布信息")
        elif method == 'max':
            print("最大池化特点: 突出显著特征")
        elif method == 'attention':
            print("注意力池化特点: 自适应权重分配")


def demonstrate_user_model():
    """演示用户行为模型"""
    
    print("\n" + "="*50)
    print("用户行为模型演示")
    print("="*50)
    
    # 创建模型
    num_items = 1000
    model = UserBehaviorModel(num_items, embedding_dim=64, hidden_dim=128)
    
    # 模拟用户行为序列
    batch_size, seq_len = 2, 5
    user_sequences = torch.randint(0, num_items, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    
    print(f"用户序列形状: {user_sequences.shape}")
    print(f"用户序列示例: {user_sequences}")
    
    # 前向传播
    with torch.no_grad():
        predictions = model(user_sequences, mask)
    
    print(f"预测结果形状: {predictions.shape}")
    print(f"预测结果: {predictions.squeeze()}")


if __name__ == "__main__":
    print("推荐系统中序列建模的池化方法演示")
    print("="*50)
    
    # 比较不同池化方法
    compare_pooling_methods()
    
    # 演示用户模型
    demonstrate_user_model()
    
    print("\n" + "="*50)
    print("总结:")
    print("1. 平均池化: 适合需要全局特征的场景")
    print("2. 最大池化: 适合需要突出关键特征的场景") 
    print("3. 注意力池化: 适合需要自适应权重的场景")
    print("4. 实际应用中可以根据具体需求选择合适的池化方法")
