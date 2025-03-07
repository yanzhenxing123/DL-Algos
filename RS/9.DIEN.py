"""
@Author: yanzx
@Date: 2025/2/26 15:13
@Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DIEN(nn.Module):
    def __init__(self, item_dim, hidden_size, embedding_dim, num_layers=1):
        super(DIEN, self).__init__()
        self.item_dim = item_dim  # 物品特征的维度
        self.hidden_size = hidden_size  # GRU隐藏层大小
        self.embedding_dim = embedding_dim  # Embedding维度
        self.num_layers = num_layers  # GRU层数

        # Embedding层
        self.item_embedding = nn.Embedding(item_dim, embedding_dim)

        # 兴趣提取层（GRU）
        self.interest_extractor = nn.GRU(
            input_size=embedding_dim,  # 这里改为embedding_dim
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 兴趣演化层（AUGRU）
        self.interest_evolver = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 注意力机制
        self.attention = nn.Linear(hidden_size * 2, 1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_behavior, target_item):
        """
        :param user_behavior: 用户行为序列 (batch_size, seq_len)
        :param target_item: 目标物品 (batch_size,)
        :return: 预测分数 (batch_size,)
        """
        batch_size, seq_len = user_behavior.size()

        # Embedding
        user_behavior_embed = self.item_embedding(user_behavior)  # (batch_size, seq_len, embedding_dim)
        target_item_embed = self.item_embedding(target_item)  # (batch_size, embedding_dim)

        # 兴趣提取层
        output, hidden = self.interest_extractor(user_behavior_embed)  # output: (batch_size, seq_len, hidden_size)  torch.Size([1, 32, 32])
        hidden = hidden[-1]  # 取最后一层的隐藏状态 (batch_size, hidden_size)

        # 兴趣演化层
        # 使用AUGRU（这里简化为GRU + Attention）
        output, _ = self.interest_evolver(output)  # output: (batch_size, seq_len, hidden_size)

        # 注意力机制
        attention_input = torch.cat([output, hidden.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)  # (batch_size, seq_len, hidden_size * 2)
        attention_score = self.attention(attention_input).squeeze(-1)  # (batch_size, seq_len)
        attention_weight = F.softmax(attention_score, dim=-1)  # (batch_size, seq_len)

        # 加权求和
        user_interest = torch.bmm(attention_weight.unsqueeze(1), output).squeeze(1)  # (batch_size, hidden_size)

        # 拼接用户兴趣和目标物品特征
        combined = torch.cat([user_interest, target_item_embed], dim=-1)  # (batch_size, hidden_size + embedding_dim)

        # 全连接层
        output = self.fc(combined)  # (batch_size, 1)

        return output.squeeze(-1)  # (batch_size,)


# 示例数据
batch_size = 32
seq_len = 10
item_dim = 100  # 假设物品ID范围为0-99
embedding_dim = 16
hidden_size = 32

user_behavior = torch.randint(0, item_dim, (batch_size, seq_len))  # 用户行为序列，作为输入
target_item = torch.randint(0, item_dim, (batch_size,))  # 目标物品， 作为物品的输入

# 初始化模型
model = DIEN(item_dim=item_dim, hidden_size=hidden_size, embedding_dim=embedding_dim)

# 前向传播
output = model(user_behavior, target_item)
print(output.shape)  # (batch_size,)
