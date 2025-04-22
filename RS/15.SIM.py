"""
@Time: 2025/4/22 22:49
@Author: yanzx
@Description: 
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSearch(nn.Module):
    """基于属性（如商品类别）的精确匹配检索"""

    def __init__(self, key_dim: int):
        super().__init__()
        # 假设通过类别/ID等离散特征进行匹配
        self.key_dim = key_dim

    def forward(self, candidate_key: torch.Tensor, user_behavior_keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_key: (batch_size, key_dim)  候选商品的属性（如类别）
            user_behavior_keys: (batch_size, seq_len, key_dim) 用户历史行为的属性
        Returns:
            mask: (batch_size, seq_len) 相似行为的布尔掩码
        """
        # 精确匹配（如类别相同）
        mask = (user_behavior_keys == candidate_key.unsqueeze(1)).all(dim=-1)
        return mask


class SoftSearch(nn.Module):
    """基于向量相似度的Top-K检索"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self,
                candidate_emb: torch.Tensor,
                user_behavior_embs: torch.Tensor,
                k: int = 100) -> torch.Tensor:
        """
        Args:
            candidate_emb: (batch_size, embedding_dim)  候选商品向量
            user_behavior_embs: (batch_size, seq_len, embedding_dim) 用户行为序列向量
            k: 检索的Top-K数量
        Returns:
            topk_indices: (batch_size, k) 最相似行为的索引
        """
        # 计算余弦相似度
        sim = F.cosine_similarity(
            candidate_emb.unsqueeze(1),  # (batch_size, 1, embedding_dim) torch.Size([32, 1, 64])
            user_behavior_embs,  # (batch_size, seq_len, embedding_dim) torch.Size([32, 10000, 64])
            dim=-1
        )  # -> (batch_size, seq_len) torch.Size([32, 10000])

        # 取Top-K
        _, topk_indices = sim.topk(k=k, dim=-1)  # (batch_size, k) torch.Size([32, 100]) _ 表示分数
        return topk_indices


class InterestExtractor(nn.Module):
    """对检索出的子序列进行兴趣提取（类似DIN的注意力机制）"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(4 * embedding_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 1)
        )

    def forward(self,
                candidate_emb: torch.Tensor,
                user_behavior_subseq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_emb: (batch_size, embedding_dim) torch.Size([32, 64])
            user_behavior_subseq: (batch_size, k, embedding_dim) 检索出的子序列 torch.Size([32, 100, 64])
        Returns:
            user_interest: (batch_size, embedding_dim) 用户兴趣向量
        """
        # 扩展候选商品向量以匹配子序列维度
        candidate_expanded = candidate_emb.unsqueeze(1).expand(
            -1, user_behavior_subseq.size(1), -1
        )  # (batch_size, k, emb_dim) torch.Size([32, 100, 64]) user_behavior_subseq.size(1) =  1000

        # 注意力权重计算
        attention_input = torch.cat([
            candidate_expanded,  # torch.Size([32, 100, 64])
            user_behavior_subseq,  # torch.Size([32, 100, 64])
            candidate_expanded - user_behavior_subseq,  # torch.Size([32, 100, 64])
            candidate_expanded * user_behavior_subseq  # torch.Size([32, 100, 64])
        ], dim=-1)  # (batch_size, k, 4*emb_dim) # torch.Size([32, 100, 256])

        weights = self.attention(attention_input).squeeze(-1)  # (batch_size, k) torch.Size([32, 100])
        weights = F.softmax(weights, dim=-1)  # torch.Size([32, 100])

        # 加权求和得到兴趣向量
        user_interest = (
                weights.unsqueeze(-1)  # torch.Size([32, 100, 1])
                *
                user_behavior_subseq  # # torch.Size([32, 100, 64])
        ).sum(dim=1)  # torch.Size([32, 64]) 历史物品的向量都对应一个权重，乘积之后做加权求和
        return user_interest


class SIM(nn.Module):
    """两阶段SIM模型"""

    def __init__(self,
                 embedding_dim: int = 64,
                 key_dim: int = 16,
                 mode: str = "soft"):
        super().__init__()
        self.mode = mode  # "hard" 或 "soft"

        # 检索模块
        if mode == "hard":
            self.search = HardSearch(key_dim)
        else:
            self.search = SoftSearch(embedding_dim)

        # 兴趣提取模块
        self.interest_extractor = InterestExtractor(embedding_dim)

    def forward(self,
                candidate: torch.Tensor,
                user_behavior: torch.Tensor,
                behavior_keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            candidate: (batch_size, embedding_dim) 候选商品向量
            user_behavior: (batch_size, seq_len, embedding_dim) 用户行为序列
            behavior_keys: (batch_size, seq_len, key_dim) 仅hard模式需要
        """
        # 第一阶段：检索子序列
        if self.mode == "hard":
            assert behavior_keys is not None, "Hard search requires behavior keys!"
            mask = self.search(candidate[:, :self.search.key_dim], behavior_keys)
            subseq = user_behavior[mask].view(user_behavior.size(0), -1, user_behavior.size(-1))
        else:
            topk_indices = self.search(candidate, user_behavior)
            # gather 按照指定维度的索引（index）从输入张量中收集（gather）元素
            subseq = torch.gather(user_behavior, 1, topk_indices.unsqueeze(-1).expand(-1, -1, user_behavior.size(-1)))

        # 第二阶段：兴趣提取，得到用户向量
        interest = self.interest_extractor(candidate, subseq)
        return interest


class SIMWithCTR(nn.Module):
    def __init__(self, embedding_dim=64, key_dim=16, mode="soft"):
        super().__init__()
        # SIM核心模块
        self.sim = SIM(embedding_dim, key_dim, mode)  # 复用之前定义的SIM模型

        # CTR预测头
        self.ctr_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # 用户兴趣 + 候选商品拼接
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, candidate, user_behavior, behavior_keys=None):
        # 1. 提取用户兴趣
        user_interest = self.sim(candidate, user_behavior, behavior_keys)  # (batch_size, emb_dim)

        # 2. 拼接用户兴趣和候选商品特征
        ctr_input = torch.cat([user_interest, candidate], dim=-1)  # (batch_size, emb_dim * 2)

        # 3. 预测CTR
        ctr_pred = self.ctr_predictor(ctr_input)  # (batch_size, 1)
        return ctr_pred.squeeze(-1)  # (batch_size)


def compute_loss(ctr_pred, target_label):
    """
    Args:
        ctr_pred: (batch_size,) 模型预测的点击概率
        target_label: (batch_size,) 真实标签（0或1）
    """
    return F.binary_cross_entropy(ctr_pred, target_label.float())


# 训练步骤
def train_step(model, batch_data):
    candidate, user_behavior, labels = batch_data
    optimizer.zero_grad()

    # 前向传播
    ctr_pred = model(candidate, user_behavior)

    # 计算损失
    loss = compute_loss(ctr_pred, labels)

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    """
    在上述代码中，candidate的shape为 (batch_size, embedding_dim)，
    即假设每个样本（用户）对应一个候选商品。这是典型的Point-wise推荐范式（如CTR预估任务）。
    使用point wise进行训练
    """

    # 初始化模型和优化器
    model = SIMWithCTR(mode="soft")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟数据
    batch_size = 32
    embedding_dim = 64
    user_behavior = torch.randn(batch_size, 10000, embedding_dim)  # 用户行为序列
    candidate = torch.randn(batch_size, embedding_dim)  # 候选商品, item_emb
    labels = torch.randint(0, 2, (batch_size,))  # 真实标签（点击/未点击）

    # 训练循环
    for epoch in range(1000):
        loss = train_step(model, (candidate, user_behavior, labels))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
