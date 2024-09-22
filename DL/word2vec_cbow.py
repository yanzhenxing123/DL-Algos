import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import numpy as np

# 定义训练数据
sentences = [
    "我 喜欢 自然语言处理",
    "我 也 喜欢 深度学习",
]


# 数据预处理
def preprocess_data(sentences):
    words = " ".join(sentences).split()
    vocab = list(set(words))  # 获取词汇表
    word2idx = {w: idx for idx, w in enumerate(vocab)}  # 词到索引的映射
    idx2word = {idx: w for idx, w in enumerate(vocab)}  # 索引到词的映射
    return words, vocab, word2idx, idx2word

# word2idx: word对应的索引
words, vocab, word2idx, idx2word = preprocess_data(sentences)
vocab_size = len(vocab)
embedding_dim = 100  # 词向量的维度
window_size = 2  # 上下文窗口大小


# 生成CBOW的训练样本
def generate_cbow_data(words, window_size):
    data = []
    for i in range(window_size, len(words) - window_size):
        context = [words[i - j] for j in range(window_size, 0, -1)] + \
                  [words[i + j] for j in range(1, window_size + 1)]
        target = words[i]
        data.append((context, target))
    return data


train_data = generate_cbow_data(words, window_size)


# 将上下文词转换为索引
def context_to_tensor(context, word2idx):
    return torch.tensor([word2idx[w] for w in context], dtype=torch.long)


# 定义CBOW模型
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        vocab_size: 5
        """
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.linear = nn.Linear(embedding_dim, vocab_size)  # 输出层

    def forward(self, context):
        """
        context: 是索引
        """
        # 查找上下文词的词向量，并取平均
        embeds = self.embeddings(context)
        avg_embed = torch.mean(embeds, dim=0).unsqueeze(0)  # 求均值并保持维度
        out = self.linear(avg_embed)  # 线性变换
        return out


# 初始化模型、损失函数和优化器
model = CBOWModel(vocab_size, embedding_dim)
loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练模型
def train_cbow(model, train_data, word2idx, idx2word, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for context, target in train_data:
            context_tensor = context_to_tensor(context, word2idx)  # 上下文转为索引
            target_tensor = torch.tensor([word2idx[target]], dtype=torch.long)  # 目标词的索引

            # 前向传播
            model.zero_grad()
            output = model(context_tensor)

            # 计算损失
            loss = loss_function(output, target_tensor)

            # 反向传播并更新参数
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss:.4f}')


# 运行训练
train_cbow(model, train_data, word2idx, idx2word)

# 测试模型词向量
word_embedding = model.embeddings(torch.tensor([word2idx["喜欢"]]))
print(f"'喜欢' 的词向量: {word_embedding}")
