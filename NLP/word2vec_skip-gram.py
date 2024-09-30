# 示例文本数据
sentences = [
    "I love natural language processing",
    "Word embeddings are great for NLP",
    "Word2Vec is a popular model",
    "I love learning about machine learning",
    "Natural language processing includes various tasks"
]

import numpy as np
from collections import Counter

# 分词
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# 构建词汇表
all_words = [word for sentence in tokenized_sentences for word in sentence]
word_counts = Counter(all_words)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}

print(f"vocab: {vocab}")  # {word: id}

# 超参数
window_size = 2  # 上下文窗口大小

# 生成训练样本
training_data = []


def skip_gram():
    for sentence in tokenized_sentences:
        """
        skip-gram
        """
        for i, target_word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            context_words = [sentence[j] for j in range(start, end) if j != i]
            for context_word in context_words:
                training_data.append((target_word, context_word))


skip_gram()  #

# cbow() #

# 查看训练样本
print(training_data)

import torch
import torch.nn as nn
import torch.optim as optim

# 超参数
embedding_dim = 10  # 嵌入维度
learning_rate = 0.01
epochs = 100


# 定义 Skip-Gram 模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context):
        target_embeddings = self.embeddings(target)  # torch.Size([82, 10])
        context_embeddings = self.embeddings(context)  # torch.Size([82, 10])
        res = torch.sum(target_embeddings * context_embeddings, dim=1)
        scores = torch.matmul(target_embeddings, context_embeddings.t())  # torch.Size([82, 82])
        return scores


# 初始化模型和优化器
model = SkipGramModel(len(vocab), embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# 准备训练数据
target_words = []
context_words = []

for target, context in training_data:
    target_words.append(vocab[target])
    context_words.append(vocab[context])

target_tensor = torch.tensor(target_words, dtype=torch.long)
context_tensor = torch.tensor(context_words, dtype=torch.long)

# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    scores = model(target_tensor, context_tensor)

    # 计算损失
    loss = loss_function(scores, context_tensor)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 查看词向量
word_vectors = model.embeddings.weight.data
print("Word Embeddings:\n", word_vectors)
