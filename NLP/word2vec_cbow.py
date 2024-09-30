import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# 示例文本数据
sentences = [
    "I love natural language processing",
    "Word embeddings are great for NLP",
    "Word2Vec is a popular model",
    "I love learning about machine learning",
    "Natural language processing includes various tasks"
]

# 分词
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# 构建词汇表
all_words = [word for sentence in tokenized_sentences for word in sentence]
word_counts = Counter(all_words)
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}

# 打印词汇表
print("Vocabulary Size:", len(vocab))
print("Vocabulary:", vocab)

# 超参数
window_size = 2  # 上下文窗口大小

# 生成训练样本
training_data = []
for sentence in tokenized_sentences:
    for i, target_word in enumerate(sentence):
        start = max(0, i - window_size)
        end = min(len(sentence), i + window_size + 1)
        context_words = [sentence[j] for j in range(start, end) if j != i]
        # 这里生成的是 (上下文单词, 目标单词) 的形式
        training_data.append((context_words, target_word))

# 准备训练数据
context_words = []
target_words = []

for context, target in training_data:
    context_words.append([vocab[word] for word in context])  # 将上下文单词转换为索引
    target_words.append(vocab[target])  # 将目标单词转换为索引

# 检查目标单词的索引是否在有效范围内
for target in target_words:
    if target >= len(vocab):
        print(f"Invalid target index: {target}")

# 将上下文单词转换为张量
context_tensor = pad_sequence([torch.tensor(cw, dtype=torch.long) for cw in context_words], batch_first=True)
target_tensor = torch.tensor(target_words, dtype=torch.long)

# 查看填充后的上下文张量
print("Context Tensor:\n", context_tensor)
print("Target Tensor:\n", target_tensor)

# 现在可以继续定义和训练 CBOW 模型
embedding_dim = 10  # 嵌入维度
learning_rate = 0.01
epochs = 100


# 定义 CBOW 模型
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # 获取上下文单词的嵌入
        context_embeddings = self.embeddings(context)
        # 返回上下文单词的均值
        out = self.linear(context_embeddings)
        out = torch.mean(out, dim=1)
        out = nn.functional.softmax(out, dim=1)
        return out  # 返回平均的作为与目标计算的 [28, 22]


# 初始化模型和优化器
model = CBOWModel(len(vocab), embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(context_tensor)  # 获取上下文的嵌入均值 # torch.Size([28, 4])

    # 计算损失

    loss = loss_function(output, target_tensor)  # torch.Size([28, 10])

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 查看词向量
word_vectors = model.embeddings.weight.data
print("Word Embeddings:\n", word_vectors)
