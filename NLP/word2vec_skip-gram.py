import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, center_word):
        # center_word: [batch_size]
        center_vec = self.embeddings(center_word)  # [batch, embed_dim]
        output = self.linear(center_vec)           # [batch, vocab_size]
        return output

def test_skipgram():
    print("\n=== Skip-gram 模型测试 ===")
    
    # 1. 准备词汇表和测试数据
    vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
    vocab_size = len(vocab)
    embedding_dim = 10
    
    # 2. 创建模型
    model = SkipGramModel(vocab_size, embedding_dim)
    
    # 3. 测试数据：中心词预测上下文
    # 句子: "the cat sat on the mat"
    # 对于中心词 "sat"，上下文是 ["the", "cat", "on", "mat"]
    center_word = torch.tensor([vocab["sat"]])  # 输入中心词
    
    # 在 Skip-gram 中，我们需要为每个上下文位置单独计算损失
    context_words = torch.tensor([
        vocab["the"], vocab["cat"], vocab["on"], vocab["mat"]
    ])  # 目标上下文词
    
    print(f"输入中心词形状: {center_word.shape}")    # torch.Size([1])
    print(f"目标上下文词形状: {context_words.shape}") # torch.Size([4])
    
    # 4. 前向传播测试
    output = model(center_word)
    print(f"模型输出形状: {output.shape}")  # torch.Size([1, 5])
    print(f"输出概率分布: {torch.softmax(output, dim=1)}")
    
    # 5. 训练测试 - Skip-gram 需要为每个上下文词计算损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    total_loss = 0
    for context_word in context_words:
        optimizer.zero_grad()
        output = model(center_word)
        
        # 为每个上下文词计算损失
        loss = criterion(output, context_word.unsqueeze(0))
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    print(f"平均损失 (4个上下文词): {total_loss / 4:.4f}")
    
    # 6. 测试批量数据
    print("\n--- 批量数据测试 ---")
    batch_centers = torch.tensor([vocab["sat"], vocab["cat"]])  # 两个中心词
    batch_output = model(batch_centers)
    print(f"批量输入形状: {batch_centers.shape}")  # torch.Size([2])
    print(f"批量输出形状: {batch_output.shape}")   # torch.Size([2, 5])
    
    # 7. 查看词向量
    print("\n--- 词向量测试 ---")
    with torch.no_grad():
        word_vectors = model.embeddings.weight
        print(f"词向量矩阵形状: {word_vectors.shape}")  # torch.Size([5, 10])
        
        # 获取特定词的向量
        sat_vector = model.embeddings(torch.tensor([vocab["sat"]]))
        on_vector = model.embeddings(torch.tensor([vocab["on"]]))
        print(f"'sat' 向量形状: {sat_vector.shape}")  # torch.Size([1, 10])
        print(f"'on' 向量形状: {on_vector.shape}")    # torch.Size([1, 10])
        
        # 计算相似度
        similarity = F.cosine_similarity(sat_vector, on_vector)
        print(f"'sat' 和 'on' 的余弦相似度: {similarity.item():.4f}")

# 运行 Skip-gram 测试
test_skipgram()