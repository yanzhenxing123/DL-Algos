import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, context_words):
        # context_words: [batch_size, context_size]
        embeds = self.embeddings(context_words)  # [batch, context_size, embed_dim]
        context_vec = embeds.mean(dim=1)         # [batch, embed_dim] ← 求平均！
        output = self.linear(context_vec)        # [batch, vocab_size]
        return output

def test_cbow():
    print("=== CBOW 模型测试 ===")
    
    # 1. 准备词汇表和测试数据
    vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
    vocab_size = len(vocab)
    embedding_dim = 10
    context_size = 4  # 窗口左右各2个词
    
    # 2. 创建模型
    model = CBOWModel(vocab_size, embedding_dim)
    
    # 3. 测试数据：上下文预测中心词
    # 句子: "the cat sat on the mat"
    # 对于中心词 "sat"，上下文是 ["the", "cat", "on", "mat"]
    context_words = torch.tensor([
        [vocab["the"], vocab["cat"], vocab["on"], vocab["mat"]]  # 预测 "sat"
    ])
    center_word = torch.tensor([vocab["sat"]])  # 目标中心词
    
    print(f"输入上下文形状: {context_words.shape}")  # torch.Size([1, 4])
    print(f"目标中心词形状: {center_word.shape}")    # torch.Size([1])
    
    # 4. 前向传播测试
    output = model(context_words)
    print(f"模型输出形状: {output.shape}")  # torch.Size([1, 5])
    print(f"输出概率分布: {torch.softmax(output, dim=1)}")
    
    # 5. 训练测试
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 模拟训练步骤
    optimizer.zero_grad()
    loss = criterion(output, center_word)
    print(f"初始损失: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    
    # 6. 训练后再次测试
    output_after = model(context_words)
    loss_after = criterion(output_after, center_word)
    print(f"训练一次后损失: {loss_after.item():.4f}")
    
    # 7. 测试批量数据
    print("\n--- 批量数据测试 ---")
    batch_contexts = torch.tensor([
        [vocab["the"], vocab["cat"], vocab["on"], vocab["mat"]],  # 预测 "sat"
        [vocab["cat"], vocab["sat"], vocab["the"], vocab["mat"]],  # 预测 "on"
    ])
    batch_centers = torch.tensor([vocab["sat"], vocab["on"]])
    
    batch_output = model(batch_contexts)
    print(f"批量输入形状: {batch_contexts.shape}")  # torch.Size([2, 4])
    print(f"批量输出形状: {batch_output.shape}")    # torch.Size([2, 5])
    
    # 8. 查看词向量
    print("\n--- 词向量测试 ---")
    with torch.no_grad():
        word_vectors = model.embeddings.weight
        print(f"词向量矩阵形状: {word_vectors.shape}")  # torch.Size([5, 10])
        
        # 获取特定词的向量
        the_vector = model.embeddings(torch.tensor([vocab["the"]]))
        cat_vector = model.embeddings(torch.tensor([vocab["cat"]]))
        print(f"'the' 向量形状: {the_vector.shape}")  # torch.Size([1, 10])
        print(f"'cat' 向量形状: {cat_vector.shape}")  # torch.Size([1, 10])
        
        # 计算相似度
        similarity = F.cosine_similarity(the_vector, cat_vector)
        print(f"'the' 和 'cat' 的余弦相似度: {similarity.item():.4f}")

# 运行 CBOW 测试
test_cbow()