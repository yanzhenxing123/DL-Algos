import torch
import torch.nn.functional as F

# 假设目标句子的最大长度 T = 5
T = 5

# 假设我们已经生成了前面的一些词（如第一个词 "The"），需要推理生成剩余的部分
generated_words = ["The"]  # 当前已经生成的词，开始时为空或者包含第一个词


# 创建一个空的 mask 矩阵，推理时会逐步更新
def create_mask(t):
    mask = torch.triu(torch.ones(t, t), diagonal=1)  # 创建一个上三角矩阵，diagonal=1表示从对角线下方为0，代表“未来”的位置
    mask = mask == 1  # 将上三角部分标记为True
    mask = mask.to(torch.float32) * float('-inf')  # 将True的位置设置为-∞，后续的softmax会将它们屏蔽掉
    return mask


# 假设每次生成一个词后，通过编码器得到一个新的上下文（这里简化为随机矩阵）
def generate_next_word(context, mask):
    # 假设 context 是一个来自编码器的上下文表示，注意力分数是通过这个上下文计算的
    attention_scores = torch.rand(T, T)  # 随机生成一个注意力分数矩阵

    # 在自注意力中应用mask
    masked_attention_scores = attention_scores + mask  # 应用mask，未来的词的分数变为负无穷

    # 使用softmax进行归一化，计算注意力权重
    attention_probs = F.softmax(masked_attention_scores, dim=-1)

    return attention_probs


# 模拟逐步生成
for t in range(1, T):  # 假设已经生成了第1个词，从第2个词开始生成
    # 更新mask矩阵
    mask = create_mask(t)
    print(f"Mask for step {t + 1}:")
    print(mask)

    # 在生成每个词时，使用当前生成的词和编码器上下文来生成下一个词
    context = torch.rand(T, T)  # 假设从编码器获得的上下文信息
    attention_probs = generate_next_word(context, mask)

    # 打印每个步骤的注意力概率（可以理解为模型对每个位置的注意力分布）
    print(f"Attention probabilities for step {t + 1}:")
    print(attention_probs)

    # 假设生成一个词并添加到生成的序列中
    generated_words.append(f"word_{t + 1}")

print(f"Generated words: {generated_words}")
