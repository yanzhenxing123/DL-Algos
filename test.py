
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, dot

# 示例文本数据
texts = [
    "the quick brown fox jumps over the lazy dog",
    "I love natural language processing",
    "word embeddings are useful in many NLP tasks"
]

# 初始化 Tokenizer 并进行文本标记化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)  # 将文本数据拟合到Tokenizer中，构建词汇表
word2id = tokenizer.word_index  # 获取词汇到索引的映射字典
id2word = {v: k for k, v in word2id.items()}  # 获取索引到词汇的映射字典
vocab_size = len(word2id) + 1  # 词汇表大小，加1是为了包括索引0

# 将文本转换为序列
sequences = tokenizer.texts_to_sequences(texts)  # 将每个文本转化为对应的词汇索引序列
print(sequences)

# 准备 skip-gram 数据
window_size = 2  # 窗口大小，决定了上下文词汇的范围
pairs = []  # 用于存储词对
labels = []  # 用于存储标签（1表示真实的上下文对，0表示负采样的对）
for sequence in sequences:
    sg_pairs, sg_labels = skipgrams(sequence, vocabulary_size=vocab_size, window_size=window_size, negative_samples=1.0)
    # skipgrams函数生成中心词和上下文词对及对应标签
    pairs.extend(sg_pairs)  # 将生成的词对添加到pairs列表中
    labels.extend(sg_labels)  # 将生成的标签添加到labels列表中

pairs = np.array(pairs)  # 将词对转换为numpy数组
labels = np.array(labels)  # 将标签转换为numpy数组

# CBOW 模型定义
embedding_dim = 50  # 词向量的维度

input_target = Input((1,))  # 输入的目标词
input_context = Input((1,))  # 输入的上下文词

embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1, name='embedding')
# Embedding层，用于将词汇索引映射到词向量

target_embedding = embedding(input_target)  # 获取目标词的词向量
target_embedding = Reshape((embedding_dim, 1))(target_embedding)  # 调整词向量的形状以便点积计算

context_embedding = embedding(input_context)  # 获取上下文词的词向量
context_embedding = Reshape((embedding_dim, 1))(context_embedding)  # 调整词向量的形状以便点积计算

# 点积层
dot_product = dot([target_embedding, context_embedding], axes=1)  # 计算目标词和上下文词的点积
dot_product = Reshape((1,))(dot_product)  # 调整点积的形状

# 输出层
output = Dense(1, activation='sigmoid')(dot_product)  # 输出层，使用sigmoid激活函数，将点积结果转换为概率

# 定义模型
model = Model(inputs=[input_target, input_context], outputs=output)  # 定义模型，指定输入和输出
model.compile(loss='binary_crossentropy', optimizer='adam')  # 编译模型，使用二元交叉熵损失和adam优化器
model.summary()  # 输出模型摘要信息

# 将数据分成输入和输出
target_words, context_words = pairs[:, 0], pairs[:, 1]  # 分割词对数据为目标词和上下文词

# 训练模型
model.fit([target_words, context_words], labels, epochs=10, batch_size=64)  # 训练模型，指定输入、输出、训练轮数和批次大小

# 获取词向量
word_embeddings = model.get_layer('embedding').get_weights()[0]  # 从模型的嵌入层中获取训练好的词向量
print(word_embeddings)