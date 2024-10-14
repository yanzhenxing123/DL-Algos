import gensim
from gensim.models import Word2Vec
import jieba

# 准备训练数据
sentences = [
    "基于图卷积神经网络的推荐算法被应用到了联邦学习中",
    "用户隐私保护的重要性日益凸显"
]

# 使用 jieba 进行中文分词
tokenized_sentences = [jieba.lcut(sentence) for sentence in sentences]

print(tokenized_sentences)

# 训练 Word2Vec 模型
model = Word2Vec(tokenized_sentences,
                 vector_size=100,  # 词向量维度
                 window=5,  # 窗口大小
                 min_count=1,  # 忽略频率低于该值的词
                 workers=4  # 并行训练的线程数
                 )

print(model.wv.key_to_index)
# 获取某个单词的词向量
word_vector = model.wv['联邦']  # 例如获取“联邦学习”的词向量
print(word_vector)
