import nltk
from nltk.util import ngrams
from collections import Counter

# 示例句子
sentence = "I love natural language processing and machine learning"

# 将句子拆分为单词
tokens = nltk.word_tokenize(sentence)

# 生成 bigrams (2-grams)
bigrams = list(ngrams(tokens, 2))

# 输出 bigram
print("Bigrams:", bigrams)

# 统计每个 bigram 的出现频率
bigram_freq = Counter(bigrams)

# 打印频率
print("Bigram Frequency:", bigram_freq)
