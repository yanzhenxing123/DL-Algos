"""
@Author: yanzx
@Date: 2025/3/7 11:53
@Description:

一句话总结价值：让排序结果的 “优劣” 从 “凭感觉判断” 变成 “可量化计算”，支撑模型优化和效果迭代。
二、核心原理：NDCG 本质是 “给排序结果的‘合理性’打分”
用一句话说清核心逻辑：
NDCG 的本质是通过 “相关性分数” 和 “位置权重”，计算排序结果的 “累积有用性”，并进行归一化处理—— 相关度高的结果排在越靠前的位置，分数越高（范围 0~1，1 表示完美排序）。
拆解关键概念：
相关性（Relevance）：结果的 “有用程度”
通常用 0~k 的整数表示（如 0 = 不相关，1 = 一般相关，2 = 高度相关）；
例：搜索 “机器学习教程” 时，“权威教材链接” 相关性 = 2，“广告页面” 相关性 = 0。
增益（Gain）：单条结果的 “价值”
直接用相关性分数表示（或其对数形式，放大高相关结果的权重）；
例：相关性 = 2 的结果，增益 = 2；相关性 = 0 的结果，增益 = 0。
折损（Discount）：位置的 “重要性衰减”
越靠后的位置，权重越低（因为用户通常只看前几页结果）；
折损系数为 log2 (位置 + 1)（位置从 1 开始计数），即第 1 位折损 = log2 (2)=1，第 2 位 = log2 (3)≈1.58，位置越后折损越大。
累积增益（CG）：前 n 个结果的 “总价值”
公式：CG@n = 第 1 位增益 + 第 2 位增益 / 折损 + ... + 第 n 位增益 / 折损；
例：前 3 位结果相关性为 [2,1,0]，则 CG@3 = 2/1 + 1/1.58 + 0/2 ≈ 2 + 0.63 + 0 = 2.63。
归一化（Normalized）：消除 “结果数量差异” 的影响
理想情况下的最大 CG（IDCG，所有相关结果按最高相关性降序排列的 CG）作为分母；
NDCG@n = CG@n / IDCG@n，确保不同任务、不同长度的排序结果可对比。
三、计算步骤：用一个例子讲透（让听众 “跟着算一遍”）
以 “推荐系统推荐 5 部电影” 为例，演示 NDCG@5 的计算：
案例：
用户真实喜好（相关性）：电影 A=3（最爱），电影 B=2（喜欢），电影 C=1（一般），电影 D=0（不喜欢），电影 E=0（不喜欢）；
模型推荐排序：[B, A, C, D, E]（即第 1 位 B，第 2 位 A，第 3 位 C，第 4 位 D，第 5 位 E）。
计算步骤：
确定各位置的相关性和增益：
第 1 位 B：相关性 = 2 → 增益 = 2；
第 2 位 A：相关性 = 3 → 增益 = 3；
第 3 位 C：相关性 = 1 → 增益 = 1；
第 4 位 D：相关性 = 0 → 增益 = 0；
第 5 位 E：相关性 = 0 → 增益 = 0。
计算折损系数（log2 (位置 + 1)）：
第 1 位：log2 (2)=1；
第 2 位：log2 (3)≈1.58；
第 3 位：log2 (4)=2；
第 4 位：log2 (5)≈2.32；
第 5 位：log2 (6)≈2.58。
计算 CG@5（折损后累积增益）：
CG@5 = 2/1 + 3/1.58 + 1/2 + 0/2.32 + 0/2.58 ≈ 2 + 1.90 + 0.5 + 0 + 0 = 4.40。
计算 IDCG@5（理想情况下的最大 CG）：
理想排序是 [A, B, C, D, E]（按相关性降序）；
IDCG@5 = 3/1 + 2/1.58 + 1/2 + 0 + 0 ≈ 3 + 1.27 + 0.5 = 4.77。
计算 NDCG@5：
NDCG@5 = CG@5 / IDCG@5 ≈ 4.40 / 4.77 ≈ 0.92（即 92 分）。
结论：这个推荐排序虽然把 “最爱” 的 A 放在了第 2 位（略有偏差），但整体高相关结果靠前，NDCG 分数较高，符合 “排序较合理” 的直观感受。


"""

import numpy as np

np.random.seed(2021)


class Model:
    def __init__(self, k):
        self.k = k
        self.item_size = 50

    def __call__(self, users):
        # 模型随机返回 k 个 item,模拟推荐结果
        res = np.random.randint(0, self.item_size, users.shape[0] * self.k)
        return res.reshape((users.shape[0], -1))


def get_implict_matrix(rec_items, test_set):
    rel_matrix = [[0] * rec_items.shape[1] for _ in range(rec_items.shape[0])]
    for user in range(len(test_set)):
        for index, item in enumerate(rec_items[user]):
            if item in test_set[user]:
                rel_matrix[user][index] = 1
    return np.array(rel_matrix)


def DCG(items):
    return np.sum(items / np.log(np.arange(2, len(items) + 2)))


def nDCG(rec_items, test_set):
    assert rec_items.shape[0] == len(test_set)
    # 获得隐式反馈的rel分数矩阵
    rel_matrix = get_implict_matrix(rec_items, test_set)
    ndcgs = []
    for user in range(len(test_set)):
        rels = rel_matrix[user]
        dcg = DCG(rels)
        idcg = DCG(sorted(rels, reverse=True))
        ndcg = dcg / idcg if idcg != 0 else 0
        ndcgs.append(ndcg)
    return ndcgs


# 假设 top-20 推荐,一共 5 个 user, 50 个 item ,隐式反馈数据集.
users = np.array([0, 1, 2, 3, 4])
# test_set 表示 5 个用户在测试集中分表交互过那些 item
test_set = [
    [0, 21, 31, 41, 49],
    [2, 3, 4, 5, 33],
    [5, 10, 20, 30, 39, 44, 45, 49],
    [4, 7, 13, 15],
    [2]
]

model = Model(20)
rec_items = model(users)
ndcgs = nDCG(rec_items, test_set)
print(ndcgs)
