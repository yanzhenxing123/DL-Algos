"""
@Author: yanzx
@Date: 2025/3/7 11:53
@Description:
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
