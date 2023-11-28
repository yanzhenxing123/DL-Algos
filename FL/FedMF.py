"""
@Time : 2023/11/24 15:44
@Author : yanzx
@Description : 
"""

import numpy as np


class FederatedALS:
    def __init__(self, num_users, num_items, num_factors=10, regularization=0.1, learning_rate=0.01, num_iterations=10):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # 初始化全局用户和物品矩阵
        self.global_user_matrix = np.random.rand(num_users, num_factors)
        self.global_item_matrix = np.random.rand(num_items, num_factors)

    def train_local_model(self, local_ratings, local_user_ids):
        local_user_matrix = self.global_user_matrix[local_user_ids, :]

        for iteration in range(self.num_iterations):
            # 更新局部物品矩阵
            local_item_matrix = np.linalg.solve(
                np.dot(local_user_matrix.T, local_user_matrix) + self.regularization * np.eye(self.num_factors),
                np.dot(local_user_matrix.T, local_ratings)
            )

            # 更新全局物品矩阵
            self.global_item_matrix += self.learning_rate * np.dot(local_item_matrix, local_user_matrix)

    def federated_train(self, devices_ratings, devices_user_ids):

        for device_ratings, device_user_ids in zip(devices_ratings, devices_user_ids):
            self.train_local_model(device_ratings, device_user_ids)


# 示例用法
num_users = 100
num_items = 50
num_devices = 5

# 模拟用户的评分数据分布在多个设备上
devices_ratings = [np.random.rand(len(range(i, num_users, num_devices)), num_items) for i in range(num_devices)]
devices_user_ids = [list(range(i, num_users, num_devices)) for i in range(num_devices)]

# 创建 FederatedALS 实例
federated_als = FederatedALS(num_users, num_items)

# 在所有设备上进行联邦训练
federated_als.federated_train(devices_ratings, devices_user_ids)

# 最终学到的全局 ALS 模型的参数
global_user_matrix = federated_als.global_user_matrix
global_item_matrix = federated_als.global_item_matrix
