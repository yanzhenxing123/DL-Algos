"""
@Time : 2023/11/24 11:42
@Author : yanzx
@Description : 
"""

import numpy as np


class MatrixFactorization:
    def __init__(self, num_users, num_items, num_factors=10, learning_rate=0.01, regularization=0.1, epochs=100):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs

        # Initialize user and item matrices with random values
        self.user_matrix = np.random.rand(num_users, num_factors)
        self.item_matrix = np.random.rand(num_items, num_factors)

    def predict_rating(self, user_id, item_id):
        return np.dot(self.user_matrix[user_id], self.item_matrix[item_id])

    def train(self, ratings):
        for epoch in range(self.epochs):
            for user_id, item_id, rating in ratings:
                prediction = self.predict_rating(user_id, item_id)
                error = rating - prediction

                # Update user and item matrices using gradient descent
                self.user_matrix[user_id] += self.learning_rate * (
                            error * self.item_matrix[item_id] - self.regularization * self.user_matrix[user_id])
                self.item_matrix[item_id] += self.learning_rate * (
                            error * self.user_matrix[user_id] - self.regularization * self.item_matrix[item_id])


def main():
    # 示例数据：(user_id, item_id, rating)
    ratings = [(0, 0, 5), (1, 1, 4), (2, 2, 3), (0, 2, 3)]

    # 创建并训练模型
    mf_model = MatrixFactorization(num_users=3, num_items=3, num_factors=2, learning_rate=0.01, regularization=0.1,
                                   epochs=100)
    mf_model.train(ratings)


if __name__ == '__main__':
    main()
