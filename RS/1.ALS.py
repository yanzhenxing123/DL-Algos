"""
@Time : 2023/11/24 11:05
@Author : yanzx
@Description : ALS 固定一个 然后更新另一个


minimize usage:

返回值：
   success: True
   status: 0
      fun: 2.121930872819836
        x: [ 1.077e+00  5.312e-01 ...  1.948e+00  1.684e+00]
      nit: 29
      jac: [ 1.874e-05  6.963e-05 ...  6.617e-05 -1.581e-05]
     nfev: 555
     njev: 37


"""

import numpy as np
from scipy.optimize import minimize


class ALSMatrixFactorization:
    def __init__(self, num_users, num_items, num_factors=10, regularization=0.1):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.regularization = regularization

        # Initialize user and item matrices with random values
        self.user_matrix = np.random.rand(num_users, num_factors)
        self.item_matrix = np.random.rand(num_items, num_factors)

    def reshape_params(self, params):
        user_matrix = params[:self.num_users * self.num_factors].reshape((self.num_users, self.num_factors))
        item_matrix = params[self.num_users * self.num_factors:].reshape((self.num_items, self.num_factors))
        return user_matrix, item_matrix

    def cost_function(self, params, ratings):
        user_matrix, item_matrix = self.reshape_params(params)
        prediction = np.dot(user_matrix, item_matrix.T)
        error = prediction - ratings
        loss = 0.5 * np.sum(error ** 2) + 0.5 * self.regularization * (
                np.sum(user_matrix ** 2) + np.sum(item_matrix ** 2))
        return loss

    def train(self, ratings):
        initial_params = np.concatenate([self.user_matrix.flatten(), self.item_matrix.flatten()])
        result = minimize(self.cost_function, initial_params, args=(ratings,), method='L-BFGS-B')
        self.user_matrix, self.item_matrix = self.reshape_params(result.x)


def main():
    # 示例数据：用户-物品评分矩阵
    ratings = np.array([[1, 0, 3], [0, 4, 0], [2, 0, 4], [0, 3, 5]])

    # 创建并训练模型
    als_model = ALSMatrixFactorization(num_users=4, num_items=3, num_factors=2, regularization=0.1)
    als_model.train(ratings)


if __name__ == '__main__':
    main()
