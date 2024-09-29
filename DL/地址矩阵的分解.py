import numpy as np

# 初始化矩阵 W
W = np.array([[4, 3, 2, 1],
              [2, 2, 2, 2],
              [1, 3, 4, 2],
              [0, 1, 2, 3]])
# 矩阵维度
d = W.shape[0]  # 4

# 秩
r = 2

# 随机初始化 A 和 B
np.random.seed(666)

# A 和 B 的元素服从标准正态分布
A = np.random.randn(d, r)
B = np.zeros((r, d))

# 定义超参数
lr = 0.01  # 学习率，用于控制梯度下降的步长。

epochs = 1000  # 迭代次数，进行多少次梯度下降更新。


# 定义损失函数
def loss_function(W, A, B):
    '''
    W：目标矩阵
    A：矩阵分解中的一个矩阵，通常是随机初始化的。
    B：矩阵分解中的另一个矩阵，通常是零矩阵初始化的。
    '''
    # 矩阵相乘，@是Python中的矩阵乘法运算符，相当于np.matmul(A, B)。
    W_approx = A @ B
    # 损失函数越小，表示 A 和 B 的乘积 W_approx越接近于目标矩阵 W
    return np.linalg.norm(W - W_approx, "fro") ** 2


# 定义梯度下降更新
def descent(W, A, B, lr, epochs):
    '''梯度下降法'''
    # 用于记录损失值
    loss_history = []
    for i in range(epochs):
        # 计算梯度
        W_approx = A @ B
        # 计算损失函数关于矩阵A的梯度
        gd_A = -2 * (W - W_approx) @ B.T
        # 计算损失函数关于矩阵B的梯度
        gd_B = -2 * A.T @ (W - W_approx)
        # 使用梯度下降更新矩阵A和B
        A -= lr * gd_A
        B -= lr * gd_B
        # 计算当前损失
        loss = loss_function(W, A, B)
        loss_history.append(loss)
        # 每100个epoch打印一次
        if i % 100 == 0:
            print(f"Epoch: {i} , 损失: {loss:.4f}")
    # 返回优化后的矩阵
    return A, B, loss_history


# 进行梯度下降优化
A, B, loss_history = descent(W, A, B, lr, epochs)

# # 绘制损失曲线
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8, 6))
# plt.plot(loss_history, label="loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss Curve")
# plt.legend()
# plt.grid(True)
# plt.show()

# 最终的近似矩阵
W_approx = A @ B
print(W_approx)

# 原始的矩阵 W
print(W)