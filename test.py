import numpy as np

# 输入特征的数量 n 和隐向量的维度 k
n_features = 5  # 特征数量
k = 3  # 隐向量维度

# 假设特征向量 x 和隐向量矩阵 V
x = np.array([1, 0, 1, 0, 1])  # 输入特征向量
V = np.random.rand(n_features, k)  # 隐向量矩阵 V，形状为 n_features x k

# 1. 计算隐向量内积矩阵 VV_T
VV_T = np.dot(V, V.T)  # 结果是 n_features x n_features 的矩阵
# 2. 计算 x_i * x_j 的矩阵 (外积)
x_outer = np.outer(x, x)  # 生成一个 n_features x n_features 的矩阵，每个元素为 x_i * x_j

# 3. 计算二阶交互项的矩阵：VV_T * x_outer
interaction_matrix = VV_T * x_outer  # 每对特征的交互项

# 4. 创建上三角掩码，只保留上三角部分 i < j
upper_tri_mask = np.triu(np.ones_like(interaction_matrix), k=1)

# 5. 将交互矩阵中的上三角部分求和
interaction = np.sum(interaction_matrix * upper_tri_mask)

print("二阶交互项结果：", interaction)



interaction = 0.0

for i in range(n_features):
    for j in range(i + 1, n_features):  # 只考虑 i < j 的情况
        # 计算隐向量的内积 <v_i, v_j>
        v_i_dot_v_j = np.dot(V[i], V[j])  # 隐向量之间的内积
        interaction += v_i_dot_v_j * x[i] * x[j]  # 累加二阶交互项

print("二阶交互项结果（未优化）：", interaction)
