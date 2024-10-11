import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, num_continuous_features, num_discrete_features, discrete_feature_dims, k):
        """
        num_continuous_features: 连续特征的数量
        num_discrete_features: 离散特征的数量
        discrete_feature_dims: 每个离散特征的类别数量（一个list）
        k: 隐向量的维度
        """
        super(FM, self).__init__()

        # 线性部分：偏置项和连续特征的线性权重
        self.bias = nn.Parameter(torch.zeros(1))  # 偏置项 w0
        self.linear_continuous = nn.Linear(num_continuous_features, 1, bias=False)  # 连续特征的线性部分

        # 离散特征的 Embedding
        self.embeddings = nn.ModuleList([nn.Embedding(feature_dim, k) for feature_dim in discrete_feature_dims])  # [[10, 4], [15, 4]]

        # 连续特征的隐向量（与离散特征做交互）
        self.v_continuous = nn.Parameter(torch.randn(num_continuous_features, k))  # 每个连续特征有一个k维的隐向量

        # 离散特征的隐向量（通过Embedding生成）
        self.v_discrete = nn.ModuleList([nn.Embedding(feature_dim, k) for feature_dim in discrete_feature_dims])

    def forward(self, x_continuous, x_discrete):
        """
        x_continuous: 连续特征的输入，形状为 (batch_size, num_continuous_features)
        x_discrete: 离散特征的输入，形状为 (batch_size, num_discrete_features)
        """
        # 1. 线性部分
        linear_part_continuous = self.linear_continuous(x_continuous)  # 连续特征的线性部分
        linear_part = linear_part_continuous + self.bias  # 加上偏置项

        # 2. 离散特征的线性部分
        for i, embedding in enumerate(self.embeddings):
            # 离散特征直接嵌入成1维向量
            linear_part += torch.sum(
                embedding(
                    x_discrete[:, i]  # torch.Size([5])
                ), dim=1, keepdim=True  # [5, 4]
            )  # [5, 1]

        # 3. 二阶交互部分
        # 计算连续特征间的二阶交互
        interaction_part_1 = torch.matmul(x_continuous, self.v_continuous) ** 2
        interaction_part_2 = torch.matmul(x_continuous ** 2, self.v_continuous ** 2)
        interaction_continuous = 0.5 * torch.sum(interaction_part_1 - interaction_part_2, dim=1, keepdim=True)

        # 计算离散特征的二阶交互
        interaction_discrete = 0.0
        for i, embedding_i in enumerate(self.v_discrete):
            v_i = embedding_i(x_discrete[:, i])  # 提取离散特征的隐向量
            for j, embedding_j in enumerate(self.v_discrete):
                if i < j:
                    v_j = embedding_j(x_discrete[:, j])
                    interaction_discrete += torch.sum(v_i * v_j, dim=1, keepdim=True)

        # 连续特征和离散特征的交互部分
        interaction_continuous_discrete = 0.0
        for i in range(x_continuous.size(1)):  # 遍历所有连续特征
            v_i = self.v_continuous[i]
            for j, embedding_j in enumerate(self.v_discrete):
                v_j = embedding_j(x_discrete[:, j])
                interaction_continuous_discrete += torch.sum(v_i * v_j, dim=1, keepdim=True)

        # 总交互部分 = 连续-连续 + 离散-离散 + 连续-离散
        interaction_part = interaction_continuous + interaction_discrete + interaction_continuous_discrete

        # 4. 最终的FM输出
        output = linear_part + interaction_part
        return output


class FMLayer(nn.Module):
    def __init__(self, n=10, k=5):
        """
        :param n: 特征维度
        :param k: 隐向量维度
        """
        super(FMLayer, self).__init__()
        self.dtype = torch.float
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)  # 前两项线性层
        '''
        torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用。它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去；而module中非nn.Parameter()的普通tensor是不在parameter中的。
注意到，nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的，这与torth.Tensor对象的默认值相反。
在nn.Module类中，pytorch也是使用nn.Parameter来对每一个module的参数进行初始化的。'''
        self.v = nn.Parameter(torch.randn(self.n, self.k))  # 交互矩阵 (n, k)
        nn.init.uniform_(self.v, -0.1, 0.1)

    def fm_layer(self, x):
        # x 属于 R^{batch*n}
        linear_part = self.linear(x)  # 线性层 (batch, n)
        # print("linear_part",linear_part.shape)
        # linear_part = torch.unsqueeze(linear_part, 1)
        # print(linear_part.shape)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v)  # out_size = (batch, n) * (n, k)  = (batch, k) # 矩阵a和b矩阵相乘。 vi,f * xi
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(
            torch.pow(x, 2),  # (batch, n)
            torch.pow(self.v, 2)  # (n, k)
        )  # out_size = (batch, k)
        # 这里torch求和一定要用sum
        inter = 0.5 * torch.sum(torch.sub(torch.pow(inter_part1, 2), inter_part2), 1, keepdim=True)
        # print("inter",inter.shape)
        output = linear_part + inter
        output = torch.sigmoid(output)
        # print(output.shape) # out_size = (batch, 1)
        return output

    def forward(self, x):
        return self.fm_layer(x)


# 模拟数据
num_continuous_features = 3  # 连续特征数量
num_discrete_features = 2  # 离散特征数量
discrete_feature_dims = [10, 15]  # 每个离散特征的类别数量
k = 4  # 隐向量的维度

# 创建FM模型
fm_model = FM(num_continuous_features, num_discrete_features, discrete_feature_dims, k)

# 随机生成输入数据
batch_size = 5
x_continuous = torch.randn(batch_size, num_continuous_features)  # 连续特征输入
x_discrete = torch.randint(0, 10, (batch_size, num_discrete_features))  # 离散特征输入

# 前向传播计算输出
output = fm_model(x_continuous, x_discrete)
print(output)



## 未优化最后一项计算

import numpy as np

# 输入
n_features = 5  # 特征的数量
k = 3  # 隐向量的维度

# 假设特征向量 x 和隐向量矩阵 V
x = np.array([1, 0, 1, 0, 1])  # 特征向量 x
V = np.random.rand(n_features, k)  # 隐向量矩阵 V，每个特征有 k 维的隐向量

# 计算二阶交互项 (未优化版本)
interaction = 0.0

for i in range(n_features):
    for j in range(i + 1, n_features):  # 只考虑 i < j 的情况
        # 计算隐向量的内积 <v_i, v_j>
        v_i_dot_v_j = np.dot(V[i], V[j])  # 隐向量之间的内积
        interaction += v_i_dot_v_j * x[i] * x[j]  # 累加二阶交互项

print("二阶交互项结果（未优化）：", interaction)
