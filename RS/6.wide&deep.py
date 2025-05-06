"""
Wide 部分：wide 主要用作学习样本中特征的共现性，产生的结果是和用户有过直接行为的item，通过少量的交叉特征转换来补充 deep 的弱点。

==》只要能够发现高频、常见模式的特征都可以放在wide侧

特征组成：wide部分是一个广义的线性模型，输入的特征主要有两部分组成，一部分是原始的部分特征，另一部分是原始特征的交叉特征(cross-product transformation) 。
特征类型：wide侧的输入可以包含了一些交叉特征、离散特征及部分连续特征。


Deep 部分：负责复杂、隐含的模式（需要 Embedding 泛化）。
deep 部分是一个前馈神经网络。通常情况下这部分特征使用的是用户行为特征，用于学习历史数据中不存在的特征组合。

deep 层的输入主要是一些稠密的连续特征和一些离散特征的embedding
对于分类特征，原始输入是特征字符串(例如 “language=en”)。这些稀疏的、高维的分类特征首先被转换成一个低维的、密集的实值向量，通常被称为嵌入向量。嵌入的维数通常是10~100。

这些向量一般使用随机的方法进行初始化，随机可以均匀随机也可以随机成正态分布，随机初始化的目的是将向量初始化到一个数量级，在训练过程中通过最小化损失函数来优化模型。

这些低维密集的嵌入向量和连续值特征concat之后被输入到神经网络的隐藏层，执行计算。
某个特征可以既放到wide侧 也放到deep侧吗


最近与同事交流中学习到的心得，记录下来备忘，读者请谨慎参考。「将那些值越大越好的连续特征放到deep侧，将那些有可能与其他特征交叉从而生成有物理解释的交叉特征的ID类特征放到wide侧」。举个例子:
-qd ctr越大，用户在当前query下越容易点击当前doc，即click概率越大。而user_age与docclick的关系存在显然的分段规律，所以并不是user_age越大越好。-user_age通过分桶离散化为ID类特征后，与doc_type(视频类型:比如 军事/偶像剧/动画)交叉后的联合特征(user_age,doc_type) 与 click有很强的相关性。比如(小朋友，动画)容易被点击,(中年人，军事剧)容易被点击。
这种做法的motivation是:DNN鼓励激活值越大越好，所以要建立 dense特征与cicklabel的强关联，从deep侧而言，要选择那些 值越大越好的单个densefature;而wide侧强调记忆，所以就挑选有可能产生有物理解释的交叉特征的单个ID类特征。编辑于 2022-06-23 14:56


模型介绍：https://zhuanlan.zhihu.com/p/53361519
特征选择：https://zhuanlan.zhihu.com/p/597001726



"""

import torch
import torch.nn as nn
import torch.optim as optim


# Wide & Deep 模型
class WideAndDeep(nn.Module):
    def __init__(self, field_dims, embed_dim, deep_layers):
        super(WideAndDeep, self).__init__()

        # Wide部分（线性模型）
        self.wide = nn.Linear(sum(field_dims), 1)  # Wide部分输入直接是原始特征

        # Deep部分（Embedding + 多层感知机）
        self.embeddings = nn.ModuleList([nn.Embedding(field_dim, embed_dim) for field_dim in field_dims])
        """
        ModuleList(
          (0): Embedding(100, 8)
          (1): Embedding(200, 8)
          (2): Embedding(300, 8)
        )
        """
        deep_input_dim = len(field_dims) * embed_dim
        layers = []
        for units in deep_layers:
            layers.append(nn.Linear(deep_input_dim, units))
            layers.append(nn.ReLU())
            deep_input_dim = units
        self.deep = nn.Sequential(*layers)

        # 最终输出层
        self.fc = nn.Linear(deep_layers[-1] + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_wide, x_deep):
        # Wide部分直接通过线性层
        wide_out = self.wide(x_wide)  # torch.Size([32, 1])

        # Deep部分通过嵌入层和多层感知机
        x_deep = [embedding(x_deep[:, i]) for i, embedding in enumerate(self.embeddings)] # [3, 32, 8]
        x_deep = torch.cat(x_deep, dim=1)  # 拼接嵌入向量 # torch.Size([32, 24])
        deep_out = self.deep(x_deep)

        # Wide 和 Deep 的输出拼接
        combined = torch.cat([wide_out, deep_out], dim=1)

        # 最终预测
        out = self.fc(combined)
        return self.sigmoid(out)


# 参数设置
field_dims = [100, 200, 300]  # 特征维度，假设有3个特征，每个特征的类别数分别是100, 200, 300
embed_dim = 8  # 每个稀疏特征嵌入到8维空间
deep_layers = [64, 32]  # Deep部分的隐藏层设置

# 创建模型
model = WideAndDeep(field_dims, embed_dim, deep_layers)

# 模拟输入数据
x_wide = torch.randn(32, sum(field_dims))  # Wide部分输入（32个样本，Wide部分的输入维度是所有特征维度之和） # (32, 600)
x_deep = torch.randint(0, 100, (32, len(field_dims)))  # Deep部分输入（32个样本，每个特征用整数表示类别） # torch.Size([32, 3])

# 前向传播
output = model(x_wide, x_deep)
print(output.shape)  # 输出 (32, 1)
