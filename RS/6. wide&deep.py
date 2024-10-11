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
        x_deep = [embedding(x_deep[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_deep = torch.cat(x_deep, dim=1)  # 拼接嵌入向量
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
