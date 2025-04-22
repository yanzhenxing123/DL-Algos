"""
用FM替换wide部分
"""

import torch
import torch.nn as nn
import torch.optim as optim


# DeepFM模型
class DeepFM(nn.Module):
    def __init__(self, feature_size, field_size, embedding_size, dnn_hidden_units, dropout_rate=0.5):
        super(DeepFM, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        # Embedding for FM part and DNN part
        self.embedding = nn.Embedding(feature_size, embedding_size)

        # FM部分：一阶线性部分
        self.fm_linear = nn.Embedding(feature_size, 1)

        # DNN部分
        dnn_layers = []
        input_size = field_size * embedding_size
        for unit in dnn_hidden_units:
            dnn_layers.append(nn.Linear(input_size, unit))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            input_size = unit
        self.dnn = nn.Sequential(*dnn_layers)
        self.dnn_output = nn.Linear(dnn_hidden_units[-1], 1)

    def forward(self, x):
        """
        x: LongTensor of shape (batch_size, field_size) 表示每个样本的多个域特征
        """
        # 一阶部分 (linear part)
        linear_part = torch.sum(self.fm_linear(x), dim=1)  # (batch_size, 1)

        # 嵌入层 (embedding for FM second-order and DNN input)
        embed_x = self.embedding(x)  # (batch_size, field_size, embedding_size)

        # FM部分：二阶交互部分
        sum_square = torch.square(torch.sum(embed_x, dim=1))  # (batch_size, embedding_size)
        square_sum = torch.sum(embed_x ** 2, dim=1)  # (batch_size, embedding_size)
        fm_part = 0.5 * (sum_square - square_sum)  # (batch_size, embedding_size)

        # DNN部分
        dnn_input = embed_x.view(embed_x.size(0), -1)  # (batch_size, field_size * embedding_size)
        dnn_output = self.dnn(dnn_input)  # (batch_size, hidden_units[-1])
        dnn_output = self.dnn_output(dnn_output)  # (batch_size, 1)

        # 最终输出，结合FM的线性、一阶、二阶和DNN部分
        output = linear_part + torch.sum(fm_part, dim=1, keepdim=True) + dnn_output
        return torch.sigmoid(output)


# 模型实例化和训练过程
def build_deepfm(feature_size, field_size, embedding_size=8, dnn_hidden_units=[64, 32], dropout_rate=0.5):
    model = DeepFM(feature_size, field_size, embedding_size, dnn_hidden_units, dropout_rate)
    return model


# 示例
feature_size = 1000  # 特征的总数量（例如：one-hot特征的总维度）
field_size = 10  # 域的数量（例如：10个不同的类别特征）
embedding_size = 8  # 嵌入维度

model = build_deepfm(feature_size, field_size, embedding_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类任务用二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 假设训练数据X_train和y_train
# X_train的shape: (num_samples, field_size)
# y_train的shape: (num_samples, 1)

# 模拟一个训练过程
def train(model, criterion, optimizer, X_train, y_train, epochs=100, batch_size=32):
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


# 假设有数据
X_train = torch.randint(0, feature_size, (1000, field_size)).long()  # 随机生成的训练数据
y_train = torch.randint(0, 2, (1000, 1)).float()  # 随机生成的标签

# 训练模型
train(model, criterion, optimizer, X_train, y_train, epochs=100)
