import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个手动实现的 RNN 模型
class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualRNN, self).__init__()
        self.hidden_size = hidden_size

        # 定义用于计算隐藏状态的权重和偏置
        self.Wx = nn.Linear(input_size, hidden_size)  # 10, 20 # 共享的
        self.Wh = nn.Linear(hidden_size, hidden_size)  # 20，20
        self.tanh = nn.Tanh()

        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)  # 20, 5
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态 h0，shape 为 (batch_size, hidden_size)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # 逐时间步计算隐藏状态
        for t in range(seq_len):
            xt = x[:, t, :]  # 获取当前时间步 t 的输入 # [32, 10]
            h = self.tanh(self.Wx(xt) + self.Wh(h))  # 更新隐藏状态 # ([32, 20])

        # 使用最后的隐藏状态计算输出
        out = self.fc(h)  # ([32, 5]) # 一个时间序列只有一个输出
        out = self.softmax(out)
        return out

    def predict(self, x):
        self.eval()  # 设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            output = self.forward(x)  # 前向传播
            probabilities = torch.softmax(output, dim=1)  # 应用 Softmax 转换为概率
            predicted_labels = torch.argmax(probabilities, dim=1)  # 获取预测的类别索引
            return predicted_labels  # 返回预测的标签


# 设置参数
input_size = 10  # 输入特征维度
hidden_size = 20  # RNN 隐藏层维度
output_size = 5  # 输出类别数
batch_size = 32  # 批量大小
sequence_length = 15  # 输入序列长度 # 时间长度，eg：小时数量
num_epochs = 1000  # 训练迭代次数
learning_rate = 0.001  # 学习率

# 实例化模型
model = ManualRNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

# 生成随机输入数据和目标
inputs = torch.randn(batch_size, sequence_length, input_size)  # 随机生成输入数据 # ([32, 15, 10])
targets = torch.randint(0, output_size, (batch_size,))  # 随机生成目标类别标签 # 32

# 训练循环
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, targets)

    # 反向传播并优化
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    # 每 10 个 epoch 输出一次损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
