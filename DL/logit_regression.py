import torch


class LogisticRegression:
    def __init__(self, input_size, learning_rate=0.01):
        # 初始化权重和偏置
        self.W = torch.randn(input_size, 1, requires_grad=True)  # 权重
        self.b = torch.zeros(1, requires_grad=True)  # 偏置
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def forward(self, X):
        # 线性部分
        z = X @ self.W + self.b  # X @ W 是矩阵乘法
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        # 计算二元交叉熵损失
        return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)

            # 计算损失
            loss = self.compute_loss(y, y_pred)

            # 反向传播
            loss.backward()

            # 更新权重和偏置
            with torch.no_grad():  # 禁用梯度计算
                self.W -= self.learning_rate * self.W.grad
                self.b -= self.learning_rate * self.b.grad

                # 清零梯度
                self.W.grad.zero_()
                self.b.grad.zero_()

            # 每 100 次输出一次损失
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).float()  # 通过阈值转换为 0 或 1


# 示例用法
if __name__ == "__main__":
    # 生成一些模拟数据
    torch.manual_seed(0)  # 设置随机种子
    X = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]], dtype=torch.float32)
    y = torch.tensor([[0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.]], dtype=torch.float32)

    X = torch.randn(100, 5)
    y = torch.randint(0, 2, (100,), dtype=torch.float32)

    print(y)


    # 创建模型
    model = LogisticRegression(input_size=5, learning_rate=0.1)

    # 训练模型
    model.fit(X, y, epochs=1000)

    # 预测
    predictions = model.predict(X)
    print("Predictions:", predictions.squeeze().numpy())
