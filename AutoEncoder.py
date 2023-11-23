"""
@Time : 2023/11/23 11:23
@Author : yanzx
@Description : 自编码器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# 生成一些示例数据
data = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(2, 1)
        self.decoder = nn.Linear(1, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 创建模型、损失函数和优化器
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# 训练自编码器
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试自编码器
with torch.no_grad():
    test_data = torch.tensor([[0.2, 0.3], [0.5, 0.7]], dtype=torch.float32)
    reconstructed_data = autoencoder(test_data)
    print("Original Data:")
    print(test_data.numpy())
    print("Reconstructed Data:")
    print(reconstructed_data.numpy())

# 可视化结果
plt.scatter(data[:, 0], data[:, 1], label='Original Data')
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label='Reconstructed Data')
plt.legend()
plt.title('Original vs Reconstructed Data')
plt.show()
