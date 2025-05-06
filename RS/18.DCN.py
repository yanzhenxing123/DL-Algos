import torch
import torch.nn as nn
import torch.optim as optim

# 假设的输入维度和特征
num_users = 1000  # 用户数量
num_ads = 500  # 广告数量
embedding_dim = 10  # 嵌入层维度
hidden_dim = 64  # 神经网络隐层维度
cross_layer_num = 3  # 交叉层数量
input_dim = 3  # 输入特征维度：用户ID，广告ID，广告预算（连续特征）


# 定义DCN模型
class DCN(nn.Module):
    def __init__(self, num_users, num_ads, embedding_dim, hidden_dim, cross_layer_num):
        super(DCN, self).__init__()

        # 定义Embedding层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.ad_embedding = nn.Embedding(num_ads, embedding_dim)

        # Deep部分：MLP部分
        self.deep_fc1 = nn.Linear(embedding_dim * 2 + 1, hidden_dim)
        self.deep_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.deep_fc3 = nn.Linear(hidden_dim // 2, 1)  # 最后一层输出1个值

        # Cross部分：特征交叉层
        self.cross_layers = nn.ModuleList(
            [nn.Linear(embedding_dim * 2 + 1, embedding_dim * 2 + 1) for _ in range(cross_layer_num)])

    def forward(self, user_id, ad_id, budget):
        """
        budget: 广告预算，也是个特征
        """
        # 获取Embedding
        user_emb = self.user_embedding(user_id)  # torch.Size([32, 10])
        ad_emb = self.ad_embedding(ad_id)  # ad_emb

        # 拼接Embedding和其他特征
        x = torch.cat([user_emb, ad_emb, budget.view(-1, 1)], dim=1)  # 拼接特征：用户ID，广告ID和广告预算 torch.Size([32, 21])

        # Deep部分：通过MLP学习特征的非线性关系
        deep_out = torch.relu(self.deep_fc1(x))
        deep_out = torch.relu(self.deep_fc2(deep_out))
        deep_out = self.deep_fc3(deep_out)  # 输出为[batch_size, 1]

        # Cross部分：通过交叉层显式捕捉特征交互
        cross_out = x
        for cross_layer in self.cross_layers:  # 逐层交叉操作
            cross_out = torch.relu(cross_layer(cross_out))

        # 合并Deep和Cross部分
        out = deep_out + cross_out[:, 0].view(-1,
                                              1)  # 确保cross部分的输出尺寸为[batch_size, 1] 在许多情况下，选择交叉操作后的 第一列（即第一个交互特征）作为交叉部分的输出是合理的，因为这个特征可能包含了交叉操作中最重要的模式。

        return torch.sigmoid(out)  # 返回广告点击的概率


# 初始化模型
model = DCN(num_users, num_ads, embedding_dim, hidden_dim, cross_layer_num)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设的数据
batch_size = 32
user_data = torch.randint(0, num_users, (batch_size,))  # 随机用户ID
ad_data = torch.randint(0, num_ads, (batch_size,))  # 随机广告ID
budget_data = torch.randn(batch_size)  # 随机广告预算
labels = torch.randint(0, 2, (batch_size,)).float()  # 随机点击标签

# 模型训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()  # 清零梯度

    # 前向传播
    outputs = model(user_data, ad_data, budget_data)

    # 计算损失（确保尺寸匹配）
    loss = criterion(outputs.view(-1), labels)  # 通过view将outputs的形状变为一维

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
