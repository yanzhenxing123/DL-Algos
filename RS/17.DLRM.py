import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
"""

This is done by taking the dot product between all pairs of embedding vectors and processed dense features. These dot products are concatenated with the original processed dense features and post-processed with another MLP (the top or output MLP)
也就是说会讲之前的embedding两两做点积，做完之后在跟之前dense features对应的embedding concat起来，喂给后续的MLP。所以这一步其实是希望特征之间做充分的交叉，组合之后，再进入上层MLP做最终的目标拟合。这一点其实follow了FM的特征交叉概念。
"""


# ========== 1. 生成合成数据 ==========
def generate_data(num_samples=10000, num_sparse_features=3, embedding_dim=4):
    # 稀疏特征（假设每个特征有5个类别）
    sparse_features = np.random.randint(0, 5, size=(num_samples, num_sparse_features))
    # 稠密特征
    dense_features = np.random.rand(num_samples, 2)
    # 标签（0/1）
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    return dense_features, sparse_features, labels


dense_data, sparse_data, labels = generate_data()


# ========== 2. 定义DLRM模型 ==========
class DLRM(nn.Module):
    def __init__(self, num_sparse_features, sparse_cardinalities, embedding_dim, dense_dim):
        super(DLRM, self).__init__()

        # 嵌入层（每个稀疏特征一个嵌入表）
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=embedding_dim)
            for card in sparse_cardinalities
        ])

        # 底部MLP处理稠密特征
        self.bottom_mlp = nn.Sequential(
            nn.Linear(dense_dim, 8),
            nn.ReLU(),
            nn.Linear(8, embedding_dim)
        )

        # 顶部MLP处理交互后的特征
        self.top_mlp = nn.Sequential(
            nn.Linear(embedding_dim * num_sparse_features + embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, dense_x, sparse_x):
        # 处理稠密特征
        dense_out = self.bottom_mlp(dense_x)

        # 处理稀疏特征
        sparse_outs = []
        for i in range(sparse_x.shape[1]):
            emb = self.embeddings[i](sparse_x[:, i])
            sparse_outs.append(emb)

        # 特征交互：拼接所有嵌入向量和稠密特征输出
        interaction = torch.cat(sparse_outs + [dense_out], dim=1)

        # 顶部MLP输出预测概率
        output = self.top_mlp(interaction)
        return output


# 假设每个稀疏特征有5个类别
sparse_cardinalities = [5, 5, 5]
model = DLRM(
    num_sparse_features=3,
    sparse_cardinalities=sparse_cardinalities,
    embedding_dim=4,
    dense_dim=2
)


# ========== 3. 训练模型 ==========
def train(model, dense_data, sparse_data, labels, epochs=10, batch_size=32):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 转换为PyTorch张量
    dense_tensor = torch.FloatTensor(dense_data)
    sparse_tensor = torch.LongTensor(sparse_data)
    label_tensor = torch.FloatTensor(labels)

    dataset = torch.utils.data.TensorDataset(dense_tensor, sparse_tensor, label_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_dense, batch_sparse, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_dense, batch_sparse)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


train(model, dense_data, sparse_data, labels)


# ========== 4. 测试预测 ==========
def predict(model, dense_x, sparse_x):
    with torch.no_grad():
        dense_x = torch.FloatTensor(dense_x)
        sparse_x = torch.LongTensor(sparse_x)
        pred = model(dense_x, sparse_x)
    return pred.numpy()


# 示例预测
test_dense = np.random.rand(1, 2)
test_sparse = np.random.randint(0, 5, size=(1, 3))
print("Predicted CTR:", predict(model, test_dense, test_sparse)[0][0])