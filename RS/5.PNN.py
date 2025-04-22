import torch
import torch.nn as nn
import torch.optim as optim

"""
Deep Crossing的加强
用户和物品的特征交叉不再是简单的特征拼接，可以是内积交互和外积交互，外积交互举了个例子
"""

# 定义PNN模型
class PNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, mlp_layers):
        super(PNN, self).__init__()

        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 内积交互层（Hadamard product）
        self.inner_product = nn.Linear(embedding_dim, mlp_layers[0])

        # 外积交互层
        self.outer_product_conv = nn.Conv2d(1, 32, kernel_size=(embedding_dim, embedding_dim))

        # 全连接层
        self.linear = nn.Linear(2 * embedding_dim, mlp_layers[0])

        # MLP层
        mlp_modules = []
        for i in range(len(mlp_layers) - 1):
            mlp_modules.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)

        # 最终输出层
        self.fc_output = nn.Linear(mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        # 嵌入层：将用户和物品ID映射为向量
        user_embed = self.user_embedding(user_input)
        item_embed = self.item_embedding(item_input)

        # 内积交互（逐元素相乘）
        inner_product_output = self.inner_product(user_embed * item_embed)

        # 外积交互
        user_embed_reshaped = user_embed.unsqueeze(2)  # (batch_size, embedding_dim, 1)
        item_embed_reshaped = item_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        outer_product_matrix = torch.matmul(user_embed_reshaped, item_embed_reshaped)  # (batch_size, embedding_dim, embedding_dim)

        # 外积交互经过卷积处理
        outer_product_matrix = outer_product_matrix.unsqueeze(1)  # 添加一个维度 (batch_size, 1, embedding_dim, embedding_dim)
        outer_product_output = self.outer_product_conv(outer_product_matrix).view(user_input.size(0), -1)  # 展平为向量

        # 拼接内积和外积的输出
        interaction_combined = torch.cat([inner_product_output, outer_product_output], dim=-1)

        # 通过 MLP 进行非线性变换
        mlp_output = self.mlp(interaction_combined)

        # 输出预测的点击率
        output = self.fc_output(mlp_output)
        output = self.sigmoid(output)
        return output
