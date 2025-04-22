import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义 NCF 模型
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_layers):
        super(NCF, self).__init__()

        # GMF部分的嵌入层
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP部分的嵌入层
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # 定义 MLP 的全连接层
        mlp_layers = []
        input_size = 2 * embedding_dim
        for layer_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_layers)  # 右侧MLP部分

        # 定义输出层，将 GMF 和 MLP 的结果结合
        self.fc_output = nn.Linear(embedding_dim + hidden_layers[-1], 1)  # 映射，没有做点积
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        # GMF 部分
        user_embedding_gmf = self.user_embedding_gmf(user_input)
        item_embedding_gmf = self.item_embedding_gmf(item_input)
        gmf_output = user_embedding_gmf * item_embedding_gmf  # Hadamard product (element-wise multiplication)

        # MLP 部分
        user_embedding_mlp = self.user_embedding_mlp(user_input)
        item_embedding_mlp = self.item_embedding_mlp(item_input)
        mlp_input = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # 拼接用户和物品的嵌入
        mlp_output = self.mlp(mlp_input)

        # 将 GMF 和 MLP 的输出结合
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)

        # 最终的全连接层输出
        output = self.fc_output(concat_output)
        output = self.sigmoid(output)
        return output
