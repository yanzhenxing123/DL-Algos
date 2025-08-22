"""
@Author: yanzx
@Time: 2025/8/19 10:29 
@Description: 
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# 补充 FeatureEmbedding 示例实现
class FeatureEmbedding(nn.Module):
    def __init__(self, feature_map, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(feature_map.num_features, embedding_dim)
        self.out_dim = feature_map.num_features * embedding_dim

    def forward(self, X):
        # 假设 X 是 (batch_size, num_features)
        return self.embedding(X)


# 补充 feature_map 示例
class DummyFeatureMap:
    def __init__(self, num_features):
        self.num_features = num_features

    def sum_emb_out_dim(self):
        # 假设每个特征都有 embedding
        return self.num_features * 10  # 10 是 embedding_dim


def get_activation(activation):
    """获取激活函数"""
    if activation == "ReLU":
        return nn.ReLU()
    elif activation == "Sigmoid":
        return nn.Sigmoid()
    elif activation is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class GateNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 hidden_activation="ReLU",
                 dropout_rate=0.0,
                 batch_norm=False):
        super(GateNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim if output_dim is not None else input_dim
        if output_dim is None:
            output_dim = hidden_dim
        gate_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        gate_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            gate_layers.append(nn.Dropout(dropout_rate))
        gate_layers.append(nn.Linear(hidden_dim, output_dim))
        gate_layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, inputs):
        return self.gate(inputs)


class PPNetBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 gate_input_dim=64,
                 gate_hidden_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False,
                 use_bias=True):
        super(PPNetBlock, self).__init__()

        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)

        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units

        for idx in range(len(hidden_units) - 1):
            dense_layers = []
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(get_activation(hidden_activations[idx]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(dropout_rates[idx]))

            self.gate_layers.append(GateNN(gate_input_dim, gate_hidden_dim, hidden_units[idx]))
            self.mlp_layers.append(nn.Sequential(*dense_layers))

        self.gate_layers.append(GateNN(gate_input_dim, gate_hidden_dim, output_dim=1))
        self.mlp_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))

    def forward(self, feature_emb, gate_emb):
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)
        hidden = feature_emb
        for i in range(len(self.mlp_layers)):
            gw = self.gate_layers[i](gate_input)
            hidden = self.mlp_layers[i](hidden * gw)
        return hidden


class PPNet(nn.Module):
    def __init__(self,
                 feature_map,
                 model_id="PPNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 gate_features=[],
                 gate_hidden_dim=64,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(PPNet, self).__init__()

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.gate_embed_layer = FeatureEmbedding(feature_map, embedding_dim)

        # 修正 gate_input_dim
        gate_input_dim = 2 * feature_map.sum_emb_out_dim()
        self.ppn = PPNetBlock(
            input_dim=feature_map.sum_emb_out_dim(),
            output_dim=1,
            gate_input_dim=gate_input_dim,
            gate_hidden_dim=gate_hidden_dim,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm
        )

        self.output_activation = nn.Sigmoid()  # 假设二分类任务

    def get_inputs(self, inputs):
        # 假设 inputs 已经是 tensor，直接返回
        return inputs

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        gate_emb = self.gate_embed_layer(X)

        # flatten 用法修正
        feature_emb_flat = feature_emb.view(feature_emb.size(0), -1)
        gate_emb_flat = gate_emb.view(gate_emb.size(0), -1)

        y_pred = self.ppn(
            feature_emb_flat,
            gate_emb_flat
        )
        y_pred = self.output_activation(y_pred)
        return {"y_pred": y_pred}


def main():
    feature_map = DummyFeatureMap(num_features=5)
    model = PPNet(feature_map)
    X = torch.randint(0, feature_map.num_features, (2, feature_map.num_features))
    out = model(X)
    print("y_pred:", out["y_pred"])
    print("y_pred shape:", out["y_pred"].shape)


if __name__ == "__main__":
    main()
