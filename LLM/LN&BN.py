"""
三维数据
"""
import torch


def manual_layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)

    if gamma is not None:
        x_norm = x_norm * gamma

    if beta is not None:
        x_norm = x_norm + beta

    return x_norm


def manual_batch_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(dim=(0, 1), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 1), keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)

    if gamma is not None:
        x_norm = x_norm * gamma

    if beta is not None:
        x_norm = x_norm + beta

    return x_norm


B, seq_len, dim = 2, 3, 4
x = torch.randn(B, seq_len, dim)

gamma = torch.ones(dim, requires_grad=True)
beta = torch.zeros(dim, requires_grad=True)

y_ln = manual_layer_norm(x, gamma, beta)
y_bn = manual_batch_norm(x, gamma, beta)

print("x shape:", x.shape)
print("LayerNorm shape:", y_ln.shape)
print("BatchNorm shape:", y_bn.shape)

print("LayerNorm mean over dim:")
print(y_ln.mean(dim=-1))

print("LayerNorm var over dim:")
print(y_ln.var(dim=-1, unbiased=False))

print("BatchNorm mean over B and seq_len:")
print(y_bn.mean(dim=(0, 1)))

print("BatchNorm var over B and seq_len:")
print(y_bn.var(dim=(0, 1), unbiased=False))