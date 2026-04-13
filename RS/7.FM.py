import torch
import torch.nn as nn

class FMLayer(nn.Module):
    def __init__(self, input_dim, k, w_reg=0.0, v_reg=0.0):
        super(FMLayer, self).__init__()
        self.input_dim = input_dim # PyTorch 通常需要显式传入特征维度
        self.k = k                 # 隐向量vi的维度
        self.w_reg = w_reg         # 权重w的正则项系数
        self.v_reg = v_reg         # 权重v的正则项系数

        # 定义并初始化参数
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(input_dim, 1))
        self.v = nn.Parameter(torch.randn(input_dim, k))

    def forward(self, x):
        # inputs维度判断，不符合则抛出异常
        if x.dim() != 2:
            raise ValueError(f"Unexpected inputs dimensions {x.dim()}, expect to be 2 dimensions")

        # 线性部分，相当于逻辑回归
        linear_part = torch.matmul(x, self.w) + self.w0  # shape: (batch_size, 1)

        # 交叉部分——第一项: (x * v)^2
        inter_part1 = torch.pow(torch.matmul(x, self.v), 2)  # shape: (batch_size, k)
        
        # 交叉部分——第二项: (x^2 * v^2)
        inter_part2 = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))  # shape: (batch_size, k)

        # 交叉结果
        inter_part = 0.5 * torch.sum(inter_part1 - inter_part2, dim=-1, keepdim=True)  # shape: (batch_size, 1)

        # 最终结果
        output = linear_part + inter_part
        return torch.sigmoid(output)  # shape: (batch_size, 1)

    def get_regularization_loss(self):
        """
        计算 L2 正则化 Loss。
        在训练的 step 中，你可以用 total_loss = bce_loss + fm_layer.get_regularization_loss()
        """
        reg_loss = 0.0
        if self.w_reg > 0:
            reg_loss += self.w_reg * torch.sum(torch.square(self.w))
        if self.v_reg > 0:
            reg_loss += self.v_reg * torch.sum(torch.square(self.v))
        return reg_loss


# ==========================================
# 测试数据和训练示例
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("FM (Factorization Machines) 层测试")
    print("=" * 60)
    
    # 参数设置
    INPUT_DIM = 10      # 输入特征维度
    K = 5               # 隐向量维度
    BATCH_SIZE = 4      # 批次大小
    NUM_SAMPLES = 100   # 训练样本数
    
    # 创建 FM 层实例
    fm_layer = FMLayer(input_dim=INPUT_DIM, k=K, w_reg=0.001, v_reg=0.001)
    
    print(f"\n✅ FM 层已创建:")
    print(f"   - 输入维度: {INPUT_DIM}")
    print(f"   - 隐向量维度: {K}")
    print(f"   - w 权重形状: {fm_layer.w.shape}")
    print(f"   - v 权重形状: {fm_layer.v.shape}")
    
    # ========== 测试 1: 前向传播 ==========
    print(f"\n{'='*60}")
    print("测试 1: 前向传播")
    print(f"{'='*60}")
    
    # 生成测试输入 (batch_size, input_dim)
    X_test = torch.randn(BATCH_SIZE, INPUT_DIM)
    print(f"输入数据形状: {X_test.shape}")
    
    # 前向传播
    y_pred = fm_layer(X_test)
    print(f"输出形状: {y_pred.shape}")
    print(f"输出值 (sigmoid after): \n{y_pred}")
    print(f"输出范围: [{y_pred.min().item():.4f}, {y_pred.max().item():.4f}]")
    
    # ========== 测试 2: 梯度计算 ==========
    print(f"\n{'='*60}")
    print("测试 2: 梯度计算与反向传播")
    print(f"{'='*60}")
    
    # 生成模拟标签 (随机 0/1)
    y_true = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    print(f"标签形状: {y_true.shape}")
    print(f"标签值: {y_true.squeeze()}")
    
    # 计算二元交叉熵损失
    loss_fn = nn.BCELoss()
    bce_loss = loss_fn(y_pred, y_true)
    
    # 加上正则化损失
    reg_loss = fm_layer.get_regularization_loss()
    total_loss = bce_loss + reg_loss
    
    print(f"\nBCE 损失: {bce_loss.item():.6f}")
    print(f"正则化损失: {reg_loss.item():.6f}")
    print(f"总损失: {total_loss.item():.6f}")
    
    # 反向传播
    total_loss.backward()
    print(f"\n✅ 反向传播成功!")
    print(f"w 权重的梯度范围: [{fm_layer.w.grad.min().item():.6f}, {fm_layer.w.grad.max().item():.6f}]")
    
    # ========== 测试 3: 批量数据训练 ==========
    print(f"\n{'='*60}")
    print("测试 3: 批量数据训练（5个迭代）")
    print(f"{'='*60}")
    
    # 重新创建模型
    fm_model = FMLayer(input_dim=INPUT_DIM, k=K, w_reg=0.0001, v_reg=0.0001)
    optimizer = torch.optim.Adam(fm_model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    
    # 生成训练数据
    X_train = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y_train = torch.randint(0, 2, (NUM_SAMPLES, 1)).float()
    
    print(f"训练数据: {X_train.shape}, 标签: {y_train.shape}\n")
    
    losses = []
    for epoch in range(5):
        epoch_loss = 0
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            X_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            
            # 前向传播
            y_pred_batch = fm_model(X_batch)
            
            # 计算损失
            bce = loss_fn(y_pred_batch, y_batch)
            reg = fm_model.get_regularization_loss()
            loss = bce + reg
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (NUM_SAMPLES // BATCH_SIZE)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.6f}")
    
    # ========== 测试 4: 特征交互演示 ==========
    print(f"\n{'='*60}")
    print("测试 4: 特征交互能力展示")
    print(f"{'='*60}")
    
    # 创建具有特定特征的输入
    fm_demo = FMLayer(input_dim=3, k=2, w_reg=0.0, v_reg=0.0)
    
    # 手动设置权重以展示特征交互
    with torch.no_grad():
        fm_demo.w.fill_(0.1)
        fm_demo.v.fill_(0.5)
        fm_demo.w0.fill_(0.0)
    
    # 第一个样本：[1, 0, 0]（只有第一个特征）
    # 第二个样本：[1, 1, 0]（第一、二特征）
    X_demo = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    
    y_demo = fm_demo(X_demo)
    print(f"输入样本:")
    print(f"  [1, 0, 0] -> 输出: {y_demo[0].item():.6f} (仅线性项)")
    print(f"  [1, 1, 0] -> 输出: {y_demo[1].item():.6f} (线性项 + 交互项)")
    print(f"  [1, 1, 1] -> 输出: {y_demo[2].item():.6f} (线性项 + 交互项)")
    print(f"\n💡 可见：特征增多时，输出值增加（体现了特征交互）")
    
    print(f"\n{'='*60}")
    print("✅ 所有测试完成!")
    print(f"{'='*60}")