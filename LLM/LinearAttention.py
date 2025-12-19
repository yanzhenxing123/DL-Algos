import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearAttention(nn.Module):
    """
    Linear Attention: 真正的线性复杂度注意力机制
    
    核心思想：
    1. 使用核函数技巧，将 softmax(QK^T) 改写为使用特征映射函数 φ
    2. 改变计算顺序：从 Q @ K^T @ V 改为 Q @ (K^T @ V)
    3. 避免显式计算 n×n 的注意力矩阵
    
    复杂度：
    - 标准注意力: O(n²·d)
    - Linear Attention: O(n·d²)
    
    核函数选择：
    - elu(x) + 1: 常用的核函数，近似 softmax
    - relu(x): 另一种选择
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, kernel='elu'):
        """
        Args:
            d_model: 输入/输出维度
            num_heads: 注意力头数
            dropout: dropout比率
            kernel: 核函数类型，可选 'elu', 'relu', 'quadratic'
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Q, K, V 投影
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.kernel = kernel
        
    def kernel_function(self, x):
        """
        核函数：将点积映射到特征空间
        """
        if self.kernel == 'elu':
            # elu(x) + 1: 近似 softmax，保证非负
            return F.elu(x) + 1
        elif self.kernel == 'relu':
            # ReLU: 简单的非负映射
            return F.relu(x)
        elif self.kernel == 'quadratic':
            # 二次核: (x + 1)^2
            return (x + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, q_seq_len, d_model)
            key: (batch_size, k_seq_len, d_model)
            value: (batch_size, v_seq_len, d_model)
            mask: (batch_size, q_seq_len, k_seq_len) 可选
        Returns:
            output: (batch_size, q_seq_len, d_model)
        """
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        
        # Step 1: 线性投影
        Q = self.wq(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        K = self.wk(key).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        V = self.wv(value).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        
        # Step 2: 转换维度
        # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Step 3: 应用核函数到 Q 和 K
        # 核函数将点积映射到特征空间
        Q_kernel = self.kernel_function(Q)  # (batch_size, num_heads, q_seq_len, head_dim)
        K_kernel = self.kernel_function(K)  # (batch_size, num_heads, k_seq_len, head_dim)
        
        # Step 4: Linear Attention 核心计算
        # 标准注意力: Q @ K^T @ V  (需要 n×n 矩阵)
        # Linear Attention: Q @ (K^T @ V)  (避免 n×n 矩阵)
        
        # 先计算 K^T @ V: (batch_size, num_heads, head_dim, head_dim)
        # K_kernel: (batch_size, num_heads, k_seq_len, head_dim)
        # V: (batch_size, num_heads, k_seq_len, head_dim)
        KV = torch.matmul(K_kernel.transpose(-2, -1), V)  #  [B, D, L] * [B, L, D]  = [B, D, D] O(D^2 * L)  (batch_size, num_heads, head_dim, head_dim)
        
        # 再计算 Q @ KV: (batch_size, num_heads, q_seq_len, head_dim)
        # Q_kernel: (batch_size, num_heads, q_seq_len, head_dim)
        attn_output = torch.matmul(Q_kernel, KV) # [B, L, D] * [B, D, D]  = [B, L, D] O(D^2 * L)  
        
        # Step 5: 归一化（重要！）
        # 需要除以 Q @ K^T 的归一化因子
        # 计算每个 query 对应的归一化因子
        # Q_kernel @ K_kernel^T 的每行和
        normalizer = torch.matmul(
            Q_kernel, 
            K_kernel.sum(dim=2, keepdim=True).transpose(-2, -1)
        )  # (batch_size, num_heads, q_seq_len, 1)
        
        # 避免除零，添加小的epsilon
        normalizer = normalizer + 1e-8
        attn_output = attn_output / normalizer
        
        # 应用 mask（如果提供）
        if mask is not None:
            # Linear Attention 的 mask 处理比较复杂
            # 这里简化处理：如果 mask 存在，我们需要重新计算被 mask 的位置
            # 实际应用中可能需要更复杂的处理
            pass
        
        attn_output = self.dropout(attn_output)
        
        # Step 6: 合并多头
        # (batch_size, q_seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 输出投影
        output = self.wo(attn_output)
        
        return output


class PerformerAttention(nn.Module):
    """
    Performer: 使用随机特征（Random Features）的 Linear Attention
    
    核心思想：
    1. 使用随机特征映射近似 softmax 核函数
    2. 通过正交随机矩阵（Orthogonal Random Features）提高近似质量
    3. 复杂度: O(n·d·m)，其中 m 是随机特征数量（通常 m << n）
    
    优势：
    - 真正的线性复杂度
    - 可以处理任意长度的序列
    - 近似质量可控（通过 m 调整）
    """
    
    def __init__(self, d_model, num_heads, num_random_features=64, dropout=0.1):
        """
        Args:
            d_model: 输入/输出维度
            num_heads: 注意力头数
            num_random_features: 随机特征数量 m（通常设为 d_model 的 1/4 到 1/2）
            dropout: dropout比率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_random_features = num_random_features
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Q, K, V 投影
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 初始化随机特征矩阵（每个头独立）
        # 使用正交随机矩阵提高近似质量
        self.register_buffer('random_matrix', self._generate_random_matrix())
        
    def _generate_random_matrix(self):
        """
        生成正交随机矩阵用于随机特征映射
        形状: (num_heads, num_random_features, head_dim)
        """
        # 为每个头生成独立的随机矩阵
        random_matrices = []
        for _ in range(self.num_heads):
            # 生成随机矩阵
            W = torch.randn(self.num_random_features, self.head_dim)
            # QR 分解得到正交矩阵
            Q, R = torch.linalg.qr(W, mode='reduced')
            random_matrices.append(Q)
        
        return torch.stack(random_matrices)  # (num_heads, num_random_features, head_dim)
    
    def random_features(self, x):
        """
        应用随机特征映射
        x: (batch_size, num_heads, seq_len, head_dim)
        返回: (batch_size, num_heads, seq_len, num_random_features)
        """
        # 计算 x @ W^T，其中 W 是随机矩阵
        # x: (batch_size, num_heads, seq_len, head_dim)
        # W: (num_heads, num_random_features, head_dim)
        # 结果: (batch_size, num_heads, seq_len, num_random_features)
        
        # 使用 einsum 进行批量矩阵乘法
        features = torch.einsum('bhnd,hfd->bhnf', x, self.random_matrix)
        
        # 应用非线性变换（cos 和 sin）
        # 这是 Performer 的关键：使用 cos 和 sin 特征
        cos_features = torch.cos(features * self.scale)
        sin_features = torch.sin(features * self.scale)
        
        # 拼接得到 2*m 个特征
        return torch.cat([cos_features, sin_features], dim=-1)  # (batch_size, num_heads, seq_len, 2*num_random_features)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, q_seq_len, d_model)
            key: (batch_size, k_seq_len, d_model)
            value: (batch_size, v_seq_len, d_model)
            mask: (batch_size, q_seq_len, k_seq_len) 可选
        Returns:
            output: (batch_size, q_seq_len, d_model)
        """
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        
        # Step 1: 线性投影
        Q = self.wq(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        K = self.wk(key).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        V = self.wv(value).view(batch_size, k_seq_len, self.num_heads, self.head_dim)
        
        # Step 2: 转换维度
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, q_seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Step 3: 应用随机特征映射
        Q_features = self.random_features(Q)  # (batch_size, num_heads, q_seq_len, 2*m)
        K_features = self.random_features(K)  # (batch_size, num_heads, k_seq_len, 2*m)
        
        # Step 4: Linear Attention 计算
        # Q_features @ K_features^T 近似 softmax(Q @ K^T)
        # 计算 K_features^T @ V
        # K_features: (batch_size, num_heads, k_seq_len, 2*m)
        # V: (batch_size, num_heads, k_seq_len, head_dim)
        KV = torch.matmul(K_features.transpose(-2, -1), V)  # (batch_size, num_heads, 2*m, head_dim)
        
        # 计算 Q_features @ KV
        attn_output = torch.matmul(Q_features, KV)  # (batch_size, num_heads, q_seq_len, head_dim)
        
        # Step 5: 归一化
        # 计算归一化因子
        normalizer = torch.matmul(
            Q_features,
            K_features.sum(dim=2, keepdim=True).transpose(-2, -1)
        )  # (batch_size, num_heads, q_seq_len, 1)
        
        normalizer = normalizer + 1e-8
        attn_output = attn_output / normalizer
        
        # 应用 mask（简化处理）
        if mask is not None:
            pass
        
        attn_output = self.dropout(attn_output)
        
        # Step 6: 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 输出投影
        output = self.wo(attn_output)
        
        return output


class EfficientAttention(nn.Module):
    """
    Efficient Attention: 另一种 Linear Attention 变体
    
    使用更简单的归一化策略，计算更稳定
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)
        
        # 投影
        Q = self.wq(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(key).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(value).view(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用 ReLU 作为核函数
        Q_kernel = F.relu(Q) + 1e-8
        K_kernel = F.relu(K) + 1e-8
        
        # Linear Attention: Q @ (K^T @ V)
        KV = torch.matmul(K_kernel.transpose(-2, -1), V)
        attn_output = torch.matmul(Q_kernel, KV)
        
        # 归一化
        normalizer = torch.matmul(
            Q_kernel,
            K_kernel.sum(dim=2, keepdim=True).transpose(-2, -1)
        ) + 1e-8
        attn_output = attn_output / normalizer
        
        attn_output = self.dropout(attn_output)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        return self.wo(attn_output)


# 示例用法和对比
if __name__ == "__main__":
    # 创建模型
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 1024  # 使用较长的序列来展示优势
    
    print("=" * 60)
    print("Linear Attention 实现对比")
    print("=" * 60)
    
    # 标准注意力（用于对比）
    class StandardAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.wq = nn.Linear(d_model, d_model)
            self.wk = nn.Linear(d_model, d_model)
            self.wv = nn.Linear(d_model, d_model)
            self.wo = nn.Linear(d_model, d_model)
            self.scale = 1.0 / math.sqrt(self.head_dim)
        
        def forward(self, query, key, value):
            batch_size, seq_len = query.size(0), query.size(1)
            Q = self.wq(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.wk(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.wv(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.wo(attn_output)
    
    # 创建各种注意力模型
    standard_attn = StandardAttention(d_model, num_heads)
    linear_attn = LinearAttention(d_model, num_heads, kernel='elu')
    performer_attn = PerformerAttention(d_model, num_heads, num_random_features=64)
    efficient_attn = EfficientAttention(d_model, num_heads)
    
    # 创建输入
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: {query.shape}")
    print(f"序列长度: {seq_len}, 模型维度: {d_model}, 头数: {num_heads}\n")
    
    # 前向传播
    with torch.no_grad():
        output_standard = standard_attn(query, key, value)
        output_linear = linear_attn(query, key, value)
        output_performer = performer_attn(query, key, value)
        output_efficient = efficient_attn(query, key, value)
    
    print("输出形状验证:")
    print(f"  标准注意力: {output_standard.shape}")
    print(f"  Linear Attention: {output_linear.shape}")
    print(f"  Performer: {output_performer.shape}")
    print(f"  Efficient Attention: {output_efficient.shape}")
    
    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数量对比:")
    print(f"  标准注意力: {count_parameters(standard_attn):,}")
    print(f"  Linear Attention: {count_parameters(linear_attn):,}")
    print(f"  Performer: {count_parameters(performer_attn):,}")
    print(f"  Efficient Attention: {count_parameters(efficient_attn):,}")
    
    # 计算复杂度对比
    print(f"\n计算复杂度对比 (序列长度 n={seq_len}, 维度 d={d_model}):")
    standard_flops = seq_len * seq_len * d_model
    linear_flops = seq_len * d_model * d_model
    performer_flops = seq_len * d_model * 64  # num_random_features = 64
    
    print(f"  标准注意力: O(n²·d) = {standard_flops:,} FLOPs")
    print(f"  Linear Attention: O(n·d²) = {linear_flops:,} FLOPs")
    print(f"  Performer: O(n·d·m) = {performer_flops:,} FLOPs (m=64)")
    
    reduction_linear = (standard_flops - linear_flops) / standard_flops * 100
    reduction_performer = (standard_flops - performer_flops) / standard_flops * 100
    
    print(f"\n复杂度降低:")
    print(f"  Linear Attention: {reduction_linear:.1f}%")
    print(f"  Performer: {reduction_performer:.1f}%")
    
    print("\n" + "=" * 60)
    print("关键区别:")
    print("=" * 60)
    print("1. 标准注意力: 显式计算 n×n 注意力矩阵，复杂度 O(n²·d)")
    print("2. Linear Attention: 使用核函数技巧，避免 n×n 矩阵，复杂度 O(n·d²)")
    print("3. Performer: 使用随机特征近似，复杂度 O(n·d·m)，m << n")
    print("4. 当 n >> d 时，Linear Attention 和 Performer 优势明显")

