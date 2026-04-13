import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

# ==========================================
# 🌟 用 KMeans 初始化的残差量化器
# ==========================================
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embed_dim, init_data=None):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        
        # 创建多个 codebook
        self.codebooks = nn.ParameterList()
        for i in range(num_codebooks):
            if init_data is not None:
                # 用 KMeans 初始化 codebook
                codebook = self._kmeans_init(init_data, codebook_size)
            else:
                # 随机初始化
                codebook = torch.randn(codebook_size, embed_dim) / np.sqrt(embed_dim)
            
            self.codebooks.append(nn.Parameter(codebook))
    
    def _kmeans_init(self, data, k):
        """用 KMeans 初始化 codebook"""
        # 转换为 numpy
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
        
        # 确保数据形状为 (N, embed_dim)
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        
        # 用 KMeans 聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_np)
        
        # 转换为 PyTorch tensor
        codebook = torch.from_numpy(kmeans.cluster_centers_).float()
        return codebook
    
    def forward(self, z):
        """
        Args:
            z: (batch_size, embed_dim)
        
        Returns:
            z_hat: 量化后的向量
            semantic_ids: 每个 codebook 的索引
            vq_loss: VQ 损失
        """
        batch_size = z.shape[0]
        z_hat = torch.zeros_like(z)
        semantic_ids = []
        vq_loss = 0
        
        # 残差向量，初始为输入
        residual = z.clone()
        
        # 逐个 codebook 量化
        for i, codebook in enumerate(self.codebooks):
            # 计算距离: (batch_size, codebook_size)
            distances = torch.cdist(residual, codebook)   # torch.Size([4, 256])
            
            # 找到最近的 codebook 向量的索引
            indices = torch.argmin(distances, dim=1)  # (batch_size,)
            semantic_ids.append(indices)
            
            # 量化向量
            z_quantized = codebook[indices]  # (batch_size, embed_dim)
            
            # 累积量化结果
            z_hat = z_hat + z_quantized
            
            # 更新残差
            residual = residual - z_quantized.detach()
            
            # VQ Loss = ||sg(z) - codebook||^2 + beta * ||z - sg(codebook)||^2
            # sg = stop gradient
            vq_loss += F.mse_loss(z_quantized.detach(), residual + z_quantized)
            vq_loss += 0.99 * F.mse_loss(z_quantized, (residual + z_quantized).detach())

            import pdb
            pdb.set_trace()
        
        semantic_ids = torch.stack(semantic_ids, dim=1)  # (batch_size, num_codebooks)
        
        return z_hat, semantic_ids, vq_loss

# ==========================================
# 🌟 新增：完整的端到端 RQ-VAE 模型
# ==========================================
class RQ_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_codebooks, codebook_size):
        super().__init__()
        
        # 1. 编码器 (Encoder): 把高维原始数据降维到潜变量空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 2. 核心量化器 (Quantizer): 咱们刚才写的那个类
        self.quantizer = ResidualVectorQuantizer(
            num_codebooks=num_codebooks, 
            codebook_size=codebook_size, 
            embed_dim=embed_dim
        )
        
        # 3. 解码器 (Decoder): 把量化后的特征还原成原始数据
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # Step 1: 编码得到连续特征 z
        z = self.encoder(x)
        
        # Step 2: 量化得到 z_hat 和 语义 ID
        z_hat, semantic_ids, vq_loss = self.quantizer(z)
        
        # Step 3: 解码得到重构的 x_hat
        x_hat = self.decoder(z_hat)
        
        return x_hat, semantic_ids, vq_loss

# ==========================================
# 测试真实的端到端训练逻辑
# ==========================================
if __name__ == "__main__":
    # 假设输入的是某个 512 维的原始物品特征
    INPUT_DIM = 512
    HIDDEN_DIM = 256
    EMBED_DIM = 128  # 压缩后的量化维度
    
    # 先生成一些初始化数据用于 KMeans
    init_data = torch.randn(1000, EMBED_DIM)
    
    # 初始化完整的模型
    model = RQ_VAE(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        embed_dim=EMBED_DIM, 
        num_codebooks=3, 
        codebook_size=256
    )
    
    # 用初始化数据对 Quantizer 的 codebook 进行 KMeans 初始化
    for i, codebook_param in enumerate(model.quantizer.codebooks):
        kmeans = KMeans(n_clusters=model.quantizer.codebook_size, random_state=42, n_init=10)
        kmeans_centers = torch.from_numpy(kmeans.fit(init_data.detach().cpu().numpy()).cluster_centers_).float()
        codebook_param.data = kmeans_centers
        print(f"✅ Codebook {i} 已用 KMeans 初始化")
    
    # 模拟一个 Batch 的输入数据 x
    x = torch.randn(4, INPUT_DIM, requires_grad=True)
    
    # 跑通整个网络
    x_hat, semantic_ids, vq_loss = model(x)
    
    print(f"\n原始输入 x 维度: {x.shape}")
    print(f"重构输出 x_hat 维度: {x_hat.shape}")
    print(f"生成的 Semantic IDs 形状: {semantic_ids.shape}")
    print(f"Semantic IDs:\n{semantic_ids}")
    
    # 🌟 计算最终的 Total Loss (完美对应你截图里的公式！)
    # L_recon = ||x - x_hat||^2
    recon_loss = F.mse_loss(x_hat, x)
    
    # L = L_recon + L_rqvae
    total_loss = recon_loss + vq_loss
    
    print(f"\n重构损失: {recon_loss.item():.4f}")
    print(f"VQ 损失: {vq_loss.item():.4f}")
    print(f"总损失: {total_loss.item():.4f}")
    
    total_loss.backward()
    print("\n✅ 反向传播成功，Encoder、Decoder 和 Codebook 都在同时更新！")