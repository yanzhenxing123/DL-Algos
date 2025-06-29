"""
@Time: 2025/5/5 15:38
@Author: yanzx
@Description: 
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 用卷积实现分块+线性投影
        self.proj = nn.Conv2d(
            in_channels,  # 3
            embed_dim,  # 768
            kernel_size=patch_size,
            stride=patch_size
        )

        # 可学习的位置编码和[CLS] token
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape # (1, 3, 224, 224)
        x = self.proj(x)  # [B, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.position_embedding
        return x


# 将图片转化为tensor
def transform_img2tensor(img):
    """

    """
    # 转换为Tensor（自动添加批次维度）
    transform = transforms.ToTensor()
    img_tensor = transform(img)  # 形状: [C, H, W]
    print(f"Tensor形状: {img_tensor.shape}")  # 输出: [3, 600, 800]（通道在前）

# 1. 读取真实图片
def load_image(image_path, img_size=224):
    img = Image.open(image_path).convert('RGB')

    # 预处理：调整大小、归一化、转为Tensor
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # 添加batch维度


# 2. 可视化分块（可选）
def show_patches(image_tensor, patch_size=16):
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())  # 反归一化

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            rect = plt.Rectangle((j, i), patch_size, patch_size, linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
    plt.title(f"Patch划分 (Size={patch_size})")
    plt.show()


# 示例：处理真实图片
if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = "images/drink.png"  # 确保图片存在

    # 读取并预处理图片
    img_tensor = load_image(image_path)
    

    print("输入图片Tensor形状:", img_tensor.shape)  # [1, 3, 224, 224]

    # 可视化分块（可选）
    show_patches(img_tensor)

    # 生成Patch Embeddings
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    patches = patch_embed(img_tensor)

    print("输出Patch Embeddings形状:", patches.shape)  # [1, 197, 768]