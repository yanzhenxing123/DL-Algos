"""
qformer：
# 第1步：Self-Attention（query tokens 之间互相交互）
self_attn_out = self.self_attn(
    query=x,      # query_tokens
    key=x,        # query_tokens
    value=x,      # query_tokens
    ...
)

# 第2步：Cross-Attention（query 去读取 image features）
cross_attn_out = self.cross_attn(
    query=x,                    # 更新后的 query_tokens
    key=image_features,         # image features
    value=image_features,       # image features
    ...
)
"""
import torch
import torch.nn as nn


class QFormerBlock(nn.Module):
    """
    一个最小版 Q-Former Block。

    包含：
    1. query token 之间的 self-attention
    2. query token 对 image features 的 cross-attention
    3. FFN
    """

    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        # Self-Attention: query token 之间互相交互
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-Attention: query token 去读取图像特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query_tokens,
        image_features,
        image_key_padding_mask=None
    ):
        """
        参数：
        query_tokens: [B, Nq, D]
            B  = batch size
            Nq = query token 数量，比如 32
            D  = hidden dim，比如 768

        image_features: [B, Ni, D]
            Ni = 图像 patch token 数量，比如 257

        image_key_padding_mask: [B, Ni]
            可选。如果图像 token 中有 padding，则 padding 位置为 True。

        返回：
        query_tokens: [B, Nq, D]
        """

        # =========================
        # 1. Self-Attention
        # =========================
        residual = query_tokens

        x = self.norm1(query_tokens)

        self_attn_out, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False
        )

        query_tokens = residual + self.dropout(self_attn_out)

        # =========================
        # 2. Cross-Attention
        # =========================
        residual = query_tokens

        x = self.norm2(query_tokens)

        cross_attn_out, cross_attn_weights = self.cross_attn(
            query=x,
            key=image_features,
            value=image_features,
            key_padding_mask=image_key_padding_mask,
            need_weights=False
        )

        query_tokens = residual + self.dropout(cross_attn_out)

        # =========================
        # 3. FFN
        # =========================
        residual = query_tokens

        x = self.norm3(query_tokens)

        ffn_out = self.ffn(x)

        query_tokens = residual + ffn_out

        return query_tokens


class MiniQFormer(nn.Module):
    """
    最小版 Q-Former。

    输入：
        image_features: [B, Ni, D]

    输出：
        query_outputs: [B, Nq, D]
    """

    def __init__(
        self,
        num_query_tokens=32,
        hidden_dim=768,
        num_heads=12,
        num_layers=6,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        self.num_query_tokens = num_query_tokens
        self.hidden_dim = hidden_dim

        # 可学习 query tokens
        # shape: [1, Nq, D]
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_dim)
        )

        self.layers = nn.ModuleList([
            QFormerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image_features,
        image_key_padding_mask=None
    ):
        """
        image_features: [B, Ni, D]
        image_key_padding_mask: [B, Ni]，可选

        return:
            query_tokens: [B, Nq, D]
        """

        B = image_features.size(0)

        # 把可学习 query token 扩展到 batch 维度
        query_tokens = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            query_tokens = layer(
                query_tokens=query_tokens,
                image_features=image_features,
                image_key_padding_mask=image_key_padding_mask
            )

        query_tokens = self.final_norm(query_tokens)

        return query_tokens


class QFormerForLLM(nn.Module):
    """
    Q-Former + 投影层。

    用于把视觉特征转成 LLM 可以接收的视觉 token。
    """

    def __init__(
        self,
        num_query_tokens=32,
        vision_dim=768,
        qformer_hidden_dim=768,
        qformer_num_heads=12,
        qformer_num_layers=6,
        llm_hidden_dim=4096,
        dropout=0.1
    ):
        super().__init__()

        # 如果视觉编码器输出维度和 Q-Former hidden_dim 不一致，先做一次投影
        if vision_dim != qformer_hidden_dim:
            self.vision_proj = nn.Linear(vision_dim, qformer_hidden_dim)
        else:
            self.vision_proj = nn.Identity()

        self.qformer = MiniQFormer(
            num_query_tokens=num_query_tokens,
            hidden_dim=qformer_hidden_dim,
            num_heads=qformer_num_heads,
            num_layers=qformer_num_layers,
            dropout=dropout
        )

        # 把 Q-Former 输出投影到 LLM hidden size
        self.llm_proj = nn.Linear(qformer_hidden_dim, llm_hidden_dim)

    def forward(
        self,
        image_features,
        image_key_padding_mask=None
    ):
        """
        image_features: [B, Ni, vision_dim]

        return:
            visual_tokens_for_llm: [B, Nq, llm_hidden_dim]
        """

        image_features = self.vision_proj(image_features)

        qformer_output = self.qformer(
            image_features=image_features,
            image_key_padding_mask=image_key_padding_mask
        )

        visual_tokens_for_llm = self.llm_proj(qformer_output)

        return visual_tokens_for_llm


def demo_qformer():
    """
    模拟运行一个 Q-Former。
    """

    # =========================
    # 1. 假设图像编码器输出
    # =========================
    batch_size = 2
    num_image_tokens = 257
    vision_dim = 768

    image_features = torch.randn(
        batch_size,
        num_image_tokens,
        vision_dim
    )

    # =========================
    # 2. 构造 Q-Former
    # =========================
    model = QFormerForLLM(
        num_query_tokens=32,
        vision_dim=768,
        qformer_hidden_dim=768,
        qformer_num_heads=12,
        qformer_num_layers=6,
        llm_hidden_dim=4096,
        dropout=0.1
    )

    # =========================
    # 3. 前向传播
    # =========================
    visual_tokens_for_llm = model(image_features)

    print("image_features shape:")
    print(image_features.shape)

    print("visual_tokens_for_llm shape:")
    print(visual_tokens_for_llm.shape)


if __name__ == "__main__":
    demo_qformer()