class GroupedQueryAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads # 每组有多少个Q头
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.head_dim = d_model // num_heads

        # GQA: 关键区别在这里！
        # Q 的投影和 MHA 一样
        self.wq = nn.Linear(d_model, d_model)
        # K 和 V 的投影输出维度变小了，因为头数更少
        self.wk = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.wv = nn.Linear(d_model, num_kv_heads * self.head_dim)
        
        self.wo = nn.Linear(d_model, d_model)

    def repeat_kv(self, x, repeat_times):
        """这个地方不会扩大参数量"""
        # 这个函数用于将K和V重复到与Q头数匹配
        # x 形状: (batch_size, seq_len, num_kv_heads, head_dim)
        batch_size, seq_len, n_kv_heads, head_dim = x.size()
        if repeat_times == 1:
            return x
        # 增加一个维度并重复，使 n_kv_heads * repeat_times = num_heads
        # (batch_size, seq_len, num_kv_heads, 1, head_dim)
        x = x.unsqueeze(3)
        # (batch_size, seq_len, num_kv_heads, repeat_times, head_dim)
        x = x.repeat(1, 1, 1, repeat_times, 1)
        # (batch_size, seq_len, num_heads, head_dim)
        return x.view(batch_size, seq_len, n_kv_heads * repeat_times, head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)

        # 1. 线性投影
        # Q: (batch_size, q_seq_len, num_heads, head_dim)
        Q = self.wq(query).view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        # K, V: (batch_size, k_seq_len, num_kv_heads, head_dim)
        K = self.wk(key).view(batch_size, k_seq_len, self.num_kv_heads, self.head_dim)
        V = self.wv(value).view(batch_size, k_seq_len, self.num_kv_heads, self.head_dim)

        # 2. 转换维度
        Q = Q.transpose(1, 2) # (batch_size, num_heads, q_seq_len, head_dim)
        K = K.transpose(1, 2) # (batch_size, num_kv_heads, k_seq_len, head_dim)
        V = V.transpose(1, 2) # (batch_size, num_kv_heads, k_seq_len, head_dim)

        # 3. GQA核心：将K和V重复，以匹配Q的头数
        # 使 K, V 形状: (batch_size, num_heads, k_seq_len, head_dim)
        K = self.repeat_kv(K, self.num_kv_groups)
        V = self.repeat_kv(V, self.num_kv_groups)

        # 4. 计算缩放点积注意力 (与MHA完全相同)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V) # (batch_size, num_heads, q_seq_len, head_dim)

        # 5. 合并多头和输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        return self.wo(attn_output)