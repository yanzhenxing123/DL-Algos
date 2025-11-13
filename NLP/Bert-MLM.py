"""
@Author: yanzx
@Time: 2025/1/6
@Description: BERT MLM (Masked Language Model) 任务完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List, Tuple


class BERTEmbedding(nn.Module):
    """BERT Embedding: Token + Position + Segment"""
    
    def __init__(self, vocab_size, d_model=768, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Segment embedding (Token type embedding)
        self.segment_embedding = nn.Embedding(2, d_model)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids):
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        token_emb = self.token_embedding(input_ids)
        
        # Position embedding
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # Segment embedding
        seg_emb = self.segment_embedding(token_type_ids)
        
        # 相加
        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 计算 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用 mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        
        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class BERTEncoder(nn.Module):
    """BERT Encoder: 多层 Transformer"""
    
    def __init__(self, d_model=768, num_heads=12, num_layers=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BERTForMLM(nn.Module):
    """BERT 模型用于 MLM 任务"""
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, d_ff=3072, max_seq_len=512):
        super().__init__()
        
        # Embedding 层
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len)
        
        # Encoder 层
        self.encoder = BERTEncoder(d_model, num_heads, num_layers, d_ff)
        
        # MLM 分类头（预测被mask的token）
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)  # 输出词汇表大小
        )
        
        # 权重共享：MLM head 和 embedding 共享权重（可选）
        # self.mlm_head[-1].weight = self.embedding.token_embedding.weight
    
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - 包含 [MASK] token
            token_type_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding(input_ids, token_type_ids)
        
        # Encoder
        hidden_states = self.encoder(x, attention_mask)
        
        # MLM 预测（对所有位置预测）
        mlm_logits = self.mlm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        return mlm_logits


class MLMDataset:
    """MLM 数据生成器"""
    
    def __init__(self, sentences: List[str], vocab: dict, max_seq_len=128, 
                 mask_prob=0.15, random_token_prob=0.1, unchanged_prob=0.1):
        """
        Args:
            sentences: 句子列表
            vocab: 词汇表字典 {word: id}
            max_seq_len: 最大序列长度
            mask_prob: 被mask的token比例（15%）
            random_token_prob: 随机替换的概率（10%）
            unchanged_prob: 保持不变的概率（10%）
        """
        self.sentences = sentences
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob
        self.unchanged_prob = unchanged_prob
        
        # 特殊 token IDs
        self.pad_id = vocab.get('[PAD]', 0)
        self.unk_id = vocab.get('[UNK]', 1) # unk_id 是 [UNK] (Unknown) 的 token ID，用于表示不在词汇表中的词（OOV）。
        self.cls_id = vocab.get('[CLS]', 2)
        self.sep_id = vocab.get('[SEP]', 3)
        self.mask_id = vocab.get('[MASK]', 4)
    
    def tokenize(self, sentence: str) -> List[int]:
        """将句子转换为 token ids"""
        words = sentence.lower().split()
        return [self.vocab.get(word, self.unk_id) for word in words]
    
    def create_masked_sample(self, sentence: str) -> dict:
        """
        创建 MLM 样本
        
        Masking 策略（对每个被选中的token）:
        - 80% 替换为 [MASK]
        - 10% 替换为随机token
        - 10% 保持不变
        """
        # Tokenize
        tokens = self.tokenize(sentence)
        
        # 添加 [CLS] 和 [SEP]
        input_ids = [self.cls_id] + tokens + [self.sep_id]
        labels = [-100] * len(input_ids)  # -100 表示不计算损失
        
        # 确定哪些位置需要mask（排除特殊token）
        candidate_positions = [i for i in range(1, len(input_ids) - 1)]  # 排除 [CLS] 和 [SEP]
        num_to_mask = max(1, int(len(candidate_positions) * self.mask_prob))
        masked_positions = random.sample(candidate_positions, min(num_to_mask, len(candidate_positions)))
        
        # 对选中的位置进行masking
        for pos in masked_positions:
            original_token = input_ids[pos]
            labels[pos] = original_token  # 保存原始token作为标签
            
            # 80% 替换为 [MASK]
            if random.random() < 0.8:
                input_ids[pos] = self.mask_id
            # 10% 替换为随机token
            elif random.random() < 0.5:  # 在剩余的20%中，50%是随机token
                input_ids[pos] = random.randint(5, len(self.vocab) - 1)  # 随机token
            # 10% 保持不变（已经在labels中记录了）
        
        # 截断到最大长度
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        # Padding
        seq_len = len(input_ids)
        padding_len = self.max_seq_len - seq_len
        input_ids = input_ids + [self.pad_id] * padding_len
        labels = labels + [-100] * padding_len
        
        # Token type IDs (MLM 通常只有一个句子，所以都是0)
        token_type_ids = [0] * self.max_seq_len
        
        # Attention mask
        attention_mask = [1] * seq_len + [0] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def get_batch(self, batch_size=32):
        """生成一批 MLM 样本"""
        batch = []
        
        for _ in range(batch_size):
            sentence = random.choice(self.sentences)
            sample = self.create_masked_sample(sentence)
            batch.append(sample)
        
        # 堆叠成 batch
        batch_dict = {
            'input_ids': torch.stack([s['input_ids'] for s in batch]),
            'token_type_ids': torch.stack([s['token_type_ids'] for s in batch]),
            'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
            'labels': torch.stack([s['labels'] for s in batch])
        }
        
        return batch_dict


def create_vocab(sentences: List[str]) -> dict:
    """创建词汇表"""
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
    word_id = 5
    
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word not in vocab:
                vocab[word] = word_id
                word_id += 1
    
    return vocab


def train_mlm():
    """MLM 任务训练示例"""
    print("BERT MLM 任务训练")
    print("="*50)
    
    # 模拟句子数据
    sentences = [
        "The cat sat on the mat.",
        "It was a sunny day.",
        "I love programming.",
        "Python is my favorite language.",
        "Machine learning is fascinating.",
        "Deep learning is a subset of machine learning.",
        "Neural networks are powerful.",
        "The sun rises in the east.",
        "Reading books is enjoyable.",
        "Knowledge is power.",
        "The weather is nice today.",
        "I enjoy reading science fiction novels.",
        "Artificial intelligence is transforming the world.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to see.",
    ]
    
    # 创建词汇表
    vocab = create_vocab(sentences)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集
    dataset = MLMDataset(sentences, vocab, max_seq_len=64, mask_prob=0.15)
    
    # 创建模型
    model = BERTForMLM(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=64
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略 -100 的标签
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    num_epochs = 5
    batch_size = 8
    
    print(f"\n开始训练 ({num_epochs} 个 epoch)...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 10
        
        for batch_idx in range(num_batches):
            # 获取 batch
            batch = dataset.get_batch(batch_size)
            
            # 前向传播
            logits = model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask']
            )
            
            # 计算损失（只对masked位置计算）
            # logits: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len]
            logits_flat = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
            labels_flat = batch['labels'].view(-1)  # [batch*seq_len]
            
            loss = criterion(logits_flat, labels_flat)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}\n")
    
    # 测试
    print("="*50)
    print("测试模型")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        test_sentence = "The cat sat on the mat."
        sample = dataset.create_masked_sample(test_sentence)
        
        logits = model(
            sample['input_ids'].unsqueeze(0),
            sample['token_type_ids'].unsqueeze(0),
            sample['attention_mask'].unsqueeze(0)
        )
        
        # 找到被mask的位置
        masked_positions = (sample['labels'] != -100).nonzero(as_tuple=True)[0]
        
        print(f"原始句子: {test_sentence}")
        print(f"Masked 输入: {sample['input_ids'].tolist()[:20]}...")  # 显示前20个
        
        # 反转词汇表
        id_to_word = {v: k for k, v in vocab.items()}
        
        print(f"\n预测结果:")
        for pos in masked_positions[:5]:  # 显示前5个masked位置
            original_id = sample['labels'][pos].item()
            predicted_id = torch.argmax(logits[0, pos, :]).item()
            
            original_word = id_to_word.get(original_id, f"ID_{original_id}")
            predicted_word = id_to_word.get(predicted_id, f"ID_{predicted_id}")
            
            print(f"  位置 {pos}: 原始='{original_word}', 预测='{predicted_word}'")



if __name__ == "__main__":
    # 运行演示
    
    print("\n" + "="*50 + "\n")
    
    # 运行训练（可选，取消注释以运行）
    train_mlm()

