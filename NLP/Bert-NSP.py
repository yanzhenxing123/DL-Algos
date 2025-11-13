"""
@Author: yanzx
@Time: 2025/1/6
@Description: BERT NSP (Next Sentence Prediction) 任务完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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
        self.segment_embedding = nn.Embedding(2, d_model)  # 0 或 1
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len) - 0 表示第一个句子，1 表示第二个句子
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        token_emb = self.token_embedding(input_ids)
        
        # Position embedding
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        
        # Segment embedding (Token type embedding)
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
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) - padding mask
        """
        batch_size, seq_len, d_model = x.shape
        
        # 计算 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 应用 mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
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


class BERTForNSP(nn.Module):
    """BERT 模型用于 NSP 任务"""
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, d_ff=3072, max_seq_len=512):
        super().__init__()
        
        # Embedding 层
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len)
        
        # Encoder 层
        self.encoder = BERTEncoder(d_model, num_heads, num_layers, d_ff)
        
        # NSP 分类头
        self.nsp_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2)  # 二分类：IsNext 或 NotNext
        )
    
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) - 1 表示有效位置，0 表示 padding
        Returns:
            nsp_logits: (batch_size, 2)
        """
        # Embedding
        x = self.embedding(input_ids, token_type_ids)
        
        # Encoder
        hidden_states = self.encoder(x, attention_mask)
        
        # 取 [CLS] 位置的输出（第一个 token）
        cls_output = hidden_states[:, 0, :]  # [batch_size, d_model]
        
        # NSP 分类
        nsp_logits = self.nsp_classifier(cls_output)  # [batch_size, 2]
        
        return nsp_logits


class NSPDataset:
    """NSP 数据生成器"""
    
    def __init__(self, documents: List[List[str]], vocab: dict, max_seq_len=128):
        """
        Args:
            documents: 文档列表，每个文档是句子列表
            vocab: 词汇表字典 {word: id}
            max_seq_len: 最大序列长度
        """
        self.documents = documents
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.unk_id = vocab.get('[UNK]', 0)
        self.cls_id = vocab.get('[CLS]', 1)
        self.sep_id = vocab.get('[SEP]', 2)
    
    def tokenize(self, sentence: str) -> List[int]:
        """将句子转换为 token ids"""
        words = sentence.lower().split()
        return [self.vocab.get(word, self.unk_id) for word in words]
    
    def create_sample(self, sentence_A: str, sentence_B: str, is_next: bool) -> dict:
        """创建 NSP 样本"""
        # Tokenize
        tokens_A = self.tokenize(sentence_A)
        tokens_B = self.tokenize(sentence_B)
        
        # 构建输入序列: [CLS] A [SEP] B [SEP]
        input_ids = [self.cls_id] + tokens_A + [self.sep_id] + tokens_B + [self.sep_id]
        
        # Token type IDs: 0 表示第一个句子，1 表示第二个句子
        len_A = len(tokens_A) + 2  # +2 for [CLS] and [SEP]
        len_B = len(tokens_B) + 1  # +1 for [SEP]
        token_type_ids = [0] * len_A + [1] * len_B
        
        # 截断到最大长度
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            token_type_ids = token_type_ids[:self.max_seq_len]
        
        # Padding
        seq_len = len(input_ids)
        padding_len = self.max_seq_len - seq_len
        input_ids = input_ids + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        
        # Attention mask
        attention_mask = [1] * seq_len + [0] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(1 if is_next else 0, dtype=torch.long)
        }
    
    def get_batch(self, batch_size=32):
        """生成一批 NSP 样本"""
        batch = []
        
        for _ in range(batch_size):
            # 50% 概率生成正样本
            is_next = random.random() > 0.5
            
            if is_next:
                # 正样本：从同一文档取连续的两个句子
                doc = random.choice(self.documents)
                if len(doc) < 2:
                    # 如果文档只有一个句子，跳过
                    continue
                idx = random.randint(0, len(doc) - 2)
                sentence_A = doc[idx]
                sentence_B = doc[idx + 1]
            else:
                # 负样本：从不同文档随机取两个句子
                doc1 = random.choice(self.documents)
                doc2 = random.choice(self.documents)
                while doc1 == doc2 or len(doc1) == 0 or len(doc2) == 0:
                    doc2 = random.choice(self.documents)
                
                sentence_A = random.choice(doc1)
                sentence_B = random.choice(doc2)
            
            sample = self.create_sample(sentence_A, sentence_B, is_next)
            batch.append(sample)
        
        # 堆叠成 batch
        batch_dict = {
            'input_ids': torch.stack([s['input_ids'] for s in batch]),
            'token_type_ids': torch.stack([s['token_type_ids'] for s in batch]),
            'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
            'labels': torch.stack([s['label'] for s in batch])
        }
        
        return batch_dict


def create_vocab(documents: List[List[str]]) -> dict:
    """创建词汇表"""
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
    word_id = 4
    
    for doc in documents:
        for sentence in doc:
            words = sentence.lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = word_id
                    word_id += 1
    
    return vocab


def train_nsp():
    """NSP 任务训练示例"""
    print("BERT NSP 任务训练")
    print("="*50)
    
    # 模拟文档数据
    documents = [
        ["The cat sat on the mat.", "It was a sunny day.", "The weather was nice."],
        ["I love programming.", "Python is my favorite language.", "It's very versatile."],
        ["Machine learning is fascinating.", "Deep learning is a subset.", "Neural networks are powerful."],
        ["The sun rises in the east.", "It sets in the west.", "The sky is blue."],
        ["Reading books is enjoyable.", "Books contain knowledge.", "Knowledge is power."],
    ]
    
    # 创建词汇表
    vocab = create_vocab(documents)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集
    dataset = NSPDataset(documents, vocab, max_seq_len=128)
    
    # 创建模型
    model = BERTForNSP(
        vocab_size=vocab_size,
        d_model=128,  # 为了演示，使用较小的维度
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=128
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    num_epochs = 5
    batch_size = 8
    
    print(f"\n开始训练 ({num_epochs} 个 epoch)...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 10  # 每个 epoch 10 个 batch
        
        for batch_idx in range(num_batches):
            # 获取 batch
            batch = dataset.get_batch(batch_size)
            
            # 前向传播
            logits = model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask']
            )
            
            # 计算损失
            loss = criterion(logits, batch['labels'])
            
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
        test_batch = dataset.get_batch(batch_size=4)
        logits = model(
            test_batch['input_ids'],
            test_batch['token_type_ids'],
            test_batch['attention_mask']
        )
        
        predictions = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        
        print(f"真实标签: {test_batch['labels'].tolist()}")
        print(f"预测标签: {predictions.tolist()}")
        print(f"预测概率: {probabilities[:, 1].tolist()}")  # IsNext 的概率
        
        # 计算准确率
        correct = (predictions == test_batch['labels']).sum().item()
        accuracy = correct / len(predictions)
        print(f"准确率: {accuracy:.2%}")


def demo_nsp():
    """演示 NSP 任务"""
    print("BERT NSP 任务演示")
    print("="*50)
    
    # 创建简单的文档
    documents = [
        ["The cat sat on the mat.", "It was a sunny day."],
        ["I love programming.", "Python is great."],
    ]
    
    # 创建词汇表
    vocab = create_vocab(documents)
    vocab_size = len(vocab)
    
    # 创建数据集
    dataset = NSPDataset(documents, vocab, max_seq_len=64)
    
    # 创建模型
    model = BERTForNSP(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        max_seq_len=64
    )
    
    # 生成一个样本
    sample = dataset.create_sample(
        sentence_A="The cat sat on the mat.",
        sentence_B="It was a sunny day.",
        is_next=True
    )
    
    print("输入样本:")
    print(f"  Sentence A: The cat sat on the mat.")
    print(f"  Sentence B: It was a sunny day.")
    print(f"  Label: IsNext (1)")
    print(f"\n输入形状:")
    print(f"  input_ids: {sample['input_ids'].shape}")
    print(f"  token_type_ids: {sample['token_type_ids'].shape}")
    print(f"  attention_mask: {sample['attention_mask'].shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(
            sample['input_ids'].unsqueeze(0),
            sample['token_type_ids'].unsqueeze(0),
            sample['attention_mask'].unsqueeze(0)
        )
        
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
    
    print(f"\n模型输出:")
    print(f"  Logits: {logits.squeeze().tolist()}")
    print(f"  概率: NotNext={probabilities[0][0]:.4f}, IsNext={probabilities[0][1]:.4f}")
    print(f"  预测: {'IsNext' if prediction == 1 else 'NotNext'}")


if __name__ == "__main__":
    # 运行演示
    demo_nsp()
    
    print("\n" + "="*50 + "\n")
    
    # 运行训练（可选，取消注释以运行）
    # train_nsp()

