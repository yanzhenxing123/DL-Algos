"""
@Author: yanzx
@Time: 2025/1/6
@Description: BERT 完整预训练：同时进行 MLM 和 NSP 任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List, Tuple, Dict


class BERTEmbedding(nn.Module):
    """BERT Embedding: Token + Position + Segment"""
    
    def __init__(self, vocab_size, d_model=768, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids):
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        seg_emb = self.segment_embedding(token_type_ids)
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
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
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
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
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


class BERTForPretraining(nn.Module):
    """BERT 模型用于预训练：同时进行 MLM 和 NSP"""
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, d_ff=3072, max_seq_len=512):
        super().__init__()
        
        # 共享的 Embedding 和 Encoder
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len)
        self.encoder = BERTEncoder(d_model, num_heads, num_layers, d_ff)
        
        # MLM 分类头
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # NSP 分类头
        self.nsp_head = nn.Sequential(
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
            attention_mask: (batch_size, seq_len)
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
            nsp_logits: (batch_size, 2)
        """
        # 共享的 Embedding 和 Encoder
        x = self.embedding(input_ids, token_type_ids)
        hidden_states = self.encoder(x, attention_mask)
        
        # MLM 预测（对所有位置）
        mlm_logits = self.mlm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # NSP 预测（使用 [CLS] 位置）
        cls_output = hidden_states[:, 0, :]  # [batch_size, d_model]
        nsp_logits = self.nsp_head(cls_output)  # [batch_size, 2]
        
        return mlm_logits, nsp_logits


class BERTPretrainingDataset:
    """BERT 预训练数据集：同时生成 MLM 和 NSP 样本"""
    
    def __init__(self, documents: List[List[str]], vocab: dict, max_seq_len=128, mask_prob=0.15):
        """
        Args:
            documents: 文档列表，每个文档是句子列表
            vocab: 词汇表字典
            max_seq_len: 最大序列长度
            mask_prob: MLM 的 mask 概率
        """
        self.documents = documents
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        # 特殊 token IDs
        self.pad_id = vocab.get('[PAD]', 0)
        self.unk_id = vocab.get('[UNK]', 1)
        self.cls_id = vocab.get('[CLS]', 2)
        self.sep_id = vocab.get('[SEP]', 3)
        self.mask_id = vocab.get('[MASK]', 4)
    
    def tokenize(self, sentence: str) -> List[int]:
        """将句子转换为 token ids"""
        words = sentence.lower().split()
        return [self.vocab.get(word, self.unk_id) for word in words]
    
    def create_pretraining_sample(self) -> dict:
        """
        创建一个预训练样本（同时包含 MLM 和 NSP）
        
        返回:
            - input_ids: 包含 [MASK] 的输入序列
            - token_type_ids: 句子类型（0 或 1）
            - attention_mask: 注意力掩码
            - mlm_labels: MLM 标签（-100 表示不计算损失）
            - nsp_label: NSP 标签（0 或 1）
        """
        # 50% 概率生成正样本（连续句子）
        is_next = random.random() > 0.5
        
        if is_next:
            # 正样本：从同一文档取连续的两个句子
            doc = random.choice(self.documents)
            if len(doc) < 2:
                # 如果文档只有一个句子，使用负样本策略
                doc1 = random.choice(self.documents)
                doc2 = random.choice(self.documents)
                while doc1 == doc2 or len(doc1) == 0 or len(doc2) == 0:
                    doc2 = random.choice(self.documents)
                sentence_A = random.choice(doc1)
                sentence_B = random.choice(doc2)
                is_next = False
            else:
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
        
        # Tokenize
        tokens_A = self.tokenize(sentence_A)
        tokens_B = self.tokenize(sentence_B)
        
        # 构建输入序列: [CLS] A [SEP] B [SEP]
        input_ids = [self.cls_id] + tokens_A + [self.sep_id] + tokens_B + [self.sep_id]
        
        # Token type IDs
        len_A = len(tokens_A) + 2
        len_B = len(tokens_B) + 1
        token_type_ids = [0] * len_A + [1] * len_B
        
        # MLM labels（初始化为 -100，表示不计算损失）
        mlm_labels = [-100] * len(input_ids)
        
        # 对第一个句子进行 MLM（排除特殊token）
        candidate_positions_A = list(range(1, len(tokens_A) + 1))  # 排除 [CLS]
        num_to_mask_A = max(1, int(len(candidate_positions_A) * self.mask_prob))
        masked_positions_A = random.sample(
            candidate_positions_A, 
            min(num_to_mask_A, len(candidate_positions_A))
        )
        
        # 对第二个句子进行 MLM（排除特殊token）
        candidate_positions_B = list(range(len_A, len_A + len(tokens_B)))  # 排除 [SEP]
        num_to_mask_B = max(1, int(len(tokens_B) * self.mask_prob))
        masked_positions_B = random.sample(
            candidate_positions_B,
            min(num_to_mask_B, len(candidate_positions_B))
        )
        
        # 合并所有需要mask的位置
        all_masked_positions = masked_positions_A + masked_positions_B
        
        # 对选中的位置进行masking
        for pos in all_masked_positions:
            original_token = input_ids[pos]
            mlm_labels[pos] = original_token  # 保存原始token作为标签
            
            # 80% 替换为 [MASK]
            if random.random() < 0.8:
                input_ids[pos] = self.mask_id
            # 10% 替换为随机token
            elif random.random() < 0.5:
                input_ids[pos] = random.randint(5, len(self.vocab) - 1)
            # 10% 保持不变（已经在labels中记录了）
        
        # 截断到最大长度
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            token_type_ids = token_type_ids[:self.max_seq_len]
            mlm_labels = mlm_labels[:self.max_seq_len]
        
        # Padding
        seq_len = len(input_ids)
        padding_len = self.max_seq_len - seq_len
        input_ids = input_ids + [self.pad_id] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        mlm_labels = mlm_labels + [-100] * padding_len
        
        # Attention mask
        attention_mask = [1] * seq_len + [0] * padding_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'nsp_label': torch.tensor(1 if is_next else 0, dtype=torch.long)
        }
    
    def get_batch(self, batch_size=32):
        """生成一批预训练样本"""
        batch = []
        for _ in range(batch_size):
            sample = self.create_pretraining_sample()
            batch.append(sample)
        
        batch_dict = {
            'input_ids': torch.stack([s['input_ids'] for s in batch]),
            'token_type_ids': torch.stack([s['token_type_ids'] for s in batch]),
            'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
            'mlm_labels': torch.stack([s['mlm_labels'] for s in batch]),
            'nsp_labels': torch.stack([s['nsp_label'] for s in batch])
        }
        
        return batch_dict


def create_vocab(documents: List[List[str]]) -> dict:
    """创建词汇表"""
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
    word_id = 5
    
    for doc in documents:
        for sentence in doc:
            words = sentence.lower().split()
            for word in words:
                if word not in vocab:
                    vocab[word] = word_id
                    word_id += 1
    
    return vocab


def train_bert_pretraining():
    """BERT 预训练：同时进行 MLM 和 NSP"""
    print("BERT 预训练：MLM + NSP")
    print("="*50)
    
    # 模拟文档数据
    documents = [
        ["The cat sat on the mat.", "It was a sunny day.", "The weather was nice."],
        ["I love programming.", "Python is my favorite language.", "It's very versatile."],
        ["Machine learning is fascinating.", "Deep learning is a subset.", "Neural networks are powerful."],
        ["The sun rises in the east.", "It sets in the west.", "The sky is blue."],
        ["Reading books is enjoyable.", "Books contain knowledge.", "Knowledge is power."],
        ["Artificial intelligence is transforming.", "Natural language processing helps computers.", "Computer vision enables machines."],
    ]
    
    # 创建词汇表
    vocab = create_vocab(documents)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 创建数据集
    dataset = BERTPretrainingDataset(documents, vocab, max_seq_len=64, mask_prob=0.15)
    
    # 创建模型
    model = BERTForPretraining(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=64
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # MLM 损失
    nsp_criterion = nn.CrossEntropyLoss()  # NSP 损失
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 损失权重（可以调整）
    mlm_weight = 1.0  # MLM 损失权重
    nsp_weight = 1.0  # NSP 损失权重
    
    print(f"\n损失权重: MLM={mlm_weight}, NSP={nsp_weight}")
    print(f"开始训练...\n")
    
    # 训练循环
    num_epochs = 5
    batch_size = 8
    
    for epoch in range(num_epochs):
        total_mlm_loss = 0
        total_nsp_loss = 0
        total_loss = 0
        num_batches = 10
        
        for batch_idx in range(num_batches):
            # 获取 batch
            batch = dataset.get_batch(batch_size)
            
            # 前向传播
            mlm_logits, nsp_logits = model(
                batch['input_ids'],
                batch['token_type_ids'],
                batch['attention_mask']
            )
            
            # 计算 MLM 损失
            mlm_logits_flat = mlm_logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
            mlm_labels_flat = batch['mlm_labels'].view(-1)  # [batch*seq_len]
            mlm_loss = mlm_criterion(mlm_logits_flat, mlm_labels_flat)
            
            # 计算 NSP 损失
            nsp_loss = nsp_criterion(nsp_logits, batch['nsp_labels'])
            
            # 加权组合损失
            total_batch_loss = mlm_weight * mlm_loss + nsp_weight * nsp_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            total_loss += total_batch_loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}")
                print(f"  MLM Loss: {mlm_loss.item():.4f}")
                print(f"  NSP Loss: {nsp_loss.item():.4f}")
                print(f"  Total Loss: {total_batch_loss.item():.4f}\n")
        
        avg_mlm_loss = total_mlm_loss / num_batches
        avg_nsp_loss = total_nsp_loss / num_batches
        avg_total_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1} 完成:")
        print(f"  平均 MLM Loss: {avg_mlm_loss:.4f}")
        print(f"  平均 NSP Loss: {avg_nsp_loss:.4f}")
        print(f"  平均 Total Loss: {avg_total_loss:.4f}\n")
    
    # 测试
    print("="*50)
    print("测试模型")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        test_batch = dataset.get_batch(batch_size=4)
        mlm_logits, nsp_logits = model(
            test_batch['input_ids'],
            test_batch['token_type_ids'],
            test_batch['attention_mask']
        )
        
        # NSP 预测
        nsp_predictions = torch.argmax(nsp_logits, dim=-1)
        nsp_probs = F.softmax(nsp_logits, dim=-1)
        
        print(f"NSP 真实标签: {test_batch['nsp_labels'].tolist()}")
        print(f"NSP 预测标签: {nsp_predictions.tolist()}")
        print(f"NSP 预测概率: {nsp_probs[:, 1].tolist()}")
        
        nsp_correct = (nsp_predictions == test_batch['nsp_labels']).sum().item()
        nsp_accuracy = nsp_correct / len(nsp_predictions)
        print(f"NSP 准确率: {nsp_accuracy:.2%}\n")


def demo_pretraining():
    """演示预训练任务"""
    print("BERT 预训练演示：MLM + NSP")
    print("="*50)
    
    documents = [
        ["The cat sat on the mat.", "It was a sunny day."],
        ["I love programming.", "Python is great."],
    ]
    
    vocab = create_vocab(documents)
    dataset = BERTPretrainingDataset(documents, vocab, max_seq_len=32, mask_prob=0.15)
    
    model = BERTForPretraining(
        vocab_size=len(vocab),
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        max_seq_len=32
    )
    
    # 生成一个样本
    sample = dataset.create_pretraining_sample()
    
    print(f"输入形状:")
    print(f"  input_ids: {sample['input_ids'].shape}")
    print(f"  mlm_labels: {sample['mlm_labels'].shape}")
    print(f"  nsp_label: {sample['nsp_label'].item()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        mlm_logits, nsp_logits = model(
            sample['input_ids'].unsqueeze(0),
            sample['token_type_ids'].unsqueeze(0),
            sample['attention_mask'].unsqueeze(0)
        )
        
        print(f"\n输出形状:")
        print(f"  MLM logits: {mlm_logits.shape}")
        print(f"  NSP logits: {nsp_logits.shape}")
        
        # 计算损失
        mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        nsp_criterion = nn.CrossEntropyLoss()
        
        mlm_loss = mlm_criterion(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            sample['mlm_labels'].view(-1)
        )
        nsp_loss = nsp_criterion(nsp_logits, sample['nsp_label'].unsqueeze(0))
        
        total_loss = mlm_loss + nsp_loss
        
        print(f"\n损失:")
        print(f"  MLM Loss: {mlm_loss.item():.4f}")
        print(f"  NSP Loss: {nsp_loss.item():.4f}")
        print(f"  Total Loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    # 运行演示
    demo_pretraining()
    
    print("\n" + "="*50 + "\n")
    
    # 运行训练（可选，取消注释以运行）
    train_bert_pretraining()

