"""
@Author: yanzx
@Date: 2024/12/15 12:16
@Description:
"""
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# === 数据准备 ===
# 示例数据
data = [
    {"text": "I love this movie!", "label": 1},  # 1: Positive
    {"text": "This is the worst film I've ever seen.", "label": 0},  # 0: Negative
    {"text": "It's okay, not great but not bad either.", "label": 1},  # Neutral treated as positive
]

# 加载数据集
dataset = Dataset.from_list(data)

# === 数据预处理 ===
# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained("./hug-models/bert-base-cased")


# 分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)


# 分词处理
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# === 标签列重命名 ===
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")


# === 评价函数 ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# === 模型加载 ===
# 加载预训练的BERT模型，用于二分类
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# === 模型微调 ===
# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 输出路径
    evaluation_strategy="epoch",  # 每个epoch后评估
    save_strategy="epoch",
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=8,  # 训练时每个设备的批量大小
    per_device_eval_batch_size=8,  # 评估时每个设备的批量大小
    num_train_epochs=3,  # 训练的总轮数
    weight_decay=0.01,  # 权重衰减
    logging_dir="./logs",  # 日志路径
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # 训练数据
    eval_dataset=tokenized_dataset,  # 验证数据（此处为同一数据，仅为示例）
    compute_metrics=compute_metrics,  # 评价函数
)

# 开始训练
trainer.train()
