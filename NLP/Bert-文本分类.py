import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, SGD, Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained(
    '/Users/bytedance/Desktop/Files/预训练模型/bert-base-chinese')  # 如果自动下载失败，请手动访问https://github.com/google-research/bert?tab=readme-ov-file下载到本地，并修改目录路径
model = BertForSequenceClassification.from_pretrained('/Users/bytedance/Desktop/Files/预训练模型/bert-base-chinese', num_labels=2)

## 读取Excel文件
# df = pd.read_excel('train_data.xlsx')
## 提取标签和文本数据
# labels = df.iloc[1:, 0].values
# texts = df.iloc[1:, 1].values
## 创建一个字典，其中键是标签，值是一个唯一的数字
# label_dict = {label: i for i, label in enumerate(set(labels))}


# print(label_dict)
## 使用字典将文本标签转换为数字
# labels = [label_dict[label] for label in labels]

texts = ["你好，世界！", "机器学习太棒了！"]
labels = [0, 1]

## 将数据集分为训练集和测试集
# train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, #test_size=0.2, random_state=13)

## 使用训练集创建训练数据加载器
# train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_len=128)
# train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

## 使用测试集创建测试数据加载器
# test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_len=128)
# test_data_loader = DataLoader(test_dataset, batch_size=128)

# 使用提取的标签和文本数据创建数据集
all_dataset = TextClassificationDataset(texts, labels, tokenizer, max_len=128)
all_data_loader = DataLoader(all_dataset, batch_size=2, shuffle=True)

# 检查是否有可用的CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载保存的状态字典
# model.load_state_dict(torch.load('my_model.pth'))

# 将模型移动到CUDA设备上
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)
epochs = 10  # 定义训练的轮数

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for batch in all_data_loader:
        # 将输入数据移动到CUDA设备上
        input_ids = batch['input_ids'].to(device)  # (2, 128)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(all_data_loader)
    print(f"Epoch {epoch + 1} / {epochs}, Training Loss: {avg_train_loss}")

    # 评估模型
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in all_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            predicted_class = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predicted_class == labels).sum().item()
            total_predictions += labels.size(0)

    # 计算准确率
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy}")

print("Training complete.")
# 将模型移动到CPU设备上
model = model.to('cpu')
# 在训练结束后保存模型的状态字典
torch.save(model.state_dict(), 'my_model.pth')

print("Model saved to model.pth.")

# ----------------------------------------------------
# 初始化一个新的模型实例
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 加载保存的状态字典
model.load_state_dict(torch.load('my_model.pth'))

# 将模型移动到CUDA设备上
model = model.to(device)

print("Model loaded from model.pth.")
model.eval()
new_text = "你好，世界！"

encoding = tokenizer.encode_plus(
    new_text,
    add_special_tokens=True,
    max_length=128,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
)

input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

with torch.no_grad():
    # 将输入数据移动到CUDA设备上
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask)

predicted_class = torch.argmax(outputs.logits, dim=1)
# 创建一个反向字典，其中键是数字，值是文本标签
# reverse_label_dict = {v: k for k, v in label_dict.items()}

# 使用反向字典将预测的类别转换回文本标签
# predicted_class = predicted_class.item()
# predicted_label = reverse_label_dict[predicted_class]

# print(f"The predicted class for the text '{new_text}' is {predicted_label}.")
print(f"The predicted class for the text '{new_text}' is {predicted_class.item()}.")
