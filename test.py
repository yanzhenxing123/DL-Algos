import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split
)
from torch import optim
from loguru import logger

epochs = 20
num_samples = 100
num_features = 3
batch_size = 4
lr = 0.01


class MyDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        samples = {
            "data": self.data[index],
            "label": self.labels[index]
        }
        return samples


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


# 定义损失函数和优化器

def train(model, dataloader, epochs):
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in dataloader:
            X, y = batch['data'], batch['label']
            optimizer.zero_grad()  # 梯度归零
            outputs = model(X)
            loss = criterion(outputs, y.view(-1, 1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = sum(losses) / len(losses)
        logger.info(f"epoch {epoch} loss: {avg_loss:.4f}")


def main():
    # 1. 生成数据
    data = torch.randn(num_samples, num_features)
    labels = []
    for data_e in data:
        y_e = data_e[0] + 2 * data_e[1] + 3 * data_e[2]
        labels.append(y_e.item())
    labels = torch.tensor(labels)
    my_dataset = MyDataSet(data, labels)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    # 2. 训练
    model = Net(input_size=num_features, output_size=1)
    train(model, my_dataloader, epochs)


if __name__ == '__main__':
    main()
