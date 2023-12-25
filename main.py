"""
@Time: 2023/12/24 17:56
@Author: yanzx
@Description: 
"""
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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyDataSet(Dataset):
    """数据集"""

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
    """模型"""

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


# 定义损失函数和优化器

def train(model, dataloader, epochs):
    """
    训练
    :param model:
    :param dataloader:
    :param epochs:
    :return:
    """
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in dataloader:
            X, y = batch['data'], batch['label']
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()  # 梯度归零
            outputs = model(X)
            loss = criterion(outputs, y.view(-1, 1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = sum(losses) / len(losses)
        logger.info(f"epoch {epoch} loss: {avg_loss:.4f}")


def test(model, testloader):
    """
    测试
    :param model:
    :param testloader:
    :return:
    """
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for data in testloader:
            X_test, y_test = data['data'], data['label']
            X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
            outputs = model(X_test)
            loss = criterion(outputs, y_test.view(-1, 1))
            losses.append(loss)
    logger.info(f"Total avg loss:{sum(losses) / len(losses):.4f}")
    return loss


def get_data():
    """
    模拟生成数据
    :return:
    """
    data = torch.randn(num_samples, num_features)
    labels = []
    for data_e in data:
        y_e = data_e[0] + 2 * data_e[1] + 3 * data_e[2]
        labels.append(y_e.item())
    labels = torch.tensor(labels)
    return data, labels


def main():
    # 1. 生成数据
    data, labels = get_data()
    my_dataset = MyDataSet(data, labels)

    # 2. 训练
    train_size = int(0.8 * len(my_dataset))
    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = Net(input_size=num_features, output_size=1)
    train(model, train_dataloader, epochs)

    # 3. 保存模型
    torch.save(model.state_dict(), 'data/model/demo_model.pth')

    # 4. 测试
    model = Net(input_size=num_features, output_size=1)
    model.load_state_dict(torch.load("data/model/demo_model.pth"))
    test(model, test_dataloader)

    # 5. 查看参数信息
    for key, value in model.state_dict().items():
        print(key, value)
    print(model.state_dict()['fc.bias'].item())


if __name__ == '__main__':
    main()
