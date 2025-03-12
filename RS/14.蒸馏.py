"""
@Author: yanzx
@Date: 2025/3/12 10:52
@Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义教师模型和学生模型
class TeacherModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# 示例数据
input_dim = 128
hidden_dim = 64
x = torch.randn(1000, input_dim)  # 输入特征
y = torch.randint(0, 2, (1000, 1)).float()  # 真实标签

# 初始化教师模型和学生模型
teacher_model = TeacherModel(input_dim, hidden_dim)
student_model = StudentModel(input_dim, hidden_dim)

# 训练教师模型（略）

# 知识蒸馏
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
criterion_real = nn.BCELoss()  # 真实标签的损失
criterion_soft = nn.KLDivLoss(reduction='batchmean')  # 软标签的损失（KL 散度）

for epoch in range(10):
    optimizer.zero_grad()
    student_outputs = student_model(x)
    teacher_outputs = teacher_model(x).detach()  # 教师模型的软标签

    # 计算损失
    loss_real = criterion_real(student_outputs, y)  # 真实标签的损失
    loss_soft = criterion_soft(F.log_softmax(student_outputs, dim=1), F.softmax(teacher_outputs, dim=1))  # 软标签的损失
    loss = loss_real + 0.5 * loss_soft  # 联合损失

    # 反向传播
    loss.backward()
    optimizer.step()
    print(f"Student Epoch {epoch + 1}, Loss: {loss.item()}")
