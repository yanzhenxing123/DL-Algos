import matplotlib.pyplot as plt
import numpy as np

# 定义3x3 RGB图像（形状：3高度 x 3宽度 x 3通道）
image = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # 第一行：红、绿、蓝
    [[255, 255, 0], [0, 0, 0], [255, 255, 255]],  # 第二行：黄、黑、白
    [[128, 128, 128], [255, 0, 255], [0, 255, 255]]  # 第三行：灰、紫、青
], dtype=np.uint8)

# 显示图像
plt.imshow(image)
plt.title("3x3 RGB Image Example")
plt.axis('off')  # 隐藏坐标轴
for y in range(3):
    for x in range(3):
        plt.text(x, y, f'({x},{y})', color='black', ha='center', va='center')
plt.show()
