import pandas as pd
import matplotlib.pyplot as plt
import os

# 文件夹路径
folder_path = 'D://pyproject//NetMut//result//4060'

# 获取文件夹中所有的.csv文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 创建一个更大的图表
plt.figure(figsize=(14, 8))

# 定义颜色列表
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

# 遍历每个CSV文件
for i, csv_file in enumerate(csv_files):
    file_path = os.path.join(folder_path, csv_file)
    data = pd.read_csv(file_path)

    # 提取数据
    times = data['Times']
    total_energy = data['Total Energy (J)']

    # 绘制折线图
    plt.plot(times, total_energy, label=csv_file.replace('_', ' ').replace('.csv', ''), color=colors[i % len(colors)])

# 添加标题和标签
plt.title('Energy Consumption Over Time', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Energy (J)', fontsize=14)
plt.legend(loc='best', fontsize=10)

# 显示网格


# 显示图表
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()