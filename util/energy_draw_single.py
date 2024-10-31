import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# 文件夹路径
folder_path = 'D://pyproject//NetMut//result//all_power//atlas'

# 颜色列表
colors = ['#223474', '#fd7a44', '#eb6466', '#1e8647', '#0dc4e0']

# 文件名列表
files = ['Power_AlexNet_30.csv', 'Power_LeNet5_30.csv', 'Power_ResNet50_30.csv', 'Power_UNet_30.csv',
         'Power_Shufflenet_30.csv']

labels = ['AlexNet', 'LeNet-5', 'ResNet-50', 'U-Net', 'ShuffleNet-v2']
# 定义图表的宽度和高度，以适应双栏页面
fig, ax = plt.subplots(figsize=(5.5, 3.5))  # 假设每栏的宽度大约为3.25英寸

# 循环读取每个文件并绘制
for i, file_name in enumerate(files):
    # 读取CSV文件
    df = pd.read_csv(os.path.join(folder_path, file_name), encoding='gbk')

    # 确保'Times'和'Average Energy (W)'列存在
    assert 'Times' in df.columns, f"'Times' 列不存在于 {file_name}"
    assert 'Average Energy (W)' in df.columns, f"'Average Energy (W)' 列不存在于 {file_name}"

    # 清理数据，去除含有NaN或Inf的行
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Times', 'Average Energy (W)'])

    # 按时间序列排序数据
    df = df.sort_values(by='Times')

    # 获取时间和平均能量数据
    times = df['Times'].values
    average_energy = df['Average Energy (W)'].values

    # 绘制折线图，并在数据点上添加三角形标记
    ax.plot(times, average_energy, linestyle='-', marker='^', markersize=5, linewidth=2, color=colors[i], label=labels[i])

# 设置图表标题和坐标轴标签
ax.set_xlabel('Times (min)', fontsize=14, fontweight='bold')
ax.set_ylabel('Power (W)', fontsize=14, fontweight='bold')

# 设置刻度标签的字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 手动设置Y轴的刻度
y_ticks = np.arange(0, 10, 2)  # 每2一个单位
ax.set_yticks(y_ticks)

# 添加图例，并将其放置在图表顶部中央
# 计算图例的列数
num_items = len(ax.get_legend_handles_labels()[0])
ncol = 3  # 设定为三列

# 添加图例，并设置图例边框样式
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=False, ncol=ncol,
          prop={'weight': 'bold'},
          frameon=False)  # 不显示边框

# 调整图表的边距，以便更好地适应页面布局
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图表
plt.savefig(os.path.join(folder_path, "atlas_all_models_power.png"))

# 显示图表
plt.show()