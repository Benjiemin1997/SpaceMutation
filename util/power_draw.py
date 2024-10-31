import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 颜色列表
colors = ['#66B2FF', '#99FF99', '#FFCC99', '#fd7a44', '#eb6466']

folder_path = 'D://pyproject//NetMut//result//power_result//all//'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 模型名称列表
model_names = [
    ['AlexNet (NVIDIA RTX 3080)', 'LeNet-5 (NVIDIA RTX 3080)', 'ResNet-50 (NVIDIA RTX 3080)', 'UNet (NVIDIA RTX 3080)', 'VGG16 (NVIDIA RTX 3080)'],
    ['AlexNet (NVIDIA RTX 4060)', 'LeNet-5 (NVIDIA RTX 4060)', 'ResNet-50 (NVIDIA RTX 4060)', 'UNet (NVIDIA RTX 4060)', 'VGG16 (NVIDIA RTX 4060)'],
    ['AlexNet (Atlas 200I DK A2)', 'LeNet-5 (Atlas 200I DK A2)', 'ResNet-50 (Atlas 200I DK A2)', 'ShuffleNet-v2 (Atlas 200I DK A2)', 'UNet (Atlas 200I DK A2)']
]

# 计算子图的数量
num_subplots = len(csv_files)

# 创建一个新的大图
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(17, 8))

# 如果只有一个子图，则 axs 是单一的对象而不是数组
if not isinstance(axs, np.ndarray):
    axs = [axs]

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 10
})

for i, csv_file in enumerate(csv_files):
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)

    # 获取温度和平均能量
    temperatures = df['temperature'].values
    average_energies = df['Average Energy (W)'].values

    # 过滤掉40度的数据点
    filtered_df = df[df['temperature'] != 40]

    # 获取过滤后的温度和平均能量
    filtered_temperatures = filtered_df['temperature'].values
    filtered_average_energies = filtered_df['Average Energy (W)'].values

    # 创建柱状图
    ax = axs.flat[i]

    # 创建柱状图，并为每个柱子设置颜色
    for temp, energy, color in zip(filtered_temperatures, filtered_average_energies, colors):
        ax.bar(temp, energy, width=8, align='center', color=color)

    # 查找40度时的平均能量值
    standard_energy_row = df[df['temperature'] == 40]
    if not standard_energy_row.empty:
        standard_energy = standard_energy_row['Average Energy (W)'].values[0]
        ax.axhline(y=standard_energy, color='r', linestyle='--', linewidth=1.5,
                   label=f'Average Power at 40°C: {standard_energy:.2f} W')

    # 设置X轴的刻度
    ax.set_xticks([30, 50, 70])

    # 设置子图的标题和标签
    row = i // 5  # 行索引
    col = i % 5  # 列索引
    ax.set_title(model_names[row][col], fontsize=10)

    # 只有在最后一行的子图时才设置 X 轴标签
    if i // 5 == 2:
        ax.set_xlabel('Temperature (°C)', fontsize=10)
    else:
        ax.set_xlabel('')  # 清空其他子图的 X 轴标签

    # 设置 Y 轴标签
    if i % 5 == 0:
        ax.set_ylabel('Average Power (W)', fontsize=10)
    else:
        ax.set_ylabel('')  # 清空其他子图的 Y 轴标签

    # 保留 Y 轴的刻度
    # ax.yaxis.set_ticklabels([])  # 清空 Y 轴的刻度标签

    # 设置X轴和Y轴的颜色为黑色
    ax.tick_params(axis='both', colors='black')

    # 设置子图的边框为黑色正方形
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # 添加图例
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
              fancybox=True, shadow=True, ncol=2)  # 增加列数使图例更宽

# 自动调整子图间距
plt.tight_layout()

# 保存图像
output_path = f'D://pyproject//NetMut//result//power_result//all//all_Average_Energy.png'
plt.savefig(output_path)

# 显示图像
plt.show()