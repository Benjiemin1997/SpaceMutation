import matplotlib.pyplot as plt

# 定义神经网络模型和对应的通过率
models = ['VGG-16', 'AlexNet', 'LeNet-5', 'ResNet-50', 'UNet', 'ShuffleNet']
passing_rates = [23, 10, 19, 14, 11,17]

# 自定义颜色列表
colors = ['#FF9999','#66B2FF','#99FF99','#FFCC99','#CCCCFF','#4fffd0']

# 设置画布大小
#fig, ax = plt.subplots(figsize=(3.31, 2.5))  # 单栏宽度大约为3.31英寸
fig, ax = plt.subplots(figsize=(4.25, 3))
# 创建一个柱状图，并设置柱子的宽度
bars = ax.bar(models, passing_rates, color=colors, width=0.4)

# 添加标题和轴标签
ax.set_xlabel('DNN Models')
ax.set_ylabel('Passing Rate (%)')

# 旋转X轴上的文字标签以便更好地适应
plt.xticks(rotation=45, ha="right")

# 计算y轴的最大值并预留一定的空间
max_passing_rate = max(passing_rates)
ax.set_ylim(top=max_passing_rate + 4)  # 给上方预留一点空间

# 添加数据标签
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}', ha='center', va='bottom', fontsize=10)  # 使用较小的字体大小，并居中

# 调整布局以防止标题与X轴标签重叠
fig.tight_layout()
plt.savefig('pass_rate.png')
# 显示图表
plt.show()