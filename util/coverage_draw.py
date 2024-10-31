import matplotlib.pyplot as plt

models = ['VGG-16', 'AlexNet', 'LeNet-5', 'ResNet-50', 'UNet', 'ShuffleNet']
passing_rates = [94.26, 70.4, 73.1, 73.52, 74.91, 74.35]


colors = ['#FF9999','#66B2FF','#99FF99','#FFCC99','#CCCCFF','#4fffd0']


fig, ax = plt.subplots(figsize=(4.25,3))

bars = ax.bar(models, passing_rates, color=colors, width=0.4)


ax.set_xlabel('DNN Models')
ax.set_ylabel('Coverage Rate (%)')

plt.xticks(rotation=45, ha="right")


max_passing_rate = max(passing_rates)
ax.set_ylim(top=max_passing_rate + 15)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}',
            ha='center', va='bottom', fontsize=10)


fig.tight_layout()
plt.savefig('coverage_rate.png')
plt.show()