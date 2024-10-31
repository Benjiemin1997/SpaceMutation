import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('D://pyproject//NetMut//result//4060//temp_and_energy_log_AlexNet_30.csv')

# 将'Times'列转换为datetime类型
df['Times'] = pd.to_datetime(df['Times'], unit='s')

# 按照分钟对数据进行分组，并计算总能量之和
grouped_df = df.groupby(pd.Grouper(key='Times', freq='30S')).sum()

# 绘制折线图
plt.plot(grouped_df.index, grouped_df['Total Energy (J)'])
plt.xlabel('Time')
plt.ylabel('Total Energy (J)')
plt.title('Energy Consumption over Time')
plt.show()