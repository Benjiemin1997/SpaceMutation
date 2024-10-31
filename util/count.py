import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./temp_and_energy_log_AlexNet_30.csv')


df['Times'] = pd.to_datetime(df['Times'], unit='s')
grouped_df = df.groupby(pd.Grouper(key='Times', freq='30S')).sum()

# 绘制折线图
plt.plot(grouped_df.index, grouped_df['Total Energy (J)'])
plt.xlabel('Time')
plt.ylabel('Total Energy (J)')
plt.title('Energy Consumption over Time')
plt.show()