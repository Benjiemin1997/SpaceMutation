import pandas as pd


def process_and_save_energy_data(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file,encoding='utf-8')

    # 创建一个区间范围，用于将 Times 划分为每分钟的区间
    bins = range(0, 1801, 60)  # 每60秒为一个区间

    # 使用cut函数根据Times的值划分区间，每60个值为一个区间（代表一分钟）
    df['Minute'] = pd.cut(df['Times'], bins=bins)

    # 按照区间计算每分钟的能量总和
    energy_per_minute = df.groupby('Minute')['Total Energy (J)'].sum().reset_index()
    energy_per_minute['Average Energy (W)'] = energy_per_minute['Total Energy (J)'] / 60

    # 将区间转换为具体的分钟数
    energy_per_minute['Times'] = [int(x.right / 60) for x in energy_per_minute['Minute']]

    # 选择需要的列，并删除不再需要的 Minute 列
    energy_per_minute = energy_per_minute[['Times', 'Average Energy (W)']]

    # 将结果写入新的CSV文件
    energy_per_minute.to_csv(output_file, index=False)


# 替换下面的路径为你实际的输入和输出CSV文件路径
process_and_save_energy_data('D://pyproject//NetMut//result//3080_10//temp_and_energy_log_VGG16_baseline.csv', 'D://pyproject//NetMut//result//3080_10/Power_VGG16_baseline.csv')