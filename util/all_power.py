import pandas as pd


def sum_average_energies(file1, file2, output_file):
    # 读取第一个CSV文件
    df1 = pd.read_csv(file1)
    # 读取第二个CSV文件
    df2 = pd.read_csv(file2)

    # 检查两列是否可加
    if 'Average Energy (W)' in df1.columns and 'Average Energy (W)' in df2.columns:
        # 对两个DataFrame的'Average Energy (W)'列求和
        df1['Average Energy (W)'] += df2['Average Energy (W)']

        # 将结果保存到新的CSV文件
        df1.to_csv(output_file, index=False)
    else:
        print("CSV文件缺少必要的'Average Energy (W)'列")


file1 = 'D://pyproject//NetMut//result//trans_power//power_trans_stand_ResNet50.csv'
file2 = 'D://pyproject//NetMut//result//30_3080//Power_ResNet_30.csv'
out_file = 'D://pyproject//NetMut//result//all_power//3080//Power_ResNet50_30.csv'
# 使用示例
sum_average_energies(file1, file2, out_file)