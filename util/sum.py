import pandas as pd

def add_energy_columns(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file,encoding='utf-8')

    # 检查必要的列是否存在
    if 'CPU Energy (J)' not in df.columns or 'GPU Energy (J)' not in df.columns:
        print("CSV 文件缺少必要的列，请确保包含 'CPU Energy (J)' 和 'GPU Energy (J)' 列。")
        return

    # 计算总能量
    df['Total Energy (J)'] = df['CPU Energy (J)'] + df['GPU Energy (J)']

    # 将修改后的 DataFrame 写回到 CSV 文件
    df.to_csv(csv_file, index=False)


def add_times_column(input_file, output_file, new_column_name='Times'):
    # 读取原始CSV文件
    df = pd.read_csv(input_file)

    # 获取行数
    num_rows = len(df)

    # 创建一个从1开始的序列，长度与DataFrame的行数相同
    times_column = pd.Series(range(1, num_rows + 1), name=new_column_name)

    # 将新列添加到DataFrame中
    df[new_column_name] = times_column

    # 将修改后的DataFrame保存到新的CSV文件中
    df.to_csv(output_file, index=False)


def remove_zero_values(input_file, output_file, column_name='GPU Energy (J)'):
    # 读取原始CSV文件
    df = pd.read_csv(input_file)

    # 移除指定列值为0的行
    df = df[df[column_name] != 0]

    # 将修改后的DataFrame保存到新的CSV文件中
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    csv_file = "D://pyproject//NetMut//result//3080_10//temp_and_energy_log_VGG16_baseline.csv"  # 替换为你的 CSV 文件路径
    remove_zero_values(csv_file,csv_file,'GPU Energy (J)')
    add_energy_columns(csv_file)
    add_times_column(csv_file,csv_file,'Times')
    print(f"已将 Total Energy (J) 列添加到 {csv_file}。")