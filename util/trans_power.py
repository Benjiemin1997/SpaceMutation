import csv
import pandas as pd

def process_satellite_columns(input_filename, output_filename, columns_to_process):
    # 打开原始CSV文件并读取内容
    with open(input_filename, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        # 创建一个新的CSV文件用于写入处理后的数据
        with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            fieldnames = reader.fieldnames  # 获取原始CSV文件的字段名
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 遍历每一行数据
            for row in reader:
                try:
                    # 对于每一列进行处理
                    for column in columns_to_process:
                        if column in row:
                            # 假设该列包含的是一个字符串形式的元组
                            satellite_value = eval(row[column])  # 将字符串转换成元组
                            if isinstance(satellite_value, tuple) and len(satellite_value) == 2:
                                # 提取第二个元素（假设是浮点数）
                                extracted_value = satellite_value[1]
                                # 更新当前行的该列值
                                row[column] = extracted_value
                            else:
                                raise ValueError(f"Invalid tuple format in column {column}")

                    # 写入处理后的行到新文件
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error processing row: {row}. Error: {e}")

def add_times_column(input_file, output_file, new_column_name='Times'):
    try:
        # 读取原始CSV文件
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 获取行数
    num_rows = len(df)

    # 创建一个从1开始的序列，长度与DataFrame的行数相同
    times_column = pd.Series(range(1, num_rows + 1), name=new_column_name)

    # 将新列添加到DataFrame中
    df[new_column_name] = times_column

    try:
        # 将修改后的DataFrame保存到新的CSV文件中
        df.to_csv(output_file, index=False)
        print(f"File written successfully to {output_file}")
    except PermissionError:
        print(f"Error: Permission denied when writing to {output_file}. Make sure the file is not open in another program.")
    except FileNotFoundError:
        print(f"Error: Directory for {output_file} does not exist. Please check the path.")
    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")

def sum_satellite_values_by_interval(input_file, output_file, interval=60):
    # 加载CSV文件
    df = pd.read_csv(input_file)

    # 确保'Times'列存在
    if 'Times' not in df.columns:
        raise ValueError("The CSV file must contain a 'Times' column.")

    # 初始化空列表来存储每个区间的结果
    results = []

    # 计算总的秒数
    total_seconds = int(df['Times'].max())  # 假设'Times'列是从1开始的递增序列

    # 计算区间数量
    num_intervals = int(total_seconds / interval)

    # 按区间累加Satellite5的值
    for i in range(num_intervals):
        start_time = i * interval + 1
        end_time = (i + 1) * interval
        subset = df[(df['Times'] >= start_time) & (df['Times'] < end_time)]
        sum_satellite5 = subset['Satellite4'].sum()
        average_energy = sum_satellite5 / interval
        results.append({
            'Start Time': start_time,
            'End Time': end_time,
            'Average Energy (W)': average_energy
        })

    # 创建DataFrame并将结果写入新的CSV文件
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

# 使用示例
input_file = "D://pyproject//NetMut//result//trans_power//tranpower_Shufflenetv2.csv"
output_file = 'D://pyproject//NetMut//result//trans_power//tranpower_power_Shufflenetv2.csv'
power_file = 'D://pyproject//NetMut//result//trans_power//power_trans_stand_Shufflenetv2.csv'
columns_to_process = ['Satellite1', 'Satellite2', 'Satellite3', 'Satellite4', 'Satellite5']

# 处理卫星列
process_satellite_columns(input_file, output_file, columns_to_process)

# 添加Times列
add_times_column(output_file, output_file)

# 按区间累加Satellite5的值，并计算平均能量
sum_satellite_values_by_interval(output_file, power_file, interval=60)