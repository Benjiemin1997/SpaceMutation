import csv
import time


def parse_log_line(line):
    """解析.log文件的一行数据"""
    # 使用正则表达式或其他合适的分割方法来提取数据
    parts = line.split()
    if len(parts) == 9:
        return parts
    return None


# 假设.log文件的名字是"example.log"
log_file_name = "info_alexnet.log"
csv_file_name = "energy_alexnet_30.csv"

# 用于存储列名
column_names = ['NpuID(Idx)', 'ChipId(Idx)', 'Pwr(W)', 'Temp(C)',
                'AI Core(%)', 'AI Cpu(%)', 'Ctrl Cpu(%)', 'Memory(%)', 'Memory BW(%)']

# 打开.log文件进行读取
with open(log_file_name, mode='r', encoding='utf-8') as log_file:
    # 打开.csv文件进行写入
    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # 写入表头
        writer.writerow(column_names)

        # 遍历.log文件的每一行
        for line in log_file:
            parsed_data = parse_log_line(line.strip())
            if parsed_data is not None:
                # 将解析后的数据写入.csv文件
                writer.writerow(parsed_data)

print(f"数据已成功从{log_file_name}写入到{csv_file_name}")

