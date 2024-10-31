import csv

# 定义输入输出文件名
input_csv_file = 'D://pyproject//NetMut//result//atlas_30//energy_alexnet_30.csv'
output_csv_file = 'D://pyproject//NetMut//result//atlas_30//energy_alexnet_sum_30.csv'

# 定义基础功率、温度阈值和最大功率调节率
base_power_watts = 25.0  # 假设基础功率为65W
T_threshold = 35  # 假设温度阈值为50℃
PRR_max = 0.3  # 假设最大功率调节率为10%

def adjust_power_with_temperature(base_power_watts, T_current, T_threshold, PRR_max):
    if 0 < T_current <= T_threshold:
        alpha_t = 1
    else:
        PRR = abs(T_threshold - T_current) / T_threshold * PRR_max
        alpha_t = 1 - PRR

    adjusted_power_watts = base_power_watts * alpha_t
    return adjusted_power_watts

# 读取原始数据
with open(input_csv_file, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['Adjusted CPU Power (W)', 'CPU Energy (J)']

    # 创建新文件
    with open(output_csv_file, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            cpu_temp = float(row['Temp(C)']) if row['Temp(C)'] else None
            current_cpu_usage = float(row['AI Cpu(%)']) if row['AI Cpu(%)'] else None

            if cpu_temp is not None and current_cpu_usage is not None:
                adjusted_cpu_power_watts = adjust_power_with_temperature(base_power_watts, cpu_temp, T_threshold,
                                                                         PRR_max)
                cpu_energy_joules = adjusted_cpu_power_watts * (current_cpu_usage / 100)

                # 更新行数据并写入
                row.update({
                    'Adjusted CPU Power (W)': adjusted_cpu_power_watts,
                    'CPU Energy (J)': cpu_energy_joules
                })
                writer.writerow(row)