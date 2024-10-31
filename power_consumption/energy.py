import subprocess
import threading
import time
import csv

def get_cpu_temperature():
    try:
        temp_str = subprocess.check_output("cat /sys/class/thermal/thermal_zone0/temp", shell=True)
        temp_c = int(temp_str) / 1000.0
        return temp_c
    except Exception as e:
        print(f"Error getting CPU temperature: {e}")
        return None

def get_gpu_temperature():
    try:
        temp_str = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", shell=True)
        temp_c = int(temp_str.strip())
        return temp_c
    except Exception as e:
        print(f"Error getting GPU temperature: {e}")
        return None

def get_cpu_usage():
    try:
        cpu_usage_str = subprocess.check_output("grep 'cpu ' /proc/stat", shell=True)
        cpu_times = list(map(int, cpu_usage_str.split()[1:]))
        idle_time = cpu_times[3]
        total_time = sum(cpu_times)
        return 100 * (1 - idle_time / total_time)
    except Exception as e:
        print(f"Error when getting CPU utilization: {e}")
        return None

def get_gpu_usage():
    try:
        gpu_usage_str = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True)
        return int(gpu_usage_str.strip())
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return None

def adjust_power_with_temperature(base_power_watts, T_current, T_threshold, PRR_max):
    if 0 < T_current <= T_threshold:
        alpha_t = 1
    else:
        PRR = (T_threshold - T_current) / T_threshold * PRR_max
        alpha_t = 1 - PRR

    adjusted_power_watts = base_power_watts * alpha_t
    return adjusted_power_watts

def calculate_energy_consumption(base_power_watts_cpu, base_power_watts_gpu, T_threshold, PRR_max):
    with open("temp_and_energy_log.csv", "w", newline='') as csvfile:
        fieldnames = ['Timestamp', 'CPU Temperature (°C)', 'CPU Usage (%)', 'Adjusted CPU Power (W)', 'CPU Energy (J)',
                      'GPU Temperature (°C)', 'GPU Usage (%)', 'Adjusted GPU Power (W)', 'GPU Energy (J)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cpu_temp = get_cpu_temperature()
            gpu_temp = get_gpu_temperature()
            current_cpu_usage = get_cpu_usage()
            current_gpu_usage = get_gpu_usage()

            log_data = {'Timestamp': timestamp}

            if cpu_temp is not None and current_cpu_usage is not None:
                adjusted_cpu_power_watts = adjust_power_with_temperature(base_power_watts_cpu, cpu_temp, T_threshold, PRR_max)
                cpu_energy_joules = (adjusted_cpu_power_watts * current_cpu_usage / 100)
                log_data.update({
                    'CPU Temperature (°C)': f"{cpu_temp:.2f}",
                    'CPU Usage (%)': f"{current_cpu_usage:.2f}",
                    'Adjusted CPU Power (W)': f"{adjusted_cpu_power_watts:.2f}",
                    'CPU Energy (J)': f"{cpu_energy_joules:.2f}"
                })

            if gpu_temp is not None and current_gpu_usage is not None:
                adjusted_gpu_power_watts = adjust_power_with_temperature(base_power_watts_gpu, gpu_temp, T_threshold, PRR_max)
                gpu_energy_joules = (adjusted_gpu_power_watts * current_gpu_usage / 100)
                log_data.update({
                    'GPU Temperature (°C)': f"{gpu_temp:.2f}",
                    'GPU Usage (%)': f"{current_gpu_usage:.2f}",
                    'Adjusted GPU Power (W)': f"{adjusted_gpu_power_watts:.2f}",
                    'GPU Energy (J)': f"{gpu_energy_joules:.2f}"
                })

            writer.writerow(log_data)
            csvfile.flush()
            time.sleep(1)

if __name__ == "__main__":
    base_power_watts_cpu = 115  # Setting the base power
    base_power_watts_gpu = 110
    T_threshold = 40  # Setting the threshold temperature
    PRR_max = 0.3  # Setting the PRR adjustment rate

    # Start a thread to monitor temperature and calculate energy consumption
    energy_thread = threading.Thread(target=calculate_energy_consumption, args=(base_power_watts_cpu,base_power_watts_gpu, T_threshold, PRR_max))
    energy_thread.start()

    try:
        subprocess.run(["python3", "process_run.py"])
    finally:
        # 结束能耗监控
        print("Energy consumption monitoring has ended")
        # 停止监控线程
        energy_thread.join()
