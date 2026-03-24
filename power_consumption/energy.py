import os
import csv
import time
import signal
import threading
import subprocess
from typing import Dict, List, Optional


def run_cmd(cmd: List[str], check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check
    )
    return result.stdout.strip()


def get_cpu_temperature() -> Optional[float]:
    candidate_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    temp_str = f.read().strip()
                return int(temp_str) / 1000.0
            except Exception as e:
                print(f"[WARN] Failed to get CPU temperature ({path}): {e}")
    print("[WARN] No available CPU temperature path found")
    return None


def get_gpu_temperature() -> Optional[float]:
    try:
        out = run_cmd([
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits"
        ])
        first_line = out.splitlines()[0].strip()
        return float(first_line)
    except Exception as e:
        print(f"[WARN] Failed to get GPU temperature: {e}")
        return None


class CPUUsageSampler:
    def __init__(self):
        self.prev_total = None
        self.prev_idle = None

    def sample(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r") as f:
                first_line = f.readline().strip()
            parts = first_line.split()
            if parts[0] != "cpu":
                return None
            values = list(map(int, parts[1:]))

            idle = values[3] + values[4]
            total = sum(values)

            if self.prev_total is None or self.prev_idle is None:
                self.prev_total = total
                self.prev_idle = idle
                return None

            delta_total = total - self.prev_total
            delta_idle = idle - self.prev_idle

            self.prev_total = total
            self.prev_idle = idle

            if delta_total <= 0:
                return None

            usage = 100.0 * (1.0 - delta_idle / delta_total)
            return max(0.0, min(100.0, usage))
        except Exception as e:
            print(f"[WARN] Failed to get CPU usage: {e}")
            return None


def get_gpu_usage() -> Optional[float]:
    try:
        out = run_cmd([
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits"
        ])
        first_line = out.splitlines()[0].strip()
        return float(first_line)
    except Exception as e:
        print(f"[WARN] Failed to get GPU usage: {e}")
        return None


def compute_alpha_t(
    T_current: float,
    T_threshold: float,
    PRR_max: float
) -> float:
    if T_current <= T_threshold:
        return 1.0

    overflow_ratio = (T_current - T_threshold) / T_threshold
    prr = min(overflow_ratio * PRR_max, PRR_max)
    alpha_t = 1.0 - prr
    return max(1.0 - PRR_max, min(1.0, alpha_t))


class CPUThermalController:
    def __init__(self):
        self.cpu_infos = self._discover_cpu_freq_interfaces()
        self.original_max_freqs: Dict[int, int] = {}
        self.original_governors: Dict[int, str] = {}

    @staticmethod
    def _discover_cpu_freq_interfaces() -> List[Dict]:
        cpu_infos = []
        cpu_root = "/sys/devices/system/cpu"
        if not os.path.isdir(cpu_root):
            return cpu_infos

        for name in os.listdir(cpu_root):
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            cpu_id = int(name[3:])
            cpufreq_dir = os.path.join(cpu_root, name, "cpufreq")
            if not os.path.isdir(cpufreq_dir):
                continue

            info = {
                "cpu_id": cpu_id,
                "scaling_max_freq": os.path.join(cpufreq_dir, "scaling_max_freq"),
                "scaling_min_freq": os.path.join(cpufreq_dir, "scaling_min_freq"),
                "cpuinfo_max_freq": os.path.join(cpufreq_dir, "cpuinfo_max_freq"),
                "cpuinfo_min_freq": os.path.join(cpufreq_dir, "cpuinfo_min_freq"),
                "scaling_governor": os.path.join(cpufreq_dir, "scaling_governor"),
                "scaling_cur_freq": os.path.join(cpufreq_dir, "scaling_cur_freq"),
            }

            if os.path.exists(info["scaling_max_freq"]):
                cpu_infos.append(info)

        return sorted(cpu_infos, key=lambda x: x["cpu_id"])

    @staticmethod
    def _read_int(path: str) -> Optional[int]:
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None

    @staticmethod
    def _read_str(path: str) -> Optional[str]:
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except Exception:
            return None

    @staticmethod
    def _write_str(path: str, value: str):
        with open(path, "w") as f:
            f.write(value)

    def available(self) -> bool:
        return len(self.cpu_infos) > 0

    def capture_original_state(self):
        for info in self.cpu_infos:
            cpu_id = info["cpu_id"]
            max_freq = self._read_int(info["scaling_max_freq"])
            governor = self._read_str(info["scaling_governor"])
            if max_freq is not None:
                self.original_max_freqs[cpu_id] = max_freq
            if governor is not None:
                self.original_governors[cpu_id] = governor

    def set_governor_userspace_or_powersave(self):
        for info in self.cpu_infos:
            gov_path = info["scaling_governor"]
            try:
                current = self._read_str(gov_path)
                if current is None:
                    continue
                self._write_str(gov_path, "powersave")
            except Exception as e:
                print(f"[WARN] Failed to set governor for CPU{info['cpu_id']}: {e}")

    def apply_alpha(self, alpha_t: float) -> Optional[float]:
        if not self.available():
            return None

        target_freqs = []

        for info in self.cpu_infos:
            cpu_id = info["cpu_id"]
            orig_max = self.original_max_freqs.get(cpu_id)
            min_freq = self._read_int(info["cpuinfo_min_freq"]) or self._read_int(info["scaling_min_freq"])

            if orig_max is None or min_freq is None:
                continue

            target_max = int(orig_max * alpha_t)
            target_max = max(min_freq, min(orig_max, target_max))

            try:
                self._write_str(info["scaling_max_freq"], str(target_max))
                target_freqs.append(target_max)
            except Exception as e:
                print(f"[WARN] Failed to set max frequency for CPU{cpu_id}: {e}")

        if not target_freqs:
            return None

        return sum(target_freqs) / len(target_freqs) / 1000.0

    def get_average_current_freq_mhz(self) -> Optional[float]:
        freqs = []
        for info in self.cpu_infos:
            cur = self._read_int(info["scaling_cur_freq"])
            if cur is not None:
                freqs.append(cur)
        if not freqs:
            return None
        return sum(freqs) / len(freqs) / 1000.0

    def restore(self):
        for info in self.cpu_infos:
            cpu_id = info["cpu_id"]

            if cpu_id in self.original_max_freqs:
                try:
                    self._write_str(info["scaling_max_freq"], str(self.original_max_freqs[cpu_id]))
                except Exception as e:
                    print(f"[WARN] Failed to restore max frequency for CPU{cpu_id}: {e}")

            if cpu_id in self.original_governors:
                try:
                    self._write_str(info["scaling_governor"], self.original_governors[cpu_id])
                except Exception as e:
                    print(f"[WARN] Failed to restore governor for CPU{cpu_id}: {e}")


class GPUThermalController:
    def __init__(self):
        self.available_flag = self._check_nvidia_smi()
        self.original_power_limit_watts: Optional[float] = None
        self.min_power_limit_watts: Optional[float] = None
        self.max_power_limit_watts: Optional[float] = None

    @staticmethod
    def _check_nvidia_smi() -> bool:
        try:
            run_cmd(["nvidia-smi", "-L"])
            return True
        except Exception:
            return False

    def available(self) -> bool:
        return self.available_flag

    def capture_original_state(self):
        if not self.available():
            return

        try:
            out = run_cmd([
                "nvidia-smi",
                "--query-gpu=power.limit,power.min_limit,power.max_limit",
                "--format=csv,noheader,nounits"
            ])
            first_line = out.splitlines()[0].strip()
            parts = [p.strip() for p in first_line.split(",")]
            self.original_power_limit_watts = float(parts[0])
            self.min_power_limit_watts = float(parts[1])
            self.max_power_limit_watts = float(parts[2])
        except Exception as e:
            print(f"[WARN] Failed to read GPU power limit info: {e}")

    def apply_alpha(self, alpha_t: float) -> Optional[float]:
        if not self.available():
            return None
        if self.original_power_limit_watts is None:
            return None
        if self.min_power_limit_watts is None or self.max_power_limit_watts is None:
            return None

        target_power = self.original_power_limit_watts * alpha_t
        target_power = max(self.min_power_limit_watts, min(self.original_power_limit_watts, target_power))

        try:
            run_cmd(["nvidia-smi", "-pl", f"{target_power:.0f}"])
            return target_power
        except Exception as e:
            print(f"[WARN] Failed to set GPU power limit: {e}")
            return None

    def restore(self):
        if not self.available():
            return
        if self.original_power_limit_watts is None:
            return
        try:
            run_cmd(["nvidia-smi", "-pl", f"{self.original_power_limit_watts:.0f}"])
        except Exception as e:
            print(f"[WARN] Failed to restore GPU power limit: {e}")


class ThermalThrottleManager:
    def __init__(
        self,
        cpu_base_power_watts: float,
        gpu_base_power_watts: float,
        cpu_temp_threshold: float,
        gpu_temp_threshold: float,
        prr_max: float,
        sample_interval: float = 1.0,
        log_csv_path: str = "temp_and_energy_log.csv",
    ):
        self.cpu_base_power_watts = cpu_base_power_watts
        self.gpu_base_power_watts = gpu_base_power_watts
        self.cpu_temp_threshold = cpu_temp_threshold
        self.gpu_temp_threshold = gpu_temp_threshold
        self.prr_max = prr_max
        self.sample_interval = sample_interval
        self.log_csv_path = log_csv_path

        self.stop_event = threading.Event()

        self.cpu_usage_sampler = CPUUsageSampler()
        self.cpu_controller = CPUThermalController()
        self.gpu_controller = GPUThermalController()

    def start(self):
        self.cpu_controller.capture_original_state()
        self.cpu_controller.set_governor_userspace_or_powersave()
        self.gpu_controller.capture_original_state()

    def stop(self):
        self.stop_event.set()
        self.cpu_controller.restore()
        self.gpu_controller.restore()

    def loop(self):
        fieldnames = [
            "Timestamp",
            "CPU Temperature (C)",
            "CPU Usage (%)",
            "CPU Alpha_t",
            "CPU Applied Max Freq (MHz)",
            "CPU Current Freq (MHz)",
            "CPU Effective Power (W)",
            "CPU Estimated Energy (J)",
            "GPU Temperature (C)",
            "GPU Usage (%)",
            "GPU Alpha_t",
            "GPU Applied Power Limit (W)",
            "GPU Effective Power (W)",
            "GPU Estimated Energy (J)",
        ]

        with open(self.log_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            _ = self.cpu_usage_sampler.sample()

            while not self.stop_event.is_set():
                start_time = time.time()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                cpu_temp = get_cpu_temperature()
                gpu_temp = get_gpu_temperature()
                cpu_usage = self.cpu_usage_sampler.sample()
                gpu_usage = get_gpu_usage()

                row = {"Timestamp": timestamp}

                cpu_alpha = None
                cpu_applied_max_freq_mhz = None
                cpu_cur_freq_mhz = None
                cpu_effective_power = None
                cpu_energy_joules = None

                if cpu_temp is not None:
                    cpu_alpha = compute_alpha_t(
                        T_current=cpu_temp,
                        T_threshold=self.cpu_temp_threshold,
                        PRR_max=self.prr_max
                    )
                    cpu_applied_max_freq_mhz = self.cpu_controller.apply_alpha(cpu_alpha)
                    cpu_cur_freq_mhz = self.cpu_controller.get_average_current_freq_mhz()
                    cpu_effective_power = self.cpu_base_power_watts * cpu_alpha

                    if cpu_usage is not None:
                        cpu_energy_joules = (
                            cpu_effective_power * (cpu_usage / 100.0) * self.sample_interval
                        )

                gpu_alpha = None
                gpu_applied_power_limit = None
                gpu_effective_power = None
                gpu_energy_joules = None

                if gpu_temp is not None and self.gpu_controller.available():
                    gpu_alpha = compute_alpha_t(
                        T_current=gpu_temp,
                        T_threshold=self.gpu_temp_threshold,
                        PRR_max=self.prr_max
                    )
                    gpu_applied_power_limit = self.gpu_controller.apply_alpha(gpu_alpha)
                    gpu_effective_power = self.gpu_base_power_watts * gpu_alpha

                    if gpu_usage is not None:
                        gpu_energy_joules = (
                            gpu_effective_power * (gpu_usage / 100.0) * self.sample_interval
                        )

                row.update({
                    "CPU Temperature (C)": f"{cpu_temp:.2f}" if cpu_temp is not None else "",
                    "CPU Usage (%)": f"{cpu_usage:.2f}" if cpu_usage is not None else "",
                    "CPU Alpha_t": f"{cpu_alpha:.4f}" if cpu_alpha is not None else "",
                    "CPU Applied Max Freq (MHz)": f"{cpu_applied_max_freq_mhz:.2f}" if cpu_applied_max_freq_mhz is not None else "",
                    "CPU Current Freq (MHz)": f"{cpu_cur_freq_mhz:.2f}" if cpu_cur_freq_mhz is not None else "",
                    "CPU Effective Power (W)": f"{cpu_effective_power:.2f}" if cpu_effective_power is not None else "",
                    "CPU Estimated Energy (J)": f"{cpu_energy_joules:.2f}" if cpu_energy_joules is not None else "",
                    "GPU Temperature (C)": f"{gpu_temp:.2f}" if gpu_temp is not None else "",
                    "GPU Usage (%)": f"{gpu_usage:.2f}" if gpu_usage is not None else "",
                    "GPU Alpha_t": f"{gpu_alpha:.4f}" if gpu_alpha is not None else "",
                    "GPU Applied Power Limit (W)": f"{gpu_applied_power_limit:.2f}" if gpu_applied_power_limit is not None else "",
                    "GPU Effective Power (W)": f"{gpu_effective_power:.2f}" if gpu_effective_power is not None else "",
                    "GPU Estimated Energy (J)": f"{gpu_energy_joules:.2f}" if gpu_energy_joules is not None else "",
                })

                writer.writerow(row)
                csvfile.flush()

                print(
                    f"[THERMAL] {timestamp} | "
                    f"CPU T={cpu_temp}C alpha={cpu_alpha} freq_cap={cpu_applied_max_freq_mhz}MHz cur={cpu_cur_freq_mhz}MHz | "
                    f"GPU T={gpu_temp}C alpha={gpu_alpha} pl={gpu_applied_power_limit}W"
                )

                elapsed = time.time() - start_time
                remaining = self.sample_interval - elapsed
                if remaining > 0:
                    self.stop_event.wait(timeout=remaining)


def main():
    cpu_base_power_watts = 100.0
    gpu_base_power_watts = 150.0

    cpu_temp_threshold = 75.0
    gpu_temp_threshold = 80.0

    prr_max = 0.5
    sample_interval = 1.0

    target_script = "main.py"

    manager = ThermalThrottleManager(
        cpu_base_power_watts=cpu_base_power_watts,
        gpu_base_power_watts=gpu_base_power_watts,
        cpu_temp_threshold=cpu_temp_threshold,
        gpu_temp_threshold=gpu_temp_threshold,
        prr_max=prr_max,
        sample_interval=sample_interval,
        log_csv_path="temp_and_energy_log.csv",
    )

    monitor_thread = None
    child = None

    def cleanup_and_exit(signum=None, frame=None):
        print("\n[INFO] Cleaning up and restoring original CPU/GPU settings...")
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
        except Exception as e:
            print(f"[WARN] Failed to terminate child process: {e}")

        try:
            manager.stop()
        except Exception as e:
            print(f"[WARN] Failed to restore controller state: {e}")

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    try:
        manager.start()

        monitor_thread = threading.Thread(target=manager.loop, daemon=True)
        monitor_thread.start()

        print(f"[INFO] Starting target program: {target_script}")
        child = subprocess.Popen(["python3", target_script])

        return_code = child.wait()
        print(f"[INFO] Target program finished with return code: {return_code}")

    finally:
        cleanup_and_exit()
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=5)
        print("[INFO] Thermal throttling monitor stopped, original settings restored.")


if __name__ == "__main__":
    main()