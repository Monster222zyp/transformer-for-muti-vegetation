import os
from pathlib import Path

# Motion profile hyperparameters
T1 = 1
T = 6

# Velocity sweep hyperparameters
VMIN = 0.1
VMAX = 0.4
VSTEP = 0.1
V_FILENAME_DECIMALS = 2

# Output timeline start and sampling step
T_START = 0.0
DT = 0.01

OUTPUT_DIRNAME = "output"
HEADERS = ["t", "x", "y", "z", "b", "c", "vx", "vy", "vz", "vb", "vc"]

# SCRIPT_DIR 是当前脚本所在的 Experiment 文件夹，REPO_ROOT 是仓库根目录。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# 运动轨迹统一写入仓库根目录的 output，避免在 Experiment 内产生第二份输出目录。
OUTPUT_DIR = REPO_ROOT / OUTPUT_DIRNAME

# 内部运动学计算仍使用 m 和 m/s；写入 CSV 前统一使用该比例转换成 mm 和 mm/s。
# 保留原有列名可避免依赖 x、vx 等列名的后续脚本失效。
M_TO_MM = 1000.0


def velocity_at_time(t: float, vm: float, t1: float, t2: float) -> float:
    """Piecewise speed profile: accel -> constant -> decel."""
    a = vm / t1
    t_acc_end = t1
    t_const_end = t1 + t2
    t_dec_end = 2.0 * t1 + t2

    if t < 0.0:
        return 0.0
    if t < t_acc_end:
        return a * t
    if t <= t_const_end:
        return vm
    if t < t_dec_end:
        return vm - a * (t - t_const_end)
    return 0.0


def calculate_t2(t1: float, total_time: float) -> float:
    """Compute constant-speed duration from total motion time."""
    t2 = total_time - 2.0 * t1
    if t2 < -1e-12:
        raise ValueError(f"T must be >= 2*T1. Got T={total_time}, T1={t1}.")
    return max(t2, 0.0)


def calculate_total_distance(vm: float, t1: float, total_time: float) -> float:
    """Total distance (area under v-t curve) for trapezoidal profile."""
    # area = vm * (t2 + t1), and t2 = total_time - 2*t1
    return vm * (total_time - t1)


def generate_profile_array(
    vm: float,
    t1: float = T1,
    total_time: float = T,
    t_start: float = T_START,
    dt: float = DT,
) -> list[list[float]]:
    """Core function: input Vm, output motion profile array."""
    if vm <= 0:
        raise ValueError("Vm must be positive.")
    if t1 <= 0 or total_time <= 0:
        raise ValueError("T1 and T must both be positive.")
    if dt <= 0:
        raise ValueError("DT must be positive.")
    if total_time < t_start:
        raise ValueError("T must be >= T_START.")

    t2 = calculate_t2(t1, total_time)
    t_end = total_time
    num_steps = int(round((t_end - t_start) / dt)) + 1

    rows: list[list[float]] = []
    for i in range(num_steps):
        t = t_start + i * dt
        v = max(velocity_at_time(t, vm, t1, t2), 0.0)
        # 以下位移和速度变量先保留 m、m/s 单位，便于保持原有的运动学计算公式。
        vc = 0.0
        vx = v
        x = vx * dt

        # position_values_m 保存五个位移列的内部 m 单位数值。
        position_values_m = [x, 0.0, 0.0, 0.0, 0.0]

        # velocity_values_m_per_s 保存五个速度列的内部 m/s 单位数值。
        velocity_values_m_per_s = [vx, 0.0, 0.0, 0.0, vc]

        # 时间 t 保持秒单位不变；其余十列都乘 1000，分别以 mm 和 mm/s 写入 CSV。
        rows.append(
            [
                t,
                *[value * M_TO_MM for value in position_values_m],
                *[value * M_TO_MM for value in velocity_values_m_per_s],
            ]
        )

    return rows


def generate_vm_values(vmin: float, vmax: float, vstep: float) -> list[float]:
    """Create an inclusive Vm sweep list from vmin to vmax with step vstep."""
    if vmin <= 0 or vmax <= 0:
        raise ValueError("Vmin and Vmax must both be positive.")
    if vmax < vmin:
        raise ValueError("Vmax must be >= Vmin.")
    if vstep <= 0:
        raise ValueError("Vstep must be positive.")

    values: list[float] = []
    i = 0
    eps = max(vstep * 1e-9, 1e-12)
    while True:
        vm = vmin + i * vstep
        if vm > vmax + eps:
            break
        values.append(round(vm, 10))
        i += 1

    return values


def write_profile_csv(rows: list[list[float]], output_file: str) -> str:
    """Write profile array to CSV file and return actual path."""
    target = output_file
    for i in range(1000):
        try:
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(",".join(HEADERS) + "\n")
                for row in rows:
                    line = ",".join(f"{v:.6f}" for v in row)
                    f.write(line + "\n")
            return target
        except PermissionError:
            base, ext = os.path.splitext(output_file)
            target = f"{base}_new{i + 1}{ext}"

    raise PermissionError(f"Could not write CSV after multiple attempts: {output_file}")


def main() -> None:
    if T <= 0 or T1 <= 0:
        raise ValueError("T and T1 must both be positive.")
    if T < 2.0 * T1 - 1e-12:
        raise ValueError(f"T must be >= 2*T1. Got T={T}, T1={T1}.")
    if V_FILENAME_DECIMALS < 0:
        raise ValueError("V_FILENAME_DECIMALS must be >= 0.")

    # 使用基于脚本位置推导出的绝对目录，因此可以从任意工作目录执行。
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for vm in generate_vm_values(VMIN, VMAX, VSTEP):
        rows = generate_profile_array(vm)
        output_file = os.path.join(
            output_dir,
            f"V_{vm:.{V_FILENAME_DECIMALS}f}.csv",
        )
        actual_output_file = write_profile_csv(rows, output_file)
        # 理论总距离由函数以 m 返回；打印时转换成 mm，与 CSV 中 x 列单位保持一致。
        total_distance_mm = calculate_total_distance(vm, T1, T) * M_TO_MM
        print(f"Saved CSV: {actual_output_file} | distance_mm={total_distance_mm:.6f}")


if __name__ == "__main__":
    main()
