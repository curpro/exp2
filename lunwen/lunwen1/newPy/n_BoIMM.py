import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 假设 imm_lib.py 在同一目录下
from lunwen1.py.imm_lib import IMMFilter

# ==========================================
# 1. 配置参数
# ==========================================
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver_a.csv'
DT = 1 / 30  # 30Hz 采样率
MEAS_NOISE_STD = 15 # 观测噪声标准差 (米)

def load_csv_data(filepath):
    """读取 CSV 并转换为 (9, N) 的状态矩阵"""
    try:
        df = pd.read_csv(filepath)
        # F-16数据包含9列: x,vx,ax, y,vy,ay, z,vz,az
        # 转置为 (9, N)
        state_matrix = df.values.T
        return state_matrix
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def create_trans_matrix(diag_val):
    """创建对角线占优的转移矩阵"""
    p = diag_val
    off = (1.0 - p) / 2.0
    return np.array([
        [p, off, off],
        [off, p, off],
        [off, off, p]
    ])

def run_filter(filter_obj, meas_pos, dt):
    num_steps = meas_pos.shape[1]
    # IMM 状态是 9 维 (x,vx,ax, y,vy,ay, z,vz,az)
    est_state = np.zeros((9, num_steps))
    model_probs = np.zeros((3, num_steps))

    # 初始状态记录
    est_state[:, 0] = filter_obj.x[0]
    model_probs[:, 0] = filter_obj.model_probs

    for i in range(1, num_steps):
        z = meas_pos[:, i]
        est, probs = filter_obj.update(z, dt)
        est_state[:, i] = est
        model_probs[:, i] = probs

    return est_state, model_probs

def main():
    # 1. 加载数据
    print(f"正在加载数据: {CSV_FILE_PATH} ...")
    true_state = load_csv_data(CSV_FILE_PATH)
    if true_state is None:
        return

    num_steps = true_state.shape[1]
    print(f"数据加载成功，共 {num_steps} 个采样点")

    # 2. 生成模拟观测值
    np.random.seed(42)  # 固定随机种子

    # --- 关键修正：适配 F-16 数据的 9维 索引 ---
    # 0:x, 1:vx, 2:ax, 3:y, 4:vy, 5:ay, 6:z, 7:vz, 8:az
    idx_pos = [0, 3, 6]
    idx_vel = [1, 4, 7]
    idx_acc = [2, 5, 8]

    true_pos = true_state[idx_pos, :]

    # 添加高斯白噪声
    meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
    meas_pos = true_pos + meas_noise

    # 生成 R 矩阵 (观测噪声协方差)
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

    # ==========================================
    # 【初始化】真值 + 微小扰动
    # ==========================================
    gt_init = true_state[:, 0]

    init_pos_err = 10.0
    init_vel_err = 5.0

    # 初始状态增加扰动
    init_noise = np.random.randn(9)
    init_noise[idx_pos] *= init_pos_err
    init_noise[idx_vel] *= init_vel_err
    init_noise[idx_acc] *= 1.0  # 加速度初始误差

    initial_state = gt_init + init_noise

    # 初始协方差 (9维对角阵)
    cov_diag = np.zeros(9)
    cov_diag[idx_pos] = init_pos_err ** 2
    cov_diag[idx_vel] = init_vel_err ** 2
    cov_diag[idx_acc] = 10.0
    initial_cov = np.diag(cov_diag)
    # ==========================================

    # 3. 定义转移概率矩阵 (Bo-IMM)
    a, b, c, d = 0.98899994, 0.01, 0.01, 0.98900043
    trans_pa = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [1 - a - c, 1 - b - d, a + b + c + d - 1]
    ])

    # 4. 初始化滤波器
    print("正在初始化滤波器...")
    imm_bo = IMMFilter(trans_pa, initial_state, initial_cov, r_cov=r_cov)
    imm_06 = IMMFilter(create_trans_matrix(0.6), initial_state, initial_cov, r_cov=r_cov)
    imm_08 = IMMFilter(create_trans_matrix(0.8), initial_state, initial_cov, r_cov=r_cov)
    imm_098 = IMMFilter(create_trans_matrix(0.98), initial_state, initial_cov, r_cov=r_cov)

    # 5. 运行滤波
    print("正在运行 Bo-IMM ...")
    est_bo, probs_bo = run_filter(imm_bo, meas_pos, DT)

    print("正在运行 0.6-IMM ...")
    est_06, probs_06 = run_filter(imm_06, meas_pos, DT)

    print("正在运行 0.8-IMM ...")
    est_08, probs_08 = run_filter(imm_08, meas_pos, DT)

    print("正在运行 0.98-IMM ...")
    est_098, probs_098 = run_filter(imm_098, meas_pos, DT)

    # 6. 计算真实误差
    true_vel = true_state[idx_vel, :]
    true_acc = true_state[idx_acc, :]

    def calc_true_metrics(est):
        # 位置误差
        err_pos = est[idx_pos, :] - true_pos
        dist_err = np.sqrt(np.sum(err_pos ** 2, axis=0))
        # 速度误差
        err_vel = est[idx_vel, :] - true_vel
        vel_err = np.sqrt(np.sum(err_vel ** 2, axis=0))
        # 加速度误差
        err_acc = est[idx_acc, :] - true_acc
        acc_err = np.sqrt(np.sum(err_acc ** 2, axis=0))
        return dist_err, vel_err, acc_err

    dist_err_bo, vel_err_bo, acc_err_bo = calc_true_metrics(est_bo)
    dist_err_06, vel_err_06, acc_err_06 = calc_true_metrics(est_06)
    dist_err_08, vel_err_08, acc_err_08 = calc_true_metrics(est_08)
    dist_err_098, vel_err_098, acc_err_098 = calc_true_metrics(est_098)

    # 打印统计结果
    def print_stats(name, dist_err_p, dist_err_v, dist_err_a):
        rmse_p = np.sqrt(np.mean(dist_err_p ** 2))
        rmse_v = np.sqrt(np.mean(dist_err_v ** 2))
        rmse_a = np.sqrt(np.mean(dist_err_a ** 2))
        print(f'{name:<10} | RMSE_p: {rmse_p:.4f} | RMSE_v: {rmse_v:.4f} | RMSE_a: {rmse_a:.4f}')

    print("-" * 100)
    print("真实误差统计 (Position, Velocity, Acceleration):")
    print_stats("Bo-IMM", dist_err_bo, vel_err_bo, acc_err_bo)
    print_stats("0.6-IMM", dist_err_06, vel_err_06, acc_err_06)
    print_stats("0.8-IMM", dist_err_08, vel_err_08, acc_err_08)
    print_stats("0.98-IMM", dist_err_098, vel_err_098, acc_err_098)
    print("-" * 100)

    # ==========================================
    # 7. 绘图 (严格还原原始颜色和样式)
    # ==========================================
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    t_axis = np.arange(num_steps) * DT

    # --- 位置误差图 ---
    plt.figure(figsize=(10, 6))
    # 颜色还原：0.8-IMM(b), 0.6-IMM(m), 0.98-IMM(orange), Bo-IMM([0, 0.85, 0])
    plt.plot(t_axis, dist_err_08, 'b', label='0.8-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_06, 'm', label='0.6-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_098, color='orange', label='0.98-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_bo, color=[0, 0.85, 0], label='Bo-IMM', linewidth=2) # 原始绿色粗线

    plt.title(f'位置误差对比 (Position RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # --- 速度误差图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, vel_err_08, 'b', label='0.8-IMM', alpha=0.6)
    plt.plot(t_axis, vel_err_06, 'm', label='0.6-IMM', alpha=0.6)
    plt.plot(t_axis, vel_err_098, color='orange', label='0.98-IMM', alpha=0.6)
    plt.plot(t_axis, vel_err_bo, color=[0, 0.85, 0], label='Bo-IMM', linewidth=2)

    plt.title('速度误差对比 (Velocity RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # --- 加速度误差图 (原代码有此图，保留) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, acc_err_08, 'b', label='0.8-IMM', alpha=0.6)
    plt.plot(t_axis, acc_err_06, 'm', label='0.6-IMM', alpha=0.6)
    plt.plot(t_axis, acc_err_098, color='orange', label='0.98-IMM', alpha=0.6)
    plt.plot(t_axis, acc_err_bo, color=[0, 0.85, 0], label='Bo-IMM', linewidth=2)

    plt.title('加速度误差对比 (Acceleration RMSE)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s^2)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # ==========================================
    # 模型概率变化对比图 (2x2 布局)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IMM Model Probability Evolution Comparison', fontsize=16)

    plot_configs = [
        (probs_bo, "Bo-IMM", axes[0, 0]),
        (probs_06, "0.6-IMM", axes[0, 1]),
        (probs_08, "0.8-IMM", axes[1, 0]),
        (probs_098, "0.98-IMM", axes[1, 1])
    ]

    # 还原原始标签和颜色
    model_names = ['Model 1 (CV)', 'Model 2 (CA)', 'Model 3 (CT)']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for probs, title, ax in plot_configs:
        time_steps = np.arange(probs.shape[1]) * DT
        for m in range(3):
            ax.plot(time_steps, probs[m, :], label=model_names[m], color=colors[m], linewidth=1.5, alpha=0.8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 3D 轨迹图 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 画真实轨迹
    ax.plot(true_state[idx_pos[0], :], true_state[idx_pos[1], :], true_state[idx_pos[2], :], 'k-', linewidth=1.5,
            label='真实轨迹')
    # 画 Bo-IMM 滤波轨迹
    ax.plot(est_bo[idx_pos[0], :], est_bo[idx_pos[1], :], est_bo[idx_pos[2], :], 'r-', linewidth=2, label='Bo-IMM 估计')

    # 画观测值 (稀疏显示)
    step_show = 10
    ax.scatter(meas_pos[0, ::step_show], meas_pos[1, ::step_show], meas_pos[2, ::step_show],
               s=1, c='gray', alpha=0.3, label='观测值')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('三维轨迹跟踪效果')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()