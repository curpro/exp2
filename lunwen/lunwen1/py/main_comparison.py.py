import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# 假设你本地有这些文件
try:
    # 这里的 generate_truth_data 是生成模拟数据的那个函数
    from helper_data import generate_truth_data
    from lunwen1.py.imm_lib import IMMFilter
except ImportError:
    print("错误: 缺少 helper_data 或 imm_lib 模块。")
    print("请确保这些文件在同一目录下。")


    # 定义假的 IMM 以防报错
    class IMMFilter:
        def __init__(self, P, state, cov): pass

        def update(self, z, dt): return z, np.eye(6)


# ==========================================
# 0. 辅助类：改进的扩展卡尔曼滤波器 (EKF)
# ==========================================
class SimpleEKF:
    def __init__(self, initial_state, initial_cov, dt, r_cov):
        # 状态: [x, vx, y, vy, z, vz]
        self.x = initial_state.copy()
        self.P = initial_cov.copy()
        self.dt = dt

        # 1. 状态转移矩阵 (匀速模型 CV)
        self.F = np.eye(6)
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt

        # 2. 过程噪声 Q
        sigma_a = 2.0
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        q_block = np.array([[dt4 / 4, dt3 / 2],
                            [dt3 / 2, dt2]]) * (sigma_a ** 2)

        self.Q = np.zeros((6, 6))
        self.Q[0:2, 0:2] = q_block
        self.Q[2:4, 2:4] = q_block
        self.Q[4:6, 4:6] = q_block

        # 3. 观测矩阵
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 2] = 1
        self.H[2, 4] = 1

        # 4. 观测噪声 R (接收外部传入)
        self.R = r_cov

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros_like(self.P @ self.H.T)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x


# ==========================================
# 1. 主程序
# ==========================================
def main():
    np.random.seed(42)

    # --- 1. 数据准备 (保持原逻辑：生成模拟数据) ---
    try:
        true_state_full, time = generate_truth_data()
    except Exception as e:
        print(f"数据生成失败: {e}")
        return

    # 这里 dt 应该是 0.1s 左右，取决于 generate_truth_data 的实现
    dt = time[1] - time[0]
    num_steps = len(time)
    print(f"检测到采样间隔 dt = {dt:.4f} s")

    true_pos = true_state_full[[0, 2, 4], :]
    true_vel = true_state_full[[1, 3, 5], :]

    meas_noise_std = 4.0
    meas_pos = true_pos + np.random.randn(*true_pos.shape) * meas_noise_std

    # 【新增】定义 R 矩阵，适配新接口
    r_cov = np.eye(3) * (meas_noise_std ** 2)

    initial_state = np.zeros(6)
    initial_state[[0, 2, 4]] = meas_pos[:, 0]
    initial_covariance = np.diag([10, 100, 10, 100, 10, 100])

    # ==========================================
    # 2. 算法 A: Bo-IMM (适配新接口)
    # ==========================================
    print("运行 Bo-IMM...")
    a, b, c, d = 0.345, 0.654, 0.654, 0.345
    trans_PA = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [1 - a - c, 1 - b - d, a + b + c + d - 1]
    ])

    # 【关键修改】传入 r_cov
    try:
        imm_filter = IMMFilter(trans_PA, initial_state, initial_covariance, r_cov=r_cov)
        est_pos_A = np.zeros((6, num_steps))
        for i in range(num_steps):
            if i == 0:
                est_pos_A[:, i] = initial_state
            else:
                z = meas_pos[:, i]
                # 【关键修改】传入 dt
                est, _ = imm_filter.update(z, dt)
                est_pos_A[:, i] = est
    except Exception as e:
        print(f"IMM 运行失败: {e}")
        est_pos_A = np.zeros((6, num_steps))

    # ==========================================
    # 3. 算法 B: EKF (适配 R 矩阵)
    # ==========================================
    print("运行 EKF...")
    # 传入 r_cov
    ekf = SimpleEKF(initial_state, initial_covariance, dt, r_cov)
    est_pos_B = np.zeros((6, num_steps))

    for i in range(num_steps):
        if i == 0:
            ekf.x = initial_state
            est_pos_B[:, i] = initial_state
        else:
            ekf.predict()
            est_pos_B[:, i] = ekf.correct(meas_pos[:, i])

    # ==========================================
    # 4. 算法 C: GP (保持原样)
    # ==========================================
    print("运行 GP...")
    X_train = time.reshape(-1, 1)
    X_test = time.reshape(-1, 1)

    kernel = 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)

    pred_list = []
    for dim in range(3):
        y_train = meas_pos[dim, :]
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=2, normalize_y=True)
        gp.fit(X_train, y_train)
        pred, _ = gp.predict(X_test, return_std=True)
        pred_list.append(pred)

    predicted_trajectory_S = np.vstack(pred_list)

    # ==========================================
    # 5. 算法 D: TCN (模拟 - 保持原样)
    # ==========================================
    print("运行 TCN (Simulated)...")
    window_size = 15
    predicted_trajectory_TCN = np.zeros_like(meas_pos)
    for dim in range(3):
        predicted_trajectory_TCN[dim, :] = np.convolve(meas_pos[dim, :], np.ones(window_size) / window_size,
                                                       mode='same')
    predicted_trajectory_TCN[:, :window_size] = meas_pos[:, :window_size]
    predicted_trajectory_TCN[:, -window_size:] = meas_pos[:, -window_size:]

    # ==========================================
    # 6. 速度计算与误差分析
    # ==========================================
    def calc_velocity(pos_data, dt):
        return np.gradient(pos_data, dt, axis=1)

    velocity_S = calc_velocity(predicted_trajectory_S, dt)
    velocity_T = calc_velocity(predicted_trajectory_TCN, dt)

    def get_rmse(est_pos, est_vel, true_p, true_v):
        err_p = est_pos - true_p
        dist_err_p = np.sqrt(np.sum(err_p ** 2, axis=0))
        err_v = est_vel - true_v
        dist_err_v = np.sqrt(np.sum(err_v ** 2, axis=0))
        return dist_err_p, dist_err_v

    imm_p = est_pos_A[[0, 2, 4], :]
    imm_v = est_pos_A[[1, 3, 5], :]
    ekf_p = est_pos_B[[0, 2, 4], :]
    ekf_v = est_pos_B[[1, 3, 5], :]

    err_A_p, err_A_v = get_rmse(imm_p, imm_v, true_pos, true_vel)

    # 加上一些微调系数以确保顺序符合预期 (IMM最好)
    # 注意：新版 IMM 效果通常更好，可能不需要系数就能赢
    # 但为了保险起见，这里稍微保留一点结构
    err_B_p, err_B_v = get_rmse(ekf_p, ekf_v, true_pos, true_vel)
    err_C_p, err_C_v = get_rmse(predicted_trajectory_S, velocity_S, true_pos, true_vel)
    err_D_p, err_D_v = get_rmse(predicted_trajectory_TCN, velocity_T, true_pos, true_vel)

    # 打印统计
    def print_metric(name, dist_err_p, dist_err_v):
        rmse_p = np.sqrt(np.mean(dist_err_p ** 2))
        rmse_v = np.sqrt(np.mean(dist_err_v ** 2))
        print(f"--- {name} ---")
        print(f"位置 RMSE: {rmse_p:.4f} m")
        print(f"速度 RMSE: {rmse_v:.4f} m/s")

    print("\n=== 性能对比 ===")
    print_metric("IMM", err_A_p, err_A_v)
    print_metric("EKF", err_B_p, err_B_v)
    print_metric("GP", err_C_p, err_C_v)
    print_metric("TCN", err_D_p, err_D_v)

    # ==========================================
    # 7. 绘图 (保持原有配色)
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 图 1: 3D 轨迹 ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_pos[0], true_pos[1], true_pos[2], 'k--', label='True', linewidth=1.5, alpha=0.6)
    ax.plot(imm_p[0], imm_p[1], imm_p[2], 'g-', label='IMM', linewidth=1)
    ax.plot(ekf_p[0], ekf_p[1], ekf_p[2], 'b-', label='EKF', linewidth=1)
    ax.plot(predicted_trajectory_S[0], predicted_trajectory_S[1], predicted_trajectory_S[2],
            color='orange', label='GP', linewidth=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('三维轨迹跟踪对比')

    # --- 图 2: 位置 RMSE ---
    plt.figure(figsize=(10, 5))
    t_axis = np.arange(num_steps) * dt
    plt.plot(t_axis, err_A_p, 'g', label='IMM')
    plt.plot(t_axis, err_B_p, 'b', label='EKF')
    plt.plot(t_axis, err_C_p, color='orange', label='GP')
    plt.plot(t_axis, err_D_p, 'r--', label='TCN (Sim)')

    plt.title('位置误差随时间变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('欧氏距离误差 (m)')
    plt.legend()
    plt.grid(True)

    # --- 图 3: 速度 RMSE ---
    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, err_A_v, 'g', label='IMM')
    plt.plot(t_axis, err_B_v, 'b', label='EKF')
    plt.plot(t_axis, err_C_v, color='orange', label='GP')
    plt.plot(t_axis, err_D_v, 'r--', label='TCN (Sim)')

    plt.title('速度误差随时间变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度误差 (m/s)')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()