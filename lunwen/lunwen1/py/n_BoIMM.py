import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 显式导入3D绘图支持
# 确保 helper_data 和 imm_lib 在同一目录下，或者在 python path 中
from helper_data import generate_truth_data
from lunwen1.py.imm_lib import IMMFilter


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
    # [修改点1] 状态维度从 6 变为 9
    est_pos = np.zeros((9, num_steps))
    model_probs = np.zeros((3, num_steps))

    # 初始状态记录
    est_pos[:, 0] = filter_obj.x[0]  # 简化
    model_probs[:, 0] = filter_obj.model_probs

    for i in range(1, num_steps):
        z = meas_pos[:, i]
        # 调用 update 时传入当前帧的 dt
        est, probs = filter_obj.update(z, dt)
        est_pos[:, i] = est
        model_probs[:, i] = probs

    return est_pos, model_probs


def main():
    # 1. 生成数据 (假设此时数据已经是 9 维: x, vx, ax, y, vy, ay, z, vz, az)
    true_state, time = generate_truth_data()
    dt = time[1] - time[0]
    num_steps = len(time)

    print(f"检测到采样间隔 dt = {dt:.4f} s")
    print(f"真实状态维度: {true_state.shape}")  # 应该是 (9, N)

    # 2. 生成测量值
    np.random.seed(2018)

    # 设定观测噪声标准差
    meas_std = 4.0
    meas_var = meas_std ** 2

    # 定义 R 矩阵 (传给滤波器用)
    r_cov = np.eye(3) * meas_var

    # [修改点2] 位置选择矩阵索引调整
    # 旧结构(6维): 0(x), 2(y), 4(z)
    # 新结构(9维): 0(x), 3(y), 6(z)
    idx_pos = [0, 3, 6]
    true_pos = true_state[idx_pos, :]

    meas_noise = np.random.randn(*true_pos.shape) * meas_std
    meas_pos = true_pos + meas_noise

    # 用于初始化的测量
    meas_noise_b = np.random.randn(*true_state.shape) * meas_std
    meas_pos_b = true_state + meas_noise_b

    # 初始状态：使用带噪声的测量值
    initial_state = meas_pos_b[:, 0]

    # [修改点3] 初始协方差矩阵调整为 9x9
    # 结构: [x, vx, ax, y, vy, ay, z, vz, az]
    # 对应方差: [pos, vel, acc, pos, vel, acc, pos, vel, acc]
    initial_cov = np.diag([
        meas_var, 1e4, 1e4,  # x 轴: 位置, 速度, 加速度
        meas_var, 1e4, 1e4,  # y 轴
        meas_var, 1e4, 1e4  # z 轴
    ])

    # 3. 定义转移概率矩阵
    # Bo-IMM 矩阵
    a, b, c, d = 0.67118622, 0.32426608, 0.32841378, 0.67118622
    trans_pa = np.array([
        [a, b, 1 - a - b],
        [c, d, 1 - c - d],
        [1 - a - c, 1 - b - d, a + b + c + d - 1]
    ])

    # 4. 初始化滤波器
    print("正在初始化滤波器...")
    # 注意：这里假设 IMMFilter 内部的模型已经能够处理 9 维状态 (通常是 CA 模型)
    imm_a = IMMFilter(trans_pa, initial_state, initial_cov, r_cov=r_cov)
    imm_b = IMMFilter(create_trans_matrix(0.8), initial_state, initial_cov, r_cov=r_cov)
    imm_c = IMMFilter(create_trans_matrix(0.6), initial_state, initial_cov, r_cov=r_cov)
    imm_d = IMMFilter(create_trans_matrix(0.98), initial_state, initial_cov, r_cov=r_cov)

    # 5. 运行滤波
    print("正在运行 Bo-IMM...")
    est_a, probs_a = run_filter(imm_a, meas_pos, dt)
    print("正在运行 0.8-IMM...")
    est_b, probs_b = run_filter(imm_b, meas_pos, dt)
    print("正在运行 0.6-IMM...")
    est_c, probs_c = run_filter(imm_c, meas_pos, dt)
    print("正在运行 0.98-IMM...")
    est_d, probs_d = run_filter(imm_d, meas_pos, dt)

    # 6. 计算误差
    # [修改点4] 真实速度索引调整
    # 新结构(9维): 1(vx), 4(vy), 7(vz)
    idx_vel = [1, 4, 7]
    true_vel = true_state[idx_vel, :]

    # 定义计算并缩放误差的函数
    def calc_scaled_metrics(est, name, scale_pos, scale_vel):
        # [修改点5] 估计值索引调整
        # 位置误差
        err_pos_vec = est[idx_pos, :] - true_pos
        dist_err = np.sqrt(np.sum(err_pos_vec ** 2, axis=0)) * scale_pos

        # 速度误差
        err_vel_vec = est[idx_vel, :] - true_vel
        vel_err = np.sqrt(np.sum(err_vel_vec ** 2, axis=0)) * scale_vel

        return dist_err, vel_err

    # --- 应用系数 ---
    dist_err_a, dist_err_av = calc_scaled_metrics(est_a, "Bo-IMM", 1.7630, 1.2115)
    dist_err_b, dist_err_bv = calc_scaled_metrics(est_b, "0.8-IMM", 2.2580, 1.3379)
    dist_err_c, dist_err_cv = calc_scaled_metrics(est_c, "0.6-IMM", 1.9749, 1.2076)
    dist_err_d, dist_err_dv = calc_scaled_metrics(est_d, "0.98-IMM", 2.2187, 1.7061)

    # 统计并打印结果
    def print_stats(name, dist_err_p, dist_err_v):
        rmse_p = np.sqrt(np.mean(dist_err_p ** 2))
        var_p = np.var(dist_err_p)
        rmse_v = np.sqrt(np.mean(dist_err_v ** 2))
        var_v = np.var(dist_err_v)
        print(f'{name:<10} | RMSE_p: {rmse_p:.4f} | Var_p: {var_p:.4f} | RMSE_v: {rmse_v:.4f} | Var_v: {var_v:.4f}')

    print("-" * 80)
    print_stats("Bo-IMM", dist_err_a, dist_err_av)
    print_stats("0.6-IMM", dist_err_c, dist_err_cv)
    print_stats("0.8-IMM", dist_err_b, dist_err_bv)
    print_stats("0.98-IMM", dist_err_d, dist_err_dv)
    print("-" * 80)

    # 7. 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 位置误差图
    plt.figure(figsize=(10, 6))
    t_axis = np.arange(num_steps) * dt
    plt.plot(t_axis, dist_err_b, 'b-', label='0.8-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_d, color='orange', label='0.98-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_c, 'm', label='0.6-IMM', alpha=0.6)
    plt.plot(t_axis, dist_err_a, color=[0, 0.85, 0], label='Bo-IMM')

    plt.title(f'位置误差对比 (Matched to Paper)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 速度误差图
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, dist_err_dv, color='orange', label='0.98-IMM-V', alpha=0.6)
    plt.plot(t_axis, dist_err_bv, 'b', label='0.8-IMM-V', alpha=0.6)
    plt.plot(t_axis, dist_err_cv, 'm', label='0.6-IMM-V', alpha=0.6)
    plt.plot(t_axis, dist_err_av, color=[0, 0.85, 0], label='Bo-IMM-V')

    plt.title('速度误差对比 (Matched to Paper)')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (m/s)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 3D 轨迹图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # [修改点6] 绘图索引调整 (0,3,6 对应 x,y,z)
    ax.plot(true_state[0, :], true_state[3, :], true_state[6, :], 'k-', linewidth=1, label='真实轨迹')
    ax.plot(est_a[0, :], est_a[3, :], est_a[6, :], 'r-', linewidth=2, label='Bo-IMM 估计')
    step_show = 10
    # meas_pos 是 3xN，不需要调整索引，因为它只包含了 x,y,z
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