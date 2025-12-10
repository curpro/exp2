import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import copy
import warnings

# 忽略运行时的某些警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 导入你的本地库
# ==========================================
try:
    from lunwen1.py.imm_lib import IMMFilter
except ImportError:
    # 备用导入路径，根据你的实际情况调整
    import sys

    sys.path.append(r"/lunwen1/py")
    from imm_lib import IMMFilter

# ==========================================
# 2. 全局配置参数
# ==========================================
# 路径请自行确认
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_multimode_test_data.csv'

# 仿真参数
DT = 1 / 30
MEAS_NOISE_STD = 4.0

# --- 第五章核心：自适应窗口参数 ---
WINDOW_SIZE = 20
UPDATE_INTERVAL = 5
BO_INIT_POINTS = 2
BO_ITERATIONS = 5
BOUNDS = {'diag_val': (0.5, 0.99)}

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 3. 辅助函数
# ==========================================
def load_csv_data(filepath):
    """
    修改点 1: 读取 9 维数据 (x, vx, ax, y, vy, ay, z, vz, az)
    """
    try:
        df = pd.read_csv(filepath)
        # 确保列名与 CSV 文件一致
        cols = ['x', 'vx', 'ax', 'y', 'vy', 'ay', 'z', 'vz', 'az']
        state_matrix = df[cols].to_numpy().T
        return state_matrix
    except Exception as e:
        print(f"读取文件失败: {e}。生成 9 维模拟数据...")
        t = np.linspace(0, 20, 600)
        zeros = np.zeros_like(t)
        # 生成一个模拟的 9 维状态: x, vx, ax, y, vy, ay, z, vz, az
        return np.vstack([
            1000 * t, np.full_like(t, 1000), zeros,  # x, vx, ax
            2000 * np.sin(0.5 * t), 1000 * np.cos(0.5 * t), -500 * np.sin(0.5 * t),  # y, vy, ay
            500 * t, np.full_like(t, 500), zeros  # z, vz, az
        ])


def get_trans_matrix(diag_val):
    M = 3
    mat = np.eye(M) * diag_val
    off_diag = (1.0 - diag_val) / (M - 1)
    mat[mat == 0] = off_diag
    return mat


def evaluate_window_performance(diag_val, start_idx, end_idx, measurements, start_snapshot, r_cov):
    """目标函数：基于'位置'观测残差的反馈"""
    x_start, P_start, prob_start = start_snapshot
    trans_mat = get_trans_matrix(diag_val)

    # 临时 IMM 回溯
    # 注意：这里假设 IMMFilter 能自动识别 x_start 是 9 维向量
    temp_imm = IMMFilter(trans_mat, x_start, P_start, r_cov=r_cov)

    # 显式赋值，确保状态一致
    temp_imm.x = copy.deepcopy(x_start)
    temp_imm.P = copy.deepcopy(P_start)
    temp_imm.model_probs = copy.deepcopy(prob_start)

    total_err = 0.0
    for k in range(start_idx, end_idx + 1):
        z_k = measurements[:, k]
        est_x, _ = temp_imm.update(z_k, DT)

        # 修改点 2: 9 维状态下，位置索引通常是 [0, 3, 6] (x, y, z)
        z_pred = est_x[[0, 3, 6]]

        # 这里反馈的是【位置误差】
        err = np.sum((z_k - z_pred) ** 2)
        total_err += err

    rmse = np.sqrt(total_err / (end_idx - start_idx + 1))
    return -rmse


# ==========================================
# 4. 主程序
# ==========================================
def main_adaptive_bo():
    print(f"正在加载数据...")
    full_state = load_csv_data(CSV_FILE_PATH)
    # full_state shape 应该是 (9, N)
    num_steps = full_state.shape[1]

    # 1. 生成观测
    # 修改点 3: 从 9 维状态中提取位置 (x=0, y=3, z=6)
    pos_indices = [0, 3, 6]
    vel_indices = [1, 4, 7]  # vx, vy, vz

    np.random.seed(42)
    true_pos = full_state[pos_indices, :]
    measurements = true_pos + np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

    # 2. 初始化
    init_diag = 0.8
    trans_mat = get_trans_matrix(init_diag)

    # 修改点 4: 初始化状态改为 9 维
    init_x = np.zeros(9)
    init_x[pos_indices] = measurements[:, 0]

    # 修改点 5: 初始化协方差矩阵改为 9x9
    init_P = np.eye(9) * 100

    # 实例化
    imm_adaptive = IMMFilter(trans_mat, init_x, init_P, r_cov=r_cov)

    # 3. 记录器
    # 修改点 6: 记录的状态维度也需要改为 9
    est_adaptive = np.zeros((9, num_steps))
    diag_history = np.zeros(num_steps)
    state_snapshots = {}

    print("开始运行在线自适应优化...")
    current_diag = init_diag

    for k in range(num_steps):
        z_curr = measurements[:, k]

        # 快照
        snapshot = (copy.deepcopy(imm_adaptive.x),
                    copy.deepcopy(imm_adaptive.P),
                    copy.deepcopy(imm_adaptive.model_probs))
        state_snapshots[k] = snapshot

        # 在线优化触发
        if k > WINDOW_SIZE and k % UPDATE_INTERVAL == 0:
            def objective_function(diag_val):
                return evaluate_window_performance(
                    diag_val, k - WINDOW_SIZE, k - 1, measurements,
                    state_snapshots[k - WINDOW_SIZE], r_cov
                )

            optimizer = BayesianOptimization(
                f=objective_function, pbounds=BOUNDS, verbose=0, random_state=1,
                allow_duplicate_points=True
            )

            # 热启动
            try:
                optimizer.probe(params={"diag_val": current_diag}, lazy=True)
            except:
                pass

            optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_ITERATIONS)
            current_diag = optimizer.max['params']['diag_val']
            # 更新转移矩阵
            if hasattr(imm_adaptive, 'trans_prob'):
                imm_adaptive.trans_prob = get_trans_matrix(current_diag)
            else:
                # 如果你的库里属性名不同，请修改这里，比如 imm_adaptive.M_tp = ...
                imm_adaptive.M = get_trans_matrix(current_diag)

        diag_history[k] = current_diag
        est, _ = imm_adaptive.update(z_curr, DT)
        est_adaptive[:, k] = est

        if k - WINDOW_SIZE - 20 in state_snapshots:
            del state_snapshots[k - WINDOW_SIZE - 20]

    print("滤波完成。正在计算指标...")

    # ==========================================
    # 5. 核心：误差计算与打印
    # ==========================================
    # 修改点 7: 使用新的索引提取真值和估计值
    true_pos_all = full_state[pos_indices, :]
    true_vel_all = full_state[vel_indices, :]

    est_pos_all = est_adaptive[pos_indices, :]
    est_vel_all = est_adaptive[vel_indices, :]

    # --- 位置误差 ---
    pos_err_vec = est_pos_all - true_pos_all
    pos_err_dist = np.sqrt(np.sum(pos_err_vec ** 2, axis=0))

    # --- 速度误差 ---
    vel_err_vec = est_vel_all - true_vel_all
    vel_err_dist = np.sqrt(np.sum(vel_err_vec ** 2, axis=0))

    # 指标统计
    metrics = {
        "位置 RMSE (m)": np.sqrt(np.mean(pos_err_dist ** 2)),
        "位置 方差 (m^2)": np.var(pos_err_dist),
        "位置 最大误差 (m)": np.max(pos_err_dist),
        "速度 RMSE (m/s)": np.sqrt(np.mean(vel_err_dist ** 2)),
        "速度 方差 (m^2/s^2)": np.var(vel_err_dist),
        "速度 最大误差 (m/s)": np.max(vel_err_dist)
    }

    # 漂亮的打印
    print("\n" + "=" * 40)
    print("   第五章 自适应算法 性能评估报告")
    print("=" * 40)
    for key, val in metrics.items():
        print(f"{key:<15} : {val:.4f}")
    print("=" * 40 + "\n")

    # ==========================================
    # 6. 增强版绘图
    # ==========================================
    t_axis = np.arange(num_steps) * DT

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 图1：位置误差
    ax1.plot(t_axis, pos_err_dist, 'r-', linewidth=1, label='Pos Error')
    ax1.set_ylabel('位置误差 (m)', fontsize=12)
    ax1.set_title('F-16 自适应跟踪性能分析 (9-State)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 图2：速度误差
    ax2.plot(t_axis, vel_err_dist, 'orange', linewidth=1, label='Vel Error')
    ax2.set_ylabel('速度误差 (m/s)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 图3：参数演化
    ax3.plot(t_axis, diag_history, 'b-', linewidth=2, label='TPM 主对角线概率(p_ii)')
    ax3.set_ylabel('自适应参数', fontsize=12)
    ax3.set_xlabel('时间 (s)', fontsize=12)
    ax3.set_ylim(0.4, 1.0)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    save_path = '../result/Chapter5_Full_Analysis.png'
    # plt.savefig(save_path, dpi=300)
    print(f"全能分析图绘制完成")
    plt.show()


if __name__ == "__main__":
    main_adaptive_bo()