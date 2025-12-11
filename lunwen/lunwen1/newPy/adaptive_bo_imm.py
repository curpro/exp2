import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import copy
import warnings

# 忽略运行时的警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 导入你的本地库
# ==========================================
from lunwen1.py.imm_lib import IMMFilter

# ==========================================
# 2. 全局配置参数
# ==========================================
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver_a.csv'
DT = 1 / 30
MEAS_NOISE_STD = 15.0  # 观测噪声标准差

# --- 第五章核心：全矩阵自适应参数 ---
WINDOW_SIZE = 30
UPDATE_INTERVAL = 2 # 更新频率
BO_INIT_POINTS = 10  # 探索点数
BO_ITERATIONS = 20  # 迭代次数
NIS_THRESHOLD = 3.0  # 机动检测门限 (归一化残差 > 3倍标准差即认为机动)

# --- 6参数优化边界 ---
# 这里的边界设置允许算法在很大范围内探索非对称性
BOUNDS = {
    'p00': (0.5, 0.98), 'p01': (0.01, 0.4),
    'p10': (0.01, 0.4), 'p11': (0.6, 0.95),
    'p20': (0.01, 0.4), 'p21': (0.01, 0.4)
}

# 绘图配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 9D 索引定义
IDX_POS = [0, 3, 6]  # x, y, z
IDX_VEL = [1, 4, 7]  # vx, vy, vz


# ==========================================
# 3. 辅助函数
# ==========================================
def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # 提取 9D 状态: x, vx, ax, y, vy, ay, z, vz, az
        state_matrix = df[['x', 'vx', 'ax', 'y', 'vy', 'ay', 'z', 'vz', 'az']].to_numpy().T
        return state_matrix
    except Exception as e:
        print(f"读取失败: {e}")
        return None


def get_trans_matrix_6d(p00, p01, p10, p11, p20, p21):
    """
    构建 3x3 转移矩阵。
    加入了自动边界约束：如果输入的概率之和超过 1，会自动按比例缩放以保证有效性。
    """

    # 辅助函数：处理单行概率
    def safe_row(prob_a, prob_b):
        # prob_a 和 prob_b 是你显式优化的两个参数
        current_sum = prob_a + prob_b

        # 预留 0.001 给第三个概率，防止为 0 导致数值计算错误
        max_allowed = 0.999

        if current_sum > max_allowed:
            # 如果超标，按比例压缩
            scale_factor = max_allowed / current_sum
            prob_a *= scale_factor
            prob_b *= scale_factor

        # 计算剩余的第三个概率
        remainder = 1.0 - prob_a - prob_b
        return prob_a, prob_b, remainder

    # 1. 第一行 (Model 0: High Noise)
    # 参数是 p00, p01 -> 剩余 p02
    p00, p01, p02 = safe_row(p00, p01)

    # 2. 第二行 (Model 1: Low Noise)
    # 参数是 p10, p11 -> 剩余 p12
    # 注意：为了逻辑统一，这里我们传入这一行的两个已知参数 p10, p11
    p10, p11, p12 = safe_row(p10, p11)

    # 3. 第三行 (Model 2: Turn/Med Noise)
    # 参数是 p20, p21 -> 剩余 p22
    p20, p21, p22 = safe_row(p20, p21)

    # 构建矩阵
    mat = np.array([
        [p00, p01, p02],
        [p10, p11, p12],
        [p20, p21, p22]
    ])

    return mat


def evaluate_window_6d(p00, p01, p10, p11, p20, p21,
                       start_idx, end_idx, measurements, start_snapshot, r_cov):
    """
    目标函数：在滑动窗口内回溯滤波，计算位置 RMSE
    """
    # 1. 构建矩阵
    trans_mat = get_trans_matrix_6d(p00, p01, p10, p11, p20, p21)

    # 罚函数：如果参数无效，返回极小值引导 BO 避开此区域
    if trans_mat is None:
        return -1e6

    # 2. 回溯测试环境搭建
    x_start, P_start, prob_start = start_snapshot
    temp_imm = IMMFilter(trans_mat, x_start[0], P_start[0], r_cov=r_cov)
    temp_imm.x = copy.deepcopy(x_start)
    temp_imm.P = copy.deepcopy(P_start)
    temp_imm.model_probs = copy.deepcopy(prob_start)

    total_err = 0.0
    # 3. 窗口内前向推演
    for k in range(start_idx, end_idx + 1):
        z_k = measurements[:, k]
        est_x, _ = temp_imm.update(z_k, DT)
        z_pred = est_x[IDX_POS]
        err = np.sum((z_k - z_pred) ** 2)
        total_err += err

    rmse = np.sqrt(total_err / (end_idx - start_idx + 1))

    # 返回负 RMSE 以最大化
    return -rmse


# ==========================================
# 4. 主程序
# ==========================================
def main_ultimate():
    print(f"正在加载数据...")
    full_state = load_csv_data(CSV_FILE_PATH)
    if full_state is None: return
    num_steps = full_state.shape[1]

    # 1. 生成带噪声的观测数据
    np.random.seed(42)
    true_pos = full_state[IDX_POS, :]
    measurements = true_pos + np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
    r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

    # 2. 初始化：定义“稳健模式”参数 (以 Model 1 Low 为核心)
    # Model 0: High Noise (机动)
    # Model 1: Low Noise (平飞)
    # Model 2: Med Noise (转弯)
    robust_params = {
        'p00': 0.8, 'p01': 0.1,  # 机动 -> 改出
        'p10': 0.05, 'p11': 0.9,  # 平飞 -> 保持 (稳)
        'p20': 0.1, 'p21': 0.1  # 转弯 -> 改出
    }

    current_params = robust_params.copy()
    init_mat = get_trans_matrix_6d(**current_params)

    # 初始状态
    init_x = full_state[:, 0].copy()
    init_P = np.diag([100] * 9)  # 初始协方差

    imm_adaptive = IMMFilter(init_mat, init_x, init_P, r_cov=r_cov)

    # 3. 记录器
    est_adaptive = np.zeros((9, num_steps))
    p00_history = np.zeros(num_steps)
    p01_history = np.zeros(num_steps)
    p10_history = np.zeros(num_steps)
    p11_history = np.zeros(num_steps)  # 原有的 p11 保持不变
    p20_history = np.zeros(num_steps)
    p21_history = np.zeros(num_steps)
    state_snapshots = {}

    print("启动全矩阵自适应跟踪 (6参数优化 + NIS门限)...")

    for k in range(num_steps):
        z_curr = measurements[:, k]

        # A. 保存状态快照 (用于回溯评估)
        state_snapshots[k] = (copy.deepcopy(imm_adaptive.x),
                              copy.deepcopy(imm_adaptive.P),
                              copy.deepcopy(imm_adaptive.model_probs))

        # B. 计算 NIS (机动检测)
        # [关键修正] 预测位置必须包含速度项，否则会有滞后误差
        if k > 0:
            pos_prev = est_adaptive[IDX_POS, k - 1]
            vel_prev = est_adaptive[IDX_VEL, k - 1]
            # 线性预测：位置 + 速度 * 时间
            z_pred_approx = pos_prev + vel_prev * DT

            residual = z_curr - z_pred_approx
            nis_score = np.linalg.norm(residual) / MEAS_NOISE_STD
        else:
            nis_score = 0.0

        # C. 自适应触发逻辑
        if k > WINDOW_SIZE and k % UPDATE_INTERVAL == 0:

            # 只有当 (检测到机动) 时，才启动昂贵的6参数优化
            if nis_score > NIS_THRESHOLD:

                # 定义优化目标 wrapper
                def objective_wrapper(p00, p01, p10, p11, p20, p21):
                    return evaluate_window_6d(
                        p00, p01, p10, p11, p20, p21,
                        k - WINDOW_SIZE, k - 1, measurements,
                        state_snapshots[k - WINDOW_SIZE], r_cov
                    )

                # 实例化优化器
                optimizer = BayesianOptimization(
                    f=objective_wrapper, pbounds=BOUNDS, verbose=0,
                    random_state=k, allow_duplicate_points=True
                )

                # --- 多点热启动 (Multi-point Warm Start) ---
                try:
                    # 1. 惯性：尝试沿用当前参数
                    optimizer.probe(params=current_params, lazy=True)
                    # 2. 复位：尝试标准稳健参数
                    optimizer.probe(params=robust_params, lazy=True)
                    # 3. 激进：尝试降低平飞保持概率，增加灵活性
                    aggressive = robust_params.copy()
                    aggressive['p11'] = 0.501
                    optimizer.probe(params=aggressive, lazy=True)
                except:
                    pass

                optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_ITERATIONS)

                # 获取最优解
                best_p = optimizer.max['params']

                # 平滑更新 (EMA, alpha=0.3)
                alpha = 0.3
                for key in current_params:
                    current_params[key] = (1 - alpha) * current_params[key] + alpha * best_p[key]

            else:
                # 平飞状态：缓慢回归稳健参数 (Reset机制)
                alpha_reset = 0.1
                for key in current_params:
                    current_params[key] = (1 - alpha_reset) * current_params[key] + alpha_reset * robust_params[key]

            # 应用新矩阵
            new_mat = get_trans_matrix_6d(**current_params)
            if new_mat is not None:
                imm_adaptive.trans_prob = new_mat

        # D. 记录参数并执行滤波
        p00_history[k] = current_params['p00']
        p01_history[k] = current_params['p01']
        p10_history[k] = current_params['p10']
        p11_history[k] = current_params['p11']
        p20_history[k] = current_params['p20']
        p21_history[k] = current_params['p21']
        est, _ = imm_adaptive.update(z_curr, DT)
        est_adaptive[:, k] = est

        # 清理旧快照
        if k - WINDOW_SIZE - 20 in state_snapshots:
            del state_snapshots[k - WINDOW_SIZE - 20]

    print("计算完毕。正在生成报表...")

    # ==========================================
    # 5. 结果评估与绘图
    # ==========================================
    est_pos = est_adaptive[IDX_POS, :]
    est_vel = est_adaptive[IDX_VEL, :]
    true_vel = full_state[IDX_VEL, :]
    true_pos = full_state[IDX_POS, :]

    pos_err = np.sqrt(np.sum((est_pos - true_pos) ** 2, axis=0))
    vel_err = np.sqrt(np.sum((est_vel - true_vel) ** 2, axis=0))

    print("\n" + "=" * 50)
    print("   全矩阵自适应 (6参数) 性能评估")
    print("=" * 50)
    print(f"位置 RMSE: {np.sqrt(np.mean(pos_err ** 2)):.4f} m")
    print(f"位置 Max : {np.max(pos_err):.4f} m")
    print(f"速度 RMSE: {np.sqrt(np.mean(vel_err ** 2)):.4f} m/s")
    print("=" * 50)

    # 计算基准模型 (Fixed 0.98) 用于对比
    base_mat = get_trans_matrix_6d(0.9, 0.05, 0.05, 0.98, 0.05, 0.05)
    imm_base = IMMFilter(base_mat, init_x, init_P, r_cov)
    est_base = np.zeros((9, num_steps))
    for k in range(num_steps):
        e, _ = imm_base.update(measurements[:, k], DT)
        est_base[:, k] = e
    est_base_vel = est_base[IDX_VEL, :]
    base_pos_err = np.sqrt(np.sum((est_base[IDX_POS] - true_pos) ** 2, axis=0))
    base_vel_err = np.sqrt(np.sum((est_base_vel - true_vel) ** 2, axis=0))

    print(f"基准模型 RMSE: {np.sqrt(np.mean(base_pos_err ** 2)):.4f} m")
    print(f"基准模型速度 RMSE: {np.sqrt(np.mean(base_vel_err ** 2)):.4f} m/s")

    # 绘图
    t = np.arange(num_steps) * DT
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, pos_err, 'r-', label='Adaptive 6-Param')
    ax1.plot(t, base_pos_err, 'k-', alpha=0.5, label='Baseline (Fixed)')
    ax1.set_ylabel('位置误差 (m)')
    ax2.set_xlabel('时间 (s)')
    ax1.legend()
    ax1.set_title('位置误差对比')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, vel_err, 'orange', label='自适应 6-Param')
    ax2.plot(t, base_vel_err, 'k-', alpha=0.7, label='基准模型 (固定)')
    ax2.set_ylabel('速度误差 (m/s)')
    ax2.set_xlabel('时间 (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fig2, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fig2.suptitle('6个核心转移概率参数演化趋势', fontsize=14)

    # 1. P00 (M0保持)
    axes[0, 0].plot(t, p00_history, 'b-')
    axes[0, 0].set_title('$p_{00}$ (M0 保持)')
    axes[0, 0].set_ylabel('概率值')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.0)

    # 2. P01 (M0 -> M1)
    axes[0, 1].plot(t, p01_history, 'b-')
    axes[0, 1].set_title('$p_{01}$ (M0 $\\to$ M1)')
    axes[0, 1].set_ylabel('概率值')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.0)

    # 3. P10 (M1 -> M0)
    axes[1, 0].plot(t, p10_history, 'b-')
    axes[1, 0].set_title('$p_{10}$ (M1 $\\to$ M0)')
    axes[1, 0].set_ylabel('概率值')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.0)

    # 4. P11 (M1保持)
    axes[1, 1].plot(t, p11_history, 'b-')
    axes[1, 1].set_title('$p_{11}$ (M1 保持)')
    axes[1, 1].set_ylabel('概率值')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.0)

    # 5. P20 (M2 -> M0)
    axes[2, 0].plot(t, p20_history, 'b-')
    axes[2, 0].set_title('$p_{20}$ (M2 $\\to$ M0)')
    axes[2, 0].set_ylabel('概率值')
    axes[2, 0].set_xlabel('时间 (s)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, 1.0)

    # 6. P21 (M2 -> M1)
    axes[2, 1].plot(t, p21_history, 'b-')
    axes[2, 1].set_title('$p_{21}$ (M2 $\\to$ M1)')
    axes[2, 1].set_ylabel('概率值')
    axes[2, 1].set_xlabel('时间 (s)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, 1.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以容纳 suptitle

    plt.show()


if __name__ == "__main__":
    main_ultimate()