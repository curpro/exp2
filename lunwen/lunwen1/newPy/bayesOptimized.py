import os
import time
import warnings
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BoTorch 依赖
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
# 引入归一化和标准化工具，这对高斯过程回归的稳定性至关重要
from botorch.models.transforms import Standardize, Normalize

# 导入本地的 IMM 算法模块
# 假设 imm_lib.py 在同一目录下
from lunwen1.py.imm_lib import IMMFilter

# 设置环境和警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# 绘图设置
matplotlib_font = 'SimHei'  # Windows请用 SimHei
plt.rcParams['font.sans-serif'] = [matplotlib_font]
plt.rcParams['axes.unicode_minus'] = False

# 颜色定义
GREEN = '\033[32m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
CYAN = '\033[36m'
RED = '\033[37m'
RESET = '\033[0m'

# ==========================================
# 1. 全局配置与数据准备
# ==========================================
# 设备配置
device = torch.device("cpu")
dtype = torch.double

# 优化参数
N_INIT = 100  # 初始随机样本点数量 (Warm-up)
N_BATCH = 150  # 迭代次数 (建议设大一点，比如50-100，让GP充分收敛)
BATCH_SIZE = 1  # 每次推荐点数
MC_SAMPLES = 500  # 蒙特卡洛采样数

# 参数范围 [a, b, c, d]
# 限制在 0.1~0.9，给 p_remain 留出空间
lower_bounds = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=dtype, device=device)
upper_bounds = torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99, 0.99], dtype=dtype, device=device)
bounds = torch.stack([lower_bounds, upper_bounds])

# ------------------------------------------
# 加载 F-16 极限机动数据
# ------------------------------------------
CSV_FILE_PATH = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver_a.csv'
DT = 1 / 30  # 30Hz 采样率
MEAS_NOISE_STD = 15.0  # 观测噪声


def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # 转换状态矩阵 [x, vx, y, vy, z, vz]
        state_matrix = df[['x', 'vx', 'ax', 'y', 'vy', 'ay', 'z', 'vz', 'az']].to_numpy().T
        return state_matrix
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


print("正在加载基准轨迹数据...")
truth_state_full = load_csv_data(CSV_FILE_PATH)
if truth_state_full is None:
    raise ValueError("数据加载失败！")

num_steps = truth_state_full.shape[1]
print(f"数据加载成功: {num_steps} 帧, DT={DT:.4f}s")

# 提取真值
true_pos = truth_state_full[[0, 3, 6], :]  # 改为 0, 3, 6
true_vel = truth_state_full[[1, 4, 7], :]  # 改为 1, 4, 7

# 生成带噪声的观测 (固定种子以保证优化过程的可重复性)
np.random.seed(2024)
meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
meas_pos = true_pos + meas_noise

# ------------------------------------------
# 初始化策略: 真值 + 微小扰动 (Perturbed GT)
# ------------------------------------------
# 这种初始化既消除了巨大的初始尖刺，又保留了一定的真实不确定性
gt_init = truth_state_full[:, 0]
init_pos_err = 10.0
init_vel_err = 5.0

# --- 修改开始 ---
# 9维状态: x, vx, ax, y, vy, ay, z, vz, az
init_noise = np.random.randn(9)
# 给位置、速度、加速度分别设置噪声
init_noise[[0, 3, 6]] *= init_pos_err
init_noise[[1, 4, 7]] *= init_vel_err
init_noise[[2, 5, 8]] *= 1.0  # 加速度初始误差

init_state = gt_init + init_noise

# 初始协方差 (9x9)
p_pos = init_pos_err ** 2
p_vel = init_vel_err ** 2
p_acc = 1.0 ** 2
# 构造对角矩阵
diag_values = [p_pos, p_vel, p_acc, p_pos, p_vel, p_acc, p_pos, p_vel, p_acc]
init_cov = np.diag(diag_values)

# 观测噪声协方差 R
r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

# 打包全局数据
GLOBAL_DATA = {
    'meas_pos': meas_pos,
    'true_pos': true_pos,
    'true_vel': true_vel,
    'dt': DT,
    'init_state': init_state,
    'init_cov': init_cov,
    'r_cov': r_cov
}


# ==========================================
# 2. 核心评估函数 (Objective Function)
# ==========================================
def run_imm_and_get_score(params):
    """ 输入 params=[a,b,c,d], 输出 -RMSE """
    p11, p12, p21, p22, p31, p32 = params

    # 计算剩余概率 (保证每一行和为1)
    p13 = 1.0 - p11 - p12
    p23 = 1.0 - p21 - p22
    p33 = 1.0 - p31 - p32

    # 物理约束检查 (概率必须 > 0)
    # 使用 0.001 作为软边界，防止数值不稳定
    if p13 < 0 or p23 < 0 or p33 < 0:
        return -300.0  # 给予巨大惩罚

    trans_matrix = np.array([
        [p11, p12, p13],
        [p21, p22, p23],
        [p31, p32, p33]
    ])

    try:
        # 初始化滤波器 (传入正确的 r_cov)
        imm = IMMFilter(
            transition_probabilities=trans_matrix,
            initial_state=GLOBAL_DATA['init_state'],
            initial_cov=GLOBAL_DATA['init_cov'],
            r_cov=GLOBAL_DATA['r_cov']
        )
    except Exception:
        return -300.0

    est_state = np.zeros((9, num_steps))
    est_state[:, 0] = GLOBAL_DATA['init_state']

    meas = GLOBAL_DATA['meas_pos']
    dt_val = GLOBAL_DATA['dt']

    # 运行滤波
    for i in range(1, num_steps):
        z = meas[:, i]
        est, _ = imm.update(z, dt_val)
        est_state[:, i] = est

    # ------------------------------------------------------
    # 【新增】计算误差 (忽略前10帧初始化阶段)
    # ------------------------------------------------------
    start_idx = 10

    # 1. 计算位置 RMSE
    err_pos = est_state[[0, 3, 6], start_idx:] - GLOBAL_DATA['true_pos'][:, start_idx:]
    rmse_pos = np.sqrt(np.mean(np.sum(err_pos ** 2, axis=0)))

    # 2. 计算速度 RMSE
    err_vel = est_state[[1, 4, 7], start_idx:] - GLOBAL_DATA['true_vel'][:, start_idx:]
    rmse_vel = np.sqrt(np.mean(np.sum(err_vel ** 2, axis=0)))

    # 3. 组合目标函数
    # 速度误差通常比位置误差大且更不稳定，给予 0.8 到 1.0 的权重比较合适
    # 如果你非常在意速度跳变，可以把 vel_weight 设为 1.5 甚至 2.0
    vel_weight = 1.0

    combined_score = -(rmse_pos + vel_weight * rmse_vel)

    return combined_score


def evaluate_y_batch(X_tensor):
    results = []
    for i in range(X_tensor.shape[0]):
        params = X_tensor[i].cpu().numpy()
        score = run_imm_and_get_score(params)
        results.append([score])
    return torch.tensor(results, device=device, dtype=dtype)


# ==========================================
# 3. 约束条件定义
# ==========================================
# 这里的约束是为了辅助 optimize_acqf 找到满足概率和要求的点
# 1. a + b <= 0.999
# 2. c + d <= 0.999
# 3. a + b + c + d >= 1.001 (保证 p33 > 0)
# Row 1: p11 + p12 <= 1  (indices 0, 1)
constraint_row1 = (torch.tensor([0, 1], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
# Row 2: p21 + p22 <= 1  (indices 2, 3)
constraint_row2 = (torch.tensor([2, 3], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)
# Row 3: p31 + p32 <= 1  (indices 4, 5)
constraint_row3 = (torch.tensor([4, 5], device=device), torch.tensor([-1.0, -1.0], dtype=dtype, device=device), -0.999)

constraints_list = [constraint_row1, constraint_row2, constraint_row3]


# ==========================================
# 4. 辅助功能函数
# ==========================================
def initialize_model(train_x, train_y, state_dict=None):
    # 使用 GP 模型拟合当前的参数-性能关系
    model = SingleTaskGP(
        train_x,
        train_y,
        # 标准化和归一化对于 GP 的收敛非常重要
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=train_y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def generate_valid_random_params(n=1):
    """【修改点4】生成满足行和约束的 6 维随机参数"""
    valid_points = []
    while len(valid_points) < n:
        # 生成 6 个 [0.01, 0.99] 的随机数
        proposal = torch.rand(6, dtype=dtype, device=device) * (upper_bounds - lower_bounds) + lower_bounds
        p = proposal.cpu().numpy()

        # 检查每一行的和是否合法
        row1_ok = (p[0] + p[1]) <= 0.999
        row2_ok = (p[2] + p[3]) <= 0.999
        row3_ok = (p[4] + p[5]) <= 0.999

        if row1_ok and row2_ok and row3_ok:
            valid_points.append(proposal)

    return torch.stack(valid_points)


# ==========================================
# 5. 主流程
# ==========================================
def main():
    print(f"{GREEN}=== 开始 F-16 轨迹跟踪参数优化 (多策略对比) ==={RESET}")
    print(f"初始样本: {N_INIT}, 迭代次数: {N_BATCH}")

    # 1. 初始化数据 (Warm-up)
    print(f"正在生成 {N_INIT} 个初始随机样本...")
    train_x = generate_valid_random_params(N_INIT)
    train_y = evaluate_y_batch(train_x)

    # 复制给四种策略 (独立维护数据集，模拟真实优化过程)
    train_x_ei, train_y_ei = train_x.clone(), train_y.clone()
    train_x_qei, train_y_qei = train_x.clone(), train_y.clone()
    train_x_ucb, train_y_ucb = train_x.clone(), train_y.clone()
    train_x_rnd, train_y_rnd = train_x.clone(), train_y.clone()

    # 记录最佳值轨迹
    best_y_ei = [train_y_ei.max().item()]
    best_y_qei = [train_y_qei.max().item()]
    best_y_ucb = [train_y_ucb.max().item()]
    best_y_rnd = [train_y_rnd.max().item()]

    print(f"初始随机最佳 RMSE: {-train_y.max().item():.4f} m")

    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # 2. 迭代优化循环
    for i in range(N_BATCH):
        t0 = time.time()

        # --- A. 更新 GP 模型 ---
        mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)
        mll_qei, model_qei = initialize_model(train_x_qei, train_y_qei)
        mll_ucb, model_ucb = initialize_model(train_x_ucb, train_y_ucb)

        fit_gpytorch_mll(mll_ei)
        fit_gpytorch_mll(mll_qei)
        fit_gpytorch_mll(mll_ucb)

        # --- B. 定义采集函数 ---
        EI = ExpectedImprovement(model=model_ei, best_f=train_y_ei.max())
        qEI = qExpectedImprovement(model=model_qei, best_f=train_y_qei.max(), sampler=qmc_sampler)
        UCB = UpperConfidenceBound(model=model_ucb, beta=3.0)  # beta 控制探索程度

        # --- C. 获取推荐点 ---
        def get_next_point(acq_f):
            candidates, _ = optimize_acqf(
                acq_function=acq_f,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=10,
                raw_samples=128,
                inequality_constraints=constraints_list
            )
            new_x = candidates.detach()
            new_y = evaluate_y_batch(new_x)
            return new_x, new_y

        # AI 策略推荐
        new_x_ei, new_y_ei = get_next_point(EI)
        new_x_qei, new_y_qei = get_next_point(qEI)
        new_x_ucb, new_y_ucb = get_next_point(UCB)

        # Random 策略推荐 (对照组)
        new_x_rnd = generate_valid_random_params(BATCH_SIZE)
        new_y_rnd = evaluate_y_batch(new_x_rnd)

        # --- D. 更新数据集 ---
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_y_ei = torch.cat([train_y_ei, new_y_ei])

        train_x_qei = torch.cat([train_x_qei, new_x_qei])
        train_y_qei = torch.cat([train_y_qei, new_y_qei])

        train_x_ucb = torch.cat([train_x_ucb, new_x_ucb])
        train_y_ucb = torch.cat([train_y_ucb, new_y_ucb])

        train_x_rnd = torch.cat([train_x_rnd, new_x_rnd])
        train_y_rnd = torch.cat([train_y_rnd, new_y_rnd])

        # --- E. 记录当前最佳值 ---
        current_best_ei = train_y_ei.max().item()
        current_best_qei = train_y_qei.max().item()
        current_best_ucb = train_y_ucb.max().item()
        current_best_rnd = train_y_rnd.max().item()

        best_y_ei.append(current_best_ei)
        best_y_qei.append(current_best_qei)
        best_y_ucb.append(current_best_ucb)
        best_y_rnd.append(current_best_rnd)

        t1 = time.time()
        print(f"\nBatch {i + 1}/{N_BATCH} | Time: {t1 - t0:.2f}s")
        print(f"{BLUE}[EI]  RMSE: {-new_y_ei.item():.4f} | Best: {-current_best_ei:.4f}{RESET}")
        print(f"{BLUE}[qEI] RMSE: {-new_y_qei.item():.4f} | Best: {-current_best_qei:.4f}{RESET}")
        print(f"{CYAN}[UCB] RMSE: {-new_y_ucb.item():.4f} | Best: {-current_best_ucb:.4f}{RESET}")
        print(f"{RED}[Rand] RMSE: {-new_y_rnd.item():.4f} | Best: {-current_best_rnd:.4f}{RESET}")

    # ==========================================
    # 6. 结果展示
    # ==========================================
    # 转换回正的 RMSE
    trace_ei = [-x for x in best_y_ei]
    trace_qei = [-x for x in best_y_qei]
    trace_ucb = [-x for x in best_y_ucb]
    trace_rnd = [-x for x in best_y_rnd]

    best_val_ei = train_y_ei.max().item()
    best_val_qei = train_y_qei.max().item()
    best_val_ucb = train_y_ucb.max().item()
    best_val_rnd = train_y_rnd.max().item()

    # 找出全局最佳值 (Score是负误差，所以越大越好)
    global_best_val = max(best_val_ei, best_val_qei, best_val_ucb, best_val_rnd)

    # 找出赢家并提取参数
    if global_best_val == best_val_ei:
        idx = train_y_ei.argmax()
        best_params = train_x_ei[idx].cpu().numpy()
    elif global_best_val == best_val_qei:
        idx = train_y_qei.argmax()
        best_params = train_x_qei[idx].cpu().numpy()
    elif global_best_val == best_val_ucb:
        idx = train_y_ucb.argmax()
        best_params = train_x_ucb[idx].cpu().numpy()
    else:
        idx = train_y_rnd.argmax()
        best_params = train_x_rnd[idx].cpu().numpy()

    best_rmse = -global_best_val  # 转回正数误差

    # 计算最终的转移矩阵
    p11, p12, p21, p22, p31, p32 = best_params
    # 计算剩余概率 (Row Sum = 1)
    p13 = 1.0 - p11 - p12
    p23 = 1.0 - p21 - p22
    p33 = 1.0 - p31 - p32

    best_matrix = np.array([
        [p11, p12, p13],
        [p21, p22, p23],
        [p31, p32, p33]
    ])

    print(f"\n{GREEN}=== 优化完成 ==={RESET}")
    print(f"最佳 RMSE: {best_rmse:.6f} m")
    print(f"最佳参数 [a, b, c, d,e, f]: {best_params.round(6)}")
    print("最佳转移矩阵 (请复制到 n_BoIMM.py):")
    print(np.array2string(best_matrix, separator=', '))

    # 绘图
    iters = np.arange(len(trace_ei))
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(iters, trace_ei, linewidth=2, label="EI")
    plt.plot(iters, trace_qei, linewidth=2, label="qEI (Rec.)")
    plt.plot(iters, trace_ucb, linewidth=2, label="UCB")
    plt.plot(iters, trace_rnd, linewidth=2, label="Random")

    plt.title(f'贝叶斯优化策略对比 (Noise={MEAS_NOISE_STD}m)')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳 RMSE (m)')
    plt.legend()

    save_path = f'BO_IMM_Comparison_{datetime.now().strftime("%H%M%S")}.png'
    plt.savefig(save_path, dpi=300)
    print(f"结果图已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()