import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

# BoTorch 依赖
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize, Normalize

# 导入本地模块
from helper_data import generate_truth_data
from lunwen1.py.imm_lib import IMMFilter

# 设置环境
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置
device = torch.device("cpu")
dtype = torch.double
N_INIT = 20
N_BATCH = 30
BATCH_SIZE = 1
MC_SAMPLES = 500

# 参数范围 [a, b, c, d]
lower_bounds = torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=dtype, device=device)
upper_bounds = torch.tensor([0.99, 0.99, 0.99, 0.99], dtype=dtype, device=device)
bounds = torch.stack([lower_bounds, upper_bounds])

# --- 1. 数据准备 ---
print("正在生成仿真基准数据 (9维状态)...")
truth_state_full, time_vec = generate_truth_data()
dt = time_vec[1] - time_vec[0]
num_steps = len(time_vec)

# 提取真值 (9维)
# 位置索引: 0(x), 3(y), 6(z)
true_pos = truth_state_full[[0, 3, 6], :]
# 速度索引: 1(vx), 4(vy), 7(vz)
true_vel = truth_state_full[[1, 4, 7], :]

# 生成观测 (仅位置)
MEAS_NOISE_STD = 4.0
np.random.seed(42)
meas_noise = np.random.randn(*true_pos.shape) * MEAS_NOISE_STD
meas_pos = true_pos + meas_noise

# 初始状态 (9维)
init_state = truth_state_full[:, 0].copy()

# 初始协方差 (9x9)
# 对角线结构: [px, vx, ax, py, vy, ay, pz, vz, az]
p_pos = 100.0;
p_vel = 10.0;
p_acc = 1.0
diag_vals = [p_pos, p_vel, p_acc, p_pos, p_vel, p_acc, p_pos, p_vel, p_acc]
init_cov = np.diag(diag_vals)

# 观测噪声矩阵
r_cov = np.eye(3) * (MEAS_NOISE_STD ** 2)

GLOBAL_DATA = {
    'meas_pos': meas_pos,
    'true_pos': true_pos,
    'dt': dt,
    'init_state': init_state,
    'init_cov': init_cov,
    'r_cov': r_cov
}


# --- 2. 评估函数 ---
def run_imm_and_get_score(params):
    a, b, c, d = params

    # 概率计算 (行和为1)
    p13 = 1 - a - b
    p23 = 1 - c - d
    # 假设第三行逻辑: CA_High 保持率较高，或与前两行对称
    p31 = (1 - a) / 2
    p32 = (1 - d) / 2
    p33 = 1 - p31 - p32

    # 约束检查
    if any(p < 0 for p in [p13, p23, p31, p32, p33]):
        return -1e5  # 强惩罚

    trans_matrix = np.array([
        [a, b, p13],
        [c, d, p23],
        [p31, p32, p33]
    ])

    try:
        # 初始化 9维 IMM
        imm = IMMFilter(
            transition_probabilities=trans_matrix,
            initial_state=GLOBAL_DATA['init_state'],
            initial_cov=GLOBAL_DATA['init_cov'],
            r_cov=GLOBAL_DATA['r_cov']
        )
    except:
        return -1e5

    # 运行滤波
    est_path = []
    meas = GLOBAL_DATA['meas_pos']
    dt_val = GLOBAL_DATA['dt']

    for i in range(meas.shape[1]):
        z = meas[:, i]
        x_est, _ = imm.update(z, dt_val)
        # 提取位置 (0, 3, 6)
        est_path.append([x_est[0], x_est[3], x_est[6]])

    est_path = np.array(est_path).T  # (3, N)

    # 计算 RMSE (忽略前50步初始化误差)
    err = np.linalg.norm(est_path[:, 50:] - GLOBAL_DATA['true_pos'][:, 50:], axis=0)
    rmse = np.sqrt(np.mean(err ** 2))

    return -rmse


def evaluate_y_batch(X_tensor):
    results = []
    for i in range(X_tensor.shape[0]):
        score = run_imm_and_get_score(X_tensor[i].cpu().numpy())
        results.append([score])
    return torch.tensor(results, device=device, dtype=dtype)


# --- 3. 约束定义 ---
# 确保 a+b <= 1, c+d <= 1
con1 = (torch.tensor([0, 1], device=device), torch.tensor([1.0, 1.0], dtype=dtype, device=device), 1.0)
con2 = (torch.tensor([2, 3], device=device), torch.tensor([1.0, 1.0], dtype=dtype, device=device), 1.0)
constraints_list = [con1, con2]


# --- 4. 辅助函数 ---
def initialize_model(train_x, train_y):
    model = SingleTaskGP(
        train_x, train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=train_y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def generate_valid_random_params(n=1):
    valid = []
    while len(valid) < n:
        prop = torch.rand(4, dtype=dtype, device=device) * (upper_bounds - lower_bounds) + lower_bounds
        if (prop[0] + prop[1] <= 0.99) and (prop[2] + prop[3] <= 0.99):
            valid.append(prop)
    return torch.stack(valid)


# --- 5. 主函数 ---
def main():
    print(f"=== 开始 9维 IMM 贝叶斯优化 ===")

    # 初始化
    train_x = generate_valid_random_params(N_INIT)
    train_y = evaluate_y_batch(train_x)

    best_y_hist = [train_y.max().item()]

    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    for i in range(N_BATCH):
        mll, model = initialize_model(train_x, train_y)
        fit_gpytorch_mll(mll)

        qEI = qExpectedImprovement(model=model, best_f=train_y.max(), sampler=qmc_sampler)

        candidate, _ = optimize_acqf(
            qEI, bounds=bounds, q=BATCH_SIZE, num_restarts=10, raw_samples=128,
            inequality_constraints=constraints_list
        )

        new_y = evaluate_y_batch(candidate)
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])

        best_y_hist.append(train_y.max().item())
        print(f"Iter {i + 1}/{N_BATCH} | Best RMSE: {-train_y.max().item():.4f} m | New: {-new_y.item():.4f}")

    # 结果
    idx = train_y.argmax()
    best_p = train_x[idx].cpu().numpy()
    print("\n=== 优化完成 ===")
    print(f"最佳参数 [a, b, c, d]: {best_p}")
    print(f"最佳 RMSE: {-train_y.max().item():.4f} m")

    plt.plot([-y for y in best_y_hist], marker='o')
    plt.ylabel('RMSE (m)')
    plt.xlabel('Iteration')
    plt.title('Bayesian Optimization Progress')
    plt.show()


if __name__ == '__main__':
    main()