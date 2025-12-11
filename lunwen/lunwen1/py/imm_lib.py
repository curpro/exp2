import numpy as np


class IMMFilter:
    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None):
        """
        IMM Filter 针对 F-16 轨迹跟踪 (9D State)
        状态向量 X = [x, vx, ax, y, vy, ay, z, vz, az]

        模型定义:
        0. CV (Constant Velocity): 匀速直线，忽略加速度输入 (Low Noise)
        1. CA (Constant Acceleration): 匀速/变加速直线 (Medium Noise)
        2. CT (Coordinated Turn): 协同转弯，X-Y 耦合 (High/Specific Noise)
        """
        self.dim = 9
        self.M = 3  # 模型数量

        # 转移概率矩阵
        self.trans_prob = transition_probabilities

        # 初始模型概率 (均匀分布或自定义)
        self.model_probs = np.array([0.4, 0.4, 0.2])

        # 初始化状态和协方差
        self.x = np.zeros((self.M, self.dim))
        self.P = np.zeros((self.M, self.dim, self.dim))

        for i in range(self.M):
            self.x[i] = initial_state.copy()
            self.P[i] = initial_cov.copy()

        # --- Q 参数设置 (针对 F-16 的经验值) ---
        # CV: 假设加速度很小，主要靠速度预测
        # CA: 允许显著的加速度变化
        # CT: 允许转弯产生的横向加速度扰动
        self.q_params = [
            1.0,  # Model 0: CV (Low noise)
            25.0,  # Model 1: CA (Medium noise)
            35.0  # Model 2: CT (High noise to accommodate turn uncertainty)
        ]

        # 测量矩阵 H (只观测位置 x, y, z)
        # indices: x(0), y(3), z(6)
        self.H = np.zeros((3, self.dim))
        self.H[0, 0] = 1
        self.H[1, 3] = 1
        self.H[2, 6] = 1

        # 观测噪声 R
        if r_cov is not None:
            self.R = r_cov
        else:
            self.R = np.eye(3) * 100.0

    # =================================================================
    #  模型 1: CV (Constant Velocity) 匀速模型
    # =================================================================
    def get_F_CV(self, dt):
        """
        CV 模型矩阵：
        认为物体做匀速运动，加速度项不参与位置和速度的更新。
        这能让滤波器在平飞阶段迅速降低对噪声的敏感度。
        """
        F = np.eye(self.dim)

        # 仅更新位置：p = p + v*dt (忽略 a)
        # 仅更新速度：v = v        (忽略 a)

        # X 轴
        F[0, 1] = dt
        # Y 轴
        F[3, 4] = dt
        # Z 轴
        F[6, 7] = dt

        return F

    # =================================================================
    #  模型 2: CA (Constant Acceleration) 常加速模型
    # =================================================================
    def get_F_CA(self, dt):
        """
        CA 模型矩阵：
        标准的牛顿运动学，适用于直线加速、爬升、俯冲。
        """
        F = np.eye(self.dim)

        # 3x3 运动学块
        # p = p + v*dt + 0.5*a*dt^2
        # v = v + a*dt
        # a = a
        block = np.array([
            [1, dt, 0.5 * dt ** 2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # 填充到 9x9 矩阵
        for i in [0, 3, 6]:  # 对应 x, y, z 的起始索引
            F[i:i + 3, i:i + 3] = block

        return F

    # =================================================================
    #  模型 3: CT (Coordinated Turn) 协同转弯模型
    # =================================================================
    def get_F_CT(self, dt, omega=0.15):  # omega 默认为约 8.5度/秒
        """
        CT 模型矩阵 (核心修改)：
        引入 X 和 Y 的耦合，描述水平面的圆周运动。
        omega: 转弯率 (rad/s)。
        对于 F-16，转弯率是变化的，这里取一个典型的平均值或通过 EKF 估算。
        为简单起见，这里使用固定转弯率的线性化模型。
        """
        F = np.eye(self.dim)

        sin_w = np.sin(omega * dt)
        cos_w = np.cos(omega * dt)

        # --- 水平面 (X-Y) 耦合更新 ---
        # 索引参考: x(0), vx(1), ax(2) | y(3), vy(4), ay(5)

        # 1. 位置更新 (XPos 由 XVel 和 YVel 共同决定)
        F[0, 1] = sin_w / omega
        F[0, 4] = -(1 - cos_w) / omega

        F[3, 1] = (1 - cos_w) / omega
        F[3, 4] = sin_w / omega

        # 2. 速度更新 (XVel 由 XVel 和 YVel 旋转得到)
        F[1, 1] = cos_w
        F[1, 4] = -sin_w

        F[4, 1] = sin_w
        F[4, 4] = cos_w

        # 3. 加速度处理
        # 简单处理：加速度作为噪声驱动的状态保持，或者也可以旋转
        # 这里保持加速度为独立的一阶马尔可夫过程 (衰减或保持)
        # 稍微加入一点衰减，防止转弯结束后加速度发散
        F[2, 2] = 1.0
        F[5, 5] = 1.0

        # --- 高度 (Z) 轴通常独立，按 CA 模型处理 ---
        # z = z + vz*dt + 0.5*az*dt^2
        F[6, 7] = dt
        F[6, 8] = 0.5 * dt ** 2
        F[7, 8] = dt

        return F

    def get_Q(self, dt, q_std):
        """
        生成过程噪声矩阵 Q
        基于离散化白噪声加速度模型 (Discrete White Noise Acceleration)
        """
        Q = np.zeros((self.dim, self.dim))
        var = q_std ** 2

        # 对每个维度 (x, y, z) 填充 Q 块
        # 对于 9D 状态 (p, v, a)，通常假设加速度的导数(jerk)是白噪声，
        # 或者假设加速度本身是随机游走。这里使用简化的 CA Q 矩阵形式。

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        # 这是一个针对 (pos, vel, acc) 的典型 Q 块近似
        # 假设加速度项由方差 var 的噪声驱动
        q_block = np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6, dt ** 2 / 2, dt]
        ]) * var

        for i in [0, 3, 6]:
            Q[i:i + 3, i:i + 3] = q_block

        return Q

    def interact(self):
        """步骤 1: 交互 (Interaction)"""
        c_bar = np.dot(self.trans_prob.T, self.model_probs)
        EPS = 1e-20  # 防止除零

        mixing_probs = np.zeros((self.M, self.M))
        for i in range(self.M):
            for j in range(self.M):
                mixing_probs[i, j] = (self.trans_prob[i, j] * self.model_probs[i]) / (c_bar[j] + EPS)

        x_mixed = np.zeros((self.M, self.dim))
        P_mixed = np.zeros((self.M, self.dim, self.dim))

        for j in range(self.M):
            for i in range(self.M):
                x_mixed[j] += mixing_probs[i, j] * self.x[i]

            for i in range(self.M):
                diff = (self.x[i] - x_mixed[j]).reshape(-1, 1)
                P_mixed[j] += mixing_probs[i, j] * (self.P[i] + diff @ diff.T)

        return x_mixed, P_mixed, c_bar

    def update(self, z, dt):
        """步骤 2 & 3: 滤波更新 (Filtering Update)"""

        # --- 关键修改：在这里分别生成不同的模型矩阵 ---
        # 模型 0: CV
        # 模型 1: CA
        # 模型 2: CT

        model_defs = [
            {'F': self.get_F_CV(dt), 'Q': self.get_Q(dt, self.q_params[0])},
            {'F': self.get_F_CA(dt), 'Q': self.get_Q(dt, self.q_params[1])},
            {'F': self.get_F_CT(dt, omega=0.22), 'Q': self.get_Q(dt, self.q_params[2])}  # 设定转弯率
        ]

        x_mixed, P_mixed, c_bar = self.interact()

        likelihoods = np.zeros(self.M)
        EPS = 1e-50

        for i in range(self.M):
            F = model_defs[i]['F']
            Q = model_defs[i]['Q']

            # 卡尔曼预测
            x_pred = F @ x_mixed[i]
            P_pred = F @ P_mixed[i] @ F.T + Q

            # 测量残差
            y_res = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R

            # 卡尔曼更新
            try:
                S_inv = np.linalg.inv(S)
                K = P_pred @ self.H.T @ S_inv
            except np.linalg.LinAlgError:
                K = np.zeros((self.dim, 3))
                S_inv = np.eye(3)

            self.x[i] = x_pred + K @ y_res

            # Joseph form 更新 P，保证正定性
            I_KH = np.eye(self.dim) - K @ self.H
            self.P[i] = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

            # 计算似然 (Likelihood)
            S_det = np.linalg.det(S)
            denom = np.sqrt((2 * np.pi) ** 3 * S_det)
            mahalanobis = -0.5 * y_res.T @ S_inv @ y_res
            likelihoods[i] = np.exp(mahalanobis) / (denom + EPS)

        # 步骤 4: 更新模型概率
        new_probs = likelihoods * c_bar
        sum_probs = np.sum(new_probs)

        if sum_probs < EPS:
            # 如果数值下溢，重置为均匀分布或保持上一时刻
            self.model_probs = c_bar
        else:
            self.model_probs = new_probs / sum_probs

        # 步骤 5: 状态融合
        x_out = np.zeros(self.dim)
        for i in range(self.M):
            x_out += self.model_probs[i] * self.x[i]

        return x_out, self.model_probs
