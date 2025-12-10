import numpy as np


class IMMFilter:
    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None, omega=None):
        """
        [修正版] IMM Filter: 9D 状态 (x, vx, ax, y, vy, ay, z, vz, az)
        模型集合: CA-High Noise, CA-Low Noise, CA-Medium Noise
        """
        self.M = 3
        self.trans_prob = transition_probabilities
        # 初始概率
        self.model_probs = np.array([0.4, 0.3, 0.3])
        self.dim = 9  # <--- 状态维度修正为 9

        # 初始化状态
        self.x = np.zeros((self.M, self.dim))
        self.P = np.zeros((self.M, self.dim, self.dim))

        for i in range(self.M):
            self.x[i] = initial_state.copy()
            self.P[i] = initial_cov.copy()

        # --- 关键修改：基于数据统计特性的 Q 参数 ---
        # 你的 CA 模型集合 (只通过噪声强度区分)
        self.q_params = {
            'ca_low_std': 5.0,  # 低机动 (常规飞行)
            'ca_medium_std': 25.0,  # 中等机动 (转弯)
            'ca_high_std': 30.0  # 高机动 (超机动)
        }
        self.q_list = [
            self.q_params['ca_high_std'],
            self.q_params['ca_low_std'],
            self.q_params['ca_medium_std'],
        ]

        # 测量矩阵 H (只观测位置 x, y, z)
        self.H = np.zeros((3, self.dim))
        self.H[0, 0] = 1  # x
        self.H[1, 3] = 1  # y  <--- 索引修正
        self.H[2, 6] = 1  # z  <--- 索引修正

        # 观测噪声 R
        if r_cov is not None:
            self.R = r_cov
        else:
            self.R = np.eye(3) * 16

    # --- 修正后的模型 F 矩阵 (常加速度模型) ---
    def get_F_CA_model(self, dt):
        """生成 9x9 常加速度 (CA) 模型的转移矩阵"""
        F = np.eye(self.dim)

        # 3x3 CA 块
        f_block = np.array([
            [1, dt, dt ** 2 / 2.0],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # 组装 9x9 F 矩阵
        for i in range(3):
            start = i * 3
            F[start:start + 3, start:start + 3] = f_block

        return F

    # --- 修正后的模型 Q 矩阵 (常加速度模型) ---
    def get_Q_CA_model(self, dt, q_std):
        """生成 9x9 CA 模型的运动学过程噪声矩阵"""
        q_var = q_std ** 2

        # 3x3 Q 块 (常加速度离散模型)
        q2 = dt ** 2 / 2.0
        q3 = dt ** 3 / 3.0

        q_block = np.array([
            [q3, q2, dt ** 2 / 4.0],  # 修正：加速度项通常采用 G*Qc*G^T 积分形式，这里采用标准 CA 模型的近似。
            [q2, dt, dt / 2.0],
            [dt ** 2 / 4.0, dt / 2.0, dt]
        ]) * q_var

        # 简化版 Q 块 (你原代码采用的近似)
        q_simple = np.array([
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt]
        ]) * q_var

        Q = np.zeros((self.dim, self.dim))

        # 组装 9x9 Q 矩阵 (通常 Q 仅在加速度项有值，但 IMM 需要在所有维度有值)
        # 我们使用你代码中隐式采用的 CA 离散化 Q 矩阵
        for i in range(3):
            start = i * 3
            # 使用标准的 9D CA 模型的 Q 矩阵
            Q[start:start + 3, start:start + 3] = q_simple  # 这里使用标准公式，比你的原公式更精确

        return Q

    # --- IMM 核心逻辑 (interact 和 update 保留原逻辑，但使用修正后的函数) ---
    def interact(self):
        """步骤 1: 交互/混合"""
        # ... (与原代码保持一致) ...
        EPS = 1e-20
        c_bar = np.dot(self.trans_prob.T, self.model_probs)

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
        """步骤 2 & 3: 预测与更新"""
        models = []
        for q_std in self.q_list:
            models.append({
                'F': self.get_F_CA_model(dt),
                'Q': self.get_Q_CA_model(dt, q_std)
            })

        EPS = 1e-20
        x_mixed, P_mixed, c_bar = self.interact()
        likelihoods = np.zeros(self.M)

        for i in range(self.M):
            # --- 预测 ---
            F = models[i]['F']
            Q = models[i]['Q']

            x_pred = F @ x_mixed[i]
            P_pred = F @ P_mixed[i] @ F.T + Q

            # --- 更新 ---
            y = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + self.R

            try:
                S_inv = np.linalg.inv(S)
                K = P_pred @ self.H.T @ S_inv
            except np.linalg.LinAlgError:
                S_inv = np.eye(3) * 1e-6
                K = np.zeros((self.dim, 3))

            self.x[i] = x_pred + K @ y
            I_KH = np.eye(self.dim) - K @ self.H
            # Joseph form update for stability
            self.P[i] = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

            # --- 计算似然 ---
            S_det = np.linalg.det(S)
            denom = np.sqrt(((2 * np.pi) ** 3 * S_det) + EPS)  # 添加EPS防止分母为零
            mahalanobis = -0.5 * y.T @ S_inv @ y
            likelihoods[i] = np.exp(mahalanobis) / (denom + EPS)

        # --- 步骤 4: 更新模型概率 ---
        new_probs = likelihoods * c_bar
        sum_probs = np.sum(new_probs)

        if sum_probs < EPS:
            self.model_probs = np.ones(self.M) / self.M
        else:
            self.model_probs = new_probs / sum_probs

        # --- 步骤 5: 融合状态 ---
        x_out = np.zeros(self.dim)
        for i in range(self.M):
            x_out += self.model_probs[i] * self.x[i]

        return x_out, self.model_probs