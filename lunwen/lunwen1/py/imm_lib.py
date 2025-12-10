import numpy as np


class IMMFilter:
    def __init__(self, transition_probabilities, initial_state, initial_cov, r_cov=None, omega=None):
        """
        [优化版] IMM Filter: 针对 F-16 数据的统计特性进行了 Q 阵调优
        """
        self.M = 3
        self.trans_prob = transition_probabilities
        # 初始概率
        self.model_probs = np.array([0.4, 0.3, 0.3])  # 稍微平衡一下初始猜测
        self.dim = 9

        # 初始化状态
        self.x = np.zeros((self.M, self.dim))
        self.P = np.zeros((self.M, self.dim, self.dim))

        for i in range(self.M):
            self.x[i] = initial_state.copy()
            self.P[i] = initial_cov.copy()

        # --- 关键修改：基于数据统计特性的 Q 参数 ---
        # 数据分析显示：平均加速度 ~15，最大 ~47
        self.q_params = {
            'cv_std': 0.5,  # 极小噪声，锁定平飞精度
            'ca_low_std': 10.0,  # [修改] 从 3.0 提至 10.0，匹配常规机动(Mean ~15)
            'ca_high_std': 50.0  # [修改] 从 80.0 降至 50.0，匹配极限机动(Max ~47)，增加模型锐度
        }

        # 观测矩阵 (只测位置 x, y, z)
        self.H = np.zeros((3, 9))
        self.H[0, 0] = 1
        self.H[1, 3] = 1
        self.H[2, 6] = 1

        # 观测噪声协方差
        if r_cov is not None:
            self.R = r_cov
        else:
            self.R = np.eye(3) * 225.0  # 15^2

    def _get_Q_CV(self, dt, sigma):
        """CV 模型 Q 阵"""
        q = sigma ** 2
        block_pos_vel = np.array([
            [dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 2 / 2, dt]
        ]) * q

        Q = np.zeros((9, 9))
        for i in range(3):  # x, y, z
            idx = i * 3
            Q[idx:idx + 2, idx:idx + 2] = block_pos_vel
            Q[idx + 2, idx + 2] = 1e-6
        return Q

    def _get_Q_CA(self, dt, sigma):
        """CA 模型 Q 阵"""
        q = sigma ** 2
        dt2 = dt ** 2 / 2
        dt3 = dt ** 3 / 6
        dt4 = dt ** 4 / 24
        dt5 = dt ** 5 / 20

        qs = np.array([
            [dt5, dt4, dt3],
            [dt4, dt ** 3 / 3, dt2],
            [dt3, dt2, dt]
        ]) * q

        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = qs
        Q[3:6, 3:6] = qs
        Q[6:9, 6:9] = qs
        return Q

    def get_F_CV(self, dt):
        """CV 转移矩阵"""
        F = np.eye(9)
        F[0, 1] = dt;
        F[3, 4] = dt;
        F[6, 7] = dt
        F[2, 2] = 0.0;
        F[5, 5] = 0.0;
        F[8, 8] = 0.0
        return F

    def get_F_CA(self, dt):
        """CA 转移矩阵"""
        F = np.eye(9)
        dt2 = 0.5 * dt ** 2
        F[0, 1] = dt;
        F[0, 2] = dt2;
        F[1, 2] = dt
        F[3, 4] = dt;
        F[3, 5] = dt2;
        F[4, 5] = dt
        F[6, 7] = dt;
        F[6, 8] = dt2;
        F[7, 8] = dt
        return F

    def get_Q(self, dt, model_idx):
        if model_idx == 0:
            return self._get_Q_CV(dt, self.q_params['cv_std'])
        elif model_idx == 1:
            return self._get_Q_CA(dt, self.q_params['ca_low_std'])
        elif model_idx == 2:
            return self._get_Q_CA(dt, self.q_params['ca_high_std'])
        return np.eye(9)

    def interact(self):
        EPS = 1e-50  # 稍微增大 EPS 防止极端数值
        c_bar = np.dot(self.trans_prob.T, self.model_probs)
        mixing_probs = (self.trans_prob * self.model_probs[:, None]) / (c_bar + EPS)

        x_mixed = np.zeros((self.M, self.dim))
        P_mixed = np.zeros((self.M, self.dim, self.dim))

        for j in range(self.M):
            for i in range(self.M):
                w = mixing_probs[i, j]
                x_mixed[j] += w * self.x[i]

            for i in range(self.M):
                w = mixing_probs[i, j]
                diff = (self.x[i] - x_mixed[j]).reshape(-1, 1)
                P_mixed[j] += w * (self.P[i] + diff @ diff.T)

        return x_mixed, P_mixed, c_bar

    def update(self, z, dt):
        models = [
            {'F': self.get_F_CV(dt), 'Q': self.get_Q(dt, 0)},
            {'F': self.get_F_CA(dt), 'Q': self.get_Q(dt, 1)},
            {'F': self.get_F_CA(dt), 'Q': self.get_Q(dt, 2)}
        ]

        x_mixed, P_mixed, c_bar = self.interact()
        likelihoods = np.zeros(self.M)
        EPS = 1e-50

        for i in range(self.M):
            F = models[i]['F']
            Q = models[i]['Q']

            x_pred = F @ x_mixed[i]
            P_pred = F @ P_mixed[i] @ F.T + Q

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
            self.P[i] = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

            S_det = np.linalg.det(S)
            denom = np.sqrt(((2 * np.pi) ** 3) * S_det)
            mahalanobis = -0.5 * (y.T @ S_inv @ y)

            # 使用 clip 防止数值溢出，保留相对大小
            likelihoods[i] = np.exp(np.clip(mahalanobis, -100, 100)) / (denom + EPS)

        new_probs = likelihoods * c_bar
        sum_probs = np.sum(new_probs)

        if sum_probs < EPS:
            self.model_probs = c_bar
        else:
            self.model_probs = new_probs / sum_probs

        x_out = np.zeros(self.dim)
        for i in range(self.M):
            x_out += self.model_probs[i] * self.x[i]

        return x_out, self.model_probs