import numpy as np


def generate_truth_data():
    """
    生成三维轨迹的真实状态 (9维: x, vx, ax, y, vy, ay, z, vz, az)
    """
    np.random.seed(2019)

    # 参数定义
    vx = 10.0  # m/s
    omegaA = np.deg2rad(5)
    omegaB = np.deg2rad(6)
    omegaC = np.deg2rad(6)
    acc = 5.0
    accD = -3.0  # m/s^2
    dt = 0.1
    num_steps = 4000
    tt = np.arange(0, num_steps * dt, dt)

    # 9维状态: [x, vx, ax, y, vy, ay, z, vz, az]
    Xgt = np.zeros((9, len(tt)))

    # 初始状态
    Xgt[1, 0] = vx  # vx
    Xgt[4, 0] = 0  # vy
    Xgt[7, 0] = 0  # vz

    # 定义各段结束索引
    seg1 = 400;
    seg2 = 800;
    seg3 = 1200;
    seg4 = 1600;
    seg5 = 2000
    seg6 = 2400;
    seg7 = 2800;
    seg8 = 3200;
    seg9 = 3600;
    seg10 = 4000

    attitude = np.zeros(len(tt))

    def update_linear(m, ax_val, ay_val, az_val):
        """线性运动更新"""
        Xgt[:, m] = Xgt[:, m - 1]  # Copy prev
        # Update Vel
        Xgt[1, m] += ax_val * dt
        Xgt[4, m] += ay_val * dt
        Xgt[7, m] += az_val * dt
        # Update Pos
        Xgt[0, m] += Xgt[1, m] * dt
        Xgt[3, m] += Xgt[4, m] * dt
        Xgt[6, m] += Xgt[7, m] * dt
        # Record Acc
        Xgt[2, m] = ax_val
        Xgt[5, m] = ay_val
        Xgt[8, m] = az_val

    def update_turn(m, omega, t_param):
        """转弯运动更新 (包含加速度计算)"""
        X0 = Xgt[:, m - 1]
        phi = np.arctan2(X0[4], np.sqrt(X0[1] ** 2 + X0[7] ** 2))
        v_total = np.sqrt(X0[1] ** 2 + X0[4] ** 2 + X0[7] ** 2)

        # 速度 (Pos 积分略)
        vx_new = v_total * np.cos(phi) * np.cos(omega * dt * t_param)
        vy_new = v_total * np.sin(phi)  # Constant in this model
        vz_new = v_total * np.cos(phi) * np.sin(omega * dt * t_param)

        # 位置
        Xgt[0, m] = Xgt[0, m - 1] + vx_new * dt
        Xgt[3, m] = Xgt[3, m - 1] + vy_new * dt
        Xgt[6, m] = Xgt[6, m - 1] + vz_new * dt

        # 速度赋值
        Xgt[1, m] = vx_new
        Xgt[4, m] = vy_new
        Xgt[7, m] = vz_new

        # 加速度 (对速度求导)
        # ax = d(vx)/dt = -V * cos(phi) * omega * sin(omega*t)
        # az = d(vz)/dt =  V * cos(phi) * omega * cos(omega*t)
        Xgt[2, m] = -v_total * np.cos(phi) * omega * np.sin(omega * dt * t_param)
        Xgt[5, m] = 0
        Xgt[8, m] = v_total * np.cos(phi) * omega * np.cos(omega * dt * t_param)

    # --- 轨迹生成循环 ---
    for m in range(1, num_steps):
        if m < seg1:  # 匀速
            update_linear(m, 0, 0, 0)
        elif m < seg2:  # 加速
            update_linear(m, acc, acc, acc)
        elif m < seg3:  # 旋转1
            update_turn(m, omegaB, m - seg3)  # 保持原代码的相位逻辑
        elif m < seg4:  # 匀速
            update_linear(m, 0, 0, 0)
        elif m < seg5:  # 减速
            update_linear(m, accD, accD, accD)
        elif m < seg6:  # 旋转2
            update_turn(m, omegaC, m - seg3)
        elif m < seg7:  # 匀速
            update_linear(m, 0, 0, 0)
        elif m < seg8:  # 旋转3
            update_turn(m, omegaC, m - seg3)
        elif m < seg9:  # 加速
            update_linear(m, acc, acc, acc)
        else:  # 旋转4
            update_turn(m, omegaA, m - 1)

    # 返回完整的 9维 真值 (不加噪声)
    return Xgt, tt