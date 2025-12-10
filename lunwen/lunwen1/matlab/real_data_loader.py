import pandas as pd
import numpy as np
from scipy import interpolate
import os


def load_real_data_as_truth(filename, ref_origin=None):
    """
    读取真实飞机数据。
    【强制模式】固定时间间隔 dt = 0.05s (20Hz)。
    """
    # 1. 读取数据
    try:
        try:
            df = pd.read_csv(filename, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filename, encoding='gbk')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None, None

    # 2. 字段匹配
    col_map = {
        'lat': ['纬度(°)', '纬度', 'lat', 'Latitude'],
        'lon': ['经度(°)', '经度', 'lon', 'Longitude'],
        'alt': ['高度(m)', '高度', 'alt', 'Altitude']
    }

    def get_col(target_key):
        candidates = col_map[target_key]
        for c in candidates:
            for df_col in df.columns:
                if c in df_col.strip():
                    return df[df_col].values
        return None

    raw_lat = get_col('lat')
    raw_lon = get_col('lon')
    raw_alt = get_col('alt')

    if raw_lat is None or raw_lon is None:
        print(f"错误: 无法识别经纬度列 {filename}")
        return None, None, None

    if raw_alt is None:
        raw_alt = np.full_like(raw_lat, 5000.0)

    # 3. 坐标系转换
    if ref_origin is not None:
        lat0, lon0, alt0 = ref_origin
    else:
        lat0, lon0, alt0 = raw_lat[0], raw_lon[0], raw_alt[0]
        print(f"  -> [原点锁定] Lat={lat0:.6f}, Lon={lon0:.6f}, Alt={alt0:.1f}")

    R = 6378137.0
    d_lat = np.deg2rad(raw_lat - lat0)
    d_lon = np.deg2rad(raw_lon - lon0)
    mean_lat = np.deg2rad(lat0)

    x_raw = d_lon * R * np.cos(mean_lat)
    y_raw = d_lat * R
    z_raw = raw_alt - alt0

    # 4. === 强制时间设定 (已修改) ===
    FIXED_DT = 0.05  # <--- 这里改成了 0.05s (20Hz)
    raw_time_axis = np.arange(len(x_raw)) * FIXED_DT

    # 5. 计算速度 (使用样条插值求导，保证平滑)
    tck_x = interpolate.splrep(raw_time_axis, x_raw, s=0)
    tck_y = interpolate.splrep(raw_time_axis, y_raw, s=0)
    tck_z = interpolate.splrep(raw_time_axis, z_raw, s=0)

    # 获取位置
    x_out = interpolate.splev(raw_time_axis, tck_x)
    y_out = interpolate.splev(raw_time_axis, tck_y)
    z_out = interpolate.splev(raw_time_axis, tck_z)

    # 求导得到速度 (vx, vy, vz)
    vx = interpolate.splev(raw_time_axis, tck_x, der=1)
    vy = interpolate.splev(raw_time_axis, tck_y, der=1)
    vz = interpolate.splev(raw_time_axis, tck_z, der=1)

    Xgt_out = np.vstack((x_out, vx, y_out, vy, z_out, vz))

    return Xgt_out, raw_time_axis, (lat0, lon0, alt0), FIXED_DT


if __name__ == "__main__":
    # === 配置 ===
    input_dir = r'/dataSet/handleCsv'
    output_dir = r'/dataSet/processed_data'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    file_list = [
        '01up.csv', '02Level Flight.csv', '03Descent.csv',
        '04Turn right.csv', '05Turn left.csv', '06Turn right up.csv',
        '07Turn right descent.csv', '08Turn left up.csv', '09Turn left descent.csv',
        '10Vertical turn up.csv', '11Roll right.csv', '12Roll left.csv',
        '1310Vertical turn descetn.csv'
    ]

    global_origin = None
    all_dfs = []
    current_time_offset = 0.0

    print("=== 开始批量处理 (强制 dt=0.05s) ===")

    for fname in file_list:
        fpath = os.path.join(input_dir, fname)
        if not os.path.exists(fpath): continue

        print(f"\n处理: {fname}")

        state, t_local, origin, dt = load_real_data_as_truth(
            fpath,
            ref_origin=global_origin
        )

        if state is not None:
            if global_origin is None: global_origin = origin

            # 保存单文件
            df = pd.DataFrame(state.T, columns=['x', 'vx', 'y', 'vy', 'z', 'vz'])
            out_path = os.path.join(output_dir, fname.replace('.csv', '_processed.csv'))
            df.to_csv(out_path, index=False, float_format='%.4f')

            # 准备合并
            continuous_time = t_local + current_time_offset
            df_merge = df.copy()
            # df_merge.insert(0, 'time', continuous_time) # 需要时间列可取消注释
            all_dfs.append(df_merge)

            # 更新下一段的起始时间
            current_time_offset = continuous_time[-1] + dt

            # 验证速度
            v_scalar = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2 + df['vz'] ** 2) * 3.6
            print(f"  -> 速度范围: {v_scalar.min():.0f} - {v_scalar.max():.0f} km/h")

    if all_dfs:
        print("\n正在合并...")
        merged = pd.concat(all_dfs, ignore_index=True)
        out_merge = os.path.join(output_dir, 'merged_trajectory.csv')
        merged.to_csv(out_merge, index=False, float_format='%.4f')
        print(f"全部完成！合并文件已生成: {out_merge}")
        print(f"总数据行数: {len(merged)}")