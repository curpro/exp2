import pandas as pd
import numpy as np
from scipy import interpolate
import os


def convert_single_file(input_path, output_path):
    print(f"正在处理文件: {input_path}")

    # 1. 读取数据
    try:
        # 尝试不同编码读取
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='gbk')
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 2. 字段映射 (适配中文列名)
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
        print("错误: 无法识别经纬度列")
        return

    if raw_alt is None:
        print("警告: 未找到高度列，默认填充 5000m")
        raw_alt = np.full_like(raw_lat, 5000.0)

    # 3. 坐标系转换 (以第一个点为原点)
    lat0, lon0, alt0 = raw_lat[0], raw_lon[0], raw_alt[0]
    print(f"  -> 原点设定 (0,0,0): Lat={lat0}, Lon={lon0}, Alt={alt0}")

    R = 6378137.0  # 地球半径
    d_lat = np.deg2rad(raw_lat - lat0)
    d_lon = np.deg2rad(raw_lon - lon0)
    mean_lat = np.deg2rad(lat0)

    x_raw = d_lon * R * np.cos(mean_lat)
    y_raw = d_lat * R
    z_raw = raw_alt - alt0

    # 4. 重采样与平滑计算
    # 设定固定时间间隔 0.05s (20Hz)
    FIXED_DT = 0.05
    raw_time_axis = np.arange(len(x_raw)) * FIXED_DT

    # 样条插值 (s=0 保证经过所有原始点)
    tck_x = interpolate.splrep(raw_time_axis, x_raw, s=0)
    tck_y = interpolate.splrep(raw_time_axis, y_raw, s=0)
    tck_z = interpolate.splrep(raw_time_axis, z_raw, s=0)

    # 计算位置
    x_out = interpolate.splev(raw_time_axis, tck_x)
    y_out = interpolate.splev(raw_time_axis, tck_y)
    z_out = interpolate.splev(raw_time_axis, tck_z)

    # 计算速度 (导数)
    vx = interpolate.splev(raw_time_axis, tck_x, der=1)
    vy = interpolate.splev(raw_time_axis, tck_y, der=1)
    vz = interpolate.splev(raw_time_axis, tck_z, der=1)

    # 5. 保存结果
    # 只需要 x, vx, y, vy, z, vz 这6列
    df_out = pd.DataFrame({
        'x': x_out, 'vx': vx,
        'y': y_out, 'vy': vy,
        'z': z_out, 'vz': vz
    })

    # 格式化保存，保留4位小数
    df_out.to_csv(output_path, index=False, float_format='%.4f')
    print(f"  -> 转换完成！已保存至: {output_path}")

    # 简单验证
    v_avg = np.mean(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)) * 3.6
    print(f"  -> 平均速度验证: {v_avg:.0f} km/h")


if __name__ == "__main__":
    # 输入文件路径 (确保文件在当前目录下，或者修改为绝对路径)
    input_csv = r'D:\AFS\lunwen\dataSet\handleCsv\04Turn right.csv'
    output_csv = r'D:\AFS\lunwen\dataSet\processed_data\04Turn right_processed.csv'

    if os.path.exists(input_csv):
        convert_single_file(input_csv, output_csv)
    else:
        print(f"找不到文件: {input_csv}，请确认路径正确。")