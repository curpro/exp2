import pandas as pd
import plotly.graph_objects as go
import os


def plot_interactive_HD(file_path):
    if not os.path.exists(file_path):
        # 尝试在当前目录下查找
        local_path = os.path.basename(file_path)
        if os.path.exists(local_path):
            file_path = local_path
        else:
            print(f"错误：找不到文件 {file_path}")
            return

    print(f"读取数据中: {file_path}")
    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # 核心优化：创建更清晰的 3D 轨迹
    # ---------------------------------------------------------
    fig = go.Figure()

    # 1. 绘制主轨迹线
    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='lines',
        name='飞行轨迹',
        line=dict(
            color=df['z'],  # 根据高度上色
            colorscale='Turbo',  # 使用高对比度色谱
            width=6,  # 线条加粗
            showscale=True,
            colorbar=dict(title='高度 (m)', x=0.85)
        ),
        hovertemplate='<b>X</b>: %{x:.1f}m<br><b>Y</b>: %{y:.1f}m<br><b>Z</b>: %{z:.1f}m<extra></extra>'
    ))

    # 2. 绘制“地面投影” (增强立体感)
    z_min = df['z'].min()
    # 为了防止投影与轨迹太远或太近，设置投影面为最低点下方 10% 范围
    offset = (df['z'].max() - z_min) * 0.1
    proj_z = z_min - (offset if offset > 10 else 10)

    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=[proj_z] * len(df),
        mode='lines',
        name='地面投影',
        line=dict(color='gray', width=3),
        opacity=0.3,
        hoverinfo='skip'
    ))

    # 3. 标记起点
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[0]], y=[df['y'].iloc[0]], z=[df['z'].iloc[0]],
        mode='markers+text',
        name='起点',
        marker=dict(size=10, color='#00FF00', symbol='diamond'),
        text=["START"],
        textposition="top center",
        textfont=dict(color='#00FF00', size=12, family="Arial Black")
    ))

    # 4. 标记终点
    fig.add_trace(go.Scatter3d(
        x=[df['x'].iloc[-1]], y=[df['y'].iloc[-1]], z=[df['z'].iloc[-1]],
        mode='markers+text',
        name='终点',
        marker=dict(size=10, color='#FF0000', symbol='x'),
        text=["END"],
        textposition="top center",
        textfont=dict(color='#FF0000', size=12, family="Arial Black")
    ))

    # ---------------------------------------------------------
    # 布局优化
    # ---------------------------------------------------------
    fig.update_layout(
        title={
            'text': f"F-16 机动轨迹可视化 (Data: {os.path.basename(file_path)})",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color='white')
        },
        template='plotly_dark',
        width=1200,
        height=900,
        margin=dict(r=0, l=0, b=0, t=50),
        scene=dict(
            xaxis=dict(title='East (X)', gridcolor='gray', showbackground=True, backgroundcolor='black'),
            yaxis=dict(title='North (Y)', gridcolor='gray', showbackground=True, backgroundcolor='black'),
            zaxis=dict(title='Altitude (Z)', gridcolor='gray', showbackground=True, backgroundcolor='black'),
            aspectmode='data',  # 强制等比例
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=0.6),  # 稍微调低视角，看起来更壮观
                center=dict(x=0, y=0, z=-0.1)
            )
        )
    )

    # 输出文件名为 _a_3d_HD.html
    output_html = file_path.replace('.csv', '_3d_HD.html')
    # 如果路径中包含目录，确保写入权限，或者直接写到当前目录
    if '\\' in output_html or '/' in output_html:
        output_html = os.path.basename(output_html)

    fig.write_html(output_html)
    print(f"✅ 高清图表已生成: {output_html}")

    try:
        fig.show()
    except:
        pass


if __name__ == "__main__":
    # --- 修改此处：适配你的新数据集路径 ---
    csv_path = r'D:\AFS\lunwen\dataSet\processed_data\f16_super_maneuver_a.csv'

    # 本地测试备用:
    # csv_path = 'f16_super_maneuver_a.csv'

    plot_interactive_HD(csv_path)