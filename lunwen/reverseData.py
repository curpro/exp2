import pandas as pd
# 读取CSV文件
# 注意：如果文件没有表头，需要添加header=None
df = pd.read_csv('/dataSet/csv/1310Vertical turn descetn.csv')  # 替换为您的文件路径

# 打印原始列名，确认数据结构
print("原始列名:")
print(df.columns.tolist())
print(f"原始数据形状: {df.shape}")

# 定义8个目标列的名称
target_columns = ['经度', '纬度', '高度', '滚转', '俯仰', '航向', '滚转率', '转弯率']

# 初始化一个空的DataFrame来存储结果
result_data = []

# 遍历每一行数据
for idx, row in df.iterrows():
    # 跳过标签列（第一列）
    row_data = row.iloc[1:].values  # 获取除标签外的所有数据

    # 将数据重塑为多个8列的块
    # 计算可以分成多少个完整的8列块
    num_blocks = len(row_data) // 8

    # 处理每个完整的8列块
    for i in range(num_blocks):
        block = row_data[i * 8:(i + 1) * 8]
        result_data.append(block)

    # 检查是否有剩余的不完整数据（最后一个块只有4列）
    remaining = len(row_data) % 8
    if remaining > 0:
        print(111)
        # 如果有剩余数据，但不足8列，可以根据需要处理
        # 这里我们丢弃不足8列的数据
        pass

# 将结果转换为DataFrame
result_df = pd.DataFrame(result_data, columns=target_columns)

# 打印结果
print("\n处理后的列名:")
print(result_df.columns.tolist())
print(f"处理后的数据形状: {result_df.shape}")
print("\n前5行数据:")
print(result_df.head())

# 保存到新文件
result_df.to_csv('D:\AFS\lunwen\dataSet\handleCsv/1310Vertical turn descetn.csv', index=False, encoding='utf-8-sig')
print("\n数据已保存到 processed_data.csv")