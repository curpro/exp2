import os
import csv

def convert_csv_format(input_folder, output_folder):
    """
    转换CSV文件格式从多列格式到标准格式
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"处理文件: {filename}")
            
            # 读取并转换数据
            with open(input_path, 'r') as infile:
                # 读取所有行
                lines = infile.readlines()
                
            # 写入新格式的文件
            with open(output_path, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                
                # 写入标题行
                writer.writerow(['经度', '纬度', '高度', '滚转', '俯仰', '航向', '滚转率', '转弯率'])
                
                # 处理每一行数据
                for line in lines:
                    values = line.strip().split(',')
                    
                    # 跳过可能的空行或标题行
                    if len(values) < 8 or values[0] == '经度':
                        continue
                    
                    # 每8个值构成一个完整的数据记录
                    num_records = len(values) // 8
                    
                    for i in range(num_records):
                        start_index = i * 8
                        record = values[start_index:start_index + 8]
                        
                        # 确保我们有正确的数值数量
                        if len(record) == 8:
                            try:
                                # 转换为浮点数并写入
                                float_record = [float(v) for v in record]
                                writer.writerow(float_record)
                            except ValueError:
                                # 如果转换失败，跳过该记录
                                print(f"警告：无法转换记录 {record}")
                                continue
            
            print(f"完成处理: {filename}")

if __name__ == "__main__":
    input_folder = r"D:\AFS\lunwen\dataSet\csv"
    output_folder = r"D:\AFS\lunwen\dataSet\handleCsv"
    
    convert_csv_format(input_folder, output_folder)
    print("所有文件处理完成!")