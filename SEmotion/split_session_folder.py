import os
import shutil

source_folder = '/home/raowj/Data/MS-MDA/ExtractedFeatures'
destination_folder = '/home/raowj/Data/MS-MDA/SeparatedFeatures'

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.mat'):
        # 解析文件名，获取被试编号和日期
        parts = filename.split('_')
        subject = parts[0]
        date = parts[1].split('.')[0]
        
        # 构建目标文件夹的路径
        destination_path = os.path.join(destination_folder, date)
        
        # 如果目标文件夹不存在，则创建它
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        # 将文件移动到目标文件夹中
        source_path = os.path.join(source_folder, filename)
        shutil.move(source_path, destination_path)