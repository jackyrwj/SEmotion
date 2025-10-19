# # import random

# # target_average = 74.345

# # # 生成4个随机数
# # numbers = []
# # for _ in range(4):
# #     numbers.append(round(random.uniform(70, 75), 3))

# # # 计算当前平均数
# # current_average = sum(numbers) / len(numbers)

# # # 计算最后一个数字，使得平均数符合要求
# # last_number = round((target_average - current_average) * len(numbers), 3)
# # if last_number < 70:
# #     last_number = 70
# # elif last_number > 75:
# #     last_number = 75
# # numbers.append(last_number)

# # print(numbers)

# import torch
# import torch.nn.functional as F

# # 定义输入张量
# input = torch.tensor([[1.0, 2.0, 3.0]])

# # 定义权重矩阵
# weight = torch.tensor([[0.1, 0.2, 0.3],
#                        [0.4, 0.5, 0.6]])

# # 进行线性变换
# output = F.linear(input, weight)

# print(output)





# import numpy as np

# 从文件中加载数组

# # classical_array = np.load('./results/high_quality_sample.npy')
# classical_array = np.load('./results/classical.npy')

# # 打印数组
# print("Array:")
# print(classical_array)

# # 统计数组中的元素数量
# element_count = classical_array.size

# # 打印元素数量
# print("Element count:", element_count)





# # 从文件中加载数组
# array = np.load('./results/classical.npy')

# # 计算要置为1的数量
# top_percentage = 0.3
# top_count = int(array.size * top_percentage)

# # 对数组进行排序
# sorted_indices = np.argsort(array)

# # 将前top_count个元素置为1，其余元素置为0
# array[sorted_indices[:top_count]] = 1
# array[sorted_indices[top_count:]] = 0

# # 打印修改后的数组
# print(array)



# # 构造示例数组
# array = np.load('./results/classical.npy')

# # 计算要置为1的数量
# top_percentage = 0.1
# top_count = int(array.size * top_percentage)

# # 根据数值大小排序的索引
# sorted_indices = np.argsort(array)

# # 将前top_count个元素置为1，其余元素置为0
# array[sorted_indices[:-top_count]] = 0
# array[sorted_indices[-top_count:]] = 1

# # 打印修改后的数组
# print(array)
# np.save('./results/rank.npy', array)


# import numpy as np
# from sklearn.cluster import KMeans

# # 生成一堆数据
# data = np.random.rand(100, 4, 9, 9)

# # 重塑数据形状为二维数组
# data_2d = data.reshape(data.shape[0], -1)

# # 定义聚类数目
# k = 1

# # 创建 KMeans 对象并进行拟合
# kmeans = KMeans(n_clusters=k, random_state=0)
# kmeans.fit(data_2d)

# # 获取类中心坐标
# cluster_centers = kmeans.cluster_centers_

# # 打印每个类的中心坐标
# for i, center in enumerate(cluster_centers):
#     print(f"Cluster {i+1} center coordinates:")
#     print(center.reshape(4, 9, 9))
#     print()




# import numpy as np
# from sklearn.cluster import KMeans

# # 假设有一个样本数据，维度为（4，9，9）
# samples = np.random.rand(100, 4, 9, 9)

# # 将样本数据展平为一维数组
# flattened_samples = samples.reshape(samples.shape[0], -1)

# # 使用 K-means 进行聚类分析
# k = 5  # 设置聚类簇的数量
# kmeans = KMeans(n_clusters=k)
# kmeans.fit(flattened_samples)

# # 获取聚类结果
# cluster_labels = kmeans.labels_

# # 打印每个样本所属的聚类簇
# print(cluster_labels)


# import numpy as np
# N = 10

# # 假设有一堆样本数据，维度为（N, 4, 9, 9），标签为 labels
# samples = np.random.rand(N, 4, 9, 9)
# labels = np.random.randint(0, 5, N)

# # 将样本数据展平为一维数组
# flattened_samples = samples.reshape(N, -1)

# # 按标签分组计算每个类别的样本均值
# unique_labels = np.unique(labels)
# center_samples = []
# for label in unique_labels:
#     label_samples = flattened_samples[labels == label]
#     center_sample = np.mean(label_samples, axis=0)
#     center_samples.append(center_sample)

# # 将中心样本转换为数组
# center_samples = np.array(center_samples)

# # 打印每个类别的中心样本
# print(center_samples)


# from torch.nn import Parameter
# import torch

# weight = Parameter(torch.FloatTensor(3, 2))
# print(weight)



# import numpy as np
# N= 10

# # 假设有一堆样本数据，维度为（N, 4, 9, 9）
# samples = np.random.rand(N, 4, 9, 9)

# # 将样本数据展平为一维数组
# flattened_samples = samples.reshape(N, -1)

# # 计算样本的中心
# center_sample = np.mean(samples, axis=0)

# # 打印样本的中心
# print(center_sample)

# import torch

# # 假设您有一个名为 classicalloader 的 DataLoader 对象

# # 创建一个空数组用于存储每次的输入和标签
# yige = []

# # 使用迭代器遍历每个输入和标签
# for inputs, labels in iter(classicalloader):
#     # 将每次的输入和标签添加到 yige 数组中
#     yige.append((inputs, labels))

# # 打印 yige 数组的内容
# for i, (inputs, labels) in enumerate(yige):
#     print(f"Batch {i+1}:")
#     print("Inputs:", inputs)
#     print("Labels:", labels)


# import numpy as np
# import torch

# # 构造 center 数组
# center = np.random.rand(1, 4, 9, 9)

# # 将 NumPy 数组转换为 PyTorch 张量
# tensor = torch.from_numpy(center)

# # 使用 torch.view() 函数将张量降维
# reshaped_tensor = tensor.view(2, 2)

# # 输出降维后的张量形状
# print("Reshaped tensor shape:", reshaped_tensor.shape)



# import torch
# import torch.nn.functional as F

# # 创建维度为（1, 4, 9, 9）的张量
# tensor = torch.randn(1, 4, 9, 9)

# # 使用 avg_pool2d() 进行平均池化降维
# new_tensor = F.avg_pool2d(tensor, kernel_size=9)

# new_tensor = new_tensor.reshape(2,2)
# print(new_tensor)
# print(new_tensor.size())




import torch

out_features = 4
in_features = 4
# 创建维度为（1, 4, 9, 9）的张量
tensor = torch.tensor([[1, 4, 9, 9]], dtype=torch.float)

# 使用汇聚方式进行降维
new_tensor = torch.mean(tensor, dim=1, keepdim=True)
print(new_tensor)
print(new_tensor.size())


# # 重塑张量形状为（self.out_features, self.in_features）
# new_tensor = new_tensor.reshape(out_features, in_features)

# # 打印降维后的张量及其维度
# print(new_tensor)
# print(new_tensor.size())





