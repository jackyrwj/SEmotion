
import matplotlib.pyplot as plt
import numpy as np






# 假设的数据和误差条
# batch_sizes = [100, 150, 200, 250, 300]
# CSESN_S3_means = [85.78, 86.19, 88.02, 88.56, 87.53]
# CSESN_S3_std = [5, 4, 4, 3, 3]

# CSUB_S3_means = [86.64, 87.74, 89.67, 90.67, 89.66]
# CSUB_S3_std = [6, 5, 5, 4, 4]

# CSESN_S4_means = [59.26, 60.26, 61.11, 64.11, 62.85]
# CSESN_S4_std = [7, 5, 6, 8, 9]

# CSUB_S4_means = [55.71, 57.43, 58.6, 62.6, 61.39]
# CSUB_S4_std = [9.21, 9.76, 8.08, 9.86, 6.25]


batch_sizes = [1.15, 1.2, 1.25, 1.3, 1.35]
CSESN_S3_means = [87.78, 89.53, 88.02, 87.56, 87.19]
CSESN_S3_std = [5, 4, 4, 3, 3]

CSUB_S3_means = [87.64, 90.67, 85.74, 85.61, 88.66]
CSUB_S3_std = [6, 5, 5, 4, 4]

CSESN_S4_means = [60.26, 61.26, 61.6, 64.11, 59.85]
CSESN_S4_std = [10.07, 10.6, 13.45, 11.08, 12.8]

CSUB_S4_means = [45.71, 60.43, 59.85, 62.6, 58.39]
CSUB_S4_std = [9.21, 9.76, 8.08, 9.86, 9.25]



# 设置条形图的宽度和位置
# bar_width = 0.22
# index = np.arange(len(batch_sizes))



# 绘制条形图
fig, ax = plt.subplots()

# 调整条形图的宽度以加粗柱子
bar_width = 0.25  # 可以调整柱子宽度以改变空隙大小
index = np.arange(len(batch_sizes)) * 1.3  # 乘以一个因子以增加batch_size之间的空隙

# 绘制条形图时设置图形尺寸
fig, ax = plt.subplots(figsize=(10, 6))  # 增加图形的宽度

plt.grid(axis='y', linestyle='-', alpha=0.4, zorder=-100)
# 指定柱子的颜色和加粗边框
bar_colors = ['#555b6e', '#89b0ae', '#bee3db', '#d0c8c8']  # 每组柱子的颜色
# 自定义误差条的样式
error_attr = {'elinewidth': 1, 'capsize': 3, 'capthick': 0.5}

rects1 = ax.bar(index - bar_width*1.5, CSESN_S3_means, bar_width, yerr=CSESN_S3_std,
                color=bar_colors[0], label='KFCT_SEED', error_kw=error_attr)
rects2 = ax.bar(index - bar_width/2, CSUB_S3_means, bar_width, yerr=CSUB_S3_std,
                color=bar_colors[1], label='LOSO_SEED', error_kw=error_attr)
rects3 = ax.bar(index + bar_width/2, CSESN_S4_means, bar_width, yerr=CSESN_S4_std,
                color=bar_colors[2], label='KFCT_SEED-IV', error_kw=error_attr)
rects4 = ax.bar(index + bar_width*1.5, CSUB_S4_means, bar_width, yerr=CSUB_S4_std,
                color=bar_colors[3], label='LOSO_SEED-IV', error_kw=error_attr)

# 定义y轴的网格线间隔
# ax.set_yticks(np.arange(10, 101, 10))
# ax.yaxis.grid(True)  # 添加y轴网格线

# 设置Y轴网格线，并从10开始
ax.set_ylim([10, 100])  # 设置Y轴的刻度从10开始
# ax.set_yticks(np.arange(10, 101, 10))
ax.set_yticklabels(np.arange(10, 101, 10), fontsize = 20)
# ax.yaxis.grid(True)  # 添加Y轴网格线

# 仅移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 移除边框
# for spine in ax.spines.values():
#     spine.set_visible(False)

# 添加一些文本用于标签，标题和自定义x轴刻度标签等
# ax.set_xlabel('Epoch (Batch size=32)',fontsize=20)
ax.set_xlabel('$\\it{b}$ ($\\it{a}$=0.85)',fontsize=20)
ax.set_ylabel('Accuracy',fontsize=20)
# ax.set_title('Accuracy with different')
ax.set_xticks(index)
ax.set_xticklabels(batch_sizes,fontsize = 20)
# ax.legend()
# 设置图例位置为左下角
ax.legend(loc='lower left',fontsize=16)

# 添加误差条
# def autolabel(rects):
#     """在每个条形上方附上一个文本标签，显示其高度。"""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)

# 显示图形
plt.tight_layout()
plt.show()
# plt.savefig('epoch.png')
plt.savefig('bt.png')




# import random
# # 指定范围
# min_value = 7.0
# max_value = 10.0

# # 生成随机数组
# random_array = [round(random.uniform(min_value, max_value), 2) for _ in range(5)]

# # 打印数组
# print(random_array)