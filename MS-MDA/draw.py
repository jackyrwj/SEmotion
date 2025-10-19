# import matplotlib.pyplot as plt
# import numpy as np


# # 用来正常显示负号
# plt.rcParams['axes.unicode_minus']=False

# # 指定分组个数
# n_bins=10

# fig,ax=plt.subplots(figsize=(8,5))

# plt.grid(axis='y', linestyle='-', alpha=0.4 ,zorder=0)

# # 分别生成10000 ， 5000 ， 2000 个值
# x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]


# # 实际绘图代码与单类型直方图差异不大，只是增加了一个图例项
# # 在 ax.hist 函数中先指定图例 label 名称
# ax.hist(x_multi, n_bins, histtype='bar',label=(12,45,67),zorder=100)


# ax.set_title('多类型直方图')

# # 通过 ax.legend 函数来添加图例
# ax.legend()

# plt.show()
# plt.savefig('条形图.png')

# import numpy as np
# import matplotlib.pyplot as plt

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)      

# # 生成3组值，每组的个数可以不一样
# x1,x2,x3 = [np.random.randn(n) for n in [10000, 5000, 2000]]

# plt.figure(figsize=(8,5))
# # 在 ax.hist 函数中先指定图例 label 名称
# plt.hist([x1, x2, x3], bins=3, density=True, histtype='bar')

# # 通过 ax.legend 函数来添加图例
# plt.legend(list("ABC"))

# plt.rcParams['font.family']='SimHei'
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['font.size']='12'
# plt.title("多类型直方图")
# #songTi = matplotlib.font_manager.FontProperties(fname='C:\\Windows\\Fonts\\msyh.ttc')
# #plt.title("多类型直方图",fontproperties=songTi,fontsize=12)
# plt.show()


# plt.savefig('条形图.png')


# # 定义数据点和标签
# data = [85.29, 87.68, 87.58, 87.86, 87.68, 86.28, 78.63, 76.63, 86.26, 75.69, 90.67, 86.56, 89.22, 86.55, 84.02]
# labels = ['csub_256', 'csub_128', 'csub_64', 'csub_32', 'csub_16', 'MS-MDA', 'MS-MDA', 'MS-MDA', 'MS-MDA', 'MS-MDA', 'cosface', 'cosface', 'cosface', 'cosface', 'cosface']

# # 创建条形图并设置颜色
# plt.bar(range(len(data)), data, tick_label=labels, color='blue')
# plt.xticks(range(len(data)), rotation=45)

# # 添加直线背景
# plt.grid(axis='y', linestyle='-', alpha=0.5)

# # 添加标签和标题
# plt.xlabel('模型大小')
# plt.ylabel('准确率')
# plt.title('模型在不同大小和数据集上的准确率')

# # 展示图表
# plt.show()
# plt.savefig('条形图.png')




# ----------------------------------------------------
#---------------------------------------------------------------
# import matplotlib.pyplot as plt

# # 用来正常显示负号
# plt.rcParams['axes.unicode_minus'] = False

# # 指定分组个数
# n_bins = 4

# # 数据整理为二维数组
# data = [[85.29, 87.68, 87.58, 87.86],
#         [87.68, 86.28, 78.63, 76.63],
#         [86.26, 75.69, 90.67, 86.56],
#         [89.22, 86.55, 84.02, 85.02]]

# fig, ax = plt.subplots(figsize=(8, 5))

# plt.grid(axis='y', linestyle='-', alpha=0.4, zorder=0)

# # 指定颜色
# colors = ['blue', 'red', 'green', 'orange']

# # 实际绘图代码与单类型直方图差异不大，只是增加了一个图例项
# # 在 ax.hist 函数中先指定图例 label 名称和颜色
# for i in range(n_bins):
#     ax.hist(data[i], n_bins, histtype='bar', label='类别{}'.format(i+1), color=colors[i], zorder=100)

# ax.set_title('多类型直方图')

# # 通过 ax.legend 函数来添加图例
# ax.legend()

# plt.show()
# plt.savefig('条形图.png')

#---------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np






# # 假设的数据和误差条
batch_sizes = [16, 32, 64, 128, 256]
CSESN_S3_means = [91.87, 92.2, 90.83, 89.03, 88.56]
CSESN_S3_std = [5, 7, 6, 8, 4]

CSUB_S3_means = [91.55, 93.02, 91.22, 90.56, 90.67]
CSUB_S3_std = [6, 5, 8,7, 4]

CSESN_S4_means = [70.27, 72.2, 68.96, 65.7, 66.11]
CSESN_S4_std = [ 8, 7, 5,6, 5]

CSUB_S4_means = [65.45,70.85, 62.6, 60.98, 63.6]
CSUB_S4_std = [4, 8, 5,6, 7]


# batch_sizes = [0.75, 0.8, 0.85, 0.9, 0.95]
# CSESN_S3_means = [86.83, 87.2, 91.33, 89.03, 88.02]
# CSESN_S3_std = [5, 4, 4, 3, 3]

# CSUB_S3_means = [84.02, 86.55, 90.67, 86.56, 89.22]
# CSUB_S3_std = [6, 5, 5, 4, 4]

# CSESN_S4_means = [65.7, 73.27, 65.7, 75.2, 64.11]
# CSESN_S4_std = [4, 4, 3, 3, 2]

# CSUB_S4_means = [63.45, 62.6, 58.34, 70.85, 62.6]
# CSUB_S4_std = [5, 5, 4, 4, 3]



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
# ax.set_yticks(np.arange(10, 101, 10),fontsize=100)
ax.set_yticklabels(np.arange(10, 101, 10), fontsize = 20)

# ax.yaxis.grid(True)  # 添加Y轴网格线

# 仅移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 移除边框
# for spine in ax.spines.values():
#     spine.set_visible(False)

# 添加一些文本用于标签，标题和自定义x轴刻度标签等
# ax.set_xlabel('Batch size (Epoch=250)',fontsize=20)
# ax.set_xlabel('a(b=1.3)',fontsize=20)
ax.set_xlabel('$\\it{a}$ ($\\it{b}$=1.3)',fontsize=20)
ax.set_ylabel('Accuracy',fontsize=20)
# ax.set_title('Accuracy with different')
ax.set_xticks(index)
ax.set_xticklabels(batch_sizes,fontsize=20)
ax.legend(loc='lower left',fontsize=16)

# 设置图例位置为左下角
# ax.legend(loc='lower left', zorder=100)

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
# plt.savefig('条形at.png')
# plt.savefig('at.png')
plt.savefig('tmp.png')