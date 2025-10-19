# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# # 创建示例的三分类数据集
# np.random.seed(0)
# n_samples = 300
# n_features = 2
# n_classes = 3

# X = np.concatenate([np.random.randn(n_samples // n_classes, n_features) + i for i in range(n_classes)])
# y = np.concatenate([[i] * (n_samples // n_classes) for i in range(n_classes)])

# # 使用T-SNE进行降维
# tsne = TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(X)

# # 绘制T-SNE图
# plt.figure(figsize=(8, 6))
# colors = ['r', 'g', 'b']
# markers = ['o', 's', 'D']

# for i in range(n_classes):
#     plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c=colors[i], marker=markers[i], label=f'Class {i+1}', s=10)

# plt.xlabel('T-SNE Dimension 1')
# plt.ylabel('T-SNE Dimension 2')
# plt.title('T-SNE Visualization')
# plt.legend()
# plt.show()
# plt.savefig('old.png')


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.cluster import KMeans

# # 生成随机的分类数据集
# X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
#                            n_clusters_per_class=1, random_state=0.8)

# # 使用K-means进行分类
# kmeans = KMeans(n_clusters=3, random_state=0.5)
# kmeans.fit(X)
# y_pred = kmeans.labels_

# # 绘制分类结果
# plt.figure(figsize=(8, 6))

# # 绘制每个类别的数据点
# for i in range(3):
#     plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], label=f'Class {i+1}')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Classification Result')
# plt.legend()
# plt.show()
# plt.savefig('xxx.png')





#=-------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt

# # 生成随机的点数据
# num_samples = 300

# # 生成类别标签
# labels = np.random.choice([0, 1, 2], size=num_samples)

# # 生成随机点的 x 和 y 坐标，限制取值范围
# x = np.random.uniform(-0.5, 0.5, size=num_samples)
# y = np.random.uniform(-0.5, 0.5, size=num_samples)

# # 绘制分类结果
# plt.figure(figsize=(8, 6))

# # 绘制每个类别的数据点
# for i in range(3):
#     plt.scatter(x[labels == i], y[labels == i], label=f'Class {i+1}')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Random Points with Three Classes')
# plt.legend()
# plt.show()
# plt.savefig('zzz.png')
#-------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC

# # 生成随机的分类数据集
# X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
#                         #    n_clusters_per_class=1, class_sep=0.1, random_state=0)
#                            n_clusters_per_class=1, class_sep=0.5, random_state=0)

# # 创建分类器对象
# logistic_regression = LogisticRegression(multi_class='auto', solver='lbfgs')
# decision_tree = DecisionTreeClassifier()
# svm = SVC(kernel='rbf')

# # 使用逻辑回归进行训练和预测
# logistic_regression.fit(X, y)
# logistic_regression_predictions = logistic_regression.predict(X)

# # 使用决策树进行训练和预测
# decision_tree.fit(X, y)
# decision_tree_predictions = decision_tree.predict(X)

# # 使用支持向量机进行训练和预测
# svm.fit(X, y)
# svm_predictions = svm.predict(X)








# # 绘制分类结果
# # plt.figure(figsize=(18, 6))

# # 绘制原始数据的分类结果
# # plt.figure(figsize=(8, 6))
# # # plt.subplot(131)
# # for i in range(3):
# #     plt.scatter(X[y == i, 0], X[y == i, 1])
# # plt.xticks([])
# # plt.yticks([])
# # # plt.title('Decision Tree')
# # plt.tight_layout()
# # # plt.legend()
# # plt.show()
# # plt.savefig('aaa.png')



# # 绘制逻辑回归分类结果
# # plt.subplot(132)
# # plt.figure(figsize=(8, 6))
# # for i in range(3):
# #     plt.scatter(X[logistic_regression_predictions == i, 0], X[logistic_regression_predictions == i, 1])
# # plt.xticks([])
# # plt.yticks([])
# # # plt.title('Decision Tree')
# # plt.tight_layout()
# # # plt.legend()
# # plt.show()
# # plt.savefig('ddd.png')


# # 绘制决策树分类结果
# plt.subplot(133)
# plt.figure(figsize=(8, 6))
# # for i in range(3):
# #     plt.scatter(X[decision_tree_predictions == i, 0], X[decision_tree_predictions == i, 1])

# plt.scatter(X[decision_tree_predictions == 0, 0], X[decision_tree_predictions == 0, 1], label=f'negative')
# plt.scatter(X[decision_tree_predictions == 1, 0], X[decision_tree_predictions == 1, 1], label=f'positive')
# plt.scatter(X[decision_tree_predictions == 2, 0], X[decision_tree_predictions == 2, 1], label=f'neutral')
# plt.xticks([])
# plt.yticks([])
# # plt.title('Decision Tree')
# plt.tight_layout()
# # plt.legend()
# plt.legend(loc='upper left',prop={'size': 20})  # 添加图例
# plt.show()
# # plt.savefig('aaa.png')
# plt.savefig('aaa_new.png')
# # plt.savefig('tmp.png')




# # # # 绘制支持向量机分类结果
# plt.figure(figsize=(8, 6))
# # plt.subplot(133)
# # for i in range(3):
# #     plt.scatter(X[svm_predictions == i, 0], X[svm_predictions == i, 1])


# # # for i in range(3):
# # #     plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], color=colors[i], label=f'Class {i+1}')
# plt.scatter(X[svm_predictions == 0, 0], X[svm_predictions == 0, 1], label=f'negative')
# plt.scatter(X[svm_predictions == 1, 0], X[svm_predictions == 1, 1], label=f'positive')
# plt.scatter(X[svm_predictions == 2, 0], X[svm_predictions == 2, 1], label=f'neutral')

# plt.xticks([])
# plt.yticks([])
# # plt.title('Decision Tree')
# plt.tight_layout()
# # plt.legend()
# plt.legend(loc='upper left',prop={'size': 20})  # 添加图例
# # plt.savefig('bbb.png')
# plt.savefig('bbb_new.png')
# # plt.savefig('.png')
# plt.show()











#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# 生成随机的分类数据集
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
                        #    n_clusters_per_class=1, class_sep=0.5, random_state=0)
                           n_clusters_per_class=1, class_sep=0.1, random_state=0)

# 使用K-means进行分类
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_pred = kmeans.labels_

# 绘制分类结果
plt.figure(figsize=(8, 6))

# 绘制每个类别的数据点
# for i in range(3):
#     plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], label=f'Class {i+1}')


plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], label=f'negative')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], label=f'positive')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], label=f'neutral')

# # 绘制每个类别的数据点
# colors = ['r', 'g', 'b']  # 定义每个类别的颜色
# for i in range(3):
#     plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], color=colors[i], label=f'Class {i+1}')


plt.xticks([])
plt.yticks([])
# plt.title('Classification Result')
plt.tight_layout()
plt.legend(loc='upper left',prop={'size': 20})  # 添加图例
plt.savefig('ccc.png')  # 在 plt.show() 之前保存图像
# plt.savefig('ccc_new.png')  # 在 plt.show() 之前保存图像
# plt.savefig('tmp.png')  # 在 plt.show() 之前保存图像
plt.show()

