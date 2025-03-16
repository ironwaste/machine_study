import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
def open_csvgetdata() :
    dataTrain = pd.read_csv("./iris.csv")
    dataTrain = dataTrain.values
    dataTrain = np.array(dataTrain)
    dataTrain = dataTrain[:,1:5]
    return dataTrain

def dist_eclud(v1,v2) :
    return sqrt(sum(power(v1-v2,2)))
# 就是正常的距离公式
# v1 的各个元素 和 v2 进行减法操作，然后对其所有进行平方
# sum 则对于所有的元素加和  (x1 - x2)^2 + (y1 - y2)^2

def update_cluster(k,mu,X,y_label) :
    for i in range(X.shape[0]):
        min_dist = float('inf')
        for index in range(k):
            dist = dist_eclud(mu[index], X[i])
            # print("第1个距离：",dist)
            if dist < min_dist:
                min_dist = dist
                y_label[i] = index
            # 根据距离 标签分类
    return y_label



# def update_centroids(k, mu, X, y_label):
#     # 更新 质心向量
#     for i in range(k):
#         sum = np.array([0.0, 0.0])
#         num = np.sum(y_label == i)
#
#         cluster_index, label = np.where(y_label == i)
#         print("cluster_index:", list(cluster_index))
#         for j in cluster_index:
#             sum = sum + X[j]
#         print("sum:", sum)
#         centroid = sum / num
#         centroid = np.mean(X[cluster_index], axis=0)
#         mu[i] = centroid
#         # print(centroid)
#     return mu
#
#
# def show_figure(dataSet, k, centroids, clusters):
#     num_samples, dim = dataSet.shape
#     marker = ['or', 'ob', 'og', 'ok']
#     marker2 = ['*r', '*b', '*g', '*k']
#     for i in range(num_samples):
#         mark_index = int(clusters[i])
#         plt.plot(dataSet[i, 0], dataSet[i, 1], marker[mark_index])
#     for i in range(k):
#         plt.plot(centroids[i, 0], centroids[i, 1], marker2[i], markersize=10)
#     plt.xlim(0.1, 0.9)
#     plt.ylim(0, 0.8)
#     plt.xlabel('密度')
#     plt.ylabel('含糖率')
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#


dataTrain = open_csvgetdata()
k = 3






# x1 =np.array([1,2])
# x2 =np.array([3,2])
# print( sum( power(x1-x2,2)) )