import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
def open_csvgetdata() :
    dataTrain = pd.read_csv("./iris.csv")
    dataTrain = dataTrain.values
    dataTrain = np.array(dataTrain)
    ylabel = dataTrain[:,5]
    dataTrain = dataTrain[:,1:5]

    y = pd.Categorical(ylabel).codes
    return dataTrain,y

def dist_eclud(v1,v2) :
    return sqrt(sum(power(v1-v2,2)))
# 就是正常的距离公式
# v1 的各个元素 和 v2 进行减法操作，然后对其所有进行平方
# sum 则对于所有的元素加和  (x1 - x2)^2 + (y1 - y2)^2

def update_cluster(k,mu,X,ylabel) :

    for i in range(X.shape[0]):
        min_dist = float('inf')
        for index in range(k):
            dist = dist_eclud(mu[index], X[i])
            # print("第1个距离：",dist)
            if dist < min_dist:
                min_dist = dist
                ylabel[i] = index
            # 根据距离 标签分类
    return ylabel





def show_figure(plt,iters,dataSet, k, centroids, clusters):

    num_samples, dim = dataSet.shape
    cnt = 0
    for i in range(4) :

        for j in range(i+1,4):
            # print("i is : "+str(i)+"  j is : "+str(j) + " cnt si : " +str(cnt))
            # print(iters)
            # print(iters * 6 + cnt)
            plt.subplot(4, 6, iters * 6 + cnt + 1, frameon=True)
            cnt += 1
            # t = '第' + str(iters + 1) + '次迭代后'
            # plt.legend(title=t)
            marker = ['or', 'ob', 'og', 'ok']
            marker2 = ['*r', '*b', '*g', '*k']

            for kk in range(num_samples):
                mark_index = int(clusters[kk])
                plt.plot(dataSet[kk, i], dataSet[kk, j], marker[mark_index])

            for kk in range(k):
                plt.plot(centroids[kk, i], centroids[kk, j], marker2[0],color='#88c999')

        # plt.xlim(0.1, 0.9)
        # plt.ylim(0, 0.8)
        # plt.xlabel('密度')
        # plt.ylabel('含糖率')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

def update_centroids(k, mu, X, y_label):
    # k 代表的是簇的个数
    # mu 代表的是 不同的簇的 质心向量
    # X[] 代表的是 样本点 集合 存储的是样本点的特征值
    # 更新 质心向量  mu[] 数组
    for i in range(k):
        sum = np.array([0.0, 0.0])
        num = np.sum(y_label == i)
        cluster_index= np.where(y_label == i)
        # print(cluster_index)
        # 根据簇的分类找出相关的变量 的 索引
        # print("cluster_index:", cluster_index)
        # 求其平均值 得到当前的质心向量
        centroid = np.mean(X[cluster_index], axis=0)

        mu[i] = centroid

        # print(centroid)
    return mu


if __name__ == '__main__' :
    # print(__name__)
    dataTrain , y_label = open_csvgetdata()
    # print(dataTrain)
    # print(y_label)
    len = dataTrain.shape[0]

    k = 2
    alldatainedx =  [ i for i in range(len)]
    mu_index = np.random.choice(alldatainedx,k)
    # 在数组 alldataindex中选择 k个不同的随机数
    # print(mu_index)
    # mu=np.zeros(k,dataTrain.shape[1])
    mu = dataTrain[mu_index]
    # print(mu)
    iters = 4

    for i in range(iters) :
        ylabel = np.empty(len)
        # ylabel = np.array(ylabel)
        ylabel = update_cluster(k,mu,dataTrain,ylabel)
        mu = update_centroids(k,mu,dataTrain,ylabel)
        show_figure(plt,i,dataTrain,k,mu,ylabel)

    plt.show()


# x1 =np.array([1,2])
# x2 =np.array([3,2])
# print( sum( power(x1-x2,2)) )