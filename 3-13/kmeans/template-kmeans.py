import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
# dataTrain = pd.read_csv("xiguadata.txt")
dataTrain = [
        [0.697, 0.460], [0.774, 0.376],[0.634, 0.264],[0.608, 0.318],
        [0.556, 0.215],[0.403, 0.237],[0.481,0.149],[0.437, 0.211],
        [0.666, 0.091],[0.243, 0.267],[0.245, 0.057],[0.343, 0.099],
        [0.639, 0.161],[0.657, 0.198],[0.360, 0.370],[0.593, 0.042],
        [0.719, 0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
        [0.748, 0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],
        [0.525, 0.369],[0.751,0.489],[0.532,0.472],[0.473,0.376],
        [0.725, 0.445],[0.446,0.459]]
dataTrain=np.array(dataTrain)
#print(dataTrain)
y_label=np.zeros((dataTrain.shape[0],1))
def dist_eclud(vec_A, vec_B):
        return sqrt(sum(power(vec_A - vec_B, 2)))
def update_clusters(k,mu,dataTrain,y_label):
        for i in range(dataTrain.shape[0]):
                min_dist = float('inf')
                for index in range(k):
                        dist = dist_eclud(mu[index], dataTrain[i])
                        #print("第1个距离：",dist)
                        if dist < min_dist:
                                min_dist = dist
                                y_label[i] = index
        return y_label

def update_centroids(k,mu,datTrain,y_label):
        # print(y_label)
        # 由于y_label 数组是二维的数组，
        # 所以np.where 返回的也是一种二维数组

        for i in range(k):
                ssum = np.array([0.0,0.0])
                num = np.sum(y_label == i)
                cluster_index,label=np.where(y_label==i)
                # print("cluster_index:",list(cluster_index))
                # print(label)
                # print(ssum)
                for j in cluster_index:
                        # print("dataTrain : " + str(dataTrain[j]))
                        ssum = ssum + dataTrain[j]
                        # print(ssum)
                # print("sum:",sum)
                # print(sum)
                # centroid=sum/num
                # print(centroid)
                centroid = np.mean(dataTrain[cluster_index],axis=0)
                mu[i]=centroid
                #print(centroid)
        return mu


def show_figure(dataSet, k, centroids, clusters):
        num_samples, dim = dataSet.shape
        marker = ['or', 'ob', 'og', 'ok']
        marker2 = ['*r', '*b', '*g', '*k']
        for i in range(num_samples):
                # print(str(i) + "  : " + str(clusters[i]) )
                mark_index = int(clusters[i])

                plt.plot(dataSet[i, 0], dataSet[i, 1], marker[mark_index])
        for i in range(k):
                plt.plot(centroids[i, 0], centroids[i, 1], marker2[i], markersize=10)
        plt.xlim(0.1, 0.9)
        plt.ylim(0, 0.8)
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False



k=3
mu1=np.array([0.403, 0.237])
mu2=np.array([0.343, 0.099])
mu3=np.array([0.478,0.437])
mu=np.zeros((k,dataTrain.shape[1]))
mu=np.array([[0.403, 0.237],[0.343, 0.099],[0.478,0.437]])
mu=np.array([[0.634, 0.264],[0.437, 0.211],[0.666, 0.091]])
iters=4
for iter in range(iters):
        # dataTrain = np.array(dataTrain)
        y_label=update_clusters(k,mu,dataTrain,y_label)
        # print("y_label:", y_label)
        # print("the first : "+str(dataTrain[0]))
        mu=update_centroids(k,mu,dataTrain,y_label)
        # print("mu=",mu)
        plt.subplot(2, 2, iter+1, frameon=True)
        t='第'+str(iter+1)+'次迭代后'
        plt.legend(title=t)
        # plt.title(t,color='red',loc='lower left')
        show_figure(dataTrain, k, mu, y_label)
plt.show()