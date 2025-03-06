import operator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math as m



# 单一样本 多特征 距离计算 dim代表特征数量

def get_euc_dist(ins1, ins2, dim):
    dist = 0
    for i in range(dim) :
        dist += pow((ins1[i]-ins2[i]) , 2)
    return m.sqrt(dist)

# 获取最接近的 k个 邻居类型
def get_neighbors(test_sample , train_set,train_set_y,k ) :
    dist_list = []
    dim = len(test_sample)
    for i in range(len(train_set ) ) :
        dist = get_euc_dist(test_sample,train_set[i],dim)
        dist_list.append( (train_set_y[i],dist ) )
    dist_list.sort(key=operator.itemgetter(1) )
    test_sample_neighbors = []
    for i in range(k):
        test_sample_neighbors.append(dist_list[i][0])
    return test_sample_neighbors

# 预测样本所属分类
def predict_class_label(neighbors) :
    class_labels = {}
    # 统计不同类别的 票数
    for i in range(len(neighbors) ) :
        neighbor_index = neighbors[i]
        if neighbor_index in class_labels :
            class_labels[neighbor_index] += 1
        else :
            class_labels[neighbor_index] = 1
        # 对于票数进行降序排序
        label_sorted = sorted(class_labels.items(), key=operator.itemgetter(1), reverse=True)

    return label_sorted[0][0]

def getAccuracy(test_label,pre_labels ) :
    correct = 0
    for x in range(len(test_label)) :
        if test_label[x] == pre_labels[x] :
            correct += 1
    return (correct / float(len(test_label)) ) * 100.0


if __name__ == '__main__':
    column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    iris_data = pd.read_csv('./iris_training.csv',header=0, names=column_names)
    print(iris_data.head())

    iris_data = np.array(iris_data)
    # [:,0:4] 代表选择所有行，但是选择0到第3列，不选第四列
    # 二维数组的 切片操作，每一个不同的维度上的切片操作用 ‘，‘逗号隔离开来

    # print(iris_data.shape)
    # print(iris_data)
    iris_train,iris_train_y = iris_data[:,0:3],iris_data[:,3]


    iris_test = pd.read_csv('./iris_test.csv',header=0, names=column_names)

    iris_test = np.array(iris_test)
    # print(iris_test.shape)
    # print(iris_test)

    iris_test,iris_test_y = iris_test[:,0:3],iris_test[:,3]

    # print('----------------------------------iris_test_y----------------------------------')
    # print(iris_test_y)


    fig = plt.figure('不同的k值条件下的 iris分类准确度')
    k = 15
    for k in range(1,25) :
        pre_labels = []
        for x in range( len(iris_test) ) :
            neighbors = get_neighbors(iris_test[x],iris_train,iris_train_y,k)
            result = predict_class_label(neighbors)
            pre_labels.append(result)
        print('当 k 的值 等于 == :  '  + str(k) )
        print('预测类型 ： ' + repr(result) + ',真实类别 = ' + repr(iris_test_y[x]))
        print('预测类别： ' + repr(pre_labels))
        accuracy = getAccuracy(iris_test_y, pre_labels)
        print('Acuracy = ' + str(accuracy) + '%')
        plt.scatter(k,accuracy)
        plt.annotate(str(k),(k,accuracy))
        plt.annotate(k,(k,accuracy))


    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('不同的k参数下 ， 所存在的不同的训练集和测试集不同的 准确度')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()





