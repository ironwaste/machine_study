import operator
import matplotlib.pyplot as plt
import xlrd
from Lib.cProfile import label
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math as m

# 鲈鱼 - 鲑鱼数据集
# long：鲈鱼 - 鲑鱼的长度特征数据集
# long     :    两类鱼的各1000个长度特征
# label    :    标签（0为鲈鱼，1为鲑鱼）
#
# light：鲈鱼 - 鲑鱼的亮度特征数据集
# light    :    两类鱼的各1000个亮度特征
# label    :    标签（0为鲈鱼，1为鲑鱼）

# joint_normal：鲈鱼 - 鲑鱼长度亮度联合正态分布样本
# long     :    两类鱼的共2000个长度特征
# light    :    两类鱼的共2000个亮度特征
# label    :    标签（0为鲈鱼，1为鲑鱼）

def get_euc_dist(ins1, ins2, dim):
    dist = 0
    for i in range(dim) :
        dist += pow((ins1[i]-ins2[i]) , 2)
    return m.sqrt(dist)

def get_neighbors(test_sample,train_set,train_set_y,k) :
    dist_list = []
    dim = len(test_sample)
    for i in range(len(train_set) ) :
        distance = get_euc_dist(test_sample, train_set[i], dim)
        dist_list.append((train_set_y[i],distance)) #train_set_y[i] is label

    dist_list.sort(key=operator.itemgetter(1) )
    test_sample_neighbors = []

    for i in range(k) :
        test_sample_neighbors.append(dist_list[i][0])
    return test_sample_neighbors


def predict_class_label(neighbors) :
    class_labels = {}
    for i in range(len(neighbors)) :
        neighbor = neighbors[i]
        if neighbor in class_labels :
            class_labels[neighbor] += 1
        else :
            class_labels[neighbor] = 1
    label_sorted = sorted(class_labels.items(), key=operator.itemgetter(1), reverse=True)
    return label_sorted[0][0]

def getAccuracy(test_label,pre_labels ) :
    correct = 0
    for x in range(len(test_label)) :
        if test_label[x] == pre_labels[x] :
            correct += 1
    return (correct / float(len(test_label)) ) * 100.0




readbook = xlrd.open_workbook('./Fish.xls')
sheet = readbook.sheet_by_name('joint_normal')


nrows = sheet.nrows
ncols = sheet.ncols

fish =[]
# print(nrows,ncols)


for i in range(1,nrows):
    v1 = sheet.cell(rowx=i, colx=0).value
    v2 = sheet.cell(rowx=i, colx=1).value
    id = sheet.cell(rowx=i, colx=2).value
    # print(id)
    fish.append([v1,v2,id])

print(len(fish))


test0 = fish[0:500]
test1 = fish[1000:1500]

label_data = test0 + test1
label_data = np.array(label_data)
# print(label_data)
label_data_y = label_data[:,2]
print(len (label_data[0]) )

test2 = fish[500:1000]
test3 = fish[1500:]
no_label_data = test2 + test3

# 判断正确值 和 准确度的部分 数据

print(len(no_label_data[0]) )

label_data=np.array(label_data)
no_label_data=np.array(no_label_data)
ans_y = no_label_data[:,2]




fig = plt.figure('fish')

for k in range(10,300) :
    pre_labels = []
    for x in range(len(no_label_data) ) :

        neighbors = get_neighbors(no_label_data[x],label_data,label_data_y,k)

        result = predict_class_label(neighbors)
        pre_labels.append(result)

        # print('类型为 ： '+repr(result) + ',真实类别 ：' + repr(ans_y[x]) )
    accuracy = getAccuracy(ans_y,pre_labels)
    # print('准确度为 ： === ' + repr(accuracy))
    plt.scatter(k, accuracy,color='green')
    # plt.annotate(str(k), (k, accuracy))
    # plt.annotate(k, (k, accuracy))


plt.xlabel('k is ')
plt.ylabel('accuracy')
plt.title('不同的k参数下 ')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()


