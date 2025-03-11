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

# 用于将 根据 标签将值进行加和 的条件 策略
# names = np.array( ['bob','joe','will'] )
# scores = np.random.randint(0,100,(3,4))
# print(scores[names == 'bob'])


def open_wwxls():
    readbook = xlrd.open_workbook('./iris.xls') # 打开csv文件
    sheet = readbook.sheet_by_index(0) # 获取表格
    return sheet

def invalue(sheet,nrows) :
    iris = []
    y = []
    # v1,v2,v3,v4 , name
    for i in range(1, nrows):
        v1 = sheet.cell(rowx=i, colx=1).value
        v2 = sheet.cell(rowx=i, colx=2).value
        v3 = sheet.cell(rowx=i, colx=3).value
        v4 = sheet.cell(rowx=i, colx=4).value
        name = sheet.cell(rowx=i, colx=5).value
        iris.append([v1, v2, v3, v4,name])
        y.append(name)
    return iris, y

def open_csv():
    path = './iris.csv';
    data = pd.read_csv(path,header=None) # none header
    return data

data = open_csv()

# 提取特征值 并且将特征值 转换为 numpy数组形式 方便下述操作
x = data[list( range(1,5) ) ]
x = data.values
x = x[2:,:-1]
# 将 其按照特征值标签进行离散化 数字化 方便后续操作
y = pd.Categorical(data[5]).codes
y[1:]
# 二维数组 ，前为列 ，后为行
# print(x)

# len(x)
print(x)

kind= y[0] # 种类个数
print(kind)

list1 = []
for i in range(kind+1) :
    list1.append(0)

for i in range( len(x) ):
    print(str(y[i]) + '    x[0][i] : '+ str(x[i][0]))
    list1[y[i]] += float(x[i][0])

print(list1)

# a = np.array()
# for i in range(kind) :
#     for  :

# irisse=np.sum(t=='Iris-setosa')
# irisve=np.sum(t=='Iris-versicolor')
# irisvi=np.sum(t=='Iris-virginica')
#
# print(t)
# print(irisse,irisve,irisvi)
#
# prior_se = irisse/nrows
# prior_ve = irisve/nrows
# prior_vi = irisvi/nrows









