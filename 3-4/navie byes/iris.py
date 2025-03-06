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


def open_wwcsv():
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


sheet = open_wwcsv()
nrows = sheet.nrows
ncols = sheet.ncols


iris,t = invalue(sheet,nrows)


dataTrain = np.array(iris)

# 二维数组 ，前为列 ，后为行


irisse=np.sum(t=='Iris-setosa')
irisve=np.sum(t=='Iris-versicolor')
irisvi=np.sum(t=='Iris-virginica')

print(t)
print(irisse,irisve,irisvi)

prior_se = irisse/nrows
prior_ve = irisve/nrows
prior_vi = irisvi/nrows









