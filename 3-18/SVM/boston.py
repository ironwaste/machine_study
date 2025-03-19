from matplotlib.pyplot import subplot
from numpy.random import random
from sklearn import svm
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn import model_selection
from sklearn.datasets import fetch_california_housing


def load_data():
    data_X = []
    data_y = []

    import pandas as pd
    import numpy as np
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data_X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    data_y = raw_df.values[1::2, 2]
    data_X = np.array(data_X)
    # data_X = np.reshape(data_X, (data_X.shape[1], data_X.shape[2]))
    data_y = np.array(data_y,dtype=int)
    # data_y = np.reshape(data_y, data_y.shape[1])
    # print(data_X.shape)

    return data_X,data_y

def split_data(data_X,data_y) :
    """
    将数据集分割为训练集和测试集
    参数：
        data_X: 特征矩阵
        data_y: 标签向量
    返回：训练集特征、测试集特征、训练集标签、测试集标签
    """
    X_train,X_test,y_train,y_test = model_selection.train_test_split(data_X,data_y,test_size=0.2,random_state=0)
    for index,item in enumerate(X_train) :
        X_train[index] = list(map(float,item))
    for index, item in enumerate(X_test):
        X_test[index] = list(map(float, item))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':
    data_X,data_y=load_data()
    # print(data_X)
    print(data_y)

    X_train,X_test,y_train,y_test = split_data(data_X,data_y)
    ansline_x = []
    ansline_y = []
    fig,axs= plt.subplots(1,2)
    for i in range(1,101100,100) :
        clf2 = svm.SVC(C=i,kernel='linear',decision_function_shape='ovr').fit(X_train,y_train.astype('int'))
        ansline_x.append(i)
        ansline_y.append(clf2.score(X_test,y_test))

    axs[0].scatter(ansline_x,ansline_y,color='blue')
    ansline_x = []
    ansline_y = []
    for i in range(20) :
        gg = random() / 15
        clf1 = svm.SVC(C=1,kernel='rbf',gamma=gg,random_state=0).fit(X_train,y_train.astype('int'))
        ansline_x.append(gg)
        ansline_y.append(clf1.score(X_test, y_test))

    axs[1].scatter(ansline_x,ansline_y,color='green')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.legend(loc='upper left')
    plt.show()

    # print("linear 线性核函数-训练集",clf2.score(X_test,y_test))
    # print("RBF 核函数-训练集",clf1.score(X_test,y_test))

