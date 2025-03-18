from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn import model_selection
from sklearn.datasets import load_iris


def load_data():
    data_set = []
    data_X = []
    data_y = []
    iris = load_iris()
    data_X.append(iris.data)
    data_y.append(iris.target)
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    data_X = np.reshape(data_X, (150, 4))
    data_y = np.reshape(data_y, (150))
    # data_X = data_X[1:3,:]
    return data_X,data_y

def split_data(data_X,data_y):

    X_train,X_test,y_train,y_test=model_selection.train_test_split(data_X,data_y,random_state=42,test_size=0.3,train_size=0.5)
    # print(y_train)
    for index, item in enumerate(X_train):
        X_train[index] = list(map(float, item))
    for index, item in enumerate(X_test):
        X_test[index] = list(map(float, item))
    # print(X_train)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    return X_train,X_test,y_train,y_test

def show_data(X_train,y_train):
    c0 = [i for i in range(len(y_train)) if y_train[i] == 1]
    c1 = [i for i in range(len(y_train)) if y_train[i] == 0]
    c2 = [i for i in range(len(y_train)) if y_train[i] == 2]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    print(X_train[c0, 0])
    plt.scatter(x=X_train[c0,0], y=X_train[c0,1], color='r', marker='s',label='Iris-virginica')
    plt.scatter(x=X_train[c1,0], y=X_train[c1,1], color='g', marker='o', label='Iris-setosa')
    plt.scatter(x=X_train[c2,0], y=X_train[c2,1], color='b', marker='v',label='Iris-versicolor')

    plt.xlabel("花萼宽度")
    plt.ylabel("花瓣长度")
    plt.title("各数据类型的散点图")
    plt.legend(loc='upper left')
    plt.show()
if __name__=='__main__':
    data_X,data_y=load_data()

    # print(data_X.shape[1])
    # print(data_X.shape[2])
    X_train,X_test,y_train,y_test=split_data(data_X,data_y)

    # print(X_train)
    # print(y_train)

    show_data(X_train,y_train)
    clf1 = svm.SVC(C=1,kernel='linear', decision_function_shape='ovr').fit(X_train,y_train)
    clf2 = svm.SVC(C=1, kernel='rbf', gamma=0.7).fit(X_train,y_train)
    clf3 = svm.SVC(kernel='poly').fit(X_train,y_train)
    clf4 = svm.SVC(kernel='sigmoid').fit(X_train,y_train)

    # print("linear线性核函数-训练集：",clf1.score(X_train, y_train))
    print("linear线性核函数-测试集：",clf1.score(X_test, y_test))
    print("RBF核函数测试集：",clf2.score(X_test, y_test))
    # print(clf1.decision_function(X_train))
    # print(clf1.predict(X_train))
