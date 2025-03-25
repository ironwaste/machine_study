# 导入必要的库
from matplotlib.pyplot import subplot
from numpy.random import random
from sklearn import svm  # 导入支持向量机模型
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
import sys
import matplotlib.pyplot as plt  # 用于数据可视化
import matplotlib as mpl
from matplotlib import colors
from sklearn import model_selection  # 用于数据集分割
from sklearn.datasets import fetch_california_housing  # 未使用

def load_data():
    """
    加载波士顿房价数据集
    返回：
        data_X: 特征数据
        data_y: 目标变量（房价）
    """
    data_X = []
    data_y = []

    import pandas as pd
    import numpy as np
    data_url = "http://lib.stat.cmu.edu/datasets/boston"  # 波士顿房价数据集URL
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)  # 读取数据，跳过前22行
    data_X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # 水平堆叠特征数据
    data_y = raw_df.values[1::2, 2]  # 提取目标变量
    data_X = np.array(data_X)  # 转换为numpy数组
    # data_X = np.reshape(data_X, (data_X.shape[1], data_X.shape[2]))  # 注释掉的重塑操作
    data_y = np.array(data_y,dtype=int)  # 转换为整型numpy数组
    # data_y = np.reshape(data_y, data_y.shape[1])  # 注释掉的重塑操作
    # print(data_X.shape)  # 注释掉的打印语句

    return data_X,data_y

def split_data(data_X,data_y) :
    """
    将数据集分割为训练集和测试集
    参数：
        data_X: 特征矩阵
        data_y: 标签向量
    返回：训练集特征、测试集特征、训练集标签、测试集标签
    """
    X_train,X_test,y_train,y_test = model_selection.train_test_split(data_X,data_y,test_size=0.1,train_size=0.2,random_state=0)  # 80%训练集，20%测试集
    for index,item in enumerate(X_train) :
        X_train[index] = list(map(float,item))  # 将训练集特征转换为浮点数
    for index, item in enumerate(X_test):
        X_test[index] = list(map(float, item))  # 将测试集特征转换为浮点数
    X_train = np.array(X_train)  # 转换为numpy数组
    X_test = np.array(X_test)  # 转换为numpy数组
    y_train = np.array(y_train)  # 转换为numpy数组
    y_test = np.array(y_test)  # 转换为numpy数组
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':
    # 程序主入口
    data_X,data_y=load_data()  # 加载数据集
    # print(data_X)  # 注释掉的打印特征数据
    # print(data_y)  # 打印目标变量

    X_train,X_test,y_train,y_test = split_data(data_X,data_y)  # 分割数据集
    ansline_x = []  # 存储不同C值
    ansline_y = []  # 存储对应的准确率
    fig,axs= plt.subplots(1,2)  # 创建1行2列的子图
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # 测试线性核函数SVM，不同的C值
    for i in range(111,11000,100) :  # C从1到101000，步长为100
        clf2 = svm.SVC(C=i,kernel='linear',decision_function_shape='ovr').fit(X_train,y_train.astype('int'))  # 训练线性核SVM
        ansline_x.append(i)  # 记录C值
        ansline_y.append(clf2.score(X_test,y_test))  # 记录准确率
        print(i)
    axs[0].scatter(ansline_x,ansline_y,color='blue')  # 左子图绘制线性核函数结果
    
    # 测试RBF核函数SVM，不同的gamma值
    ansline_x = []  # 重置存储不同gamma值
    ansline_y = []  # 重置存储对应的准确率
    for i in range(20) :  # 测试20次
        gg = random() / 40  # 随机生成gamma值
        clf1 = svm.SVC(C=1,kernel='rbf',gamma=gg,random_state=0).fit(X_train,y_train.astype('int'))  # 训练RBF核SVM
        ansline_x.append(gg)  # 记录gamma值
        ansline_y.append(clf1.score(X_test, y_test))  # 记录准确率

    axs[1].scatter(ansline_x,ansline_y,color='green')  # 右子图绘制RBF核函数结果
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.legend(loc='upper left')  # 设置图例位置
    plt.show()  # 显示图形

    # print("linear 线性核函数-训练集",clf2.score(X_test,y_test))  # 注释掉的打印线性核函数测试结果
    # print("RBF 核函数-训练集",clf1.score(X_test,y_test))  # 注释掉的打印RBF核函数测试结果