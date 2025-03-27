from matplotlib.pyplot import subplot
from numpy.random import random
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import plot_tree
from matplotlib import colors
from sklearn import model_selection
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
from sklearn.datasets import load_digits
from sklearn import tree
import numpy as np
from sklearn.preprocessing import StandardScaler

def show_num():
    """
    显示手写数字数据集中的前4个数字图像
    使用matplotlib的matshow函数显示灰度图像
    """
    digits = load_digits()
    plt.gray()  # 设置为灰度显示模式
    plt.matshow(digits.images[0])  # 显示第一个数字
    plt.matshow(digits.images[1])  # 显示第二个数字
    plt.matshow(digits.images[2])  # 显示第三个数字
    plt.matshow(digits.images[3])  # 显示第四个数字
    plt.show()

def load_data():
    """
    加载手写数字数据集并进行预处理
    返回：
        data_X: 特征矩阵，每行代表一个数字图像（8x8像素）
        data_y: 标签向量，表示每个图像对应的数字（0-9）
    """
    data_set = []
    data_X = []
    data_y = []
    digits = load_digits()  # 加载手写数字数据集

    data_X.append(digits.data)  # 添加特征数据
    data_y.append(digits.target)  # 添加标签数据
    data_X = np.array(data_X)
    data_X = np.reshape(data_X, (data_X.shape[1], data_X.shape[2]))  # 重塑数组维度
    data_y = np.array(data_y)
    data_y = np.reshape(data_y, data_y.shape[1])  # 重塑标签数组
    return data_X, data_y

def split_data(data_X, data_y):
    """
    将数据集分割为训练集和测试集
    参数：
        data_X: 特征矩阵
        data_y: 标签向量
    返回：
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集标签
        y_test: 测试集标签
    """
    # 使用sklearn的train_test_split函数分割数据集
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_X, data_y, test_size=0.2, random_state=0)
    
    # 将特征数据转换为浮点数类型
    for index, item in enumerate(X_train):
        X_train[index] = list(map(float, item))
    for index, item in enumerate(X_test):
        X_test[index] = list(map(float, item))
    
    # 将数据转换为numpy数组
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test

# 加载数据
data_X, data_y = load_data()
# 分割数据集
x_train, x_test, y_train, y_test = split_data(data_X, data_y)

# 数据标准化处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # 对训练集进行标准化
x_test = scaler.transform(x_test)  # 对测试集进行标准化

# 创建并训练决策树分类器
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)  # 训练模型

# 评估模型性能
score = classifier.score(x_test, y_test)
print(score)  # 打印准确率

# 将决策树导出为.dot文件，用于可视化
with open("kk.dot", 'w') as f:
    f = export_graphviz(classifier, out_file=f)

# 使用matplotlib绘制决策树
plt.figure(figsize=(20, 10))
plot_tree(
    classifier,
    # feature_names=digits.feature_names,
    # class_names=[str(i) for i in range(10)],
    # filled=True,
    # rounded=True
)
plt.show()

# 以下代码被注释掉，用于生成PDF格式的决策树图
# dot_data = tree.export_graphviz(classifier, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("Handwritten numeral recognition.pdf")




















