import xlrd  # 导入xlrd库，用于读取Excel文件
from operator import itemgetter  # 导入itemgetter，用于获取列表或元组中的某个元素
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

# 打开Excel文件
readbook = xlrd.open_workbook('./Fish.xls')
# 通过索引获取Excel中的第二个工作表
sheet = readbook.sheet_by_index(1)
# 通过名称获取Excel中的工作表，名称为'long'
sheet = readbook.sheet_by_name('long')

# 获取工作表的行数和列数
nrows = sheet.nrows  # 行数
ncols = sheet.ncols  # 列数
print(nrows, ncols)  # 打印行数和列数

# 初始化一个空列表，用于存储从Excel中读取的数据
fish = []
# 遍历每一行（从第二行开始，跳过表头）
for i in range(1, nrows):
    v1 = sheet.cell(i, 0).value  # 获取第i行第1列的值
    v2 = sheet.cell(i, 1).value  # 获取第i行第2列的值
    print(v1, v2)  # 打印读取的值
    fish.append([v1, v2, 0, 0, 0])  # 将读取的值存入fish列表，并初始化后三列为0

print(fish)  # 打印fish列表

# 将fish列表中的数据分为训练集和测试集
train0 = fish[0:500]  # 取前500行作为训练集0
train1 = fish[1000:1500]  # 取1000到1500行作为训练集1
train = train0 + train1  # 合并训练集0和训练集1
print(train)  # 打印训练集
print(len(train))  # 打印训练集的长度

test0 = fish[500:1000]  # 取500到1000行作为测试集0
test1 = fish[1500:]  # 取1500行之后的数据作为测试集1
test = test0 + test1  # 合并测试集0和测试集1
print(len(test))  # 打印测试集的长度

# 设置参数
v = 3  # 窗口宽度
n = 500  # 训练集的大小

# 初始化计数器
count0 = 0  # 用于统计分类0正确的次数
count1 = 0  # 用于统计分类1正确的次数

# 遍历测试集
for i in range(1000):
    k0 = 0  # 用于统计训练集0中落在窗口内的数据个数
    k1 = 0  # 用于统计训练集1中落在窗口内的数据个数
    # newData新来的测试数据
    newData = test[i][0]  # 获取测试集中第i行的第0列数据
    # 根据窗口大小确定值域范围
    iRange = newData - v / 2  # 窗口的左边界
    lRange = newData + v / 2  # 窗口的右边界

    # 统计训练集中的数据有多少个落在窗口内
    for j in range(500):
        trainData0 = train0[j][0]  # 第0类数据
        trainData1 = train1[j][0]  # 第1类数据
        if (trainData0 >= iRange) and (trainData0 <= lRange):
            k0 = k0 + 1  # 如果第0类数据落在窗口内，则增加k0
        if (trainData1 >= iRange) and (trainData1 <= lRange):
            k1 = k1 + 1  # 如果第1类数据落在窗口内，则增加k1

    # 根据公式估计类条件概率密度，并放入测试数据的第3列和第4列
    classPro0 = k0 / (n * v)  # 计算第0类的类条件概率
    classPro1 = k1 / (n * v)  # 计算第1类的类条件概率
    test[i][2] = classPro0  # 将第0类的类条件概率存入测试集的第3列
    test[i][3] = classPro1  # 将第1类的类条件概率存入测试集的第4列

    # 贝叶斯分类器进行分类，直接用类条件概率大小进行比较进行分类
    # 第5列存放测试数据的分类结果
    if classPro0 > classPro1:
        test[i][4] = 0  # 如果第0类的概率大于第1类，则分类为0
    if classPro0 < classPro1:
        test[i][4] = 1  # 如果第1类的概率大于第0类，则分类为1

    # 统计分类正确次数
    if (i <= 500) and (test[i][4] == test[i][1]):
        count0 = count0 + 1  # 如果分类正确，则增加count0
    # 统计分类正确次数
    if (i > 500) and (test[i][4] == test[i][1]):
        count1 = count1 + 1  # 如果分类正确，则增加count1

# 计算分类0和分类1的正确性
accurate0 = count0 / 500  # 计算分类0的正确率
accurate1 = count1 / 500  # 计算分类1的正确率
print(accurate0, accurate1)  # 打印分类0和分类

# 打印测试集中第1到500行的数据（索引从1到499）
print(test[1:500])
# 设置图表标题，标题内容为 "一维数据Parzen矩形窗估计(width=窗口宽度), precision1=分类0正确率, precision2=分类1正确率"
title = "一维数据Parzen矩形窗估计(width=%.1f), precision1=%.3f, precision2=%.3f" % (v, accurate0, accurate1)
# 将测试集转换为numpy数组，方便后续操作
test = np.array(test)
# 打印测试集中所有行的第0列数据（即特征值）
print(test[:, 0])
# 创建一个新的图表
fig = plt.figure()
# 添加第一个子图（2行1列的第1个子图）
plt.subplot(211)
# 绘制分类0的数据散点图
# test[:500, 0]: 分类0的特征值（第0列）
# test[:500, 2]: 分类0的类条件概率密度（第2列）
# marker='s': 使用正方形标记点
plt.scatter(test[:500, 0], test[:500, 2], marker='s')
# 设置x轴标签
plt.xlabel('Class 0 Data')
# 设置y轴标签
plt.ylabel('Density')
# 设置子图标题
plt.title(title)
# 添加第二个子图（2行1列的第2个子图）
plt.subplot(2, 1, 2)
# 绘制分类1的数据散点图
# test[500:1000, 0]: 分类1的特征值（第0列）
# test[500:1000, 3]: 分类1的类条件概率密度（第3列）
# marker='v': 使用倒三角形标记点
plt.scatter(test[500:1000, 0], test[500:1000, 3], marker='v')
# 设置x轴标签
plt.xlabel('Class 1 Data')
# 设置y轴标签
plt.ylabel('Density')
# 设置matplotlib的字体为SimHei（黑体），以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 显示绘制的图表
plt.show()


'''
1.数据打印：打印测试集中第1到500行的数据;打印测试集中所有行的第0列数据（特征值）。
2.图表标题设置：设置图表标题，包含窗口宽度和分类正确率信息。
3.数据可视化：创建图表，包含两个子图：第一个子图绘制分类0的特征值与类条件概率密度的散点图;第二个子图绘制分类1的特征值与类条件概率密度的散点图。
4.字体设置：设置中文字体支持，解决中文显示问题。
5.显示图表：调用plt.show() 显示绘制好的图表。
'''