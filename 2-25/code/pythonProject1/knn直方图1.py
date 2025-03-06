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
    # print(v1, v2)  # 打印读取的值（注释掉）
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
v = 0.5  # 窗口宽度0.3   1.0
n = 500  # 训练集的大小
# 初始化计数器
count0 = 0  # 用于统计分类0正确的次数
count1 = 0  # 用于统计分类1正确的次数
# 计算训练集中第0列的最大值和最小值
max0 = max(train, key=itemgetter(0))[0]
min0 = min(train, key=itemgetter(0))[0]
# 计算分箱的数量
bin = (int)((max0 - min0) / v)
print(max0, min0)  # 打印最大值和最小值
print("bin", bin)  # 打印分箱数量
bin0 = -1  # 初始化分箱索引
# 遍历测试集
for i in range(1000):
    k0 = 0  # 用于统计训练集0中落在窗口内的数据个数
    k1 = 0  # 用于统计训练集1中落在窗口内的数据个数
    # newData新来的测试数据
    newData = test[i][0]  # 获取测试集中第i行的第0列数据
    # 确定newData所属的分箱
    for binx in range(bin):
        if (newData >= min0 + (binx - 1) * v) and (newData <= min0 + binx * v):
            bin0 = binx
    # 计算当前分箱的区间范围
    iRange = min0 + (bin0 - 1) * v
    lRange = min0 + bin0 * v
    # 统计训练集中的数据有多少个落在窗口内
    for j in range(500):
        trainData0 = train0[j][0]  # 第0类数据
        trainData1 = train1[j][0]  # 第1类数据
        if (trainData0 >= iRange) and (trainData0 <= lRange):
            k0 = k0 + 1
        if (trainData1 >= iRange) and (trainData1 <= lRange):
            k1 = k1 + 1
    # 根据公式估计类条件概率密度，并放入测试数据的第3列
    classPro0 = k0 / (n * v)  # 计算第0类的类条件概率
    classPro1 = k1 / (n * v)  # 计算第1类的类条件概率
    # 第3列存放属于第0类的类条件概率，第4列存放属于第1类的类条件概率
    print(test[i][1])  # 打印测试集中第i行的第1列数据
    test[i][2] = classPro0  # 将第0类的类条件概率存入测试集的第3列
    test[i][3] = classPro1  # 将第1类的类条件概率存入测试集的第4列
    # 贝叶斯分类器进行分类,直接用类条件概率大小进行比较进行分类
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
print("类别0：", accurate0)  # 打印分类0的正确率
print("类别1：", accurate1)  # 打印分类1的正确率

# 数据分布直方图
# 从测试集 test中提取分类结果为0的数据的第0列值（x）
ret_x0 = [x for [x, y, z, a, b] in test if b == 0]
# 从测试集 test 中提取分类结果为1的数据的第0列值（x）
ret_x1 = [x for [x, y, z, a, b] in test if b == 1]
# 打印分类结果为0的数据的第0列值
print("ret_x0:")
print(ret_x0)
# 绘制分类结果为0的数据的直方图
# ret_x0:数据; bin:分箱数量; color='r': 直方图颜色为红色; label='class 0': 设置图例标签为 'class 0'
n, bins, patches = plt.hist(ret_x0, bin, color='r', label='class 0')
# 绘制分类结果为1的数据的直方图
n, bins, patches = plt.hist(ret_x1, bin, color='b', label='class 1')
# 显示图例，位置在右上角
plt.legend(loc='upper right')
# 设置图表标题，标题内容为 "一维数据分布直方图(width=窗口宽度)"
title = "一维数据分布直方图(width=%.1f)" % (v)
plt.title(title)
# 设置 matplotlib 的字体为 SimHei（黑体），以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 显示绘制的图表
plt.show()
'''
1.数据提取：从测试集 test中提取分类结果为 0 和 1 的数据的第 0 列值。
2.直方图绘制：分别绘制分类结果为 0 和 1 的数据的直方图，并用不同颜色区分。
3.图表设置：添加图例、标题，并设置中文字体支持。
4.显示图表：调用 plt.show() 显示绘制好的直方图。
'''
