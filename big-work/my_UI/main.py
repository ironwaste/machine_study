from PyQt5 import QtCore, QtGui, QtWidgets
import sch
import sys
from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import *
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QTableWidgetItem, QFileDialog, QMessageBox
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shutil

class T(sch.Ui_MainWindow,QtWidgets.QMainWindow) :

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.predict_Button.clicked.connect(self.open_png)
        self.Train_Button.clicked.connect(self.train_entry)

    def open_png(self) :
        # getOpenFileNames 和 getOpenFileName的区别 一个是多个一个是单个文件
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",# title
            "", # starting dir,
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        # 成员变量 - 存储
        if not filename:  # 用户取消选择
            return
        pixmap = QPixmap(filename)
        if pixmap.isNull():  # 检查是否加载成功
            QMessageBox.warning(self, "错误", "无法加载图片文件！")
            return
        self.preview_picture.setPixmap(pixmap)
        self.preview_picture.setScaledContents(True)  # 自适应QLabel大小
        self.output_num.setNum(self.img_to_arr(filename))

    def img_to_arr(self,filename) : # 将 需要预测的图片转化为 数组形式
        arr = []
        fh = open(filename)
        im = Image.open(filename)
        for i in range(0,32) :
            for j in range(0, 32):
                pix = im.getpixel((i,j)) # RGB 三通道 颜色值
                pixs = pix[0] + pix[1] + pix[2] # 转化为单通道颜色值
                if pixs == 0 :
                    arr.append(1)
                else :
                    arr.append(0)
        print(self.predict(arr,self.W1,self.W2))
        return np.argmax(self.predict(arr,self.W1,self.W2))





    def train_entry(self):
        lr = self.lr_num.toPlainText()
        iter = self.iter_num.toPlainText()
        if not iter.isnumeric() or not self.IsFloatNum(lr) :
            QMessageBox.warning(self, "错误", "输入不是浮点数和整数!")
            return
        self.lr = float(lr)
        self.iter = int(iter)
        self.img_path = "./img"  # 手写体数字图片路径
        self.txt_path = "./txt"  # 转换后的数字矩阵的保存路径
        self.train_path = "./train/"
        self.test_path = "./test/"
        self.img_to_txt(self.img_path, self.txt_path)
        self.get_split_data()
        X_train, y_train = self.get_traindata()
        X_test, y_test = self.get_testdata()

        y_test = np.array(list(map(int, y_test)))
        y_train = LabelBinarizer().fit_transform(y_train)
        self.train(X_train, y_train, X_test, y_test)


    # 建立训练数据
    def get_traindata(self):
        labels = []
        train_file = listdir(self.train_path)

        trainarr = np.zeros((len(train_file), 1024))

        for i in range(0, len(train_file)):
            this_label = train_file[i].split("_")[0]
            if len(this_label) != 0:
                labels.append(int( train_file[i].split(".")[0].split("_")[0] ))
            trainarr[i, :] = self.data_to_array(self.train_path + train_file[i])
            print(this_label+"   :   " + train_file[i])

        return trainarr, labels

    def get_testdata(self):

        test_files = listdir(self.test_path)
        true_label_set = []
        test_set = np.zeros((len(test_files), 1024))

        for i in range(0, len(test_files)):
            true_label = (int)(test_files[i].split("_")[0])
            testarr = self.data_to_array(self.test_path + test_files[i])
            true_label_set.append(true_label)
            test_set[i, :] = testarr
        return test_set, true_label_set

    def data_to_array(self,fname):
        arr = []
        fh = open(fname)
        for i in range(0, 32):
            thisline = fh.readline()
            # print(thisline)
            for j in range(0, 32):
                arr.append(int(thisline[j]))
        return arr

    def img_to_txt(self,png_path, txt_path):
        png_list = listdir(png_path + "/" + 'png')
        for f in png_list:
            fname = f.split(".")[0]
            im = Image.open(png_path + "/" + 'png' + "/" + f)
            fh = open(txt_path + "/" + fname + ".txt", "w")
            for m in range(0, 32):
                for n in range(0, 32):
                    pix = im.getpixel((n, m))
                    pixs = pix[0] + pix[1] + pix[2]
                    if pixs == 0:
                        fh.write("1")
                    else:
                        fh.write("0")
                fh.write("\n")
            fh.close()

    def get_split_data(self):
        txt_list = listdir(self.txt_path)

        for txt in txt_list:
            try:
                shutil.move(self.txt_path + "/" + txt, self.test_path)
            except:
                pass

    def predict(self,X_data, W1, W2):
        L1 = self.sigmoid(np.dot(X_data, W1))
        L2 = self.sigmoid(np.dot(L1, W2))
        return L2

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    def train(self,X_train, y_train, X_test, y_test) :

        # 定义神经网络由三层组成，每一层的神经元个数分别为1024、50、10，其中，1024像素即1024通道输入，隐藏层神经元个数为50，输出层为0~9，所以是10个
        # 输入层到隐藏层的权值为W1，隐藏层到输出层的权值为W2
        iter = self.iter
        lr = self.lr
        W1 = np.random.random((1024, 50)) - 0.5
        W2 = np.random.random((50, 10)) - 0.5
        accx = []
        accy = []
        for it in range(iter):
            # 随机选取一个数据
            # k = np.random.randint(X_train.shape[0])
            k = it % X_train.shape[0]
            X_data = X_train[k]

            # 把数据变为矩阵形式
            X_data = np.atleast_2d(X_data)

            # 利用BP神经网络算法公式更新W1和W2
            L1 = self.sigmoid(np.dot(X_data, W1))
            L2 = self.sigmoid(np.dot(L1, W2))

            L2_delta = (y_train[k] - L2) * self.dsigmoid(np.dot(L1, W2))
            L1_delta = np.dot(L2_delta, W2.T) * self.dsigmoid(np.dot(X_data, W1))

            # 更新权值
            W2 += lr * np.dot(L1.T, L2_delta)
            W1 += lr * np.dot(X_data.T, L1_delta)

            # 每训练1000次，预测一次准确率

            if it % 1000 == 0:
                output = self.predict(X_test, W1, W2)
                pre_y = np.argmax(output, axis=1)
                acc = np.mean(np.equal(pre_y, y_test))
                accx.append(acc)
                accy.append(it)
                new_entry = f"[迭代次数：{it}; 准确度 ： {acc}]"
                current_logs = self.log_model.stringList()
                current_logs.append(new_entry)
                self.log_model.setStringList(current_logs)
                self.listView.scrollToBottom()
                # print("iterations:", it, "accuracy:", acc)
        print(accx)
        print(accy)
        plt.plot(accy,accx)
        plt.show()
        # print('suceess 训练结束')
        new_entry = f"[训练结束success]"
        current_logs = self.log_model.stringList()
        current_logs.append(new_entry)
        self.log_model.setStringList(current_logs)
        self.listView.scrollToBottom()
        self.W2=W2
        self.W1=W1


    def IsFloatNum(self,str):
        s = str.split('.')
        if len(s) > 2:
            return False
        else:
            for si in s:
                if not si.isdigit():
                    return False
            return True

if __name__ == '__main__' :
    app = QtWidgets.QApplication(sys.argv)
    win = T()

    win.show()
    sys.exit(app.exec())




