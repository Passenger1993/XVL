from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import sys, time

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1024, 512)
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(350, 430, 371, 20))
        self.label.setStyleSheet("color: rgb(0, 255, 0);\n"
"font: 14pt \"GOST type A\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(160, 60, 700, 350))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("D:/Documents/Выборка/Без дефектов/Б1.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(170, 460, 741, 16))
        self.progressBar.setStyleSheet("font: 12pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);")
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Загрузка изображения, ожидайте...."))


app = QApplication(sys.argv)

