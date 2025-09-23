from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt, QSize

window_size = [1366, 768]

class settings_window(QWidget):
    def setupUi(self, Dialog):
        global window_size
        Dialog.setObjectName("Dialog")
        self.button_size = 20
        Dialog.resize(window_size[0], window_size[1])
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                             "color: rgb(0, 255, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        Dialog.setWindowIcon(icon)
        self.label_1 = QtWidgets.QLabel(Dialog)
        self.label_1.setGeometry(QtCore.QRect(600, 10, 165, 40))
        self.label_1.setStyleSheet(f"font: 24pt \"GOST type A\";")
        self.label_1.setObjectName("label_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(50, 150, 131, 40))
        self.label_2.setStyleSheet(f"font: {self.button_size}pt \"GOST type A\";")
        self.label_2.setObjectName("label_3")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(50, 100, 211, 40))
        self.label_3.setStyleSheet(f"font: {self.button_size}pt \"GOST type A\";\n"
                                   "gridline-color: rgb(0, 255, 0);")
        self.label_3.setObjectName("label_4")
        self.comboBox_1 = QtWidgets.QComboBox(Dialog)
        self.comboBox_1.setGeometry(QtCore.QRect(280, 150, 380, 40))
        self.comboBox_1.setStyleSheet(f"border: 2px solid rgb(0, 255, 0);"
                                      f"font: {self.button_size - 2}pt \"GOST type A\";")
        self.comboBox_1.setObjectName("comboBox_2")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(280, 100, 380, 40))
        self.comboBox_2.setStyleSheet(f"border: 2px solid rgb(0, 255, 0);"
                                      f"font: {self.button_size}pt \"GOST type A\";")
        self.comboBox_2.setObjectName("comboBox_3")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.pushButton_1 = QtWidgets.QPushButton(Dialog)
        self.pushButton_1.setGeometry(QtCore.QRect(700, 700, 200, 50))
        self.pushButton_1.setStyleSheet("\n"
                                        "QPushButton{\n"
                                        "color: rgb(0, 255, 0);\n"
                                        f"font: 16pt \"GOST type A\";\n"
                                        "border: 2px solid rgb(0, 255, 0)\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:pressed{\n"
                                        "background-color:  rgb(0, 255, 0);\n"
                                        "color: rgb(0, 0, 0);\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:hover{\n"
                                        "background-color:  rgb(0, 255, 0);\n"
                                        "color: rgb(0, 0, 0);\n"
                                        "}")
        self.pushButton_1.setObjectName("pushButton_5")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(450, 700, 200, 50))
        self.pushButton_2.setStyleSheet("\n"
                                        "QPushButton{\n"
                                        "color: rgb(0, 255, 0);\n"
                                        f"font: 16pt \"GOST type A\";\n"
                                        "border: 2px solid rgb(0, 255, 0)\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:pressed{\n"
                                        "background-color:  rgb(0, 255, 0);\n"
                                        "color: rgb(0, 0, 0);\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:hover{\n"
                                        "background-color:  rgb(0, 255, 0);\n"
                                        "color: rgb(0, 0, 0);\n"
                                        "}")
        self.pushButton_2.setObjectName("pushButton_6")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Settings"))
        self.label_1.setText(_translate("Dialog", "Настройки"))
        self.label_2.setText(_translate("Dialog", "Язык"))
        self.label_3.setText(_translate("Dialog", "Качество анализа"))
        self.comboBox_1.setItemText(0, _translate("Dialog", "                Русский"))
        self.comboBox_1.setItemText(1, _translate("Dialog", "                English"))
        self.comboBox_1.setItemText(2, _translate("Dialog", "                Deutch"))
        self.comboBox_2.setItemText(0, _translate("Dialog", "   Высокое (глубокий анализ)"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "             Среднее"))
        self.comboBox_2.setItemText(2, _translate("Dialog", " Низкое(высокая погрешность)"))
        self.pushButton_1.setText(_translate("Dialog", "Закрыть"))
        self.pushButton_2.setText(_translate("Dialog", "Принять"))
