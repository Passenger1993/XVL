from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import *
from settings import *
from settings import window_size


class archieve_directory(QDialog):
    def setupUi(self, Dialog):
        global window_size
        Dialog.setObjectName("Dialog")
        Dialog.resize(window_size[0], window_size[1])
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        Dialog.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(Dialog)  # Заголовок
        self.label.setGeometry(QtCore.QRect(580, 10, 310, 80))
        self.label.setStyleSheet("color: rgb(0, 255, 0);\n"
                                 "font: 24pt \"GOST type A\";")
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Dialog)  # Картинка
        self.label_2.setGeometry(QtCore.QRect(20, 150, 650, 430))
        self.label_2.setPixmap(QtGui.QPixmap("Обложка\Т11.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(690, 150, 150, 100))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(lambda: self.change_image("venv/Обложка/Т11.png", "Трещина", self.pushButton))
        self.pixmap = QPixmap("Обложка\Т11.png")
        self.pushButton.setIcon(QIcon(self.pixmap.scaled(QtCore.QSize(146, 96))))
        self.pushButton.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(860, 150, 150, 100))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(lambda: self.change_image("venv/Обложка/Т21.png", "Трещина", self.pushButton_2))
        self.pixmap_2 = QPixmap("Обложка\Т21.png")
        self.pushButton_2.setIcon(QIcon(self.pixmap_2.scaled(QtCore.QSize(146, 96))))
        self.pushButton_2.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(1030, 150, 150, 100))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(
            lambda: self.change_image("venv/Обложка/Т31.png", "Трещина", self.pushButton_3))
        self.pixmap_3 = QPixmap("Обложка\Т31.png")
        self.pushButton_3.setIcon(QIcon(self.pixmap_3.scaled(QtCore.QSize(146, 96))))
        self.pushButton_3.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(1200, 150, 150, 100))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(
            lambda: self.change_image("venv/Обложка/Т41.png", "Трещина", self.pushButton_4))
        self.pixmap_4 = QPixmap("Обложка\Т41.png")
        self.pushButton_4.setIcon(QIcon(self.pixmap_4.scaled(QtCore.QSize(146, 96))))
        self.pushButton_4.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_6 = QtWidgets.QPushButton(Dialog)
        self.pushButton_6.setGeometry(QtCore.QRect(690, 260, 150, 100))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(
            lambda: self.change_image("venv/Обложка/C120.png", "Скопление пор", self.pushButton_6))
        self.pixmap_6 = QPixmap("Обложка\C120.png")
        self.pushButton_6.setIcon(QIcon(self.pixmap_6.scaled(QtCore.QSize(146, 96))))
        self.pushButton_6.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_7 = QtWidgets.QPushButton(Dialog)
        self.pushButton_7.setGeometry(QtCore.QRect(860, 260, 150, 100))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(
            lambda: self.change_image("venv/Обложка/C130.png", "Скопление пор", self.pushButton_7))
        self.pixmap_7 = QPixmap("Обложка\C130.png")
        self.pushButton_7.setIcon(QIcon(self.pixmap_7.scaled(QtCore.QSize(146, 96))))
        self.pushButton_7.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_8 = QtWidgets.QPushButton(Dialog)
        self.pushButton_8.setGeometry(QtCore.QRect(1030, 260, 150, 100))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.clicked.connect(
            lambda: self.change_image("venv/Обложка/C140.png", "Скопление пор", self.pushButton_8))
        self.pixmap_8 = QPixmap("Обложка\C140.png")
        self.pushButton_8.setIcon(QIcon(self.pixmap_8.scaled(QtCore.QSize(146, 96))))
        self.pushButton_8.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_9 = QtWidgets.QPushButton(Dialog)
        self.pushButton_9.setGeometry(QtCore.QRect(1200, 260, 150, 100))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(
            lambda: self.change_image("venv/Обложка/C150.png", "Скопление пор", self.pushButton_9))
        self.pixmap_9 = QPixmap("Обложка\C150.png")
        self.pushButton_9.setIcon(QIcon(self.pixmap_9.scaled(QtCore.QSize(146, 96))))
        self.pushButton_9.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_10 = QtWidgets.QPushButton(Dialog)
        self.pushButton_10.setGeometry(QtCore.QRect(690, 370, 150, 100))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.clicked.connect(
            lambda: self.change_image("venv/Обложка/C160.png", "Скопление пор", self.pushButton_10))
        self.pixmap_10 = QPixmap("Обложка\C160.png")
        self.pushButton_10.setIcon(QIcon(self.pixmap_10.scaled(QtCore.QSize(146, 96))))
        self.pushButton_10.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_11 = QtWidgets.QPushButton(Dialog)
        self.pushButton_11.setGeometry(QtCore.QRect(860, 370, 150, 100))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.clicked.connect(
            lambda: self.change_image("venv/Обложка/Н1.png", "Непровар", self.pushButton_11))
        self.pixmap_11 = QPixmap("Обложка\Н1.png")
        self.pushButton_11.setIcon(QIcon(self.pixmap_11.scaled(QtCore.QSize(146, 96))))
        self.pushButton_11.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_12 = QtWidgets.QPushButton(Dialog)
        self.pushButton_12.setGeometry(QtCore.QRect(1030, 370, 150, 100))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.clicked.connect(
            lambda: self.change_image("venv/Обложка/Н11.png", "Непровар", self.pushButton_12))
        self.pixmap_12 = QPixmap("Обложка\Н11.png")
        self.pushButton_12.setIcon(QIcon(self.pixmap_12.scaled(QtCore.QSize(146, 96))))
        self.pushButton_12.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_13 = QtWidgets.QPushButton(Dialog)
        self.pushButton_13.setGeometry(QtCore.QRect(1200, 370, 150, 100))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.clicked.connect(
            lambda: self.change_image("venv/Обложка/Н21.png", "Непровар", self.pushButton_13))
        self.pixmap_13 = QPixmap("Обложка\Н21.png")
        self.pushButton_13.setIcon(QIcon(self.pixmap_13.scaled(QtCore.QSize(146, 96))))
        self.pushButton_13.setIconSize(QtCore.QSize(146, 96))


        self.pushButton_14 = QtWidgets.QPushButton(Dialog)
        self.pushButton_14.setGeometry(QtCore.QRect(690, 480, 150, 100))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_14.clicked.connect(
            lambda: self.change_image("venv/Обложка/Н1039.png", "Непровар", self.pushButton_14))
        self.pixmap_14 = QPixmap("Обложка\Н1039.png")
        self.pushButton_14.setIcon(QIcon(self.pixmap_14))
        self.pushButton_14.setIcon(QIcon(self.pixmap_14.scaled(QtCore.QSize(146, 96))))
        self.pushButton_14.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_15 = QtWidgets.QPushButton(Dialog)
        self.pushButton_15.setGeometry(QtCore.QRect(860, 480, 150, 100))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.clicked.connect(
            lambda: self.change_image("venv/Обложка/Н1040.png", "Непровар", self.pushButton_15))
        self.pixmap_15 = QPixmap("Обложка\Н1040.png")
        self.pushButton_15.setIcon(QIcon(self.pixmap_15.scaled(QtCore.QSize(146, 96))))
        self.pushButton_15.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_16 = QtWidgets.QPushButton(Dialog)
        self.pushButton_16.setGeometry(QtCore.QRect(1030, 480, 150, 100))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_16.clicked.connect(
            lambda: self.change_image("venv/Обложка/О900.png", "Одиночное включение", self.pushButton_16))
        self.pixmap_16 = QPixmap("Обложка\О900.png")
        self.pushButton_16.setIcon(QIcon(self.pixmap_16.scaled(QtCore.QSize(146, 96))))
        self.pushButton_16.setIconSize(QtCore.QSize(146, 96))

        self.pushButton_17 = QtWidgets.QPushButton(Dialog)
        self.pushButton_17.setGeometry(QtCore.QRect(1200, 480, 150, 100))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_17.clicked.connect(
            lambda: self.change_image("venv/Обложка/О910.png", "Одиночное включение", self.pushButton_17))
        self.pixmap_17 = QPixmap("Обложка\О910.png")
        self.pushButton_17.setIcon(QIcon(self.pixmap_17.scaled(QtCore.QSize(146, 96))))
        self.pushButton_17.setIconSize(QtCore.QSize(146, 96))

        self.label_18 = QtWidgets.QLabel(Dialog)
        self.label_18.setGeometry(QtCore.QRect(30, 525, 100, 50))
        self.label_18.setStyleSheet("color: rgb(0, 255, 0);\n"
                                    "font: 17pt \"GOST type A\";\n"
                                    "border:2px solid rgb(0, 255, 0)")
        self.label_18.setObjectName("label_18")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(650, 700, 200, 50))
        self.pushButton_5.setStyleSheet("\n"
                                        "QPushButton{\n"
                                        "color: rgb(0, 255, 0);\n"
                                        "font: 16pt \"GOST type A\";\n"
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
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(lambda: self.change_image(None,None,None,1))
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def change_image(self, image_path, name, button, terminator = None):
        self.button_list = [self.pushButton,self.pushButton_2,self.pushButton_3,self.pushButton_4,
                           self.pushButton_6,self.pushButton_7,self.pushButton_8,self.pushButton_9,
                           self.pushButton_10,self.pushButton_11,self.pushButton_12,self.pushButton_13,
                           self.pushButton_14,self.pushButton_15,self.pushButton_16,self.pushButton_17]
        if not terminator:
            _translate = QtCore.QCoreApplication.translate
            self.label_2.setPixmap(QtGui.QPixmap(image_path))
            "self.label_2.setPixmap(QtGui.QPixmap(image_path))"
            self.label_18.setText(_translate("Dialog", name))
            border_sizes = {"Непровар":100,"Скопление пор":155,"Одиночное включение":220,"Без дефектов":180,"Трещина":100}
            self.label_18.resize(border_sizes[name],50)
            for page in self.button_list:
                if page == button:
                    page.setStyleSheet("border: 2px solid rgb(0, 255, 0); padding: 10px;")
                else:
                    page.setStyleSheet("border: 2px solid transparent; padding: 10px;")
        else:
            for page in self.button_list:
                page.setStyleSheet("border: 2px solid transparent; padding: 10px;")

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "image_archieve"))
        self.label.setText(_translate("Dialog", "Архив изображений"))
        self.label_18.setText(_translate("Dialog", "Непровар"))
        self.pushButton_5.setText(_translate("Dialog", "Закрыть"))
