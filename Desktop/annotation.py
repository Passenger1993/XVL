from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from settings import *
from settings import window_size


class annotation_window(QWidget):
    def setupUi(self, Dialog):
        global window_size
        self.button_size = 17
        Dialog.setObjectName("Dialog")
        self.window_size = window_size
        Dialog.resize(self.window_size[0], self.window_size[1])
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        Dialog.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(550,10,220,35))
        self.label.setStyleSheet("font: 24pt \"GOST type A\";\n"
                                 "color: rgb(0, 255, 0);")
        self.label.setObjectName("label")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(580,680,200,50))
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
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(560,360,250,250))
        self.label_2.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("Обложка/Ярлык.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(50,80,1200,300))
        self.label_3.setStyleSheet("color: rgb(0, 255, 0);\n"
                                   f"font: {self.button_size}pt \"GOST type A\";")
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Annotation"))
        self.label.setText(_translate("Dialog", " О программе"))
        self.pushButton_5.setText(_translate("Dialog", "Закрыть"))
        self.label_3.setText(_translate("Dialog",
                                        "Приветствуем вас в X-Ray Vision Lab! Это программа позволяет обнаруживать дефекты сварных швов и соединений,\n"
                                        " используя в качестве основного инструмента обученную нейросеть CNN3-130 softmax. Программа способна визуально\n"
                                        " зарегистрировать 4 вида дефектов: непровар, трещина, скопления пор и одиночные включения. Для того, чтобы получить\n"
                                        " изображение, вам необходимо загрузить его через панель управления, далее дождаться загрузки и на выходе получить\n"
                                        " обработанное изображение с \"отмеченными\" областями дефектов.\n"
                                        "\n"
                                        "Программа предельно проста. Удобный интерфейс и оформление нацелено на широкий круг пользователей. В качестве\n"
                                        " примера изображений вы можете взять образцы во вкладке \"Архив изображений\".\n"
                                        "\n"
                                        "Мы надеемся, что наш продукт прийдётся вам по душе, и Vision Lab прольёт свет на все изъяны ваших изделий....  "))
