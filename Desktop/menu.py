from PyQt5 import QtWidgets, QtCore , QtGui
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import *
from settings import *
from settings import window_size
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class MenuProcess(QMainWindow):
    def setupUi(self, MainWindow):
        global window_size
        self.button_size = 18
        MainWindow.setObjectName("MainWindow")
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        MainWindow.resize(window_size[0], window_size[1])
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        MainWindow.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.label.setGeometry(QtCore.QRect(420, 10,700, 50))
        self.label.setStyleSheet(f"font: {self.button_size+6}pt \"GOST type A\";\n"
                                 "color: rgb(0, 255, 0);"
                                 "background-color: rgba(0, 0, 0, 100);")
        self.label.setObjectName("label")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.label_2 = QVideoWidget(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 70, 500, 300))
        self.label_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_2.setObjectName("label_2")


        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.label_2)
        self.media_content = QMediaContent(QUrl.fromLocalFile("wallpapers.mp4"))
        self.media_player.setMedia(self.media_content)
        self.media_player.play()

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setSizePolicy(self.pushButton.sizePolicy().horizontalPolicy(), self.pushButton.sizePolicy().verticalPolicy())
        self.pushButton.setGeometry(
            QtCore.QRect(1000, 140, 300,50))
        self.pushButton.setStyleSheet("QPushButton{\n"
f"font: {self.button_size}pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);\n"
"border: 2px solid rgb(0, 255, 0)\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"background-color:  rgb(0, 255, 0);\n"
"color: rgb(0, 0, 0);\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setSizePolicy(self.pushButton_2.sizePolicy().horizontalPolicy(), self.pushButton_2.sizePolicy().verticalPolicy())
        self.pushButton_2.setGeometry(QtCore.QRect(1000, 230, 300, 50))
        self.pushButton_2.setStyleSheet("QPushButton{\n"
f"font: {self.button_size}pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);\n"
"border: 2px solid rgb(0, 255, 0)\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"background-color:  rgb(0, 255, 0);\n"
"color: rgb(0, 0, 0);\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setSizePolicy(self.pushButton_3.sizePolicy().horizontalPolicy(), self.pushButton_3.sizePolicy().verticalPolicy())
        self.pushButton_3.setGeometry(
            QtCore.QRect(1000,320,300,50))
        self.pushButton_3.setStyleSheet("QPushButton{\n"
f"font: {self.button_size}pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);\n"
"border: 2px solid rgb(0, 255, 0)\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"background-color:  rgb(0, 255, 0);\n"
"color: rgb(0, 0, 0);\n"
"}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setSizePolicy(self.pushButton_4.sizePolicy().horizontalPolicy(), self.pushButton_4.sizePolicy().verticalPolicy())
        self.pushButton_4.setGeometry(
            QtCore.QRect(1000, 410, 300,50))
        self.pushButton_4.setStyleSheet("QPushButton{\n"
f"font: {self.button_size}pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);\n"
"border: 2px solid rgb(0, 255, 0)\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"background-color:  rgb(0, 255, 0);\n"
"color: rgb(0, 0, 0);\n"
"}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setSizePolicy(self.pushButton_5.sizePolicy().horizontalPolicy(), self.pushButton_5.sizePolicy().verticalPolicy())
        self.pushButton_5.setGeometry(
            QtCore.QRect(1000,500,300 ,50))
        self.pushButton_5.setStyleSheet("QPushButton{\n"
f"font: {self.button_size}pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);\n"
"border: 2px solid rgb(0, 255, 0)\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"background-color:  rgb(0, 255, 0);\n"
"color: rgb(0, 0, 0);\n"
"}")
        self.pushButton_5.setObjectName("pushButton_5")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.textBrowser.setGeometry(QtCore.QRect(1000, 700, 150 ,60))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.textBrowser.setAlignment(QtCore.Qt.AlignLeft)
        self.label.setAlignment(QtCore.Qt.AlignLeft)
        self.central_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.title_layout = QtWidgets.QVBoxLayout()
        self.button_layout = QtWidgets.QVBoxLayout()
        self.mark_layout = QtWidgets.QVBoxLayout()
        self.title_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        self.button_layout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)

        self.middle_layout = QtWidgets.QHBoxLayout()
        self.middle_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.middle_layout.addWidget(self.label_2)
        self.middle_layout.addLayout(self.button_layout)

        self.mark_layout.setAlignment(QtCore.Qt.AlignRight)
        self.button_layout.addWidget(self.pushButton)
        self.button_layout.addSpacing(15)
        self.button_layout.addWidget(self.pushButton_2)
        self.button_layout.addSpacing(15)
        self.button_layout.addWidget(self.pushButton_3)
        self.button_layout.addSpacing(15)
        self.button_layout.addWidget(self.pushButton_4)
        self.button_layout.addSpacing(15)
        self.button_layout.addWidget(self.pushButton_5)
        self.title_layout.addWidget(self.label)
        self.mark_layout.addWidget(self.textBrowser)
        self.central_layout.addLayout(self.title_layout)
        self.central_layout.addLayout(self.middle_layout)
        self.central_layout.addLayout(self.mark_layout)
        self.textBrowser.setFixedHeight(76)
        self.textBrowser.setFixedWidth(144)
        self.label.setFixedHeight(36)
        self.label.setFixedWidth(550)

        MainWindow.resizeEvent = self.onResize

    def onResize(self, event):
        widget_height = int(self.centralwidget.height())
        widget_width = int(self.centralwidget.width())
        for i in range(0,self.button_layout.count(),2):
            widget = self.button_layout.itemAt(i).widget()
            widget.setFixedHeight(int(widget_height*0.1))
            widget.setFixedWidth(int(widget_width/4))
            widget.setStyleSheet(("QPushButton{\n"
f"font: {int(widget_height/45)}pt \"GOST type A\";\n"
"color: rgb(0, 255, 0);\n"
"border: 2px solid rgb(0, 255, 0)\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"background-color:  rgb(0, 255, 0);\n"
"color: rgb(0, 0, 0);\n"
"}"))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "X-Ray"))
        self.label.setText(_translate("MainWindow", "Добро пожаловать в X-Ray Vision Lab!"))
        self.pushButton.setText(_translate("MainWindow", "Загрузить изображение"))
        self.pushButton_2.setText(_translate("MainWindow", "Архив дефектов"))
        self.pushButton_3.setText(_translate("MainWindow", "О программе"))
        self.pushButton_4.setText(_translate("MainWindow", "Настройки"))
        self.pushButton_5.setText(_translate("MainWindow", "Выход"))
        self.textBrowser.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            f"</style></head><body style=\" font-family:\'GOST type A\'; font-size:20pt; font-weight:400; font-style:normal;\">\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; color:#00ff00;\">         V 2.0 (Beta)</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; color:#00ff00;\">All rights reserved</span></p></body></html>"))
