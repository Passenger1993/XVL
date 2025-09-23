from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap ,QPainter, QColor, QFont, QPen
from settings import *
from settings import window_size
import os

class load_process(QDialog):
    def setupUi(self, Dialog):
        global window_size
        self.load_finish = False
        Dialog.setObjectName("Dialog")
        Dialog.resize(window_size[0], window_size[1])
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        Dialog.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(450, 650, 450, 35))
        self.label.setStyleSheet("color: rgb(0, 255, 0);\n"
                                 "font: 22pt \"GOST type A\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(183, 60, 1000, 570))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.progress_bar = QtWidgets.QProgressBar(Dialog)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setGeometry(QtCore.QRect(183, 700, 1000, 20))
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgb(0, 255, 0);
            }
            QProgressBar::chunk {
                background-color: rgb(0, 255, 0);
            }
        """)
        self.progress_label = QtWidgets.QLabel(Dialog)
        self.progress_label.setGeometry(QtCore.QRect(1190, 700, 40, 20))
        self.progress_label.setStyleSheet("font: 14pt \"GOST type A\";\n"
                                          "color: rgb(0, 255, 0);")
        self.progress_bar.setObjectName("progressBar")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Обработка изображения, ожидайте...."))

class load_succeful(QDialog):
    def setupUi(self, Dialog):
        global window_size
        global window_cords
        Dialog.setObjectName("Dialog")
        Dialog.resize(window_size[0], window_size[1])
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        Dialog.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(540, 650, 450, 35))
        self.label.setStyleSheet("color: rgb(0, 255, 0);\n"
                                 "font: 20pt \"GOST type A\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(183, 60, 1000, 570))
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Загрузка завершена!"))

class analysis_process(QDialog):
    def setupUi(self, Dialog):
        global window_size
        global window_cords
        global scaling_factor
        Dialog.setObjectName("Dialog")
        Dialog.resize(window_size[0], window_size[1])
        Dialog.setStyleSheet("background-color: rgb(0, 0, 0);\n"
                             "font: 8pt \"GOST type A\";")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Обложка/Ярлык.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        Dialog.setWindowIcon(icon)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(560, 20, 320, 41))
        self.label.setStyleSheet("color: rgb(0, 255, 0);\n"
                                 "font: 24pt \"GOST type A\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 80, 920, 570))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")

        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(980, 80, 360, 570))
        self.textBrowser.setStyleSheet("border: 2px solid rgb(0, 255, 0)")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(650, 700, 150, 50))
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
                                        "}")
        self.pushButton_5.setObjectName("pushButton_5")

        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def target_acquisition(self,x,y,width,height, text):
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        self.painter = QPainter(self.label_2.pixmap())
        self.painter.setPen(pen)
        self.painter.drawRect(x, y, width, height)
        self.painter.setFont(QFont("GOST type A", 10))
        self.painter.drawText(x+round(width/2.5), y-3, text)
        self.painter.end()

    def retranslateUi(self, Dialog, defect_num, leader, variation, damage, quality, resolution, bright, constrast,sharpness):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Image analysis"))
        self.label.setText(_translate("Dialog", "Результаты анализа:"))
        self.pushButton_5.setText(_translate("Dialog", "Закрыть"))
        self.textBrowser.setHtml(_translate("Dialog",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            f"</style></head><body style=\" font-family:\'GOST type A\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Дефектов обнаружено: {defect_num}</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Ведущий: {leader}</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Число вариаций дефектов: {variation}</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Степень повреждения: {damage}%</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Качество изображения: {quality}</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Разрешение: {resolution} пкс.</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Яркость: {bright}%</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Контрастность: {constrast}%</span></p>\n"
                                            f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; color:#00ff00;\">Резкость: {sharpness}%</span></p>\n"
                                            "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:18pt; color:#00ff00;\"><br /></p></body></html>"))
