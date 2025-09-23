from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from progress.bar import IncrementalBar
import matplotlib.pyplot as plt
from Desktop.image_analysis import *
from settings import *
from Desktop.archieve import *
from Desktop.annotation import *
from PIL import Image
from Desktop.menu import *
import numpy as np
import sys, time
import cv2

class WorkerThread(QThread):
    progress_changed = pyqtSignal(int)

    def run(self):
        for i in range(101):
            time.sleep(0.1)  # имитация длительной операции
            self.progress_changed.emit(i)

class MainProcess():
    def __init__(self):
        s

        self.Dialog_3 = QDialog()
        self.Annotation = annotation_window()
        self.Annotation.setupUi(self.Dialog_3)
        self.Annotation.retranslateUi(self.Dialog_3)

        self.Dialog_4 = QDialog()
        self.Settings = settings_window()
        self.Settings.setupUi(self.Dialog_4)
        self.Settings.retranslateUi(self.Dialog_4)

        # ------------------------------------------Блок переходов------------------------------------------
        self.main_window.pushButton.clicked.connect(self.openload)
        self.main_window.pushButton_2.clicked.connect(lambda: self.transition(self.MainWindow, self.Dialog_2))
        self.main_window.pushButton_3.clicked.connect(lambda: self.transition(self.MainWindow, self.Dialog_3))
        self.main_window.pushButton_4.clicked.connect(lambda: self.transition(self.MainWindow, self.Dialog_4))
        self.View.pushButton_5.clicked.connect(lambda: self.transition(self.Dialog_1_7, self.MainWindow))
        self.Archieve.pushButton_5.clicked.connect(lambda: self.transition(self.Dialog_2, self.MainWindow))
        self.Annotation.pushButton_5.clicked.connect(lambda: self.transition(self.Dialog_3, self.MainWindow))
        self.Settings.pushButton_1.clicked.connect(lambda: self.transition(self.Dialog_4, self.MainWindow))
        self.Settings.pushButton_2.clicked.connect(self.accept_parameters)
        self.main_window.pushButton_5.clicked.connect(sys.exit)

    def transition(self, window_1, window_2):
        window_1.close()
        window_2.show()

    def type_error(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Неверный формат")
        msg.setWindowTitle("Ошибка")
        msg.setStandardButtons(QMessageBox.Close)
        msg.show()
        msg.exec_()

    def accept_parameters(self):
        self.transition(self.Dialog_4, self.MainWindow)

    # ------------------------------------------Особые переходы------------------------------------------
    def openload(self):
        self.Analysis.image = QFileDialog.getOpenFileName(self.Analysis, "D:\\Documents\\Выборка", ".")
        if self.Analysis.image[0][-4:] == ".png" or self.Analysis.image[0][-4:] == ".jpg":
            pixmap = self.load_image(self.Analysis.image[0])
            print(pixmap.size().width())
            image_division = self.split_image(pixmap)

            self.Analysis.label_2.setPixmap(pixmap)
            pic_val = self.graphic_parameter(pixmap)
            pic_dam = self.damage_control(pixmap)
            self.painter = QPainter(pixmap)
            self.pen = QPen(QColor("green"))
            self.pen.setWidth(2)
            self.painter.setPen(self.pen)

            for x in range(0, self.Analysis.label_2.size().width(), 15):
                self.painter.drawLine(x, 0, x, self.Analysis.label_2.size().height())
            for y in range(0, self.Analysis.label_2.size().height(), 10):
                self.painter.drawLine(0, y, self.Analysis.label_2.size().width(), y)

            self.painter.end()
            self.transition(self.MainWindow, self.Dialog_1)
            self.View.retranslateUi(self.View, 4, "Трещина", 3, pic_dam, pic_val[3],
                                    f"{pixmap.size().width()}x{pixmap.size().height()}",
                                    pic_val[0], pic_val[1], pic_val[2])

            self.Analysis.label_2.setPixmap(pixmap.scaled(QtCore.QSize(700, 350)))
            self.Succeful.label_2.setPixmap(pixmap.scaled(QtCore.QSize(700, 350)))
            self.worker_thread = WorkerThread()
            self.worker_thread.progress_changed.connect(self.update_analys)
            self.worker_thread.finished.connect(self.open_analysis)
            self.worker_thread.start()
            self.timer = QTimer(self.Analysis)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(lambda: self.transition(self.Dialog_1_5, self.Dialog_1_7))
            self.View.label_2.setPixmap(QtGui.QPixmap(self.Analysis.image[0]))
            self.View.target_acquisition(40, 125, 520, 100,"Непровар")
            """self.View.target_acquisition(500, 300, 120, 100,"Трещина")"""

        elif self.Analysis.image[0][-4:] == '':
            pass
        else:
            self.type_error()

    def open_analysis(self):
        self.Analysis.accept()
        self.transition(self.Dialog_1, self.Dialog_1_5)
        self.timer.start(3000)

    # ------------------------------------------Блок цветовых параметров------------------------------------------
    def graphic_parameter(self, pixmap):
        image = pixmap.toImage()
        brightness_sum = 0
        contrast_sum = 0
        for x in range(image.width()):
            for y in range(image.height()):
                pixel_color = image.pixelColor(x, y)
                brightness_sum += pixel_color.lightnessF()
                contrast_sum += pixel_color.valueF()
        total_pixels = image.width() * image.height()
        average_brightness = (brightness_sum / (total_pixels+1)) * 100
        average_contrast = (contrast_sum / (total_pixels+1)) * 100
        average_sharpness = self.calculate_image_sharpness(image)

        if (image.width() > 1920) & (image.height() > 1080) & (average_brightness > 70) & (average_contrast > 50) & (
                average_sharpness > 70):
            image_quality = "Высокое"
        elif (image.width() > 800) & (image.height() > 600) & (30 < average_brightness <= 70) & (
                10 < average_contrast <= 50) & (30 < average_sharpness <= 70):
            image_quality = "Среднее"
        else:
            image_quality = "Низкое"
            return (round(average_brightness),round(average_contrast),round(average_sharpness), image_quality)

    def get_pixel_brightness(self, pixel):
        r, g, b, _ = pixel.getRgb()  # Получение значений R, G, B каналов цвета
        brightness = (r + g + b) / 3  # Вычисление яркости пикселя
        return brightness

    def calculate_image_sharpness(self,image):
        arr = self.convert_qimage_to_numpy(image)
        if image is not None and image.depth() == 32:
             gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
             sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
             return sharpness
        else:
            print("Ошибка: Не удалось загрузить изображение")
            return 0

    def load_image(self,path):
        try:
            img = Image.open(path)
            img.verify()  # Проверка корректности изображения
            return QPixmap(path)
        except (IOError, SyntaxError) as e:
            print("Ошибка: Некорректный формат изображения:", e)
            return None

    # ------------------------------------------Блок контроля ущерба------------------------------------------
    def convert_qimage_to_numpy(self,qimage):
        if qimage.isNull():
            return None
        buffer = qimage.bits()
        buffer.setsize(qimage.byteCount())
        arr = np.frombuffer(buffer, np.uint8).reshape(qimage.height(), qimage.width(), 4)
        return arr

    def qimage_to_cv2(self,qimage):
        # Преобразование QImage в массив NumPy
        width = qimage.width()
        height = qimage.height()
        if qimage.format() == QImage.Format_Grayscale8:
            image_data = qimage.bits().asstring(width * height)
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width)
        else:
            image_data = qimage.bits().asstring(width * height * 4)
            image = np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 4)

        # Преобразование формата изображения, если оно цветное
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    def damage_control(self,pixmap):
        qimage = pixmap.toImage()
        image = self.qimage_to_cv2(qimage)

        # Если изображение цветное, конвертируем его в оттенки серого
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Если изображение уже в оттенках серого, используем его

        # Создаем маску для светлой области
        light_area_mask = cv2.inRange(gray, 120 , 255)

        # Создаем маску для темных пикселей
        dark_pixels_mask = cv2.inRange(gray, 0, 120)

        # Выявляем тёмные пиксели внутри светлой области
        dark_pixels_inside_light_area = cv2.bitwise_and(dark_pixels_mask, light_area_mask)

        # Подсчитываем количество темных пикселей внутри светлой области
        counter = np.sum(dark_pixels_inside_light_area == 255)/100

        return round(counter)

    # ------------------------------------------Блок разбиения------------------------------------------
    def split_image(self,pixmap):
        elements = []
        height = pixmap.height()
        width = pixmap.width()
        for y in range(0, height, 256):
            for x in range(0, width, 256):
                if x + 256 > width or y + 256 > height:
                    continue
                element = pixmap.copy(x, y, 256, 256)
                img = element.toImage()
                byte_array = img.constBits()
                byte_array.setsize(256 * 256 * 4)
                arr = np.frombuffer(byte_array, np.uint8).reshape((256, 256, 4))
                elements.append(arr)

        return elements

    # ------------------------------------------Блок обновления------------------------------------------
    def update_analys(self, value):
        self.Analysis.progress_bar.setValue(value)
        self.Analysis.progress_label.setText(f"{value}%")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.processEvents()
    main_thread = MainProcess()
    main_thread.MainWindow.show()

    sys.exit(app.exec_())
