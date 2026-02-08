# -*- coding: utf-8 -*-
"""test_interface.py
Простой интерфейс для тестирования модели YOLOv8 на дефектах сварки
"""

import sys
import os
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

# Импорт YOLO (убедитесь, что ultralytics установлен)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Установите ultralytics: pip install ultralytics")


class DetectionThread(QThread):
    """Поток для выполнения детекции, чтобы не блокировать интерфейс"""
    detection_finished = pyqtSignal(np.ndarray, list, list, list)  # изображение, боксы, классы, уверенность
    detection_error = pyqtSignal(str)

    def __init__(self, model_path, image_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path

    def run(self):
        try:
            # Загружаем модель
            model = YOLO(self.model_path)

            # Загружаем изображение
            img = cv2.imread(str(self.image_path))
            if img is None:
                self.detection_error.emit(f"Не удалось загрузить изображение: {self.image_path}")
                return

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Выполняем предсказание
            results = model(img_rgb, conf=0.25)  # порог уверенности 25%

            # Извлекаем результаты
            boxes = []
            classes = []
            confidences = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        # Координаты
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([x1, y1, x2, y2])

                        # Класс
                        class_id = int(box.cls[0])
                        classes.append(class_id)

                        # Уверенность
                        conf = float(box.conf[0])
                        confidences.append(conf)

            # Отправляем результаты
            self.detection_finished.emit(img_rgb, boxes, classes, confidences)

        except Exception as e:
            self.detection_error.emit(f"Ошибка детекции: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Настройка окна
        self.setWindowTitle("Детектор дефектов сварки")
        self.setGeometry(100, 100, 1200, 800)

        # Переменные
        self.model = None
        self.current_image = None
        self.current_results = None
        self.model_path = self.find_model_file()

        # Создаем интерфейс
        self.init_ui()

        # Загружаем модель
        self.load_model()

    def find_model_file(self):
        """Ищет файл модели в различных местах"""
        path = Path("C:/PycharmProjects/XVL/src/model/best.pt")

        if path.exists():
            print(f"✅ Найдена модель: {path}")
            return str(path)

        # Если модель не найдена, предложим выбрать
        return None

    def load_model(self):
        """Загружает модель YOLO"""
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "Ошибка",
                               "Библиотека ultralytics не установлена!\n"
                               "Установите: pip install ultralytics")
            return False

        if self.model_path is None:
            reply = QMessageBox.question(self, "Модель не найдена",
                                       "Файл модели best.pt не найден.\n"
                                       "Хотите указать путь вручную?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.select_model_file()
            else:
                QMessageBox.warning(self, "Внимание",
                                  "Модель не загружена. Функционал детекции недоступен.")
                return False

        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Модель загружена: {self.model_path}")

            # Показываем информацию о модели
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"📊 Классы модели: {self.class_names}")
            else:
                self.class_names = {}

            QMessageBox.information(self, "Успех",
                                  f"Модель загружена успешно!\n"
                                  f"Классов: {len(self.class_names)}")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки модели",
                               f"Не удалось загрузить модель:\n{str(e)}")
            return False

    def select_model_file(self):
        """Позволяет пользователю выбрать файл модели"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели",
            str(Path.home()),
            "Модели PyTorch (*.pt);;Все файлы (*)"
        )

        if file_path:
            self.model_path = file_path
            self.load_model()

    def init_ui(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный вертикальный лейаут
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Верхняя панель с кнопками
        top_layout = QHBoxLayout()

        self.load_button = QPushButton("📁 Загрузить фото")
        self.load_button.setFixedSize(150, 40)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.load_button.clicked.connect(self.load_image)

        self.reset_button = QPushButton("🔄 Сбросить")
        self.reset_button.setFixedSize(120, 40)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #f0ad4e;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ec971f;
            }
        """)
        self.reset_button.clicked.connect(self.reset_interface)

        self.info_button = QPushButton("ℹ️ Информация")
        self.info_button.setFixedSize(120, 40)
        self.info_button.setStyleSheet("""
            QPushButton {
                background-color: #5bc0de;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #46b8da;
            }
        """)
        self.info_button.clicked.connect(self.show_info)

        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.reset_button)
        top_layout.addWidget(self.info_button)
        top_layout.addStretch()

        main_layout.addLayout(top_layout)

        # Метка для изображения
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        self.image_label.setText("Загрузите изображение для анализа")
        self.image_label.setFont(QFont("Arial", 14))

        main_layout.addWidget(self.image_label, 1)

        # Панель статуса
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Готов к работе")
        self.status_label.setFont(QFont("Arial", 10))

        self.detection_label = QLabel("Дефектов не обнаружено")
        self.detection_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.detection_label.setStyleSheet("color: #666;")

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.detection_label)

        main_layout.addLayout(status_layout)

        # Нижняя панель с кнопками
        bottom_layout = QHBoxLayout()

        self.close_button = QPushButton("✖ Закрыть и вернуться")
        self.close_button.setFixedSize(200, 50)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
            QPushButton:pressed {
                background-color: #ac2925;
            }
        """)
        self.close_button.clicked.connect(self.reset_interface)
        self.close_button.setEnabled(False)

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.close_button)
        bottom_layout.addStretch()

        main_layout.addLayout(bottom_layout)

    def load_image(self):
        """Открывает проводник для выбора изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение",
            str(Path.home()),
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tiff);;Все файлы (*)"
        )

        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        """Обрабатывает выбранное изображение"""
        # Обновляем статус
        self.status_label.setText("Анализ изображения...")
        self.status_label.setStyleSheet("color: #f0ad4e; font-weight: bold;")

        # Загружаем и показываем оригинальное изображение
        self.current_image_path = image_path
        pixmap = QPixmap(image_path)

        # Масштабируем для отображения
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size() * 0.9,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 10px;")

        # Запускаем детекцию в отдельном потоке
        if self.model is not None:
            self.detection_thread = DetectionThread(self.model_path, image_path)
            self.detection_thread.detection_finished.connect(self.on_detection_finished)
            self.detection_thread.detection_error.connect(self.on_detection_error)
            self.detection_thread.start()
        else:
            QMessageBox.warning(self, "Ошибка", "Модель не загружена!")
            self.status_label.setText("Модель не загружена")
            self.status_label.setStyleSheet("color: #d9534f;")

    def on_detection_finished(self, image, boxes, classes, confidences):
        """Обрабатывает завершение детекции"""
        # Сохраняем результаты
        self.current_results = (boxes, classes, confidences)

        # Рисуем bounding boxes на изображении
        result_image = self.draw_boxes(image, boxes, classes, confidences)

        # Конвертируем numpy array в QPixmap
        height, width, channel = result_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Масштабируем для отображения
        scaled_pixmap = pixmap.scaled(
            self.image_label.size() * 0.9,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

        # Обновляем статус
        num_defects = len(boxes)
        if num_defects > 0:
            self.status_label.setText(f"✅ Анализ завершен. Обнаружено дефектов: {num_defects}")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

            # Группируем по классам
            class_counts = {}
            for class_id in classes:
                class_name = self.class_names.get(class_id, f"Дефект {class_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Формируем текст
            stats_text = "Обнаружено: "
            stats_parts = []
            for class_name, count in class_counts.items():
                stats_parts.append(f"{class_name}: {count}")

            self.detection_label.setText(" | ".join(stats_parts))
            self.detection_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        else:
            self.status_label.setText("✅ Анализ завершен. Дефекты не обнаружены")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.detection_label.setText("Дефектов не обнаружено")
            self.detection_label.setStyleSheet("color: #5bc0de; font-weight: bold;")

        # Активируем кнопку закрыть
        self.close_button.setEnabled(True)

    def on_detection_error(self, error_message):
        """Обрабатывает ошибки детекции"""
        self.status_label.setText("❌ Ошибка анализа")
        self.status_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.detection_label.setText("Ошибка")

        QMessageBox.critical(self, "Ошибка анализа", error_message)

        # Сбрасываем изображение
        self.image_label.setText("Ошибка при анализе изображения")
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #d9534f;
                border-radius: 10px;
                color: #d9534f;
            }
        """)

    def draw_boxes(self, image, boxes, classes, confidences):
        """Рисует bounding boxes на изображении"""
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # Цвета для разных классов
        colors = [
            (255, 0, 0),    # Красный
            (0, 255, 0),    # Зеленый
            (0, 0, 255),    # Синий
            (255, 255, 0),  # Желтый
            (255, 0, 255),  # Пурпурный
            (0, 255, 255),  # Голубой
            (255, 165, 0),  # Оранжевый
            (128, 0, 128),  # Фиолетовый
        ]

        # Пытаемся загрузить шрифт
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        for i, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box

            # Выбираем цвет для класса
            color = colors[class_id % len(colors)]

            # Рисуем прямоугольник
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Подготовка текста
            class_name = self.class_names.get(class_id, f"Дефект {class_id}")
            label = f"{class_name}: {confidence:.1%}"

            # Рисуем фон для текста
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill=color)

            # Рисуем текст
            draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

        return np.array(img_pil)

    def reset_interface(self):
        """Сбрасывает интерфейс к начальному состоянию"""
        self.current_image = None
        self.current_results = None

        self.image_label.clear()
        self.image_label.setText("Загрузите изображение для анализа")
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        self.image_label.setFont(QFont("Arial", 14))

        self.status_label.setText("Готов к работе")
        self.status_label.setStyleSheet("color: black;")

        self.detection_label.setText("Дефектов не обнаружено")
        self.detection_label.setStyleSheet("color: #666;")

        self.close_button.setEnabled(False)

    def show_info(self):
        """Показывает информацию о программе"""
        info_text = """
        <h3>Детектор дефектов сварки</h3>
        <p>Программа для обнаружения дефектов сварки с использованием YOLOv8.</p>
        
        <h4>Инструкция:</h4>
        <ol>
            <li>Нажмите кнопку "Загрузить фото"</li>
            <li>Выберите изображение с дефектами сварки</li>
            <li>Дождитесь завершения анализа</li>
            <li>Просмотрите результат с отмеченными дефектами</li>
            <li>Нажмите "Закрыть и вернуться" для анализа нового изображения</li>
        </ol>
        
        <h4>Информация о модели:</h4>
        """

        if self.model_path:
            info_text += f"<p>Модель: {Path(self.model_path).name}</p>"

        if hasattr(self, 'class_names') and self.class_names:
            info_text += "<p>Классы:</p><ul>"
            for class_id, class_name in self.class_names.items():
                info_text += f"<li>{class_id}: {class_name}</li>"
            info_text += "</ul>"

        QMessageBox.information(self, "Информация", info_text)

    def closeEvent(self, event):
        """Обрабатывает закрытие окна"""
        reply = QMessageBox.question(
            self, "Выход",
            "Вы уверены, что хотите выйти?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль

    # Проверяем доступность библиотек
    if not YOLO_AVAILABLE:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Предупреждение")
        msg.setText("Библиотека ultralytics не установлена!")
        msg.setInformativeText(
            "Для работы программы требуется установить библиотеку ultralytics.\n\n"
            "Установите её, выполнив команду:\n"
            "pip install ultralytics\n\n"
            "Продолжить без детекции?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        if msg.exec_() == QMessageBox.No:
            sys.exit(1)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()