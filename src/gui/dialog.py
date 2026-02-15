# -*- coding: utf-8 -*-
"""Графический интерфейс приложения"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

from src import ModelLoader, DetectionThread, ResultVisualizer


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()

        # Настройка окна
        self.setWindowTitle("XVL - Детекция дефектов сварки")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: rgb(0,0,0); color: rgb(0,255,0);")

        # Инициализация компонентов
        self.model_loader = ModelLoader()
        self.result_visualizer = ResultVisualizer()
        self.current_image_path = None
        self.current_results = None

        # Создание интерфейса
        self._init_ui()

        # Загрузка модели
        self._load_model()

    def _init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный вертикальный лейаут
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Верхняя панель с кнопками
        main_layout.addLayout(self._create_top_panel())

        # Область отображения изображения
        main_layout.addWidget(self._create_image_display(), 1)

        # Панель статуса
        main_layout.addLayout(self._create_status_panel())

        # Нижняя панель
        main_layout.addLayout(self._create_bottom_panel())

    def _create_top_panel(self):
        """Создает верхнюю панель с кнопками"""
        top_layout = QHBoxLayout()

        self.load_button = self._create_styled_button(
            "📁 Загрузить фото",
            "#4CAF50",
            150, 40,
            self._load_image
        )

        self.reset_button = self._create_styled_button(
            "🔄 Сбросить",
            "#f0ad4e",
            120, 40,
            self.reset_interface
        )

        self.info_button = self._create_styled_button(
            "ℹ️ Информация",
            "#5bc0de",
            120, 40,
            self._show_info
        )

        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.reset_button)
        top_layout.addWidget(self.info_button)
        top_layout.addStretch()

        return top_layout

    def _create_image_display(self):
        """Создает область для отображения изображений"""
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: rgb(0,0,0);
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        self.image_label.setText("Загрузите изображение для анализа")
        self.image_label.setFont(QFont("Arial", 14))

        return self.image_label

    def _create_status_panel(self):
        """Создает панель статуса"""
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Готов к работе")
        self.status_label.setFont(QFont("Arial", 10))

        self.detection_label = QLabel("Дефектов не обнаружено")
        self.detection_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.detection_label.setStyleSheet("color: #666;")

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.detection_label)

        return status_layout

    def _create_bottom_panel(self):
        """Создает нижнюю панель с кнопками"""
        bottom_layout = QHBoxLayout()

        self.close_button = self._create_styled_button(
            "✖ Закрыть и вернуться",
            "#d9534f",
            200, 50,
            self.reset_interface
        )
        self.close_button.setEnabled(False)

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.close_button)
        bottom_layout.addStretch()

        return bottom_layout

    def _create_styled_button(self, text, color, width, height, callback):
        """Создает стилизованную кнопку"""
        button = QPushButton(text)
        button.setFixedSize(width, height)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: {14 if height <= 40 else 16}px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 20)};
            }}
        """)
        button.clicked.connect(callback)
        return button

    def _darken_color(self, hex_color, percent=10):
        """Затемняет цвет на указанный процент"""
        import colorsys
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        new_l = max(0, hls[1] - percent/100)
        new_rgb = colorsys.hls_to_rgb(hls[0], new_l, hls[2])
        return f'rgb({int(new_rgb[0]*255)},{int(new_rgb[1]*255)},{int(new_rgb[2]*255)})'

    def _load_model(self):
        """Загружает модель YOLO"""
        try:
            success = self.model_loader.load_model()
            if success:
                QMessageBox.information(
                    self,
                    "Успех",
                    f"Модель загружена успешно!\nКлассов: {len(self.model_loader.class_names)}"
                )
        except FileNotFoundError:
            reply = QMessageBox.question(
                self, "Модель не найдена",
                "Файл модели best.pt не найден.\nХотите указать путь вручную?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._select_model_file()
            else:
                QMessageBox.warning(
                    self, "Внимание",
                    "Модель не загружена. Функционал детекции недоступен."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка загрузки модели",
                f"Не удалось загрузить модель:\n{str(e)}"
            )

    def _select_model_file(self):
        """Позволяет пользователю выбрать файл модели"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели",
            str(Path.home()),
            "Модели PyTorch (*.pt);;Все файлы (*)"
        )

        if file_path:
            try:
                self.model_loader.load_model(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель: {str(e)}")

    def _load_image(self):
        """Открывает проводник для выбора изображения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение",
            str(Path.home()),
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tiff);;Все файлы (*)"
        )

        if file_path:
            self._process_image(file_path)

    def _process_image(self, image_path):
        """Обрабатывает выбранное изображение"""
        self.current_image_path = image_path

        # Обновляем статус
        self._update_status("Анализ изображения...", "#f0ad4e")

        # Загружаем и показываем оригинальное изображение
        self._display_image(image_path)

        # Запускаем детекцию в отдельном потоке
        if self.model_loader.model is not None:
            self.detection_thread = DetectionThread(
                self.model_loader,
                image_path,
                confidence_threshold=0.25
            )
            self.detection_thread.detection_finished.connect(self._on_detection_finished)
            self.detection_thread.detection_error.connect(self._on_detection_error)
            self.detection_thread.start()
        else:
            QMessageBox.warning(self, "Ошибка", "Модель не загружена!")
            self._update_status("Модель не загружена", "#d9534f")

    def _display_image(self, image_path):
        """Отображает изображение в интерфейсе"""
        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_label.size() * 0.9,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 10px;")

    def _on_detection_finished(self, image, boxes, classes, confidences):
        """Обрабатывает завершение детекции"""
        # Сохраняем результаты
        self.current_results = (boxes, classes, confidences)

        # Рисуем bounding boxes на изображении
        result_image = self.result_visualizer.draw_boxes(
            image, boxes, classes, confidences, self.model_loader.class_names
        )

        # Отображаем результат
        self._display_result_image(result_image)

        # Обновляем статус
        num_defects = len(boxes)
        if num_defects > 0:
            self._update_status(
                f"✅ Анализ завершен. Обнаружено дефектов: {num_defects}",
                "#4CAF50"
            )

            # Показываем статистику
            stats_text = self.result_visualizer.get_statistics_text(
                classes, self.model_loader.class_names
            )
            self.detection_label.setText(stats_text)
            self.detection_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        else:
            self._update_status(
                "✅ Анализ завершен. Дефекты не обнаружены",
                "#4CAF50"
            )
            self.detection_label.setText("Дефектов не обнаружено")
            self.detection_label.setStyleSheet("color: #5bc0de; font-weight: bold;")

        # Активируем кнопку закрыть
        self.close_button.setEnabled(True)

    def _on_detection_error(self, error_message):
        """Обрабатывает ошибки детекции"""
        self._update_status("❌ Ошибка анализа", "#d9534f")
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

    def _display_result_image(self, image_array):
        """Отображает результат в виде QPixmap"""
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Масштабируем для отображения
        scaled_pixmap = pixmap.scaled(
            self.image_label.size() * 0.9,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def _update_status(self, text, color):
        """Обновляет текст статуса"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def reset_interface(self):
        """Сбрасывает интерфейс к начальному состоянию"""
        self.current_image_path = None
        self.current_results = None

        self.image_label.clear()
        self.image_label.setText("Загрузите изображение для анализа")
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: rgb(0,0,0);
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        self.image_label.setFont(QFont("Arial", 14))

        self._update_status("Готов к работе", "rgb(0,255,0)")

        self.detection_label.setText("Дефектов не обнаружено")
        self.detection_label.setStyleSheet("color: #666;")

        self.close_button.setEnabled(False)

    def _show_info(self):
        """Показывает информацию о программе"""
        info_text = """
        <h3>XVL - Детекция дефектов сварки</h3>
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

        if self.model_loader.model_path:
            info_text += f"<p>Модель: {Path(self.model_loader.model_path).name}</p>"

        if self.model_loader.class_names:
            info_text += "<p>Классы:</p><ul>"
            for class_id, class_name in self.model_loader.class_names.items():
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


# Для прямого запуска GUI (если нужно)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())