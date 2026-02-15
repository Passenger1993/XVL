# -*- coding: utf-8 -*-
"""Главный файл приложения для детекции дефектов сварки"""

import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from src import MainWindow
from src import check_yolo_availability


def main():
    """Главная функция приложения"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Проверяем доступность библиотек
    if not check_yolo_availability():
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