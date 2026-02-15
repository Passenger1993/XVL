# -*- coding: utf-8 -*-
"""Тесты для GUI (dialog.py)"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dialog import MainWindow


class TestMainWindow:
    """Тесты для главного окна приложения"""

    def test_initialization(self, qapp):
        """Тест инициализации главного окна"""
        # Act
        window = MainWindow()

        # Assert
        assert window.windowTitle() == "XVL - Детекция дефектов сварки"
        assert window.current_image_path is None
        assert window.current_results is None
        assert hasattr(window, 'model_loader')
        assert hasattr(window, 'result_visualizer')

    @patch('src.dialog.ModelLoader')
    def test_load_model_success(self, MockModelLoader, qapp):
        """Тест успешной загрузки модели"""
        # Arrange
        mock_loader = Mock()
        mock_loader.load_model.return_value = True
        mock_loader.class_names = {0: "defect1", 1: "defect2"}
        MockModelLoader.return_value = mock_loader

        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

        # Act
        window._load_model()

        # Assert
        mock_loader.load_model.assert_called_once()

    @patch('src.dialog.QMessageBox')
    @patch('src.dialog.ModelLoader')
    def test_load_model_file_not_found(self, MockModelLoader, mock_messagebox, qapp):
        """Тест загрузки модели с отсутствующим файлом"""
        # Arrange
        mock_loader = Mock()
        mock_loader.load_model.side_effect = FileNotFoundError()
        MockModelLoader.return_value = mock_loader

        mock_msg_instance = Mock()
        mock_msg_instance.exec_.return_value = Mock()
        mock_messagebox.question.return_value = mock_msg_instance

        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

        # Act
        window._load_model()

        # Assert
        mock_messagebox.question.assert_called_once()

    def test_create_styled_button(self, qapp):
        """Тест создания стилизованной кнопки"""
        # Arrange
        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

        # Act
        button = window._create_styled_button(
            "Test Button",
            "#FF0000",
            100,
            30,
            lambda: None
        )

        # Assert
        assert button.text() == "Test Button"
        assert button.width() == 100
        assert button.height() == 30

    def test_darken_color(self, qapp):
        """Тест функции затемнения цвета"""
        # Arrange
        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

        # Act
        result = window._darken_color("#FF0000", 20)

        # Assert
        assert result.startswith("rgb(")
        assert result.endswith(")")

    @patch('src.dialog.QFileDialog')
    def test_select_model_file(self, mock_filedialog, qapp):
        """Тест выбора файла модели"""
        # Arrange
        test_path = "/path/to/model.pt"
        mock_filedialog.getOpenFileName.return_value = (test_path, "")

        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()
            window.model_loader = Mock()

        # Act
        window._select_model_file()

        # Assert
        mock_filedialog.getOpenFileName.assert_called_once()
        window.model_loader.load_model.assert_called_once_with(test_path)

    @patch('src.dialog.QFileDialog')
    def test_load_image(self, mock_filedialog, qapp):
        """Тест загрузки изображения"""
        # Arrange
        test_path = "/path/to/image.jpg"
        mock_filedialog.getOpenFileName.return_value = (test_path, "")

        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()
            window._process_image = Mock()

        # Act
        window._load_image()

        # Assert
        mock_filedialog.getOpenFileName.assert_called_once()
        window._process_image.assert_called_once_with(test_path)

    def test_update_status(self, qapp):
        """Тест обновления статуса"""
        # Arrange
        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()
            window.status_label = Mock()

        # Act
        window._update_status("Test status", "#FF0000")

        # Assert
        window.status_label.setText.assert_called_once_with("Test status")
        window.status_label.setStyleSheet.assert_called_once()

    def test_reset_interface(self, qapp):
        """Тест сброса интерфейса"""
        # Arrange
        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

            # Мокаем виджеты
            window.image_label = Mock()
            window.status_label = Mock()
            window.detection_label = Mock()
            window.close_button = Mock()

            window.current_image_path = "/test/path.jpg"
            window.current_results = ([], [], [])

            window._update_status = Mock()

        # Act
        window.reset_interface()

        # Assert
        assert window.current_image_path is None
        assert window.current_results is None

        window.image_label.clear.assert_called_once()
        window.image_label.setText.assert_called_once()
        window.image_label.setStyleSheet.assert_called_once()

        window._update_status.assert_called_once_with("Готов к работе", "rgb(0,255,0)")

        window.detection_label.setText.assert_called_once_with("Дефектов не обнаружено")
        window.close_button.setEnabled.assert_called_once_with(False)

    @patch('src.dialog.QMessageBox')
    def test_show_info(self, mock_messagebox, qapp):
        """Тест показа информации о программе"""
        # Arrange
        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()
            window.model_loader = Mock()
            window.model_loader.model_path = "/path/to/model.pt"
            window.model_loader.class_names = {0: "defect1"}

        # Act
        window._show_info()

        # Assert
        mock_messagebox.information.assert_called_once()

    @patch('src.dialog.QMessageBox')
    def test_close_event_accepted(self, mock_messagebox, qapp):
        """Тест обработки закрытия окна (принято)"""
        # Arrange
        mock_msg_instance = Mock()
        mock_msg_instance.exec_.return_value = Mock()
        mock_messagebox.question.return_value = mock_msg_instance

        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

        mock_event = Mock()

        # Настраиваем mock для QMessageBox.Yes
        from PyQt5.QtWidgets import QMessageBox
        mock_msg_instance.exec_.return_value = QMessageBox.Yes

        # Act
        window.closeEvent(mock_event)

        # Assert
        mock_event.accept.assert_called_once()

    @patch('src.dialog.QMessageBox')
    def test_close_event_rejected(self, mock_messagebox, qapp):
        """Тест обработки закрытия окна (отклонено)"""
        # Arrange
        mock_msg_instance = Mock()
        mock_messagebox.question.return_value = mock_msg_instance

        with patch.object(MainWindow, '_init_ui'):
            window = MainWindow()

        mock_event = Mock()

        # Настраиваем mock для QMessageBox.No
        from PyQt5.QtWidgets import QMessageBox
        mock_msg_instance.exec_.return_value = QMessageBox.No

        # Act
        window.closeEvent(mock_event)

        # Assert
        mock_event.ignore.assert_called_once()