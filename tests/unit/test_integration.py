# -*- coding: utf-8 -*-
"""Интеграционные тесты"""

import sys
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ModelLoader, DetectionThread
from src import MainWindow


class TestIntegration:
    """Интеграционные тесты"""

    @patch('src.predictor.YOLO')
    @patch('src.dialog.ModelLoader')
    def test_full_pipeline(self, MockModelLoader, MockYOLO, qapp, temp_image_file):
        """Тест полного пайплайна от загрузки модели до предсказания"""
        # Arrange
        # Настраиваем mock для YOLO
        mock_yolo_instance = Mock()
        mock_result = Mock()
        mock_box = Mock()

        mock_box.xyxy = [[50, 50, 150, 150]]
        mock_box.cls = [0]
        mock_box.conf = [0.9]
        mock_result.boxes = [mock_box]
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo_instance.names = {0: "porosity"}
        MockYOLO.return_value = mock_yolo_instance

        # Настраиваем ModelLoader
        model_loader = ModelLoader()
        model_loader.model = mock_yolo_instance
        model_loader.class_names = {0: "porosity"}

        # Act & Assert
        # Тест загрузки модели
        with patch.object(ModelLoader, 'find_model_file', return_value="dummy.pt"):
            success = model_loader.load_model()
            assert success == True

        # Тест предсказания
        img_rgb, boxes, classes, confidences = model_loader.predict(
            temp_image_file,
            confidence_threshold=0.25
        )

        assert len(boxes) == 1
        assert classes == [0]
        assert confidences == [0.9]

        # Тест создания GUI
        with patch('src.dialog.ModelLoader', return_value=model_loader):
            window = MainWindow()
            assert window.model_loader == model_loader

    def test_detection_thread_integration(self, temp_image_file):
        """Интеграционный тест DetectionThread"""
        # Arrange
        mock_model_loader = Mock()
        mock_model_loader.predict.return_value = (
            Mock(),  # изображение
            [[10, 10, 50, 50]],  # boxes
            [0],  # classes
            [0.8]  # confidences
        )

        # Создаем поток
        thread = DetectionThread(mock_model_loader, temp_image_file)

        # Mock сигналов
        thread.detection_finished = Mock()

        # Act
        thread.run()

        # Assert
        mock_model_loader.predict.assert_called_once_with(
            temp_image_file,
            0.25
        )
        thread.detection_finished.emit.assert_called_once()