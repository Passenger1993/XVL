# -*- coding: utf-8 -*-
"""Тесты для модуля predictor.py"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
from pathlib import Path

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictor import (
    check_yolo_availability,
    ModelLoader,
    DetectionThread,
    ResultVisualizer
)


class TestCheckYoloAvailability:
    """Тесты для функции проверки доступности YOLO"""

    @patch('src.predictor.YOLO_AVAILABLE', True)
    def test_yolo_available(self):
        """Тест когда YOLO доступен"""
        assert check_yolo_availability() == True

    @patch('src.predictor.YOLO_AVAILABLE', False)
    def test_yolo_not_available(self):
        """Тест когда YOLO недоступен"""
        assert check_yolo_availability() == False


class TestModelLoader:
    """Тесты для класса ModelLoader"""

    def test_initialization(self):
        """Тест инициализации ModelLoader"""
        loader = ModelLoader()
        assert loader.model is None
        assert loader.class_names == {}
        assert loader.model_path is None

    @patch('src.predictor.Path')
    @patch('src.predictor.YOLO_AVAILABLE', True)
    def test_find_model_file_found(self, mock_path):
        """Тест поиска файла модели (успешный)"""
        # Arrange
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        loader = ModelLoader()

        # Act
        result = loader.find_model_file()

        # Assert
        assert result is not None
        mock_path_instance.exists.assert_called()

    @patch('src.predictor.Path')
    def test_find_model_file_not_found(self, mock_path):
        """Тест поиска файла модели (не найден)"""
        # Arrange
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        loader = ModelLoader()

        # Act
        result = loader.find_model_file()

        # Assert
        assert result is None

    @patch('src.predictor.YOLO')
    @patch('src.predictor.YOLO_AVAILABLE', True)
    def test_load_model_success(self, mock_yolo_class, mock_model):
        """Тест успешной загрузки модели"""
        # Arrange
        mock_model_instance = Mock()
        mock_model_instance.names = {0: "defect1", 1: "defect2"}
        mock_yolo_class.return_value = mock_model_instance

        loader = ModelLoader()
        test_model_path = "test_model.pt"

        # Act
        result = loader.load_model(test_model_path)

        # Assert
        assert result == True
        assert loader.model_path == test_model_path
        assert loader.model == mock_model_instance
        assert loader.class_names == {0: "defect1", 1: "defect2"}
        mock_yolo_class.assert_called_once_with(test_model_path)

    @patch('src.predictor.YOLO_AVAILABLE', False)
    def test_load_model_yolo_not_available(self):
        """Тест загрузки модели без YOLO"""
        # Arrange
        loader = ModelLoader()

        # Act & Assert
        with pytest.raises(ImportError, match="Библиотека ultralytics не установлена"):
            loader.load_model("test.pt")

    @patch('src.predictor.YOLO')
    @patch('src.predictor.YOLO_AVAILABLE', True)
    def test_load_model_file_not_found(self, mock_yolo_class):
        """Тест загрузки модели с несуществующим файлом"""
        # Arrange
        mock_yolo_class.side_effect = FileNotFoundError("File not found")
        loader = ModelLoader()

        # Act & Assert
        with pytest.raises(Exception, match="Не удалось загрузить модель"):
            loader.load_model("non_existent.pt")

    @patch('src.predictor.cv2.imread')
    @patch('src.predictor.YOLO_AVAILABLE', True)
    def test_predict_success(self, mock_imread, mock_model):
        """Тест успешного предсказания"""
        # Arrange
        # Mock изображение
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        mock_imread.return_value = test_image

        # Mock модели и результата
        mock_model_instance = Mock()
        mock_result = Mock()
        mock_box = Mock()

        # Настройка mock объектов
        mock_box.xyxy = [[100, 100, 200, 200]]
        mock_box.cls = [0]
        mock_box.conf = [0.85]
        mock_result.boxes = [mock_box]
        mock_model_instance.return_value = [mock_result]

        loader = ModelLoader()
        loader.model = mock_model_instance
        loader.class_names = {0: "porosity"}

        # Act
        img_rgb, boxes, classes, confidences = loader.predict(
            "test_image.jpg",
            confidence_threshold=0.25
        )

        # Assert
        assert img_rgb is not None
        assert len(boxes) == 1
        assert classes == [0]
        assert confidences == [0.85]
        mock_imread.assert_called_once()
        mock_model_instance.assert_called_once()

    @patch('src.predictor.cv2.imread')
    def test_predict_image_not_found(self, mock_imread):
        """Тест предсказания с несуществующим изображением"""
        # Arrange
        mock_imread.return_value = None
        loader = ModelLoader()
        loader.model = Mock()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            loader.predict("non_existent.jpg")

    def test_predict_model_not_loaded(self):
        """Тест предсказания без загруженной модели"""
        # Arrange
        loader = ModelLoader()

        # Act & Assert
        with pytest.raises(ValueError, match="Модель не загружена"):
            loader.predict("test.jpg")


class TestDetectionThread:
    """Тесты для DetectionThread"""

    def test_thread_initialization(self):
        """Тест инициализации потока"""
        # Arrange
        mock_model_loader = Mock()
        image_path = "test.jpg"
        confidence = 0.3

        # Act
        thread = DetectionThread(mock_model_loader, image_path, confidence)

        # Assert
        assert thread.model_loader == mock_model_loader
        assert thread.image_path == image_path
        assert thread.confidence_threshold == confidence

    @patch('src.predictor.ModelLoader')
    def test_run_success(self, MockModelLoader):
        """Тест успешного выполнения потока"""
        # Arrange
        mock_loader_instance = Mock()
        mock_loader_instance.predict.return_value = (
            np.zeros((100, 100, 3), dtype=np.uint8),
            [[10, 10, 50, 50]],
            [0],
            [0.9]
        )

        thread = DetectionThread(mock_loader_instance, "test.jpg")
        thread.detection_finished = Mock()

        # Act
        thread.run()

        # Assert
        mock_loader_instance.predict.assert_called_once_with(
            "test.jpg",
            0.25  # default confidence
        )
        thread.detection_finished.emit.assert_called_once()

    @patch('src.predictor.ModelLoader')
    def test_run_with_error(self, MockModelLoader):
        """Тест выполнения потока с ошибкой"""
        # Arrange
        mock_loader_instance = Mock()
        mock_loader_instance.predict.side_effect = Exception("Test error")

        thread = DetectionThread(mock_loader_instance, "test.jpg")
        thread.detection_error = Mock()

        # Act
        thread.run()

        # Assert
        thread.detection_error.emit.assert_called_once_with(
            "Ошибка детекции: Test error"
        )


class TestResultVisualizer:
    """Тесты для ResultVisualizer"""

    def test_get_statistics_text_no_defects(self):
        """Тест статистики без дефектов"""
        # Act
        result = ResultVisualizer.get_statistics_text([], {})

        # Assert
        assert result == "Дефектов не обнаружено"

    def test_get_statistics_text_with_defects(self):
        """Тест статистики с дефектами"""
        # Arrange
        classes = [0, 1, 0, 2]
        class_names = {
            0: "porosity",
            1: "crack",
            2: "inclusion"
        }

        # Act
        result = ResultVisualizer.get_statistics_text(classes, class_names)

        # Assert
        assert "porosity: 2" in result
        assert "crack: 1" in result
        assert "inclusion: 1" in result

    def test_draw_boxes_empty(self):
        """Тест рисования bounding boxes без дефектов"""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        result = ResultVisualizer.draw_boxes(
            image, [], [], [], {}
        )

        # Assert
        assert result.shape == image.shape
        np.testing.assert_array_equal(result, image)

    @patch('src.predictor.ImageFont')
    def test_draw_boxes_with_defects(self, mock_font):
        """Тест рисования bounding boxes с дефектами"""
        # Arrange
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        boxes = [[50, 50, 150, 150]]
        classes = [0]
        confidences = [0.85]
        class_names = {0: "porosity"}

        mock_font_instance = Mock()
        mock_font.truetype.return_value = mock_font_instance

        # Act
        result = ResultVisualizer.draw_boxes(
            image, boxes, classes, confidences, class_names
        )

        # Assert
        assert result.shape == image.shape
        # Проверяем, что изображение изменилось
        assert not np.array_equal(result, image)

    def test_color_selection(self):
        """Тест выбора цветов для классов"""
        # Проверяем, что цвета определены
        assert len(ResultVisualizer.COLORS) > 0
        assert all(len(color) == 3 for color in ResultVisualizer.COLORS)
        assert all(0 <= c <= 255 for color in ResultVisualizer.COLORS for c in color)