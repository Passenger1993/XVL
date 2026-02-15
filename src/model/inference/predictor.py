# -*- coding: utf-8 -*-
"""Модуль для работы с моделью YOLO и детекцией объектов"""

import os
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Установите ultralytics: pip install ultralytics")


def check_yolo_availability():
    """Проверяет доступность библиотеки YOLO"""
    return YOLO_AVAILABLE


class ModelLoader:
    """Класс для загрузки и управления моделью YOLO"""

    def __init__(self):
        self.model = None
        self.class_names = {}
        self.model_path = None

    def find_model_file(self, default_path="C:/PycharmProjects/XVL/src/model/best.pt"):
        """Ищет файл модели в различных местах"""
        possible_paths = [
            Path(default_path),
            Path("model/best.pt"),
            Path("best.pt"),
            Path.cwd() / "best.pt",
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Найдена модель: {path}")
                return str(path)

        return None

    def load_model(self, model_path=None):
        """Загружает модель YOLO"""
        if not YOLO_AVAILABLE:
            raise ImportError("Библиотека ultralytics не установлена!")

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.find_model_file()

        if not self.model_path:
            raise FileNotFoundError("Файл модели best.pt не найден")

        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Модель загружена: {self.model_path}")

            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"📊 Классы модели: {self.class_names}")

            return True

        except Exception as e:
            raise Exception(f"Не удалось загрузить модель: {str(e)}")

    def predict(self, image_path, confidence_threshold=0.25):
        """Выполняет предсказание на изображении"""
        if self.model is None:
            raise ValueError("Модель не загружена")

        # Загружаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Выполняем предсказание
        results = self.model(img_rgb, conf=confidence_threshold)

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

        return img_rgb, boxes, classes, confidences


class DetectionThread(QThread):
    """Поток для выполнения детекции в фоновом режиме"""
    detection_finished = pyqtSignal(np.ndarray, list, list, list)  # изображение, боксы, классы, уверенность
    detection_error = pyqtSignal(str)

    def __init__(self, model_loader, image_path, confidence_threshold=0.25):
        super().__init__()
        self.model_loader = model_loader
        self.image_path = image_path
        self.confidence_threshold = confidence_threshold

    def run(self):
        try:
            # Выполняем предсказание
            img_rgb, boxes, classes, confidences = self.model_loader.predict(
                self.image_path,
                self.confidence_threshold
            )

            # Отправляем результаты
            self.detection_finished.emit(img_rgb, boxes, classes, confidences)

        except Exception as e:
            self.detection_error.emit(f"Ошибка детекции: {str(e)}")


class ResultVisualizer:
    """Класс для визуализации результатов детекции"""

    COLORS = [
        (255, 0, 0),    # Красный
        (0, 255, 0),    # Зеленый
        (0, 0, 255),    # Синий
        (255, 255, 0),  # Желтый
        (255, 0, 255),  # Пурпурный
        (0, 255, 255),  # Голубой
        (255, 165, 0),  # Оранжевый
        (128, 0, 128),  # Фиолетовый
    ]

    @staticmethod
    def draw_boxes(image, boxes, classes, confidences, class_names):
        """Рисует bounding boxes на изображении"""
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # Загружаем шрифт
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        for i, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box

            # Выбираем цвет для класса
            color = ResultVisualizer.COLORS[class_id % len(ResultVisualizer.COLORS)]

            # Рисуем прямоугольник
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Подготовка текста
            class_name = class_names.get(class_id, f"Дефект {class_id}")
            label = f"{class_name}: {confidence:.1%}"

            # Рисуем фон для текста
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill=color)

            # Рисуем текст
            draw.text((x1, y1), label, fill=(255, 255, 255), font=font)

        return np.array(img_pil)

    @staticmethod
    def get_statistics_text(classes, class_names):
        """Формирует текстовую статистику по результатам"""
        if not classes:
            return "Дефектов не обнаружено"

        class_counts = {}
        for class_id in classes:
            class_name = class_names.get(class_id, f"Дефект {class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        stats_parts = [f"{name}: {count}" for name, count in class_counts.items()]
        return " | ".join(stats_parts)