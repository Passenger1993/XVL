"""
XVL Predictor Module
Ядро для загрузки модели с Hugging Face Hub и выполнения предсказаний.
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Импорты для работы с моделью и изображениями
from huggingface_hub import hf_hub_download, snapshot_download
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes для структурированных результатов
# ============================================================================

@dataclass
class Detection:
    """Информация об одном обнаруженном дефекте."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max] в пикселях
    normalized_bbox: List[float]  # [x_center, y_center, width, height] нормализованные
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PredictionResult:
    """Полный результат предсказания для одного изображения."""
    image_path: str
    image_hash: str
    image_size: Tuple[int, int]  # (width, height)
    detections: List[Detection] = field(default_factory=list)
    processing_time: float = 0.0
    model_name: str = ""
    timestamp: str = ""
    
    # Изображения (опционально, хранятся как numpy массивы или PIL Image)
    original_image: Optional[np.ndarray] = None
    annotated_image: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        """Конвертирует результат в словарь."""
        result = {
            "image_path": self.image_path,
            "image_hash": self.image_hash,
            "image_size": self.image_size,
            "detections": [det.to_dict() for det in self.detections],
            "processing_time": self.processing_time,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "summary": self.get_summary()
        }
        if include_images and self.annotated_image is not None:
            result["annotated_image_base64"] = self._image_to_base64(self.annotated_image)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводную статистику."""
        return {
            "total_defects": len(self.detections),
            "defects_by_class": {
                det.class_name: sum(1 for d in self.detections if d.class_name == det.class_name)
                for det in self.detections
            },
            "max_confidence": max([det.confidence for det in self.detections]) if self.detections else 0.0,
            "has_defects": len(self.detections) > 0
        }
    
    def _image_to_base64(self, image_array: np.ndarray) -> str:
        """Конвертирует numpy array в base64 строку (для сериализации)."""
        import base64
        from io import BytesIO
        
        # Конвертируем BGR (OpenCV) в RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
            
        pil_image = Image.fromarray(image_rgb)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def save_annotated_image(self, output_path: str, quality: int = 95) -> str:
        """Сохраняет аннотированное изображение на диск."""
        if self.annotated_image is None:
            raise ValueError("No annotated image available")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Конвертируем BGR в RGB для сохранения
        if len(self.annotated_image.shape) == 3 and self.annotated_image.shape[2] == 3:
            save_image = cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB)
        else:
            save_image = self.annotated_image
            
        pil_image = Image.fromarray(save_image)
        pil_image.save(output_path, quality=quality, optimize=True)
        
        logger.debug(f"Annotated image saved to: {output_path}")
        return str(output_path)
    
    def save_metadata(self, output_path: str) -> str:
        """Сохраняет метаданные предсказания в JSON файл."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = self.to_dict()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Metadata saved to: {output_path}")
        return str(output_path)

# ============================================================================
# Основной класс предсказателя
# ============================================================================

class XVLPredictor:
    """Основной класс для работы с моделью XVL."""
    
    # Цвета для разных классов дефектов (BGR формат для OpenCV)
    CLASS_COLORS = {
        0: (0, 0, 255),    # 'incomplete_fusion' - Красный
        1: (255, 0, 0),    # 'crack' - Синий
        2: (0, 255, 0),    # 'single_pore' - Зеленый
        3: (0, 255, 255),  # 'cluster_pores' - Желтый
        4: (255, 0, 255),  # 'empty' - Пурпурный (для визуализации, если нужно)
    }
    
    # Имена классов (должны соответствовать вашей обученной модели)
    CLASS_NAMES = {
        0: "incomplete_fusion",
        1: "crack", 
        2: "single_pore",
        3: "cluster_pores",
        4: "empty"  # Без дефектов
    }
    
    def __init__(
        self,
        model_repo: str = "yourusername/xvl-weld-defect-detection",  # ← ЗАМЕНИТЕ на ваш!
        model_file: str = "best.pt",  # или "pytorch_model.bin" если конвертировали
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Инициализация предсказателя.
        
        Args:
            model_repo: Имя репозитория на Hugging Face Hub
            model_file: Имя файла с весами модели
            device: Устройство для инференса ('cuda', 'cpu', 'mps' или None для auto)
            confidence_threshold: Порог уверенности для детекций
            iou_threshold: Порог IoU для NMS
        """
        self.model_repo = model_repo
        self.model_file = model_file
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Определяем устройство
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device
        
        self.model = None
        self.class_names = self.CLASS_NAMES
        self._is_loaded = False
        
        logger.info(f"Initializing XVLPredictor:")
        logger.info(f"  Model: {model_repo}/{model_file}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
    
    def _auto_detect_device(self) -> str:
        """Автоматически определяет доступное устройство."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple MPS (Metal) detected")
            return "mps"
        else:
            logger.info("Using CPU")
            return "cpu"
    
    def load_model(self, force_reload: bool = False) -> None:
        """Загружает модель с Hugging Face Hub."""
        if self._is_loaded and not force_reload:
            logger.debug("Model already loaded")
            return
        
        try:
            logger.info(f"Downloading model from Hugging Face Hub: {self.model_repo}")
            
            # Скачиваем модель в локальный кэш
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                cache_dir="models/huggingface",  # Локальная папка для кэша
                force_download=force_reload
            )
            
            logger.info(f"Model downloaded to: {model_path}")
            
            # Загружаем модель через YOLO
            self.model = YOLO(model_path)
            
            # Переопределяем имена классов, если они есть в модели
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
                logger.info(f"Loaded class names from model: {self.class_names}")
            else:
                logger.info(f"Using default class names: {self.class_names}")
            
            self._is_loaded = True
            
            # Тестовый прогон для компиляции (если нужно)
            if self.device == "cuda":
                logger.debug("Running warm-up inference on GPU...")
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Вычисляет хэш изображения для идентификации."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return hashlib.md5(image_bytes).hexdigest()[:12]
    
    def _process_yolo_results(self, results, image_size: Tuple[int, int]) -> List[Detection]:
        """Обрабатывает сырые результаты YOLO в список Detection объектов."""
        detections = []
        
        # YOLO возвращает список Results объектов
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i in range(len(boxes.xyxy)):
                    # Координаты bounding box
                    bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    
                    # Нормализованные координаты (YOLO формат)
                    img_w, img_h = image_size
                    x_center = (bbox[0] + bbox[2]) / 2 / img_w
                    y_center = (bbox[1] + bbox[3]) / 2 / img_h
                    width = (bbox[2] - bbox[0]) / img_w
                    height = (bbox[3] - bbox[1]) / img_h
                    normalized_bbox = [x_center, y_center, width, height]
                    
                    # Информация о классе
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # Пропускаем если уверенность ниже порога
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Получаем имя класса
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Создаем объект Detection
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        normalized_bbox=normalized_bbox
                    )
                    detections.append(detection)
        
        # Сортируем по уверенности (от высокой к низкой)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def _annotate_image(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Рисует bounding boxes и labels на изображении."""
        if len(detections) == 0:
            return image.copy()
        
        annotated = image.copy()
        img_h, img_w = annotated.shape[:2]
        
        for det in detections:
            # Получаем координаты bbox
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Получаем цвет для класса
            color = self.CLASS_COLORS.get(det.class_id, (128, 128, 128))  # Серый по умолчанию
            
            # Рисуем прямоугольник
            thickness = max(2, int(min(img_w, img_h) / 400))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Подпись с классом и уверенностью
            label = f"{det.class_name} {det.confidence:.2f}"
            
            # Вычисляем размер текста
            font_scale = max(0.5, min(img_w, img_h) / 1500)
            font_thickness = max(1, int(font_scale * 2))
            
            # Получаем размер текста
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Рисуем фон для текста
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1  # Заполненный прямоугольник
            )
            
            # Рисуем текст
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # Белый текст
                font_thickness,
                cv2.LINE_AA
            )
        
        return annotated
    
    def predict(self, image_path: str, annotate: bool = True) -> PredictionResult:
        """
        Основной метод для предсказания дефектов на изображении.
        
        Args:
            image_path: Путь к изображению
            annotate: Создавать ли аннотированное изображение
            
        Returns:
            PredictionResult объект с результатами
        """
        import time
        start_time = time.time()
        
        # Убеждаемся, что модель загружена
        if not self._is_loaded:
            self.load_model()
        
        # Проверяем существование файла
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Загружаем изображение
        try:
            # Используем PIL для загрузки (лучше сохраняет цвета)
            pil_image = Image.open(image_path)
            # Конвертируем в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Конвертируем в numpy array для OpenCV
            image_np = np.array(pil_image)
            # Конвертируем RGB в BGR для OpenCV
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            image_size = (image_np.shape[1], image_np.shape[0])  # (width, height)
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            raise
        
        # Вычисляем хэш изображения
        image_hash = self._calculate_image_hash(str(image_path))
        
        # Выполняем предсказание
        try:
            with torch.no_grad():
                yolo_results = self.model(
                    image_np,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False  # Отключаем вывод YOLO
                )
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {str(e)}")
            raise
        
        # Обрабатываем результаты
        processing_time = time.time() - start_time
        detections = self._process_yolo_results(yolo_results, image_size)
        
        # Создаем аннотированное изображение если нужно
        annotated_image = None
        if annotate and len(detections) > 0:
            annotated_image = self._annotate_image(image_np.copy(), detections)
        
        # Создаем результат
        result = PredictionResult(
            image_path=str(image_path),
            image_hash=image_hash,
            image_size=image_size,
            detections=detections,
            processing_time=processing_time,
            model_name=self.model_repo,
            original_image=image_np,
            annotated_image=annotated_image
        )
        
        # Логируем статистику
        logger.info(f"  → Found {len(detections)} defect(s) in {processing_time:.2f}s")
        if len(detections) > 0:
            for det in detections[:3]:  # Показываем только топ-3
                logger.info(f"    - {det.class_name}: {det.confidence:.1%}")
        
        return result
    
    def predict_batch(
        self, 
        image_paths: List[str], 
        output_dir: Optional[str] = None,
        batch_size: int = 4
    ) -> Dict[str, PredictionResult]:
        """
        Пакетная обработка изображений.
        
        Args:
            image_paths: Список путей к изображениям
            output_dir: Директория для сохранения результатов (опционально)
            batch_size: Размер батча для обработки
            
        Returns:
            Словарь {image_path: PredictionResult}
        """
        if not self._is_loaded:
            self.load_model()
        
        results = {}
        total_images = len(image_paths)
        
        logger.info(f"Starting batch processing of {total_images} images")
        
        for i in range(0, total_images, batch_size):
            batch = image_paths[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_images + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} "
                       f"({len(batch)} images)")
            
            for img_path in batch:
                try:
                    result = self.predict(img_path, annotate=True)
                    results[img_path] = result
                    
                    # Сохраняем результаты если указана output_dir
                    if output_dir:
                        self.save_result(result, output_dir)
                        
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {str(e)}")
                    continue
        
        logger.info(f"✅ Batch processing completed: {len(results)}/{total_images} successful")
        return results
    
    def save_result(self, result: PredictionResult, output_dir: str) -> Dict[str, str]:
        """
        Сохраняет все результаты предсказания в указанную директорию.
        
        Returns:
            Словарь с путями к сохраненным файлам
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Базовое имя файла (без расширения)
        image_stem = Path(result.image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{image_stem}_{timestamp}_{result.image_hash[:8]}"
        
        saved_files = {}
        
        # 1. Сохраняем аннотированное изображение
        if result.annotated_image is not None:
            img_path = output_path / f"{base_name}_annotated.jpg"
            result.save_annotated_image(str(img_path))
            saved_files["annotated_image"] = str(img_path)
        
        # 2. Сохраняем метаданные в JSON
        json_path = output_path / f"{base_name}_metadata.json"
        result.save_metadata(str(json_path))
        saved_files["metadata"] = str(json_path)
        
        # 3. Сохраняем текстовый файл с детекциями в YOLO формате
        txt_path = output_path / f"{base_name}_detections.txt"
        with open(txt_path, 'w') as f:
            for det in result.detections:
                # YOLO формат: class_id x_center y_center width height
                line = f"{det.class_id} {det.normalized_bbox[0]:.6f} "
                line += f"{det.normalized_bbox[1]:.6f} {det.normalized_bbox[2]:.6f} "
                line += f"{det.normalized_bbox[3]:.6f}\n"
                f.write(line)
        saved_files["detections_txt"] = str(txt_path)
        
        # 4. Создаем краткий отчет
        report_path = output_path / f"{base_name}_summary.txt"
        with open(report_path, 'w') as f:
            f.write(f"XVL Prediction Summary\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Image: {Path(result.image_path).name}\n")
            f.write(f"Size: {result.image_size[0]}x{result.image_size[1]}\n")
            f.write(f"Processing time: {result.processing_time:.3f}s\n")
            f.write(f"Model: {result.model_name}\n")
            f.write(f"Total defects: {len(result.detections)}\n\n")
            
            if result.detections:
                f.write("Defects found:\n")
                for i, det in enumerate(result.detections, 1):
                    f.write(f"  {i}. {det.class_name} ({det.confidence:.1%})\n")
                    f.write(f"     BBox: [{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, ")
                    f.write(f"{det.bbox[2]:.0f}, {det.bbox[3]:.0f}]\n")
            else:
                f.write("No defects detected\n")
        
        saved_files["summary"] = str(report_path)
        
        logger.debug(f"Results saved to {output_dir}")
        return saved_files
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о загруженной модели."""
        if not self._is_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_repo": self.model_repo,
            "model_file": self.model_file,
            "device": self.device,
            "class_names": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold
        }
        
        # Добавляем информацию о CUDA если используется GPU
        if self.device == "cuda" and torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info

# ============================================================================
# Вспомогательные функции
# ============================================================================

def create_predictor(
    model_repo: Optional[str] = None,
    **kwargs
) -> XVLPredictor:
    """Фабричная функция для создания предсказателя."""
    if model_repo is None:
        # Дефолтный репозиторий (замените на ваш!)
        model_repo = "yourusername/xvl-weld-defect-detection"
    
    return XVLPredictor(model_repo=model_repo, **kwargs)