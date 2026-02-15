"""
# Обычный режим
python yolo_trainer.py --configs configs.yaml --data ./data

# Режим Colab
python yolo_trainer.py --configs configs.yaml --data /content/drive/data --colab --base-dir /content

# Возобновление обучения
python yolo_trainer.py --configs configs.yaml --data ./data --resume
"""

# yolo_trainer.py
import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
import shutil
from datetime import datetime

import torch
from ultralytics import YOLO
import cv2


# ========== КОНФИГУРАЦИЯ ПУТЕЙ ДЛЯ COLAB ==========
# В Colab пути могут быть разными, поэтому делаем их настраиваемыми
DEFAULT_PATHS = {
    'log_dir': './logs/training',
    'models_dir': './models',
    'metrics_dir': './metrics',
    'dataset_dir': './data',
    'yolo_dataset_dir': './yolo_dataset'
}


def setup_colab_paths(base_path: str = None):
    """Настройка путей для работы в Colab"""
    if base_path:
        base_path = Path(base_path)
        DEFAULT_PATHS['log_dir'] = str(base_path / 'logs' / 'training')
        DEFAULT_PATHS['models_dir'] = str(base_path / 'models')
        DEFAULT_PATHS['metrics_dir'] = str(base_path / 'metrics')
        DEFAULT_PATHS['dataset_dir'] = str(base_path / 'data')
        DEFAULT_PATHS['yolo_dataset_dir'] = str(base_path / 'yolo_dataset')

    # Создаем директории если их нет
    for dir_path in DEFAULT_PATHS.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# ========== НАСТРОЙКА ЛОГГИРОВАНИЯ ==========
def setup_logging(log_dir: str = None):
    """Настройка логирования с учетом путей Colab"""
    if log_dir is None:
        log_dir = DEFAULT_PATHS['log_dir']

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ========== КЛАССЫ ДЛЯ КОНФИГУРАЦИИ ==========
@dataclass
class TrainingConfig:
    """Конфигурация обучения из YAML файла"""
    img_size: int = 512
    model_name: str = "yolov8s.pt"
    epochs: int = 150
    batch_size: int = 48
    learning_rate: float = 0.00005
    num_folds: int = 5
    samples_per_epoch: Optional[int] = None
    val_samples: Optional[int] = None
    class_weights: List[float] = None
    class_names: List[str] = None
    mixup: float = 0.05
    cutmix: float = 0.05
    dropout: float = 0.1
    use_cosine_annealing: bool = True
    warmup_epochs: int = 20
    weight_decay: float = 0.005
    # Новые поля для управления путями
    project_dir: str = DEFAULT_PATHS['models_dir']
    log_dir: str = DEFAULT_PATHS['log_dir']
    metrics_dir: str = DEFAULT_PATHS['metrics_dir']

    def __post_init__(self):
        if self.class_weights is None:
            self.class_weights = [1.0] * len(self.class_names) if self.class_names else [1.0]
        if self.class_names is None:
            self.class_names = []


class ConfigLoader:
    """Загрузчик конфигурации из YAML файла"""

    @staticmethod
    def load_config(config_path: str, base_dir: str = None) -> TrainingConfig:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # Извлечение параметров с fallback на значения по умолчанию
            training = config_data.get('training', {})
            data = config_data.get('data', {})
            augmentation = config_data.get('augmentation', {})
            optimization = config_data.get('optimization', {})
            paths = config_data.get('paths', {})

            # Определение базовой директории для путей
            if base_dir:
                base_path = Path(base_dir)
            else:
                base_path = Path.cwd()

            return TrainingConfig(
                img_size=training.get('img_size', 512),
                model_name=training.get('model_name', 'yolov8s.pt'),
                epochs=training.get('epochs', 150),
                batch_size=training.get('batch_size', 48),
                learning_rate=training.get('learning_rate', 0.00005),
                num_folds=training.get('num_folds', 5),
                samples_per_epoch=data.get('samples_per_epoch'),
                val_samples=data.get('val_samples'),
                class_weights=data.get('class_weights'),
                class_names=data.get('class_names', []),
                mixup=augmentation.get('mixup', 0.05),
                cutmix=augmentation.get('cutmix', 0.05),
                dropout=augmentation.get('dropout', 0.1),
                use_cosine_annealing=optimization.get('use_cosine_annealing', True),
                warmup_epochs=optimization.get('warmup_epochs', 20),
                weight_decay=optimization.get('weight_decay', 0.005),
                project_dir=str(base_path / paths.get('project_dir', DEFAULT_PATHS['models_dir'])),
                log_dir=str(base_path / paths.get('log_dir', DEFAULT_PATHS['log_dir'])),
                metrics_dir=str(base_path / paths.get('metrics_dir', DEFAULT_PATHS['metrics_dir']))
            )

        except FileNotFoundError:
            logger.error(f"Конфигурационный файл не найден: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке конфигурации: {e}")
            raise


# ========== КЛАССЫ ДЛЯ ОБРАБОТКИ ДАННЫХ (остаются без изменений) ==========
class DataConverter:
    """Конвертер аннотаций из JSON в YOLO формат"""

    # Маппинг русских названий классов на английские
    CLASS_MAPPING = {
        "Непровар": "incomplete_fusion",
        "трещина": "crack",
        "Одиночное_включение": "single_pore",
        "Скопление_пор": "cluster_pores",
        "пусто": "empty"
    }

    @staticmethod
    def convert_annotation_to_yolo(
        annotation: Dict[str, Any],
        image_path: Path,
        class_names: List[str],
        output_dir: Path,
        logger: logging.Logger
    ) -> bool:
        """
        Конвертирует одну аннотацию в YOLO формат

        Args:
            annotation: Словарь с аннотациями из JSON
            image_path: Путь к изображению
            class_names: Список имен классов
            output_dir: Директория для сохранения аннотаций YOLO

        Returns:
            bool: Успешность конвертации
        """
        try:
            # Загрузка изображения для получения размеров
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Не удалось загрузить изображение: {image_path}")
                return False

            height, width = img.shape[:2]

            # Создание файла аннотаций
            txt_path = output_dir / f"{image_path.stem}.txt"

            with open(txt_path, 'w') as f:
                # Обработка каждого дефекта в аннотации
                for obj_list in annotation.values():
                    if not isinstance(obj_list, dict):
                        continue

                    for obj_name, bbox in obj_list.items():
                        # Извлечение базового имени класса (убираем номер)
                        base_class_name = None
                        for ru_name, en_name in DataConverter.CLASS_MAPPING.items():
                            if obj_name.startswith(ru_name):
                                base_class_name = en_name
                                break

                        if base_class_name not in class_names:
                            logger.warning(f"Неизвестный класс: {base_class_name} в {obj_name}")
                            continue

                        class_id = class_names.index(base_class_name)

                        # Конвертация bbox в формат YOLO
                        x1, y1, x2, y2 = map(float, bbox)

                        # Расчет центра и размеров в относительных координатах
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height

                        # Запись в файл
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

                # Если файл пустой (нет объектов), создаем пустой файл для класса empty
                if txt_path.stat().st_size == 0 and 'empty' in class_names:
                    f.write("")  # Пустая строка

            return True

        except Exception as e:
            logger.error(f"Ошибка конвертации аннотации для {image_path}: {e}")
            return False

    @classmethod
    def convert_dataset(
        cls,
        images_dir: Path,
        annotations_path: Path,
        class_names: List[str],
        output_dir: Path
    ) -> Tuple[List[str], List[str]]:
        """
        Конвертирует весь датасет в YOLO формат

        Returns:
            Tuple[List[str], List[str]]: Списки успешных и неуспешных конвертаций
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(annotations_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            successful = []
            failed = []

            for img_name, annotation in annotations.items():
                # Поиск изображения с любым расширением
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    potential_path = images_dir / f"{img_name}{ext}"
                    if potential_path.exists():
                        img_path = potential_path
                        break

                if img_path is None:
                    logger.warning(f"Изображение не найдено: {img_name}")
                    failed.append(img_name)
                    continue

                if cls.convert_annotation_to_yolo(annotation, img_path, class_names, output_dir):
                    successful.append(img_name)
                else:
                    failed.append(img_name)

            logger.info(f"Конвертация завершена. Успешно: {len(successful)}, Ошибок: {len(failed)}")
            return successful, failed

        except Exception as e:
            logger.error(f"Ошибка конвертации датасета: {e}")
            raise


class DatasetPreparator:
    """Подготовка датасета для обучения"""

    @staticmethod
    def prepare_yolo_dataset_structure(
        base_dir: Path,
        class_names: List[str],
        num_folds: int = 1
    ) -> Dict[str, Any]:
        """
        Подготавливает структуру датасета в формате YOLO

        Args:
            base_dir: Базовая директория с данными
            class_names: Список имен классов
            num_folds: Количество фолдов для кросс-валидации

        Returns:
            Dict с информацией о датасете
        """
        # Директории для YOLO формата
        yolo_dir = base_dir.parent / "yolo_dataset"
        yolo_dir.mkdir(exist_ok=True)

        # Конвертация train данных
        train_images_dir = base_dir / "train"
        train_ann_path = base_dir / "train" / "annotations.json"
        train_labels_dir = yolo_dir / "labels" / "train"

        if train_ann_path.exists():
            DataConverter.convert_dataset(
                train_images_dir, train_ann_path, class_names, train_labels_dir
            )

            # Копирование изображений
            train_images_yolo = yolo_dir / "images" / "train"
            train_images_yolo.mkdir(parents=True, exist_ok=True)

            for img_file in train_images_dir.glob("*.*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    shutil.copy2(img_file, train_images_yolo)

        # Конвертация val данных
        val_images_dir = base_dir / "val"
        val_ann_path = base_dir / "val" / "annotations.json"
        val_labels_dir = yolo_dir / "labels" / "val"

        if val_ann_path.exists():
            DataConverter.convert_dataset(
                val_images_dir, val_ann_path, class_names, val_labels_dir
            )

            # Копирование изображений
            val_images_yolo = yolo_dir / "images" / "val"
            val_images_yolo.mkdir(parents=True, exist_ok=True)

            for img_file in val_images_dir.glob("*.*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    shutil.copy2(img_file, val_images_yolo)

        # Создание data.yaml файла
        data_yaml = {
            'path': str(yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }

        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return {
            'yaml_path': yaml_path,
            'data_dir': yolo_dir,
            'num_folds': num_folds
        }


# ========== УЛУЧШЕННЫЙ ТРЕНЕР ДЛЯ COLAB ==========
class YOLOTrainer:
    """Тренировочный класс для YOLOv8 с поддержкой Colab"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Определение устройства с приоритетом для Colab
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        logger.info(f"Используемое устройство: {self.device}")

        # Создаем директории для сохранения результатов
        Path(self.config.project_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.metrics_dir).mkdir(parents=True, exist_ok=True)

    def setup_training_args(self, fold_idx: Optional[int] = None) -> Dict[str, Any]:
        """Настройка аргументов для обучения"""
        # Формируем имя проекта для удобства в Colab
        if fold_idx is not None:
            project_name = f"weld_defects_fold_{fold_idx}"
        else:
            project_name = "weld_defects"

        args = {
            'data': str(self.dataset_info['yaml_path']),
            'epochs': self.config.epochs,
            'imgsz': self.config.img_size,
            'batch': self.config.batch_size,
            'lr0': self.config.learning_rate,
            'device': self.device,
            'save': True,
            'save_period': 10,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'weight_decay': self.config.weight_decay,
            'dropout': self.config.dropout,
            'mixup': self.config.mixup,
            'cutmix': self.config.cutmix,
            'warmup_epochs': self.config.warmup_epochs,
            'cos_lr': self.config.use_cosine_annealing,
            'label_smoothing': 0.1,
            'patience': 30,
            'project': self.config.project_dir,  # Используем настраиваемый путь
            'name': project_name,
            'verbose': True,  # Добавляем подробный вывод для Colab
        }

        # Добавление весов классов если указаны
        if self.config.class_weights:
            args['cls'] = self.config.class_weights

        # Ограничение количества выборок если указано
        if self.config.samples_per_epoch:
            args['subsample'] = self.config.samples_per_epoch

        return args

    def train_single_fold(self, fold_idx: Optional[int] = None, resume: bool = False):
        """Обучение на одном фолде"""
        try:
            # Загрузка модели
            model = YOLO(self.config.model_name)

            # Настройка аргументов обучения
            train_args = self.setup_training_args(fold_idx)

            # Путь для возобновления обучения
            if resume:
                last_checkpoint = self.find_last_checkpoint(train_args['project'], train_args['name'])
                if last_checkpoint:
                    train_args['resume'] = last_checkpoint
                    logger.info(f"Возобновление обучения с: {last_checkpoint}")

            # Обучение
            logger.info(f"Начало обучения фолда {fold_idx if fold_idx is not None else 'single'}")
            results = model.train(**train_args)

            logger.info(f"Обучение фолда {fold_idx if fold_idx is not None else 'single'} завершено")
            return results

        except Exception as e:
            logger.error(f"Ошибка обучения фолда {fold_idx}: {e}")
            raise

    def train_kfold(self, dataset_info: Dict[str, Any], resume_fold: int = 0):
        """K-fold кросс-валидация"""
        results = []

        for fold_idx in range(resume_fold, self.config.num_folds):
            logger.info(f"Начало обучения фолда {fold_idx + 1}/{self.config.num_folds}")

            # Здесь должна быть логика создания фолдов
            # Для простоты используем стандартное разделение

            fold_result = self.train_single_fold(fold_idx, resume=(fold_idx == resume_fold))
            results.append(fold_result)

            # Сохранение метрик фолда
            self.save_fold_metrics(fold_idx, fold_result)

        return results

    def find_last_checkpoint(self, project_dir: str, run_name: str) -> Optional[str]:
        """Поиск последнего чекпоинта для возобновления обучения"""
        checkpoint_dir = Path(project_dir) / run_name / 'weights'
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('last*.pt'))
            if checkpoints:
                return str(checkpoints[0])
        return None

    def save_fold_metrics(self, fold_idx: int, results: Any):
        """Сохранение метрик фолда"""
        metrics_dir = Path('metrics')
        metrics_dir.mkdir(exist_ok=True)

        # Здесь можно сохранить метрики в файл
        # Например, results.box.map, results.box.map50 и т.д.

    def train(self, dataset_info: Dict[str, Any], resume: bool = False):
        """Основной метод обучения"""
        self.dataset_info = dataset_info

        if self.config.num_folds > 1:
            logger.info(f"Запуск {self.config.num_folds}-fold кросс-валидации")
            return self.train_kfold(dataset_info, resume_fold=0)
        else:
            logger.info("Запуск обучения на одном наборе данных")
            return self.train_single_fold(resume=resume)


# ========== ОСНОВНОЙ ПАЙПЛАЙН ДЛЯ COLAB ==========
def train_pipeline(
    config_path: str,
    data_dir: str,
    resume: bool = False,
    device: Optional[str] = None,
    project_name: str = "weld_defects",
    base_dir: str = None,  # НОВЫЙ ПАРАМЕТР для указания базовой директории
    colab_mode: bool = False  # НОВЫЙ ПАРАМЕТР для включения режима Colab
) -> Dict[str, Any]:
    """
    Основной пайплайн обучения для использования в Colab и других средах

    Args:
        config_path: Путь к конфигурационному файлу
        data_dir: Путь к директории с данными
        resume: Возобновить обучение с последнего чекпоинта
        device: Устройство для обучения ('cuda', 'cpu' или None для auto)
        project_name: Название проекта для сохранения результатов
        base_dir: Базовая директория для всех путей (для Colab)
        colab_mode: Режим работы в Colab (автоматически настраивает пути)

    Returns:
        Словарь с результатами обучения
    """
    try:
        logger.info("=" * 60)
        logger.info("ЗАПУСК ОБУЧЕНИЯ В COLAB-СОВМЕСТИМОМ РЕЖИМЕ")
        logger.info("=" * 60)

        # Настройка путей для Colab
        if colab_mode:
            logger.info("Включен режим Colab, настраиваем пути...")
            if base_dir:
                setup_colab_paths(base_dir)
            else:
                setup_colab_paths()

        logger.info(f"Параметры запуска:")
        logger.info(f"  Конфиг: {config_path}")
        logger.info(f"  Данные: {data_dir}")
        logger.info(f"  Resume: {resume}")
        logger.info(f"  Device: {device if device else 'auto'}")
        logger.info(f"  Base dir: {base_dir if base_dir else 'текущая директория'}")
        logger.info(f"  Colab mode: {colab_mode}")

        # Загрузка конфигурации
        config = ConfigLoader.load_config(config_path, base_dir)

        # Подготовка датасета
        data_base_dir = Path(data_dir)
        dataset_preparator = DatasetPreparator()
        dataset_info = dataset_preparator.prepare_yolo_dataset_structure(
            data_base_dir,
            config.class_names,
            config.num_folds
        )

        # Инициализация тренера
        trainer = YOLOTrainer(config)

        # Переопределение устройства если указано
        if device:
            trainer.device = device
            logger.info(f"Устройство переопределено на: {device}")

        # Запуск обучения
        results = trainer.train(dataset_info, resume=resume)

        # Сбор результатов
        output = {
            'success': True,
            'configs': config.__dict__,
            'results': results,
            'checkpoint_dir': Path(config.project_dir) / project_name,
            'metrics_dir': Path(config.metrics_dir),
            'log_dir': Path(config.log_dir),
            'dataset_info': dataset_info
        }

        logger.info("=" * 60)
        logger.info("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        logger.info(f"Результаты сохранены в: {output['checkpoint_dir']}")
        logger.info("=" * 60)

        return output

    except Exception as e:
        logger.error(f"Ошибка в пайплайне обучения: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


# ========== УПРОЩЕННАЯ ФУНКЦИЯ ДЛЯ ИМПОРТА В COLAB ==========
def colab_train(
    config_path: str,
    data_dir: str,
    base_dir: str = "/content",
    device: str = "cuda",
    resume: bool = False
):
    """
    Упрощенная функция для запуска обучения в Google Colab

    Пример использования в Colab:
    ```
    from yolo_trainer import colab_train

    result = colab_train(
        config_path="/content/drive/MyDrive/configs.yaml",
        data_dir="/content/drive/MyDrive/weld_data",
        base_dir="/content",
        device="cuda"
    )
    ```
    """
    return train_pipeline(
        config_path=config_path,
        data_dir=data_dir,
        resume=resume,
        device=device,
        base_dir=base_dir,
        colab_mode=True
    )


# ========== КОМАНДНАЯ СТРОКА ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Trainer for Weld Defects')
    parser.add_argument('--configs', type=str, required=True, help='Path to configs YAML file', default='model/configs.yaml')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--base-dir', type=str, help='Base directory for all paths')
    parser.add_argument('--colab', action='store_true', help='Enable Colab mode')

    args = parser.parse_args()

    result = train_pipeline(
        config_path=args.config,
        data_dir=args.data,
        resume=args.resume,
        device=args.device,
        base_dir=args.base_dir,
        colab_mode=args.colab
    )

    if not result['success']:
        sys.exit(1)