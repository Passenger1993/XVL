"""Модуль для визуализации bounding box на изображениях дефектов."""

import json
import os
import random
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from enum import Enum

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ColorPalette(Enum):
    """Цветовая палитра для различных типов дефектов."""
    CRACK = (255, 0, 0)      # Красный
    CORROSION = (0, 255, 0)  # Зеленый
    SPALLING = (0, 0, 255)   # Синий
    DEFAULT = (255, 165, 0)  # Оранжевый


@dataclass
class BoundingBox:
    """DTO для bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def is_valid(self) -> bool:
        return (self.x1 < self.x2 and self.y1 < self.y2 and
                self.x1 >= 0 and self.y1 >= 0)


def get_project_root() -> Path:
    """
    Определяет корневую директорию проекта.

    Returns:
        Path: Корневая директория проекта
    """
    # Текущий файл: C:\PycharmProjects\XVL\src\generators\markup_test.py
    current_file = Path(__file__).resolve()

    # Поднимаемся на 3 уровня вверх для получения C:\PycharmProjects\XVL
    project_root = current_file.parent.parent.parent

    # Проверяем, что это действительно корень проекта
    if (project_root / "data").exists():
        return project_root

    # Альтернативная проверка
    markers = [".git", "pyproject.toml", "setup.py", "requirements.txt"]
    for marker in markers:
        if (project_root / marker).exists():
            return project_root

    # Если не нашли маркеров, используем текущую рабочую директорию
    logger.warning("Не удалось определить корень проекта, используется текущая директория")
    return Path.cwd()


class AnnotationLoader:
    """Загрузчик и валидатор аннотаций."""

    def __init__(self, annotations_path: Path):
        self.path = annotations_path
        self._annotations: Dict[str, Dict[str, List[int]]] = {}

    def load(self) -> Dict[str, Dict[str, List[int]]]:
        """Загружает и валидирует аннотации."""
        # Проверяем существование файла
        if not self.path.exists():
            # Пробуем найти файл относительно текущей директории
            current_dir = Path.cwd()
            possible_paths = [
                self.path,
                current_dir / self.path.name,
                get_project_root() / "data" / "training" / self.path.name
            ]

            for path in possible_paths:
                if path.exists():
                    logger.info(f"Найден файл аннотаций по альтернативному пути: {path}")
                    self.path = path
                    break
            else:
                raise FileNotFoundError(
                    f"Файл аннотаций не найден: {self.path}\n"
                    f"Искал в следующих местах:\n" +
                    "\n".join(f"  - {p}" for p in possible_paths)
                )

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Невалидный JSON файл: {e}")

        self._validate_annotations(data)
        self._annotations = data
        logger.info(f"Загружено {len(data)} аннотаций")
        return data

    def _validate_annotations(self, data: Dict) -> None:
        """Валидация структуры аннотаций."""
        if not isinstance(data, dict):
            raise ValueError("Аннотации должны быть словарем")

        for img_id, bboxes in data.items():
            if not isinstance(bboxes, dict):
                raise ValueError(f"Неверный формат bbox для изображения {img_id}")

            for label, coords in bboxes.items():
                if not isinstance(coords, list) or len(coords) != 4:
                    raise ValueError(
                        f"Неверный формат координат для {img_id}/{label}"
                    )
                if not all(isinstance(c, (int, float)) for c in coords):
                    raise ValueError(
                        f"Координаты должны быть числами для {img_id}/{label}"
                    )


class BBoxVisualizer:
    """Визуализатор bounding box на изображениях."""

    def __init__(
        self,
        images_dir: Path,
        output_dir: Optional[Path] = None,
        font_path: Optional[Path] = None
    ):
        self.images_dir = images_dir
        self.output_dir = output_dir or images_dir / "visualized"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Сначала попробовать путь от пользователя или системный шрифт
        potential_paths = []
        if font_path and font_path.exists():
            potential_paths.append(font_path)

        # Распространенные пути к шрифтам в Windows
        windows_fonts_dir = Path(r"C:\Windows\Fonts")
        potential_paths.extend([
            windows_fonts_dir / "arial.ttf",  # Arial
            windows_fonts_dir / "calibri.ttf", # Calibri
        ])

        self.font = None
        for font_candidate in potential_paths:
            try:
                self.font = ImageFont.truetype(str(font_candidate), 12)  # Размер 12
                logger.info(f"Загружен шрифт: {font_candidate}")
                break  # Использовать первый успешно загруженный
            except IOError as e:
                logger.debug(f"Не удалось загрузить шрифт {font_candidate}: {e}")
                continue

        # Если ни один TTF не загрузился, использовать базовый
        if self.font is None:
            logger.warning("Не удалось загрузить TTF-шрифт, используется системный по умолчанию.")
            self.font = ImageFont.load_default()

    def draw_bboxes(
        self,
        image: Image.Image,
        bboxes: Dict[str, List[int]],
        line_width: int = 2,
        show_labels: bool = True
    ) -> Image.Image:
        """Рисует bbox на изображении с подписями."""
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        draw = ImageDraw.Draw(image)

        for label, coords in bboxes.items():
            bbox = BoundingBox(*coords, label=label)

            if not bbox.is_valid():
                logger.warning(f"Пропущен невалидный bbox: {bbox}")
                continue

            # Выбор цвета в зависимости от типа дефекта
            color = self._get_color_for_label(label)

            # Рисуем прямоугольник
            draw.rectangle(
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                outline=color,
                width=line_width
            )

            # Добавляем подпись
            if show_labels:
                self._draw_label(draw, bbox, color)

        return image

    def _get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """Возвращает цвет для метки дефекта."""
        label_lower = label.lower()
        for defect_type in ColorPalette:
            if defect_type.name.lower() in label_lower:
                return defect_type.value
        return ColorPalette.DEFAULT.value

    def _draw_label(
        self,
        draw: ImageDraw.Draw,
        bbox: BoundingBox,
        color: Tuple[int, int, int]
    ) -> None:
        """Рисует подпись для bounding box."""
        text = f"{bbox.label}"
        text_bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Позиция подписи
        x = max(bbox.x1, 0)
        y = max(bbox.y1 - text_height - 5, 0)

        # Фон для текста
        draw.rectangle(
            [x, y, x + text_width + 4, y + text_height + 4],
            fill=color
        )

        # Текст
        draw.text(
            (x + 2, y + 2),
            text,
            fill=(255, 255, 255),
            font=self.font
        )


class ImageSelector:
    """Выбор изображений для визуализации."""

    @staticmethod
    def get_random(
        annotations: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Выбирает случайное изображение из аннотаций."""
        if not annotations:
            return None, None

        image_id = random.choice(list(annotations.keys()))
        return image_id, annotations[image_id]

    @staticmethod
    def get_by_id(
        annotations: Dict[str, Any],
        image_id: str
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Получает аннотации по ID изображения."""
        return image_id, annotations.get(image_id)

    @staticmethod
    def get_sample(
        annotations: Dict[str, Any],
        sample_size: int = 5
    ) -> List[Tuple[str, Dict]]:
        """Выборка нескольких случайных изображений."""
        if not annotations:
            return []

        sample_size = min(sample_size, len(annotations))
        sampled_ids = random.sample(list(annotations.keys()), sample_size)
        return [(img_id, annotations[img_id]) for img_id in sampled_ids]


def setup_argparse() -> argparse.Namespace:
    """Настройка аргументов командной строки."""
    # Определяем корень проекта для значений по умолчанию
    PROJECT_ROOT = get_project_root()
    DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "training" / "annotations.json"
    DEFAULT_IMAGES = PROJECT_ROOT / "data" / "training" / "train"

    parser = argparse.ArgumentParser(
        description='Визуализация bounding box на изображениях'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f'Путь к файлу аннотаций (по умолчанию: {DEFAULT_ANNOTATIONS})'
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        default=DEFAULT_IMAGES,
        help=f'Директория с изображениями (по умолчанию: {DEFAULT_IMAGES})'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Директория для сохранения результатов'
    )
    parser.add_argument(
        '--image-id',
        type=str,
        default=None,
        help='ID конкретного изображения'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1,
        help='Количество изображений для обработки'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Сохранять изображения вместо показа'
    )

    return parser.parse_args()


def main():
    """Основная функция."""
    args = setup_argparse()

    # Логируем используемые пути для отладки
    logger.info(f"Корень проекта: {get_project_root()}")
    logger.info(f"Файл аннотаций: {args.annotations}")
    logger.info(f"Директория изображений: {args.images_dir}")
    logger.info(f"Файл аннотаций существует: {args.annotations.exists()}")
    logger.info(f"Директория изображений существует: {args.images_dir.exists()}")

    try:
        # Загрузка аннотаций
        loader = AnnotationLoader(args.annotations)
        annotations = loader.load()

        # Инициализация визуализатора
        visualizer = BBoxVisualizer(
            images_dir=args.images_dir,
            output_dir=args.output_dir
        )

        # Выбор изображений
        selector = ImageSelector()

        if args.image_id:
            # Обработка конкретного изображения
            image_id, bboxes = selector.get_by_id(annotations, args.image_id)
            if not bboxes:
                logger.error(f"Изображение {args.image_id} не найдено в аннотациях")
                return
            samples = [(image_id, bboxes)]
        else:
            # Случайная выборка
            samples = selector.get_sample(annotations, args.sample_size)

        # Обработка изображений
        for image_id, bboxes in samples:
            image_path = args.images_dir / f"{image_id}.png"

            if not image_path.exists():
                logger.warning(f"Изображение не найдено: {image_path}")
                continue

            # Загрузка и обработка изображения
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения {image_id}: {e}")
                continue

            # Визуализация bbox
            image_with_bbox = visualizer.draw_bboxes(
                image=image,
                bboxes=bboxes,
                show_labels=True
            )

            # Вывод или сохранение
            if args.save:
                output_path = visualizer.output_dir / f"{image_id}_visualized.png"
                image_with_bbox.save(output_path)
                logger.info(f"Сохранено: {output_path}")
            else:
                image_with_bbox.show()
                logger.info(f"Обработано: {image_id}")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()