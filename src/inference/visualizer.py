"""
🎨 visualizer.py - Визуализация результатов детекции дефектов
Модуль для отрисовки bounding boxes, создания отчётов и визуализации метрик.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import logging
from dataclasses import dataclass

# Настройка логирования
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Конфигурация визуализации"""
    # Цвета для разных классов (BGR формат для OpenCV)
    COLORS = {
        'incomplete_fusion': (0, 165, 255),    # Оранжевый
        'crack': (0, 0, 255),                  # Красный
        'single_pore': (0, 255, 0),            # Зелёный
        'cluster_pores': (255, 255, 0),        # Голубой
        'empty': (128, 128, 128),              # Серый
    }

    # Настройки bounding boxes
    BOX_THICKNESS = 2
    TEXT_THICKNESS = 1
    FONT_SCALE = 0.5
    CONFIDENCE_THRESHOLD = 0.3  # Порог для отображения

    # Настройки графиков
    PLOT_DPI = 150
    FIGURE_SIZE = (12, 8)

    # Пути
    DEFAULT_OUTPUT_DIR = Path("visualizations")

class DetectionVisualizer:
    """Основной класс для визуализации детекций"""

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.output_dir = self.config.DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

        # Кэш для шрифтов
        self._font_cache = {}

        logger.info(f"DetectionVisualizer initialized. Output dir: {self.output_dir}")

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Получает цвет для класса"""
        return self.config.COLORS.get(class_name, (255, 255, 255))  # Белый по умолчанию

    def _get_font(self, scale: float = None):
        """Получает или создаёт шрифт"""
        scale = scale or self.config.FONT_SCALE
        key = f"scale_{scale}"

        if key not in self._font_cache:
            try:
                # Пробуем загрузить шрифт, иначе используем дефолтный
                font = ImageFont.truetype("arial.ttf", int(20 * scale))
            except:
                font = ImageFont.load_default()
            self._font_cache[key] = font

        return self._font_cache[key]

    def draw_detections_pil(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[Dict],
        show_confidence: bool = True,
        show_class: bool = True
    ) -> Image.Image:
        """
        Отрисовывает bounding boxes на изображении с использованием PIL.
        Возвращает новое изображение с аннотациями.
        """
        # Конвертируем numpy array в PIL Image если нужно
        if isinstance(image, np.ndarray):
            # Конвертируем BGR в RGB если нужно
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()

        draw = ImageDraw.Draw(pil_image)
        font = self._get_font()

        img_width, img_height = pil_image.size

        for detection in detections:
            confidence = detection.get('confidence', 0)

            # Пропускаем низковероятные детекции
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                continue

            bbox = detection['bbox']  # [x1, y1, x2, y2]
            class_name = detection.get('class', 'unknown')

            # Конвертируем относительные координаты в абсолютные если нужно
            if all(0 <= coord <= 1 for coord in bbox):
                x1 = bbox[0] * img_width
                y1 = bbox[1] * img_height
                x2 = bbox[2] * img_width
                y2 = bbox[3] * img_height
            else:
                x1, y1, x2, y2 = bbox

            # Получаем цвет для класса
            color = self._get_color(class_name)
            color_rgb = color[::-1]  # BGR to RGB

            # Рисуем bounding box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color_rgb,
                width=self.config.BOX_THICKNESS
            )

            # Подготовка текста
            text_parts = []
            if show_class:
                text_parts.append(class_name)
            if show_confidence:
                text_parts.append(f"{confidence:.1%}")

            if text_parts:
                text = " ".join(text_parts)

                # Вычисляем размер текста
                try:
                    bbox_text = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    text_width = len(text) * 10
                    text_height = 20

                # Рисуем фон для текста
                text_bg = [
                    x1,
                    max(y1 - text_height - 5, 0),
                    x1 + text_width + 10,
                    max(y1, text_height + 5)
                ]

                draw.rectangle(text_bg, fill=color_rgb)

                # Рисуем текст
                draw.text(
                    (x1 + 5, max(y1 - text_height - 2, 2)),
                    text,
                    fill=(255, 255, 255),  # Белый текст
                    font=font
                )

        return pil_image

    def draw_detections_cv2(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_class: bool = True
    ) -> np.ndarray:
        """
        Отрисовывает bounding boxes с использованием OpenCV.
        Быстрее чем PIL, но менее гибко в оформлении текста.
        """
        img_copy = image.copy()
        img_height, img_width = img_copy.shape[:2]

        for detection in detections:
            confidence = detection.get('confidence', 0)

            if confidence < self.config.CONFIDENCE_THRESHOLD:
                continue

            bbox = detection['bbox']
            class_name = detection.get('class', 'unknown')

            # Конвертируем координаты если нужно
            if all(0 <= coord <= 1 for coord in bbox):
                x1 = int(bbox[0] * img_width)
                y1 = int(bbox[1] * img_height)
                x2 = int(bbox[2] * img_width)
                y2 = int(bbox[3] * img_height)
            else:
                x1, y1, x2, y2 = map(int, bbox)

            # Получаем цвет
            color = self._get_color(class_name)

            # Рисуем bounding box
            cv2.rectangle(
                img_copy,
                (x1, y1),
                (x2, y2),
                color,
                self.config.BOX_THICKNESS
            )

            # Подготовка текста
            text_parts = []
            if show_class:
                text_parts.append(class_name.replace('_', ' '))
            if show_confidence:
                text_parts.append(f"{confidence:.0%}")

            if text_parts:
                text = " ".join(text_parts)

                # Вычисляем размер текста
                (text_width, text_height), baseline = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.FONT_SCALE,
                    self.config.TEXT_THICKNESS
                )

                # Рисуем фон для текста
                cv2.rectangle(
                    img_copy,
                    (x1, max(y1 - text_height - 10, 0)),
                    (x1 + text_width + 10, y1),
                    color,
                    -1  # Заполненный прямоугольник
                )

                # Рисуем текст
                cv2.putText(
                    img_copy,
                    text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.FONT_SCALE,
                    (255, 255, 255),  # Белый текст
                    self.config.TEXT_THICKNESS,
                    cv2.LINE_AA
                )

        return img_copy

    def create_detection_report(
        self,
        image_path: Union[str, Path],
        detections: List[Dict],
        output_path: Optional[Union[str, Path]] = None,
        save_json: bool = True
    ) -> Dict:
        """
        Создает полный отчет по детекциям на изображении.
        Возвращает словарь с метаданными и путями к файлам.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Загружаем изображение
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Создаем выходные пути
        if output_path is None:
            timestamp = Path(image_path).stem
            output_path = self.output_dir / f"report_{timestamp}"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Визуализируем детекции
        img_with_boxes = self.draw_detections_cv2(img, detections)

        # Сохраняем визуализацию
        vis_path = output_path / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(vis_path), img_with_boxes)

        # Создаем JSON с метаданными если нужно
        metadata = {
            "image": str(image_path),
            "detections_count": len(detections),
            "detections": detections,
            "visualization": str(vis_path),
            "statistics": self._calculate_statistics(detections)
        }

        if save_json:
            json_path = output_path / f"{image_path.stem}_detections.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            metadata["json_report"] = str(json_path)

        # Создаем график с распределением классов
        plot_path = self._create_class_distribution_plot(detections, output_path)
        metadata["class_distribution_plot"] = str(plot_path)

        logger.info(f"Report created: {output_path}")
        logger.info(f"  - Detections: {len(detections)}")
        logger.info(f"  - Visualization: {vis_path.name}")
        logger.info(f"  - Statistics: {metadata['statistics']}")

        return metadata

    def create_comparison_grid(
        self,
        images_data: List[Dict],  # Список {'path': Path, 'detections': List}
        output_path: Union[str, Path],
        grid_size: Tuple[int, int] = (3, 3),
        titles: Optional[List[str]] = None
    ) -> Path:
        """
        Создает grid из нескольких изображений с детекциями.
        Полезно для демо-отчетов.
        """
        output_path = Path(output_path)

        # Ограничиваем количество изображений по размеру grid
        max_images = grid_size[0] * grid_size[1]
        images_data = images_data[:max_images]

        fig, axes = plt.subplots(
            grid_size[0],
            grid_size[1],
            figsize=(grid_size[1] * 4, grid_size[0] * 3)
        )
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, (ax, img_data) in enumerate(zip(axes, images_data)):
            # Загружаем и визуализируем изображение
            img_path = Path(img_data['path'])
            detections = img_data.get('detections', [])

            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_with_boxes = self.draw_detections_cv2(img_rgb, detections)

                ax.imshow(img_with_boxes)

                # Устанавливаем заголовок
                title_parts = [img_path.stem]
                if detections:
                    title_parts.append(f"({len(detections)} def)")
                if titles and idx < len(titles):
                    title_parts.append(titles[idx])

                ax.set_title(" | ".join(title_parts), fontsize=10)
                ax.axis('off')

                # Добавляем аннотацию с количеством дефектов
                if detections:
                    ax.text(
                        0.02, 0.98,
                        f"Detections: {len(detections)}",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5)
                    )
            else:
                ax.text(0.5, 0.5, f"Failed to load\n{img_path.name}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        # Скрываем пустые subplots
        for idx in range(len(images_data), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Сохраняем grid
        grid_path = output_path / "detection_grid.jpg"
        plt.savefig(grid_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison grid created: {grid_path}")
        return grid_path

    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """Вычисляет статистику по детекциям"""
        if not detections:
            return {"total": 0, "by_class": {}, "avg_confidence": 0}

        stats = {
            "total": len(detections),
            "by_class": {},
            "confidences": [],
            "avg_confidence": 0
        }

        # Группируем по классам
        for det in detections:
            class_name = det.get('class', 'unknown')
            confidence = det.get('confidence', 0)

            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'confidences': []
                }

            stats['by_class'][class_name]['count'] += 1
            stats['by_class'][class_name]['confidences'].append(confidence)
            stats['confidences'].append(confidence)

        # Вычисляем средние значения
        if stats['confidences']:
            stats['avg_confidence'] = np.mean(stats['confidences'])

        for class_name, class_stats in stats['by_class'].items():
            if class_stats['confidences']:
                class_stats['avg_confidence'] = np.mean(class_stats['confidences'])
            del class_stats['confidences']  # Удаляем временный список

        return stats

    def _create_class_distribution_plot(
        self,
        detections: List[Dict],
        output_dir: Path
    ) -> Path:
        """Создает график распределения классов"""
        if not detections:
            return None

        # Подсчитываем количество по классам
        class_counts = {}
        confidences_by_class = {}

        for det in detections:
            class_name = det.get('class', 'unknown')
            confidence = det.get('confidence', 0)

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name not in confidences_by_class:
                confidences_by_class[class_name] = []
            confidences_by_class[class_name].append(confidence)

        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Гистограмма количества
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = [self._get_color(cls) for cls in classes]
        colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]  # BGR to RGB

        bars = ax1.bar(classes, counts, color=colors_rgb, edgecolor='black')
        ax1.set_title('Количество детекций по классам', fontsize=12)
        ax1.set_xlabel('Класс дефекта')
        ax1.set_ylabel('Количество')
        ax1.tick_params(axis='x', rotation=45)

        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Box plot уверенностей
        confidence_data = []
        labels = []
        for class_name in classes:
            if class_name in confidences_by_class and confidences_by_class[class_name]:
                confidence_data.append(confidences_by_class[class_name])
                labels.append(f"{class_name}\n(n={len(confidences_by_class[class_name])})")

        if confidence_data:
            bp = ax2.boxplot(confidence_data, labels=labels, patch_artist=True)

            # Раскрашиваем box plots
            for patch, color in zip(bp['boxes'], colors_rgb):
                patch.set_facecolor(color)

            ax2.set_title('Распределение уверенностей по классам', fontsize=12)
            ax2.set_ylabel('Уверенность')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохраняем график
        plot_path = output_dir / "class_distribution.jpg"
        plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        return plot_path

# Фабричная функция для удобства
def create_visualizer(config: VisualizationConfig = None) -> DetectionVisualizer:
    """Создает и возвращает визуализатор"""
    return DetectionVisualizer(config)

# Пример использования
if __name__ == "__main__":
    # Тестирование визуализатора
    import tempfile

    # Создаем тестовые данные
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Серый фон

    test_detections = [
        {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.85,
            'class': 'crack'
        },
        {
            'bbox': [300, 150, 400, 250],
            'confidence': 0.72,
            'class': 'single_pore'
        },
        {
            'bbox': [200, 300, 350, 380],
            'confidence': 0.45,
            'class': 'incomplete_fusion'
        }
    ]

    # Инициализируем визуализатор
    visualizer = DetectionVisualizer()

    # Тестируем разные методы
    print("Testing visualization methods...")

    # Метод OpenCV
    result_cv2 = visualizer.draw_detections_cv2(test_image, test_detections)

    # Метод PIL
    pil_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    result_pil = visualizer.draw_detections_pil(pil_image, test_detections)

    # Создаем отчет
    with tempfile.TemporaryDirectory() as tmpdir:
        # Сохраняем тестовое изображение
        test_path = Path(tmpdir) / "test_image.jpg"
        cv2.imwrite(str(test_path), test_image)

        # Создаем отчет
        report = visualizer.create_detection_report(
            test_path,
            test_detections,
            output_path=Path(tmpdir) / "report"
        )

        print(f"Report created at: {report['visualization']}")
        print(f"Statistics: {report['statistics']}")

    print("Visualizer test completed successfully!")