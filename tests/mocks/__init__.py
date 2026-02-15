"""
Моки для тестирования проекта XVL
"""

from unittest.mock import Mock, MagicMock, patch
import sys
import numpy as np
from PIL import Image

# Моки для генераторов изображений

class MockCrackGenerator:
    """Мок генератора трещин"""
    @staticmethod
    def generate_crack(crack_type="single"):
        """Генерация тестовой трещины"""
        img = Image.new('L', (256, 256), color=128)
        bbox = {"crack_1": [50, 50, 100, 100]}
        return img, bbox

class MockPoreGenerator:
    """Мок генератора пор"""
    @staticmethod
    def generate_pore(num_pores=1):
        """Генерация тестовых пор"""
        img = Image.new('L', (256, 256), color=128)
        if num_pores == 1:
            bbox = {"pore_1": [70, 70, 80, 80]}
        else:
            bbox = {f"pore_{i}": [60 + i*10, 60 + i*10, 70 + i*10, 70 + i*10]
                   for i in range(num_pores)}
        return img, bbox

class MockFusionGenerator:
    """Мок генератора непроваров"""
    @staticmethod
    def generate_fusion():
        """Генерация тестового непровара"""
        img = Image.new('L', (256, 256), color=128)
        bbox = {"fusion_1": [60, 60, 110, 110]}
        return img, bbox

class MockEmptySeam:
    """Мок пустого шва"""
    @staticmethod
    def generate_empty():
        """Генерация пустого шва"""
        return Image.new('L', (256, 256), color=128)

# Моки для evaluator

class MockCV2:
    """Мок для OpenCV"""
    @staticmethod
    def imread(path, flags=None):
        """Мок загрузки изображения"""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    @staticmethod
    def Canny(image, threshold1, threshold2):
        """Мок детектора краев"""
        return np.random.randint(0, 255, image.shape, dtype=np.uint8)

    @staticmethod
    def imdecode(buf, flags):
        """Мок декодирования изображения"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    @staticmethod
    def imencode(ext, img, params=None):
        """Мок кодирования изображения"""
        class MockEncoded:
            def tobytes(self):
                return b"mock_image_data"
        return True, MockEncoded()

    @staticmethod
    def rectangle(img, pt1, pt2, color, thickness):
        """Мок рисования прямоугольника"""
        return img

    @staticmethod
    def putText(img, text, org, font, fontScale, color, thickness):
        """Мок добавления текста"""
        return img

    @staticmethod
    def getTextSize(text, font, fontScale, thickness):
        """Мок получения размера текста"""
        return (len(text) * 10, 20), 0

    IMREAD_COLOR = 1
    IMREAD_GRAYSCALE = 0
    FONT_HERSHEY_SIMPLEX = 0
    FILLED = -1

# Моки для manager

class MockPILImage:
    """Мок для PIL Image"""
    def __init__(self, size=(256, 256), mode='L'):
        self.size = size
        self.mode = mode

    def copy(self):
        return self

    def crop(self, box):
        return self

    def save(self, path, format=None):
        return True

    def convert(self, mode):
        self.mode = mode
        return self

    def filter(self, filter_obj):
        return self

class MockImageDraw:
    """Мок для ImageDraw"""
    def __init__(self, image):
        self.image = image

    def rectangle(self, xy, outline=None, width=None, fill=None):
        return self

    def ellipse(self, xy, outline=None, width=None, fill=None):
        return self

    def polygon(self, xy, outline=None, width=None, fill=None):
        return self

    def line(self, xy, fill=None, width=None):
        return self

# Моки для аннотаций

def get_sample_annotations():
    """Возвращает пример аннотаций для тестирования"""
    return {
        "1": {
            "Трещина_№1": [10, 10, 50, 50],
            "Непровар_№1": [20, 20, 60, 60]
        },
        "2": {
            "Одиночное_включение_№1": [30, 30, 40, 40],
            "Скопление_пор_№1": [40, 40, 80, 80]
        },
        "3": {},
        "4": {
            "Трещина_№1": [15, 15, 45, 45]
        },
        "5": {}
    }

def get_sample_summary():
    """Возвращает пример summary для тестирования evaluator"""
    return {
        'total_images': 100,
        'successful': 100,
        'failed': 0,
        'average_scores': {'overall': 75.5},
        'average_metrics': {
            'brightness': 125,
            'contrast': 35,
            'entropy': 5.5,
            'edge_density': 0.1,
            'file_size_kb': 150
        },
        'defect_statistics': {
            'Трещина': 25,
            'Непровар': 20,
            'Одиночное_включение': 15,
            'Скопление_пор': 10,
            'empty': 30
        },
        'images_with_defects': 70,
        'average_defects_per_image': 1.5,
        'score_distribution': {
            'excellent': 20,
            'good': 40,
            'fair': 30,
            'poor': 10
        },
        'detailed_results': [
            {
                'overall_score': 75 + i % 20,
                'brightness': 125 + (i % 40 - 20),
                'contrast': 35 + (i % 20 - 10),
                'entropy': 5.5 + (i % 3 - 1.5),
                'edge_density': 0.1 + (i % 20) / 100,
                'file_size_kb': 150,
                'filename': f'image_{i}.png',
                'image_id': str(i + 1),
                'defect_count': i % 3
            } for i in range(100)
        ],
        'generation_date': '2024-01-01 12:00:00'
    }

# Экспортируем все моки
__all__ = [
    'MockCrackGenerator',
    'MockPoreGenerator',
    'MockFusionGenerator',
    'MockEmptySeam',
    'MockCV2',
    'MockPILImage',
    'MockImageDraw',
    'get_sample_annotations',
    'get_sample_summary'
]