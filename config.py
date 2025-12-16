# configs.py или src/configs/settings.py
import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Конфигурация модели"""
    weights_path: str = "models/best.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
@dataclass
class GenerationConfig:
    """Конфигурация генерации данных"""
    image_size: tuple = (640, 640)
    crack_probability: float = 0.3
    pore_probability: float = 0.2
    max_defects_per_image: int = 5
    
@dataclass
class PathsConfig:
    """Конфигурация путей"""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    examples_dir: Path = project_root / "examples"
    
    def setup_directories(self):
        """Создаёт необходимые директории"""
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.examples_dir.mkdir(exist_ok=True)

# Экспортируем конфигурации
model_config = ModelConfig()
generation_config = GenerationConfig()
paths = PathsConfig()