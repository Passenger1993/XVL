# src/config.py
from dataclasses import dataclass, asdict
from typing import Dict, Any
import yaml

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""

    # Общие параметры
    model_name: str = "yolov8s.pt"
    img_size: int = 512
    epochs: int = 150
    batch_size: int = 48
    checkpoint_interval: int = 10
    max_checkpoints: int = 5
    verbose: bool = True

    # Пути
    data_config_path: str = "configs/data.yaml"

    # Параметры обучения
    learning_rate: float = 0.00005
    weight_decay: float = 0.005
    momentum: float = 0.9

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Загружает конфиг из YAML"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Объединяем с default значениями
        defaults = asdict(cls())
        defaults.update(config_dict)

        return cls(**defaults)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return asdict(self)

    @property
    def training_kwargs(self) -> Dict:
        """Возвращает параметры для YOLO.train()"""
        return {
            'lr0': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'patience': 50,
            'save': True,
            'exist_ok': True,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'plots': True
        }