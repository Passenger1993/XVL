import logging
import logging.config
import yaml
from pathlib import Path
import sys
from datetime import datetime

def setup_logging(
    config_path: str = 'configs/logging.yaml',
    default_level=logging.INFO,
    experiment_name: str = None
):
    """Настройка логирования с созданием директорий для эксперимента"""

    # Создаем имя эксперимента
    if not experiment_name:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Создаем директорию для логов эксперимента
    log_dir = Path('logs') / 'training' / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем конфиг из YAML если есть
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Динамически подставляем пути в конфиг
        for handler in config['handlers'].values():
            if 'filename' in handler:
                # Заменяем {experiment_name} в путях
                handler['filename'] = handler['filename'].format(
                    experiment_name=experiment_name,
                    log_dir=str(log_dir)
                )

        logging.config.dictConfig(config)
    else:
        # Дефолтная конфигурация если YAML нет
        logging.basicConfig(level=default_level)

    # Создаем симлинк current -> текущий эксперимент
    current_link = Path('logs/training/current')
    if current_link.exists():
        current_link.unlink()
    current_link.symlink_to(experiment_name)

    logger = logging.getLogger(__name__)
    logger.info(f"Логирование инициализировано: {log_dir}")

    return log_dir

class MetricLogger:
    """Специальный логгер для метрик ML"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_file = log_dir / 'metrics.json'
        self.metrics = []

    def log_epoch(self, epoch: int, metrics: dict):
        """Логирует метрики эпохи в JSON"""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)

        # Сохраняем в JSON
        import json
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Также пишем в обычный лог
        logger = logging.getLogger('metrics')
        logger.info(f"Epoch {epoch}: {metrics}")