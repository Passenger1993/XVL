# train.py (для командной строки)
import argparse
import sys
from pathlib import Path

# Импортируем модуль
from yolo_trainer import train_pipeline, logger, setup_logging

def main():
    """Точка входа для командной строки"""
    # Определяем корень проекта
    project_root = Path(__file__).parent.parent.parent  # Из src/trainer поднимаемся до XVL

    parser = argparse.ArgumentParser(
        description='Обучение YOLO для детектирования дефектов сварных швов'
    )
    parser.add_argument('--config', type=str,
                       default=str(project_root / "configs" / "train_configuration.yaml"),
                       help='Путь к конфигурационному файлу YAML')
    parser.add_argument('--data-dir', type=str,
                       default=str(project_root / "data/training"),
                       help='Путь к директории с данными')
    parser.add_argument('--resume', action='store_true',
                       help='Возобновить обучение с последнего чекпоинта')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', 'mps'],
                       help='Устройство для обучения (cuda/cpu/mps)')
    parser.add_argument('--project', type=str, default='weld_defects',
                       help='Название проекта для сохранения результатов')
    parser.add_argument('--log-file', type=str, default='training.log',
                       help='Файл для логирования')
    parser.add_argument('--save-dir', type=str, default=str(project_root / "models"),
                       help='Директория для сохранения моделей')

    args = parser.parse_args()

    # Проверяем существование файлов и директорий
    if not Path(args.config).exists():
        logger.error(f"Конфигурационный файл не найден: {args.config}")
        sys.exit(1)

    if not Path(args.data_dir).exists():
        logger.error(f"Директория с данными не найдена: {args.data_dir}")
        sys.exit(1)

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(args.log_file)

    # Запуск пайплайна
    results = train_pipeline(
        config_path=args.config,
        data_dir=args.data_dir,
        resume=args.resume,
        device=args.device,
        project_name=args.project
    )

    if not results['success']:
        logger.error(f"Обучение завершилось с ошибкой: {results.get('error')}")
        sys.exit(1)

    logger.info(f"Результаты сохранены в: {results['checkpoint_dir']}")

if __name__ == "__main__":
    main()