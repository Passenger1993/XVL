# train.py (для командной строки)
import argparse
import sys
from pathlib import Path

# Импортируем модуль
from yolo_trainer import train_pipeline, logger, setup_logging

def main():
    """Точка входа для командной строки"""
    parser = argparse.ArgumentParser(
        description='Обучение YOLO для детектирования дефектов сварных швов'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Путь к конфигурационному файлу YAML')
    parser.add_argument('--data-dir', type=str, required=True,
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