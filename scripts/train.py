#!/usr/bin/env python3
import sys
import traceback
from pathlib import Path
import logging

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.exсeptions import XVLBaseException

def main():
    """Основная функция с глобальной обработкой ошибок"""
    try:
        # Настройка логирования ПЕРВЫМ делом
        log_dir = setup_logging(experiment_name="training_run")
        logger = logging.getLogger('xvl.script')

        # Логируем старт
        logger.info("=" * 60)
        logger.info("ЗАПУСК ОБУЧЕНИЯ XVL")
        logger.info("=" * 60)

        # Инициализация и обучение
        from src.trainer import YOLOTrainer
        from src.config import TrainingConfig

        config = TrainingConfig.from_yaml('configs/training.yaml')
        logger.info(f"Загружена конфигурация: {config}")

        trainer = YOLOTrainer(config)
        logger.info("Тренер инициализирован")

        # Запуск обучения
        results = trainer.train()
        logger.info(f"Обучение завершено: {results}")

        return 0

    except KeyboardInterrupt:
        logger = logging.getLogger('xvl.script')
        logger.warning("Обучение прервано пользователем (Ctrl+C)")

        # Пытаемся сохранить состояние
        if 'trainer' in locals():
            trainer.save_checkpoint('interrupted.pth')
            logger.info("Сохранён чекпоинт 'interrupted.pth'")

        return 130  # Стандартный код для SIGINT

    except XVLBaseException as e:
        # Обработка наших кастомных исключений
        logger = logging.getLogger('xvl.script')
        logger.error(
            f"Ошибка XVL: {e.message}",
            extra={'details': e.details, 'type': type(e).__name__}
        )
        return 1

    except Exception as e:
        # Ловим всё остальное
        logger = logging.getLogger('xvl.script')
        logger.critical(
            f"Критическая ошибка: {str(e)}",
            extra={'traceback': traceback.format_exc()}
        )
        return 1

    finally:
        # Всегда выполняем этот блок
        logger = logging.getLogger('xvl.script')
        logger.info("=" * 60)
        logger.info("ЗАВЕРШЕНИЕ РАБОТЫ СКРИПТА")
        logger.info("=" * 60)

        # Закрываем все файловые дескрипторы логгеров
        logging.shutdown()

if __name__ == "__main__":
    sys.exit(main())