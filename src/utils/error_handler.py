import logging
import traceback
from functools import wraps
from typing import Callable, Any
from exсeptions import ResourceExhaustedError, TrainingError, DataGenerationError

logger = logging.getLogger(__name__)

def error_handler(logger_name: str = None, raise_again: bool = False):
    """
    Декоратор для автоматического логирования ошибок

    Args:
        logger_name: Имя логгера (если None, используется имя модуля)
        raise_again: Поднимать исключение снова после логирования
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Получаем логгер
            func_logger = logging.getLogger(logger_name or func.__module__)

            try:
                func_logger.debug(f"Вызов {func.__name__} с args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                func_logger.debug(f"Функция {func.__name__} завершилась успешно")
                return result

            except ResourceExhaustedError as e:
                # Критическая ошибка ресурсов
                func_logger.critical(
                    f"Ресурсы исчерпаны в {func.__name__}: {e.message}",
                    extra={'details': e.details}
                )
                if raise_again:
                    raise
                return None

            except (TrainingError, DataGenerationError) as e:
                # Ошибки обучения/данных
                func_logger.error(
                    f"Ошибка в {func.__name__}: {e.message}",
                    extra={'details': e.details, 'traceback': traceback.format_exc()}
                )
                if raise_again:
                    raise
                return None

            except Exception as e:
                # Непредвиденные ошибки
                func_logger.exception(
                    f"Непредвиденная ошибка в {func.__name__}: {str(e)}",
                    extra={'traceback': traceback.format_exc()}
                )
                if raise_again:
                    raise
                return None

        return wrapper
    return decorator