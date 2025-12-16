class XVLBaseException(Exception):
    """Базовое исключение для всего проекта"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class TrainingError(XVLBaseException):
    """Ошибки обучения модели"""
    pass

class DataGenerationError(XVLBaseException):
    """Ошибки генерации данных"""
    pass

class ModelLoadError(XVLBaseException):
    """Ошибки загрузки модели"""
    pass

class ConfigError(XVLBaseException):
    """Ошибки конфигурации"""
    pass

class ResourceExhaustedError(XVLBaseException):
    """Закончились ресурсы (память, GPU)"""
    pass