import logging
import os
from datetime import datetime

class LoggerManager:
    _loggers = {}
    _level = logging.INFO

    @classmethod
    def initialize(cls, level=None):
        # Si pasan level explícito, lo usa
        if level is not None:
            if level != logging.INFO:
                cls._level = level
                return


    @classmethod
    def get_logger(cls, name: str = "AppLogger", log_dir: str = "logs"):
        if name in cls._loggers:
            return cls._loggers[name]

        # Crear logger base
        logger = logging.getLogger(name)
        logger.setLevel(cls._level)
        logger.propagate = False  # Evita duplicar logs

        if not logger.handlers:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        cls._loggers[name] = logger
        return logger

class LoggerMixin:
    def __init__(self):
        class_name = self.__class__.__name__
        self.logger = LoggerManager.get_logger(name=class_name)