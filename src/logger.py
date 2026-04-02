import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Определяем путь к папке логов (на уровень выше от src, в папку data/logs)
LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_system.log"

def setup_logger():
    # Создаем основной логгер проекта
    logger = logging.getLogger("RegLabRAG")
    logger.setLevel(logging.DEBUG) # Базовый уровень - ловим всё

    # Формат сообщений: Время | Уровень | Файл:Строка | Сообщение
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Запись в файл (храним 3 файла по 5 Мегабайт, чтобы не забить диск)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG) # В файл пишем абсолютно всё
    file_handler.setFormatter(formatter)

    # 2. Вывод в терминал (консоль)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # В терминал выводим только главное (INFO, WARNING, ERROR)
    console_handler.setFormatter(formatter)

    # Очищаем старые обработчики, чтобы логи не дублировались
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Создаем глобальный объект, который будут импортировать другие скрипты
log = setup_logger()