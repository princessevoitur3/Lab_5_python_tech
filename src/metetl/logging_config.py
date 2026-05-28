from __future__ import annotations

import logging
import sys
from pathlib import Path


LOGGER_NAME = "metetl"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"


def setup_logger() -> logging.Logger:
    """
    Создает и настраивает общий логгер проекта.

    Требования лабораторной:
    - подробные DEBUG-логи пишутся в файл logs/app.log;
    - краткие INFO-логи выводятся в консоль;
    - файловая запись содержит время, имя файла и номер строки.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="%(levelname)s | %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()
