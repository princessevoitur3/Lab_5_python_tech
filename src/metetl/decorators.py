from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from metetl.logging_config import logger


P = ParamSpec("P")
R = TypeVar("R")


def timing(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для логирования времени выполнения функции."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        logger.debug("Function %s started", func.__name__)
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug("Function %s finished in %.3f sec", func.__name__, elapsed)

    return wrapper
