from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DownloadJob:
    """Описание одной задачи на скачивание изображения."""

    index: int
    object_id: str
    image_url: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DownloadedImage:
    """Изображение после скачивания, но до обработки."""

    job: DownloadJob
    image_bytes: bytes


@dataclass(frozen=True)
class ProcessedImage:
    """Изображение после обработки."""

    job: DownloadJob
    original_bytes: bytes
    processed_png_bytes: bytes
    worker_pid: int
    width: int
    height: int


@dataclass(frozen=True)
class SavedImage:
    """Информация о сохраненных файлах результата."""

    job: DownloadJob
    original_path: Path
    processed_path: Path
    metadata_path: Path


class Artwork:
    """Класс с базовыми операциями над изображением для тестов и обработки."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def to_grayscale(self) -> np.ndarray:
        """Переводит RGB/BGR-изображение в оттенки серого."""
        if self.image.ndim == 2:
            return self.image.astype(np.float32)

        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    @staticmethod
    def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Создает ядро Гаусса для размытия."""
        axis = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        kernel = np.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2))).astype(np.float32)
        kernel /= np.sum(kernel)
        return kernel

    @staticmethod
    def convolve2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Выполняет ручную двумерную свертку изображения с ядром."""
        image = image.astype(np.float32)
        kernel = np.flipud(np.fliplr(kernel.astype(np.float32)))

        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2

        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        height, width = image.shape
        result = np.zeros((height, width), dtype=np.float32)

        for row in range(height):
            for col in range(width):
                window = padded[row: row + kh, col: col + kw]
                result[row, col] = np.sum(window * kernel)

        return result
