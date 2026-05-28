from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import cv2
import numpy as np

from metetl.images.models import DownloadedImage, DownloadJob, ProcessedImage, SavedImage
from metetl.logging_config import logger


USER_AGENT = "Mozilla/5.0 Lab5MetETL/1.0"


def load_jobs(input_json: Path, num: int) -> list[DownloadJob]:
    """Читает JSON с метаданными и превращает записи в DownloadJob."""
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    raw_jobs = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(raw_jobs, list):
        raise ValueError("Input JSON must contain a list")

    jobs: list[DownloadJob] = []
    for item in raw_jobs[:num]:
        jobs.append(
            DownloadJob(
                index=int(item["index"]),
                object_id=str(item["object_id"]),
                image_url=str(item["image_url"]),
                metadata=dict(item.get("metadata", {})),
            )
        )

    if not jobs:
        raise RuntimeError("No jobs found in input JSON")

    logger.info("Loaded %d download jobs from %s", len(jobs), input_json)
    return jobs


async def download_one(
    session: aiohttp.ClientSession,
    job: DownloadJob,
    semaphore: asyncio.Semaphore,
) -> DownloadedImage:
    """Асинхронно скачивает одно изображение."""
    logger.info("Downloading image %s started, object_id=%s", job.index, job.object_id)

    async with semaphore:
        async with session.get(job.image_url) as response:
            response.raise_for_status()
            image_bytes = await response.read()

    if not image_bytes:
        raise RuntimeError(f"Image {job.index}: empty response")

    logger.info("Downloading image %s finished, bytes=%d", job.index, len(image_bytes))
    return DownloadedImage(job=job, image_bytes=image_bytes)


async def download_stage(
    jobs: Iterable[DownloadJob],
    download_concurrency: int,
) -> AsyncIterator[DownloadedImage]:
    """Стадия асинхронного скачивания изображений."""
    timeout = aiohttp.ClientTimeout(total=120)
    semaphore = asyncio.Semaphore(download_concurrency)
    headers = {"User-Agent": USER_AGENT}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        tasks = [asyncio.create_task(download_one(session, job, semaphore)) for job in jobs]

        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            except Exception as error:
                logger.exception("Download error: %s", error)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Создает ядро Гаусса."""
    axis = np.linspace(-(size // 2), size // 2, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2))).astype(np.float32)
    kernel /= np.sum(kernel)
    return kernel


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Ручная двумерная свертка, перенесенная из ЛР4."""
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


def resize_for_processing(image: np.ndarray, max_side: int) -> np.ndarray:
    """Уменьшает изображение, чтобы ручная свертка не работала слишком долго."""
    if max_side <= 0:
        return image

    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image

    scale = max_side / largest_side
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def process_image_cpu(index: int, object_id: str, image_bytes: bytes, max_side: int) -> tuple[bytes, int, int, int]:
    """CPU-обработка изображения: grayscale, blur, Sobel, PNG-кодирование."""
    worker_pid = os.getpid()
    logger.debug("Processing image %s started in PID %s", index, worker_pid)

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Image {index}: OpenCV cannot decode image")

    image = resize_for_processing(image, max_side=max_side)
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = convolve2d_manual(gray, gaussian_kernel(size=5, sigma=1.0))

    sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float32,
    )
    sobel_y = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=np.float32,
    )

    gx = convolve2d_manual(blur, sobel_x)
    gy = convolve2d_manual(blur, sobel_y)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    edges = np.uint8(np.clip(edges, 0, 255))

    ok, encoded = cv2.imencode(".png", edges)
    if not ok:
        raise RuntimeError(f"Image {index}: cannot encode result as PNG")

    logger.debug("Processing image %s finished in PID %s", index, worker_pid)
    return encoded.tobytes(), worker_pid, width, height


async def process_one(
    downloaded: DownloadedImage,
    executor: ProcessPoolExecutor,
    max_side: int,
) -> ProcessedImage:
    """Передает CPU-тяжелую обработку в ProcessPoolExecutor."""
    loop = asyncio.get_running_loop()
    processed_png_bytes, worker_pid, width, height = await loop.run_in_executor(
        executor,
        process_image_cpu,
        downloaded.job.index,
        downloaded.job.object_id,
        downloaded.image_bytes,
        max_side,
    )

    return ProcessedImage(
        job=downloaded.job,
        original_bytes=downloaded.image_bytes,
        processed_png_bytes=processed_png_bytes,
        worker_pid=worker_pid,
        width=width,
        height=height,
    )


async def convolution_stage(
    downloaded_images: AsyncIterator[DownloadedImage],
    executor: ProcessPoolExecutor,
    max_in_flight: int,
    max_side: int,
) -> AsyncIterator[ProcessedImage]:
    """Стадия параллельной обработки скачанных изображений."""
    pending: set[asyncio.Task[ProcessedImage]] = set()

    async for downloaded in downloaded_images:
        task = asyncio.create_task(process_one(downloaded, executor, max_side=max_side))
        pending.add(task)

        if len(pending) >= max_in_flight:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                try:
                    yield finished.result()
                except Exception as error:
                    logger.exception("Processing error: %s", error)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            try:
                yield finished.result()
            except Exception as error:
                logger.exception("Processing error: %s", error)


async def write_bytes(path: Path, data: bytes) -> None:
    """Асинхронно записывает байты в файл."""
    async with aiofiles.open(path, "wb") as file:
        await file.write(data)


async def write_text(path: Path, text: str) -> None:
    """Асинхронно записывает текст в файл."""
    async with aiofiles.open(path, "w", encoding="utf-8") as file:
        await file.write(text)


async def save_one(processed: ProcessedImage, output_dir: Path) -> SavedImage:
    """Сохраняет оригинал, обработанную картинку и metadata JSON."""
    job = processed.job
    prefix = f"{job.index}_{job.object_id}"

    original_path = output_dir / f"{prefix}_original.jpg"
    processed_path = output_dir / f"{prefix}_processed.png"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    info_path = output_dir / f"{prefix}_processing_info.json"

    logger.info("Saving image %s started", job.index)

    metadata_json = json.dumps(job.metadata, ensure_ascii=False, indent=2)
    info_json = json.dumps(
        {
            "index": job.index,
            "object_id": job.object_id,
            "worker_pid": processed.worker_pid,
            "processed_width": processed.width,
            "processed_height": processed.height,
            "image_url": job.image_url,
        },
        ensure_ascii=False,
        indent=2,
    )

    await asyncio.gather(
        write_bytes(original_path, processed.original_bytes),
        write_bytes(processed_path, processed.processed_png_bytes),
        write_text(metadata_path, metadata_json),
        write_text(info_path, info_json),
    )

    logger.info("Saving image %s finished -> %s", job.index, processed_path)
    return SavedImage(
        job=job,
        original_path=original_path,
        processed_path=processed_path,
        metadata_path=metadata_path,
    )


async def save_stage(
    processed_images: AsyncIterator[ProcessedImage],
    output_dir: Path,
    save_concurrency: int,
) -> AsyncIterator[SavedImage]:
    """Стадия асинхронного сохранения файлов."""
    pending: set[asyncio.Task[SavedImage]] = set()

    async for processed in processed_images:
        pending.add(asyncio.create_task(save_one(processed, output_dir)))

        if len(pending) >= save_concurrency:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                try:
                    yield finished.result()
                except Exception as error:
                    logger.exception("Save error: %s", error)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            try:
                yield finished.result()
            except Exception as error:
                logger.exception("Save error: %s", error)


async def run_processing_pipeline(
    input_json: Path,
    output_root: Path,
    num: int,
    workers: int,
    download_concurrency: int,
    save_concurrency: int,
    max_side: int,
) -> list[SavedImage]:
    """Главная функция пайплайна скачивания и обработки изображений."""
    if num <= 0:
        raise ValueError("num must be positive")

    output_dir = output_root / datetime.now().strftime("lab5_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing pipeline started")
    logger.info("Input JSON: %s", input_json)
    logger.info("Output dir: %s", output_dir)
    logger.info("Requested images: %d", num)

    jobs = load_jobs(input_json, num)
    start = time.perf_counter()
    saved: list[SavedImage] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        downloaded_iter = download_stage(jobs, download_concurrency=download_concurrency)
        processed_iter = convolution_stage(
            downloaded_iter,
            executor=executor,
            max_in_flight=max(1, workers),
            max_side=max_side,
        )
        saved_iter = save_stage(
            processed_iter,
            output_dir=output_dir,
            save_concurrency=save_concurrency,
        )

        async for item in saved_iter:
            saved.append(item)

    elapsed = time.perf_counter() - start
    logger.info("Saved images: %d", len(saved))
    logger.info("Pipeline finished in %.3f sec", elapsed)
    logger.info("Result directory: %s", output_dir)

    return sorted(saved, key=lambda item: item.job.index)


class ImageProcessor:
    """Небольшая обертка над функцией обработки для тестов unittest."""

    def __init__(self, max_side: int = 700) -> None:
        self.max_side = max_side

    def process_bytes(self, image_bytes: bytes, index: int = 1, object_id: str = "mock") -> bytes:
        processed_png_bytes, _, _, _ = process_image_cpu(index, object_id, image_bytes, self.max_side)
        return processed_png_bytes
