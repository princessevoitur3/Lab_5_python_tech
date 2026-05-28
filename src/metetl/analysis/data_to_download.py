from __future__ import annotations

import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Any

import aiohttp

from metetl.decorators import timing
from metetl.logging_config import logger


MET_API_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
USER_AGENT = "Mozilla/5.0 Lab5MetETL/1.0"


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict[str, Any] | None:
    """Асинхронно загружает JSON по URL с несколькими попытками."""
    for attempt in range(1, retries + 1):
        try:
            async with semaphore:
                logger.debug("GET metadata: %s, attempt=%s", url, attempt)
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.debug("Metadata request failed: status=%s url=%s", response.status, url)
                        return None

                    data = await response.json(content_type=None)
                    if isinstance(data, dict):
                        return data
                    return None
        except Exception as error:
            logger.warning("Metadata request error, attempt %s/%s: %s", attempt, retries, error)
            await asyncio.sleep(0.2 * attempt)

    return None


def load_object_ids_from_csv(csv_path: Path) -> list[str]:
    """
    Читает Object ID из MetObjects.csv/MetObjects.csv.

    Приоритет отдается объектам с Classification == Paintings,
    потому что у них чаще есть изображения.
    Если таких объектов нет, берутся все Object ID.
    """
    logger.info("Loading object IDs from CSV: %s", csv_path)

    painting_ids: list[str] = []
    all_ids: list[str] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            object_id = (row.get("Object ID") or "").strip()
            if not object_id:
                continue

            all_ids.append(object_id)

            if row.get("Classification") == "Paintings":
                painting_ids.append(object_id)

    if painting_ids:
        logger.info("Loaded %d painting object IDs", len(painting_ids))
        return painting_ids

    if all_ids:
        logger.warning("No Paintings found, using all %d object IDs", len(all_ids))
        return all_ids

    raise RuntimeError("CSV does not contain Object ID values")


async def build_random_download_jobs(
    object_ids: list[str],
    limit: int,
    max_attempts: int,
    metadata_concurrency: int,
) -> list[dict[str, Any]]:
    """
    Берет случайные Object ID, запрашивает MET API и выбирает объекты с картинкой.

    Важное отличие от медленного последовательного варианта:
    - список ID перемешивается случайно;
    - если у объекта нет картинки, он просто пропускается;
    - как только найдено нужное количество картинок, работа останавливается.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    shuffled_ids = object_ids[:]
    random.shuffle(shuffled_ids)
    shuffled_ids = shuffled_ids[:max_attempts]

    timeout = aiohttp.ClientTimeout(total=60)
    headers = {"User-Agent": USER_AGENT}
    semaphore = asyncio.Semaphore(metadata_concurrency)
    jobs: list[dict[str, Any]] = []

    logger.info("Searching random image metadata: need=%d, max_attempts=%d", limit, max_attempts)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        cursor = 0
        batch_size = max(metadata_concurrency * 2, 10)

        while len(jobs) < limit and cursor < len(shuffled_ids):
            batch = shuffled_ids[cursor: cursor + batch_size]
            cursor += batch_size

            tasks = [
                asyncio.create_task(
                    fetch_json(
                        session=session,
                        url=MET_API_OBJECT_URL.format(object_id=object_id),
                        semaphore=semaphore,
                    )
                )
                for object_id in batch
            ]

            for object_id, task in zip(batch, tasks):
                metadata = await task
                if not metadata:
                    logger.debug("Object skipped, metadata missing: object_id=%s", object_id)
                    continue

                image_url = metadata.get("primaryImage") or metadata.get("primaryImageSmall")
                if not image_url:
                    logger.debug("Object skipped, image missing: object_id=%s", object_id)
                    continue

                job = {
                    "index": len(jobs) + 1,
                    "object_id": str(metadata.get("objectID", object_id)),
                    "title": metadata.get("title") or "Untitled",
                    "artist": metadata.get("artistDisplayName") or "Unknown",
                    "classification": metadata.get("classification") or "Unknown",
                    "image_url": image_url,
                    "metadata": metadata,
                }
                jobs.append(job)
                logger.info("Image metadata found: %d/%d, object_id=%s", len(jobs), limit, job["object_id"])

                if len(jobs) >= limit:
                    logger.info("Required number of images found, stopping search")
                    break

    if not jobs:
        raise RuntimeError("No objects with images were found. Try increasing --max-attempts")

    if len(jobs) < limit:
        logger.warning("Only %d images found instead of %d", len(jobs), limit)

    return jobs


@timing
def prepare_download_metadata(
    csv_path: Path,
    output_path: Path,
    limit: int = 1,
    max_attempts: int = 200,
    metadata_concurrency: int = 12,
) -> list[dict[str, Any]]:
    """Готовит JSON-файл с метаданными изображений."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    object_ids = load_object_ids_from_csv(csv_path)
    jobs = asyncio.run(
        build_random_download_jobs(
            object_ids=object_ids,
            limit=limit,
            max_attempts=max_attempts,
            metadata_concurrency=metadata_concurrency,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Download metadata saved to %s", output_path)
    return jobs
