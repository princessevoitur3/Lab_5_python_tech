from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from metetl.analysis.aggregations import run_analysis
from metetl.analysis.data_to_download import prepare_download_metadata
from metetl.images.processing import run_processing_pipeline
from metetl.logging_config import logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metetl",
        description="ЛР5: логирование, CLI, модули, пакеты, сборка Python-проекта",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Подготовить JSON с метаданными изображений по случайным Object ID из CSV",
    )
    prepare_parser.add_argument("--csv", required=True, help="Путь к MetObjects.csv")
    prepare_parser.add_argument("--output", required=True, help="Куда сохранить JSON с метаданными")
    prepare_parser.add_argument("--num", type=int, default=1, help="Сколько изображений найти")
    prepare_parser.add_argument("--max-attempts", type=int, default=200, help="Максимум случайных ID для проверки")
    prepare_parser.add_argument("--metadata-concurrency", type=int, default=12, help="Сколько запросов к MET API выполнять параллельно")

    process_parser = subparsers.add_parser(
        "process",
        help="Скачать и обработать изображения из подготовленного JSON",
    )
    process_parser.add_argument("--input", required=True, help="JSON с метаданными изображений")
    process_parser.add_argument("--output", required=True, help="Папка для результатов")
    process_parser.add_argument("--num", type=int, default=1, help="Сколько изображений скачать и обработать")
    process_parser.add_argument("--workers", type=int, default=1, help="Количество процессов для обработки")
    process_parser.add_argument("--download-concurrency", type=int, default=4, help="Количество одновременных скачиваний")
    process_parser.add_argument("--save-concurrency", type=int, default=4, help="Количество одновременных сохранений")
    process_parser.add_argument("--max-side", type=int, default=700, help="Максимальная сторона изображения перед сверткой")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Запустить анализ CSV из лабораторной №3",
    )
    analyze_parser.add_argument("--csv", required=True, help="Путь к MetObjects.csv или MetObjects.csv")
    analyze_parser.add_argument("--output", required=True, help="Папка для CSV-результатов и графиков")
    analyze_parser.add_argument("--chunksize", type=int, default=50_000, help="Размер чанка для чтения большого CSV")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        logger.info("Program started")
        logger.debug("CLI args: %s", args)

        if args.command == "prepare":
            prepare_download_metadata(
                csv_path=Path(args.csv),
                output_path=Path(args.output),
                limit=args.num,
                max_attempts=args.max_attempts,
                metadata_concurrency=args.metadata_concurrency,
            )
        elif args.command == "process":
            asyncio.run(
                run_processing_pipeline(
                    input_json=Path(args.input),
                    output_root=Path(args.output),
                    num=args.num,
                    workers=args.workers,
                    download_concurrency=args.download_concurrency,
                    save_concurrency=args.save_concurrency,
                    max_side=args.max_side,
                )
            )
        elif args.command == "analyze":
            run_analysis(
                csv_path=Path(args.csv),
                output_dir=Path(args.output_dir),
                chunksize=args.chunksize,
            )
        else:
            parser.print_help()
            return 1

        logger.info("Program finished successfully")
        return 0

    except Exception as error:
        logger.exception("Program finished with error: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
