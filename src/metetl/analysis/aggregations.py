from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Counter as CounterType, Dict, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metetl.decorators import timing
from metetl.logging_config import logger


CHUNK_SIZE = 50_000
MIN_COUNT = 10
TOP_N = 10
MOVING_WINDOW = 5

VARIANT_USECOLS = ["Culture", "AccessionYear", "Object Begin Date"]
EXTRA_FIELDS = ["Department", "Culture", "Medium", "Classification", "Country"]


def read_chunks(file_path: Path, chunksize: int, usecols: list[str]) -> Iterator[pd.DataFrame]:
    """Читает большой CSV частями, чтобы не загружать весь файл в память."""
    logger.info("Reading CSV chunks from %s", file_path)
    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        logger.debug("Chunk loaded: rows=%d usecols=%s", len(chunk), usecols)
        yield chunk


def extract_accession_year(series: pd.Series) -> pd.Series:
    """Извлекает 4-значный год из поля AccessionYear."""
    years = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(years, errors="coerce")


def filter_variant2_chunks(chunks: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Фильтрует данные для варианта 2 из ЛР3."""
    for chunk in chunks:
        df = chunk[VARIANT_USECOLS].copy()

        df["Culture"] = df["Culture"].astype(str).str.strip()
        df.loc[df["Culture"].isin(["", "nan", "None"]), "Culture"] = np.nan

        df["AccessionYear"] = extract_accession_year(df["AccessionYear"])
        df["Object Begin Date"] = pd.to_numeric(df["Object Begin Date"], errors="coerce")

        df = df.dropna(subset=["Culture", "AccessionYear", "Object Begin Date"])

        df["AccessionYear"] = df["AccessionYear"].astype(int)
        df["Object Begin Date"] = df["Object Begin Date"].astype(int)
        df["ObjectAgeAtAccession"] = df["AccessionYear"] - df["Object Begin Date"]

        df = df[df["ObjectAgeAtAccession"] >= 0]

        if not df.empty:
            logger.debug("Filtered chunk rows=%d", len(df))
            yield df[["Culture", "AccessionYear", "ObjectAgeAtAccession"]]


def aggregate_variant2_chunks(chunks: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Агрегирует чанки по Culture и AccessionYear."""
    aggregated_df = pd.DataFrame(
        columns=["count", "sum_age", "sum_sq_age"],
        dtype=float,
    )

    aggregated_df.index = pd.MultiIndex.from_tuples(
        [],
        names=["Culture", "AccessionYear"],
    )

    for chunk in chunks:
        chunk = chunk.copy()
        chunk["age_sq"] = chunk["ObjectAgeAtAccession"] ** 2

        chunk_agg = (
            chunk
            .groupby(["Culture", "AccessionYear"])
            .agg(
                count=("ObjectAgeAtAccession", "count"),
                sum_age=("ObjectAgeAtAccession", "sum"),
                sum_sq_age=("age_sq", "sum"),
            )
        )

        aggregated_df = aggregated_df.add(chunk_agg, fill_value=0)

    aggregated_df = aggregated_df.reset_index()

    aggregated_df["count"] = aggregated_df["count"].astype(int)
    aggregated_df["AccessionYear"] = aggregated_df["AccessionYear"].astype(int)
    aggregated_df["sum_age"] = aggregated_df["sum_age"].astype(float)
    aggregated_df["sum_sq_age"] = aggregated_df["sum_sq_age"].astype(float)

    logger.info("Aggregation finished: groups=%d", len(aggregated_df))
    yield aggregated_df


def finalize_variant2(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Считает среднее, дисперсию, стандартное отклонение, CI и PI."""
    culture_stats = (
        aggregated_df
        .groupby("Culture", as_index=False)
        .agg(
            count=("count", "sum"),
            sum_age=("sum_age", "sum"),
            sum_sq_age=("sum_sq_age", "sum"),
        )
    )

    culture_stats = culture_stats[culture_stats["count"] >= MIN_COUNT].copy()

    if culture_stats.empty:
        raise RuntimeError("After filtering there is no data for analysis")

    culture_stats["mean_age"] = culture_stats["sum_age"] / culture_stats["count"]

    culture_stats["variance"] = (
        culture_stats["sum_sq_age"]
        - (culture_stats["sum_age"] ** 2) / culture_stats["count"]
    ) / (culture_stats["count"] - 1)

    culture_stats["variance"] = culture_stats["variance"].fillna(0).clip(lower=0)
    culture_stats["std_age"] = np.sqrt(culture_stats["variance"])

    culture_stats["ci_95"] = 1.96 * culture_stats["std_age"] / np.sqrt(culture_stats["count"])
    culture_stats["pi_95"] = 1.96 * culture_stats["std_age"]

    culture_stats["ci_lower"] = culture_stats["mean_age"] - culture_stats["ci_95"]
    culture_stats["ci_upper"] = culture_stats["mean_age"] + culture_stats["ci_95"]
    culture_stats["pi_lower"] = culture_stats["mean_age"] - culture_stats["pi_95"]
    culture_stats["pi_upper"] = culture_stats["mean_age"] + culture_stats["pi_95"]

    year_sum_dict = (
        aggregated_df
        .groupby("Culture")
        .apply(lambda x: dict(zip(x["AccessionYear"], x["sum_age"])))
        .to_dict()
    )

    year_count_dict = (
        aggregated_df
        .groupby("Culture")
        .apply(lambda x: dict(zip(x["AccessionYear"], x["count"])))
        .to_dict()
    )

    culture_stats["year_sum_age"] = culture_stats["Culture"].map(year_sum_dict)
    culture_stats["year_count"] = culture_stats["Culture"].map(year_count_dict)
    culture_stats = culture_stats.rename(columns={"Culture": "culture"})

    return (
        culture_stats[
            [
                "culture",
                "count",
                "mean_age",
                "std_age",
                "ci_95",
                "pi_95",
                "ci_lower",
                "ci_upper",
                "pi_lower",
                "pi_upper",
                "year_sum_age",
                "year_count",
            ]
        ]
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def run_variant2_pipeline(file_path: Path, chunksize: int) -> pd.DataFrame:
    """Запускает основной анализ ЛР3."""
    chunks = read_chunks(file_path, chunksize, VARIANT_USECOLS)
    filtered = filter_variant2_chunks(chunks)
    aggregated = aggregate_variant2_chunks(filtered)
    aggregated_df = next(aggregated)
    return finalize_variant2(aggregated_df)


def plot_top_cultures(results: pd.DataFrame, output_path: Path) -> None:
    """Строит график топ-10 культур по среднему возрасту объекта."""
    top10 = results.nlargest(TOP_N, "count").sort_values("mean_age")
    x = np.arange(len(top10))

    _, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x, top10["mean_age"], label="Mean object age at accession")

    ci_yerr = np.vstack(
        [
            top10["mean_age"] - top10["ci_lower"],
            top10["ci_upper"] - top10["mean_age"],
        ]
    )

    ax.errorbar(
        x,
        top10["mean_age"],
        yerr=ci_yerr,
        fmt="none",
        ecolor="black",
        elinewidth=2.5,
        capsize=7,
        capthick=2.5,
        label="95% CI",
        zorder=5,
    )

    for i, (_, row) in enumerate(top10.iterrows()):
        ax.plot([i, i], [row["pi_lower"], row["pi_upper"]], "--", linewidth=1.5, alpha=0.7)

    ax.plot([], [], "--", linewidth=1.5, alpha=0.7, label="95% PI")
    ax.set_xticks(x)
    ax.set_xticklabels(top10["culture"], rotation=45, ha="right")
    ax.set_ylabel("Object age at accession, years")
    ax.set_title("Variant 2: Top-10 cultures by frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved: %s", output_path)


def build_timeline(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит временной ряд среднего возраста по годам поступления."""
    year_sum_age: Dict[int, float] = row["year_sum_age"]
    year_count: Dict[int, int] = row["year_count"]

    min_year = min(year_count.keys())
    max_year = max(year_count.keys())
    full_years = np.arange(min_year, max_year + 1, dtype=np.int32)

    mean_ages = []
    for year in full_years:
        year = int(year)
        if year in year_count:
            mean_ages.append(year_sum_age[year] / year_count[year])
        else:
            mean_ages.append(np.nan)

    mean_ages = np.array(mean_ages, dtype=np.float64)
    smooth = (
        pd.Series(mean_ages)
        .rolling(window=MOVING_WINDOW, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    return full_years, mean_ages, smooth


def plot_longest_history_timeline(results: pd.DataFrame, output_path: Path) -> None:
    """Строит график динамики для культуры с длинной историей."""
    candidates = results.copy()
    candidates["n_years"] = candidates["year_count"].apply(len)
    candidates = candidates[(candidates["count"] >= 100) & (candidates["n_years"] >= 5)]

    if candidates.empty:
        culture_row = results.loc[results["mean_age"].idxmax()]
    else:
        culture_row = candidates.loc[candidates["mean_age"].idxmax()]

    culture_name = culture_row["culture"]
    years, mean_ages, smooth = build_timeline(culture_row)

    _, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(years, mean_ages, alpha=0.6, label="Mean age by accession year")

    smooth_line = pd.Series(smooth).interpolate(limit_direction="both").to_numpy()
    ax.plot(years, smooth_line, linewidth=2.5, label=f"Moving average ({MOVING_WINDOW} years)")

    ax.set_xlabel("Accession year")
    ax.set_ylabel("Mean object age at accession, years")
    ax.set_title(f"Culture with the longest history: {culture_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved: %s", output_path)


def basic_clean(value: object) -> str:
    """Базовая очистка категориального значения."""
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "unknown", "not assigned"}:
        return "Unknown"

    return re.sub(r"\s+", " ", text)


def normalize_category(value: object) -> str:
    """Нормализует категориальное значение для дополнительного анализа ЛР3."""
    text = basic_clean(value)
    if text == "Unknown":
        return text

    text = text.replace("?", "").strip()

    prefixes = ["probably ", "possibly ", "perhaps ", "maybe ", "ca. ", "circa "]
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            lower = text.lower()

    for separator in [" or ", " and ", ";", "|", "/"]:
        if separator in text:
            text = text.split(separator)[0].strip()

    aliases = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.": "United States",
        "united states of america": "United States",
        "england": "United Kingdom",
        "scotland": "United Kingdom",
        "holland": "Netherlands",
        "oil paint on canvas": "Oil on canvas",
        "watercolour": "Watercolor",
        "painting": "Paintings",
        "print": "Prints",
        "drawing": "Drawings",
    }

    key = text.lower()
    if key in aliases:
        return aliases[key]

    return text[:1].upper() + text[1:] if text else "Unknown"


def gini_from_counter(counter: CounterType[str]) -> float:
    """Считает коэффициент Джини для распределения категорий."""
    counts = np.array(list(counter.values()), dtype=np.float64)
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0

    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)

    return float((2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n)


def normalized_entropy_from_counter(counter: CounterType[str]) -> float:
    """Считает нормированную энтропию Шеннона."""
    counts = np.array(list(counter.values()), dtype=np.float64)
    if len(counts) <= 1 or counts.sum() == 0:
        return 0.0

    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    return float(entropy / np.log(len(probabilities)))


def enc_from_counter(counter: CounterType[str]) -> float:
    """Считает эффективное количество категорий."""
    counts = np.array(list(counter.values()), dtype=np.float64)
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0

    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    return float(np.exp(entropy))


def metrics_row(field: str, stage: str, counter: CounterType[str]) -> dict[str, float | int | str]:
    """Формирует одну строку таблицы метрик качества категорий."""
    total = sum(counter.values())
    unique = len(counter)
    gini = gini_from_counter(counter)
    entropy = normalized_entropy_from_counter(counter)
    enc = enc_from_counter(counter)
    enc_normalized = enc / unique if unique else 0.0

    return {
        "field": field,
        "stage": stage,
        "total_values": total,
        "unique_categories": unique,
        "gini": gini,
        "normalized_shannon_entropy": entropy,
        "effective_number_categories": enc,
        "enc_normalized": enc_normalized,
    }


def run_extra_analysis(file_path: Path, chunksize: int) -> pd.DataFrame:
    """Запускает дополнительный категориальный анализ из ЛР3."""
    raw_counters = {field: Counter() for field in EXTRA_FIELDS}
    normalized_counters = {field: Counter() for field in EXTRA_FIELDS}

    for chunk in read_chunks(file_path, chunksize, EXTRA_FIELDS):
        for field in EXTRA_FIELDS:
            raw_values = chunk[field].map(basic_clean)
            normalized_values = chunk[field].map(normalize_category)

            raw_counters[field].update(raw_values.tolist())
            normalized_counters[field].update(normalized_values.tolist())

    rows = []
    for field in EXTRA_FIELDS:
        raw = metrics_row(field, "raw", raw_counters[field])
        normalized = metrics_row(field, "normalized", normalized_counters[field])

        rows.append(raw)
        rows.append(normalized)

        rows.append(
            {
                "field": field,
                "stage": "improvement",
                "total_values": raw["total_values"],
                "unique_categories": raw["unique_categories"] - normalized["unique_categories"],
                "gini": normalized["gini"] - raw["gini"],
                "normalized_shannon_entropy": normalized["normalized_shannon_entropy"] - raw["normalized_shannon_entropy"],
                "effective_number_categories": normalized["effective_number_categories"] - raw["effective_number_categories"],
                "enc_normalized": normalized["enc_normalized"] - raw["enc_normalized"],
            }
        )

    return pd.DataFrame(rows)


def plot_extra_heatmap(metrics: pd.DataFrame, stage: str, output_path: Path) -> None:
    """Строит heatmap качества категориальных полей."""
    data = metrics[metrics["stage"] == stage].set_index("field")
    heatmap = data[["gini", "normalized_shannon_entropy", "enc_normalized"]].astype(float)

    labels = ["Gini", "Normalized entropy", "ENC normalized"]

    _, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_title(f"Categorical field quality heatmap: {stage}")

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(j, i, f"{heatmap.iloc[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(image, ax=ax, label="0..1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved: %s", output_path)


def write_extra_report(metrics: pd.DataFrame, output_path: Path) -> None:
    """Создает текстовый отчет по дополнительному анализу ЛР3."""
    raw = metrics[metrics["stage"] == "raw"].set_index("field")
    normalized = metrics[metrics["stage"] == "normalized"].set_index("field")
    improvement = metrics[metrics["stage"] == "improvement"].set_index("field")

    lines = [
        "Дополнительное задание: оценка качества категориальных полей",
        "",
        "Исследованные поля: " + ", ".join(EXTRA_FIELDS),
        "",
        "Источники проблем:",
        "- пропуски и разные обозначения пустых значений;",
        "- лишние пробелы и знаки вопроса;",
        "- приставки неопределенности: probably, possibly, ca., circa;",
        "- составные значения через or, and, ;, |, /;",
        "- синонимы и разные варианты записи одной категории.",
        "",
        "Правила нормализации:",
        "- пустые значения приводятся к Unknown;",
        "- лишние пробелы удаляются;",
        "- знаки вопроса и приставки неопределенности удаляются;",
        "- составные категории упрощаются до первого основного значения;",
        "- часть частых синонимов приводится к единому виду.",
        "",
        "Оценка потенциального улучшения:",
    ]

    for field in EXTRA_FIELDS:
        lines.append("")
        lines.append(f"{field}:")
        lines.append(
            f"- уникальных категорий: {int(raw.loc[field, 'unique_categories'])} → "
            f"{int(normalized.loc[field, 'unique_categories'])}; "
            f"уменьшение: {int(improvement.loc[field, 'unique_categories'])}"
        )
        lines.append(f"- Gini: {raw.loc[field, 'gini']:.3f} → {normalized.loc[field, 'gini']:.3f}")
        lines.append(
            f"- normalized Shannon entropy: "
            f"{raw.loc[field, 'normalized_shannon_entropy']:.3f} → "
            f"{normalized.loc[field, 'normalized_shannon_entropy']:.3f}"
        )
        lines.append(f"- ENC normalized: {raw.loc[field, 'enc_normalized']:.3f} → {normalized.loc[field, 'enc_normalized']:.3f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved: %s", output_path)


@timing
def run_analysis(csv_path: Path, output_dir: Path, chunksize: int = CHUNK_SIZE) -> None:
    """Главная функция анализа CSV для CLI-команды metetl analyze."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting main analysis from lab 3")
    results = run_variant2_pipeline(csv_path, chunksize)

    results_to_save = results.drop(columns=["year_sum_age", "year_count"])
    results_path = output_dir / "variant2_results.csv"
    results_to_save.to_csv(results_path, index=False, encoding="utf-8-sig")
    logger.info("Analysis table saved: %s", results_path)

    plot_top_cultures(results, output_dir / "variant2_top10_cultures.png")
    plot_longest_history_timeline(results, output_dir / "variant2_longest_history_timeline.png")

    logger.info("Starting extra categorical analysis from lab 3")
    extra_metrics = run_extra_analysis(csv_path, chunksize)
    extra_metrics_path = output_dir / "extra_categorical_quality_metrics.csv"
    extra_metrics.to_csv(extra_metrics_path, index=False, encoding="utf-8-sig")
    logger.info("Extra metrics saved: %s", extra_metrics_path)

    plot_extra_heatmap(extra_metrics, "raw", output_dir / "extra_quality_heatmap_raw.png")
    plot_extra_heatmap(extra_metrics, "normalized", output_dir / "extra_quality_heatmap_normalized.png")
    write_extra_report(extra_metrics, output_dir / "extra_normalization_report.txt")

    logger.info("CSV analysis completed")
