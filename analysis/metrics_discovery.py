"""Descoberta automática e consolidação de métricas dos modelos."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

METRIC_KEYS = ("accuracy", "precision", "recall", "f1")

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "acuracia", "acurácia"),
    "precision": ("precision", "precisao", "precisão"),
    "recall": ("recall", "revocacao", "revocação", "sensibilidade"),
    "f1": ("f1", "f1_score", "f1-score", "f1 score"),
}

SKIP_FILE_PATTERNS = (
    "por_fold",
    "per_fold",
    "feature_importance",
    "all_experiments",
    "confusion_matrix",
    "benchmark",
    "hiperparametr",
    "hyperparam",
    "best_params",
    "metrics_by_fold",
)

FOLDER_DISPLAY_NAMES: dict[str, str] = {
    "naive_bayes": "Naive Bayes",
    "regrassao_logisticca": "Regressão Logística",
    "regressao_logistica": "Regressão Logística",
    "logistic_regression": "Regressão Logística",
    "arvore_de_decisao": "Árvore de Decisão",
    "decision_tree": "Árvore de Decisão",
    "svm": "SVM",
    "redes_neurais": "Rede Neural",
    "neural_network": "Rede Neural",
}


def _normalize_name(value: str) -> str:
    """Normaliza nomes de colunas e métricas para comparação."""
    normalized = (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("ó", "o")
        .replace("ã", "a")
        .replace("ç", "c")
        .replace("é", "e")
    )
    return re.sub(r"_+", "_", normalized)


def _match_metric(column_name: str) -> str | None:
    """Mapeia um nome de coluna para uma métrica canônica."""
    normalized = _normalize_name(column_name)
    normalized = normalized.removesuffix("_mean").removesuffix("_media")
    normalized = normalized.removesuffix("_score")

    for metric, aliases in METRIC_ALIASES.items():
        if normalized in aliases:
            return metric
    return None


def _should_skip_file(path: Path) -> bool:
    """Ignora arquivos que não representam resumo agregado de métricas."""
    name = path.name.lower()
    return any(pattern in name for pattern in SKIP_FILE_PATTERNS)


def _display_name_from_folder(folder_name: str) -> str:
    """Converte nome da pasta em rótulo legível."""
    key = _normalize_name(folder_name)
    if key in FOLDER_DISPLAY_NAMES:
        return FOLDER_DISPLAY_NAMES[key]
    return folder_name.replace("_", " ").title()


def _parse_metric_rows(df: pd.DataFrame) -> dict[str, float] | None:
    """Interpreta CSVs no formato metrica/media ou metric/mean."""
    metric_col = None
    value_col = None

    for column in df.columns:
        normalized = _normalize_name(column)
        if normalized in {"metric", "metrica", "metricas", "métrica"}:
            metric_col = column
        if normalized in {"mean", "media", "média", "valor", "value"}:
            value_col = column

    if metric_col is None or value_col is None:
        return None

    metrics: dict[str, float] = {}
    for _, row in df.iterrows():
        metric = _match_metric(str(row[metric_col]))
        if metric and pd.notna(row[value_col]):
            metrics[metric] = float(row[value_col])

    return metrics if len(metrics) >= 3 else None


def _parse_summary_columns(df: pd.DataFrame) -> dict[str, float] | None:
    """Interpreta CSVs com colunas *_mean ou métricas diretas."""
    if df.empty:
        return None

    metrics: dict[str, float] = {}
    row = df.iloc[0]

    for column in df.columns:
        metric = _match_metric(str(column))
        if metric and pd.notna(row[column]):
            metrics[metric] = float(row[column])

    return metrics if len(metrics) >= 3 else None


def _parse_file(path: Path) -> dict[str, float] | None:
    """Tenta extrair métricas agregadas de um arquivo CSV."""
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning("Não foi possível ler %s: %s", path, exc)
        return None

    if df.empty:
        return None

    parsers = (_parse_metric_rows, _parse_summary_columns)
    for parser in parsers:
        metrics = parser(df)
        if metrics:
            return metrics

    return None


def _score_candidate_file(path: Path) -> int:
    """Atribui pontuação de relevância para escolher o melhor arquivo."""
    name = path.name.lower()
    score = 0

    if "5fold" in name or "summary" in name or "resultados" in name:
        score += 3
    if "metrics_summary" in name:
        score += 5
    if "artifacts" in str(path).lower():
        score += 1
    if path.suffix.lower() != ".csv":
        score -= 10

    return score


def discover_model_metrics(models_dir: Path | None = None) -> pd.DataFrame:
    """
    Percorre models/ e consolida métricas de cada subpasta.

    Returns:
        DataFrame com colunas Modelo, Accuracy, Precision, Recall, F1.
    """
    base_dir = models_dir or MODELS_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Pasta de modelos não encontrada: {base_dir}")

    consolidated: list[dict[str, float | str]] = []

    for model_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        candidates = sorted(
            [
                path
                for path in model_dir.rglob("*.csv")
                if not _should_skip_file(path)
            ],
            key=_score_candidate_file,
            reverse=True,
        )

        if not candidates:
            logger.warning("Nenhum arquivo de métricas encontrado em %s", model_dir)
            continue

        metrics: dict[str, float] | None = None
        best_file: Path | None = None
        display_name = _display_name_from_folder(model_dir.name)

        for candidate in candidates:
            parsed = _parse_file(candidate)
            if not parsed:
                continue

            metrics = parsed
            best_file = candidate

            try:
                preview = pd.read_csv(candidate)
                if "modelo" in preview.columns and not preview.empty:
                    modelo_nome = preview.iloc[0]["modelo"]
                    if pd.notna(modelo_nome):
                        display_name = str(modelo_nome)
            except (OSError, pd.errors.ParserError):
                pass
            break

        if metrics is None or best_file is None:
            logger.warning("Métricas não extraídas de %s", model_dir.name)
            continue

        record: dict[str, float | str] = {
            "Modelo": display_name,
            "fonte": str(best_file.relative_to(PROJECT_ROOT)),
        }
        for metric in METRIC_KEYS:
            column_name = "F1" if metric == "f1" else metric.capitalize()
            record[column_name] = metrics.get(metric)

        if all(record.get(col) is not None for col in ("Accuracy", "Precision", "Recall", "F1")):
            consolidated.append(record)
            logger.info(
                "Métricas carregadas para %s a partir de %s",
                record["Modelo"],
                best_file.name,
            )

    if not consolidated:
        raise RuntimeError("Nenhuma métrica foi consolidada a partir de models/.")

    df = pd.DataFrame(consolidated)
    metric_columns = ["Accuracy", "Precision", "Recall", "F1"]
    df = df.sort_values("Recall", ascending=False).reset_index(drop=True)
    return df[["Modelo", *metric_columns, "fonte"]]


def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula score composto e ranking global dos modelos."""
    ranking = df.copy()
    ranking["Score"] = (
        0.45 * ranking["Recall"]
        + 0.30 * ranking["F1"]
        + 0.15 * ranking["Precision"]
        + 0.10 * ranking["Accuracy"]
    )
    ranking["Rank_Global"] = (
        ranking["Score"].rank(ascending=False, method="min").astype(int)
    )
    ranking = ranking.sort_values("Rank_Global").reset_index(drop=True)
    return ranking


def save_consolidated_outputs(
    comparison_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    results_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Salva final_comparison.csv e ranking.csv."""
    output_dir = results_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "final_comparison.csv"
    ranking_path = output_dir / "ranking.csv"

    comparison_df.drop(columns=["fonte"], errors="ignore").to_csv(
        comparison_path,
        index=False,
    )
    ranking_df.to_csv(ranking_path, index=False)

    logger.info("Comparação salva em %s", comparison_path)
    logger.info("Ranking salvo em %s", ranking_path)
    return comparison_path, ranking_path
