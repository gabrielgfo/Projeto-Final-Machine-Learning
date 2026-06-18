"""Avaliação cruzada estratificada do modelo Naive Bayes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from model_factory import create_model
from preprocessing import (
    aplicar_smote_treino,
    get_preprocessing_pipeline,
    transformar_dados,
)

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

CV_FOLDS = 5
CV_RANDOM_STATE = 42


def _carregar_config() -> dict[str, Any]:
    """Carrega a melhor configuração salva pela etapa de treino."""
    params_path = ARTIFACTS_DIR / "best_params.json"
    try:
        with params_path.open("r", encoding="utf-8") as arquivo:
            config = json.load(arquivo)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Não foi possível ler {params_path}.") from exc

    if "model_type" not in config:
        raise ValueError("best_params.json não contém 'model_type'.")

    return config


def _calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas de classificação para um fold."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _salvar_metricas_por_fold(metricas_folds: list[dict[str, float]]) -> None:
    """Persiste métricas individuais de cada fold."""
    df = pd.DataFrame(metricas_folds)
    df.insert(0, "fold", range(1, len(metricas_folds) + 1))
    caminho = ARTIFACTS_DIR / "metrics_by_fold.csv"
    try:
        df.to_csv(caminho, index=False)
        logger.info("Métricas por fold salvas em %s.", caminho)
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar {caminho}.") from exc


def _salvar_resumo_metricas(metricas_folds: list[dict[str, float]]) -> None:
    """Calcula média e desvio padrão das métricas entre folds."""
    df = pd.DataFrame(metricas_folds)
    resumo = pd.DataFrame(
        {
            "metric": df.columns,
            "mean": df.mean().values,
            "std": df.std(ddof=1).values,
        }
    )
    caminho = ARTIFACTS_DIR / "metrics_summary.csv"
    try:
        resumo.to_csv(caminho, index=False)
        logger.info("Resumo de métricas salvo em %s.", caminho)
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar {caminho}.") from exc


def _salvar_matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Gera e salva a matriz de confusão agregada."""
    matriz = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(
        matriz,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    caminho = ARTIFACTS_DIR / "confusion_matrix.csv"
    try:
        df.to_csv(caminho)
        logger.info("Matriz de confusão salva em %s.", caminho)
    except OSError as exc:
        raise RuntimeError(f"Falha ao salvar {caminho}.") from exc


def executar_avaliacao(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Executa avaliação com validação cruzada estratificada.

    Para cada fold, aplica o pré-processamento da variante vencedora,
    SMOTE apenas no treino, e calcula métricas no conjunto de teste.

    Args:
        X: Features brutas.
        y: Vetor alvo.
        config: Configuração do modelo. Se None, lê de best_params.json.

    Returns:
        DataFrame com o resumo das métricas (média e desvio padrão).
    """
    if config is None:
        config = _carregar_config()

    model_type = config["model_type"]
    binarize_threshold = float(config.get("binarize", 0.0))

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    metricas_folds: list[dict[str, float]] = []
    y_true_agregado: list[int] = []
    y_pred_agregado: list[int] = []

    try:
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx].to_numpy()
            y_test = y.iloc[test_idx].to_numpy()

            pipeline = get_preprocessing_pipeline(
                model_type,
                binarize_threshold=binarize_threshold,
            )
            X_train_proc, X_test_proc = transformar_dados(
                pipeline,
                X_train,
                X_test,
                y.iloc[train_idx],
                fit=True,
            )
            assert X_test_proc is not None

            X_train_bal, y_train_bal = aplicar_smote_treino(X_train_proc, y_train)

            modelo = create_model(config)
            modelo.fit(X_train_bal, y_train_bal)

            y_pred = modelo.predict(X_test_proc)
            y_proba = modelo.predict_proba(X_test_proc)[:, 1]

            metricas = _calcular_metricas(y_test, y_pred, y_proba)
            metricas_folds.append(metricas)

            y_true_agregado.extend(y_test.tolist())
            y_pred_agregado.extend(y_pred.tolist())

            logger.info(
                "Fold %d - recall: %.4f, f1: %.4f, roc_auc: %.4f",
                fold,
                metricas["recall"],
                metricas["f1"],
                metricas["roc_auc"],
            )
    except ValueError as exc:
        raise RuntimeError("Falha durante a avaliação cruzada.") from exc

    _salvar_metricas_por_fold(metricas_folds)
    _salvar_resumo_metricas(metricas_folds)
    _salvar_matriz_confusao(
        np.array(y_true_agregado),
        np.array(y_pred_agregado),
    )

    df_resumo = pd.DataFrame(metricas_folds).agg(["mean", "std"]).T.reset_index()
    df_resumo.columns = ["metric", "mean", "std"]
    return df_resumo
