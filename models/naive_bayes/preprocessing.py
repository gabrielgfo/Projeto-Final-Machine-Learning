"""Pré-processamento específico para variantes de Naive Bayes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessamento import get_df_preprocessado

logger = logging.getLogger(__name__)

ModelType = Literal["gaussian", "bernoulli", "complement"]

CORRELATION_THRESHOLD = 0.90
VARIANCE_THRESHOLD = 0.01
SMOTE_RANDOM_STATE = 42


class CorrelationRemover(BaseEstimator, TransformerMixin):
    """Remove atributos com correlação absoluta acima do limiar."""

    def __init__(self, threshold: float = CORRELATION_THRESHOLD) -> None:
        self.threshold = threshold
        self.columns_to_keep_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> CorrelationRemover:
        """Identifica colunas a manter com base na matriz de correlação."""
        del y
        frame = self._as_dataframe(X)
        if frame.shape[1] <= 1:
            self.columns_to_keep_ = np.arange(frame.shape[1])
            return self

        corr_matrix = frame.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = {
            coluna
            for coluna in upper.columns
            if any(upper[coluna] > self.threshold)
        }
        self.columns_to_keep_ = np.array(
            [idx for idx, col in enumerate(frame.columns) if col not in to_drop]
        )
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Remove colunas altamente correlacionadas."""
        if self.columns_to_keep_ is None:
            raise ValueError("CorrelationRemover deve ser ajustado antes do transform.")
        frame = self._as_dataframe(X)
        return frame.iloc[:, self.columns_to_keep_].to_numpy(dtype=np.float64)

    @staticmethod
    def _as_dataframe(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def carregar_dados() -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega os dados brutos numéricos para modelagem.

    Returns:
        Tupla (X, y) sem transformações específicas de variante.
    """
    try:
        X, y = get_df_preprocessado()
        logger.info(
            "Dados carregados: %d amostras, %d atributos.",
            X.shape[0],
            X.shape[1],
        )
        return X, y
    except (FileNotFoundError, ValueError, RuntimeError):
        raise
    except Exception as exc:
        logger.exception("Erro inesperado ao carregar os dados.")
        raise RuntimeError("Falha ao carregar os dados.") from exc


def get_preprocessing_pipeline(
    model_type: ModelType,
    binarize_threshold: float = 0.0,
) -> Pipeline:
    """
    Retorna o pipeline de pré-processamento adequado para cada variante.

    O SMOTE não faz parte deste pipeline; é aplicado separadamente no treino.

    Args:
        model_type: Variante de Naive Bayes (gaussian, bernoulli, complement).
        binarize_threshold: Limiar do Binarizer para o pipeline Bernoulli.

    Returns:
        Pipeline sklearn com as transformações da variante.

    Raises:
        ValueError: Se o tipo de modelo for inválido.
    """
    if model_type == "gaussian":
        return Pipeline([
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("correlation", CorrelationRemover(threshold=CORRELATION_THRESHOLD)),
            ("scaler", StandardScaler()),
        ])

    if model_type == "bernoulli":
        return Pipeline([
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("correlation", CorrelationRemover(threshold=CORRELATION_THRESHOLD)),
            ("binarize", Binarizer(threshold=binarize_threshold)),
        ])

    if model_type == "complement":
        return Pipeline([
            ("scaler", MinMaxScaler()),
        ])

    raise ValueError(
        f"model_type inválido: {model_type!r}. "
        "Use 'gaussian', 'bernoulli' ou 'complement'."
    )


def transformar_dados(
    pipeline: Pipeline,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray | None = None,
    y_train: pd.Series | np.ndarray | None = None,
    *,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Ajusta e aplica o pipeline de pré-processamento em treino e teste.

    Args:
        pipeline: Pipeline de pré-processamento da variante.
        X_train: Dados de treino.
        X_test: Dados de teste opcionais.
        y_train: Alvo de treino (usado apenas no fit).
        fit: Se True, ajusta o pipeline no conjunto de treino.

    Returns:
        Tupla com arrays transformados de treino e teste.
    """
    try:
        if fit:
            X_train_proc = pipeline.fit_transform(X_train, y_train)
        else:
            X_train_proc = pipeline.transform(X_train)

        X_test_proc = None
        if X_test is not None:
            X_test_proc = pipeline.transform(X_test)

        return (
            np.asarray(X_train_proc, dtype=np.float64),
            None if X_test_proc is None else np.asarray(X_test_proc, dtype=np.float64),
        )
    except ValueError as exc:
        raise ValueError("Falha ao transformar os dados.") from exc


def aplicar_smote_treino(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica SMOTE apenas nos dados de treino.

    O balanceamento não deve ser aplicado antes da separação dos folds.
    """
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    try:
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    except ValueError as exc:
        raise ValueError("Falha ao aplicar SMOTE nos dados de treino.") from exc

    logger.debug(
        "SMOTE aplicado: %d -> %d amostras de treino.",
        len(y_train),
        len(y_balanced),
    )
    return X_balanced, y_balanced
