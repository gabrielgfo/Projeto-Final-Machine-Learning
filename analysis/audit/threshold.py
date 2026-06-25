"""Utilitários de seleção de threshold com guarda-corpo metodológico."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve


@dataclass
class ThresholdSelection:
    """Resultado da seleção de threshold."""

    threshold: float
    criterion: str
    calibration_f1: float
    positive_rate: float
    rejected_extreme: bool
    fallback_used: bool


def _youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


def encontrar_melhor_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    search_min: float = 0.2,
    search_max: float = 0.8,
    n_thresholds: int = 81,
    prevalence_cap: float = 1.5,
    fallback: float = 0.5,
) -> ThresholdSelection:
    """
    Seleciona threshold por Youden's J com restrições de sanidade.

    Rejeita thresholds cuja taxa de positivos previstos exceda
    ``prevalence_cap`` vezes a prevalência real no conjunto de calibração.
    Se nenhum threshold passar, usa ``fallback`` (0.5).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    prevalence = float(y_true.mean()) if len(y_true) else 0.5

    thresholds = np.linspace(search_min, search_max, n_thresholds)
    best_j = -np.inf
    best_thr = fallback
    best_f1 = 0.0
    best_pos_rate = float((y_proba >= fallback).mean())
    rejected_extreme = False
    fallback_used = True

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        pos_rate = float(y_pred.mean())

        if pos_rate > prevalence_cap * max(prevalence, 1e-6):
            rejected_extreme = True
            continue

        j = _youden_j(y_true, y_pred)
        if j > best_j:
            best_j = j
            best_thr = float(thr)
            best_f1 = float(f1_score(y_true, y_pred, zero_division=0))
            best_pos_rate = pos_rate
            fallback_used = False

    if fallback_used:
        y_pred_fallback = (y_proba >= fallback).astype(int)
        best_f1 = float(f1_score(y_true, y_pred_fallback, zero_division=0))
        best_pos_rate = float(y_pred_fallback.mean())

    return ThresholdSelection(
        threshold=best_thr,
        criterion="youden_j_constrained",
        calibration_f1=best_f1,
        positive_rate=best_pos_rate,
        rejected_extreme=rejected_extreme,
        fallback_used=fallback_used,
    )


def threshold_from_decision_function(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    **kwargs,
) -> ThresholdSelection:
    """Seleciona threshold para modelos com decision_function (ex.: SVM)."""
    scores = np.asarray(decision_scores, dtype=float)
    proba_proxy = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return encontrar_melhor_threshold(y_true, proba_proxy, **kwargs)
