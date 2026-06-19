"""Verificações de sanidade metodológica (protocolo seção 4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class SanityReport:
    """Relatório de verificação de sanidade de um modelo/fold."""

    model_name: str
    fold: int | str
    passed: bool
    checks: dict[str, str] = field(default_factory=dict)
    implied_confusion: dict[str, int] = field(default_factory=dict)
    trivial_baseline: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _implied_confusion(
    y_true: np.ndarray,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
) -> dict[str, int]:
    """Estima TP, FP, FN, TN a partir das métricas e tamanho do fold."""
    n = len(y_true)
    positives = int(y_true.sum())
    negatives = n - positives

    tp = int(round(recall * positives))
    fn = positives - tp
    fp = int(round(tp * (1 / precision - 1))) if precision > 0 else 0
    fp = max(0, min(fp, negatives))
    tn = negatives - fp

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": n, "positives": positives}


def _trivial_always_positive(y_true: np.ndarray) -> dict[str, float]:
    y_pred = np.ones_like(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _trivial_majority(y_true: np.ndarray) -> dict[str, float]:
    majority = int(y_true.mean() >= 0.5)
    y_pred = np.full_like(y_true, majority)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_sanity_checks(
    model_name: str,
    fold: int | str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    roc_auc: float,
    pr_auc: float,
    *,
    threshold_search_bounds: tuple[float, float] = (0.2, 0.8),
) -> SanityReport:
    """Executa as 6 verificações obrigatórias do protocolo de auditoria."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    implied = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "n": len(y_true)}

    report = SanityReport(
        model_name=model_name,
        fold=fold,
        passed=True,
        implied_confusion=implied,
        trivial_baseline=_trivial_always_positive(y_true),
    )

    # 1. Invariante matemático
    prevalence = y_true.mean()
    plausible = (
        0 <= tp <= y_true.sum()
        and 0 <= tn <= (len(y_true) - y_true.sum())
        and abs((tp + tn) / len(y_true) - acc) < 0.02
    )
    report.checks["1_invariante"] = (
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}; prevalência={prevalence:.3f}; "
        f"plausível={'sim' if plausible else 'não'}"
    )
    if not plausible:
        report.passed = False
        report.flags.append("matriz_de_confusao_implausivel")

    # 2. Classificador trivial
    trivial_pos = report.trivial_baseline
    trivial_maj = _trivial_majority(y_true)
    beats_trivial = (
        acc > trivial_pos["accuracy"] + 0.01
        or (rec < 0.98 and f1 > trivial_pos["f1"] + 0.02)
        or (prec > trivial_pos["precision"] + 0.05 and rec < 0.95)
    )
    degenerate_like_trivial = (
        abs(acc - prec) < 0.02 and rec > 0.98
    ) or rec >= 0.995
    report.checks["2_trivial"] = (
        f"Sempre positivo: Acc={trivial_pos['accuracy']:.3f}, Rec={trivial_pos['recall']:.3f}; "
        f"Maioria: Acc={trivial_maj['accuracy']:.3f}; "
        f"Degenerado={'sim' if degenerate_like_trivial else 'não'}"
    )
    if degenerate_like_trivial:
        report.passed = False
        report.flags.append("comportamento_degenerado_tipo_sempre_positivo")

    # 3. Threshold plausível
    at_extreme = (
        threshold <= threshold_search_bounds[0] + 0.001
        or threshold >= threshold_search_bounds[1] - 0.001
    )
    report.checks["3_threshold"] = (
        f"threshold={threshold:.4f}; extremo={'sim' if at_extreme else 'não'}"
    )
    if at_extreme:
        report.passed = False
        report.flags.append("threshold_no_extremo_da_busca")

    # 4. Padrão Accuracy ≈ Precision com Recall ≈ 1
    acc_prec_pattern = abs(acc - prec) < 0.02 and rec > 0.98
    report.checks["4_padrao_degenerado"] = (
        f"Acc≈Prec com Rec alto: {'sim' if acc_prec_pattern else 'não'}"
    )
    if acc_prec_pattern:
        report.passed = False
        report.flags.append("padrao_acc_igual_prec_recall_unidade")

    # 5. Coerência AUC vs métricas threshold
    auc_median = roc_auc < 0.78 and (rec > 0.95 or f1 > 0.85)
    report.checks["5_coerencia_auc"] = (
        f"ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}; "
        f"inflação suspeita={'sim' if auc_median and rec > 0.95 else 'não'}"
    )
    if auc_median and rec > 0.95:
        report.passed = False
        report.flags.append("auc_mediano_com_metricas_threshold_infladas")

    # 6. Sustentação para relatório
    report.checks["6_sustentacao"] = (
        f"Métricas: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}; "
        f"AUC={roc_auc:.3f} {'corrobora' if roc_auc >= 0.75 else 'não corrobora'} excelência"
    )

    return report
