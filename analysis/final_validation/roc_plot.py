"""Plotagem aprimorada da curva ROC comparativa (sem alterar métricas oficiais)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from analysis.final_validation.consistency import PROJECT_ROOT, REAL_MODELS
from analysis.final_validation.plots import _export_predictions_for_curves, _slug

logger = logging.getLogger(__name__)

RESULTS = PROJECT_ROOT / "results"
VALIDATION_DIR = RESULTS / "validation"
ROC_NPZ = VALIDATION_DIR / "roc_curves.npz"
ROC_PNG = VALIDATION_DIR / "roc_comparison.png"

# Ordem visual: melhor AUC primeiro (valores de validated_metrics.csv)
MODEL_ORDER = [
    "Árvore de Decisão",
    "Rede Neural",
    "SVM",
    "Regressão Logística",
    "Naive Bayes",
]

MODEL_STYLES: dict[str, dict[str, str | float]] = {
    "Árvore de Decisão": {"color": "#1B9E77", "lw": 2.8},
    "Rede Neural": {"color": "#D95F02", "lw": 2.8},
    "SVM": {"color": "#7570B3", "lw": 2.6},
    "Regressão Logística": {"color": "#E7298A", "lw": 2.4},
    "Naive Bayes": {"color": "#66A61E", "lw": 2.2},
}

MODEL_SHORT = {
    "Árvore de Decisão": "Árvore de Decisão",
    "Rede Neural": "Rede Neural",
    "SVM": "SVM",
    "Regressão Logística": "Regressão Logística",
    "Naive Bayes": "Naive Bayes",
}


def _load_official_auc() -> dict[str, float]:
    val = pd.read_csv(RESULTS / "validated_metrics.csv")
    return dict(zip(val["modelo"], val["roc_auc"], strict=True))


def _curves_from_predictions(
    preds: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in REAL_MODELS:
        y_true, y_score = preds[model]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves[model] = (fpr, tpr)
    return curves


def save_roc_curves_npz(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    dest: Path = ROC_NPZ,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for model, (fpr, tpr) in curves.items():
        key = _slug(model)
        payload[f"{key}_fpr"] = fpr
        payload[f"{key}_tpr"] = tpr
    np.savez_compressed(dest, **payload)
    return dest


def load_roc_curves_npz(src: Path = ROC_NPZ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not src.exists():
        raise FileNotFoundError(f"Dados de curva ROC não encontrados: {src}")
    data = np.load(src)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in REAL_MODELS:
        key = _slug(model)
        curves[model] = (data[f"{key}_fpr"], data[f"{key}_tpr"])
    return curves


def plot_roc_comparison(
    dest: Path = ROC_PNG,
    curves_npz: Path = ROC_NPZ,
    *,
    export_if_missing: bool = True,
) -> Path:
    """
    Gera PNG comparativo das curvas ROC.

    Usa pontos (FPR, TPR) em cache (``roc_curves.npz``). Os valores de AUC na
    legenda vêm de ``validated_metrics.csv`` (métricas oficiais, sem recálculo).
    """
    if curves_npz.exists():
        curves = load_roc_curves_npz(curves_npz)
        logger.info("Curvas ROC carregadas de %s", curves_npz)
    elif export_if_missing:
        logger.info("Cache ROC ausente — exportando predições uma vez para %s", curves_npz)
        preds = _export_predictions_for_curves()
        curves = _curves_from_predictions(preds)
        save_roc_curves_npz(curves, curves_npz)
    else:
        raise FileNotFoundError(
            f"Arquivo {curves_npz} não existe. Execute com export_if_missing=True."
        )

    official_auc = _load_official_auc()

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    for model in MODEL_ORDER:
        fpr, tpr = curves[model]
        style = MODEL_STYLES[model]
        auc_val = official_auc[model]
        label = f"{MODEL_SHORT[model]} (AUC = {auc_val:.3f})"
        ax.plot(
            fpr,
            tpr,
            color=style["color"],
            linewidth=style["lw"],
            label=label,
            alpha=0.95,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2, label="Classificador aleatório")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Taxa de Falsos Positivos (1 − Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)")
    ax.set_title(
        "Curvas ROC — Comparação dos Modelos\n"
        "(predições agregadas em validação cruzada 5-fold, protocolo auditado)",
        fontweight="bold",
        pad=12,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC comparativa salva em %s", dest)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    plot_roc_comparison()


if __name__ == "__main__":
    main()
