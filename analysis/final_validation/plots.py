"""Fases 6–8: curvas ROC/PR e matrizes de confusão."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from analysis.final_validation.consistency import FOLD_FILES, PROJECT_ROOT, REAL_MODELS

logger = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("ã", "a")
        .replace("õ", "o")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def _avg_cm_from_folds(model: str) -> np.ndarray:
    df = pd.read_csv(FOLD_FILES[model])
    tp, fp, fn, tn = df[["tp", "fp", "fn", "tn"]].mean()
    return np.array([[tn, fp], [fn, tp]])


def _plot_confusion_matrix(cm: np.ndarray, title: str, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Real 0", "Real 1"],
        ax=ax,
        cbar_kws={"label": "Contagem média"},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _export_predictions_for_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Exporta y_true e y_score agregados via protocolo auditado (sem grid search).
    Reutiliza evaluator existente — única reexecução permitida para curvas.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from analysis.audit.data import load_notebook_dataset
    from analysis.audit.evaluator import RANDOM_STATE, get_model_specs, FoldPreprocessor, MinMaxFoldPreprocessor
    from analysis.audit.threshold import encontrar_melhor_threshold
    from imblearn.over_sampling import SMOTE

    X, y = load_notebook_dataset()
    specs = get_model_specs()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out: dict[str, tuple[list, list]] = {m: ([], []) for m in REAL_MODELS}

    for model_name in REAL_MODELS:
        spec = specs[model_name]
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE,
            )
            prep_cls = spec["custom_preprocess"] or FoldPreprocessor
            prep = prep_cls() if spec["custom_preprocess"] else FoldPreprocessor()
            X_tr_p = prep.fit_transform(X_tr, y_tr)
            X_cal_p = prep.transform(X_cal)
            X_te_p = prep.transform(X_test)
            if spec["smote"]:
                X_tr_p, y_tr_a = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_p, y_tr)
            else:
                y_tr_a = y_tr.to_numpy()
            model = spec["factory"]()
            if hasattr(model, "fit"):
                from sklearn.neural_network import MLPClassifier
                if isinstance(model, MLPClassifier):
                    classes, counts = np.unique(y_tr_a, return_counts=True)
                    w = {c: len(y_tr_a) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
                    model.fit(X_tr_p, y_tr_a, sample_weight=np.array([w[c] for c in y_tr_a]))
                else:
                    model.fit(X_tr_p, y_tr_a)
            y_proba_cal = model.predict_proba(X_cal_p)[:, 1]
            y_proba = model.predict_proba(X_te_p)[:, 1]
            thr = encontrar_melhor_threshold(y_cal.to_numpy(), y_proba_cal).threshold
            y_pred = (y_proba >= thr).astype(int)
            out[model_name][0].extend(y_test.to_numpy().tolist())
            out[model_name][1].extend(y_proba.tolist())

    return {m: (np.array(v[0]), np.array(v[1])) for m, v in out.items()}


def _write_cm_summary(dest: Path, cms: dict[str, np.ndarray]) -> None:
    lines = [
        "# Resumo das Matrizes de Confusão (média 5 folds)",
        "",
        "| Modelo | TN | FP | FN | TP | Interpretação |",
        "|--------|----|----|----|-----|---------------|",
    ]
    for model, cm in cms.items():
        tn, fp, fn, tp = cm.ravel()
        interp = []
        if fn > tp:
            interp.append("Alto FN — perde positivos (evasão)")
        if fp > 500:
            interp.append("FP elevado — falsos alarmes")
        if tn > 2400 and fn < 800:
            interp.append("Equilíbrio TP/TN favorável")
        lines.append(
            f"| {model} | {tn:.0f} | {fp:.0f} | {fn:.0f} | {tp:.0f} | {', '.join(interp) or '—'} |"
        )
    lines += [
        "",
        "## Implicações práticas",
        "",
        "- **FP:** Cursos sinalizados como alto risco sem serem — custo de intervenção desnecessária.",
        "- **FN:** Cursos em risco não detectados — falha crítica para política de retenção.",
        "- Modelos com Recall alto exigem FN baixo; TN alto isolado não basta (caso Naive Bayes).",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")


def generate_all_validation_plots(validation_dir: Path) -> list[Path]:
    """Gera matrizes, ROC e PR (fases 6–8)."""
    validation_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cms: dict[str, np.ndarray] = {}

    for model in REAL_MODELS:
        slug = _slug(model)
        cm = _avg_cm_from_folds(model)
        cms[model] = cm
        p = validation_dir / f"confusion_matrix_{slug}.png"
        _plot_confusion_matrix(cm, f"Matriz de Confusão — {model}", p)
        paths.append(p)

    summary_path = validation_dir / "confusion_matrix_summary.md"
    _write_cm_summary(summary_path, cms)
    paths.append(summary_path)

    logger.info("Exportando predições para curvas ROC/PR (protocolo auditado, sem grid search)...")
    try:
        preds = _export_predictions_for_curves()
    except Exception as exc:
        logger.warning("Falha ao exportar predições: %s — curvas ROC/PR omitidas", exc)
        return paths

    from analysis.final_validation.roc_plot import (
        _curves_from_predictions,
        plot_roc_comparison,
        save_roc_curves_npz,
    )

    curves = _curves_from_predictions(preds)
    npz_path = validation_dir / "roc_curves.npz"
    save_roc_curves_npz(curves, npz_path)
    roc_cmp = plot_roc_comparison(
        validation_dir / "roc_comparison.png",
        npz_path,
        export_if_missing=False,
    )
    paths.append(roc_cmp)

    for model in REAL_MODELS:
        y_true, y_score = preds[model]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#3498DB", lw=2)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.title(f"ROC — {model}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.tight_layout()
        p = validation_dir / f"roc_{_slug(model)}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)

    plt.figure(figsize=(8, 6))
    for model in REAL_MODELS:
        y_true, y_score = preds[model]
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.plot(rec, prec, label=model)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall — Comparação")
    plt.legend(fontsize=8)
    plt.tight_layout()
    pr_cmp = validation_dir / "pr_comparison.png"
    plt.savefig(pr_cmp, dpi=150)
    plt.close()
    paths.append(pr_cmp)

    for model in REAL_MODELS:
        y_true, y_score = preds[model]
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, color="#27AE60", lw=2)
        plt.title(f"Precision-Recall — {model}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        p = validation_dir / f"pr_curve_{_slug(model)}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        paths.append(p)

    return paths
