"""Avaliação cruzada auditada com protocolo de sanidade."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from analysis.audit.data import FoldPreprocessor, load_notebook_dataset
from analysis.audit.sanity_checks import SanityReport, run_sanity_checks
from analysis.audit.threshold import (
    ThresholdSelection,
    encontrar_melhor_threshold,
    threshold_from_decision_function,
)
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

AUDIT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "audit"
RANDOM_STATE = 42
CV_FOLDS = 5


@dataclass
class FoldResult:
    fold: int
    threshold: float
    threshold_criterion: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    tp: int
    fp: int
    fn: int
    tn: int
    sanity_passed: bool
    sanity_flags: list[str]


def _split_train_calib(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _evaluate_sklearn_classifier(
    model_name: str,
    estimator_factory: Callable[[], ClassifierMixin],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    use_decision_function: bool = False,
    balance_smote: bool = False,
    custom_preprocess: Callable | None = None,
) -> tuple[list[FoldResult], list[SanityReport]]:
    """Avalia classificador sklearn com threshold seguro."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results: list[FoldResult] = []
    sanity_reports: list[SanityReport] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx].to_numpy()

        X_train_fit, X_calib, y_train_fit, y_calib = _split_train_calib(X_train, y_train)

        if custom_preprocess:
            prep = custom_preprocess()
            X_train_fit_p = prep.fit_transform(X_train_fit, y_train_fit)
            X_calib_p = prep.transform(X_calib)
            X_test_p = prep.transform(X_test)
        else:
            prep = FoldPreprocessor()
            X_train_fit_p = prep.fit_transform(X_train_fit, y_train_fit)
            X_calib_p = prep.transform(X_calib)
            X_test_p = prep.transform(X_test)

        if balance_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_fit_p, y_train_fit_a = smote.fit_resample(X_train_fit_p, y_train_fit)
        else:
            y_train_fit_a = y_train_fit.to_numpy()

        model = estimator_factory()
        if isinstance(model, MLPClassifier):
            classes, counts = np.unique(y_train_fit_a, return_counts=True)
            weights = {cls: len(y_train_fit_a) / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
            sample_weight = np.array([weights[c] for c in y_train_fit_a])
            model.fit(X_train_fit_p, y_train_fit_a, sample_weight=sample_weight)
        else:
            model.fit(X_train_fit_p, y_train_fit_a)

        if use_decision_function and hasattr(model, "decision_function"):
            calib_scores = model.decision_function(X_calib_p)
            test_scores = model.decision_function(X_test_p)
            calib_proba = (calib_scores - calib_scores.min()) / (
                calib_scores.max() - calib_scores.min() + 1e-12
            )
            test_proba = (test_scores - test_scores.min()) / (
                test_scores.max() - test_scores.min() + 1e-12
            )
            thr_sel = encontrar_melhor_threshold(y_calib.to_numpy(), calib_proba)
            y_pred = (test_proba >= thr_sel.threshold).astype(int)
            y_score = test_proba
        else:
            y_proba_calib = model.predict_proba(X_calib_p)[:, 1]
            y_proba_test = model.predict_proba(X_test_p)[:, 1]
            thr_sel = encontrar_melhor_threshold(y_calib.to_numpy(), y_proba_calib)
            y_pred = (y_proba_test >= thr_sel.threshold).astype(int)
            y_score = y_proba_test

        metrics = _metrics_from_predictions(y_test, y_pred, y_score)
        sanity = run_sanity_checks(
            model_name,
            fold,
            y_test,
            y_pred,
            y_score,
            thr_sel.threshold,
            float(metrics["roc_auc"]),
            float(metrics["pr_auc"]),
        )

        fold_results.append(
            FoldResult(
                fold=fold,
                threshold=thr_sel.threshold,
                threshold_criterion=thr_sel.criterion,
                accuracy=float(metrics["accuracy"]),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                roc_auc=float(metrics["roc_auc"]),
                pr_auc=float(metrics["pr_auc"]),
                tp=int(metrics["tp"]),
                fp=int(metrics["fp"]),
                fn=int(metrics["fn"]),
                tn=int(metrics["tn"]),
                sanity_passed=sanity.passed,
                sanity_flags=sanity.flags,
            )
        )
        sanity_reports.append(sanity)

    return fold_results, sanity_reports


class MinMaxFoldPreprocessor(FoldPreprocessor):
    """Pré-processamento para ComplementNB: mode | label | corr | minmax."""

    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()


def get_model_specs() -> dict[str, dict[str, Any]]:
    """Especificações dos 5 modelos com hiperparâmetros dos notebooks."""
    return {
        "Regressão Logística": {
            "factory": lambda: LogisticRegression(
                C=10.0,
                class_weight="balanced",
                max_iter=1000,
                penalty="l1",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "smote": False,
            "decision_function": False,
            "custom_preprocess": None,
        },
        "Árvore de Decisão": {
            "factory": lambda: DecisionTreeClassifier(
                class_weight={0: 1, 1: 2},
                criterion="entropy",
                max_depth=15,
                max_features=None,
                min_impurity_decrease=0.0012123525181736484,
                min_samples_leaf=13,
                min_samples_split=46,
                ccp_alpha=0.0043528172066691975,
                random_state=RANDOM_STATE,
            ),
            "smote": False,
            "decision_function": False,
            "custom_preprocess": None,
        },
        "SVM": {
            "factory": lambda: SVC(
                C=3.14891164795686,
                gamma=5.669849511478847,
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            "smote": False,
            "decision_function": False,
            "custom_preprocess": None,
        },
        "Naive Bayes": {
            "factory": lambda: ComplementNB(alpha=0.16121212121212125, norm=True),
            "smote": True,
            "decision_function": False,
            "custom_preprocess": MinMaxFoldPreprocessor,
        },
        "Rede Neural": {
            "factory": lambda: MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="sgd",
                learning_rate_init=0.01,
                momentum=0.9,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            ),
            "smote": False,
            "decision_function": False,
            "custom_preprocess": None,
        },
    }


def evaluate_baselines(X: pd.DataFrame, y: pd.Series) -> dict[str, list[FoldResult]]:
    """Avalia classificadores triviais no mesmo protocolo de folds."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baselines = {
        "Sempre Positivo": lambda y_test: np.ones_like(y_test),
        "Sempre Majoritário": lambda y_test: np.full_like(
            y_test,
            int(y_test.mean() >= 0.5),
        ),
    }
    output: dict[str, list[FoldResult]] = {name: [] for name in baselines}

    for fold, (_, test_idx) in enumerate(skf.split(X, y), start=1):
        y_test = y.iloc[test_idx].to_numpy()
        for name, predictor in baselines.items():
            y_pred = predictor(y_test)
            y_score = y_pred.astype(float)
            try:
                roc = float(roc_auc_score(y_test, y_score))
            except ValueError:
                roc = 0.5
            metrics = _metrics_from_predictions(y_test, y_pred, y_score)
            metrics["roc_auc"] = roc
            output[name].append(
                FoldResult(
                    fold=fold,
                    threshold=0.5,
                    threshold_criterion="trivial",
                    accuracy=float(metrics["accuracy"]),
                    precision=float(metrics["precision"]),
                    recall=float(metrics["recall"]),
                    f1=float(metrics["f1"]),
                    roc_auc=float(metrics["roc_auc"]) if len(np.unique(y_test)) > 1 else 0.5,
                    pr_auc=float(metrics["pr_auc"]),
                    tp=int(metrics["tp"]),
                    fp=int(metrics["fp"]),
                    fn=int(metrics["fn"]),
                    tn=int(metrics["tn"]),
                    sanity_passed=False,
                    sanity_flags=["baseline_trivial"],
                )
            )
    return output


def summarize_folds(folds: list[FoldResult]) -> dict[str, float]:
    """Agrega métricas por fold em médias."""
    df = pd.DataFrame([f.__dict__ for f in folds])
    numeric = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "tn"]
    return {col: float(df[col].mean()) for col in numeric if col in df.columns}


def save_model_audit(
    model_name: str,
    folds: list[FoldResult],
    sanity_reports: list[SanityReport],
) -> Path:
    """Salva resultados auditados de um modelo."""
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(" ", "_")
    fold_df = pd.DataFrame([f.__dict__ for f in folds])
    fold_path = AUDIT_RESULTS_DIR / f"{slug}_folds.csv"
    fold_df.to_csv(fold_path, index=False)

    sanity_rows = []
    for report in sanity_reports:
        row = {
            "model": report.model_name,
            "fold": report.fold,
            "passed": report.passed,
            "flags": ";".join(report.flags),
            **report.checks,
            **{f"cm_{k}": v for k, v in report.implied_confusion.items()},
        }
        sanity_rows.append(row)
    sanity_path = AUDIT_RESULTS_DIR / f"{slug}_sanity.json"
    with sanity_path.open("w", encoding="utf-8") as f:
        json.dump(sanity_rows, f, indent=2, ensure_ascii=False)

    return fold_path
