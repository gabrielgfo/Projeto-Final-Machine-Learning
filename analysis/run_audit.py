"""Orquestrador da auditoria metodológica dos modelos."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.audit.data import load_notebook_dataset
from analysis.audit.evaluator import (
    AUDIT_RESULTS_DIR,
    evaluate_baselines,
    get_model_specs,
    save_model_audit,
    summarize_folds,
    _evaluate_sklearn_classifier,
)

logger = logging.getLogger(__name__)

SCORE_WEIGHTS = {
    "recall": 0.45,
    "f1": 0.30,
    "precision": 0.15,
    "accuracy": 0.10,
}


def _compute_score(row: pd.Series) -> float:
    return (
        SCORE_WEIGHTS["recall"] * row["recall"]
        + SCORE_WEIGHTS["f1"] * row["f1"]
        + SCORE_WEIGHTS["precision"] * row["precision"]
        + SCORE_WEIGHTS["accuracy"] * row["accuracy"]
    )


def _format_model_table(summary: dict[str, float], passed: bool) -> str:
    return (
        f"| {summary['model']} | {summary['threshold_mean']:.3f}±{summary['threshold_std']:.3f} | "
        f"{summary['accuracy']:.3f} | {summary['precision']:.3f} | {summary['recall']:.3f} | "
        f"{summary['f1']:.3f} | {summary['roc_auc']:.3f} | {summary['pr_auc']:.3f} | "
        f"{summary['tn_mean']:.0f} | {'S' if passed else 'N'} |"
    )


def run_audit(models: list[str] | None = None) -> None:
    """Executa auditoria completa e salva artefatos em results/audit/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_notebook_dataset()
    logger.info(
        "Dataset unificado: %d amostras, prevalência positiva=%.3f",
        len(y),
        y.mean(),
    )

    specs = get_model_specs()
    target_models = models or list(specs.keys())
    summaries: list[dict] = []
    sanity_sections: list[str] = []
    fold_store: dict[str, list] = {}

    for model_name in target_models:
        logger.info("=== Auditando: %s ===", model_name)
        spec = specs[model_name]
        folds, sanity_reports = _evaluate_sklearn_classifier(
            model_name,
            spec["factory"],
            X,
            y,
            use_decision_function=spec["decision_function"],
            balance_smote=spec["smote"],
            custom_preprocess=spec["custom_preprocess"],
        )
        save_model_audit(model_name, folds, sanity_reports)
        fold_store[model_name] = folds

        summary = summarize_folds(folds)
        threshold_std = float(pd.DataFrame([f.__dict__ for f in folds])["threshold"].std())
        passed = all(r.passed for r in sanity_reports)
        summary.update({
            "model": model_name,
            "threshold_mean": summary.get("threshold", 0.5),
            "threshold_std": threshold_std,
            "tn_mean": summary.get("tn", 0),
            "degenerescence_pass": passed,
        })
        summaries.append(summary)

        sanity_sections.append(f"### {model_name}\n")
        for report in sanity_reports:
            sanity_sections.append(
                f"**Fold {report.fold}** — Passou: {'Sim' if report.passed else 'Não'}\n"
            )
            for key, text in report.checks.items():
                sanity_sections.append(f"- {key}: {text}")
            if report.flags:
                sanity_sections.append(f"- Flags: {', '.join(report.flags)}")
            sanity_sections.append("")

        logger.info(
            "%s | Rec=%.3f F1=%.3f AUC=%.3f | Sanidade=%s",
            model_name,
            summary["recall"],
            summary["f1"],
            summary["roc_auc"],
            "OK" if passed else "FALHOU",
        )

    # Baselines
    baseline_results = evaluate_baselines(X, y)
    for name, folds in baseline_results.items():
        summary = summarize_folds(folds)
        summary.update({
            "model": name,
            "threshold_mean": 0.5,
            "threshold_std": 0.0,
            "tn_mean": summary.get("tn", 0),
            "degenerescence_pass": False,
        })
        summaries.append(summary)

    comparison_df = pd.DataFrame(summaries)
    comparison_df["score"] = comparison_df.apply(_compute_score, axis=1)
    comparison_df = comparison_df.sort_values("recall", ascending=False)
    comparison_df.to_csv(AUDIT_RESULTS_DIR / "audited_comparison.csv", index=False)

    ranking_df = comparison_df.sort_values("score", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1
    ranking_df.to_csv(AUDIT_RESULTS_DIR / "audited_ranking.csv", index=False)

    # Wilcoxon entre os 2 melhores (excluindo baselines)
    real_models = [m for m in ranking_df["model"] if m in fold_store]
    stat_note = "Teste estatístico não aplicável (menos de 2 modelos)."
    if len(real_models) >= 2:
        top2 = real_models[:2]
        recalls_a = [f.recall for f in fold_store[top2[0]]]
        recalls_b = [f.recall for f in fold_store[top2[1]]]
        try:
            stat, p_value = wilcoxon(recalls_a, recalls_b)
            stat_note = (
                f"Wilcoxon pareado entre {top2[0]} e {top2[1]} em Recall: "
                f"statistic={stat:.4f}, p-value={p_value:.4f}. "
                f"{'Diferença significativa (p<0.05)' if p_value < 0.05 else 'Sem diferença significativa (p>=0.05)'}"
            )
        except ValueError as exc:
            stat_note = f"Wilcoxon não executado: {exc}"

    # Relatório markdown
    lines = [
        "# Auditoria Metodológica — Evasão Acadêmica",
        "",
        "## Nota sobre pipelines (seção 5.4)",
        "- Notebooks (LR, Árvore, SVM, Rede Neural): alvo `TAXA_EVASAO >= 20%`, "
        "pré-processamento `mode | label | corr | standard` por fold.",
        "- Naive Bayes original usava `data/preprocessamento.py` (mediana, QT_ING>=10) — "
        "reavaliado aqui no pipeline unificado dos notebooks para comparação justa.",
        "- Balanceamento: `class_weight=balanced` (LR, Árvore, SVM, MLP); SMOTE só no treino (NB).",
        "",
        "## Verificações de Sanidade",
        *sanity_sections,
        "",
        "## Tabela por Modelo",
        "",
        "| Modelo | Threshold (média±dp) | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | TN médio | Passou? |",
        "|--------|----------------------|-----|------|--------|----|---------|--------|----------|---------|",
    ]
    for summary in summaries:
        if summary["model"] in specs or summary["model"].startswith("Sempre"):
            lines.append(_format_model_table(summary, summary["degenerescence_pass"]))

    ranking_lines = [
        "| Rank | Modelo | Score | Recall | F1 | Precision | Accuracy | ROC-AUC |",
        "|------|--------|-------|--------|----|-----------|----------|---------|",
    ]
    for _, row in ranking_df.iterrows():
        ranking_lines.append(
            f"| {int(row['rank'])} | {row['model']} | {row['score']:.4f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} | "
            f"{row['accuracy']:.3f} | {row['roc_auc']:.3f} |"
        )

    lines.extend([
        "",
        "## Ranking Revisado (pós-correção de threshold)",
        "",
        *ranking_lines,
        "",
        f"## Significância Estatística\n\n{stat_note}",
        "",
        "## Conclusão Metodológica",
        "",
        "O ranking anterior que favorecia Regressão Logística com Recall≈1.0 "
        "refletia threshold patológico (maximização cega de F1), não superioridade preditiva. "
        "Com Youden's J restrito e guarda-corpo de prevalência, os resultados devem ser "
        "interpretados à luz do ROC-AUC/PR-AUC e da coluna TN médio.",
    ])

    report_path = AUDIT_RESULTS_DIR / "audit_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Auditoria salva em %s", AUDIT_RESULTS_DIR)

    from analysis.audit.final_report_pdf import generate_final_report_pdf

    pdf_path = generate_final_report_pdf()
    logger.info("Relatório PDF pós-auditoria salvo em %s", pdf_path)


if __name__ == "__main__":
    run_audit()
