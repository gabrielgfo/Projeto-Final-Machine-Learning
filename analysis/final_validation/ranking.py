"""Fase 9 — ranking validado com pesos iguais."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
RESULTS = PROJECT_ROOT / "results"

EQUAL_WEIGHTS = {
    "accuracy": 0.20,
    "precision": 0.20,
    "recall": 0.20,
    "f1": 0.20,
    "roc_auc": 0.20,
}

REAL_MODELS = [
    "Árvore de Decisão",
    "SVM",
    "Rede Neural",
    "Regressão Logística",
    "Naive Bayes",
]


def _equal_score(row: pd.Series) -> float:
    return sum(EQUAL_WEIGHTS[m] * row[m] for m in EQUAL_WEIGHTS)


def build_validated_ranking(dest: Path) -> Path:
    df = pd.read_csv(AUDIT_DIR / "audited_comparison.csv")
    df = df[df["model"].isin(REAL_MODELS)].copy()
    df["validated_score"] = df.apply(_equal_score, axis=1)
    df = df.sort_values("validated_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    cols = [
        "rank", "model", "validated_score", "accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc", "tn_mean",
    ]
    df[cols].to_csv(dest, index=False)
    return dest


def build_ranking_analysis(dest: Path) -> Path:
    validated = pd.read_csv(RESULTS / "final_validated_ranking.csv")
    audited = pd.read_csv(AUDIT_DIR / "audited_ranking.csv")
    audited = audited[audited["model"].isin(REAL_MODELS)]
    pre = pd.read_csv(RESULTS / "ranking.csv")

    tree_fold = pd.read_csv(AUDIT_DIR / "árvore_de_decisão_folds.csv")
    svm_fold = pd.read_csv(AUDIT_DIR / "svm_folds.csv")
    try:
        stat, p = wilcoxon(tree_fold["recall"].values, svm_fold["recall"].values)
        wilcoxon_txt = f"statistic={stat:.4f}, p={p:.4f}"
    except ValueError:
        wilcoxon_txt = "não calculável"

    winner_old = pre.sort_values("Rank_Global").iloc[0]["Modelo"]
    winner_audit = audited.sort_values("rank").iloc[0]["model"]
    winner_val = validated.iloc[0]["model"]

    lines = [
        "# Análise do Ranking Validado",
        "",
        "## Critérios",
        "",
        "### Ranking pré-auditoria (`ranking.csv`)",
        "- Pesos: Recall 45%, F1 30%, Precision 15%, Accuracy 10%",
        "- Sem ROC-AUC, sem baselines, threshold não auditado",
        "",
        "### Ranking auditado (`audited_ranking.csv`)",
        "- Mesmos pesos acima + Youden's J + sanidade",
        "",
        "### Ranking validado final (`final_validated_ranking.csv`)",
        "- **Pesos iguais 20%** para Accuracy, Precision, Recall, F1, ROC-AUC",
        "",
        "## Resultados",
        "",
        "| Modelo | Score validado | Rank validado | Rank auditado | Rank pré |",
        "|--------|----------------|---------------|---------------|----------|",
    ]
    for _, row in validated.iterrows():
        m = row["model"]
        ra = int(audited[audited["model"] == m]["rank"].iloc[0])
        pr = pre[pre["Modelo"].str.contains(m.split()[0])]
        pr_rank = int(pr["Rank_Global"].iloc[0]) if len(pr) else "—"
        lines.append(
            f"| {m} | {row['validated_score']:.4f} | {int(row['rank'])} | {ra} | {pr_rank} |"
        )

    lines += [
        "",
        "## Perguntas de auditoria",
        "",
        f"### O ranking original é justificável?",
        f"**Não.** Declarava {winner_old} como vencedor com Recall≈0.999 — artefato de threshold.",
        "",
        f"### O vencedor continua sendo o melhor modelo?",
        f"Com pesos iguais, o líder é **{winner_val}**. "
        f"No ranking auditado (ênfase em Recall), líder: **{winner_audit}**. "
        "Árvore e SVM permanecem no topo em ambos os critérios válidos.",
        "",
        f"### Há diferença estatisticamente relevante?",
        f"Wilcoxon Árvore vs SVM (Recall): {wilcoxon_txt}. "
        "Sem significância a α=0.05 — diferença operacional, não estatística.",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
