"""Fases 1–3: inventário, extração e consistência."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"

REAL_MODELS = [
    "Árvore de Decisão",
    "SVM",
    "Rede Neural",
    "Regressão Logística",
    "Naive Bayes",
]

FOLD_FILES = {
    "Árvore de Decisão": AUDIT_DIR / "árvore_de_decisão_folds.csv",
    "SVM": AUDIT_DIR / "svm_folds.csv",
    "Rede Neural": AUDIT_DIR / "rede_neural_folds.csv",
    "Regressão Logística": AUDIT_DIR / "regressão_logística_folds.csv",
    "Naive Bayes": AUDIT_DIR / "naive_bayes_folds.csv",
}


def _glob_list(base: Path, pattern: str) -> list[str]:
    return sorted(str(p.relative_to(PROJECT_ROOT)) for p in base.rglob(pattern) if p.is_file())


def build_project_inventory(dest: Path, today: str) -> Path:
    """Fase 1 — inventário completo."""
    dataset_candidates = [
        PROJECT_ROOT / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
        PROJECT_ROOT / "data" / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
        Path.home()
        / "Downloads"
        / "microdados_censo_da_educacao_superior_2024"
        / "dados"
        / "MICRODADOS_CADASTRO_CURSOS_2024.CSV",
    ]
    dataset_status = {str(p): p.exists() for p in dataset_candidates}

    lines = [
        "# Inventário do Projeto — Auditoria Científica Final",
        "",
        f"**Data:** {today}",
        "",
        "## Estrutura de diretórios",
        "",
        "| Diretório | Função |",
        "|-----------|--------|",
        "| `models/` | Notebooks e artefatos por modelo |",
        "| `analysis/` | Pipelines comparativo e auditoria |",
        "| `analysis/audit/` | Protocolo unificado 5-fold pós-correção |",
        "| `analysis/final_validation/` | Validação científica final |",
        "| `results/` | Rankings e relatórios pré-auditoria |",
        "| `results/audit/` | Resultados auditados (oficiais) |",
        "| `results/validation/` | Curvas ROC/PR e matrizes validadas |",
        "| `slides/` | Apresentações HTML |",
        "",
        "## Modelos avaliados",
        "",
        "1. Regressão Logística (`models/regrassao_logisticca/`)",
        "2. Árvore de Decisão (`models/arvore_de_decisao/`)",
        "3. SVM (avaliado via `analysis/audit/evaluator.py`; notebook em `main`)",
        "4. Rede Neural MLP (`models/Redes_Neurais/`)",
        "5. Naive Bayes ComplementNB (`models/naive_bayes/` em branch `main`)",
        "",
        "## Dataset",
        "",
        f"- Censo INEP 2024 — `MICRODADOS_CADASTRO_CURSOS_2024.CSV`",
        f"- Status dos caminhos: `{json.dumps(dataset_status, ensure_ascii=False)}`",
        "- Alvo auditado: `ALTO_RISCO_EVASAO` (TAXA_EVASAO ≥ 20%)",
        "- Amostra: 30.000 estratificada 50/50",
        "",
        "## Artefatos serializados (*.pkl)",
        "",
    ]
    pkls = _glob_list(PROJECT_ROOT, "*.pkl")
    lines.append("Nenhum no working tree atual." if not pkls else "\n".join(f"- `{p}`" for p in pkls))
    lines += [
        "",
        "> `best_model.pkl` (Naive Bayes) disponível em `main` e `cursor/naive-bayes-multi-variant-pipeline`.",
        "",
        "## PDFs de relatório",
        "",
    ]
    for p in _glob_list(PROJECT_ROOT, "*.pdf"):
        lines.append(f"- `{p}`")

    lines += ["", "## CSVs/JSON em results/", ""]
    for p in sorted((RESULTS).rglob("*.csv")) + sorted((RESULTS).rglob("*.json")):
        lines.append(f"- `{p.relative_to(PROJECT_ROOT)}`")

    lines += ["", "## Notebooks", ""]
    for p in _glob_list(MODELS, "*.ipynb"):
        lines.append(f"- `{p}`")

    lines += [
        "",
        "## Protocolo de validação oficial",
        "",
        "Métricas **oficiais** para entrega acadêmica: `results/audit/audited_comparison.csv`",
        "(Youden's J, 6 checagens de sanidade, baselines triviais).",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def build_extracted_metrics(dest: Path) -> Path:
    """Fase 2 — consolida métricas de todas as fontes."""
    rows: list[dict] = []

    if (RESULTS / "final_comparison.csv").exists():
        df = pd.read_csv(RESULTS / "final_comparison.csv")
        for _, r in df.iterrows():
            rows.append({
                "modelo": r["Modelo"],
                "fonte": "final_comparison.csv (pré-auditoria)",
                "accuracy": r.get("Accuracy"),
                "precision": r.get("Precision"),
                "recall": r.get("Recall"),
                "f1": r.get("F1"),
                "roc_auc": None,
                "hiperparametros": "notebooks originais",
                "validacao": "5-fold CV notebooks",
            })

    if (AUDIT_DIR / "audited_comparison.csv").exists():
        df = pd.read_csv(AUDIT_DIR / "audited_comparison.csv")
        for _, r in df.iterrows():
            rows.append({
                "modelo": r["model"],
                "fonte": "audited_comparison.csv (pós-auditoria)",
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
                "tn_mean": r["tn_mean"],
                "threshold_mean": r["threshold_mean"],
                "hiperparametros": "evaluator.py get_model_specs()",
                "validacao": "5-fold CV auditado + Youden J",
            })

    rn_csv = MODELS / "Redes_Neurais" / "resultados_rede_neural_5fold.csv"
    if rn_csv.exists():
        r = pd.read_csv(rn_csv).iloc[0]
        rows.append({
            "modelo": "Rede Neural",
            "fonte": "resultados_rede_neural_5fold.csv",
            "accuracy": r["accuracy_mean"],
            "precision": r["precision_mean"],
            "recall": r["recall_mean"],
            "f1": r["f1_mean"],
            "roc_auc": None,
            "hiperparametros": "MLP 64, ReLU, SGD",
            "validacao": "5-fold notebook",
        })

    out = pd.DataFrame(rows)
    out.to_csv(dest, index=False)
    return dest


def build_inconsistencies_report(dest: Path) -> Path:
    """Fase 3 — divergências entre fontes."""
    issues: list[str] = [
        "# Relatório de Inconsistências Metodológicas",
        "",
        "## Resposta à pergunta de auditoria",
        "",
        "**As métricas apresentadas nos PDFs pré-auditoria NÃO são consistentes** "
        "com os arquivos auditados em `results/audit/`. Os PDFs refletem threshold "
        "patológico; os CSVs auditados são a fonte oficial.",
        "",
        "## Divergências documentadas",
        "",
    ]

    pre = pd.read_csv(RESULTS / "ranking.csv")
    post = pd.read_csv(AUDIT_DIR / "audited_ranking.csv")
    post_real = post[post["model"].isin(REAL_MODELS)]

    lr_pre = pre[pre["Modelo"] == "Regressão Logística"].iloc[0]
    lr_post = post_real[post_real["model"] == "Regressão Logística"].iloc[0]
    issues += [
        "### 1. Regressão Logística — Recall inflado",
        "",
        "| Fonte | Recall | Score | Rank |",
        "|-------|--------|-------|------|",
        f"| `ranking.csv` (pré) | {lr_pre['Recall']:.4f} | {lr_pre['Score']:.4f} | 1º |",
        f"| `audited_ranking.csv` (pós) | {lr_post['recall']:.4f} | {lr_post['score']:.4f} | 6º |",
        "",
        "- **Impacto:** Conclusão de 'melhor modelo' inválida no relatório pré-auditoria.",
        "- **Causa:** Maximização cega de F1 com threshold ~0.01 (TN≈0).",
        "- **Valor oficial:** Pós-auditoria.",
        "",
    ]

    nb_pre = pre[pre["Modelo"] == "Naive Bayes"].iloc[0]
    nb_post = post_real[post_real["model"] == "Naive Bayes"].iloc[0]
    issues += [
        "### 2. Naive Bayes — pipelines distintos",
        "",
        f"| Fonte | Recall | ROC-AUC |",
        f"|-------|--------|---------|",
        f"| Pré-auditoria / pipeline original | {nb_pre['Recall']:.4f} | ~0.727 (branch main) |",
        f"| Pipeline unificado auditado | {nb_post['recall']:.4f} | {nb_post['roc_auc']:.4f} |",
        "",
        "- **Impacto:** Comparação injusta entre NB original (mediana, SMOTE) e demais modelos (alvo ≥20%).",
        "- **Valor oficial:** Reavaliação unificada em `naive_bayes_folds.csv`.",
        "",
    ]

    tree_pre = pre[pre["Modelo"] == "Árvore de Decisão"].iloc[0]
    tree_post = post_real[post_real["model"] == "Árvore de Decisão"].iloc[0]
    issues += [
        "### 3. Árvore de Decisão — mudança de posição no ranking",
        "",
        f"| Fonte | Rank | Recall | Precision |",
        f"|-------|------|--------|-----------|",
        f"| Pré-auditoria | 4º | {tree_pre['Recall']:.4f} | {tree_pre['Precision']:.4f} |",
        f"| Pós-auditoria | 1º | {tree_post['recall']:.4f} | {tree_post['precision']:.4f} |",
        "",
        "- **Impacto:** Vencedor real só emerge após correção metodológica.",
        "- **Valor oficial:** Pós-auditoria.",
        "",
    ]

    rn_path = MODELS / "Redes_Neurais" / "resultados_rede_neural_5fold.csv"
    if rn_path.exists():
        rn = pd.read_csv(rn_path).iloc[0]
        rn_post = post_real[post_real["model"] == "Rede Neural"].iloc[0]
        issues += [
            "### 4. Rede Neural — notebook vs auditoria",
            "",
            f"| Métrica | Notebook | Auditado | Δ |",
            f"|---------|----------|----------|---|",
            f"| Recall | {rn['recall_mean']:.4f} | {rn_post['recall']:.4f} | {rn_post['recall']-rn['recall_mean']:+.4f} |",
            f"| F1 | {rn['f1_mean']:.4f} | {rn_post['f1']:.4f} | {rn_post['f1']-rn['f1_mean']:+.4f} |",
            "",
            "- **Impacto:** Moderado; mesmo modelo, protocolo de threshold diferente.",
            "- **Valor oficial:** Auditado.",
            "",
        ]

    issues += [
        "### 5. Artefatos ausentes no working tree",
        "",
        "- CSVs de LR/Árvore/SVM não exportados no disco (só em `main`).",
        "- Nenhum `.pkl` carregável exceto via branch `main` (Naive Bayes).",
        "- **Mitigação:** Validação recalculada a partir de `*_folds.csv` auditados.",
        "",
    ]
    dest.write_text("\n".join(issues), encoding="utf-8")
    return dest


def _metrics_from_cm(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def validate_metrics_from_folds(dest: Path) -> Path:
    """Fases 4–5 — recalcula métricas a partir dos folds auditados."""
    audited = pd.read_csv(AUDIT_DIR / "audited_comparison.csv")
    audited = audited[audited["model"].isin(REAL_MODELS)]

    rows = []
    for model, path in FOLD_FILES.items():
        df = pd.read_csv(path)
        # Recalcular a partir da matriz de confusão média
        tp_m, fp_m, fn_m, tn_m = df[["tp", "fp", "fn", "tn"]].mean()
        recalc = _metrics_from_cm(tp_m, fp_m, fn_m, tn_m)

        aud = audited[audited["model"] == model].iloc[0]
        rows.append({
            "modelo": model,
            "accuracy": float(df["accuracy"].mean()),
            "precision": float(df["precision"].mean()),
            "recall": float(df["recall"].mean()),
            "f1": float(df["f1"].mean()),
            "roc_auc": float(df["roc_auc"].mean()),
            "pr_auc": float(df["pr_auc"].mean()),
            "accuracy_recalc_from_cm": recalc["accuracy"],
            "precision_recalc_from_cm": recalc["precision"],
            "recall_recalc_from_cm": recalc["recall"],
            "f1_recalc_from_cm": recalc["f1"],
            "max_metric_delta": max(
                abs(recalc["accuracy"] - df["accuracy"].mean()),
                abs(recalc["recall"] - df["recall"].mean()),
            ),
            "tn_mean": float(df["tn"].mean()),
            "tp_mean": float(df["tp"].mean()),
            "fonte": str(path.relative_to(PROJECT_ROOT)),
            "validado": "sim" if max(
                abs(recalc["accuracy"] - df["accuracy"].mean()),
                abs(recalc["f1"] - df["f1"].mean()),
            ) < 0.002 else "revisar",
        })

    out = pd.DataFrame(rows)
    out.to_csv(dest, index=False)
    return dest
