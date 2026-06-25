"""Fase 12 — discussão científica crítica."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
AUDIT = PROJECT_ROOT / "results" / "audit"


def build_scientific_discussion(dest: Path) -> Path:
    val = pd.read_csv(RESULTS / "validated_metrics.csv")
    rank = pd.read_csv(RESULTS / "final_validated_ranking.csv")

    lines = [
        "# Discussão Científica Crítica",
        "",
        "## Sobre o dataset",
        "",
        "- **Adequação:** Microdados INEP 2024 são representativos do censo de cursos superiores.",
        "- **Desbalanceamento:** Amostra auditada 50/50 (15k/classe); SMOTE apenas no Naive Bayes.",
        "- **Limitações:** Alvo binário por taxa ≥20%; não captura evasão longitudinal individual; "
        "possível vazamento mitigado por exclusão de colunas de situação no pipeline original.",
        "",
        "## Sobre os modelos",
        "",
        "| Critério | Melhor opção | Evidência |",
        "|----------|--------------|-----------|",
        f"| Equilíbrio geral (pesos iguais) | {rank.iloc[0]['model']} | Score={rank.iloc[0]['validated_score']:.4f} |",
        "| Interpretabilidade | Árvore de Decisão | Regras exportáveis, feature importance |",
        "| Robustez entre folds | SVM / Árvore | Wilcoxon p=0.625 — empate estatístico |",
        "| PR-AUC | Rede Neural | 0.883 no protocolo auditado |",
        "",
        "## Sobre as métricas",
        "",
        "- **Accuracy sozinha:** Insuficiente em 50/50; baselines triviais atingem 50%.",
        "- **ROC-AUC:** Altera conclusão — separa discriminação de threshold; LR cai de 'líder' a mediano.",
        "- **Precision vs Recall:** Naive Bayes (Prec=0.84, Rec=0.05) vs Árvore (Prec=0.82, Rec=0.77) "
        "contam histórias opostas — foco em evasão exige Recall.",
        "",
        "## Conclusão da validação",
        "",
        "Métricas oficiais: `validated_metrics.csv` derivado de `*_folds.csv` auditados. "
        "Divergências com PDFs pré-auditoria documentadas em `inconsistencies_report.md`.",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest
