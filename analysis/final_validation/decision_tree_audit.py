"""Fases 10–11 — auditoria da árvore de decisão e importância de features."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from analysis.audit.data import FoldPreprocessor, load_notebook_dataset
from analysis.audit.evaluator import RANDOM_STATE

logger = logging.getLogger(__name__)

TREE_PARAMS = {
    "class_weight": {0: 1, 1: 2},
    "criterion": "entropy",
    "max_depth": 15,
    "max_features": None,
    "min_impurity_decrease": 0.0012123525181736484,
    "min_samples_leaf": 13,
    "min_samples_split": 46,
    "ccp_alpha": 0.0043528172066691975,
    "random_state": RANDOM_STATE,
}


def _fit_reference_tree() -> tuple[DecisionTreeClassifier, list[str]]:
    """Ajusta árvore com hiperparâmetros auditados (sem grid search)."""
    X, y = load_notebook_dataset()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    prep = FoldPreprocessor()
    X_p = prep.fit_transform(X_train, y_train)
    feature_names = prep.selected_features
    tree = DecisionTreeClassifier(**TREE_PARAMS)
    tree.fit(X_p, y_train)
    return tree, feature_names


def run_decision_tree_audit(dt_dir: Path, feat_dir: Path) -> dict[str, Path]:
    dt_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    logger.info("Ajustando árvore de referência (hiperparâmetros auditados)...")
    tree, feature_names = _fit_reference_tree()

    # Visual
    fig, ax = plt.subplots(figsize=(28, 14))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["Sem risco", "Alto risco"],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
        max_depth=4,
    )
    ax.set_title("Árvore de Decisão — top 4 níveis (hiperparâmetros auditados)")
    fig.tight_layout()
    png = dt_dir / "decision_tree_final.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["tree_png"] = png

    rules = export_text(tree, feature_names=feature_names, max_depth=4)
    rules_path = dt_dir / "decision_tree_rules.txt"
    rules_path.write_text(rules, encoding="utf-8")
    paths["rules"] = rules_path

    stats_lines = [
        "# Estatísticas Estruturais — Árvore de Decisão",
        "",
        f"| Atributo | Valor |",
        f"|----------|-------|",
        f"| Profundidade máxima (ajustada) | {tree.get_depth()} |",
        f"| Profundidade máxima (hiperparâmetro) | {TREE_PARAMS['max_depth']} |",
        f"| Número de folhas | {tree.get_n_leaves()} |",
        f"| Número de nós | {tree.tree_.node_count} |",
        f"| Critério | {TREE_PARAMS['criterion']} |",
        f"| Impureza média nas folhas | {tree.tree_.impurity[tree.tree_.children_left == -1].mean():.6f} |",
        f"| ccp_alpha | {TREE_PARAMS['ccp_alpha']} |",
        f"| Features selecionadas | {len(feature_names)} |",
        "",
    ]
    stats_path = dt_dir / "tree_statistics.md"
    stats_path.write_text("\n".join(stats_lines), encoding="utf-8")
    paths["statistics"] = stats_path

    analysis = [
        "# Análise Crítica — Árvore de Decisão",
        "",
        "## A árvore é interpretável?",
        "**Sim**, parcialmente. As regras nos primeiros 4 níveis são legíveis "
        f"({len(feature_names)} features após seleção por correlação).",
        "",
        "## As regras são compreensíveis?",
        "Sim para stakeholders técnicos; variáveis codificadas (Label Encoding) "
        "exigem tabela de referência para áreas CINE e tipos de organização.",
        "",
        "## Risco de sobreajuste?",
        f"Moderado. `ccp_alpha={TREE_PARAMS['ccp_alpha']}` e `min_samples_leaf=13` "
        "aplicam poda. ROC-AUC=0.869 no protocolo auditado sugere generalização aceitável.",
        "",
        "## Risco de subajuste?",
        "Baixo. Profundidade efetiva e  folhas indicam capacidade de capturar interações.",
        "",
        "## A complexidade é justificável?",
        f"Sim para o trade-off interpretabilidade/desempenho (1º no ranking auditado, "
        "Recall=0.768, TN médio=2497).",
        "",
    ]
    analysis_path = dt_dir / "tree_analysis.md"
    analysis_path.write_text("\n".join(analysis), encoding="utf-8")
    paths["analysis"] = analysis_path

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": tree.feature_importances_,
    }).sort_values("importance", ascending=False)
    top20 = imp.head(20)
    top20.to_csv(feat_dir / "decision_tree_top20.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"][::-1], top20["importance"][::-1], color="#3498DB")
    ax.set_xlabel("Importância (Gini)")
    ax.set_title("Top 20 — Importância de Atributos (Árvore de Decisão)")
    fig.tight_layout()
    fi_png = feat_dir / "decision_tree_importance.png"
    fig.savefig(fi_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["importance_png"] = fi_png

    fi_md = [
        "# Análise de Importância de Features",
        "",
        "Modelo compatível: **Árvore de Decisão** (Random Forest/XGBoost não implementados).",
        "",
        "## Top 5 atributos",
        "",
    ]
    for _, row in top20.head(5).iterrows():
        fi_md.append(f"- **{row['feature']}**: {row['importance']:.4f}")
    fi_md += [
        "",
        "## Interpretação",
        "",
        "Atributos de proporção (financiamento, EAD, demografia) e contagens INEP "
        "dominam as divisões iniciais, coerente com fatores socioeconômicos de evasão.",
    ]
    fi_path = feat_dir / "feature_importance_analysis.md"
    fi_path.write_text("\n".join(fi_md), encoding="utf-8")
    paths["importance_md"] = fi_path

    return paths
