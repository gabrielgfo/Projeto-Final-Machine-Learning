"""Geração do relatório PDF final pós-auditoria metodológica."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
AUDIT_VIZ_DIR = AUDIT_DIR / "visualizations"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PDF = RESULTS_DIR / "final_report.pdf"
OUTPUT_PDF_AUDIT = AUDIT_DIR / "final_report_audited.pdf"

REAL_MODELS = [
    "Regressão Logística",
    "Árvore de Decisão",
    "SVM",
    "Naive Bayes",
    "Rede Neural",
]


def _styled_table(data: list[list], font_size: int = 8) -> Table:
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ])
    )
    return table


def _audited_comparison_df(new_ranking: pd.DataFrame) -> pd.DataFrame:
    """Converte ranking auditado para formato das visualizações."""
    real = new_ranking[new_ranking["Modelo"].isin(REAL_MODELS)].copy()
    return pd.DataFrame({
        "Modelo": real["Modelo"],
        "Accuracy": real["accuracy"],
        "Precision": real["precision"],
        "Recall": real["recall"],
        "F1": real["f1"],
    })


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    old_ranking = pd.read_csv(RESULTS_DIR / "ranking.csv")
    new_ranking = pd.read_csv(AUDIT_DIR / "audited_ranking.csv")
    new_ranking = new_ranking.rename(columns={"model": "Modelo"})
    return old_ranking, new_ranking


def _generate_audited_visualizations(new_ranking: pd.DataFrame) -> dict[str, Path]:
    from analysis.visualizations import generate_all_visualizations

    comparison = _audited_comparison_df(new_ranking)
    return generate_all_visualizations(comparison, AUDIT_VIZ_DIR)


def _changes_section(body: ParagraphStyle) -> list:
    changes = [
        (
            "<b>1. Seleção de threshold</b><br/>"
            "<b>Antes:</b> maximização cega de F1 em np.linspace(0.01, 0.99, 100) sobre split "
            "de calibração (20% do treino).<br/>"
            "<b>Depois:</b> Youden's J (sensibilidade + especificidade − 1) com busca restrita "
            "a [0.2, 0.8] e rejeição de thresholds cuja taxa de positivos previstos exceda "
            "1,5× a prevalência real.<br/>"
            "<b>Motivo científico:</b> maximizar F1 sem restrições permite thresholds no extremo "
            "inferior da grade, classificando quase tudo como positivo. Isso produz Recall≈1 e "
            "Accuracy≈Precision (padrão TN≈0), indistinguível do classificador trivial "
            "\"sempre positivo\". Youden's J penaliza explicitamente falsos positivos e falsos "
            "negativos de forma equilibrada."
        ),
        (
            "<b>2. Métricas independentes de threshold</b><br/>"
            "<b>Antes:</b> comparação baseada principalmente em Accuracy, Precision, Recall e F1 "
            "no threshold otimizado.<br/>"
            "<b>Depois:</b> inclusão obrigatória de ROC-AUC, PR-AUC e matriz de confusão "
            "(TP, FP, FN, TN) por fold.<br/>"
            "<b>Motivo científico:</b> AUC mede discriminação independente do limiar. Quando "
            "ROC-AUC é mediano (~0.70) mas Recall no threshold escolhido é excelente (~0.99), "
            "há evidência de inflação por threshold, não de modelo superior."
        ),
        (
            "<b>3. Verificações de sanidade obrigatórias</b><br/>"
            "<b>Antes:</b> ausentes no pipeline comparativo.<br/>"
            "<b>Depois:</b> seis checagens por fold (invariante matemático, comparador trivial, "
            "plausibilidade do threshold, padrão degenerado, coerência AUC, sustentação para "
            "conclusões).<br/>"
            "<b>Motivo científico:</b> em problemas de classificação com classes balanceadas "
            "(~50/50), um modelo que prevê sempre a classe positiva obtém Recall=1.0 e "
            "Accuracy=0.5 — métricas de threshold isoladas não detectam esse artefato."
        ),
        (
            "<b>4. Baselines triviais no ranking</b><br/>"
            "<b>Antes:</b> apenas os cinco classificadores reais.<br/>"
            "<b>Depois:</b> inclusão de \"Sempre Positivo\" e \"Sempre Majoritário\" com o "
            "mesmo score composto (0.45·Recall + 0.30·F1 + 0.15·Precision + 0.10·Accuracy).<br/>"
            "<b>Motivo científico:</b> qualquer modelo real deve superar claramente baselines "
            "triviais. Se não supera, o ganho reportado é metodológico, não preditivo."
        ),
        (
            "<b>5. Unificação do protocolo de avaliação</b><br/>"
            "<b>Antes:</b> notebooks com pipelines distintos; Naive Bayes com "
            "data/preprocessamento.py (mediana, QT_ING≥10); comparação sem ROC-AUC em todos.<br/>"
            "<b>Depois:</b> reavaliação com amostra 30k estratificada 50/50, alvo TAXA_EVASAO≥20%, "
            "pré-processamento mode|label|corr|standard por fold; SMOTE apenas no treino (NB).<br/>"
            "<b>Motivo científico:</b> comparação válida exige mesmo alvo, mesma população e "
            "pré-processamento ajustado exclusivamente no treino de cada fold."
        ),
        (
            "<b>6. Teste de significância estatística</b><br/>"
            "<b>Antes:</b> ausente; declaração categórica de \"vencedor\".<br/>"
            "<b>Depois:</b> Wilcoxon pareado entre os dois melhores modelos em Recall (5 folds).<br/>"
            "<b>Motivo científico:</b> diferenças em cinco folds podem ser instáveis; testes "
            "pareados avaliam se a diferença observada é estatisticamente distinguível do acaso."
        ),
    ]
    return [Paragraph(text, body) for text in changes]


def generate_final_report_pdf(output_path: Path | None = None) -> Path:
    """Gera PDF final com dados pós-auditoria e contraste com versão anterior."""
    destino = output_path or OUTPUT_PDF
    destino.parent.mkdir(parents=True, exist_ok=True)

    old_ranking, new_ranking = _load_data()
    viz_paths = _generate_audited_visualizations(new_ranking)

    old_real = old_ranking[old_ranking["Modelo"].isin([
        "Regressão Logística",
        "Árvore de Decisão",
        "SVM (Support Vector Machine)",
        "Naive Bayes",
        "Rede Neural (MLP 1 camada + SGD)",
    ])].copy()
    name_map = {
        "SVM (Support Vector Machine)": "SVM",
        "Rede Neural (MLP 1 camada + SGD)": "Rede Neural",
    }
    old_real["Modelo"] = old_real["Modelo"].replace(name_map)

    new_real = new_ranking[new_ranking["Modelo"].isin(REAL_MODELS)].copy()

    compare_rows = [["Métrica", "Regressão Logística (antes)", "Regressão Logística (depois)"]]
    old_lr = old_real[old_real["Modelo"] == "Regressão Logística"].iloc[0]
    new_lr = new_real[new_real["Modelo"] == "Regressão Logística"].iloc[0]
    _metric_pairs = [
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Accuracy", "accuracy"),
    ]
    for old_col, new_col in _metric_pairs:
        compare_rows.append([
            old_col,
            f"{old_lr[old_col]:.3f}",
            f"{new_lr[new_col]:.3f}",
        ])
    compare_rows.append(["ROC-AUC", "N/A", f"{new_lr['roc_auc']:.3f}"])
    compare_rows.append([
        "Rank global",
        str(int(old_ranking[old_ranking["Modelo"] == "Regressão Logística"]["Rank_Global"].iloc[0])),
        str(int(new_real[new_real["Modelo"] == "Regressão Logística"]["rank"].iloc[0])),
    ])

    old_table = [["Rank", "Modelo", "Recall", "F1", "Score"]]
    for _, row in old_real.sort_values("Rank_Global").iterrows():
        old_table.append([
            str(int(row["Rank_Global"])),
            str(row["Modelo"]),
            f"{row['Recall']:.3f}",
            f"{row['F1']:.3f}",
            f"{row['Score']:.3f}",
        ])

    new_table = [["Rank", "Modelo", "Recall", "F1", "ROC-AUC", "Score"]]
    for _, row in new_real.sort_values("rank").iterrows():
        new_table.append([
            str(int(row["rank"])),
            str(row["Modelo"]),
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['roc_auc']:.3f}",
            f"{row['score']:.3f}",
        ])

    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    section = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=8)

    elements: list = []

    elements.append(Paragraph(
        "Relatório Final — Comparação de Modelos (Pós-Auditoria Metodológica)",
        title,
    ))
    elements.append(Paragraph(
        "Classificação de cursos com alto risco de evasão acadêmica (ALTO_RISCO_EVASAO). "
        "Este documento substitui o relatório anterior em results/final_report.pdf, "
        "incorporando correções metodológicas validadas por auditoria estatística.",
        body,
    ))

    elements.append(Paragraph("1. Resumo Executivo", section))
    winner = new_real.sort_values("rank").iloc[0]
    elements.append(Paragraph(
        f"A auditoria invalidou a conclusão anterior de que <b>Regressão Logística</b> seria "
        f"o melhor modelo (Recall≈0.999). Com threshold corrigido, seu Recall cai para "
        f"<b>{new_lr['recall']:.3f}</b> e o score composto ({new_lr['score']:.3f}) fica "
        f"<b>abaixo dos baselines triviais</b> (0.775).<br/><br/>"
        f"O novo líder é <b>{winner['Modelo']}</b> (Score={winner['score']:.3f}, "
        f"Recall={winner['recall']:.3f}, ROC-AUC={winner['roc_auc']:.3f}). "
        f"Wilcoxon entre Árvore de Decisão e SVM em Recall: p=0.625 — "
        f"<b>sem diferença estatisticamente significativa</b> a α=0.05.",
        body,
    ))

    elements.append(Paragraph("2. Alterações Metodológicas (Antes vs Depois)", section))
    elements.extend(_changes_section(body))

    elements.append(PageBreak())
    elements.append(Paragraph("3. Impacto na Regressão Logística (Caso Crítico)", section))
    elements.append(Paragraph(
        "A Regressão Logística ilustra o bug raiz: com threshold em 0.01–0.05, todos os folds "
        "apresentavam Accuracy≈Precision≈0.754 e Recall≈1.0 — assinatura matemática de "
        "TN≈0 (classificar quase tudo como positivo). O GridSearchCV interno (F1=0.6675 com "
        "threshold 0.5) já indicava desempenho moderado; o salto para F1≈0.86 vinha apenas "
        "do pós-ajuste de threshold degenerado.",
        body,
    ))
    elements.append(_styled_table(compare_rows))

    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph("4. Ranking Anterior (Pré-Auditoria)", section))
    elements.append(_styled_table(old_table))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("5. Ranking Revisado (Pós-Auditoria)", section))
    elements.append(_styled_table(new_table))

    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("6. Tabela Completa Pós-Auditoria", section))
    full_table = [[
        "Modelo", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "PR-AUC", "TN médio", "Sanidade",
    ]]
    for _, row in new_ranking.iterrows():
        full_table.append([
            str(row["Modelo"]),
            f"{row['accuracy']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['roc_auc']:.3f}",
            f"{row['pr_auc']:.3f}",
            f"{row['tn_mean']:.0f}",
            "S" if row.get("degenerescence_pass", False) else "N",
        ])
    elements.append(_styled_table(full_table, font_size=7))

    elements.append(PageBreak())
    elements.append(Paragraph("7. Discussão Científica por Modelo", section))

    discussions = [
        (
            "<b>Árvore de Decisão (1º no ranking revisado):</b> ROC-AUC≈0.87 com TN médio≈2497 "
            "confirma discriminação real. Interpretabilidade via regras é vantagem prática. "
            "Risco de overfitting mitigado por poda (ccp_alpha) nos hiperparâmetros originais."
        ),
        (
            "<b>SVM (2º, estatisticamente equivalente à Árvore em Recall):</b> margem máxima com "
            "kernel RBF captura fronteiras não lineares. Precision e Recall equilibrados (~0.77–0.81). "
            "Custo computacional maior, porém desempenho estável entre folds."
        ),
        (
            "<b>Rede Neural (3º):</b> maior ROC-AUC (≈0.87) e bom Recall (≈0.75). Capacidade de "
            "representação não linear, porém menor interpretabilidade e dependência de tuning."
        ),
        (
            "<b>Regressão Logística (6º):</b> após correção, Recall≈0.50 e ROC-AUC≈0.70 — desempenho "
            "mediano, compatível com separabilidade linear limitada. A versão anterior era "
            "metodologicamente inválida; o modelo mantém valor para interpretabilidade, não "
            "como líder em Recall."
        ),
        (
            "<b>Naive Bayes (7º):</b> no pipeline unificado, Recall≈0.05 apesar de Precision≈0.84. "
            "A hipótese de independência e o pré-processamento ComplementNB+SMOTE não se "
            "adequaram a este dataset tabular com atributos correlacionados."
        ),
    ]
    for text in discussions:
        elements.append(Paragraph(text, body))

    elements.append(Paragraph("8. Visualizações (dados pós-auditoria)", section))
    for key in ("bar_chart", "metric_heatmap", "radar_chart"):
        img_path = viz_paths[key]
        if img_path.exists():
            elements.append(Image(str(img_path), width=16 * cm, height=9 * cm))
            elements.append(Spacer(1, 0.2 * cm))

    elements.append(PageBreak())
    elements.append(Paragraph("9. Conclusão para Relatório Acadêmico", section))
    elements.append(Paragraph(
        "A comparação original entre modelos de evasão acadêmica continha um vício metodológico "
        "sistemático na seleção de threshold, que inflava artificialmente Recall e F1 da "
        "Regressão Logística e Naive Bayes. Após correção com Youden's J restrito, verificações "
        "de sanidade, baselines triviais e métricas AUC, conclui-se que:<br/><br/>"
        "• <b>Árvore de Decisão</b> e <b>SVM</b> são os modelos mais defensáveis para "
        "monitoramento de evasão com foco em Recall.<br/>"
        "• <b>Regressão Logística</b> não deve ser declarada vencedora; seu desempenho anterior "
        "era artefato estatístico.<br/>"
        "• Qualquer afirmação de superioridade deve citar ROC-AUC e TN, não apenas Recall "
        "no threshold otimizado.<br/>"
        "• Para implantação institucional, recomenda-se SVM ou Árvore de Decisão, com "
        "Rede Neural como alternativa quando capacidade não linear é prioritária.",
        body,
    ))

    doc = SimpleDocTemplate(
        str(destino),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(elements)
    logger.info("Relatório final pós-auditoria salvo em %s", destino)

    if destino == OUTPUT_PDF:
        import shutil
        shutil.copy2(destino, OUTPUT_PDF_AUDIT)
        logger.info("Cópia salva em %s", OUTPUT_PDF_AUDIT)

    return destino


if __name__ == "__main__":
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO)
    generate_final_report_pdf()
