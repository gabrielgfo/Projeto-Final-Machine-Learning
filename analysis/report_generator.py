"""Geração do relatório PDF comparativo final."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1"]


def _paragraphs_from_text(text: str, style: ParagraphStyle) -> list[Paragraph]:
    """Converte blocos de texto em parágrafos do ReportLab."""
    blocks = [block.strip() for block in text.split("<br/><br/>") if block.strip()]
    return [Paragraph(block, style) for block in blocks]


def _create_metrics_table(df: pd.DataFrame, title_columns: list[str]) -> Table:
    """Cria tabela genérica para o PDF."""
    data = [title_columns]
    for _, row in df.iterrows():
        data.append([str(row[col]) if col == "Modelo" else f"{row[col]:.4f}" for col in title_columns])

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ])
    )
    return table


def _create_usage_table(scenarios: list[dict[str, str]]) -> Table:
    """Cria tabela de cenários de uso."""
    data = [["Modelo", "Quando usar", "Quando evitar"]]
    for item in scenarios:
        data.append([item["Modelo"], item["Quando usar"], item["Quando evitar"]])

    table = Table(data, colWidths=[3.2 * cm, 6.2 * cm, 6.2 * cm])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ])
    )
    return table


def _add_image(elements: list, image_path: Path, width: float = 16 * cm) -> None:
    """Adiciona imagem ao PDF com tratamento de erro."""
    if not image_path.exists():
        logger.warning("Imagem não encontrada para o relatório: %s", image_path)
        return
    elements.append(Image(str(image_path), width=width, height=width * 0.55))
    elements.append(Spacer(1, 0.3 * cm))


def generate_final_report(
    comparison_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    analysis_bundle: dict[str, Any],
    visualization_paths: dict[str, Path],
    output_path: Path,
) -> Path:
    """Gera o relatório PDF final da comparação entre modelos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=10,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    elements: list = []

    elements.append(Paragraph("Relatório Comparativo Final — Evasão Acadêmica", title_style))
    elements.append(Spacer(1, 0.2 * cm))

    elements.append(Paragraph("1. Introdução", section_style))
    elements.append(Paragraph(
        "Este relatório consolida os resultados dos modelos de Machine Learning "
        "desenvolvidos para classificar cursos de educação superior com alto risco "
        "de evasão (ALTO_RISCO_EVASAO). A comparação prioriza Recall, por refletir "
        "o custo institucional de não identificar cursos críticos, mas também "
        "considera F1, Precision e Accuracy por meio de um score composto.",
        body_style,
    ))

    elements.append(Paragraph("2. Metodologia", section_style))
    elements.append(Paragraph(
        "O pipeline percorreu automaticamente a pasta <b>models/</b>, localizou "
        "arquivos de métricas agregadas em cada subpasta, extraiu Accuracy, "
        "Precision, Recall e F1, consolidou os resultados e calculou um ranking "
        "global com pesos 0,45 (Recall), 0,30 (F1), 0,15 (Precision) e "
        "0,10 (Accuracy). Foram geradas visualizações comparativas e uma análise "
        "científica automatizada baseada na teoria de cada algoritmo.",
        body_style,
    ))

    elements.append(Paragraph("3. Comparação Visual", section_style))
    _add_image(elements, visualization_paths["metric_heatmap"])
    _add_image(elements, visualization_paths["ranking_heatmap"])
    elements.append(PageBreak())
    _add_image(elements, visualization_paths["bar_chart"])
    _add_image(elements, visualization_paths["radar_chart"])

    elements.append(Paragraph("4. Comparação Quantitativa", section_style))
    elements.append(_create_metrics_table(
        comparison_df,
        ["Modelo", *METRIC_COLUMNS],
    ))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("5. Ranking Final", section_style))
    ranking_display = ranking_df[[
        "Rank_Global", "Modelo", "Score", *METRIC_COLUMNS,
    ]].copy()
    ranking_display.columns = [
        "Rank", "Modelo", "Score", *METRIC_COLUMNS,
    ]
    elements.append(_create_metrics_table(ranking_display, list(ranking_display.columns)))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("6. Discussão Científica", section_style))
    for paragraph in _paragraphs_from_text(analysis_bundle["theoretical"], body_style):
        elements.append(paragraph)

    elements.append(Paragraph("Análise Individual dos Modelos", section_style))
    for text in analysis_bundle["individual"].values():
        elements.append(Paragraph(text, body_style))

    elements.append(Paragraph("7. Justificativa do Modelo Vencedor", section_style))
    winner = ranking_df.iloc[0]
    elements.append(Paragraph(
        f"O modelo <b>{winner['Modelo']}</b> foi classificado em 1º lugar no ranking "
        f"composto (Score={winner['Score']:.4f}). Esse resultado indica o melhor "
        f"equilíbrio entre detectar cursos de alto risco (Recall={winner['Recall']:.3f}) "
        f"e manter desempenho global consistente (F1={winner['F1']:.3f}, "
        f"Precision={winner['Precision']:.3f}, Accuracy={winner['Accuracy']:.3f}). "
        f"A escolha não se baseia apenas em uma métrica isolada, mas na adequação "
        f"ao objetivo institucional de monitoramento preventivo da evasão.",
        body_style,
    ))

    elements.append(Paragraph("8. Cenários de Uso", section_style))
    elements.append(_create_usage_table(analysis_bundle["usage_scenarios"]))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("9. Conclusão", section_style))
    elements.append(Paragraph(analysis_bundle["conclusion"], body_style))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(elements)
    logger.info("Relatório final salvo em %s", output_path)
    return output_path
