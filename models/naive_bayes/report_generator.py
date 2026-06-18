"""Geração automática do relatório PDF do modelo Naive Bayes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPORT_DIR = Path(__file__).resolve().parent / "report"
REPORT_PATH = REPORT_DIR / "naive_bayes_report.pdf"

EXPERIMENTS_PER_MODEL = 30

NOMES_MODELO = {
    "gaussian": "GaussianNB",
    "bernoulli": "BernoulliNB",
    "complement": "ComplementNB",
}


def _ler_json(caminho: Path) -> dict:
    """Lê arquivo JSON com tratamento de erros."""
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            return json.load(arquivo)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Não foi possível ler {caminho}.") from exc


def _ler_csv(caminho: Path) -> pd.DataFrame:
    """Lê arquivo CSV com tratamento de erros."""
    try:
        return pd.read_csv(caminho)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Não foi possível ler {caminho}.") from exc


def _melhor_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a melhor configuração de cada variante de Naive Bayes."""
    melhores = []
    for model_type in ("gaussian", "bernoulli", "complement"):
        subset = df[df["model_type"] == model_type]
        if subset.empty:
            continue
        melhor = subset.sort_values(
            by=["recall", "f1", "roc_auc", "recall_std"],
            ascending=[False, False, False, True],
        ).iloc[0]
        melhores.append(melhor)
    return pd.DataFrame(melhores)


def _gerar_texto_discussao_variantes(
    comparacao: pd.DataFrame,
    melhor_config: dict,
) -> str:
    """Gera discussão comparativa entre as variantes de Naive Bayes."""
    vencedor = NOMES_MODELO.get(melhor_config.get("model_type", ""), "N/A")
    metricas_vencedor = comparacao[
        comparacao["model_type"] == melhor_config.get("model_type")
    ]
    recall_vencedor = (
        float(metricas_vencedor.iloc[0]["recall"])
        if not metricas_vencedor.empty
        else 0.0
    )

    linhas_variantes = []
    for _, linha in comparacao.iterrows():
        nome = NOMES_MODELO.get(linha["model_type"], linha["model_type"])
        linhas_variantes.append(
            f"{nome} obteve recall médio de {linha['recall']:.3f}, "
            f"F1 de {linha['f1']:.3f} e ROC-AUC de {linha['roc_auc']:.3f}."
        )

    texto_variantes = " ".join(linhas_variantes)

    return (
        f"{texto_variantes} "
        f"<b>GaussianNB</b> assume distribuição normal dos atributos contínuos, "
        "sendo adequado quando variáveis apresentam comportamento aproximadamente "
        "gaussiano após padronização. "
        "<b>BernoulliNB</b> assume atributos binários, útil quando a presença ou "
        "ausência de características é mais informativa que sua magnitude. "
        "<b>ComplementNB</b> é derivado do MultinomialNB e tende a ser mais "
        "robusto em cenários com desbalanceamento de classes, ao modelar "
        "complementarmente a probabilidade das classes. "
        f"Empiricamente, <b>{vencedor}</b> apresentou o melhor recall médio "
        f"({recall_vencedor:.3f}) entre as variantes testadas, indicando que sua "
        "hipótese distributiva foi a mais compatível com o perfil do dataset de "
        "evasão — que combina atributos contínuos, categóricos codificados e "
        "proporções derivadas. "
        "O SMOTE aplicado apenas no treino de cada fold favoreceu a detecção da "
        "classe de alto risco sem contaminar a avaliação. "
        "Limitações permanecem na hipótese de independência condicional e na "
        "simplificação distributiva imposta por cada variante."
    )


def _gerar_texto_discussao_final(resumo: pd.DataFrame, melhor_config: dict) -> str:
    """Gera texto interpretativo do modelo final selecionado."""
    metricas = {row["metric"]: row for _, row in resumo.iterrows()}
    recall = float(metricas.get("recall", {}).get("mean", 0.0))
    f1 = float(metricas.get("f1", {}).get("mean", 0.0))
    roc_auc = float(metricas.get("roc_auc", {}).get("mean", 0.0))
    vencedor = NOMES_MODELO.get(melhor_config.get("model_type", ""), "Naive Bayes")

    if recall >= 0.75:
        desempenho = (
            f"O {vencedor} selecionado apresentou recall elevado, indicando boa "
            "capacidade de identificar cursos com alto risco de evasão."
        )
    elif recall >= 0.55:
        desempenho = (
            f"O {vencedor} selecionado obteve recall moderado, capturando parte "
            "relevante dos cursos de alto risco."
        )
    else:
        desempenho = (
            f"O {vencedor} selecionado apresentou recall limitado na detecção de "
            "cursos de alto risco."
        )

    return (
        f"{desempenho} Com F1 médio de {f1:.3f} e ROC-AUC médio de {roc_auc:.3f}, "
        "o modelo final reflete o trade-off entre sensibilidade e precisão "
        "observado na comparação entre variantes."
    )


def _criar_tabela_metricas(resumo: pd.DataFrame) -> Table:
    """Monta tabela de métricas do modelo final."""
    dados = [["Métrica", "Média", "Desvio Padrão"]]
    nomes = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }

    for _, linha in resumo.iterrows():
        nome = nomes.get(linha["metric"], str(linha["metric"]))
        dados.append([
            nome,
            f"{linha['mean']:.4f}",
            f"{linha['std']:.4f}",
        ])

    tabela = Table(dados, colWidths=[5 * cm, 4 * cm, 4 * cm])
    tabela.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ])
    )
    return tabela


def _criar_tabela_comparacao(comparacao: pd.DataFrame) -> Table:
    """Monta tabela comparativa entre variantes de Naive Bayes."""
    dados = [["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]

    for _, linha in comparacao.iterrows():
        dados.append([
            NOMES_MODELO.get(linha["model_type"], linha["model_type"]),
            f"{linha['accuracy']:.4f}",
            f"{linha['precision']:.4f}",
            f"{linha['recall']:.4f}",
            f"{linha['f1']:.4f}",
            f"{linha['roc_auc']:.4f}",
        ])

    tabela = Table(
        dados,
        colWidths=[3.2 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm],
    )
    tabela.setStyle(
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
    return tabela


def _formatar_parametros(config: dict) -> str:
    """Formata hiperparâmetros para exibição no relatório."""
    model_type = config.get("model_type", "N/A")
    nome = NOMES_MODELO.get(model_type, model_type)
    partes = [f"<b>Modelo:</b> {nome}"]

    if "var_smoothing" in config:
        partes.append(f"<b>var_smoothing:</b> {config['var_smoothing']}")
    if "alpha" in config:
        partes.append(f"<b>alpha:</b> {config['alpha']}")
    if "binarize" in config:
        partes.append(f"<b>binarize:</b> {config['binarize']}")
    if "norm" in config:
        partes.append(f"<b>norm:</b> {config['norm']}")

    return "<br/>".join(partes)


def gerar_relatorio(caminho_saida: Path | None = None) -> Path:
    """
    Gera relatório PDF com objetivo, comparação de variantes e resultados.

    Args:
        caminho_saida: Caminho opcional do PDF. Padrão: report/naive_bayes_report.pdf.

    Returns:
        Caminho do relatório gerado.
    """
    destino = caminho_saida or REPORT_PATH
    destino.parent.mkdir(parents=True, exist_ok=True)

    try:
        best_params = _ler_json(ARTIFACTS_DIR / "best_params.json")
        resumo = _ler_csv(ARTIFACTS_DIR / "metrics_summary.csv")
        experimentos = _ler_csv(ARTIFACTS_DIR / "all_experiments.csv")
        comparacao = _melhor_por_modelo(experimentos)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Falha ao carregar artefatos para o relatório.") from exc

    estilos = getSampleStyleSheet()
    titulo = ParagraphStyle(
        "Titulo",
        parent=estilos["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    subtitulo = ParagraphStyle(
        "Subtitulo",
        parent=estilos["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )
    corpo = ParagraphStyle(
        "Corpo",
        parent=estilos["BodyText"],
        fontSize=11,
        leading=14,
        spaceAfter=8,
    )

    elementos: list = []

    elementos.append(Paragraph("Relatório — Naive Bayes (Evasão Acadêmica)", titulo))
    elementos.append(Spacer(1, 0.3 * cm))

    elementos.append(Paragraph("1. Objetivo", subtitulo))
    elementos.append(
        Paragraph(
            "Este trabalho aborda a classificação de cursos de educação superior "
            "quanto ao risco de evasão acadêmica, utilizando microdados do Censo "
            "da Educação Superior (INEP). O objetivo é identificar cursos com "
            "elevada taxa de desvinculação de alunos, priorizando a métrica "
            "<b>Recall</b> para maximizar a detecção de casos de alto risco.",
            corpo,
        )
    )

    elementos.append(Paragraph("2. Estratégia de Pré-processamento", subtitulo))
    elementos.append(
        Paragraph(
            "Foram definidos três pipelines independentes:<br/>"
            "<b>GaussianNB:</b> VarianceThreshold, remoção de correlação "
            "(|r| &gt; 0,90), StandardScaler e SMOTE no treino.<br/>"
            "<b>BernoulliNB:</b> VarianceThreshold, remoção de correlação, "
            "Binarizer e SMOTE no treino.<br/>"
            "<b>ComplementNB:</b> MinMaxScaler para intervalo [0, 1] e SMOTE "
            "no treino.",
            corpo,
        )
    )

    elementos.append(Paragraph("3. Busca de Hiperparâmetros", subtitulo))
    elementos.append(
        Paragraph(
            f"Foram executados aproximadamente {EXPERIMENTS_PER_MODEL} experimentos "
            f"por variante ({EXPERIMENTS_PER_MODEL * 3} no total), com "
            "validação cruzada estratificada de 5 folds. "
            "Espaços de busca: GaussianNB (var_smoothing), BernoulliNB "
            "(alpha, binarize) e ComplementNB (alpha, norm). "
            "Critério de seleção: <b>Recall</b> médio, com desempate por F1, "
            "ROC-AUC e menor desvio padrão do Recall.",
            corpo,
        )
    )

    elementos.append(
        Paragraph("4. Comparação entre variantes de Naive Bayes", subtitulo)
    )
    elementos.append(_criar_tabela_comparacao(comparacao))
    elementos.append(Spacer(1, 0.4 * cm))
    elementos.append(
        Paragraph(_gerar_texto_discussao_variantes(comparacao, best_params), corpo)
    )

    elementos.append(Paragraph("5. Melhor Configuração Encontrada", subtitulo))
    elementos.append(Paragraph(_formatar_parametros(best_params), corpo))

    elementos.append(Paragraph("6. Resultados do Modelo Final", subtitulo))
    elementos.append(_criar_tabela_metricas(resumo))
    elementos.append(Spacer(1, 0.4 * cm))
    elementos.append(
        Paragraph(_gerar_texto_discussao_final(resumo, best_params), corpo)
    )

    try:
        doc = SimpleDocTemplate(
            str(destino),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        doc.build(elementos)
    except Exception as exc:
        raise RuntimeError(f"Falha ao gerar o relatório em {destino}.") from exc

    logger.info("Relatório PDF gerado em %s.", destino)
    return destino
