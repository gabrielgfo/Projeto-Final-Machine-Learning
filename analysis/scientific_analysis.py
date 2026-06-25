"""Geração automática de análise científica dos resultados comparativos."""

from __future__ import annotations

from typing import Any

import pandas as pd

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1"]

MODEL_THEORY: dict[str, dict[str, str]] = {
    "Regressão Logística": {
        "pontos_fortes": (
            "separabilidade linear, interpretabilidade dos coeficientes e "
            "baixo custo computacional"
        ),
        "limitacoes": (
            "dificuldade em capturar relações não lineares e interações "
            "complexas entre variáveis educacionais"
        ),
        "quando_usar": (
            "quando explicabilidade e rapidez de inferência forem requisitos "
            "do sistema de monitoramento"
        ),
        "quando_evitar": (
            "quando a fronteira de decisão for fortemente não linear ou "
            "houver muitas interações entre atributos"
        ),
    },
    "Árvore de Decisão": {
        "pontos_fortes": (
            "alta interpretabilidade, regras de decisão explícitas e "
            "capacidade de modelar não linearidades por partições"
        ),
        "limitacoes": (
            "sensibilidade a pequenas mudanças nos dados e risco de overfitting "
            "sem poda ou regularização adequada"
        ),
        "quando_usar": (
            "quando gestores precisam entender regras claras de risco de evasão"
        ),
        "quando_evitar": (
            "quando estabilidade preditiva entre coortes é mais importante "
            "que regras simples"
        ),
    },
    "SVM": {
        "pontos_fortes": (
            "maximização de margem, bom desempenho em espaços de alta dimensão "
            "e fronteiras complexas com kernels"
        ),
        "limitacoes": (
            "custo computacional elevado em grandes volumes e menor "
            "interpretabilidade dos resultados"
        ),
        "quando_usar": (
            "quando se busca equilíbrio entre generalização e fronteiras "
            "não lineares com dados tabulares estruturados"
        ),
        "quando_evitar": (
            "quando explicabilidade imediata ou treinamento em tempo real "
            "forem necessários"
        ),
    },
    "Naive Bayes": {
        "pontos_fortes": (
            "simplicidade, rapidez de treinamento e inferência, e robustez "
            "em cenários com muitos atributos"
        ),
        "limitacoes": (
            "hipótese de independência condicional entre atributos, frequentemente "
            "violada em dados educacionais correlacionados"
        ),
        "quando_usar": (
            "quando velocidade, simplicidade e baseline probabilístico "
            "forem prioridades"
        ),
        "quando_evitar": (
            "quando atributos altamente correlacionados dominam o fenômeno "
            "de evasão"
        ),
    },
    "Rede Neural": {
        "pontos_fortes": (
            "alta capacidade de representação de relações não lineares e "
            "combinações complexas entre variáveis"
        ),
        "limitacoes": (
            "maior necessidade de dados, risco de overfitting e baixa "
            "interpretabilidade direta"
        ),
        "quando_usar": (
            "quando a prioridade é maximizar capacidade preditiva e há volume "
            "suficiente de dados para generalização"
        ),
        "quando_evitar": (
            "quando interpretabilidade, transparência institucional ou "
            "explicabilidade forem exigências centrais"
        ),
    },
}


def _find_row(df: pd.DataFrame, keywords: tuple[str, ...]) -> pd.Series | None:
    """Localiza linha do DataFrame por palavras-chave no nome do modelo."""
    for _, row in df.iterrows():
        name = row["Modelo"].lower()
        if any(keyword in name for keyword in keywords):
            return row
    return None


def _best_by_metric(df: pd.DataFrame, metric: str) -> str:
    """Retorna o nome do modelo com melhor valor em uma métrica."""
    return str(df.loc[df[metric].idxmax(), "Modelo"])


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def generate_individual_analyses(df: pd.DataFrame) -> dict[str, str]:
    """Gera análise individual para cada modelo presente no ranking."""
    analyses: dict[str, str] = {}

    model_lookup = {
        "Regressão Logística": ("regress", "logist"),
        "Árvore de Decisão": ("arvore", "decis", "tree"),
        "SVM": ("svm", "support vector"),
        "Naive Bayes": ("naive", "bayes"),
        "Rede Neural": ("rede", "neural", "mlp"),
    }

    for display_name, keywords in model_lookup.items():
        row = _find_row(df, keywords)
        if row is None:
            continue

        theory = MODEL_THEORY[display_name]
        analyses[display_name] = (
            f"<b>{display_name}</b> obteve Accuracy={_format_metric(row['Accuracy'])}, "
            f"Precision={_format_metric(row['Precision'])}, "
            f"Recall={_format_metric(row['Recall'])} e F1={_format_metric(row['F1'])}. "
            f"Do ponto de vista teórico, destaca-se por {theory['pontos_fortes']}. "
            f"Suas principais limitações incluem {theory['limitacoes']}. "
            f"No contexto de evasão acadêmica, o resultado observado sugere que "
            f"este algoritmo {'prioriza fortemente a detecção de casos positivos' if row['Recall'] >= 0.9 else 'busca um compromisso entre sensibilidade e precisão' if row['Recall'] >= 0.8 else 'é mais conservador na identificação de alto risco'}, "
            f"enquanto {'mantém precisão elevada' if row['Precision'] >= 0.85 else 'aceita mais falsos positivos para aumentar a cobertura dos casos críticos'}."
        )

    return analyses


def generate_theoretical_comparison(df: pd.DataFrame, ranking_df: pd.DataFrame) -> str:
    """Gera discussão teórica comparativa respondendo às perguntas-chave."""
    winner = ranking_df.iloc[0]
    best_recall = _best_by_metric(df, "Recall")
    best_f1 = _best_by_metric(df, "F1")
    best_precision = _best_by_metric(df, "Precision")

    balanced_row = ranking_df.sort_values(
        by=["Score", "F1", "Precision"],
        ascending=False,
    ).iloc[0]

    interpretable_candidates = []
    for name in ("Regressão Logística", "Árvore de Decisão"):
        row = _find_row(df, (name.split()[0].lower(),))
        if row is not None:
            interpretable_candidates.append((name, float(row["Recall"]), float(row["F1"])))

    interpretable_text = (
        ", ".join(f"{name} (Recall={rec:.3f}, F1={f1:.3f})" for name, rec, f1 in interpretable_candidates)
        if interpretable_candidates
        else "modelos interpretáveis disponíveis no experimento"
    )

    return (
        f"<b>1. Por que o modelo vencedor superou os demais?</b><br/>"
        f"O ranking composto favoreceu <b>{winner['Modelo']}</b> (Score={winner['Score']:.3f}) "
        f"por combinar Recall={winner['Recall']:.3f}, F1={winner['F1']:.3f}, "
        f"Precision={winner['Precision']:.3f} e Accuracy={winner['Accuracy']:.3f}. "
        f"Com peso de 45% para Recall, o critério reflete a gravidade dos falsos negativos "
        f"em evasão acadêmica.<br/><br/>"
        f"<b>2. Quais características do dataset favoreceram esse comportamento?</b><br/>"
        f"O dataset reúne atributos contínuos, categóricos codificados e proporções derivadas "
        f"do Censo da Educação Superior. Após balanceamento nos pipelines de treino, os modelos "
        f"mais expressivos conseguiram explorar interações entre modalidade, perfil de ingressantes "
        f"e estrutura institucional. O melhor desempenho global sugere que o problema possui "
        f"estrutura estatística aproveitável, mas não necessariamente linear.<br/><br/>"
        f"<b>3. O desempenho observado era esperado teoricamente?</b><br/>"
        f"Em parte, sim. Era esperado que modelos flexíveis (Rede Neural, SVM, Árvore) "
        f"superassem baselines simples em F1, enquanto modelos com forte viés de recall "
        f"(como Regressão Logística e Naive Bayes com SMOTE) se destacariam na sensibilidade. "
        f"A liderança de {best_recall} em Recall e de {best_f1} em F1 confirma trade-offs "
        f"clássicos entre sensibilidade e equilíbrio global.<br/><br/>"
        f"<b>4. Modelos não vencedores com vantagens práticas?</b><br/>"
        f"Sim. {best_precision} apresenta a maior Precision, útil para reduzir alarmes falsos. "
        f"Modelos interpretáveis como {interpretable_text} permanecem valiosos para "
        f"explicar decisões a gestores e órgãos reguladores, mesmo sem liderar o ranking.<br/><br/>"
        f"<b>5. Melhor equilíbrio entre desempenho, interpretabilidade e custo?</b><br/>"
        f"<b>{balanced_row['Modelo']}</b> oferece o melhor compromisso segundo o score composto, "
        f"mas a Árvore de Decisão e a Regressão Logística seguem como opções pragmáticas "
        f"quando transparência e custo computacional são prioritários.<br/><br/>"
        f"<b>6. Recomendação para monitoramento real de evasão?</b><br/>"
        f"Para triagem institucional com foco em não perder cursos de alto risco, recomenda-se "
        f"priorizar modelos com Recall elevado ({best_recall}). Para implantação operacional "
        f"com revisão humana, o ideal é combinar alta sensibilidade com precisão aceitável, "
        f"o que aponta para <b>{winner['Modelo']}</b> como candidato principal e "
        f"<b>{best_precision}</b> como alternativa quando falsos positivos forem custosos."
    )


def generate_usage_scenarios() -> list[dict[str, str]]:
    """Retorna tabela de cenários de uso para cada modelo."""
    scenarios = []
    for model_name, theory in MODEL_THEORY.items():
        scenarios.append({
            "Modelo": model_name,
            "Quando usar": theory["quando_usar"],
            "Quando evitar": theory["quando_evitar"],
        })
    return scenarios


def generate_conclusion(df: pd.DataFrame, ranking_df: pd.DataFrame) -> str:
    """Gera conclusão pronta para relatório acadêmico e apresentação."""
    winner = ranking_df.iloc[0]
    best_recall_model = _best_by_metric(df, "Recall")

    return (
        f"Este estudo comparou cinco abordagens de classificação para identificação de "
        f"cursos com alto risco de evasão no Censo da Educação Superior, priorizando "
        f"Recall por refletir o custo institucional dos falsos negativos. "
        f"Os resultados mostram que <b>{best_recall_model}</b> alcançou a maior sensibilidade "
        f"({df.loc[df['Modelo'] == best_recall_model, 'Recall'].iloc[0]:.3f}), "
        f"enquanto <b>{winner['Modelo']}</b> obteve o melhor score composto "
        f"({winner['Score']:.3f}), equilibrando Recall, F1, Precision e Accuracy. "
        f"A comparação evidencia que não existe modelo universalmente superior: a escolha "
        f"depende do objetivo institucional — triagem ampla de risco, precisão em alertas "
        f"ou interpretabilidade para tomada de decisão. "
        f"Para o relatório final e a apresentação, recomenda-se apresentar o ranking "
        f"quantitativo, discutir os trade-offs visualizados nos heatmaps e radar chart, "
        f"e justificar a adoção de <b>{winner['Modelo']}</b> como solução principal, "
        f"com <b>{best_recall_model}</b> como referência quando a prioridade absoluta "
        f"for não deixar cursos de alto risco sem detecção."
    )


def build_analysis_bundle(
    comparison_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> dict[str, Any]:
    """Monta pacote completo de textos analíticos para o relatório."""
    return {
        "individual": generate_individual_analyses(comparison_df),
        "theoretical": generate_theoretical_comparison(comparison_df, ranking_df),
        "usage_scenarios": generate_usage_scenarios(),
        "conclusion": generate_conclusion(comparison_df, ranking_df),
    }
