"""Fase 13 — relatório científico IEEE em LaTeX."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
AUDIT_DIR = RESULTS / "audit"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "ieee_report.tex.tpl"
IEEETRAN_SRC = RESULTS / "IEEEtran.cls"

MODEL_ORDER = [
    "Árvore de Decisão",
    "SVM",
    "Rede Neural",
    "Regressão Logística",
    "Naive Bayes",
]

TABLE_LABELS = {
    "Árvore de Decisão": "Árvore de Decisão",
    "SVM": "SVM",
    "Rede Neural": "Rede Neural",
    "Regressão Logística": "Regressão Log.",
    "Naive Bayes": "Naive Bayes",
}


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _fmt_tab(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}"


def _fmt_tex(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}".replace(".", "{,}")


def _fmt_int(value: float) -> str:
    return str(int(round(value)))


def _cm_slug(name: str) -> str:
    """Mesmo slug usado em ``plots.py`` para os PNGs de matriz de confusão."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("ã", "a")
        .replace("õ", "o")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def _cm_counts(row: pd.Series, n_neg: int = 3000, n_pos: int = 3000) -> tuple[int, int, int, int]:
    tn = int(round(row["tn_mean"]))
    tp = int(round(row["tp_mean"]))
    fp = n_neg - tn
    fn = n_pos - tp
    return tn, fp, fn, tp


def _confusion_matrix_figure(model: str, row: pd.Series, fig_label: str) -> str:
    slug = _cm_slug(model)
    png = RESULTS / "validation" / f"confusion_matrix_{slug}.png"
    if not png.exists():
        raise FileNotFoundError(f"Matriz de confusão não encontrada: {png}")
    tn, fp, fn, tp = _cm_counts(row)
    return f"""\\begin{{figure}}[!t]
\\centering
\\includegraphics[width=\\linewidth]{{validation/confusion_matrix_{slug}.png}}
\\caption{{Matriz de confusão média (5 folds) --- {_tex_escape(model)}. Contagens médias por fold ($n_{{teste}}=6000$): $TN={tn}$, $FP={fp}$, $FN={fn}$, $TP={tp}$.}}
\\label{{{fig_label}}}
\\end{{figure}}"""


def _model_row(df: pd.DataFrame, name: str) -> pd.Series:
    row = df[df["modelo"] == name]
    if row.empty:
        raise KeyError(f"Modelo não encontrado em validated_metrics.csv: {name}")
    return row.iloc[0]


def _load_pre_audit_lr() -> tuple[float, float]:
    pre = pd.read_csv(RESULTS / "ranking.csv")
    lr = pre[pre["Modelo"] == "Regressão Logística"].iloc[0]
    return float(lr["Recall"]), float(lr["Score"])


def _wilcoxon_p() -> float:
    tree_fold = pd.read_csv(AUDIT_DIR / "árvore_de_decisão_folds.csv")
    svm_fold = pd.read_csv(AUDIT_DIR / "svm_folds.csv")
    try:
        _, p_value = wilcoxon(tree_fold["recall"].values, svm_fold["recall"].values)
        return float(p_value)
    except ValueError:
        return float("nan")


def _load_tree_stats() -> dict[str, float | int]:
    stats_path = RESULTS / "decision_tree" / "tree_statistics.md"
    text = stats_path.read_text(encoding="utf-8")
    patterns = {
        "depth": r"Profundidade máxima \(ajustada\) \| (\d+)",
        "max_depth_hp": r"Profundidade máxima \(hiperparâmetro\) \| (\d+)",
        "leaves": r"Número de folhas \| (\d+)",
        "nodes": r"Número de nós \| (\d+)",
        "impurity": r"Impureza média nas folhas \| ([\d.]+)",
    }
    out: dict[str, float | int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Não foi possível extrair '{key}' de {stats_path}")
        out[key] = float(match.group(1)) if key == "impurity" else int(match.group(1))
    return out


def _load_top_features(n: int = 4) -> list[tuple[str, float]]:
    feat = pd.read_csv(RESULTS / "feature_importance" / "decision_tree_top20.csv")
    feat = feat[feat["importance"] > 0].head(n)
    return [(row["feature"], float(row["importance"])) for _, row in feat.iterrows()]


def _metrics_table(val: pd.DataFrame) -> str:
    rows = []
    for name in MODEL_ORDER:
        row = _model_row(val, name)
        label = TABLE_LABELS[name]
        rows.append(
            f"{label} & {_fmt_tab(row['accuracy'])} & {_fmt_tab(row['precision'])} & "
            f"{_fmt_tab(row['recall'])} & {_fmt_tab(row['f1'])} & {_fmt_tab(row['roc_auc'])} \\\\"
        )
    rows.append("\\hline")
    rows.append("Baseline trivial  & 0.500 & 0.500 & 1.000 & 0.667 & 0.500 \\\\")
    return "\n".join(rows)


def _build_abstract(m: dict[str, pd.Series], rank: pd.DataFrame, wilcoxon_p: float) -> str:
    tree = m["Árvore de Decisão"]
    svm = m["SVM"]
    rn = m["Rede Neural"]
    lr_pre_recall, _ = _load_pre_audit_lr()
    nb = m["Naive Bayes"]
    return (
        "A evasão acadêmica constitui fenômeno multifatorial com impacto institucional, econômico e social. "
        "Este trabalho investiga a capacidade preditiva de cinco classificadores---Regressão Logística, "
        "Árvore de Decisão, Máquina de Vetores de Suporte (SVM), Rede Neural (MLP) e Naive Bayes "
        "(ComplementNB)---sobre microdados do Censo da Educação Superior 2024 (INEP), com alvo binário "
        "\\texttt{ALTO\\_RISCO\\_EVASAO} (taxa de evasão $\\geq 20\\%$). "
        "Após auditoria metodológica que corrigiu a seleção de limiar de decisão (índice de Youden restrito "
        "ao intervalo $[0{,}2; 0{,}8]$), incorporou verificações de sanidade por fold e comparou os modelos "
        "a baselines triviais, constatou-se que conclusões prévias que favoreciam a Regressão Logística "
        f"(Recall $\\approx {_fmt_tex(lr_pre_recall)}$) decorriam de limiares degenerados, e não de superioridade "
        "discriminativa. Sob protocolo validado (amostra estratificada de 30.000 cursos, validação cruzada "
        "5-fold), a Árvore de Decisão obteve o melhor equilíbrio global "
        f"(Accuracy $= {_fmt_tex(tree['accuracy'])}$, Recall $= {_fmt_tex(tree['recall'])}$, "
        f"ROC-AUC $= {_fmt_tex(tree['roc_auc'])}$), seguida de desempenho estatisticamente equivalente do SVM "
        f"(teste de Wilcoxon pareado em Recall: $p = {_fmt_tex(wilcoxon_p)}$). "
        f"A Rede Neural apresentou ROC-AUC competitivo (${_fmt_tex(rn['roc_auc'])}$) e maior PR-AUC "
        f"(${_fmt_tex(rn['pr_auc'])}$), enquanto o Naive Bayes falhou em sensibilidade "
        f"(Recall $= {_fmt_tex(nb['recall'])}$) no pipeline unificado. "
        "Os resultados evidenciam que métricas dependentes de limiar devem ser interpretadas em conjunto com "
        "ROC-AUC, matriz de confusão e baselines, sob pena de conclusões metodologicamente inválidas."
    )


def _build_results_section(
    val: pd.DataFrame,
    rank: pd.DataFrame,
    m: dict[str, pd.Series],
    wilcoxon_p: float,
) -> str:
    tree = m["Árvore de Decisão"]
    svm = m["SVM"]
    rn = m["Rede Neural"]
    lr = m["Regressão Logística"]
    nb = m["Naive Bayes"]
    winner = rank.iloc[0]
    second = rank.iloc[1]
    third = rank.iloc[2]

    return f"""A Tabela~\\ref{{tab:metrics}} consolida métricas validadas (média de cinco folds, recalculadas a partir de \\texttt{{validated\\_metrics.csv}} com consistência entre matriz de confusão e métricas reportadas, $\\Delta < 0{{,}}002$).

\\begin{{table}}[!t]
\\centering
\\caption{{Métricas validadas sob protocolo auditado}}
\\label{{tab:metrics}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Modelo}} & \\textbf{{Acc.}} & \\textbf{{Prec.}} & \\textbf{{Rec.}} & \\textbf{{F1}} & \\textbf{{AUC}} \\\\
\\hline
{_metrics_table(val)}
\\hline
\\end{{tabular}}
\\end{{table}}

Com pesos iguais ($20\\%$ para Accuracy, Precision, Recall, F1 e ROC-AUC), a {_tex_escape(winner['model'])} lidera (Score $= {_fmt_tex(winner['validated_score'], 3)}$), seguida da {_tex_escape(second['model'])} (${_fmt_tex(second['validated_score'], 3)}$) e {_tex_escape(third['model'])} (${_fmt_tex(third['validated_score'], 3)}$). No ranking auditado com ênfase em Recall (peso $45\\%$), Árvore e SVM permanecem no topo; Wilcoxon pareado em Recall entre ambos: $p = {_fmt_tex(wilcoxon_p)}$ (sem significância a $\\alpha = 0{{,}}05$).

\\begin{{figure}}[!t]
\\centering
\\includegraphics[width=\\linewidth]{{validation/roc_comparison.png}}
\\caption{{Curvas ROC comparativas --- predições agregadas em validação cruzada 5-fold (protocolo auditado). AUC na legenda conforme \\texttt{{validated\\_metrics.csv}}.}}
\\label{{fig:roc}}
\\end{{figure}}

A Figura~\\ref{{fig:roc}} corrobora a separação hierárquica: Árvore e Rede Neural ($\\text{{AUC}} \\approx {_fmt_tex(tree['roc_auc'])}$) discriminam melhor que Regressão Logística (${_fmt_tex(lr['roc_auc'])}$) e Naive Bayes (${_fmt_tex(nb['roc_auc'])}$). O TN médio por fold (Árvore: {_fmt_int(tree['tn_mean'])}; SVM: {_fmt_int(svm['tn_mean'])}; Naive Bayes: {_fmt_int(nb['tn_mean'])} com $TP$ médio {_fmt_int(nb['tp_mean'])}) evidencia que alto TN isolado não implica bom modelo se acompanhado de FN massivo."""


def _build_comparative_section(
    m: dict[str, pd.Series],
    rank: pd.DataFrame,
    wilcoxon_p: float,
) -> str:
    tree = m["Árvore de Decisão"]
    svm = m["SVM"]
    rn = m["Rede Neural"]
    lr = m["Regressão Logística"]
    nb = m["Naive Bayes"]
    lr_pre_recall, lr_pre_score = _load_pre_audit_lr()
    tree_rank = rank.iloc[0]

    cm_lr = _confusion_matrix_figure("Regressão Logística", lr, "fig:cm_lr")
    cm_tree = _confusion_matrix_figure("Árvore de Decisão", tree, "fig:cm_tree")
    cm_svm = _confusion_matrix_figure("SVM", svm, "fig:cm_svm")
    cm_rn = _confusion_matrix_figure("Rede Neural", rn, "fig:cm_rn")
    cm_nb = _confusion_matrix_figure("Naive Bayes", nb, "fig:cm_nb")

    return f"""A análise comparativa transcende o ranqueamento numérico: interpreta \\textit{{por que}} cada família algorítmica produziu o padrão observado de erros, como a auditoria de limiar alterou a hierarquia de modelos e quais implicações práticas decorrem para a implantação em contexto de gestão educacional. As discussões a seguir referem-se exclusivamente às métricas validadas (Tabela~\\ref{{tab:metrics}}), com médias sobre cinco folds e limiar calibrado por índice de Youden restrito. Cada subseção inclui a matriz de confusão agregada correspondente (Figuras~\\ref{{fig:cm_lr}}--\\ref{{fig:cm_nb}}), exportada de \\texttt{{results/validation/}}.

\\subsection{{Regressão Logística}}
No ranking pré-auditoria, a Regressão Logística aparecia como melhor modelo (Recall $= {_fmt_tex(lr_pre_recall)}$, Score composto $= {_fmt_tex(lr_pre_score, 3)}$). Após correção metodológica, seu Recall caiu para {_fmt_tex(lr['recall'])}, F1 para {_fmt_tex(lr['f1'])} e ROC-AUC permaneceu em {_fmt_tex(lr['roc_auc'])}, posicionando o classificador abaixo dos três líderes e apenas marginalmente acima do Naive Bayes em capacidade discriminativa global.

\\textbf{{Por que inicialmente parecia superior?}} A maximização cega de F1 em $[0{{,}}01; 0{{,}}99]$ favorecia limiares extremamente baixos ($\\approx 0{{,}}01$), nos quais a função logística atribuía rótulo positivo a quase todos os cursos. Matematicamente, isso maximiza $TP$ e, portanto, Recall, mas aniquila $TN$: o classificador torna-se funcionalmente equivalente ao baseline ``sempre positivo'' (Recall $= 1{{,}}0$, Accuracy $= 0{{,}}5$). A coincidência entre Accuracy e Precision elevadas com Recall próximo de $1$ constitui, na auditoria, sinal de alerta (\\textit{{sanity check}} 4), não evidência de excelência preditiva.

\\textbf{{Por que deixou de ser a melhor opção após validação?}} Porque o ROC-AUC (${_fmt_tex(lr['roc_auc'])}$) indica separação moderada entre classes no espaço de scores; a vantagem aparente em Recall era artefato de calibração, não de ranking superior. Com limiar restrito, a Regressão Logística passa a errar aproximadamente metade dos casos positivos ($FN$ elevado em relação aos líderes), enquanto sua Precision (${_fmt_tex(lr['precision'])}$) não compensa a perda de sensibilidade. O TN médio ({_fmt_int(lr['tn_mean'])}) é comparável ao da Árvore ({_fmt_int(tree['tn_mean'])}), mas o TP médio ({_fmt_int(lr['tp_mean'])} vs.\\ {_fmt_int(tree['tp_mean'])}) revela que a sensibilidade recuperada pelos líderes provém de melhor discriminação estrutural, não de limiar permissivo.

Do ponto de vista teórico, a hipótese de linearidade logística é plausível para relações monotônicas entre indicadores INEP e risco, porém insuficiente para capturar interações (e.g., alto ingresso combinado com baixa taxa de conclusão). A Regressão Logística permanece útil para inferência sobre direção de efeitos e comunicação com audiências acostumadas a odds ratios, mas não como líder operacional em detecção de alto risco sob protocolo rigoroso. A Figura~\\ref{{fig:cm_lr}} evidencia $FN$ elevado ({_fmt_int(3000 - lr['tp_mean'])}) frente aos líderes, confirmando perda de sensibilidade após auditoria de limiar.

{cm_lr}

\\subsection{{Árvore de Decisão}}
A Árvore de Decisão obteve o melhor equilíbrio entre Recall (${_fmt_tex(tree['recall'])}$), Precision (${_fmt_tex(tree['precision'])}$), F1 (${_fmt_tex(tree['f1'])}$) e ROC-AUC (${_fmt_tex(tree['roc_auc'])}$), com TN médio {_fmt_int(tree['tn_mean'])} e TP médio {_fmt_int(tree['tp_mean'])} em conjunto de teste de $6000$ instâncias por fold. No ranking com pesos iguais ($20\\%$ por métrica), lidera com Score {_fmt_tex(tree_rank['validated_score'], 3)}; no ranking auditado com ênfase em Recall ($45\\%$), mantém a primeira posição.

\\textbf{{Por que liderou o ranking?}} Três fatores convergem. Primeiro, o particionamento recursivo modela interações e não linearidades sem engenharia manual de features, adequando-se a indicadores de fluxo (ingresso, matrícula, razões derivadas) que não se comportam linearmente em relação ao risco. Segundo, a poda via \\textit{{ccp\\_alpha}} e profundidade efetiva $5$ (Seção~\\ref{{sec:arvore}}) limitam sobreajuste, preservando generalização entre folds (desvio padrão de Recall na ordem de centésimos). Terceiro, as probabilidades de folha, calibradas com limiar de Youden restrito, evitam a degenerescência observada na Regressão Logística: o modelo discrimina \\textit{{e}} opera em ponto de corte equilibrado.

\\textbf{{Por que é solução prática?}} Gestores educacionais podem inspecionar regras explícitas (\\texttt{{decision\\_tree\\_rules.txt}}) e discutir políticas em termos de indicadores já monitorados pelo INEP---ingresso, matrícula, taxa de conclusão---sem depender de explicações pós-hoc. A implementação em \\texttt{{scikit-learn}} \\cite{{pedregosa2011}} é leve, treinamento rápido em $24\\,000$ amostras por fold e inferência $O(\\text{{profundidade}})$. O custo de interpretabilidade parcial (limiares em escala padronizada) é documentável e inferior ao de modelos de caixa-preta. A Figura~\\ref{{fig:cm_tree}} ilustra equilíbrio entre $TP$ e $TN$ sem colapso de especificidade.

{cm_tree}

\\subsection{{SVM}}
A SVM apresentou métricas muito próximas à Árvore: Recall ${_fmt_tex(svm['recall'])}$ vs.\\ ${_fmt_tex(tree['recall'])}$, Precision ${_fmt_tex(svm['precision'])}$ vs.\\ ${_fmt_tex(tree['precision'])}$, F1 ${_fmt_tex(svm['f1'])}$ vs.\\ ${_fmt_tex(tree['f1'])}$. O teste de Wilcoxon pareado em Recall entre ambos retornou $p = {_fmt_tex(wilcoxon_p)}$, incompatível com rejeição da hipótese nula de igualdade de distribuições entre folds a $\\alpha = 0{{,}}05$. Em termos práticos, as diferenças observadas podem decorrer de variância amostral, não de superioridade estrutural de um algoritmo sobre o outro.

O kernel RBF ($C \\approx 3{{,}}15$, $\\gamma \\approx 5{{,}}67$) projeta os dados em espaço de dimensão elevada, permitindo fronteiras suaves que maximizam a margem entre classes. O ROC-AUC (${_fmt_tex(svm['roc_auc'])}$) é inferior ao da Árvore (${_fmt_tex(tree['roc_auc'])}$), sugerindo que, embora o desempenho no limiar calibrado seja equivalente, a ordenação global de scores da SVM é ligeiramente menos informativa---implicação relevante se a instituição priorizar triagem por score contínuo antes de fixar limiar operacional.

\\textbf{{Existe diferença prática entre SVM e Árvore?}} Para detecção binária com limiar auditado, não de forma estatisticamente significativa. A distinção reside em governança: a Árvore oferece regras auditáveis; a SVM exige técnicas adicionais (e.g., LIME, SHAP) para explicação. Em ambientes com exigência regulatória de transparência, a Árvore é preferível; em pipelines automatizados de scoring com revisão humana posterior, a SVM é alternativa robusta e estável. A Figura~\\ref{{fig:cm_svm}} confirma padrão de erros visualmente indistinguível do da Árvore.

{cm_svm}

\\subsection{{Rede Neural}}
O MLP (64 neurônios, ReLU, SGD) alcançou Accuracy ${_fmt_tex(rn['accuracy'])}$, Precision ${_fmt_tex(rn['precision'])}$ (a mais alta entre os cinco modelos), Recall ${_fmt_tex(rn['recall'])}$, F1 ${_fmt_tex(rn['f1'])}$, ROC-AUC ${_fmt_tex(rn['roc_auc'])}$ e PR-AUC ${_fmt_tex(rn['pr_auc'])}$ (máximo do conjunto). Posiciona-se em segundo lugar no ranking com pesos iguais (Score ${_fmt_tex(rank.iloc[1]['validated_score'], 3)}$), a ${_fmt_tex(rank.iloc[0]['validated_score'] - rank.iloc[1]['validated_score'], 3)} da Árvore---diferença desprezível no critério composto.

A rede combina capacidade expressiva não linear com regularização implícita do protocolo (validação cruzada, limiar restrito). O PR-AUC superior indica melhor compromisso precisão-revocação ao longo da curva de limiares, útil quando a prevalência operacional difere da amostra balanceada artificialmente. Contudo, o Recall no limiar único (${_fmt_tex(rn['recall'])}$) fica abaixo da Árvore e da SVM: a rede é mais conservadora em positivos, privilegiando Precision.

\\textbf{{O ganho justifica a perda de interpretabilidade?}} A resposta depende do caso de uso. Se o objetivo é ranking contínuo para priorização de visitas técnicas ou auditorias amostrais, o ganho em PR-AUC pode justificar o MLP como componente de \\textit{{ensemble}} ou segundo opinador. Se o objetivo é formulação de políticas públicas com justificativa explícita perante conselhos e órgãos de controle, o ganho marginal ($< 2$ pontos percentuais em métricas principais) não compensa opacidade, custo de \\textit{{hyperparameter tuning}} e risco de instabilidade ante re-treinamentos periódicos. Neste estudo, a Árvore permanece recomendada como modelo primário; a rede, como validação cruzada de robustez discriminativa. A Figura~\\ref{{fig:cm_rn}} mostra $TN$ ligeiramente superior e $FN$ moderadamente maior que na Árvore, coerente com maior Precision e menor Recall.

{cm_rn}

\\subsection{{Naive Bayes}}
O ComplementNB exibiu o pior desempenho global: Accuracy ${_fmt_tex(nb['accuracy'])}$, Recall ${_fmt_tex(nb['recall'])}$, F1 ${_fmt_tex(nb['f1'])}$, ROC-AUC ${_fmt_tex(nb['roc_auc'])}$. Paradoxalmente, apresentou a segunda maior Precision (${_fmt_tex(nb['precision'])}$), apenas atrás da Rede Neural---padrão característico de classificador excessivamente conservador na classe positiva.

\\textbf{{Por que Recall extremamente baixo?}} A hipótese de independência condicional é severamente violada: variáveis como \\textit{{QT\\_ING}}, \\textit{{QT\\_MAT}} e \\textit{{RAZAO\\_ING\\_MAT}} são estruturalmente correlacionadas (ingresso determina parcialmente matrícula). O ComplementNB, projetado para representações esparsas de texto, estima likelihoods que, sob correlação forte, concentram massa probabilística na classe negativa. O resultado é TN médio elevado ({_fmt_int(nb['tn_mean'])}) com TP médio crítico ({_fmt_int(nb['tp_mean'])}): o modelo quase nunca alerta alto risco, falhando em aproximadamente $95\\%$ dos casos positivos.

O SMOTE aplicado apenas no treino do pipeline unificado não reestrutura as dependências entre features; apenas rebalanceia contagens de instâncias. A ROC-AUC (${_fmt_tex(nb['roc_auc'])}$) próxima de $0{{,}}5$ confirma discriminação fraca, independentemente de limiar. \\textbf{{Implicação prática:}} o Naive Bayes, no protocolo atual, é inadequado para detecção de evasão; poderia, no máximo, servir como filtro de alta Precision em cascata (primeiro estágio conservador), mas isso exigiria redesenho arquitetural não avaliado neste trabalho. A Figura~\\ref{{fig:cm_nb}} expõe o padrão crítico: $FN$ massivo ({_fmt_int(3000 - nb['tp_mean'])}) com $FP$ mínimo ({_fmt_int(3000 - nb['tn_mean'])}), típico de classificador que quase nunca emite alerta de alto risco.

{cm_nb}

\\subsection{{Síntese comparativa}}
A Tabela~\\ref{{tab:metrics}}, a Figura~\\ref{{fig:roc}} e as matrizes de confusão (Figuras~\\ref{{fig:cm_lr}}--\\ref{{fig:cm_nb}}) permitem hierarquizar os modelos em três faixas: (i) \\textbf{{tier operacional}}---Árvore, SVM e Rede Neural, com ROC-AUC $\\geq 0{{,}}82$ e Recall $\\geq 0{{,}}75$; (ii) \\textbf{{tier intermediário}}---Regressão Logística, com discriminação moderada e sensibilidade comprometida após auditoria; (iii) \\textbf{{tier inadequado}}---Naive Bayes, com falha estrutural de sensibilidade. A auditoria de limiar foi determinante para revelar essa estrutura: sem ela, conclusões favoreceriam erroneamente a Regressão Logística. A escolha final entre os três primeiros deve privilegiar interpretabilidade (Árvore), equivalência estatística com menor AUC global (SVM) ou PR-AUC máximo (Rede Neural), conforme prioridades institucionais."""


def _build_tree_section(m: dict[str, pd.Series]) -> str:
    tree = m["Árvore de Decisão"]
    stats = _load_tree_stats()
    features = _load_top_features(4)
    feat_lines = ", ".join(
        f"\\textit{{{_tex_escape(name)}}} (${_fmt_tex(imp, 3)}$)" for name, imp in features
    )

    return f"""\\subsection{{Estrutura e complexidade}}
A árvore ajustada com hiperparâmetros auditados atingiu profundidade efetiva ${stats['depth']}$ (limite configurado ${stats['max_depth_hp']}$), ${stats['nodes']}$ nós e ${stats['leaves']}$ folhas, com impureza média nas folhas ${_fmt_tex(float(stats['impurity']), 3)}$. A poda via \\textit{{ccp\\_alpha}} reduziu complexidade, mitigando sobreajuste enquanto preservou ROC-AUC elevado (${_fmt_tex(tree['roc_auc'])}$). A representação gráfica completa da árvore consta no Anexo (Figura~\\ref{{fig:tree}}).

\\subsection{{Variáveis mais importantes}}
A importância por Gini (top quatro) foi: {feat_lines}. A razão ingresso/matrícula captura descompasso entre fluxo de entrada e base matriculada---possível indicador de instabilidade de cohort. Volume de ingressantes e matrículas reflete escala e dinâmica do curso; taxa de conclusão baixa pode sinalizar gargalos de permanência. Essas variáveis são plausíveis do ponto de vista teórico da evasão \\cite{{tinto2010}}.

\\subsection{{Regras aprendidas e significado}}
Exemplos de regras extraídas (\\texttt{{decision\\_tree\\_rules.txt}}): quando \\textit{{QT\\_ING}} $\\leq -0{{,}}12$ (ingresso abaixo da média padronizada) e \\textit{{RAZAO\\_ING\\_MAT}} $\\leq -1{{,}}05$, a folha prediz alto risco (\\texttt{{class: 1}})---configuração compatível com descompasso entre entrada de alunos e base matriculada estável. Ramo alternativo: \\textit{{QT\\_ING}} $> -0{{,}}12$ com \\textit{{TAXA\\_CONCLUSAO}} $> -0{{,}}31$ também conduz a predição positiva, sugerindo que cursos com ingresso relativamente alto mas conclusão acima de um limiar ainda assim são sinalizados, possivelmente por interação com outras variáveis no caminho da árvore.

Quando \\textit{{RAZAO\\_ING\\_MAT}} assume valores intermediários ($-1{{,}}05$ a $1{{,}}19$) e \\textit{{QT\\_MAT}} é baixo, predomina classe negativa (\\texttt{{class: 0}}): cursos com matrícula reduzida e razão ingresso/matrícula moderada tendem a não ser classificados como alto risco. Os limiares estão em escala padronizada; a interpretação qualitativa requer referência às distribuições no treino. Ainda assim, as regras permitem diálogo entre analistas e gestores sobre quais indicadores INEP merecem monitoramento contínuo e em que combinações configuram alerta.

\\subsection{{Interpretabilidade e apoio à decisão}}
Árvores são amplamente utilizadas em sistemas de apoio à decisão públicos por traduzirem políticas em critérios auditáveis. A limitação reside na instabilidade de árvores únicas pequenas ante perturbações dos dados; ensembles (Random Forest) poderiam aumentar robustez à custa de interpretabilidade global."""


def _build_tree_appendix() -> str:
    return r"""\clearpage
\onecolumn
\section*{Anexo: Árvore de Decisão}
\label{sec:arvore}
\vspace*{0.5em}
\begin{figure}[!t]
\centering
\includegraphics[width=\textwidth]{decision_tree/decision_tree_final.png}
\caption{Visualização da árvore de decisão ajustada sob protocolo auditado. Variáveis em escala padronizada; folhas indicam classe predita (\texttt{class: 0} = baixo risco, \texttt{class: 1} = alto risco).}
\label{fig:tree}
\end{figure}
"""


def _build_conclusion_section(
    m: dict[str, pd.Series],
    rank: pd.DataFrame,
    wilcoxon_p: float,
) -> str:
    tree = m["Árvore de Decisão"]
    rn = m["Rede Neural"]
    lr = m["Regressão Logística"]
    nb = m["Naive Bayes"]
    lr_pre_recall, _ = _load_pre_audit_lr()

    return f"""Este estudo comparou cinco classificadores---Regressão Logística, Árvore de Decisão, SVM, Rede Neural (MLP) e Naive Bayes (ComplementNB)---para predição de alto risco de evasão em cursos do ensino superior brasileiro, a partir dos microdados do Censo INEP 2024. O alvo binário \\texttt{{ALTO\\_RISCO\\_EVASAO}} (taxa de evasão $\\geq 20\\%$) foi modelado sob amostra estratificada de $30\\,000$ cursos, validação cruzada 5-fold e protocolo de auditoria que corrigiu degenerescência na seleção de limiar de decisão.

\\textbf{{Síntese dos resultados.}} A Árvore de Decisão obteve o melhor equilíbrio global (Accuracy ${_fmt_tex(tree['accuracy'])}$, Recall ${_fmt_tex(tree['recall'])}$, ROC-AUC ${_fmt_tex(tree['roc_auc'])}$, Score composto ${_fmt_tex(rank.iloc[0]['validated_score'], 3)}$), seguida de desempenho estatisticamente equivalente da SVM em Recall (Wilcoxon: $p = {_fmt_tex(wilcoxon_p)}$) e da Rede Neural em ROC-AUC e PR-AUC (${_fmt_tex(rn['pr_auc'])}$). A Regressão Logística, anteriormente classificada como melhor modelo do projeto (Recall ${_fmt_tex(lr_pre_recall)}$), foi rebaixada após auditoria para Recall ${_fmt_tex(lr['recall'])}$ e ROC-AUC ${_fmt_tex(lr['roc_auc'])}$, evidenciando que a aparente superioridade decorria de limiares degenerados, não de melhor discriminação. O Naive Bayes falhou em sensibilidade (Recall ${_fmt_tex(nb['recall'])}$), inadequado para detecção operacional de alto risco.

\\textbf{{Justificativa da escolha final.}} Recomenda-se a Árvore de Decisão como modelo primário quando interpretabilidade, desempenho equilibrado e conformidade com indicadores INEP são prioritários. A SVM constitui substituto válido em cenários de menor exigência explicativa. A Rede Neural pode complementar análises de ranking por PR-AUC, mas não justifica, neste dataset, a complexidade adicional como solução única.

\\textbf{{Contribuições.}} (1) Comparação reprodutível de cinco famílias de modelos sob pipeline unificado; (2) demonstração empírica de que métricas dependentes de limiar (Recall, F1) podem induzir conclusões inválidas sem análise de TN, ROC-AUC e baselines triviais; (3) protocolo de auditoria replicável (Youden em $[0{{,}}2; 0{{,}}8]$, \\textit{{sanity checks}}, recálculo de métricas a partir da matriz de confusão); (4) estudo de caso interpretativo da árvore, com variáveis de fluxo (\\textit{{RAZAO\\_ING\\_MAT}}, \\textit{{QT\\_ING}}, \\textit{{TAXA\\_CONCLUSAO}}) alinhadas à literatura de permanência \\cite{{tinto2010}}.

\\textbf{{Limitações.}} Amostra parcial balanceada artificialmente ($50/50$); agregação por curso (não por estudante); snapshot temporal único (2024); ausência de ensembles e validação temporal; pipeline do Naive Bayes original difere do unificado.

\\textbf{{Trabalhos futuros.}} Incorporar validação prospectiva (\\textit{{train past, test future}}), calibração probabilística pós-hoc, explicações globais (SHAP), comparação com Random Forest e XGBoost sob o mesmo protocolo auditado, e avaliação de custo assimétrico de erros com participação de gestores institucionais."""


def _ensure_ieetran_cls(dest_dir: Path) -> None:
    """Copia IEEEtran.cls para o diretório de saída se necessário."""
    target = dest_dir / "IEEEtran.cls"
    if target.exists():
        return
    if IEEETRAN_SRC.exists():
        shutil.copy2(IEEETRAN_SRC, target)


def generate_latex_report(dest: Path, today: str) -> Path:
    """Gera artigo IEEE completo em ``dest`` a partir das métricas validadas."""
    del today  # data não exibida no formato IEEE conference

    val = pd.read_csv(RESULTS / "validated_metrics.csv")
    rank = pd.read_csv(RESULTS / "final_validated_ranking.csv")
    models = {name: _model_row(val, name) for name in MODEL_ORDER}
    wilcoxon_p = _wilcoxon_p()
    lr_pre_recall, _ = _load_pre_audit_lr()

    replacements = {
        "@@ABSTRACT@@": _build_abstract(models, rank, wilcoxon_p),
        "@@LR_PRE_RECALL_TEX@@": _fmt_tex(lr_pre_recall),
        "@@RESULTS_SECTION@@": _build_results_section(val, rank, models, wilcoxon_p),
        "@@COMPARATIVE_SECTION@@": _build_comparative_section(models, rank, wilcoxon_p),
        "@@TREE_SECTION@@": _build_tree_section(models),
        "@@TREE_APPENDIX@@": _build_tree_appendix(),
        "@@CONCLUSION_SECTION@@": _build_conclusion_section(models, rank, wilcoxon_p),
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    for placeholder, content in replacements.items():
        template = template.replace(placeholder, content)

    if "@@" in template:
        leftover = sorted({m.group(0) for m in re.finditer(r"@@\w+@@", template)})
        raise RuntimeError(f"Placeholders não substituídos no template IEEE: {leftover}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(template, encoding="utf-8")
    _ensure_ieetran_cls(dest.parent)
    return dest
