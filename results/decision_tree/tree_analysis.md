# Análise Crítica — Árvore de Decisão

## A árvore é interpretável?
**Sim**, parcialmente. As regras nos primeiros 4 níveis são legíveis (13 features após seleção por correlação).

## As regras são compreensíveis?
Sim para stakeholders técnicos; variáveis codificadas (Label Encoding) exigem tabela de referência para áreas CINE e tipos de organização.

## Risco de sobreajuste?
Moderado. `ccp_alpha=0.0043528172066691975` e `min_samples_leaf=13` aplicam poda. ROC-AUC=0.869 no protocolo auditado sugere generalização aceitável.

## Risco de subajuste?
Baixo. Profundidade efetiva e  folhas indicam capacidade de capturar interações.

## A complexidade é justificável?
Sim para o trade-off interpretabilidade/desempenho (1º no ranking auditado, Recall=0.768, TN médio=2497).
