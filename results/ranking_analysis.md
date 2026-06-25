# Análise do Ranking Validado

## Critérios

### Ranking pré-auditoria (`ranking.csv`)
- Pesos: Recall 45%, F1 30%, Precision 15%, Accuracy 10%
- Sem ROC-AUC, sem baselines, threshold não auditado

### Ranking auditado (`audited_ranking.csv`)
- Mesmos pesos acima + Youden's J + sanidade

### Ranking validado final (`final_validated_ranking.csv`)
- **Pesos iguais 20%** para Accuracy, Precision, Recall, F1, ROC-AUC

## Resultados

| Modelo | Score validado | Rank validado | Rank auditado | Rank pré |
|--------|----------------|---------------|---------------|----------|
| Árvore de Decisão | 0.8111 | 1 | 1 | 4 |
| Rede Neural | 0.8087 | 2 | 3 | 2 |
| SVM | 0.7987 | 3 | 2 | 3 |
| Regressão Logística | 0.6384 | 4 | 6 | 1 |
| Naive Bayes | 0.4175 | 5 | 7 | 5 |

## Perguntas de auditoria

### O ranking original é justificável?
**Não.** Declarava Regressão Logística como vencedor com Recall≈0.999 — artefato de threshold.

### O vencedor continua sendo o melhor modelo?
Com pesos iguais, o líder é **Árvore de Decisão**. No ranking auditado (ênfase em Recall), líder: **Árvore de Decisão**. Árvore e SVM permanecem no topo em ambos os critérios válidos.

### Há diferença estatisticamente relevante?
Wilcoxon Árvore vs SVM (Recall): statistic=5.0000, p=0.6250. Sem significância a α=0.05 — diferença operacional, não estatística.
