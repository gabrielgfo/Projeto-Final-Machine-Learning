# Resumo das Matrizes de Confusão (média 5 folds)

| Modelo | TN | FP | FN | TP | Interpretação |
|--------|----|----|----|-----|---------------|
| Árvore de Decisão | 2497 | 503 | 695 | 2305 | FP elevado — falsos alarmes, Equilíbrio TP/TN favorável |
| SVM | 2457 | 543 | 680 | 2320 | FP elevado — falsos alarmes, Equilíbrio TP/TN favorável |
| Rede Neural | 2538 | 462 | 739 | 2261 | Equilíbrio TP/TN favorável |
| Regressão Logística | 2450 | 550 | 1490 | 1510 | FP elevado — falsos alarmes |
| Naive Bayes | 2973 | 27 | 2860 | 140 | Alto FN — perde positivos (evasão) |

## Implicações práticas

- **FP:** Cursos sinalizados como alto risco sem serem — custo de intervenção desnecessária.
- **FN:** Cursos em risco não detectados — falha crítica para política de retenção.
- Modelos com Recall alto exigem FN baixo; TN alto isolado não basta (caso Naive Bayes).