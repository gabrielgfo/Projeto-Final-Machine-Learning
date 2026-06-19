# Auditoria Metodológica — Evasão Acadêmica

> **Relatório PDF completo (pós-auditoria):** `results/final_report.pdf` e `results/audit/final_report_audited.pdf`

## Nota sobre pipelines (seção 5.4)
- Notebooks (LR, Árvore, SVM, Rede Neural): alvo `TAXA_EVASAO >= 20%`, pré-processamento `mode | label | corr | standard` por fold.
- Naive Bayes original usava `data/preprocessamento.py` (mediana, QT_ING>=10) — reavaliado aqui no pipeline unificado dos notebooks para comparação justa.
- Balanceamento: `class_weight=balanced` (LR, Árvore, SVM, MLP); SMOTE só no treino (NB).

## Verificações de Sanidade
### Regressão Logística

**Fold 1** — Passou: Sim

- 1_invariante: TP=1570, FP=596, FN=1430, TN=2404; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.4925; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.707, PR-AUC=0.706; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.662, Prec=0.725, Rec=0.523, F1=0.608; AUC=0.707 não corrobora excelência

**Fold 2** — Passou: Sim

- 1_invariante: TP=1630, FP=615, FN=1370, TN=2385; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5000; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.692, PR-AUC=0.690; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.669, Prec=0.726, Rec=0.543, F1=0.622; AUC=0.692 não corrobora excelência

**Fold 3** — Passou: Sim

- 1_invariante: TP=1396, FP=487, FN=1604, TN=2513; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5150; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.693, PR-AUC=0.689; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.651, Prec=0.741, Rec=0.465, F1=0.572; AUC=0.693 não corrobora excelência

**Fold 4** — Passou: Sim

- 1_invariante: TP=1434, FP=481, FN=1566, TN=2519; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5225; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.701, PR-AUC=0.700; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.659, Prec=0.749, Rec=0.478, F1=0.584; AUC=0.701 não corrobora excelência

**Fold 5** — Passou: Sim

- 1_invariante: TP=1520, FP=569, FN=1480, TN=2431; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5075; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.700, PR-AUC=0.697; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.658, Prec=0.728, Rec=0.507, F1=0.597; AUC=0.700 não corrobora excelência

### Árvore de Decisão

**Fold 1** — Passou: Sim

- 1_invariante: TP=2047, FP=253, FN=953, TN=2747; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.6650; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.868, PR-AUC=0.856; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.799, Prec=0.890, Rec=0.682, F1=0.772; AUC=0.868 corrobora excelência

**Fold 2** — Passou: Sim

- 1_invariante: TP=2379, FP=587, FN=621, TN=2413; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5225; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.870, PR-AUC=0.862; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.799, Prec=0.802, Rec=0.793, F1=0.798; AUC=0.870 corrobora excelência

**Fold 3** — Passou: Sim

- 1_invariante: TP=2369, FP=524, FN=631, TN=2476; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5225; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.872, PR-AUC=0.866; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.807, Prec=0.819, Rec=0.790, F1=0.804; AUC=0.872 corrobora excelência

**Fold 4** — Passou: Sim

- 1_invariante: TP=2387, FP=594, FN=613, TN=2406; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5225; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.872, PR-AUC=0.866; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.799, Prec=0.801, Rec=0.796, F1=0.798; AUC=0.872 corrobora excelência

**Fold 5** — Passou: Sim

- 1_invariante: TP=2345, FP=556, FN=655, TN=2444; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5300; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.864, PR-AUC=0.854; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.798, Prec=0.808, Rec=0.782, F1=0.795; AUC=0.864 corrobora excelência

### SVM

**Fold 1** — Passou: Sim

- 1_invariante: TP=2340, FP=527, FN=660, TN=2473; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.3350; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.831, PR-AUC=0.822; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.802, Prec=0.816, Rec=0.780, F1=0.798; AUC=0.831 corrobora excelência

**Fold 2** — Passou: Sim

- 1_invariante: TP=2336, FP=562, FN=664, TN=2438; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.3050; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.824, PR-AUC=0.810; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.796, Prec=0.806, Rec=0.779, F1=0.792; AUC=0.824 corrobora excelência

**Fold 3** — Passou: Sim

- 1_invariante: TP=2252, FP=479, FN=748, TN=2521; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.4775; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.818, PR-AUC=0.814; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.795, Prec=0.825, Rec=0.751, F1=0.786; AUC=0.818 corrobora excelência

**Fold 4** — Passou: Sim

- 1_invariante: TP=2351, FP=584, FN=649, TN=2416; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.3425; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.820, PR-AUC=0.802; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.794, Prec=0.801, Rec=0.784, F1=0.792; AUC=0.820 corrobora excelência

**Fold 5** — Passou: Sim

- 1_invariante: TP=2320, FP=561, FN=680, TN=2439; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.4100; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.817, PR-AUC=0.801; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.793, Prec=0.805, Rec=0.773, F1=0.789; AUC=0.817 corrobora excelência

### Naive Bayes

**Fold 1** — Passou: Sim

- 1_invariante: TP=159, FP=26, FN=2841, TN=2974; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5000; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.614, PR-AUC=0.618; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.522, Prec=0.859, Rec=0.053, F1=0.100; AUC=0.614 não corrobora excelência

**Fold 2** — Passou: Sim

- 1_invariante: TP=143, FP=31, FN=2857, TN=2969; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5000; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.596, PR-AUC=0.599; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.519, Prec=0.822, Rec=0.048, F1=0.090; AUC=0.596 não corrobora excelência

**Fold 3** — Passou: Sim

- 1_invariante: TP=151, FP=33, FN=2849, TN=2967; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5000; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.583, PR-AUC=0.588; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.520, Prec=0.821, Rec=0.050, F1=0.095; AUC=0.583 não corrobora excelência

**Fold 4** — Passou: Sim

- 1_invariante: TP=105, FP=18, FN=2895, TN=2982; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5000; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.593, PR-AUC=0.593; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.514, Prec=0.854, Rec=0.035, F1=0.067; AUC=0.593 não corrobora excelência

**Fold 5** — Passou: Sim

- 1_invariante: TP=144, FP=26, FN=2856, TN=2974; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5000; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.578, PR-AUC=0.585; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.520, Prec=0.847, Rec=0.048, F1=0.091; AUC=0.578 não corrobora excelência

### Rede Neural

**Fold 1** — Passou: Sim

- 1_invariante: TP=2243, FP=423, FN=757, TN=2577; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5450; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.877, PR-AUC=0.890; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.803, Prec=0.841, Rec=0.748, F1=0.792; AUC=0.877 corrobora excelência

**Fold 2** — Passou: Sim

- 1_invariante: TP=2248, FP=475, FN=752, TN=2525; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5225; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.864, PR-AUC=0.879; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.795, Prec=0.826, Rec=0.749, F1=0.786; AUC=0.864 corrobora excelência

**Fold 3** — Passou: Sim

- 1_invariante: TP=2220, FP=419, FN=780, TN=2581; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5225; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.866, PR-AUC=0.883; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.800, Prec=0.841, Rec=0.740, F1=0.787; AUC=0.866 corrobora excelência

**Fold 4** — Passou: Sim

- 1_invariante: TP=2274, FP=451, FN=726, TN=2549; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.5375; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.872, PR-AUC=0.887; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.804, Prec=0.834, Rec=0.758, F1=0.794; AUC=0.872 corrobora excelência

**Fold 5** — Passou: Sim

- 1_invariante: TP=2319, FP=541, FN=681, TN=2459; prevalência=0.500; plausível=sim
- 2_trivial: Sempre positivo: Acc=0.500, Rec=1.000; Maioria: Acc=0.500; Degenerado=não
- 3_threshold: threshold=0.4850; extremo=não
- 4_padrao_degenerado: Acc≈Prec com Rec alto: não
- 5_coerencia_auc: ROC-AUC=0.866, PR-AUC=0.874; inflação suspeita=não
- 6_sustentacao: Métricas: Acc=0.796, Prec=0.811, Rec=0.773, F1=0.791; AUC=0.866 corrobora excelência


## Tabela por Modelo

| Modelo | Threshold (média±dp) | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | TN médio | Passou? |
|--------|----------------------|-----|------|--------|----|---------|--------|----------|---------|
| Regressão Logística | 0.508±0.012 | 0.660 | 0.734 | 0.503 | 0.596 | 0.699 | 0.696 | 2450 | S |
| Árvore de Decisão | 0.552±0.063 | 0.800 | 0.824 | 0.768 | 0.793 | 0.869 | 0.861 | 2497 | S |
| SVM | 0.374±0.069 | 0.796 | 0.811 | 0.773 | 0.791 | 0.822 | 0.810 | 2457 | S |
| Naive Bayes | 0.500±0.000 | 0.519 | 0.841 | 0.047 | 0.089 | 0.593 | 0.597 | 2973 | S |
| Rede Neural | 0.523±0.023 | 0.800 | 0.831 | 0.754 | 0.790 | 0.869 | 0.883 | 2538 | S |
| Sempre Positivo | 0.500±0.000 | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 | 0.500 | 0 | N |
| Sempre Majoritário | 0.500±0.000 | 0.500 | 0.500 | 1.000 | 0.667 | 0.500 | 0.500 | 0 | N |

## Ranking Revisado (pós-correção de threshold)

| Rank | Modelo | Score | Recall | F1 | Precision | Accuracy | ROC-AUC |
|------|--------|-------|--------|----|-----------|----------|---------|
| 1 | Árvore de Decisão | 0.7875 | 0.768 | 0.793 | 0.824 | 0.800 | 0.869 |
| 2 | SVM | 0.7866 | 0.773 | 0.791 | 0.811 | 0.796 | 0.822 |
| 3 | Rede Neural | 0.7807 | 0.754 | 0.790 | 0.831 | 0.800 | 0.869 |
| 4 | Sempre Positivo | 0.7750 | 1.000 | 0.667 | 0.500 | 0.500 | 0.500 |
| 5 | Sempre Majoritário | 0.7750 | 1.000 | 0.667 | 0.500 | 0.500 | 0.500 |
| 6 | Regressão Logística | 0.5815 | 0.503 | 0.596 | 0.734 | 0.660 | 0.699 |
| 7 | Naive Bayes | 0.2256 | 0.047 | 0.089 | 0.841 | 0.519 | 0.593 |

## Significância Estatística

Wilcoxon pareado entre Árvore de Decisão e SVM em Recall: statistic=5.0000, p-value=0.6250. Sem diferença significativa (p>=0.05)

## Conclusão Metodológica

O ranking anterior que favorecia Regressão Logística com Recall≈1.0 refletia threshold patológico (maximização cega de F1), não superioridade preditiva. Com Youden's J restrito e guarda-corpo de prevalência, os resultados devem ser interpretados à luz do ROC-AUC/PR-AUC e da coluna TN médio.