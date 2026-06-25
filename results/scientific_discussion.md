# Discussão Científica Crítica

## Sobre o dataset

- **Adequação:** Microdados INEP 2024 são representativos do censo de cursos superiores.
- **Desbalanceamento:** Amostra auditada 50/50 (15k/classe); SMOTE apenas no Naive Bayes.
- **Limitações:** Alvo binário por taxa ≥20%; não captura evasão longitudinal individual; possível vazamento mitigado por exclusão de colunas de situação no pipeline original.

## Sobre os modelos

| Critério | Melhor opção | Evidência |
|----------|--------------|-----------|
| Equilíbrio geral (pesos iguais) | Árvore de Decisão | Score=0.8111 |
| Interpretabilidade | Árvore de Decisão | Regras exportáveis, feature importance |
| Robustez entre folds | SVM / Árvore | Wilcoxon p=0.625 — empate estatístico |
| PR-AUC | Rede Neural | 0.883 no protocolo auditado |

## Sobre as métricas

- **Accuracy sozinha:** Insuficiente em 50/50; baselines triviais atingem 50%.
- **ROC-AUC:** Altera conclusão — separa discriminação de threshold; LR cai de 'líder' a mediano.
- **Precision vs Recall:** Naive Bayes (Prec=0.84, Rec=0.05) vs Árvore (Prec=0.82, Rec=0.77) contam histórias opostas — foco em evasão exige Recall.

## Conclusão da validação

Métricas oficiais: `validated_metrics.csv` derivado de `*_folds.csv` auditados. Divergências com PDFs pré-auditoria documentadas em `inconsistencies_report.md`.
