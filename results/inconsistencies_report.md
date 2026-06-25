# Relatório de Inconsistências Metodológicas

## Resposta à pergunta de auditoria

**As métricas apresentadas nos PDFs pré-auditoria NÃO são consistentes** com os arquivos auditados em `results/audit/`. Os PDFs refletem threshold patológico; os CSVs auditados são a fonte oficial.

## Divergências documentadas

### 1. Regressão Logística — Recall inflado

| Fonte | Recall | Score | Rank |
|-------|--------|-------|------|
| `ranking.csv` (pré) | 0.9991 | 0.8960 | 1º |
| `audited_ranking.csv` (pós) | 0.5033 | 0.5815 | 6º |

- **Impacto:** Conclusão de 'melhor modelo' inválida no relatório pré-auditoria.
- **Causa:** Maximização cega de F1 com threshold ~0.01 (TN≈0).
- **Valor oficial:** Pós-auditoria.

### 2. Naive Bayes — pipelines distintos

| Fonte | Recall | ROC-AUC |
|-------|--------|---------|
| Pré-auditoria / pipeline original | 0.9731 | ~0.727 (branch main) |
| Pipeline unificado auditado | 0.0468 | 0.5927 |

- **Impacto:** Comparação injusta entre NB original (mediana, SMOTE) e demais modelos (alvo ≥20%).
- **Valor oficial:** Reavaliação unificada em `naive_bayes_folds.csv`.

### 3. Árvore de Decisão — mudança de posição no ranking

| Fonte | Rank | Recall | Precision |
|-------|------|--------|-----------|
| Pré-auditoria | 4º | 0.8201 | 0.9513 |
| Pós-auditoria | 1º | 0.7685 | 0.8240 |

- **Impacto:** Vencedor real só emerge após correção metodológica.
- **Valor oficial:** Pós-auditoria.

### 4. Rede Neural — notebook vs auditoria

| Métrica | Notebook | Auditado | Δ |
|---------|----------|----------|---|
| Recall | 0.9342 | 0.7536 | -0.1806 |
| F1 | 0.8820 | 0.7901 | -0.0918 |

- **Impacto:** Moderado; mesmo modelo, protocolo de threshold diferente.
- **Valor oficial:** Auditado.

### 5. Artefatos ausentes no working tree

- CSVs de LR/Árvore/SVM não exportados no disco (só em `main`).
- Nenhum `.pkl` carregável exceto via branch `main` (Naive Bayes).
- **Mitigação:** Validação recalculada a partir de `*_folds.csv` auditados.
