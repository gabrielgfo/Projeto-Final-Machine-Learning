# Inventário do Projeto — Auditoria Científica Final

**Data:** 2026-06-24

## Estrutura de diretórios

| Diretório | Função |
|-----------|--------|
| `models/` | Notebooks e artefatos por modelo |
| `analysis/` | Pipelines comparativo e auditoria |
| `analysis/audit/` | Protocolo unificado 5-fold pós-correção |
| `analysis/final_validation/` | Validação científica final |
| `results/` | Rankings e relatórios pré-auditoria |
| `results/audit/` | Resultados auditados (oficiais) |
| `results/validation/` | Curvas ROC/PR e matrizes validadas |
| `slides/` | Apresentações HTML |

## Modelos avaliados

1. Regressão Logística (`models/regrassao_logisticca/`)
2. Árvore de Decisão (`models/arvore_de_decisao/`)
3. SVM (avaliado via `analysis/audit/evaluator.py`; notebook em `main`)
4. Rede Neural MLP (`models/Redes_Neurais/`)
5. Naive Bayes ComplementNB (`models/naive_bayes/` em branch `main`)

## Dataset

- Censo INEP 2024 — `MICRODADOS_CADASTRO_CURSOS_2024.CSV`
- Status dos caminhos: `{"/home/jo-o-pedro/Projeto-Final-Machine-Learning/MICRODADOS_CADASTRO_CURSOS_2024.CSV": false, "/home/jo-o-pedro/Projeto-Final-Machine-Learning/data/MICRODADOS_CADASTRO_CURSOS_2024.CSV": false, "/home/jo-o-pedro/Downloads/microdados_censo_da_educacao_superior_2024/dados/MICRODADOS_CADASTRO_CURSOS_2024.CSV": true}`
- Alvo auditado: `ALTO_RISCO_EVASAO` (TAXA_EVASAO ≥ 20%)
- Amostra: 30.000 estratificada 50/50

## Artefatos serializados (*.pkl)

- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.10.0_pickle_py27_np17.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.10.0_pickle_py33_np18.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.10.0_pickle_py34_np19.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.10.0_pickle_py35_np19.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.11.0_pickle_py36_np111.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.9.2_pickle_py27_np16.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.9.2_pickle_py27_np17.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.9.2_pickle_py33_np18.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.9.2_pickle_py34_np19.pkl`
- `.venv/lib/python3.14/site-packages/joblib/test/data/joblib_0.9.2_pickle_py35_np19.pkl`
- `.venv/lib/python3.14/site-packages/numpy/_core/tests/data/astype_copy.pkl`

> `best_model.pkl` (Naive Bayes) disponível em `main` e `cursor/naive-bayes-multi-variant-pipeline`.

## PDFs de relatório

- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/back.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/filesave.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/forward.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/hand.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/help.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/home.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/matplotlib.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/move.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/qt4_editor_options.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/subplots.pdf`
- `.venv/lib/python3.14/site-packages/matplotlib/mpl-data/images/zoom_to_rect.pdf`
- `models/Redes_Neurais/Redes_Neurais-1.pdf`
- `results/audit/final_report_audited.pdf`
- `results/final_report.pdf`

## CSVs/JSON em results/

- `results/audit/audited_comparison.csv`
- `results/audit/audited_ranking.csv`
- `results/audit/naive_bayes_folds.csv`
- `results/audit/rede_neural_folds.csv`
- `results/audit/regressão_logística_folds.csv`
- `results/audit/svm_folds.csv`
- `results/audit/árvore_de_decisão_folds.csv`
- `results/final_comparison.csv`
- `results/ranking.csv`
- `results/audit/naive_bayes_sanity.json`
- `results/audit/rede_neural_sanity.json`
- `results/audit/regressão_logística_sanity.json`
- `results/audit/svm_sanity.json`
- `results/audit/árvore_de_decisão_sanity.json`

## Notebooks

- `models/Redes_Neurais/RedesNeurais.ipynb`
- `models/arvore_de_decisao/ArvoreDecisao.ipynb`
- `models/regrassao_logisticca/RegressaoLogistica.ipynb`

## Protocolo de validação oficial

Métricas **oficiais** para entrega acadêmica: `results/audit/audited_comparison.csv`
(Youden's J, 6 checagens de sanidade, baselines triviais).
