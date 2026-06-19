# Predição de risco de evasão no ensino superior

Projeto final da disciplina de Aprendizado de Máquina e Ciência de Dados do CIn-UFPE.

O projeto utiliza os microdados do **Censo da Educação Superior 2024**, disponibilizados pelo INEP, para classificar cursos de graduação quanto ao risco de evasão. São comparadas cinco famílias de modelos:

- Árvore de Decisão;
- Naive Bayes;
- Rede Neural;
- Regressão Logística;
- Support Vector Machine (SVM).

## Equipe

- Gabriel Albertin
- Gabriel Fonseca
- João Pedro
- Luiz Veloso
- Victor Conde

## Dados

O conjunto de dados utilizado é o arquivo:

```text
MICRODADOS_CADASTRO_CURSOS_2024.CSV
```

Download oficial:

[Microdados do Censo da Educação Superior 2024 — INEP](https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_2024.zip)

Os notebooks utilizam o caminho relativo `./MICRODADOS_CADASTRO_CURSOS_2024.CSV`. Por isso, atualmente o arquivo precisa estar na mesma pasta do notebook executado.

O dataset não é versionado no Git por possuir mais de 100 MB. Evite adicioná-lo aos commits.

## Definição do problema

A tarefa é uma classificação binária de cursos com alto ou baixo risco de evasão.

No pré-processamento compartilhado, a taxa de evasão é calculada pela razão entre alunos desvinculados e o total formado por matriculados e desvinculados. O alvo `alto_risco_evasao` identifica os registros cuja taxa é maior ou igual à mediana do conjunto filtrado.

Para reduzir vazamento de dados, variáveis diretamente relacionadas à construção do alvo, identificadores e colunas de matrícula ou conclusão são removidas das features utilizadas pelo pipeline compartilhado.

## Estrutura do repositório

```text
.
├── data/
│   └── preprocessamento.py
├── models/
│   ├── arvore_de_decisao/
│   │   └── ArvoreDecisao.ipynb
│   ├── naive_bayes/
│   │   ├── train.py
│   │   ├── preprocessing.py
│   │   ├── model_factory.py
│   │   ├── evaluate.py
│   │   ├── report_generator.py
│   │   ├── artifacts/
│   │   └── report/
│   ├── Redes_Neurais/
│   │   └── RedesNeurais.ipynb
│   ├── regrassao_logisticca/
│   │   └── RegressaoLogistica.ipynb
│   └── svm/
│       └── SVM.ipynb
├── requirements.txt
└── README.md
```

## Instalação

Recomenda-se Python 3.10 ou superior e o uso de um ambiente virtual.

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para executar os notebooks, instale também as dependências de visualização e o Jupyter:

```powershell
pip install jupyter matplotlib seaborn scipy
```

A Rede Neural requer ainda o TensorFlow:

```powershell
pip install tensorflow
```

## Execução

### Naive Bayes

O Naive Bayes possui um pipeline executável por linha de comando. A partir da raiz do projeto:

```powershell
python .\models\naive_bayes\train.py
```

O processo:

1. carrega e prepara os microdados;
2. compara GaussianNB, BernoulliNB e ComplementNB;
3. realiza validação cruzada estratificada com 5 folds;
4. aplica SMOTE somente aos dados de treino;
5. seleciona a configuração com melhor recall;
6. salva o modelo e as métricas;
7. gera o relatório em PDF.

### Demais modelos

Inicie o Jupyter a partir da raiz do repositório:

```powershell
jupyter notebook
```

Abra e execute o notebook desejado:

| Modelo | Notebook |
|---|---|
| Árvore de Decisão | `models/arvore_de_decisao/ArvoreDecisao.ipynb` |
| Rede Neural | `models/Redes_Neurais/RedesNeurais.ipynb` |
| Regressão Logística | `models/regrassao_logisticca/RegressaoLogistica.ipynb` |
| SVM | `models/svm/SVM.ipynb` |
| Naive Bayes | `models/naive_bayes` |

Esses notebooks carregam o dataset por caminho relativo. Portanto, antes da execução, copie `MICRODADOS_CADASTRO_CURSOS_2024.CSV` para a mesma pasta do notebook escolhido, conforme indicado na seção [Dados](#dados).

## Avaliação

Os experimentos utilizam validação cruzada estratificada com 5 folds. Dependendo do modelo, são produzidas as seguintes métricas:

- acurácia;
- precisão;
- recall;
- F1-score;
- ROC-AUC;
- matriz de confusão;
- média e desvio padrão entre os folds.

O recall recebe atenção especial por representar a capacidade de identificar cursos que realmente apresentam alto risco de evasão.

## Resultados e relatórios

O repositório contém resultados previamente gerados, incluindo:

- métricas agregadas e por fold em CSV;
- melhores hiperparâmetros;
- gráficos de desempenho;
- importância das features da Árvore de Decisão;
- modelo Naive Bayes serializado;
- relatórios individuais em PDF.

Cada resultado está armazenado na pasta do modelo correspondente em `models/`.

## Tecnologias

- Python;
- pandas e NumPy;
- scikit-learn;
- imbalanced-learn;
- TensorFlow/Keras;
- Matplotlib e Seaborn;
- ReportLab;
- Jupyter Notebook.

## Observação sobre reprodutibilidade

Os experimentos utilizam sementes aleatórias fixas sempre que suportado pelos modelos e métodos de validação. Os resultados ainda podem variar entre versões das bibliotecas, sistemas operacionais e implementações com paralelismo ou aceleração por hardware.
