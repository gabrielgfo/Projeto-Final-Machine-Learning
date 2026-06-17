# Relatório Detalhado de Experimentação: Redes Neurais

## Predição de Risco de Evasão em Cursos de Educação Superior
**Autor:** João Pedro Barbosa Aragão  
**Instituição:** Centro de Informática -- UFPE

---

## Resumo
Este documento detalha o processo de modelagem, treinamento e validação de Redes Neurais Artificiais (RNA) para o problema de classificação binária de risco de evasão. Abordamos os desafios inerentes ao desbalanceamento de classes, propondo uma solução baseada em *class weighting* e calibração dinâmica de limiares de decisão, validadas através de um protocolo rigoroso de *Stratified 5-Fold Cross-Validation* para assegurar a generalização do modelo e a ausência de *data leakage*.

---

## 1. Contextualização e Fundamentação
O fenômeno da evasão escolar no ensino superior possui uma natureza multivariada, onde a variável-alvo, `ALTO_RISCO_EVASAO`, foi construída com base na relação entre ingressantes e alunos desvinculados/trancados. A distribuição de classes (75,4% de Alto Risco) impõe um viés de indução que favorece classificadores que optam pela classe majoritária.

## 2. Metodologia e Arquitetura
Selecionamos uma arquitetura do tipo *Multilayer Perceptron* (MLP) com uma camada oculta. A escolha por essa estrutura baseia-se no teorema da aproximação universal para redes com uma camada oculta, que, para este dataset de 21 atributos, provou ser o ponto de equilíbrio entre capacidade de generalização e risco de *overfitting*.

### Protocolo de Validação Cruzada
Para cada *fold*, o conjunto de treino foi subdividido internamente. O ajuste do limiar de decisão ($t \in [0.1, 0.85]$) foi realizado exclusivamente nos dados de calibração, garantindo que o *fold* de validação permanecesse como um conjunto de teste "invisível" ao processo de otimização de hiperparâmetros.

## 3. Resultados Estatísticos
A tabela abaixo consolida o desempenho do modelo (MLP 1 camada + SGD). Os desvios-padrão observados indicam uma estabilidade aceitável do modelo entre as diferentes partições dos dados.

| Métrica | Média | Desvio-Padrão |
| :--- | :---: | :---: |
| **Acurácia** | 0,8105 | 0,0283 |
| **F1-Score** | 0,8820 | 0,0112 |
| **Precisão** | 0,8383 | 0,0425 |
| **Recall** | 0,9342 | 0,0337 |

## 4. Discussão Técnica
* **Análise do Recall:** O alto valor de *recall* (0,9342) sugere que o modelo é extremamente eficaz na identificação de casos de risco, o que é estratégico para instituições que priorizam a redução de falsos negativos em programas de retenção estudantil.
* **Estabilidade entre Folds:** A baixa variação no F1-Score (1,12%) confirma que a estratégia de calibração de limiar sem *leakage* foi bem-sucedida em estabilizar o desempenho do modelo em diferentes estratos do dataset.
* **Limiar Ótimo:** A convergência da média do limiar para 0,20 reflete a necessidade de ajustar o ponto de corte do modelo para compensar a prevalência da classe minoritária.

## 5. Conclusão
O modelo de rede neural, aliado às técnicas de reponderação de gradientes e calibração de limiar, estabeleceu-se como um classificador robusto. Este experimento reforça que, para datasets desbalanceados, a arquitetura da rede é apenas uma variável; o protocolo de validação e o ajuste do limiar de decisão são os verdadeiros catalisadores do ganho de performance preditiva.
