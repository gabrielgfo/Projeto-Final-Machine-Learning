\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{url}
\usepackage{lmodern}
\renewcommand{\rmdefault}{lmr}
\renewcommand{\sfdefault}{lmss}
\renewcommand{\ttdefault}{lmtt}

\hyphenation{evasão acadêmica micro-dados}

\begin{document}

\title{Comparação Auditada de Modelos de Aprendizado de Máquina para Predição de Alto Risco de Evasão em Cursos do Ensino Superior Brasileiro}

\author{
\IEEEauthorblockN{João Pedro, Luiz Miguel, Victor Conde, Gabriel Veloso, and Gabriel Albertan}
\IEEEauthorblockA{Centro de Informática --- Universidade Federal de Pernambuco (CIn-UFPE)}
}

\maketitle

\begin{abstract}
@@ABSTRACT@@
\end{abstract}

\begin{IEEEkeywords}
Evasão acadêmica, aprendizado de máquina, INEP, classificação binária, auditoria metodológica, árvores de decisão.
\end{IEEEkeywords}

%==============================================================================
\section{Introdução}
%==============================================================================

\subsection{Contexto}
A evasão acadêmica designa a interrupção prematura da trajetória formativa do estudante antes da conclusão do curso, seja por desligamento, abandono ou trancamento prolongado. No contexto do ensino superior brasileiro, o fenômeno compromete a eficiência institucional, reduz a taxa de formação de capital humano qualificado e impacta indicadores de avaliação de políticas públicas, incluindo o Sistema Nacional de Avaliação da Educação Superior (Sinaes) e os instrumentos de regulação conduzidos pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP).

Do ponto de vista econômico, a evasão implica subutilização de vagas, custos irrecuperáveis de infraestrutura e perda de retorno sobre investimentos públicos e privados em educação. Socialmente, a não conclusão do curso pode restringir mobilidade ocupacional, perpetuar desigualdades e afetar a autopercepção de sucesso educacional dos estudantes. Por essas razões, a evasão deixou de ser tratada apenas como problema pedagógico individual e passou a ser compreendida como desafio estratégico de gestão institucional, demandando instrumentos de monitoramento, alerta precoce e alocação de recursos de permanência.

\subsection{Motivação}
A predição de risco de evasão em nível agregado (curso ou cohort) permite que gestores identifiquem contextos vulneráveis antes que o processo de abandono se consolide. Modelos preditivos baseados em dados censitários podem apoiar a priorização de programas de apoio psicopedagógico, revisão de políticas de ingresso, ajustes de oferta (turno, modalidade, financiamento) e avaliação de impacto de intervenções. Contudo, a utilidade prática desses modelos depende não apenas da acurácia global, mas da capacidade de detectar verdadeiros casos de alto risco (Recall) sem gerar alarmes espúrios em massa (controle de falsos positivos e de degenerescência do limiar).

\subsection{Dados educacionais e mineração de dados}
Os microdados do Censo da Educação Superior disponibilizados pelo INEP constituem fonte pública de elevado valor analítico, por reunirem informações demográficas, estruturais e de fluxo acadêmico em escala nacional. A mineração de dados educacionais (Educational Data Mining --- EDM) e a aprendizagem analítica em larga escala (Learning Analytics) têm explorado tais bases para modelagem de desempenho, evasão e permanência \cite{romero2020}. A aplicação de aprendizado de máquina a políticas educacionais exige, entretanto, rigor metodológico na definição do alvo, no pré-processamento sem vazamento de informação e na avaliação com métricas adequadas ao custo assimétrico dos erros.

\subsection{Objetivos e hipótese}
\textbf{Objetivo geral:} comparar, sob protocolo auditado e reprodutível, o desempenho de cinco famílias de classificadores na identificação de cursos com alto risco de evasão a partir dos microdados INEP 2024.

\textbf{Objetivos específicos:}
\begin{enumerate}
    \item implementar pipeline unificado de pré-processamento e validação cruzada estratificada;
    \item auditar a seleção de limiar de decisão e detectar métricas artificialmente infladas;
    \item ranquear modelos com base em métricas validadas e baselines triviais;
    \item analisar interpretabilidade e aplicabilidade da Árvore de Decisão como estudo de caso.
\end{enumerate}

\textbf{Hipótese implícita:} modelos com maior capacidade discriminativa (ROC-AUC) e Recall sustentado por especificidade não degenerada superarão baselines triviais e serão preferíveis a classificadores cuja aparente superioridade dependa exclusivamente de limiares extremos.

%==============================================================================
\section{Fundamentação Teórica}
%==============================================================================

\subsection{Regressão Logística}
A Regressão Logística modela a probabilidade de pertencimento à classe positiva mediante função logística sobre combinação linear das features \cite{hosmer2013}. É interpretável via odds ratios e computacionalmente eficiente. Suas limitações incluem a hipótese de separabilidade aproximadamente linear no espaço de atributos e sensibilidade a multicolinearidade e outliers. Em problemas com fronteiras não lineares e interações entre variáveis socioeconômicas, sua capacidade expressiva pode ser insuficiente, o que deve refletir-se em ROC-AUC moderado, independentemente do Recall observado em um limiar específico.

\subsection{Árvores de Decisão}
Árvores de decisão realizam particionamento recursivo do espaço de atributos, maximizando ganho de informação (entropia ou índice de Gini) em cada nó \cite{breiman1984}. Oferecem interpretabilidade direta por regras SE-ENTÃO e toleram features heterogêneas após codificação. O principal risco é o \textit{overfitting}, mitigado por poda (\textit{ccp\_alpha}), restrições de profundidade e tamanho mínimo de folha. Em contextos de política educacional, a transparência das regras é vantagem operacional relevante.

\subsection{Máquinas de Vetores de Suporte (SVM)}
A SVM busca o hiperplano de separação que maximiza a margem entre classes, podendo empregar kernels (e.g., RBF) para fronteiras não lineares \cite{cortes1995}. São reconhecidas pela robustez em espaços de alta dimensão e boa generalização quando hiperparâmetros são adequadamente calibrados. A interpretabilidade é inferior à das árvores, e o custo computacional cresce com o tamanho da amostra, embora ainda viável na escala deste estudo.

\subsection{Redes Neurais}
Perceptrons multicamadas (MLP) aproximam funções não lineares por composição de transformações afins e ativações \cite{haykin2009}. Possuem capacidade expressiva elevada, porém demandam tuning de arquitetura, regularização e inicialização. Em dados tabulares com amostra moderada, podem igualar ou superar modelos clássicos em métricas de ranking (ROC-AUC), com custo de opacidade decisória.

\subsection{Naive Bayes}
Classificadores Naive Bayes assumem independência condicional entre features dado a classe, estimando likelihoods com suavização de Laplace \cite{domingos2012}. O ComplementNB foi projetado para classes desbalanceadas e representações esparsas. Em dados tabulares com forte correlação entre atributos derivados de contagens INEP, a violação da independência pode comprometer a estimativa posterior, especialmente na classe positiva.

\subsection{Métricas de avaliação}
Seja $TP$, $FP$, $FN$, $TN$ os elementos da matriz de confusão. Accuracy mede a fração de acertos globais, mas é insuficiente em cenários balanceados artificialmente ou quando classes têm custos assimétricos. Precision ($TP/(TP+FP)$) quantifica a confiabilidade dos alertas; Recall ($TP/(TP+FN)$) quantifica a cobertura dos casos positivos. O F1-score harmoniza Precision e Recall, mas permanece dependente do limiar de decisão. A ROC-AUC resume a capacidade de discriminação ao variar o limiar, enquanto a PR-AUC é informativa em prevalências desafiadoras \cite{fawcett2006}. Neste estudo, a evasão é tratada como evento crítico a ser detectado; portanto, Recall e ROC-AUC possuem peso analítico superior à Accuracy isolada.

%==============================================================================
\section{Metodologia}
%==============================================================================

\subsection{Dataset e definição do alvo}
Utilizaram-se microdados de cadastro de cursos do Censo da Educação Superior 2024 (\texttt{MICRODADOS\_CADASTRO\_CURSOS\_2024.CSV}), com separador ponto-e-vírgula e codificação Latin-1. A variável resposta \texttt{ALTO\_RISCO\_EVASAO} foi definida como indicador binário de taxa de evasão $\geq 20\%$, calculada a partir de desligamentos e trancamentos sobre ingressantes ($QT\_ING > 0$). Foram derivadas nove variáveis de proporção (e.g., taxa de conclusão, proporção EAD, índices de financiamento) e selecionadas oito atributos categóricos institucionais codificados por \textit{Label Encoding}.

\subsection{Amostragem}
Extraiu-se amostra estratificada de $N = 30\,000$ cursos ($15\,000$ por classe), fixando prevalência em $50\%$ para evitar que métricas fossem dominadas por desbalanceamento extremo e para tornar explícita a comparação com baselines triviais (Accuracy $= 0{,}5$, Recall $= 1{,}0$ para classificador ``sempre positivo''). A amostragem estratificada preserva representatividade de ambas as classes e estabiliza a variância das métricas entre folds \cite{kohavi1995}, ao custo de não refletir a distribuição natural populacional---limitação explicitada na Seção~\ref{sec:ameacas}.

\subsection{Pré-processamento}
O pipeline por fold compreende: (i) imputação de ausentes pela moda, ajustada no treino; (ii) codificação ordinal de categorias no treino com mapeamento de categorias não vistas para valor reservado; (iii) seleção de features com $|\rho| \geq 0{,}05$ em relação ao alvo; (iv) padronização (\texttt{StandardScaler}), exceto ComplementNB (\texttt{MinMaxScaler}). O ajuste exclusivo no treino de cada fold previne vazamento de informação. A seleção por correlação reduz dimensionalidade e ruído, mas pode omitir variáveis preditivas não lineares; trata-se de compromisso interpretável entre parcimônia e completude.

\subsection{Modelos e hiperparâmetros}
Hiperparâmetros fixos provenientes dos experimentos originais e consolidados em \texttt{analysis/audit/evaluator.py}: Regressão Logística ($L_1$, $C=10$), Árvore de Decisão (entropia, \textit{max\_depth}=15, \textit{ccp\_alpha}$\approx 0{,}00435$, \textit{class\_weight} assimétrico), SVM (RBF, $C \approx 3{,}15$, $\gamma \approx 5{,}67$), MLP (64 neurônios, ReLU, SGD) e ComplementNB ($\alpha \approx 0{,}161$, \textit{norm}=True com SMOTE apenas no treino). Não houve reexecução de \textit{grid search} na validação final.

\subsection{Validação cruzada e calibração de limiar}
Adotou-se Stratified K-Fold com $K=5$ e \textit{shuffle} ($seed=42$). Em cada fold, $20\%$ do treino foi reservado para calibração do limiar, evitando que o conjunto de teste influenciasse a escolha do ponto de corte---prática alinhada à literatura de avaliação honesta \cite{kohavi1995}.

\subsubsection{Problema dos limiares degenerados}
A função original \texttt{encontrar\_melhor\_threshold} maximizava F1 no intervalo $[0{,}01; 0{,}99]$. Em classificadores com scores mal calibrados ou com massa probabilística concentrada em valores baixos, o ótimo de F1 frequentemente ocorre em limiares $\rightarrow 0$, classificando quase todas as instâncias como positivas. Nesse regime, Recall $\rightarrow 1$, mas $TN \rightarrow 0$ e Accuracy $\rightarrow$ prevalência da classe positiva. Para amostra balanceada ($50\%$), obtém-se Accuracy $\approx 0{,}5$ com aparente ``excelência'' em Recall---indistinguível do baseline trivial ``sempre positivo''. Esse fenômeno explica o Recall @@LR_PRE_RECALL_TEX@@ da Regressão Logística pré-auditoria: não refletia poder discriminativo, mas colapso do limiar.

\subsubsection{Critério de Youden restrito}
A auditoria substituiu a maximização cega de F1 pelo índice de Youden $J = \text{sensibilidade} + \text{especificidade} - 1$, que busca explicitamente equilíbrio entre $TP$ e $TN$. O intervalo $[0{,}2; 0{,}8]$ foi adotado porque: (i) limiares $< 0{,}2$ continuavam a produzir taxas de positivos previstos incompatíveis com prevalência calibrada; (ii) limiares $> 0{,}8$ geravam Recall insuficiente para o problema de detecção de risco; (iii) o intervalo central força soluções operacionalmente plausíveis. Critérios adicionais rejeitam limiares cuja taxa de positivos previstos exceda $1{,}5\times$ a prevalência no conjunto de calibração, com \textit{fallback} em $0{,}5$ quando nenhum limiar satisfaz as restrições.

\subsection{Protocolo de sanidade}
Por fold, executaram-se seis verificações: (1) consistência da matriz de confusão; (2) comparação com classificadores triviais; (3) plausibilidade do limiar; (4) detecção do padrão Accuracy $\approx$ Precision com Recall elevado; (5) coerência entre ROC-AUC/PR-AUC e métricas no limiar; (6) sustentação para conclusões de excelência. Baselines ``Sempre Positivo'' e ``Sempre Majoritário'' foram incluídos no ranking (Score composto $= 0{,}775$).

%==============================================================================
\section{Resultados}
%==============================================================================

@@RESULTS_SECTION@@

%==============================================================================
\section{Análise Comparativa dos Modelos}
%==============================================================================

@@COMPARATIVE_SECTION@@

%==============================================================================
\section{Discussão}
%==============================================================================

\subsection{Sobre as métricas}
Accuracy isolada não conta toda a história: com prevalência $50\%$, um classificador trivial pode atingir $0{,}5$ sem valor preditivo. Neste problema, \textbf{Recall é operacionalmente central}, pois representa a fração de cursos de alto risco efetivamente sinalizados; contudo, Recall sem TN (especificidade) é enganoso---como demonstrado pela Regressão Logística pré-auditoria e pelo Naive Bayes pós-auditoria. O ROC-AUC complementa a análise ao medir qualidade global da ordenação por score, independentemente de um único limiar; discrepâncias entre Recall alto e AUC moderado sinalizam calibração inadequada.

\subsection{Sobre interpretabilidade}
Em educação, decisões algorítmicas afetam alocação de bolsas, monitoramento de cursos e imagem institucional. Gestores educacionais frequentemente necessitam justificar ações a conselhos e órgãos reguladores; modelos interpretáveis reduzem assimetria de informação entre equipe técnica e decisores. A Árvore de Decisão atende parcialmente a esse requisito, embora variáveis codificadas exijam documentação auxiliar.

\subsection{Sobre aplicabilidade}
Em cenário real com foco em detecção precoce e transparência, recomenda-se Árvore de Decisão ou SVM, priorizando a primeira quando a explicabilidade for mandatória. A Rede Neural é reserva para análises secundárias de ranking. Regressão Logística e Naive Bayes não devem liderar implantação sem reformulação metodológica substancial.

\subsection{Sobre robustez}
Os resultados são consistentes entre folds (variação pequena de Recall e F1), e a recalibração de métricas a partir da matriz de confusão confirmou coerência numérica. Não há evidência de instabilidade grave entre partições; a equivalência estatística Árvore--SVM sugere que a escolha final pode basear-se em critérios de governança (interpretabilidade, custo) mais que em ganho preditivo marginal.

%==============================================================================
\section{Estudo de Caso: Árvore de Decisão}
\label{sec:arvore}
%==============================================================================

@@TREE_SECTION@@

%==============================================================================
\section{Ameaças à Validade}
\label{sec:ameacas}
%==============================================================================

\subsection{Validade interna}
A amostragem balanceada $50/50$ facilita comparação e detecção de degenerescência, mas não representa prevalência populacional real de alto risco---métricas absolutas de Precision/Recall em produção podem diferir. O pré-processamento por fold minimiza vazamento; a seleção por correlação pode excluir preditores não lineares não correlacionados marginalmente. A calibração de limiar em subconjunto do treino introduz variância adicional, embora necessária para evitar otimismo no teste.

\subsection{Validade externa}
Os modelos foram estimados em snapshot de 2024; generalização para outros anos pressupõe estabilidade das relações estruturais e das definições INEP. Generalização para instituições específicas (micro em vez de agregado por curso) não foi avaliada. Resultados aplicam-se ao nível de agregação curricular do Censo.

\subsection{Limitações computacionais e metodológicas}
Não foram avaliados ensembles (Random Forest, XGBoost), validação temporal (\textit{train past, test future}) nem calibração probabilística pós-hoc (Platt, isotônica). O pipeline do Naive Bayes original (mediana, \textit{QT\_ING} $\geq 10$) difere do protocolo unificado, inviabilizando comparação histórica direta sem reexecução controlada.

%==============================================================================
\section{Conclusão}
%==============================================================================

@@CONCLUSION_SECTION@@

%==============================================================================
\section*{Agradecimentos}
%==============================================================================

Os autores agradecem ao INEP pela disponibilização dos microdados do Censo da Educação Superior e ao CIn-UFPE pelo suporte acadêmico.


\begin{thebibliography}{99}

\bibitem{breiman1984}
L. Breiman, J. Friedman, R. Olshen, and C. Stone,
\emph{Classification and Regression Trees}.
Belmont, CA, USA: Wadsworth, 1984.

\bibitem{cortes1995}
C. Cortes and V. Vapnik,
``Support-vector networks,''
\emph{Machine Learning}, vol. 20, no. 3, pp. 273--297, 1995.

\bibitem{domingos2012}
P. Domingos and G. Hulten,
``Mining high-speed data streams,''
in \emph{Proc. ACM SIGKDD}, 2000, pp. 71--80.

\bibitem{fawcett2006}
T. Fawcett,
``An introduction to ROC analysis,''
\emph{Pattern Recognition Letters}, vol. 27, no. 8, pp. 861--874, 2006.

\bibitem{haykin2009}
S. Haykin,
\emph{Neural Networks and Learning Machines}, 3rd ed.
Upper Saddle River, NJ, USA: Prentice Hall, 2009.

\bibitem{hosmer2013}
D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant,
\emph{Applied Logistic Regression}, 3rd ed.
Hoboken, NJ, USA: Wiley, 2013.

\bibitem{inep2024}
INEP,
\emph{Microdados do Censo da Educação Superior 2024}.
Brasília, DF, Brasil, 2024. [Online]. Available: \url{https://www.gov.br/inep}

\bibitem{kohavi1995}
R. Kohavi,
``A study of cross-validation and bootstrap for accuracy estimation and model selection,''
in \emph{Proc. IJCAI}, 1995, pp. 1137--1145.

\bibitem{pedregosa2011}
F. Pedregosa \emph{et al.},
``Scikit-learn: Machine learning in Python,''
\emph{Journal of Machine Learning Research}, vol. 12, pp. 2825--2830, 2011.

\bibitem{romero2020}
C. Romero and S. Ventura,
``Educational data mining and learning analytics: An updated survey,''
\emph{Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery}, vol. 10, no. 3, e1355, 2020.

\bibitem{tinto2010}
V. Tinto,
\emph{Leaving College: Rethinking the Causes and Cures of Student Attrition}, 2nd ed.
Chicago, IL, USA: Univ. Chicago Press, 2010.

\end{thebibliography}

@@TREE_APPENDIX@@

\end{document}
