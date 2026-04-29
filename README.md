# Projeto Mineracao de Dados

## Grupo 5 -- Interpretabilidade de Modelos

Repositorio oficial do trabalho final da disciplina de Mineracao de Dados, Departamento de Estatistica, Universidade Federal de Sao Carlos (UFSCar), ministrada pelo Prof. Dr. Helton Graziadei.

---

## Integrantes

| Nome | RA |
|---|---|
| Arthur Pereira Gon | 811464 |
| Giovanni Roberti | 811676 |
| Igor Figueiredo | 791420 |
| Lucas Yukio | 813650 |
| Murilo Cassiavilani | 812067 |

---

## Tema e Modalidade

**Tema sorteado:** 5 -- Interpretabilidade de Modelos.

**Modalidade:** A -- Projeto aplicado de mineracao de dados.

---

## Visao Geral do Projeto

O objetivo deste trabalho e realizar um estudo abrangente de interpretabilidade de modelos preditivos aplicado aos principais paradigmas de modelagem abordados na disciplina. Em vez de restringir a analise a um unico tipo de problema, o grupo desenvolveu aplicacoes em cada area sorteada aos demais grupos (regressao, classificacao, agrupamento, mineracao de textos e deteccao de anomalias).

Em cada paradigma, foram ajustados modelos representativos e aplicadas tecnicas especificas de interpretabilidade para comparar resultados, explicar comportamentos e responder a pergunta central: por que o modelo prediz o que prediz?

---

## Estrutura do Repositorio

```
Projeto Mineracao de Dados /
|
├── compile/                          # PDF final da proposta (overleaf/main.pdf)
│
├── conteudo materia/                 # Material didatico da disciplina
│   ├── AulaR.pdf                     # Introducao ao tidyverse e tidymodels
│   ├── Avaliacao_Classificadores_Logistica.pdf
│   ├── Fund_Aprendizagem_Supervisionada.pdf
│   ├── Metodos_Nao_Param_Reg.pdf
│   ├── Mineracao_Dados-Lista1.pdf    # Lista de exercicios I
│   ├── Mineracao_Dados-Lista2.pdf    # Lista de exercicios II
│   ├── Mineracao_Dados-Lista3.pdf    # Lista de exercicios III
│   ├── Regressao_Linear.pdf
│   ├── Selecao de Variaveis e Regularizacao.pdf
│   ├── Seminario.pdf                 # Instrucoes para o trabalho final
│   └── Seminario (1).pdf
│
├── overleaf/                         # Projeto LaTeX (proposta inicial)
│   ├── images/                       # Pasta para figuras
│   ├── main.tex                      # Codigo-fonte LaTeX (proposta antiga)
│   ├── main.pdf                      # PDF compilado da proposta
│   ├── Projeto_Grupo5_Overleaf.zip   # Arquivo compactado para upload no Overleaf
│   └── references.bib                # Bibliografia
│
|   NOTA: A pasta overleaf contem a escrita da PROPOSTA INICIAL enviada ao
|   professor. Esta proposta deve ser ATUALIZADA para refletir os modelos
|   especificos e fontes de dados utilizados em cada paradigma.
|
├── Regressao/                        # Paradigma 1: Regressao
│   ├── dataset/
│   │   ├── nba_salaries.csv
│   │   └── nba_2022-23_all_stats_with_salary.csv
│   ├── ADED/                         # Analise Descritiva Exploratoria
│   ├── PROCESSAMENTO/                # Pre-processamento e divisao treino/teste
│   ├── modelagem/                    # Modelos ajustados e resultados
│   ├── interpretabilidade/           # Tecnicas de interpretabilidade
│   ├── interpretacao.md              # Discussao e conclusoes
│   └── pipeline_regressao.py         # Script reprodutivel
│
├── classificacao/                    # Paradigma 2: Classificacao
│   ├── dataset/
│   │   ├── results.csv
│   │   └── stats.csv
│   ├── ADED/
│   ├── PROCESSAMENTO/
│   ├── modelagem/
│   ├── interpretabilidade/
│   ├── interpretacao.md
│   └── pipeline_classificacao.py
│
├── agrupamento/                      # Paradigma 3: Agrupamento
│   ├── dataset/
│   │   ├── players(man).csv
│   │   ├── players_tournament(man).csv
│   │   ├── raw_kaggle.csv
│   │   ├── return_kaggle.csv
│   │   └── serve_kaggle.csv
│   ├── ADED/
│   ├── PROCESSAMENTO/
│   ├── modelagem/
│   ├── interpretabilidade/
│   ├── interpretacao.md
│   └── pipeline_agrupamento.py
│
├── textos/                           # Paradigma 4: Mineracao de Textos (PENDENTE)
│
├── anomalias/                        # Paradigma 5: Deteccao de Anomalias (PENDENTE)
│
└── README.md                         # Este arquivo
```

---

## Paradigmas Implementados

### 1. Regressao -- Predicao de Salarios NBA

**Dataset:** NBA Player Salaries (2022-23 Season) [Kaggle](https://www.kaggle.com/datasets/jamiewelsh2/nba-player-salaries-2022-23-season)
- 467 jogadores com estatisticas de performance e salarios

**Variavel resposta:** `Salary` (salario anual em USD, log-transformado)

**Modelos:**
- Regressao Linear Multipla (OLS)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

**Tecnicas de Interpretabilidade:**
- Coeficientes padronizados (OLS, Ridge, Lasso)
- Importancia por Permutacao (Random Forest)
- Partial Dependence Plots (PDP)

**Resultados:**
| Modelo | R2 Teste | MAE (USD) |
|---|---|---|
| OLS | 0.669 | $4.82M |
| Ridge | 0.627 | $4.68M |
| Lasso | 0.605 | $4.86M |
| **Random Forest** | **0.709** | **$4.04M** |

**Insights:** A disponibilidade (GP, MP) e mais importante que eficiencia pura. A idade tem efeito nao-linear (pico aos 30 anos).

---

### 2. Classificacao -- Vitoria do Mandante na Premier League

**Dataset:** Premier League Stats (2006-2018) [Kaggle](https://www.kaggle.com/datasets/zaeemnalla/premier-league)
- 4.560 partidas + estatisticas agregadas por time-temporada

**Variavel resposta:** `HomeWin` (1 = vitoria do mandante, 0 = empate ou derrota)

**Divisao temporal:** Treino 2006-2015, Teste 2016-2018

**Modelos:**
- Regressao Logistica com Lasso
- Random Forest Classifier

**Tecnicas de Interpretabilidade:**
- Coeficientes Lasso (selecao de variaveis)
- Importancia por Permutacao
- Partial Dependence Plots

**Resultados:**
| Modelo | AUC-ROC | Acuracia | F1 |
|---|---|---|---|
| **Logistica Lasso** | **0.767** | 0.687 | 0.628 |
| Random Forest | 0.751 | 0.689 | 0.624 |

**Insights:** Historico de vitorias/derrotas (forma do time) e o unico preditor realmente importante. Estatisticas de estilo de jogo nao acrescentam poder preditivo.

---

### 3. Agrupamento -- Perfis de Jogadores de Tenis

**Dataset:** Tennis Player Data (Serve, Return e Raw) [Kaggle](https://www.kaggle.com/datasets/mohammadkumail110/tennis-player-data-serve-return-and-raw)
- 462 jogadores ATP com estatisticas de saque e devolucao

**Algoritmos:**
- K-Means (K = 2, escolhido via Silhouette Score)
- Agrupamento Hierarquico (Ward)

**Features:** 12 metricas agregadas (saque + devolucao)

**Tecnicas de Interpretabilidade:**
- Centroides dos clusters (heatmap)
- Graficos radar por perfil
- Boxplots comparativos
- PCA para visualizacao
- Jogadores mais proximos do centroide

**Resultados:**
| Cluster | Perfil | Tamanho | Jogadores representativos |
|---|---|---|---|
| 0 | **Saqueadores / Agressivos** | 42 | Kokkinakis, Dimitrov, Hurkacz |
| 1 | **Devolvedores / Baseline** | 56 | Carreno Busta, Musetti, De Minaur |

**Insights:** No circuito ATP atual, jogadores se especializam em dois polos. Jogadores "completos" sao raros.

---

## Paradigmas Pendentes

### 4. Mineracao de Textos
- Status: **NAO INICIADO**
- Dataset: a definir
- Tarefa: Classificacao de sentimento, extracao de topicos (LDA) ou agrupamento de documentos
- Interpretabilidade: Inspecao de palavras por topico, analise de contribuicoes de termos

### 5. Deteccao de Anomalias
- Status: **NAO INICIADO**
- Dataset: a definir
- Tarefa: Identificar comportamentos atipicos em dados reais
- Interpretabilidade: Isolation Forest, LOF, explicacoes locais

---

## Sobre a Pasta Overleaf

A pasta `overleaf/` contem a **proposta inicial** do projeto em formato LaTeX:
- `main.tex` -- Codigo-fonte da proposta
- `main.pdf` -- PDF compilado
- `Projeto_Grupo5_Overleaf.zip` -- Arquivo compactado para upload no Overleaf

**IMPORTANTE:** A proposta inicial deve ser ATUALIZADA para o relatorio final. A nova versao deve:
1. Especificar cada modelo e dataset utilizado (NBA, Premier League, Tenis, etc.)
2. Incluir referencias bibliograficas para cada dataset e metodo
3. Adicionar secoes de resultados e interpretabilidade de cada paradigma
4. Manter o limite de 8 paginas conforme instrucoes do seminario

A pasta `compile/` contem o PDF final da proposta para referencia.

---

## Referencias

### Datasets
1. Welsh, J. NBA Player Salaries (2022-23 Season). Kaggle, 2023. Disponivel em: https://www.kaggle.com/datasets/jamiewelsh2/nba-player-salaries-2022-23-season
2. Kumail, M. Tennis Player Data (Serve, Return and Raw). Kaggle, 2024. Disponivel em: https://www.kaggle.com/datasets/mohammadkumail110/tennis-player-data-serve-return-and-raw
3. Zaeem, N. Premier League Stats. Kaggle, 2019. Disponivel em: https://www.kaggle.com/datasets/zaeemnalla/premier-league

### Metodos de Interpretabilidade
4. Tibshirani, R. Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society, 1996.
5. Breiman, L. Random Forests. Machine Learning, 2001.
6. Lundberg, S. M.; Lee, S. I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017. (SHAP)
7. Friedman, J. H. Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 2001.

### Disciplina
8. Graziadei, H. Mineracao de Dados. Departamento de Estatistica, UFSCar, 2025.

---

## Cronograma e Entregas

| Etapa | Data Limite | Status |
|---|---|---|
| Formacao dos grupos | 14/04 | Concluido |
| Sorteio dos temas | 16/04 | Concluido |
| Envio da proposta | 30/04 | Concluido |
| Devolutiva e aprovacao | 08/05 | Pendente |
| Entrega do relatorio e codigo | 25/06 | Em desenvolvimento |
| Inicio dos seminarios | 02/07 | Pendente |

---

## Uso de Ferramentas de Inteligencia Artificial

O uso de assistentes generativos de IA e permitido como ferramenta de apoio ao aprendizado. Este projeto utilizou IA para:
- Depuracao de codigo Python
- Estruturacao de pipelines de modelagem
- Revisao textual da interpretacao.md

Os alunos sao integralmente responsaveis por qualquer erro tecnico, metodologico ou conceitual.

---

*Sao Carlos -- SP, 2025.*
