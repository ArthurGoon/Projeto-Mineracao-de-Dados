# Relatorio Central de Auditoria de Qualidade de Dados
## Projeto Mineracao de Dados - Grupo 5 (Interpretabilidade de Modelos)

**Data da auditoria:** 28/05/2026
**Paradigmas auditados:** Regressao (NBA), Classificacao (Premier League), Agrupamento (Tenis ATP)

---

## Resumo Executivo

| Paradigma | Status Atual | Problema Critico #1 | Problema Critico #2 | Impacto na Predicao |
|---|---|---|---|---|
| Regressao NBA | R2 = 0.71 (RF) | **10 outliers salariais < $100k** distorcem o target | **VIF catastrofico** (TS%=419, FG%=259) | MAPE > 100%; coeficientes OLS instaveis |
| Classificacao PL | AUC = 0.77 (Lasso) | **Leakage temporal** (stats acumuladas da temporada inteira) | **1.520 linhas perdidas** (33%) no dropna silencioso | AUC superestimado; Lasso zerou 36/39 features |
| Agrupamento Tenis | Silhouette = 0.24 | **~73% registros com valores missing-like** (`-`, `nan` string) | **Nao separa por superficie** (Hard/Clay/Grass) | Clusters fracos; sem profiling estatistico formal |

---

## 1. Regressao - Predicao de Salarios NBA

### 1.1 Visao Geral do Dataset
- **Dimensao:** 467 jogadores x 51 variaveis
- **Target:** Salary (USD anual), log-transformado no pipeline
- **Duplicatas:** 0

### 1.2 Problemas Identificados

#### A. OUTLIERS NO TARGET (SEVERIDADE: ALTA)
| Problema | Quantidade | Detalhe |
|---|---|---|
| Salario < $100k | 10 jogadores | Minimo: $5.849 (RaiQuan Gray, Jacob Gilyard) |
| Salario > $40M | 11 jogadores | Maximo: $48.07M (Stephen Curry) |
| Outliers IQR | 48 | Limite superior: $23.9M |
| Outliers Z>3 | 11 | |

**Justificativa do problema:** Os 10 jogadores com salario < $100k sao contratos two-way/G-League (Wigginton $99k, Schakel $96k, Umude $58k, etc.). Eles nao representam o mercado salarial NBA padrao. O MAPE > 100% e impulsionado por esses extremos.

**Recomendacao:** Investigacao individual. Se forem contratos two-way, documentar e aplicar winsorizacao no percentil 1% inferior. Se forem erros de entrada, corrigir.

#### B. MISSING VALUES (SEVERIDADE: MEDIA)
| Coluna | Missing | % | Justificativa |
|---|---|---|---|
| FT% | 23 | 4.9% | Jogadores que nunca foram a linha de lance livre |
| 3P% | 13 | 2.8% | Pivos que nunca arremessaram de 3 |
| 2P% | 4 | 0.9% | |
| FG% | 1 | 0.2% | |

**Problema:** O pipeline usa `fillna(median)` global. Pivots (C) e alas (SF) tem padroes de arremesso completamente diferentes. Imputar a mediana global de 3P% (~0.32) em um pivot que nunca arremessou de 3 e vieso.

**Recomendacao:** Imputacao condicional por `Position_Clean`.
- C com missing 3P% -> imputar 0 (centers raramente arremessam de 3)
- PG com missing FT% -> imputar mediana dos PGs (guards tem FT% mais alto)

#### C. MULTICOLINEARIDADE CATASTROFICA (SEVERIDADE: CRITICA)
| Feature | VIF | Interpretacao |
|---|---|---|
| TS% | 419.3 | Variancia inflacionada 419x por causa da correlacao com FG%, FT%, 3P% |
| FG% | 259.1 | |
| MP | 149.7 | Minutos correlacionam com praticamente todas as stats de volume |
| PER | 107.1 | |
| USG% | 64.4 | |
| PTS | 59.6 | |
| FT% | 41.1 | |
| WS | 36.3 | |
| Age | 33.9 | |
| VORP | 23.5 | |
| GP | 15.9 | |
| TRB | 15.7 | |
| BPM | 15.3 | |
| 3P% | 11.6 | |
| STL | 11.0 | |

**15 de 17 features tem VIF > 10** (limiar de alta multicolinearidade). **16 de 17 tem VIF > 5**.

**Correlacoes criticas entre stats avancadas:**
- PER <-> BPM: r = 0.90
- WS <-> VORP: r = 0.89

**Impacto:** Os coeficientes OLS sao instaveis (variance inflacionada). PER apareceu negativo no pipeline (-0.72) — isso e um artefato da multicolinearidade, nao um insight real.

**Recomendacao:**
1. Remover uma de {PER, BPM} (manter BPM, que e mais interpretavel)
2. Remover uma de {WS, VORP} (manter VORP)
3. Remover TS% (derivado de FG%, FT%, 3P%)
4. Ou: aplicar PCA nas stats avancadas
5. Ridge e preferivel a OLS nesse cenario, mas o pipeline ja usa Ridge

#### D. INCONSISTENCIAS LOGICAS (SEVERIDADE: BAIXA)
- GS > GP: 0 casos (OK)
- Total Minutes != GP*MP por >10 min: 0 casos (OK)
- 0 GP mas stats > 0: 0 casos (OK)
- MP>0 mas 0 pontos/chutes: 1 caso (Alondes Williams: 1 jogo, 5 min, 0 pts, 0 chutes — possivel)

**Veredito:** Dados consistentes do ponto de vista logico.

#### E. DISTRIBUICAO POR POSICAO
| Posicao | N | Salario Medio | Salario Mediana |
|---|---|---|---|
| PG | 79 | $11.83M | $4.83M |
| PF | 86 | $8.89M | $4.36M |
| SF | 94 | $8.17M | $3.54M |
| C | 91 | $7.28M | $2.91M |
| SG | 117 | $6.85M | $3.10M |

**Insight:** PGs tem salario medio 72% maior que Centers. O pipeline atual usa OneHotEncoder(drop='first') com base em Position_Clean, mas nao explora interacoes posicao x stats.

---

## 2. Classificacao - Vitoria do Mandante Premier League

### 2.1 Visao Geral do Dataset
- **Results:** 4.560 partidas x 6 colunas (temporadas 2006-2007 a 2017-2018)
- **Stats:** 240 registros x 42 colunas (20 times x 12 temporadas)
- **Target:** HomeWin (1 = vitoria mandante, 0 = empate/derrota)
- **Taxa baseline:** 46.2% de vitorias do mandante

### 2.2 Problemas Identificados

#### A. LEAKAGE TEMPORAL (SEVERIDADE: CRITICA)
**Este e o problema mais grave do pipeline.**

As estatisticas em `stats.csv` sao **acumuladas ao longo de toda a temporada** (ex: Manchester City teve 32 vitorias na temporada 2017-2018). O pipeline mergea essas stats em CADA partida daquela temporada.

**Problema:** Quando o modelo preve uma partida da **rodada 8** (outubro), ele "ve" o desempenho do time ao longo de **38 jogos** — incluindo 30 jogos que ainda NAO aconteceram. O modelo aprende do futuro.

**Impacto:**
- AUC-ROC = 0.77 e superestimada
- Na vida real, no inicio da temporada, voce so conhece o desempenho ate aquela rodada
- A diferenca de forma (diff_wins) no inicio da temporada seria muito menor

**Recomendacao:**
1. **Ideal:** Reconstruir stats acumuladas ATE cada rodada (requer dados por rodada, nao disponiveis)
2. **Alternativa imediata:** Usar stats da TEMPORADA ANTERIOR como proxy de forma
3. **Feature temporal:** Incluir rodada da temporada como feature (inicio vs final tem dinamicas diferentes)

#### B. LINHAS PERDIDAS NO MERGE/DROPNA (SEVERIDADE: ALTA)
| Etapa | Linhas |
|---|---|
| Results original | 4.560 |
| Apos merge home stats | 4.560 |
| Apos merge away stats | 4.560 |
| Linhas com QUALQUER missing pos-merge | 1.520 |
| **Apos dropna (pipeline)** | **3.040** |
| **Linhas PERDIDAS** | **1.520 (33.3%)** |

**Causa:** Missing values em `stats.csv` propagam-se para o merge:
- `backward_pass`: 80 missing (33.3% das 240 stats)
- `big_chance_missed`: 80 missing (33.3%)
- `saves`, `head_clearance`, `total_through_ball`, `dispossessed`: 20 missing cada (8.3%)

**O pipeline faz dropna silenciosamente**, sem documentar que 33% das partidas foram descartadas.

**Recomendacao:**
1. Imputar missing em stats ANTES do merge (mediana por time ou KNN)
2. Ou: remover features com >30% missing (`backward_pass`, `big_chance_missed`)
3. Documentar quantas linhas foram perdidas e por que

#### C. FEATURE ENGINEERING INSUFICIENTE (SEVERIDADE: ALTA)
O pipeline atual usa apenas **features diferenciais** (`diff_wins`, `diff_losses`, etc.) com 39 features.

**Features criticas AUSENTES:**
| Feature | Justificativa |
|---|---|
| Forma recente (ultimos 5 jogos) | O preditor mais importante no futebol; momentum recente > acumulado da temporada |
| Posicao na tabela ate aquela rodada | Times na zona de rebaixamento tem dinamica diferente |
| Head-to-head historico | Alguns times tem vantagem psicologica (derbys, historico) |
| Desempenho em casa vs fora separado | O diferencial global esconde especialistas em casa |
| Rodada da temporada | Inicio (1-10), meio (11-28), final (29-38) tem dinamicas diferentes |
| Fase da temporada | Tres blocos com comportamentos distintos |

**Recomendacao:** Implementar pelo menos forma recente e posicao na tabela.

#### D. VALIDACAO INCORRETA (SEVERIDADE: MEDIA)
O pipeline usa **CV 5-fold aleatorio** em dados temporais. Isso viola a ordem temporal — o modelo pode ser treinado em dados de 2018 e testado em dados de 2006.

**Recomendacao:** Usar walk-forward validation ou time-series split.

#### E. FEATURE SELECTION MUITO AGRESSIVA (SEVERIDADE: MEDIA)
Lasso selecionou apenas **3 de 39 features** (diff_wins, diff_losses, diff_att_ibox_goal).

**Problema:** Pode estar descartando informacoes relevantes. Com C muito pequeno (lambda grande), Lasso e agressivo.

**Recomendacao:**
1. Testar diferentes valores de C (mais amplo)
2. Usar RFECV com Random Forest para selecao wrapper
3. Ou Elastic Net (L1 + L2) para balancear selecao e estabilidade

---

## 3. Agrupamento - Perfis de Jogadores de Tenis

### 3.1 Visao Geral do Dataset
- **Players:** 462 jogadores x 3 colunas
- **Serve:** 237.185 registros x 17 colunas
- **Return:** 237.196 registros x 17 colunas
- **Raw:** 237.205 registros x 17 colunas

### 3.2 Problemas Identificados

#### A. MISSING-LIKE MASSIVO (SEVERIDADE: ALTA)
| Dataset | Total Registros | Missing/NaN | `-` string | Validos | % Validos |
|---|---|---|---|---|---|
| serve A% | 237.185 | 86.819 | 16 | 150.350 | **63.4%** |
| serve 1st% | 237.185 | 86.819 | 18 | 150.348 | **63.4%** |
| serve 2nd% | 237.185 | 86.819 | 36 | 150.330 | **63.4%** |
| return TPW | 237.196 | 86.806 | 0 | 150.390 | **63.4%** |
| return RPW | 237.196 | 86.806 | 15 | 150.375 | **63.4%** |

**Causa:** Os valores "-" e strings "nan" representam partidas onde o jogador:
- Nao teve saque (WO — walkover)
- Foi eliminado precocemente (Q1, Q2 — partidas incompletas)
- Dados de duplas (nao aplicam percentuais de saque)

**O pipeline atual:** Usa `fillna(median)` global e filtra jogadores com >= 20 partidas.

**Problema:** 36.6% dos registros sao "missing-like". A agregacao por media dos validos ainda funciona, mas jogadores com muitos "-" tem estatisticas baseadas em poucos registros.

**Recomendacao:**
1. NAO imputar os "-" — eles sao informativos
2. Documentar confiabilidade: quanto mais "-", menos confiavel a media
3. Para jogadores com < 20 registros validos, considerar exclusao ou peso menor

#### B. NAO SEPARACAO POR SUPERFICIE (SEVERIDADE: ALTA)
| Superficie | Registros Serve | % |
|---|---|---|
| Hard | 117.554 | 49.6% |
| Clay | 107.647 | 45.4% |
| Grass | 9.479 | 4.0% |
| Carpet | 2.477 | 1.0% |

**Problema:** O pipeline agrega TODAS as superficies. Um jogador que e especialista em saque em grass (Wimbledon) e fraco em clay (Roland Garros) vai parecer "medio" na agregacao global.

**Recomendacao:**
1. Criar features de especializacao: `Aces_pct_hard`, `Aces_pct_clay`, `Aces_pct_grass`
2. Ou: agregar separadamente e criar ratios (especialista grass / especialista clay)
3. Ou: incluir superficie como feature no clustering

#### C. QUALIDADE DO CLUSTERING (SEVERIDADE: MEDIA)
| Metrica | Valor | Interpretacao |
|---|---|---|
| Silhouette (K=2) | 0.239 | Baixo — clusters se sobrepoe significativamente |
| Clusters encontrados | 2 | Possivelmente simplista |
| Jogadores analisados | 98 (apos filtro >=20) | 21% dos 462 originais |

**Problema:** Silhouette = 0.24 e baixo. Para contexto: > 0.5 e considerado bom, 0.25-0.5 e razoavel, < 0.25 e fraco.

**Recomendacao:**
1. Testar GMM (Gaussian Mixture Model) para soft clustering
2. Testar DBSCAN para detectar outliers/jogadores hibridos
3. Usar metricas adicionais: Calinski-Harabasz, Dunn Index
4. Bootstrap de estabilidade (subamostragem repetida)

#### D. INTERPRETABILIDADE INCOMPLETA (SEVERIDADE: MEDIA)
**Faltando no pipeline atual:**
- Profiling estatistico formal (testes t/ANOVA entre clusters, Cohen's d)
- Comparacao com ranking ATP medio entre clusters
- Identificacao de jogadores de fronteira (proximos a ambos os centroides)
- Visualizacao com t-SNE ou UMAP alem do PCA
- Evolucao temporal (jogadores que mudaram de cluster ao longo da carreira)

---

## 4. Prioridade de Acoes

### Prioridade 1 (CRITICA) — Impacto direto na performance
| # | Acao | Paradigma | Justificativa |
|---|---|---|---|
| 1.1 | Tratar leakage temporal das stats | Classificacao | Modelo "ve o futuro"; AUC superestimada |
| 1.2 | Remover/reduzir multicolinearidade | Regressao | VIF catastrofico; coeficientes OLS instaveis |
| 1.3 | Documentar e tratar 1.520 linhas perdidas | Classificacao | 33% dos dados descartados silenciosamente |

### Prioridade 2 (ALTA) — Melhoria substancial na predicao
| # | Acao | Paradigma | Justificativa |
|---|---|---|---|
| 2.1 | Adicionar forma recente (ultimos 5 jogos) | Classificacao | Preditor mais importante no futebol |
| 2.2 | Separar clustering por superficie | Agrupamento | Estilo de jogo muda drasticamente entre superficies |
| 2.3 | Feature engineering avancado (Age^2, interacoes, stats/min) | Regressao | Capturar nao-linearidades ja observadas |
| 2.4 | Imputacao condicional por posicao | Regressao | Mediana global mascara diferencas entre posicoes |
| 2.5 | Adicionar posicao na tabela e head-to-head | Classificacao | Contexto competitivo ausente |

### Prioridade 3 (MEDIA) — Robustez e interpretabilidade
| # | Acao | Paradigma | Justificativa |
|---|---|---|---|
| 3.1 | Walk-forward validation | Classificacao | CV aleatorio invalido em series temporais |
| 3.2 | Testar GMM e DBSCAN | Agrupamento | Silhouette baixo; clusters podem ser frágeis |
| 3.3 | Profiling estatistico formal dos clusters | Agrupamento | Testes t, Cohen's d, comparacao com ranking |
| 3.4 | SHAP local + LIME | Regressao/Classificacao | Interpretabilidade por individuo |
| 3.5 | Analise de erros extremos | Regressao | Entender porque MAPE > 100% |

---

## 5. Checklist: O que ja esta OK

| Paradigma | Aspecto | Status | Justificativa |
|---|---|---|---|
| Regressao | Log-transformacao do target | OK | Reduz assimetria (1.836 -> ~0) |
| Regressao | Consistencia logica (GS<=GP, etc.) | OK | 0 inconsistencias |
| Classificacao | Divisao temporal treino/teste | OK | 2006-2015 / 2016-2018 e sensata |
| Classificacao | Merge results x stats | OK | 0 team-season faltando |
| Classificacao | Gols consistentes com resultado | OK | 0 discrepancias |
| Agrupamento | Agregacao por jogador | OK | Media dos registros validos e razoavel |
| Agrupamento | Filtro minimo de partidas | OK | >= 20 partidas e sensato |

---

## 6. Proximos Passos Sugeridos

1. **Iniciar Fase 2 (Feature Engineering)** com as acoes de Prioridade 1 e 2
2. **Documentar cada decisao** em README_MELHORIAS.md com justificativa estatistica
3. **Manter comparacao ANTES vs DEPOIS** para cada mudanca
4. **Apos cada mudanca, re-executar o pipeline** e comparar metricas

---

*Relatorio gerado automaticamente pela auditoria de qualidade de dados.*
*Scripts disponiveis em: `auditoria_dados/auditoria_*.py`*
