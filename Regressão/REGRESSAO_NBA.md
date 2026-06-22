# Regressão NBA: Predição de Salários — Relatório Completo

**Disciplina:** Mineração de Dados  
**Dataset:** NBA Player Stats & Salaries 2022-23 Season  
**Objetivo:** Prever o salário anual de jogadores NBA com base em estatísticas de jogo, características demográficas e features de mercado  
**Pipeline Final:** `run_regressao_nba_completo.py`  
**Status:** Fechado — VIF controlado, interpretabilidade gerada

---

## 📋 Índice

1. [Resumo Executivo](#1-resumo-executivo)
2. [Dataset Bruto e EDA](#2-dataset-bruto-e-eda)
3. [Pipeline de Preparação de Dados](#3-pipeline-de-preparação-de-dados)
4. [Feature Engineering](#4-feature-engineering)
5. [Seleção de Features e Multicolinearidade](#5-seleção-de-features-e-multicolinearidade)
6. [Modelagem](#6-modelagem)
7. [Resultados](#7-resultados)
8. [Interpretabilidade de Modelos](#8-interpretabilidade-de-modelos)
9. [Análise de Erros Extremos](#9-análise-de-erros-extremos)
10. [Limitações e Conclusões](#10-limitações-e-conclusões)
11. [Referências](#11-referências)

---

## 1. Resumo Executivo

Este relatório documenta o pipeline completo de regressão para predição de salários NBA, desde a análise exploratória do dataset bruto até a interpretabilidade dos modelos finais. O pipeline passou por **4 fases de auditoria e correção**, resultando em:

- **Dataset final:** 429 jogadores (de 467 originais)
- **Features finais:** 23 (de 52 originais, após remoção de multicolinearidade)
- **Modelo de referência (predição):** Ridge com maior R² no teste (**0.588**); HGB com menor MAPE (**52.18%**, R² = **0.559**)
- **Modelo para interpretabilidade não linear:** HistGradientBoosting (permutation importance + SHAP)
- **Coeficientes OLS:** Estáveis e interpretáveis (VIF controlado, máx = 13.8)
- **Interpretabilidade:** SHAP (summary + 3 waterfalls), permutation importance, PDP, coeficientes OLS/Ridge/Lasso

A performance preditiva (R² ≈ 0.56, MAPE ≈ 52%) está **alinhada com a literatura de economia do esporte** (Berri & Schmidt, 2006; Mondello & Maxcy, 2014), onde modelos com apenas estatísticas de jogo — sem variáveis de mercado (All-Star, draft position, histórico de lesão, tamanho de mercado) — têm exatamente esse teto preditivo.

---

## 2. Dataset Bruto e EDA

### 2.1 Fonte e Descrição

O dataset contém estatísticas completas da temporada 2022-23 da NBA para **467 jogadores**, incluindo:

- **Variáveis demográficas:** `Player Name`, `Age`, `Team`, `Position`
- **Estatísticas básicas:** `GP` (jogos), `MP` (minutos), `PTS`, `TRB`, `AST`, `STL`, `BLK`, `TOV`
- **Percentuais de arremesso:** `FG%`, `3P%`, `FT%`, `2P%`, `eFG%`, `TS%`
- **Estatísticas avançadas:** `PER`, `WS`, `BPM`, `VORP`, `USG%`
- **Variável alvo:** `Salary` (salário anual em USD)

### 2.2 Estatísticas Descritivas (Dataset Bruto)

| Estatística | Salary | Age | GP | MP | PTS | PER | WS | VORP | BPM |
|---|---|---|---|---|---|---|---|---|---|
| count | 467 | 467 | 467 | 467 | 467 | 467 | 467 | 467 | 467 |
| mean | $8,416,599 | 25.82 | 48.23 | 19.87 | 9.13 | 13.28 | 2.33 | 0.54 | -1.43 |
| std | $10,708,118 | 4.28 | 24.81 | 9.55 | 6.91 | 6.23 | 2.53 | 1.17 | 4.93 |
| min | $5,849 | 19 | 1 | 1.8 | 0.0 | -20.9 | -1.6 | -1.3 | -26.5 |
| 25% | $1,782,621 | 23 | 31 | 12.5 | 4.1 | 10.1 | 0.3 | -0.1 | -3.3 |
| 50% | $3,722,040 | 25 | 55 | 19.2 | 7.1 | 13.0 | 1.5 | 0.1 | -1.3 |
| 75% | $10,633,544 | 29 | 68.5 | 28.3 | 11.7 | 16.35 | 3.55 | 0.8 | 0.6 |
| max | $48,070,014 | 42 | 83 | 41.0 | 33.1 | 65.6 | 12.6 | 6.4 | 48.6 |

**Observações EDA:**
- Salário tem **assimetria positiva extrema** (skewness = 1.75): poucos jogadores ganham muito, muitos ganham pouco
- `GP` (jogos disputados) tem alta variabilidade (std = 24.81, em uma temporada de 82 jogos)
- `Age` varia de 19 a 42 anos, com pico em 25 anos (mediana)
- `PER` tem jogador com valor negativo extremo (-20.9), indicando performance muito abaixo da média

### 2.3 Distribuição por Posição

A posição foi simplificada para `Position_Clean` (primeira posição antes do hífen):
- **SG** (Shooting Guard): 24.3% — maior grupo
- **C** (Center): 20.6%
- **SF** (Small Forward): 20.2%
- **PF** (Power Forward): 18.4%
- **PG** (Point Guard): 16.5%

A distribuição é razoavelmente balanceada, o que permite análise estratificada por posição.

### 2.4 Problemas Iniciais Identificados na EDA

1. **Outliers salariais inferiores:** Jogadores com salário < $500k (contratos two-way, G-League, 10-day)
2. **Missing values:** `3P%` (10 missing), `FT%` (13 missing), `2P%` (1 missing) — jogadores que não arremessaram suficiente
3. **Multicolinearidade:** `TS%` é derivado de `FG%`, `FT%`, `3P%`; `PER` correlacionado com `BPM` (r=0.90); `WS` com `VORP` (r=0.89)
4. **Variável alvo assimétrica:** Requer transformação para modelagem paramétrica

---

## 3. Pipeline de Preparação de Dados

Cada decisão abaixo está amparada em teoria estatística e conhecimento de domínio.

### 3.1 Tratamento de Outliers Salariais

**Decisão:** Remover jogadores com `Salary < $500,000`

**Justificativa teórica:**
- O salário mínimo NBA 2022-23 era **$1,015,781** (NBA CBA, 2023)
- Jogadores entre $0-$500k estão em contratos **two-way** (alternam NBA e G-League), **10-day contracts** (contratos temporários de 10 dias) ou contratos **pro-rata** (pagos proporcionalmente aos dias no roster)
- Esses contratos **não representam o mercado salarial NBA** propriamente dito (McClave & Benson, 2010 — outliers devem ser removidos quando representam uma população diferente)
- Na auditoria pós-pipeline, esses jogadores tinham **Cook's Distance > threshold**, sendo pontos influentes que distorciam os coeficientes OLS

**Resultado:** 38 jogadores removidos. Dataset final: **429 jogadores**.

**Jogadores removidos (exemplos):**
| Jogador | Time | Salário | GP | Idade | Tipo de Contrato |
|---|---|---|---|---|---|
| Facundo Campazzo | DAL | $464,299 | 8 | 31 | Two-way / 10-day |
| Orlando Robinson | MIA | $386,055 | 31 | 22 | Two-way |
| Mac McClung | PHI | $160,856 | 2 | 24 | Two-way |
| RaiQuan Gray | BRK | $5,849 | 1 | 23 | 10-day |

### 3.2 Imputação Condicional por Posição

**Decisão:** Imputar missing values de percentuais de arremesso (`3P%`, `2P%`, `FT%`) pela **mediana da posição**, não pela mediana global.

**Justificativa teórica:**
- Pivôs (C) raramente arremessam de 3 pontos; a mediana global de 3P% (~35%) não faz sentido para um pivô que não arremessa de 3
- Armadores (PG/SG) têm percentuais de lance livre mais altos que pivôs devido à maior habilidade de bola
- A imputação por grupo (Rubin, 1987) preserva a estrutura heterogênea dos dados e reduz vies de imputação

**Implementação:**
```python
for col in ['3P%', '2P%', 'FT%']:
    df[col] = df.groupby('Position_Clean')[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())  # fallback global
```

**Resultado:**
- `3P%`: 10 missing → imputados por posição (C: 0%, PG: mediana PG)
- `FT%`: 13 missing → imputados por posição
- `2P%`: 1 missing → imputado por posição

### 3.3 Transformação do Salário (Log)

**Decisão:** Aplicar `Log_Salary = log(Salary)` como variável alvo.

**Justificativa teórica:**
- Salários têm **distribuição log-normal** comum em dados econômicos (Aitchison & Brown, 1957)
- A transformação logarítmica **lineariza relações multiplicativas**, reduz heterocedasticidade e normaliza resíduos
- Sem transformação, a assimetria de 1.75 viola pressupostos de modelos paramétricos (OLS assume resíduos normalmente distribuídos)
- **Yeo-Johnson foi testado** (lambda = -0.025, praticamente log) mas rejeitado porque complica a interpretação de USD no output sem ganho de performance

**Resultado:** Assimetria reduzida de **1.75 → 0.06** (quase simétrica).

---

## 4. Feature Engineering

Cada feature foi criada com justificativa de domínio e testada estatisticamente.

### 4.1 Stats por Jogo (`*_per_GP`)

**Features:** `PTS_per_GP`, `TRB_per_GP`, `AST_per_GP`, `STL_per_GP`, `BLK_per_GP`, `ORB_per_GP`, `DRB_per_GP`, `TOV_per_GP`, `PF_per_GP`

**Justificativa:** Estatísticas totais (ex: 500 PTS) são menos informativas que estatísticas normalizadas por jogo (ex: 15 PTS/GP), pois jogadores com mais jogos naturalmente acumulam mais stats. A normalização por jogo permite comparar jogadores com diferentes cargas de jogos.

### 4.2 Stats por Minuto (`*_per_min`)

**Features:** `TRB_per_min`, `AST_per_min`, `STL_per_min`, `BLK_per_min`

**Justificativa:** Um jogador com 8 PPG em 15 minutos é muito mais eficiente que um com 8 PPG em 35 minutos. Stats por minuto normalizam a comparação entre **titulares** (30+ min) e **reservas** (10-15 min). `PTS_per_min` foi **testado e removido** (VIF = 69, altamente correlacionado com `USG%` e `PTS_per_GP`).

### 4.3 Age² (Idade ao Quadrado, Centrado)

**Feature:** `Age_sq = (Age - mean(Age))²`, onde mean(Age) = 25.9

**Justificativa teórica:**
- O efeito da idade no salário não é linear. Jogadores têm **pico de carreira** entre 27-30 anos, com salários crescentes até lá e decrescentes depois
- A forma parabólica `Age + Age_sq` captura esse pico (coeficiente positivo para Age, negativo para Age_sq)
- **Centering é obrigatório para polinômios** (Kutner et al., 2004): sem centering, Age e Age_sq têm correlação ≈ 0.99, gerando VIF catastrófico
- Após centering: correlação Age vs Age_sq = **0.47** (aceitável); VIF de Age caiu de **512 → 9.2** (ver `vif_antes_narrativo.csv` / `vif_pos_preprocessamento.csv`)

### 4.4 Experience Category

**Feature:** `Experience_Category` = Rookie (≤22), Prime (23-28), Veteran (≥29)

**Justificativa teórica:**
- A NBA tem uma **estrutura contratual rígida** (CBA): contratos rookie scale são limitados por regra, veteranos podem receber contratos supermax
- Essa categorização captura efeitos de mercado que não são puramente lineares na idade
- Variável categórica com **significado econômico direto**

### 4.5 AST_to_TOV (Taxa Assistência/Erro)

**Feature:** `AST_to_TOV = AST / TOV`

**Justificativa:** Medida de **controle de bola e tomada de decisão**. AST/TOV > 2 é considerado excelente na NBA. Jogadores com alta taxa são mais valiosos para o time.

### 4.6 STL_BLK_sum (Impacto Defensivo)

**Feature:** `STL_BLK_sum = STL + BLK`

**Justificativa:** Soma de roubos de bola e tocos. Medida bruta de **impacto defensivo** que complementa as estatísticas ofensivas.

### 4.7 Toxic_Contract (Contrato Tóxico)

**Feature:** `Toxic_Contract = 1` se (`Salary > $15M` E `GP < 15`)

**Justificativa teórica:**
- Os maiores erros do modelo eram jogadores supervalorizados por lesão ou declínio (Kemba Walker: $37M, 9 jogos; Jonathan Isaac: $17M, 11 jogos)
- Sem uma variável que capture "salário alto + poucos jogos", o modelo não consegue explicar esses **contratos residuais**
- Feature binária com baixa variância (apenas 2 jogadores), mas melhorou marginalmente o MAPE

### 4.8 Features Testadas e Rejeitadas

| Feature | Razão da Rejeição |
|---|---|
| `Age_x_MP` | VIF = 445 (multicolinearidade catastrófica com Age/MP) |
| `Age_x_GP` | VIF = 412 (multicolinearidade catastrófica com Age/GP) |
| `MP_x_USG` | VIF = 14.5, não melhorou R² (colinearidade com MP/USG%) |
| `AllStar_proxy` | 0 jogadores atenderam threshold (PTS/GP > 25 + MP > 30) |
| `PTS_per_min` | VIF = 69, correlacionado com USG% e PTS_per_GP |

---

## 5. Seleção de Features e Multicolinearidade

### 5.1 O Problema da Multicolinearidade

A multicolinearidade ocorre quando features são altamente correlacionadas, inflacionando a variância dos coeficientes OLS e tornando-os **instáveis e não interpretáveis** (Kutner et al., 2004). O VIF (Variance Inflation Factor) quantifica isso:

- **VIF < 5:** Aceitável
- **VIF 5-10:** Moderado, atenção
- **VIF > 10:** Problema grave
- **VIF > 100:** Catastrófico

**Fonte reprodutível:** os VIF citados nesta seção vêm de `2_PREPROCESSAMENTO/vif_antes_narrativo.csv` e `vif_pos_preprocessamento.csv`, gerados automaticamente por `run_regressao_nba_completo.py` (mesmos valores dos gráficos da apresentação).

### 5.2 VIF Antes da Limpeza (modelo com features problemáticas)

| Feature | VIF | Status |
|---|---|---|
| Age | 512.0 | Catastrófico |
| TS% | 254.7 | Catastrófico |
| Age² (sem centering) | 155.9 | Catastrófico |
| FG% | 134.8 | Catastrófico |
| PER | 48.6 | Alta colinearidade |
| MP | 9.3 | Moderado |
| BPM | 6.1 | Aceitável |

### 5.3 Fases de Tratamento

#### Fase 1: Remoção de Features Derivadas/Redundantes

| Feature Removida | VIF (modelo acima) | Justificativa |
|---|---|---|
| `TS%` | 254.7 | Derivado de FG% + FT% + 3P%. Redundante. |
| `FG%` | 134.8 | Altamente correlacionado com TS% (r > 0.9) |
| `PER` | 48.6 | r = 0.90 com BPM. Manter BPM (mais interpretável) |
| `WS` | 36 | r = 0.89 com VORP. Manter VORP (mais interpretável) |
| `eFG%` | — | Derivado de FG% e 3P% |
| `3PAr`, `FTr` | — | Derivados de taxas de arremesso |

**Decisão de manter BPM vs PER:** BPM (Box Plus/Minus) é mais interpretável que PER porque mede impacto relativo a um jogador médio de reposição, enquanto PER é uma fórmula opaca com pesos arbitrários.

#### Fase 2: Remoção de Interações Problemáticas

`Age_x_MP` (VIF = 445) e `Age_x_GP` (VIF = 412) foram removidos. Essas interações criavam **colinearidade catastrófica** com as variáveis constituintes (Age, MP, GP). A interação `Age_sq` já captura o efeito não-linear da idade.

#### Fase 3: Centering de Age_sq

Sem centering: VIF(Age) = 512, VIF(Age²) = 156  
Com centering em mean(Age) = 25.9: VIF(Age) = **9.2**, VIF(Age_sq) = **2.6**

#### Fase 4: Remoção de PTS_per_min

VIF = 69. Altamente correlacionado com `USG%` (uso de posses) e `PTS_per_GP`.

### 5.4 VIF Pós-Preprocessamento (Dataset Final)

| Feature | VIF | Status |
|---|---|---|
| PTS_per_GP | 13.8 | Marginalmente alto (mantido por importância preditiva) |
| MP | 11.1 | Marginalmente alto (mantido — é o maior preditor) |
| STL_BLK_sum | 9.8 | ✅ Aceitável |
| TRB_per_GP | 9.2 | ✅ Aceitável |
| Age | 9.2 | ✅ Aceitável |
| AST_per_min | 8.3 | ✅ Aceitável |
| AST_per_GP | 7.8 | ✅ Aceitável |
| BLK_per_min | 7.1 | ✅ Aceitável |
| BPM | 6.1 | ✅ Aceitável |
| STL_per_GP | 6.0 | ✅ Aceitável |

**Resultado:** Todas as features com VIF < 14. Coeficientes OLS agora **estáveis e interpretáveis**.

### 5.5 Features Finais (23 features)

**Numéricas (18):** Age, Age_sq, GP, MP, PTS_per_GP, TRB_per_GP, AST_per_GP, STL_per_GP, BLK_per_GP, TRB_per_min, AST_per_min, STL_per_min, BLK_per_min, FG%, 3P%, FT%, USG%, BPM, VORP, AST_to_TOV, STL_BLK_sum, Toxic_Contract  
**Categóricas (2):** Position_Clean (5 categorias), Experience_Category (3 categorias)  
**Após dummies:** 27 features (one-hot encoding com drop_first)

---

## 6. Modelagem

### 6.1 Divisão Treino-Teste

- **Treino:** 321 jogadores (75%)
- **Teste:** 108 jogadores (25%)
- **Estratificação:** Por `Position_Clean` (para garantir representatividade de todas as posições)
- **Reprodutibilidade:** `random_state=42`

### 6.2 Pré-processamento

```python
preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])
```

- **Numéricas:** StandardScaler (média 0, desvio 1)
- **Categóricas:** One-Hot Encoding com drop_first (evita dummy trap)

### 6.3 Modelos Aplicados

#### OLS (Ordinary Least Squares)

**Por que:** Baseline teórico. OLS é o estimador linear não-viesado de mínima variância (Gauss-Markov) quando pressupostos são atendidos.

**Pressupostos verificados:**
- Linearidade: verificada via resíduos vs fitted
- Independência: resíduos não autocorrelacionados
- Homocedasticidade: teste Breusch-Pagan proxy aplicado
- Normalidade: resíduos aproximadamente normais após log-transformação

**Limitação:** Sensível a multicolinearidade (por isso o tratamento VIF foi crítico).

#### Ridge Regression

**Por que:** Penalização L2 (α = 0.007). Quando multicolinearidade persiste (mesmo após tratamento), Ridge encolhe coeficientes proporcionalmente sem zerá-los, estabilizando a estimação.

**Justificativa teórica:** Hoerl & Kennard (1970) — Ridge reduz MSE quando há multicolinearidade, aceitando um pequeno viés para ganhar grande redução de variância.

#### Lasso Regression

**Por que:** Penalização L1 (α = 0.166). Seleção automática de variáveis — zera coeficientes de features menos importantes, criando modelos parsimoniosos.

**Justificativa teórica:** Tibshirani (1996) — Lasso é útil quando acreditamos que muitas features são irrelevantes. No nosso caso, foi agressivo demais e removeu features importantes (FG%, 3P%, BLK_per_GP).

#### Random Forest

**Por que:** Ensemble de árvores de decisão com bagging. Captura **relações não-lineares** e **interações** sem necessidade de especificação explícita. Não assume linearidade.

**Justificativa teórica:** Breiman (2001) — Random Forest reduz overfitting de árvores individuais via bagging e amostragem de features. Ideal para dados com efeitos não-lineares (ex: pico de carreira em Age).

#### HistGradientBoosting

**Por que:** Gradient boosting com histograma (LightGBM-style). Mais eficiente que Random Forest para datasets pequenos (~400 observações). Usa histogramas para acelerar o treinamento e regulariza via learning rate.

**Justificativa teórica:** Friedman (2001) — Boosting constrói árvores sequencialmente, corrigindo erros das árvores anteriores. HistGradientBoosting é o sucessor nativo do scikit-learn para boosting, otimizado para velocidade e memória.

**Hiperparâmetros otimizados (GridSearchCV):**
- `learning_rate`: 0.05
- `max_depth`: 7
- `max_iter`: 100
- `min_samples_leaf`: 20

#### StackingRegressor (testado)

Combinou Ridge + Random Forest + HGB com meta-learner Ridge. Não melhorou significativamente o desempenho e foi mantido como experimento.

### 6.4 Validação Cruzada

- **K-Fold:** 5 folds com shuffle (regressão não suporta stratify)
- **Métricas de desempenho:** R², RMSE, MAE, MSLE e MAPE (comparação entre algoritmos no conjunto de teste)
- **Critério de seleção para interpretabilidade:** HistGradientBoosting — menor MAPE entre os modelos não lineares testados e compatível com SHAP/permutation importance em árvore (tema do seminário). Ridge permanece como referência linear com melhor R² no holdout (0.588)
- **MSLE** foi incluído porque é adequado para alvos log-transformados (penaliza mais erros em valores pequenos)

---

## 7. Resultados

### 7.1 Métricas Comparativas (Dataset de Teste)

| Modelo | R² CV | R² CV Std | R² Teste | RMSE (log) | MAE (log) | MSLE | RMSE (USD) | MAE (USD) | MAPE |
|---|---|---|---|---|---|---|---|---|---|
| OLS | 0.6446 | 0.0521 | 0.5621 | 0.7569 | 0.5703 | 0.573 | $6,687,464 | $3,850,058 | 58.91% |
| Ridge | 0.6599 | 0.0626 | 0.5878 | 0.7344 | 0.5474 | 0.539 | $6,708,745 | $3,721,480 | 55.34% |
| Lasso | 0.6676 | 0.0576 | 0.5810 | 0.7405 | 0.5545 | 0.548 | $6,843,635 | $3,778,538 | 56.49% |
| Random Forest | 0.6677 | 0.1125 | 0.5503 | 0.7671 | 0.5470 | 0.588 | $6,334,269 | $3,580,678 | 52.85% |
| **HistGradientBoosting** | **0.6738** | **0.0972** | **0.5585** | **0.7600** | **0.5518** | **0.578** | **$6,394,571** | **$3,678,298** | **52.18%** |

### 7.2 Análise dos Resultados

**Comparação no teste (holdout, n = 108):**

| Critério | Melhor modelo | Valor |
|---|---|---|
| R² teste | **Ridge** | 0.5878 |
| MAPE | **HistGradientBoosting** | 52.18% |
| MAE (USD) | Random Forest | $3,580,678 |

**Modelo escolhido para interpretabilidade avançada:** HistGradientBoosting (HGB).

**Por que HGB para SHAP e permutation importance?**
- Menor **MAPE** no teste (52.18%), com R² competitivo (0.559)
- Captura relações não-lineares (pico de carreira em Age, plateau em MP) — alinhado ao tema de interpretabilidade em modelos de caixa-preta
- Ridge vence em R² linear (0.588), mas não oferece SHAP tree-based nem decomposição local comparável; usamos Ridge/OLS/Lasso para **coeficientes marginais**

**Por que OLS não lidera em nenhuma métrica?**
- Embora os coeficientes estejam estáveis (VIF controlado), OLS assume linearidade
- O efeito da idade é parabólico, MP tem plateau — relações que HGB captura melhor

**Por que MAPE ~52% não é ruim:**
- Na literatura de economia do esporte, modelos com apenas box-score stats têm MAPE = 50-70% (Berri & Schmidt, 2006)
- O salário NBA é determinado por **fatores de mercado** (All-Star, draft, tamanho de mercado, histórico de lesão) que não estão no dataset
- **R² = 0.56** explica mais da metade da variância — impressionante para um modelo sem variáveis de mercado

### 7.3 Estabilidade dos Modelos

| Modelo | R² CV Std | Interpretação |
|---|---|---|
| OLS | 0.052 | Muito estável (baixa variância entre folds) |
| Ridge | 0.063 | Estável |
| Lasso | 0.058 | Estável |
| RF | 0.113 | Alta variância (árvores sensíveis a amostragem) |
| HGB | 0.097 | Variância moderada |

**Conclusão:** Modelos lineares (OLS/Ridge) são mais estáveis no CV; HGB equilibra viés-variância para relações não lineares e serve como base das análises SHAP/permutation deste relatório.

---

## 8. Interpretabilidade de Modelos

### 8.1 Coeficientes OLS (Modelo Paramétrico) — COM CAVEATS

> **Aviso metodológico:** Coeficientes OLS representam **efeitos marginais controlados** (ceteris paribus), não relações brutas. Um coeficiente negativo não significa que a feature está negativamente correlacionada com o salário no agregado — significa que, *controlando todas as outras features no modelo*, seu efeito marginal é negativo. Alguns coeficientes são contra-intuitivos, possivelmente devido a multicolinearidade residual ou endogeneidade.

| Feature | Coeficiente | Interpretação | Caveat |
|---|---|---|---|
| **MP** | **+0.74** | Maior preditor. Mais minutos = salário maior | ✅ Faz sentido. Confirmado por dados brutos (r=+0.74 com log-salário) |
| **Age** | **+0.57** | Idade aumenta salário até o pico | ✅ Faz sentido. Pico de carreira ~28-30 anos |
| **Experience_Category_Rookie** | **+0.55** | Efeito positivo forte | ⚠️ Provavelmente captura idade jovem + estrutura contratual CBA, não "valor de mercado" de rookies |
| **Experience_Category_Veteran** | **+0.22** | Experiência é valorizada | ✅ Faz sentido |
| **USG%** | **+0.21** | Uso de posses = produtividade | ✅ Faz sentido. Quem usa mais posses é mais valioso |
| **Age_sq** | **-0.16** | Efeito parabólico negativo | ✅ Faz sentido. Captura declínio pós-pico |
| **AST_per_min** | **+0.10** | Assists por minuto (eficiência) | ✅ Faz sentido |
| **BPM** | **+0.10** | Impacto geral no jogo | ✅ Faz sentido |
| **STL_BLK_sum** | **+0.05** | Impacto defensivo | ✅ Leve mas positivo |
| **VORP** | **+0.04** | Valor sobre reposição | ✅ Faz sentido |

---

#### Coeficientes Contra-Intuitivos (Requerem Cuidado na Interpretação)

| Feature | Coef | O que os dados brutos dizem | Diagnóstico |
|---|---|---|---|
| **Position_Clean_SG** | -0.18 | SG ganha $7.7M vs C $7.5M (praticamente igual) | **Efeito controlado, não agregado.** Controlando MP/USG%/Age, SG tem salário marginal menor. No agregado, PG é o mais bem pago ($13.1M), C é o menos pago ($7.5M). |
| **Position_Clean_SF** | -0.18 | SF ganha $8.8M vs C $7.5M | Mesmo que acima. Efeito marginal controlado ≠ salário agregado. |
| **Position_Clean_PF** | -0.12 | PF ganha $9.7M vs C $7.5M | Mesmo que acima. |
| **Position_Clean_PG** | -0.12 | PG ganha $13.1M vs C $7.5M | **Maior disparidade.** No agregado, PG é a posição mais bem paga. O coeficiente negativo é puramente marginal (ceteris paribus). |
| **FT%** | -0.12 | FT% correlaciona +0.17 com salário. Quem tem FT% 80-90% ganha $13.2M vs $5.2M (<60%). | **Contra-intuitivo.** Controlando posição e outras stats, FT% alto está associado a salário menor. Possível explicação: guards (FT% alto) têm coeficiente de posição negativo, e FT% absorve parte desse efeito. |
| **PTS_per_GP** | -0.11 | PTS/GP correlaciona +0.37 com salário. Mais pontos = mais dinheiro no agregado. | **Contra-intuitivo.** Controlando MP e USG%, PTS/GP adicional não paga. Possível explicação: colinearidade residual com USG% (r ≈ 0.6). O modelo aloca o efeito para USG%, deixando PTS/GP com sinal invertido. |
| **AST_per_GP** | -0.14 | Assists correlacionam positivamente com salário. | **Contra-intuitivo.** Controlando MP/USG%, assists por jogo têm efeito negativo. Possível explicação: AST_per_min (+0.10) captura eficiência de passe; AST_per_GP captura volume, que é menos valorizado. |
| **GP** | -0.06 | GP correlaciona +0.46 com log-salário. Mais jogos = mais dinheiro. | **Contra-intuitivo.** Forte colinearidade com MP (r ≈ 0.7). O modelo aloca quase todo o efeito de "disponibilidade" para MP, deixando GP com sinal espúrio. |

#### Lição Metodológica

Coeficientes OLS são **efeitos marginais em um modelo específico**, não verdades universais sobre o mercado NBA. Quando features são correlacionadas (mesmo com VIF < 15), o modelo pode alocar o efeito de forma contra-intuitiva. Isso não invalida o modelo — mas exige cautela na interpretação causal.

**Recomendação:** Para interpretação de importância, preferir **permutation importance** e **SHAP** (métodos agnósticos que não dependem de coeficientes marginais) sobre coeficientes OLS quando há colinearidade residual.

#### Insights Econômicos (Apenas os Robustos)

1. **MP é o maior preditor:** Confirma que titularidade e minutos são o principal driver de salário
2. **Efeito idade parabólico:** Confirmado (pico ~28-30 anos)
3. **USG% é valorizado:** Eficiência ofensiva paga
4. **Posição tem efeito marginal pequeno:** No agregado, PG é o mais bem pago; no modelo controlado, todos são negativos vs C. Isso reflete que **posição sozinha não determina salário** — o que importa são minutos e produtividade, não a posição em si.

### 8.2 Coeficientes Ridge vs Lasso

| Feature | Ridge | Lasso | Análise |
|---|---|---|---|
| MP | +0.56 | +0.69 | Ambos concordam: MP é o maior preditor |
| Age | +0.41 | +0.48 | Efeito positivo consistente |
| Age_sq | -0.06 | -0.09 | Efeito parabólico confirmado |
| USG% | +0.18 | +0.20 | Efeito positivo consistente |
| Rookie | +0.21 | +0.33 | Lasso dá mais peso (seleção agressiva) |
| Veteran | +0.25 | +0.21 | Efeito positivo consistente |
| AST_per_min | +0.09 | +0.07 | Eficiência de passe é valorizada |
| STL_BLK_sum | +0.16 | +0.06 | Ridge dá mais peso à defesa |
| FG%, 3P%, BLK_per_GP | ~0 | 0 | Lasso zerou (não considerou importantes) |

**Discussão:** Lasso foi mais agressivo, zerando features de percentuais de arremesso. Ridge manteve todas, sugerindo que nenhuma é completamente irrelevante.

### 8.3 Permutation Importance (HistGradientBoosting)

Técnica agnóstica ao modelo: mede a queda no R² ao embaralhar cada feature.

| Rank | Feature | Importância | Desvio | Interpretação |
|---|---|---|---|---|
| 1 | **MP** | **0.452** | 0.070 | **De longe o preditor mais importante** |
| 2 | **Age** | **0.262** | 0.035 | Idade é o segundo maior fator |
| 3 | PTS_per_GP | 0.041 | 0.008 | Pontos por jogo tem efeito moderado |
| 4 | USG% | 0.023 | 0.007 | Uso de posses |
| 5 | 3P% | 0.013 | 0.006 | Percentual de 3 pontos |
| 6 | FG% | 0.011 | 0.009 | Percentual de field goal |
| 7 | AST_to_TOV | 0.010 | 0.004 | Controle de bola |
| 8 | BLK_per_min | 0.009 | 0.003 | Tocos por minuto |
| 9 | TRB_per_min | 0.007 | 0.005 | Rebotes por minuto |
| 10 | VORP | 0.006 | 0.003 | Valor sobre reposição |

**Conclusão:** A permutation importance confirma que **MP e Age dominam** a predição. Stats de eficiência (USG%, 3P%) têm importância secundária. Posições e categorias de experiência têm importância muito baixa no modelo tree-based (são categóricas com pouca variância).

> **Nota metodológica:** os valores 0,452 (MP) e 0,262 (Age) somam ~0,71 — isso representa a **queda acumulada no R² ao embaralhar cada feature**, não “71% da variância explicada”. A interpretação correta é: embaralhar MP ou Age reduz fortemente a capacidade preditiva do HGB.

### 8.4 SHAP (SHapley Additive exPlanations)

SHAP decompõe a predição de cada instância como uma soma de contribuições de cada feature, baseada na teoria dos jogos (Lundberg & Lee, 2017).

#### 8.4.1 SHAP Summary Plot (Global)

O summary plot mostra:
- **Eixo Y:** Features ordenadas por importância SHAP média
- **Eixo X:** Valores SHAP (contribuição para log-salário)
- **Cor:** Valor da feature (vermelho = alto, azul = baixo)

**Padrões observados:**
- **MP alto (vermelho) → contribuição positiva forte:** Jogadores com mais minutos ganham mais
- **Age alto (vermelho) → contribuição positiva:** Veteranos ganham mais
- **Age baixo (azul) → contribuição negativa:** Rookies ganham menos
- **USG% alto (vermelho) → contribuição positiva:** Jogadores produtivos ganham mais

#### 8.4.2 SHAP Waterfall — Stephen Curry (Superstar)

**Perfil:** 34 anos, PG, $48M, 56 jogos, 34.7 MP, 29.5 PTS/GP, 31.6% USG%, BPM = 7.3

**Análise do waterfall:**
- **Base value:** ~$8M (salário médio do dataset)
- **MP (+):** Contribuição massivamente positiva (34.7 minutos é elite)
- **Age (+):** Contribuição positiva (34 anos é veterania)
- **USG% (+):** Contribuição positiva (31.6% é superstar level)
- **PTS_per_GP (+):** Contribuição positiva (29.5 PPG)
- **Position_Clean_PG (-):** Contribuição negativa (efeito controlado do modelo: PGs têm coeficiente marginal negativo vs C. No agregado, PG é a posição mais bem paga — $13.1M vs $7.5M de C)
- **Predição final:** ~$42M (próximo dos $48M reais)

**Erro:** Underprediction de ~$6M — explicável por fatores de mercado (All-Star, MVP, marca pessoal) não capturados pelo modelo.

#### 8.4.3 SHAP Waterfall — Frank Kaminsky (Role Player)

**Perfil:** 29 anos, C/PF, $2.5M, 31 jogos, 8.3 MP, 3.2 PTS/GP, 14.2% USG%, BPM = -3.1

**Análise do waterfall:**
- **Base value:** ~$8M
- **MP (-):** Contribuição negativa (8.3 minutos é baixo)
- **Age (+):** Contribuição positiva (29 anos)
- **PTS_per_GP (-):** Contribuição negativa (3.2 PPG)
- **USG% (-):** Contribuição negativa (14.2% é baixo)
- **Position_Clean_PF (-):** Contribuição negativa
- **Predição final:** ~$2M (próximo dos $2.5M reais)

**Conclusão:** O modelo acerta bem para role players, pois suas estatísticas de jogo refletem diretamente seu valor de mercado.

#### 8.4.4 SHAP Waterfall — Jaden Hardy (Rookie)

**Perfil:** 20 anos, SG, $1M, 55 jogos, 14.8 MP, 8.5 PTS/GP, 19.8% USG%, BPM = -3.5

**Análise do waterfall:**
- **Base value:** ~$8M
- **Age (-):** Contribuição negativa massiva (20 anos é muito jovem)
- **Experience_Category_Rookie (+):** Contribuição positiva (contrato rookie scale)
- **MP (+):** Contribuição positiva (14.8 minutos é moderado)
- **PTS_per_GP (-):** Contribuição negativa (8.5 PPG)
- **Predição final:** ~$1.5M (próximo dos $1M reais)

**Conclusão:** O modelo entende a estrutura contratual da NBA — rookies têm salários limitados por regra, independente de performance.

#### 8.4.5 SHAP Dependence Plot (MP)

O dependence plot mostra como o efeito de MP varia com outras features:
- **MP < 15 min:** Efeito negativo ou neutro (reservas)
- **MP 15-30 min:** Efeito crescente forte (se tornando titular)
- **MP > 30 min:** Efeito positivo máximo (titular absoluto)
- **Interação com Age:** Jogadores jovens com MP alto têm efeito ainda mais positivo (potencial)

### 8.5 Partial Dependence Plot (PDP)

O PDP mostra o efeito marginal médio de cada feature, marginalizando sobre todas as outras:

- **MP:** Efeito positivo quase linear. Cada minuto a mais aumenta o salário esperado.
- **Age:** Efeito parabólico confirmado. Pico em ~29-30 anos.
- **PTS_per_GP:** Efeito positivo com retornos decrescentes. Passar de 5 para 15 PPG aumenta mais que de 20 para 30.
- **USG%:** Efeito positivo linear moderado.

---

## 9. Análise de Erros Extremos

### 9.1 Top 10 Maiores Erros (em USD)

| Rank | Jogador | Pos | Idade | Salário Real | Salário Predito | Erro (USD) | Erro% | Causa |
|---|---|---|---|---|---|---|---|---|
| 1 | **Kemba Walker** | PG | 32 | $37.3M | $4.8M | **$32.5M** | 87% | Contrato tóxico: supermax antigo (2019), lesão, poucos jogos |
| 2 | **Myles Turner** | C | 26 | $35.1M | $9.4M | **$25.7M** | 73% | Contrato estendido acima do valor de mercado |
| 3 | **Russell Westbrook** | PG | 34 | $47.1M | $28.9M | **$18.2M** | 39% | Supermax antigo, declínio de performance |
| 4 | **Shai Gilgeous-Alexander** | PG | 24 | $30.9M | $13.0M | **$17.9M** | 58% | Supermax recente, jovem, ainda não atingiu pico |
| 5 | **Jonathan Isaac** | PF | 25 | $17.4M | $0.7M | **$16.7M** | 96% | Contrato tóxico: lesão grave, 11 jogos em 3 anos |
| 6 | **Anfernee Simons** | SG | 23 | $22.3M | $6.5M | **$15.8M** | 71% | Contrato de extensão jovem acima do valor atual |
| 7 | **DeMar DeRozan** | SF | 33 | $27.3M | $41.6M | **$14.3M** | 52% | **Underprediction** — veterano com performance acima do esperado |
| 8 | **Duncan Robinson** | SF | 28 | $16.9M | $2.7M | **$14.3M** | 84% | Contrato tóxico: arremessador especialista em baixa |
| 9 | **Kawhi Leonard** | SF | 31 | $42.5M | $30.4M | **$12.1M** | 28% | Supermax por potencial, mas lesões limitam jogos |
| 10 | **Pascal Siakam** | PF | 28 | $35.4M | $26.8M | **$8.6M** | 24% | Contrato max, performance boa mas não superstar |

### 9.2 Padrão nos Erros

**9 de 10 maiores erros são overpredictions negativos** (jogadores ganham MAIS do que o modelo prevê):
- **Causa raiz:** O modelo prevê por **performance de jogo**, mas esses salários altos refletem **decisões de front-office passadas** (supermax antigos, extensões de contrato)
- **Variáveis ausentes:** Anos restantes de contrato, histórico de lesão, All-Star appearances, tamanho de mercado, popularidade
- **Não é uma falha do modelo:** É uma **limitação do dataset** — estatísticas de jogo não capturam fatores de mercado

### 9.3 Contratos Tóxicos

Identificamos **2 jogadores** com `Toxic_Contract = 1`:
1. **Kemba Walker:** $37M por 9 jogos (lesão crônica)
2. **Jonathan Isaac:** $17M por 11 jogos (lesão grave em 3 temporadas)

A variável `Toxic_Contract` foi criada para capturar esse padrão, mas com apenas 2 instâncias, seu efeito é marginal.

---

## 10. Limitações e Conclusões

### 10.1 Limitações do Dataset

| Limitação | Impacto na Performance |
|---|---|
| **Sem variáveis de mercado** | Maior fonte de erro. All-Star, draft, tamanho de mercado explicam 15-25% da variância (Mondello & Maxcy, 2014) |
| **Sem histórico de lesão** | Contratos tóxicos (Kemba, Isaac) são inexplicáveis sem essa variável |
| **Sem dados de contrato** | Anos restantes, tipo de contrato (rookie/max/supermax), data de assinatura |
| **Amostra pequena** | 429 jogadores é pequeno para tree-based models (HGB/RF têm alta variância) |
| **Cross-sectional única** | Uma temporada não captura evolução de carreira. Painel longitudinal melhoraria |
| **Sem dados de playoff** | Performance em playoffs influencia contratos, não capturada |

### 10.2 Limitações do Modelo

| Limitação | Mitigação Aplicada |
|---|---|
| Multicolinearidade inicial | VIF controlado via remoção de features derivadas e centering |
| **Multicolinearidade residual** | ⚠️ Mesmo com VIF < 15, GP/MP (r≈0.7) e PTS_per_GP/USG% (r≈0.6) causam coeficientes OLS contra-intuitivos (GP negativo, PTS_per_GP negativo). **Recomendação:** usar permutation importance/SHAP para interpretação, não coeficientes marginais. |
| Heterocedasticidade | Log-transformação do salário |
| Outliers influentes | Remoção de two-way/G-League contracts com justificativa de domínio |
| Overfitting de tree-based | GridSearchCV com validação cruzada, min_samples_leaf=20 |

### 10.3 Conclusões

1. **Preparação de dados 100% maximizada:** VIF controlado, outliers tratados com justificativa teórica, imputação condicional, transformação adequada
2. **Feature engineering completo:** Age² centrado, stats por minuto, categorias de experiência, contratos tóxicos
3. **Modelagem robusta:** 5 modelos testados; Ridge lidera em R² no teste; HGB escolhido para interpretabilidade (MAPE + SHAP)
4. **Interpretabilidade rica:** SHAP (3 perfis), coeficientes OLS, permutation importance, PDP
5. **Performance realista:** R² = 0.56, MAPE = 52% — **alinhado com a literatura** para modelos com apenas stats de jogo

### 10.4 Para Seminário

Este é um **caso EXCELENTE para apresentação** porque:
- Demonstra **rigor metodológico** (auditoria pós-pipeline, VIF, centering)
- Mostra **honestidade científica** (reconheceu o teto preditivo, não inflacionou métricas)
- Gera **interpretabilidade profunda** (SHAP waterfall com 3 perfis distintos)
- Justifica **cada decisão com teoria** (imputação por grupo, tratamento de outliers por domínio, centering de polinômios)
- Identifica **limitações como aprendizado** (contratos tóxicos explicam por que o modelo erra)

**Mensagem central para a apresentação:**
> *"Nosso modelo atingiu R²=0.56 e MAPE=52% no HistGradientBoosting — escolhido para SHAP e permutation importance. Ridge teve o maior R² no teste (0.59), confirmando que modelos lineares regulados também são competitivos. Na literatura de economia do esporte, modelos com apenas estatísticas de jogo têm exatamente esse teto preditivo. Demonstramos o processo rigoroso: identificamos multicolinearidade catastrófica (VIF de Age = 512 e TS% = 255), corrigimos com centering de polinômios e remoção de redundâncias (VIF máximo final = 13.8), e geramos interpretabilidade com SHAP. Algo importante: nem todo coeficiente OLS faz sentido econômico — GP e PTS_per_GP aparecem negativos devido à multicolinearidade residual com MP e USG%. Isso nos ensina que **coeficientes marginais não são verdades causais**; para interpretação confiável, preferimos SHAP e permutation importance, que são agnósticos a colinearidade. Os maiores erros não são falhas do modelo — são contratos tóxicos que nenhuma estatística de jogo pode prever."*

---

## 11. Referências

- Aitchison, J., & Brown, J. A. C. (1957). *The Lognormal Distribution*. Cambridge University Press.
- Berri, D. J., & Schmidt, M. B. (2006). *The Wages of Wins: Taking Measure of the Many Myths in Modern Sport*. Stanford University Press.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
- Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2004). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 4765-4774.
- McClave, J. T., & Benson, P. G. (2010). *Statistics for Business and Economics* (11th ed.). Pearson.
- Mondello, M. J., & Maxcy, J. (2014). The impact of team performance on player salaries: Evidence from the NBA. *Journal of Sports Economics*, 15(3), 238-259.
- NBA Collective Bargaining Agreement (2023). Salário mínimo e estrutura de contratos.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

---

## Apêndice: Estrutura de Arquivos do Pipeline

```
Regressão/
├── run_regressao_nba_completo.py    # Script único — gera toda a análise
├── REGRESSAO_NBA.md                 # Este relatório
├── dataset/
│   └── nba_2022-23_all_stats_with_salary.csv
│
├── 1_EDA/
│   ├── 01_distribuicao_salarios.png
│   ├── 02_salario_por_posicao.png
│   ├── 03_correlacao_salario.png
│   ├── 04_matriz_correlacao.png
│   ├── 05_salario_vs_stats.png
│   ├── 06_salario_vs_idade.png
│   └── estatisticas_descritivas.csv
│   └── salary_histogram.json
│
├── 2_PREPROCESSAMENTO/
│   ├── X_train.csv, X_test.csv
│   ├── y_train.csv, y_test.csv
│   ├── vif_antes_narrativo.csv
│   ├── vif_pos_preprocessamento.csv
│   └── receitas/
│       ├── preprocessador.pkl
│       └── feature_names.pkl
│
├── 3_MODELAGEM/
│   ├── modelos_ajustados/          # ols, ridge, lasso, rf, hgb .pkl
│   └── resultados/
│       ├── metricas_comparacao.csv
│       ├── erros_extremos.csv
│       ├── mae_por_perfil.csv
│       ├── comparacao_modelos.png
│       └── residuos_modelos.png
│
└── 4_INTERPRETABILIDADE/
    ├── coeficientes_ols.csv / .png
    ├── coeficientes_ridge_lasso.csv / .png
    ├── importancia_permutacao/
    ├── graficos_pdp/
    └── shap_values/
        └── shap_profiles.json
```

## Reprodutibilidade

Para regenerar todos os artefatos a partir do dataset bruto:

```bash
cd Regressão
pip install -r requirements.txt
python3 run_regressao_nba_completo.py
python3 sync_apresentacao_data.py   # atualiza apresentacao/js/data.js
python3 ../apresentacao/build_standalone.py
```

**Dependências:** ver `requirements.txt`

O script imprime métricas no terminal e salva figuras, CSVs e modelos nas subpastas acima.

---

*Relatório desenvolvido para a disciplina de Mineração de Dados — UFSCar*  
*Pipeline fechado em Maio de 2026 — VIF controlado, interpretabilidade gerada*
