# Interpretacao da Classificacao - Vitoria do Mandante na Premier League

## Dataset
**Fonte:** Kaggle - Premier League Stats (temporadas 2006-2018)
- **results.csv:** 4.560 partidas com resultados, gols, times e temporada
- **stats.csv:** Estatisticas agregadas por time-temporada (gols, chutes, defesas, passes, cartoes, etc.)

## Problema
Classificacao binaria: prever se o time **mandante vence** (1) ou **nao vence** (0 = empate ou derrota).

**Divisao temporal:**
- **Treino:** 2.280 partidas (temporadas 2006-2007 a 2015-2016)
- **Teste:** 760 partidas (temporadas 2016-2017 a 2017-2018)

A divisao temporal simula um cenario real de previsao: o modelo e treinado em dados passados e testado em dados futuros.

---

## Resumo dos Modelos

| Modelo | AUC-ROC | Acuracia | Precisao | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistica Lasso | **0.767** | 0.687 | 0.718 | 0.558 | 0.628 |
| Random Forest | 0.751 | 0.689 | 0.731 | 0.544 | 0.624 |

**Melhor modelo:** Logistica com Lasso (maior AUC-ROC, mais interpretavel).

**Baseline:** A taxa de vitoria do mandante na Premier League e de aproximadamente 46%. Um classificador trivial que sempre preve "nao-vitoria" acertaria ~54% das vezes. Nossos modelos atingem ~69% de acuracia, representando um ganho preditivo relevante.

---

## 1. Coeficientes Lasso (Regressao Logistica Regularizada)

A penalizacao L1 selecionou automaticamente apenas **3 variaveis** entre 39 candidatas:

| Feature | Coeficiente | Interpretacao |
|---|---|---|
| diff_losses | **-0.408** | Quanto mais derrotas o mandante tem em relacao ao visitante, menor a chance de vitoria em casa |
| diff_wins | **+0.406** | Quanto mais vitorias o mandante tem em relacao ao visitante, maior a chance de vitoria em casa |
| diff_att_ibox_goal | +0.029 | Gols de dentro da area (diferencial) tem efeito positivo mas pequeno |

### Discussao
- **diff_wins** e **diff_losses** dominam completamente o modelo. Isso faz sentido: o historico de resultados (forma do time) e o melhor preditor de resultados futuros.
- O fato de o Lasso zerar todas as outras 36 variaveis indica que, uma vez conhecido o historico de vitorias/derrotas, estatisticas de estilo de jogo (chutes, passes, defesas) nao acrescentam poder preditivo adicional.
- **diff_att_ibox_goal** sobrevive como um sinal residual de qualidade ofensiva: times que marcam mais de dentro da area tendem a ser mais perigosos.
- A simetria dos coeficientes (wins ~ +0.41, losses ~ -0.41) sugere que o modelo equilibra forma positiva e negativa de forma quase simetrica.

---

## 2. Importancia por Permutacao (Random Forest)

Tecnica agnostica que mede a queda no AUC-ROC ao embaralhar cada variavel:

| Rank | Feature | Importancia | Interpretacao |
|---|---|---|---|
| 1 | diff_wins | 0.0218 | Historico de vitorias do mandante vs visitante |
| 2 | diff_losses | 0.0182 | Historico de derrotas do mandante vs visitante |
| 3 | diff_att_ibox_goal | 0.0107 | Gols de dentro da area (diferencial) |
| 4 | diff_goals_conceded | 0.0080 | Gols sofridos (diferencial) |
| 5 | diff_ontarget_scoring_att | 0.0022 | Chutes a gol (diferencial) |

### Discussao
- O Random Forest concorda com a Logistica: **diff_wins** e **diff_losses** sao as features mais importantes.
- A importancia por permutacao revela que **diff_goals_conceded** (gols sofridos) tambem importa: mandantes que tomam menos gols que o visitante tem vantagem.
- Stats de estilo de jogo (passes, cruzamentos, desarmes, interceptacoes) tem importancia quase nula. Isso sugere que o "como" o time joga e menos importante que o "quanto" ganha/perde.
- A vantagem do mandante intrinseca (46% das vezes) parece ser capturada pela diferenca de forma, nao por uma variavel binaria de local.

---

## 3. Partial Dependence Plots (PDP)

Os PDPs ilustram o efeito marginal medio de cada variavel sobre a probabilidade de vitoria do mandante:

- **diff_wins:** Efeito positivo e monotonico. A cada vitoria a mais do mandante em relacao ao visitante, a probabilidade de vitoria em casa aumenta linearmente.
- **diff_losses:** Efeito negativo e monotonico. Mais derrotas do mandante = menor chance de vitoria em casa.
- **diff_att_ibox_goal:** Efeito positivo com retornos decrescentes. Passar de 0 a +5 gols de dentro da area aumenta a probabilidade, mas de +10 a +15 ja nao aumenta tanto.
- **diff_goals_conceded:** Efeito negativo. Mandantes que sofrem muito mais gols que o visitante tem chance menor de vencer em casa.
- **diff_ontarget_scoring_att:** Efeito positivo mas fraco. Mais chutes a gol ajudam, mas nao tanto quanto vitorias acumuladas.

---

## 4. SHAP (SHapley Additive exPlanations)

O pacote SHAP nao estava disponivel neste ambiente. Em um ambiente completo, seria gerado:

- **SHAP Summary:** Distribuicao global dos impactos de cada feature
- **SHAP Waterfall:** Explicacao individual de uma partida especifica (ex: porque o Manchester City venceu em casa?)
- **SHAP Dependence:** Como o efeito de uma feature varia com outra

**O que esperariamos ver:**
- Para um grande time jogando em casa contra um rebaixado: diff_wins altamente positivo, impulsionando a predicao para vitoria
- Para um time medio contra outro medio: valores SHAP pequenos, predicao proxima do baseline (~46%)

---

## 5. Sintese e Conclusoes

### O que os modelos aprendem?
1. **Forma do time > Estilo de jogo:** O historico de vitorias/derrotas do time e o unico preditor realmente importante para prever resultados na Premier League.
2. **Estatisticas de processo nao ajudam:** Chutes, passes, cruzamentos, desarmes e interceptacoes nao acrescentam poder preditivo uma vez que a forma do time e conhecida.
3. **Qualidade ofensiva marginal:** Gols de dentro da area e chutes a gol tem efeito pequeno mas positivo.
4. **Vantagem do mandante e real, mas capturada pela forma:** O fato de jogar em casa (46% de vitorias) parece estar embutido na diferenca de forma entre os times.

### Limitacoes
- **Apenas 3 variaveis selecionadas:** O Lasso foi muito agressivo, possivelmente descartando informacoes relevantes.
- **Sem dados de elenco:** Lesoes, suspensoes, contratacoes e saidas de jogadores-chave nao sao considerados.
- **Sem contexto de fixture:** Derbys, jogos apos competicoes europeias, rotacao de elenco nao sao modelados.
- **AUC 0.77:** Bem melhor que aleatorio (0.5) e baseline (0.54), mas longe de perfeito. Futebol e intrinsecamente imprevisivel.

### Proximos passos
- Incluir forma recente (ultimos 5 jogos) ao inves de acumulada na temporada
- Adicionar variaveis de mercado (odds de apostas, posicao na tabela)
- Testar modelos sequenciais (LSTM, GRU) que capturam momentum temporal
- Incluir SHAP para explicacoes locais de partidas especificas

---

*Analise desenvolvida para a disciplina de Mineracao de Dados - UFSCar - Grupo 5*
