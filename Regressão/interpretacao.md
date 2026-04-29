# Interpretacao da Regressao - Predicao de Salarios NBA

## Dataset
University Football Injury Prediction Dataset foi substituido por dados reais da NBA: **NBA Player Salaries (2022-23 Season)** com estatisticas completas de performance. O objetivo e prever o salario anual de jogadores com base em suas estatisticas de jogo e caracteristicas demograficas.

---

## Resumo dos Modelos

| Modelo | R2 (Teste) | RMSE (USD) | MAE (USD) | MAPE |
|---|---|---|---|---|
| OLS | 0.669 | $7,995,659 | $4,820,124 | 136.09% |
| Ridge | 0.627 | $7,949,589 | $4,684,855 | 163.71% |
| Lasso | 0.605 | $8,303,874 | $4,861,438 | 186.58% |
| Random Forest | **0.709** | **$6,745,905** | **$4,036,764** | **99.55%** |

**Melhor modelo:** Random Forest (R2 = 0.709, menor erro absoluto medio).

---

## 1. Coeficientes OLS (Regressao Linear Multipla)

Os coeficientes padronizados indicam a direcao e magnitude do efeito de cada variavel sobre o log-salario:

- **BPM (+0.77):** Box Plus/Minus e o preditor mais positivo. Jogadores com maior impacto geral no jogo (ofensiva + defesa) recebem salarios maiores. Isso faz sentido: superstars como Curry e LeBron tem BPM elevado.
- **PER (-0.72):** Surpreendentemente negativo. Isso pode indicar multicolinearidade com outras estatisticas (PER e altamente correlacionado com PTS e USG%).
- **MP (+0.68):** Minutos por jogo. Jogadores que jogam mais minutos tendem a ser titulares e receber mais.
- **Posicao (PG/SG):** Negativo em relacao ao centro (base). PGs e SGs ganham menos que centers na base, possivelmente devido a uma amostra com muitos armadores reservas.
- **Age (+0.40):** Idade tem efeito positivo, indicando que veteranos experientes ganham mais (efeito de pico de carreira).

**Limitacao:** OLS sofre com multicolinearidade (VIF alto entre stats de arremesso e minutos), tornando alguns coeficientes instaveis.

---

## 2. Ridge vs Lasso (Regularizacao)

### Ridge (alpha = 0.078)
- Encolhe coeficientes proporcionalmente
- Mantem todas as 21 variaveis no modelo
- R2 teste = 0.627 (menor que OLS, indicando que a regularizacao forte nao ajudou neste caso)

### Lasso (alpha = 0.166)
- Seleciona variaveis (alguns coeficientes zerados)
- Zerou: FG%, FT%, 3P%, VORP, WS, TRB_per_GP, BLK_per_GP
- R2 teste = 0.605 (mais parsimonioso, mas com erro maior)

**Discussao:** O Lasso foi agressivo demais, removendo variaveis importantes. O Ridge nao melhorou sobre OLS, sugerindo que a multicolinearidade nao e catastrofica neste dataset. Random Forest capturou melhor as nao-linearidades.

---

## 3. Importancia por Permutacao (Random Forest)

Esta tecnica agnostica ao modelo quantifica a queda no R2 ao embaralhar cada variavel:

1. **GP (0.46):** Numero de jogos disputados e de longe o mais importante. Jogadores que disputam mais jogos sao titulares confiaveis e recebem mais.
2. **MP (0.22):** Minutos por jogo. Titulares jogam mais e ganham mais.
3. **Age (0.10):** Idade importa (veteranos vs rookies).
4. **PTS_per_GP (0.03):** Pontos por jogo tem efeito menor do que esperado, possivelmente porque ja esta capturado por MP e USG%.
5. **USG% (0.01):** Taxa de uso ofensivo.

**Discussao:** A predominancia de GP e MP sobre stats de performance pura sugere que o mercado da NBA valoriza **disponibilidade e minutos** quase tanto quanto eficiencia. Isso reflete a realidade: contratos garantidos e titularidade sao mais previsores de salario do que estatisticas avancadas.

---

## 4. Partial Dependence Plots (PDP)

Os PDPs mostram o efeito marginal medio de cada variavel:

- **GP:** Efeito positivo forte e quase linear. A cada jogo a mais disputado, o salario esperado aumenta.
- **MP:** Efeito positivo ate cerca de 35 minutos, depois estabiliza (plateau). Jogadores com 35+ minutos ja sao titulares maximos.
- **Age:** Efeito nao-linear em forma de U invertido. Salario aumenta com a idade ate o pico (28-32 anos), depois cai.
- **PTS_per_GP:** Efeito positivo mas com retornos decrescentes. Passar de 10 para 20 pontos aumenta mais o salario do que de 25 para 30.

---

## 5. SHAP (SHapley Additive exPlanations)

Infelizmente o pacote SHAP nao estava disponivel neste ambiente. Em um ambiente completo, seria gerado:

- **SHAP Summary:** Ranking global das features com distribuicao dos valores SHAP
- **SHAP Waterfall (Stephen Curry):** Decomposicao do salario de 48M em contribuicoes de cada estatistica
- **SHAP Dependence:** Como o efeito de uma variavel varia com outra

**O que esperariamos ver:**
- Para Curry: BPM, PER e USG% teriam contribuicoes massivamente positivas
- Para um rookie: Idade baixa e poucos GP teriam contribuicoes negativas

---

## 6. Sintese e Conclusoes

### O que os modelos aprendem?
1. **Disponibilidade > Eficiencia:** GP e MP sao os melhores preditores, mais que PER ou VORP.
2. **Efeito nao-linear da idade:** Salario sobe ate o pico de carreira (~30 anos), depois cai.
3. **Posicao importa:** Armadores base ganham menos que alas e pivos (possivelmente devido a amostra).
4. **Stats avancadas tem poder:** BPM e PER aparecem em todos os modelos, validando sua utilidade.

### Limitacoes
- MAPE alto (>100%) indica dificuldade em prever salarios de jogadores de extremo (superstars e rookies)
- Apenas 467 jogadores (amostra pequena)
- Nao considera fatores de mercado (cap salarial, ano do contrato, agente, popularidade)

### Proximos passos
- Coletar dados de multiplas temporadas para painel longitudinal
- Incluir variaveis de mercado (All-Star, MVP, contrato rookie vs max)
- Aplicar SHAP para explicacoes locais individuais

---

*Analise desenvolvida para a disciplina de Mineração de Dados - UFSCar - Grupo 5*
