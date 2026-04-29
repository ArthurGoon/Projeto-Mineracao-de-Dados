# Interpretacao do Agrupamento - Perfis de Jogadores de Tenis

## Dataset
**Fonte:** Kaggle - Tennis Player Data (Serve, Return e Raw)
- **462 jogadores** do circuito ATP masculino
- **237.185 registros** de estatisticas de saque
- **237.196 registros** de estatisticas de devolucao
- Dados de partidas entre 2019 e 2024

## Preparacao dos Dados
1. **Agregacao:** Media das estatisticas de saque e devolucao por jogador
2. **Filtro:** Jogadores com **minimo de 20 partidas** (98 jogadores finais)
3. **Features:** 12 metricas padronizadas
4. **Padronizacao:** StandardScaler para K-Means

## Escolha do Numero de Clusters
O metodo do cotovelo e o silhouette score indicaram **K = 2** como o numero ideal de clusters.

| K | Silhouette Score |
|---|---|
| 2 | **0.239** (melhor) |
| 3 | 0.224 |
| 4 | 0.160 |
| 5 | 0.181 |

A divisao em 2 grupos e estatisticamente a mais coesa e interpretavel.

---

## 1. K-Means (K = 2)

### Tamanho dos Clusters
- **Cluster 0:** 42 jogadores (42.9%)
- **Cluster 1:** 56 jogadores (57.1%)

### Centroides (padronizados)

| Feature | Cluster 0 | Cluster 1 | Interpretacao |
|---|---|---|---|
| Aces_pct | **+0.82** | -0.61 | Cluster 0: muitos aces |
| Df_pct | +0.24 | -0.18 | Cluster 0: mais duplas faltas |
| FirstServe_In | -0.41 | +0.31 | Cluster 1: primeiro saque mais consistente |
| FirstServe_Win | **+0.82** | -0.61 | Cluster 0: muito forte no primeiro saque |
| SecondServe_Win | +0.03 | -0.02 | Segundo saque similar |
| TotalPoints_Won | -0.09 | +0.07 | Cluster 1: leve vantagem |
| ReturnPoints_Won | **-0.83** | **+0.62** | Cluster 1: muito melhor na devolucao |
| vAces_pct | +0.17 | -0.13 | Cluster 0: sofre menos aces (provavelmente jogam mais rapido) |
| v1stReturn_Win | -0.74 | +0.56 | Cluster 1: ganha mais pontos na devolucao do 1o saque |
| v2ndReturn_Win | -0.81 | +0.60 | Cluster 1: ganha mais pontos na devolucao do 2o saque |
| BP_Saved_rate | +0.53 | -0.40 | Cluster 0: salva mais break points (saque forte) |
| BP_Converted_rate | **-0.83** | **+0.63** | Cluster 1: converte muito mais break points |

### Perfis Identificados

#### Cluster 0: **"Saqueadores / Agressivos"** (42 jogadores)
**Caracteristicas:**
- Alto percentual de aces (+0.82 z-score)
- Muito forte no primeiro saque (+0.82)
- Salvam mais break points (+0.53)
- **Fracos na devolucao:** ReturnPoints_Won muito baixo (-0.83)
- Convertem poucos break points (-0.83)

**Jogadores representativos:**
- Thanasi Kokkinakis, Alexei Popyrin, Grigor Dimitrov, Hubert Hurkacz, Karen Khachanov

**Interpretacao:** Esses jogadores dependem do saque para ganhar. Quando o saque funciona, dominam; quando falha, tem dificuldade na devolucao. Sao jogadores de estilo agressivo que buscam finalizar pontos rapidamente.

---

#### Cluster 1: **"Devolvedores / Baseline"** (56 jogadores)
**Caracteristicas:**
- Baixo percentual de aces (-0.61)
- Primeiro saque mais consistente (+0.31)
- **Fortes na devolucao:** ReturnPoints_Won alto (+0.62)
- Convertem muitos break points (+0.63)
- Menos duplas faltas (-0.18)

**Jogadores representativos:**
- Mackenzie Mcdonald, Pablo Carreno Busta, Lorenzo Musetti, Albert Ramos, Tommy Paul, Kei Nishikori, Alex De Minaur

**Interpretacao:** Esses jogadores sao mais consistentes, menos dependentes do saque e mais fortes na troca de bolas de fundo de quadra (baseline). Eles ganham partidas na devolucao e na consistencia, nao no saque avassalador.

---

## 2. Agrupamento Hierarquico

O agrupamento hierarquico (Ward, K=2) produziu uma divisao diferente:
- Cluster 0: 49 jogadores
- Cluster 1: 49 jogadores

**Matriz de contingencia K-Means vs Hierarquico:**

|  | Hier 0 | Hier 1 |
|---|---|---|
| **KM 0** | 1 | 41 |
| **KM 1** | 48 | 8 |

**Concordancia:** Os dois metodos concordam parcialmente. A maioria dos jogadores do Cluster 1 do K-Means (devolvedores) foi para o Cluster 0 do Hierarquico. A separacao basica "saqueadores vs devolvedores" se mantem, mas os metodos divergem em alguns casos de fronteira.

---

## 3. Visualizacao PCA

A Analise de Componentes Principais mostra:
- **PC1 (44.1%):** Eixo principal que separa saqueadores de devolvedores
- **PC2 (20.0%):** Eixo secundario relacionado a consistencia/erros

Os dois clusters aparecem bem separados no espaco PCA, validando a qualidade do agrupamento.

---

## 4. Sintese e Conclusoes

### Os clusters fazem sentido no tenis?
**Sim.** A dicotomia encontrada reflete perfeitamente o debate classico do tenis:

1. **Jogadores de saque (Serve-and-volley / Agressivos):** Depedem de um saque forte para impor o ritmo. Exemplos historicos: Pete Sampras, Goran Ivanisevic.
2. **Jogadores de devolucao (Baseline / Consistentes):** Dependem da troca de bolas, movimentacao e consistencia. Exemplos historicos: Rafael Nadal, David Ferrer.

### Surpreendente: nao ha cluster "equilibrado"
O algoritmo nao encontrou um grupo intermediario. Isso sugere que, no circuito ATP masculino atual, jogadores tendem a se especializar em um dos dois extremos. Jogadores "completos" (bons no saque E na devolucao) sao raros e acabam sendo atribuidos a um dos dois polos.

### Limitacoes
1. **Apenas 98 jogadores:** O filtro de 20 partidas eliminou muitos jogadores jovens ou de segundo escalao.
2. **Sem separacao por superficie:** O estilo de jogo muda entre grama (favorece saque) e saibro (favorece devolucao).
3. **Estatisticas agregadas:** Nao capturam evolucao temporal (jogadores que mudaram de estilo ao longo da carreira).
4. **K=2 pode ser simplista:** Alguns especialistas argumentariam que jogadores de "saque-voleio" sao diferentes de "saque-baseline", mas os dados nao suportaram essa subdivisao.

### Proximos passos
- Incluir superficie como feature (alguns jogadores sao especialistas de grama)
- Adicionar estatisticas de rede (voleios) para separar saque-voleio puro
- Incluir dados de WTA (tenis feminino) para comparar perfis entre generos

---

*Analise desenvolvida para a disciplina de Mineracao de Dados - UFSCar - Grupo 5*
