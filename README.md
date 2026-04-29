# Projeto Mineração de Dados

## Grupo 5 -- Interpretabilidade de Modelos

Repositório oficial do trabalho final da disciplina de Mineração de Dados, Departamento de Estatística, Universidade Federal de São Carlos (UFSCar), ministrada pelo Prof. Dr. Helton Graziadei.

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

**Modalidade:** A -- Projeto aplicado de mineração de dados.

---

## Visão Geral do Projeto

O objetivo deste trabalho é realizar um estudo abrangente de interpretabilidade de modelos preditivos aplicado aos principais paradigmas de modelagem abordados na disciplina. Em vez de restringir a análise a um único tipo de problema, o grupo desenvolverá aplicações em cada área sorteada aos demais grupos (regressão, classificação, agrupamento, mineração de textos e detecção de anomalias).

Em cada paradigma, serão ajustados modelos representativos e aplicadas técnicas específicas de interpretabilidade para comparar resultados, explicar comportamentos e responder à pergunta central: por que o modelo prediz o que prediz?

---

## Estrutura por Paradigma

### 1. Regressão
Modelos lineares regularizados (Ridge, Lasso) e árvores de regressão (Random Forest, XGBoost). Interpretação via inspeção de coeficientes, Gráficos de Efeito Parcial (PDP) e valores SHAP para quantificar a contribuição de cada preditor nas predições quantitativas.

### 2. Classificação
Regressão Logística com penalização Lasso, Random Forest e XGBoost para predição binária ou multiclasse. Uso de coeficientes regularizados, importância por permutação, PDP e SHAP para decompor as predições e interpretar os efeitos das variáveis sobre as probabilidades de classe.

### 3. Agrupamento
Aplicação de K-Means e agrupamento hierárquico para descoberta de estruturas latentes. Interpretação dos perfis encontrados por meio da análise das variáveis que mais caracterizam cada grupo, permitindo segmentação e perfilamento acionável.

### 4. Mineração de Textos
Modelagem de tópicos via LDA e classificação de documentos. Interpretação por inspeção das palavras mais relevantes por tópico e análise de contribuições de termos nas predições, conectando os padrões estatísticos ao significado linguístico.

### 5. Detecção de Anomalias
Abordagens supervisionadas e não supervisionadas (Isolation Forest, LOF). Interpretação das anomalias detectadas via identificação das variáveis que mais se desviam do perfil médio, utilizando técnicas de explicação local.

---

## Estrutura do Repositório

```
Projeto Mineração de Dados /
├── compile/                          # PDFs compilados e prontos para entrega
│   └── main.pdf                      # Proposta de projeto (PDF final)
│
├── conteudo materia/                 # Material didático da disciplina
│   ├── AulaR.pdf                     # Introdução ao tidyverse e tidymodels
│   ├── Avaliacao_Classificadores_Logistica.pdf
│   ├── Fund_Aprendizagem_Supervisionada.pdf
│   ├── Metodos_Nao_Param_Reg.pdf
│   ├── Mineracao_Dados-Lista1.pdf    # Lista de exercícios I
│   ├── Mineracao_Dados-Lista2.pdf    # Lista de exercícios II
│   ├── Mineracao_Dados-Lista3.pdf    # Lista de exercícios III
│   ├── Regressao_Linear.pdf
│   ├── Selecao de Variaveis e Regularizacao.pdf
│   ├── Seminario.pdf                 # Instruções para o trabalho final
│   └── Seminario (1).pdf
│
├── overleaf/                         # Projeto LaTeX (Overleaf)
│   ├── images/                       # Pasta para figuras
│   ├── main.tex                      # Código-fonte LaTeX
│   ├── main.pdf                      # PDF compilado
│   ├── Projeto_Grupo5_Overleaf.zip   # Arquivo compactado para upload no Overleaf
│   └── references.bib                # Bibliografia (se necessário)
│
└── README.md                         # Este arquivo
```

---

## Base de Dados

As bases de dados serão selecionadas de repositórios públicos (UCI Machine Learning Repository, Kaggle, bases acadêmicas) de forma a cobrir adequadamente cada paradigma. A escolha definitiva será submetida à aprovação do professor, garantindo que cada aplicação possua variáveis interpretáveis no contexto do domínio.

---

## Implementação

O trabalho será desenvolvido integralmente em linguagem R, utilizando os seguintes pacotes e ecossistemas:

- **Modelagem:** `tidymodels` (pré-processamento, divisão estratificada, validação cruzada, ajuste de hiperparâmetros).
- **Interpretabilidade:** `DALEX` e `fastshap` (PDP, importância por permutação, SHAP).
- **Mineração de Textos:** `tidytext` (tokenização, remoção de stopwords, LDA).
- **Detecção de Anomalias:** `dbscan`, `isolationForest`.

O código será organizado em notebooks reprodutíveis, documentando desde a importação dos dados brutos até a geração das figuras e tabelas finais.

---

## Cronograma e Entregas

| Etapa | Data Limite |
|---|---|
| Formação dos grupos | 14/04 |
| Sorteio dos temas | 16/04 |
| Envio da proposta | 30/04 |
| Devolutiva e aprovação | 08/05 |
| Entrega do relatório e código | 25/06 |
| Início dos seminários | 02/07 |

Todas as entregas devem ser enviadas para o e-mail do professor: **helton@ufscar.br**, com o assunto no formato: `[Mineração de Dados] - Entregável - Grupo 5`.

---

## Critérios de Avaliação

1. **Qualidade do relatório (50%):** Rigor técnico e metodológico, clareza na escrita, correta formatação e interpretação adequada dos resultados estatísticos.
2. **Seminário (50%):** Organização da apresentação, gestão do tempo, organização e reprodutibilidade do código entregue e domínio técnico demonstrado na arguição.

---

## Uso de Ferramentas de Inteligência Artificial

O uso de assistentes generativos de IA é permitido como ferramenta de apoio ao aprendizado (depuração de código, revisão textual, brainstorming). Em caso de uso substancial, uma declaração deve ser incluída na seção de Metodologia do relatório. Os alunos são integralmente responsáveis por qualquer erro técnico, metodológico ou conceitual gerado pela IA.

---

## Instruções para o Grupo

1. **Proposta atual:** o PDF `compile/main.pdf` e o zip `overleaf/Projeto_Grupo5_Overleaf.zip` contêm a proposta de projeto já formatada.
2. **Próximos passos:** após aprovação da proposta pelo professor, iniciar a coleta e preparação das bases de dados para cada paradigma.
3. **Divisão de tarefas:** sugerimos que cada integrante assuma a liderança de um ou dois paradigmas, mantendo todos alinhados via este repositório.
4. **Controle de versão:** utilizem commits frequentes e descritivos para acompanhar o progresso do projeto.

---

*São Carlos -- SP, Abril de 2025.*
