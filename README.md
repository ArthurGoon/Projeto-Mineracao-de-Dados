# Projeto Mineração de Dados — Grupo 5

**Tema:** Interpretabilidade de Modelos  
**Disciplina:** Mineração de Dados — UFSCar  
**Prof.:** Dr. Helton Graziadei

## Integrantes

| Nome | RA |
|------|-----|
| Arthur Pereira Gon | 811464 |
| Giovanni Roberti | 811676 |
| Igor Figueiredo | 791420 |
| Lucas Yukio | 813650 |
| Murilo Cassiavilani | 812067 |

---

## Entrega do seminário — Regressão NBA

Predição de salários da NBA (temporada 2022-23) com foco em **auditoria metodológica** e **interpretabilidade** (OLS, Ridge, Lasso, permutation importance, SHAP).

### Estrutura do repositório

```
├── apresentacao/          # Deck scrollytelling (HTML final + fonte)
├── Regressão/             # Dataset, script único e resultados
└── README.md
```

---

## Regressão — como reproduzir

```bash
cd Regressão
python3 run_regressao_nba_completo.py
```

**Entrada:** `dataset/nba_2022-23_all_stats_with_salary.csv` (467 jogadores)

**Saídas geradas:**

| Pasta | Conteúdo |
|-------|----------|
| `1_EDA/` | Gráficos exploratórios + estatísticas descritivas |
| `2_PREPROCESSAMENTO/` | Treino/teste + preprocessador |
| `3_MODELAGEM/` | Modelos `.pkl`, métricas, gráficos comparativos |
| `4_INTERPRETABILIDADE/` | Coeficientes, permutation, PDP, SHAP |

**Documentação completa:** [`Regressão/REGRESSAO_NBA.md`](Regressão/REGRESSAO_NBA.md)

**Dependências Python:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `shap`, `joblib`, `scipy`

---

## Apresentação — como visualizar

Abra diretamente no navegador (offline):

```
apresentacao/index.html
```

Para editar e regenerar o HTML autocontido:

```bash
# Edite source.html, css/ e js/
python3 apresentacao/build_standalone.py
```
