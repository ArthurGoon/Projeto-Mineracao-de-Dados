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
├── relatorio/             # Relatório PDF de entrega (LaTeX + figuras)
├── apresentacao/          # Deck scrollytelling (HTML final + fonte)
├── Regressão/             # Dataset, script único e resultados
└── README.md
```

---

## Regressão — como reproduzir

```bash
cd Regressão
pip install -r requirements.txt
python3 run_regressao_nba_completo.py
```

Para atualizar a apresentação após rodar o pipeline:

```bash
python3 sync_apresentacao_data.py
python3 ../apresentacao/build_standalone.py
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

**Dependências:** `Regressão/requirements.txt`

---

## Relatório — PDF de entrega

Arquivo final: [`relatorio/main.pdf`](relatorio/main.pdf) (máx. 8 páginas).

Para recompilar após editar o texto ou trocar figuras:

```bash
cd relatorio
pdflatex main.tex
```

As figuras em `relatorio/` são PDFs vetoriais gerados em `Regressão/4_INTERPRETABILIDADE/figuras_relatorio/` e copiadas para `relatorio/` ao rodar o pipeline.

Para validar números e tabelas do relatório contra os artefatos do pipeline:

```bash
cd Regressão
python3 auditoria_relatorio.py
```

---

## Apresentação — como visualizar

Abra diretamente no navegador (offline):

```
apresentacao/index.html
```

Para editar e regenerar o HTML autocontido:

```bash
cd Regressão
python3 run_regressao_nba_completo.py
python3 sync_apresentacao_data.py
python3 ../apresentacao/build_standalone.py
```
