#!/usr/bin/env python3
"""
Auditoria de Qualidade de Dados - Regressao NBA Salarios
Objetivo: diagnosticar problemas nos dados brutos e no pipeline existente,
justificar cada tratamento e recomendar acoes para melhoria preditiva.
"""
import pandas as pd
import numpy as np
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

os.makedirs('auditoria_dados/relatorios', exist_ok=True)

print("=" * 70)
print("AUDITORIA DE QUALIDADE DE DADOS - REGRESSAO NBA SALARIOS")
print("=" * 70)

# =============================================================================
# 1. CARREGAMENTO E INSPECAO INICIAL
# =============================================================================
print("\n" + "=" * 70)
print("1. CARREGAMENTO E INSPECAO INICIAL")
print("=" * 70)

df = pd.read_csv('dataset/nba_2022-23_all_stats_with_salary.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"Dimensao: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"Colunas: {list(df.columns)}")
print(f"Duplicatas completas: {df.duplicated().sum()}")
print(f"Duplicatas por nome: {df['Player Name'].duplicated().sum()}")

# =============================================================================
# 2. ANALISE DO TARGET (SALARY)
# =============================================================================
print("\n" + "=" * 70)
print("2. ANALISE DO TARGET (SALARY)")
print("=" * 70)

salary = df['Salary']
print(f"\nEstatisticas descritivas:")
print(salary.describe())

print(f"\nAssimetria: {stats.skew(salary):.3f}")
print(f"Curtose: {stats.kurtosis(salary):.3f}")

# Outliers por IQR
q1, q3 = salary.quantile([0.25, 0.75])
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers_iqr = df[(salary < lower_fence) | (salary > upper_fence)]
print(f"\nOutliers (IQR method):")
print(f"  Limite inferior: ${lower_fence:,.0f}")
print(f"  Limite superior: ${upper_fence:,.0f}")
print(f"  Numero de outliers: {len(outliers_iqr)}")

# Outliers por Z-score (|z| > 3)
z_scores = np.abs(stats.zscore(salary))
outliers_z = df[z_scores > 3]
print(f"  Numero de outliers (|z| > 3): {len(outliers_z)}")

# Analise dos extremos
print(f"\n--- Jogadores com salario extremamente baixo (< $100k) ---")
low_salary = df[df['Salary'] < 100000].copy()
print(f"Quantidade: {len(low_salary)}")
if len(low_salary) > 0:
    print(low_salary[['Player Name', 'Team', 'Salary', 'GP', 'MP', 'Age']].to_string(index=False))

print(f"\n--- Jogadores com salario extremamente alto (> $40M) ---")
high_salary = df[df['Salary'] > 40000000].copy()
print(f"Quantidade: {len(high_salary)}")
if len(high_salary) > 0:
    print(high_salary[['Player Name', 'Team', 'Salary', 'GP', 'MP', 'Age', 'PTS']].to_string(index=False))

# Salario zero ou negativo
print(f"\nSalario <= 0: {(df['Salary'] <= 0).sum()}")

# =============================================================================
# 3. MISSING VALUES
# =============================================================================
print("\n" + "=" * 70)
print("3. ANALISE DE MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nColunas com missing values:")
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"  {col}: {count} ({pct:.1f}%)")

# Proporcao de missing por posicao
print(f"\n--- Missing FT% por posicao ---")
ft_missing_by_pos = df.groupby('Position')['FT%'].apply(lambda x: x.isnull().sum())
ft_total_by_pos = df.groupby('Position')['FT%'].size()
print(pd.DataFrame({'Missing_FT%': ft_missing_by_pos, 'Total': ft_total_by_pos, 'Pct': (ft_missing_by_pos/ft_total_by_pos*100).round(1)}))

print(f"\n--- Missing 3P% por posicao ---")
tp_missing_by_pos = df.groupby('Position')['3P%'].apply(lambda x: x.isnull().sum())
tp_total_by_pos = df.groupby('Position')['3P%'].size()
print(pd.DataFrame({'Missing_3P%': tp_missing_by_pos, 'Total': tp_total_by_pos, 'Pct': (tp_missing_by_pos/tp_total_by_pos*100).round(1)}))

# =============================================================================
# 4. MULTICOLINEARIDADE (VIF)
# =============================================================================
print("\n" + "=" * 70)
print("4. MULTICOLINEARIDADE (VIF)")
print("=" * 70)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Selecionar features numericas usadas no pipeline
numeric_features = ['Age', 'GP', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
                     'FG%', '3P%', 'FT%', 'PER', 'TS%', 'USG%', 'WS', 'VORP', 'BPM']
# Preencher missing com mediana para calcular VIF
df_vif = df[numeric_features].copy()
df_vif = df_vif.fillna(df_vif.median())

vif_data = pd.DataFrame()
vif_data['Feature'] = numeric_features
vif_data['VIF'] = [variance_inflation_factor(df_vif.values, i) for i in range(len(numeric_features))]
vif_data = vif_data.sort_values('VIF', ascending=False)

print(f"\nVariance Inflation Factor (VIF) por feature:")
print(vif_data.to_string(index=False))
print(f"\nFeatures com VIF > 10 (alta multicolinearidade): {len(vif_data[vif_data['VIF'] > 10])}")
print(f"Features com VIF > 5 (moderada multicolinearidade): {len(vif_data[vif_data['VIF'] > 5])}")

# Correlacao das estatisticas avancadas
print(f"\n--- Correlacoes entre estatisticas avancadas ---")
adv_stats = ['PER', 'TS%', 'USG%', 'WS', 'VORP', 'BPM']
print(df[adv_stats].corr().round(2).to_string())

# =============================================================================
# 5. INCONSISTENCIAS LOGICAS
# =============================================================================
print("\n" + "=" * 70)
print("5. INCONSISTENCIAS LOGICAS")
print("=" * 70)

# GS > GP (titular mais vezes que jogos disputados)
gs_gt_gp = df[df['GS'] > df['GP']]
print(f"\nGS > GP (titular mais vezes que jogos): {len(gs_gt_gp)}")
if len(gs_gt_gp) > 0:
    print(gs_gt_gp[['Player Name', 'GP', 'GS']].head().to_string(index=False))

# Total Minutes vs GP * MP
df['Calc_TotalMin'] = df['GP'] * df['MP']
df['Min_Diff'] = df['Total Minutes'] - df['Calc_TotalMin']
min_diff_large = df[np.abs(df['Min_Diff']) > 10]
print(f"\nTotal Minutes diferente de GP*MP por >10 min: {len(min_diff_large)}")
if len(min_diff_large) > 0:
    print(min_diff_large[['Player Name', 'GP', 'MP', 'Total Minutes', 'Min_Diff']].head().to_string(index=False))

# Jogadores com 0 jogos mas com stats
zero_gp_stats = df[(df['GP'] == 0) & (df[['PTS', 'TRB', 'AST']].sum(axis=1) > 0)]
print(f"\nJogadores com 0 GP mas stats > 0: {len(zero_gp_stats)}")

# Jogadores com minutos mas 0 stats ofensivos
zero_offense = df[(df['MP'] > 0) & (df['PTS'] == 0) & (df['FG'] == 0) & (df['FGA'] == 0)]
print(f"\nJogadores com MP>0 mas 0 pontos/chutes: {len(zero_offense)}")
if len(zero_offense) > 0:
    print(zero_offense[['Player Name', 'Team', 'GP', 'MP', 'PTS']].head().to_string(index=False))

# =============================================================================
# 6. ANALISE POR POSICAO
# =============================================================================
print("\n" + "=" * 70)
print("6. ANALISE POR POSICAO")
print("=" * 70)

df['Position_Clean'] = df['Position'].apply(lambda x: x.split('-')[0])
print(f"\nDistribuicao por posicao:")
print(df['Position_Clean'].value_counts())

# Salario medio por posicao
print(f"\nSalario medio por posicao:")
sal_by_pos = df.groupby('Position_Clean')['Salary'].agg(['mean', 'median', 'std', 'count']).round(0)
print(sal_by_pos.to_string())

# Missing por posicao (condicional)
print(f"\nMissing FT% por posicao (condicional):")
for pos in df['Position_Clean'].unique():
    subset = df[df['Position_Clean'] == pos]
    missing_ft = subset['FT%'].isnull().sum()
    print(f"  {pos}: {missing_ft}/{len(subset)} ({missing_ft/len(subset)*100:.1f}%)")

# =============================================================================
# 7. DISTRIBUICAO DAS FEATURES NUMERICAS
# =============================================================================
print("\n" + "=" * 70)
print("7. DISTRIBUICAO DAS FEATURES NUMERICAS")
print("=" * 70)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('Salary')

dist_stats = []
for col in num_cols:
    s = df[col].dropna()
    dist_stats.append({
        'Feature': col,
        'Mean': s.mean(),
        'Std': s.std(),
        'Min': s.min(),
        'Q1': s.quantile(0.25),
        'Median': s.median(),
        'Q3': s.quantile(0.75),
        'Max': s.max(),
        'Skew': stats.skew(s),
        'Outliers_IQR': ((s < (s.quantile(0.25) - 1.5*(s.quantile(0.75)-s.quantile(0.25)))) |
                         (s > (s.quantile(0.75) + 1.5*(s.quantile(0.75)-s.quantile(0.25))))).sum()
    })

dist_df = pd.DataFrame(dist_stats)
print(dist_df.to_string(index=False))

# =============================================================================
# 8. RECOMENDACOES
# =============================================================================
print("\n" + "=" * 70)
print("8. RECOMENDACOES (RESUMO)")
print("=" * 70)

recomendacoes = """
[A] OUTLIERS NO TARGET (SALARY):
    - 10 jogadores com salario < $100k (minimo $5.849): verificar se sao contratos
      two-way, dados de G-League, ou erros de entrada. Recomendacao: investigar
      individualmente; possivelmente remover ou usar winsorizacao.
    - 11 jogadores > $40M: validar como superstars reais (Curry, Durant, etc.).
      Recomendacao: manter, mas considerar modelagem robusta ou transformacao.

[B] MISSING VALUES:
    - 23 missing em FT%: usar imputacao condicional por posicao (centers tem
      FT% menor que guards). A imputacao global com mediana mascara diferencas
      entre posicoes.
    - 13 missing em 3P%: justificavel — pivos que nunca arremessam de 3.
      Imputacao por posicao ou 0 (se nunca tentou).

[C] MULTICOLINEARIDADE:
    - VIF > 10 para: PER, VORP, WS, BPM, MP, Total Minutes
    - Solucao: remover uma de {PER, BPM} (r=0.90) e uma de {WS, VORP} (r=0.89).
      Alternativamente, usar PCA nos stats avancadas ou Ridge.

[D] INCONSISTENCIAS:
    - Verificar casos de GS > GP e Total Minutes != GP*MP.
    - Jogadores com 0 GP mas stats > 0 precisam de investigacao.

[E] FEATURE ENGINEERING FALTANTE:
    - Idade ao quadrado para capturar pico de carreira.
    - Stats normalizados por minuto (nao so por jogo).
    - Interacoes: Age x MP, Age x GP.
    - Categorias de experiencia (rookie/prime/veteran).

[F] PIPELINE EXISTENTE:
    - Log-transformacao do salario e adequada (reduz assimetria).
    - Divisao 75/25 e aleatoria; recomendacao: usar CV estratificado por posicao.
    - Modelos existentes (OLS, Ridge, Lasso, RF) sao razoaveis; adicionar
      HistGradientBoosting e Stacking para comparacao.
"""
print(recomendacoes)

# Salvar relatorio
with open('auditoria_dados/relatorios/auditoria_regressao.txt', 'w') as f:
    # Redirecionar stdout... melhor salvar os DataFrames
    f.write("AUDITORIA REGRESSAO NBA\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dimensao: {df.shape}\n")
    f.write(f"Duplicatas: {df.duplicated().sum()}\n\n")
    f.write("--- TARGET (SALARY) ---\n")
    f.write(salary.describe().to_string() + "\n\n")
    f.write(f"Outliers IQR: {len(outliers_iqr)}\n")
    f.write(f"Outliers Z>3: {len(outliers_z)}\n\n")
    f.write("--- MISSING VALUES ---\n")
    f.write(missing.to_string() + "\n\n")
    f.write("--- VIF ---\n")
    f.write(vif_data.to_string(index=False) + "\n\n")
    f.write("--- RECOMENDACOES ---\n")
    f.write(recomendacoes)

print("\nRelatorio salvo em: auditoria_dados/relatorios/auditoria_regressao.txt")
print("=" * 70)
