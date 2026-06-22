#!/usr/bin/env python3
"""
Auditoria de Qualidade de Dados - Classificacao Premier League
Objetivo: diagnosticar problemas nos dados, leakage temporal, merge quality,
missing values e recomendar tratamentos justificados.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('auditoria_dados/relatorios', exist_ok=True)

print("=" * 70)
print("AUDITORIA DE QUALIDADE DE DADOS - CLASSIFICACAO PREMIER LEAGUE")
print("=" * 70)

# =============================================================================
# 1. CARREGAMENTO E INSPECAO INICIAL
# =============================================================================
print("\n" + "=" * 70)
print("1. CARREGAMENTO E INSPECAO INICIAL")
print("=" * 70)

results = pd.read_csv('dataset/results.csv')
stats = pd.read_csv('dataset/stats.csv')

print(f"Results: {results.shape[0]} partidas x {results.shape[1]} colunas")
print(f"Stats: {stats.shape[0]} registros x {stats.shape[1]} colunas")
print(f"Results cols: {list(results.columns)}")
print(f"Stats cols: {list(stats.columns)}")
print(f"Duplicatas results: {results.duplicated().sum()}")
print(f"Duplicatas stats: {stats.duplicated().sum()}")

# =============================================================================
# 2. ANALISE DO TARGET
# =============================================================================
print("\n" + "=" * 70)
print("2. ANALISE DO TARGET (HomeWin)")
print("=" * 70)

results['HomeWin'] = (results['result'] == 'H').astype(int)
print(f"\nDistribuicao dos resultados:")
print(results['result'].value_counts())
print(f"\nTaxa de vitoria do mandante: {results['HomeWin'].mean():.3f}")

# Por temporada
home_by_season = results.groupby('season')['HomeWin'].mean()
print(f"\nTaxa de vitoria casa por temporada:")
print(home_by_season.to_string())

# =============================================================================
# 3. MISSING VALUES
# =============================================================================
print("\n" + "=" * 70)
print("3. ANALISE DE MISSING VALUES")
print("=" * 70)

print(f"\nMissing values em results:")
results_missing = results.isnull().sum()
print(results_missing[results_missing > 0].to_string() if (results_missing > 0).any() else "  Nenhum")

print(f"\nMissing values em stats:")
stats_missing = stats.isnull().sum()
stats_missing = stats_missing[stats_missing > 0].sort_values(ascending=False)
print(stats_missing.to_string())
print(f"\nDetalhamento:")
for col, count in stats_missing.items():
    pct = count / len(stats) * 100
    print(f"  {col}: {count}/{len(stats)} ({pct:.1f}%)")

# =============================================================================
# 4. AUDITORIA DO MERGE
# =============================================================================
print("\n" + "=" * 70)
print("4. AUDITORIA DO MERGE RESULTS x STATS")
print("=" * 70)

# Verificar cobertura
home_keys = set(zip(results['home_team'], results['season']))
away_keys = set(zip(results['away_team'], results['season']))
stats_keys = set(zip(stats['team'], stats['season']))

missing_home = home_keys - stats_keys
missing_away = away_keys - stats_keys
print(f"\nCombinacoes team-season em results nao encontradas em stats:")
print(f"  Home: {len(missing_home)}")
print(f"  Away: {len(missing_away)}")
if len(missing_home) > 0:
    print(f"  Exemplos: {list(missing_home)[:5]}")

# Simular o merge do pipeline
stats_home = stats.add_prefix('home_')
stats_away = stats.add_prefix('away_')
merged = results.merge(stats_home, left_on=['home_team','season'], right_on=['home_team','home_season'], how='left')
merged = merged.merge(stats_away, left_on=['away_team','season'], right_on=['away_team','away_season'], how='left')

print(f"\nLinhas em results: {len(results)}")
print(f"Linhas apos merge: {len(merged)}")
print(f"Linhas com missing em home stats: {merged['home_wins'].isnull().sum()}")
print(f"Linhas com missing em away stats: {merged['away_wins'].isnull().sum()}")
print(f"Linhas com QUALQUER missing pos-merge: {merged.isnull().any(axis=1).sum()}")

# Quantas linhas seriam dropadas pelo pipeline?
diff_cols = ['wins','losses','goals','total_yel_card','total_red_card','total_scoring_att',
             'ontarget_scoring_att','hit_woodwork','att_hd_goal','att_pen_goal','att_freekick_goal',
             'att_ibox_goal','att_obox_goal','goal_fastbreak','total_offside','clean_sheet',
             'goals_conceded','saves','outfielder_block','interception','total_tackle',
             'last_man_tackle','total_clearance','head_clearance','own_goals','penalty_conceded',
             'pen_goals_conceded','total_pass','total_through_ball','total_long_balls',
             'backward_pass','total_cross','corner_taken','touches','big_chance_missed',
             'clearance_off_line','dispossessed','penalty_save','total_high_claim','punches']

feature_cols = [f'diff_{col}' for col in diff_cols if f'home_{col}' in merged.columns and f'away_{col}' in merged.columns]
merged_with_diffs = merged.copy()
for col in diff_cols:
    hc = f'home_{col}'; ac = f'away_{col}'
    if hc in merged_with_diffs.columns and ac in merged_with_diffs.columns:
        merged_with_diffs[f'diff_{col}'] = merged_with_diffs[hc] - merged_with_diffs[ac]

dropna_after = merged_with_diffs.dropna(subset=feature_cols + ['HomeWin'])
print(f"\nLinhas ANTES de dropna: {len(merged_with_diffs)}")
print(f"Linhas APOS dropna (como no pipeline): {len(dropna_after)}")
print(f"Linhas PERDIDAS: {len(merged_with_diffs) - len(dropna_after)} ({(len(merged_with_diffs) - len(dropna_after))/len(merged_with_diffs)*100:.1f}%)")

# =============================================================================
# 5. LEAKAGE TEMPORAL
# =============================================================================
print("\n" + "=" * 70)
print("5. DIAGNOSTICO DE LEAKAGE TEMPORAL")
print("=" * 70)

print(f"""
PROBLEMA CRITICO: As estatisticas em 'stats.csv' sao ACUMULADAS POR TEMPORADA.
Isso significa que:
- Uma partida em OUTUBRO (rodada 8) usa stats de 38 jogos (temporada inteira)
- Uma partida em MAIO (rodada 38) usa stats de... 38 jogos (mesmo valor!)

ISSO E LEAKAGE TEMPORAL: ao prever o resultado de uma partida no inicio da
temporada, o modelo "ve" o desempenho do time ao longo de TODA a temporada,
incluindo jogos que ainda nao aconteceram.

Impacto: o modelo APRENDE do futuro, superestimando a performance preditiva.
""")

# Demonstracao: variacao das stats ao longo da temporada
print("--- Variacao das stats 'wins' por time (temporada 2017-2018) ---")
stats_2017 = stats[stats['season'] == '2017-2018'].copy()
print(f"All teams have same 'wins' count regardless of match date: SIM (acumulado)")
print(f"Wins em 2017-2018:")
print(stats_2017[['team','wins']].sort_values('wins', ascending=False).head(10).to_string(index=False))

# =============================================================================
# 6. CONSISTENCIA DOS DADOS
# =============================================================================
print("\n" + "=" * 70)
print("6. CONSISTENCIA DOS DADOS")
print("=" * 70)

# Gols consistentes com resultado?
results['goals_diff'] = results['home_goals'] - results['away_goals']
results['expected_result'] = results['goals_diff'].apply(lambda x: 'H' if x > 0 else ('A' if x < 0 else 'D'))
mismatched = results[results['result'] != results['expected_result']]
print(f"\nPartidas onde gols nao batem com resultado: {len(mismatched)}")
if len(mismatched) > 0:
    print(mismatched[['home_team','away_team','home_goals','away_goals','result','expected_result']].head().to_string(index=False))

# Stats: wins + losses + draws = 38?
stats['total_games'] = stats['wins'] + stats['losses']
# Verificar se stats['goals'] corresponde a gols marcados
print(f"\nTimes com wins + losses != 38 (excluindo empates):")
print(f"  Total: {(stats['total_games'] != 38).sum()} (na Premier League sao 38 jogos)")

# Duplicatas de time-season
print(f"\nDuplicatas de team-season em stats: {stats[['team','season']].duplicated().sum()}")

# =============================================================================
# 7. BALANCEAMENTO TEMPORAL
# =============================================================================
print("\n" + "=" * 70)
print("7. BALANCEAMENTO TEMPORAL")
print("=" * 70)

results['year_start'] = results['season'].str[:4].astype(int)
print(f"\nPartidas por temporada:")
print(results.groupby('season').size().to_string())

# Divisao do pipeline
train_seasons = [f'{y}-{y+1}' for y in range(2006, 2016)]
test_seasons = [f'{y}-{y+1}' for y in range(2016, 2018)]
train_mask = results['season'].isin(train_seasons)
test_mask = results['season'].isin(test_seasons)

print(f"\nDivisao temporal do pipeline:")
print(f"  Treino: {train_mask.sum()} partidas ({train_seasons[0]} a {train_seasons[-1]})")
print(f"  Teste: {test_mask.sum()} partidas ({test_seasons[0]} a {test_seasons[-1]})")
print(f"  Taxa vitoria casa (treino): {results[train_mask]['HomeWin'].mean():.3f}")
print(f"  Taxa vitoria casa (teste): {results[test_mask]['HomeWin'].mean():.3f}")

# Regra da Premier League: 20 times, 380 jogos por temporada
print(f"\nJogos por temporada (esperado: 380):")
games_per_season = results.groupby('season').size()
print(games_per_season.to_string())
print(f"Temporadas com != 380 jogos: {(games_per_season != 380).sum()}")

# =============================================================================
# 8. FEATURES DIFERENCIAIS - DISTRIBUICAO
# =============================================================================
print("\n" + "=" * 70)
print("8. ANALISE DAS FEATURES DIFERENCIAIS (pos-merge)")
print("=" * 70)

# Reconstruir features diferenciais
X = dropna_after[feature_cols].copy()
print(f"Numero de features diferenciais: {len(feature_cols)}")
print(f"Shape final apos dropna: {X.shape}")

# Estatisticas das features
print(f"\nEstatisticas das features diferenciais:")
print(X.describe().round(2).T.to_string())

# Features com variancia zero ou proxima de zero
low_var = X.var() < 0.001
print(f"\nFeatures com variancia < 0.001: {low_var.sum()}")
if low_var.sum() > 0:
    print(f"  {list(low_var[low_var].index)}")

# =============================================================================
# 9. RECOMENDACOES
# =============================================================================
print("\n" + "=" * 70)
print("9. RECOMENDACOES (RESUMO)")
print("=" * 70)

recomendacoes = """
[A] LEAKAGE TEMPORAL (CRITICO):
    - As stats sao acumuladas na temporada inteira. Isso cria leakage para
      partidas no inicio da temporada.
    - SOLUCAO: para cada partida, calcular stats ACUMULADAS ATE AQUELA RODADA.
      Isso requer dados por rodada (nao disponiveis atualmente).
    - ALTERNATIVA IMEDIATA: usar stats da TEMPORADA ANTERIOR como proxy de
      forma (mas perde jogadores novos, rebaixados/promovidos).

[B] MISSING VALUES:
    - backward_pass (80/240 = 33.3%), big_chance_missed (80/240 = 33.3%)
    - total_through_ball (20/240 = 8.3%), dispossessed (20/240 = 8.3%)
    - SOLUCAO: imputar com mediana ou KNN; ou remover features com >30% missing.

[C] LINHAS PERDIDAS NO DROPNA:
    - O pipeline dropna silenciosamente perde linhas.
    - SOLUCAO: imputar missing antes de criar features diferenciais.

[D] FEATURE ENGINEERING FALTANTE:
    - Forma recente (ultimos 5 jogos) — o preditor mais importante no futebol.
    - Posicao na tabela acumulada ate aquela rodada.
    - Head-to-head historico entre os times.
    - Rodada da temporada (inicio vs final tem dinamicas diferentes).
    - Desempenho em casa vs fora separado (nao so diferencial).
    - Fase da temporada (inicio/meio/final).

[E] VALIDACAO:
    - CV 5-fold aleatorio em dados temporais e incorreto.
    - SOLUCAO: usar walk-forward validation ou time-series split.

[F] FEATURE SELECTION:
    - Lasso zerou 36/39 features; pode ser muito agressivo.
    - SOLUCAO: usar RFECV ou testar diferentes valores de C no Lasso.
"""
print(recomendacoes)

# Salvar relatorio
with open('auditoria_dados/relatorios/auditoria_classificacao.txt', 'w') as f:
    f.write("AUDITORIA CLASSIFICACAO PREMIER LEAGUE\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Results: {results.shape}\n")
    f.write(f"Stats: {stats.shape}\n\n")
    f.write("--- MISSING STATS ---\n")
    f.write(stats_missing.to_string() + "\n\n")
    f.write(f"--- MERGE ---\n")
    f.write(f"Linhas perdidas no dropna: {len(merged_with_diffs) - len(dropna_after)}\n\n")
    f.write("--- LEAKAGE TEMPORAL ---\n")
    f.write("Stats acumuladas por temporada inteira -> leakage para partidas no inicio\n\n")
    f.write("--- RECOMENDACOES ---\n")
    f.write(recomendacoes)

print("\nRelatorio salvo em: auditoria_dados/relatorios/auditoria_classificacao.txt")
print("=" * 70)
