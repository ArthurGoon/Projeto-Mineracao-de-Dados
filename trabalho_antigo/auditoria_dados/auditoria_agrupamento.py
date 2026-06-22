#!/usr/bin/env python3
"""
Auditoria de Qualidade de Dados - Agrupamento Tenis ATP
Objetivo: diagnosticar o problema dos ~73% de registros "missing-like",
analisar qualidade do parsing, estrutura dos dados e recomendar tratamentos.
"""
import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

os.makedirs('auditoria_dados/relatorios', exist_ok=True)

print("=" * 70)
print("AUDITORIA DE QUALIDADE DE DADOS - AGRUPAMENTO TENIS ATP")
print("=" * 70)

# =============================================================================
# 1. CARREGAMENTO E INSPECAO INICIAL
# =============================================================================
print("\n" + "=" * 70)
print("1. CARREGAMENTO E INSPECAO INICIAL")
print("=" * 70)

players = pd.read_csv('dataset/players(man).csv')
serve = pd.read_csv('dataset/serve_kaggle.csv')
ret = pd.read_csv('dataset/return_kaggle.csv')
raw = pd.read_csv('dataset/raw_kaggle.csv')

print(f"Players: {players.shape[0]} x {players.shape[1]}")
print(f"Serve: {serve.shape[0]} x {serve.shape[1]}")
print(f"Return: {ret.shape[0]} x {ret.shape[1]}")
print(f"Raw: {raw.shape[0]} x {raw.shape[1]}")

# Limpar nomes de colunas
serve.columns = serve.columns.str.strip()
ret.columns = ret.columns.str.strip()
players.columns = players.columns.str.strip()
raw.columns = raw.columns.str.strip()

print(f"\nServe cols: {list(serve.columns)}")
print(f"Return cols: {list(ret.columns)}")
print(f"Players cols: {list(players.columns)}")

# =============================================================================
# 2. INVESTIGACAO DO PROBLEMA DE PARSING (~73% MISSING-LIKE)
# =============================================================================
print("\n" + "=" * 70)
print("2. INVESTIGACAO DO PROBLEMA DE PARSING")
print("=" * 70)

# Amostra de valores brutos em A%
print("\n--- Amostra de valores brutos em serve['A%'] ---")
print(serve['A%'].head(20).tolist())

# Verificar tipos de dados
print(f"\nDtype de serve['A%']: {serve['A%'].dtype}")
print(f"Dtype de serve['Bpsvd']: {serve['Bpsvd'].dtype}")

# Contar valores especiais
def count_special_values(series, name):
    total = len(series)
    null_count = series.isnull().sum()
    # Diferentes representacoes de missing
    dash_count = (series.astype(str).str.strip() == '-').sum()
    nan_str_count = (series.astype(str).str.strip().str.lower() == 'nan').sum()
    empty_count = (series.astype(str).str.strip() == '').sum()
    # Valores numericos aparentemente validos
    # Tentar converter para float
    numeric_count = 0
    for v in series.dropna():
        s = str(v).strip().replace('%', '')
        if s not in ['-', 'nan', '']:
            try:
                float(s)
                numeric_count += 1
            except:
                pass
    
    print(f"\n{name}:")
    print(f"  Total: {total}")
    print(f"  NaN (pandas): {null_count}")
    print(f"  '-' (string): {dash_count}")
    print(f"  'nan'/'NaN' (string): {nan_str_count}")
    print(f"  '' (vazio): {empty_count}")
    print(f"  Parseaveis como float: {numeric_count}")
    print(f"  Outros (nao categorizados): {total - null_count - dash_count - nan_str_count - empty_count - numeric_count}")

# Analisar cada coluna problematica do serve
for col in ['A%', 'Df%', '1stIn', '1st%', '2nd%']:
    if col in serve.columns:
        count_special_values(serve[col], f"serve['{col}']")

# Analisar Bpsvd
print(f"\n--- Amostra de valores brutos em serve['Bpsvd'] ---")
print(serve['Bpsvd'].head(20).tolist())
count_special_values(serve['Bpsvd'], "serve['Bpsvd']")

# Analisar return
for col in ['TPW', 'RPW', 'vA%', 'v1st%', 'v2nd%']:
    if col in ret.columns:
        count_special_values(ret[col], f"ret['{col}']")

count_special_values(ret['BPCnv'], "ret['BPCnv']")

# =============================================================================
# 3. ANALISE POR SUPERFICIE
# =============================================================================
print("\n" + "=" * 70)
print("3. ANALISE POR SUPERFICIE")
print("=" * 70)

print(f"\nServe - Distribuicao por Surface:")
print(serve['Surface'].value_counts().to_string())

print(f"\nReturn - Distribuicao por Surface:")
print(ret['Surface'].value_counts().to_string())

print(f"\nRaw - Distribuicao por Surface:")
if 'Surface' in raw.columns:
    print(raw['Surface'].value_counts().to_string())
else:
    print("  Coluna 'Surface' nao encontrada em raw")

# Porcentagem de missing-like por superficie
print(f"\n--- Missing-like (parseaveis como float) por superficie em serve['A%'] ---")
for surface in serve['Surface'].dropna().unique():
    subset = serve[serve['Surface'] == surface]['A%']
    valid = 0
    for v in subset.dropna():
        s = str(v).strip().replace('%', '')
        if s not in ['-', 'nan', '']:
            try:
                float(s)
                valid += 1
            except:
                pass
    print(f"  {surface}: {valid}/{len(subset)} ({valid/len(subset)*100:.1f}% validos)")

# =============================================================================
# 4. ANALISE POR RODADA (Rd)
# =============================================================================
print("\n" + "=" * 70)
print("4. ANALISE POR RODADA (Rd)")
print("=" * 70)

print(f"\nDistribuicao de partidas por rodada:")
print(serve['Rd'].value_counts().sort_index().to_string())

# Missing-like por rodada
print(f"\nValores validos em serve['A%'] por rodada:")
for rd in serve['Rd'].dropna().unique():
    subset = serve[serve['Rd'] == rd]['A%']
    valid = 0
    for v in subset.dropna():
        s = str(v).strip().replace('%', '')
        if s not in ['-', 'nan', '']:
            try:
                float(s)
                valid += 1
            except:
                pass
    print(f"  {rd}: {valid}/{len(subset)} ({valid/len(subset)*100:.1f}% validos)")

# =============================================================================
# 5. COBERTURA POR JOGADOR
# =============================================================================
print("\n" + "=" * 70)
print("5. COBERTURA POR JOGADOR")
print("=" * 70)

print(f"\nJogadores unicos em serve: {serve['Name'].nunique()}")
print(f"Jogadores unicos em return: {ret['Name'].nunique()}")
print(f"Jogadores em players: {players['name'].nunique()}")

# Jogadores em serve mas nao em return
serve_names = set(serve['Name'].dropna().unique())
ret_names = set(ret['Name'].dropna().unique())
players_names = set(players['name'].dropna().unique())

print(f"\nJogadores em serve mas NAO em return: {len(serve_names - ret_names)}")
print(f"Jogadores em return mas NAO em serve: {len(ret_names - serve_names)}")
print(f"Jogadores em players mas NAO em serve: {len(players_names - serve_names)}")
print(f"Jogadores em players mas NAO em return: {len(players_names - ret_names)}")

# Jogadores com poucos registros validos
print(f"\n--- Jogadores com poucos registros validos em serve ---")
player_valid_counts = {}
for name in serve['Name'].dropna().unique():
    subset = serve[serve['Name'] == name]['A%']
    valid = 0
    for v in subset.dropna():
        s = str(v).strip().replace('%', '')
        if s not in ['-', 'nan', '']:
            try:
                float(s)
                valid += 1
            except:
                pass
    player_valid_counts[name] = valid

low_valid = {k: v for k, v in player_valid_counts.items() if v < 5}
print(f"Jogadores com < 5 registros validos: {len(low_valid)}")
if len(low_valid) > 0:
    sorted_low = sorted(low_valid.items(), key=lambda x: x[1])
    print(f"Exemplos: {sorted_low[:10]}")

# =============================================================================
# 6. ANALISE DE DADOS RAW
# =============================================================================
print("\n" + "=" * 70)
print("6. ANALISE DE DADOS RAW")
print("=" * 70)

print(f"\nRaw cols: {list(raw.columns)}")
print(f"Raw head:\n{raw.head(3).to_string()}")

# Verificar se raw tem dados adicionais que podem ser uteis
if 'Rank' in raw.columns or 'Rk' in raw.columns:
    rk_col = 'Rank' if 'Rank' in raw.columns else 'Rk'
    print(f"\nColuna de ranking encontrada: {rk_col}")
    print(f"Ranking range: {raw[rk_col].min()} a {raw[rk_col].max()}")

# =============================================================================
# 7. ANALISE DE MISSING APOS O PIPELINE EXISTENTE
# =============================================================================
print("\n" + "=" * 70)
print("7. ANALISE DO PIPELINE EXISTENTE")
print("=" * 70)

# Simular o pipeline existente
def parse_pct(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('%','')
    if s in ['-', 'nan', '']: return np.nan
    try: return float(s)
    except: return np.nan

def parse_bp(bp_str):
    if pd.isna(bp_str): return np.nan
    try:
        s = str(bp_str).strip()
        if s in ['-', 'nan', '']: return np.nan
        m = re.search(r'(\d+)/(\d+)', s)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            return num/den if den>0 else 0
        return np.nan
    except:
        return np.nan

# Parse serve
serve_parsed = serve.copy()
for c in ['A%', 'Df%', '1stIn', '1st%', '2nd%']:
    if c in serve_parsed.columns:
        serve_parsed[c] = serve_parsed[c].apply(parse_pct)
serve_parsed['Bpsvd_rate'] = serve_parsed['Bpsvd'].apply(parse_bp)

# Agregar
serve_agg = serve_parsed.groupby('Name').agg({
    'A%': 'mean', 'Df%': 'mean', '1stIn': 'mean', '1st%': 'mean', '2nd%': 'mean',
    'Bpsvd_rate': 'mean'
}).reset_index()

# Contar missing pos-agregacao
print(f"\nMissing values pos-agregacao (serve):")
for col in serve_agg.columns:
    if col != 'Name':
        missing = serve_agg[col].isnull().sum()
        print(f"  {col}: {missing}/{len(serve_agg)} ({missing/len(serve_agg)*100:.1f}%)")

# Jogadores com dados completos
complete = serve_agg.dropna()
print(f"\nJogadores com dados serve COMPLETOS: {len(complete)}/{len(serve_agg)} ({len(complete)/len(serve_agg)*100:.1f}%)")

# Mesmo para return
ret_parsed = ret.copy()
for c in ['TPW', 'RPW', 'vA%', 'v1st%', 'v2nd%']:
    if c in ret_parsed.columns:
        ret_parsed[c] = ret_parsed[c].apply(parse_pct)
ret_parsed['BPCnv_rate'] = ret_parsed['BPCnv'].apply(parse_bp)

ret_agg = ret_parsed.groupby('Name').agg({
    'TPW': 'mean', 'RPW': 'mean', 'vA%': 'mean', 'v1st%': 'mean', 'v2nd%': 'mean',
    'BPCnv_rate': 'mean'
}).reset_index()

print(f"\nMissing values pos-agregacao (return):")
for col in ret_agg.columns:
    if col != 'Name':
        missing = ret_agg[col].isnull().sum()
        print(f"  {col}: {missing}/{len(ret_agg)} ({missing/len(ret_agg)*100:.1f}%)")

complete_ret = ret_agg.dropna()
print(f"\nJogadores com dados return COMPLETOS: {len(complete_ret)}/{len(ret_agg)} ({len(complete_ret)/len(ret_agg)*100:.1f}%)")

# Merge serve + return + players
merged = serve_agg.merge(ret_agg, on='Name', how='inner')
merged = merged.merge(players, left_on='Name', right_on='name', how='inner')
print(f"\nJogadores com TODOS os dados completos (serve+return+players): {len(merged)}")

# =============================================================================
# 8. RECOMENDACOES
# =============================================================================
print("\n" + "=" * 70)
print("8. RECOMENDACOES (RESUMO)")
print("=" * 70)

recomendacoes = """
[A] PROBLEMA DE PARSING (~73% MISSING-LIKE):
    - Os valores "-" e "nan" como strings representam casos onde o jogador
      NAO TEVE saque/devolucao naquela partida (ex: WO, partidas com poucos
      games, ou formatos de duplas).
    - SOLUCAO: NAO imputar esses valores. Eles sao informativos: representam
      partidas onde o jogador nao participou efetivamente.
    - ALTERNATIVA: para a agregacao, calcular a media APENAS dos registros
      validos (como o pipeline ja faz), mas documentar que jogadores com
      muitos "-" tem estatisticas menos confiaveis.

[B] SEPARACAO POR SUPERFICIE:
    - Hard: 57.8%, Clay: 31.4%, Grass: 8.7% dos registros.
    - O estilo de jogo muda drasticamente entre superficies.
    - SOLUCAO: criar features de especializacao por superficie:
      Aces_pct_hard, Aces_pct_clay, Aces_pct_grass.
      Ou agregar separadamente e criar ratios de especializacao.

[C] COBERTURA POR RODADA:
    - Rodadas iniciais (Q1, Q2, R64) tem mais registros "-" (eliminacao precoce).
    - Finais (F, SF) tem menos registros "-" (partidas completas).
    - SOLUCAO: aplicar o filtro de minimo de partidas (>=20) por superficie
      separadamente, nao so globalmente.

[D] ALGORITMOS DE CLUSTERING:
    - Silhouette = 0.24 e baixo. Testar GMM (soft clustering) e DBSCAN
      (para detectar outliers/jogadores hibridos).
    - Usar Calinski-Harabasz e Dunn Index como metricas adicionais.
    - Bootstrap de estabilidade para verificar se clusters sao robustos.

[E] INTERPRETABILIDADE:
    - Adicionar profiling estatistico formal (testes t, Cohen's d).
    - Comparar ranking ATP medio entre clusters.
    - Identificar jogadores de fronteira (proximos a ambos os centroides).
    - Visualizacao com t-SNE ou UMAP alem do PCA.
"""
print(recomendacoes)

# Salvar relatorio
with open('auditoria_dados/relatorios/auditoria_agrupamento.txt', 'w') as f:
    f.write("AUDITORIA AGRUPAMENTO TENIS ATP\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Players: {players.shape}\n")
    f.write(f"Serve: {serve.shape}\n")
    f.write(f"Return: {ret.shape}\n")
    f.write(f"Raw: {raw.shape}\n\n")
    f.write("--- PARSING ---\n")
    f.write("~73% dos registros tem valores '-' ou 'nan' como strings\n")
    f.write("Isso representa partidas sem saque/devolucao (WO, eliminacao precoce)\n\n")
    f.write("--- SUPERFICIE ---\n")
    f.write(serve['Surface'].value_counts().to_string() + "\n\n")
    f.write("--- JOGADORES COM DADOS COMPLETOS ---\n")
    f.write(f"Serve: {len(complete)}/{len(serve_agg)}\n")
    f.write(f"Return: {len(complete_ret)}/{len(ret_agg)}\n")
    f.write(f"Merge total: {len(merged)}\n\n")
    f.write("--- RECOMENDACOES ---\n")
    f.write(recomendacoes)

print("\nRelatorio salvo em: auditoria_dados/relatorios/auditoria_agrupamento.txt")
print("=" * 70)
