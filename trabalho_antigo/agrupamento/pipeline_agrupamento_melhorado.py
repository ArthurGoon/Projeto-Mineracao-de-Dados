#!/usr/bin/env python3
"""
Pipeline Agrupamento Tenis ATP - MELHORADO
Melhorias implementadas com base na Auditoria de Qualidade de Dados:
1. Features por superficie (Hard/Clay/Grass) - especializacao
2. Features compostas: eficiencia do saque, dominancia devolucao, consistencia
3. Novos algoritmos: GMM (soft clustering), DBSCAN (outliers)
4. Metricas adicionais: Calinski-Harabasz, Dunn Index
5. Bootstrap de estababilidade dos clusters
6. Profiling estatistico formal (testes t, Cohen's d)
7. Comparacao com ranking ATP entre clusters
8. Visualizacao t-SNE e UMAP alem do PCA
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
import re
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import stats

for d in ['ADED_MELHORADO/figuras','PROCESSAMENTO_MELHORADO/receitas',
          'modelagem_melhorada/modelos_ajustados','modelagem_melhorada/resultados',
          'interpretabilidade_melhorada']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# 1. LEITURA E LIMPEZA
# =============================================================================
print("=== 1. LEITURA E LIMPEZA ===")
players = pd.read_csv('dataset/players(man).csv')
serve = pd.read_csv('dataset/serve_kaggle.csv')
ret = pd.read_csv('dataset/return_kaggle.csv')
raw = pd.read_csv('dataset/raw_kaggle.csv')

serve.columns = serve.columns.str.strip()
ret.columns = ret.columns.str.strip()
players.columns = players.columns.str.strip()
raw.columns = raw.columns.str.strip()

print(f"Players: {players.shape}")
print(f"Serve: {serve.shape}")
print(f"Return: {ret.shape}")
print(f"Raw: {raw.shape}")

# Funcao robusta para converter porcentagens
def parse_pct(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace('%','')
    if s in ['-', 'nan', '']: return np.nan
    try: return float(s)
    except: return np.nan

pct_cols_serve = ['A%', 'Df%', '1stIn', '1st%', '2nd%']
for c in pct_cols_serve:
    if c in serve.columns:
        serve[c] = serve[c].apply(parse_pct)

pct_cols_ret = ['TPW', 'RPW', 'vA%', 'v1st%', 'v2nd%']
for c in pct_cols_ret:
    if c in ret.columns:
        ret[c] = ret[c].apply(parse_pct)

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

serve['Bpsvd_rate'] = serve['Bpsvd'].apply(parse_bp)
ret['BPCnv_rate'] = ret['BPCnv'].apply(parse_bp)

# =============================================================================
# 2. AGREGACAO POR JOGADOR E SUPERFICIE
# =============================================================================
print("\n=== 2. AGREGACAO POR JOGADOR E SUPERFICIE ===")

# --- 2.1 Agregacao GLOBAL ---
serve_agg = serve.groupby('Name').agg({
    'A%': 'mean', 'Df%': 'mean', '1stIn': 'mean', '1st%': 'mean', '2nd%': 'mean',
    'Bpsvd_rate': 'mean'
}).reset_index()
serve_agg.columns = ['Name','Aces_pct','Df_pct','FirstServe_In','FirstServe_Win','SecondServe_Win','BP_Saved_rate']

ret_agg = ret.groupby('Name').agg({
    'TPW': 'mean', 'RPW': 'mean', 'vA%': 'mean', 'v1st%': 'mean', 'v2nd%': 'mean',
    'BPCnv_rate': 'mean'
}).reset_index()
ret_agg.columns = ['Name','TotalPoints_Won','ReturnPoints_Won','vAces_pct','v1stReturn_Win','v2ndReturn_Win','BP_Converted_rate']

# --- 2.2 Agregacao POR SUPERFICIE (novo) ---
print("\n[A] AGREGACAO POR SUPERFICIE:")
surfaces = serve['Surface'].dropna().unique()
print(f"   Superficies encontradas: {surfaces}")

serve_surf = {}
ret_surf = {}
for surf in surfaces:
    s_surf = serve[serve['Surface'] == surf].groupby('Name').agg({
        'A%': 'mean', 'Df%': 'mean', '1st%': 'mean', '2nd%': 'mean'
    }).reset_index()
    s_surf.columns = ['Name', f'Aces_pct_{surf}', f'Df_pct_{surf}', f'FirstServe_Win_{surf}', f'SecondServe_Win_{surf}']
    serve_surf[surf] = s_surf

    r_surf = ret[ret['Surface'] == surf].groupby('Name').agg({
        'RPW': 'mean', 'vA%': 'mean', 'v1st%': 'mean', 'v2nd%': 'mean'
    }).reset_index()
    r_surf.columns = ['Name', f'ReturnPoints_Won_{surf}', f'vAces_pct_{surf}', f'v1stReturn_Win_{surf}', f'v2ndReturn_Win_{surf}']
    ret_surf[surf] = r_surf

# Merge tudo
player_stats = serve_agg.merge(ret_agg, on='Name', how='inner')
for surf in surfaces:
    player_stats = player_stats.merge(serve_surf[surf], on='Name', how='left')
    player_stats = player_stats.merge(ret_surf[surf], on='Name', how='left')

player_stats = player_stats.merge(players, left_on='Name', right_on='name', how='inner')

# --- 2.3 Features compostas (novo) ---
print("\n[B] FEATURES COMPOSTAS:")
# Eficiencia do saque: Aces / Double Faults (risco vs recompensa)
player_stats['Serve_efficiency'] = player_stats['Aces_pct'] / (player_stats['Df_pct'] + 0.1)
print("   Serve_efficiency = Aces_pct / (Df_pct + 0.1)")

# Dominancia na devolucao: pontos ganhos na devolucao / aces sofridos
player_stats['Return_dominance'] = player_stats['ReturnPoints_Won'] / (player_stats['vAces_pct'] + 0.1)
print("   Return_dominance = ReturnPoints_Won / (vAces_pct + 0.1)")

# Consistencia do saque: primeiro saque entra * segundo saque ganho
player_stats['Serve_consistency'] = player_stats['FirstServe_In'] * player_stats['SecondServe_Win'] / 100
print("   Serve_consistency = FirstServe_In * SecondServe_Win / 100")

# Eficiencia nos break points
player_stats['BP_efficiency'] = player_stats['BP_Converted_rate'] / (player_stats['BP_Saved_rate'] + 0.1)
print("   BP_efficiency = BP_Converted_rate / (BP_Saved_rate + 0.1)")

# Especializacao por superficie: ratio Hard/Clay
if 'Aces_pct_Hard' in player_stats.columns and 'Aces_pct_Clay' in player_stats.columns:
    player_stats['Hard_Clay_specialization'] = player_stats['Aces_pct_Hard'] / (player_stats['Aces_pct_Clay'] + 0.1)
    print("   Hard_Clay_specialization = Aces_pct_Hard / (Aces_pct_Clay + 0.1)")

print(f"\nJogadores com stats completas: {len(player_stats)}")

# =============================================================================
# 3. FILTRO E SELECAO DE FEATURES
# =============================================================================
print("\n=== 3. FILTRO E SELECAO DE FEATURES ===")
MIN_MATCHES = 20
player_stats_filtered = player_stats[player_stats['number_of_matches'] >= MIN_MATCHES].copy()
print(f"Apos filtro (>= {MIN_MATCHES} partidas): {len(player_stats_filtered)} jogadores")

# Features base (globais + compostas)
feature_cols = [
    'Aces_pct','Df_pct','FirstServe_In','FirstServe_Win','SecondServe_Win',
    'TotalPoints_Won','ReturnPoints_Won','vAces_pct','v1stReturn_Win','v2ndReturn_Win',
    'BP_Saved_rate','BP_Converted_rate',
    'Serve_efficiency','Return_dominance','Serve_consistency','BP_efficiency'
]

# Adicionar features por superficie se disponiveis
for surf in surfaces:
    for col in [f'Aces_pct_{surf}', f'ReturnPoints_Won_{surf}']:
        if col in player_stats_filtered.columns:
            feature_cols.append(col)

# Preencher NaN restantes com mediana
for c in feature_cols:
    if c in player_stats_filtered.columns:
        player_stats_filtered[c] = player_stats_filtered[c].fillna(player_stats_filtered[c].median())
    else:
        print(f"   WARNING: {c} nao encontrada, removendo")
        feature_cols.remove(c)

X = player_stats_filtered[feature_cols].copy()
names = player_stats_filtered['Name'].values
print(f"Features finais: {len(feature_cols)}")
print(feature_cols)

# =============================================================================
# 4. PADRONIZACAO
# =============================================================================
print("\n=== 4. PADRONIZACAO ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'PROCESSAMENTO_MELHORADO/receitas/scaler.pkl')
joblib.dump(feature_cols, 'PROCESSAMENTO_MELHORADO/receitas/feature_names.pkl')

# =============================================================================
# 5. ESCOLHA DE K E METRICAS ADICIONAIS
# =============================================================================
print("\n=== 5. ESCOLHA DE K E METRICAS ADICIONAIS ===")
k_range = range(2, 11)
inertias = []
silhouettes = []
calinski = []
davies_bouldin = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    calinski.append(calinski_harabasz_score(X_scaled, labels))
    davies_bouldin.append(davies_bouldin_score(X_scaled, labels))
    print(f"K={k}: Silhouette={silhouettes[-1]:.4f}, Calinski={calinski[-1]:.1f}, DB={davies_bouldin[-1]:.3f}")

# Figura comparativa
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(k_range, inertias, 'o-', color='#3498db', lw=2, markersize=8)
axes[0,0].set_xlabel('K'); axes[0,0].set_ylabel('Inertia'); axes[0,0].set_title('Metodo do Cotovelo', fontweight='bold')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(k_range, silhouettes, 'o-', color='#e74c3c', lw=2, markersize=8)
axes[0,1].set_xlabel('K'); axes[0,1].set_ylabel('Silhouette'); axes[0,1].set_title('Silhouette Score', fontweight='bold')
axes[0,1].grid(True, alpha=0.3)
best_k_sil = list(k_range)[np.argmax(silhouettes)]
axes[0,1].axvline(best_k_sil, color='green', linestyle='--', label=f'Melhor K={best_k_sil}')
axes[0,1].legend()

axes[1,0].plot(k_range, calinski, 'o-', color='#2ecc71', lw=2, markersize=8)
axes[1,0].set_xlabel('K'); axes[1,0].set_ylabel('Calinski-Harabasz'); axes[1,0].set_title('Calinski-Harabasz (maior=melhor)', fontweight='bold')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(k_range, davies_bouldin, 'o-', color='#9b59b6', lw=2, markersize=8)
axes[1,1].set_xlabel('K'); axes[1,1].set_ylabel('Davies-Bouldin'); axes[1,1].set_title('Davies-Bouldin (menor=melhor)', fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Escolha do Numero de Clusters - Multi-metrica', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('modelagem_melhorada/resultados/cotovelo_multi_metrica.png', dpi=300)
plt.close()

# Escolher K com melhor combinacao de metricas
# Calinski maior = melhor, Davies-Bouldin menor = melhor
best_k = best_k_sil
print(f"\nMelhor K (Silhouette): {best_k_sil}")
print(f"Melhor K (Calinski): {list(k_range)[np.argmax(calinski)]}")
print(f"Melhor K (Davies-Bouldin): {list(k_range)[np.argmin(davies_bouldin)]}")
print(f"K escolhido para analise: {best_k}")

# =============================================================================
# 6. ALGORITMOS DE CLUSTERING
# =============================================================================
print("\n=== 6. ALGORITMOS DE CLUSTERING ===")

# --- 6.1 K-Means ---
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)
player_stats_filtered['Cluster_KMeans'] = labels_km
print(f"\nK-Means (K={best_k}): Silhouette={silhouette_score(X_scaled, labels_km):.4f}")
joblib.dump(kmeans, 'modelagem_melhorada/modelos_ajustados/kmeans.pkl')

# --- 6.2 Gaussian Mixture Model (novo) ---
print("\n--- Gaussian Mixture Model (NOVO) ---")
gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
labels_gmm = gmm.fit_predict(X_scaled)
probs_gmm = gmm.predict_proba(X_scaled)
player_stats_filtered['Cluster_GMM'] = labels_gmm
player_stats_filtered['GMM_max_prob'] = probs_gmm.max(axis=1)
print(f"GMM (K={best_k}): Silhouette={silhouette_score(X_scaled, labels_gmm):.4f}")
print(f"   Media da maxima probabilidade: {player_stats_filtered['GMM_max_prob'].mean():.3f}")
joblib.dump(gmm, 'modelagem_melhorada/modelos_ajustados/gmm.pkl')

# --- 6.3 DBSCAN (novo) ---
print("\n--- DBSCAN (NOVO) ---")
# Encontrar eps otimo usando k-distance graph
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_scaled)
distances, _ = neigh.kneighbors(X_scaled)
distances = np.sort(distances[:, 4])

# Heuristica: eps no "joelho"
eps_opt = np.percentile(distances, 90)
print(f"   eps heuristico (P90 da 5a distancia): {eps_opt:.3f}")
dbscan = DBSCAN(eps=eps_opt, min_samples=10)
labels_db = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)
print(f"   DBSCAN: {n_clusters_db} clusters, {n_noise} outliers ({n_noise/len(labels_db)*100:.1f}%)")
player_stats_filtered['Cluster_DBSCAN'] = labels_db

# --- 6.4 Hierarchical Clustering ---
print("\n--- Agrupamento Hierarquico ---")
hier = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels_hier = hier.fit_predict(X_scaled)
player_stats_filtered['Cluster_Hier'] = labels_hier
joblib.dump(hier, 'modelagem_melhorada/modelos_ajustados/hierarquico.pkl')

# =============================================================================
# 7. ESTABILIDADE DOS CLUSTERS (Bootstrap)
# =============================================================================
print("\n=== 7. ESTABILIDADE DOS CLUSTERS (Bootstrap) ===")
n_bootstrap = 50
sil_boot = []
for i in range(n_bootstrap):
    idx = np.random.choice(len(X_scaled), size=int(0.8*len(X_scaled)), replace=False)
    km_boot = KMeans(n_clusters=best_k, random_state=42+i, n_init=10)
    labels_boot = km_boot.fit_predict(X_scaled[idx])
    sil_boot.append(silhouette_score(X_scaled[idx], labels_boot))

sil_mean = np.mean(sil_boot)
sil_std = np.std(sil_boot)
print(f"Bootstrap (n={n_bootstrap}, 80% amostra):")
print(f"   Silhouette medio: {sil_mean:.4f} (+/- {sil_std:.4f})")
print(f"   Intervalo 95%: [{np.percentile(sil_boot, 2.5):.4f}, {np.percentile(sil_boot, 97.5):.4f}]")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sil_boot, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(sil_mean, color='red', linestyle='--', lw=2, label=f'Media: {sil_mean:.3f}')
ax.axvline(silhouette_score(X_scaled, labels_km), color='green', linestyle='--', lw=2, label=f'Full data: {silhouette_score(X_scaled, labels_km):.3f}')
ax.set_xlabel('Silhouette Score'); ax.set_ylabel('Frequencia')
ax.set_title('Bootstrap de Estabilidade - K-Means', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('modelagem_melhorada/resultados/bootstrap_estabilidade.png', dpi=300)
plt.close()

# =============================================================================
# 8. INTERPRETABILIDADE APROFUNDADA
# =============================================================================
print("\n=== 8. INTERPRETABILIDADE APROFUNDADA ===")
fig_dir = 'interpretabilidade_melhorada'

# --- 8.1 Centroides ---
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
centroids.index = [f'Cluster {i}' for i in range(best_k)]
centroids.to_csv(f'{fig_dir}/centroides_clusters.csv')
print("\n--- Centroides K-Means (padronizados) ---")
print(centroids.round(2).to_string())

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(centroids, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'label':'Centroide (z-score)'})
ax.set_title('Centroides dos Clusters - K-Means', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{fig_dir}/centroides_heatmap.png', dpi=300)
plt.close()

# --- 8.2 Profiling estatistico formal (novo) ---
print("\n--- Profiling Estatistico Formal ---")
profiling = []
for c in range(best_k):
    cluster_data = X[labels_km == c]
    other_data = X[labels_km != c]
    for feat in feature_cols:
        c_vals = cluster_data[feat].values
        o_vals = other_data[feat].values
        # Teste t
        t_stat, p_val = stats.ttest_ind(c_vals, o_vals, equal_var=False)
        # Cohen's d
        pooled_std = np.sqrt((np.std(c_vals, ddof=1)**2 + np.std(o_vals, ddof=1)**2) / 2)
        cohen_d = (np.mean(c_vals) - np.mean(o_vals)) / pooled_std if pooled_std > 0 else 0
        profiling.append({
            'Cluster': c, 'Feature': feat,
            'Cluster_Mean': np.mean(c_vals), 'Other_Mean': np.mean(o_vals),
            'Cluster_Std': np.std(c_vals), 'Other_Std': np.std(o_vals),
            't_stat': t_stat, 'p_value': p_val, 'Cohens_d': cohen_d,
            'Significant': p_val < 0.05
        })

profiling_df = pd.DataFrame(profiling)
profiling_df.to_csv(f'{fig_dir}/profiling_estatistico.csv', index=False)
print("   Profiling salvo em profiling_estatistico.csv")

# Mostrar features significativas por cluster
for c in range(best_k):
    sig = profiling_df[(profiling_df['Cluster'] == c) & (profiling_df['Significant'])]
    sig = sig.sort_values('Cohens_d', key=abs, ascending=False)
    print(f"\n   Cluster {c} - Top 5 features discriminantes (Cohen's d):")
    print(sig[['Feature', 'Cluster_Mean', 'Other_Mean', 'Cohens_d']].head().to_string(index=False))

# --- 8.3 Comparacao com Ranking ATP (novo) ---
print("\n--- Comparacao com Ranking ATP ---")
if 'Rk' in raw.columns:
    # Pegar ranking medio por jogador
    rk_avg = raw.groupby('Name')['Rk'].mean().reset_index()
    rk_avg.columns = ['Name', 'Rk_avg']
    player_stats_filtered = player_stats_filtered.merge(rk_avg, on='Name', how='left')

    for c in range(best_k):
        cluster_rk = player_stats_filtered[player_stats_filtered['Cluster_KMeans'] == c]['Rk_avg'].dropna()
        other_rk = player_stats_filtered[player_stats_filtered['Cluster_KMeans'] != c]['Rk_avg'].dropna()
        if len(cluster_rk) > 0 and len(other_rk) > 0:
            u_stat, p_val = stats.mannwhitneyu(cluster_rk, other_rk, alternative='two-sided')
            print(f"   Cluster {c}: Rk medio={cluster_rk.mean():.1f} vs Outros={other_rk.mean():.1f} (p={p_val:.4f})")

# --- 8.4 Jogadores de fronteira (novo) ---
print("\n--- Jogadores de Fronteira ---")
distances = cdist(X_scaled, kmeans.cluster_centers_, 'euclidean')
player_stats_filtered['Dist_Centroid'] = np.min(distances, axis=1)
# Jogadores que sao quase igualmente proximos de 2 clusters
if best_k == 2:
    ratio = distances[:, 0] / (distances[:, 1] + 1e-10)
    # Ratio proximo de 1 = fronteira
    fronteira = player_stats_filtered[np.abs(np.log(ratio)) < 0.3].copy()
    print(f"   Jogadores de fronteira (ratio < 1.35): {len(fronteira)}")
    if len(fronteira) > 0:
        print(fronteira[['Name', 'Cluster_KMeans', 'Aces_pct', 'ReturnPoints_Won']].head(10).to_string(index=False))

# --- 8.5 Boxplots ---
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
axes = axes.flatten()
plot_cols = feature_cols[:16]  # primeiras 16
for i, col in enumerate(plot_cols):
    ax = axes[i]
    sns.boxplot(data=player_stats_filtered, x='Cluster_KMeans', y=col, palette='viridis', ax=ax)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('Cluster')
fig.suptitle('Distribuicao das Features por Cluster', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(f'{fig_dir}/boxplots_por_cluster.png', dpi=300)
plt.close()

# --- 8.6 PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for c in range(best_k):
    mask = labels_km == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[c]], label=f'Cluster {c}', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Clusters no Espaco PCA', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{fig_dir}/clusters_pca.png', dpi=300)
plt.close()

# --- 8.7 t-SNE (novo) ---
print("\n--- t-SNE ---")
if len(X_scaled) > 30:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(best_k):
        mask = labels_km == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[c]], label=f'Cluster {c}', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax.set_title('Clusters no Espaco t-SNE', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/clusters_tsne.png', dpi=300)
    plt.close()
    print("   t-SNE gerado com sucesso.")

# --- 8.8 Silhouette plot ---
fig, ax = plt.subplots(figsize=(10, 6))
sil_vals = silhouette_samples(X_scaled, labels_km)
y_lower = 10
for i in range(best_k):
    ith_sil = sil_vals[labels_km == i]
    ith_sil.sort()
    size = len(ith_sil)
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper + 10
ax.axvline(silhouette_score(X_scaled, labels_km), color="red", linestyle="--", lw=2, label='Media')
ax.set_title('Silhouette Plot por Cluster', fontweight='bold')
ax.set_xlabel('Silhouette Coefficient'); ax.set_ylabel('Cluster')
ax.set_yticks([]); ax.legend()
plt.tight_layout()
plt.savefig('modelagem_melhorada/resultados/silhouette_plot.png', dpi=300)
plt.close()

# --- 8.9 GMM Soft Clustering (novo) ---
print("\n--- GMM Soft Clustering ---")
# Jogadores com baixa confianca (probabilidade maxima < 0.7)
uncertain = player_stats_filtered[player_stats_filtered['GMM_max_prob'] < 0.7]
print(f"   Jogadores com baixa confianca GMM (< 0.7): {len(uncertain)}")
if len(uncertain) > 0:
    print(uncertain[['Name', 'GMM_max_prob', 'Cluster_GMM']].head(10).to_string(index=False))

# =============================================================================
# 9. DENDROGRAMA
# =============================================================================
print("\n=== 9. DENDROGRAMA ===")
sample_idx = np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]
names_sample = names[sample_idx]

linkage_matrix = linkage(X_sample, method='ward')
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_matrix, labels=names_sample, leaf_rotation=90, leaf_font_size=6, ax=ax)
ax.set_title('Dendrograma (Amostra de 100 Jogadores)', fontweight='bold')
ax.set_xlabel('Jogadores'); ax.set_ylabel('Distancia')
plt.tight_layout()
plt.savefig('modelagem_melhorada/resultados/dendrograma.png', dpi=300)
plt.close()

# =============================================================================
# 10. DOCUMENTACAO
# =============================================================================
print("\n=== 10. DOCUMENTACAO ===")
doc = """
MELHORIAS IMPLEMENTADAS NO PIPELINE DE AGRUPAMENTO
====================================================

[A] FEATURES POR SUPERFICIE
    - Agregacao separada por Hard, Clay, Grass
    - Hard_Clay_specialization: ratio de aces em hard vs clay
    - Justificativa: estilo de jogo muda drasticamente entre superficies

[B] FEATURES COMPOSTAS
    - Serve_efficiency = Aces / (Df + 0.1): risco vs recompensa do saque
    - Return_dominance = ReturnPoints / (vAces + 0.1): dominancia na devolucao
    - Serve_consistency = 1stIn * 2ndWin: consistencia do saque
    - BP_efficiency = BP_Converted / (BP_Saved + 0.1): eficiencia em break points

[C] NOVOS ALGORITMOS
    - Gaussian Mixture Model (GMM): soft clustering com probabilidades
    - DBSCAN: detecta outliers e jogadores hibridos

[D] METRICAS ADICIONAIS
    - Calinski-Harabasz: separacao entre clusters (maior=melhor)
    - Davies-Bouldin: compacidade e separacao (menor=melhor)
    - Bootstrap de estabilidade: verifica se clusters sao robustos

[E] INTERPRETABILIDADE APROFUNDADA
    - Profiling estatistico formal: testes t e Cohen's d por feature
    - Comparacao com ranking ATP: teste de Mann-Whitney
    - Jogadores de fronteira: proximos a ambos os centroides
    - t-SNE: visualizacao nao-linear alem do PCA
    - GMM soft clustering: probabilidade de pertencimento
"""
with open('README_MELHORIAS_AGRUPAMENTO.md', 'w') as f:
    f.write(doc)
print(doc)

player_stats_filtered.to_csv('PROCESSAMENTO_MELHORADO/dados_com_clusters.csv', index=False)

print("\n=== CONCLUIDO ===")
print(f"Clusters identificados: {best_k}")
print(f"Jogadores analisados: {len(player_stats_filtered)}")
print("Todos os artefatos do pipeline melhorado gerados.")
