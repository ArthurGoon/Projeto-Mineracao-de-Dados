import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

for d in ['ADED/figuras','PROCESSAMENTO/receitas','modelagem/modelos_ajustados',
          'modelagem/resultados','interpretabilidade']:
    os.makedirs(d, exist_ok=True)

print("=== 1. LEITURA ===")
players = pd.read_csv('dataset/players(man).csv')
serve = pd.read_csv('dataset/serve_kaggle.csv')
ret = pd.read_csv('dataset/return_kaggle.csv')
print(f"Players: {players.shape}")
print(f"Serve: {serve.shape}")
print(f"Return: {ret.shape}")

# Limpar nomes de colunas
serve.columns = serve.columns.str.strip()
ret.columns = ret.columns.str.strip()
players.columns = players.columns.str.strip()

print(f"\nServe cols: {list(serve.columns)}")
print(f"Return cols: {list(ret.columns)}")
print(f"Players cols: {list(players.columns)}")

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

# Converter Bpsvd e BPCnv para taxa
import re
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

print("\n=== 2. AGREGACAO POR JOGADOR ===")
# Agregar serve por jogador
serve_agg = serve.groupby('Name').agg({
    'A%': 'mean',
    'Df%': 'mean',
    '1stIn': 'mean',
    '1st%': 'mean',
    '2nd%': 'mean',
    'Bpsvd_rate': 'mean'
}).reset_index()
serve_agg.columns = ['Name','Aces_pct','Df_pct','FirstServe_In','FirstServe_Win','SecondServe_Win','BP_Saved_rate']

# Agregar return por jogador
ret_agg = ret.groupby('Name').agg({
    'TPW': 'mean',
    'RPW': 'mean',
    'vA%': 'mean',
    'v1st%': 'mean',
    'v2nd%': 'mean',
    'BPCnv_rate': 'mean'
}).reset_index()
ret_agg.columns = ['Name','TotalPoints_Won','ReturnPoints_Won','vAces_pct','v1stReturn_Win','v2ndReturn_Win','BP_Converted_rate']

# Merge
player_stats = serve_agg.merge(ret_agg, on='Name', how='inner')
player_stats = player_stats.merge(players, left_on='Name', right_on='name', how='inner')

print(f"Jogadores com stats completas: {len(player_stats)}")
print(f"Stats summary:\n{player_stats[['number_of_matches']].describe()}")

# ADED: distribuicao de partidas
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(player_stats['number_of_matches'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
ax.set_title('Distribuicao de Partidas por Jogador', fontweight='bold')
ax.set_xlabel('Numero de Partidas'); ax.set_ylabel('Frequencia')
ax.axvline(player_stats['number_of_matches'].median(), color='red', linestyle='--', label=f"Mediana: {player_stats['number_of_matches'].median():.0f}")
ax.legend()
plt.tight_layout(); plt.savefig('ADED/figuras/01_distribuicao_partidas.png', dpi=300); plt.close()

# Filtrar jogadores com minimo de partidas (minimo = 20 para ter estatistica estavel)
MIN_MATCHES = 20
player_stats_filtered = player_stats[player_stats['number_of_matches'] >= MIN_MATCHES].copy()
print(f"\nApos filtro (>= {MIN_MATCHES} partidas): {len(player_stats_filtered)} jogadores")

# Features para agrupamento
feature_cols = ['Aces_pct','Df_pct','FirstServe_In','FirstServe_Win','SecondServe_Win',
                'TotalPoints_Won','ReturnPoints_Won','vAces_pct','v1stReturn_Win','v2ndReturn_Win',
                'BP_Saved_rate','BP_Converted_rate']

# Preencher NaN restantes com mediana
for c in feature_cols:
    player_stats_filtered[c] = player_stats_filtered[c].fillna(player_stats_filtered[c].median())

X = player_stats_filtered[feature_cols].copy()
names = player_stats_filtered['Name'].values

print(f"\nFeatures para agrupamento: {len(feature_cols)}")
print(feature_cols)

# ADED: distribuicao das features
fig, axes = plt.subplots(3,4, figsize=(14,10))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    ax = axes[i]
    X[col].hist(bins=25, color='#2ecc71', edgecolor='black', alpha=0.7, ax=ax)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('')
fig.suptitle('Distribuicao das Features de Performance', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('ADED/figuras/02_distribuicao_features.png', dpi=300); plt.close()

# Salvar dados processados
X.to_csv('PROCESSAMENTO/dados_agrupados.csv', index=False)
pd.DataFrame({'Name':names}).to_csv('PROCESSAMENTO/nomes_jogadores.csv', index=False)

print("\n=== 3. PADRONIZACAO ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'PROCESSAMENTO/receitas/scaler.pkl')
joblib.dump(feature_cols, 'PROCESSAMENTO/receitas/feature_names.pkl')

print("\n=== 4. ESCOLHA DE K ===")
# Metodo do cotovelo e silhouette
k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"K={k}: Inertia={km.inertia_:.1f}, Silhouette={silhouette_score(X_scaled, km.labels_):.4f}")

# Figura cotovelo + silhouette
fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].plot(k_range, inertias, 'o-', color='#3498db', lw=2, markersize=8)
axes[0].set_xlabel('K (numero de clusters)'); axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Metodo do Cotovelo', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, 'o-', color='#e74c3c', lw=2, markersize=8)
axes[1].set_xlabel('K (numero de clusters)'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por K', fontweight='bold')
axes[1].grid(True, alpha=0.3)
best_k = list(k_range)[np.argmax(silhouettes)]
axes[1].axvline(best_k, color='green', linestyle='--', lw=2, label=f'Melhor K={best_k}')
axes[1].legend()

plt.suptitle('Escolha do Numero de Clusters', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('modelagem/resultados/cotovelo_silhouette.png', dpi=300); plt.close()

print(f"\nMelhor K segundo Silhouette: {best_k}")

# Usar K com melhor silhouette, mas tambem testar K=4 para comparacao
K_FINAL = best_k

print("\n=== 5. K-MEANS ===")
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)
player_stats_filtered['Cluster_KMeans'] = labels_km

print(f"Clusters K-Means: {K_FINAL}")
print(f"Silhouette final: {silhouette_score(X_scaled, labels_km):.4f}")
print(f"Tamanho dos clusters:")
print(pd.Series(labels_km).value_counts().sort_index())

joblib.dump(kmeans, 'modelagem/modelos_ajustados/kmeans.pkl')

print("\n=== 6. AGRUPAMENTO HIERARQUICO ===")
# Dendrograma amostra (amostra de 100 jogadores para visualizacao)
sample_idx = np.random.choice(len(X_scaled), min(100, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]
names_sample = names[sample_idx]

linkage_matrix = linkage(X_sample, method='ward')
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_matrix, labels=names_sample, leaf_rotation=90, leaf_font_size=6, ax=ax)
ax.set_title('Dendrograma (Amostra de 100 Jogadores)', fontweight='bold')
ax.set_xlabel('Jogadores'); ax.set_ylabel('Distancia')
plt.tight_layout(); plt.savefig('modelagem/resultados/dendrograma.png', dpi=300); plt.close()

# Agrupamento hierarquico completo
hier = AgglomerativeClustering(n_clusters=K_FINAL, linkage='ward')
labels_hier = hier.fit_predict(X_scaled)
player_stats_filtered['Cluster_Hier'] = labels_hier
print(f"\nAgrupamento Hierarquico (K={K_FINAL}) concluido.")
print(f"Tamanho dos clusters:")
print(pd.Series(labels_hier).value_counts().sort_index())

joblib.dump(hier, 'modelagem/modelos_ajustados/hierarquico.pkl')

# Matriz de contingencia
contingencia = pd.crosstab(player_stats_filtered['Cluster_KMeans'], player_stats_filtered['Cluster_Hier'])
print(f"\nContingencia K-Means vs Hierarquico:")
print(contingencia)

print("\n=== 7. INTERPRETABILIDADE ===")

# 7.1 Centroides dos clusters
fig_dir_int = 'interpretabilidade'
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
centroids.index = [f'Cluster {i}' for i in range(K_FINAL)]
print("\n--- Centroides K-Means (padronizados) ---")
print(centroids.round(2).to_string())
centroids.to_csv(f'{fig_dir_int}/centroides_clusters.csv')

# Heatmap dos centroides
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(centroids, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'label':'Centroide (z-score)'})
ax.set_title('Centroides dos Clusters - K-Means', fontweight='bold')
ax.set_xlabel('Features'); ax.set_ylabel('Cluster')
plt.tight_layout(); plt.savefig(f'{fig_dir_int}/centroides_heatmap.png', dpi=300); plt.close()

# 7.2 Grafico radar por cluster
from math import pi
fig, axes = plt.subplots(1, K_FINAL, figsize=(5*K_FINAL, 5), subplot_kw=dict(polar=True))
if K_FINAL==1: axes=[axes]

for idx, (ax, cluster_id) in enumerate(zip(axes, range(K_FINAL))):
    values = centroids.iloc[cluster_id].values.tolist()
    values += values[:1]
    angles = [n / float(len(feature_cols)) * 2 * pi for n in range(len(feature_cols))]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=8)
    ax.set_title(f'Cluster {cluster_id}', fontweight='bold', size=12)
    ax.grid(True)

plt.suptitle('Perfis dos Clusters (Radar)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f'{fig_dir_int}/perfis_radar.png', dpi=300); plt.close()

# 7.3 Jogadores representativos de cada cluster
print("\n--- Top Jogadores por Cluster (mais proximos do centroide) ---")
from scipy.spatial.distance import cdist

distances = cdist(X_scaled, kmeans.cluster_centers_, 'euclidean')
player_stats_filtered['Dist_Centroid'] = np.min(distances, axis=1)

top_jogadores = []
for c in range(K_FINAL):
    mask = player_stats_filtered['Cluster_KMeans'] == c
    cluster_players = player_stats_filtered[mask].copy()
    cluster_players['Dist'] = distances[mask, c]
    top5 = cluster_players.nsmallest(10, 'Dist')[['Name','number_of_matches','Aces_pct','ReturnPoints_Won','Cluster_KMeans','Dist']]
    print(f"\nCluster {c} ({len(cluster_players)} jogadores) - Top 10 mais proximos do centroide:")
    print(top5.to_string(index=False))
    top5['Cluster'] = c
    top_jogadores.append(top5)

top_jogadores_df = pd.concat(top_jogadores, ignore_index=True)
top_jogadores_df.to_csv(f'{fig_dir_int}/top_jogadores_cluster.csv', index=False)

# 7.4 Boxplots das features por cluster
fig, axes = plt.subplots(3,4, figsize=(16,12))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    ax = axes[i]
    sns.boxplot(data=player_stats_filtered, x='Cluster_KMeans', y=col, palette='viridis', ax=ax)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('Cluster')
fig.suptitle('Distribuicao das Features por Cluster', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f'{fig_dir_int}/boxplots_por_cluster.png', dpi=300); plt.close()

# 7.5 Silhouette plot para K_FINAL
fig, ax = plt.subplots(figsize=(10, 6))
sil_vals = silhouette_samples(X_scaled, labels_km)
y_lower = 10
for i in range(K_FINAL):
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
plt.tight_layout(); plt.savefig('modelagem/resultados/silhouette_plot.png', dpi=300); plt.close()

# 7.6 PCA para visualizacao 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, K_FINAL))
for c in range(K_FINAL):
    mask = labels_km == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[c]], label=f'Cluster {c}', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Clusters no Espaco PCA', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{fig_dir_int}/clusters_pca.png', dpi=300); plt.close()

print(f"\nVariancia explicada PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

# Salvar resultado final
player_stats_filtered.to_csv('PROCESSAMENTO/dados_com_clusters.csv', index=False)

print("\n=== CONCLUIDO ===")
print(f"Clusters identificados: {K_FINAL}")
print(f"Jogadores analisados: {len(player_stats_filtered)}")
print("Todos os artefatos gerados.")
