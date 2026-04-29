import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuracoes
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
fig_dir = 'ADED/figuras'

print("=== 1. LEITURA ===")
df = pd.read_csv('dataset/nba_2022-23_all_stats_with_salary.csv')
df = df.drop('Unnamed: 0', axis=1)

print(f"Dimensoes: {df.shape}")
print(f"Colunas: {list(df.columns)}")
print(f"Dados faltantes:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Tratar posicoes hibridas
df['Position_Clean'] = df['Position'].apply(lambda x: x.split('-')[0])

print("\n=== 2. DISTRIBUICAO DO SALARIO ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Salario original
axes[0].hist(df['Salary']/1e6, bins=40, color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribuicao do Salario (Milhoes USD)', fontweight='bold')
axes[0].set_xlabel('Salario (milhoes)')
axes[0].set_ylabel('Frequencia')
axes[0].axvline(df['Salary'].mean()/1e6, color='red', linestyle='--', label=f'Media: {df["Salary"].mean()/1e6:.1f}M')
axes[0].legend()

# Log salario
log_sal = np.log(df['Salary'])
axes[1].hist(log_sal, bins=40, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribuicao do Log(Salario)', fontweight='bold')
axes[1].set_xlabel('Log(Salario)')
axes[1].set_ylabel('Frequencia')
axes[1].axvline(log_sal.mean(), color='red', linestyle='--', label=f'Media: {log_sal.mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{fig_dir}/01_distribuicao_salarios.png')
plt.close()

print(f"Assimetria salario: {stats.skew(df['Salary']):.2f}")
print(f"Assimetria log(salario): {stats.skew(np.log(df['Salary'])):.2f}")
print(f"Media salario: ${df['Salary'].mean():,.0f}")
print(f"Mediana salario: ${df['Salary'].median():,.0f}")
print(f"Max salario: ${df['Salary'].max():,.0f} ({df.loc[df['Salary'].idxmax(), 'Player Name']})")
print(f"Min salario: ${df['Salary'].min():,.0f} ({df.loc[df['Salary'].idxmin(), 'Player Name']})")

print("\n=== 3. SALARIO POR POSICAO ===")
salario_pos = df.groupby('Position_Clean')['Salary'].agg(['mean', 'median', 'count']).reset_index()
salario_pos = salario_pos.sort_values('mean', ascending=False)
print(salario_pos.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='Position_Clean', y='Salary', order=salario_pos['Position_Clean'], palette='viridis', ax=ax)
ax.set_title('Salario por Posicao', fontweight='bold')
ax.set_xlabel('Posicao')
ax.set_ylabel('Salario (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
plt.tight_layout()
plt.savefig(f'{fig_dir}/02_salario_por_posicao.png')
plt.close()

print("\n=== 4. TOP CORRELACOES COM SALARIO ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Salary')
corr_salary = df[numeric_cols + ['Salary']].corr()['Salary'].drop('Salary').abs().sort_values(ascending=False)
print(corr_salary.head(15).to_string())

# Figura correlacoes top
fig, ax = plt.subplots(figsize=(8, 6))
top_corr = corr_salary.head(15).sort_values()
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in df[top_corr.index.tolist() + ['Salary']].corr()['Salary'].drop('Salary').loc[top_corr.index]]
ax.barh(top_corr.index, top_corr.values, color=colors)
ax.set_title('Correlacao Absoluta com Salario', fontweight='bold')
ax.set_xlabel('|Correlacao de Pearson|')
plt.tight_layout()
plt.savefig(f'{fig_dir}/03_correlacao_salario.png')
plt.close()

print("\n=== 5. MATRIZ DE CORRELACAO (VARIAVEIS SELECIONADAS) ===")
# Selecionar vars representativas para evitar figura gigante
vars_sel = ['Salary', 'Age', 'GP', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK',
            'FG%', '3P%', 'FT%', 'PER', 'TS%', 'USG%', 'WS', 'VORP', 'BPM']
fig, ax = plt.subplots(figsize=(12, 10))
corr_mat = df[vars_sel].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Matriz de Correlacao - Variaveis Selecionadas', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{fig_dir}/04_matriz_correlacao.png')
plt.close()

print("\n=== 6. SALARIO VS STATS AVANCADAS ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# PTS vs Salario
axes[0,0].scatter(df['PTS'], df['Salary']/1e6, alpha=0.6, c=df['Age'], cmap='viridis', edgecolors='black', linewidth=0.5)
axes[0,0].set_xlabel('Pontos por Jogo')
axes[0,0].set_ylabel('Salario (milhoes)')
axes[0,0].set_title('Salario vs Pontos (cor = Idade)')
cbar = plt.colorbar(axes[0,0].collections[0], ax=axes[0,0])
cbar.set_label('Idade')

# VORP vs Salario
axes[0,1].scatter(df['VORP'], df['Salary']/1e6, alpha=0.6, c='#e74c3c', edgecolors='black', linewidth=0.5)
axes[0,1].set_xlabel('VORP (Value Over Replacement Player)')
axes[0,1].set_ylabel('Salario (milhoes)')
axes[0,1].set_title('Salario vs VORP')

# WS vs Salario
axes[1,0].scatter(df['WS'], df['Salary']/1e6, alpha=0.6, c='#3498db', edgecolors='black', linewidth=0.5)
axes[1,0].set_xlabel('Win Shares')
axes[1,0].set_ylabel('Salario (milhoes)')
axes[1,0].set_title('Salario vs Win Shares')

# PER vs Salario
axes[1,1].scatter(df['PER'], df['Salary']/1e6, alpha=0.6, c='#2ecc71', edgecolors='black', linewidth=0.5)
axes[1,1].set_xlabel('Player Efficiency Rating')
axes[1,1].set_ylabel('Salario (milhoes)')
axes[1,1].set_title('Salario vs PER')

plt.suptitle('Salario vs Estatisticas de Performance', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{fig_dir}/05_salario_vs_stats.png')
plt.close()

print("\n=== 7. SALARIO VS IDADE ===")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Age'], df['Salary']/1e6, alpha=0.6, edgecolors='black', linewidth=0.5)
# Trend line
z = np.polyfit(df['Age'], df['Salary']/1e6, 2)
p = np.poly1d(z)
x_line = np.linspace(df['Age'].min(), df['Age'].max(), 100)
ax.plot(x_line, p(x_line), "r--", lw=2, label='Tendencia polinomial (grau 2)')
ax.set_xlabel('Idade (anos)')
ax.set_ylabel('Salario (milhoes)')
ax.set_title('Salario vs Idade', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{fig_dir}/06_salario_vs_idade.png')
plt.close()

print("\n=== 8. ESTATISTICAS DESCRITIVAS ===")
print(df[['Salary', 'Age', 'GP', 'MP', 'PTS', 'PER', 'WS', 'VORP', 'BPM']].describe().round(2).to_string())

# Salvar estatisticas
stats_df = df[['Salary', 'Age', 'GP', 'MP', 'PTS', 'PER', 'WS', 'VORP', 'BPM']].describe().round(2)
stats_df.to_csv(f'{fig_dir}/estatisticas_descritivas.csv')

print("\n=== ADED CONCLUIDA ===")
print(f"Figuras salvas em {fig_dir}/")
