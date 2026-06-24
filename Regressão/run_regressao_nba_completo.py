#!/usr/bin/env python3
"""Pipeline de regressão NBA — dataset bruto até modelos e interpretabilidade. Ver REGRESSAO_NBA.md."""
import json as _json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
REPORT_DIR = REPO_ROOT / 'relatorio'
os.chdir(ROOT)

DIR_EDA = ROOT / '1_EDA'
DIR_PREPROC = ROOT / '2_PREPROCESSAMENTO'
DIR_MODEL = ROOT / '3_MODELAGEM'
DIR_INTERP = ROOT / '4_INTERPRETABILIDADE'
DIR_DATASET = ROOT / 'dataset'
DATA_CSV = DIR_DATASET / 'nba_2022-23_all_stats_with_salary.csv'

for d in [
    DIR_EDA,
    DIR_PREPROC / 'receitas',
    DIR_MODEL / 'modelos_ajustados',
    DIR_MODEL / 'resultados',
    DIR_INTERP / 'importancia_permutacao',
    DIR_INTERP / 'graficos_pdp',
    DIR_INTERP / 'shap_values',
    DIR_INTERP / 'figuras_relatorio',
]:
    d.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error
)
from sklearn.inspection import permutation_importance, partial_dependence

print("=== EDA ===")
df_raw = pd.read_csv(DATA_CSV)
if 'Unnamed: 0' in df_raw.columns:
    df_raw = df_raw.drop('Unnamed: 0', axis=1)
df_raw['Position_Clean'] = df_raw['Position'].apply(lambda x: str(x).split('-')[0])
print(f"Dados brutos: {df_raw.shape}")

eda_cols = ['Salary', 'Age', 'GP', 'MP', 'PTS', 'PER', 'WS', 'VORP', 'BPM']
eda_stats = df_raw[eda_cols].describe().T
eda_stats.to_csv(DIR_EDA / 'estatisticas_descritivas.csv')
print(f"Estatísticas descritivas salvas em {DIR_EDA / 'estatisticas_descritivas.csv'}")

sns.set_theme(style='whitegrid', font_scale=0.9)

REPORT_FIG_DIR = DIR_INTERP / 'figuras_relatorio'
REPORT_LABELS = {
    'MP': 'Minutos (MP)',
    'Age': 'Idade',
    'PTS_per_GP': 'Pontos por jogo',
    'USG%': 'Uso ofensivo (USG%)',
    '3P%': 'Aproveitamento 3P%',
    'FG%': 'Aproveitamento FG%',
    'AST_to_TOV': 'AST/TOV',
    'BLK_per_min': 'Tocos por minuto',
    'TRB_per_min': 'Rebotes por minuto',
    'VORP': 'VORP',
    'Age_sq': 'Idade²',
    'AST_per_GP': 'Assistências por jogo',
    'FT%': 'Lances livres (FT%)',
    'STL_BLK_sum': 'Roubos + tocos',
    'STL_per_GP': 'Roubos por jogo',
    'GP': 'Jogos disputados',
    'BPM': 'BPM',
    'BLK_per_GP': 'Tocos por jogo',
    'AST_per_min': 'Assistências/min',
    'STL_per_min': 'Roubos/min',
    'TRB_per_GP': 'Rebotes por jogo',
}

def _report_label(feature):
    return REPORT_LABELS.get(feature, feature)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_raw['Salary'] / 1e6, bins=40, color='#a78bfa', edgecolor='white')
ax.set_xlabel('Salário (US$ milhões)'); ax.set_ylabel('Frequência')
ax.set_title('Distribuição Salarial — Dataset Bruto (467 jogadores)', fontweight='bold')
plt.tight_layout(); plt.savefig(DIR_EDA / '01_distribuicao_salarios.png', dpi=200); plt.close()

pos_sal = df_raw.groupby('Position_Clean')['Salary'].mean().sort_values(ascending=False) / 1e6
fig, ax = plt.subplots(figsize=(8, 5))
pos_sal.plot(kind='bar', ax=ax, color='#2dd4bf')
ax.set_ylabel('Salário médio (US$ milhões)'); ax.set_title('Salário Médio por Posição', fontweight='bold')
plt.tight_layout(); plt.savefig(DIR_EDA / '02_salario_por_posicao.png', dpi=200); plt.close()

corr_cols = ['Salary', 'MP', 'PTS', 'USG%', 'PER', 'BPM', 'WS', 'VORP', 'Age']
corr_mat = df_raw[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Matriz de Correlação — Variáveis-Chave', fontweight='bold')
plt.tight_layout(); plt.savefig(DIR_EDA / '04_matriz_correlacao.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_raw['Age'], df_raw['Salary'] / 1e6, alpha=0.5, c='#fb923c', edgecolors='none')
ax.set_xlabel('Idade'); ax.set_ylabel('Salário (US$ milhões)')
ax.set_title('Salário vs Idade', fontweight='bold')
plt.tight_layout(); plt.savefig(DIR_EDA / '06_salario_vs_idade.png', dpi=200); plt.close()

sal_corr = df_raw[['Salary', 'MP', 'PTS', 'USG%', 'PER', 'BPM', 'Age']].corr()['Salary'].drop('Salary').sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
sal_corr.plot(kind='barh', ax=ax, color='#a78bfa')
ax.set_xlabel('Correlação com Salary'); ax.set_title('Correlação Salarial — Variáveis-Chave', fontweight='bold')
plt.tight_layout(); plt.savefig(DIR_EDA / '03_correlacao_salario.png', dpi=200); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, color in zip(axes, ['MP', 'PTS', 'BPM'], ['#2dd4bf', '#fb923c', '#60a5fa']):
    ax.scatter(df_raw[col], df_raw['Salary'] / 1e6, alpha=0.45, c=color, edgecolors='none')
    ax.set_xlabel(col); ax.set_ylabel('Salário (US$ M)')
fig.suptitle('Salário vs Estatísticas de Jogo', fontweight='bold')
plt.tight_layout(); plt.savefig(DIR_EDA / '05_salario_vs_stats.png', dpi=200); plt.close()

# Figura vetorial para o relatório: EDA com rótulos maiores e menos ruído visual.
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
ax = axes[0, 0]
ax.hist(df_raw['Salary'] / 1e6, bins=32, color='#8b5cf6', edgecolor='white')
ax.set_title('Distribuição dos salários', fontsize=15, fontweight='bold')
ax.set_xlabel('Salário (US$ milhões)', fontsize=12)
ax.set_ylabel('Jogadores', fontsize=12)
ax.tick_params(axis='both', labelsize=11)
ax.grid(axis='y', alpha=0.25)

ax = axes[0, 1]
salary_corr = df_raw[['Salary', 'PTS', 'VORP', 'MP', 'WS', 'USG%', 'Age', 'BPM']].corr()['Salary'].drop('Salary').sort_values()
salary_corr.plot(kind='barh', ax=ax, color='#2563eb')
ax.set_title('Correlação com salário', fontsize=15, fontweight='bold')
ax.set_xlabel('Correlação de Pearson', fontsize=12)
ax.tick_params(axis='both', labelsize=11)
ax.grid(axis='x', alpha=0.25)

ax = axes[1, 0]
ax.scatter(df_raw['Age'], df_raw['Salary'] / 1e6, alpha=0.55, s=22, c='#f97316', edgecolors='none')
ax.set_title('Salário por idade', fontsize=15, fontweight='bold')
ax.set_xlabel('Idade', fontsize=12)
ax.set_ylabel('Salário (US$ milhões)', fontsize=12)
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, alpha=0.25)

ax = axes[1, 1]
pos_sal = df_raw.groupby('Position_Clean')['Salary'].mean().sort_values(ascending=False) / 1e6
pos_sal.plot(kind='bar', ax=ax, color='#14b8a6')
ax.set_title('Salário médio por posição', fontsize=15, fontweight='bold')
ax.set_xlabel('Posição', fontsize=12)
ax.set_ylabel('Salário médio (US$ milhões)', fontsize=12)
ax.tick_params(axis='both', labelsize=11, rotation=0)
ax.grid(axis='y', alpha=0.25)

fig.suptitle('Análise exploratória dos salários da NBA', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(REPORT_FIG_DIR / 'eda_relatorio.pdf', bbox_inches='tight')
plt.savefig(REPORT_FIG_DIR / 'eda_relatorio.png', dpi=300, bbox_inches='tight')
plt.close()

# Histograma salarial (export para apresentação)
_sal_edges = [0, 1e6, 2e6, 3e6, 5e6, 8e6, 12e6, 18e6, 25e6, 35e6, 50e6]
_sal_labels = ['<1M', '1-2M', '2-3M', '3-5M', '5-8M', '8-12M', '12-18M', '18-25M', '25-35M', '35-50M']
_low_thr = 500_000
_sal_cat = pd.cut(df_raw['Salary'], bins=_sal_edges, labels=_sal_labels, right=False)
_hist_bins = []
for lab in _sal_labels:
    mask = _sal_cat == lab
    _hist_bins.append({
        'label': lab,
        'count': int(mask.sum()),
        'outliers': int((mask & (df_raw['Salary'] < _low_thr)).sum()),
    })

(DIR_EDA / 'salary_histogram.json').write_text(_json.dumps({
    'total': len(df_raw),
    'removedOutliers': int((df_raw['Salary'] < _low_thr).sum()),
    'cleanTotal': int((df_raw['Salary'] >= _low_thr).sum()),
    'bins': _hist_bins,
}, indent=2), encoding='utf-8')

print(f"Gráficos EDA salvos em {DIR_EDA}/")

print("\n=== Limpeza ===")
df = df_raw.copy()

LOW_SALARY_THRESHOLD = 500_000
low_salary_count = (df['Salary'] < LOW_SALARY_THRESHOLD).sum()
print(f"Removendo {low_salary_count} jogadores com salário < ${LOW_SALARY_THRESHOLD:,}")
df = df[df['Salary'] >= LOW_SALARY_THRESHOLD].copy()
print(f"Shape após filtro: {df.shape}")

for col in ['FG%', '3P%', '2P%', 'eFG%', 'FT%', 'TS%', '3PAr', 'FTr']:
    if col in df.columns:
        df[col] = df.groupby('Position_Clean')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

df['Log_Salary'] = np.log(df['Salary'])

print("\n=== Feature engineering ===")

for col in ['PTS','TRB','AST','STL','BLK','ORB','DRB','TOV','PF']:
    df[f'{col}_per_GP'] = df[col] / df['GP'].replace(0, np.nan)
    df[f'{col}_per_GP'] = df[f'{col}_per_GP'].fillna(0)

for col in ['PTS','TRB','AST','STL','BLK']:
    df[f'{col}_per_min'] = df[col] / df['MP'].replace(0, np.nan)
    df[f'{col}_per_min'] = df[f'{col}_per_min'].fillna(0)

age_mean = df['Age'].mean()
df['Age_sq'] = (df['Age'] - age_mean) ** 2

def experience_cat(age):
    if age <= 22: return 'Rookie'
    elif age <= 28: return 'Prime'
    else: return 'Veteran'
df['Experience_Category'] = df['Age'].apply(experience_cat)

df['AST_to_TOV'] = df['AST'] / df['TOV'].replace(0, np.nan)
df['AST_to_TOV'] = df['AST_to_TOV'].fillna(df['AST_to_TOV'].median())
df['STL_BLK_sum'] = df['STL'] + df['BLK']
df['Toxic_Contract'] = ((df['Salary'] > 15_000_000) & (df['GP'] < 15)).astype(int)

print("\n=== Seleção de features ===")

features_base = [
    'Age', 'Age_sq', 'GP', 'MP',
    'PTS_per_GP', 'TRB_per_GP', 'AST_per_GP', 'STL_per_GP', 'BLK_per_GP',
    'TRB_per_min', 'AST_per_min', 'STL_per_min', 'BLK_per_min',
    'FG%', '3P%', 'FT%', 'USG%', 'BPM', 'VORP',
    'AST_to_TOV', 'STL_BLK_sum', 'Toxic_Contract',
    'Position_Clean', 'Experience_Category'
]

X = df[features_base].copy()
y = df['Log_Salary'].copy()
print(f"{len(features_base)} features selecionadas")

print("\n=== Split treino/teste ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=X['Position_Clean']
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

X_train.to_csv(DIR_PREPROC / 'X_train.csv', index=False)
X_test.to_csv(DIR_PREPROC / 'X_test.csv', index=False)
pd.DataFrame({'Salary': y_train}).to_csv(DIR_PREPROC / 'y_train.csv', index=False)
pd.DataFrame({'Salary': y_test}).to_csv(DIR_PREPROC / 'y_test.csv', index=False)

print("\n=== Preprocessamento ===")
cat_features = ['Position_Clean', 'Experience_Category']
num_features = [c for c in features_base if c not in cat_features]

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)
cat_names = list(preprocess.named_transformers_['cat'].get_feature_names_out(cat_features))
feature_names = num_features + cat_names
print(f"Features pos-preproc: {len(feature_names)}")

from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif_check = pd.DataFrame(X_train_proc, columns=feature_names)
X_vif_check = X_vif_check.loc[:, X_vif_check.var() > 0.001]
vif_df = pd.DataFrame({'Feature': X_vif_check.columns,
    'VIF': [variance_inflation_factor(X_vif_check.values, i) for i in range(len(X_vif_check.columns))]})
vif_df = vif_df.sort_values('VIF', ascending=False)
print(vif_df.head(10).to_string(index=False))
vif_df.to_csv(DIR_PREPROC / 'vif_pos_preprocessamento.csv', index=False)

# VIF narrativo pré-limpeza (Age² sem centering)
df_vif_before = df.copy()
df_vif_before['Age_sq_raw'] = df_vif_before['Age'] ** 2
vif_before_cols = ['TS%', 'FG%', 'Age', 'Age_sq_raw', 'PER', 'BPM', 'MP']
X_vb = df_vif_before[vif_before_cols].astype(float)
vif_before = pd.DataFrame({
    'Feature': vif_before_cols,
    'VIF': [variance_inflation_factor(X_vb.values, i) for i in range(len(vif_before_cols))],
}).sort_values('VIF', ascending=False)
vif_before.to_csv(DIR_PREPROC / 'vif_antes_narrativo.csv', index=False)

joblib.dump(preprocess, DIR_PREPROC / 'receitas' / 'preprocessador.pkl')
joblib.dump(feature_names, DIR_PREPROC / 'receitas' / 'feature_names.pkl')

print("\n=== Modelagem ===")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def avaliar_melhorado(modelo, Xtr_raw, Xte_raw, ytr, yte, nome):
    modelo.fit(Xtr_raw, ytr)
    y_pred = modelo.predict(Xte_raw)

    rmse_log = np.sqrt(mean_squared_error(yte, y_pred))
    mae_log = mean_absolute_error(yte, y_pred)
    r2 = r2_score(yte, y_pred)
    msle = mean_squared_log_error(np.exp(yte), np.exp(y_pred))

    y_pred_sal = np.exp(y_pred)
    yte_sal = np.exp(yte)
    rmse_sal = np.sqrt(mean_squared_error(yte_sal, y_pred_sal))
    mae_sal = mean_absolute_error(yte_sal, y_pred_sal)
    mape = np.mean(np.abs((yte_sal - y_pred_sal) / yte_sal)) * 100

    cv_scores = cross_val_score(modelo, Xtr_raw, ytr, cv=kf, scoring='r2')

    print(f"\n--- {nome} ---")
    print(f"  R2 CV (KFold shuffle): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  R2 Teste: {r2:.4f} | RMSE(log): {rmse_log:.4f} | MAE(log): {mae_log:.4f}")
    print(f"  MSLE: {msle:.6f}")
    print(f"  RMSE(USD): ${rmse_sal:,.0f} | MAE(USD): ${mae_sal:,.0f} | MAPE: {mape:.2f}%")

    return {
        'Modelo': nome, 'R2_CV': cv_scores.mean(), 'R2_CV_std': cv_scores.std(),
        'R2_Test': r2, 'RMSE_Log': rmse_log, 'MAE_Log': mae_log,
        'MSLE': msle, 'RMSE_USD': rmse_sal, 'MAE_USD': mae_sal, 'MAPE': mape,
        'Modelo_Fit': modelo, 'y_pred': y_pred
    }

resultados = []

# OLS
pipe_ols = Pipeline([('preprocess', preprocess), ('reg', LinearRegression())])
res_ols = avaliar_melhorado(pipe_ols, X_train, X_test, y_train, y_test, "OLS")
resultados.append(res_ols)
joblib.dump(pipe_ols, DIR_MODEL / 'modelos_ajustados' / 'ols.pkl')

# Ridge
pipe_ridge = Pipeline([('preprocess', preprocess), ('reg', Ridge())])
grid_ridge = GridSearchCV(
    pipe_ridge, {'reg__alpha': np.logspace(-2, 3, 20)},
    cv=kf, scoring='r2', n_jobs=-1
)
grid_ridge.fit(X_train, y_train)
print(f"  Melhor alpha: {grid_ridge.best_params_['reg__alpha']:.4f}")
res_ridge = avaliar_melhorado(grid_ridge.best_estimator_, X_train, X_test, y_train, y_test, "Ridge")
resultados.append(res_ridge)
joblib.dump(grid_ridge.best_estimator_, DIR_MODEL / 'modelos_ajustados' / 'ridge.pkl')

# Lasso
pipe_lasso = Pipeline([('preprocess', preprocess), ('reg', Lasso(max_iter=10000))])
grid_lasso = GridSearchCV(
    pipe_lasso, {'reg__alpha': np.logspace(-4, 1, 20)},
    cv=kf, scoring='r2', n_jobs=-1
)
grid_lasso.fit(X_train, y_train)
print(f"  Melhor alpha: {grid_lasso.best_params_['reg__alpha']:.4f}")
res_lasso = avaliar_melhorado(grid_lasso.best_estimator_, X_train, X_test, y_train, y_test, "Lasso")
resultados.append(res_lasso)
joblib.dump(grid_lasso.best_estimator_, DIR_MODEL / 'modelos_ajustados' / 'lasso.pkl')

# Random Forest
pipe_rf = Pipeline([('preprocess', preprocess), ('reg', RandomForestRegressor(random_state=42, n_jobs=-1))])
param_rf = {
    'reg__n_estimators': [200, 300],
    'reg__max_depth': [10, 15, None],
    'reg__min_samples_split': [2, 5],
    'reg__min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=kf, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"  Melhores params: {grid_rf.best_params_}")
res_rf = avaliar_melhorado(grid_rf.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest")
resultados.append(res_rf)
joblib.dump(grid_rf.best_estimator_, DIR_MODEL / 'modelos_ajustados' / 'random_forest.pkl')

print("\n--- HistGradientBoosting ---")
pipe_hgb = Pipeline([('preprocess', preprocess), ('reg', HistGradientBoostingRegressor(random_state=42))])
param_hgb = {
    'reg__max_iter': [100, 200],
    'reg__max_depth': [3, 5, 7],
    'reg__learning_rate': [0.05, 0.1],
    'reg__min_samples_leaf': [10, 20]
}
grid_hgb = GridSearchCV(pipe_hgb, param_hgb, cv=kf, scoring='r2', n_jobs=-1)
grid_hgb.fit(X_train, y_train)
print(f"  Melhores params: {grid_hgb.best_params_}")
res_hgb = avaliar_melhorado(grid_hgb.best_estimator_, X_train, X_test, y_train, y_test, "HistGradientBoosting")
resultados.append(res_hgb)
joblib.dump(grid_hgb.best_estimator_, DIR_MODEL / 'modelos_ajustados' / 'hist_gradient_boosting.pkl')

# XGBoost
try:
    import xgboost as xgb
    pipe_xgb = Pipeline([('preprocess', preprocess), ('reg', xgb.XGBRegressor(random_state=42, n_jobs=-1))])
    param_xgb = {
        'reg__n_estimators': [200, 300],
        'reg__max_depth': [3, 5, 7],
        'reg__learning_rate': [0.05, 0.1],
        'reg__subsample': [0.8, 1.0]
    }
    grid_xgb = GridSearchCV(pipe_xgb, param_xgb, cv=kf, scoring='r2', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    print(f"  Melhores params: {grid_xgb.best_params_}")
    res_xgb = avaliar_melhorado(grid_xgb.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")
    resultados.append(res_xgb)
    joblib.dump(grid_xgb.best_estimator_, DIR_MODEL / 'modelos_ajustados' / 'xgboost.pkl')
    has_xgb = True
except ImportError:
    print("XGBoost nao disponivel. Pulando.")
    has_xgb = False

print("\n=== Resumo comparativo ===")
res_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['Modelo_Fit', 'y_pred']}
    for r in resultados
])
print(res_df.to_string(index=False))
res_df.to_csv(DIR_MODEL / 'resultados' / 'metricas_comparacao.csv', index=False)

# Graficos comparativos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
colors = colors[:len(res_df)]

axes[0].bar(res_df['Modelo'], res_df['R2_Test'], color=colors)
axes[0].set_ylabel('R2 (Teste)'); axes[0].set_title('R2 por Modelo', fontweight='bold'); axes[0].set_ylim(0, 1)
for i, v in enumerate(res_df['R2_Test']): axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(res_df['Modelo'], res_df['MAE_Log'], color=colors)
axes[1].set_ylabel('MAE (Log)'); axes[1].set_title('MAE (Log) por Modelo', fontweight='bold')
for i, v in enumerate(res_df['MAE_Log']): axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(res_df['Modelo'], res_df['MAPE'], color=colors)
axes[2].set_ylabel('MAPE (%)'); axes[2].set_title('MAPE por Modelo', fontweight='bold')
for i, v in enumerate(res_df['MAPE']): axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

axes[3].bar(res_df['Modelo'], res_df['MSLE'], color=colors)
axes[3].set_ylabel('MSLE'); axes[3].set_title('MSLE por Modelo', fontweight='bold')
for i, v in enumerate(res_df['MSLE']): axes[3].text(i, v + 0.0001, f'{v:.4f}', ha='center', fontweight='bold')
axes[3].tick_params(axis='x', rotation=45)

plt.suptitle('Comparacao de Modelos', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(DIR_MODEL / 'resultados' / 'comparacao_modelos.png', dpi=300)
plt.close()

print("\n=== Residuos ===")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

modelos_plot = [('OLS', res_ols), ('Ridge', res_ridge), ('Lasso', res_lasso),
                ('Random Forest', res_rf), ('HGB', res_hgb), ('XGBoost', res_xgb if has_xgb else res_hgb)]

for ax, (nome, res) in zip(axes, modelos_plot):
    yp = res['y_pred']
    resid = y_test - yp
    ax.scatter(yp, resid, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Preditos (Log)'); ax.set_ylabel('Residuos')
    ax.set_title(nome, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if nome in ['OLS', 'Ridge']:
        resid_sq = resid ** 2
        bp_corr = np.corrcoef(yp, resid_sq)[0, 1]
        ax.text(0.05, 0.95, f'BP corr: {bp_corr:.3f}', transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Analise de Residuos', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(DIR_MODEL / 'resultados' / 'residuos_modelos.png', dpi=300)
plt.close()

print("\n=== Interpretabilidade ===")

# OLS
ols_coef = pd.DataFrame({'Feature': feature_names, 'Coeficiente': pipe_ols.named_steps['reg'].coef_})
ols_coef = ols_coef.sort_values('Coeficiente', key=abs, ascending=False)
print("\n--- Coeficientes OLS ---")
print(ols_coef.to_string(index=False))
ols_coef.to_csv(DIR_INTERP / 'coeficientes_ols.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 10))
cp = ols_coef.sort_values('Coeficiente')
ax.barh(cp['Feature'], cp['Coeficiente'], color=['#e74c3c' if c < 0 else '#2ecc71' for c in cp['Coeficiente']])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficientes OLS', fontweight='bold')
ax.set_xlabel('Coeficiente padronizado')
plt.tight_layout()
plt.savefig(DIR_INTERP / 'coeficientes_ols.png', dpi=300)
plt.close()

# Ridge vs Lasso
ridge_coef = pd.DataFrame({
    'Feature': feature_names,
    'Ridge': grid_ridge.best_estimator_.named_steps['reg'].coef_,
    'Lasso': grid_lasso.best_estimator_.named_steps['reg'].coef_
})
ridge_coef.to_csv(DIR_INTERP / 'coeficientes_ridge_lasso.csv', index=False)
fig, ax = plt.subplots(figsize=(10, 10))
xp = np.arange(len(feature_names)); w = 0.35
ax.barh(xp - w / 2, ridge_coef['Ridge'], w, label='Ridge', color='#3498db')
ax.barh(xp + w / 2, ridge_coef['Lasso'], w, label='Lasso', color='#e74c3c')
ax.set_yticks(xp); ax.set_yticklabels(feature_names, fontsize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Ridge vs Lasso', fontweight='bold')
ax.set_xlabel('Coeficiente'); ax.legend()
plt.tight_layout()
plt.savefig(DIR_INTERP / 'coeficientes_ridge_lasso.png', dpi=300)
plt.close()

# Permutation importance (melhor modelo tree-based)
if has_xgb and res_xgb['R2_Test'] >= res_hgb['R2_Test']:
    best_tree_model = res_xgb
    best_tree_name = 'XGBoost'
else:
    best_tree_model = res_hgb
    best_tree_name = 'HGB'
print(f"\n--- Importancia Permutacao ({best_tree_name}) ---")

if best_tree_name == 'XGBoost':
    X_test_proc_best = grid_xgb.best_estimator_.named_steps['preprocess'].transform(X_test)
    tree_model = grid_xgb.best_estimator_.named_steps['reg']
else:
    X_test_proc_best = grid_hgb.best_estimator_.named_steps['preprocess'].transform(X_test)
    tree_model = grid_hgb.best_estimator_.named_steps['reg']

perm_imp = permutation_importance(
    tree_model, X_test_proc_best, y_test,
    n_repeats=10, random_state=42, scoring='r2'
)

imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importancia': perm_imp.importances_mean,
    'Desvio': perm_imp.importances_std
})
imp_df = imp_df.sort_values('Importancia', ascending=False)
print(imp_df.head(10).to_string(index=False))
imp_df.to_csv(DIR_INTERP / 'importancia_permutacao' / 'importancia.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 10))
ip = imp_df.sort_values('Importancia')
ax.barh(ip['Feature'], ip['Importancia'], xerr=ip['Desvio'], color='#9b59b6', capsize=3)
ax.set_title(f'Importancia por Permutacao - {best_tree_name}', fontweight='bold')
ax.set_xlabel('Queda no R2')
plt.tight_layout()
plt.savefig(DIR_INTERP / 'importancia_permutacao' / 'importancia.png', dpi=300)
plt.close()

# Figura para o relatório: somente variáveis relevantes + grupo agregado.
imp_report = imp_df.copy()
imp_report['Importancia_pos'] = imp_report['Importancia'].clip(lower=0)
top_imp = imp_report.head(9).copy()
rest_imp = imp_report.iloc[9:]
if len(rest_imp) > 0:
    outros = pd.DataFrame([{
        'Feature': f'Outras variáveis ({len(rest_imp)})',
        'Importancia': rest_imp['Importancia_pos'].sum(),
        'Desvio': 0.0,
        'Importancia_pos': rest_imp['Importancia_pos'].sum(),
    }])
    top_imp = pd.concat([top_imp, outros], ignore_index=True)
top_imp['Feature_plot'] = top_imp['Feature'].map(_report_label)
top_imp = top_imp.sort_values('Importancia_pos')
fig, ax = plt.subplots(figsize=(8.8, 6.2))
colors = ['#8b5cf6' if 'Outras' not in f else '#94a3b8' for f in top_imp['Feature']]
ax.barh(top_imp['Feature_plot'], top_imp['Importancia_pos'], color=colors)
ax.set_title('Importância por permutação (HGB)', fontsize=16, fontweight='bold')
ax.set_xlabel('Queda no $R^2$ ao permutar a variável', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.grid(axis='x', alpha=0.25)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(REPORT_FIG_DIR / 'importancia_relatorio.pdf', bbox_inches='tight')
plt.savefig(REPORT_FIG_DIR / 'importancia_relatorio.png', dpi=300, bbox_inches='tight')
plt.close()

# PDP (top 6 features)
top6 = imp_df.head(6)['Feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for ax, feat in zip(axes, top6):
    fi = feature_names.index(feat)
    pd_res = partial_dependence(tree_model, X_test_proc_best, features=[fi], kind='average', grid_resolution=50)
    ax.plot(pd_res['grid_values'][0], pd_res['average'][0], color='#e74c3c', lw=2.5)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Efeito marginal (Log)')
    ax.grid(True, alpha=0.3)
fig.suptitle(f'PDP - {best_tree_name}', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(DIR_INTERP / 'graficos_pdp' / 'pdp_modelo.png', dpi=300)
plt.close()

# Figura para relatório: PDPs principais, com fonte maior.
pdp_features = [f for f in ['MP', 'Age', 'PTS_per_GP', 'USG%'] if f in feature_names]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
axes = axes.flatten()
for ax, feat in zip(axes, pdp_features):
    fi = feature_names.index(feat)
    pd_res = partial_dependence(tree_model, X_test_proc_best, features=[fi], kind='average', grid_resolution=60)
    ax.plot(pd_res['grid_values'][0], pd_res['average'][0], color='#ef4444', lw=3)
    ax.set_title(_report_label(feat), fontsize=14, fontweight='bold')
    ax.set_xlabel(_report_label(feat), fontsize=12)
    ax.set_ylabel('Efeito parcial no log(salário)', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.28)
for ax in axes[len(pdp_features):]:
    ax.axis('off')
fig.suptitle('Efeitos parciais (PDP) do HGB', fontsize=17, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(REPORT_FIG_DIR / 'pdp_relatorio.pdf', bbox_inches='tight')
plt.savefig(REPORT_FIG_DIR / 'pdp_relatorio.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP
print("\n--- SHAP ---")
try:
    import shap

    if has_xgb:
        explainer = shap.TreeExplainer(grid_xgb.best_estimator_.named_steps['reg'])
        X_test_shap = grid_xgb.best_estimator_.named_steps['preprocess'].transform(X_test)
        modelo_shap = "XGBoost"
    else:
        explainer = shap.TreeExplainer(grid_hgb.best_estimator_.named_steps['reg'])
        X_test_shap = grid_hgb.best_estimator_.named_steps['preprocess'].transform(X_test)
        modelo_shap = "HistGradientBoosting"

    sv = explainer.shap_values(X_test_shap)

    if isinstance(sv, list):
        sv_plot = sv[0]
    else:
        sv_plot = sv

    ev = explainer.expected_value
    if isinstance(ev, (np.ndarray, list)):
        ev_scalar = float(ev[0]) if len(np.atleast_1d(ev)) > 0 else float(ev)
    else:
        ev_scalar = float(ev)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_test_shap, feature_names=feature_names, show=False, plot_size=(10, 8))
    plt.title(f'SHAP Summary - {modelo_shap}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(DIR_INTERP / 'shap_values' / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot global
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_test_shap, feature_names=feature_names, plot_type='bar', show=False, plot_size=(10, 8))
    plt.title(f'SHAP Global - {modelo_shap}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(DIR_INTERP / 'shap_values' / 'shap_importancia_global.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figuras para relatório: cortar variáveis próximas de zero e agregar o restante.
    X_test_dense = X_test_shap.toarray() if hasattr(X_test_shap, 'toarray') else np.asarray(X_test_shap)
    mean_abs_shap = np.abs(sv_plot).mean(axis=0)
    shap_order = np.argsort(mean_abs_shap)[::-1]
    top_n_shap = 8
    top_idx = shap_order[:top_n_shap]
    rest_idx = shap_order[top_n_shap:]
    top_names = [_report_label(feature_names[i]) for i in top_idx]

    shap.summary_plot(
        sv_plot[:, top_idx],
        X_test_dense[:, top_idx],
        feature_names=top_names,
        show=False,
        plot_size=(9.5, 6.2),
        max_display=top_n_shap,
    )
    fig = plt.gcf()
    if len(fig.axes) > 1:
        color_axis = fig.axes[-1]
        color_axis.set_ylabel('Valor da variável', fontsize=12)
        color_axis.set_yticklabels(['Baixo', 'Alto'])
    plt.title('Resumo SHAP - principais variáveis', fontsize=16, fontweight='bold')
    plt.xlabel('Impacto na predição do log(salário)', fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORT_FIG_DIR / 'shap_summary_relatorio.pdf', bbox_inches='tight')
    plt.savefig(REPORT_FIG_DIR / 'shap_summary_relatorio.png', dpi=300, bbox_inches='tight')
    plt.close()

    shap_bar_vals = list(mean_abs_shap[top_idx])
    shap_bar_names = top_names.copy()
    if len(rest_idx) > 0:
        shap_bar_vals.append(float(mean_abs_shap[rest_idx].sum()))
        shap_bar_names.append(f'Outras variáveis ({len(rest_idx)})')
    order_bar = np.argsort(shap_bar_vals)
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    bar_vals = np.array(shap_bar_vals)[order_bar]
    bar_names = np.array(shap_bar_names)[order_bar]
    colors = ['#2563eb' if 'Outras' not in name else '#94a3b8' for name in bar_names]
    ax.barh(bar_names, bar_vals, color=colors)
    ax.set_title('Importância global SHAP', fontsize=16, fontweight='bold')
    ax.set_xlabel('Média de |SHAP| no log(salário)', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(axis='x', alpha=0.25)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(REPORT_FIG_DIR / 'shap_global_relatorio.pdf', bbox_inches='tight')
    plt.savefig(REPORT_FIG_DIR / 'shap_global_relatorio.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Waterfall por perfil
    names_test = df.loc[X_test.index, 'Player Name'].reset_index(drop=True)
    y_test_sal = np.exp(y_test).reset_index(drop=True)

    perfis = [
        ('Stephen Curry', 'superstar', 'curry'),
        ('Frank Kaminsky', 'role_player', 'kaminsky'),
        ('Jaden Hardy', 'rookie', 'hardy'),
    ]

    SHAP_LABELS = {
        'MP': 'Minutos (MP)', 'Age': 'Idade (Age)', 'USG%': 'Uso (USG%)',
        'BPM': 'Eficiência (BPM)', 'PTS_per_GP': 'Pontos (PTS/GP)', 'PTS': 'Pontos (PTS)',
        '3P%': '3P%', 'VORP': 'VORP', 'GP': 'Jogos (GP)', 'Age_sq': 'Age²',
        'BLK_per_GP': 'Tocos/GP', 'AST_to_TOV': 'AST/TOV',
    }

    def _shap_profile_usd(idx):
        sh = sv_plot[idx]
        base_log = ev_scalar
        pred_log = base_log + float(sh.sum())
        order = np.argsort(-np.abs(sh))[:5]
        forces = []
        cum = base_log
        for fi in order:
            nxt = cum + sh[fi]
            delta = (np.exp(nxt) - np.exp(cum)) / 1e6
            forces.append({
                'name': SHAP_LABELS.get(feature_names[fi], feature_names[fi]),
                'val': round(float(delta), 2),
                'positive': bool(delta >= 0),
            })
            cum = nxt
        outros = (np.exp(pred_log) - np.exp(cum)) / 1e6
        if abs(outros) >= 0.05:
            forces.append({
                'name': 'Outros',
                'val': round(float(outros), 2),
                'positive': bool(outros >= 0),
            })
        return {
            'baseVal': round(float(np.exp(base_log) / 1e6), 2),
            'predVal': round(float(np.exp(pred_log) / 1e6), 2),
            'realVal': round(float(y_test_sal.iloc[idx] / 1e6), 2),
            'forces': forces,
        }

    shap_profiles_export = {}

    for player_name, perfil, js_key in perfis:
        player_idx = names_test[names_test == player_name].index
        if len(player_idx) > 0:
            idx = player_idx[0]
            shap_profiles_export[js_key] = _shap_profile_usd(idx)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv_plot[idx],
                    base_values=ev_scalar,
                    data=X_test_shap[idx],
                    feature_names=feature_names
                ),
                show=False, max_display=15
            )
            plt.title(f'SHAP Waterfall - {player_name} ({perfil})', fontweight='bold')
            plt.tight_layout()
            plt.savefig(DIR_INTERP / 'shap_values' / f'shap_waterfall_{perfil}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   Waterfall para {player_name} ({perfil}) gerado.")

    (DIR_INTERP / 'shap_values' / 'shap_profiles.json').write_text(
        _json.dumps(shap_profiles_export, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print("   shap_profiles.json exportado.")

    if 'curry' in shap_profiles_export:
        profile = shap_profiles_export['curry']
        forces = profile['forces']
        contrib = pd.DataFrame(forces)
        contrib = contrib.sort_values('val')
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        colors = np.where(contrib['val'] >= 0, '#16a34a', '#dc2626')
        ax.barh(contrib['name'], contrib['val'], color=colors, alpha=0.92)
        ax.axvline(0, color='#111827', lw=1)
        for y, val in enumerate(contrib['val']):
            ha = 'left' if val >= 0 else 'right'
            offset = 0.35 if val >= 0 else -0.35
            ax.text(val + offset, y, f'{val:+.1f}M', va='center', ha=ha, fontsize=12, fontweight='bold')
        ax.text(
            0.99, 0.05,
            f"Base: US$ {profile['baseVal']:.1f}M   |   Predito: US$ {profile['predVal']:.1f}M   |   Real: US$ {profile['realVal']:.1f}M",
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#cbd5e1')
        )
        ax.set_xlabel('Contribuição na predição (US$ milhões)', fontsize=13)
        ax.set_title('Contribuições SHAP locais - Stephen Curry', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(axis='x', alpha=0.25)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(REPORT_FIG_DIR / 'shap_curry_contribuicoes_relatorio.pdf', bbox_inches='tight')
        plt.savefig(REPORT_FIG_DIR / 'shap_curry_contribuicoes_relatorio.png', dpi=300, bbox_inches='tight')
        plt.close()

    topf = imp_df.iloc[0]['Feature']
    tfi = feature_names.index(topf)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(tfi, sv_plot, X_test_shap, feature_names=feature_names, show=False, ax=ax)
    plt.title(f'SHAP Dependence - {topf}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(DIR_INTERP / 'shap_values' / 'shap_dependence_top.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP ({modelo_shap}) gerado com sucesso.")

except ImportError:
    print("SHAP nao instalado. Pulando.")
except Exception as e:
    print(f"Erro SHAP: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Erros extremos ===")
best_y_pred = best_tree_model['y_pred']
best_y_pred_sal = np.exp(best_y_pred)
yte_sal = np.exp(y_test)
erros = np.abs(yte_sal - best_y_pred_sal)

error_df = pd.DataFrame({
    'Player': df.loc[X_test.index, 'Player Name'].values,
    'Position': df.loc[X_test.index, 'Position_Clean'].values,
    'Age': df.loc[X_test.index, 'Age'].values,
    'Actual_Salary': yte_sal.values,
    'Predicted_Salary': best_y_pred_sal,
    'Error_USD': erros.values,
    'Error_Pct': (erros / yte_sal * 100).values
}).sort_values('Error_USD', ascending=False)

print("\nTop 10 maiores erros (em USD):")
print(error_df.head(10).to_string(index=False))
error_df.to_csv(DIR_MODEL / 'resultados' / 'erros_extremos.csv', index=False)

# MAE por perfil (conjunto de teste)
def player_profile(row):
    if row['Salary'] > 15_000_000 and row['GP'] < 15:
        return 'Lesionados'
    if row['Salary'] > 30_000_000 and row['Age'] >= 30:
        return 'Supermax antigos'
    if row['Salary'] > 28_000_000 and row['BPM'] > 4:
        return 'Superstars'
    if row['Age'] <= 22:
        return 'Rookies (CBA)'
    return 'Role players'

profile_df = pd.DataFrame({
    'Player': df.loc[X_test.index, 'Player Name'].values,
    'Salary': yte_sal.values,
    'GP': df.loc[X_test.index, 'GP'].values,
    'Age': df.loc[X_test.index, 'Age'].values,
    'BPM': df.loc[X_test.index, 'BPM'].values,
    'Error_USD': erros.values,
    'Predicted_Salary': best_y_pred_sal,
})
profile_df['group'] = profile_df.apply(player_profile, axis=1)
mae_rows = []
for group, g in profile_df.groupby('group'):
    mae_rows.append({
        'group': group,
        'n': len(g),
        'mae_m_usd': round(g['Error_USD'].mean() / 1e6, 2),
    })
mae_profile_df = pd.DataFrame(mae_rows).sort_values('group')
mae_profile_df.to_csv(DIR_MODEL / 'resultados' / 'mae_por_perfil.csv', index=False)
print("\nMAE por perfil (HGB, teste):")
print(mae_profile_df.to_string(index=False))
print(f"MAE global teste: ${erros.mean():,.0f}")

print("\n=== Concluído ===")
print(f"Artefatos em {DIR_EDA.name}/, {DIR_PREPROC.name}/, {DIR_MODEL.name}/, {DIR_INTERP.name}/")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
for report_name in [
    'eda_relatorio.pdf',
    'importancia_relatorio.pdf',
    'pdp_relatorio.pdf',
    'shap_summary_relatorio.pdf',
    'shap_global_relatorio.pdf',
    'shap_curry_contribuicoes_relatorio.pdf',
]:
    src = REPORT_FIG_DIR / report_name
    if src.exists():
        shutil.copy2(src, REPORT_DIR / report_name)
print(f"Figuras do relatório copiadas para {REPORT_DIR.relative_to(REPO_ROOT)}/")
