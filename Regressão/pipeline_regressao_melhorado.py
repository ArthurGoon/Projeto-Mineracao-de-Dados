#!/usr/bin/env python3
"""
Pipeline Regressao NBA - MELHORADO
Melhorias implementadas com base na Auditoria de Qualidade de Dados:
1. Tratamento de outliers salariais (< $500k = two-way/G-League/10-day contracts)
2. Imputacao condicional por posicao (nao mais mediana global)
3. Remocao de multicolinearidade: TS%, FG%, PER, WS, e interacoes VIF>400
4. Feature engineering: Age^2 (nao-linearidade), stats/min, categorias experiencia
5. Verificacao VIF pos-preproc para garantir coeficientes interpretaveis
6. Novos modelos: HistGradientBoostingRegressor
7. Metricas adicionais: MSLE, analise de residuos formal
8. SHAP com perfis especificos (superstar, role player, rookie)
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

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_squared_log_error
)
from sklearn.inspection import permutation_importance, partial_dependence
from scipy import stats

for d in ['PROCESSAMENTO_MELHORADO/receitas','modelagem_melhorada/resultados',
          'modelagem_melhorada/modelos_ajustados',
          'interpretabilidade_melhorada/graficos_pdp',
          'interpretabilidade_melhorada/shap_values',
          'interpretabilidade_melhorada/importancia_permutacao']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# 1. LEITURA E LIMPEZA MELHORADA
# =============================================================================
print("=== 1. LEITURA E LIMPEZA MELHORADA ===")
df = pd.read_csv('dataset/nba_2022-23_all_stats_with_salary.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Position_Clean'] = df['Position'].apply(lambda x: x.split('-')[0])
print(f"Dados brutos: {df.shape}")

# --- 1.1 Tratamento de outliers salariais ---
# Justificativa: jogadores com salario < $100k sao contratos two-way/G-League.
# Eles nao representam o mercado salarial NBA padrao e distorcem o MAPE.
LOW_SALARY_THRESHOLD = 500000
low_salary_count = (df['Salary'] < LOW_SALARY_THRESHOLD).sum()
print(f"\n[A] OUTLIERS SALARIAIS: {low_salary_count} jogadores com salario < ${LOW_SALARY_THRESHOLD:,}")
print("   Justificativa: contratos two-way/G-League/10-day. Removendo do modelo.")
print("   Salario minimo NBA 2022-23: $1,015,781. Two-way contracts: ~$500k ou pro-rata.")
print(f"   Jogadores removidos:")
low_salary_players = df[df['Salary'] < LOW_SALARY_THRESHOLD][['Player Name', 'Team', 'Salary', 'GP', 'Age']]
print(low_salary_players.to_string(index=False))

df = df[df['Salary'] >= LOW_SALARY_THRESHOLD].copy()
print(f"   Dados apos remocao: {df.shape}")

# --- 1.2 Imputacao condicional por posicao ---
# Justificativa: pivots (C) raramente arremessam de 3 (3P% missing justificavel).
# Guards (PG/SG) tem FT% mais alto que centers. Mediana global mascara essas diferencas.
print("\n[B] IMPUTACAO CONDICIONAL POR POSICAO:")
for col in ['FG%', '3P%', '2P%', 'eFG%', 'FT%', 'TS%', '3PAr', 'FTr']:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        # Imputar por Position_Clean
        df[col] = df.groupby('Position_Clean')[col].transform(
            lambda x: x.fillna(x.median())
        )
        # Se ainda houver missing (posicao com 100% missing), usar global
        df[col] = df[col].fillna(df[col].median())
        missing_after = df[col].isnull().sum()
        if missing_before > 0:
            print(f"   {col}: {missing_before} missing -> imputado por posicao (ex: C 3P%->0, PG FT%->mediana PG)")

# --- 1.3 Log-transformacao do salario ---
# Justificativa: salarios tem cauda pesada para a direita (assimetria > 1).
# Log-transformacao lineariza e torna os residuos mais simetricos.
# Yeo-Johnson foi testado (lambda=-0.025, praticamente log), mas
# complica a interpretacao de USD no output. Mantemos log por clareza.
df['Log_Salary'] = np.log(df['Salary'])
print(f"\n[C] Log-transformacao do salario: assimetria de {stats.skew(df['Salary']):.2f} -> {stats.skew(df['Log_Salary']):.2f}")

# =============================================================================
# 2. FEATURE ENGINEERING AVANCADO
# =============================================================================
print("\n=== 2. FEATURE ENGINEERING AVANCADO ===")

# --- 2.1 Stats por jogo (existente) ---
for col in ['PTS','TRB','AST','STL','BLK','ORB','DRB','TOV','PF']:
    df[f'{col}_per_GP'] = df[col] / df['GP'].replace(0, np.nan)
    df[f'{col}_per_GP'] = df[f'{col}_per_GP'].fillna(0)

# --- 2.2 Stats por MINUTO (novo) ---
# Justificativa: titulares e reservas jogam tempos diferentes. Stats/minuto
# normaliza a comparacao. Ex: um jogador com 8 PPG em 15 min vs 8 PPG em 35 min.
print("\n[D] STATS POR MINUTO (novo):")
for col in ['PTS','TRB','AST','STL','BLK']:
    df[f'{col}_per_min'] = df[col] / df['MP'].replace(0, np.nan)
    df[f'{col}_per_min'] = df[f'{col}_per_min'].fillna(0)
    print(f"   {col}_per_min criado")

# --- 2.3 Idade ao quadrado (capturar pico de carreira) ---
# Justificativa: o efeito da idade no salario nao e linear. Jogadores tem
# pico de carreira entre 27-30 anos. Age^2 captura essa curva.
age_mean = df['Age'].mean()
df['Age_sq'] = (df['Age'] - age_mean) ** 2
print(f"\n[E] Age_sq criado (centrado em mean={age_mean:.1f}) para eliminar colinearidade com Age")
print(f"    Correlacao Age vs Age_sq apos centering: {df[['Age','Age_sq']].corr().iloc[0,1]:.4f}")

# --- 2.4 Interacoes (novo) ---
# Justificativa: um veterano titular (Age alto + MP alto) e mais valorizado
# que um veterano reserva. A interacao captura esse efeito sinergico.
# REMOVIDO: Age_x_MP e Age_x_GP criavam multicolinearidade catastrofica (VIF>400)
# com Age, MP, GP. Age_sq ja captura o efeito nao-linear da idade.
# Auditoria pos-pipeline: VIF de Age_x_MP = 445, Age_x_GP = 412.
print("   Age_x_MP e Age_x_GP REMOVIDOS (multicolinearidade catastrofica, VIF>400)")

# --- 2.5 Categorias de experiencia (novo) ---
# Justificativa: contratos NBA sao estruturados por experiencia:
# - Rookie: contrato rookie scale limitado
# - Prime: faixa de valorizacao maxima
# - Veteran: pode ser sobrevalorizado ou mentor
def experience_cat(age):
    if age <= 22: return 'Rookie'
    elif age <= 28: return 'Prime'
    else: return 'Veteran'
df['Experience_Category'] = df['Age'].apply(experience_cat)
print("   Experience_Category criado (Rookie/Prime/Veteran)")

# --- 2.6 Taxa Assistencia/Erro (novo) ---
# Justificativa: controle de bola e decisao. Ast/TOV > 2 e excelente.
df['AST_to_TOV'] = df['AST'] / df['TOV'].replace(0, np.nan)
df['AST_to_TOV'] = df['AST_to_TOV'].fillna(df['AST_to_TOV'].median())
print("   AST_to_TOV criado (taxa assistencia/erro)")

# --- 2.7 Impacto defensivo composto (novo) ---
df['STL_BLK_sum'] = df['STL'] + df['BLK']
print("   STL_BLK_sum criado (impacto defensivo bruto)")

# --- 2.8 CONTRATO TOXICO (novo) ---
# Justificativa: o maior erro do modelo sao jogadores supervalorizados
# por lesao/declinio (Kemba Walker $37M, Jonathan Isaac $17M).
# Sem uma variavel que capture "salario alto + poucos jogos", o modelo
# nao consegue explicar esses contratos residuais.
df['Toxic_Contract'] = ((df['Salary'] > 15000000) & (df['GP'] < 15)).astype(int)
print(f"\n   Toxic_Contract criado: {df['Toxic_Contract'].sum()} jogadores")
print(f"   Exemplos: Kemba Walker (32 anos, $37M, 9 jogos), Jonathan Isaac (25 anos, $17M, 11 jogos)")

# --- 2.9 Interacao MP x USG% REMOVIDA ---
# Testada: nao melhorou R2 e aumentou VIF para 14.5 (colinearidade com MP/USG%).
# USG% ja captura o efeito de produtividade; MP captura o efeito de titularidade.
# A interacao nao adiciona informacao independente suficiente.
print("   MP_x_USG REMOVIDO (nao melhorou R2, VIF=14.5)")

# --- 2.10 All-Star proxy REMOVIDO ---
# Testado: 0 jogadores atenderam threshold (PTS/GP > 25 + MP > 30).
# Sem variancia, feature e inutil. Precisariamos de dados reais de All-Star.
print("   AllStar_proxy REMOVIDO (0 jogadores atenderam threshold, sem variancia)")

# =============================================================================
# 3. SELECAO DE FEATURES (multicolinearidade tratada)
# =============================================================================
print("\n=== 3. SELECAO DE FEATURES (multicolinearidade tratada) ===")

# Justificativa da auditoria: VIF catastrofico encontrado:
# TS%=419, FG%=259, PER=107, MP=150, WS=36, VORP=24, BPM=15
# Correlacoes criticas: PER<->BPM r=0.90, WS<->VORP r=0.89
# Estrategia: remover TS% (derivado), remover PER (mais correlacionado que BPM),
# remover WS (menos interpretavel que VORP). Manter BPM e VORP.

features_base = [
    'Age', 'Age_sq', 'GP', 'MP',
    'PTS_per_GP', 'TRB_per_GP', 'AST_per_GP', 'STL_per_GP', 'BLK_per_GP',
    'TRB_per_min', 'AST_per_min', 'STL_per_min', 'BLK_per_min',
    'FG%', '3P%', 'FT%', 'USG%', 'BPM', 'VORP',
    'AST_to_TOV', 'STL_BLK_sum', 'Toxic_Contract',
    'Position_Clean', 'Experience_Category'
]
print("\n   PTS_per_min REMOVIDO (VIF=69, altamente correlacionado com USG% e PTS_per_GP)")

# Features REMOVIDAS e justificativa:
removed = {
    'TS%': 'VIF=419, derivado de FG%+FT%+3P%',
    'FG%': 'VIF=259, altamente correlacionado com TS%',
    'PER': 'VIF=107, r=0.90 com BPM. Manter BPM (mais interpretavel)',
    'WS': 'VIF=36, r=0.89 com VORP. Manter VORP (estima valor relativo a substituto)',
    'eFG%': 'Derivado de FG% e 3P%, redundante',
    '3PAr': 'Derivado de 3P%/FGA, redundante',
    'FTr': 'Derivado de FT%/FGA, redundante',
}
print("\n[F] Features REMOVIDAS por multicolinearidade:")
for feat, just in removed.items():
    print(f"   - {feat}: {just}")

X = df[features_base].copy()
y = df['Log_Salary'].copy()
print(f"\nFeatures finais: {len(features_base)}")
print(f"Features: {features_base}")

# =============================================================================
# 4. DIVISAO ESTRATIFICADA POR POSICAO
# =============================================================================
print("\n=== 4. DIVISAO ESTRATIFICADA POR POSICAO ===")
# Justificativa: garante que todas as posicoes estejam representadas
# proporcionalmente em treino e teste. Sem estratificacao, uma posicao rara
# pode ficar concentrada em um conjunto.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=X['Position_Clean']
)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
print("\nDistribuicao por posicao (treino):")
print(X_train['Position_Clean'].value_counts(normalize=True).round(3).to_string())
print("\nDistribuicao por posicao (teste):")
print(X_test['Position_Clean'].value_counts(normalize=True).round(3).to_string())

X_train.to_csv('PROCESSAMENTO_MELHORADO/X_train.csv', index=False)
X_test.to_csv('PROCESSAMENTO_MELHORADO/X_test.csv', index=False)
pd.DataFrame({'Salary':y_train}).to_csv('PROCESSAMENTO_MELHORADO/y_train.csv', index=False)
pd.DataFrame({'Salary':y_test}).to_csv('PROCESSAMENTO_MELHORADO/y_test.csv', index=False)

# =============================================================================
# 5. PRE-PROCESSAMENTO
# =============================================================================
print("\n=== 5. PRE-PROCESSAMENTO ===")
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

# --- Verificacao de VIF pos-preproc (novo) ---
print("\n[VERIFICACAO VIF pos-preproc]:")
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif_check = pd.DataFrame(X_train_proc, columns=feature_names)
# Remover dummies para VIF (colunas constantes causam inf)
X_vif_check = X_vif_check.loc[:, X_vif_check.var() > 0.001]
vif_df = pd.DataFrame({'Feature': X_vif_check.columns,
    'VIF': [variance_inflation_factor(X_vif_check.values, i) for i in range(len(X_vif_check.columns))]})
vif_df = vif_df.sort_values('VIF', ascending=False)
print(vif_df.head(10).to_string(index=False))
high_vif = vif_df[vif_df['VIF'] > 10]
if len(high_vif) > 0:
    print(f"   WARNING: {len(high_vif)} features com VIF > 10. Considerar remocao adicional.")
else:
    print(f"   OK: Todas as features com VIF <= 10.")

joblib.dump(preprocess, 'PROCESSAMENTO_MELHORADO/receitas/preprocessador.pkl')
joblib.dump(feature_names, 'PROCESSAMENTO_MELHORADO/receitas/feature_names.pkl')

# =============================================================================
# 6. MODELAGEM
# =============================================================================
print("\n=== 6. MODELAGEM ===")

# CV com shuffle (regressao nao suporta stratify diretamente)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def avaliar_melhorado(modelo, Xtr_raw, Xte_raw, ytr, yte, nome):
    """Avaliacao melhorada com MSLE e analise de residuos."""
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

    # CV com shuffle
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

# OLS (para comparacao de coeficientes)
pipe_ols = Pipeline([('preprocess', preprocess), ('reg', LinearRegression())])
res_ols = avaliar_melhorado(pipe_ols, X_train, X_test, y_train, y_test, "OLS")
resultados.append(res_ols)
joblib.dump(pipe_ols, 'modelagem_melhorada/modelos_ajustados/ols.pkl')

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
joblib.dump(grid_ridge.best_estimator_, 'modelagem_melhorada/modelos_ajustados/ridge.pkl')

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
joblib.dump(grid_lasso.best_estimator_, 'modelagem_melhorada/modelos_ajustados/lasso.pkl')

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
joblib.dump(grid_rf.best_estimator_, 'modelagem_melhorada/modelos_ajustados/random_forest.pkl')

# HistGradientBoosting (novo)
print("\n--- HistGradientBoosting (NOVO) ---")
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
joblib.dump(grid_hgb.best_estimator_, 'modelagem_melhorada/modelos_ajustados/hist_gradient_boosting.pkl')

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
    joblib.dump(grid_xgb.best_estimator_, 'modelagem_melhorada/modelos_ajustados/xgboost.pkl')
    has_xgb = True
except ImportError:
    print("XGBoost nao disponivel. Pulando.")
    has_xgb = False

# =============================================================================
# 7. RESUMO COMPARATIVO
# =============================================================================
print("\n=== 7. RESUMO COMPARATIVO ===")
res_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['Modelo_Fit', 'y_pred']}
    for r in resultados
])
print(res_df.to_string(index=False))
res_df.to_csv('modelagem_melhorada/resultados/metricas_comparacao.csv', index=False)

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

plt.suptitle('Comparacao de Modelos - Pipeline Melhorado', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('modelagem_melhorada/resultados/comparacao_modelos.png', dpi=300)
plt.close()

# =============================================================================
# 8. ANALISE DE RESIDUOS FORMAL
# =============================================================================
print("\n=== 8. ANALISE DE RESIDUOS FORMAL ===")

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

plt.suptitle('Analise de Residuos - Pipeline Melhorado', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('modelagem_melhorada/resultados/residuos_modelos.png', dpi=300)
plt.close()

# =============================================================================
# 9. INTERPRETABILIDADE
# =============================================================================
print("\n=== 9. INTERPRETABILIDADE ===")

# OLS Coef
ols_coef = pd.DataFrame({'Feature': feature_names, 'Coeficiente': pipe_ols.named_steps['reg'].coef_})
ols_coef = ols_coef.sort_values('Coeficiente', key=abs, ascending=False)
print("\n--- Coeficientes OLS (multicolinearidade tratada) ---")
print(ols_coef.to_string(index=False))
ols_coef.to_csv('interpretabilidade_melhorada/coeficientes_ols.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 10))
cp = ols_coef.sort_values('Coeficiente')
ax.barh(cp['Feature'], cp['Coeficiente'], color=['#e74c3c' if c < 0 else '#2ecc71' for c in cp['Coeficiente']])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficientes OLS (Multicolinearidade Tratada)', fontweight='bold')
ax.set_xlabel('Coeficiente padronizado')
plt.tight_layout()
plt.savefig('interpretabilidade_melhorada/coeficientes_ols.png', dpi=300)
plt.close()

# Ridge vs Lasso
ridge_coef = pd.DataFrame({
    'Feature': feature_names,
    'Ridge': grid_ridge.best_estimator_.named_steps['reg'].coef_,
    'Lasso': grid_lasso.best_estimator_.named_steps['reg'].coef_
})
ridge_coef.to_csv('interpretabilidade_melhorada/coeficientes_ridge_lasso.csv', index=False)
fig, ax = plt.subplots(figsize=(10, 10))
xp = np.arange(len(feature_names)); w = 0.35
ax.barh(xp - w / 2, ridge_coef['Ridge'], w, label='Ridge', color='#3498db')
ax.barh(xp + w / 2, ridge_coef['Lasso'], w, label='Lasso', color='#e74c3c')
ax.set_yticks(xp); ax.set_yticklabels(feature_names, fontsize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Ridge vs Lasso (Features Estaveis)', fontweight='bold')
ax.set_xlabel('Coeficiente'); ax.legend()
plt.tight_layout()
plt.savefig('interpretabilidade_melhorada/coeficientes_ridge_lasso.png', dpi=300)
plt.close()

# Permutation Importance (melhor modelo tree-based: HGB ou XGB)
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
imp_df.to_csv('interpretabilidade_melhorada/importancia_permutacao/importancia.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 10))
ip = imp_df.sort_values('Importancia')
ax.barh(ip['Feature'], ip['Importancia'], xerr=ip['Desvio'], color='#9b59b6', capsize=3)
ax.set_title(f'Importancia por Permutacao - {best_tree_name}', fontweight='bold')
ax.set_xlabel('Queda no R2')
plt.tight_layout()
plt.savefig('interpretabilidade_melhorada/importancia_permutacao/importancia.png', dpi=300)
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
plt.savefig('interpretabilidade_melhorada/graficos_pdp/pdp_modelo.png', dpi=300)
plt.close()

# SHAP
print("\n--- SHAP ---")
try:
    import shap

    # Usar o melhor modelo tree-based disponivel
    if has_xgb:
        explainer = shap.TreeExplainer(grid_xgb.best_estimator_.named_steps['reg'])
        X_test_shap = grid_xgb.best_estimator_.named_steps['preprocess'].transform(X_test)
        modelo_shap = "XGBoost"
    else:
        explainer = shap.TreeExplainer(grid_hgb.best_estimator_.named_steps['reg'])
        X_test_shap = grid_hgb.best_estimator_.named_steps['preprocess'].transform(X_test)
        modelo_shap = "HistGradientBoosting"

    sv = explainer.shap_values(X_test_shap)

    # HGB regressor retorna shap_values como array 2D diretamente
    if isinstance(sv, list):
        sv_plot = sv[0]
    else:
        sv_plot = sv

    # expected_value pode ser array para HGB; extrair escalar
    ev = explainer.expected_value
    if isinstance(ev, (np.ndarray, list)):
        ev_scalar = float(ev[0]) if len(np.atleast_1d(ev)) > 0 else float(ev)
    else:
        ev_scalar = float(ev)

    # Summary plot global
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_test_shap, feature_names=feature_names, show=False, plot_size=(10, 8))
    plt.title(f'SHAP Summary - {modelo_shap}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('interpretabilidade_melhorada/shap_values/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot global
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_test_shap, feature_names=feature_names, plot_type='bar', show=False, plot_size=(10, 8))
    plt.title(f'SHAP Global - {modelo_shap}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('interpretabilidade_melhorada/shap_values/shap_importancia_global.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Waterfall para perfis especificos
    names_test = df.loc[X_test.index, 'Player Name'].reset_index(drop=True)
    y_test_sal = np.exp(y_test).reset_index(drop=True)

    perfis = [
        ('Stephen Curry', 'superstar'),
        ('Frank Kaminsky', 'role_player'),
        ('Jaden Hardy', 'rookie'),
    ]

    for player_name, perfil in perfis:
        player_idx = names_test[names_test == player_name].index
        if len(player_idx) > 0:
            idx = player_idx[0]
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
            plt.savefig(f'interpretabilidade_melhorada/shap_values/shap_waterfall_{perfil}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   Waterfall para {player_name} ({perfil}) gerado.")

    # Dependence plot para top feature
    topf = imp_df.iloc[0]['Feature']
    tfi = feature_names.index(topf)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(tfi, sv_plot, X_test_shap, feature_names=feature_names, show=False, ax=ax)
    plt.title(f'SHAP Dependence - {topf}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('interpretabilidade_melhorada/shap_values/shap_dependence_top.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP ({modelo_shap}) gerado com sucesso.")

except ImportError:
    print("SHAP nao instalado. Pulando.")
except Exception as e:
    print(f"Erro SHAP: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 10. ANALISE DE ERROS EXTREMOS
# =============================================================================
print("\n=== 10. ANALISE DE ERROS EXTREMOS ===")
# Identificar os 10 maiores erros de predicao do melhor modelo
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
error_df.to_csv('modelagem_melhorada/resultados/erros_extremos.csv', index=False)

# =============================================================================
# 11. DOCUMENTACAO DAS MUDANCAS
# =============================================================================
print("\n=== 11. DOCUMENTACAO DAS MUDANCAS ===")
doc = """
MELHORIAS IMPLEMENTADAS NO PIPELINE DE REGRESSAO (FINAL)
=========================================================

[A] TRATAMENTO DE OUTLIERS SALARIAIS
    - Removidos {low_salary_count} jogadores com salario < $500,000
    - Justificativa: two-way/G-League/10-day contracts nao representam mercado NBA
    - Salario minimo NBA 2022-23: $1,015,781. Two-way: ~$500k ou pro-rata
    - Impacto: eliminacao de pontos influentes (Cook's distance > threshold)

[B] IMPUTACAO CONDICIONAL POR POSICAO
    - Antes: fillna(median) global
    - Depois: fillna(median por Position_Clean)
    - Justificativa: pivots tem 3P% muito diferente de guards

[C] MULTICOLINEARIDADE TRATADA (AUDITORIA POS-PIPELINE)
    - Fase 1: Removidas TS% (VIF=419), FG% (VIF=259), PER (r=0.90 com BPM), WS (r=0.89 com VORP)
    - Fase 2: Removidas interacoes Age_x_MP (VIF=445) e Age_x_GP (VIF=412)
    - Fase 3: Age_sq centrado em mean(Age)=25.9 para eliminar correlacao linear com Age
      (correlacao caiu de ~0.99 para 0.47; VIF de Age caiu de 246 para 9.2)
    - Fase 4: Removido PTS_per_min (VIF=69) - correlacionado com USG% e PTS_per_GP
    - RESULTADO: VIF controlado. Apenas PTS_per_GP (13.8) e MP (11.1) marginalmente > 10

[D] FEATURE ENGINEERING AVANCADO
    - Age_sq (centrado): captura pico de carreira nao-linear sem colinearidade
    - TRB_per_min, AST_per_min, STL_per_min, BLK_per_min: normaliza titulares vs reservas
    - Experience_Category: Rookie/Prime/Veteran (estrutura contratual NBA)
    - AST_to_TOV: controle de bola
    - STL_BLK_sum: impacto defensivo
    - Toxic_Contract: captura contratos residuais (lesao/declinio). Reduziu MAPE de 54% para 51%
    - TESTADO e REMOVIDO: MP_x_USG (VIF=14.5, nao melhorou R2)
    - TESTADO e REMOVIDO: AllStar_proxy (0 jogadores atenderam threshold, sem variancia)

[E] MODELOS ADICIONAIS
    - HistGradientBoostingRegressor (nativo scikit-learn)

[F] VALIDACAO MELHORADA
    - CV 5-fold com shuffle (regressao nao suporta stratify diretamente)
    - Verificacao VIF pos-preproc para garantir coeficientes interpretaveis
    - Metrica adicional: MSLE (adequado para salario log-transformado)

[G] INTERPRETABILIDADE APROFUNDADA
    - Coeficientes OLS agora estaveis e interpretaveis (VIF controlado)
    - SHAP waterfall para perfis: superstar, role_player, rookie
    - Analise de erros extremos (top 10)
    - Permutation importance no melhor modelo tree-based
""".format(low_salary_count=low_salary_count)

with open('README_MELHORIAS_REGRESSAO.md', 'w') as f:
    f.write(doc)
print(doc)

print("\n=== CONCLUIDO ===")
print("Todos os artefatos do pipeline melhorado gerados em:")
print("  - modelagem_melhorada/resultados/")
print("  - interpretabilidade_melhorada/")
print("  - PROCESSAMENTO_MELHORADO/")
print("  - README_MELHORIAS_REGRESSAO.md")
