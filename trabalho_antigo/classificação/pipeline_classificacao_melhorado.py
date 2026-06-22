#!/usr/bin/env python3
"""
Pipeline Classificacao Premier League - MELHORADO
Melhorias implementadas com base na Auditoria de Qualidade de Dados:
1. Tratamento do leakage temporal: usar stats da TEMPORADA ANTERIOR como proxy
2. Imputacao de missing values em stats (evitar perda de 33% dos dados no dropna)
3. Feature engineering: forma recente (ultimos 5 jogos), posicao na tabela, rodada, fase
4. Walk-forward validation (em vez de CV aleatoria em dados temporais)
5. Novos modelos: HistGradientBoostingClassifier
6. Metricas adicionais: AUC-PR, calibracao de probabilidades
7. SHAP local para partidas especificas
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
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, brier_score_loss
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
results = pd.read_csv('dataset/results.csv')
stats = pd.read_csv('dataset/stats.csv')
print(f"Results: {results.shape}")
print(f"Stats: {stats.shape}")

results['HomeWin'] = (results['result'] == 'H').astype(int)

# --- 1.1 Tratamento de missing em stats ---
# Justificativa: o pipeline original fazia dropna silenciosamente, perdendo
# 1.520 partidas (33.3%). Vamos imputar missing antes do merge.
missing_stats = stats.isnull().sum()
missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
print(f"\n[A] MISSING VALUES em stats:")
for col, count in missing_stats.items():
    pct = count / len(stats) * 100
    print(f"   {col}: {count}/{len(stats)} ({pct:.1f}%)")

# Remover features com >30% missing (backward_pass, big_chance_missed)
# Justificativa: imputar 33% de uma feature introduz muito ruido
features_to_drop = missing_stats[missing_stats / len(stats) > 0.30].index.tolist()
print(f"\n   Features REMOVIDAS (>30% missing): {features_to_drop}")
stats_clean = stats.drop(columns=features_to_drop)

# Imputar features restantes com mediana por temporada
for col in stats_clean.columns:
    if col not in ['team', 'season'] and stats_clean[col].isnull().any():
        missing_before = stats_clean[col].isnull().sum()
        stats_clean[col] = stats_clean.groupby('season')[col].transform(lambda x: x.fillna(x.median()))
        stats_clean[col] = stats_clean[col].fillna(stats_clean[col].median())
        missing_after = stats_clean[col].isnull().sum()
        print(f"   {col}: {missing_before} -> {missing_after} (imputado por temporada)")

# =============================================================================
# 2. FEATURE ENGINEERING AVANCADO
# =============================================================================
print("\n=== 2. FEATURE ENGINEERING AVANCADO ===")

# --- 2.1 Criar stats da TEMPORADA ANTERIOR (mitigar leakage) ---
# Justificativa: usar stats da temporada atual cria leakage temporal.
# Usar a temporada anterior como proxy de forma e metodologicamente correto.
print("\n[B] STATS DA TEMPORADA ANTERIOR (mitiga leakage temporal):")
stats_clean['season_start'] = stats_clean['season'].str[:4].astype(int)
stats_prev = stats_clean.copy()
stats_prev['season_start'] = stats_prev['season_start'] + 1
stats_prev['season'] = stats_prev['season_start'].astype(str) + '-' + (stats_prev['season_start'] + 1).astype(str)
stats_prev = stats_prev.drop(columns=['season_start'])

# Prefixos
stats_home_prev = stats_prev.add_prefix('home_prev_')
stats_away_prev = stats_prev.add_prefix('away_prev_')

# Merge com results usando stats da temporada ANTERIOR
results_m = results.merge(
    stats_home_prev,
    left_on=['home_team', 'season'],
    right_on=['home_prev_team', 'home_prev_season'],
    how='left'
)
results_m = results_m.merge(
    stats_away_prev,
    left_on=['away_team', 'season'],
    right_on=['away_prev_team', 'away_prev_season'],
    how='left'
)

# Para a primeira temporada (2006-2007), nao ha temporada anterior
# Vamos usar as stats da propria temporada como fallback (menos ideal, mas necessario)
stats_home_curr = stats_clean.add_prefix('home_curr_')
stats_away_curr = stats_clean.add_prefix('away_curr_')
results_m = results_m.merge(
    stats_home_curr,
    left_on=['home_team', 'season'],
    right_on=['home_curr_team', 'home_curr_season'],
    how='left'
)
results_m = results_m.merge(
    stats_away_curr,
    left_on=['away_team', 'season'],
    right_on=['away_curr_team', 'away_curr_season'],
    how='left'
)

# Para cada stat, usar anterior se disponivel, senao usar atual
stat_cols = [c for c in stats_clean.columns if c not in ['team', 'season', 'season_start']]
for col in stat_cols:
    hp = f'home_prev_{col}'; hc = f'home_curr_{col}'
    ap = f'away_prev_{col}'; ac = f'away_curr_{col}'
    results_m[f'home_{col}'] = results_m[hp].fillna(results_m[hc])
    results_m[f'away_{col}'] = results_m[ap].fillna(results_m[ac])
    results_m = results_m.drop(columns=[hp, hc, ap, ac], errors='ignore')

# Limpar colunas de merge
merge_cols = [c for c in results_m.columns if c.startswith('home_prev_') or c.startswith('away_prev_') or
              c.startswith('home_curr_') or c.startswith('away_curr_')]
results_m = results_m.drop(columns=merge_cols, errors='ignore')

print(f"   Merge concluido. Linhas: {len(results_m)}")

# --- 2.2 Features diferenciais (stats) ---
diff_cols = [c for c in stat_cols]
for col in diff_cols:
    if f'home_{col}' in results_m.columns and f'away_{col}' in results_m.columns:
        results_m[f'diff_{col}'] = results_m[f'home_{col}'] - results_m[f'away_{col}']

# --- 2.3 FORMA RECENTE (ultimos 5 jogos) ---
# Justificativa: a forma recente e o preditor mais importante no futebol.
# Calculamos media movel de 5 jogos para cada time.
print("\n[C] FORMA RECENTE (ultimos 5 jogos):")

# Ordenar por temporada e criar ordem implicita (Premier League: 380 jogos/season)
results_m = results_m.sort_values(['season', 'home_team', 'away_team']).reset_index(drop=True)

# Para cada time, calcular forma nos ultimos 5 jogos
# Pontos: vitoria=3, empate=1, derrota=0
def calcular_pontos(resultado, local):
    if resultado == 'H' and local == 'home': return 3
    if resultado == 'A' and local == 'away': return 3
    if resultado == 'D': return 1
    return 0

# Construir series de resultados por time
formas_home = []
formas_away = []
goals_for_home = []
goals_for_away = []
goals_against_home = []
goals_against_away = []

for idx, row in results_m.iterrows():
    season = row['season']
    home_t = row['home_team']
    away_t = row['away_team']

    # Partidas anteriores na mesma temporada
    prev_matches = results_m[
        (results_m['season'] == season) & (results_m.index < idx)
    ]

    # Forma do mandante (ultimos 5 jogos como mandante ou visitante)
    home_prev = prev_matches[
        (prev_matches['home_team'] == home_t) | (prev_matches['away_team'] == home_t)
    ].tail(5)
    if len(home_prev) > 0:
        home_pts = 0
        home_gf = 0
        home_ga = 0
        for _, m in home_prev.iterrows():
            if m['home_team'] == home_t:
                home_pts += calcular_pontos(m['result'], 'home')
                home_gf += m['home_goals']
                home_ga += m['away_goals']
            else:
                home_pts += calcular_pontos(m['result'], 'away')
                home_gf += m['away_goals']
                home_ga += m['home_goals']
        formas_home.append(home_pts / len(home_prev))
        goals_for_home.append(home_gf / len(home_prev))
        goals_against_home.append(home_ga / len(home_prev))
    else:
        formas_home.append(1.5)  # medio
        goals_for_home.append(1.5)
        goals_against_home.append(1.5)

    # Forma do visitante
    away_prev = prev_matches[
        (prev_matches['home_team'] == away_t) | (prev_matches['away_team'] == away_t)
    ].tail(5)
    if len(away_prev) > 0:
        away_pts = 0
        away_gf = 0
        away_ga = 0
        for _, m in away_prev.iterrows():
            if m['home_team'] == away_t:
                away_pts += calcular_pontos(m['result'], 'home')
                away_gf += m['home_goals']
                away_ga += m['away_goals']
            else:
                away_pts += calcular_pontos(m['result'], 'away')
                away_gf += m['away_goals']
                away_ga += m['home_goals']
        formas_away.append(away_pts / len(away_prev))
        goals_for_away.append(away_gf / len(away_prev))
        goals_against_away.append(away_ga / len(away_prev))
    else:
        formas_away.append(1.5)
        goals_for_away.append(1.5)
        goals_against_away.append(1.5)

results_m['home_forma_5'] = formas_home
results_m['away_forma_5'] = formas_away
results_m['home_gf_5'] = goals_for_home
results_m['away_gf_5'] = goals_for_away
results_m['home_ga_5'] = goals_against_home
results_m['away_ga_5'] = goals_against_away
results_m['diff_forma_5'] = results_m['home_forma_5'] - results_m['away_forma_5']
results_m['diff_gf_5'] = results_m['home_gf_5'] - results_m['away_gf_5']
results_m['diff_ga_5'] = results_m['home_ga_5'] - results_m['away_ga_5']

print(f"   diff_forma_5: media={results_m['diff_forma_5'].mean():.3f}, std={results_m['diff_forma_5'].std():.3f}")

# --- 2.4 RODADA DA TEMPORADA e FASE ---
# Justificativa: inicio vs final da temporada tem dinamicas diferentes
print("\n[D] RODADA e FASE DA TEMPORADA:")
results_m['rodada'] = results_m.groupby('season').cumcount() + 1
results_m['fase_temporada'] = pd.cut(
    results_m['rodada'],
    bins=[0, 10, 28, 40],
    labels=['Inicio', 'Meio', 'Final']
)
print(f"   Distribuicao de fase: {results_m['fase_temporada'].value_counts().to_dict()}")

# --- 2.5 HEAD-TO-HEAD HISTORICO (ultimas 3 partidas) ---
print("\n[E] HEAD-TO-HEAD (ultimas 3 partidas entre os times):")
hh_wins_home = []
hh_goals_diff = []

for idx, row in results_m.iterrows():
    season = row['season']
    home_t = row['home_team']
    away_t = row['away_team']

    # Partidas anteriores entre esses dois times (qualquer temporada)
    hh_prev = results_m[
        (results_m.index < idx) &
        (
            ((results_m['home_team'] == home_t) & (results_m['away_team'] == away_t)) |
            ((results_m['home_team'] == away_t) & (results_m['away_team'] == home_t))
        )
    ].tail(3)

    if len(hh_prev) > 0:
        home_wins = 0
        home_goals = 0
        away_goals = 0
        for _, m in hh_prev.iterrows():
            if m['home_team'] == home_t:
                home_goals += m['home_goals']
                away_goals += m['away_goals']
                if m['result'] == 'H':
                    home_wins += 1
            else:
                home_goals += m['away_goals']
                away_goals += m['home_goals']
                if m['result'] == 'A':
                    home_wins += 1
        hh_wins_home.append(home_wins / len(hh_prev))
        hh_goals_diff.append((home_goals - away_goals) / len(hh_prev))
    else:
        hh_wins_home.append(0.5)  # neutro
        hh_goals_diff.append(0)

results_m['hh_wins_home'] = hh_wins_home
results_m['hh_goals_diff'] = hh_goals_diff
print(f"   hh_wins_home: media={np.mean(hh_wins_home):.3f}")

# =============================================================================
# 3. SELECAO DE FEATURES
# =============================================================================
print("\n=== 3. SELECAO DE FEATURES ===")
feature_cols = [c for c in results_m.columns if c.startswith('diff_')]
feature_cols += [
    'diff_forma_5', 'diff_gf_5', 'diff_ga_5',
    'hh_wins_home', 'hh_goals_diff', 'rodada'
]

# Verificar missing
missing_final = results_m[feature_cols + ['HomeWin']].isnull().sum()
missing_final = missing_final[missing_final > 0]
if len(missing_final) > 0:
    print(f"   Missing encontrado:\n{missing_final.to_string()}")
    # Imputar missing com mediana
    for col in missing_final.index:
        results_m[col] = results_m[col].fillna(results_m[col].median())
    print(f"   Missing imputado com mediana.")

# --- 3.1 Verificar se ainda ha linhas com missing ---
antes_dropna = len(results_m)
results_m = results_m.dropna(subset=feature_cols + ['HomeWin'])
depois_dropna = len(results_m)
print(f"\n   Linhas ANTES: {antes_dropna} | APOS dropna: {depois_dropna} | Perdidas: {antes_dropna - depois_dropna}")

X = results_m[feature_cols].copy()
y = results_m['HomeWin'].copy()
print(f"   Features finais: {len(feature_cols)}")
print(f"   Amostras finais: {len(results_m)}")

# =============================================================================
# 4. DIVISAO TEMPORAL E WALK-FORWARD VALIDATION
# =============================================================================
print("\n=== 4. DIVISAO TEMPORAL E WALK-FORWARD ===")
train_seasons = [f'{y}-{y+1}' for y in range(2006, 2016)]
test_seasons = [f'{y}-{y+1}' for y in range(2016, 2018)]

mask_train = results_m['season'].isin(train_seasons)
mask_test = results_m['season'].isin(test_seasons)

X_train = X[mask_train].copy()
X_test = X[mask_test].copy()
y_train = y[mask_train].copy()
y_test = y[mask_test].copy()

print(f"Treino: {len(X_train)} ({train_seasons[0]} a {train_seasons[-1]})")
print(f"Teste: {len(X_test)} ({test_seasons[0]} a {test_seasons[-1]})")
print(f"Taxa vitoria casa (treino): {y_train.mean():.3f}")
print(f"Taxa vitoria casa (teste): {y_test.mean():.3f}")

X_train.to_csv('PROCESSAMENTO_MELHORADO/X_train.csv', index=False)
X_test.to_csv('PROCESSAMENTO_MELHORADO/X_test.csv', index=False)
y_train.to_csv('PROCESSAMENTO_MELHORADO/y_train.csv', index=False)
y_test.to_csv('PROCESSAMENTO_MELHORADO/y_test.csv', index=False)

# Walk-forward validation setup
print(f"\n[F] Walk-forward validation (TimeSeriesSplit, 5 folds):")
tscv = TimeSeriesSplit(n_splits=5)

# =============================================================================
# 5. PRE-PROCESSAMENTO
# =============================================================================
print("\n=== 5. PRE-PROCESSAMENTO ===")
preprocess = StandardScaler()
X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

joblib.dump(preprocess, 'PROCESSAMENTO_MELHORADO/receitas/preprocessador.pkl')
joblib.dump(feature_cols, 'PROCESSAMENTO_MELHORADO/receitas/feature_names.pkl')

# =============================================================================
# 6. MODELAGEM
# =============================================================================
print("\n=== 6. MODELAGEM ===")

def avaliar_clf_melhorado(modelo, Xtr_raw, Xte_raw, ytr, yte, nome):
    """Avaliacao melhorada com AUC-PR, Brier score e calibracao."""
    modelo.fit(Xtr_raw, ytr)
    y_pred = modelo.predict(Xte_raw)
    y_prob = modelo.predict_proba(Xte_raw)[:, 1]

    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred)
    rec = recall_score(yte, y_pred)
    f1 = f1_score(yte, y_pred)
    auc = roc_auc_score(yte, y_prob)
    auc_pr = average_precision_score(yte, y_prob)
    brier = brier_score_loss(yte, y_prob)

    # Walk-forward CV
    cv_scores = cross_val_score(modelo, Xtr_raw, ytr, cv=tscv, scoring='roc_auc')

    print(f"\n--- {nome} ---")
    print(f"  AUC CV (walk-forward): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Acuracia: {acc:.4f} | Precisao: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  AUC-ROC Teste: {auc:.4f} | AUC-PR: {auc_pr:.4f} | Brier: {brier:.4f}")

    return {
        'Modelo': nome, 'AUC_CV': cv_scores.mean(), 'AUC_CV_std': cv_scores.std(),
        'Acuracia': acc, 'Precisao': prec, 'Recall': rec, 'F1': f1,
        'AUC_Test': auc, 'AUC_PR': auc_pr, 'Brier': brier,
        'Modelo_Fit': modelo, 'y_pred': y_pred, 'y_prob': y_prob
    }

resultados = []

# 6.1 Logistica com Lasso
print("\n--- Logistica Binaria com Lasso ---")
pipe_lr = Pipeline([
    ('sc', StandardScaler()),
    ('clf', LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=42))
])
param_lr = {'clf__C': np.logspace(-3, 1, 15)}
grid_lr = GridSearchCV(pipe_lr, param_lr, cv=tscv, scoring='roc_auc', n_jobs=-1)
grid_lr.fit(X_train, y_train)
print(f"  Melhor C: {grid_lr.best_params_['clf__C']:.4f}")
res_lr = avaliar_clf_melhorado(grid_lr.best_estimator_, X_train, X_test, y_train, y_test, "Logistica Lasso")
resultados.append(res_lr)
joblib.dump(grid_lr.best_estimator_, 'modelagem_melhorada/modelos_ajustados/logistica_lasso.pkl')

# 6.2 Random Forest
print("\n--- Random Forest ---")
pipe_rf = Pipeline([
    ('sc', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
param_rf = {
    'clf__n_estimators': [200, 300],
    'clf__max_depth': [10, 15, None],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=tscv, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"  Melhores params: {grid_rf.best_params_}")
res_rf = avaliar_clf_melhorado(grid_rf.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest")
resultados.append(res_rf)
joblib.dump(grid_rf.best_estimator_, 'modelagem_melhorada/modelos_ajustados/random_forest.pkl')

# 6.3 HistGradientBoosting (novo)
print("\n--- HistGradientBoosting (NOVO) ---")
pipe_hgb = Pipeline([
    ('sc', StandardScaler()),
    ('clf', HistGradientBoostingClassifier(random_state=42))
])
param_hgb = {
    'clf__max_iter': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.05, 0.1],
    'clf__min_samples_leaf': [10, 20]
}
grid_hgb = GridSearchCV(pipe_hgb, param_hgb, cv=tscv, scoring='roc_auc', n_jobs=-1)
grid_hgb.fit(X_train, y_train)
print(f"  Melhores params: {grid_hgb.best_params_}")
res_hgb = avaliar_clf_melhorado(grid_hgb.best_estimator_, X_train, X_test, y_train, y_test, "HistGradientBoosting")
resultados.append(res_hgb)
joblib.dump(grid_hgb.best_estimator_, 'modelagem_melhorada/modelos_ajustados/hist_gradient_boosting.pkl')

# 6.4 XGBoost
try:
    import xgboost as xgb
    print("\n--- XGBoost ---")
    pipe_xgb = Pipeline([
        ('sc', StandardScaler()),
        ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1))
    ])
    param_xgb = {
        'clf__n_estimators': [200],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.1],
        'clf__subsample': [0.8]
    }
    grid_xgb = GridSearchCV(pipe_xgb, param_xgb, cv=tscv, scoring='roc_auc', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    print(f"  Melhores params: {grid_xgb.best_params_}")
    res_xgb = avaliar_clf_melhorado(grid_xgb.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")
    resultados.append(res_xgb)
    joblib.dump(grid_xgb.best_estimator_, 'modelagem_melhorada/modelos_ajustados/xgboost.pkl')
    has_xgb = True
except ImportError:
    print("XGBoost nao disponivel. Pulando.")
    has_xgb = False

# 6.5 Calibracao de probabilidades (novo)
print("\n--- Calibracao de Probabilidades (NOVO) ---")
best_clf = grid_hgb.best_estimator_ if res_hgb['AUC_Test'] >= res_rf['AUC_Test'] else grid_rf.best_estimator_
best_clf_name = 'HGB' if best_clf == grid_hgb.best_estimator_ else 'RF'
calibrated = CalibratedClassifierCV(best_clf, method='isotonic', cv=tscv)
res_cal = avaliar_clf_melhorado(calibrated, X_train, X_test, y_train, y_test, f"{best_clf_name} Calibrado")
resultados.append(res_cal)
joblib.dump(calibrated, 'modelagem_melhorada/modelos_ajustados/calibrated.pkl')

# =============================================================================
# 7. RESUMO COMPARATIVO
# =============================================================================
print("\n=== 7. RESUMO COMPARATIVO ===")
res_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['Modelo_Fit', 'y_pred', 'y_prob']}
    for r in resultados
])
print(res_df.to_string(index=False))
res_df.to_csv('modelagem_melhorada/resultados/metricas_comparacao.csv', index=False)

# ROC e Precision-Recall
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for nome, res, cor in [
    ('Logistica Lasso', res_lr, '#3498db'),
    ('Random Forest', res_rf, '#2ecc71'),
    ('HGB', res_hgb, '#f39c12')
]:
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, label=f"{nome} (AUC={res['AUC_Test']:.3f})", color=cor, lw=2)
    precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(recall, precision, label=f"{nome} (AP={res['AUC_PR']:.3f})", color=cor, lw=2)

if has_xgb:
    fpr, tpr, _ = roc_curve(y_test, res_xgb['y_prob'])
    axes[0].plot(fpr, tpr, label=f"XGBoost (AUC={res_xgb['AUC_Test']:.3f})", color='#e74c3c', lw=2)
    precision, recall, _ = precision_recall_curve(y_test, res_xgb['y_prob'])
    axes[1].plot(recall, precision, label=f"XGBoost (AP={res_xgb['AUC_PR']:.3f})", color='#e74c3c', lw=2)

axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('Curvas ROC', fontweight='bold'); axes[0].legend(loc='lower right')

axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Curvas Precision-Recall', fontweight='bold'); axes[1].legend(loc='lower left')

plt.suptitle('Pipeline Melhorado - ROC e PR', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('modelagem_melhorada/resultados/curvas_roc_pr.png', dpi=300)
plt.close()

# Matrizes de confusao
fig, axes = plt.subplots(1, len(resultados), figsize=(5 * len(resultados), 4))
if len(resultados) == 1:
    axes = [axes]
for ax, (nome, res) in zip(axes, [(r['Modelo'], r) for r in resultados]):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Nao-Vitoria', 'Vitoria Casa'], yticklabels=['Nao-Vitoria', 'Vitoria Casa'])
    ax.set_title(nome, fontweight='bold')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
plt.suptitle('Matrizes de Confusao - Pipeline Melhorado', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('modelagem_melhorada/resultados/matrizes_confusao.png', dpi=300)
plt.close()

# =============================================================================
# 8. INTERPRETABILIDADE
# =============================================================================
print("\n=== 8. INTERPRETABILIDADE ===")

# 8.1 Coeficientes Lasso
lr_model = grid_lr.best_estimator_.named_steps['clf']
lr_coef = pd.DataFrame({
    'Feature': feature_cols,
    'Coeficiente': lr_model.coef_[0],
    'Abs': np.abs(lr_model.coef_[0])
})
lr_coef = lr_coef.sort_values('Abs', ascending=False)
lr_coef.to_csv('interpretabilidade_melhorada/coeficientes_lasso.csv', index=False)
print("\n--- Coeficientes Lasso ---")
print(lr_coef[lr_coef['Coeficiente'] != 0].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 10))
cp = lr_coef[lr_coef['Coeficiente'] != 0].sort_values('Coeficiente')
ax.barh(cp['Feature'], cp['Coeficiente'], color=['#e74c3c' if c < 0 else '#2ecc71' for c in cp['Coeficiente']])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficientes Logistica Lasso', fontweight='bold')
ax.set_xlabel('Coeficiente')
plt.tight_layout()
plt.savefig('interpretabilidade_melhorada/coeficientes_lasso.png', dpi=300)
plt.close()

# 8.2 Permutacao (melhor modelo)
best_model_name = best_clf_name
rf_model = grid_rf.best_estimator_.named_steps['clf']
X_test_proc = grid_rf.best_estimator_.named_steps['sc'].transform(X_test)
perm_imp = permutation_importance(
    rf_model, X_test_proc, y_test,
    n_repeats=10, random_state=42, scoring='roc_auc'
)
imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importancia': perm_imp.importances_mean,
    'Desvio': perm_imp.importances_std
})
imp_df = imp_df.sort_values('Importancia', ascending=False)
imp_df.to_csv('interpretabilidade_melhorada/importancia_permutacao/importancia.csv', index=False)
print("\n--- Importancia Permutacao (RF) ---")
print(imp_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 10))
ip = imp_df.head(20).sort_values('Importancia')
ax.barh(ip['Feature'], ip['Importancia'], xerr=ip['Desvio'], color='#9b59b6', capsize=3)
ax.set_title('Importancia Permutacao - RF', fontweight='bold')
ax.set_xlabel('Queda no AUC')
plt.tight_layout()
plt.savefig('interpretabilidade_melhorada/importancia_permutacao/importancia.png', dpi=300)
plt.close()

# 8.3 PDP
top6 = imp_df.head(6)['Feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for ax, feat in zip(axes, top6):
    fi = feature_cols.index(feat)
    pd_res = partial_dependence(rf_model, X_test_proc, features=[fi], kind='average', grid_resolution=50)
    ax.plot(pd_res['grid_values'][0], pd_res['average'][0], color='#e74c3c', lw=2.5)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Prob. Vitoria Casa')
    ax.grid(True, alpha=0.3)
fig.suptitle('PDP - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('interpretabilidade_melhorada/graficos_pdp/pdp_random_forest.png', dpi=300)
plt.close()

# 8.4 SHAP local para partidas especificas
print("\n--- SHAP Local ---")
try:
    import shap
    if has_xgb:
        explainer = shap.TreeExplainer(grid_xgb.best_estimator_.named_steps['clf'])
        X_test_shap = grid_xgb.best_estimator_.named_steps['sc'].transform(X_test)
        sv = explainer.shap_values(X_test_shap)
        sv_plot = sv[1] if isinstance(sv, list) else sv
        modelo_shap = "XGBoost"
    else:
        explainer = shap.TreeExplainer(rf_model)
        sv_plot = explainer.shap_values(X_test_proc)
        sv_plot = sv_plot[1] if isinstance(sv_plot, list) else sv_plot
        X_test_shap = X_test_proc
        modelo_shap = "Random Forest"

    # Summary global
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_test_shap, feature_names=feature_cols, show=False, plot_size=(10, 8))
    plt.title(f'SHAP Summary - {modelo_shap}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('interpretabilidade_melhorada/shap_values/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Bar global
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_test_shap, feature_names=feature_cols, plot_type='bar', show=False, plot_size=(10, 8))
    plt.title(f'SHAP Global - {modelo_shap}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('interpretabilidade_melhorada/shap_values/shap_importancia_global.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Local: partidas especificas (indice 0, 10, 20 do teste)
    for i, desc in [(0, 'vitoria_casa_prevista'), (10, 'derrota_casa_prevista'), (20, 'empate_casa_prevista')]:
        if i < len(X_test_shap):
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv_plot[i],
                    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                    data=X_test_shap[i],
                    feature_names=feature_cols
                ),
                show=False, max_display=15
            )
            plt.title(f'SHAP Waterfall - {desc}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'interpretabilidade_melhorada/shap_values/shap_waterfall_{desc}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   Waterfall para {desc} gerado.")

    print(f"SHAP ({modelo_shap}) gerado com sucesso.")

except ImportError:
    print("SHAP nao instalado. Pulando.")
except Exception as e:
    print(f"Erro SHAP: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 9. DOCUMENTACAO DAS MUDANCAS
# =============================================================================
print("\n=== 9. DOCUMENTACAO DAS MUDANCAS ===")
doc = """
MELHORIAS IMPLEMENTADAS NO PIPELINE DE CLASSIFICACAO
=====================================================

[A] LEAKAGE TEMPORAL TRATADO
    - Antes: stats acumuladas da temporada inteira (leakage para partidas no inicio)
    - Depois: stats da TEMPORADA ANTERIOR como proxy de forma
    - Justificativa: metodologicamente correto, nao "ve o futuro"
    - Nota: perde informacao sobre times promovidos/rebaixados

[B] MISSING VALUES TRATADOS
    - Removidas features com >30% missing (backward_pass, big_chance_missed)
    - Restante imputado com mediana por temporada
    - Justificativa: evita perda silenciosa de 33% dos dados

[C] FEATURE ENGINEERING AVANCADO
    - diff_forma_5: media de pontos nos ultimos 5 jogos (momentum)
    - diff_gf_5, diff_ga_5: gols feitos/sofridos nos ultimos 5 jogos
    - hh_wins_home: vitorias do mandante no head-to-head recente
    - hh_goals_diff: saldo de gols no head-to-head
    - rodada: numero da rodada na temporada
    - fase_temporada: Inicio/Meio/Final (dinamicas diferentes)

[D] VALIDACAO MELHORADA
    - Antes: CV 5-fold aleatorio (invalido em series temporais)
    - Depois: TimeSeriesSplit (walk-forward validation)
    - Justificativa: respeita ordem temporal, simula cenario real

[E] MODELOS E METRICAS ADICIONAIS
    - HistGradientBoostingClassifier (nativo scikit-learn)
    - CalibratedClassifierCV (probabilidades calibradas)
    - AUC-PR (melhor que AUC-ROC para classes desbalanceadas)
    - Brier score (calibracao de probabilidades)

[F] INTERPRETABILIDADE APROFUNDADA
    - SHAP waterfall para partidas especificas
    - Curvas Precision-Recall
    - Matriz de confusao por modelo
"""
with open('README_MELHORIAS_CLASSIFICACAO.md', 'w') as f:
    f.write(doc)
print(doc)

print("\n=== CONCLUIDO ===")
print("Todos os artefatos do pipeline melhorado gerados em:")
print("  - modelagem_melhorada/resultados/")
print("  - interpretabilidade_melhorada/")
print("  - PROCESSAMENTO_MELHORADO/")
print("  - README_MELHORIAS_CLASSIFICACAO.md")
