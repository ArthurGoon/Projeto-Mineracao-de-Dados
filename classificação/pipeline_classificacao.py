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

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.inspection import permutation_importance, partial_dependence

for d in ['PROCESSAMENTO/receitas','modelagem/resultados','modelagem/modelos_ajustados',
          'interpretabilidade/graficos_pdp','interpretabilidade/shap_values',
          'interpretabilidade/importancia_permutacao']:
    os.makedirs(d, exist_ok=True)

print("=== 1. LEITURA ===")
results = pd.read_csv('dataset/results.csv')
stats = pd.read_csv('dataset/stats.csv')
print(f"Results: {results.shape}")
print(f"Stats: {stats.shape}")
print(f"Columns results: {list(results.columns)}")
print(f"Columns stats: {list(stats.columns)}")

print("\n=== 2. ADED ===")
fig_dir = 'ADED/figuras'

# Target: H = 1, D/A = 0
results['HomeWin'] = (results['result'] == 'H').astype(int)
print(f"HomeWin dist:\n{results['HomeWin'].value_counts()}")

# Figura 1: Distribuicao resultados
fig, ax = plt.subplots(figsize=(6,4))
results['result'].value_counts().plot(kind='bar', color=['#2ecc71','#f39c12','#e74c3c'], ax=ax)
ax.set_xticklabels(['Vitória Casa','Empate','Vitória Fora'], rotation=0)
ax.set_title('Distribuicao dos Resultados', fontweight='bold')
ax.set_ylabel('Frequencia')
plt.tight_layout(); plt.savefig(f'{fig_dir}/01_distribuicao_resultados.png', dpi=300); plt.close()

# Figura 2: Vantagem casa por temporada
home_advantage = results.groupby('season').agg(
    HomeWin_pct=('HomeWin','mean'),
    Draw_pct=('result', lambda x: (x=='D').mean()),
    AwayWin_pct=('result', lambda x: (x=='A').mean())
).reset_index()

fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(home_advantage))
width = 0.25
ax.bar(x - width, home_advantage['HomeWin_pct'], width, label='Vitória Casa', color='#2ecc71')
ax.bar(x, home_advantage['Draw_pct'], width, label='Empate', color='#f39c12')
ax.bar(x + width, home_advantage['AwayWin_pct'], width, label='Vitória Fora', color='#e74c3c')
ax.set_xticks(x)
ax.set_xticklabels(home_advantage['season'], rotation=45, ha='right')
ax.set_ylabel('Proporcao')
ax.set_title('Resultados por Temporada', fontweight='bold')
ax.legend()
ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='50%')
plt.tight_layout(); plt.savefig(f'{fig_dir}/02_resultados_por_temporada.png', dpi=300); plt.close()

print(f"\nVantagem casa media: {results['HomeWin'].mean():.3f}")

# Figura 3: Gols
fig, axes = plt.subplots(1,2, figsize=(10,4))
results['home_goals'].hist(bins=range(0,10), alpha=0.7, color='#3498db', ax=axes[0], edgecolor='black')
axes[0].set_title('Gols Mandante', fontweight='bold'); axes[0].set_xlabel('Gols'); axes[0].set_ylabel('Frequencia')
results['away_goals'].hist(bins=range(0,10), alpha=0.7, color='#e74c3c', ax=axes[1], edgecolor='black')
axes[1].set_title('Gols Visitante', fontweight='bold'); axes[1].set_xlabel('Gols'); axes[1].set_ylabel('Frequencia')
plt.tight_layout(); plt.savefig(f'{fig_dir}/03_gols_distribuicao.png', dpi=300); plt.close()

print("\n=== 3. FEATURE ENGINEERING ===")
# Merge stats com results: para cada partida, trazer stats do time mandante e visitante daquela temporada
# Renomear stats para home_ e away_
stats_home = stats.add_prefix('home_')
stats_away = stats.add_prefix('away_')

# Merge
results = results.merge(stats_home, left_on=['home_team','season'], right_on=['home_team','home_season'], how='left')
results = results.merge(stats_away, left_on=['away_team','season'], right_on=['away_team','away_season'], how='left')

# Criar features diferenciais
diff_cols = ['wins','losses','goals','total_yel_card','total_red_card','total_scoring_att',
             'ontarget_scoring_att','hit_woodwork','att_hd_goal','att_pen_goal','att_freekick_goal',
             'att_ibox_goal','att_obox_goal','goal_fastbreak','total_offside','clean_sheet',
             'goals_conceded','saves','outfielder_block','interception','total_tackle',
             'last_man_tackle','total_clearance','head_clearance','own_goals','penalty_conceded',
             'pen_goals_conceded','total_pass','total_through_ball','total_long_balls',
             'backward_pass','total_cross','corner_taken','touches','big_chance_missed',
             'clearance_off_line','dispossessed','penalty_save','total_high_claim','punches']

for col in diff_cols:
    if f'home_{col}' in results.columns and f'away_{col}' in results.columns:
        results[f'diff_{col}'] = results[f'home_{col}'] - results[f'away_{col}']

# Selecionar features para modelagem
feature_cols = [c for c in results.columns if c.startswith('diff_')]
feature_cols += ['home_goals','away_goals']  # vamos usar como proxy de forma (media movel)
# Na verdade, em previsao real nao podemos usar gols da partida. Vamos remover.
feature_cols = [c for c in results.columns if c.startswith('diff_')]

# Remover possiveis NaN
results = results.dropna(subset=feature_cols + ['HomeWin'])

print(f"Features criadas: {len(feature_cols)}")
print(f"Amostras finais: {len(results)}")

X = results[feature_cols].copy()
y = results['HomeWin'].copy()

print("\n=== 4. DIVISAO TEMPORAL ===")
train_seasons = [f'{y}-{y+1}' for y in range(2006, 2016)]
test_seasons = [f'{y}-{y+1}' for y in range(2016, 2018)]

mask_train = results['season'].isin(train_seasons)
mask_test = results['season'].isin(test_seasons)

X_train = X[mask_train].copy()
X_test = X[mask_test].copy()
y_train = y[mask_train].copy()
y_test = y[mask_test].copy()

print(f"Treino: {len(X_train)} ({train_seasons[0]} a {train_seasons[-1]})")
print(f"Teste: {len(X_test)} ({test_seasons[0]} a {test_seasons[-1]})")
print(f"Taxa vitória casa treino: {y_train.mean():.3f}")
print(f"Taxa vitória casa teste: {y_test.mean():.3f}")

X_train.to_csv('PROCESSAMENTO/X_train.csv', index=False)
X_test.to_csv('PROCESSAMENTO/X_test.csv', index=False)
y_train.to_csv('PROCESSAMENTO/y_train.csv', index=False)
y_test.to_csv('PROCESSAMENTO/y_test.csv', index=False)

print("\n=== 5. PRE-PROCESSAMENTO ===")
preprocess = StandardScaler()
X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)

joblib.dump(preprocess, 'PROCESSAMENTO/receitas/preprocessador.pkl')
joblib.dump(feature_cols, 'PROCESSAMENTO/receitas/feature_names.pkl')

print("\n=== 6. MODELAGEM ===")
def avaliar_clf(modelo, Xtr, Xte, ytr, yte, nome):
    modelo.fit(Xtr, ytr)
    y_pred = modelo.predict(Xte)
    y_prob = modelo.predict_proba(Xte)[:,1]
    
    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred)
    rec = recall_score(yte, y_pred)
    f1 = f1_score(yte, y_pred)
    auc = roc_auc_score(yte, y_prob)
    
    cv_scores = cross_val_score(modelo, Xtr, ytr, cv=5, scoring='roc_auc')
    
    print(f"\n--- {nome} ---")
    print(f"  AUC CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Acuracia: {acc:.4f} | Precisao: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"  AUC-ROC Teste: {auc:.4f}")
    
    return {'Modelo':nome,'AUC_CV':cv_scores.mean(),'Acuracia':acc,'Precisao':prec,
            'Recall':rec,'F1':f1,'AUC_Test':auc,'Modelo_Fit':modelo,'y_pred':y_pred,'y_prob':y_prob}

resultados = []

# 6.1 Logistica com Lasso
print("\n--- Logistica Binaria com Lasso ---")
pipe_lr = Pipeline([('sc',StandardScaler()),('clf',LogisticRegression(penalty='l1',solver='saga',max_iter=10000,random_state=42))])
param_lr = {'clf__C': np.logspace(-3, 1, 15)}
grid_lr = GridSearchCV(pipe_lr, param_lr, cv=5, scoring='roc_auc', n_jobs=-1)
grid_lr.fit(X_train, y_train)
print(f"  Melhor C: {grid_lr.best_params_['clf__C']:.4f}")
res_lr = avaliar_clf(grid_lr.best_estimator_, X_train, X_test, y_train, y_test, "Logistica Lasso")
resultados.append(res_lr)
joblib.dump(grid_lr.best_estimator_, 'modelagem/modelos_ajustados/logistica_lasso.pkl')

# 6.2 Random Forest
print("\n--- Random Forest ---")
pipe_rf = Pipeline([('sc',StandardScaler()),('clf',RandomForestClassifier(random_state=42,n_jobs=-1))])
param_rf = {'clf__n_estimators':[200,300],'clf__max_depth':[10,15,None],'clf__min_samples_split':[2,5],'clf__min_samples_leaf':[1,2]}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"  Melhores params: {grid_rf.best_params_}")
res_rf = avaliar_clf(grid_rf.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest")
resultados.append(res_rf)
joblib.dump(grid_rf.best_estimator_, 'modelagem/modelos_ajustados/random_forest.pkl')

# 6.3 XGBoost
try:
    import xgboost as xgb
    print("\n--- XGBoost ---")
    pipe_xgb = Pipeline([('sc',StandardScaler()),('clf',xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=42,n_jobs=-1))])
    param_xgb = {'clf__n_estimators':[200],'clf__max_depth':[3,5],'clf__learning_rate':[0.1],'clf__subsample':[0.8]}
    grid_xgb = GridSearchCV(pipe_xgb, param_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    print(f"  Melhores params: {grid_xgb.best_params_}")
    res_xgb = avaliar_clf(grid_xgb.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")
    resultados.append(res_xgb)
    joblib.dump(grid_xgb.best_estimator_, 'modelagem/modelos_ajustados/xgboost.pkl')
    has_xgb = True
except ImportError:
    print("XGBoost nao disponivel. Pulando.")
    has_xgb = False

# Resumo
res_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['Modelo_Fit','y_pred','y_prob']} for r in resultados])
print("\n=== RESUMO ===")
print(res_df.to_string(index=False))
res_df.to_csv('modelagem/resultados/metricas_comparacao.csv', index=False)

# ROC Curves
fig, ax = plt.subplots(figsize=(7,5))
for nome, res, cor in [('Logistica Lasso',res_lr,'#3498db'),('Random Forest',res_rf,'#2ecc71')]:
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{nome} (AUC={res['AUC_Test']:.3f})", color=cor, lw=2)
if has_xgb:
    fpr, tpr, _ = roc_curve(y_test, res_xgb['y_prob'])
    ax.plot(fpr, tpr, label=f"XGBoost (AUC={res_xgb['AUC_Test']:.3f})", color='#e74c3c', lw=2)
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('Curvas ROC', fontweight='bold'); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig('modelagem/resultados/curvas_roc.png', dpi=300); plt.close()

# Matrizes de confusao
fig, axes = plt.subplots(1, len(resultados), figsize=(5*len(resultados),4))
if len(resultados)==1: axes=[axes]
for ax, (nome,res) in zip(axes, [(r['Modelo'],r) for r in resultados]):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Nao-Vitoria','Vitoria Casa'], yticklabels=['Nao-Vitoria','Vitoria Casa'])
    ax.set_title(nome, fontweight='bold'); ax.set_xlabel('Predito'); ax.set_ylabel('Real')
plt.suptitle('Matrizes de Confusao', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('modelagem/resultados/matrizes_confusao.png', dpi=300); plt.close()

print("\n=== 7. INTERPRETABILIDADE ===")

# 7.1 Coeficientes Lasso
lr_model = grid_lr.best_estimator_.named_steps['clf']
lr_coef = pd.DataFrame({'Feature':feature_cols,'Coeficiente':lr_model.coef_[0],'Abs':np.abs(lr_model.coef_[0])})
lr_coef = lr_coef.sort_values('Abs', ascending=False)
lr_coef.to_csv('interpretabilidade/coeficientes_lasso.csv', index=False)
print("\n--- Coeficientes Lasso ---")
print(lr_coef[lr_coef['Coeficiente']!=0].to_string(index=False))

fig, ax = plt.subplots(figsize=(8,10))
cp = lr_coef[lr_coef['Coeficiente']!=0].sort_values('Coeficiente')
ax.barh(cp['Feature'], cp['Coeficiente'], color=['#e74c3c' if c<0 else '#2ecc71' for c in cp['Coeficiente']])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficientes Logistica Lasso', fontweight='bold'); ax.set_xlabel('Coeficiente')
plt.tight_layout(); plt.savefig('interpretabilidade/coeficientes_lasso.png', dpi=300); plt.close()

# 7.2 Permutacao RF
rf_model = grid_rf.best_estimator_.named_steps['clf']
X_test_proc = grid_rf.best_estimator_.named_steps['sc'].transform(X_test)
perm_imp = permutation_importance(rf_model, X_test_proc, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
imp_df = pd.DataFrame({'Feature':feature_cols,'Importancia':perm_imp.importances_mean,'Desvio':perm_imp.importances_std})
imp_df = imp_df.sort_values('Importancia', ascending=False)
imp_df.to_csv('interpretabilidade/importancia_permutacao/importancia.csv', index=False)
print("\n--- Importancia Permutacao (RF) ---")
print(imp_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8,10))
ip = imp_df.head(20).sort_values('Importancia')
ax.barh(ip['Feature'], ip['Importancia'], xerr=ip['Desvio'], color='#9b59b6', capsize=3)
ax.set_title('Importancia Permutacao - RF', fontweight='bold'); ax.set_xlabel('Queda no AUC')
plt.tight_layout(); plt.savefig('interpretabilidade/importancia_permutacao/importancia.png', dpi=300); plt.close()

# 7.3 PDP
top6 = imp_df.head(6)['Feature'].tolist()
fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.flatten()
for ax, feat in zip(axes, top6):
    fi = feature_cols.index(feat)
    pd_res = partial_dependence(rf_model, X_test_proc, features=[fi], kind='average', grid_resolution=50)
    ax.plot(pd_res['grid_values'][0], pd_res['average'][0], color='#e74c3c', lw=2.5)
    ax.set_title(feat, fontweight='bold'); ax.set_xlabel(feat); ax.set_ylabel('Prob. Vitoria Casa')
    ax.grid(True, alpha=0.3)
fig.suptitle('PDP - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('interpretabilidade/graficos_pdp/pdp_random_forest.png', dpi=300); plt.close()

# 7.4 SHAP
try:
    import shap
    if has_xgb:
        explainer = shap.TreeExplainer(grid_xgb.best_estimator_.named_steps['clf'])
        X_test_shap = grid_xgb.best_estimator_.named_steps['sc'].transform(X_test)
        sv = explainer.shap_values(X_test_shap); modelo_shap = "XGBoost"
    else:
        explainer = shap.TreeExplainer(rf_model)
        sv = explainer.shap_values(X_test_proc); X_test_shap = X_test_proc; modelo_shap = "Random Forest"
    
    fig, ax = plt.subplots(figsize=(10,8))
    shap.summary_plot(sv[1] if isinstance(sv, list) else sv, X_test_shap, feature_names=feature_cols, show=False, plot_size=(10,8))
    plt.title(f'SHAP Summary - {modelo_shap}', fontweight='bold')
    plt.tight_layout(); plt.savefig('interpretabilidade/shap_values/shap_summary.png', dpi=300, bbox_inches='tight'); plt.close()
    
    fig, ax = plt.subplots(figsize=(10,8))
    shap.summary_plot(sv[1] if isinstance(sv, list) else sv, X_test_shap, feature_names=feature_cols, plot_type='bar', show=False, plot_size=(10,8))
    plt.title(f'SHAP Global - {modelo_shap}', fontweight='bold')
    plt.tight_layout(); plt.savefig('interpretabilidade/shap_values/shap_importancia_global.png', dpi=300, bbox_inches='tight'); plt.close()
    
    print(f"SHAP ({modelo_shap}) gerado.")
except ImportError:
    print("SHAP nao instalado. Pulando.")
except Exception as e:
    print(f"Erro SHAP: {e}")

print("\n=== CONCLUIDO ===")
