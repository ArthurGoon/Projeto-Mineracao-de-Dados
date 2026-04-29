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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence

for d in ['PROCESSAMENTO/receitas','modelagem/resultados','modelagem/modelos_ajustados',
          'interpretabilidade/graficos_pdp','interpretabilidade/shap_values',
          'interpretabilidade/importancia_permutacao']:
    os.makedirs(d, exist_ok=True)

print("=== 1. LEITURA E LIMPEZA ===")
df = pd.read_csv('dataset/nba_2022-23_all_stats_with_salary.csv')
df = df.drop('Unnamed: 0', axis=1)
df['Position_Clean'] = df['Position'].apply(lambda x: x.split('-')[0])
for col in ['FG%','3P%','2P%','eFG%','FT%','TS%','3PAr','FTr']:
    df[col] = df[col].fillna(df[col].median())
df['Log_Salary'] = np.log(df['Salary'])
print(f"Dados: {df.shape}")

print("\n=== 2. FEATURES ===")
for col in ['PTS','TRB','AST','STL','BLK','ORB','DRB','TOV','PF']:
    df[f'{col}_per_GP'] = df[col] / df['GP'].replace(0, np.nan)
    df[f'{col}_per_GP'] = df[f'{col}_per_GP'].fillna(0)

features = ['Age','GP','MP','PTS_per_GP','TRB_per_GP','AST_per_GP','STL_per_GP','BLK_per_GP',
            'FG%','3P%','FT%','PER','TS%','USG%','WS','VORP','BPM','Position_Clean']
X = df[features].copy()
y = df['Log_Salary'].copy()
print(f"Features: {len(features)}")

print("\n=== 3. DIVISAO ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
X_train.to_csv('PROCESSAMENTO/X_train.csv', index=False)
X_test.to_csv('PROCESSAMENTO/X_test.csv', index=False)
pd.DataFrame({'Salary':y_train}).to_csv('PROCESSAMENTO/y_train.csv', index=False)
pd.DataFrame({'Salary':y_test}).to_csv('PROCESSAMENTO/y_test.csv', index=False)

print("\n=== 4. PRE-PROCESSAMENTO ===")
cat_features = ['Position_Clean']
num_features = [c for c in features if c not in cat_features]
preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])
X_train_proc = preprocess.fit_transform(X_train)
X_test_proc = preprocess.transform(X_test)
cat_names = list(preprocess.named_transformers_['cat'].get_feature_names_out(cat_features))
feature_names = num_features + cat_names
print(f"Features pos-preproc: {len(feature_names)}")
joblib.dump(preprocess, 'PROCESSAMENTO/receitas/preprocessador.pkl')
joblib.dump(feature_names, 'PROCESSAMENTO/receitas/feature_names.pkl')

print("\n=== 5. MODELAGEM ===")
def avaliar(modelo, Xtr, Xte, ytr, yte, nome):
    modelo.fit(Xtr, ytr)
    y_pred = modelo.predict(Xte)
    rmse_log = np.sqrt(mean_squared_error(yte, y_pred))
    mae_log = mean_absolute_error(yte, y_pred)
    r2 = r2_score(yte, y_pred)
    y_pred_sal = np.exp(y_pred)
    yte_sal = np.exp(yte)
    rmse_sal = np.sqrt(mean_squared_error(yte_sal, y_pred_sal))
    mae_sal = mean_absolute_error(yte_sal, y_pred_sal)
    mape = np.mean(np.abs((yte_sal - y_pred_sal)/yte_sal))*100
    cv_scores = cross_val_score(modelo, Xtr, ytr, cv=5, scoring='r2')
    print(f"\n--- {nome} ---")
    print(f"  R2 CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  R2 Teste: {r2:.4f} | RMSE(log): {rmse_log:.4f} | MAE(log): {mae_log:.4f}")
    print(f"  RMSE(USD): ${rmse_sal:,.0f} | MAE(USD): ${mae_sal:,.0f} | MAPE: {mape:.2f}%")
    return {'Modelo':nome,'R2_CV':cv_scores.mean(),'R2_Test':r2,'RMSE_Log':rmse_log,
            'MAE_Log':mae_log,'RMSE_USD':rmse_sal,'MAE_USD':mae_sal,'MAPE':mape,
            'Modelo_Fit':modelo,'y_pred':y_pred}

resultados = []

# OLS
pipe_ols = Pipeline([('preprocess',preprocess),('reg',LinearRegression())])
res_ols = avaliar(pipe_ols, X_train, X_test, y_train, y_test, "OLS")
resultados.append(res_ols); joblib.dump(pipe_ols, 'modelagem/modelos_ajustados/ols.pkl')

# Ridge
pipe_ridge = Pipeline([('preprocess',preprocess),('reg',Ridge())])
grid_ridge = GridSearchCV(pipe_ridge, {'reg__alpha':np.logspace(-2,3,20)}, cv=5, scoring='r2', n_jobs=-1)
grid_ridge.fit(X_train, y_train)
print(f"  Melhor alpha: {grid_ridge.best_params_['reg__alpha']:.4f}")
res_ridge = avaliar(grid_ridge.best_estimator_, X_train, X_test, y_train, y_test, "Ridge")
resultados.append(res_ridge); joblib.dump(grid_ridge.best_estimator_, 'modelagem/modelos_ajustados/ridge.pkl')

# Lasso
pipe_lasso = Pipeline([('preprocess',preprocess),('reg',Lasso(max_iter=10000))])
grid_lasso = GridSearchCV(pipe_lasso, {'reg__alpha':np.logspace(-4,1,20)}, cv=5, scoring='r2', n_jobs=-1)
grid_lasso.fit(X_train, y_train)
print(f"  Melhor alpha: {grid_lasso.best_params_['reg__alpha']:.4f}")
res_lasso = avaliar(grid_lasso.best_estimator_, X_train, X_test, y_train, y_test, "Lasso")
resultados.append(res_lasso); joblib.dump(grid_lasso.best_estimator_, 'modelagem/modelos_ajustados/lasso.pkl')

# Random Forest
pipe_rf = Pipeline([('preprocess',preprocess),('reg',RandomForestRegressor(random_state=42,n_jobs=-1))])
param_rf = {'reg__n_estimators':[200,300],'reg__max_depth':[10,15,None],'reg__min_samples_split':[2,5],'reg__min_samples_leaf':[1,2]}
grid_rf = GridSearchCV(pipe_rf, param_rf, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print(f"  Melhores params: {grid_rf.best_params_}")
res_rf = avaliar(grid_rf.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest")
resultados.append(res_rf); joblib.dump(grid_rf.best_estimator_, 'modelagem/modelos_ajustados/random_forest.pkl')

# XGBoost
try:
    import xgboost as xgb
    pipe_xgb = Pipeline([('preprocess',preprocess),('reg',xgb.XGBRegressor(random_state=42,n_jobs=-1))])
    param_xgb = {'reg__n_estimators':[200,300],'reg__max_depth':[3,5,7],'reg__learning_rate':[0.05,0.1],'reg__subsample':[0.8,1.0]}
    grid_xgb = GridSearchCV(pipe_xgb, param_xgb, cv=5, scoring='r2', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    print(f"  Melhores params: {grid_xgb.best_params_}")
    res_xgb = avaliar(grid_xgb.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")
    resultados.append(res_xgb); joblib.dump(grid_xgb.best_estimator_, 'modelagem/modelos_ajustados/xgboost.pkl')
    has_xgb = True
except ImportError:
    print("XGBoost nao disponivel. Pulando.")
    has_xgb = False

# Resumo
res_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['Modelo_Fit','y_pred']} for r in resultados])
print("\n=== RESUMO ===")
print(res_df.to_string(index=False))
res_df.to_csv('modelagem/resultados/metricas_comparacao.csv', index=False)

# Graficos comparativos
fig, axes = plt.subplots(1,2, figsize=(12,5))
colors = ['#3498db','#2ecc71','#e74c3c','#9b59b6','#f39c12'][:len(res_df)]
axes[0].bar(res_df['Modelo'], res_df['R2_Test'], color=colors)
axes[0].set_ylabel('R2 (Teste)'); axes[0].set_title('R2 por Modelo', fontweight='bold'); axes[0].set_ylim(0,1)
for i,v in enumerate(res_df['R2_Test']): axes[0].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].bar(res_df['Modelo'], res_df['MAE_Log'], color=colors)
axes[1].set_ylabel('MAE (Log)'); axes[1].set_title('MAE por Modelo', fontweight='bold')
for i,v in enumerate(res_df['MAE_Log']): axes[1].text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
plt.suptitle('Comparacao de Modelos', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('modelagem/resultados/comparacao_modelos.png', dpi=300); plt.close()

# Residuos
fig, axes = plt.subplots(2,2, figsize=(12,10))
axes = axes.flatten()
for ax, (nome,res) in zip(axes, [('OLS',res_ols),('Ridge',res_ridge),('Lasso',res_lasso),('Random Forest',res_rf)]):
    yp = res['y_pred']; resid = y_test - yp
    ax.scatter(yp, resid, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Preditos (Log)'); ax.set_ylabel('Residuos'); ax.set_title(nome, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle('Analise de Residuos', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('modelagem/resultados/residuos_modelos.png', dpi=300); plt.close()

print("\n=== 6. INTERPRETABILIDADE ===")

# OLS Coef
ols_coef = pd.DataFrame({'Feature':feature_names,'Coeficiente':pipe_ols.named_steps['reg'].coef_})
ols_coef = ols_coef.sort_values('Coeficiente', key=abs, ascending=False)
print("\n--- Coeficientes OLS ---")
print(ols_coef.to_string(index=False))
ols_coef.to_csv('interpretabilidade/coeficientes_ols.csv', index=False)
fig, ax = plt.subplots(figsize=(8,8))
cp = ols_coef.sort_values('Coeficiente')
ax.barh(cp['Feature'], cp['Coeficiente'], color=['#e74c3c' if c<0 else '#2ecc71' for c in cp['Coeficiente']])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficientes OLS', fontweight='bold'); ax.set_xlabel('Coeficiente padronizado')
plt.tight_layout(); plt.savefig('interpretabilidade/coeficientes_ols.png', dpi=300); plt.close()

# Ridge vs Lasso
ridge_coef = pd.DataFrame({'Feature':feature_names,
    'Ridge':grid_ridge.best_estimator_.named_steps['reg'].coef_,
    'Lasso':grid_lasso.best_estimator_.named_steps['reg'].coef_})
ridge_coef.to_csv('interpretabilidade/coeficientes_ridge_lasso.csv', index=False)
fig, ax = plt.subplots(figsize=(10,8))
xp = np.arange(len(feature_names)); w = 0.35
ax.barh(xp-w/2, ridge_coef['Ridge'], w, label='Ridge', color='#3498db')
ax.barh(xp+w/2, ridge_coef['Lasso'], w, label='Lasso', color='#e74c3c')
ax.set_yticks(xp); ax.set_yticklabels(feature_names, fontsize=8); ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coeficientes Ridge vs Lasso', fontweight='bold'); ax.set_xlabel('Coeficiente'); ax.legend()
plt.tight_layout(); plt.savefig('interpretabilidade/coeficientes_ridge_lasso.png', dpi=300); plt.close()

# Permutation Importance
rf_model = grid_rf.best_estimator_.named_steps['reg']
X_test_proc_rf = grid_rf.best_estimator_.named_steps['preprocess'].transform(X_test)
perm_imp = permutation_importance(rf_model, X_test_proc_rf, y_test, n_repeats=10, random_state=42, scoring='r2')
imp_df = pd.DataFrame({'Feature':feature_names,'Importancia':perm_imp.importances_mean,'Desvio':perm_imp.importances_std})
imp_df = imp_df.sort_values('Importancia', ascending=False)
print("\n--- Importancia Permutacao (RF) ---")
print(imp_df.head(10).to_string(index=False))
imp_df.to_csv('interpretabilidade/importancia_permutacao/importancia.csv', index=False)
fig, ax = plt.subplots(figsize=(8,8))
ip = imp_df.sort_values('Importancia')
ax.barh(ip['Feature'], ip['Importancia'], xerr=ip['Desvio'], color='#9b59b6', capsize=3)
ax.set_title('Importancia por Permutacao - RF', fontweight='bold'); ax.set_xlabel('Queda no R2')
plt.tight_layout(); plt.savefig('interpretabilidade/importancia_permutacao/importancia.png', dpi=300); plt.close()

# PDP
top6 = imp_df.head(6)['Feature'].tolist()
fig, axes = plt.subplots(2,3, figsize=(15,10))
axes = axes.flatten()
for ax, feat in zip(axes, top6):
    fi = feature_names.index(feat)
    pd_res = partial_dependence(rf_model, X_test_proc_rf, features=[fi], kind='average', grid_resolution=50)
    ax.plot(pd_res['grid_values'][0], pd_res['average'][0], color='#e74c3c', lw=2.5)
    ax.set_title(feat, fontweight='bold'); ax.set_xlabel(feat); ax.set_ylabel('Efeito marginal (Log)')
    ax.grid(True, alpha=0.3)
fig.suptitle('PDP - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig('interpretabilidade/graficos_pdp/pdp_random_forest.png', dpi=300); plt.close()

# SHAP
try:
    import shap
    if has_xgb:
        explainer = shap.TreeExplainer(grid_xgb.best_estimator_.named_steps['reg'])
        X_test_shap = grid_xgb.best_estimator_.named_steps['preprocess'].transform(X_test)
        sv = explainer.shap_values(X_test_shap); modelo_shap = "XGBoost"
    else:
        explainer = shap.TreeExplainer(rf_model)
        sv = explainer.shap_values(X_test_proc_rf); X_test_shap = X_test_proc_rf; modelo_shap = "Random Forest"
    
    fig, ax = plt.subplots(figsize=(10,8))
    shap.summary_plot(sv, X_test_shap, feature_names=feature_names, show=False, plot_size=(10,8))
    plt.title(f'SHAP Summary - {modelo_shap}', fontweight='bold')
    plt.tight_layout(); plt.savefig('interpretabilidade/shap_values/shap_summary.png', dpi=300, bbox_inches='tight'); plt.close()
    
    fig, ax = plt.subplots(figsize=(10,8))
    shap.summary_plot(sv, X_test_shap, feature_names=feature_names, plot_type='bar', show=False, plot_size=(10,8))
    plt.title(f'SHAP Global - {modelo_shap}', fontweight='bold')
    plt.tight_layout(); plt.savefig('interpretabilidade/shap_values/shap_importancia_global.png', dpi=300, bbox_inches='tight'); plt.close()
    
    # Waterfall para Stephen Curry
    names_test = df.loc[X_test.index, 'Player Name'].reset_index(drop=True)
    curry_idx = names_test[names_test == 'Stephen Curry'].index
    if len(curry_idx) > 0:
        idx = curry_idx[0]
        fig, ax = plt.subplots(figsize=(10,6))
        shap.waterfall_plot(shap.Explanation(values=sv[idx], base_values=explainer.expected_value,
            data=X_test_shap[idx], feature_names=feature_names), show=False, max_display=15)
        plt.title('SHAP Waterfall - Stephen Curry', fontweight='bold')
        plt.tight_layout(); plt.savefig('interpretabilidade/shap_values/shap_waterfall_curry.png', dpi=300, bbox_inches='tight'); plt.close()
    
    # Dependence top feature
    topf = imp_df.iloc[0]['Feature']; tfi = feature_names.index(topf)
    fig, ax = plt.subplots(figsize=(8,5))
    shap.dependence_plot(tfi, sv, X_test_shap, feature_names=feature_names, show=False, ax=ax)
    plt.title(f'SHAP Dependence - {topf}', fontweight='bold')
    plt.tight_layout(); plt.savefig('interpretabilidade/shap_values/shap_dependence_top.png', dpi=300, bbox_inches='tight'); plt.close()
    
    print(f"SHAP ({modelo_shap}) gerado com sucesso.")
except ImportError:
    print("SHAP nao instalado. Pulando.")
except Exception as e:
    print(f"Erro SHAP: {e}")

print("\n=== CONCLUIDO ===")
print("Todos os artefatos gerados.")
