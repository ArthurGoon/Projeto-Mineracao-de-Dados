
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
