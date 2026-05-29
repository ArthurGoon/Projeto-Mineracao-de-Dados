
MELHORIAS IMPLEMENTADAS NO PIPELINE DE REGRESSAO (FINAL)
=========================================================

[A] TRATAMENTO DE OUTLIERS SALARIAIS
    - Removidos 38 jogadores com salario < $500,000
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
