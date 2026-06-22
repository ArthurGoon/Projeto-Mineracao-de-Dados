
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
