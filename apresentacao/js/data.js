// Data repository for NBA Salaries Regression Presentation
window.NBA_DATA = [
    {"name": "Stephen Curry", "pos": "PG", "salary": 48070014.0, "age": 34, "gp": 56, "mp": 34.7, "pts": 29.4, "usg": 31.0, "bpm": 7.5, "per": 24.1, "ws": 7.8, "type": "Superstar"},
    {"name": "John Wall", "pos": "PG", "salary": 47345760.0, "age": 32, "gp": 34, "mp": 22.2, "pts": 11.4, "usg": 27.0, "bpm": -1.2, "per": 13.6, "ws": 0.3, "type": "Role Player"},
    {"name": "Russell Westbrook", "pos": "PG", "salary": 47080179.0, "age": 34, "gp": 73, "mp": 29.1, "pts": 15.9, "usg": 27.7, "bpm": 0.2, "per": 16.1, "ws": 1.9, "type": "Toxic"},
    {"name": "LeBron James", "pos": "PF", "salary": 44474988.0, "age": 38, "gp": 55, "mp": 35.5, "pts": 28.9, "usg": 33.3, "bpm": 6.1, "per": 23.9, "ws": 5.6, "type": "Superstar"},
    {"name": "Kevin Durant", "pos": "PF", "salary": 44119845.0, "age": 34, "gp": 47, "mp": 35.6, "pts": 29.1, "usg": 30.7, "bpm": 7.1, "per": 25.9, "ws": 6.8, "type": "Superstar"},
    {"name": "Bradley Beal", "pos": "SG", "salary": 43279250.0, "age": 29, "gp": 50, "mp": 33.5, "pts": 23.2, "usg": 29.2, "bpm": 1.8, "per": 19.7, "ws": 3.4, "type": "Superstar"},
    {"name": "Kawhi Leonard", "pos": "SF", "salary": 42492492.0, "age": 31, "gp": 52, "mp": 33.6, "pts": 23.8, "usg": 27.0, "bpm": 6.1, "per": 23.9, "ws": 7.1, "type": "Superstar"},
    {"name": "Paul George", "pos": "SF", "salary": 42492492.0, "age": 32, "gp": 56, "mp": 34.6, "pts": 23.8, "usg": 29.5, "bpm": 2.8, "per": 19.6, "ws": 4.6, "type": "Superstar"},
    {"name": "Giannis Antetokounmpo", "pos": "PF", "salary": 42492492.0, "age": 28, "gp": 63, "mp": 32.1, "pts": 31.1, "usg": 38.8, "bpm": 8.5, "per": 29.0, "ws": 8.6, "type": "Superstar"},
    {"name": "Damian Lillard", "pos": "PG", "salary": 42492492.0, "age": 32, "gp": 58, "mp": 36.3, "pts": 32.2, "usg": 33.8, "bpm": 7.1, "per": 26.7, "ws": 9.0, "type": "Superstar"},
    {"name": "Klay Thompson", "pos": "SF", "salary": 40600080.0, "age": 32, "gp": 69, "mp": 33.0, "pts": 21.9, "usg": 26.4, "bpm": -0.3, "per": 14.7, "ws": 3.1, "type": "Role Player"},
    {"name": "Kyrie Irving", "pos": "PG-SG", "salary": 38917057.0, "age": 30, "gp": 60, "mp": 37.4, "pts": 27.1, "usg": 28.9, "bpm": 4.1, "per": 22.4, "ws": 7.4, "type": "Superstar"},
    {"name": "Rudy Gobert", "pos": "C", "salary": 38172414.0, "age": 30, "gp": 70, "mp": 30.7, "pts": 13.4, "usg": 16.3, "bpm": 0.7, "per": 18.9, "ws": 7.8, "type": "Role Player"},
    {"name": "Khris Middleton", "pos": "SF", "salary": 37984276.0, "age": 31, "gp": 33, "mp": 24.3, "pts": 15.1, "usg": 27.4, "bpm": 0.8, "per": 17.4, "ws": 1.9, "type": "Role Player"},
    {"name": "Anthony Davis", "pos": "C", "salary": 37980720.0, "age": 29, "gp": 56, "mp": 34.0, "pts": 25.9, "usg": 28.4, "bpm": 6.3, "per": 27.8, "ws": 9.0, "type": "Superstar"},
    {"name": "Jimmy Butler", "pos": "PF", "salary": 37653300.0, "age": 33, "gp": 64, "mp": 33.4, "pts": 22.9, "usg": 25.6, "bpm": 8.7, "per": 27.6, "ws": 12.3, "type": "Superstar"},
    {"name": "Tobias Harris", "pos": "SF", "salary": 37633050.0, "age": 30, "gp": 74, "mp": 32.9, "pts": 14.7, "usg": 18.2, "bpm": 0.7, "per": 14.8, "ws": 5.9, "type": "Role Player"},
    {"name": "Kemba Walker", "pos": "PG", "salary": 37281261.0, "age": 32, "gp": 9, "mp": 16.0, "pts": 8.0, "usg": 22.1, "bpm": -0.3, "per": 15.0, "ws": 0.3, "type": "Toxic"},
    {"name": "Trae Young", "pos": "PG", "salary": 37096500.0, "age": 24, "gp": 73, "mp": 34.8, "pts": 26.2, "usg": 32.6, "bpm": 3.3, "per": 22.0, "ws": 6.7, "type": "Superstar"},
    {"name": "Zach LaVine", "pos": "SG", "salary": 37096500.0, "age": 27, "gp": 77, "mp": 35.9, "pts": 24.8, "usg": 28.3, "bpm": 1.9, "per": 19.0, "ws": 7.1, "type": "Superstar"},
    {"name": "Ben Simmons", "pos": "PG", "salary": 35448672.0, "age": 26, "gp": 42, "mp": 26.3, "pts": 6.9, "usg": 14.3, "bpm": 0.4, "per": 13.4, "ws": 2.2, "type": "Role Player"},
    {"name": "Pascal Siakam", "pos": "PF", "salary": 35448672.0, "age": 28, "gp": 71, "mp": 37.4, "pts": 24.2, "usg": 27.2, "bpm": 3.1, "per": 20.3, "ws": 7.8, "type": "Role Player"},
    {"name": "Myles Turner", "pos": "C", "salary": 35096500.0, "age": 26, "gp": 62, "mp": 29.4, "pts": 18.0, "usg": 22.0, "bpm": 2.1, "per": 20.0, "ws": 5.4, "type": "Role Player"},
    {"name": "Jrue Holiday", "pos": "PG", "salary": 34319520.0, "age": 32, "gp": 67, "mp": 32.6, "pts": 19.3, "usg": 25.0, "bpm": 3.1, "per": 19.2, "ws": 6.7, "type": "Role Player"},
    {"name": "Karl-Anthony Towns", "pos": "PF", "salary": 33833400.0, "age": 27, "gp": 29, "mp": 33.0, "pts": 20.8, "usg": 25.6, "bpm": 3.0, "per": 18.8, "ws": 2.7, "type": "Superstar"},
    {"name": "Devin Booker", "pos": "SG", "salary": 33833400.0, "age": 26, "gp": 53, "mp": 34.6, "pts": 27.8, "usg": 31.8, "bpm": 4.2, "per": 22.0, "ws": 6.0, "type": "Superstar"},
    {"name": "Andrew Wiggins", "pos": "SF", "salary": 33616770.0, "age": 27, "gp": 37, "mp": 32.2, "pts": 17.1, "usg": 21.5, "bpm": -0.5, "per": 14.7, "ws": 2.3, "type": "Role Player"},
    {"name": "Joel Embiid", "pos": "C", "salary": 33616770.0, "age": 28, "gp": 66, "mp": 34.6, "pts": 33.1, "usg": 37.0, "bpm": 9.2, "per": 31.4, "ws": 12.3, "type": "Superstar"},
    {"name": "CJ McCollum", "pos": "SG", "salary": 33333333.0, "age": 31, "gp": 75, "mp": 35.3, "pts": 20.9, "usg": 26.4, "bpm": 0.8, "per": 15.6, "ws": 4.3, "type": "Role Player"},
    {"name": "James Harden", "pos": "PG", "salary": 33000000.0, "age": 33, "gp": 58, "mp": 36.8, "pts": 21.0, "usg": 25.0, "bpm": 5.4, "per": 21.6, "ws": 8.4, "type": "Superstar"},
    {"name": "Jamal Murray", "pos": "PG", "salary": 31650600.0, "age": 25, "gp": 65, "mp": 32.8, "pts": 20.0, "usg": 26.1, "bpm": 1.3, "per": 18.0, "ws": 5.1, "type": "Role Player"},
    {"name": "Brandon Ingram", "pos": "SF", "salary": 31650600.0, "age": 25, "gp": 45, "mp": 34.2, "pts": 24.7, "usg": 30.8, "bpm": 1.7, "per": 19.2, "ws": 3.5, "type": "Role Player"},
    {"name": "D'Angelo Russell", "pos": "PG", "salary": 31377750.0, "age": 26, "gp": 71, "mp": 32.5, "pts": 17.8, "usg": 22.7, "bpm": 1.5, "per": 16.3, "ws": 5.1, "type": "Role Player"},
    {"name": "Donovan Mitchell", "pos": "SG", "salary": 30913750.0, "age": 26, "gp": 68, "mp": 35.8, "pts": 28.3, "usg": 32.1, "bpm": 6.3, "per": 22.9, "ws": 8.9, "type": "Superstar"},
    {"name": "Shai Gilgeous-Alexander", "pos": "PG", "salary": 30913750.0, "age": 24, "gp": 68, "mp": 35.5, "pts": 31.4, "usg": 32.8, "bpm": 7.3, "per": 27.2, "ws": 11.4, "type": "Superstar"},
    {"name": "Deandre Ayton", "pos": "C", "salary": 30913750.0, "age": 24, "gp": 67, "mp": 30.4, "pts": 18.0, "usg": 22.9, "bpm": 0.9, "per": 19.9, "ws": 6.2, "type": "Role Player"},
    {"name": "Kevin Love", "pos": "PF", "salary": 30556968.0, "age": 34, "gp": 62, "mp": 20.0, "pts": 8.2, "usg": 19.2, "bpm": 0.6, "per": 13.1, "ws": 2.6, "type": "Role Player"},
    {"name": "Jayson Tatum", "pos": "SF", "salary": 30351780.0, "age": 24, "gp": 74, "mp": 36.9, "pts": 30.1, "usg": 32.7, "bpm": 5.5, "per": 23.7, "ws": 10.5, "type": "Superstar"},
    {"name": "Bam Adebayo", "pos": "C", "salary": 30351780.0, "age": 25, "gp": 75, "mp": 34.6, "pts": 20.4, "usg": 25.2, "bpm": 1.5, "per": 20.1, "ws": 7.4, "type": "Superstar"},
    {"name": "De'Aaron Fox", "pos": "PG", "salary": 30351780.0, "age": 25, "gp": 73, "mp": 33.4, "pts": 25.0, "usg": 30.1, "bpm": 2.5, "per": 21.8, "ws": 7.4, "type": "Superstar"},
    {"name": "Gordon Hayward", "pos": "SF", "salary": 30075000.0, "age": 32, "gp": 50, "mp": 31.5, "pts": 14.7, "usg": 20.0, "bpm": -1.4, "per": 13.5, "ws": 2.0, "type": "Role Player"},
    {"name": "Jaylen Brown", "pos": "SF", "salary": 29776785.0, "age": 26, "gp": 67, "mp": 35.9, "pts": 26.6, "usg": 31.4, "bpm": 1.3, "per": 19.1, "ws": 5.0, "type": "Superstar"},
    {"name": "Chris Paul", "pos": "PG", "salary": 28400000.0, "age": 37, "gp": 59, "mp": 32.0, "pts": 13.9, "usg": 19.2, "bpm": 3.2, "per": 17.7, "ws": 6.2, "type": "Role Player"},
    {"name": "Kyle Lowry", "pos": "PG", "salary": 28333334.0, "age": 36, "gp": 55, "mp": 31.2, "pts": 11.2, "usg": 16.7, "bpm": 0.2, "per": 12.6, "ws": 3.4, "type": "Role Player"},
    {"name": "Jalen Brunson", "pos": "PG", "salary": 27733332.0, "age": 26, "gp": 68, "mp": 35.0, "pts": 24.0, "usg": 27.2, "bpm": 3.9, "per": 21.2, "ws": 8.7, "type": "Superstar"},
    {"name": "DeMar DeRozan", "pos": "SF", "salary": 27300000.0, "age": 33, "gp": 74, "mp": 36.2, "pts": 24.5, "usg": 27.8, "bpm": 2.0, "per": 20.6, "ws": 8.5, "type": "Superstar"},
    {"name": "Al Horford", "pos": "C", "salary": 26500000.0, "age": 36, "gp": 63, "mp": 30.5, "pts": 9.8, "usg": 11.9, "bpm": 3.3, "per": 13.8, "ws": 6.3, "type": "Role Player"},
    {"name": "Draymond Green", "pos": "PF", "salary": 25806468.0, "age": 32, "gp": 73, "mp": 31.5, "pts": 8.5, "usg": 13.2, "bpm": 0.8, "per": 12.2, "ws": 4.7, "type": "Role Player"},
    {"name": "Julius Randle", "pos": "PF", "salary": 23760000.0, "age": 28, "gp": 77, "mp": 35.5, "pts": 25.1, "usg": 29.5, "bpm": 3.7, "per": 20.3, "ws": 8.1, "type": "Role Player"},
    {"name": "John Collins", "pos": "PF", "salary": 23500000.0, "age": 25, "gp": 71, "mp": 30.0, "pts": 13.1, "usg": 17.1, "bpm": -1.3, "per": 13.6, "ws": 4.2, "type": "Role Player"},
    {"name": "Mike Conley", "pos": "PG", "salary": 22680000.0, "age": 35, "gp": 67, "mp": 30.3, "pts": 11.9, "usg": 16.3, "bpm": 0.9, "per": 14.7, "ws": 5.5, "type": "Role Player"},
    {"name": "Malcolm Brogdon", "pos": "PG", "salary": 22600000.0, "age": 30, "gp": 67, "mp": 26.0, "pts": 14.9, "usg": 22.8, "bpm": 2.8, "per": 18.2, "ws": 5.8, "type": "Role Player"},
    {"name": "Anfernee Simons", "pos": "SG", "salary": 22321429.0, "age": 23, "gp": 62, "mp": 35.0, "pts": 21.1, "usg": 25.1, "bpm": -1.1, "per": 14.8, "ws": 2.5, "type": "Role Player"},
    {"name": "Terry Rozier", "pos": "SG", "salary": 21486316.0, "age": 28, "gp": 63, "mp": 35.3, "pts": 21.1, "usg": 26.9, "bpm": -0.6, "per": 14.7, "ws": 1.7, "type": "Role Player"},
    {"name": "Fred VanVleet", "pos": "PG", "salary": 21250000.0, "age": 28, "gp": 69, "mp": 36.7, "pts": 19.3, "usg": 23.2, "bpm": 2.5, "per": 17.0, "ws": 6.5, "type": "Role Player"},
    {"name": "Buddy Hield", "pos": "SF", "salary": 21177750.0, "age": 30, "gp": 80, "mp": 31.0, "pts": 16.8, "usg": 20.8, "bpm": 1.9, "per": 15.4, "ws": 4.4, "type": "Role Player"},
    {"name": "Domantas Sabonis", "pos": "C", "salary": 21100000.0, "age": 26, "gp": 79, "mp": 34.6, "pts": 19.1, "usg": 21.3, "bpm": 5.8, "per": 23.5, "ws": 12.6, "type": "Superstar"},
    {"name": "Jerami Grant", "pos": "PF", "salary": 20955000.0, "age": 28, "gp": 63, "mp": 35.7, "pts": 20.5, "usg": 22.8, "bpm": -0.2, "per": 16.0, "ws": 4.1, "type": "Role Player"},
    {"name": "Aaron Gordon", "pos": "PF", "salary": 20690909.0, "age": 27, "gp": 68, "mp": 30.2, "pts": 16.3, "usg": 21.1, "bpm": 2.1, "per": 19.5, "ws": 6.8, "type": "Role Player"},
    {"name": "Mikal Bridges", "pos": "SF-SG", "salary": 20100000.0, "age": 26, "gp": 83, "mp": 35.7, "pts": 20.1, "usg": 22.6, "bpm": 1.7, "per": 16.8, "ws": 7.5, "type": "Role Player"},
    {"name": "Jonathan Isaac", "pos": "PF", "salary": 17400000.0, "age": 25, "gp": 11, "mp": 11.3, "pts": 5.0, "usg": 21.1, "bpm": 4.0, "per": 20.3, "ws": 0.3, "type": "Toxic"},
    {"name": "Duncan Robinson", "pos": "SF", "salary": 16902000.0, "age": 28, "gp": 42, "mp": 16.5, "pts": 6.4, "usg": 17.9, "bpm": -5.0, "per": 7.8, "ws": 0.5, "type": "Toxic"},
    {"name": "Frank Kaminsky", "pos": "C", "salary": 2463490.0, "age": 29, "gp": 36, "mp": 6.5, "pts": 2.5, "usg": 14.1, "bpm": 2.8, "per": 15.3, "ws": 0.7, "type": "Role Player"},
    {"name": "Jaden Hardy", "pos": "SG", "salary": 1017781.0, "age": 20, "gp": 48, "mp": 14.8, "pts": 8.8, "usg": 26.4, "bpm": -2.3, "per": 14.6, "ws": 0.7, "type": "Rookie"},
    
    // Outliers that will be filtered out visually in Section 2
    {"name": "Facundo Campazzo", "pos": "PG", "salary": 464299.0, "age": 31, "gp": 8, "mp": 5.2, "pts": 1.3, "usg": 11.2, "bpm": -3.5, "per": 6.4, "ws": 0.0, "type": "Outlier"},
    {"name": "Orlando Robinson", "pos": "C", "salary": 386055.0, "age": 22, "gp": 31, "mp": 13.7, "pts": 3.7, "usg": 14.5, "bpm": -1.2, "per": 12.1, "ws": 0.5, "type": "Outlier"},
    {"name": "Mac McClung", "pos": "PG", "salary": 160856.0, "age": 24, "gp": 2, "mp": 12.5, "pts": 9.0, "usg": 24.2, "bpm": 2.5, "per": 18.0, "ws": 0.1, "type": "Outlier"},
    {"name": "RaiQuan Gray", "pos": "PF", "salary": 5849.0, "age": 23, "gp": 1, "mp": 35.0, "pts": 16.0, "usg": 21.0, "bpm": 1.1, "per": 15.5, "ws": 0.0, "type": "Outlier"}
];

// Auxiliar visual datasets
window.vifBeforeData = [
    { name: "Idade", val: 512.0, status: "Colineariedade Crítica" },
    { name: "TS%", val: 254.7, status: "Colineariedade Crítica" },
    { name: "Idade² (sem centering)", val: 155.9, status: "Colineariedade Crítica" },
    { name: "FG%", val: 134.8, status: "Colineariedade Crítica" },
    { name: "PER", val: 48.6, status: "Alta Colineariedade" },
    { name: "MP", val: 9.3, status: "Moderado" }
];

window.vifAfterData = [
    { name: "PTS (Pontos)", val: 13.8, status: "Alta Colineariedade" },
    { name: "MP (Minutos)", val: 11.1, status: "Alta Colineariedade" },
    { name: "STL_BLK_sum", val: 9.8, status: "Moderado" },
    { name: "Rebotes/GP", val: 9.2, status: "Moderado" },
    { name: "Idade Centrada", val: 9.2, status: "Moderado" },
    { name: "AST_per_min", val: 8.3, status: "Moderado" }
];

window.modelMetrics = [
    { name: "HistGradientBoosting (HGB)", mape: 52.18, r2: 0.559, maeUsd: 3678298, color: "var(--accent-teal)", interpret: "Permutation + SHAP" },
    { name: "Random Forest", mape: 52.85, r2: 0.550, maeUsd: 3580678, color: "var(--accent-purple)", interpret: "Importância por impureza" },
    { name: "Ridge Regression", mape: 55.34, r2: 0.588, maeUsd: 3721480, color: "var(--accent-blue)", interpret: "Coeficientes L2" },
    { name: "Lasso Regression", mape: 56.49, r2: 0.581, maeUsd: 3778538, color: "var(--accent-orange)", interpret: "Coeficientes L1" },
    { name: "OLS Baseline", mape: 58.91, r2: 0.562, maeUsd: 3850058, color: "var(--accent-pink)", interpret: "Coeficientes OLS" }
];

window.coefCompareData = [
    { name: "Minutos (MP)", ols: 0.74, ridge: 0.56, lasso: 0.69 },
    { name: "Idade (Age)", ols: 0.57, ridge: 0.41, lasso: 0.48 },
    { name: "USG%", ols: 0.21, ridge: 0.18, lasso: 0.20 },
    { name: "Age²", ols: -0.16, ridge: -0.06, lasso: -0.09 },
    { name: "Rookie (CBA)", ols: 0.55, ridge: 0.21, lasso: 0.33 },
    { name: "PTS/GP", ols: -0.11, ridge: -0.03, lasso: -0.06, warning: true }
];

window.ridgeCoefs = [
    { name: "Minutos (MP)", val: 0.56, warning: false },
    { name: "Idade (Age)", val: 0.41, warning: false },
    { name: "USG%", val: 0.18, warning: false },
    { name: "Veteran", val: 0.25, warning: false },
    { name: "STL_BLK_sum", val: 0.16, warning: false },
    { name: "PTS/GP", val: -0.03, warning: true }
];

window.lassoCoefs = [
    { name: "Minutos (MP)", val: 0.69, warning: false },
    { name: "Idade (Age)", val: 0.48, warning: false },
    { name: "Rookie (CBA)", val: 0.33, warning: false },
    { name: "USG%", val: 0.20, warning: false },
    { name: "Veteran", val: 0.21, warning: false },
    { name: "PTS/GP", val: -0.06, warning: true }
];

window.olsCoefs = [
    { name: "Minutos (MP)", val: 0.74, warning: false },
    { name: "Idade (Age)", val: 0.57, warning: false },
    { name: "Rookie (CBA)", val: 0.55, warning: false },
    { name: "Uso (USG%)", val: 0.21, warning: false },
    { name: "Age²", val: -0.16, warning: false },
    { name: "Pontos (PTS/GP)", val: -0.11, warning: true }
];

window.permutationData = [
    { name: "Minutos Jogados (MP)", val: 0.452 },
    { name: "Idade (Age)", val: 0.262 },
    { name: "Pontos (PTS/GP)", val: 0.041 },
    { name: "Uso (USG%)", val: 0.023 },
    { name: "3P%", val: 0.013 }
];

window.shapPlayerStats = {
    curry: [
        { lbl: "Perfil", val: "Superstar" },
        { lbl: "Salário real", val: "US$ 48,1M" },
        { lbl: "Previsto HGB", val: "US$ 46,3M" }
    ],
    kaminsky: [
        { lbl: "Perfil", val: "Role player" },
        { lbl: "Salário real", val: "US$ 2,5M" },
        { lbl: "Previsto HGB", val: "US$ 1,9M" }
    ],
    hardy: [
        { lbl: "Perfil", val: "Rookie scale" },
        { lbl: "Salário real", val: "US$ 1,0M" },
        { lbl: "Previsto HGB", val: "US$ 2,7M" }
    ]
};

window.shapProfiles = {
    curry: {
        baseVal: 4.58,
        predVal: 46.26,
        forces: [
            { name: "Minutos (MP)", val: 7.96, positive: true },
            { name: "Idade (Age)", val: 10.67, positive: true },
            { name: "Uso (USG%)", val: 2.74, positive: true },
            { name: "Pontos (PTS/GP)", val: 3.06, positive: true },
            { name: "Tocos/GP", val: 2.86, positive: true },
            { name: "Outros", val: 14.39, positive: true }
        ]
    },
    kaminsky: {
        baseVal: 4.58,
        predVal: 1.86,
        forces: [
            { name: "Minutos (MP)", val: -2.37, positive: false },
            { name: "Idade (Age)", val: 0.54, positive: true },
            { name: "Pontos (PTS/GP)", val: -0.25, positive: false },
            { name: "Tocos/GP", val: -0.18, positive: false },
            { name: "AST/TOV", val: -0.16, positive: false },
            { name: "Outros", val: -0.3, positive: false }
        ]
    },
    hardy: {
        baseVal: 4.58,
        predVal: 2.74,
        forces: [
            { name: "Minutos (MP)", val: -1.6, positive: false },
            { name: "Idade (Age)", val: -0.84, positive: false },
            { name: "Jogos (GP)", val: 0.39, positive: true },
            { name: "Uso (USG%)", val: 0.29, positive: true },
            { name: "Age²", val: 0.16, positive: true },
            { name: "Outros", val: -0.25, positive: false }
        ]
    }
};

// Distribuição salarial completa — 467 jogadores (dataset bruto 2022-23)
window.SALARY_HISTOGRAM = {
    total: 467,
    removedOutliers: 38,
    cleanTotal: 429,
    bins: [
        { label: "<1M", count: 75, outliers: 38 },
        { label: "1-2M", count: 73, outliers: 0 },
        { label: "2-3M", count: 67, outliers: 0 },
        { label: "3-5M", count: 53, outliers: 0 },
        { label: "5-8M", count: 47, outliers: 0 },
        { label: "8-12M", count: 47, outliers: 0 },
        { label: "12-18M", count: 36, outliers: 0 },
        { label: "18-25M", count: 21, outliers: 0 },
        { label: "25-35M", count: 25, outliers: 0 },
        { label: "35-50M", count: 23, outliers: 0 }
    ]
};

// Desvio absoluto médio (MAE) por perfil — HistGradientBoosting, conjunto de teste
// Valores derivados da análise de erros do relatório (US$)
window.fitMapData = [
    {
        group: "Role players",
        desc: "Salário acompanha minutos e produção",
        maeM: 3.13,
        n: 71,
        color: "var(--accent-teal)",
        example: "Frank Kaminsky: real US$ 2,5M vs previsto US$ 1,9M (desvio US$ 0,6M)"
    },
    {
        group: "Rookies (CBA)",
        desc: "Contrato limitado pela regra da liga",
        maeM: 1.95,
        n: 27,
        color: "var(--accent-blue)",
        example: "Jaden Hardy: real US$ 1,0M vs previsto US$ 2,7M (desvio US$ 1,7M)"
    },
    {
        group: "Superstars",
        desc: "Marca e legado elevam o contrato",
        maeM: 10.99,
        n: 2,
        color: "var(--accent-purple)",
        example: "Stephen Curry: real US$ 48,1M vs previsto US$ 46,3M (desvio US$ 1,8M)"
    },
    {
        group: "Lesionados",
        desc: "Poucos jogos, contrato antigo alto",
        maeM: 24.61,
        n: 2,
        color: "var(--accent-rose)",
        example: "Jonathan Isaac: real US$ 17,4M vs previsto US$ 0,7M (desvio US$ 16,7M)"
    },
    {
        group: "Supermax antigos",
        desc: "Contrato herdado do auge da carreira",
        maeM: 8.57,
        n: 6,
        color: "var(--accent-orange)",
        example: "Kemba Walker: real US$ 37,3M vs previsto US$ 4,8M (desvio US$ 32,5M)"
    }
];

window.HGB_OVERALL_MAE_M = 3.68;

window.extremeErrors = [
    { name: "Kemba Walker", real: 37.3, pred: 4.8, error: 32.5, age: 32, gp: 9, pos: "PG", cause: "Assinou contrato máximo anterior, mas jogou apenas 9 partidas devido a lesões crônicas no joelho." },
    { name: "Myles Turner", real: 35.1, pred: 9.4, error: 25.7, age: 26, gp: 62, pos: "C", cause: "Contrato estendido acima do valor de mercado atual; o modelo prevê salário compatível com produção recente." },
    { name: "Russell Westbrook", real: 47.1, pred: 28.9, error: 18.2, age: 34, gp: 73, pos: "PG", cause: "Supermax antigo: minutos consistentes, mas eficiência abaixo do que o contrato sugere." },
    { name: "Shai Gilgeous-Alexander", real: 30.9, pred: 13.0, error: 17.9, age: 24, gp: 68, pos: "PG", cause: "Supermax recente antes do pico estatístico; modelo subestima contrato de franquia jovem." }
];
