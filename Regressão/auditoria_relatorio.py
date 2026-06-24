#!/usr/bin/env python3
"""Auditoria: relatorio/main.tex vs artefatos do pipeline."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TEX = REPO / "relatorio" / "main.tex"


@dataclass
class AuditResult:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


def _close(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol


def _round2(x: float) -> float:
    return round(x, 2)


def _load_data() -> dict:
    df_raw = pd.read_csv(ROOT / "dataset" / "nba_2022-23_all_stats_with_salary.csv")
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])
    df_raw["Position_Clean"] = df_raw["Position"].apply(lambda x: str(x).split("-")[0])

    metrics = pd.read_csv(ROOT / "3_MODELAGEM" / "resultados" / "metricas_comparacao.csv")
    ols = pd.read_csv(ROOT / "4_INTERPRETABILIDADE" / "coeficientes_ols.csv")
    ridge_lasso = pd.read_csv(ROOT / "4_INTERPRETABILIDADE" / "coeficientes_ridge_lasso.csv")
    imp = pd.read_csv(ROOT / "4_INTERPRETABILIDADE" / "importancia_permutacao" / "importancia.csv")
    vif_pos = pd.read_csv(ROOT / "2_PREPROCESSAMENTO" / "vif_pos_preprocessamento.csv")
    vif_antes = pd.read_csv(ROOT / "2_PREPROCESSAMENTO" / "vif_antes_narrativo.csv")
    erros = pd.read_csv(ROOT / "3_MODELAGEM" / "resultados" / "erros_extremos.csv")
    shap = json.loads((ROOT / "4_INTERPRETABILIDADE" / "shap_values" / "shap_profiles.json").read_text())
    eda_stats = pd.read_csv(ROOT / "1_EDA" / "estatisticas_descritivas.csv", index_col=0)
    X_test = pd.read_csv(ROOT / "2_PREPROCESSAMENTO" / "X_test.csv")

    df = df_raw[df_raw["Salary"] >= 500_000].copy()
    log_salary = np.log(df["Salary"])

    return {
        "df_raw": df_raw,
        "df": df,
        "metrics": metrics,
        "ols": ols,
        "ridge_lasso": ridge_lasso,
        "imp": imp,
        "vif_pos": vif_pos,
        "vif_antes": vif_antes,
        "erros": erros,
        "shap": shap,
        "eda_stats": eda_stats,
        "n_test": len(X_test),
        "skew_before": float(df["Salary"].skew()),
        "skew_after": float(log_salary.skew()),
    }


def audit_counts(data: dict, r: AuditResult) -> None:
    n_raw = len(data["df_raw"])
    n_final = len(data["df"])
    removed = n_raw - n_final
    if n_raw == 467 and removed == 38 and n_final == 429:
        r.ok(f"Contagens: {n_raw} -> {n_final} (removidos {removed})")
    else:
        r.fail(f"Contagens: esperado 467/38/429, obtido {n_raw}/{removed}/{n_final}")
    if data["n_test"] == 108:
        r.ok("n_teste = 108")
    else:
        r.fail(f"n_teste: esperado 108, obtido {data['n_test']}")


def audit_eda(data: dict, r: AuditResult) -> None:
    stats = data["eda_stats"]
    mean_sal_m = stats.loc["Salary", "mean"] / 1e6
    med_sal_m = stats.loc["Salary", "50%"] / 1e6
    mean_age = stats.loc["Age", "mean"]

    if _close(mean_sal_m, 8.4, 0.05):
        r.ok(f"Média salário ~8,4M (pipeline {mean_sal_m:.2f}M)")
    else:
        r.fail(f"Média salário: relatório 8,4M vs pipeline {mean_sal_m:.2f}M")

    if _close(med_sal_m, 3.7, 0.05):
        r.ok(f"Mediana salário ~3,7M (pipeline {med_sal_m:.2f}M)")
    else:
        r.fail(f"Mediana salário: relatório 3,7M vs pipeline {med_sal_m:.2f}M")

    if _close(mean_age, 25.8, 0.05):
        r.ok(f"Idade média ~25,8 (pipeline {mean_age:.2f})")
    else:
        r.fail(f"Idade média: relatório 25,8 vs pipeline {mean_age:.2f}")

    df_raw = data["df_raw"]
    corr_pts = df_raw[["Salary", "PTS"]].corr().iloc[0, 1]
    corr_vorp = df_raw[["Salary", "VORP"]].corr().iloc[0, 1]
    corr_mp = df_raw[["Salary", "MP"]].corr().iloc[0, 1]
    corr_per_bpm = df_raw[["PER", "BPM"]].corr().iloc[0, 1]

    for name, val, expected in [
        ("corr(Salary, PTS)", corr_pts, 0.73),
        ("corr(Salary, VORP)", corr_vorp, 0.68),
        ("corr(Salary, MP)", corr_mp, 0.64),
        ("corr(PER, BPM)", corr_per_bpm, 0.90),
    ]:
        if _close(val, expected, 0.02):
            r.ok(f"{name} = {val:.2f} (relatório {expected:.2f})")
        else:
            r.fail(f"{name}: relatório {expected:.2f} vs pipeline {val:.2f}")

    if _close(data["skew_before"], 1.75, 0.05):
        r.ok(f"Assimetria pré-log ~1,75 (pipeline {data['skew_before']:.2f})")
    else:
        r.fail(f"Assimetria pré-log: relatório 1,75 vs pipeline {data['skew_before']:.2f}")

    if _close(data["skew_after"], 0.06, 0.03):
        r.ok(f"Assimetria pós-log ~0,06 (pipeline {data['skew_after']:.2f})")
    else:
        r.fail(f"Assimetria pós-log: relatório 0,06 vs pipeline {data['skew_after']:.2f}")

    pos_mean = df_raw.groupby("Position_Clean")["Salary"].mean()
    if pos_mean.idxmax() == "PG":
        r.ok("Maior salário médio por posição: PG")
    else:
        r.fail(f"Maior salário médio por posição: PG vs pipeline {pos_mean.idxmax()}")


def audit_vif(data: dict, r: AuditResult) -> None:
    vif_age_antes = data["vif_antes"].loc[data["vif_antes"]["Feature"] == "Age", "VIF"].iloc[0]
    vif_age_pos = data["vif_pos"].loc[data["vif_pos"]["Feature"] == "Age", "VIF"].iloc[0]
    if _close(vif_age_antes, 512, 1.0):
        r.ok(f"VIF Age antes ~512 ({vif_age_antes:.1f})")
    else:
        r.fail(f"VIF Age antes: relatório 512 vs {vif_age_antes:.1f}")
    if _close(vif_age_pos, 9.2, 0.1):
        r.ok(f"VIF Age depois ~9,2 ({vif_age_pos:.1f})")
    else:
        r.fail(f"VIF Age depois: relatório 9,2 vs {vif_age_pos:.1f}")

    tex_vif = {
        "PTS_per_GP": 13.8, "MP": 11.1, "STL_BLK_sum": 9.8,
        "Age": 9.2, "AST_per_min": 8.3, "BPM": 6.1,
    }
    for feat, expected in tex_vif.items():
        row = data["vif_pos"].loc[data["vif_pos"]["Feature"] == feat, "VIF"]
        if row.empty:
            r.fail(f"VIF {feat}: não encontrado no pipeline")
            continue
        val = row.iloc[0]
        if _close(val, expected, 0.15):
            r.ok(f"VIF {feat}: {expected} (pipeline {val:.1f})")
        else:
            r.fail(f"VIF {feat}: relatório {expected} vs pipeline {val:.1f}")


def audit_metrics(data: dict, r: AuditResult) -> None:
    tex_rows = {
        "OLS": (0.5621, 0.7569, 0.5703, 0.573, 3850058, 58.91),
        "Ridge": (0.5878, 0.7344, 0.5474, 0.539, 3721480, 55.34),
        "Lasso": (0.5810, 0.7405, 0.5545, 0.548, 3778538, 56.49),
        "Random Forest": (0.5503, 0.7671, 0.5470, 0.588, 3580678, 52.85),
        "HistGradientBoosting": (0.5585, 0.7600, 0.5518, 0.578, 3678298, 52.18),
    }
    m = data["metrics"]
    for modelo, (r2, rmse, mae, msle, mae_usd, mape) in tex_rows.items():
        row = m.loc[m["Modelo"] == modelo].iloc[0]
        checks = [
            ("R2_Test", row["R2_Test"], r2, 0.0005),
            ("RMSE_Log", row["RMSE_Log"], rmse, 0.0005),
            ("MAE_Log", row["MAE_Log"], mae, 0.0005),
            ("MSLE", row["MSLE"], msle, 0.001),
            ("MAE_USD", row["MAE_USD"], mae_usd, 5000),
            ("MAPE", row["MAPE"], mape, 0.05),
        ]
        for name, got, exp, tol in checks:
            if abs(got - exp) <= tol:
                r.ok(f"Métricas {modelo} {name}: OK")
            else:
                r.fail(f"Métricas {modelo} {name}: relatório {exp} vs pipeline {got}")

    stability = {
        "OLS": 0.052, "Ridge": 0.063, "Lasso": 0.058,
        "Random Forest": 0.113, "HistGradientBoosting": 0.097,
    }
    for modelo, exp in stability.items():
        got = m.loc[m["Modelo"] == modelo, "R2_CV_std"].iloc[0]
        if _close(got, exp, 0.002):
            r.ok(f"Estabilidade {modelo}: {exp}")
        else:
            r.fail(f"Estabilidade {modelo}: relatório {exp} vs {got:.3f}")


def audit_ols(data: dict, r: AuditResult) -> None:
    ols = data["ols"].set_index("Feature")
    tex_coef = {
        "MP": 0.74, "Age": 0.57, "Experience_Category_Rookie": 0.55,
        "USG%": 0.21, "Age_sq": -0.16, "AST_per_min": 0.10, "BPM": 0.10,
        "PTS_per_GP": -0.11, "Position_Clean_PG": -0.12,
    }
    for feat, exp in tex_coef.items():
        got = ols.loc[feat, "Coeficiente"]
        if _close(got, exp, 0.015):
            r.ok(f"OLS {feat}: {exp}")
        else:
            r.fail(f"OLS {feat}: relatório {exp} vs pipeline {got:.3f}")


def audit_ridge_lasso(data: dict, r: AuditResult) -> None:
    rl = data["ridge_lasso"].set_index("Feature")
    tex = {
        "MP": (0.56, 0.69), "Age": (0.41, 0.48), "Age_sq": (-0.06, -0.09),
        "USG%": (0.18, 0.20),
        "Experience_Category_Rookie": (0.21, 0.33),
        "Experience_Category_Veteran": (0.25, 0.21),
    }
    for feat, (exp_r, exp_l) in tex.items():
        got_r = rl.loc[feat, "Ridge"]
        got_l = rl.loc[feat, "Lasso"]
        if _close(got_r, exp_r, 0.015) and _close(got_l, exp_l, 0.015):
            r.ok(f"Ridge/Lasso {feat}: {exp_r}/{exp_l}")
        else:
            r.fail(
                f"Ridge/Lasso {feat}: relatório {exp_r}/{exp_l} "
                f"vs pipeline {got_r:.3f}/{got_l:.3f}"
            )


def audit_permutation(data: dict, r: AuditResult) -> None:
    imp = data["imp"].set_index("Feature")
    tex = {"MP": 0.452, "Age": 0.262, "PTS_per_GP": 0.041, "USG%": 0.023}
    for feat, exp in tex.items():
        got = imp.loc[feat, "Importancia"]
        if _close(got, exp, 0.002):
            r.ok(f"Permutation {feat}: {exp}")
        else:
            r.fail(f"Permutation {feat}: relatório {exp} vs pipeline {got:.3f}")

    mp_drop = imp.loc["MP", "Importancia"]
    age_drop = imp.loc["Age", "Importancia"]
    hgb_r2 = data["metrics"].loc[
        data["metrics"]["Modelo"] == "HistGradientBoosting", "R2_Test"
    ].iloc[0]
    rel_drop = mp_drop / hgb_r2
    sum_mp_age = mp_drop + age_drop

    if _close(sum_mp_age, 0.71, 0.02):
        r.ok(f"Permutation MP+Age acumulado ~0,71 ({sum_mp_age:.3f})")
    if rel_drop > 0.75:
        r.ok(
            f"Queda relativa MP/R² HGB ~{rel_drop*100:.0f}% "
            f"({mp_drop:.3f}/{hgb_r2:.4f})"
        )


def audit_shap_curry(data: dict, r: AuditResult) -> None:
    curry = data["shap"]["curry"]
    if _close(curry["predVal"], 46.3, 0.1):
        r.ok(f"Curry pred ~46,3M (pipeline {curry['predVal']})")
    else:
        r.fail(f"Curry pred: relatório 46,3M vs {curry['predVal']}")
    if _close(curry["realVal"], 48.1, 0.1):
        r.ok(f"Curry real ~48,1M (pipeline {curry['realVal']})")
    else:
        r.fail(f"Curry real: relatório 48,1M vs {curry['realVal']}")

    forces = {f["name"]: f["val"] for f in curry["forces"]}
    for label in ["Minutos (MP)", "Idade (Age)", "Outros"]:
        if label in forces:
            r.ok(f"Curry SHAP force '{label}': {forces[label]:+.2f}M")
        else:
            r.fail(f"Curry SHAP: falta contribuição '{label}'")


def audit_erros(data: dict, r: AuditResult) -> None:
    erros = data["erros"].head(5)
    tex_players = [
        ("Kemba Walker", 37.3, 4.8, 32.5, 87.19),
        ("Myles Turner", 35.1, 9.4, 25.7, 73.16),
        ("Russell Westbrook", 47.1, 28.9, 18.2, 38.59),
        ("Shai Gilgeous-Alexander", 30.9, 13.0, 17.9, 57.79),
        ("Jonathan Isaac", 17.4, 0.7, 16.7, 96.04),
    ]
    for i, (name, real_m, pred_m, err_m, pct) in enumerate(tex_players):
        row = erros.iloc[i]
        if row["Player"] != name:
            r.fail(f"Erro #{i+1}: esperado {name}, obtido {row['Player']}")
            continue
        real = row["Actual_Salary"] / 1e6
        pred = row["Predicted_Salary"] / 1e6
        err = row["Error_USD"] / 1e6
        pct_got = row["Error_Pct"]
        if (
            _close(real, real_m, 0.15)
            and _close(pred, pred_m, 0.15)
            and _close(err, err_m, 0.15)
            and abs(pct_got - pct) <= 1.5
        ):
            r.ok(f"Erro {name}: OK")
        else:
            r.fail(
                f"Erro {name}: relatório {real_m}/{pred_m}/{err_m}/{pct}% "
                f"vs pipeline {real:.1f}/{pred:.1f}/{err:.1f}/{pct_got:.1f}%"
            )


def audit_interpretations(r: AuditResult) -> None:
    tex = TEX.read_text(encoding="utf-8")

    if re.search(r"71\\% no \$R\^2\$", tex) or "71% no $R^2$" in tex:
        r.fail(
            "INTERPRETAÇÃO: frase '71% no R²' ao embaralhar MP está incorreta "
            "(0,452 é queda absoluta; 71% ≈ soma MP+Age)"
        )
    elif "importância acumulada de 0,71" in tex and "0,45 pontos" in tex:
        r.ok("INTERPRETAÇÃO permutation: queda absoluta 0,45 e acumulado MP+Age 0,71 corretos")

    if "matriz de correlação" in tex.lower():
        r.fail("TEXTO: ainda cita 'matriz de correlação' (figura é painel com barras)")
    elif "correlações com salário" in tex.lower() or "painel de correlações" in tex.lower():
        r.ok("TEXTO EDA: descreve correlações com salário (alinhado à figura)")

    if re.search(r"0\{,\}56.*0\{,\}59", tex):
        r.fail(
            "CONCLUSÃO: faixa R² 0,56–0,59 exclui Random Forest (0,5503); "
            "usar 0,55–0,59"
        )
    elif re.search(r"0\{,\}55.*0\{,\}59", tex):
        r.ok("CONCLUSÃO: faixa R² 0,55–0,59 inclui todos os modelos")

    if "HistGradientBoosting" in tex:
        r.ok("Erros extremos implicitamente do melhor modelo tree (HGB)")


def audit_figures(data: dict, r: AuditResult) -> None:
    fig_dir = REPO / "relatorio"
    required = [
        "eda_relatorio.pdf",
        "importancia_relatorio.pdf",
        "pdp_relatorio.pdf",
        "shap_summary_relatorio.pdf",
        "shap_global_relatorio.pdf",
        "shap_curry_contribuicoes_relatorio.pdf",
    ]
    for name in required:
        path = fig_dir / name
        if path.exists() and path.stat().st_size > 1000:
            r.ok(f"Figura existe: {name} ({path.stat().st_size // 1024} KB)")
        else:
            r.fail(f"Figura ausente ou vazia: {name}")

    imp = data["imp"]
    top9 = imp.head(9)
    rest = imp.iloc[9:]
    rest_pos = rest["Importancia"].clip(lower=0).sum()
    if len(rest) == 19:
        r.ok(f"Permutation 'Outras (19)': {len(rest)} variáveis, soma {rest_pos:.3f}")
    else:
        r.warn(f"Permutation agregado: esperado 19 restantes, obtido {len(rest)}")

    n_features = len(imp)
    rest_shap = n_features - 8
    if rest_shap == 20:
        r.ok("SHAP global 'Outras (20)': 8 top + 20 restantes = 28 features")
    else:
        r.warn(f"SHAP agregado: 8 top + {rest_shap} restantes (esperado 20)")


def print_report(r: AuditResult) -> int:
    print("=" * 60)
    print("AUDITORIA: relatorio/main.tex vs pipeline")
    print("=" * 60)
    print(f"\nPASS: {len(r.passed)}")
    for msg in r.passed:
        print(f"  [OK] {msg}")
    if r.warnings:
        print(f"\nAVISOS: {len(r.warnings)}")
        for msg in r.warnings:
            print(f"  [!!] {msg}")
    if r.failed:
        print(f"\nFALHAS: {len(r.failed)}")
        for msg in r.failed:
            print(f"  [XX] {msg}")
    print("\n" + "=" * 60)
    if r.success and not r.warnings:
        print("RESULTADO: PASS (100% conforme)")
    elif r.success:
        print("RESULTADO: PASS com avisos (correções de texto recomendadas)")
    else:
        print("RESULTADO: FAIL — correções necessárias")
    print("=" * 60)
    return 0 if r.success else 1


def main() -> int:
    r = AuditResult()
    data = _load_data()
    audit_counts(data, r)
    audit_eda(data, r)
    audit_vif(data, r)
    audit_metrics(data, r)
    audit_ols(data, r)
    audit_ridge_lasso(data, r)
    audit_permutation(data, r)
    audit_shap_curry(data, r)
    audit_erros(data, r)
    audit_interpretations(r)
    audit_figures(data, r)
    return print_report(r)


if __name__ == "__main__":
    sys.exit(main())
