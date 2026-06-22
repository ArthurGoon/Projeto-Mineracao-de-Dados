#!/usr/bin/env python3
"""Sincroniza apresentacao/js/data.js com artefatos exportados pelo pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA_JS = REPO / "apresentacao" / "js" / "data.js"

VIF_BEFORE_LABELS = {
    "TS%": "TS%",
    "FG%": "FG%",
    "Age": "Idade",
    "Age_sq_raw": "Idade² (sem centering)",
    "PER": "PER",
    "BPM": "BPM",
    "MP": "MP",
}

VIF_AFTER_LABELS = {
    "Age": "Idade Centrada",
    "MP": "MP (Minutos)",
    "PTS_per_GP": "PTS (Pontos)",
    "USG%": "USG% (Uso)",
    "BPM": "BPM (Eficiência)",
    "TRB_per_GP": "Rebotes/GP",
    "Age_sq": "Age²",
}

PROFILE_LABELS = {
    "curry": "Superstar",
    "kaminsky": "Role player",
    "hardy": "Rookie scale",
}


def _vif_status(v: float) -> str:
    if v > 100:
        return "Colineariedade Crítica"
    if v > 10:
        return "Alta Colineariedade"
    if v > 5:
        return "Moderado"
    return "Aceitável"


def _fmt_vif_array(rows, label_map) -> str:
    lines = ["["]
    for i, row in enumerate(rows):
        name = label_map.get(row["Feature"], row["Feature"])
        comma = "," if i < len(rows) - 1 else ""
        lines.append(
            f'    {{ name: "{name}", val: {row["VIF"]:.1f}, status: "{_vif_status(row["VIF"])}" }}{comma}'
        )
    lines.append("]")
    return "\n".join(lines)


def _fmt_shap_profiles(shap: dict) -> str:
    lines = ["{"]
    keys = list(shap.keys())
    for ki, key in enumerate(keys):
        p = shap[key]
        lines.append(f"    {key}: {{")
        lines.append(f"        baseVal: {p['baseVal']},")
        lines.append(f"        predVal: {p['predVal']},")
        lines.append("        forces: [")
        for fi, f in enumerate(p["forces"]):
            pos = "true" if f["positive"] else "false"
            comma = "," if fi < len(p["forces"]) - 1 else ""
            lines.append(
                f'            {{ name: "{f["name"]}", val: {f["val"]}, positive: {pos} }}{comma}'
            )
        lines.append("        ]")
        comma = "," if ki < len(keys) - 1 else ""
        lines.append(f"    }}{comma}")
    lines.append("}")
    return "\n".join(lines)


def _fmt_shap_player_stats(shap: dict) -> str:
    lines = ["{"]
    keys = list(shap.keys())
    for ki, key in enumerate(keys):
        p = shap[key]
        real = f"US$ {p['realVal']:.1f}M".replace(".", ",")
        pred = f"US$ {p['predVal']:.1f}M".replace(".", ",")
        lines.append(f"    {key}: [")
        lines.append(f'        {{ lbl: "Perfil", val: "{PROFILE_LABELS[key]}" }},')
        lines.append(f'        {{ lbl: "Salário real", val: "{real}" }},')
        lines.append(f'        {{ lbl: "Previsto HGB", val: "{pred}" }}')
        comma = "," if ki < len(keys) - 1 else ""
        lines.append(f"    ]{comma}")
    lines.append("}")
    return "\n".join(lines)


def _fmt_histogram(hist: dict) -> str:
    lines = [
        "{",
        f"    total: {hist['total']},",
        f"    removedOutliers: {hist['removedOutliers']},",
        f"    cleanTotal: {hist['cleanTotal']},",
        "    bins: [",
    ]
    for i, b in enumerate(hist["bins"]):
        comma = "," if i < len(hist["bins"]) - 1 else ""
        lines.append(
            f'        {{ label: "{b["label"]}", count: {b["count"]}, outliers: {b["outliers"]} }}{comma}'
        )
    lines.append("    ]")
    lines.append("}")
    return "\n".join(lines)


def _replace_block(text: str, var_name: str, body: str) -> str:
    pattern = rf"window\.{var_name}\s*=\s*[\s\S]*?;"
    replacement = f"window.{var_name} = {body};"
    if not re.search(pattern, text):
        raise SystemExit(f"Bloco window.{var_name} não encontrado em data.js")
    return re.sub(pattern, replacement, text, count=1)


def main() -> None:
    shap = json.loads((ROOT / "4_INTERPRETABILIDADE/shap_values/shap_profiles.json").read_text())
    hist = json.loads((ROOT / "1_EDA/salary_histogram.json").read_text())
    vif_before = pd.read_csv(ROOT / "2_PREPROCESSAMENTO/vif_antes_narrativo.csv").head(6).to_dict("records")
    vif_after = pd.read_csv(ROOT / "2_PREPROCESSAMENTO/vif_pos_preprocessamento.csv").head(6).to_dict("records")

    text = DATA_JS.read_text(encoding="utf-8")
    text = _replace_block(text, "vifBeforeData", _fmt_vif_array(vif_before, VIF_BEFORE_LABELS))
    text = _replace_block(text, "vifAfterData", _fmt_vif_array(vif_after, VIF_AFTER_LABELS))
    if "window.shapPlayerStats" not in text:
        text = text.replace(
            "window.shapProfiles = {",
            "window.shapPlayerStats = {};\n\nwindow.shapProfiles = {",
        )
    text = _replace_block(text, "shapPlayerStats", _fmt_shap_player_stats(shap))
    text = _replace_block(text, "shapProfiles", _fmt_shap_profiles(shap))
    text = _replace_block(text, "SALARY_HISTOGRAM", _fmt_histogram(hist))

    DATA_JS.write_text(text, encoding="utf-8")
    print(f"OK: {DATA_JS} sincronizado.")


if __name__ == "__main__":
    main()
