#!/usr/bin/env python3
"""Gera index.html autocontido a partir de source.html + css/js + vídeo."""

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = ROOT / "source.html"
OUT = ROOT / "index.html"

VIDEO_CANDIDATES = [
    ROOT / "assets" / "grok-video-39b77359-8f5b-4680-a080-ce5c86bd859f.mp4",
    REPO / "grok-video-39b77359-8f5b-4680-a080-ce5c86bd859f.mp4",
]


def load_video_data_uri() -> str:
    for path in VIDEO_CANDIDATES:
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:video/mp4;base64,{encoded}"
    raise FileNotFoundError("Vídeo de fundo não encontrado.")


def sanitize_css_for_inline(css: str) -> str:
    return re.sub(
        r"@import\s+url\([^)]+\)\s*;",
        "/* fontes web removidas no bundle offline — usa system-ui */",
        css,
    )


def sanitize_js_for_inline(js: str) -> str:
    return js.replace("</script>", "<\\/script>").replace("</SCRIPT>", "<\\/SCRIPT>")


def build() -> None:
    html = SOURCE.read_text(encoding="utf-8")
    css_files = ["base.css", "slides.css", "animations.css"]
    css = sanitize_css_for_inline(
        "\n\n".join((ROOT / "css" / name).read_text(encoding="utf-8") for name in css_files)
    )
    data_js = sanitize_js_for_inline((ROOT / "js" / "data.js").read_text(encoding="utf-8"))
    app_js = sanitize_js_for_inline((ROOT / "js" / "app.js").read_text(encoding="utf-8"))
    video_uri = load_video_data_uri()

    html = re.sub(
        r'\s*<link rel="stylesheet" href="css/[^"]+">\s*',
        "",
        html,
        flags=re.MULTILINE,
    )
    html = html.replace(
        "</head>",
        f"    <style>\n{css}\n    </style>\n</head>",
        1,
    )
    html = re.sub(
        r'<source src="[^"]+" type="video/mp4">',
        f'<source src="{video_uri}" type="video/mp4">',
        html,
        count=1,
    )
    html = re.sub(
        r'\s*<script src="js/data\.js"></script>\s*<script src="js/app\.js"></script>\s*',
        f"\n    <script>\n{data_js}\n    </script>\n    <script>\n{app_js}\n    </script>\n",
        html,
        count=1,
    )

    banner = (
        "<!-- APRESENTAÇÃO NBA — arquivo autocontido (~6 MB). Abra direto no navegador. -->\n"
        "<!-- Se aparecer sem estilo (fundo branco), você baixou a versão errada/antiga. -->\n"
        "<!-- Baixe de: github.com/ArthurGoon/Projeto-Mineracao-de-Dados/raw/main/apresentacao/index.html -->\n"
        "<!-- Para editar: altere source.html, css/ e js/, depois rode: python3 build_standalone.py -->\n"
    )
    if not html.startswith("<!-- APRESENTAÇÃO NBA"):
        html = banner + html

    OUT.write_text(html, encoding="utf-8")
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"OK: {OUT}")
    print(f"Tamanho: {size_mb:.1f} MB")


if __name__ == "__main__":
    build()
