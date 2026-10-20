"""Build one printable handbook (HTML and PDF) per person, plus one for the shared Start Here documents.

    /Users/advikdivekar/miniconda3/bin/python tools/build_handbooks.py       (any Python with markdown-it-py)

Each handbook = the person's four documents + all their diagrams drawn in. The PDF is made with Google Chrome (headless).
Re-run after editing any .md file or after re-running tools/make_diagrams.py.
"""

import html
import re
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font: 11pt/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2937; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 21pt; border-bottom: 3px solid #6d28d9; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 15pt; margin-top: 22px; color: #111827; border-bottom: 1px solid #d1d5db; padding-bottom: 2px; }
h3 { font-size: 12.5pt; margin-top: 16px; color: #374151; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 4px 7px; vertical-align: top; text-align: left; }
th { background: #f3f4f6; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f3f4f6; padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 9pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #6d28d9; margin: 10px 0; padding: 2px 12px; background: #faf5ff; }
.doc { page-break-before: always; }
.doc:first-of-type { page-break-before: auto; }
figure { margin: 14px 0; page-break-inside: avoid; text-align: center; }
figure svg { max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
figcaption { font-size: 9pt; color: #6b7280; }
.cover { text-align: center; padding: 45mm 0 20mm; page-break-after: always; }
.cover h1 { border: 0; font-size: 28pt; }
.cover p { font-size: 13pt; color: #4b5563; }
.toc li { margin: 4px 0; }
"""


def render(markdown_text):
    md = MarkdownIt("commonmark").enable("table")
    return md.render(markdown_text)


def inline_svg(path):
    text = path.read_text()
    text = re.sub(r"^<\?xml[^>]*\?>", "", text).strip()
    return f"<figure>{text}<figcaption>{html.escape(path.stem.replace('_', ' '))}</figcaption></figure>"


def build(title, subtitle, docs, diagram_dirs, out_base, extra_note=""):
    parts = [f'<div class="cover"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p><p>{html.escape(extra_note)}</p>'
             f'<p>Machine Learning (702DB0C015), B.Tech CSE (Data Science), Semester V, 2026-27</p></div>']
    toc = "".join(f'<li>{html.escape(name)}</li>' for name, _ in docs)
    parts[0] = parts[0].replace("</div>", f'<h2>Contents</h2><ol class="toc" style="display:inline-block;text-align:left">{toc}<li>Diagrams</li></ol></div>')
    for name, path in docs:
        parts.append(f'<div class="doc">{render(path.read_text())}</div>')
    diagrams = []
    for directory in diagram_dirs:
        diagrams += sorted(directory.glob("*.svg"))
    if diagrams:
        parts.append('<div class="doc"><h1>Diagrams</h1>' + "".join(inline_svg(p) for p in diagrams) + "</div>")
    page = f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{"".join(parts)}</body></html>'
    html_path = out_base.with_suffix(".html")
    html_path.write_text(page)
    pdf_path = out_base.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", f"file://{html_path}"],
                   check=True, capture_output=True, timeout=180)
    print("built", html_path.name, "and", pdf_path.name)


def main():
    people = [("01_Advik", "Advik", "System, evaluation, deep search, live viewer (lead)"),
              ("02_Aaryan", "Aaryan", "Problem, domain, data and preprocessing"),
              ("03_Paras", "Paras", "Labels, dataset and the neural network (CNN)"),
              ("04_Satvik", "Satvik", "Linking, tree models and metrics")]
    for folder, name, role in people:
        base = ROOT / folder
        docs = [(label, base / file) for label, file in [("Part brief", "PART_BRIEF.md"), ("Knowledge pack", "KNOWLEDGE_PACK.md"),
                                                          ("Presentation guide", "PRESENTATION_GUIDE.md"), ("Viva questions and answers", "VIVA_QA.md")]]
        build(f"{name}: handbook", role, docs, [base / "diagrams"], base / f"Handbook_{name}",
              "Read 00_Start_Here first; then this handbook.")
    start = ROOT / "00_Start_Here"
    docs = [(path.stem.replace("_", " "), path) for path in sorted(start.glob("0*.md"))]
    build("Asteroid Hunter: Start Here", "Everyone reads this first", docs, [start / "diagrams"], start / "Handbook_Start_Here")


if __name__ == "__main__":
    main()
