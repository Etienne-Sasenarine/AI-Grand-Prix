"""Render paper.md to paper.html and paper.pdf. Needs the `markdown` package and Google Chrome (headless)."""
import os, re, subprocess, sys
import markdown
HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
src = open(os.path.join(HERE, "paper.md"), encoding="utf-8").read()
body = markdown.markdown(src, extensions=["tables", "smarty"])
# figure captions: an <em> paragraph directly after an image paragraph
body = re.sub(r'<p><img alt="" src="([^"]+)" /></p>\s*<p><em>(.*?)</em></p>', r'<figure><img src="\1"><figcaption>\2</figcaption></figure>', body, flags=re.S)
css = """
@page { size: A4; margin: 22mm 20mm 24mm 20mm; @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #555; } }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #111; max-width: 170mm; margin: 0 auto; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 8pt; font-weight: 600; }
h2 { font-size: 13.5pt; margin: 22pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 14pt 0 4pt; }
p { margin: 0 0 8pt; text-align: justify; hyphens: auto; }
h1 + p { text-align: left; }
figure { margin: 12pt 0 14pt; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 165mm; }
figcaption { font-size: 9pt; color: #333; text-align: left; margin-top: 4pt; line-height: 1.35; }
table { border-collapse: collapse; font-size: 8.8pt; margin: 8pt 0 12pt; width: 100%; }
tr { page-break-inside: avoid; }
h2, h3 { page-break-after: avoid; }
th, td { border-top: 1px solid #ccc; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1px solid #555; font-weight: 600; }
tbody tr:last-child td { border-bottom: 1px solid #555; }
code { font-family: Menlo, monospace; font-size: 8.8pt; }
hr { border: 0; border-top: 1px solid #bbb; margin: 14pt 0; }
ol li, ul li { margin-bottom: 4pt; }
a { color: #1a4d8f; text-decoration: none; }
h2:first-of-type { page-break-before: auto; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Vision-Only Autonomous Drone Racing on Borrowed Hardware</title><style>{css}</style></head><body>{body}</body></html>"
open(os.path.join(HERE, "paper.html"), "w", encoding="utf-8").write(html)
out = os.path.join(HERE, "paper.pdf")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", "file://" + os.path.join(HERE, "paper.html")], check=True, capture_output=True)
print("wrote", out, os.path.getsize(out), "bytes")
