"""Convert paper.md into a NeurIPS-style LaTeX source (neurips/paper.tex).

The Markdown is deliberately regular (ATX headings, pipe tables, image + italic
caption pairs, numbered and bulleted lists), so a small converter is enough and
keeps one source of truth: edit paper.md, rerun this, rebuild.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
STY = "neurips_2025" if os.path.exists(os.path.join(HERE, "neurips", "neurips_2025.sty")) else "neurips_2023"

SPECIAL = {"×": r"$\times$", "°": r"$^\circ$", "≈": r"$\approx$", "±": r"$\pm$", "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$",
           "µ": r"$\mu$", "⁻⁶": r"$^{-6}$", "−": "--", "–": "--", "—": "---", "…": r"\ldots{}", "’": "'", "‘": "`", "“": "``", "”": "''",
           "√": r"$\surd$", "★": "*"}
def esc(t):
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    for a, b in SPECIAL.items(): t = t.replace(a, b)
    t = re.sub(r'"([^"]+)"', r"``\1''", t)
    return t

def inline(t):
    """Markdown inline -> LaTeX, escaping text but not the markup we emit."""
    out = []; i = 0
    pat = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|\*(.+?)\*")
    for m in pat.finditer(t):
        out.append(esc(t[i:m.start()]))
        if m.group(1) is not None: out.append(r"\texttt{" + esc(m.group(1)) + "}")
        elif m.group(2) is not None: out.append(r"\href{" + m.group(3) + "}{" + inline(m.group(2)) + "}")
        elif m.group(4) is not None: out.append(r"\textbf{" + inline(m.group(4)) + "}")
        elif m.group(5) is not None: out.append(r"\emph{" + inline(m.group(5)) + "}")
        i = m.end()
    out.append(esc(t[i:]))
    return "".join(out)

def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\s*\|?\s*-", l)]
    ncol = max(len(r) for r in rows)
    wide = ncol >= 3
    first, rest = (0.28, 0.64) if ncol < 5 else (0.16, 0.70)
    spec = "@{}" + ("p{%.2f\\linewidth}" % first + "p{%.2f\\linewidth}" % (rest / (ncol - 1)) * (ncol - 1) if wide else "p{0.34\\linewidth}p{0.60\\linewidth}") + "@{}"
    body = [r"\begin{table}[htbp]\centering\small", r"\begin{tabular}{" + spec + "}", r"\toprule"]
    body.append(" & ".join(inline(c) for c in rows[0]) + r" \\ \midrule")
    for r in rows[1:]:
        r = r + [""] * (ncol - len(r)); body.append(" & ".join(inline(c) for c in r) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(body)

src = open(os.path.join(HERE, "paper.md"), encoding="utf-8").read().split("\n")
# header
title = src[0].lstrip("# ").strip()
authors = re.sub(r"\*\*", "", src[2]).strip()
affil = src[3].strip()
i = 4
while not src[i].startswith("## Abstract"): i += 1
i += 1
abstract = []
while not src[i].startswith("---"): 
    if src[i].strip(): abstract.append(src[i].strip())
    i += 1
i += 1
body = []
# Any front-matter lines between title and abstract worth keeping: the draft note and code links
front = [l for l in src[4:] if l.startswith("*Draft") or l.startswith("Code:")]

tex = []
tex.append(r"\documentclass{article}")
tex.append(r"\usepackage[final]{" + STY + "}")
tex.append(r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{hyperref}\usepackage{url}\usepackage{booktabs}\usepackage{amsfonts}\usepackage{amsmath}\usepackage{nicefrac}\usepackage{microtype}\usepackage{graphicx}\usepackage{xcolor}\usepackage{enumitem}")
tex.append(r"\graphicspath{{../figures/}}")
tex.append(r"\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue!60!black}")
tex.append(r"\title{" + inline(title) + "}")
names = [n.strip() for n in authors.split(",")]
tex.append(r"\author{" + (r" \And ".join(inline(n) for n in names[:3])) + r" \AND " + (r" \And ".join(inline(n) for n in names[3:])) + r" \AND " + r" \\ ".join(inline(x) for x in affil.split(", AI Grand Prix")).replace(r"\\ ", r"\\ AI Grand Prix", 1) + "}")
tex.append(r"\begin{document}\maketitle")
tex.append(r"\begin{abstract}" + "\n" + inline(" ".join(abstract)) + "\n" + r"\end{abstract}")
if front: tex.append(r"\begin{center}\small " + r" \\ ".join(inline(l) for l in front) + r"\end{center}")

n = len(src); para = []; figcount = 0
def flush():
    global para
    if para: tex.append(inline(" ".join(para))); tex.append(""); para = []
while i < n:
    l = src[i]
    if l.startswith("## "):
        flush(); t = re.sub(r"^\d+\.\s*", "", l[3:]).strip()
        if t.startswith("Appendix A"): tex.append(r"\appendix")
        if t.startswith("Appendix "): t = re.sub(r"^Appendix [A-Z]\.\s*", "", t)
        tex.append((r"\section*{" if t in ("Acknowledgements", "References") else r"\section{") + inline(t) + "}")
        if t == "References": tex.append(r"\small")
    elif l.startswith("### "):
        flush(); tex.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+\s*", "", l[4:]).strip()) + "}")
    elif l.startswith("![]("):
        flush(); img = re.match(r"!\[\]\((.+)\)", l).group(1).replace("figures/", "")
        # caption = next non-empty italic line
        j = i + 1
        while j < n and not src[j].strip(): j += 1
        cap = src[j].strip().strip("*") if j < n and src[j].startswith("*Figure") else ""
        cap = re.sub(r"^Figure \d+\.\s*", "", cap)
        h = "0.60\\textheight" if img.endswith(("course_overlay.png", "race_contact_sheet.jpg", "video_posters.png", "race40drop_first_lap.png", "project_timeline.png", "stress_causes.png", "observation_contract.png")) else "0.42\\textheight"
        stem = os.path.splitext(img)[0]
        if img.endswith(".gif"): img = stem + "_still.png"     # the PDF gets one frame; GitHub animates the GIF
        elif os.path.exists(os.path.join(HERE, "figures", stem + ".pdf")): img = stem + ".pdf"
        tex.append(r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth,height=" + h + r",keepaspectratio]{" + img + "}")
        tex.append(r"\caption{" + inline(cap) + "}\end{figure}")
        i = j
    elif l.startswith("*Figure") or l.startswith("*Table"):
        pass  # captions handled with their figure / table
    elif l.startswith("|"):
        flush(); block = []
        while i < n and src[i].startswith("|"): block.append(src[i]); i += 1
        j = i
        while j < n and not src[j].strip(): j += 1
        cap = ""
        if j < n and src[j].startswith("*Table"): cap = re.sub(r"^Table \d+\.\s*", "", src[j].strip().strip("*")); i = j
        t = table(block)
        if cap: t = t.replace(r"\end{table}", r"\caption{" + inline(cap) + r"}\end{table}")
        tex.append(t); tex.append("")
    elif re.match(r"^\d+\.\s", l):
        flush(); items = []
        while i < n and re.match(r"^\d+\.\s", src[i]): items.append(re.sub(r"^\d+\.\s", "", src[i])); i += 1
        tex.append(r"\begin{enumerate}[leftmargin=*]"); tex += [r"\item " + inline(x) for x in items]; tex.append(r"\end{enumerate}"); continue
    elif l.startswith("- "):
        flush(); items = []
        while i < n and src[i].startswith("- "): items.append(src[i][2:]); i += 1
        tex.append(r"\begin{itemize}[leftmargin=*]"); tex += [r"\item " + inline(x) for x in items]; tex.append(r"\end{itemize}"); continue
    elif l.startswith("---"):
        flush()
    elif not l.strip():
        flush()
    else:
        para.append(l.strip())
    i += 1
flush()
tex.append(r"\end{document}")
os.makedirs(os.path.join(HERE, "neurips"), exist_ok=True)
open(os.path.join(HERE, "neurips", "paper.tex"), "w", encoding="utf-8").write("\n".join(tex))
print("wrote neurips/paper.tex using", STY)
