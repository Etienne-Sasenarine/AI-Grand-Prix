"""Figures for the paper, from the numbers recorded in the team's documents and the logs on disk.

House style, borrowed from the lap figure the team liked: a plain title with a
second line that says what the marks mean; reference lines labelled where
they sit; the one number that matters annotated on the chart; colour for
magnitude where there is one, and one blue for "all of them" against one
warm colour for "the one we are talking about".
"""
import os, sys, glob, csv, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import tfevents

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, INK3, GRID, PALE = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#f3f2ee"
rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                 "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                 "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                 "axes.titlesize": 9, "axes.titleweight": "normal", "svg.fonttype": "none", "pdf.fonttype": 42})

HEADLINES = False   # titles and subtitles live in the paper's captions, where the text is selectable

def save(fig, name, vector=True):
    """PNG for GitHub, plus PDF and SVG with live text for the two typeset renderings (charts only; photo composites stay raster)."""
    p = os.path.join(OUT, name); fig.savefig(p)
    if vector:
        stem = os.path.splitext(p)[0]
        fig.savefig(stem + ".pdf"); fig.savefig(stem + ".svg")
        # the SVG names matplotlib's bundled font, which browsers do not have; give them a sans-serif stack instead
        svg = open(stem + ".svg", encoding="utf-8").read().replace("'DejaVu Sans'", "'Helvetica Neue', Helvetica, Arial, sans-serif")
        open(stem + ".svg", "w", encoding="utf-8").write(svg)
    plt.close(fig); print("wrote", p)

def headline(fig, title, subtitle, y=1.0):
    if not HEADLINES:
        return
    h = fig.get_size_inches()[1]
    fig.text(0.0, y + 0.30 / h, title, fontsize=10.5, fontweight="semibold", color=INK, ha="left", va="bottom", transform=fig.transFigure)
    w = fig.get_size_inches()[0]
    sub = textwrap.fill(subtitle, int(w * 17.5))
    fig.text(0.0, y + 0.10 / h, sub, fontsize=8.0, color=INK2, ha="left", va="bottom", transform=fig.transFigure, linespacing=1.25)
    if "\n" in sub:
        t = fig.texts[-2]; t.set_y(y + 0.44 / h)

def refline(ax, y, label, color=INK2, ls=":", x=0.01, va="bottom", **kw):
    ax.axhline(y, color=color, lw=0.9, ls=ls, zorder=1)
    ax.text(x, y, " " + label, transform=ax.get_yaxis_transform(), fontsize=7.3, color=color, ha="left", va=va, **kw)

def note(ax, text, xy, xytext, color=INK, fontsize=7.6, ha="left", arrow=True):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize, color=color, ha=ha, va="center", zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=3, connectionstyle="arc3,rad=0.15") if arrow else None,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

def dumbbell(ax, labels, a, b, ca, cb, la, lb, unit="%", fmt="{:.0f}"):
    y = np.arange(len(labels))[::-1]
    for yi, va, vb in zip(y, a, b):
        ax.plot([vb, va], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(a, y, s=42, color=ca, zorder=3, label=la, edgecolor="white", linewidth=0.8)
    ax.scatter(b, y, s=42, color=cb, zorder=3, label=lb, edgecolor="white", linewidth=0.8)
    for yi, va, vb in zip(y, a, b):
        ax.text(va + 1.2, yi, fmt.format(va) + unit, va="center", ha="left", fontsize=7.3, color=ca)
        ax.text(vb - 1.2, yi, fmt.format(vb) + unit, va="center", ha="right", fontsize=7.3, color=cb)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
    return y

# ------------------------------------------------------------------ 1. Hover stress sweep
conds = ["nominal", "rate filter\ntau 60 ms", "mass x0.90", "wind 0.6 N", "mass x1.15", "command delay\n4 steps", "hover stick 0.19\n(outside band)", "hover stick 0.31\n(outside band)"]
surv = np.array([95.3, 94.1, 95.7, 89.8, 77.2, 70.1, 76.4, 55.7]); sett = np.array([75.8, 73.9, 73.2, 36.5, 20.4, 13.0, 7.0, 1.9])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = dumbbell(ax, conds, surv, sett, BLUE, ORANGE, "survived the 15 s episode", "settled: within 0.30 m and under 0.30 m/s")
ax.set_xlim(-6, 108); ax.set_xlabel("% of 4096 episodes")
ax.legend(loc="lower left", fontsize=7.5, ncol=1, bbox_to_anchor=(0.0, -0.32))
note(ax, "a 60 Hz policy run at 40 Hz\nsees this condition", xy=(13.0, y[5]), xytext=(30, y[5] - 0.55), color=ORANGE)
ax.text(36, y[1] + 0.02, "robust to how the aircraft responds", fontsize=7.4, color=INK2, va="center", ha="center", style="italic")
ax.text(40, y[4] + 0.02, "fragile to how much thrust it needs", fontsize=7.4, color=INK2, va="center", ha="center", style="italic")
headline(fig, "The hover policy under eight perturbations",
         "best 60 Hz checkpoint, stress_hover.py, 20 Sep; blue = survived, orange = settled, grey = the gap between them")
save(fig, "hover_stress_sweep.png")

# ------------------------------------------------------------------ 2. 40 Hz vs 60 Hz
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.45})
rates = [40, 60]
a1.plot(rates, [52.6, 87.0], "-o", color=BLUE, lw=2, ms=7, mec="white"); a1.plot(rates, [85.5, 89.0], "-o", color=ORANGE, lw=2, ms=7, mec="white")
a1.text(61.5, 84.0, "trained at 60 Hz", color=BLUE, fontsize=8, va="center"); a1.text(61.5, 92.5, "retrained at 40 Hz", color=ORANGE, fontsize=8, va="center")
for x, v, c, dx, dy in [(40, 52.6, BLUE, 0, -5), (60, 87.0, BLUE, -2.8, -3), (40, 85.5, ORANGE, 0, 4.5), (60, 89.0, ORANGE, -2.8, 3.5)]:
    a1.text(x + dx, v + dy, f"{v:.1f}%", color=c, fontsize=7.5, ha="center", va="center")
a1.annotate("", xy=(40, 53.5), xytext=(40, 84.5), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
a1.text(41, 69, "34 points", color=INK2, fontsize=7.5, va="center")
a1.set_xticks(rates); a1.set_xticklabels(["run at 40 Hz\n(the link)", "run at 60 Hz\n(training)"], fontsize=7.8); a1.set_xlim(33, 82); a1.set_ylim(40, 100)
a1.set_ylabel("hover episodes settled (%)"); a1.grid(axis="x", visible=False)
vals2 = [8.246, 3.086]
a2.bar([0, 1], vals2, 0.55, color=[BLUE, "#a9c8ee"], linewidth=0)
for i, v in enumerate(vals2): a2.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
a2.set_xticks([0, 1]); a2.set_xticklabels(["at 60 Hz", "at 40 Hz"]); a2.set_ylabel("gates passed in 15 s"); a2.grid(axis="x", visible=False); a2.set_ylim(0, 9.5)
a2.text(0.5, 9.0, "racing policy trained at 60 Hz\nkeeps 37% of its gates", ha="center", fontsize=7.8, color=INK2)
headline(fig, "The control rate is part of the plant",
         "left: hover settled fraction for two checkpoints, each run at both rates; right: the 60 Hz racing checkpoint at both rates (4096 episodes)")
save(fig, "rate_40_vs_60.png")

# ------------------------------------------------------------------ 3. MSP link timing
fig, ax = plt.subplots(figsize=(7.2, 3.3))
streams = ["no RC stream\n(read-only)", "RC stream at 40 Hz", "RC stream at 60 Hz"]
bm = [10.03, 16.75, 25.10]; bp = [10.27, 20.28, 30.30]; fb = [40.09, 66.90, 100.43]; budget = [None, 25.0, 16.67]
x = np.arange(3); w = 0.25
ax.bar(x - w, bm, w, color=BLUE, label="one batched request (MSP_MULTIPLE_MSP), mean", linewidth=0)
ax.bar(x, bp, w, color="#a9c8ee", label="batched, 95th percentile", linewidth=0)
ax.bar(x + w, fb, w, color=ORANGE, label="four separate requests, mean", linewidth=0)
for xi, v in zip(x - w, bm): ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
for xi, v in zip(x, bp): ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
for xi, v in zip(x + w, fb): ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
for xi, b, p95 in zip(x, budget, bp):
    if b is None: continue
    ax.plot([xi - 1.6 * w, xi + 1.6 * w], [b, b], color=INK, lw=1.2, ls="--", zorder=4)
    ok = p95 <= b
    if xi == 1: ax.text(xi - 0.5 * w, b + 2.5, f"budget {b:g} ms, p95 fits", fontsize=7.2, color=GREEN, va="bottom", ha="center", fontweight="semibold")
    else: ax.text(xi + 1.7 * w, b, f" budget {b:g} ms\n p95 does not fit", fontsize=7.2, color=RED, va="center", ha="left", fontweight="semibold")
ax.set_xticks(x); ax.set_xticklabels(streams); ax.set_ylabel("telemetry snapshot round trip (ms)"); ax.set_ylim(0, 112); ax.grid(axis="x", visible=False); ax.set_xlim(-0.6, 2.9)
ax.legend(loc="upper left", fontsize=7.3, bbox_to_anchor=(0.0, 1.0))
note(ax, "without message 230 the four separate requests\nfail the runner's timing gate at every stream rate", xy=(1 + w, 66.9), xytext=(1.45, 86), color=ORANGE)
headline(fig, "One 115200-baud UART carries both the telemetry and the stick stream",
         "snapshot = attitude + raw IMU + status + RC in one round trip; 300 samples per condition, measured on the aircraft over USB, 21 Sep")
save(fig, "msp_link_timing.png")

# ------------------------------------------------------------------ 4. Rate map
rc_rate, super_rate = 0.55, 0.75
stick = np.linspace(0, 1, 400)
delivered = 200 * rc_rate * stick / (1 - super_rate * stick)
asked = stick * 3.2 * 180 / np.pi
cross = asked[np.argmin(np.abs(delivered - asked)[10:]) + 10]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.38})
a1.fill_between(asked, delivered, asked, where=delivered < asked, color="#a9c8ee", alpha=0.6, lw=0, label="under-responds")
a1.fill_between(asked, delivered, asked, where=delivered >= asked, color="#f6c3ad", alpha=0.7, lw=0, label="over-responds")
a1.plot(asked, asked, color=INK2, lw=1.2, ls="--"); a1.plot(asked, delivered, color=ORANGE, lw=2.2)
a1.text(120, 112, "what the policy assumes", color=INK2, fontsize=7.5, rotation=32, ha="left", va="bottom")
a1.text(92, 330, "what Betaflight sets\n(rc_rate 55, super 75, expo 0)", color=ORANGE, fontsize=7.3, ha="left")
a1.set_xlabel("body rate the policy asked for (deg/s)"); a1.set_ylabel("rate setpoint the PID loop receives (deg/s)"); a1.set_xlim(0, 184); a1.set_ylim(0, 450)
a1.legend(loc="upper left", fontsize=7.3)
ratio = delivered[1:] / asked[1:]
a2.fill_between(asked[1:], ratio, 1, where=ratio < 1, color="#a9c8ee", alpha=0.6, lw=0)
a2.fill_between(asked[1:], ratio, 1, where=ratio >= 1, color="#f6c3ad", alpha=0.7, lw=0)
a2.plot(asked[1:], ratio, color=ORANGE, lw=2.2); a2.axhline(1, color=INK2, lw=0.9, ls="--")
for r, e in [(0.2, 0.62), (1.0, 0.78), (2.4, 1.37), (3.2, 2.40)]:
    a2.plot([r * 180 / np.pi], [e], "o", color=ORANGE, ms=4.5, mec="white"); a2.text(r * 180 / np.pi + 6, e + 0.09, f"{e:.2f}x", fontsize=7.3, ha="center", color=INK2)
a2.axvline(cross, color=INK3, lw=0.8, ls=":"); a2.text(cross, 0.5, f" loop gain crosses 1\n at {cross:.0f} deg/s", fontsize=7.2, color=INK3, va="bottom")
note(a2, "full stick: 2.4x what was asked.\nUnder-respond, over-correct, diverge.", xy=(181, 2.38), xytext=(8, 2.25), color=ORANGE)
a2.set_xlabel("body rate the policy asked for (deg/s)"); a2.set_ylabel("delivered / asked"); a2.set_xlim(0, 190); a2.set_ylim(0.4, 2.65)
headline(fig, "The shipped linear stick map against the flight controller's rate curve",
         "the policy's 3.2 rad/s (183 deg/s) envelope was sent to full stick, where this profile commands 440 deg/s; read off the aircraft over MSP_RC_TUNING")
save(fig, "rate_map_error.png")

# ------------------------------------------------------------------ 5. Barometer collapse
rows = list(csv.DictReader(open(os.path.expanduser("~/dev/ai-grand-prix/pq/logs/orin/hover_20260918_003956.csv"))))
t = np.array([float(r["t_s"]) for r in rows]); alt = np.array([float(r["altitude_m"]) for r in rows]); thr = np.array([float(r["throttle_us"]) for r in rows])
spinning = thr > 1010
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True, gridspec_kw={"hspace": 0.1})
for ax in (a1, a2):
    ax.fill_between(t, 0, 1, where=spinning, transform=ax.get_xaxis_transform(), color="#f6c3ad", alpha=0.45, lw=0)
a1.plot(t, thr, color=BLUE, lw=1.3); a1.set_ylabel("throttle stick (us)"); a1.set_ylim(980, 1300)
a1.text(t[spinning][0], 1280, "shaded = propellers turning", fontsize=7.5, color=ORANGE, ha="left", va="top")
a2.plot(t, alt, color=ORANGE, lw=1.3); a2.set_ylabel("barometer altitude (m)"); a2.set_xlabel("time (s)")
refline(a2, 0, "the floor, where the aircraft actually was", color=INK2, x=0.62)
i_min = int(np.argmin(alt)); note(a2, f"reads {alt[i_min]:.1f} m below the floor\nwhile sitting on it", xy=(t[i_min], alt[i_min]), xytext=(t[i_min] + 18, alt[i_min] + 0.8), color=ORANGE)
headline(fig, "Aircraft armed on the ground, 18 Sep: every throttle rise reads as a descent",
         "MSP telemetry at 23.6 Hz over two minutes; prop wash on the DPS310 barometer makes height unusable, so every hover design had to take height from the camera")
save(fig, "baro_collapse.png")

# ------------------------------------------------------------------ 6. Perception numbers
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.2, 0.9, 1.5], "wspace": 0.75})
def slope(ax, pairs, colors, labels, ylim, fmt="{:.3f}"):
    for (b, a), c, l in zip(pairs, colors, labels):
        ax.plot([0, 1], [b, a], "-o", color=c, lw=2, ms=6, mec="white")
        ax.text(-0.08, b, fmt.format(b), ha="right", va="center", fontsize=7, color=c); ax.text(1.08, a, fmt.format(a) + "\n" + l, ha="left", va="center", fontsize=7, color=c)
    ax.set_xticks([0, 1]); ax.set_xlim(-0.75, 2.3); ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)
slope(a1, [(0.283, 0.636), (0.760, 0.854)], [BLUE, AQUA], ["box mAP50-95", "pose mAP50-95"], (0.2, 0.95))
a1.set_xticklabels(["incumbent", "fine-tuned"], fontsize=7, rotation=20, ha="right"); a1.set_title("Held-out contiguous blocks", loc="left", fontsize=8.5)
slope(a2, [(0.517, 0.283)], [INK2], ["box mAP50-95"], (0.2, 0.6))
a2.set_xticklabels(["random", "blocks"], fontsize=7, rotation=20, ha="right"); a2.set_title("Same model, two splits", loc="left", fontsize=8.5)
th = [0.10, 0.25, 0.40, 0.50]; usable = [89, 67, 60, 53]; ready = [79, 59, 50, 36]
a3.fill_betweenx([0, 100], 0.25, 0.5, color=PALE, lw=0)
a3.plot(th, usable, "-o", color=BLUE, ms=4.5, mec="white", lw=2); a3.plot(th, ready, "-o", color=ORANGE, ms=4.5, mec="white", lw=2)
a3.text(0.515, 53, "frames with\n4+ corners", color=BLUE, fontsize=7.2, va="center"); a3.text(0.515, 36, "policy ready\n(6 good frames\nin a row)", color=ORANGE, fontsize=7.2, va="center")
a3.axvline(0.5, color=INK2, lw=0.8, ls=":"); a3.text(0.51, 99, "library\ndefault", fontsize=6.8, ha="left", va="top", color=INK2)
a3.axvline(0.25, color=INK2, lw=0.8, ls="--"); a3.text(0.24, 99, "chosen at\nthe gate", fontsize=6.8, ha="right", va="top", color=INK2)
a3.annotate("", xy=(0.25, 59), xytext=(0.5, 36), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9, connectionstyle="arc3,rad=-0.3"))
a3.text(0.40, 44, "+23 points", color=ORANGE, fontsize=7.5, ha="center")
a3.set_xlabel("keypoint confidence threshold"); a3.set_ylabel("% of 1345 frames at a real gate"); a3.set_ylim(0, 100); a3.set_xlim(0.08, 0.72); a3.set_xticks([0.1, 0.25, 0.5])
a3.set_title("The corner-visibility threshold", loc="left", fontsize=8.5)
headline(fig, "What decided the detector",
         "left: the fine-tune against the incumbent; centre: why every earlier mAP was optimistic; right: the corner-visibility threshold, measured at a real gate on 21 Sep")
save(fig, "perception_numbers.png")

# ------------------------------------------------------------------ 7. Close-gate models
fig, ax = plt.subplots(figsize=(7.2, 2.6))
y = dumbbell(ax, ["hybrid model alone", "teammate's model alone", "hybrid, else teammate\n(fallback)"], [72, 100, 100], [41, 16, 46], BLUE, ORANGE, "gate detected", "detection became a pose (PnP solved)")
ax.set_xlim(0, 118); ax.set_xlabel("% of 548 close-gate frames (gate larger than 30% of the frame)")
ax.legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(0.0, -0.45), ncol=2)
ax.set_ylim(-0.6, 2.75)
note(ax, "blind to 28% of close gates, but poses what it sees", xy=(72, y[0]), xytext=(8, y[0] + 0.5), color=BLUE)
note(ax, "sees every gate, returns corners too rough to solve", xy=(16, y[1]), xytext=(24, y[1] - 0.5), color=ORANGE)
headline(fig, "Close gates are the ones flown through; the two models fail in opposite directions",
         "walk-around capture, 640x360 on a laptop CPU; blue = gate found, orange = corners good enough for a pose")
save(fig, "close_gate_models.png")

# ------------------------------------------------------------------ 8. Training history
base = os.path.expanduser("~/dev/ai-grand-prix/isaac_drone_racer/logs/skrl/drone_racer")
runs = [("2026-09-04_17-03-36_ppo_torch", "pq_scratch: PQ track from scratch, 4 Sep 17:03", BLUE),
        ("2026-09-04_20-38-19_ppo_torch", "pq_vel: + velocity alignment, 20:38", AQUA),
        ("2026-09-04_23-48-28_ppo_torch", "pq_hook: + run-in progress, 23:48", YELLOW),
        ("2026-09-05_01-08-00_ppo_torch", "pq_speed_best: + speed through gate, 5 Sep 01:08", ORANGE),
        ("2026-09-06_01-35-30_ppo_torch", "continued 6 Sep with identical settings", RED)]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ends = []
for d, lab, c in runs:
    ev = glob.glob(os.path.join(base, d, "events.out.tfevents.*"))
    if not ev: continue
    s_ = tfevents.read(ev[0]).get("Info / Episode_Reward/gate_passed", [])
    if not s_: continue
    st, v = map(np.array, zip(*s_)); ax.plot(st, v, color=c, lw=1.5, alpha=0.95)
    ends.append((st[-1], v[-1], lab, c))
# direct labels at the right edge, spread out so they do not collide
ends.sort(key=lambda e: e[1]); ys = [e[1] for e in ends]
for k in range(1, len(ys)):
    ys[k] = max(ys[k], ys[k - 1] + 0.55)
for (st, v, lab, c), yy in zip(ends, ys):
    ax.annotate(lab, xy=(st, v), xytext=(51500, yy), fontsize=7.3, color=c, va="center", arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.6))
note(ax, "the gate-count reset bug: whenever any environment reset,\nthe previous-position buffer was overwritten for all of them,\nso no crossing could register. 0.0 for 50 000 steps.", xy=(25000, 0.0), xytext=(4000, 2.6), color=RED)
ax.set_xlabel("PPO timesteps (4096 environments each)"); ax.set_ylabel("gate_passed reward term\n(per-step mean, exploration noise on)"); ax.set_xlim(0, 78000); ax.set_ylim(-0.3, 8.8)
ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
headline(fig, "The lineage that produced pq_speed_best, from the surviving TensorBoard logs",
         "each run warm-started from the one before it on the team's copy of the PQ track; the last one is the same recipe a day later")
save(fig, "training_history.png")

# ------------------------------------------------------------------ 9. Image composites
def load(p, w=None):
    im = Image.open(os.path.expanduser(p)).convert("RGB")
    if w: im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    return im
def panel(imgs, titles, name, ncol, figw=7.2):
    n = len(imgs); nrow = int(np.ceil(n / ncol)); ratios = [im.height / im.width for im in imgs]
    fig, axs = plt.subplots(nrow, ncol, figsize=(figw, figw / ncol * max(ratios) * nrow + 0.3 * nrow)); axs = np.atleast_1d(axs).ravel()
    for ax, im, tt in zip(axs, imgs, titles):
        ax.imshow(im); ax.set_title(tt, fontsize=8, loc="left"); ax.axis("off")
    for ax in axs[n:]: ax.axis("off")
    fig.tight_layout(pad=0.4); save(fig, name, vector=False)

try:
    panel([load("~/Downloads/gate frames/0919_220524_000107.jpg", 960), load("~/dev/aigp-runs/drone_frames/0920_224905_f00150.jpg", 960),
           load("~/dev/ai-grand-prix/datasets/autolabel/overlays/0919_214639_000000.jpg", 960), load("~/dev/ai-grand-prix/datasets/hybrid/preview/img/0919_214639_000004.jpg", 960)],
          ["(a) onboard camera, hand-carried walk-around, 19 Sep", "(b) onboard camera at a close gate, 20 Sep: corners leave the frame",
           "(c) geometric auto-label: eight numbered corners", "(d) hybrid label as written for YOLO-pose training"], "what_the_drone_sees.png", 2)
    panel([load("~/dev/aigp-runs/gallery/sample/0919_214639_000160.jpg", 1400), load("~/dev/aigp-runs/gallery/worst_DISAGREEMENT_not_a_fair_sample/0919_214639_000366.jpg", 1400)],
          ["(a) a typical frame: hand-labelled model (left) and the teammate's model (right) agree", "(b) a selected worst case: the incumbent stacks seven boxes on two gates"], "detector_ab_gallery.png", 1)
    panel([load("~/Downloads/stack_sim_videos/paths_12_drones.png", 1600)],
          ["Twelve simulated attempts of the perception stack from the pad (plan view): blue = true path, dotted = the stack's own position estimate, X = crash"], "stack_sim_paths.png", 1)
    cs = load("~/dev/aigp-runs/videos/flight_contact_sheet.png", 1400); cs.save(os.path.join(OUT, "race_contact_sheet.jpg"), quality=80); print("wrote race_contact_sheet.jpg")
    kp = os.path.expanduser("~/dev/ai-grand-prix/pq/eval/pq_speed_best_play.png")
    if os.path.exists(kp): Image.open(kp).convert("RGB").save(os.path.join(OUT, "pq_speed_best_play.png"))
except FileNotFoundError as e:
    print("keeping committed composites:", e)

# The 17 Sep briefing's vector figures, rasterised from the flight repo when it is present.
briefing = os.path.expanduser("~/dev/ai-grand-prix/pq/briefing/figs")
for n in ["course_overlay", "camera_fov", "stress_results", "retrain_curves", "bug_ab", "planb_tradeoff"]:
    src = os.path.join(briefing, n + ".pdf")
    if not os.path.exists(src):
        print("keeping committed", n + ".png (briefing PDF not found)"); continue
    try:
        import fitz  # PyMuPDF
        fitz.open(src)[0].get_pixmap(dpi=200).save(os.path.join(OUT, n + ".png")); print("wrote", n + ".png")
    except ImportError:
        print("keeping committed", n + ".png (PyMuPDF not installed)")

# 10. Video poster frames (extracted with OpenCV in a venv; see README) ----------------
try:
    posters = [("poster_race_start_from_pad.jpg", "(a) race_start_from_pad.mp4: the 40 Hz racing policy lifts off the competition pad and takes gate 1"),
               ("poster_race_best.jpg", "(b) race_best.mp4: the best racing checkpoint, chase camera"),
               ("poster_stack_run_02.jpg", "(c) stack_run_02.mp4: the classical stack, chase view (left) and its own camera with detections (right)")]
    imgs = [load(os.path.join(OUT, p), 1280) for p, _ in posters]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 3.9 + 3.9 + 2.2))
    for ax, im, (_, tt) in zip(axs, imgs, posters):
        ax.imshow(im); ax.set_title(tt, fontsize=8, loc="left"); ax.axis("off")
    fig.tight_layout(pad=0.4); save(fig, "video_posters.png", vector=False)
except FileNotFoundError as e:
    print("skipping video posters:", e)
print("done")
