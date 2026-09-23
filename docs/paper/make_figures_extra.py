"""Five more figures, each built from something the team made rather than from a table of numbers.

  system_diagram.png      the whole chain from photons to motors, with every measured rate on its arrow
  project_timeline.png    four months of eras, then the on-site week with its incidents
  autolabel_funnel.png    the geometric labeller's 2467 candidates, where they went, and accuracy against gate size
  stress_causes.png       how episodes ended in each of the 16 Sep stress scenarios
  observation_contract.png  the 55 x 32 vector the policy sees, coloured by where each channel comes from

Run after make_figures.py (same style helpers, same output directory).
"""
import os, sys, json, glob, csv, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, INK, INK2, INK3, GRID, PALE,  # noqa: E402
                          headline, note, save, OUT)

FLIGHT = os.path.expanduser("~/dev/ai-grand-prix")

# ------------------------------------------------------------------ A. System diagram
def box(ax, x, y, w, h, title, body, fc, ec=None, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=7.6, fontweight="semibold", color=tc, zorder=4, linespacing=1.1)
    ax.text(x + w / 2, y + 0.10, body, ha="center", va="bottom", fontsize=5.9, color=tc, zorder=4, linespacing=1.3)

def arrow(ax, p, q, label="", color=INK2, above=True, rad=0.0, lw=1.2, fs=6.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, zorder=5, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx, my + (0.14 if above else -0.14), label, ha="center", va="bottom" if above else "top", fontsize=fs, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

fig, ax = plt.subplots(figsize=(9.2, 5.2)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, 10); ax.set_ylim(0, 6.1); ax.axis("off")
J, F, R = "#e4eefb", "#fbe6dc", "#e8f6ef"
ax.add_patch(Rectangle((0.15, 2.9), 6.75, 3.05, fc=PALE, ec="none", zorder=1)); ax.text(0.25, 5.88, "on the Jetson Orin NX: Python, NumPy, OpenCV, no deep-learning framework", fontsize=6.8, color=INK2, va="top")
ax.add_patch(Rectangle((7.0, 2.9), 2.85, 3.05, fc="#faece6", ec="none", zorder=1)); ax.text(7.1, 5.85, "on the flight controller: Betaflight 4.4.3", fontsize=6.8, color=INK2, va="top")
ax.add_patch(Rectangle((0.15, 0.2), 9.7, 2.45, fc="#eef7f2", ec="none", zorder=1)); ax.text(0.25, 2.55, "the radio pilot, who holds the only abort", fontsize=6.8, color=INK2, va="top")
bh, y0 = 1.55, 3.85
def B(x, w, *a, **k): box(ax, x, y0, w, bh, *a, **k); return x + w
e = B(0.30, 1.55, "camera", "IMX477, 20 deg up-tilt\n1920x1080 at 30 fps\nHFOV 74 deg, fx 425 px", J)
e = B(2.05, 1.60, "gate detector", "YOLOv8n-pose hand497\n640x360, 27 ms on GPU\n8 corners, kp conf 0.25", J)
e = B(3.85, 1.70, "observation", "55 ch x 32 frames = 1760\nlatched 30 Hz, delayed 1-4\ntracker picks the next gate", J)
e = B(5.70, 1.05, "policy", "NumPy MLP\n3 x 256, 40 Hz\nthrust + 3 rates", J)
e = B(7.10, 1.30, "MSP override", "mask 15 on AUX5\nno timeout on\nthe last frame", F)
e = B(8.55, 1.20, "PID loop,\nmotors", "440 deg/s at\nfull stick\nDShot300 x4", F)
arrow(ax, (1.85, y0 + 0.85), (2.05, y0 + 0.85)); ax.text(1.95, y0 + 1.62, "30 fps", ha="center", fontsize=6.2, color=INK2)
arrow(ax, (3.65, y0 + 0.85), (3.85, y0 + 0.85)); ax.text(3.75, y0 + 1.62, "~28 fps, 47 ms packet age", ha="center", fontsize=6.2, color=INK2)
arrow(ax, (5.55, y0 + 0.85), (5.70, y0 + 0.85))
arrow(ax, (6.75, y0 + 0.85), (7.10, y0 + 0.85), color=ORANGE, lw=1.6); ax.text(6.92, y0 + 1.62, "4 RC channels at 40 Hz", ha="center", fontsize=6.2, color=ORANGE)
arrow(ax, (8.40, y0 + 0.85), (8.55, y0 + 0.85))
ax.add_patch(FancyArrowPatch((7.10, y0 + 0.2), (4.7, y0 + 0.2), arrowstyle="-|>", mutation_scale=11, color=BLUE, lw=1.2, zorder=5, shrinkA=2, shrinkB=2))
ax.text(2.2, y0 - 0.12, "telemetry back: attitude, raw gyro, status and RC in one MSP_MULTIPLE_MSP request,\np95 20.3 ms with the stream running, on the same 115200-baud UART", ha="left", va="top", fontsize=6.0, color=BLUE)
ax.text(9.75, y0 - 0.62, "adapter on the command path: inverts the rate and throttle curves, pitch sign -1, gyro counts / 16.384", ha="right", va="top", fontsize=6.0, color=ORANGE)
box(ax, 0.35, 0.5, 2.0, 1.45, "transmitter", "AUX1 = ARM (radio only)\nAUX5 down = policy flies\nAUX2 = ANGLE", R)
arrow(ax, (2.35, 1.5), (7.75, 2.9), "arming, and the override switch: the abort path", color=GREEN, rad=-0.12, fs=6.2)
box(ax, 3.6, 0.5, 2.7, 1.45, "props-off handover check", "streams neutral sticks first (the buffer\nstarts at zero, which reads as failsafe),\nthen proves AUX5 off returns control", R)
box(ax, 7.0, 0.5, 2.7, 1.45, "the missing sensor", "no usable height: the barometer collapses\nunder prop wash, so height comes\nfrom the gate in view", "#fdecec", tc=RED)
headline(fig, "The chain from photons to motors, with what was measured on each link",
         "blue = telemetry back to the observation; orange = the command path where the sign, scale and curve faults of Section 6 lived")
save(fig, "system_diagram.png")

# ------------------------------------------------------------------ B. Timeline
def d(s): return dt.datetime.strptime(s, "%Y-%m-%d")
eras = [("2026-06-03", "2026-06-28", "Virtual qualifier 1\nHSV mask, PnP,\nwaypoint replay", BLUE),
        ("2026-06-28", "2026-08-04", "Virtual qualifier 2\nYOLO corners + IMU, dual-gate EKF,\nDreamerV3", AQUA),
        ("2026-08-04", "2026-08-30", "Classical stack,\nsnake gate,\nHG-DAgger", YELLOW),
        ("2026-08-30", "2026-09-16", "Isaac Lab PPO on\na digitised course", ORANGE),
        ("2026-09-16", "2026-09-23", "on\nsite", RED)]
events = [("2026-06-06", "first gate passed"), ("2026-06-20", "replay laps the course in <14 s\n(on simulator position)"), ("2026-07-25", "DreamerV3: 168k steps, 0 gates"),
          ("2026-07-27", "PnP + IMU passes gates 1-2"), ("2026-08-16", "human best 14.04 s"), ("2026-09-05", "pq_speed_best: 59 gates / 40 s"),
          ("2026-09-16", "audit: camera, track, reset bug;\nstress study")]
import textwrap
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 7.0), gridspec_kw={"height_ratios": [1.0, 2.4], "hspace": 0.5})
for s, e, lab, c in eras:
    a1.axvspan(d(s), d(e), ymin=0.55, ymax=0.95, color=c, alpha=0.22, lw=0)
    a1.text(d(s) + (d(e) - d(s)) / 2, 0.75, lab, ha="center", va="center", fontsize=6.0, color=INK)
for i, (s, lab) in enumerate(events):
    y = 0.40 - 0.11 * (i % 3)
    a1.plot([d(s), d(s)], [0.55, y + 0.03], color=INK3, lw=0.6); a1.plot(d(s), 0.55, "o", color=INK, ms=3)
    a1.text(d(s), y, lab, fontsize=6.2, color=INK2, ha="center", va="top")
a1.set_xlim(d("2026-06-01"), d("2026-09-24")); a1.set_ylim(-0.05, 1.0); a1.set_yticks([]); a1.grid(False)
a1.xaxis.set_major_locator(matplotlib.dates.MonthLocator()); a1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
for sp in ("left", "top", "right"): a1.spines[sp].set_visible(False)
a1.set_title("June to September: three generations of approach on the virtual qualifiers, then a fourth for the hall", loc="left", fontsize=8.5)
# on-site week
days = ["15 Sep", "16 Sep", "17 Sep", "18 Sep", "19 Sep", "20 Sep", "21 Sep", "22 Sep"]
good = {2: ["arrive; aircraft weighed 1745 g", "organizers' MSP library found"], 3: ["camera capture works headless"],
        4: ["auto-labeller; 1207 gate frames", "Plan B chosen over the policy"], 5: ["hand434 deployed, threshold 0.25", "rate-map audit: 0.62x to 2.40x"],
        6: ["40 Hz link, gyro counts, pitch sign measured", "40 Hz retrains; race40drop 10.2 gates/ep", "hand497 measured better"], 7: ["scored day"]}
bad = {3: ["Betaflight console wedges the FC, twice", "'ANGLE' switch kills the aircraft (zero buffer)", "tip-overs in ACRO; airframe replaced"],
       4: ["FC silent again; recovered by hand"], 5: ["Betaflight reset; battery at 2.55 V/cell", "first policy flight: 0.45 s, into a wall"],
       6: ["FC hangs with two MSP clients attached; armed on power-up twice", "fallback stack's first flight hits the ceiling", "replacement Jetson; RL shelved"],
       7: ["no autonomous run; all four aircraft grounded"]}
for i, day in enumerate(days):
    a2.text(i, 1.02, day, ha="center", va="bottom", fontsize=7.5, fontweight="semibold", color=INK)
    yy = 0.95
    for t in good.get(i, []):
        t = textwrap.fill(t, 17); a2.text(i, yy, t, ha="center", va="top", fontsize=5.4, color=BLUE, linespacing=1.15); yy -= 0.062 * (t.count("\n") + 1) + 0.045
    yy = 0.22
    for t in bad.get(i, []):
        t = textwrap.fill(t, 15); a2.text(i, yy, t, ha="center", va="top", fontsize=5.4, color=RED, linespacing=1.15); yy -= 0.062 * (t.count("\n") + 1) + 0.045
a2.axhline(0.30, color=GRID, lw=0.8); a2.text(-0.6, 0.95, "progress", fontsize=7, color=BLUE, rotation=90, va="top", ha="center"); a2.text(-0.6, 0.22, "setbacks", fontsize=7, color=RED, rotation=90, va="top", ha="center")
a2.set_xlim(-0.75, 7.5); a2.set_ylim(-0.7, 1.15); a2.axis("off")
a2.set_title("The week in the hall: what moved forward, and what the hardware did", loc="left", fontsize=8.5)
headline(fig, "Four months, one week, no scored run", "top: eras with the milestone each one reached; bottom: the physical qualifier day by day, from the commit history and session notes", y=0.98)
save(fig, "project_timeline.png")

# ------------------------------------------------------------------ C. Auto-label funnel
rows = list(csv.DictReader(open(os.path.join(FLIGHT, "datasets/autolabel/report.csv"))))
hyb = list(csv.DictReader(open(os.path.join(FLIGHT, "datasets/hybrid/report.csv"))))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
from collections import Counter
v = Counter(r["verdict"] for r in rows); src = Counter(h["source"] for h in hyb)
stages = [("gate candidates\nfrom the colour mask", [(len(rows), INK3, "")]),
          ("geometry alone", [(v["auto"], BLUE, "accepted"), (v["review"], YELLOW, "to review"), (v["rejected"], INK3, "rejected"), (v["quarantined"], RED, "quarantined")]),
          ("hybrid: detector proposes,\ngeometry disposes", [(src["both"], BLUE, "both rings"), (src["outer"] + src["inner"], "#8fb8ea", "one ring"), (src["network"], AQUA, "network")])]
for i, (lab, parts) in enumerate(stages):
    y = 2 - i; x = 0
    for n, c, t in parts:
        a1.barh(y, n, left=x, color=c, height=0.55, linewidth=0)
        if n > 400: a1.text(x + n / 2, y, f"{t}\n{n}" if t else f"{n}", ha="center", va="center", fontsize=6.2, color="white" if c not in (YELLOW, "#8fb8ea") else INK)
        x += n
    a1.text(x + 40, y, f"{x}" + ("\nrejected 265,\nquarantined 21" if i == 1 else ""), va="center", fontsize=6.6, color=INK2)
a1.set_yticks([2, 1, 0]); a1.set_yticklabels([s for s, _ in stages], fontsize=7.2); a1.set_xlabel("gate instances, 1207 frames"); a1.grid(axis="y", visible=False); a1.set_xlim(0, 2800)
a1.set_title("Where the candidates went", loc="left", fontsize=8.5)
side = np.array([float(r["outer_side_px"]) if r["outer_side_px"] not in ("", "inf", "nan") else np.nan for r in rows])
al = np.array([float(r["align_px"]) if r["align_px"] not in ("", "inf", "nan") else np.nan for r in rows])
verd = np.array([r["verdict"] for r in rows])
for name, c, z, a in [("rejected", INK3, 2, 0.35), ("review", YELLOW, 3, 0.5), ("auto", BLUE, 4, 0.55)]:
    m = (verd == name) & np.isfinite(side) & np.isfinite(al)
    a2.scatter(side[m], al[m], s=7, color=c, alpha=a, lw=0, zorder=z, label=f"{name} ({m.sum()})")
a2.set_xscale("log"); a2.set_xlim(30, 2000); a2.set_ylim(0, 9)
a2.axhline(4.0, color=INK2, lw=0.9, ls="--"); a2.text(1950, 3.9, "accept cap 4.0 px", fontsize=6.8, color=INK2, va="top", ha="right")
a2.axhline(4.36, color=RED, lw=0.9, ls=":"); a2.text(32, 4.46, "4.36 px: median error against human labels", fontsize=6.8, color=RED, va="bottom", ha="left")
m = (verd == "auto") & np.isfinite(al); a2.axhline(np.nanmedian(al[m]), color=BLUE, lw=0.9, ls=":"); a2.text(32, np.nanmedian(al[m]) + 0.1, f"accepted labels' own median {np.nanmedian(al[m]):.2f} px (self-scored)", fontsize=6.8, color=BLUE, va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
a2.set_xlabel("gate size in the image (outer side, px, log)"); a2.set_ylabel("edge alignment residual (px)"); a2.legend(fontsize=6.8, loc="upper left", markerscale=2)
a2.set_title("Self-scored accuracy against gate size", loc="left", fontsize=8.5)
headline(fig, "The geometric auto-labeller: yield, and the number that turned out to be optimistic",
         "1207 walk-around frames; the residual is the corners' distance from the image's own colour edge, which is what the labeller solves for, so it cannot see its own systematic error")
save(fig, "autolabel_funnel.png")

# ------------------------------------------------------------------ D. Stress causes
files = sorted(glob.glob(os.path.join(FLIGHT, "analysis/2026-09-16/experiments/results/stress/*.json")))
recs = [json.load(open(f)) for f in files]
recs = [r for r in recs if "cause_counts" in r]
order = ["collision", "gate_missed", "flyaway", "time_out"]
cols = {"collision": RED, "gate_missed": ORANGE, "flyaway": VIOLET, "time_out": AQUA}
labs = {"collision": "crashed into something", "gate_missed": "flew past a gate", "flyaway": "flew away", "time_out": "still flying at the cut"}
fig, ax = plt.subplots(figsize=(7.4, 5.2))
names = [r["name"] for r in recs]; ys = np.arange(len(recs))[::-1]
for r, y in zip(recs, ys):
    cc = r["cause_counts"]; tot = sum(cc.values()) or 1; x = 0
    for k in order:
        f = cc.get(k, 0) / tot
        ax.barh(y, f, left=x, color=cols[k], height=0.72, linewidth=0)
        x += f
    ax.text(1.01, y, f"{r['crash_per_100_gates']:.0f}", va="center", fontsize=6.6, color=INK2)
ax.text(1.01, len(recs) - 0.35, "crashes per\n100 gates", fontsize=6.4, color=INK2, va="bottom")
ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=6.4); ax.set_xlim(0, 1.0); ax.set_xlabel("share of episode endings"); ax.grid(axis="y", visible=False)
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=cols[k]) for k in order], labels=[labs[k] for k in order], fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4)
headline(fig, "How episodes ended for pq_speed_best in each stress scenario, 16 Sep",
         "512 simulated aircraft per scenario, one perturbation at a time; the gate-count bug fixed for every run. R = reference, A = expected at the event, B = rougher, C = stress only")
save(fig, "stress_causes.png")

# ------------------------------------------------------------------ E. Observation contract
groups = [("corner u,v x8", 16, BLUE, "detector, through a 30 Hz latch,\na 1-4 step delay and sticky dropout"),
          ("visible x8", 8, "#8fb8ea", "-1 sentinel when unseen"),
          ("roll, pitch", 2, ORANGE, "flight controller attitude\n(nose-down positive on this FC)"),
          ("gyro x3", 3, "#f6a07a", "raw counts / 16.384, clipped +-8 rad/s"),
          ("velocity x3", 3, AQUA, "dead-reckoned from issued thrust,\nattitude and a drag model; never the accelerometer"),
          ("next gate one-hot x18", 18, YELLOW, "from the gate tracker;\nnothing on the course publishes it"),
          ("lap", 1, "#c9a300", ""),
          ("last actions x4 (v2)", 4, VIOLET, "the issued command, not the delayed one")]
n = sum(g[1] for g in groups); H = 32
img = np.zeros((H, n)); c0 = 0; bounds = []
for gi, (_, w, _, _) in enumerate(groups):
    img[:, c0:c0 + w] = gi; bounds.append((c0, w)); c0 += w
fig, ax = plt.subplots(figsize=(7.4, 5.6))
cmap = ListedColormap([g[2] for g in groups])
ax.imshow(img, cmap=cmap, aspect="auto", interpolation="nearest", alpha=0.9, extent=(0, n, H, 0))
for k in range(1, H): ax.axhline(k, color="white", lw=0.4)
for c, w in bounds: ax.axvline(c, color="white", lw=1.2)
ax.add_patch(Rectangle((0, H - 1), n, 1, fc="none", ec=INK, lw=1.6, zorder=5)); ax.text(n + 0.4, H - 0.5, "newest frame: the one the runner\nbuilds this step", fontsize=6.8, color=INK, va="center")
ax.text(n + 0.4, 0.5, "oldest of 32 frames\n= 0.8 s of history at 40 Hz", fontsize=6.8, color=INK2, va="center")
ax.set_xlim(0, n + 14); ax.set_ylim(H, -0.5); ax.set_yticks([]); ax.set_xticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.grid(False)
for (lab, w, c, src), (c0, _) in zip(groups, bounds):
    ax.text(c0 + w / 2, -0.6, lab, ha="center", va="bottom", fontsize=6.4, color=INK, rotation=0 if w > 6 else 90)
ax.set_ylim(H + 15.5, -3)
row = 0
for lab, w, c, src in groups:
    if not src: continue
    x = 0.5; y = H + 1.8 + row * 1.7
    ax.add_patch(Rectangle((x, y - 0.5), 1.4, 1.1, fc=c, ec="none", clip_on=False))
    ax.text(x + 2.0, y, f"{lab.split(' x')[0]}: " + src.replace("\n", " "), fontsize=6.3, color=INK2, va="center", ha="left", clip_on=False)
    row += 1
ax.text(0.5, H + 14.6, "51 channels (v1, 1632 inputs) + 4 (v2, 1760). Every value both ends must agree on is hashed: v2 = a20c14d6a335.\nA wrong channel order raises nothing; it flies.", ha="left", va="center", fontsize=6.4, color=INK)
headline(fig, "What the policy sees: the observation contract, one row per frame, one column per channel",
         "identical by construction between the torch builder in the simulator and the NumPy builder on the aircraft; colour = where the channel comes from")
save(fig, "observation_contract.png")
print("done")
