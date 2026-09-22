"""Figures for the paper, from the numbers recorded in the team's documents and the logs on disk."""
import os, sys, glob, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import tfevents

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                 "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                 "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})

def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p); plt.close(fig); print("wrote", p)

def bars(ax, cats, series, colors, labels, width=0.36, fmt="{:.0f}", ylim=None, unit=""):
    x = np.arange(len(cats)); n = len(series)
    for k, (vals, c, lab) in enumerate(zip(series, colors, labels)):
        xs = x + (k - (n - 1) / 2) * (width + 0.02)
        b = ax.bar(xs, vals, width, color=c, label=lab, linewidth=0)
        for xi, v in zip(xs, vals):
            ax.text(xi, v, fmt.format(v) + unit, ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)

# 1. Hover stress sweep --------------------------------------------------------
conds = ["nominal", "rate tau\n60 ms", "mass\nx0.90", "wind\n0.6 N", "mass\nx1.15", "delay\n4 steps", "hover\n0.19", "hover\n0.31"]
surv = [95.3, 94.1, 95.7, 89.8, 77.2, 70.1, 76.4, 55.7]
sett = [75.8, 73.9, 73.2, 36.5, 20.4, 13.0, 7.0, 1.9]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
bars(ax, conds, [surv, sett], [BLUE, ORANGE], ["survived the 15 s episode", "settled (within 0.30 m and 0.30 m/s)"], ylim=(0, 108), unit="%")
ax.set_ylabel("% of 4096 episodes"); ax.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.2))
ax.set_title("Hover policy under eight perturbations (best 60 Hz checkpoint, stress_hover.py, 20 Sep)", fontsize=9, loc="left")
save(fig, "hover_stress_sweep.png")

# 2. 40 Hz vs 60 Hz ------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [3, 2]})
cats = ["60 Hz policy\nrun at 60 Hz", "60 Hz policy\nrun at 40 Hz", "40 Hz retrain\nrun at 40 Hz", "40 Hz retrain\nrun at 60 Hz"]
vals = [87.0, 52.6, 85.5, 89.0]
cols = [BLUE, BLUE, ORANGE, ORANGE]
b = a1.bar(range(4), vals, 0.6, color=cols, linewidth=0)
for i, v in enumerate(vals): a1.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
a1.set_xticks(range(4)); a1.set_xticklabels(cats, fontsize=7.5); a1.set_ylim(0, 100); a1.set_ylabel("hover: % settled"); a1.grid(axis="x", visible=False)
a1.set_title("Hover: the control rate the policy trained at matters", fontsize=9, loc="left")
vals2 = [8.246, 3.086]
b2 = a2.bar([0, 1], vals2, 0.55, color=[BLUE, BLUE], linewidth=0)
for i, v in enumerate(vals2): a2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
a2.set_xticks([0, 1]); a2.set_xticklabels(["60 Hz policy\nat 60 Hz", "60 Hz policy\nat 40 Hz"], fontsize=7.5); a2.set_ylabel("racing: gates in 15 s"); a2.grid(axis="x", visible=False)
a2.set_title("Racing keeps 37% of its gates", fontsize=9, loc="left")
save(fig, "rate_40_vs_60.png")

# 3. MSP link timing -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.8))
streams = ["no RC stream\n(read-only)", "RC stream 40 Hz", "RC stream 60 Hz"]
batched_mean = [10.03, 16.75, 25.10]; batched_p95 = [10.27, 20.28, 30.30]; fallback = [40.09, 66.90, 100.43]
x = np.arange(3); w = 0.26
ax.bar(x - w, batched_mean, w, color=BLUE, label="MSP_MULTIPLE_MSP, mean", linewidth=0)
ax.bar(x, batched_p95, w, color="#8fb8ea", label="MSP_MULTIPLE_MSP, p95", linewidth=0)
ax.bar(x + w, fallback, w, color=ORANGE, label="four separate requests, mean", linewidth=0)
for xi, v in zip(x - w, batched_mean): ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
for xi, v in zip(x, batched_p95): ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
for xi, v in zip(x + w, fallback): ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
ax.axhline(25.0, color=INK2, lw=0.8, ls="--"); ax.text(-0.42, 25.8, "40 Hz budget 25 ms", fontsize=7, color=INK2, ha="left")
ax.axhline(16.67, color=INK2, lw=0.8, ls=":"); ax.text(0.55, 12.5, "60 Hz budget 16.7 ms", fontsize=7, color=INK2, ha="left")
ax.set_xticks(x); ax.set_xticklabels(streams); ax.set_ylabel("telemetry snapshot round trip (ms)"); ax.set_ylim(0, 112); ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 0.98)); ax.set_title("One 115200-baud UART carries both telemetry and the stick stream (300 samples per condition, 21 Sep)", fontsize=9, loc="left")
save(fig, "msp_link_timing.png")

# 4. Rate map --------------------------------------------------------------------
rc_rate, super_rate = 0.55, 0.75
stick = np.linspace(0, 1, 200)
delivered = 200 * rc_rate * stick / (1 - super_rate * stick)      # deg/s, Betaflight rates, expo 0
asked = stick * 3.2 * 180 / np.pi                                  # the linear map sends stick = rate / 3.2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
a1.plot(asked, asked, color=INK2, lw=1.2, ls="--", label="what the policy assumes (linear)")
a1.plot(asked, delivered, color=ORANGE, lw=2, label="what Betaflight delivers (rc_rate 55, super 75)")
a1.set_xlabel("body rate the policy asked for (deg/s)"); a1.set_ylabel("rate the PID loop is set to (deg/s)")
a1.set_xlim(0, 184); a1.set_ylim(0, 450); a1.legend(fontsize=7.5, loc="upper left")
ratio = delivered[1:] / asked[1:]
a2.plot(asked[1:], ratio, color=ORANGE, lw=2); a2.axhline(1.0, color=INK2, lw=0.8, ls="--")
for r, e in [(0.2, 0.62), (1.0, 0.78), (1.6, 0.96), (2.4, 1.37), (3.2, 2.40)]:
    a2.plot([r * 180 / np.pi], [e], "o", color=ORANGE, ms=4); a2.text(r * 180 / np.pi, e + 0.06, f"{e:.2f}x", fontsize=7, ha="center", color=INK2)
a2.set_xlabel("body rate the policy asked for (deg/s)"); a2.set_ylabel("delivered / asked"); a2.set_xlim(0, 184); a2.set_ylim(0.4, 2.6)
a1.set_title("Linear stick map vs the FC's rate curve", fontsize=9, loc="left"); a2.set_title("Loop gain crosses unity mid-range", fontsize=9, loc="left"); fig.tight_layout(w_pad=2.5)
save(fig, "rate_map_error.png")

# 5. Barometer collapse ----------------------------------------------------------
rows = list(csv.DictReader(open(os.path.expanduser("~/dev/ai-grand-prix/pq/logs/orin/hover_20260918_003956.csv"))))
t = np.array([float(r["t_s"]) for r in rows]); alt = np.array([float(r["altitude_m"]) for r in rows]); thr = np.array([float(r["throttle_us"]) for r in rows])
gz = np.array([float(r["gz_dps"]) for r in rows]); gx = np.array([float(r["gx_dps"]) for r in rows]); gy = np.array([float(r["gy_dps"]) for r in rows])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw={"hspace": 0.12})
a1.plot(t, thr, color=BLUE, lw=1.2); a1.set_ylabel("throttle stick (us)")
a1.set_title("Aircraft armed on the ground, 18 Sep: the barometer reads a descent whenever the props spin", fontsize=9, loc="left")
a2.plot(t, alt, color=ORANGE, lw=1.2); a2.set_ylabel("baro altitude (m)"); a2.set_xlabel("time (s)")
save(fig, "baro_collapse.png")
print("gyro raw peak", np.abs(np.c_[gx, gy, gz]).max())

# 6. Perception numbers ----------------------------------------------------------
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
bars(a1, ["box\nmAP50-95", "pose\nmAP50-95"], [[0.283, 0.760], [0.636, 0.854]], [INK2, BLUE], ["teammate's model", "fine-tuned on hybrid labels"], fmt="{:.3f}", ylim=(0, 1.05))
a1.legend(fontsize=7, loc="upper left"); a1.set_title("Held-out contiguous blocks", fontsize=9, loc="left")
bars(a2, ["random\nsplit", "contiguous\nblocks"], [[0.517, 0.283]], [INK2], ["same model, box mAP50-95"], fmt="{:.3f}", ylim=(0, 0.65), width=0.5)
a2.set_title("Neighbouring frames leak", fontsize=9, loc="left")
th = [0.10, 0.25, 0.40, 0.50]; usable = [89, 67, 60, 53]; ready = [79, 59, 50, 36]
a3.plot(th, usable, "-o", color=BLUE, ms=4, label="frames with 4+ corners"); a3.plot(th, ready, "-o", color=ORANGE, ms=4, label="policy ready (6 in a row)")
a3.axvline(0.5, color=INK2, lw=0.8, ls=":"); a3.text(0.5, 92, "library\ndefault", fontsize=7, ha="center", color=INK2)
a3.axvline(0.25, color=INK2, lw=0.8, ls="--"); a3.text(0.25, 92, "chosen", fontsize=7, ha="center", color=INK2)
a3.set_xlabel("keypoint confidence threshold"); a3.set_ylabel("% of 1345 frames at a real gate"); a3.set_ylim(0, 100); a3.legend(fontsize=7, loc="lower left")
a3.set_title("An inherited threshold cost the data", fontsize=9, loc="left")
fig.tight_layout(w_pad=1.5)
save(fig, "perception_numbers.png")

# 7. Close-gate detector comparison ----------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.4))
cats = ["hybrid alone", "teammate alone", "hybrid, else teammate"]
bars(ax, cats, [[72, 100, 100], [41, 16, 46]], [BLUE, ORANGE], ["gate detected", "detection became a pose (PnP solved)"], ylim=(0, 115), unit="%")
ax.set_ylabel("% of 548 close-gate frames"); ax.legend(fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
ax.set_title("Close gates (>30% of the frame) are the ones flown through, and the two models fail in opposite directions", fontsize=9, loc="left")
save(fig, "close_gate_models.png")

# 8. Training history from tfevents ----------------------------------------------
base = os.path.expanduser("~/dev/ai-grand-prix/isaac_drone_racer/logs/skrl/drone_racer")
runs = {"2026-09-04_17-03-36_ppo_torch": ("4 Sep 17:03  PQ track from scratch (pq_scratch)", BLUE),
        "2026-09-04_20-38-19_ppo_torch": ("4 Sep 20:38  + velocity alignment (pq_vel)", AQUA),
        "2026-09-04_23-48-28_ppo_torch": ("4 Sep 23:48  + run-in progress (pq_hook)", YELLOW),
        "2026-09-05_01-08-00_ppo_torch": ("5 Sep 01:08  + speed through gate (pq_speed_best)", ORANGE),
        "2026-09-06_01-35-30_ppo_torch": ("6 Sep 01:35  identical settings, continued: the gate-count reset bug", RED)}
fig, a1 = plt.subplots(figsize=(7.2, 3.4))
for d, (lab, c) in runs.items():
    ev = glob.glob(os.path.join(base, d, "events.out.tfevents.*"))
    if not ev: continue
    r = tfevents.read(ev[0]); s_ = r.get("Info / Episode_Reward/gate_passed", [])
    if not s_: continue
    st, v = zip(*s_); a1.plot(st, v, color=c, lw=1.4, label=lab)
a1.set_xlabel("PPO timesteps (4096 environments each)"); a1.set_ylabel("gate_passed reward term\n(per-step mean, exploration noise on)")
a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
a1.set_title("The lineage that produced pq_speed_best on the team's PQ track, from the surviving TensorBoard logs", fontsize=9, loc="left")
save(fig, "training_history.png")

# 9. Image composites -------------------------------------------------------------
def load(p, w=None):
    im = Image.open(os.path.expanduser(p)).convert("RGB")
    if w: im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    return im
def panel(imgs, titles, name, ncol, figw=7.2):
    n = len(imgs); nrow = int(np.ceil(n / ncol))
    ratios = [im.height / im.width for im in imgs]
    fig, axs = plt.subplots(nrow, ncol, figsize=(figw, figw / ncol * max(ratios) * nrow + 0.3 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax, im, tt in zip(axs, imgs, titles):
        ax.imshow(im); ax.set_title(tt, fontsize=8, loc="left"); ax.axis("off")
    for ax in axs[n:]: ax.axis("off")
    fig.tight_layout(pad=0.4); save(fig, name)

panel([load("~/Downloads/gate frames/0919_220524_000107.jpg", 960), load("~/dev/aigp-runs/drone_frames/0920_224905_f00150.jpg", 960),
       load("~/dev/ai-grand-prix/datasets/autolabel/overlays/0919_214639_000000.jpg", 960), load("~/dev/ai-grand-prix/datasets/hybrid/preview/img/0919_214639_000004.jpg", 960)],
      ["(a) onboard camera, hand-carried walk-around, 19 Sep", "(b) onboard camera at a close gate, 20 Sep: corners leave the frame",
       "(c) geometric auto-label: eight numbered corners", "(d) hybrid label as written for YOLO-pose training"], "what_the_drone_sees.png", 2)
panel([load("~/dev/aigp-runs/gallery/sample/0919_214639_000160.jpg", 1400), load("~/dev/aigp-runs/gallery/worst_DISAGREEMENT_not_a_fair_sample/0919_214639_000366.jpg", 1400)],
      ["(a) a typical frame: hand-labelled model (left) and the teammate's model (right) agree", "(b) a selected worst case: the incumbent stacks seven boxes on two gates"], "detector_ab_gallery.png", 1)
panel([load("~/Downloads/stack_sim_videos/paths_12_drones.png", 1600)],
      ["Twelve simulated attempts of the perception stack from the pad (plan view): blue = true path, dotted = the stack's own position estimate, X = crash"], "stack_sim_paths.png", 1)
cs = load("~/dev/aigp-runs/videos/flight_contact_sheet.png", 1400); cs.save(os.path.join(OUT, "race_contact_sheet.jpg"), quality=80); print("wrote race_contact_sheet.jpg")
Image.open(os.path.expanduser("~/dev/ai-grand-prix/pq/eval/pq_speed_best_play.png")).convert("RGB").save(os.path.join(OUT, "pq_speed_best_play.png"))
for n in ["course_overlay", "camera_fov", "stress_results", "retrain_curves", "bug_ab", "planb_tradeoff"]:
    Image.open(os.path.join(HERE, "raw", n + ".png")).convert("RGB").save(os.path.join(OUT, n + ".png"))
print("done")
