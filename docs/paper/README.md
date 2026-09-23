# The paper

`paper.md` is the write-up of the whole AI Grand Prix physical-qualifier effort:
the perception pipeline, the simulator and training stack, the measurements
taken on the aircraft, and the sequence of failures that ended the attempt.
It is the one source; two renderings are kept beside it:

* `paper.pdf` — the Markdown rendered as-is (headless Chrome).
* `neurips/paper.pdf` — the same text typeset in the official NeurIPS style
  (`neurips/neurips_2025.sty`, `final` option), produced by `md2tex.py` and
  `pdflatex`. The 2026 style file was not published at the time of writing.

`videos/` holds the five recordings the paper links to (63 MB, plain files,
not LFS):

| file | what it shows |
|---|---|
| `race_start_from_pad.mp4` | the 40 Hz racing policy taking off from the competition pad and passing gate 1 (Isaac, 6.7 s) |
| `race_best.mp4` | the best racing checkpoint from a chase camera (Isaac, 9.9 s) |
| `stack_run_01..03.mp4` | the classical fallback stack in Isaac: chase view on the left, the drone's own camera with the simulated detector's corners on the right (60 s each) |

## Rebuilding it

    python3 make_figures.py     # regenerates figures/ from the numbers and logs cited in the paper
    python3 build_pdf.py        # paper.md -> paper.html -> paper.pdf (needs the `markdown` package and Google Chrome)
    python3 md2tex.py && (cd neurips && pdflatex paper.tex && pdflatex paper.tex)   # the NeurIPS-style PDF

`make_figures.py` reads the team's recorded numbers (session notes, bench
findings, checkpoint provenance), the 18 Sep telemetry CSV, the TensorBoard
event files of the pre-event training runs (via `tfevents.py`, a
dependency-free reader), the briefing's vector figures, and the onboard camera
captures and detector galleries. Those sources live in the flight repo and on
the team laptop, not here; the generated PNGs are committed so the paper
renders on GitHub without them.

## Figures

| file | what it is |
|---|---|
| `course_overlay.png`, `camera_fov.png`, `stress_results.png`, `retrain_curves.png`, `bug_ab.png`, `planb_tradeoff.png` | from the 17 Sep engineering journal (`pq/briefing/figs`), rasterised |
| `pq_speed_best_play.png` | the pre-event keeper policy in playback, 19 Sep |
| `training_history.png` | `gate_passed` reward over the 4–6 Sep run lineage, parsed from TensorBoard events |
| `hover_stress_sweep.png`, `rate_40_vs_60.png` | hover and racing numbers from the 20–21 Sep session notes |
| `msp_link_timing.png`, `rate_map_error.png` | bench measurements on the aircraft, 21 Sep (`pq/flight/FINDINGS_*.md`) |
| `baro_collapse.png` | `pq/logs/orin/hover_20260918_003956.csv` |
| `perception_numbers.png`, `close_gate_models.png` | numbers from `aigp-perception` READMEs and the 21 Sep live comparison |
| `what_the_drone_sees.png`, `detector_ab_gallery.png` | onboard camera frames, auto-label overlays and the model A/B gallery |
| `stack_sim_paths.png` | the classical fallback stack flown in Isaac, 21 Sep |
| `race_contact_sheet.jpg` | the racing policy's onboard camera in Isaac, 20 Sep |
| `video_posters.png`, `poster_*.jpg` | one frame from each recording in `videos/` (extracted with OpenCV) |
