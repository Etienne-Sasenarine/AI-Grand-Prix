# The paper

`paper.md` is the write-up of the whole AI Grand Prix physical-qualifier effort:
the perception pipeline, the simulator and training stack, the measurements
taken on the aircraft, and the sequence of failures that ended the attempt.
`paper.pdf` is the same document rendered for reading offline.

## Rebuilding it

    python3 make_figures.py     # regenerates figures/ from the numbers and logs cited in the paper
    python3 build_pdf.py        # paper.md -> paper.html -> paper.pdf (needs the `markdown` package and Google Chrome)

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
