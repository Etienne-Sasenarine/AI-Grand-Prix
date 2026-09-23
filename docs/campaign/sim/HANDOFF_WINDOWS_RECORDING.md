# Handoff: film the AI Grand Prix racing policies on this Windows laptop

## Goal

Produce side-by-side videos (Isaac chase view | the drone's onboard camera with detections drawn) of two trained racing policies, and copy them to my Mac. Everything needed is in the private repo https://github.com/bojro/aigp-sim (GitHub account: bojro). The one-script path is `scripts\windows\record_race_pov.ps1`. Read its header and `scripts\diag\record_race_pov.py` before running anything.

## Context you should know

- This laptop (RTX 4060, Windows with WSL) was the team's Isaac Sim machine on 16 Sep 2026 and has not been used for this project since. Expect stale state. Two Isaac installs may exist: a Windows-native venv, historically `D:\isaacsim_venv` with the repo copy at `D:\Code\Competitions\AIGP\isaac_drone_racer`, and a WSL one behind `~/aigp/isaac_env.sh` (Isaac Sim 4.5, Isaac Lab 2.1, skrl 1.4.2, Python 3.10). Use the **Windows-native** one: the recorder needs camera rendering, which is fragile under WSL2. Find it first (`Get-ChildItem D:\ -Directory`, look for `isaacsim_venv`, `python.exe` with `isaaclab` importable); do not reinstall Isaac unless it is genuinely gone.
- `aigp-sim` is a cleaned-up fork of the old `isaac_drone_racer` tree. Package names (`contract`, `tasks`, `utils`, `dynamics`, `assets`) are unchanged. `README.md` and `docs/` explain the layout; `docs/RUNBOOK_RUNPOD.md` has the Isaac setup sequence if something must be installed. Old copies of the repo on this machine are NOT to be used; clone fresh.
- The two checkpoints are committed in `checkpoints/` (race40_best_agent.pt, race40drop_best_agent.pt; both 40 Hz, v2 contract, 1760 inputs). `checkpoints/README.md` has their provenance. Run them with `AIGP_POLICY_HZ=40`; the ps1 sets this.
- The cyan overlay needs the sibling repo https://github.com/bojro/aigp-perception (private) and `onnxruntime-gpu`; the ps1 handles both. If onnxruntime-gpu will not install against this CUDA, fall back to `onnxruntime` (CPU) or pass `-NoDetector`.
- The recorder has never run on a live Isaac install (the pod it was written for is gone). Expect to fix small things: import paths, the tiled-camera key name, the skrl reset re-arm, imageio needing ffmpeg. Fix them in place, keep the fixes minimal, and commit them with a clear message. Do not rewrite the script.
- Isaac's exit code is meaningless. The `RECORDED=` line printed by the recorder, and `SMOKE_RESULT=PASS` from the smoke test, are the results.
- Do not touch anything to do with the drone, Betaflight, MSP, or the Code-Red-Cables/AI_GP repository. This task is simulation only.

## Steps

1. Authenticate to GitHub if needed (`gh auth login` or a credential helper); the repos are private.
2. `git lfs install`, then `git clone https://github.com/bojro/aigp-sim.git D:\aigp-sim` (or another drive with ~5 GB free; the drone USD is 98 MB in LFS).
3. Locate the Isaac venv python. Verify with `<python> -c "import isaaclab, isaaclab_tasks, skrl; print(skrl.__version__)"` (should print 1.4.x). If the venv is missing or broken, stop and report what you found before installing anything large.
4. `cd D:\aigp-sim` then `.\scripts\windows\record_race_pov.ps1 -Python <that python>`. Watch for `SMOKE_RESULT=PASS`, then two recording runs. Keep the laptop on AC power and awake; each run is a few minutes on this GPU.
5. If a run fails, read the traceback, make the minimal fix, rerun that run only (`-SkipSmoke` once the smoke test has passed). Commit fixes to aigp-sim main and push.
6. Look at one output video (open it) and confirm: two panes, left chase view, right onboard camera with green corners (and cyan hand497 detections if the detector ran), a status line, and the drone actually leaving the pad. A file that plays but shows a static scene means the skrl reset trap (see record_start.py's docstring).
7. Copy the videos to the Mac: `scp -r <out dir> bojro@Bojros-MacBook-Air-2.local:~/dev/aigp-sim/paper/videos/race_pov` (the Mac was at 10.48.99.15 on this network on 23 Sep; use the IP if mDNS fails). If scp is not available, zip them and say where they are.

## Report back

- Which python/venv was used and its Isaac Sim / Isaac Lab / skrl versions.
- The `SMOKE_RESULT=` line and every `RECORDED=` line, verbatim.
- Every fix you had to make, as commit hashes.
- Where the videos are, with sizes, and whether hand497 detected the simulated gates at all (the cyan count in the legend line).
