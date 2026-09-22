# Windows

`launch_play.ps1` opens a checkpoint in the Isaac GUI from PowerShell:

    .\scripts\windows\launch_play.ps1 -Checkpoint <path-to-best_agent.pt>

It takes `-GateLookahead` and `-SeparateNets` for older checkpoints that were
trained with those set. Training itself runs on Linux (`docs/RUNBOOK_RUNPOD.md`).
