# Play a checkpoint in the Isaac GUI on Windows. One script, parameterised;
# it replaced thirteen copies that each hard-coded one checkpoint path.
#
#     .\scripts\windows\launch_play.ps1 -Checkpoint logs\skrl\drone_racer\<run>\checkpoints\best_agent.pt
#     .\scripts\windows\launch_play.ps1 -Checkpoint <path> -GateLookahead 1 -SeparateNets 1
#
# Assumes the Isaac Sim venv python is on PATH or given via -Python, and that
# you run it from the repo root. The VK_DRIVER_FILES pin some machines need
# for Vulkan is left to the caller's environment.
param(
    [Parameter(Mandatory = $true)] [string] $Checkpoint,
    [string] $Task = 'Isaac-Drone-Racer-Play-v0',
    [string] $GateLookahead = '0',
    [string] $SeparateNets = '0',
    [string] $Python = 'python'
)
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:ENABLE_CAMERAS = '1'
$env:GATE_LOOKAHEAD = $GateLookahead
$env:SEPARATE_NETS = $SeparateNets
& $Python scripts\rl\play.py --task $Task --num_envs 1 --real-time --enable_cameras --checkpoint $Checkpoint
