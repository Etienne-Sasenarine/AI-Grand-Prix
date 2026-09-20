$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:ENABLE_CAMERAS = '1'
$env:GATE_LOOKAHEAD = '0'
$env:SEPARATE_NETS = '0'
$env:OVERHEAD_LIGHTS = '1'
$env:VK_DRIVER_FILES = 'C:\WINDOWS\System32\DriverStore\FileRepository\nvami.inf_amd64_07a5b3dbac82d20b\nv-vk64.json'
$env:VK_ICD_FILENAMES = $env:VK_DRIVER_FILES
$ckpt = 'D:\Code\Competitions\AIGP\isaac_drone_racer\logs\skrl\drone_racer\2026-09-02_23-56-34_ppo_torch\checkpoints\best_agent.pt'
$p = Start-Process -FilePath 'D:\isaacsim_venv\Scripts\python.exe' `
    -ArgumentList @(
        'scripts\rl\play.py',
        '--task', 'Isaac-Drone-Racer-Play-v0',
        '--num_envs', '1',
        '--real-time',
        '--enable_cameras',
        '--checkpoint', $ckpt
    ) `
    -WorkingDirectory 'D:\Code\Competitions\AIGP\isaac_drone_racer' `
    -RedirectStandardOutput 'D:\Code\Competitions\AIGP\isaac_drone_racer\logs\runlogs\gui_heading_nola.log' `
    -RedirectStandardError 'D:\Code\Competitions\AIGP\isaac_drone_racer\logs\runlogs\gui_heading_nola.err' `
    -PassThru
Write-Output "PID=$($p.Id)"
