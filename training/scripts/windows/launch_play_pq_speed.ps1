$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:ENABLE_CAMERAS = '1'
$env:GATE_LOOKAHEAD = '0'
$env:SEPARATE_NETS = '0'
$env:OVERHEAD_LIGHTS = '1'
$env:VISUAL_GATE_COUNTER = '1'
$env:PLAY_START_OFFICIAL = '1'
$env:PLAY_START_IDX = '6'
$env:GATE_COUNTER_START = '6'
# Post-policy speed brake (m/s). Leave unset or 0 for the raw trained model.
# $env:PLAY_SPEED_CAP_MPS = '10'
$env:VK_DRIVER_FILES = 'C:\WINDOWS\System32\DriverStore\FileRepository\nvami.inf_amd64_07a5b3dbac82d20b\nv-vk64.json'
$env:VK_ICD_FILENAMES = $env:VK_DRIVER_FILES
$ckpt = 'D:\Code\Competitions\AIGP\isaac_drone_racer\models\pq_speed_best.pt'
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
    -RedirectStandardOutput 'D:\Code\Competitions\AIGP\isaac_drone_racer\logs\runlogs\gui_pq_speed.log' `
    -RedirectStandardError 'D:\Code\Competitions\AIGP\isaac_drone_racer\logs\runlogs\gui_pq_speed.err' `
    -PassThru
Write-Output "PID=$($p.Id)"
