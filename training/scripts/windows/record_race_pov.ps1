# Record the racing policies on the Windows laptop: chase view | onboard camera.
#
# One script from a fresh clone to five videos. It updates this repo, fetches
# aigp-perception for the real-detector overlay, installs what the recorder
# needs into the Isaac Sim venv, checks that Isaac Lab imports, runs the smoke
# test once, and then records both checkpoints in checkpoints/: race40drop
# under the measured corner dropout, race40 with perfect corners.
#
#     git lfs install
#     git clone https://github.com/bojro/aigp-sim.git D:\aigp-sim
#     cd D:\aigp-sim
#     .\scripts\windows\record_race_pov.ps1 -Python D:\isaacsim_venv\Scripts\python.exe
#
# Useful switches:
#     -SkipSmoke        skip the plant smoke test (it takes a few minutes)
#     -NoDetector       green overlay only; skips aigp-perception and onnxruntime
#     -Attempts 3 -Seconds 40 -Speed 1
#     -Out D:\aigp-videos
#
# When it finishes it prints the RECORDED= lines and the scp command that
# copies the videos to the Mac. Isaac's exit code is meaningless; if a run
# fails, paste the traceback back, the recorder has not run on a live install
# before this.
param(
    [string] $Python = 'python',
    [string] $Out = (Join-Path $PSScriptRoot '..\..\videos' | Resolve-Path -ErrorAction SilentlyContinue),
    [string] $PerceptionRepo = '',
    [int]    $Attempts = 3,
    [double] $Seconds = 40,
    [double] $Speed = 1,
    [switch] $SkipSmoke,
    [switch] $NoDetector,
    [string] $MacUser = 'bojro',
    [string] $MacHost = ''
)
$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repo
if (-not $Out) { $Out = Join-Path $repo 'videos' }
if (-not $PerceptionRepo) { $PerceptionRepo = Join-Path (Split-Path $repo -Parent) 'aigp-perception' }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }

Step "aigp-sim at $repo"
git lfs install | Out-Null
git pull --ff-only
git lfs pull

if (-not $NoDetector) {
    Step "aigp-perception at $PerceptionRepo (for the hand497 overlay)"
    if (Test-Path $PerceptionRepo) { git -C $PerceptionRepo pull --ff-only }
    else { git clone https://github.com/bojro/aigp-perception.git $PerceptionRepo }
    $onnx = Join-Path $PerceptionRepo 'models\gate_pose_hand497.onnx'
    if (-not (Test-Path $onnx)) { throw "missing $onnx" }
}

Step "python packages into the Isaac venv ($Python)"
& $Python -m pip install -q -e . imageio imageio-ffmpeg
if (-not $NoDetector) { & $Python -m pip install -q onnxruntime-gpu }

Step "does Isaac Lab import?"
& $Python -c "import isaaclab, isaaclab_tasks, skrl; print('isaaclab ok, skrl', skrl.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Isaac Lab is not importable from $Python; activate the Isaac Sim venv or pass -Python" }

$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:ENABLE_CAMERAS = '1'

if (-not $SkipSmoke) {
    Step "smoke test (the plant on live PhysX; look for SMOKE_RESULT=)"
    $smoke = & $Python scripts\smoke_test.py --headless 2>&1 | Tee-Object -Variable smokeOut
    if (-not ($smokeOut -match 'SMOKE_RESULT=\s*(PASS|OK|pass)')) {
        Write-Host ($smokeOut | Select-String 'SMOKE_RESULT') -ForegroundColor Yellow
        throw "smoke test did not report a pass; fix the plant before recording (or -SkipSmoke if you know why)"
    }
}

$common = @('--headless', '--enable_cameras', '--attempts', $Attempts, '--seconds', $Seconds, '--speed', $Speed)
if (-not $NoDetector) { $common += @('--detector', $onnx, '--perception-repo', $PerceptionRepo) }
$runs = @(
    @{ name = 'race40drop'; ckpt = 'checkpoints\race40drop_best_agent.pt'; drop = '0.13'; sticky = '0.81' },
    @{ name = 'race40';     ckpt = 'checkpoints\race40_best_agent.pt';     drop = '';     sticky = '' }
)
$recorded = @()
foreach ($r in $runs) {
    Step "recording $($r.name): $($r.ckpt)"
    $env:AIGP_POLICY_HZ = '40'
    if ($r.drop) { $env:AIGP_KP_DROP = $r.drop; $env:AIGP_KP_STICKY = $r.sticky }
    else { Remove-Item Env:AIGP_KP_DROP -ErrorAction SilentlyContinue; Remove-Item Env:AIGP_KP_STICKY -ErrorAction SilentlyContinue }
    $dest = Join-Path $Out $r.name
    & $Python scripts\diag\record_race_pov.py --checkpoint $r.ckpt --out $dest @common 2>&1 | Tee-Object -Variable runOut
    $lines = $runOut | Select-String 'RECORDED'
    if (-not $lines) { Write-Host "no RECORDED= line for $($r.name); see the output above" -ForegroundColor Red }
    else { $recorded += $lines }
}

Step "result"
$recorded | ForEach-Object { Write-Host $_ }
Get-ChildItem -Recurse -Filter *.mp4 $Out | ForEach-Object { Write-Host ("  {0,8:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName) }
if ($MacHost) {
    Write-Host "`nto copy to the Mac:`n  scp -r `"$Out`" ${MacUser}@${MacHost}:~/dev/aigp-sim/paper/videos/race_pov"
} else {
    Write-Host "`nto copy to the Mac (fill in its address):`n  scp -r `"$Out`" ${MacUser}@<mac>:~/dev/aigp-sim/paper/videos/race_pov"
}
