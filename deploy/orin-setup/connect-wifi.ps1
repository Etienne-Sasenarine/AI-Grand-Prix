$ErrorActionPreference = 'Stop'
$keyPath = Join-Path $env:USERPROFILE '.ssh\orin_dcl_setup'
Write-Host 'Connect the Orin to AI Grand Prix Wi-Fi. Keep USB connected until verified.'
Write-Host 'Enter the Orin Ubuntu password at the sudo prompt, then the Wi-Fi password if requested.'
Write-Host 'Passwords stay in this terminal; do not paste them into chat.'
& ssh -t -i $keyPath -o IdentitiesOnly=yes -o ConnectTimeout=5 dcl@192.168.55.1 'sudo nmcli --ask --wait 40 device wifi connect "AI Grand Prix" ifname wlP1p1s0 && sudo nmcli connection modify "AI Grand Prix" connection.autoconnect yes && ip -brief address show wlP1p1s0'
if ($LASTEXITCODE -ne 0) { throw 'Wi-Fi setup did not complete. Keep USB connected and report the error without passwords.' }
Write-Host 'Wi-Fi configured. Return to Codex for a Wi-Fi SSH connectivity check before unplugging USB.' -ForegroundColor Green
