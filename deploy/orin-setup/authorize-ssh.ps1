$ErrorActionPreference = 'Stop'
$keyPath = Join-Path $env:USERPROFILE '.ssh\orin_dcl_setup'
Write-Host 'Refreshing the Orin USB network address...'
$usbAdapter = Get-NetAdapter | Where-Object { $_.InterfaceDescription -eq 'Remote NDIS Compatible Device' -and $_.Status -eq 'Up' }
if (@($usbAdapter).Count -eq 1) {
    $ipConfig = Get-NetIPInterface -InterfaceIndex $usbAdapter.ifIndex -AddressFamily IPv4
    if ($ipConfig.Dhcp -eq 'Enabled') { & ipconfig /renew $usbAdapter.Name | Out-Null }
}
Write-Host 'Authorize this PC to access your Orin as dcl.'
Write-Host 'Enter the Orin Ubuntu password at the SSH prompt. Characters will not appear.'
Write-Host 'This adds a dedicated public key; it does not change your password or control motors.'
$remoteCommand = 'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; key=$(cat); grep -qxF "$key" ~/.ssh/authorized_keys || printf "%s\n" "$key" >> ~/.ssh/authorized_keys'
Get-Content -LiteralPath ($keyPath + '.pub') | & ssh -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes -o PubkeyAuthentication=no -o PreferredAuthentications=password -o NumberOfPasswordPrompts=3 dcl@192.168.55.1 $remoteCommand
if ($LASTEXITCODE -ne 0) { throw 'SSH authorization failed. Tell Codex the error, not your password.' }
& ssh -i $keyPath -o IdentitiesOnly=yes -o BatchMode=yes dcl@192.168.55.1 'uname -srmo'
if ($LASTEXITCODE -ne 0) { throw 'Key verification failed.' }
Write-Host 'SUCCESS: SSH key access verified. Return to Codex and say ready.' -ForegroundColor Green
