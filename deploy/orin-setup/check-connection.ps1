param([string]$Address = '192.168.55.1')
$ErrorActionPreference = 'Stop'
Get-Command ssh | Select-Object Name, Source
Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in @('Ports', 'Net') -and $_.FriendlyName -notmatch 'WAN Miniport' } | Select-Object Class, FriendlyName, Status
$client = [System.Net.Sockets.TcpClient]::new()
try {
    $pending = $client.ConnectAsync($Address, 22)
    if ($pending.Wait(3000) -and $client.Connected) {
        Write-Output "SSH is reachable at $Address. Device identity and login still need verification."
    } else {
        Write-Output "No SSH response at $Address. Check power, data cable, carrier-board device port, or use its LAN address."
    }
} catch {
    Write-Output "SSH is unavailable at ${Address}: $($_.Exception.Message)"
} finally {
    $client.Dispose()
}
