$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in PowerShell as Administrator.'
}
$adapter = @(Get-CimInstance Win32_NetworkAdapter | Where-Object {
    $_.PNPDeviceID -like 'USB\VID_0955&PID_7020&MI_00\*' -and $_.NetEnabled
})
if ($adapter.Count -ne 1) { throw 'Expected exactly one connected NVIDIA USB RNDIS adapter.' }
$adapterName = $adapter[0].NetConnectionID
# NCM can receive .100 through DHCP. Give RNDIS a distinct address and prefer
# its directly connected route. Active-store changes last until reboot.
$benchAddress = '192.168.55.101'
$conflicts = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object {
    $_.IPAddress -eq $benchAddress -and $_.InterfaceIndex -ne $adapter[0].InterfaceIndex
})
if ($conflicts.Count) { throw "$benchAddress is already assigned to another adapter; no changes made." }
& netsh interface ipv4 set address "name=$adapterName" source=static "address=$benchAddress" mask=255.255.255.0 gateway=none store=active
if ($LASTEXITCODE -ne 0) { throw 'Could not configure the USB adapter.' }
& netsh interface ipv4 set interface "interface=$adapterName" metric=5 store=active
if ($LASTEXITCODE -ne 0) { throw 'Address configured, but could not set the USB route metric.' }
Write-Host "Orin USB adapter configured at $benchAddress. SSH target remains 192.168.55.1." -ForegroundColor Green
& netsh interface ipv4 show addresses "name=$adapterName"
# To restore DHCP manually: netsh interface ipv4 set address name="Ethernet 4" source=dhcp
