# Wireless SSH

Orin joined `AI Grand Prix` using its `wlP1p1s0` Wi-Fi adapter.
NetworkManager auto-connect is enabled and the SSH service is enabled at boot.

Verified SSH from this PC (`192.168.13.90`) to Orin Wi-Fi (`192.168.13.205`).
The PC's SSH configuration contains an `orin` alias using the existing dedicated
key and the previously verified USB host identity.

```powershell
ssh orin
```

USB can be unplugged; the Orin still needs power. This provides access on the
same reachable local network, not access over the public Internet.
The address is assigned by DHCP and may change. If the shortcut later stops
working, find the Orin's new address in the router's client list (hostname
`dcl-orin`) and update `HostName` in the PC's `~/.ssh/config`, or reconnect USB
and inspect `ip -brief address show wlP1p1s0`.
No router port forwarding or public SSH exposure was configured.
