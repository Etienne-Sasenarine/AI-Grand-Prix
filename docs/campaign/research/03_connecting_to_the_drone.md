# 03 — Connecting to the drone: runbook

Written 19 Sep 2026. Scope: Jetson Orin NX 16 GB / Seeed A603 / JetPack 6.2, user `dcl`; laptop is
Windows 11 + WSL2 (NAT). Nothing here has been run against the drone — commands are from documents
and from reading our own files. **Never paste the Wi-Fi password into a file or a script; use
`nmcli --ask`.**

Tags: **[CITED: url]** public source · **[ORGANIZER-DOC: file:line]** · **[TEAM-OBSERVED]** ·
**[MY ESTIMATE]** reasoning, not verified. File short names: `QS` = `reference_text/orin-nx-quickstart.txt`,
`SG` = `reference_text/orin-nx-software-guide.txt`, `target/…` = organizer scripts.

## 0. Five findings that change what we do

1. **The RC link is not on 2.4 GHz.** The receiver is serial CRSF (`serialrx_provider = CRSF`, dump
   line 686). The vendor ELRS block (dump lines 1110–1117) reads `7850/8100/8350` and
   `9035/9150/9269`, 20 channels each — read as tenths of a MHz that is **785–835 MHz and
   903.5–926.9 MHz**, a dual-band sub-GHz link [MY ESTIMATE on the units; the numbers are
   TEAM-OBSERVED]. `expresslrs_domain = AU433` (line 1105) belongs to the unused on-board SPI
   receiver (its UID is all zeros), ignore it. ELRS is sold in separate 868/915 MHz and 2.4 GHz
   families [CITED: https://oscarliang.com/expresslrs-receivers/]. So the Jetson's 2.4 GHz Wi-Fi does
   **not** share a band with the pilot link. See §5.
2. **The recorder is not yet power-cut-safe.** `tools/flight_recorder.py` calls `fh.flush()` per row
   but never `os.fsync()`. `flush()` only hands data to the Linux page cache. ext4 commits every
   5 s, and with delayed allocation "even older data can be lost on power failure" because writeback
   waits for `dirty_expire_centisecs` (30 s default) [CITED:
   https://www.kernel.org/doc/html/latest/admin-guide/ext4.html]. A battery pull at the end of a
   flight can lose the last 5–30 s — exactly the landing. Fix in §4.
3. **Swapped boards break SSH by design.** Every board answers on the same `192.168.55.1` with a
   *different* host key, so SSH refuses with "REMOTE HOST IDENTIFICATION HAS CHANGED". The
   teammate's `authorize-ssh.ps1` uses `StrictHostKeyChecking=yes` (line 13) and will fail on every
   new board until the old key is removed. Fix: §1 step B and the SSH config in §2.
4. **`nvgetty` is not the thing fighting for `/dev/ttyTHS1` on Orin.** The organizer script says
   nvgetty owns `ttyTCU0` (the debug console) and should be left alone; the real suspects are
   `serial-getty@ttyTHS1` and a `console=` kernel argument [ORGANIZER-DOC:
   target/msp/setup_jetson_uart.sh:95-117]. `provision.sh:152-153` mentions nvgetty only for
   `ttyTHS0`. Just run the script; do not disable nvgetty by hand.
5. **`nmcli radio wifi off` survives reboots.** NetworkManager remembers the radio state. Turn Wi-Fi
   off, pull the battery, and the board comes back with Wi-Fi off — reachable by cable only
   [MY ESTIMATE, standard NetworkManager behaviour]. Do not use it casually.

---

## 1. Get connected in 5 minutes (try in this order)

Rule: **test from Windows PowerShell first.** WSL2 in NAT mode reaches the drone *through* Windows,
so if Windows cannot ping it, WSL never will [CITED: https://learn.microsoft.com/en-us/windows/wsl/networking]
[MY ESTIMATE that this is why WSL "often" fails].

### A. USB cable (most reliable thing we have today)

Data-capable cable into the A603's **micro-USB port next to the barrel jack (W6)**, not a USB-A port
[ORGANIZER-DOC: QS:35-40] [CITED: https://wiki.seeedstudio.com/headless_setup_and_recovery_for_a603/
— "Power-only cables are electrically indistinguishable from nothing happening"]. Wait ~60 s after
power-on.

```powershell
# PowerShell. What did Windows see?
Get-PnpDevice -PresentOnly | ? { $_.InstanceId -like 'USB\VID_0955*' } | ft Status,Class,FriendlyName,InstanceId
```

| You see | Meaning | Go to |
|---|---|---|
| `PID_7020`, a "Remote NDIS" / "NCM" network adapter and a COM port | Normal device mode | step B |
| `PID_7323`, "APX", error state | **Force-recovery mode.** Not a Windows driver problem | §3 row 2 |
| nothing with VID_0955 | Charge-only cable, wrong port, board not booted, or device mode never started [ORGANIZER-DOC: target/usb-device-mode.sh:10-17] | swap cable, then C or D |

### B. Give Windows a fixed address, then SSH

Windows DHCP on this adapter cycles (15-second leases [TEAM-OBSERVED]; the same ~13 s up / 10 s down
pattern is reported on NVIDIA's forum and blamed on the Windows RNDIS driver [CITED:
https://forums.developer.nvidia.com/t/jetson-nano-ndis-usb-network-periodically-disconnects-with-windows-computer/170491]).
Windows 11 also shows **two** adapters, RNDIS and NCM, on the same subnet [CITED:
https://forums.developer.nvidia.com/t/dual-ethernet-universal-device-controller-linux-gadget-configuration/287416],
which is why the teammate's script uses `.101` and a low metric rather than `.100`.

```powershell
# PowerShell AS ADMINISTRATOR. Must be re-run for every new board: a new board is a new Windows adapter.
& "\\wsl.localhost\Ubuntu\home\badip\aigp\aigp_hover10s\orin-setup\fix-usb-network.ps1"   # [TEAM-OBSERVED to work]
ping 192.168.55.1
ssh-keygen -R 192.168.55.1          # forget the previous drone's host key (finding 3)
ssh dcl@192.168.55.1                # password dcl  [ORGANIZER-DOC: QS:38]
```

Then from WSL: `ssh orin-usb` (config in §2). If WSL hangs while PowerShell works, do not debug it —
use Windows' SSH from inside WSL, which exists on this laptop:
`/mnt/c/Windows/System32/OpenSSH/ssh.exe dcl@192.168.55.1` (it uses the keys in
`C:\Users\badip\.ssh`, not `~/.ssh`).

### C. Serial console on the same cable (works when the network half does not)

The cable also carries a login console: a COM port on Windows, 115200 baud [ORGANIZER-DOC: QS:49-53].

```powershell
Get-PnpDevice -Class Ports -PresentOnly | ft FriendlyName      # find COMn
plink -serial COM5 -sercfg 115200,8,n,1                        # PuTTY's CLI; install PuTTY BEFORE you need it
```

Log in `dcl`/`dcl`, then `ip -br addr` tells you the Wi-Fi address and `nmcli device` the Wi-Fi state.
Caveat: network, console and the README drive are **one gadget** — if Windows sees no VID_0955
device at all, the console is dead too [ORGANIZER-DOC: target/usb-device-mode.sh:5-8].

### D. Ethernet cable, static addresses (best bench link once set up)

The organizer script says the board can be reached on Ethernet at `192.168.0.1`
[ORGANIZER-DOC: target/usb-device-mode.sh:8] — **unverified; check with `ip -br addr` while you
are in over USB.** If it is not configured, add it once per board (§2 script does this):

```bash
# Jetson (find the wired interface name with `nmcli device`; assumed eth0)
sudo nmcli con add type ethernet ifname eth0 con-name bench-eth ipv4.method manual \
     ipv4.addresses 192.168.0.1/24 ipv6.method disabled connection.autoconnect yes
```
```powershell
# Windows, admin, once per laptop (adapter name from Get-NetAdapter; a USB-Ethernet dongle is fine)
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.0.100 -PrefixLength 24
ssh dcl@192.168.0.1
```
No DHCP, no driver quirks, no radio. [MY ESTIMATE: most reliable option; needs a cable run to the drone.]

### E. Wi-Fi (convenience only — never on the critical path)

```powershell
ping dcl-orin.local                      # Windows resolves .local itself; WSL NAT cannot (multicast does not cross NAT)
arp -a | findstr /i "<wifi-mac-of-this-board>"   # MAC recorded by the §2 script; Windows prints it with dashes (aa-bb-cc-…)
1..254 | % { Test-Connection "192.168.13.$_" -Count 1 -TimeoutSeconds 1 -Quiet | Out-Null }; arp -a   # sweep, then read arp
```
Venue networks often block client-to-client multicast, so mDNS failing proves nothing
[MY ESTIMATE]. Update `HostName` under `Host orin` in `~/.ssh/config` (currently `192.168.13.16`).

### F. None of the above

Monitor + keyboard on the carrier, or hand the board back: the organizers "have a diagnostic for
exactly this" [ORGANIZER-DOC: QS:51-53]. If you get in any other way, run
`sudo ~/target/usb-device-mode.sh`, then `--force` [ORGANIZER-DOC: target/usb-device-mode.sh:19-21,102-107].

---

## 2. New drone checklist (target: under 10 minutes)

Props **off**. Flight battery **in** for step 6 only (the FC must be powered to answer).

1. Cable in, §1 A–B until `ssh dcl@192.168.55.1` works from PowerShell.
2. Install our SSH key (one password prompt). From WSL: `ssh-copy-id -i ~/.ssh/orin_key.pub dcl@192.168.55.1`.
   From Windows: `authorize-ssh.ps1` after `ssh-keygen -R 192.168.55.1`.
3. Record the board's identity (serial number, Wi-Fi MAC, host name) — this is how you find it later.
4. Free the UART: `sudo ~/target/msp/setup_jetson_uart.sh --apply` [ORGANIZER-DOC: QS:144-147; SG:165-168].
5. Copy `tools/` and `flight/` to `~/aigp/`. Also copy `~/target` **off** the board if we still lack it.
6. Confirm the FC answers: `python3 ~/target/msp/msp_bench.py --port /dev/ttyTHS1 info` — `--port`
   goes *before* the subcommand [ORGANIZER-DOC: QS:149-152].
7. Harden Wi-Fi, give the board a unique host name, add the bench Ethernet profile.
8. Install the boot-time recorder last, and confirm it is the *only* holder of the port.
9. `sudo shutdown -h now` before pulling power [ORGANIZER-DOC: QS:203-206].

Put this once in WSL `~/.ssh/config`. The USB address is shared by every drone, so its host key is
deliberately not remembered; the link is a private cable, so this costs nothing [MY ESTIMATE]:

```
Host orin-usb
    HostName 192.168.55.1
    User dcl
    IdentityFile ~/.ssh/orin_key
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ConnectTimeout 5
```

### Draft script — UNTESTED

```bash
#!/usr/bin/env bash
# new_drone.sh — UNTESTED DRAFT. Never run against hardware. Read it before trusting it.
# Run from WSL in the workspace root, AFTER `ping 192.168.55.1` works from PowerShell (section 1 A-B).
# Usage: ./new_drone.sh <board-number>      e.g. ./new_drone.sh 3
# Asks for the password `dcl` at most twice: once for ssh-copy-id, once for sudo.
set -euo pipefail
N="${1:?board number, e.g. 3}"
H=192.168.55.1
KEY="$HOME/.ssh/orin_key"
WS=/mnt/c/Users/badip/Downloads/ai-grand-prix
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 "dcl@$H")
SCP=(scp -q -r -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

echo "== 1. reachable?"
ping -c2 -W2 "$H" >/dev/null || { echo "No ping. Fix the Windows adapter first (fix-usb-network.ps1)."; exit 1; }

echo "== 2. SSH key (password: dcl)"
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -f "$KEY"
ssh-keygen -R "$H" >/dev/null 2>&1 || true
"${SSH[@]}" -o BatchMode=yes true 2>/dev/null || \
  ssh-copy-id -i "$KEY.pub" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "dcl@$H"

echo "== 3. identity"
mkdir -p "$WS/logs/orin/boards"
"${SSH[@]}" 'echo "serial: $(tr -d "\0" </proc/device-tree/serial-number)"; echo "host: $(hostname)";
  echo "wifi-mac: $(cat /sys/class/net/wlP1p1s0/address 2>/dev/null)"; head -1 /etc/nv_tegra_release;
  ip -br addr' | tee "$WS/logs/orin/boards/board${N}_$(date +%Y%m%d_%H%M).txt"

echo "== 4. copy tools (and pull ~/target if we do not have it)"
"${SSH[@]}" 'mkdir -p ~/aigp ~/flights'
"${SCP[@]}" "$WS/tools" "$WS/flight" "dcl@$H:~/aigp/"
[ -d "$WS/target/msp" ] || "${SCP[@]}" "dcl@$H:~/target" "$WS/"

echo "== 5. root-level setup on the board (sudo password: dcl)"
cat > /tmp/aigp_board_setup.sh <<EOF
set -u
systemctl stop aigp-recorder 2>/dev/null || true             # it holds the FC port
/home/dcl/target/msp/setup_jetson_uart.sh --apply            # organizers' script; idempotent
hostnamectl set-hostname dcl-orin-${N}                       # unique name per board for mDNS
sed -i "s/^127\.0\.1\.1.*/127.0.1.1 dcl-orin-${N}/" /etc/hosts
printf '[connection]\nwifi.powersave = 2\n' > /etc/NetworkManager/conf.d/99-wifi-powersave-off.conf
for c in \$(nmcli -t -f NAME,TYPE con show | awk -F: '\$2=="802-11-wireless"{print \$1}' | tr ' ' '~'); do
  c="\${c//\~/ }"
  nmcli con mod "\$c" connection.autoconnect yes connection.autoconnect-priority 100 \
        connection.autoconnect-retries 0 connection.permissions "" 802-11-wireless-security.psk-flags 0 || true
done
ETH=\$(nmcli -t -f DEVICE,TYPE device | awk -F: '\$2=="ethernet"{print \$1; exit}')
nmcli con show bench-eth >/dev/null 2>&1 || { [ -n "\$ETH" ] && nmcli con add type ethernet ifname "\$ETH" \
  con-name bench-eth ipv4.method manual ipv4.addresses 192.168.0.1/24 ipv6.method disabled; }
systemctl enable --now ssh avahi-daemon 2>/dev/null || true
mkdir -p /var/log/journal && systemctl restart systemd-journald   # keep logs across power cuts
systemctl reload NetworkManager || true
sync
EOF
"${SCP[@]}" /tmp/aigp_board_setup.sh "dcl@$H:/tmp/"
"${SSH[@]}" -t 'sudo bash /tmp/aigp_board_setup.sh'

echo "== 6. does the flight controller answer? (battery in, PROPS OFF)"
"${SSH[@]}" 'sudo fuser -v /dev/ttyTHS1 || echo "port is free"'
"${SSH[@]}" 'python3 ~/target/msp/msp_bench.py --port /dev/ttyTHS1 info' \
  || echo "!! FC silent: is the battery in? has someone used the Betaflight CLI (needs a battery pull)?"

echo "== 7. recorder on boot — check what --install prints; it should write /tmp/aigp-recorder.service"
"${SSH[@]}" 'python3 ~/aigp/tools/flight_recorder.py --install'
"${SSH[@]}" -t 'sudo cp /tmp/aigp-recorder.service /etc/systemd/system/ && sudo systemctl daemon-reload \
  && sudo systemctl enable --now aigp-recorder && sleep 6 && systemctl is-active aigp-recorder \
  && ls -la ~/flights | tail -3 && sync'

echo "== done. Wi-Fi join is manual so the password never touches a file:"
echo "   ssh -t orin-usb 'sudo nmcli --ask device wifi connect \"<SSID>\" ifname wlP1p1s0'   then re-run step 5"
```
Known weak points of the draft: sudo may or may not prompt; the recorder script path on the board
(`~/aigp/tools/` here vs `~/aigp/` in `FLIGHT_SESSION.md`) must match whatever is already in use;
the Wi-Fi loop's name handling is clumsy; `psk-flags 0` only helps if the key was stored at all.

---

## 3. Diagnosis table

| # | Symptom | Likely cause | Check | Fix |
|---|---|---|---|---|
| 1 | `192.168.55.1` unreachable from WSL | Windows lost its address on the RNDIS adapter (15 s leases [TEAM-OBSERVED]); or RNDIS + NCM both on one subnet; WSL only relays what Windows has | PowerShell: `ping 192.168.55.1`; `Get-NetIPAddress -IPAddress 192.168.55.*` | `fix-usb-network.ps1` as admin (static `.101`, metric 5). Note the README says `.100` but the script sets `.101` on purpose. Last resort: `ssh.exe` from WSL. **Not recommended mid-week:** `networkingMode=mirrored` in `.wslconfig` needs `wsl --shutdown`, which kills any running training [CITED: https://learn.microsoft.com/en-us/windows/wsl/networking]; usbipd-win takes the *whole* USB device away from Windows while attached [CITED: https://learn.microsoft.com/en-us/windows/wsl/connect-usb] and the WSL kernel may lack RNDIS/NCM drivers [MY ESTIMATE] |
| 1b | Windows sees no NVIDIA adapter at all | Charge-only cable; wrong port; device mode did not start because the VBUS-detect line never asserted [ORGANIZER-DOC: target/usb-device-mode.sh:10-17] | `Get-PnpDevice` line in §1A | New cable → Ethernet/monitor → `sudo ~/target/usb-device-mode.sh --force` |
| 1c | "REMOTE HOST IDENTIFICATION HAS CHANGED" | Different drone, same IP | — | `ssh-keygen -R 192.168.55.1` or the `orin-usb` config |
| 2 | Board shows as **APX, VID_0955 PID_7323** | `0955:7323` *is* "Orin NX 16 GB in Force Recovery Mode" [CITED: https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/IN/QuickStart.html]. Windows' "error state" is only the missing APX driver, not a second fault. Causes: (a) recovery pin held low — on the A603 that is **W7 pin 3 shorted to pin 4 (GND)** at power-on [CITED: https://wiki.seeedstudio.com/headless_setup_and_recovery_for_a603/] — a left-in jumper, bent pin, debris or crash damage; a marginal jumper caused exactly this in one forum case [CITED: https://forums.developer.nvidia.com/t/nvidia-jetson-orin-nx-stuck-in-recovery-mode/325065]; (b) the boot ROM found no valid bootloader in the module's flash and fell back to recovery [MY ESTIMATE]; (c) a faulty module or SSD — one forum case was only cured by swapping the module [CITED: https://forums.developer.nvidia.com/t/orin-nx-stuck-at-recovery-mode/360927]; (d) brown-out during power-up [MY ESTIMATE] | Look at W7 pins 3–4 with a torch. Remove **all** power (battery *and* USB) for 10 s, power up with USB unplugged, plug USB in after 30 s | If it still says 7323: **hand it back.** Re-flashing needs an x86 Ubuntu host [CITED: NVIDIA quick start above] *and* the organizers' custom image — the guide says a re-flash "is the organisers' job" [ORGANIZER-DOC: QS:199-201]. We have neither. Do not spend a slot on it |
| 3 | Jetson not on Wi-Fi after power-up | (a) autoconnect gave up — default is a few tries; (b) the saved key is "agent-owned", so at boot with nobody logged in there is no key [MY ESTIMATE]; (c) the profile file was written seconds before a power cut and never reached disk (finding 2); (d) Wi-Fi power-save; (e) radio left off (finding 5) | Console: `nmcli device`; `nmcli -g connection.autoconnect,connection.autoconnect-retries,802-11-wireless-security.psk-flags con show "<SSID>"`; `journalctl -b -u NetworkManager \| tail -30`; `iw dev wlP1p1s0 get power_save` | `autoconnect-retries 0` = retry forever [CITED: https://oneuptime.com/blog/post/2026-03-20-connection-autoconnect-priority-nmcli/view]; `psk-flags 0`; `wifi.powersave = 2` in `/etc/NetworkManager/conf.d/`, and if `iw` still says "on" use `iw dev wlP1p1s0 set power_save off` from a boot service — the config file is reported not to stick on Jetsons [CITED: https://github.com/robwaat/Tutorial/blob/master/Jetson%20Disable%20Wifi%20Power%20Management.md]; always `sync` after changing network settings |
| 3b | On Wi-Fi but address changed, `dcl-orin` does not resolve | DHCP; all boards share one host name; mDNS blocked by venue or by WSL NAT | §1E | Unique host name per board; record the MAC; ask organizers for a DHCP reservation; or own the network (§4) |
| 4 | Worry: Wi-Fi next to the RC receiver | Different bands (finding 1). Residual risk is broadband noise from an antenna centimetres away, and the unknown video-transmitter band [MY ESTIMATE] | Bench: props off, link-quality/RSSI on the transmitter with Jetson Wi-Fi idle vs. running `iperf`/`scp` | If no change, leave Wi-Fi alone. See §5 |
| 5 | `/dev/ttyTHS1` busy, or MSP replies missing/garbled | Our own `aigp-recorder` service (pyserial `exclusive=True`, `target/msp/msp.py:213`); a serial getty / `console=` on the port | `sudo fuser -v /dev/ttyTHS1`; `sudo lsof /dev/ttyTHS1`; `systemctl is-active aigp-recorder serial-getty@ttyTHS1` | `sudo systemctl stop aigp-recorder` — **killing the PID is useless, the unit has `Restart=always`, 5 s** (`tools/flight_recorder.py:204-205`). Once per board: `setup_jetson_uart.sh --apply`. Use the `fc` wrapper in §4 |
| 6 | FC silent on every baud after a CLI session | Entering the Betaflight CLI over this UART wedges the FC until power is removed; 2 of 2 [TEAM-OBSERVED, SPEC.md:223] | `msp_bench.py … info` times out | `sudo shutdown -h now` **first** (the Jetson is on the same battery), then pull the battery. Never CLI on the flight line; use MSP (`setup_angle_mode.py`). For real CLI work use Betaflight Configurator on the FC's own USB port at the bench [MY ESTIMATE that this avoids the wedge] |
| 7 | New drone, nothing works | Nothing is set up: no key, UART not freed, no tools, no recorder, old host key | — | §2, one script |

---

## 4. Making capture and autonomy independent of any network

Principle: **a network link is for setup and for pulling data afterwards. Nothing that must happen
during a flight may start from, or be kept alive by, a laptop.** `FLIGHT_SESSION.md` already moved
the recorder to a boot service; these close the remaining holes.

1. **`fsync` the recorder.** After `fh.flush()`, call `os.fsync(fh.fileno())` about once a second
   (not every row). Also `fsync` the directory once after creating the file, or the file itself may
   not exist after a power cut. Cost is small on NVMe [MY ESTIMATE]. Belt and braces: `sudo sysctl -w
   vm.dirty_expire_centisecs=300 vm.dirty_writeback_centisecs=100`.
2. **One owner of the FC port, with a hand-over wrapper.** Install on the board as `~/aigp/fc`:
   ```bash
   #!/bin/bash
   # fc <command...> : stop the recorder, run the command, always restart the recorder.  UNTESTED
   sudo systemctl stop aigp-recorder; trap 'sudo systemctl start aigp-recorder' EXIT; "$@"
   ```
   Better for race day: the autonomy process owns the port and writes the log itself, started by
   systemd with `Conflicts=aigp-recorder.service`, so the two can never both run [MY ESTIMATE].
3. **Autonomy starts from systemd or a transmitter switch, never from an SSH session.** If it must be
   launched by hand, launch inside `tmux new -s fly` so a dropped link does not kill it (check
   `which tmux` on the board; there is no network for `apt` at the venue unless we provide it).
4. **A status signal that needs no network.** The recorder already knows if the FC answers; expose it
   on something visible — an LED, or the FC beeper over MSP if we find a safe command [MY ESTIMATE].
   At minimum, a 10-second pre-flight check over the USB cable: `systemctl is-active aigp-recorder &&
   ls -la ~/flights | tail -1` and see the newest file growing.
5. **Survive hard power cuts.** "Pulling the battery while it is writing will eventually corrupt it"
   [ORGANIZER-DOC: QS:203-206]. Shut down cleanly whenever a cable is attached. Make journald
   persistent so we can see *why* Wi-Fi failed on the previous boot (`journalctl -b -1`). A read-only
   root with an overlay (`overlayroot`) plus a separate writable log partition is the full fix
   [CITED: https://www.forecr.io/blogs/programming/how-to-protect-the-root-filesystem-on-jetson-with-overlayroot]
   — **too invasive to attempt this week** on a board we cannot re-flash [MY ESTIMATE]. Copy flights
   off after every session; a swapped drone takes its SSD with it.
6. **Own the network if allowed.** A pocket travel router (own SSID, 5 GHz, DHCP reservation per
   board MAC) removes the changing-address and mDNS problems entirely. A Windows "Mobile hotspot"
   does the same with no extra hardware but tends to switch itself off when idle and wants an
   upstream connection [MY ESTIMATE]. Give the team network `autoconnect-priority 200` and the venue
   network `100` so either works. Ask first (§6).

---

## 5. Wi-Fi in flight: recommendation

Leave the Jetson's Wi-Fi **on, idle and unimportant**. The pilot link is sub-GHz (finding 1), so
there is no co-channel conflict to avoid, and switching the radio off risks locking ourselves out
after a power cut (finding 5). Do not stream video or `scp` during a flight — that is the only time
the radio is a strong transmitter next to the receiver antenna. Do the bench link-quality check in
§3 row 4 once. If the organizers require radios off, use `sudo rfkill block wifi` from the autonomy
service at arm and `rfkill unblock wifi` at disarm, plus a boot-time `nmcli radio wifi on`, and test
that a battery pull mid-flight still comes back reachable. Ask which band the video transmitter
uses: 5.8 GHz analogue video overlaps the top of the 5 GHz Wi-Fi band, which matters if we bring a
5 GHz router [MY ESTIMATE].

---

## 6. Ask the organizers

1. A board boots to **APX / 0955:7323** with nothing on the recovery pins — will you re-flash or swap
   it, and how long does that take? Is there a spare on the shelf now?
2. When a drone is swapped, can the **NVMe SSD (or the Jetson module) move with us**, so keys, tools
   and flight logs survive? If not, is there a supported way to clone our setup?
3. Is the **Ethernet port configured at 192.168.0.1** on every image, as `usb-device-mode.sh` implies?
   May we run a cable to the drone in the pit?
4. May we bring a **travel router or laptop hotspot**? Any band/channel/power restrictions? If not,
   can the venue network give each of our boards a **fixed DHCP reservation** (we supply the MACs),
   and does it allow client-to-client traffic and mDNS?
5. Which bands do the **pilot link and the video transmitter** actually use (our dump suggests
   785–835 and 903–927 MHz)? Must companion-computer Wi-Fi be off during scored runs?
6. Is the Betaflight **CLI-over-UART wedge** known? Is there a supported way to change settings that
   need the CLI — Configurator over the FC's USB port?
7. Is there a recommended **clean power-down** on the flight line, given the Jetson runs off the
   flight battery and the guide warns that battery pulls corrupt the SSD?
8. Does `sudo` for `dcl` stay enabled on all boards, and is `tmux` on the image?

## Sources not already linked inline

- NVIDIA Jetson Linux r36.4.3 Quick Start (recovery-mode USB IDs, host requirements): https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/IN/QuickStart.html
- Seeed A603 flashing page (pins 3–4 of the 14-pin header, micro-USB, Ubuntu host): https://wiki.seeedstudio.com/reComputer_A603_Flash_System/
- NVIDIA forum, default gadget is RNDIS + NCM + serial, config in `/opt/nvidia/l4t-usb-device-mode/`: https://forums.developer.nvidia.com/t/help-with-usb-gadget-and-existing-l4t-config/274895
- NVIDIA forum, ext4/NVMe corruption after power loss on Orin: https://forums.developer.nvidia.com/t/nvme-corruption/352124
- NetworkManager settings reference: https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nm-settings-nmcli.html
