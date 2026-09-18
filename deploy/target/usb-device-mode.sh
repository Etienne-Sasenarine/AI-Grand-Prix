#!/bin/bash
#
# Diagnose (and optionally force) USB device mode on the A603 — run ON THE JETSON.
#
# Device mode is what serves 192.168.55.1, /dev/ttyACM0 and the L4T-README
# drive to a host over the micro-USB port. All three come from the SAME USB
# gadget, so if tethering is dead the serial console is dead too -- get in over
# Ethernet (192.168.0.1) or a UART header to run this.
#
# Why it fails on this carrier: the device tree marks usb2-0 as
#   mode = "otg";  usb-role-switch;  connector { gpio-usb-b-connector,
#                                                vbus-gpio = PZ.1 ACTIVE_LOW }
# so Linux only switches the port to *peripheral* when PZ.1 reports VBUS from
# the host. If that line is not wired, is inverted, or reads high, the port
# stays in host mode and no gadget is ever created. Recovery-mode flashing
# still works because that runs in the bootloader and never uses this path --
# so "flashing works" does not prove device mode will.
#
# Usage:
#   sudo ./usb-device-mode.sh              # diagnose
#   sudo ./usb-device-mode.sh --force      # force peripheral role, restart it
#
set -uo pipefail
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
val() { printf '   %-34s %s\n' "$1" "$2"; }

say "USB device controller (UDC)"
# Without a UDC there is no gadget, and the l4t service cannot do anything.
if compgen -G "/sys/class/udc/*" >/dev/null; then
	for u in /sys/class/udc/*; do
		val "$(basename "$u")" "state=$(cat "$u/state" 2>/dev/null) \
current=$(cat "$u/current_speed" 2>/dev/null)"
	done
else
	val "/sys/class/udc" "EMPTY — no device controller bound"
	echo "     The XUDC never came up as a peripheral. That is the whole problem;"
	echo "     everything below explains why."
fi

say "USB role switch"
FOUND=0
for r in /sys/class/usb_role/*; do
	[ -e "$r/role" ] || continue
	FOUND=1
	val "$(basename "$r")" "$(cat "$r/role" 2>/dev/null)"
done
[ "$FOUND" = 0 ] && val "/sys/class/usb_role" "no role switch registered"

say "VBUS detect line (PZ.1, active low)"
if command -v gpiofind >/dev/null && command -v gpioget >/dev/null; then
	# libgpiod numbers lines densely, so the DT macro value is NOT the line
	# number -- look the line up by name rather than computing it.
	LINE="$(gpiofind PZ.01 2>/dev/null || gpiofind PZ.1 2>/dev/null)"
	if [ -n "$LINE" ]; then
		val "gpiofind" "$LINE"
		val "raw level" "$(gpioget $LINE 2>/dev/null) (active-low: 0 means VBUS present)"
	else
		val "PZ.1" "not found by name; dumping candidates:"
		gpioinfo 2>/dev/null | grep -iE "vbus|PZ\." | sed 's/^/     /'
	fi
else
	val "gpiod tools" "not installed (apt install gpiod)"
fi

say "l4t usb device-mode service"
val "unit" "$(systemctl is-enabled nv-l4t-usb-device-mode 2>/dev/null) / $(systemctl is-active nv-l4t-usb-device-mode 2>/dev/null)"
journalctl -u nv-l4t-usb-device-mode -b --no-pager 2>/dev/null | tail -8 | sed 's/^/     /'

say "gadget network interface"
if ip -br addr show l4tbr0 >/dev/null 2>&1; then
	val "l4tbr0" "$(ip -br addr show l4tbr0 | tr -s ' ')"
else
	val "l4tbr0" "absent — the gadget was never created"
fi
ip -br addr show 2>/dev/null | grep -E "usb|l4t" | sed 's/^/     /'

if [ "$FORCE" = 1 ]; then
	say "forcing peripheral role"
	DONE=0
	for r in /sys/class/usb_role/*; do
		[ -w "$r/role" ] || continue
		echo device > "$r/role" 2>/dev/null && {
			val "$(basename "$r")" "-> $(cat "$r/role")"; DONE=1; }
	done
	if [ "$DONE" = 0 ]; then
		echo "   No writable role switch. The port is not in OTG mode at all;"
		echo "   the durable fix is to patch the DTB to mode = \"peripheral\""
		echo "   (host-side: build/stage-usb-peripheral.sh, then reflash)."
	else
		systemctl restart nv-l4t-usb-device-mode
		sleep 3
		val "l4tbr0 after restart" "$(ip -br addr show l4tbr0 2>/dev/null | tr -s ' ' || echo 'still absent')"
	fi
fi

cat <<'EOF'

Reading this:
  * UDC present + role "device" + l4tbr0 192.168.55.1  -> board side is fine,
    the problem is on the HOST (no address on its usb0/enx interface).
  * UDC empty and role "host"                          -> VBUS detect never
    asserted. Try --force; if that sticks, patch the DTB to peripheral mode.
  * No role switch at all                              -> usb2-0 is not in OTG
    mode in the running DTB.

Host side, once the board says "device":
    ip -br link                       # find the new usb0 / enx... interface
    sudo ip addr add 192.168.55.100/24 dev <iface>
    ping -c3 192.168.55.1
EOF
