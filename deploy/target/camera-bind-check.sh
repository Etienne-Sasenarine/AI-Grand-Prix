#!/bin/bash
#
# Why is the camera driver not bound? — run ON THE JETSON.
#
# Walks the bind chain in order and stops at the first broken link, because
# every later symptom is downstream of it. "Driver not bound" is a conclusion,
# not a cause: it happens when the DT node is absent, when the i2c client is
# never instantiated, when the module cannot load, when the sensor does not
# answer on the bus, or when probe runs and fails. Those need different fixes.
#
#   1 device tree node        does the kernel see a sensor in the DT at all?
#   2 i2c adapter             is the controller up and numbered?
#   3 i2c client              did the kernel instantiate the device from the DT?
#   4 driver binding          is a driver attached to that client?
#   5 module                  is the driver even loadable?
#   6 bus response            does the sensor ACK its address?
#   7 probe log               what did the driver say when it tried?
#   8 platform stack          nvcsi / VI / tegra-camera-platform up?
#   9 reset line              is the sensor held in reset?
#
# Works for imx477, imx219 and tevs: the expected node and address are read
# from the live device tree rather than hardcoded.
#
# Usage: sudo ./camera-bind-check.sh
#
set -uo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

hdr() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()  { printf '   [ok]   %s\n' "$*"; }
bad() { printf '   [BAD]  %s\n' "$*"; FAULT="${FAULT:-$1}"; }
inf() { printf '          %s\n' "$*"; }
FAULT=""
# Pre-declare: under `set -u` an unset one aborts the run at the first
# reference, which would hide every section after the real fault.
NODE=""; COMPAT=""; ADDR=""; ST=""; BUS=""; CLIENT=""

I2CDT=/proc/device-tree/bus@0/i2c@3180000

hdr "1. device tree node"
if [ ! -d "$I2CDT" ]; then
	bad "no i2c@3180000 in the live device tree — wrong DTB flashed"
else
	NODE=""
	for n in "$I2CDT"/*/; do
		n="$(basename "$n")"
		case "$n" in
			rbpcv*imx*|tevs_*|*imx477*|*imx219*) NODE="$n"; break ;;
		esac
	done
	if [ -z "$NODE" ]; then
		bad "no sensor node under i2c@3180000"
		inf "present: $(ls "$I2CDT" | tr '\n' ' ')"
	else
		COMPAT="$(tr -d '\0' < "$I2CDT/$NODE/compatible" 2>/dev/null)"
		ADDR="$(printf '%02x' "$(od -An -tu1 -j3 -N1 "$I2CDT/$NODE/reg" 2>/dev/null | tr -d ' ')" 2>/dev/null)"
		ST="$(tr -d '\0' < "$I2CDT/$NODE/status" 2>/dev/null || echo okay)"
		ok "node $NODE  compatible=$COMPAT  reg=0x$ADDR  status=$ST"
		[ "$ST" = okay ] || bad "node status is '$ST', not okay"
	fi
fi

hdr "2. i2c adapter"
BUS="$(i2cdetect -l 2>/dev/null | awk '/3180000/{print $1}' | sed 's/i2c-//')"
if [ -n "$BUS" ]; then ok "3180000.i2c is /dev/i2c-$BUS"
else bad "no i2c adapter for 3180000.i2c — the controller did not probe"; fi

hdr "3. i2c client instantiated from the DT"
CLIENT=""
if [ -n "${BUS:-}" ] && [ -n "${ADDR:-}" ]; then
	CLIENT="$BUS-00$ADDR"
	if [ -d "/sys/bus/i2c/devices/$CLIENT" ]; then
		ok "client $CLIENT exists"
	else
		bad "no i2c client $CLIENT — the DT node did not become a device"
		inf "clients present: $(ls /sys/bus/i2c/devices/ 2>/dev/null | tr '\n' ' ')"
	fi
fi

hdr "4. driver binding"
if [ -z "$CLIENT" ] || [ ! -d "/sys/bus/i2c/devices/$CLIENT" ]; then
	inf "no client to check — fix the earlier failure first"
elif [ -e "/sys/bus/i2c/devices/$CLIENT/driver" ]; then
	ok "bound to $(basename "$(readlink -f "/sys/bus/i2c/devices/$CLIENT/driver")")"
else
	bad "client exists but NO driver is bound — this is the reported symptom"
fi

hdr "5. module"
MOD=""
case "$COMPAT" in *imx477*) MOD=nv_imx477 ;; *imx219*) MOD=nv_imx219 ;; *tevs*) MOD=tevs ;; esac
if [ -n "$MOD" ]; then
	if lsmod | grep -q "^${MOD//-/_}\b"; then
		ok "$MOD is loaded"
	else
		bad "$MOD is NOT loaded"
		inf "modinfo: $(modinfo -F filename "$MOD" 2>&1 | head -1)"
		inf "trying modprobe, errors below:"
		modprobe "$MOD" 2>&1 | sed 's/^/            /'
		lsmod | grep -q "^${MOD//-/_}\b" && inf "-> it loaded on demand; it simply was not autoloaded" \
			|| inf "-> it will not load; that is the root cause"
	fi
	inf "tegra-camera: $(lsmod | grep -c '^tegra_camera') loaded"
fi

hdr "6. does the sensor answer on the bus?"
if [ -n "${BUS:-}" ]; then
	i2cdetect -y -r "$BUS" 2>/dev/null | sed 's/^/          /'
	if [ -n "${ADDR:-}" ]; then
		# UU = claimed by a bound driver; a bare number = present but unclaimed;
		# -- = no response at all, which is electrical, not software.
		LINE="$(i2cdetect -y -r "$BUS" 2>/dev/null | grep -E "^$(printf '%d' 0x$ADDR | awk '{printf "%02x", int($1/16)*16}'):")"
		case "$LINE" in
			*UU*) ok "0x$ADDR is claimed by a driver" ;;
			*"$ADDR"*) ok "0x$ADDR responds but is unclaimed — driver side problem" ;;
			*) bad "0x$ADDR does NOT respond — power, reset or cable, not software" ;;
		esac
	fi
fi

hdr "7. probe log"
dmesg 2>/dev/null | grep -iE "imx477|imx219|tevs|tegracam|camera_common|nvcsi|tegra-capture-vi|fw_devlink" \
	| tail -25 | sed 's/^/          /'
DEF="$(dmesg 2>/dev/null | grep -ciE 'deferred|EPROBE_DEFER')"
[ "$DEF" -gt 0 ] && inf "$DEF deferral lines — a supplier that FAILS (not defers) blocks the consumer permanently"

hdr "8. platform stack"
for d in /sys/bus/platform/drivers/tegra194-vi /sys/bus/platform/drivers/tegra194-nvcsi \
         /sys/bus/platform/drivers/tegra-camera-platform; do
	[ -d "$d" ] && ok "$(basename "$d") present" || inf "$(basename "$d") absent"
done
ls /dev/video* /dev/media* 2>/dev/null | sed 's/^/          /' || inf "no /dev/video* or /dev/media* nodes"

hdr "9. reset line"
if command -v gpioinfo >/dev/null; then
	gpioinfo 2>/dev/null | grep -iE "cam|reset|PH\." | head -8 | sed 's/^/          /'
else
	inf "gpiod not installed (apt install gpiod)"
fi

hdr "verdict"
if [ -z "$FAULT" ]; then
	echo "   nothing obviously broken — re-read section 7 for the probe's own words."
else
	echo "   first broken link: $FAULT"
	cat <<'EOF'

   Reading it:
     no sensor node          -> wrong DTB flashed; re-run stage-stock.sh imx477
     no i2c client           -> DTB node malformed, or i2c controller failed
     driver not bound + 0x1a silent
                             -> electrical: camera power, reset (PH.6), FPC seating
                                or orientation. Not a software fix.
     driver not bound + 0x1a responds
                             -> module did not load, or probe failed: section 7
     module will not load    -> missing dependency or vermagic mismatch; the
                                message from modprobe above names it
EOF
fi
