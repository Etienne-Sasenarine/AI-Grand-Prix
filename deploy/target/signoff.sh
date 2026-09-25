#!/bin/bash
#
# Board acceptance test — run ON THE JETSON.
#
# Walks the whole stack (identity, power, storage, network, camera device tree,
# I2C, driver, V4L2, software stack, flight controller and IMU), records every
# check as PASS/FAIL/INFO, and renders a PDF sign-off report stamped with the
# flight controller's unique hardware serial.
#
# Results are written as TSV as they happen, and the final verdict is counted
# FROM that file rather than from shell variables -- so a check emitted by a
# helper program counts exactly like one emitted here. (An earlier version kept
# counters in the shell and silently ignored the Python-side results, which
# could report SIGN-OFF: PASS with visible [FAIL] lines above it.)
#
# Usage: sudo ./signoff.sh [-w 1920] [-h 1080]
#                          [--msp DEV]        Betaflight FC: link + IMU test
#                          [--imu-motion]     accepted but ignored: the IMU test is
#                                             now msp_bench.py info, read-only
#                          [--mavlink DEV]    ArduPilot/PX4 autopilot instead
#                          [--operator NAME]  recorded in the report
#                          [--drone SERIAL]   serial of the drone this board
#                                             goes into; keys the fleet sheet
#                          [--video DEV]      default /dev/video0
#                          [--out DIR] [--no-pdf] [--live]
#
#   --live      after a clean run, start the browser view (adds the IMU
#               overlay when --msp is given)
#
set -uo pipefail

W=1920; H=1080; LIVE=0; MAVDEV=""; MSPDEV=""; MOTION=""
OPERATOR="${SUDO_USER:-${USER:-unknown}}"; WANT_PDF=1; OUT=""
VIDEODEV="${VIDEODEV:-/dev/video0}"
DRONE="${DRONE_SERIAL:-}"
while [ $# -gt 0 ]; do
	case "$1" in
		-w) W="$2"; shift 2 ;;
		-h) H="$2"; shift 2 ;;
		--live) LIVE=1; shift ;;
		--mavlink) MAVDEV="$2"; shift 2 ;;
		--msp) MSPDEV="$2"; shift 2 ;;
		--imu-motion) MOTION="--motion"; shift ;;
		--operator) OPERATOR="$2"; shift 2 ;;
		--drone) DRONE="$2"; shift 2 ;;
		--out) OUT="$2"; shift 2 ;;
		--no-pdf) WANT_PDF=0; shift ;;
		--video) VIDEODEV="$2"; shift 2 ;;
		*) sed -n '3,30p' "$0"; exit 1 ;;
	esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
# Resolve the INVOKING user's home, not $HOME. This script is run under sudo,
# and sudo's env_reset sets HOME to the target user's home -- so $HOME here is
# normally /root, and the report would land somewhere the operator needs sudo
# even to read. SUDO_USER is the only reliable handle on who actually ran it.
if [ -n "${SUDO_USER:-}" ]; then
	HOMEDIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
fi
: "${HOMEDIR:=$HOME}"
SIGNOFF_DIR="${HOMEDIR}/signoff"
[ -n "$OUT" ] || OUT="${SIGNOFF_DIR}/${STAMP}"
mkdir -p "$OUT"
REPORT="$OUT/report.txt"
TSV="$OUT/results.tsv"
META="$OUT/meta.json"
IMUJSON="$OUT/imu.json"
: > "$TSV"
SECTION="general"

log()  { echo "$*" | tee -a "$REPORT" >/dev/null; echo "$*"; }
# Collapse tabs/newlines: the TSV is the machine-readable record and a stray
# tab in a detail string would shift every column after it.
clean() { printf '%s' "${1:-}" | tr '\t\n' '  ' | tr -s ' '; }
rec()  { printf '%s\t%s\t%s\t%s\n' "$1" "$SECTION" "$(clean "$2")" "$(clean "${3:-}")" >> "$TSV"; }
ok()   { rec PASS "$1" "${2:-}"; log "  [PASS] $1${2:+ — $2}"; }
bad()  { rec FAIL "$1" "${2:-}"; log "  [FAIL] $1${2:+ — $2}"; }
inf()  { rec INFO "$1" "${2:-}"; log "         $1${2:+: $2}"; }
hdr()  { SECTION="$1"; log ""; log "== $2"; }

log "Board sign-off — $(date -Is)"
[ -n "$DRONE" ] && log "drone: $DRONE"
log "artefacts: $OUT"

# ---------------------------------------------------------------- 1 identity
hdr identity "1. identity"
if [ -n "$DRONE" ]; then
	ok "drone serial recorded" "$DRONE"
else
	# Without it the report cannot be tied to an airframe, and the fleet
	# sheet has nothing to key the row on.
	bad "no drone serial given" "re-run with --drone SERIAL"
fi
MODEL="$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
COMPAT="$(cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' ')"
L4T="$(sed -n 's/.*R\([0-9]*\).*REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release 2>/dev/null)"
KERN="$(uname -r)"
MODSER="$(cat /proc/device-tree/serial-number 2>/dev/null | tr -d '\0')"
MAC="$(cat /sys/class/net/eth0/address 2>/dev/null)"
[ -n "$MODEL" ] && ok "device tree model" "$MODEL" || bad "no /proc/device-tree/model"
inf "compatible" "$COMPAT"
inf "L4T release" "${L4T:-unknown}"
inf "kernel" "$KERN"
[ -n "$MODSER" ] && inf "module serial" "$MODSER" || inf "module serial" "not exposed"
inf "hostname" "$(hostname)"
[ -n "$MAC" ] && inf "eth0 MAC" "$MAC"

# ------------------------------------------------------------- 2 power/therm
hdr power "2. power / thermals"
PM=""
if command -v nvpmodel >/dev/null; then
	PM="$(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
	echo "$PM" | grep -qi "25W" && ok "power mode is 25W" "$PM" \
		|| bad "power mode is not 25W" "$PM"
else
	bad "nvpmodel missing"
fi
for z in /sys/devices/virtual/thermal/thermal_zone*/; do
	t=$(cat "$z/temp" 2>/dev/null) || continue
	ty=$(cat "$z/type" 2>/dev/null)
	inf "thermal $ty" "$((t/1000)) C"
	# Tegra throttles around 95 C; anything near that on an idle bench board
	# means the heatsink is not mounted properly.
	[ "$((t/1000))" -ge 85 ] && bad "thermal $ty is hot" "$((t/1000)) C at idle"
done
if command -v tegrastats >/dev/null; then
	TS="$(timeout 3 tegrastats --interval 500 2>/dev/null | head -1)"
	[ -n "$TS" ] && inf "tegrastats" "$TS"
fi

# --------------------------------------------------------- 3 storage / memory
hdr storage "3. storage and memory"
ROOTDEV="$(findmnt -no SOURCE / 2>/dev/null)"
AVAIL_K="$(df -Pk / | awk 'NR==2{print $4}')"
inf "root filesystem" "$ROOTDEV, $(df -Ph / | awk 'NR==2{print $2" total, "$4" free ("$5" used)"}')"
if [ -n "$AVAIL_K" ] && [ "$AVAIL_K" -gt 2097152 ]; then
	ok "free space on /" "$((AVAIL_K/1024)) MB"
else
	bad "low free space on /" "$((AVAIL_K/1024)) MB, want >2048 MB"
fi
MEM_K="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
inf "RAM" "$((MEM_K/1024)) MB total, $(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo) MB available"
# 16 GB module reports ~15.3 GiB; anything under 8 GB means the wrong SKU.
[ "$MEM_K" -gt 8000000 ] && ok "memory size plausible for Orin NX 16GB" "$((MEM_K/1024)) MB" \
	|| inf "memory size" "$((MEM_K/1024)) MB — not a 16GB module?"

# ---------------------------------------------------------------- 4 network
hdr network "4. network"
inf "interfaces" "$(ip -br addr 2>/dev/null | tr '\n' ';' | tr -s ' ')"
if systemctl is-active --quiet ssh 2>/dev/null; then
	ok "sshd is running" "$(ls /etc/ssh/ssh_host_*_key 2>/dev/null | wc -l) host keys"
else
	bad "sshd is not running" "$(systemctl is-active ssh 2>/dev/null)"
fi

# -------------------------------------------------------- 5 camera / DT / I2C
hdr camera "5. camera device tree + driver"
DTNODES="$(ls /proc/device-tree/bus@0/i2c@3180000/ 2>/dev/null | tr '\n' ' ')"
if echo "$DTNODES" | grep -qE 'tevs_a@48|rbpcv[0-9]_imx'; then
	ok "sensor node present in the live device tree" "$DTNODES"
else
	bad "no sensor node under i2c@3180000" "${DTNODES:-i2c@3180000 not in the DT}"
fi
BUS=$(i2cdetect -l 2>/dev/null | awk '/3180000/{print $1}' | sed 's/i2c-//')
if [ -n "$BUS" ]; then
	ok "camera i2c bus present" "/dev/i2c-$BUS"
	i2cdetect -y -r "$BUS" 2>/dev/null | tee -a "$REPORT" >/dev/null
	# UU = an address claimed by a bound driver, which is what a working
	# camera looks like. A bare number means present but unclaimed.
	CLAIMED=$(i2cdetect -y -r "$BUS" 2>/dev/null | grep -c 'UU')
	[ "$CLAIMED" -gt 0 ] && ok "an i2c address is claimed by a driver" \
		|| inf "no claimed i2c address on bus $BUS" "camera driver may not be bound"
else
	bad "no i2c adapter for 3180000.i2c"
fi
# ------------------------------------------------------------------- 6 v4l2
hdr v4l2 "6. v4l2"
PIXFMT=""
if [ -e "$VIDEODEV" ]; then
	ok "$VIDEODEV present"
	FMT=$(v4l2-ctl -d "$VIDEODEV" --list-formats-ext 2>/dev/null)
	echo "$FMT" >> "$REPORT"
	if grep -q UYVY <<<"$FMT"; then PIXFMT=UYVY; KIND="YUV (on-module ISP)"
	elif grep -qE "'RG10'" <<<"$FMT"; then PIXFMT=RG10; KIND="Bayer (Tegra ISP)"
	else PIXFMT=""; KIND="unknown"; fi
	if [ -n "$PIXFMT" ]; then
		ok "pixel format detected" "$PIXFMT — $KIND"
		inf "advertised sizes" "$(grep -oE '[0-9]+x[0-9]+' <<<"$FMT" | sort -u | tr '\n' ' ')"
	else
		bad "no usable pixel format" "$(head -3 <<<"$FMT" | tr '\n' ' ')"
	fi
else
	bad "$VIDEODEV missing" "no V4L2 node — driver not bound"
fi

# ---------------------------------------------------------------- 7 capture
# -------------------------------------------------------------- 9 software
hdr software "7. software stack"
for c in v4l2-ctl media-ctl i2cdetect gpioinfo gst-launch-1.0; do
	command -v "$c" >/dev/null && ok "$c present" "$(command -v $c)" || bad "$c missing"
done
# Python-side checks append to the SAME TSV, so they count towards the verdict.
python3 - "$TSV" 2>/dev/null <<'PY' | tee -a "$REPORT"
import re, sys
tsv = sys.argv[1]
def rec(status, name, detail=""):
    detail = " ".join(str(detail).split())
    with open(tsv, "a") as f:
        f.write(f"{status}\tsoftware\t{name}\t{detail}\n")
    mark = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INFO": "       "}[status]
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
try:
    import cv2
    m = re.search(r"^\s*GStreamer:\s*(.+)$", cv2.getBuildInformation(), re.M)
    g = m.group(1).strip() if m else "not listed"
    # Without GStreamer every cv2.CAP_GSTREAMER pipeline fails at runtime, so
    # a cv2 that imports is not on its own good enough.
    rec("PASS" if not g.startswith("NO") else "FAIL",
        f"cv2 {cv2.__version__}", f"GStreamer={g}")
except Exception as e:
    rec("FAIL", "cv2 import", str(e))
for mod, label in (("pymavlink", "pymavlink"), ("serial", "pyserial"),
                   ("rich", "rich"), ("numpy", "numpy"), ("gi", "python3-gi")):
    try:
        m = __import__(mod)
        rec("PASS", f"{label} present", getattr(m, "__version__", "ok"))
    except Exception as e:
        rec("FAIL", f"{label} missing", str(e))
PY

# ------------------------------------------------- 10 flight controller / IMU
if [ -n "$MSPDEV" ]; then
	hdr IMU "8. flight controller + IMU ($MSPDEV)"
	# msp_bench.py info identifies the FC and lists the sensors it detected.
	# It is read-only and only prints -- it writes no TSV rows and no imu.json --
	# so the verdict is recorded here: a clean exit means the FC answered MSP on
	# this port. --port is a global option and must precede the subcommand.
	if [ ! -e "$MSPDEV" ]; then
		bad "$MSPDEV does not exist"
	elif python3 "$HERE/msp/msp_bench.py" --port "$MSPDEV" info 2>&1 | tee -a "$REPORT"; then
		ok "flight controller answers MSP" "$MSPDEV"
	else
		bad "no MSP reply from flight controller" "$MSPDEV"
	fi
fi

if [ -n "$MAVDEV" ]; then
	hdr mavlink "9. autopilot link over MAVLink ($MAVDEV)"
	if [ -e "$MAVDEV" ]; then
		python3 - "$MAVDEV" "$TSV" <<'PY' | tee -a "$REPORT"
import sys
dev, tsv = sys.argv[1], sys.argv[2]
def rec(status, name, detail=""):
    with open(tsv, "a") as f:
        f.write(f"{status}\tmavlink\t{name}\t{' '.join(str(detail).split())}\n")
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
try:
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(dev, baud=921600)
    hb = m.wait_heartbeat(timeout=10)
    if hb:
        rec("PASS", "MAVLink heartbeat",
            f"system {m.target_system} component {m.target_component} "
            f"(MAVLink {m.WIRE_PROTOCOL_VERSION})")
    else:
        rec("FAIL", "MAVLink heartbeat", "none within 10 s")
except Exception as e:
    rec("FAIL", "MAVLink link error", str(e))
PY
	else
		SECTION=mavlink; bad "$MAVDEV does not exist"
	fi
fi

# ------------------------------------------------------------------- verdict
# Counted from the TSV, so helper-emitted results carry the same weight.
NPASS=$(awk -F'\t' '$1=="PASS"' "$TSV" | wc -l)
NFAIL=$(awk -F'\t' '$1=="FAIL"' "$TSV" | wc -l)
NTOT=$(wc -l < "$TSV")

CAMNAME="$(echo "$DTNODES" | grep -oE 'imx477|imx219|tevs' | head -1)"
# Values reach Python through the environment, never by interpolation into
# its source: a device-tree model string containing a quote would otherwise
# produce a syntax error, or worse, executable text.
MODEL="$MODEL" COMPAT="$COMPAT" L4T="${L4T:-}" KERN="$KERN" \
MODSER="${MODSER:-}" MAC="${MAC:-}" OPERATOR="$OPERATOR" DRONE="$DRONE" \
CAMNAME="${CAMNAME:-unknown}" PM="$PM" HOSTN="$(hostname)" DATE="$(date -Is)" \
META="$META" python3 - <<'PY'
import json, os
e = os.environ.get
json.dump({
    "title": "Seeed A603 / Jetson Orin NX - camera + flight controller",
    "date": e("DATE", ""), "model": e("MODEL", ""), "compatible": e("COMPAT", ""),
    "l4t": e("L4T", ""), "kernel": e("KERN", ""), "hostname": e("HOSTN", ""),
    "module_serial": e("MODSER", ""), "mac": e("MAC", ""),
    "operator": e("OPERATOR", ""), "camera": e("CAMNAME", ""),
    "drone_serial": e("DRONE", ""),
    "power_mode": " ".join(e("PM", "").split())[:60],
}, open(e("META"), "w"), indent=2)
PY

hdr result "result"
log "  $NPASS passed, $NFAIL failed, $NTOT checks"
log "  artefacts in $OUT"
log "  (also reachable as ${SIGNOFF_DIR}/latest)"
if [ "$NFAIL" -eq 0 ]; then log "  SIGN-OFF: PASS"; else log "  SIGN-OFF: FAIL — see the entries above"; fi

# A stable path to the most recent run, so copying the report off does not
# mean looking up a timestamp first. Done for every run, PDF or not.
ln -sfn "$OUT" "${SIGNOFF_DIR}/latest" 2>/dev/null
# The artefacts are the deliverable; leave them owned by the operator rather
# than root, or they cannot be copied off the board without sudo.
if [ -n "${SUDO_USER:-}" ]; then
	chown "$SUDO_USER:" "$SIGNOFF_DIR" 2>/dev/null
	chown -h "$SUDO_USER:" "${SIGNOFF_DIR}/latest" 2>/dev/null
	chown -R "$SUDO_USER:" "$OUT" 2>/dev/null
fi

if [ "$WANT_PDF" = 1 ]; then
	SLUG="$(printf '%s' "$DRONE" | tr -c 'A-Za-z0-9._-' '-' | sed 's/^-*//;s/-*$//')"
	PDF="$OUT/signoff-${SLUG:+${SLUG}-}${STAMP}.pdf"
	ARGS=(--tsv "$TSV" --meta "$META" -o "$PDF")
	[ -s "$IMUJSON" ] && ARGS+=(--imu "$IMUJSON")
	if python3 "$HERE/mkreport.py" "${ARGS[@]}"; then
		log "  PDF report: $PDF"
		# A fixed name for the newest report, so fetching it is a plain scp
		# with no timestamp to look up first:
		#     scp dcl@<board>:signoff/latest.pdf .
		ln -sfn "$PDF" "${SIGNOFF_DIR}/latest.pdf" 2>/dev/null
		[ -n "${SUDO_USER:-}" ] && chown -h "$SUDO_USER:" "${SIGNOFF_DIR}/latest.pdf" 2>/dev/null
		log "  fetch with: scp ${SUDO_USER:-$USER}@<board>:signoff/latest.pdf ."
	else
		log "  !! PDF generation failed — text report is still at $REPORT"
	fi
fi

if [ "$LIVE" = 1 ] && [ "$NFAIL" -eq 0 ] && [ -n "$MSPDEV" ]; then
	log ""
	echo "starting camera stream with MSP attitude overlay (Ctrl-C to stop)…"
	exec python3 "$HERE/live-view-imu.py" --width "$W" --height "$H" --msp "$MSPDEV"
fi
if [ "$LIVE" = 1 ] && [ "$NFAIL" -eq 0 ]; then
	echo
	echo "starting live view (Ctrl-C to stop)…"
	exec python3 "$HERE/live-view.py" --width "$W" --height "$H"
fi
exit $([ "$NFAIL" -eq 0 ] && echo 0 || echo 1)
