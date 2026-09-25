#!/bin/bash
#
# Run ON THE JETSON.  Walks the TEVS bring-up checks in dependency order and
# stops at the first hard failure, so you always know which layer broke.
#
# Usage: sudo ./bringup-check.sh
#
set -uo pipefail

PASS=0 FAIL=0
ok()   { echo "  [ok]   $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "         $*"; }
hdr()  { echo; echo "== $* =="; }

DT=/proc/device-tree/bus@0/i2c@3180000

hdr "1. device tree"
if [ -d "$DT" ]; then
	SENSOR=$(ls "$DT" | grep -E '^tevs_a@48$' || true)
	if [ -n "$SENSOR" ]; then
		ok "sensor node present: $DT/$SENSOR"
		info "compatible: $(tr -d '\0' < "$DT/$SENSOR/compatible")"
	else
		bad "no tevs_a@48 under $DT -- the new DTB did not take"
		info "present: $(ls "$DT" | tr '\n' ' ')"
		echo; echo "Stopping: nothing downstream can work."; exit 1
	fi
	[ -d "$DT/pca9554_a@27" ] && ok "pca9554_a@27 declared" \
		|| info "no pca9554_a@27 in DT (fine if this build drops it)"
	[ -d /proc/device-tree/tegra-camera-platform/modules/module0 ] \
		&& ok "tegra-camera-platform/module0 present" \
		|| bad "tegra-camera-platform missing -- nvargus/bandwidth data absent"
else
	bad "$DT missing -- wrong DTB entirely"; exit 1
fi
info "booted model: $(tr -d '\0' < /proc/device-tree/model)"
info "compatible  : $(tr -d '\0' < /proc/device-tree/compatible)"

hdr "2. gpio hogs"
# Use gpioinfo, not /sys/kernel/debug/gpio: libgpiod reports the line name AND
# the consumer, e.g.
#   line  46: "PH.03" "camera-control-output-low" output active-high [used]
# whereas the debugfs dump formats hogs differently and cost two false FAILs
# earlier in this bring-up.
if command -v gpioinfo >/dev/null; then
	GI=$(gpioinfo 2>/dev/null)
	grep -qE '"(cam0-rst|camera-control-output-low)"' <<<"$GI" \
		&& ok "cam0-rst hog applied (PH.03 claimed)" \
		|| bad "cam0-rst hog not claimed"
	grep -qE '"(gpio06_high|GPIO06-high)"' <<<"$GI" \
		&& ok "gpio06_high hog applied (AON CC.03 claimed)" \
		|| bad "gpio06_high hog not claimed"
	# Reset line state is the useful datum when the camera is silent.
	info "reset line: $(grep -E '"PH\.06"' <<<"$GI" | sed 's/^[[:space:]]*//')"
else
	info "install gpiod (apt install gpiod) for the hog check"
fi

hdr "3. i2c -- run BEFORE loading tevs, or bound devices show as UU"
BUS=$(i2cdetect -l 2>/dev/null | awk '/3180000/{print $1}' | sed 's/i2c-//')
if [ -n "$BUS" ]; then
	ok "cam_i2c is /dev/i2c-$BUS"
	SCAN=$(i2cdetect -y -r "$BUS" 2>/dev/null)
	echo "$SCAN" | sed 's/^/         /'
	echo "$SCAN" | grep -qE ' (48|UU)' && ok "0x48 responds (TEVS ISP)" \
		|| bad "0x48 silent -- ISP held in reset? check FPC seating/orientation and the gpio06_high hog"
	if echo "$SCAN" | grep -qE ' (27|UU)'; then
		ok "0x27 responds (PCA9554 -- standby-gpios usable)"
	else
		info "0x27 absent: this RPI15 adapter has no PCA9554."
		info "  -> drop the pca9554 node + standby-gpios from the camera dtsi."
		info "  -> the tevs standby patch (IS_ERR_OR_NULL -> IS_ERR) makes that OK."
	fi
else
	bad "no i2c adapter for 3180000.i2c"
fi

hdr "4. driver probe"
# Do NOT swallow modprobe's stderr: a vermagic/CRC mismatch or a missing
# dependency is reported there and nowhere else, and it looks identical to
# "camera absent" if you only read dmesg.
if ! MPERR=$(modprobe tevs 2>&1); then
	bad "modprobe tevs failed: $MPERR"
else
	[ -n "$MPERR" ] && info "modprobe: $MPERR"
fi
lsmod | grep -qE '^tevs ' && ok "tevs module loaded" || bad "tevs not in lsmod"
ls /sys/bus/i2c/devices/ 2>/dev/null | grep -q '2-0048' \
	&& ok "i2c device 2-0048 instantiated from DT" \
	|| bad "no 2-0048 -- the i2c device was never created from the DT node"
sleep 1
LOG=$(dmesg | grep -i tevs | tail -20)
echo "$LOG" | sed 's/^/         /'
echo "$LOG" | grep -q 'probe success' && ok "tevs probe succeeded" || bad "tevs did not probe"
echo "$LOG" | grep -oE 'Product:[^,]*'   | tail -1 | sed 's/^/         /'
echo "$LOG" | grep -oE 'Chip ID: 0x[0-9A-Fa-f]+' | tail -1 | sed 's/^/         /'
# The module's OTP-programmed lane rate -- tells you whether raising
# link-frequencies to 600000000 (1200 Mbps/lane) is a legal operating point.
echo "$LOG" | grep -oE 'MIPI_Rate:[0-9]+' | tail -1 | sed 's/^/         note: /'
echo "$LOG" | grep -q 'Unknown symbol max_' && bad "max_serdes_all_tn.ko not installed"
echo "$LOG" | grep -q 'disagrees about version' && bad "MODVERSIONS CRC mismatch vs this kernel"

hdr "5. v4l2"
if [ -e /dev/video0 ]; then
	ok "/dev/video0 exists"
	v4l2-ctl -d /dev/video0 --list-formats-ext 2>/dev/null | sed 's/^/         /'
	media-ctl -p -d /dev/media0 2>/dev/null | grep -E 'entity|pad' | head -20 | sed 's/^/         /'
else
	bad "no /dev/video0"
fi

hdr "6. capture (the acceptance test)"
if [ -e /dev/video0 ]; then
	if v4l2-ctl -d /dev/video0 \
		--set-fmt-video=width=1920,height=1080,pixelformat=UYVY \
		--set-parm=30 --stream-mmap --stream-count=100 --stream-to=/dev/null 2>&1 \
		| tee /tmp/capture.log | sed 's/^/         /'; then
		grep -qi 'error\|timeout' /tmp/capture.log && bad "capture reported errors" \
			|| ok "captured 100 frames of 1920x1080 UYVY @30"
	else
		bad "capture failed"
		info "enable CSI/VI tracing:"
		info "  echo 1 > /sys/kernel/debug/tracing/tracing_on"
		info "  echo 1 > /sys/kernel/debug/tracing/events/tegra_rtcpu/enable"
		info "  echo 2 > /sys/kernel/debug/camrtc/log-level"
		info "  cat /sys/kernel/debug/tracing/trace"
		info "PXL_SOF timeout -> lane_polarity / FPC orientation."
		info "'no reply from camera processor' -> tegra_sinterface or port-index wrong."
	fi
fi

hdr "7. IMU (TEVM-AR0234 builds only)"
if [ -d /proc/device-tree/bus@0/i2c@3180000/lsm6dso16is@6a ]; then
	WHO=$(i2cget -y "${BUS:-2}" 0x6a 0x0f 2>/dev/null || echo "n/a")
	info "WHO_AM_I = $WHO (expect 0x22 for LSM6DSO16IS)"
	NAME=$(cat /sys/bus/iio/devices/iio:device0/name 2>/dev/null || true)
	if [ "$NAME" = lsm6dso16is ]; then
		ok "IIO device bound: $NAME"
	else
		info "not bound -- the shipped st_lsm6dsx_i2c.ko match table stops at"
		info "st,lsm6dsop; LSM6DSO16IS needs the v6.6 backport."
	fi
fi

echo; echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
