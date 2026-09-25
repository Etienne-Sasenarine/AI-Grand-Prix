#!/bin/bash
#
# Run ON THE JETSON. Installs everything the board needs for camera work and
# flight-controller comms. Idempotent -- safe to re-run.
#
# Needs network. If the board has none, bake these in host-side instead with
# build/stage-rootfs-extras.sh + stage-app-stack.sh before flashing.
#
# Usage: sudo ./provision.sh [-u USER] [-m POWERMODE] [--no-python]
#
set -euo pipefail

USERNAME="${SUDO_USER:-$(logname 2>/dev/null || echo '')}"
MODE="25W"; WANT_PY=1

while [ $# -gt 0 ]; do
	case "$1" in
		-u) USERNAME="$2"; shift 2 ;;
		-m) MODE="$2"; shift 2 ;;
		--no-python) WANT_PY=0; shift ;;
		*) sed -n '3,12p' "$0"; exit 1 ;;
	esac
done
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "NVIDIA apt source"
# Shipped with a literal <SOC> placeholder that makes `apt update` fail on that
# line until it is substituted. t234 = Orin.
SRC=/etc/apt/sources.list.d/nvidia-l4t-apt-source.list
if [ -f "$SRC" ] && grep -q '<SOC>' "$SRC"; then
	sed -i 's|<SOC>|t234|' "$SRC"; echo "   substituted <SOC> -> t234"
else
	echo "   already correct"
fi
apt-get update

say "camera + debug tooling"
apt-get install -y --no-install-recommends \
	v4l-utils i2c-tools gpiod gstreamer1.0-tools \
	gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
	python3-gi gir1.2-gstreamer-1.0

say "nvarguscamerasrc (Bayer sensors: imx219/imx477)"
# Not in the BSP tarball; only from NVIDIA's apt repo.
apt-get install -y --no-install-recommends nvidia-l4t-gstreamer || \
	echo "   !! unavailable -- Bayer cameras will have no ISP path (v4l2 raw still works)"

say "OpenCV"
# Prefer the packaged build: it has GStreamer, which pip's opencv-python does
# NOT -- and without it cv2.CAP_GSTREAMER pipelines fail at runtime.
#
# `nvidia-opencv` is the C++ libraries ONLY. On a JetPack image it is usually
# already installed, so apt reports success, installs nothing, and there is
# still no cv2 module. The Python binding is `libopencv-python`, a separate
# package. Test the import, not the exit status.
if ! python3 -c 'import cv2' 2>/dev/null; then
	apt-get install -y --no-install-recommends nvidia-opencv libopencv-python || true
fi
if ! python3 -c 'import cv2' 2>/dev/null; then
	# Ubuntu's binding pulls its own libopencv, which can collide with
	# NVIDIA's -- only worth it if the NVIDIA one is genuinely unavailable.
	apt-get install -y --no-install-recommends python3-opencv || true
fi
if ! python3 -c 'import cv2' 2>/dev/null; then
	echo "   !! no cv2 -- the camera viewers will not run. Do NOT 'pip install"
	echo "      opencv-python': that build has no GStreamer and every pipeline"
	echo "      fails at runtime. Check: apt-cache policy libopencv-python"
elif ! python3 - <<-'PY' 2>/dev/null
	import cv2, re, sys
	# The field is column-aligned and the alignment varies between builds, so
	# match the label and whatever follows it rather than a fixed string.
	m = re.search(r"^\s*GStreamer:\s*(.+)$", cv2.getBuildInformation(), re.M)
	sys.exit(0 if m and not m.group(1).strip().startswith("NO") else 1)
	PY
then
	echo "   !! cv2 present but built WITHOUT GStreamer -- cv2.CAP_GSTREAMER"
	echo "      pipelines will fail. A pip opencv-python usually shadows the"
	echo "      packaged one: pip3 uninstall opencv-python opencv-python-headless"
else
	echo "   cv2 $(python3 -c 'import cv2; print(cv2.__version__)') with GStreamer"
fi

if [ "$WANT_PY" = 1 ]; then
	say "MAVLink 2"
	apt-get install -y --no-install-recommends python3-pip
	# python3-serial and python3-rich are installed WITHOUT
	# --no-install-recommends: the ask was for these plus everything they pull
	# in, and recommends are exactly what that flag would drop.
	apt-get install -y python3-serial python3-rich
	apt-get install -y --no-install-recommends python3-pymavlink 2>/dev/null || true
	python3 -c 'import pymavlink' 2>/dev/null || pip3 install --no-cache-dir pymavlink
	apt-get install -y --no-install-recommends mavproxy 2>/dev/null || \
		echo "   mavproxy not in apt (optional; pip install MAVProxy)"
fi

say "power mode -> $MODE"
if command -v nvpmodel >/dev/null; then
	conf=$(readlink -f /etc/nvpmodel.conf)
	if [[ "$MODE" =~ ^[0-9]+$ ]]; then id="$MODE"; else
		id=$(grep -oE "^< POWER_MODEL ID=[0-9]+ NAME=${MODE} >" "$conf" |
		     grep -oE 'ID=[0-9]+' | cut -d= -f2 | head -1)
	fi
	if [ -n "${id:-}" ]; then
		nvpmodel -m "$id" || true
		echo "   $(nvpmodel -q | tr '\n' ' ')"
	else
		echo "   '$MODE' not offered by $(basename "$conf"); available:"
		grep -oE 'NAME=[A-Za-z0-9_]+' "$conf" | cut -d= -f2 | sed 's/^/     /'
	fi
fi

if [ -n "$USERNAME" ] && id "$USERNAME" >/dev/null 2>&1; then
	say "groups for '$USERNAME'"
	for g in dialout video i2c gpio plugdev; do
		getent group "$g" >/dev/null && usermod -aG "$g" "$USERNAME" && echo "   + $g"
	done
	echo "   (takes effect at next login)"
fi

say "summary"
for c in v4l2-ctl media-ctl i2cdetect gpioinfo gst-launch-1.0; do
	printf '   %-16s %s\n' "$c" "$(command -v $c || echo MISSING)"
done
python3 - <<'PY'
import re
try:
    import cv2
    info = cv2.getBuildInformation()
    g = re.search(r"^\s*GStreamer:\s*(\S+)", info, re.M)
    print(f"   cv2              {cv2.__version__}  GStreamer={g.group(1) if g else '?'}")
except Exception as e:
    print("   cv2              MISSING:", e)
try:
    import pymavlink; print(f"   pymavlink        {pymavlink.__version__}")
except Exception as e:
    print("   pymavlink        MISSING:", e)
for mod, label in (("serial", "pyserial"), ("rich", "rich")):
    try:
        m = __import__(mod)
        print(f"   {label:16s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"   {label:16s} MISSING:", e)
PY

cat <<'EOF'

Next: ./signoff.sh   (acceptance test + still image + report)
      ./live-view.py (live stream in a browser)

If you wire the flight controller to /dev/ttyTHS0, free the UART first:
      sudo systemctl disable --now nvgetty
EOF
