#!/usr/bin/env bash
# Prepare a Jetson Orin NX (Seeed A603 carrier) UART for talking to a Betaflight FC.
#
# Dry-run by default: it only reports. Pass --apply to actually change anything.
#
#   ./setup_jetson_uart.sh                 # inspect
#   sudo ./setup_jetson_uart.sh --apply    # fix permissions, console, udev rule
#
set -uo pipefail

PORT="${PORT:-/dev/ttyTHS1}"
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --port=*) PORT="${arg#*=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
act()  { printf '  \033[36m->\033[0m   %s\n' "$*"; }

need_root() {
  if [[ $APPLY -eq 1 && $EUID -ne 0 ]]; then
    bad "--apply needs root. Re-run with sudo."
    exit 1
  fi
}
need_root

say "Platform"
if [[ -r /proc/device-tree/model ]]; then
  ok "$(tr -d '\0' < /proc/device-tree/model)"
else
  warn "no /proc/device-tree/model - this does not look like a Jetson"
fi
[[ -r /etc/nv_tegra_release ]] && ok "$(head -1 /etc/nv_tegra_release)"

say "Serial devices present"
# ttyTHS* = Tegra high-speed UARTs (the 40-pin header lives here)
# ttyTCU* = Tegra Combined UART, the debug console - do NOT use for the FC
# ttyUSB*/ttyACM* = USB-serial adapters
found=0
for d in /dev/ttyTHS* /dev/ttyTCU* /dev/ttyUSB* /dev/ttyACM*; do
  [[ -e "$d" ]] || continue
  found=1
  printf '  %-16s %s\n' "$d" "$(stat -c '%U:%G %a' "$d")"
done
[[ $found -eq 1 ]] || bad "no serial devices found at all"

say "Target port: $PORT"
if [[ ! -e "$PORT" ]]; then
  bad "$PORT does not exist."
  echo "     On Orin NX the 40-pin header UART1 (pin 8 TX / pin 10 RX) is normally"
  echo "     /dev/ttyTHS1. If it is missing, the pinmux may not expose UART1 - run"
  echo "     'sudo /opt/nvidia/jetson-io/jetson-io.py' and confirm the header config."
  exit 1
fi
ok "$PORT exists ($(stat -c '%U:%G %a' "$PORT"))"

say "Console / getty conflict"
# Anything else holding the port will silently eat MSP replies.
conflict=0
if command -v fuser >/dev/null 2>&1; then
  holders="$(fuser "$PORT" 2>/dev/null || true)"
  if [[ -n "$holders" ]]; then
    conflict=1
    names="$(ps -o comm= -p $holders 2>/dev/null | sort -u | head -5 | tr '\n' ' ')"
    bad "another process holds $PORT: ${names}(pids: $(echo $holders | cut -c1-60))"
    echo "     The Betaflight Configurator is the usual culprit - close it."
  else
    ok "no process is holding $PORT"
  fi
fi

unit="serial-getty@$(basename "$PORT").service"
if systemctl list-unit-files "$unit" >/dev/null 2>&1 && \
   systemctl is-enabled "$unit" >/dev/null 2>&1; then
  bad "$unit is enabled - it will fight you for the port"
  if [[ $APPLY -eq 1 ]]; then
    systemctl stop "$unit"; systemctl disable "$unit"
    act "stopped and disabled $unit"
  else
    act "fix: sudo systemctl disable --now $unit"
  fi
else
  ok "$unit is not enabled"
fi

# nvgetty binds ttyTCU0 (debug console) on Orin, not ttyTHS1. Only touch it if the
# port you actually chose is the one it owns.
if systemctl is-enabled nvgetty >/dev/null 2>&1; then
  if [[ "$PORT" == "/dev/ttyTCU0" ]]; then
    bad "nvgetty owns $PORT"
    if [[ $APPLY -eq 1 ]]; then
      systemctl stop nvgetty; systemctl disable nvgetty
      act "stopped and disabled nvgetty"
    else
      act "fix: sudo systemctl disable --now nvgetty"
    fi
  else
    ok "nvgetty is enabled but owns ttyTCU0, not $PORT - leave it alone"
  fi
fi

if grep -qE 'console=tty(TCU0|S0|THS[0-9])' /proc/cmdline 2>/dev/null; then
  c="$(tr ' ' '\n' < /proc/cmdline | grep '^console=' | tr '\n' ' ')"
  if [[ "$c" == *"$(basename "$PORT")"* ]]; then
    bad "kernel cmdline puts a console on $PORT: $c"
    act "fix: edit /boot/extlinux/extlinux.conf and drop that console= entry, then reboot"
  else
    ok "kernel console is on $c, not $PORT"
  fi
fi

say "Permissions"
grp="$(stat -c '%G' "$PORT")"
who="${SUDO_USER:-$USER}"
if id -nG "$who" | tr ' ' '\n' | grep -qx "$grp"; then
  ok "$who is in group '$grp'"
elif [[ "$grp" != "dialout" && "$grp" != "tty" ]]; then
  # Never hand out membership of a privileged group just to open a device.
  bad "$PORT is owned by group '$grp', which is not a serial group"
  act "do NOT add yourself to '$grp'. Check you picked the right device;"
  act "a real Tegra UART is owned by 'dialout'."
else
  bad "$who is NOT in group '$grp'"
  if [[ $APPLY -eq 1 ]]; then
    usermod -aG "$grp" "$who"
    act "added to '$grp' - log out and back in for it to take effect"
  else
    act "fix: sudo usermod -aG $grp $who   (then log out/in)"
  fi
fi

say "Stable device name"
RULE=/etc/udev/rules.d/99-betaflight.rules
if [[ -e "$RULE" ]]; then
  ok "$RULE already installed"
else
  warn "no udev rule; USB adapters can renumber between boots"
  if [[ $APPLY -eq 1 ]]; then
    cat > "$RULE" <<'RULEEOF'
# Stable /dev/betaflight symlink for the FC serial link.
# Tegra 40-pin header UART1 (pins 8/10):
KERNEL=="ttyTHS1", SYMLINK+="betaflight", GROUP="dialout", MODE="0660"
# FTDI FT232 USB-UART adapter (matches companion_listener.py's auto-detect):
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", \
  SYMLINK+="betaflight", GROUP="dialout", MODE="0660"
RULEEOF
    udevadm control --reload-rules && udevadm trigger
    act "installed $RULE -> use /dev/betaflight from now on"
  else
    act "fix: re-run with --apply to install $RULE"
  fi
fi

say "Next steps"
cat <<'NEXT'
  1. python3 msp/test_msp_loopback.py         # protocol self-test, no hardware
  2. python3 msp/msp_bench.py rc --port PORT  # verify control path, never arms
NEXT
[[ $APPLY -eq 0 ]] && printf '\n  (dry run - nothing was changed. Re-run with sudo ... --apply)\n'
exit 0
