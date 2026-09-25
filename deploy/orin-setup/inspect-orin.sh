#!/bin/sh
# Read-only inventory. Does not open the flight-controller serial port or command motors.
set -eu
printf '\n--- Board ---\n'
if [ -r /proc/device-tree/model ]; then tr -d '\000' < /proc/device-tree/model; printf '\n'; fi
uname -srmo
if [ -r /etc/nv_tegra_release ]; then cat /etc/nv_tegra_release; fi
printf '\n--- OS ---\n'
cat /etc/os-release
printf '\n--- User and serial access ---\n'
id
printf '\n--- USB devices ---\n'
if command -v lsusb >/dev/null 2>&1; then lsusb; fi
printf '\n--- Serial device paths ---\n'
for device in /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB* /dev/ttyTHS*; do
    if [ -e "$device" ]; then ls -l "$device"; fi
done
printf '\n--- Python ---\n'
if command -v python3 >/dev/null 2>&1; then
    python3 --version
    python3 -c 'import importlib.util; print("pyserial installed:", importlib.util.find_spec("serial") is not None)'
fi
printf '\n--- Network ---\n'
ip -brief address
