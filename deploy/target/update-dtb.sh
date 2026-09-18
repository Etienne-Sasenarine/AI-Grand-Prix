#!/bin/bash
#
# !!  DOES NOT WORK ON THIS PLATFORM AS FLASHED.  MEASURED, NOT THEORISED.  !!
#
# Writing the kernel-dtb partition has NO EFFECT on an unsigned JetPack 6
# flash.  Two things conspire:
#
#   * flash.sh only appends a DTB signature when uefi_sign=True; otherwise it
#     prints "Not signing of kernel-dtb" (flash.sh:3971) and writes the blob
#     raw.
#   * L4TLauncher validates that signature.  Finding none it takes its
#     "DTB signature missing" / "DTB on partition was corrupted, attempt use
#     to UEFI DTB" path and uses the DTB embedded in the bootloader image
#     instead -- which flash.sh put there via TBCDTB_FILE="${DTB_FILE}".
#
# Net effect: the partition copy is ignored, and the authoritative device tree
# lives in A_cpu-bootloader.  A dd here verifies clean and changes nothing,
# which is a genuinely misleading failure mode.
#
# To change the device tree you must re-flash so TBCDTB is regenerated:
#   host$ sudo ./build/stage-bsp.sh ar0821
#   host$ cd Linux_for_Tegra && sudo ./tools/kernel_flash/l4t_initrd_flash.sh \
#           --external-device nvme0n1p1 -c tools/kernel_flash/flash_l4t_t234_nvme.xml \
#           -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
#           --showlogs --network usb0 jetson-orin-nano-devkit-super internal
#
# This script is kept only for signed-boot setups, or to stage the partition
# copy alongside a bootloader reflash.  It refuses to run without --i-know.
#
# Usage:  sudo ./update-dtb.sh <new.dtb> [slot]
#         slot defaults to the currently-booted one.
#
set -euo pipefail

if [ "${1:-}" = "--i-know" ]; then
	shift
else
	cat >&2 <<'EOF'
REFUSING: writing kernel-dtb does not change the booted device tree on an
unsigned JetPack 6 flash -- UEFI ignores the unsigned partition copy and uses
the DTB embedded in A_cpu-bootloader (TBCDTB).  See the comment at the top of
this script.  Re-flash instead; pass --i-know to override.
EOF
	exit 1
fi

DTB="${1:-}"
[ -f "$DTB" ] || { echo "usage: $0 --i-know <new.dtb> [A|B]" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "must run as root" >&2; exit 1; }

SLOT="${2:-}"
if [ -z "$SLOT" ]; then
	# nvbootctrl reports the active boot slot as 0 (A) or 1 (B).
	case "$(nvbootctrl get-current-slot 2>/dev/null || echo 0)" in
		1) SLOT=B ;;
		*) SLOT=A ;;
	esac
fi
PART="/dev/disk/by-partlabel/${SLOT}_kernel-dtb"

[ -e "$PART" ] || {
	echo "no $PART." >&2
	echo "Available partlabels:" >&2
	ls /dev/disk/by-partlabel/ >&2
	echo "If kernel-dtb is not here, this layout keeps it in QSPI --" >&2
	echo "use recovery mode: sudo ./flash.sh -k ${SLOT}_kernel-dtb jetson-orin-nano-devkit-super internal" >&2
	exit 1
}

SIZE=$(blockdev --getsize64 "$PART")
DSIZE=$(stat -c%s "$DTB")
echo "target : $PART ($SIZE bytes, slot $SLOT)"
echo "source : $DTB ($DSIZE bytes)"
[ "$DSIZE" -le "$SIZE" ] || { echo "DTB larger than the partition!" >&2; exit 1; }

BAK="/root/kernel-dtb-${SLOT}.bak"
if [ ! -f "$BAK" ]; then
	echo "backing up current DTB -> $BAK  (recovery needs this)"
	dd if="$PART" of="$BAK" bs=1M status=none
	# Keep a decompilable copy too, so you can inspect what was replaced.
	dtc -I dtb -O dts "$BAK" -o "${BAK%.bak}.dts" 2>/dev/null || true
else
	echo "backup already exists: $BAK (not overwriting)"
fi

echo "writing..."
# Write the DTB FIRST, and never pre-wipe with a fixed block count.
#
# The earlier version zeroed with `bs=1M count=1`, which overruns a 768 KiB
# partition: dd writes the partition full of zeros, then fails ENOSPC on the
# remainder, and `set -e` aborts BEFORE the real write -- leaving an erased,
# unbootable kernel-dtb.  A pre-wipe is unnecessary anyway: the FDT header
# carries its own totalsize, so any stale trailing bytes are ignored.
dd if="$DTB" of="$PART" bs=4096 conv=fsync status=none
sync

# Read back and compare -- a truncated or failed write must not reach a reboot.
if cmp -n "$DSIZE" "$DTB" "$PART"; then
	echo "verified OK"
else
	echo "READ-BACK MISMATCH -- restoring backup!" >&2
	dd if="$BAK" of="$PART" bs=4096 conv=fsync status=none; sync
	exit 1
fi

echo
echo "reboot to apply.  After boot, check it took with:"
echo "  ls /proc/device-tree/bus@0/i2c@3180000/"
echo "If the board fails to boot, recover from $BAK via USB recovery-mode flash."
