# MSP — talking to a Betaflight flight controller

Imported from the CommBetaFlight project. This is the **MSP subset only**: the
files here have no dependency on the rest of that project, just on each other
plus `pyserial` and `rich`, both of which are baked into the flashed image.

| File | What it is |
|---|---|
| `msp.py` | MSP v1/v2 protocol + `MSPLink`, a thread-safe connection |
| `msp_rc.py` | `RCTransmitter` — RC channel encoding, `MSP_SET_RAW_RC` |
| `msp_bench.py` | round-trip latency / throughput benchmark |
| `companion_listener_msp.py` | companion-computer listener over MSP |
| `test_msp_loopback.py` | loopback test, no FC required (uses a pty) |
| `test_companion_listener_msp.py` | listener test |
| `setup_jetson_uart.sh` | prepares the Jetson UART: permissions, console, udev |

## Why MSP and not MAVLink here

Betaflight speaks **MSP**, not MAVLink. `signoff.sh --mavlink` uses `pymavlink`
and is for an ArduPilot/PX4 autopilot; for a Betaflight FC use
`signoff.sh --msp` instead. Both are present because which one applies depends
on the flight controller, not on the Jetson.

## First run

The UART must be freed from the serial console before anything will work:

```bash
./setup_jetson_uart.sh                  # inspect only, changes nothing
sudo ./setup_jetson_uart.sh --apply     # fix permissions, console, udev rule
```

Then verify the link:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from msp import MSPLink
with MSPLink('/dev/ttyTHS1', 115200) as fc:
    print('API', fc.api_version(), 'variant', fc.fc_variant(), 'version', fc.fc_version())
"
```

No FC to hand? `test_msp_loopback.py` exercises the protocol over a pty.

## Camera + IMU on one page

`../live-view-imu.py` serves the camera stream with the FC's attitude burned
into the JPEG (a horizon indicator) and a numeric panel beside it fed from
`/imu.json`. That combination is the point: a screenshot is evidence that the
camera and the IMU were live *at the same instant*, which two separate tools
cannot give you.

    sudo ./setup_jetson_uart.sh --apply
    sudo ../signoff.sh --live --msp /dev/ttyTHS1      # full acceptance run
    ../live-view-imu.py --msp /dev/ttyTHS1            # viewer on its own

Then open `http://<jetson-ip>:8080/`.

The two halves fail independently. No FC still gives you video, with a red
**NO LINK** panel; the panel also distinguishes **STALE** (port open, FC gone
quiet) from **DOWN** (port unusable), because a link that stops replying must
never keep displaying the last attitude as though it were current.

Scaling follows Betaflight's `MSP_RAW_IMU` convention — accel 512 counts = 1 G,
gyro already in deg/s. INAV and older firmware differ, so the raw counts are
shown alongside the converted values rather than replaced by them.

Validate the IMU half with no flight controller and no camera attached:

    python3 test_imu_panel.py

It runs a simulated Betaflight FC on a pty and checks that live attitude
reaches the panel, that a silenced FC is reported rather than frozen, and that
the link recovers when the FC comes back.

## IMU acceptance test

`imu_check.py` is the functional half of the sign-off. It goes well beyond
"did the FC answer", because a dead IMU usually still returns *plausible*
numbers — an all-zero gyro is indistinguishable from a perfectly still one, and
a frozen accel buffer holding 1 g looks like a level board.

    ./imu_check.py --dev /dev/ttyTHS1 [--motion] [--json imu.json]

| check | catches |
|---|---|
| FC reports ACC and GYRO | sensor not detected by the firmware at all |
| accel reads 1 g at rest | wrong scaling, dead axis, uncalibrated part |
| gyro near zero at rest | saturated or drifting gyro |
| **data is changing** | **a frozen buffer, which passes every check above** |
| MSP CRC errors | marginal wiring or baud mismatch |
| MSP_UID readable | the serial the report is stamped with |

`--motion` additionally asks the operator to rotate the board and checks that
the gyro and the attitude estimate both follow.

Verify the test itself, with no hardware:

    python3 test_imu_check.py

`fake_fc.py` simulates a Betaflight FC on a pty in each fault mode — healthy,
frozen, saturated, badscale, noaccel, i2cerr, nouid — and the harness asserts
that each fault is caught, and caught by the *right* check. The frozen case is
the one that matters: it passes the 1 g test and the gyro-at-rest test, and is
caught only by the liveness check.
