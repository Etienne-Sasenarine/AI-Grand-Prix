#!/usr/bin/env python3
"""MSP (MultiWii Serial Protocol) client for a Betaflight FC over UART.

Speaks MSPv1 (``$M<``) and MSPv2 (``$X<``) and runs a background reader thread so
telemetry polling and RC transmission never block each other on the same port.

Wire format
-----------
v1 out:  '$' 'M' '<' len(u8) cmd(u8) payload[len]  crc=XOR(len,cmd,payload)
v1 in:   '$' 'M' '>'|'!' len(u8) cmd(u8) payload[len] crc
v2 out:  '$' 'X' '<' flag(u8) cmd(u16le) len(u16le) payload[len] crc8_dvb_s2(flag..payload)
v2 in:   '$' 'X' '>'|'!' flag  cmd(u16le) len(u16le) payload[len] crc

'!' means the FC parsed the frame but rejected the command (unknown / bad payload).

Requires: pip install pyserial
"""

from __future__ import annotations

import queue
import struct
import threading
import time

import serial

# --------------------------------------------------------------------------------------
# Command IDs (betaflight/src/main/msp/msp_protocol.h)
# --------------------------------------------------------------------------------------
MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_BOARD_INFO = 4
MSP_BUILD_INFO = 5
MSP_NAME = 10
MSP_SET_ARMING_DISABLED = 99
MSP_STATUS = 101
MSP_RAW_IMU = 102
MSP_MOTOR = 104
MSP_RC = 105
MSP_RAW_GPS = 106
MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_ANALOG = 110
MSP_BOXNAMES = 116
MSP_BOXIDS = 119
MSP_STATUS_EX = 150
MSP_UID = 160
MSP_SET_RAW_RC = 200

# Permanent box ID of the ARM mode. Its position inside MSP_BOXIDS gives us the bit
# index of ARM within MSP_STATUS.flight_mode_flags.
BOX_ARM_PERMANENT_ID = 0

# Bits of MSP_STATUS.sensor_flags.
SENSOR_ACC = 1 << 0
SENSOR_BARO = 1 << 1
SENSOR_MAG = 1 << 2
SENSOR_GPS = 1 << 3
SENSOR_RANGEFINDER = 1 << 4
SENSOR_GYRO = 1 << 5

# armingDisableFlags_e, Betaflight 4.4/4.5. Order matters: index == bit position.
# Older/newer firmware may shift the tail of this list; decode is best-effort and any
# unknown bit is reported as "BIT_n" rather than silently dropped.
ARMING_DISABLE_FLAGS = (
    "NO_GYRO", "FAILSAFE", "RX_FAILSAFE", "BAD_RX_RECOVERY", "BOXFAILSAFE",
    "RUNAWAY_TAKEOFF", "CRASH_DETECTED", "THROTTLE", "ANGLE", "BOOT_GRACE_TIME",
    "NOPREARM", "LOAD", "CALIBRATING", "CLI", "CMS_MENU", "BST", "MSP", "PARALYZE",
    "GPS", "RESC", "RPMFILTER", "REBOOT_REQUIRED", "DSHOT_BITBANG", "ACC_CALIBRATION",
    "MOTOR_PROTOCOL", "ARM_SWITCH",
)


class MSPError(IOError):
    """FC replied '!' (command rejected) or the reply was malformed."""


class MSPTimeout(TimeoutError):
    """No reply within the deadline."""


def crc8_dvb_s2(data, crc=0):
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_v1(cmd, payload=b""):
    if len(payload) > 255:
        raise ValueError("MSPv1 payload limited to 255 bytes; use MSPv2")
    body = bytes((len(payload), cmd)) + payload
    crc = 0
    for b in body:
        crc ^= b
    return b"$M<" + body + bytes((crc,))


def encode_v2(cmd, payload=b"", flag=0):
    body = struct.pack("<BHH", flag, cmd, len(payload)) + payload
    return b"$X<" + body + bytes((crc8_dvb_s2(body),))


# --------------------------------------------------------------------------------------
# Incremental parser
# --------------------------------------------------------------------------------------
class _Decoder:
    """Feed bytes, get (cmd, payload, ok) tuples. Never blocks, never raises on junk."""

    def __init__(self):
        self._buf = bytearray()
        self.crc_errors = 0

    def feed(self, chunk):
        self._buf.extend(chunk)
        out = []
        while True:
            frame = self._try_one()
            if frame is None:
                return out
            out.append(frame)

    def _try_one(self):
        buf = self._buf
        # Resync: drop everything before the next '$'.
        start = buf.find(0x24)
        if start < 0:
            buf.clear()
            return None
        if start:
            del buf[:start]
        if len(buf) < 3:
            return None

        proto, direction = buf[1], buf[2]
        if direction not in (0x3E, 0x21):  # '>' '!'
            del buf[:1]
            return None

        if proto == 0x4D:  # 'M' -> v1
            if len(buf) < 5:
                return None
            size, cmd = buf[3], buf[4]
            total = 6 + size
            if len(buf) < total:
                return None
            payload = bytes(buf[5:5 + size])
            crc = 0
            for b in buf[3:5 + size]:
                crc ^= b
            got = buf[total - 1]
        elif proto == 0x58:  # 'X' -> v2
            if len(buf) < 8:
                return None
            _flag, cmd, size = struct.unpack_from("<BHH", buf, 3)
            total = 9 + size
            if len(buf) < total:
                return None
            payload = bytes(buf[8:8 + size])
            crc = crc8_dvb_s2(buf[3:8 + size])
            got = buf[total - 1]
        else:
            del buf[:1]
            return None

        del buf[:total]
        if crc != got:
            self.crc_errors += 1
            return None
        return cmd, payload, direction == 0x3E


# --------------------------------------------------------------------------------------
# Link
# --------------------------------------------------------------------------------------
class MSPLink:
    """Thread-safe MSP connection.

    One background thread drains the serial port. ``request()`` blocks the calling
    thread until the matching reply arrives; ``send()`` is fire-and-forget and is what
    the RC loop uses (MSP_SET_RAW_RC replies with an empty ack we do not wait on).
    """

    def __init__(self, port, baudrate=115200, timeout=0.5, use_v2=False):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.use_v2 = use_v2

        self._ser = None
        self._decoder = _Decoder()
        self._rx_thread = None
        self._running = False
        self._write_lock = threading.Lock()
        # cmd -> Queue of (payload, ok)
        self._waiters = {}
        self._waiters_lock = threading.Lock()
        self._arm_bit = None

        self.rx_frames = 0
        self.tx_frames = 0

    # -- lifecycle ---------------------------------------------------------------------
    def open(self):
        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=0.02,
            write_timeout=0.5,
            exclusive=True,
        )
        time.sleep(0.15)  # let the FC's UART settle before we trust the stream
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, name="msp-rx", daemon=True)
        self._rx_thread.start()
        return self

    def close(self):
        self._running = False
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @property
    def is_open(self):
        return self._ser is not None and self._ser.is_open

    @property
    def crc_errors(self):
        """Frames the decoder threw away for a bad checksum since open()."""
        return self._decoder.crc_errors

    # -- io ----------------------------------------------------------------------------
    def _rx_loop(self):
        while self._running:
            try:
                chunk = self._ser.read(max(1, self._ser.in_waiting))
            except (serial.SerialException, OSError, TypeError):
                break
            if not chunk:
                continue
            for cmd, payload, ok in self._decoder.feed(chunk):
                self.rx_frames += 1
                with self._waiters_lock:
                    q = self._waiters.get(cmd)
                if q is not None:
                    q.put((payload, ok))

    def send(self, cmd, payload=b""):
        """Fire-and-forget. Used for the RC stream where latency beats confirmation."""
        frame = encode_v2(cmd, payload) if self.use_v2 else encode_v1(cmd, payload)
        with self._write_lock:
            self._ser.write(frame)
        self.tx_frames += 1

    def request(self, cmd, payload=b"", timeout=None, retries=2):
        """Send ``cmd`` and return its reply payload. Raises MSPTimeout / MSPError."""
        deadline = self.timeout if timeout is None else timeout
        q = queue.Queue(maxsize=4)
        with self._waiters_lock:
            self._waiters[cmd] = q
        try:
            for attempt in range(retries + 1):
                while not q.empty():   # discard anything stale from a previous attempt
                    q.get_nowait()
                self.send(cmd, payload)
                try:
                    reply, ok = q.get(timeout=deadline)
                except queue.Empty:
                    continue
                if not ok:
                    raise MSPError(f"FC rejected MSP command {cmd}")
                return reply
            raise MSPTimeout(f"no reply to MSP command {cmd} after {retries + 1} attempts")
        finally:
            with self._waiters_lock:
                self._waiters.pop(cmd, None)

    # -- identity ----------------------------------------------------------------------
    def api_version(self):
        p = self.request(MSP_API_VERSION)
        return (p[0], p[1], p[2])  # (msp_protocol, api_major, api_minor)

    def fc_variant(self):
        return self.request(MSP_FC_VARIANT)[:4].decode("ascii", "replace")

    def fc_version(self):
        p = self.request(MSP_FC_VERSION)
        return f"{p[0]}.{p[1]}.{p[2]}"

    def board_id(self):
        p = self.request(MSP_BOARD_INFO)
        return p[:4].decode("ascii", "replace")

    def uid(self):
        """The FC's unique hardware serial, as 24 lowercase hex characters.

        MSP_UID is the MCU's factory-programmed 96-bit unique device ID (three
        32-bit words), which is what Betaflight Configurator shows and what
        identifies one airframe's controller from another. It is burned into
        silicon, so it survives reflashing, config wipes and target changes --
        unlike the craft name, which is whatever the last person typed.
        """
        p = self.request(MSP_UID)
        if len(p) < 12:
            raise MSPError(f"MSP_UID too short ({len(p)} bytes)")
        return "".join(f"{w:08x}" for w in struct.unpack("<3I", p[:12]))

    def craft_name(self):
        return self.request(MSP_NAME).decode("ascii", "replace")

    def build_info(self):
        """(build_date, build_time, git_revision) as the firmware reports them.

        Betaflight packs this as 11 bytes of date, 8 of time, then a 7-character
        short git revision. That revision is how you tell one build from another
        when the version number is identical -- e.g. confirming a custom firmware
        was actually flashed.
        """
        p = self.request(MSP_BUILD_INFO)
        date = p[0:11].decode("ascii", "replace").strip()
        clock = p[11:19].decode("ascii", "replace").strip()
        rev = p[19:26].decode("ascii", "replace").strip()
        return date, clock, rev

    # -- telemetry ---------------------------------------------------------------------
    def box_ids(self):
        """Permanent box IDs, in the same order as the bits of flight_mode_flags."""
        return list(self.request(MSP_BOXIDS))

    def arm_bit(self):
        """Bit index of ARM inside flight_mode_flags. Cached; falls back to 0."""
        if self._arm_bit is None:
            try:
                self._arm_bit = self.box_ids().index(BOX_ARM_PERMANENT_ID)
            except (ValueError, MSPError, MSPTimeout):
                self._arm_bit = 0
        return self._arm_bit

    def status(self):
        """Parse MSP_STATUS. Tolerant of the trailing fields moving between releases."""
        p = self.request(MSP_STATUS)
        if len(p) < 11:
            raise MSPError(f"MSP_STATUS too short ({len(p)} bytes)")
        cycle_us, i2c_errors, sensors, mode_flags, pid_profile = struct.unpack_from("<HHHIB", p, 0)
        out = {
            "cycle_time_us": cycle_us,
            "i2c_errors": i2c_errors,
            "sensor_flags": sensors,
            "flight_mode_flags": mode_flags,
            "pid_profile": pid_profile,
            "system_load_pct": None,
            "arming_disable_flags": None,
            "arming_disable_reasons": [],
        }
        i = 11
        if len(p) >= i + 2:
            out["system_load_pct"] = struct.unpack_from("<H", p, i)[0]
            i += 2
        i += 2                      # gyro cycle time (u16), unused
        if len(p) > i:
            i += 1 + p[i]           # extended flight mode flags: count byte + that many bytes
        if len(p) > i:
            i += 1                  # arming disable flag count
        if len(p) >= i + 4:
            flags = struct.unpack_from("<I", p, i)[0]
            out["arming_disable_flags"] = flags
            out["arming_disable_reasons"] = decode_arming_flags(flags)

        out["armed"] = bool(mode_flags & (1 << self.arm_bit()))
        return out

    def attitude(self):
        """(roll_deg, pitch_deg, yaw_deg). Yaw is 0..360 as reported by the FC."""
        roll, pitch, yaw = struct.unpack("<3h", self.request(MSP_ATTITUDE)[:6])
        return roll / 10.0, pitch / 10.0, float(yaw)

    def raw_imu(self):
        """(acc[3], gyro[3], mag[3]) in raw FC units."""
        v = struct.unpack("<9h", self.request(MSP_RAW_IMU)[:18])
        return v[0:3], v[3:6], v[6:9]

    def altitude(self):
        """(altitude_m, vario_m_s) from the FC's estimator."""
        alt_cm, vario_cm_s = struct.unpack("<ih", self.request(MSP_ALTITUDE)[:6])
        return alt_cm / 100.0, vario_cm_s / 100.0

    def raw_gps(self):
        """MSP_RAW_GPS. All-zero when no GPS is fitted; ``fix`` says whether to trust it."""
        p = self.request(MSP_RAW_GPS)
        if len(p) < 16:
            raise MSPError(f"MSP_RAW_GPS too short ({len(p)} bytes)")
        fix, num_sat, lat, lon, alt_m, speed_cms, course_ddeg = struct.unpack_from(
            "<BBiiHHH", p, 0
        )
        return {
            "fix": fix,
            "num_sat": num_sat,
            "lat_deg": lat / 1e7,
            "lon_deg": lon / 1e7,
            "altitude_m": float(alt_m),
            "ground_speed_m_s": speed_cms / 100.0,
            "ground_course_deg": course_ddeg / 10.0,
        }

    def analog(self):
        """Battery / RSSI. Prefers the 0.01 V field when the firmware sends it."""
        p = self.request(MSP_ANALOG)
        volts = p[0] / 10.0                       # legacy 0.1 V field
        mah, rssi, amps = struct.unpack_from("<HHh", p, 1)
        if len(p) >= 9:                           # Betaflight appends 0.01 V (9 bytes total)
            volts = struct.unpack_from("<H", p, 7)[0] / 100.0
        return {
            "voltage_v": volts,
            "mah_drawn": mah,
            "rssi": rssi,
            "current_a": amps / 100.0,
        }

    def rc_channels(self):
        """Channel values the FC is actually acting on, in rcmap order."""
        p = self.request(MSP_RC)
        return list(struct.unpack(f"<{len(p) // 2}H", p[:len(p) // 2 * 2]))

    def motors(self):
        return list(struct.unpack("<8H", self.request(MSP_MOTOR)[:16]))

    # -- control -----------------------------------------------------------------------
    def set_raw_rc(self, channels):
        """MSP_SET_RAW_RC. ``channels`` are microseconds, rcmap order (default AETR)."""
        n = len(channels)
        if not 4 <= n <= 18:
            raise ValueError("MSP_SET_RAW_RC takes 4..18 channels")
        self.send(MSP_SET_RAW_RC, struct.pack(f"<{n}H", *(int(c) for c in channels)))

    def set_arming_disabled(self, disabled, disable_runaway_takeoff=False):
        """Betaflight blocks arming while an MSP client is connected; clear that here.

        ``disabled=False`` releases the MSP arming lock. It does NOT override any other
        arming-disable reason (throttle high, angle, calibrating, ...).
        """
        payload = bytes((1 if disabled else 0, 1 if disable_runaway_takeoff else 0))
        try:
            self.request(MSP_SET_ARMING_DISABLED, payload, timeout=0.3, retries=1)
        except MSPError:
            pass  # firmware without the command; the FC simply never set the lock


def decode_arming_flags(flags):
    if not flags:
        return []
    out = []
    for bit in range(32):
        if flags & (1 << bit):
            out.append(ARMING_DISABLE_FLAGS[bit] if bit < len(ARMING_DISABLE_FLAGS) else f"BIT_{bit}")
    return out


def decode_sensor_flags(flags):
    names = [
        (SENSOR_GYRO, "GYRO"), (SENSOR_ACC, "ACC"), (SENSOR_BARO, "BARO"),
        (SENSOR_MAG, "MAG"), (SENSOR_GPS, "GPS"), (SENSOR_RANGEFINDER, "RANGEFINDER"),
    ]
    return [name for bit, name in names if flags & bit]
