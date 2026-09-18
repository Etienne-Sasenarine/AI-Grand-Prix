#!/usr/bin/env python3
"""Render a sign-off PDF from the results signoff.sh collects.

Stdlib only — no reportlab, no Pillow, nothing to apt-get. A sign-off report
that cannot be produced because a dependency is missing is worse than no report
at all, and this board's whole point is that it ships as an image. PDF is a
simple enough container to emit directly; the same reasoning produced the
hand-rolled PNG writer in tools/raw2png.py.

Body text is Courier throughout. That is a deliberate layout decision, not a
stylistic one: wrapping text correctly needs per-character widths, and for a
monospaced base-14 font those are exactly 0.6 em, so line breaking is provably
right without shipping font metrics. Headings use Helvetica, where the strings
are short and fixed and never need wrapping.

The camera still is embedded losslessly by handing the JPEG bytes straight to
PDF's DCTDecode filter — no decode, no re-encode, no imaging library.

Usage:
    ./mkreport.py --tsv results.tsv --meta meta.json [--imu imu.json]
                  [--image still.jpg] [-o report.pdf]
"""
import argparse
import datetime
import json
import struct
import sys
import zlib

A4 = (595.28, 841.89)
MARGIN = 44.0
LEAD = 11.0                      # body line height
MONO_W = 0.6                     # Courier advance width, in em

GREY = (0.42, 0.42, 0.42)
BLACK = (0.11, 0.11, 0.11)
GREEN = (0.05, 0.48, 0.18)
RED = (0.75, 0.10, 0.10)
AMBER = (0.70, 0.45, 0.0)
RULE = (0.80, 0.80, 0.80)
BAND = (0.94, 0.94, 0.94)


def esc(s):
    """PDF literal-string escaping, in WinAnsi's ASCII subset."""
    s = "".join(c if 32 <= ord(c) < 127 else "?" for c in str(s))
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def wrap(text, cols):
    """Greedy wrap to `cols` monospaced columns. Never loses characters."""
    text = " ".join(str(text).split())
    if not text:
        return [""]
    out, line = [], ""
    for word in text.split(" "):
        while len(word) > cols:            # a single over-long token: hard-split
            if line:
                out.append(line); line = ""
            out.append(word[:cols]); word = word[cols:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= cols:
            line += " " + word
        else:
            out.append(line); line = word
    if line:
        out.append(line)
    return out


def jpeg_info(data):
    """(width, height, components, progressive) from a JPEG's SOF marker."""
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xD8 or m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xDA:                       # start of scan: no SOF found
            break
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h, data[i + 9], m in (0xC2, 0xC6, 0xCA, 0xCE)
        i += 2 + ln
    raise ValueError("no JPEG SOF marker found")


class PDF:
    """Minimal PDF writer: pages, base-14 text, rules, and DCTDecode images."""

    def __init__(self, size=A4):
        self.w, self.h = size
        self.objs = [b""]                   # index 0 unused; objects are 1-based
        self.pages = []                     # list of (ops, xobjects)
        self.ops, self.xobj = [], {}
        self.y = self.h - MARGIN
        self.images = {}

    # -- object plumbing --------------------------------------------------
    def _add(self, body):
        self.objs.append(body)
        return len(self.objs) - 1

    # -- drawing ----------------------------------------------------------
    def text(self, x, y, s, font="F3", size=8.2, colour=BLACK):
        r, g, b = colour
        self.ops.append(f"BT /{font} {size:.2f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
                        f"1 0 0 1 {x:.2f} {y:.2f} Tm ({esc(s)}) Tj ET")

    def rect(self, x, y, w, h, colour):
        r, g, b = colour
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} "
                        f"{w:.2f} {h:.2f} re f")

    def rule(self, y, colour=RULE, width=0.6):
        r, g, b = colour
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width} w "
                        f"{MARGIN:.2f} {y:.2f} m {self.w - MARGIN:.2f} {y:.2f} l S")

    def image(self, jpeg, x, y, w, h):
        if id(jpeg) not in self.images:
            iw, ih, comps, prog = jpeg_info(jpeg)
            cs = {1: "/DeviceGray", 3: "/DeviceRGB", 4: "/DeviceCMYK"}.get(comps,
                                                                           "/DeviceRGB")
            n = self._add(b"<</Type/XObject/Subtype/Image/Width %d/Height %d"
                          b"/ColorSpace %s/BitsPerComponent 8/Filter/DCTDecode"
                          b"/Length %d>>stream\n%s\nendstream"
                          % (iw, ih, cs.encode(), len(jpeg), jpeg))
            self.images[id(jpeg)] = n
        n = self.images[id(jpeg)]
        name = f"Im{n}"
        self.xobj[name] = n
        self.ops.append(f"q {w:.2f} 0 0 {h:.2f} {x:.2f} {y:.2f} cm /{name} Do Q")

    # -- flow -------------------------------------------------------------
    def space(self, n=1):
        self.y -= LEAD * n

    def need(self, pts):
        if self.y - pts < MARGIN + 22:
            self.page_break()

    def page_break(self):
        self.pages.append((self.ops, self.xobj))
        self.ops, self.xobj = [], {}
        self.y = self.h - MARGIN

    def line(self, s, font="F3", size=8.2, colour=BLACK, indent=0.0):
        self.need(LEAD)
        self.y -= LEAD
        self.text(MARGIN + indent, self.y, s, font, size, colour)

    def heading(self, s):
        self.need(30)
        self.y -= 16
        self.text(MARGIN, self.y, s, "F2", 10.5, BLACK)
        self.y -= 4
        self.rule(self.y)
        self.y -= 2

    # -- output -----------------------------------------------------------
    def save(self, path, footer=""):
        self.pages.append((self.ops, self.xobj))
        total = len(self.pages)
        font_ids = {}
        for key, base in (("F1", "Helvetica"), ("F2", "Helvetica-Bold"),
                          ("F3", "Courier"), ("F4", "Courier-Bold")):
            font_ids[key] = self._add(
                b"<</Type/Font/Subtype/Type1/BaseFont/%s/Encoding/WinAnsiEncoding>>"
                % base.encode())
        # Reserve the page-tree slot up front. Predicting its object number
        # from a count of what gets written later is exactly the kind of
        # arithmetic that breaks the moment a page carries an image.
        pages_id = self._add(b"")
        page_ids = []
        for i, (ops, xobj) in enumerate(self.pages, 1):
            # Footers go on last, once the total page count is known.
            f = f"{footer}   ·   page {i} of {total}".replace("·", "-")
            ops = ops + [
                f"BT /F3 7 Tf {GREY[0]:.2f} {GREY[1]:.2f} {GREY[2]:.2f} rg "
                f"1 0 0 1 {MARGIN:.2f} {MARGIN - 12:.2f} Tm ({esc(f)}) Tj ET"]
            stream = zlib.compress("\n".join(ops).encode("latin-1", "replace"))
            cid = self._add(b"<</Length %d/Filter/FlateDecode>>stream\n%s\nendstream"
                            % (len(stream), stream))
            res = (b"<</Font<<" +
                   b"".join(b"/%s %d 0 R" % (k.encode(), v) for k, v in font_ids.items()) +
                   b">>")
            if xobj:
                res += (b"/XObject<<" +
                        b"".join(b"/%s %d 0 R" % (k.encode(), v) for k, v in xobj.items()) +
                        b">>")
            res += b">>"
            pid = self._add(b"<</Type/Page/Parent %d 0 R/MediaBox[0 0 %.2f %.2f]"
                            b"/Resources %s/Contents %d 0 R>>"
                            % (pages_id, self.w, self.h, res, cid))
            page_ids.append(pid)
        kids = b"[" + b" ".join(b"%d 0 R" % p for p in page_ids) + b"]"
        self.objs[pages_id] = b"<</Type/Pages/Kids %s/Count %d>>" % (kids, total)
        root = self._add(b"<</Type/Catalog/Pages %d 0 R>>" % pages_id)
        info = self._add(b"<</Producer(signoff mkreport.py)/Title(Board sign-off report)>>")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0] * len(self.objs)
        for n in range(1, len(self.objs)):
            offsets[n] = len(out)
            out += b"%d 0 obj\n" % n + self.objs[n] + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n" % len(self.objs)
        out += b"0000000000 65535 f \n"
        for n in range(1, len(self.objs)):
            out += b"%010d 00000 n \n" % offsets[n]
        out += (b"trailer\n<</Size %d/Root %d 0 R/Info %d 0 R>>\nstartxref\n%d\n%%%%EOF\n"
                % (len(self.objs), root, info, xref))
        with open(path, "wb") as f:
            f.write(bytes(out))
        return len(out)


# ------------------------------------------------------------------ report

def read_tsv(path):
    rows = []
    with open(path) as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append({"status": parts[0], "section": parts[1],
                             "name": parts[2],
                             "detail": parts[3] if len(parts) > 3 else ""})
    return rows


def build(rows, meta, imu, image, out_path):
    npass = sum(1 for r in rows if r["status"] == "PASS")
    nfail = sum(1 for r in rows if r["status"] == "FAIL")
    verdict = "PASS" if nfail == 0 else "FAIL"
    vcol = GREEN if nfail == 0 else RED

    p = PDF()
    cols = int((p.w - 2 * MARGIN) / (8.2 * MONO_W))

    # -- title -----------------------------------------------------------
    p.rect(MARGIN, p.y - 46, p.w - 2 * MARGIN, 52, BAND)
    p.text(MARGIN + 10, p.y - 12, "BOARD SIGN-OFF REPORT", "F2", 15)
    p.text(MARGIN + 10, p.y - 27, meta.get("title", "Jetson Orin NX / A603"), "F1", 9.5,
           GREY)
    p.text(MARGIN + 10, p.y - 39, meta.get("date", ""), "F3", 8, GREY)
    bw = 84
    p.rect(p.w - MARGIN - bw - 8, p.y - 40, bw, 30, vcol)
    p.text(p.w - MARGIN - bw + 10, p.y - 30, verdict, "F2", 18, (1, 1, 1))
    p.y -= 62

    # The drone serial is what files this report in the fleet sheet, so it
    # sits above everything else, in the same style as the FC serial band.
    drone = meta.get("drone_serial") or ""
    p.rect(MARGIN, p.y - 30, p.w - 2 * MARGIN, 34, (0.97, 0.99, 0.97))
    p.text(MARGIN + 10, p.y - 10, "DRONE SERIAL", "F2", 8, GREY)
    p.text(MARGIN + 10, p.y - 24, drone or "NOT RECORDED", "F4", 13,
           BLACK if drone else RED)
    p.y -= 46

    # The flight controller's serial is the identity the whole report hangs
    # off, so it gets its own band rather than a row in a table.
    uid = (imu or {}).get("uid") or meta.get("fc_uid") or ""
    p.rect(MARGIN, p.y - 30, p.w - 2 * MARGIN, 34, (0.97, 0.97, 0.99))
    p.text(MARGIN + 10, p.y - 10, "FLIGHT CONTROLLER SERIAL (MSP_UID)", "F2", 8, GREY)
    p.text(MARGIN + 10, p.y - 24, uid or "NOT READ", "F4", 13,
           BLACK if uid else RED)
    fw = (imu or {}).get("firmware")
    if fw:
        p.text(p.w - MARGIN - 210, p.y - 24,
               f"{fw} {(imu or {}).get('version','')} "
               f"{(imu or {}).get('board','')}".strip(), "F3", 8.5, GREY)
    p.y -= 46

    # -- identity + summary ----------------------------------------------
    p.heading("1  Unit under test")
    for k in ("drone_serial", "model", "compatible", "l4t", "kernel", "hostname", "module_serial",
              "mac", "operator", "camera", "power_mode"):
        if meta.get(k):
            p.line(f"{k.replace('_', ' '):<16}{meta[k]}", size=8.2)
    if imu:
        for k, label in (("board", "fc board"), ("craft", "craft name"),
                         ("build", "fc build")):
            if imu.get(k):
                p.line(f"{label:<16}{imu[k]}", size=8.2)

    p.space()
    p.heading("2  Summary")
    p.line(f"{npass} passed, {nfail} failed, {len(rows)} checks total", "F4", 9,
           GREEN if nfail == 0 else RED)
    if nfail:
        p.line("failing checks:", size=8.2, colour=RED)
        for r in rows:
            if r["status"] == "FAIL":
                for i, ln in enumerate(wrap(f"{r['section']}: {r['name']} "
                                            f"{'- ' + r['detail'] if r['detail'] else ''}",
                                            cols - 4)):
                    p.line(("  " if i == 0 else "    ") + ln, size=8.2, colour=RED)

    # -- results ----------------------------------------------------------
    p.space()
    p.heading("3  Results")
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            p.need(24)
            p.space(0.4)
            p.line(section.upper(), "F2", 8.6, GREY)
        colour = {"PASS": GREEN, "FAIL": RED}.get(r["status"], GREY)
        tag = {"PASS": "PASS", "FAIL": "FAIL", "INFO": "  · "}.get(r["status"], "    ")
        p.need(LEAD)
        p.y -= LEAD
        p.text(MARGIN + 4, p.y, tag.replace("·", "-"), "F4", 8.2, colour)
        p.text(MARGIN + 34, p.y, r["name"][:cols - 8], "F3", 8.2, BLACK)
        if r["detail"]:
            for ln in wrap(r["detail"], cols - 12):
                p.need(LEAD)
                p.y -= LEAD
                p.text(MARGIN + 42, p.y, ln, "F3", 7.6, GREY)

    # -- IMU measurements --------------------------------------------------
    if imu:
        p.space()
        p.heading("4  IMU measurements")
        fields = [
            ("accel magnitude at rest", imu.get("accel_mag_g"), "g", "1.000 expected"),
            ("gyro magnitude at rest", imu.get("gyro_mag_dps"), "dps", "0 expected"),
            ("gyro noise (1 sigma)", imu.get("gyro_sd_dps"), "dps", "per axis"),
            ("accel noise (1 sigma)", imu.get("accel_sd_g"), "g", "per axis"),
            ("sample rate", imu.get("rate_hz"), "Hz", f"{imu.get('samples','?')} samples"),
            ("MSP CRC errors", imu.get("crc_errors"), "", ""),
            ("sensors detected", ",".join(imu.get("sensors", [])), "", ""),
            ("peak rate under motion", imu.get("motion_peak_dps"), "dps",
             "operator rotation"),
            ("attitude travel under motion", imu.get("motion_att_deg"), "deg",
             "operator rotation"),
        ]
        for label, val, unit, note in fields:
            if val is None or val == "":
                continue
            if isinstance(val, list):
                val = " / ".join("--" if v is None else f"{v}" for v in val)
            p.line(f"{label:<30}{val} {unit}".rstrip() +
                   (f"     ({note})" if note else ""), size=8.2)

    # -- captured frame ----------------------------------------------------
    if image:
        try:
            iw, ih, _c, prog = jpeg_info(image)
            p.page_break()
            p.heading("5  Captured frame")
            p.line(f"{iw} x {ih} JPEG, {len(image)} bytes, embedded losslessly"
                   + (" (progressive)" if prog else ""), size=8.2, colour=GREY)
            p.space(0.5)
            avail_w = p.w - 2 * MARGIN
            avail_h = p.y - MARGIN - 24
            sc = min(avail_w / iw, avail_h / ih)
            w, h = iw * sc, ih * sc
            p.image(image, MARGIN, p.y - h, w, h)
            p.y -= h + 6
        except Exception as e:
            p.line(f"still image could not be embedded: {e}", colour=AMBER)

    footer = f"{meta.get('hostname', '')}  {meta.get('date', '')}  FC {uid or 'n/a'}"
    return p.save(out_path, footer), verdict, npass, nfail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--imu")
    ap.add_argument("--image")
    ap.add_argument("-o", "--out", default="report.pdf")
    a = ap.parse_args()

    rows = read_tsv(a.tsv)
    meta = json.load(open(a.meta))
    meta.setdefault("date", datetime.datetime.now().astimezone().isoformat(
        timespec="seconds"))
    imu = json.load(open(a.imu)) if a.imu and __import__("os").path.exists(a.imu) else None
    image = None
    if a.image:
        try:
            with open(a.image, "rb") as f:
                image = f.read()
        except OSError:
            image = None

    size, verdict, npass, nfail = build(rows, meta, imu, image, a.out)
    print(f"{a.out}  ({size} bytes)  {verdict}: {npass} passed, {nfail} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
