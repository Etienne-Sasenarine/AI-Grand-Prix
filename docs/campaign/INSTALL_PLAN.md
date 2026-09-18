# Installs and downloads, ordered by how badly they can block us

> **Revised 17 Sep after reading the organizers' Orin NX guides.** Most of what this document
> originally told you to install is **already on the drone**, and one recommendation (building the
> Betaflight software-in-the-loop simulator) is now unnecessary. The corrections are marked below.

Assumption that drives this whole list: **the venue network will be slow, filtered, or shared with
twenty other teams.** Anything that needs downloading should be downloaded now, on a known-good
connection, onto this laptop and a USB drive.

## What you do NOT need to install

The Jetson image ships with everything the onboard stack needs, and the organizers say so
explicitly: *"Nothing here needs pip; every dependency is already in the image."*

Already present on the board: **OpenCV built with GStreamer support**, NumPy, pyserial, python3-gi,
rich, pymavlink, MAVProxy, v4l-utils, i2c-tools, gpiod — plus the whole `~/target/` toolchain
(MSP library, RC transmitter with watchdog, fake flight controller, camera viewers, timing tools,
stack diagnostics).

**Two things you must not do**, both from the organizers' guide:
- **Never `pip install opencv-python`.** The image's OpenCV is built with GStreamer; the pip wheel is
  not, and `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` will then fail silently — `isOpened()`
  returns False with no explanation.
- **Never try to change the device tree.** Writing the `kernel-dtb` partition verifies clean and does
  nothing at all on this platform; changing it needs a full re-flash, which is the organizers' job.
  This has already cost somebody a debugging cycle.

**What is NOT on the image:** any deep-learning framework. No PyTorch, no ultralytics. This makes the
NumPy policy runtime mandatory rather than merely sensible, and means the gate detector must be
exported to ONNX and run through OpenCV's DNN module or TensorRT.

Current state of this laptop (WSL): **no compiler at all** — `gcc`, `g++`, `make`, `cmake` are all
missing, and sudo needs a password. `git` and `node` are present. `venv-client` has OpenCV 5.0,
NumPy 2.5, Torch 2.14+cu126, Ultralytics 8.4 — but **no `pip` and no `pyserial`**.

---

## Blocker 0 — the one command that unblocks several others

**Downgraded from blocker to convenience** — the Betaflight SITL build is no longer needed. Still
worth running when convenient, for blackbox log tools, video handling and PDF text extraction:

```
sudo apt install -y build-essential cmake ffmpeg poppler-utils
```

Also fix pip in the flight-client environment:

```
~/aigp/venv-client/bin/python -m ensurepip --upgrade
~/aigp/venv-client/bin/python -m pip install pyserial
```

---

## Tier 1 — do today, on this laptop

### 1. ~~Betaflight SITL~~ → copy the organizers' toolchain instead

**Superseded.** The plan was to build Betaflight's software-in-the-loop target so we could test the
link against real firmware without a drone. The organizers already ship something better for this
purpose: `~/target/msp/fake_fc.py` simulates a Betaflight flight controller on a pty, **with
deliberate fault modes** (`frozen`, `saturated`, `badscale`, `noaccel`, `i2cerr`, `nouid`), and the
whole test suite runs with nothing plugged in. It is plain Python — no compiler, no build, no
toolchain.

**So the action is:**

```
scp -r dcl@192.168.55.1:~/target ./
```

Do this the moment anybody can reach a board, and share it with the team. It unblocks everyone who
wants to write adapter code without the drone in front of them.

Betaflight SITL is still marginally useful for verifying the rate-curve inversion against real
firmware, but it is now a nice-to-have, not a blocker.

### 2. Betaflight Configurator + Blackbox Explorer — install on the Windows side

Configurator version must match firmware 4.4.3. Needed for: the `diff all` settings dump, setting
`msp_override_channels_mask` to 15, enabling blackbox logging, and downloading logs. Blackbox
Explorer (or the `blackbox_decode` command-line tool) is how we read the flight recorder afterwards
— which is how we get hover throttle, response time, drag and battery sag.

**Download the installers now.** Do not plan to download them at the venue.

### 3. A printed calibration target

The spec says plainly: *"Formal camera calibration matrices (intrinsics/extrinsics) will not be
provided."* The lens is an M12 screw mount, so the real field of view can differ from the nominal
75° by degrees, and there will be lens distortion that the simulator models as exactly zero.

Print a checkerboard or ChArUco board, **glued to something rigid and flat** — a clipboard, foam
board, the back of a notebook. A wavy printout gives a wrong calibration that looks fine.
This is a five-dollar item that blocks a Tier-1 measurement, and nobody ever remembers it.

### 4. Cloud training box, set up before we need it

Isaac Sim container plus Isaac Lab and skrl, on the rented GPU. It is roughly a 20 GB pull and about
an hour of setup the first time. **Do it now and snapshot the image**, so that on Thursday evening
"launch a training run" is five minutes and not ninety.

---

## Tier 2 — the Jetson, and the trap waiting there

### 5. Do not install PyTorch on the Jetson — now confirmed

`pip install torch` does not work on a Jetson. ARM64 plus CUDA means NVIDIA's own wheel built for
JetPack 6.2, or their container: a multi-hour, network-heavy step that breaks in interesting ways,
on venue Wi-Fi, on the day we can least afford it. **The organizers' package list confirms no
framework is installed, and their guidance is that nothing on the board should need pip at all.**

**We don't need it.** The trained model is a three-layer, 256-unit-wide network. That is three matrix
multiplies and three activation functions. Export the weights to a `.npz` file and evaluate it in
NumPy — about fifty lines, no PyTorch, no CUDA, no install. It will run in well under a millisecond
on the Orin's CPU, which is nothing against a 16-millisecond control period.

I'll write the exporter and the NumPy runtime, and verify it reproduces the PyTorch output to
floating-point precision on this laptop. Then the flight path has no deep-learning framework in it
at all.

### 6. The detector is the part that actually needs acceleration

Two routes, and we should prepare both:

- **Learned detector:** with no PyTorch and no ultralytics on the image, the route is **export to
  ONNX and run through OpenCV's DNN module** (which is present and needs nothing installed), or
  through TensorRT if it is on the image — verify, because it ships with JetPack but is absent from
  the guide's package list. TensorRT engines must be built **on the Jetson itself**; they are
  specific to that device and software version and cannot be prepared here.
- **Classical detector** (`vision/snake_gate_detector.py`, tuned with `tools/hsv_tuner.py`): pure
  OpenCV and NumPy, nothing to install, tunable in an hour on the real gates. Noisier and
  shorter-ranged, but it will never be the reason we can't fly, and the slow backup plan tolerates a
  noisy detector far better than the trained model does.

Bring both. Expect to start on the classical one.

### 7. Camera bring-up

The driver is already installed and the organizers ship the viewers and the diagnostics. Nothing to
install — but three constraints from their guide will otherwise eat an afternoon each:

1. **Only two of the camera's four data lanes are wired** on this carrier, capping resolution ×
   framerate. Run `v4l2-ctl -d /dev/video0 --list-formats-ext` and treat that list as the truth.
   The spec's 75° field of view is quoted for 1920×1080 @ 60 fps — **if that mode isn't available,
   our calibration is wrong before we start.**
2. **The image processor was expected to need a graphics context — measured 18 Sep, it does not.**
   Colour, auto-exposure and white balance ran correctly over a plain SSH session on the race Orin.
   The paragraph below records the documented behaviour, which we did not reproduce.
   The guides say colour, auto-exposure and white balance run in
   NVIDIA's ISP, which fails over a plain SSH session with `Failed to initialize EGLDisplay` and
   falls back to a flat, dim software debayer. That is documented behaviour, not a broken camera.
   For flight we must either arrange a display context on the board or design for minimally-processed
   frames.
3. **`cv2.VideoCapture` discards the hardware capture timestamp** and gives arrival time instead,
   with tens of milliseconds of pipeline jitter. Use the GStreamer/`python3-gi` path
   (`live-view-pts.py`, `frame-timestamps.py`) wherever timing matters.

If the camera node is missing entirely: `sudo ~/target/camera-bind-check.sh` walks the bind chain and
names which link broke — "driver not bound" is a conclusion, not a cause.

### 8. Small things on the Jetson that still need a network

`pyserial`, and whatever our code imports. Pre-download the wheels (`pip download`) onto the USB
drive rather than assuming the venue lets you reach the package index.

---

## Tier 3 — physical kit that behaves like an install

- **USB-C to micro-USB cable** — this is the link to the board. The micro-USB port is the small one
  *next to the barrel jack*, not one of the full-size USB-A ports. The board always answers on
  `192.168.55.1` over it, regardless of Wi-Fi, and the same cable also provides a serial console at
  `/dev/ttyACM0` if the network half fails. **Bring several.**
- A USB-Ethernet adapter and cable remain useful as a backup path, but the USB gadget link above is
  the primary and it needs no venue network at all.
- Spare USB-C and micro-USB cables, a powered USB hub.
- A USB drive with: our repo, pre-downloaded pip wheels, YOLO weights, Betaflight source and
  Configurator installers, and this set of documents.
- Tape measure, laser rangefinder if we can get one, painter's tape for marking the survey datum,
  a phone with an inclinometer app.
- Spare props (8-inch, 4.1 pitch) and batteries, counted honestly — that's the real limit on how
  many attempts we get.

---

## What I can build right now, with no installs

1. **The control adapter** — the layer between our policy's output and the organizers'
   `RCTransmitter.set_control()`. Not the protocol; they wrote that.
2. **The Betaflight curve inversion** — radians-per-second and throttle into stick values, through
   Betaflight's actual non-linear rate and throttle curves, with unit tests and an explicit sign/axis
   table (Isaac uses forward-left-up, Betaflight uses forward-right-down; a sign error here flips the
   drone on takeoff).
3. **The policy exporter and NumPy runtime** — now mandatory, since the board has no framework.
4. **The gate counter**, testable offline against recorded video.

All four are pure software and on the critical path. Items 1 and 2 can be tested end-to-end against
`fake_fc.py` — **so the only thing standing between us and tested link code is somebody copying
`~/target/` off a board.**
