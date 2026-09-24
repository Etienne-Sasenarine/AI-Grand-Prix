"""Animated GIFs of the recordings in videos/, for inline display on GitHub, plus a still frame for the PDF.

OpenCV reads the mp4 (needs a Python with cv2); Pillow writes the GIF (needs a Python with PIL). On the
team Mac those are two interpreters, so the frame dump and the assembly are separate steps:

    ~/dev/ai-grand-prix/.venv_mac/bin/python make_gifs.py dump
    python3 make_gifs.py assemble
"""
import os, sys, glob, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(tempfile.gettempdir(), "aigp_gif_frames")
CLIPS = [  # name, source video, start s, end s, width px, fps, colours, stack the two panes vertically
    ("race_start_from_pad", "race_start_from_pad.mp4", 0.0, 6.7, 800, 10, 64, False),
    ("race_best", "race_best.mp4", 0.0, 9.9, 800, 8, 64, False),
    ("stack_pov", "stack_run_02.mp4", 6.0, 15.0, 720, 6, 48, True),   # 1280x360 side by side -> 720x810 stacked
]

def restack(fr):
    """A 2:1 side-by-side frame becomes the two panes one above the other."""
    import numpy as np
    h, w = fr.shape[:2]; half = w // 2
    return np.vstack([fr[:, :half], fr[:, half:]])

def dump():
    import cv2
    for name, src, t0, t1, width, fps_out, _, stack in CLIPS:
        d = os.path.join(TMP, name); os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.png")): os.remove(f)
        cap = cv2.VideoCapture(os.path.join(HERE, "videos", src)); fps = cap.get(cv2.CAP_PROP_FPS)
        n = 0
        for k in range(int((t1 - t0) * fps_out)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + k / fps_out) * fps)); ok, fr = cap.read()
            if not ok: break
            if stack: fr = restack(fr)
            h = int(fr.shape[0] * width / fr.shape[1]); fr = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(d, f"{k:04d}.png"), fr); n += 1
        print(name, n, "frames")

def assemble():
    from PIL import Image
    for name, _, _, _, _, fps_out, colours, _ in CLIPS:
        paths = sorted(glob.glob(os.path.join(TMP, name, "*.png")))
        frames = [Image.open(p).convert("RGB").quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) for p in paths]
        out = os.path.join(HERE, "figures", name + ".gif")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps_out), loop=0, optimize=True)
        Image.open(paths[len(paths) // 3]).convert("RGB").save(os.path.join(HERE, "figures", name + "_still.png"))
        print(name, len(frames), "frames,", round(os.path.getsize(out) / 1e6, 1), "MB")

if __name__ == "__main__":
    {"dump": dump, "assemble": assemble}[sys.argv[1]]()
