"""Video probing, screen-recording crop detection and HUD-dial detection."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
Crop = tuple[int, int, int, int]   # x, y, w, h


def probe(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    info["duration_s"] = round(info["frames"] / info["fps"], 2)
    return info


def count_frames(video: Path) -> int | None:
    """Number of frames really stored in the file (stream copy, no decoding: a few seconds for an
    hour of video). None when no ffmpeg binary is available."""
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    try:
        out = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
                              "-c", "copy", "-f", "framecrc", "-"], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return sum(1 for ln in out.stdout.splitlines() if ln and not ln.startswith("#"))


def dropped_frames(video: Path, info: dict | None = None) -> tuple[int, float] | None:
    """(frames missing, seconds missing) against a constant frame rate, or None if it can't be counted.

    A screen capture that cannot keep up drops frames, and the file then has gaps in its time stamps.
    OpenCV seeks by time stamp but reads frame after frame, so any code that counts frames while
    reading (`time = n / fps`) runs late by the frames dropped so far. `to_constant_fps` fixes the file."""
    info = info or probe(video)
    n = count_frames(video)
    if n is None:
        return None
    missing = max(0, info["frames"] - n)
    return missing, missing / info["fps"]


def warn_if_dropped(video: Path, info: dict | None = None, tol_s: float = 0.05) -> float:
    """Log a warning when the video lost more than `tol_s` seconds of frames; returns the seconds lost."""
    d = dropped_frames(video, info)
    if d is None or d[1] <= tol_s:
        return 0.0 if d is None else d[1]
    log.warning("%s has %d dropped frame(s) = %.2f s missing: frames read one after another will show a "
                "later moment than `n / fps`. Fix the file first: python scripts/to_constant_fps.py \"%s\"",
                Path(video).name, d[0], d[1], video)
    return d[1]


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, fr = cap.read()
    return fr if ok else None


def apply_crop(frame: np.ndarray, crop: Crop | None) -> np.ndarray:
    if crop is None:
        return frame
    x, y, w, h = crop
    return frame[y:y + h, x:x + w]


def detect_content_crop(video: Path, n_samples: int = 40, margin: int = 8, min_cover: float = 0.25) -> Crop | None:
    """Find the live-video area inside a screen recording (browser chrome, side panels).

    Static UI has ~zero temporal variance; the playing video does not. We take the rows/columns where
    a good share of pixels vary across `n_samples` frames. Returns None when the whole frame is live
    (a normal video) or nothing varies. Heuristic: for letterboxed layouts or captions that sit on
    top of the video, set `crop: [x, y, w, h]` in clip.yaml instead.
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 4:
        return None
    stack = []
    for f in np.linspace(2, n - 3, min(n_samples, n - 4)).astype(int):
        fr = read_frame(cap, f)
        if fr is not None:
            stack.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)[::2, ::2])
    cap.release()
    if len(stack) < 4:
        return None
    moving = np.std(np.stack(stack), axis=0) > 12
    cols = np.flatnonzero(moving.mean(0) > min_cover)
    rows = np.flatnonzero(moving.mean(1) > min_cover)
    if cols.size == 0 or rows.size == 0:
        return None
    h2, w2 = moving.shape
    x0, x1 = max(0, cols[0] * 2 - margin), min(w2 * 2, cols[-1] * 2 + margin)
    y0, y1 = max(0, rows[0] * 2 - margin), min(h2 * 2, rows[-1] * 2 + margin)
    if (x1 - x0) * (y1 - y0) > 0.95 * (w2 * 2) * (h2 * 2):
        return None                                   # whole frame is live: no crop needed
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def f1tv_dial_present(frame: np.ndarray) -> bool:
    """Does the F1 TV circular telemetry dial (blue ring + green throttle arc) sit bottom-left?

    Uses the same reference geometry as ingest/hud.py (1920x1080), scaled to the frame.
    """
    h, w = frame.shape[:2]
    sx, sy = w / 1920, h / 1080
    roi = frame[int(400 * sy):int(790 * sy), int(10 * sx):int(400 * sx)]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] > 100) & (hsv[..., 0] < 125) & (hsv[..., 1] > 150) & (hsv[..., 2] > 120)).mean()
    green = ((hsv[..., 0] > 40) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 100)).mean()
    return bool(blue > 0.04 and green > 0.01)
