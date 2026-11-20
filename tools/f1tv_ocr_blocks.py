"""Split an F1 TV screen recording (player NOT full screen: clock bottom-left, driver list right) into playing
blocks: one OCR read of the player clock and the highlighted driver per second, then runs where the clock advances
1 s per second with one driver. CPU only (rapidocr). Convert the recording first: scripts/to_constant_fps.py.

    python scripts/f1tv_ocr_blocks.py <recording.mkv> --out runs/<name>/ [--session race]
-> <out>/ocr_scan.csv (t, player_s, clock, driver) and <out>/blocks.csv (driver, off, t0, t1, n, from, to, video,
   session); player time = recording time + off.

Layout (1920x1080 capture, browser player 2026-11-17): clock box y 985-1020, x 30-170; driver list x 1650-1890 (the
selected driver is the brightest box); onboard picture y 50-910, x 40-1578 (see f1tv_scan_blocks.VIDEO_RECT).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CLOCK = (985, 1020, 30, 170)
DRIVERS_X = (1650, 1890)


def parse_clock(clk: str) -> float:
    """Player clock text -> seconds: "01:41:08" (replays >= 1 h) or "26:12" (shorter replays, e.g. a sprint)."""
    # (?:\2)? = overlapping OCR boxes "01:16" + "16:00" joined as "01:1616:00"
    m = re.search(r"(\d{1,2}):(\d\d)(?:\2)?:(\d\d)", str(clk))
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = re.fullmatch(r"\s*(\d{1,2}):(\d\d)\s*", str(clk))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


def read_frame(ocr, im):
    y0, y1, x0, x1 = CLOCK
    r, _ = ocr(cv2.resize(im[y0:y1, x0:x1], None, fx=2, fy=2))
    # rapidocr sometimes splits the clock into boxes ("01:16" + ":00", 2025 quali): join them left to right
    clk = "".join(t[1].strip() for t in sorted(r, key=lambda t: min(p[0] for p in t[0]))) if r else ""
    ps = parse_clock(clk)
    a, b = DRIVERS_X
    pan = cv2.cvtColor(im[:, a:b], cv2.COLOR_BGR2GRAY).mean(axis=1)
    y = int(np.argmax(np.convolve(pan, np.ones(30) / 30, "same")))
    r, _ = ocr(cv2.resize(im[max(0, y - 25):y + 25, a:b], None, fx=2, fy=2))
    return clk, ps, (r[0][1].strip() if r else "")


def blocks_from_scan(d: pd.DataFrame, min_len: int = 8) -> pd.DataFrame:
    """Runs of seconds with one driver and a constant offset (player - recording time, +-1.5 s)."""
    d = d.copy(); d["off"] = d.player_s - d.t
    ok = d.player_s.notna() & d.driver.str.fullmatch(r"[A-Z]{3}", na=False)
    out, cur = [], None
    for r in d[ok].itertuples():
        if cur and r.driver == cur["driver"] and abs(r.off - cur["off"]) <= 1.5 and r.t - cur["t1"] <= 8:
            cur["t1"] = r.t; cur["n"] += 1
        else:
            if cur: out.append(cur)
            cur = dict(driver=r.driver, off=r.off, t0=r.t, t1=r.t, n=1)
    if cur: out.append(cur)
    b = pd.DataFrame(out, columns=["driver", "off", "t0", "t1", "n"])
    b = b[b.n >= min_len].reset_index(drop=True)
    f = lambda x: "%d:%02d:%02d" % (x // 3600, x % 3600 // 60, x % 60)
    b["from"] = (b.t0 + b.off).map(f); b["to"] = (b.t1 + b.off).map(f)
    return b


def main() -> None:
    from rapidocr_onnxruntime import RapidOCR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session", default=""); ap.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ocr = RapidOCR()
    cap = cv2.VideoCapture(str(a.video)); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps * a.every))); rows = []
    for fi in range(n):                       # sequential read (constant-fps file): frame index = time * fps
        if fi % step:
            if not cap.grab(): break
            continue
        ok, im = cap.read()
        if not ok: break
        clk, ps, drv = read_frame(ocr, im)
        rows.append(dict(t=round(fi / fps, 2), player_s=ps, clock=clk, driver=drv))
    d = pd.DataFrame(rows); d.to_csv(a.out / "ocr_scan.csv", index=False)
    b = blocks_from_scan(d); b["video"] = str(a.video).replace("\\", "/"); b["session"] = a.session
    b.to_csv(a.out / "blocks.csv", index=False)
    print(b[["driver", "from", "to", "t0", "t1", "off"]].to_string())


if __name__ == "__main__":
    main()
