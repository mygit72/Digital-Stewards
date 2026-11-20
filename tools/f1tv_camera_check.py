"""Mark the seconds of an F1 TV onboard recording that are NOT the standard forward view (the director switches to
rear-facing, side/halo or high cockpit cameras). The standard view has the car body (nose, wing, halo, front tyres)
fixed in the lower part of the picture: compare each sampled frame with a reference median of known-standard
frames on the pixels that stay constant there.

    python scripts/f1tv_camera_check.py <blocks.csv> <ref h:mm:ss> <out.csv> [--ref-block 1]
-> out.csv: player_s, video, diff (mean abs diff on the static mask), standard (diff < thr)
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1tv_scan_blocks import VIDEO_RECT

def body(im):
    y0, y1, x0, x1 = VIDEO_RECT
    v = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(v[int(0.62 * (y1 - y0)):], (384, 82)).astype(np.float32)

def hms(s):
    h, m, x = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(x)

ap = argparse.ArgumentParser(); ap.add_argument("blocks"); ap.add_argument("ref"); ap.add_argument("out")
ap.add_argument("--ref-block", type=int, default=1); ap.add_argument("--thr", type=float, default=30.0)
a = ap.parse_args()
B = pd.read_csv(a.blocks); caps = {}
def grab(b, p):
    c = caps.setdefault(b.video, cv2.VideoCapture(b.video)); fps = c.get(cv2.CAP_PROP_FPS)
    c.set(cv2.CAP_PROP_POS_FRAMES, int(round((p - b.off) * fps))); ok, im = c.read(); return im if ok else None
rb = B.loc[a.ref_block]; r0 = hms(a.ref)
refs = np.stack([body(grab(rb, r0 + k)) for k in range(0, 30)])
ref = np.median(refs, 0); mask = refs.std(0) < 12
print("static mask pixels:", int(mask.sum()))
rows = []
for _, b in B.iterrows():
    for p in np.arange(b.t0 + b.off, b.t1 + b.off + 1e-6, 1.0):
        im = grab(b, p)
        if im is None: continue
        d = float(np.abs(body(im) - ref)[mask].mean())
        rows.append(dict(player_s=p, video=b.video, diff=round(d, 1)))
d = pd.DataFrame(rows); d["standard"] = d["diff"] < a.thr; d.to_csv(a.out, index=False)
print(d["diff"].describe().to_string())
