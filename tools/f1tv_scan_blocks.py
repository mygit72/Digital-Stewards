"""Run a frame classifier over the playing blocks of an F1 TV screen recording (player not full screen).
Blocks from runs/real_cota2026/multi_blocks.csv (driver, t0, t1, off = player_s - rec_t). Output: one row per
sampled frame: block, driver, rec_t, player_s, p_out."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from digital_stewards.classifier import crop
VIDEO_RECT = (50, 910, 40, 1578)      # y0, y1, x0, x1 of the onboard picture in the recording
ROWS = (0.12, 0.72)

def frame_crop(im):
    y0, y1, x0, x1 = VIDEO_RECT
    return crop(im[y0:y1, x0:x1], rows=ROWS, square=True)

if __name__ == "__main__":
    from ultralytics import YOLO
    video, blocks_csv, weights, out = sys.argv[1:5]
    hz = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    device = sys.argv[6] if len(sys.argv) > 6 else None      # 'cpu' to keep the GPU free
    m = YOLO(weights); oi = [k for k, v in m.names.items() if v == "out"][0]
    b = pd.read_csv(blocks_csv); cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    # Read the whole file from frame 0 and count frames: "seek once, then read on" lands 3-5 s late in these
    # screen recordings (checked against the on-screen F1 TV clock, 2026-11-17).
    want = {}
    for bi, r in b.iterrows():
        f0, f1 = int(r.t0 * fps), int(r.t1 * fps); step = max(1, int(round(fps / hz)))
        for f in range(f0, f1 + 1, step):
            want[f] = (bi, r)
    last = max(want); buf, meta = [], []

    def flush():
        for (f, bi, r), res in zip(meta, m.predict(buf, imgsz=448, verbose=False, **({'device': device} if device else {}))):
            rows.append(dict(block=bi, driver=r.driver, frame=f, rec_t=round(f / fps, 3),
                             player_s=round(f / fps + r.off, 2), p_out=float(res.probs.data[oi])))
        buf.clear(); meta.clear()

    for f in range(last + 1):
        if f not in want:
            if not cap.grab(): break
            continue
        ok, im = cap.read()
        if not ok: break
        bi, r = want[f]; buf.append(frame_crop(im)); meta.append((f, bi, r))
        if len(buf) >= 64: flush()
    if buf: flush()
    pd.DataFrame(rows).to_csv(out, index=False); print(len(rows), "frames ->", out)
