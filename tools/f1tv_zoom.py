"""Zoom strips of an F1 TV recording at given player times. usage: f1tv_zoom.py <out.jpg> <block> <h:mm:ss.s> <dur_s> <step_s> [cols]"""
import sys, cv2, numpy as np, pandas as pd
V = "recordings/f1tv/f1tv_multi_2026-11-17 19-09-42.mkv"
out, bi, start, dur, step = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
cols = int(sys.argv[6]) if len(sys.argv) > 6 else 3
h, m, s = start.split(":"); p0 = int(h) * 3600 + int(m) * 60 + float(s)
import os
b = pd.read_csv(os.environ.get("F1TV_BLOCKS", "runs/real_cota2026/blocks_all.csv")).loc[bi]; V = b.video
c = cv2.VideoCapture(V); fps = c.get(cv2.CAP_PROP_FPS); tiles = []
for k in np.arange(0, dur + 1e-6, step):
    p = p0 + k; t = p - b.off
    c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, im = c.read()
    v = im[50:910, 40:1578][120:700]                     # road + front tyres
    v = cv2.resize(v, (640, 241)); cv2.putText(v, f"{int(p//3600)}:{int(p%3600//60):02d}:{p%60:05.2f}", (6, 22), 0, 0.7, (0, 255, 255), 2)
    tiles.append(v)
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
