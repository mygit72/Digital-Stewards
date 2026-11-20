"""
The OUT/IN frame classifier: is any part of the car still on the track, or are all four wheels off?

Model: YOLOv8s-cls on a 448 x 448 crop of the road strip in front of and beside the front tyres.

    digital-stewards train --data datasets/<dataset>          (folder with train/{in,out} and val/{in,out})
    digital-stewards eval  --data datasets/<dataset> --weights models/digital_steward_v13.pt
    digital-stewards scan  <video> --weights models/digital_steward_v13.pt --out scan.csv [--rows 0.22,0.68]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BN_STAT_LIMIT = 1e8      # healthy running statistics here are < ~2e3; the fault leaves 1e26+ / inf / NaN


def repair_bn_stats(model_sd: dict, ema_sd: dict, snap: dict, limit: float = BN_STAT_LIMIT) -> list[str]:
    """Put back BatchNorm running statistics that one bad batch destroyed; return the repaired keys.

    Seen 2026-11-15: in a single batch one conv of block `model.2` returns garbage (~1e28 with normal weights
    and inputs). Train mode normalises with batch statistics, so the loss stays finite, but the layer's
    running_var becomes inf, the EMA copy turns it into NaN, and validation / last.pt (both use the EMA) are
    broken from then on. Running statistics are not trained, so restoring the last healthy values (`snap`,
    updated in place with the current healthy ones) loses nothing.
    """
    import torch
    fixed = []
    for k in [k for k in model_sd if k.endswith(("running_mean", "running_var"))]:
        ok = all(bool(torch.isfinite(sd[k]).all()) and float(sd[k].abs().max()) < limit for sd in (model_sd, ema_sd))
        if ok:
            snap[k] = (model_sd[k].clone(), ema_sd[k].clone())
        elif k in snap:
            model_sd[k].copy_(snap[k][0])
            ema_sd[k].copy_(snap[k][1])
            fixed.append(k)
    return fixed


class BNGuard:
    """Ultralytics callbacks: snapshot the BatchNorm running statistics, repair them after every epoch."""

    def __init__(self):
        self.snap: dict = {}
        self.repairs: dict[int, list[str]] = {}

    def start(self, trainer) -> None:
        repair_bn_stats(trainer.model.state_dict(), trainer.ema.ema.state_dict(), self.snap)

    def check(self, trainer) -> None:
        fixed = repair_bn_stats(trainer.model.state_dict(), trainer.ema.ema.state_dict(), self.snap)
        if fixed:
            self.repairs[trainer.epoch + 1] = fixed
            log.warning("epoch %d: BatchNorm running statistics were corrupt in %s - restored from the last healthy "
                        "epoch", trainer.epoch + 1, fixed)
CROP = (0.22, 0.68)


def crop(frame: np.ndarray, size: int = 448, rows: tuple[float, float] = CROP, square: bool = False) -> np.ndarray:
    """The road strip in front of and beside the front tyres.

    square=False (models up to v5): a 448 x ~115 strip. Ultralytics classification resizes the SHORT side to
    imgsz and centre-crops a square (and its RandomResizedCrop falls back to a centre crop on such a wide
    image), so those models only ever saw the middle ~26-34% of the width - never the front tyres or the
    line beside them (found 2026-11-16). square=True squashes the whole strip into size x size, so every
    pixel column reaches the model; train it with `--rrc-scale 0` so no random crop cuts a tyre away."""
    h = frame.shape[0]
    c = frame[int(rows[0] * h):int(rows[1] * h)]
    if square:
        return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(c, (size, int(size * c.shape[0] / c.shape[1])), interpolation=cv2.INTER_AREA)


def evaluate(data: Path, weights: Path, split: str = "val", out_thr: float = 0.5) -> pd.DataFrame:
    """Held-out accuracy per group: OUT frames, clean IN, near-miss IN (the one that matters for
    the line), plus OUT episodes decided by their mean p(out). Writes <data>/eval_<split>.csv."""
    from ultralytics import YOLO
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["split"] == split].reset_index(drop=True)
    model = YOLO(str(weights))
    out_idx = [k for k, v in model.names.items() if v == "out"][0]
    p = []
    for i in range(0, len(man), 64):
        res = model.predict([str(f) for f in man["file"].iloc[i:i + 64]], imgsz=448, verbose=False)
        p += [float(r.probs.data[out_idx]) for r in res]
    man["p_out"] = p
    man["kind"] = np.where(man["cls"] == "out", "out", man["group"])
    man["right"] = (man["p_out"] >= out_thr) == (man["cls"] == "out")
    man.to_csv(data / f"eval_{split}.csv", index=False)
    return man


def eval_report(man: pd.DataFrame) -> str:
    rows = [f"{k:<6} frames {int(g['right'].sum()):>4}/{len(g):<4} = {g['right'].mean():.1%}"
            for k, g in man.groupby("kind")]
    eps = man[man["cls"] == "out"].groupby(["clip", "group"])["p_out"].mean()
    if len(eps):
        rows.append(f"OUT episodes (mean p_out >= 0.5): {int((eps >= 0.5).sum())}/{len(eps)}")
    return "\n".join(rows)


def scan(video: Path, weights: Path, t0: float = 0.0, t1: float | None = None, step: int = 3,
         out_thr: float = 0.5, batch: int = 64, rows: tuple[float, float] = CROP, square: bool = False) -> pd.DataFrame:
    """p(out) for every `step`-th frame of any video between t0 and t1 (s), no labels needed - e.g. a real
    onboard lap, where every OUT call is a candidate false alarm to look at. Frames are cropped as in
    training (`crop`); `rows` moves the crop for a camera that sits higher or lower than the F1 25 T-cam
    (pick it from the picture: horizon near the top, front tyres at the bottom). Columns: frame, t, p_out, out."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out_idx = [k for k, v in model.names.items() if v == "out"][0]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0, f1 = int(round(t0 * fps)), min(n, int(round(t1 * fps)) if t1 is not None else n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    found, buf, idx = [], [], []

    def flush():
        if buf:
            res = model.predict(buf, imgsz=448, verbose=False)
            found.extend({"frame": i, "t": round(i / fps, 3), "p_out": float(r.probs.data[out_idx])} for i, r in zip(idx, res))
            buf.clear(); idx.clear()

    for f in range(f0, f1):                      # sequential read: fine for constant-fps files
        ok, fr = cap.read()
        if not ok:
            break
        if (f - f0) % step == 0:
            buf.append(crop(fr, rows=rows, square=square)); idx.append(f)
            if len(buf) >= batch:
                flush()
    flush()
    cap.release()
    df = pd.DataFrame(found)
    df["out"] = df["p_out"] >= out_thr
    return df


MIN_DUR_S = 0.1   # an OUT call must last this long (2026-11-17: on v10 removes ~2/3 of false episodes on
                  # valid laps, 35 -> 13, and keeps all 83 held-out OUT episodes; >= 0.2 s starts losing them)


def out_episodes(t, p, thr: float = 0.5, min_dur_s: float = MIN_DUR_S, gap_s: float = 0.12,
                 dt: float | None = None) -> list[tuple[float, float]]:
    """OUT episodes from per-frame p(out): runs of p >= thr (frames closer than gap_s joined) lasting at
    least min_dur_s. A run's duration is last - first + one frame interval (dt, default: median spacing)."""
    t = np.asarray(t, dtype=float)
    hit = np.asarray(p, dtype=float) >= thr
    if dt is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    runs = []
    for ti in t[hit]:
        if runs and ti - runs[-1][1] <= gap_s:
            runs[-1][1] = ti
        else:
            runs.append([ti, ti])
    return [(a, b) for a, b in runs if b - a + dt >= min_dur_s - 1e-9]


def scan_report(df: pd.DataFrame, gap_s: float = 0.5, min_dur_s: float = MIN_DUR_S, thr: float = 0.5) -> str:
    """Share of frames called OUT, the OUT runs (consecutive OUT frames, gaps < gap_s merged) and how many
    of them pass the minimum-duration rule (`out_episodes`)."""
    o = df[df["out"]]
    runs = []
    for t in o["t"]:
        if runs and t - runs[-1][1] <= gap_s:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    eps = out_episodes(df["t"], df["p_out"], thr=thr, min_dur_s=min_dur_s)
    lines = [f"frames {len(df)}, called OUT {len(o)} = {len(o) / max(len(df), 1):.1%}, OUT runs {len(runs)}, "
             f"OUT episodes >= {min_dur_s:.2f} s: {len(eps)}"]
    lines += [f"  OUT {a:7.2f}-{b:7.2f} s ({b - a:.2f} s), max p_out "
              f"{df[(df['t'] >= a) & (df['t'] <= b)]['p_out'].max():.2f}" for a, b in runs]
    return "\n".join(lines)


def train(data: Path, epochs: int = 30, model: str = "yolov8s-cls.pt", project: str = "runs/train",
          rrc_scale: float = 0.0, erasing: float = 0.1) -> None:
    """The settings the final model was trained with. Left/right flips are off: which side of the car the white
    line is on matters. AMP is off (it went NaN on this data) and the BatchNorm guard repairs a rare corrupt batch."""
    from ultralytics import YOLO
    import torch
    m = YOLO(model)
    guard = BNGuard()
    m.add_callback("on_train_start", guard.start)
    m.add_callback("on_train_epoch_end", guard.check)
    m.train(data=str(Path(data).resolve()), epochs=epochs, imgsz=448, fliplr=0.0, hsv_h=0.02,
            project=str(Path(project).resolve()), name="train", exist_ok=True, device=0 if torch.cuda.is_available() else "cpu",
            batch=32, workers=4, scale=rrc_scale, erasing=erasing, amp=False)
    if guard.repairs:
        print(f"BatchNorm guard repaired running statistics in {len(guard.repairs)} epoch(s): {guard.repairs}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="digital-stewards", description="Track-limits steward: OUT/IN frame classifier.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=30); t.add_argument("--model", default="yolov8s-cls.pt")
    t.add_argument("--project", default="runs/train")
    e = sub.add_parser("eval"); e.add_argument("--data", type=Path, required=True)
    e.add_argument("--weights", type=Path, required=True); e.add_argument("--split", default="val")
    s = sub.add_parser("scan", help="p(out) over any onboard video")
    s.add_argument("video", type=Path); s.add_argument("--weights", type=Path, required=True)
    s.add_argument("--from", dest="t0", type=float, default=0.0); s.add_argument("--to", dest="t1", type=float, default=None)
    s.add_argument("--step", type=int, default=3, help="every Nth frame"); s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--out", type=Path, required=True, help="CSV of frame, t, p_out, out")
    s.add_argument("--rows", default=None, help="crop rows 'top,bottom' as fractions of the height (default 0.22,0.68)")
    s.add_argument("--min-dur", type=float, default=MIN_DUR_S)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.cmd == "train":
        train(a.data, a.epochs, a.model, a.project)
    elif a.cmd == "eval":
        print(eval_report(evaluate(a.data, a.weights, a.split)))
    else:
        rows = tuple(float(x) for x in a.rows.split(",")) if a.rows else CROP
        df = scan(a.video, a.weights, a.t0, a.t1, a.step, a.thr, rows=rows, square=True)
        a.out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
        print(scan_report(df, min_dur_s=a.min_dur, thr=a.thr))
