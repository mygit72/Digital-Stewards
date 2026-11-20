"""Real-onboard OUT/IN frames from the F1 TV recordings of the 2026 COTA race (FIA-deleted laps, every OUT window
judged by eye from a zoom on both front tyres and the line - see configs/f1tv_cota2026_labels.yaml).

    python scripts/f1tv_build.py --out datasets/real_cota2026

train = drivers in `train_drivers`, val (= TEST, never train on it) = all other drivers. OUT frames: every frame of
each confirmed window; IN frames: every 0.5 s of the playing blocks, >= 2.5 s away from any OUT / unsure window
and >= 1 s from any `exclude` window (non-standard camera, overlays)
(the player clock is read to 1 s). Optional `hard_in` windows (model false alarms judged IN by eye) add IN frames
every 0.1 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from f1tv_scan_blocks import frame_crop  # noqa: E402


def secs(s: str) -> float:
    h, m, x = str(s).split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "configs" / "f1tv_cota2026_labels.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--in-gap", type=float, default=0.5)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.labels.read_text(encoding="utf8"))
    blocks = pd.read_csv(ROOT / cfg["blocks"])
    train_drv = set(cfg["train_drivers"])
    outs = [(e["block"], secs(e["from"]), secs(e["to"]), e["name"]) for e in cfg["out"]]
    avoid = [(b, f - 2.5, t + 2.5) for b, f, t, _ in outs] + \
            [(u["block"], secs(u["from"]) - 2.5, secs(u["to"]) + 2.5) for u in cfg.get("unsure", [])] +             [(u["block"], secs(u["from"]) - 1.0, secs(u["to"]) + 1.0) for u in cfg.get("exclude", [])]   # e.g. camera switches
    rows = []
    for bi, b in blocks.iterrows():
        split = "train" if b.driver in train_drv else "val"
        cap = cv2.VideoCapture(str(ROOT / b.video)); fps = cap.get(cv2.CAP_PROP_FPS)
        want = {}
        for ob, f, t, name in outs:
            if ob != bi:
                continue
            for p in np.arange(f, t + 1e-6, 1 / 30):
                want[int(round((p - b.off) * fps))] = ("out", name, p)
        for p in np.arange(b.t0 + b.off + 1, b.t1 + b.off - 1, a.in_gap):
            if any(ab == bi and lo <= p <= hi for ab, lo, hi in avoid):
                continue
            want.setdefault(int(round((p - b.off) * fps)), ("in", "clean", p))
        for h in cfg.get("hard_in", []):          # model false alarms judged IN by eye: sampled densely
            if h["block"] != bi:
                continue
            for p in np.arange(secs(h["from"]), secs(h["to"]) + 1e-6, 0.1):
                if not any(ab == bi and lo <= p <= hi for ab, lo, hi in avoid):
                    want.setdefault(int(round((p - b.off) * fps)), ("in", "hard_in", p))
        for fr in sorted(want):
            cls, grp, p = want[fr]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read()
            if not ok:
                continue
            path = a.out / split / cls / f"f1tv_b{bi:02d}_{b.driver}_{fr:06d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame_crop(im), [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(dict(wall_time=p, cls=cls, split=split, group=grp, clip=f"f1tv_{b.driver}", file=str(path)))
    man = pd.DataFrame(rows); man.to_csv(a.out / "manifest.csv", index=False)
    print(man.groupby(["split", "cls"]).size().to_string())
    print(man[man.cls == "out"].groupby(["split", "group"]).size().to_string())


if __name__ == "__main__":
    main()
