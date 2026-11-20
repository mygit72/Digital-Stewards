"""Make a new classifier training set = an existing one + the TRAIN frames of one or more `frame-clf build`
datasets (+ YouTube-like degraded copies of those frames). The added datasets' held-out frames are NOT
copied: they stay separate test sets.

    python scripts/frameclf_add_clip.py --base datasets/f1_offtrack_v9 --add datasets/own_1359sq \
        --out datasets/f1_offtrack_v10 --degrade-copies 1
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digital_stewards.augment import photometric_augment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--add", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--degrade-copies", type=int, default=1)
    ap.add_argument("--out-copies", type=int, default=None, help="degraded copies for OUT frames (default: --degrade-copies)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists - pick a new name")
    shutil.copytree(a.base, a.out, ignore=shutil.ignore_patterns("*.cache", "eval_*.csv"))
    man = pd.read_csv(a.out / "manifest.csv")
    man["file"] = [str(a.out / Path(f).relative_to(a.base)) for f in man["file"]]
    rng = np.random.default_rng(0)
    rows = []
    for d in a.add:
        m = pd.read_csv(d / "manifest.csv")
        m = m[m["split"] == "train"]
        for r in m.itertuples():
            src = Path(r.file)
            dst = a.out / "train" / r.cls / f"{d.name}_{src.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            rows.append({"wall_time": r.wall_time, "cls": r.cls, "split": "train", "group": r.group,
                         "clip": r.clip, "file": str(dst)})
            nk = a.out_copies if (r.cls == "out" and a.out_copies is not None) else a.degrade_copies
            if nk:
                img = cv2.imread(str(src))
                for k in range(nk):
                    p = dst.with_name(f"{dst.stem}_deg{k}.jpg")
                    cv2.imwrite(str(p), photometric_augment(img, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    rows.append({"wall_time": r.wall_time, "cls": r.cls, "split": "train",
                                 "group": f"{r.group}_deg", "clip": r.clip, "file": str(p)})
        print(f"{d}: {len(m)} train frames added (+{len(m) * a.degrade_copies} degraded copies)")
    man = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)
    man.to_csv(a.out / "manifest.csv", index=False)
    print(man.groupby(["split", "cls"]).size().to_string())


if __name__ == "__main__":
    main()
