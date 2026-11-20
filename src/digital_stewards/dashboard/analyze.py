"""
Analyse one recording for the dashboard, writing results as it goes (the page shows them live).

    python -m digital_stewards.dashboard.analyze <rec_dir>

<rec_dir>/meta.json: {"id", "name", "video", "profile": "f1tv" | "game" | "full", "weights", "hz", "laps"?}
<rec_dir>/timeline.csv (appended every batch): t, p_out, camera score (onboard = score >= camgate.THR and t not in
a skip window). <rec_dir>/state.json (small, rewritten every batch): status, processed_until (s), camera skip
windows, steward log, offence count, penalty total.

Profiles = where the onboard picture is and which rows the model sees (as in training):
  f1tv - F1 TV player not full screen (recordings/f1tv/*): picture y 50-910 x 40-1578, rows 0.12-0.72, square
  game - F1 25 T-cam full frame: rows 0.22-0.68, square
  full - any full-frame onboard: rows 0.12-0.72, square
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..penalties import Steward, load_rules
from .. import camgate
from ..classifier import crop

ROOT = Path(__file__).resolve().parents[3]
PROFILES = {
    "f1tv": {"rect": (50, 910, 40, 1578), "rows": (0.12, 0.72)},
    "game": {"rect": None, "rows": (0.22, 0.68)},
    "full": {"rect": None, "rows": (0.12, 0.72)},
}


def picture(im: np.ndarray, rect) -> np.ndarray:
    if rect is None:
        return im
    y0, y1, x0, x1 = rect
    return im[y0:y1, x0:x1]


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf8")
    for _ in range(20):                 # the server may be reading it (Windows): retry
        try:
            tmp.replace(path); return
        except PermissionError:
            time.sleep(0.05)


def analyse(rec_dir: Path) -> None:
    from ultralytics import YOLO
    meta = json.loads((rec_dir / "meta.json").read_text(encoding="utf8"))
    prof = PROFILES[meta.get("profile", "full")]
    video = ROOT / meta["video"]
    model = YOLO(str(ROOT / meta["weights"])); oi = [k for k, v in model.names.items() if v == "out"][0]
    rules = load_rules(ROOT / "configs" / "penalties.yaml")
    st = Steward(rules, laps=meta.get("laps"), cuts=meta.get("cuts"))
    run_id = time.time()                  # the page resets its cached timeline when this changes (restart)
    (rec_dir / "timeline.csv").write_text("", encoding="utf8")
    if meta.get("after"):                 # next part of the same race: wait for the previous part, carry its counts
        prev = rec_dir.parent / meta["after"] / "state.json"
        while True:
            try:
                ps_ = json.loads(prev.read_text(encoding="utf8"))
                if ps_.get("status") == "done":
                    break
            except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                pass
            write_json(rec_dir / "state.json", {"status": "waiting", "waiting_for": meta["after"], "run": run_id})
            time.sleep(5)
        st.offences, st.penalty_s = int(ps_.get("offences", 0)), float(ps_.get("penalty_s", 0))
        st.log.append({"t": 0.0, "kind": "carry", "text": f"Continues {meta['after']}: {st.offences} track-limits "
                       f"offence(s), {st.penalty_s:g} s penalty so far"
                       + (" - black and white flag already shown" if st.offences >= rules["bw_flag_after"] else "")})
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); dur = n / fps
    step = max(1, int(round(fps / float(meta.get("hz", 10)))))
    state = {"status": "reference", "duration": dur, "carry_offences": st.offences, "carry_penalty_s": st.penalty_s, "processed_until": 0.0, "skips": [], "log": [],
             "offences": 0, "penalty_s": 0.0, "started": time.time(), "cam_thr": camgate.THR,
             "run": run_id}
    tl = open(rec_dir / "timeline.csv", "w", encoding="utf8", newline="\n")
    write_json(rec_dir / "state.json", state)

    # camera reference from frames spread over the whole recording
    maps = []
    for f in np.linspace(0, n - 1, 300).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
        if ok:
            maps.append(camgate.edge_map(picture(im, prof["rect"])))
    ref = camgate.fit_reference(maps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    state["status"] = "running"
    ts, ps, ss, buf = [], [], [], []
    fed = 0                               # timeline entries already fed to the steward

    def flush():
        nonlocal fed
        if not buf:
            return
        res = model.predict([b[1] for b in buf], imgsz=448, verbose=False)
        for (t, _, s), r in zip(buf, res):
            ts.append(t); ps.append(float(r.probs.data[oi])); ss.append(s)
            tl.write(f"{t:.2f},{ps[-1]:.3f},{s:.3f}\n")
        tl.flush(); buf.clear()
        # skip windows over everything so far; only feed the steward up to where a window can't still grow
        skips = camgate.skip_windows(ts, ss)
        safe = ts[-1] - (camgate.MIN_DUR_S + camgate.PAD_S + 1.0)
        on = ~camgate.in_windows(ts, skips) & (np.asarray(ss) >= camgate.THR)
        while fed < len(ts) and ts[fed] <= safe:
            st.feed(ts[fed], ps[fed], bool(on[fed])); fed += 1
        state.update(processed_until=ts[-1], skips=[[round(a, 2), round(b, 2)] for a, b in skips], log=st.log,
                     offences=st.offences, penalty_s=st.penalty_s)
        write_json(rec_dir / "state.json", state)

    for f in range(n):
        if f % step:
            if not cap.grab():
                break
            continue
        ok, im = cap.read()
        if not ok:
            break
        pic = picture(im, prof["rect"])
        buf.append((round(f / fps, 2), crop(pic, rows=prof["rows"], square=True), camgate.score(camgate.edge_map(pic), ref)))
        if len(buf) >= 64:
            flush()
    flush()
    skips = camgate.skip_windows(ts, ss)
    on = ~camgate.in_windows(ts, skips) & (np.asarray(ss) >= camgate.THR)
    while fed < len(ts):
        st.feed(ts[fed], ps[fed], bool(on[fed])); fed += 1
    st.finish(); tl.close()
    state.update(status="done", processed_until=dur, log=st.log, offences=st.offences, penalty_s=st.penalty_s,
                 skips=[[round(a, 2), round(b, 2)] for a, b in skips])
    write_json(rec_dir / "state.json", state)


if __name__ == "__main__":
    analyse(Path(sys.argv[1]))
