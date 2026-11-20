"""
Steward dashboard: recording on the left, the live steward log on the right.

    digital-stewards dashboard add <video> --name "NOR 2025 race p1" --profile f1tv [--blocks runs/.../blocks.csv]
                          [--after <id>] [--no-proxy] [--weights ...] [--hz 10]
    digital-stewards dashboard clock <id> --blocks runs/.../blocks.csv      (set / replace the player clock)
    digital-stewards dashboard serve [--port 8765]          -> http://127.0.0.1:8765
    digital-stewards dashboard list

`add` registers a recording in runs/dashboard/<id>/ and starts its analysis in the background
(steward.dashboard.analyze): the page shows results as they arrive. The page plays a 720p30 viewing copy
(runs/dashboard/<id>/play.mp4, < 2 GiB - Chrome won't open bigger files or our .mkv captures); the analysis reads
the original. `--blocks` = an F1 TV OCR blocks file
(scripts/f1tv_ocr_blocks.py): the page then shows the F1 TV player clock (= FIA-comparable) next to the
recording time. `--after <id>` = this recording continues that one (same race, next part): offence counts,
the flag and penalties carry over. Serves on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DASH = ROOT / "runs" / "dashboard"
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_WEIGHTS = "models/digital_steward_v13.pt"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "rec"


def rec_dirs() -> list[Path]:
    return sorted((d for d in DASH.glob("*") if (d / "meta.json").exists()), key=lambda d: d.name)


def start_analysis(d: Path) -> None:
    log = open(d / "analyze.log", "w")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([sys.executable, "-m", "digital_stewards.dashboard.analyze", str(d)], cwd=ROOT, stdout=log,
                     stderr=subprocess.STDOUT, creationflags=flags, env={**__import__("os").environ,
                                                                        "PYTHONPATH": str(ROOT / "src")})


MAX_PLAY_BYTES = 1_900_000_000   # Chrome 157 won't open a video file over 2 GiB (Format error; 2026-11-19 test)


def make_proxy(video: Path, out: Path) -> Path:
    """Viewing copy for the page: 1280x720, 30 fps, H.264 (GPU), bitrate chosen to stay under MAX_PLAY_BYTES.
    The analysis keeps reading the full-quality original; only playback uses this."""
    import imageio_ffmpeg
    import cv2
    if out.exists():
        return out
    c = cv2.VideoCapture(str(video)); dur = c.get(cv2.CAP_PROP_FRAME_COUNT) / (c.get(cv2.CAP_PROP_FPS) or 30); c.release()
    kbps = int(min(6000, MAX_PLAY_BYTES * 8 / max(dur, 1) / 1000 * 0.92))
    print(f"viewing copy (720p30, {kbps} kbit/s) -> {out}")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    base = [ff, "-v", "error", "-y", "-i", str(video), "-vf", "scale=1280:-2,fps=30", "-an", "-movflags", "+faststart"]
    rate = ["-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k"]
    try:
        subprocess.run(base + ["-c:v", "h264_nvenc", "-preset", "p4"] + rate + [str(out)], check=True)
    except subprocess.CalledProcessError:                       # no NVIDIA GPU: CPU encoder
        subprocess.run(base + ["-c:v", "libx264", "-preset", "veryfast"] + rate + [str(out)], check=True)
    return out


def clock_from_blocks(blocks: Path, video: Path) -> list[list[float]]:
    """[[rec_t0, rec_t1, offset], ...] for the rows of an OCR blocks file that belong to this video
    (player time = recording time + offset)."""
    import pandas as pd
    b = pd.read_csv(blocks)
    names = {video.name, video.with_suffix(".mkv").name}
    b = b[[Path(v).name in names for v in b["video"]]]
    return [[float(r.t0), float(r.t1), float(r.off)] for r in b.itertuples()]


def add(video: Path, name: str, profile: str, weights: str, hz: float, remux: bool, laps: Path | None,
        blocks: Path | None = None, after: str | None = None, cuts: Path | None = None) -> Path:
    video = video.resolve()
    rel = lambda v: v.relative_to(ROOT).as_posix() if v.is_relative_to(ROOT) else str(v)
    d = DASH / slug(name); d.mkdir(parents=True, exist_ok=True)
    meta = {"id": d.name, "name": name, "video": rel(video), "video_play": rel(video), "profile": profile,
            "weights": weights, "hz": hz, "added": time.time()}
    if laps:
        meta["laps"] = [float(x) for x in laps.read_text().split()]
    if blocks:
        meta["clock"] = clock_from_blocks(blocks, video)
    if cuts:
        meta["cuts"] = [float(x) for x in cuts.read_text().split()]
    if after:
        if not (DASH / after / "meta.json").exists():
            raise SystemExit(f"--after: no recording {after}")
        meta["after"] = after
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    start_analysis(d)
    print(f"added {d.name}; analysis started (log {d / 'analyze.log'})")
    if remux:                                    # viewing copy while the analysis runs
        meta["video_play"] = rel(make_proxy(video, d / "play.mp4"))
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    return d


def serve(port: int) -> None:
    from aiohttp import web

    def rec(req) -> Path:
        d = DASH / req.match_info["id"]
        if not (d / "meta.json").exists() or d.parent != DASH:
            raise web.HTTPNotFound()
        return d

    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def recordings(_):
        out = []
        for d in rec_dirs():
            m = json.loads((d / "meta.json").read_text(encoding="utf8"))
            s = d / "state.json"
            st = json.loads(s.read_text(encoding="utf8")) if s.exists() else {}
            out.append({**m, "status": st.get("status", "queued"), "duration": st.get("duration"),
                        "processed_until": st.get("processed_until", 0), "offences": st.get("offences", 0),
                        "penalty_s": st.get("penalty_s", 0)})
        return web.json_response(out)

    async def state(req):
        s = rec(req) / "state.json"
        for _ in range(5):
            try:
                return web.json_response(json.loads(s.read_text(encoding="utf8")) if s.exists() else {"status": "queued"})
            except (json.JSONDecodeError, PermissionError):
                time.sleep(0.05)
        raise web.HTTPServiceUnavailable()

    async def timeline(req):
        f = rec(req) / "timeline.csv"; off = int(req.query.get("from", 0))
        if not f.exists():
            return web.Response(text="", headers={"X-Next": str(off)})
        with open(f, "rb") as fh:
            fh.seek(off); data = fh.read()
        cut = data.rfind(b"\n") + 1                  # complete lines only
        return web.Response(body=data[:cut], content_type="text/plain", headers={"X-Next": str(off + cut)})

    async def trackpos(req):
        f = rec(req) / "trackpos.json"            # optional: car position on the circuit map, per 0.1 s
        return web.FileResponse(f) if f.exists() else web.json_response(None)

    async def video(req):
        m = json.loads((rec(req) / "meta.json").read_text(encoding="utf8"))
        p = Path(m.get("video_play") or m["video"]); p = p if p.is_absolute() else ROOT / p
        ctype = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm"}.get(p.suffix.lower(), "video/mp4")
        return web.FileResponse(p, headers={"Content-Type": ctype})

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/recordings", recordings), web.get("/api/rec/{id}/state", state),
                    web.get("/api/rec/{id}/timeline", timeline), web.get("/api/rec/{id}/video", video),
                    web.get("/api/rec/{id}/trackpos", trackpos),
                    web.static("/static", STATIC)])
    print(f"steward dashboard: http://127.0.0.1:{port}")
    web.run_app(app, host="127.0.0.1", port=port, print=None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="digital-stewards dashboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("video", type=Path); a.add_argument("--name", required=True)
    a.add_argument("--profile", choices=["f1tv", "game", "full"], default="full")
    a.add_argument("--weights", default=DEFAULT_WEIGHTS); a.add_argument("--hz", type=float, default=10)
    a.add_argument("--no-proxy", dest="remux", action="store_false", help="play the original file (only if < 2 GiB and browser-playable)"); a.add_argument("--laps", type=Path, help="text file: lap start times (s)")
    a.add_argument("--blocks", type=Path, help="F1 TV OCR blocks.csv -> show the player clock")
    a.add_argument("--after", help="id of the recording this one continues (offences carry over)")
    a.add_argument("--cuts", type=Path, help="edit points of a trimmed clip, one time (s) per line (make_highlight_clip.py writes <clip>.cuts.txt)")
    c = sub.add_parser("clock"); c.add_argument("id"); c.add_argument("--blocks", type=Path, required=True)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765)
    sub.add_parser("list")
    sub.add_parser("restart").add_argument("id")
    sub.add_parser("proxy").add_argument("id")
    x = ap.parse_args(argv)
    if x.cmd == "add":
        add(x.video, x.name, x.profile, x.weights, x.hz, x.remux, x.laps, x.blocks, x.after, x.cuts)
    elif x.cmd == "clock":
        mp = DASH / x.id / "meta.json"; m = json.loads(mp.read_text(encoding="utf8"))
        v = Path(m["video"]); m["clock"] = clock_from_blocks(x.blocks, v if v.is_absolute() else ROOT / v)
        mp.write_text(json.dumps(m, indent=1), encoding="utf8"); print("clock segments:", m["clock"])
    elif x.cmd == "serve":
        serve(x.port)
    elif x.cmd == "proxy":
        mp = DASH / x.id / "meta.json"; m = json.loads(mp.read_text(encoding="utf8")); v = Path(m["video"])
        out = make_proxy(v if v.is_absolute() else ROOT / v, DASH / x.id / "play.mp4")
        m["video_play"] = out.relative_to(ROOT).as_posix(); mp.write_text(json.dumps(m, indent=1), encoding="utf8")
        print("viewing copy:", out)
    elif x.cmd == "restart":
        start_analysis(DASH / x.id); print("analysis restarted:", x.id)
    else:
        for d in rec_dirs():
            st = json.loads((d / "state.json").read_text(encoding="utf8")) if (d / "state.json").exists() else {}
            print(f"{d.name:40s} {st.get('status', 'queued'):9s} offences {st.get('offences', 0)}")


if __name__ == "__main__":
    main()
