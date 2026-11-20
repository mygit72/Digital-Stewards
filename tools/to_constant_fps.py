"""Rewrite a screen capture at a constant frame rate, repeating a frame wherever the capture dropped one.

    python scripts/to_constant_fps.py "recordings/video/2026-11-15 16-43-12.mkv" [more.mkv ...] [--fps 60]

The original is kept in `<folder>/_vfr_original/`; the converted file takes its place under the same
name, so clip.yaml paths and the file-name time stamp stay valid. Files with (almost) no dropped
frames are left alone unless --force is given.

Why: FFmpeg's `ddagrab` capture stamps each frame with the real time, and when the encoder cannot keep
up, frames are missing. OpenCV seeks by time stamp but reads frame after frame, so code that counts
frames while reading (`time = n / fps`) shows a later moment than it thinks.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digital_stewards.video_utils import dropped_frames, probe  # noqa: E402


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert(video: Path, fps: int, force: bool, cq: int) -> bool:
    info = probe(video)
    d = dropped_frames(video, info)
    if d is None:
        print(f"{video.name}: cannot count frames (no ffmpeg)")
        return False
    print(f"{video.name}: {d[0]} dropped frame(s) = {d[1]:.2f} s missing of {info['duration_s']:.0f} s")
    if d[1] <= 0.05 and not force:
        print("   nothing to do")
        return False
    tmp = video.with_name(video.stem + ".cfr_tmp" + video.suffix)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-y", "-i", str(video), "-vf", f"fps={fps}",
           "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(cq), "-g", str(2 * fps), "-an", str(tmp)]
    if subprocess.run(cmd).returncode != 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"ffmpeg failed on {video}")
    left = dropped_frames(tmp)
    if left is None or left[1] > 0.05:
        raise SystemExit(f"{tmp}: still {left} dropped after conversion - original left untouched")
    keep = video.parent / "_vfr_original"
    keep.mkdir(exist_ok=True)
    st = video.stat()
    shutil.move(str(video), str(keep / video.name))
    tmp.rename(video)
    os.utime(video, (st.st_atime, st.st_mtime))          # keep the end-of-capture time stamp
    print(f"   converted ({left[0]} frame(s) off); original in {keep / video.name}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--cq", type=int, default=21, help="NVENC quality (lower = better, bigger)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for v in a.videos:
        if not v.exists():
            raise SystemExit(f"not found: {v}")
        convert(v, a.fps, a.force, a.cq)


if __name__ == "__main__":
    main()
