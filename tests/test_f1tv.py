"""F1 TV real-onboard helpers: block splitting from the OCR scan, player-time parsing, crop shape."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from f1tv_build import secs  # noqa: E402
from f1tv_ocr_blocks import blocks_from_scan, parse_clock  # noqa: E402
from f1tv_scan_blocks import frame_crop  # noqa: E402


def _scan(segments):
    """segments: (driver, player_start_s, rec_start_s, n_seconds); one OCR row per recording second."""
    rows = []
    for drv, p0, r0, n in segments:
        rows += [dict(t=float(r0 + k), player_s=float(p0 + k), clock="", driver=drv) for k in range(n)]
    return pd.DataFrame(rows)


def test_seek_and_driver_change_split_blocks():
    d = _scan([("RUS", 6000, 0, 30),      # plays 30 s
               ("RUS", 600, 31, 20),      # user seeks back: same driver, new offset -> new block
               ("ALO", 620, 52, 15)])     # driver switch -> new block
    b = blocks_from_scan(d)
    assert list(b.driver) == ["RUS", "RUS", "ALO"]
    assert list(b.off) == [6000.0, 569.0, 568.0]
    assert b["from"].iloc[0] == "1:40:00"


def test_short_and_unreadable_runs_dropped():
    d = _scan([("RUS", 100, 0, 20), ("ALO", 500, 21, 5)])       # 5 s of ALO: shorter than min_len
    d.loc[3, "player_s"] = np.nan; d.loc[4, "driver"] = "?"       # OCR misses inside a block are tolerated
    b = blocks_from_scan(d, min_len=8)
    assert list(b.driver) == ["RUS"] and b.t1.iloc[0] == 19


def test_pause_inside_block_breaks_it():
    d = _scan([("HAD", 3000, 0, 20)])
    d.loc[d.t >= 10, "t"] += 30            # recording runs on 30 s while the player clock was paused
    b = blocks_from_scan(d)
    assert len(b) == 2


def test_secs_parses_player_times():
    assert secs("1:41:08.3") == 6068.3
    assert secs("0:10:03") == 603


def test_frame_crop_is_square():
    im = np.zeros((1080, 1920, 3), np.uint8)
    c = frame_crop(im)
    assert c.shape[0] == c.shape[1] == 448


def test_parse_clock_both_formats():
    assert parse_clock("01:41:08") == 6068
    assert parse_clock("26:12") == 1572
    assert np.isnan(parse_clock("29:"))
    assert parse_clock("01:1616:04") == 4564      # overlapping OCR boxes "01:16" + "16:04" (2025 quali)
    assert parse_clock("01:16") == 76             # a lone mm:ss stays mm:ss
    assert len(blocks_from_scan(_scan([("RUS", 0, 0, 3)]))) == 0      # nothing long enough -> empty, no crash
