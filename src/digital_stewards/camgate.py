"""
Camera gate: is this frame the forward onboard view? If not, the steward gives no verdict (skip).

Broadcast onboard feeds (F1 TV, YouTube) switch cameras: rear-facing, halo / cockpit, close-ups. The OUT/IN model
only knows the forward view (car nose, front tyres and road ahead), so on other views its verdict is meaningless
(2026-11-19: v12 / v13 called 10.6% / 7.3% of the 2025 race's rear-camera frames OUT).

Method, per clip (works for any car / livery without setup): in the forward view the car body (nose, wing, halo,
front tyres, helmet) sits still in the picture, so its EDGES are in the same pixels in every frame, while the road
changes. Reference = median gradient map over frames sampled from the clip (most of an onboard clip is the forward
view); its strongest pixels = the car body's edges. A frame's score = share of that edge mass still present.
Kerbs, paint and run-off under the car only ADD edges, so an excursion does not lower the score; a different
camera removes the car's edges and the score drops.

    g = edge_map(picture)                     # per sampled frame (picture = the onboard video area)
    ref = fit_reference(list_of_edge_maps)
    s = score(g, ref)                         # ~1 forward view, low = other camera
    skip = skip_windows(t, scores)            # [(t0, t1), ...] runs below the threshold, padded
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = (192, 108)        # edge maps are computed at this size (w, h)
REF_TOP = 0.12           # reference mask = the strongest 12% of the median edge map (the car body's edges)
THR = 0.58               # score below this = not the forward onboard view (2025 race, log edges: forward view
                         # min 0.62, deletions >= 0.67; rear / halo / close-up cameras 95% < 0.53)
MIN_DUR_S = 0.5          # a skip window must last this long (single dark / glare frames are not camera cuts)
PAD_S = 0.5              # padding around a skip window (cuts often have a fade / motion blur)


def edge_map(picture: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the log brightness (so the car's edges keep their strength in deep shadow - the
    2025 race's main straight under the grandstand dropped to 0.51 with plain brightness, 0.81 with log),
    small and normalised by its own 95th percentile."""
    g = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY) if picture.ndim == 3 else picture
    g = np.log1p(cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)) * 40
    g = cv2.GaussianBlur(g, (3, 3), 0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return np.clip(m / (np.percentile(m, 95) + 1e-6), 0, 1.5)


def fit_reference(maps: list[np.ndarray] | np.ndarray, max_n: int = 400) -> dict:
    """Reference from edge maps sampled across the clip (median: the forward view must be the majority)."""
    maps = np.asarray(maps)
    if len(maps) > max_n:
        maps = maps[np.linspace(0, len(maps) - 1, max_n).astype(int)]
    med = np.median(maps, axis=0)
    mask = med >= np.quantile(med, 1 - REF_TOP)
    return {"median": med, "mask": mask, "mass": float(med[mask].sum())}


def score(g: np.ndarray, ref: dict) -> float:
    """Share of the reference's car-body edge mass present in this frame (0..1)."""
    m = ref["mask"]
    return float(np.minimum(g[m], ref["median"][m]).sum() / (ref["mass"] + 1e-6))


def skip_windows(t, s, thr: float = THR, min_dur_s: float = MIN_DUR_S, pad_s: float = PAD_S,
                 gap_s: float = 1.0) -> list[tuple[float, float]]:
    """Runs of score < thr (samples closer than gap_s joined) lasting >= min_dur_s, padded by pad_s."""
    t = np.asarray(t, dtype=float); low = np.asarray(s, dtype=float) < thr
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    runs = []
    for ti in t[low]:
        if runs and ti - runs[-1][1] <= gap_s:
            runs[-1][1] = ti
        else:
            runs.append([ti, ti])
    return [(a - pad_s, b + pad_s) for a, b in runs if b - a + dt >= min_dur_s - 1e-9]


def in_windows(t, windows) -> np.ndarray:
    """Boolean mask: t inside any (t0, t1) window."""
    t = np.asarray(t, dtype=float); m = np.zeros(len(t), bool)
    for a, b in windows:
        m |= (t >= a) & (t <= b)
    return m
