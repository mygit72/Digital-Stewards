"""
Track-limits steward log with FIA-style escalation, fed frame by frame (so a dashboard can show it live).

FIA practice in races (race director's notes, 2023+): every track-limits offence deletes the lap time; after
the 3rd the driver gets the black-and-white flag; the 4th is a 5-second time penalty; further offences are
reported to the stewards. All of it is configurable (`configs/penalties.yaml`).

    st = Steward(rules)
    for t, p_out, onboard in frames:          # in time order
        st.feed(t, p_out, onboard)
    st.finish()
    st.log                                    # list of dicts: t, kind, text (+ offence no., penalty_s)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RULES = {
    "out_thr": 0.5,          # p(out) >= this = the model says OUT
    "min_dur_s": 0.1,        # an OUT episode must last this long (frameclf.MIN_DUR_S)
    "gap_s": 0.12,           # OUT frames closer than this are one episode
    "merge_s": 3.0,          # episodes closer than this are one offence (one corner)
    "bw_flag_after": 3,      # black-and-white flag after this many offences
    "penalties": {4: 5},     # offence number -> time penalty (s)
    "report_after": 4,       # offences after this one are reported to the stewards
}


def load_rules(path: Path | None = None) -> dict:
    r = dict(DEFAULT_RULES)
    if path and Path(path).exists():
        r.update(yaml.safe_load(Path(path).read_text(encoding="utf8")) or {})
    r["penalties"] = {int(k): float(v) for k, v in r["penalties"].items()}
    return r


@dataclass
class Steward:
    rules: dict = field(default_factory=lambda: dict(DEFAULT_RULES))
    laps: list[float] | None = None            # lap start times (s), optional
    cuts: list[float] | None = None            # edit points of a trimmed clip (s): nothing is joined across them
    log: list[dict] = field(default_factory=list)
    offences: int = 0
    penalty_s: float = 0.0
    _run: list | None = None                   # current OUT run [t0, t1, pmax]
    _last_offence_t: float = -1e9
    _dt: float = 0.1
    _prev_t: float | None = None

    def lap_at(self, t: float) -> int | None:
        if not self.laps:
            return None
        n = sum(1 for s in self.laps if s <= t)
        return n or None

    def feed(self, t: float, p_out: float, onboard: bool = True) -> None:
        if self.cuts and self._prev_t is not None and any(self._prev_t < c <= t for c in self.cuts):
            if self._run:                          # a new piece of footage starts: close what was running
                self._close()
            self._last_offence_t = -1e9
        if self._prev_t is not None and t > self._prev_t:
            self._dt = min(self._dt, t - self._prev_t) if self._dt else t - self._prev_t
        self._prev_t = t
        hit = onboard and p_out >= self.rules["out_thr"]
        if self._run and (t - self._run[1] > self.rules["gap_s"] or not onboard):
            self._close()
        if hit:
            if self._run:
                self._run[1] = t; self._run[2] = max(self._run[2], p_out)
            else:
                self._run = [t, t, p_out]

    def finish(self) -> None:
        if self._run:
            self._close()

    def _close(self) -> None:
        t0, t1, pmax = self._run; self._run = None
        if t1 - t0 + self._dt < self.rules["min_dur_s"] - 1e-9:
            return
        if t0 - self._last_offence_t <= self.rules["merge_s"]:
            self._last_offence_t = t1           # same corner, same offence
            return
        self._last_offence_t = t1
        self.offences += 1; n = self.offences; lap = self.lap_at(t0)
        where = f"lap {lap}" if lap else "this lap"
        self._add(t0, "offence", f"Track limits #{n}: all four wheels off (model p={pmax:.2f}, {t1 - t0 + self._dt:.1f} s)"
                  f" - {where} time deleted", offence=n, t_end=t1, p=round(pmax, 3), lap=lap)
        if n == self.rules["bw_flag_after"]:
            self._add(t0, "flag", "Black and white flag - track limits")
        if n in self.rules["penalties"]:
            s = self.rules["penalties"][n]; self.penalty_s += s
            self._add(t0, "penalty", f"{s:g}-second time penalty - track limits (offence #{n})", penalty_s=s)
        elif n > self.rules["report_after"]:
            self._add(t0, "report", f"Track limits offence #{n} reported to the stewards")

    def _add(self, t: float, kind: str, text: str, **kw) -> None:
        self.log.append({"t": round(t, 2), "kind": kind, "text": text, **kw})
