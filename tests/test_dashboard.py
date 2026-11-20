import numpy as np

from digital_stewards.penalties import Steward, load_rules
from digital_stewards import camgate


def run(st, spans, dur=200.0, dt=0.1, skip=()):
    """Feed a timeline with OUT (p=0.9) inside `spans`, IN elsewhere; `skip` = not-onboard spans."""
    for t in np.arange(0, dur, dt):
        out = any(a <= t <= b for a, b in spans)
        on = not any(a <= t <= b for a, b in skip)
        st.feed(round(t, 2), 0.9 if out else 0.01, on)
    st.finish()
    return st


def test_fia_escalation():
    st = run(Steward(load_rules()), [(10, 10.5), (40, 40.4), (70, 70.3), (100, 100.5), (130, 130.5), (160, 160.5)])
    kinds = [e["kind"] for e in st.log]
    assert st.offences == 6
    assert kinds.count("flag") == 1 and kinds.index("flag") == 3          # after offence #3
    assert [e["penalty_s"] for e in st.log if e["kind"] == "penalty"] == [5.0]   # offence #4 only
    assert kinds.count("report") == 2                                     # #5, #6
    assert st.penalty_s == 5.0


def test_custom_rule_three_strikes_five_seconds():
    r = load_rules(); r["penalties"] = {3: 5.0}
    st = run(Steward(r), [(10, 10.5), (40, 40.5), (70, 70.5)])
    assert st.penalty_s == 5.0


def test_short_blip_and_same_corner_merge():
    # at 30 Hz one frame (0.033 s) is below min_dur_s = 0.1 s; two OUT bursts 1 s apart are one offence
    st = run(Steward(load_rules()), [(10, 10.0)], dt=1 / 30)
    assert st.offences == 0
    st = run(Steward(load_rules()), [(50, 50.3), (51.3, 51.6)])
    assert st.offences == 1


def test_no_verdict_on_other_camera():
    st = run(Steward(load_rules()), [(20, 25)], skip=[(15, 30)])
    assert st.offences == 0


def test_laps_tagged():
    st = run(Steward(load_rules(), laps=[0, 100]), [(120, 120.5)])
    assert st.log[0]["lap"] == 2


def test_skip_windows():
    t = np.arange(0, 60, 0.25)
    s = np.where((t >= 10) & (t < 20), 0.1, 0.9)
    s[t == 40] = 0.1                                      # single low sample = glare, not a camera cut
    w = camgate.skip_windows(t, s)
    assert len(w) == 1 and w[0][0] <= 10 and w[0][1] >= 19.75


def test_gate_scores_same_view_high():
    rng = np.random.default_rng(0)
    car = np.zeros((300, 500, 3), np.uint8); car[200:, 150:350] = 200      # fixed "car body"
    frames = []
    for _ in range(20):
        f = car.copy(); f[:200] = rng.integers(0, 255, (200, 500, 3), dtype=np.uint8)   # changing road / sky
        frames.append(f)
    ref = camgate.fit_reference([camgate.edge_map(f) for f in frames])
    other = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    assert camgate.score(camgate.edge_map(frames[0]), ref) > camgate.score(camgate.edge_map(other), ref)


def test_no_merge_across_cut():
    # two OUT bursts 2 s apart are one offence - unless a trimmed clip cuts between them
    assert run(Steward(load_rules()), [(50, 50.5), (52.5, 53.0)]).offences == 1
    assert run(Steward(load_rules(), cuts=[51.5]), [(50, 50.5), (52.5, 53.0)]).offences == 2
