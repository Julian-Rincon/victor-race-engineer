"""Detector de curva genérico: en curva solo habla el spotter."""
from __future__ import annotations

import math

import victor_corner as vc


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def _drive(det, clk, pts, speed=200.0, brake=0.0, dt=0.1):
    for x, z in pts:
        clk.t += dt
        det.update(x, z, speed, brake)


def _straight(n, step=5.5, x0=0.0):
    return [(x0 + i * step, 0.0) for i in range(n)]


def _arc(n, radius=60.0, step=5.5):
    out = []
    for i in range(n):
        a = i * step / radius
        out.append((radius * math.sin(a), radius * (1 - math.cos(a))))
    return out


def test_straight_is_not_a_corner():
    clk = Clock(); d = vc.CornerDetector(clock=clk)
    _drive(d, clk, _straight(30))
    assert not d.in_corner()
    assert d.yaw_rate < 1.0


def test_arc_is_a_corner_and_hold_after_exit():
    clk = Clock(); d = vc.CornerDetector(clock=clk)
    _drive(d, clk, _arc(20))                       # ~52 °/s
    assert d.in_corner()
    last = _arc(20)[-1]
    # salida: recta corta (< HOLD_S) sigue bloqueando...
    _drive(d, clk, [(last[0] + 0.1, last[1] + 5.5 * (i + 1)) for i in range(3)])
    assert d.in_corner()
    # ...y tras recta suficiente vuelve a hablar
    _drive(d, clk, [(last[0] + 0.1, last[1] + 5.5 * (i + 4)) for i in range(20)])
    assert not d.in_corner()


def test_braking_counts_as_corner():
    clk = Clock(); d = vc.CornerDetector(clock=clk)
    _drive(d, clk, _straight(10), brake=0.8)
    assert d.in_corner()


def test_slow_speed_never_blocks_voice():
    clk = Clock(); d = vc.CornerDetector(clock=clk)
    _drive(d, clk, _arc(20, radius=10, step=1.0), speed=20.0)
    assert not d.in_corner()


def test_stale_data_does_not_block():
    clk = Clock(); d = vc.CornerDetector(clock=clk)
    _drive(d, clk, _arc(20))
    clk.t += 5
    assert not d.in_corner()


def test_teleport_resets_heading():
    clk = Clock(); d = vc.CornerDetector(clock=clk)
    _drive(d, clk, _straight(10))
    _drive(d, clk, [(5000.0, 5000.0)] + _straight(10, x0=5000.0))
    assert d.yaw_rate < 1.0 and not d.in_corner()


def test_tts_worker_splits_long_messages(eng):
    assert eng._split_sentences("Uno. Dos! Tres?") == ["Uno.", "Dos!", "Tres?"]
