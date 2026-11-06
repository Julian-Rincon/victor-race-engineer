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


# ─── Mapa de curvas (mirar adelante) ────────────────────────────────────────

import struct


def _write_ai(path, pts):
    """fast_lane.ai mínimo: cabecera + (x, y, z, dist, id) por punto."""
    b = bytearray(struct.pack("<4i", 7, len(pts), 0, 0))
    d = 0.0
    prev = None
    for i, (x, z) in enumerate(pts):
        if prev:
            d += math.hypot(x - prev[0], z - prev[1])
        b += struct.pack("<4fi", x, 0.0, z, d, i)
        prev = (x, z)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(b))


def _stadium(straight=1000.0, radius=50.0, step=2.0):
    """Óvalo: recta 1000 m, curva de 50 m de radio, recta, curva."""
    pts = []
    n = int(straight / step)
    pts += [(i * step, 0.0) for i in range(n)]
    m = int(math.pi * radius / step)
    pts += [(straight + radius * math.sin(k * step / radius), radius - radius * math.cos(k * step / radius)) for k in range(m)]
    pts += [(straight - i * step, 2 * radius) for i in range(n)]
    pts += [(-radius * math.sin(k * step / radius), radius + radius * math.cos(k * step / radius)) for k in range(m)]
    return pts


def test_corner_map_from_ai_line(tmp_path):
    _write_ai(tmp_path / "content/tracks/oval/ai/fast_lane.ai", _stadium())
    m = vc.CornerMap(buckets=1000)
    m.reset(("oval", ""), tmp_path)
    assert m.source == "línea IA"
    L = m.track_len
    s_mid_straight = 500 / L
    s_curve = (1000 + 80) / L
    assert not m.is_corner(s_mid_straight)
    assert m.is_corner(s_curve)
    # a mitad de recta quedan ~500 m menos la frenada previa
    ahead = m.clear_ahead_m(s_mid_straight, L)
    assert 300 < ahead < 520
    # pegado a la curva: la zona de frenada ya cuenta
    assert m.clear_ahead_m((1000 - 60) / L, L) == 0.0


def test_corner_map_layout_dir_preferred(tmp_path):
    _write_ai(tmp_path / "content/tracks/t/gp/ai/fast_lane.ai", _stadium())
    m = vc.CornerMap(buckets=500)
    m.reset(("t", "gp"), tmp_path)
    assert m.source == "línea IA"
    m.reset(("nope", ""), tmp_path)
    assert m.source == "aprendido" and m.clear_ahead_m(0.5, 4000) == math.inf


def test_corner_map_learns_only_after_two_laps():
    m = vc.CornerMap(buckets=100)
    m.reset(("x", ""), None)
    m.learn(0.505, lap=1, in_corner=True)
    m.learn(0.505, lap=1, in_corner=True)      # misma vuelta no cuenta doble
    assert not m.is_corner(0.505)             # un trompo no marca la recta
    m.learn(0.505, lap=2, in_corner=True)
    assert m.is_corner(0.505)
    assert m.clear_ahead_m(0.40, 1000) == 100.0


def test_speech_seconds_is_capped():
    assert vc.speech_seconds("Hola.") < 1.0
    assert vc.speech_seconds("x" * 1000) == vc.SPEECH_CAP_S
