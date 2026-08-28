"""Spotter v6 — overlap métrico estilo CrewChief. Migrado a pytest 2026-08-28.

Antes: "al lado" = gap temporal en spline < 1.2 s (≈50 m a ritmo de carrera →
falsos "coche a la derecha" constantes) y sin debounce (jitter a 10 Hz).
Ahora: overlap real por geometría (largo de coche), zona de 20 m, confirmación
de aparición y retardo de "despejado", y aviso de 3-wide.

No necesita AC ni audio real. El reloj se fake-ea con la fixture `clock`.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def clock(eng, monkeypatch):
    """Reloj monotónico controlado. `clock.advance(s)` mueve el tiempo que ve
    engineer (time.time). monkeypatch lo restaura al terminar el test."""
    now = [1000.0]
    monkeypatch.setattr(eng.time, "time", lambda: now[0])

    class _Clock:
        def advance(self, secs: float) -> None:
            now[0] += secs

    return _Clock()


@pytest.fixture
def spotter(eng, clock, spoken):
    """engineer con spotter reseteado + reloj fake + captura de TTS.

    Devuelve un helper con:
      - car(long_m, lat_m): Telemetry con un rival en esa posición relativa
      - clear(): Telemetry sin rivales
      - feed(t, times, dt): corre _run_spotter avanzando el reloj
      - said (lista de textos anunciados)
    """
    eng._reset_spotter()
    spoken.clear()

    class _Spotter:
        said = spoken  # lista de (prioridad, texto)

        def texts(self) -> list[str]:
            return [txt for _pri, txt in spoken]

        def car(self, long_m: float, lat_m: float, lap: int = 5):
            t = eng.Telemetry()
            t.status = 2
            t.session = 2
            t.track_len = 4318.0
            t.speed = 150.0
            t.lap = lap
            t.spline = 0.500000
            t.wx, t.wz = 0.0, 0.0
            t.vx, t.vz = 0.0, 41.7
            t.cars_data = [{
                "idx": 1, "race_pos": 2, "lap": lap,
                "spline": t.spline + long_m / t.track_len,
                "speed": 150.0,
                "wx": lat_m, "wz": long_m,
                "driver_name": "Ayrton Senna", "in_pit": False,
            }]
            return t

        def clear(self):
            t = self.car(0, 0)
            t.cars_data = []
            return t

        def feed(self, t, times: int = 1, dt: float = 0.1) -> None:
            for _ in range(times):
                clock.advance(dt)
                eng._run_spotter(t)

        def reset(self) -> None:
            spoken.clear()
            eng._reset_spotter()
            spoken.clear()

    return _Spotter()


def test_car_directly_behind_is_not_alongside(spotter):
    spotter.feed(spotter.car(long_m=-10.0, lat_m=0.5), times=8)
    assert not spotter.said, f"no debería anunciar nada: {spotter.texts()}"


def test_car_25m_ahead_is_out_of_spotter_zone(spotter):
    spotter.feed(spotter.car(long_m=25.0, lat_m=2.0), times=8)
    assert not spotter.said, f"25 m está fuera de zona: {spotter.texts()}"


def test_real_overlap_right_not_announced_on_first_frame(spotter):
    spotter.feed(spotter.car(long_m=1.0, lat_m=2.2), times=1)
    assert not spotter.said, "el debounce debe tragarse el primer frame"


def test_real_overlap_right_announced_after_confirmation(spotter):
    spotter.feed(spotter.car(long_m=1.0, lat_m=2.2), times=4)
    assert any("derecha" in s.lower() for s in spotter.texts()), spotter.texts()


def test_clear_not_announced_immediately(spotter):
    spotter.feed(spotter.car(1.0, 2.2), times=4)  # primero anunciar al rival
    spotter.said.clear()
    spotter.feed(spotter.clear(), times=2)  # 0.2 s limpio
    assert not spotter.said, f"0.2 s limpio es muy poco para 'despejado': {spotter.texts()}"


def test_clear_announced_after_delay(spotter):
    spotter.feed(spotter.car(1.0, 2.2), times=4)
    spotter.said.clear()
    spotter.feed(spotter.clear(), times=11)  # ~1.1 s limpio
    assert any("despejado" in s.lower() for s in spotter.texts()), spotter.texts()


def test_single_frame_jitter_does_not_trigger_clear_and_reannounce(spotter):
    spotter.feed(spotter.car(1.0, 2.2), times=4)  # anunciado
    spotter.said.clear()
    spotter.feed(spotter.clear(), times=1)        # 1 frame perdido
    spotter.feed(spotter.car(1.0, 2.2), times=3)  # vuelve
    assert not spotter.said, f"jitter de 1 frame no debe generar mensajes: {spotter.texts()}"


def test_three_wide_announces_both_sides(spotter):
    t3 = spotter.car(1.0, 2.2)
    t3.cars_data.append({
        "idx": 2, "race_pos": 4, "lap": 5,
        "spline": t3.spline, "speed": 150.0,
        "wx": -2.2, "wz": -0.5,
        "driver_name": "Alain Prost", "in_pit": False,
    })
    spotter.feed(t3, times=4)
    assert any("ambos" in s.lower() for s in spotter.texts()), spotter.texts()
