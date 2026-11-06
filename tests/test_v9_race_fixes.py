"""Regresiones de la carrera real en Spa (2026-11-06, Lanzo V10, 11 coches).

Lo que pasó en el log y no debe volver a pasar:
  - empezaba frases largas al final de la recta y seguía hablando en la curva
  - "Bajaste 2.3s... Rival cerrando" 7 veces en la misma vuelta
  - "¡Fuera de pista!" en cada piano, y dicho 20 s tarde
  - la vuelta 1 salía "récord personal"
  - el mismo mensaje diferido encolado varias veces
"""
from __future__ import annotations

import math

import pytest

import victor_corner as vc


@pytest.fixture
def race(eng, spoken, monkeypatch):
    eng._alert_times.clear()
    eng._deferred_msgs.clear()
    t = eng.Telemetry()
    t.session, t.status, t.track, t.lap = 2, 2, "spa", 3
    t.speed, t.spline, t.track_len = 250.0, 0.30, 7000.0
    t.cars, t.position = 11, 4
    eng._current = t
    monkeypatch.setattr(eng, "_in_braking_zone", lambda: False)
    monkeypatch.setattr(eng._corner, "in_corner", lambda: False)
    m = vc.CornerMap(buckets=1000)
    m.reset(("spa", ""), None)
    monkeypatch.setattr(eng, "_corner_map", m)
    yield eng, t, m
    eng._deferred_msgs.clear()


def _texts(spoken):
    return [txt for _p, txt in spoken]


def test_no_empieza_frase_si_la_curva_esta_cerca(race, spoken):
    eng, t, m = race
    # curva aprendida 100 m por delante (2 vueltas)
    for lap in (1, 2):
        for k in range(10):
            m.learn(0.30 + (100 + k * 7) / 7000, lap, True)
    msg = "Combustible bajo, te quedan tres vueltas. Prepara la estrategia de pit."
    eng._say(msg, priority=1)
    assert not spoken, "a 250 km/h 100 m son 1,4 s: no cabe la frase"
    assert any(m_[1] == msg for m_ in eng._deferred_msgs)


def test_habla_si_la_recta_alcanza(race, spoken):
    eng, t, m = race
    for lap in (1, 2):
        m.learn(0.30 + 1500 / 7000, lap, True)   # curva a 1,5 km
    eng._say("Gomas en temperatura.", priority=2)
    assert _texts(spoken) == ["Gomas en temperatura."]


def test_spotter_habla_siempre(race, spoken, monkeypatch):
    eng, t, m = race
    monkeypatch.setattr(eng._corner, "in_corner", lambda: True)
    eng._say("Coche izquierda.", priority=0)
    assert _texts(spoken) == ["Coche izquierda."]


def test_diferidos_sin_duplicados(race, spoken, monkeypatch):
    eng, t, m = race
    monkeypatch.setattr(eng._corner, "in_corner", lambda: True)
    for _ in range(5):
        eng._say("Esa vuelta fue más lenta.", priority=2)
    assert len(eng._deferred_msgs) == 1


def test_ttl_propio_del_mensaje(race, monkeypatch):
    eng, t, m = race
    monkeypatch.setattr(eng._corner, "in_corner", lambda: True)
    eng._say("Fuera de pista, vuelve con cuidado.", priority=1, ttl=8)
    assert eng._deferred_msgs[0][3] == 8


def test_vuelta_lenta_una_sola_vez_por_vuelta(race, spoken, monkeypatch):
    eng, t, m = race
    laps = [eng.LapRecord(lap=1, time_ms=153000, fuel_used=2.7),
            eng.LapRecord(lap=2, time_ms=156300, fuel_used=2.7)]
    eng._pace_drop_lap = -1
    monkeypatch.setattr(eng, "_connected", True)
    t.car = "rss_gtm_lanzo_v10"
    with eng._lock:
        eng._lap_history.clear(); eng._lap_history.extend(laps)
    for _ in range(4):
        eng._alert_times.pop("rival_close", None)   # aunque el cooldown venza
        eng._check_proactive_alerts()
    hits = [x for x in _texts(spoken) if "más lenta" in x]
    assert len(hits) == 1, _texts(spoken)


def test_vuelta_1_no_es_record(race, spoken, monkeypatch):
    eng, t, m = race
    prompts = []
    monkeypatch.setattr(eng, "_groq", lambda p, **k: prompts.append(p) or "ok")
    t.best_lap_ms = 153935          # AC ya puso esta vuelta como mejor
    eng._announce_lap(1, 153935, 2.7, 4)
    assert "RÉCORD" not in prompts[-1].upper() or "NO DIGAS RÉCORD" in prompts[-1].upper()
    assert "VUELTA RÁPIDA" not in prompts[-1]
    assert "2:33.935" in prompts[-1]          # tiempo en m:ss, no "153.935 s"
    eng._announce_lap(2, 152000, 2.7, 4)      # mejora la 1 → ahora sí
    assert "VUELTA RÁPIDA" in prompts[-1]
    eng._announce_lap(3, 154000, 2.7, 4)
    assert "VUELTA RÁPIDA" not in prompts[-1]
