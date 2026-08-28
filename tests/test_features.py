"""Features de Victor — tiering LLM, gomas por compuesto, braking zones, gap con
tendencia, proyección de vuelta, modo mando, rotación de crash.

Migrado a pytest 2026-08-28 (antes: script standalone con check()/prints).
No necesita AC ni audio real.
"""
from __future__ import annotations

import collections

import pytest


@pytest.fixture
def victor(eng, spoken):
    """engineer con estado global reseteado a una carrera RBR en curso.

    Equivale al reset() del script viejo. `spoken` (de conftest) ya captura el
    TTS y restaura _tts_pq.put al terminar.
    """
    eng._alert_times.clear()
    eng._deferred_msgs.clear()
    eng._gap_snapshots.clear()
    eng._reset_zone_state()
    eng._connected = True
    t = eng.Telemetry()
    t.session = 2
    t.track = "ks_red_bull_ring"
    t.lap = 3
    t.speed = 150.0
    # 0.30 = recta larga RBR, FUERA de las 3 zonas de frenada
    t.spline = 0.30
    t.fuel_laps = 10.0
    t.fuel = 20.0
    t.cars = 10
    t.position = 4
    t.best_lap_ms = 90000.0  # 1:30, por encima del umbral de 30 s
    eng._current = t
    eng._car_db["ks_red_bull_ring_gt3"] = {"name": "Test GT3", "class": "GT3", "tags": ["gt3"]}
    eng._prev_dmg_max = 0.0
    eng._prev_position = -1
    return eng


@pytest.fixture
def no_groq(victor, monkeypatch):
    """Cuenta llamadas a _groq y devuelve "" — para verificar que las alertas
    proactivas son rule-based y no gastan una llamada al LLM."""
    calls: list = []
    monkeypatch.setattr(victor, "_groq", lambda *a, **k: calls.append(a) or "")
    return calls


def _texts(spoken):
    return [txt for _pri, txt in spoken]


def _rival_telemetry(eng):
    t = eng.Telemetry(speed=150, spline=0.50, track_len=4326, position=4,
                      cars=10, track="ks_red_bull_ring")
    t.cars_data = [{"idx": 5, "race_pos": 3, "speed": 148, "spline": 0.47, "driver_name": "Torres"}]
    return t


# ─── 1. LLM tiering — alertas proactivas son rule-based ───────────────────────

def test_fuel_critical_does_not_call_groq(victor, no_groq):
    victor._current.fuel_laps = 1.4  # crítico ≤ 1.8
    victor._check_proactive_alerts()
    assert no_groq == []


def test_fuel_critical_speaks_fuel_alert(victor, no_groq, spoken):
    victor._current.fuel_laps = 1.4
    victor._check_proactive_alerts()
    assert any("combustible" in t.lower() for t in _texts(spoken)), _texts(spoken)


def test_fuel_critical_is_priority_one(victor, no_groq, spoken):
    victor._current.fuel_laps = 1.4
    victor._check_proactive_alerts()
    assert any(pri == 1 for pri, _t in spoken)


def test_fuel_warn_does_not_call_groq(victor, no_groq):
    victor._current.fuel_laps = 3.0  # warn 1.8–3.5
    victor._check_proactive_alerts()
    assert no_groq == []


def test_fuel_warn_speaks_fuel_alert(victor, no_groq, spoken):
    victor._current.fuel_laps = 3.0
    victor._check_proactive_alerts()
    assert any("combustible" in t.lower() for t in _texts(spoken))


def test_tyre_overheat_does_not_call_groq(victor, no_groq):
    victor._current.tyre_fl = 108.0  # GT3 >95+10
    victor._current.tyre_fr = victor._current.tyre_rl = victor._current.tyre_rr = 85.0
    victor._car_db[""] = {"name": "GT3", "class": "GT3", "tags": ["gt3"]}
    victor._check_proactive_alerts()
    assert no_groq == []


def test_tyre_overheat_speaks_temperature_alert(victor, no_groq, spoken):
    victor._current.tyre_fl = 108.0
    victor._current.tyre_fr = victor._current.tyre_rl = victor._current.tyre_rr = 85.0
    victor._car_db[""] = {"name": "GT3", "class": "GT3", "tags": ["gt3"]}
    victor._check_proactive_alerts()
    assert any(("sobrecalentamiento" in t.lower() or "temperatura" in t.lower())
               for t in _texts(spoken)), _texts(spoken)


# ─── 2. Tyre temp dinámica por compuesto ─────────────────────────────────────

def test_gt3_tyre_range(victor):
    victor._car_db["ks_ferrari_488_gt3"] = {"name": "Ferrari 488 GT3", "class": "GT3", "tags": ["gt3"]}
    lo, hi = victor._parse_tyre_range("ks_ferrari_488_gt3")
    assert (lo, hi) == (75.0, 95.0)


def test_tyre_warn_fires_above_range_hi(victor, spoken):
    victor._car_db["ks_ferrari_488_gt3"] = {"name": "Ferrari 488 GT3", "class": "GT3", "tags": ["gt3"]}
    victor._current.car = "ks_ferrari_488_gt3"
    victor._current.tyre_fl = 97.0  # entre hi=95 y hi+10=105 -> warn
    victor._current.tyre_fr = victor._current.tyre_rl = victor._current.tyre_rr = 80.0
    victor._check_proactive_alerts()
    assert any(("fuera del rango" in t.lower() or "temperatura" in t.lower())
               for t in _texts(spoken)), _texts(spoken)


def test_cold_tyres_detected_below_threshold(victor, spoken):
    victor._car_db["ks_ferrari_488_gt3"] = {"name": "Ferrari 488 GT3", "class": "GT3", "tags": ["gt3"]}
    victor._current.car = "ks_ferrari_488_gt3"
    victor._current.tyre_fl = victor._current.tyre_fr = 60.0  # óptimo 75, frío ≈ 63
    victor._current.tyre_rl = victor._current.tyre_rr = 58.0
    victor._current.lap = 2
    victor._check_proactive_alerts()
    assert any(("fría" in t.lower() or "frías" in t.lower()) for t in _texts(spoken)), _texts(spoken)


# ─── 3. Braking zone — supresión en frenada ──────────────────────────────────

def test_in_braking_zone_true_at_turn_one(victor):
    victor._current.spline = 0.02  # RBR T1 wrap zone (0.96, 0.06)
    victor._current.speed = 220.0
    victor._current.track = "ks_red_bull_ring"
    assert victor._in_braking_zone()


def test_in_braking_zone_false_on_straight(victor):
    victor._current.spline = 0.30
    assert not victor._in_braking_zone()


def test_priority_two_message_is_deferred_in_braking_zone(victor, spoken):
    victor._current.spline = 0.02
    victor._current.speed = 220.0
    victor._say("Test mensaje diferido", priority=2)
    assert len(victor._deferred_msgs) == 1
    assert spoken == []


def test_priority_one_message_passes_through_braking_zone(victor, spoken):
    victor._current.spline = 0.02
    victor._current.speed = 220.0
    victor._say("Test mensaje crítico", priority=1)
    assert len(spoken) == 1


def test_deferred_messages_drain_on_leaving_braking_zone(victor, spoken):
    victor._current.spline = 0.02
    victor._current.speed = 220.0
    victor._say("Test mensaje diferido", priority=2)
    # el main loop drena _deferred_msgs al salir de zona
    victor._current.spline = 0.30
    with victor._deferred_lock:
        msgs = list(victor._deferred_msgs)
        victor._deferred_msgs.clear()
    for pri, txt, *_ in msgs:
        victor._tts_pq.put((pri, txt))
    assert any("diferido" in t for t in _texts(spoken)), _texts(spoken)


# ─── 4. Gap con tendencia ────────────────────────────────────────────────────

def test_gap_trend_closing(victor):
    victor._gap_snapshots[5] = collections.deque([4.0, 3.2], maxlen=5)
    trend = victor._gap_trend(5, 2.5)  # last 3.2 -> delta -0.7
    assert "cerrando" in trend


def test_gap_trend_opening(victor):
    victor._gap_snapshots[7] = collections.deque([1.2, 1.8], maxlen=5)
    trend = victor._gap_trend(7, 2.4)  # last 1.8 -> alejando
    assert "alejando" in trend


def test_gap_trend_empty_without_history(victor):
    assert victor._gap_trend(99, 3.0) == ""


def test_gap_desc_includes_trend(victor):
    t = _rival_telemetry(victor)
    victor._gap_snapshots[5] = collections.deque([4.0, 3.2, 2.5], maxlen=5)
    desc = victor._gap_desc(t)
    assert "cerrando" in desc or "alejando" in desc, desc


# ─── 5. Mid-lap projection ───────────────────────────────────────────────────

def test_projection_fires_at_half_distance(victor, spoken):
    victor._zone_best = [4.5] * victor._N_ZONES
    victor._zone_cur_lap = [4.7] * 10 + [None] * 10  # 10 zonas completadas (50 %)
    victor._current.best_lap_ms = 90000.0
    victor._current.session = 2
    victor._project_lap_time()
    assert any("proyectada" in t.lower() for t in _texts(spoken)), _texts(spoken)


def test_projection_detects_personal_best(victor, spoken):
    victor._zone_best = [4.5] * victor._N_ZONES
    victor._zone_cur_lap = [4.2] * 10 + [None] * 10  # 42 + 45 = 87 s vs best 90 s
    victor._current.best_lap_ms = 90000.0
    victor._current.session = 2
    victor._project_lap_time()
    assert any("récord" in t.lower() for t in _texts(spoken)), _texts(spoken)


# ─── 6. Controller mode en el system prompt ──────────────────────────────────

def test_controller_mode_enabled(eng):
    assert eng.CONTROLLER_MODE is True


@pytest.mark.parametrize("needle", ["MANDO", "suavidad", "stick"])
def test_system_prompt_mentions_controller_coaching(eng, needle):
    assert needle.lower() in eng._SYSTEM.lower()


# ─── 7. Crash template rotation ──────────────────────────────────────────────

def test_crash_templates_rotate(victor, spoken):
    victor._current.spline = 0.50  # fuera de braking zone
    first_words = []
    for i in range(4):
        victor._alert_times.pop("crash_reaction", None)
        victor._prev_dmg_max = 0.0
        victor._current.dmg_front = 20.0 + i  # spike > 15 cada vez
        victor._check_proactive_alerts()
        if spoken:
            first_words.append(spoken[-1][1][:20])
        spoken.clear()
    assert len(set(first_words)) >= 2, f"esperaba ≥2 variantes en 4 crashes: {first_words}"
