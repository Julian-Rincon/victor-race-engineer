"""v6 — mejoras post-auditoría (2026-07-03), migrado a pytest 2026-08-28.

1. fuel_laps calculado desde el histórico de vueltas (la fuente SHM no lo trae)
2. Splits acumulados desde transiciones de sector (SHM trae sector_index +
   last_sector_ms, no arrays de splits)
3. Gating por status del juego (LIVE=2): sin alertas en menú/replay/pausa

No necesita AC ni audio real. Antes era un script standalone con check()/sys.exit().
"""
from __future__ import annotations

import pytest


# ─── 1. fuel_laps desde histórico ────────────────────────────────────────────

@pytest.fixture
def fuel(eng):
    eng._lap_history.clear()
    yield eng
    eng._lap_history.clear()


def test_fuel_laps_without_history_is_zero(fuel):
    assert fuel._calc_fuel_laps(30.0) == 0.0


def test_fuel_laps_averages_last_three_laps(fuel):
    for lap, used in ((1, 2.0), (2, 2.2), (3, 2.1)):
        fuel._lap_history.append(fuel.LapRecord(lap=lap, time_ms=95000, fuel_used=used, pos=3))
    # 10.5 L / 2.1 L·vuelta ≈ 5.0 vueltas
    assert fuel._calc_fuel_laps(10.5) == pytest.approx(5.0, abs=0.01)


def test_fuel_laps_with_zero_fuel_is_zero(fuel):
    for lap, used in ((1, 2.0), (2, 2.2), (3, 2.1)):
        fuel._lap_history.append(fuel.LapRecord(lap=lap, time_ms=95000, fuel_used=used, pos=3))
    assert fuel._calc_fuel_laps(0.0) == 0.0


def test_fuel_laps_ignores_pit_lap_outlier(fuel):
    for lap, used in ((1, 2.0), (2, 2.2), (3, 2.1)):
        fuel._lap_history.append(fuel.LapRecord(lap=lap, time_ms=95000, fuel_used=used, pos=3))
    fuel._lap_history.append(fuel.LapRecord(lap=4, time_ms=180000, fuel_used=0.01, pos=3))
    assert fuel._calc_fuel_laps(10.5) == pytest.approx(5.0, abs=0.01), "la vuelta de pits (0.01 L) no debe bajar el promedio"


# ─── 2. Splits por transiciones de sector ────────────────────────────────────

@pytest.fixture
def sectors(eng):
    eng._reset_sector_state()
    return eng


def _run_lap(eng, s1, s2, s3, lap):
    """Vuelta completa: sector 0→1→2→(vuelta+1) 0."""
    eng._ingest_sector(1, s1, lap)
    eng._ingest_sector(2, s2, lap)
    eng._ingest_sector(0, s3, lap + 1)


def test_last_splits_of_completed_lap(sectors):
    sectors._ingest_sector(0, 0, 5)  # arranque de vuelta
    _run_lap(sectors, 30500, 31200, 29900, lap=5)
    assert sectors._last_lap_splits() == [30500, 31200, 29900]


def test_best_splits_seeded_from_first_full_lap(sectors):
    sectors._ingest_sector(0, 0, 5)
    _run_lap(sectors, 30500, 31200, 29900, lap=5)
    assert sectors._best_sector_splits() == [30500, 31200, 29900]


def test_best_splits_are_elementwise_ideal_lap(sectors):
    sectors._ingest_sector(0, 0, 5)
    _run_lap(sectors, 30500, 31200, 29900, lap=5)
    _run_lap(sectors, 30800, 30900, 30100, lap=6)  # mejora S2, empeora S1
    assert sectors._last_lap_splits() == [30800, 30900, 30100]
    assert sectors._best_sector_splits() == [30500, 30900, 29900]


def test_repeated_sector_packets_do_not_duplicate(sectors):
    sectors._ingest_sector(0, 0, 5)
    _run_lap(sectors, 30500, 31200, 29900, lap=5)
    _run_lap(sectors, 30800, 30900, 30100, lap=6)
    sectors._ingest_sector(1, 31000, 7)
    sectors._ingest_sector(1, 31000, 7)  # duplicado
    sectors._ingest_sector(2, 30000, 7)
    sectors._ingest_sector(0, 30000, 8)
    assert sectors._last_lap_splits() == [31000, 30000, 30000]


def test_incomplete_lap_does_not_contaminate_best(sectors):
    sectors._ingest_sector(0, 0, 1)
    sectors._ingest_sector(2, 45000, 1)  # salto 0→2 (telemetría perdida)
    sectors._ingest_sector(0, 31000, 2)
    assert sectors._last_lap_splits() == []


# ─── 3. Gating por status del juego ──────────────────────────────────────────

def test_is_live_true_for_status_live(eng):
    t = eng.Telemetry()
    t.status = 2
    assert eng._is_live(t) is True


@pytest.mark.parametrize("status, name", [(0, "OFF"), (1, "REPLAY"), (3, "PAUSE")])
def test_is_live_false_for_non_live_status(eng, status, name):
    t = eng.Telemetry()
    t.status = status
    assert eng._is_live(t) is False


def test_is_live_default_dataclass_is_live(eng):
    """Compat con el bridge viejo que no manda status."""
    assert eng._is_live(eng.Telemetry()) is True
