"""Pipeline SHM de Victor SIN Assetto Corsa. Migrado a pytest 2026-08-28.

  native/shm_simulator.exe  (crea páginas acpmf_* con valores conocidos, offsets
  del C# de CrewChief)  ->  native/ac_shm_reader.exe  ->  ac_telemetry_{fast,slow}.json

Valida campo a campo que el lector interpreta el layout correctamente y que emite
todo lo que engineer.py necesita (track_len, status, oponentes, etc.).

Marcado `slow`: lanza wine/Proton. Se salta solo si wine o los .exe no están, o
si AC está corriendo. `pytest -m "not slow"` lo excluye.
"""
from __future__ import annotations

import json
import subprocess
import time

import pytest

from _wine_support import NATIVE_DIR, WINE_BIN, ac_is_running, requires_wine, wine_env

pytestmark = [pytest.mark.slow, requires_wine]

FAST = NATIVE_DIR / "ac_telemetry_fast.json"
SLOW = NATIVE_DIR / "ac_telemetry_slow.json"


@pytest.fixture(scope="module")
def telemetry():
    """Lanza simulador + lector bajo wine una sola vez y devuelve
    {"fast": dict, "slow": dict} con lo que escribió el lector."""
    if ac_is_running():
        pytest.skip("Assetto Corsa está corriendo — cerralo para usar el simulador")

    subprocess.run(["pkill", "-f", "shm_simulator.exe"], capture_output=True)
    subprocess.run(["pkill", "-f", "ac_shm_reader.exe"], capture_output=True)
    time.sleep(1)
    for f in (FAST, SLOW):
        f.unlink(missing_ok=True)

    env = wine_env()
    sim = subprocess.Popen([str(WINE_BIN), str(NATIVE_DIR / "shm_simulator.exe")], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(6)  # que wineserver arranque y las páginas existan
    rdr = subprocess.Popen([str(WINE_BIN), str(NATIVE_DIR / "ac_shm_reader.exe")], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if FAST.exists() and SLOW.exists():
                break
            time.sleep(0.5)
        else:
            rdr.terminate()
            log = (rdr.stdout.read() or b"").decode(errors="replace")[-2000:]
            pytest.fail(f"el lector no escribió los JSON en 30 s. Log:\n{log}")
        time.sleep(1.5)  # un par de ciclos más

        yield {"fast": json.loads(FAST.read_text()), "slow": json.loads(SLOW.read_text())}
    finally:
        sim.terminate()
        rdr.terminate()
        subprocess.run(["pkill", "-f", "shm_simulator.exe"], capture_output=True)
        subprocess.run(["pkill", "-f", "ac_shm_reader.exe"], capture_output=True)
        FAST.unlink(missing_ok=True)
        SLOW.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def fast(telemetry):
    return telemetry["fast"]


@pytest.fixture(scope="module")
def slow(telemetry):
    return telemetry["slow"]


# ─── fast: física / gráficos ─────────────────────────────────────────────────

FAST_FIELDS = [
    ("speed", 123.4, 0.11),
    ("rpm", 5500, None),
    ("gear", 4, None),
    ("fuel", 33.5, 0.01),
    ("lap", 5, None),
    ("position", 3, None),
    ("lap_time_ms", 61234, None),
    ("best_lap_ms", 94500, None),
    ("last_lap_ms", 95321, None),
    ("spline", 0.421337, 1e-5),
    ("tyre_compound", "Soft (S)", None),
    ("tyre_fl", 82.1, 0.05),
    ("tyre_wear_fr", 93.2, 0.05),
    ("wx", 100.5, 0.01),
    ("wz", -200.25, 0.01),
    ("vx", 1.5, 0.01),
    ("vz", 34.2, 0.01),
    ("dmg_front", 10.0, 0.05),
    ("cars", 3, None),
    # campos que engineer.py NECESITA
    ("track_len", 4318.0, 0.5),
    ("status", 2, None),
    ("sector_index", 1, None),
    ("last_sector_ms", 30500, None),
    ("in_pitlane", False, None),
]


@pytest.mark.parametrize("key, expected, tol", FAST_FIELDS, ids=[f[0] for f in FAST_FIELDS])
def test_fast_field(fast, key, expected, tol):
    got = fast.get(key)
    if tol is None:
        assert got == expected
    else:
        assert got is not None and abs(got - expected) <= tol, f"{key}: {got!r} != {expected!r} (±{tol})"


# ─── fast: oponentes vía crewchief SHM ───────────────────────────────────────

@pytest.fixture(scope="module")
def opponents(fast):
    return fast.get("cars_data", [])


def _by_name(cars, name):
    return next((c for c in cars if c.get("driver_name") == name), None)


def test_two_opponents(opponents):
    assert len(opponents) == 2


def test_senna_present(opponents):
    assert _by_name(opponents, "Ayrton Senna") is not None


def test_prost_present(opponents):
    assert _by_name(opponents, "Alain Prost") is not None


@pytest.mark.parametrize("key, expected, tol", [
    ("race_pos", 2, None),
    ("spline", 0.4230, 1e-4),
    ("speed", 124.2, 0.2),
    ("lap", 5, None),
    ("wx", 104.5, 0.01),
    ("wz", -198.0, 0.01),
    ("in_pit", False, None),
])
def test_senna_field(opponents, key, expected, tol):
    senna = _by_name(opponents, "Ayrton Senna")
    assert senna is not None
    got = senna.get(key)
    if tol is None:
        assert got == expected
    else:
        assert abs(got - expected) <= tol, f"Senna {key}: {got!r} != {expected!r} (±{tol})"


def test_prost_race_pos(opponents):
    assert _by_name(opponents, "Alain Prost").get("race_pos") == 1


# ─── slow: static / session ─────────────────────────────────────────────────

@pytest.mark.parametrize("key, expected", [
    ("track", "ks_test_track"),
    ("track_layout", "layout_a"),
    ("car", "test_car_gt3"),
    ("session", 2),
    ("time_left", 1800),
])
def test_slow_field(slow, key, expected):
    assert slow.get(key) == expected


# ─── slow: all_drivers (nombres por posición) ────────────────────────────────

@pytest.fixture(scope="module")
def drivers_by_pos(slow):
    drivers = slow.get("all_drivers", {})
    return {v.get("pos"): v.get("name") for v in drivers.values()}, drivers


def test_three_drivers(drivers_by_pos):
    _by_pos, drivers = drivers_by_pos
    assert len(drivers) == 3


@pytest.mark.parametrize("pos, name", [(1, "Alain Prost"), (2, "Ayrton Senna"), (3, "Julian Rincon")])
def test_driver_at_position(drivers_by_pos, pos, name):
    by_pos, _drivers = drivers_by_pos
    assert by_pos.get(pos) == name
