"""E2E: native/shm_simulator.exe -> native/ac_shm_reader.exe (wine real) ->
engineer._file_worker. Migrado a pytest 2026-08-28.

Valida que el ingeniero ingiere los archivos REALES escritos por el lector:
conexión, track, track_len (spotter habilitado), status, oponentes con nombres,
y el mapa de pilotos por posición. Sin audio (el TTS se captura).

Marcado `slow`: lanza wine/Proton. Se salta si wine/los .exe no están o si AC
está corriendo. `pytest -m "not slow"` lo excluye.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import replace

import pytest

from _wine_support import NATIVE_DIR, WINE_BIN, ac_is_running, requires_wine, wine_env

pytestmark = [pytest.mark.slow, requires_wine]

FAST = NATIVE_DIR / "ac_telemetry_fast.json"
SLOW = NATIVE_DIR / "ac_telemetry_slow.json"


@pytest.fixture(scope="module")
def ingested(eng):
    """Corre el pipeline completo bajo wine y arranca eng._file_worker hasta que
    el ingeniero reporta conexión. Devuelve un snapshot inmutable de su estado:
    (telemetry, connected, drivers)."""
    if ac_is_running():
        pytest.skip("Assetto Corsa está corriendo — cerralo primero")

    orig_put = eng._tts_pq.put
    eng._tts_pq.put = lambda item: None  # mudo durante la ingesta

    subprocess.run(["pkill", "-f", "shm_simulator.exe"], capture_output=True)
    subprocess.run(["pkill", "-f", "ac_shm_reader.exe"], capture_output=True)
    time.sleep(1)
    FAST.unlink(missing_ok=True)
    SLOW.unlink(missing_ok=True)

    env = wine_env()
    sim = subprocess.Popen([str(WINE_BIN), str(NATIVE_DIR / "shm_simulator.exe")], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    rdr = subprocess.Popen([str(WINE_BIN), str(NATIVE_DIR / "ac_shm_reader.exe")], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not (FAST.exists() and SLOW.exists()):
            time.sleep(0.5)
        if not (FAST.exists() and SLOW.exists()):
            pytest.fail("TIMEOUT esperando los JSON del lector")

        threading.Thread(target=eng._file_worker, daemon=True, name="telem-test").start()
        deadline = time.time() + 10
        while time.time() < deadline and not eng._connected:
            time.sleep(0.2)
        time.sleep(2.5)  # dejar que procese el slow packet + varios fast

        with eng._lock:
            snap_current = replace(eng._current)
        with eng._drivers_lock:
            snap_drivers = dict(eng._drivers)
        yield snap_current, eng._connected, snap_drivers
    finally:
        sim.terminate()
        rdr.terminate()
        subprocess.run(["pkill", "-f", "shm_simulator.exe"], capture_output=True)
        subprocess.run(["pkill", "-f", "ac_shm_reader.exe"], capture_output=True)
        FAST.unlink(missing_ok=True)
        SLOW.unlink(missing_ok=True)
        eng._connected = False
        eng._tts_pq.put = orig_put


@pytest.fixture(scope="module")
def telem(ingested):
    return ingested[0]


def test_engineer_reports_connected(ingested):
    assert ingested[1] is True


def test_track_name_ingested(telem):
    assert telem.track == "ks_test_track"


def test_car_model_ingested(telem):
    assert telem.car == "test_car_gt3"


def test_session_type_is_race(telem):
    assert telem.session == 2


def test_track_len_ingested_enables_spotter(telem):
    assert abs(telem.track_len - 4318.0) < 1


def test_status_is_live(telem):
    assert telem.status == 2


def test_fuel_ingested(telem):
    assert abs(telem.fuel - 33.5) < 0.01


def test_position_ingested(telem):
    assert telem.position == 3


def test_two_rivals_in_cars_data(telem):
    assert len(telem.cars_data) == 2


def test_sector_index_ingested(telem):
    assert telem.sector_index == 1


def test_rival_names_ingested(telem):
    names = {c.get("driver_name") for c in telem.cars_data}
    assert names == {"Ayrton Senna", "Alain Prost"}


def test_driver_map_position_one_is_prost(ingested):
    _telem, _connected, drivers = ingested
    assert drivers.get(1, {}).get("name") == "Alain Prost"


def test_driver_map_position_two_is_senna(ingested):
    _telem, _connected, drivers = ingested
    assert drivers.get(2, {}).get("name") == "Ayrton Senna"
