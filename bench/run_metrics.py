#!/usr/bin/env python3
"""Métricas reproducibles de Victor que NO necesitan Assetto Corsa abierto.

Todo corre en local, sin red externa y sin llamar a ninguna API real:

  telemetry_parse   lectura + decodificación del JSON rápido (_read_json_file),
                    con un paquete con las mismas claves que escribe
                    native/ac_shm_reader.c y una parrilla de 24 coches.
  spotter           coste por frame de _run_spotter y retardo de anuncio
                    (tiempo simulado a 10 Hz desde que empieza el solape).
  alerts            coste por llamada de _check_fast_alerts y
                    _check_proactive_alerts.
  pit_strategy      coste de _calc_fuel_laps / _local_reply("pit") y acierto
                    de la conclusión "llega / no llega" en los mismos 3 casos
                    de combustible del benchmark de LLM (bench/llm).
  llm_chain         VictorBrain.complete() contra un servidor HTTP local que
                    imita la API OpenAI-compatible (respuesta inmediata):
                    mide el coste propio de Victor (HTTP loopback, JSON,
                    failover), no la latencia de ningún proveedor real.
  voice_turn        engineer._handle_voice() de punta a punta con STT simulado
                    (devuelve el texto al instante) y el LLM simulado de arriba:
                    wake word -> enrutado -> contexto -> LLM -> _say().
  file_bridge       latencia real del puente por archivo (lo que en carrera
                    escribe ac_shm_reader.exe): os.replace() del JSON ->
                    valor visible en engineer._current, con el _file_worker
                    de producción corriendo en un hilo.
  tests (--tests)   pytest -q completo; con pytest-cov instalado añade la
                    cobertura de src/victor.

Uso (desde la raíz del repo):
  python3 bench/run_metrics.py                  # -> bench/results/<AAAA-MM-DD>/metrics.json
  python3 bench/run_metrics.py --tests          # + pytest -q (+ cobertura si hay pytest-cov)
  python3 bench/run_metrics.py --quick          # menos iteraciones, para probar
  python3 bench/run_metrics.py --out otra/ruta.json

Los tiempos dependen de la máquina: el JSON guarda CPU, Python y fecha.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "victor"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "bench" / "llm"))

import victor_brain as vb  # noqa: E402

TRACK_LEN = 5793.0          # Monza
N_CARS = 24                 # parrilla típica de GT3


# ─── utilidades ───────────────────────────────────────────────────────────────
def import_engineer():
    """Igual que la fixture `eng` de tests/conftest.py: stubs de audio solo
    durante el import."""
    keys = ("pyaudio", "piper", "piper.voice")
    saved = {k: sys.modules.get(k) for k in keys}
    piper_stub = mock.MagicMock()
    piper_stub.PiperVoice.load.return_value = mock.MagicMock()
    sys.modules["pyaudio"] = mock.MagicMock()
    sys.modules["piper"] = piper_stub
    sys.modules["piper.voice"] = piper_stub
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import engineer  # noqa: PLC0415
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return engineer


def stats_us(samples_ns: list[int]) -> dict:
    """Resumen en microsegundos (percentiles por interpolación lineal)."""
    us = [s / 1000.0 for s in samples_ns]
    q = statistics.quantiles(us, n=100, method="inclusive")
    return {
        "n": len(us),
        "p50_us": round(q[49], 2),
        "p95_us": round(q[94], 2),
        "p99_us": round(q[98], 2),
        "max_us": round(max(us), 2),
        "mean_us": round(statistics.fmean(us), 2),
    }


def stats_ms(samples_s: list[float]) -> dict:
    ms = [s * 1000.0 for s in samples_s]
    q = statistics.quantiles(ms, n=100, method="inclusive")
    return {
        "n": len(ms),
        "p50_ms": round(q[49], 3),
        "p95_ms": round(q[94], 3),
        "p99_ms": round(q[98], 3),
        "max_ms": round(max(ms), 3),
        "mean_ms": round(statistics.fmean(ms), 3),
    }


@contextlib.contextmanager
def quiet():
    """Los print() de producción van a /dev/null (en carrera van a un log)."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


@contextlib.contextmanager
def fake_clock(eng, start: float = 1_000_000.0):
    """Reloj controlado para time.time() (debounce/cooldowns), como en tests/test_spotter.py.
    perf_counter_ns (lo que se mide) no se toca."""
    now = [start]
    real = eng.time.time
    eng.time.time = lambda: now[0]
    try:
        yield now
    finally:
        eng.time.time = real


def machine_info() -> dict:
    cpu = "desconocido"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "cpu": cpu,
        "cpu_threads": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
    }


# ─── telemetría sintética (mismas claves que native/ac_shm_reader.c) ──────────
def rival(idx: int, spline: float, wx: float, wz: float, lap: int = 18) -> dict:
    return {
        "idx": idx, "race_pos": idx + 1, "lap": lap, "spline": round(spline % 1.0, 6),
        "speed": 248.3, "last_lap": 107_482, "best_lap": 107_204,
        "wx": round(wx, 2), "wz": round(wz, 2), "driver_name": f"Piloto Rival {idx:02d}",
        "real_pos": idx + 1, "in_pitlane": False, "in_pit": False,
    }


def grid(our_spline: float = 0.5, near: list[tuple[float, float]] | None = None) -> list[dict]:
    """23 rivales repartidos por la pista (lejos) + los `near` (long_m, lat_m) cerca."""
    cars = []
    near = near or []
    for i, (long_m, lat_m) in enumerate(near, start=1):
        cars.append(rival(i, our_spline + long_m / TRACK_LEN, -lat_m, long_m))
    for i in range(len(near) + 1, N_CARS):
        frac = i / N_CARS
        cars.append(rival(i, our_spline + frac, 800.0 * frac, -1200.0 * frac))
    return cars


def fast_packet(lap_time_ms: int = 63_211, cars_data: list[dict] | None = None) -> dict:
    return {
        "t": "fast", "status": 2, "speed": 248.3, "rpm": 7450, "gear": 5, "fuel": 26.0,
        "fuel_laps": 0.0, "lap": 18, "lap_time_ms": lap_time_ms, "best_lap_ms": 107_204,
        "last_lap_ms": 107_482, "position": 2, "cars": N_CARS, "spline": 0.5,
        "tyre_compound": "M", "tyre_fl": 92.1, "tyre_fr": 94.3, "tyre_rl": 88.0, "tyre_rr": 91.2,
        "tyre_wear_fl": 96.1, "tyre_wear_fr": 95.4, "tyre_wear_rl": 97.0, "tyre_wear_rr": 96.6,
        "last_splits": [], "best_splits": [], "track_len": TRACK_LEN, "sector_index": 1,
        "last_sector_ms": 35_120, "in_pitlane": False, "wx": 0.0, "wz": 0.0,
        "vx": 0.0, "vz": 68.97, "dmg_front": 0.0, "dmg_rear": 0.0, "dmg_left": 0.0,
        "dmg_right": 0.0, "dmg_centre": 0.0, "tyres_out": 0, "brake": 0.0,
        "slip_fl": 0.02, "slip_fr": 0.02, "slip_rl": 0.03, "slip_rr": 0.03,
        "cars_data": grid() if cars_data is None else cars_data,
    }


def make_telemetry(eng, **over):
    t = eng.Telemetry()
    p = fast_packet()
    for k in ("status", "speed", "rpm", "gear", "fuel", "lap", "lap_time_ms", "best_lap_ms",
              "last_lap_ms", "position", "cars", "spline", "tyre_compound", "tyre_fl", "tyre_fr",
              "tyre_rl", "tyre_rr", "tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr",
              "track_len", "wx", "wz", "vx", "vz", "cars_data", "brake", "tyres_out",
              "slip_fl", "slip_fr", "slip_rl", "slip_rr"):
        setattr(t, k, p[k])
    t.session = 2
    t.track = "ks_monza"
    t.car = "ks_ferrari_488_gt3"
    t.fuel_laps = 11.3
    for k, v in over.items():
        setattr(t, k, v)
    return t


def seed_laps(eng, fuel_per_lap: float = 2.3, n: int = 5) -> None:
    eng._lap_history.clear()
    for i in range(n):
        eng._lap_history.append(eng.LapRecord(lap=13 + i, time_ms=107_300 + 40 * i,
                                              fuel_used=fuel_per_lap, pos=2))


# ─── 1. parser de telemetría ──────────────────────────────────────────────────
def bench_parse(eng, iters: int) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as d:
        fast = Path(d) / "ac_telemetry_fast.json"
        fast.write_text(json.dumps(fast_packet()))
        # paquete lento real de una carrera por vueltas: time_left = inf crudo (printf de C)
        slow = Path(d) / "ac_telemetry_slow.json"
        slow.write_text('{"t":"slow","track":"ks_monza","track_layout":"","car":"ks_ferrari_488_gt3",'
                        '"session":2,"time_left":inf,"all_drivers":{}}')
        for name, path in (("fast_packet", fast), ("slow_packet_nonfinite", slow)):
            assert eng._read_json_file(str(path)) is not None, name
            samples = []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                eng._read_json_file(str(path))
                samples.append(time.perf_counter_ns() - t0)
            s = stats_us(samples)
            size = path.stat().st_size
            total_s = sum(samples) / 1e9
            s.update({
                "bytes": size,
                "packets_per_s": round(iters / total_s),
                "mb_per_s": round(size * iters / total_s / 1e6, 1),
            })
            out[name] = s
    out["note"] = ("Lectura de disco + json.loads de engineer._read_json_file; el paquete rápido "
                   f"lleva las claves de native/ac_shm_reader.c y {N_CARS - 1} rivales en cars_data. "
                   "El lector real escribe a ~10 Hz.")
    return out


# ─── 2. spotter ────────────────────────────────────────────────────────────────
def bench_spotter(eng, frames: int) -> dict:
    eng._tts_pq.put = lambda item: None
    t_clear = make_telemetry(eng)
    t_right = make_telemetry(eng, cars_data=grid(near=[(1.0, 2.2)]))
    t_left = make_telemetry(eng, cars_data=grid(near=[(-0.5, -2.3)]))
    t_three = make_telemetry(eng, cars_data=grid(near=[(0.5, 2.3), (-0.8, -2.4)]))
    phases = [t_clear, t_right, t_clear, t_left, t_three]
    samples = []
    eng._reset_spotter()
    with fake_clock(eng) as now, quiet():
        for i in range(frames):
            now[0] += 0.1                                  # 10 Hz, como el lector SHM
            t = phases[(i // 50) % len(phases)]
            t0 = time.perf_counter_ns()
            eng._run_spotter(t)
            samples.append(time.perf_counter_ns() - t0)

        # Retardo de anuncio en tiempo simulado (frames a 10 Hz)
        said: list[tuple[int, str]] = []
        eng._tts_pq.put = lambda item: said.append(item)
        eng._reset_spotter()
        for _ in range(20):
            now[0] += 0.1
            eng._run_spotter(t_clear)
        said.clear()
        start = now[0]
        alongside_ms = None
        for _ in range(50):
            now[0] += 0.1
            eng._run_spotter(t_right)
            if said:
                alongside_ms = round((now[0] - start) * 1000)
                alongside_text = said[0][1]
                break
        said.clear()
        start = now[0]
        clear_ms = None
        for _ in range(50):
            now[0] += 0.1
            eng._run_spotter(t_clear)
            if said:
                clear_ms = round((now[0] - start) * 1000)
                clear_text = said[0][1]
                break
    eng._reset_spotter()
    eng._tts_pq.put = lambda item: None
    return {
        "per_frame": stats_us(samples),
        "announce_delay_sim": {
            "frame_rate_hz": 10,
            "overlap_to_call_ms": alongside_ms,
            "call_text": alongside_text if alongside_ms is not None else None,
            "clear_to_call_ms": clear_ms,
            "clear_text": clear_text if clear_ms is not None else None,
            "config": {"SPOTTER_CONFIRM_S": getattr(eng, "SPOTTER_CONFIRM_S", None),
                       "SPOTTER_CLEAR_S": getattr(eng, "SPOTTER_CLEAR_S", None)},
        },
        "note": (f"_run_spotter con {N_CARS - 1} rivales, ciclando despejado / derecha / izquierda / 3-wide. "
                 "announce_delay_sim es tiempo SIMULADO a 10 Hz (debounce de diseño), no tiempo de CPU."),
    }


# ─── 3. alertas ────────────────────────────────────────────────────────────────
def bench_alerts(eng, iters: int) -> dict:
    eng._tts_pq.put = lambda item: None
    normal = make_telemetry(eng)
    lockup = make_telemetry(eng, brake=0.95, slip_fl=0.9)
    offtrack = make_telemetry(eng, tyres_out=3)
    seq = [normal] * 8 + [lockup, offtrack]
    fast_s = []
    pro_s = []
    seed_laps(eng)
    prev_connected = eng._connected
    with fake_clock(eng) as now, quiet():
        for i in range(iters):
            now[0] += 0.1
            t = seq[i % len(seq)]
            t0 = time.perf_counter_ns()
            eng._check_fast_alerts(t, "SHM")
            fast_s.append(time.perf_counter_ns() - t0)
        eng._connected = True
        with eng._lock:
            eng._current = make_telemetry(eng, fuel_laps=2.9, tyre_fr=108.0)
        for _ in range(iters):
            now[0] += 6.0                                  # el daemon lo llama cada ~6 s
            t0 = time.perf_counter_ns()
            eng._check_proactive_alerts()
            pro_s.append(time.perf_counter_ns() - t0)
    eng._connected = prev_connected
    with eng._lock:
        eng._current = eng.Telemetry()
    return {
        "fast_alerts_per_call": stats_us(fast_s),
        "proactive_alerts_per_call": stats_us(pro_s),
        "note": ("_check_fast_alerts con 1 de cada 10 frames con bloqueo de freno y 1 con salida de pista; "
                 "_check_proactive_alerts con combustible bajo y neumático caliente (dispara avisos, "
                 "sujeto a cooldown). TTS reemplazado por un no-op."),
    }


# ─── 4. estrategia de pits / combustible ──────────────────────────────────────
def bench_pit(eng, iters: int) -> dict:
    from bench_llm import FUEL_CASES  # mismos casos que el benchmark de LLM

    cases = []
    for cid, c in FUEL_CASES.items():
        seed_laps(eng, fuel_per_lap=c["consumo"], n=3)
        laps = eng._calc_fuel_laps(c["litros"])
        llega = laps >= c["vueltas"]
        cases.append({"case": cid, "litros": c["litros"], "l_por_vuelta": c["consumo"],
                      "vueltas_restantes": c["vueltas"], "vueltas_de_combustible": round(laps, 2),
                      "conclusion": "llega" if llega else "no llega",
                      "correcta": llega == c["llega"]})

    seed_laps(eng)
    t = make_telemetry(eng)
    calc, reply = [], []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        eng._calc_fuel_laps(26.0)
        calc.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        eng._local_reply("pit", t)
        reply.append(time.perf_counter_ns() - t0)
    return {
        "calc_fuel_laps": stats_us(calc),
        "local_reply_pit": stats_us(reply),
        "fuel_cases": cases,
        "fuel_cases_correct": f"{sum(c['correcta'] for c in cases)}/{len(cases)}",
        "note": ("Camino determinista (sin LLM). Los casos son los 3 de bench/llm; la conclusión sale de "
                 "_calc_fuel_laps(litros) >= vueltas restantes con el consumo de las últimas vueltas."),
    }


# ─── 5. cadena de LLM con proveedores simulados ───────────────────────────────
MOCK_REPLY = "Copiado, Julián. Te quedan unas once vueltas de combustible, vas bien."


class _MockProvider(BaseHTTPRequestHandler):
    """POST /<modo>/chat/completions — modo: ok | e500 | e429."""
    protocol_version = "HTTP/1.1"
    # Sin esto, cabeceras y cuerpo salen en dos send() y Nagle + ACK retardado
    # añaden ~40 ms por respuesta en loopback: artefacto del mock, no de Victor.
    disable_nagle_algorithm = True

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        mode = self.path.strip("/").split("/")[0]
        headers = {"Content-Type": "application/json"}
        if mode == "ok":
            json.loads(body)
            status = 200
            payload = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": MOCK_REPLY}}],
                                  "usage": {"prompt_tokens": 1000, "completion_tokens": 20}}).encode()
        elif mode == "e429":
            status, payload = 429, b'{"error":{"message":"rate limit"}}'
            headers["retry-after"] = "30"
        else:
            status, payload = 500, b'{"error":{"message":"internal error"}}'
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_mock_server() -> tuple[ThreadingHTTPServer, str, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockProvider)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    refused = f"http://127.0.0.1:{s.getsockname()[1]}"
    s.close()                                     # puerto cerrado -> conexión rechazada
    return srv, base, refused


def make_brain(urls: list[str]) -> vb.VictorBrain:
    keys = {k: "bench-key" for k in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "NVIDIA_API_KEY",
                                      "GEMINI_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY")}
    with mock.patch.multiple(vb.cfg, CUSTOM_LLM_BASE_URL="", LOCAL_LLM_MODEL="", **keys):
        brain = vb.VictorBrain()
    brain.backends = brain.backends[:len(urls)]
    for b, u in zip(brain.backends, urls):
        b.base_url = u
    return brain


def reset_brain(brain: vb.VictorBrain) -> None:
    for b in brain.backends:
        b.fails, b.paused_until, b.pause_reason = 0, 0.0, ""


def bench_llm_chain(iters: int, base: str, refused: str) -> dict:
    ok, e500, e429 = f"{base}/ok", f"{base}/e500", f"{base}/e429"
    scenarios = {
        "first_backend_ok": [ok],
        "failover_after_http_500": [e500, ok],
        "failover_after_429": [e429, ok],
        "failover_after_connection_refused": [refused, ok],
        "all_6_backends_down": [e500] * 6,
    }
    msgs = [{"role": "user", "content": "Piloto pregunta por combustible. Fuel: 26.0L ~11.3 vueltas."}]
    out = {}
    with quiet():
        for name, urls in scenarios.items():
            brain = make_brain(urls)
            samples = []
            for i in range(iters + 5):
                reset_brain(brain)
                t0 = time.perf_counter()
                reply = brain.complete("Eres Victor.", msgs)
                dt = time.perf_counter() - t0
                if i >= 5:                         # 5 de calentamiento (conexión keep-alive)
                    samples.append(dt)
            expected = "" if name == "all_6_backends_down" else MOCK_REPLY
            assert reply == expected, (name, reply)
            out[name] = stats_ms(samples) | {"backend": brain.last_backend}

        # Circuit breaker: tras 3 fallos seguidos el backend caído deja de intentarse
        brain = make_brain([e500, ok])
        tried = []
        for _ in range(5):
            brain.complete("Eres Victor.", msgs)
            tried.append(brain.backends[0].fails)
        out["circuit_breaker"] = {
            "primary_paused_after_calls": next((i + 1 for i, f in enumerate(tried) if f >= 3), None),
            "primary_available_after": brain.backends[0].available(),
        }
    out["note"] = ("VictorBrain.complete() real contra un servidor HTTP local OpenAI-compatible que responde al "
                   "instante: es el sobrecoste de Victor (HTTP loopback + JSON + failover), NO la latencia de "
                   "un proveedor real. El estado del breaker se reinicia en cada iteración.")
    return out


# ─── 6. turno de voz de punta a punta (STT y LLM simulados) ───────────────────
def bench_voice_turn(eng, iters: int, base: str) -> dict:
    ok, e500 = f"{base}/ok", f"{base}/e500"
    said: list[tuple[int, str]] = []
    eng._tts_pq.put = lambda item: said.append(item)
    seed_laps(eng)
    with eng._lock:
        eng._current = make_telemetry(eng)
    orig_brain = eng._brain
    scenarios = {
        "fuel_question_llm_ok": ("Victor, ¿cuánto combustible me queda?", [ok]),
        "free_question_llm_ok": ("Victor, ¿cómo ves la carrera?", [ok]),
        "fuel_question_failover": ("Victor, ¿cuánto combustible me queda?", [e500, ok]),
        "llm_down_deterministic_fallback": ("Victor, ¿cuánto combustible me queda?", [e500] * 6),
        "local_command_no_llm": ("Victor, modo gap", [ok]),
    }
    out = {}
    try:
        with quiet():
            for name, (text, urls) in scenarios.items():
                eng._brain = make_brain(urls)
                samples = []
                last = ""
                with mock.patch.object(eng, "_transcribe", return_value=text):
                    for i in range(iters + 5):
                        reset_brain(eng._brain)
                        eng._convo.clear()
                        said.clear()
                        t0 = time.perf_counter()
                        eng._handle_voice("/nonexistent/victor-bench.wav")
                        dt = time.perf_counter() - t0
                        if i >= 5:
                            samples.append(dt)
                        last = said[-1][1] if said else ""
                assert last, f"{name}: Victor no dijo nada"
                out[name] = stats_ms(samples) | {"transcript": text, "reply": last}
    finally:
        eng._brain = orig_brain
        eng._gap_mode = False
        with eng._lock:
            eng._current = eng.Telemetry()
        eng._tts_pq.put = lambda item: None
    out["note"] = ("engineer._handle_voice() real, de la transcripción a la respuesta encolada en _say(). "
                   "STT simulado (devuelve el texto al instante) y LLM simulado (servidor local inmediato): "
                   "excluye grabación, VAD, STT real, red, inferencia del LLM y síntesis Piper.")
    return out


# ─── 7. puente por archivo: os.replace -> engineer._current ───────────────────
def bench_file_bridge(eng, packets: int) -> dict:
    """Corre el _file_worker de producción en un hilo contra un directorio
    temporal y escribe paquetes a 10 Hz igual que ac_shm_reader.exe
    (archivo .tmp + rename). Debe ir AL FINAL: el hilo no se puede parar."""
    eng._tts_pq.put = lambda item: None
    d = Path(tempfile.mkdtemp(prefix="victor-bridge-"))
    eng.TELEM_FAST_SHM = str(d / "ac_telemetry_fast.json")
    eng.TELEM_SLOW_SHM = str(d / "ac_telemetry_slow.json")
    eng.TELEM_FAST_BRIDGE = str(d / "no-bridge" / "ac_telemetry_fast.json")
    eng.TELEM_SLOW_BRIDGE = str(d / "no-bridge" / "ac_telemetry_slow.json")
    fast = Path(eng.TELEM_FAST_SHM)
    tmp = fast.with_suffix(".json.tmp")
    cars = grid()

    def write(marker: int) -> float:
        tmp.write_text(json.dumps(fast_packet(lap_time_ms=marker, cars_data=cars)))
        os.replace(tmp, fast)
        return time.time()

    write(1)
    lat = []
    missed = 0
    devnull = open(os.devnull, "w")
    with contextlib.redirect_stdout(devnull):
        threading.Thread(target=eng._file_worker, daemon=True, name="bench-file-worker").start()
        time.sleep(0.5)
        next_t = time.monotonic()
        for i in range(packets):
            marker = 100_000 + i
            t_written = write(marker)
            deadline = time.monotonic() + 1.0
            seen = None
            while time.monotonic() < deadline:
                cur = eng._current
                if cur.lap_time_ms == marker:
                    seen = time.time()          # visible en _current (sondeo cada 0,5 ms)
                    break
                time.sleep(0.0005)
            if seen is None:
                missed += 1
            else:
                lat.append(max(0.0, seen - t_written))
            next_t += 0.1                               # 10 Hz
            time.sleep(max(0.0, next_t - time.monotonic()))
    return {
        "write_to_current": stats_ms(lat),
        "packets_written": packets,
        "packets_missed_within_1s": missed,
        "note": ("Latencia del puente JSON-en-disco del CASE_STUDY: desde que os.replace() deja el paquete "
                 "visible hasta que engineer._current lo refleja (sondeado cada 0,5 ms), con el _file_worker real "
                 "(sondeo cada 40 ms, compuerta de 80 ms). Incluye en el mismo hilo el spotter y las alertas "
                 "rápidas de cada paquete. Escritura a 10 Hz como el lector real."),
    }


# ─── 8. pytest (+ cobertura) ───────────────────────────────────────────────────
def run_tests() -> dict:
    out: dict = {}
    try:
        import pytest_cov  # noqa: F401, PLC0415
        has_cov = True
    except ImportError:
        has_cov = False
    with tempfile.TemporaryDirectory() as d:
        cov_json = Path(d) / "cov.json"
        cmd = [sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider"]
        if has_cov:
            cmd += [f"--cov={SRC}", f"--cov-report=json:{cov_json}", "--cov-report="]
        t0 = time.perf_counter()
        env = os.environ | {"COVERAGE_FILE": str(Path(d) / ".coverage")}   # no ensucia la raíz
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        out["duration_s"] = round(time.perf_counter() - t0, 1)
        out["command"] = "python3 -m pytest -q" + (" --cov=src/victor" if has_cov else "")
        out["exit_code"] = r.returncode
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
        out["summary_line"] = tail[0]
        for key in ("passed", "failed", "skipped", "error", "errors", "deselected"):
            m = re.search(rf"(\d+) {key}\b", tail[0])
            if m:
                out[key] = int(m.group(1))
        if has_cov and cov_json.exists():
            cov = json.loads(cov_json.read_text())
            out["coverage"] = {
                "percent_covered": round(cov["totals"]["percent_covered"], 1),
                "covered_lines": cov["totals"]["covered_lines"],
                "num_statements": cov["totals"]["num_statements"],
                "per_file": {Path(f).name: round(v["summary"]["percent_covered"], 1)
                             for f, v in sorted(cov["files"].items())},
            }
        else:
            out["coverage"] = None
            out["coverage_note"] = "pytest-cov no está instalado: pip install pytest-cov"
    return out


# ─── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / time.strftime("%Y-%m-%d") / "metrics.json"))
    ap.add_argument("--quick", action="store_true", help="menos iteraciones (prueba rápida)")
    ap.add_argument("--tests", action="store_true", help="correr también pytest -q (+ cobertura)")
    args = ap.parse_args()

    n = 200 if args.quick else 5000
    n_net = 30 if args.quick else 300
    n_bridge = 30 if args.quick else 300

    eng = import_engineer()
    orig_put = eng._tts_pq.put
    res: dict = {"machine": machine_info(), "quick": args.quick}
    srv, base, refused = start_mock_server()
    try:
        steps = [
            ("telemetry_parse", lambda: bench_parse(eng, n)),
            ("spotter", lambda: bench_spotter(eng, n)),
            ("alerts", lambda: bench_alerts(eng, n)),
            ("pit_strategy", lambda: bench_pit(eng, n)),
            ("llm_chain", lambda: bench_llm_chain(n_net, base, refused)),
            ("voice_turn", lambda: bench_voice_turn(eng, n_net, base)),
            ("file_bridge", lambda: bench_file_bridge(eng, n_bridge)),   # siempre al final
        ]
        for name, fn in steps:
            t0 = time.perf_counter()
            print(f"[metrics] {name}...", flush=True)
            res[name] = fn()
            print(f"[metrics] {name} ok ({time.perf_counter() - t0:.1f}s)", flush=True)
    finally:
        srv.shutdown()
        eng._tts_pq.put = orig_put

    if args.tests:
        print("[metrics] pytest -q ...", flush=True)
        res["tests"] = run_tests()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[metrics] escrito {out}")


if __name__ == "__main__":
    main()
