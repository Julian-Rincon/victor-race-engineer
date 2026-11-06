"""Victor — cerebro multi-backend.

Cadena cloud con failover: Groq -> Cerebras -> NVIDIA NIM -> Gemini, todos
OpenAI-compatibles. Deliberadamente SIN LLM local: Victor corre durante las
sesiones de juego y un modelo local se come ~5 GB de VRAM que la RTX 4060 ya
casi agota con el juego (incidente de freeze 2026-07-03). Si TODO falla,
engineer.py contesta con respuestas deterministas (sin LLM) — nunca silencio.

Robustez (v8, 2026-11):
- Sesión HTTP keep-alive (ahorra el handshake TLS en cada pregunta) con
  User-Agent propio: Groq rechaza el UA por defecto de python-requests.
- Autocuración de modelos: si el proveedor responde "ese modelo no existe /
  fue retirado", se pide /models y se elige uno vivo según
  cfg.MODEL_PREFERENCES. Pasó dos veces: llama-3.3-70b (Groq, 2026-08) y
  gemini-2.0-flash (2026-06) murieron y Victor quedó sin cerebro.
- Errores permanentes (401/402/403: key inválida, sin crédito) apartan el
  backend toda la sesión en vez de reintentarlo cada 5 min; 429 respeta
  Retry-After; fallos transitorios usan el circuit breaker (3 fallos -> 5 min).
- gpt-oss es de razonamiento: se pide reasoning_effort=low y se reserva margen
  de tokens para que llegue a escribir `content`.

STT: Groq whisper-large-v3-turbo -> Deepgram Nova-3 -> faster-whisper local
en CPU (0 VRAM, ~1 s por comando en este PC) para seguir oyendo sin red.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

import victor_config as cfg

_BREAKER_FAILS   = 3
_BREAKER_PAUSE_S = 300.0
_PERMANENT_PAUSE_S = 6 * 3600.0     # "toda la sesión" en la práctica
_USER_AGENT = "victor-race-engineer/8.0 (+https://github.com/Julian-Rincon/victor-race-engineer)"

_session = requests.Session()
_session.headers.update({"User-Agent": _USER_AGENT})


class _PermanentError(Exception):
    """Key inválida / sin crédito — reintentar no sirve."""


class _RateLimited(Exception):
    """429 — el backend ya quedó pausado según Retry-After."""


class _ModelGone(Exception):
    """El modelo configurado no existe o fue retirado."""


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(k in m for k in ("gpt-oss", "qwen3", "qwen-3", "nemotron", "deepseek-r",
                                 "reason", "thinking"))


def pick_model(available: list[str], prefs: list[str]) -> Optional[str]:
    """Primer modelo de `available` que contenga la preferencia más alta.
    Ignora modelos que no son de chat (whisper, tts, embeddings, guard)."""
    skip = ("whisper", "tts", "embed", "guard", "vision", "audio", "image", "playai", "orpheus",
            "safety", "code", "omni", "-vl")
    chat = [m for m in available if not any(x in m.lower() for x in skip)]
    for pref in prefs:
        hits = sorted(m for m in chat if pref in m.lower())
        if hits:
            return hits[0]
    return None


@dataclass
class _Backend:
    name: str
    api_key: str
    base_url: str
    model: str
    fails: int = 0
    paused_until: float = 0.0
    pause_reason: str = ""
    # Margen extra de tokens para modelos de razonamiento (ver docstring).
    reasoning_reserve: int = 0
    healed: bool = False
    kind: str = "openai"            # "openai" | "ollama"

    def available(self) -> bool:
        return bool(self.api_key) and time.time() >= self.paused_until

    def record_success(self) -> None:
        self.fails = 0
        self.paused_until = 0.0
        self.pause_reason = ""

    def record_failure(self) -> None:
        self.fails += 1
        if self.fails >= _BREAKER_FAILS:
            self.paused_until = time.time() + _BREAKER_PAUSE_S
            self.pause_reason = "circuit breaker"

    def pause(self, seconds: float, reason: str) -> None:
        self.paused_until = time.time() + seconds
        self.pause_reason = reason


def _classify_http(r: requests.Response) -> None:
    """Lanza la excepción adecuada para respuestas no-2xx."""
    if r.status_code < 400:
        return
    body = (r.text or "")[:400].lower()
    if r.status_code in (401, 402, 403):
        raise _PermanentError(f"HTTP {r.status_code}: {body[:160]}")
    if r.status_code == 404 or (r.status_code == 400 and "model" in body and any(
            k in body for k in ("not found", "does not exist", "decommission", "deprecat",
                                "no longer", "not supported", "invalid model", "unknown model"))):
        raise _ModelGone(f"HTTP {r.status_code}: {body[:160]}")
    r.raise_for_status()


class VictorBrain:
    """Chat completion con failover en cadena sobre backends OpenAI-compatibles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.backends: list[_Backend] = [
            _Backend("groq",     cfg.GROQ_API_KEY,     "https://api.groq.com/openai/v1",         cfg.GROQ_MODEL,
                     reasoning_reserve=250),
            _Backend("cerebras", cfg.CEREBRAS_API_KEY, "https://api.cerebras.ai/v1",              cfg.CEREBRAS_MODEL,
                     reasoning_reserve=250),
            _Backend("nvidia",   cfg.NVIDIA_API_KEY,   "https://integrate.api.nvidia.com/v1",     cfg.NVIDIA_MODEL),
            _Backend("gemini",   cfg.GEMINI_API_KEY,
                     "https://generativelanguage.googleapis.com/v1beta/openai", cfg.GEMINI_MODEL),
            _Backend("mistral",  cfg.MISTRAL_API_KEY,  "https://api.mistral.ai/v1",               cfg.MISTRAL_MODEL),
            _Backend("openrouter", cfg.OPENROUTER_API_KEY, "https://openrouter.ai/api/v1",        cfg.OPENROUTER_MODEL,
                     reasoning_reserve=250),
        ]
        if cfg.CUSTOM_LLM_BASE_URL and cfg.CUSTOM_LLM_MODEL:
            self.backends.insert(0, _Backend("custom", cfg.CUSTOM_LLM_API_KEY,
                                             cfg.CUSTOM_LLM_BASE_URL, cfg.CUSTOM_LLM_MODEL,
                                             reasoning_reserve=250))
        if cfg.LOCAL_LLM_MODEL:
            # Último de la cadena: solo si toda la nube falla.
            self.backends.append(_Backend("local", "local", cfg.LOCAL_LLM_URL,
                                          cfg.LOCAL_LLM_MODEL, kind="ollama"))
        self.last_backend: Optional[str] = None

    def _headers(self, b: _Backend) -> dict:
        return {"Authorization": f"Bearer {b.api_key}", "Content-Type": "application/json"}

    def list_models(self, b: _Backend, timeout: float = 8.0) -> list[str]:
        r = _session.get(f"{b.base_url}/models", headers=self._headers(b), timeout=timeout)
        _classify_http(r)
        data = r.json().get("data", [])
        return [str(m.get("id", "")).removeprefix("models/") for m in data if m.get("id")]

    def _heal(self, b: _Backend) -> bool:
        """Cambia b.model por uno vivo. Solo una vez por backend y sesión."""
        if b.healed or b.kind != "openai":
            return False
        b.healed = True
        try:
            models = self.list_models(b)
            if b.name == "openrouter":          # solo gratis
                models = [m for m in models if m.endswith(":free")]
            new = pick_model(models, cfg.MODEL_PREFERENCES.get(b.name, []))
        except Exception as e:
            print(f"[Brain:{b.name}] no pude listar modelos: {e}")
            return False
        if b.name == "openrouter" and new and not new.endswith(":free"):
            new = None                  # jamás cambiar a un modelo de pago
        if not new or new == b.model:
            return False
        print(f"[Brain:{b.name}] modelo '{b.model}' retirado -> uso '{new}'")
        b.model = new
        return True

    def _call_ollama(self, b: _Backend, system: str, messages: list, max_tokens: int, timeout: float) -> str:
        r = _session.post(f"{b.base_url}/api/chat", timeout=max(timeout, 25.0), json={
            "model": b.model, "stream": False, "think": False, "keep_alive": "15m",
            "messages": [{"role": "system", "content": system}] + messages,
            # num_gpu=0: NUNCA cargar capas en la GPU (es del juego).
            "options": {"num_gpu": 0, "num_thread": cfg.LOCAL_LLM_THREADS,
                        "num_predict": max_tokens, "temperature": 0.3},
        })
        if r.status_code == 404:
            raise _ModelGone(f"ollama: {r.text[:120]}")
        r.raise_for_status()
        content = (r.json().get("message", {}).get("content") or "").strip()
        if not content:
            raise ValueError("ollama: respuesta vacía")
        return content

    def _call_one(self, b: _Backend, system: str, messages: list, max_tokens: int, timeout: float) -> str:
        if b.kind == "ollama":
            return self._call_ollama(b, system, messages, max_tokens, timeout)
        payload = {
            "model": b.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens + (b.reasoning_reserve if _is_reasoning_model(b.model) else 0),
            "temperature": 0.3,
        }
        if _is_reasoning_model(b.model) and b.name in ("groq", "cerebras", "openrouter"):
            payload["reasoning_effort"] = "low"
        r = _session.post(f"{b.base_url}/chat/completions", headers=self._headers(b),
                          json=payload, timeout=timeout)
        if r.status_code == 429 and r.headers.get("x-ratelimit-limit-req-minute") == "0":
            # Mistral sin plan activo responde 429 con límite 0: no es un
            # límite pasajero, es "esta cuenta no tiene cuota".
            raise _PermanentError("429 con cuota 0 (plan no activado)")
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("retry-after", "30"))
            except ValueError:
                wait = 30.0
            b.pause(max(5.0, min(wait, 600.0)), "rate limit (429)")
            raise _RateLimited(f"429 rate limit, pausado {wait:.0f}s")
        _classify_http(r)
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            # Truncado antes de escribir content (razonamiento sin margen) —
            # cuenta como fallo real, no como respuesta vacía válida.
            raise ValueError("respuesta sin content (probable corte en razonamiento)")
        return content

    def complete(self, system: str, messages: list, max_tokens: int = 110, timeout: float = 10.0,
                 allow_local: bool = True) -> str:
        """Intenta cada backend disponible en orden. Devuelve "" si todos fallan.

        allow_local=False salta el LLM local (CPU): para datos concretos
        (combustible, gaps…) la respuesta determinista es instantánea y exacta,
        y gemma4:e2b falló la cuenta de combustible en la prueba real."""
        with self._lock:
            candidates = [b for b in self.backends if b.available()
                          and (allow_local or b.kind != "ollama")]
        for b in candidates:
            t0 = time.monotonic()
            for attempt in (1, 2):
                try:
                    reply = self._call_one(b, system, messages, max_tokens, timeout)
                    b.record_success()
                    self.last_backend = b.name
                    print(f"[Brain] {b.name}/{b.model} respondió en {time.monotonic()-t0:.2f}s")
                    return reply
                except _ModelGone as e:
                    print(f"[Brain:{b.name}] modelo no disponible: {e}")
                    if attempt == 1 and self._heal(b):
                        continue                       # reintento con el modelo nuevo
                    b.pause(_PERMANENT_PAUSE_S, f"modelo '{b.model}' no existe")
                except _PermanentError as e:
                    print(f"[Brain:{b.name}] {e} — apartado esta sesión")
                    b.pause(_PERMANENT_PAUSE_S, str(e)[:60])
                except _RateLimited as e:
                    print(f"[Brain:{b.name}] {e}")      # no cuenta para el breaker
                except Exception as e:
                    print(f"[Brain:{b.name}] {e} (tras {time.monotonic()-t0:.2f}s)")
                    b.record_failure()
                break
        self.last_backend = None
        return ""

    def status(self) -> dict:
        out = {}
        for b in self.backends:
            if not b.api_key:
                out[b.name] = "sin key"
            elif b.available():
                out[b.name] = f"activo ({b.model})"
            else:
                out[b.name] = f"pausado {int(b.paused_until - time.time())}s: {b.pause_reason}"
        return out


# ─── Servidor Ollama local (solo CPU) ────────────────────────────────────────

_ollama_proc = None


def ensure_local_llm() -> bool:
    """Si hay LOCAL_LLM_MODEL y Ollama no responde, lo levanta SIN GPU
    (CUDA_VISIBLE_DEVICES vacío) y lo apaga al salir. Si ya corría (p.ej. para
    otro proyecto) lo reutiliza tal cual: num_gpu=0 por petición igual
    mantiene a Victor fuera de la VRAM."""
    global _ollama_proc
    if not cfg.LOCAL_LLM_MODEL:
        return False
    try:
        _session.get(f"{cfg.LOCAL_LLM_URL}/api/version", timeout=1.5).raise_for_status()
        return True
    except Exception:
        pass
    import atexit, os, shutil, subprocess
    if not shutil.which("ollama"):
        print("[Brain:local] ollama no está instalado")
        return False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OLLAMA_HOST=cfg.LOCAL_LLM_URL.split("//")[-1])
    _ollama_proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
    atexit.register(_ollama_proc.terminate)
    for _ in range(20):
        time.sleep(0.5)
        try:
            _session.get(f"{cfg.LOCAL_LLM_URL}/api/version", timeout=1).raise_for_status()
            print(f"[Brain:local] ollama levantado en CPU para {cfg.LOCAL_LLM_MODEL}")
            return True
        except Exception:
            continue
    print("[Brain:local] ollama no arrancó")
    return False


# ─── STT ──────────────────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """Groq (turbo) -> Deepgram -> faster-whisper local en CPU.

    El local solo se usa si la nube no dio NADA por fallo/ausencia de key —
    no cuando Groq rechazó el audio por anti-alucinación (eso es "no había
    voz", y repetirlo localmente reintroduciría la alucinación)."""
    text, rejected = _transcribe_groq(wav_path)
    if text or rejected:
        return text
    if cfg.DEEPGRAM_API_KEY:
        text = _transcribe_deepgram(wav_path)
        if text:
            return text
    return _transcribe_local(wav_path)


def _transcribe_groq(wav_path: str) -> tuple[str, bool]:
    """Devuelve (texto, rechazado_por_anti_alucinación)."""
    if not cfg.GROQ_API_KEY:
        return "", False
    try:
        with open(wav_path, "rb") as f:
            r = _session.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {cfg.GROQ_API_KEY}"},
                files={"file": ("rec.wav", f, "audio/wav")},
                data={"model": cfg.GROQ_STT_MODEL, "language": "es", "response_format": "verbose_json",
                      "prompt": cfg.STT_PROMPT},
                timeout=15,
            )
        r.raise_for_status()
        data = r.json()
        text = data.get("text", "").strip()
        segments = data.get("segments", [])
        if segments:
            no_speech = max(s.get("no_speech_prob", 0) for s in segments)
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            if no_speech > 0.35 or avg_logprob < -0.5:
                print(f"[STT groq] Rechazado (no_speech={no_speech:.2f} logprob={avg_logprob:.2f}): {text!r}")
                return "", True
        return text, False
    except Exception as e:
        print(f"[STT groq] {e}")
        return "", False


def _transcribe_deepgram(wav_path: str) -> str:
    try:
        with open(wav_path, "rb") as f:
            audio = f.read()
        r = _session.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": "nova-3", "language": "es", "smart_format": "true"},
            headers={"Authorization": f"Token {cfg.DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            data=audio,
            timeout=15,
        )
        r.raise_for_status()
        alts = r.json()["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"].strip() if alts else ""
    except Exception as e:
        print(f"[STT deepgram] {e}")
        return ""


# ─── STT local (faster-whisper, CPU) ──────────────────────────────────────────

_local_model = None
_local_lock = threading.Lock()
_local_failed = False


def local_stt_enabled() -> bool:
    return (cfg.LOCAL_STT_MODEL or "off").lower() not in ("off", "none", "0", "")


def _get_local_model():
    global _local_model, _local_failed
    if _local_model is not None or _local_failed or not local_stt_enabled():
        return _local_model
    with _local_lock:
        if _local_model is None and not _local_failed:
            try:
                from faster_whisper import WhisperModel
                t0 = time.monotonic()
                # device=cpu A PROPÓSITO: la VRAM de la 4060 es del juego.
                _local_model = WhisperModel(cfg.LOCAL_STT_MODEL, device="cpu",
                                            compute_type="int8", cpu_threads=cfg.LOCAL_STT_THREADS)
                print(f"[STT local] faster-whisper '{cfg.LOCAL_STT_MODEL}' listo en {time.monotonic()-t0:.1f}s")
            except Exception as e:
                _local_failed = True
                print(f"[STT local] no disponible: {e}")
    return _local_model


def preload_local_stt() -> None:
    """Carga el modelo en segundo plano para que el primer fallback no espere."""
    if local_stt_enabled():
        threading.Thread(target=_get_local_model, daemon=True, name="stt-local-load").start()


def _transcribe_local(wav_path: str) -> str:
    model = _get_local_model()
    if model is None:
        return ""
    try:
        t0 = time.monotonic()
        segs, _info = model.transcribe(wav_path, language="es", beam_size=1,
                                       condition_on_previous_text=False,
                                       initial_prompt=cfg.STT_PROMPT)
        segs = list(segs)
        # Mismo criterio anti-alucinación que con Groq.
        if segs and (max(s.no_speech_prob for s in segs) > 0.5
                     or sum(s.avg_logprob for s in segs) / len(segs) < -0.8):
            print("[STT local] Rechazado (probable no-voz)")
            return ""
        text = "".join(s.text for s in segs).strip()
        print(f"[STT local] {time.monotonic()-t0:.2f}s: {text!r}")
        return text
    except Exception as e:
        print(f"[STT local] {e}")
        return ""
