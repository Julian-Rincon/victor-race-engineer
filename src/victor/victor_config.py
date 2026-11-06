"""Victor — configuración central.

API keys: se buscan en varios .env por orden (el primero que exista gana, el
entorno del proceso siempre tiene prioridad):

  1. $VICTOR_ENV_FILE                       (ruta explícita)
  2. <repo>/.env                            (no versionado)
  3. ~/.config/victor/victor.env            (recomendado — fuera del repo)
  4. ~/Documentos/Proyectos/nexus-core/.env (legado NEXUS v1, ya no existe)

Cada clave acepta el nombre corto (GROQ_API_KEY) o el de NEXUS
(NEXUS_GROQ_API_KEY). Antes solo se leía el .env de nexus-core: al archivarse
NEXUS v1 Victor quedó SIN NINGUNA key — mudo — sin que nada lo avisara.

Assets locales: voz Piper, Silero VAD, calibración del pipeline de voz.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DIR  = Path(__file__).parent.resolve()          # src/victor/
_ROOT = Path(__file__).resolve().parents[2]       # raíz del repo

# Legado: se conserva el nombre por compatibilidad (tests / scripts viejos).
NEXUS_ENV_PATH = Path.home() / "Documentos/Proyectos/nexus-core/.env"
USER_ENV_PATH  = Path.home() / ".config/victor/victor.env"


def env_candidates() -> list[Path]:
    out: list[Path] = []
    if os.getenv("VICTOR_ENV_FILE"):
        out.append(Path(os.environ["VICTOR_ENV_FILE"]).expanduser())
    out += [_ROOT / ".env", USER_ENV_PATH, NEXUS_ENV_PATH]
    return out


# Assets locales (no versionados, en la raíz del repo).
PIPER_MODEL_PATH = _ROOT / "voices/es_davefx_medium/model.onnx"
SILERO_VAD_PATH  = _ROOT / "models/silero_vad.onnx"
CALIBRATION_PATH = _ROOT / "data/victor_calibration.json"
DRIVER_MEMORY_PATH = _ROOT / "data/driver_memory.json"


def _load_dotenv(p: Path) -> dict:
    env: dict = {}
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def _load_env_chain(paths: list[Path]) -> tuple[dict, list[Path]]:
    """Fusiona los .env existentes; los primeros de la lista ganan."""
    merged: dict = {}
    used: list[Path] = []
    for p in paths:
        if p.is_file():
            used.append(p)
            for k, v in _load_dotenv(p).items():
                merged.setdefault(k, v)
    return merged, used


_env, ENV_FILES_USED = _load_env_chain(env_candidates())


def get_key(name: str, default: str = "") -> str:
    """Entorno del proceso > .env. Prueba el nombre tal cual y su par
    con/sin prefijo NEXUS_ (GROQ_API_KEY <-> NEXUS_GROQ_API_KEY)."""
    short = name[len("NEXUS_"):] if name.startswith("NEXUS_") else name
    for n in (name, short, f"NEXUS_{short}", f"VICTOR_{short}"):
        v = os.getenv(n) or _env.get(n)
        if v:
            return v
    return default


# ── API keys / modelos ───────────────────────────────────────────────────────
GROQ_API_KEY     = get_key("GROQ_API_KEY") or get_key("JARVIS_GROQ_API_KEY")
CEREBRAS_API_KEY = get_key("CEREBRAS_API_KEY")
NVIDIA_API_KEY   = get_key("NVIDIA_API_KEY")
GEMINI_API_KEY   = get_key("GEMINI_API_KEY") or get_key("GOOGLE_API_KEY")
DEEPGRAM_API_KEY = get_key("DEEPGRAM_API_KEY")

# Defaults revisados 2026-11: llama-3.3-70b-versatile (Groq) se apagó el
# 2026-08-16 y gemini-2.0-flash el 2026-06-01 — Victor llamaba a modelos
# muertos. Aun así los nombres caducan rápido: victor_brain se autocura
# (si el proveedor responde "modelo no existe" elige uno vivo de /models
# según MODEL_PREFERENCES).
GROQ_MODEL     = get_key("GROQ_MODEL", "openai/gpt-oss-120b")
CEREBRAS_MODEL = get_key("CEREBRAS_MODEL", "gpt-oss-120b")
NVIDIA_MODEL   = get_key("NVIDIA_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
GEMINI_MODEL   = get_key("GEMINI_MODEL", "gemini-flash-latest")

# Backend propio OpenAI-compatible (OpenRouter, llama.cpp server, LM Studio…).
# Si se define CUSTOM_LLM_BASE_URL va PRIMERO en la cadena (es elección
# explícita). Para un servidor local sin auth basta cualquier key ("local").
CUSTOM_LLM_BASE_URL = get_key("CUSTOM_LLM_BASE_URL").rstrip("/")
CUSTOM_LLM_MODEL    = get_key("CUSTOM_LLM_MODEL")
CUSTOM_LLM_API_KEY  = get_key("CUSTOM_LLM_API_KEY") or ("local" if CUSTOM_LLM_BASE_URL else "")

# Audio ducking: volumen relativo del juego mientras Victor habla (1.0 = off).
DUCK_LEVEL = float(get_key("DUCK_LEVEL", "0.35") or 0.35)
DUCK_APPS  = tuple(a.strip() for a in get_key("DUCK_APPS", "").split(",") if a.strip())

# Subcadenas en orden de preferencia para la autocuración por backend.
MODEL_PREFERENCES = {
    "groq":     ["gpt-oss-120b", "gpt-oss-20b", "qwen3", "llama-4", "kimi", "llama"],
    "cerebras": ["gpt-oss-120b", "qwen-3", "llama"],
    "nvidia":   ["llama-4-maverick", "llama-3.3-70b", "llama"],
    "gemini":   ["flash-latest", "flash-lite", "flash"],
}

# STT: whisper-large-v3-turbo es ~2x más rápido que large-v3 en Groq con
# calidad casi igual en español — latencia es lo que importa en pista.
GROQ_STT_MODEL = get_key("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# STT local de respaldo (faster-whisper en CPU — 0 VRAM, no compite con el
# juego). Medido en este PC (i7 16 hilos, int8, 4 hilos): small ≈ 1 s por
# comando; large-v3-turbo ≈ 4.6 s (demasiado para pista). "off" lo desactiva.
LOCAL_STT_MODEL   = get_key("LOCAL_STT_MODEL", "small")
LOCAL_STT_THREADS = int(get_key("LOCAL_STT_THREADS", "4") or 4)

# Prompt de vocabulario para Whisper (Groq y local). Sin él, en la prueba real
# con comandos sintetizados, `small` transcribía "tengo daños" -> "Tecmodaneos",
# "modo gap" -> "Modogap", "los gaps" -> "los gafos" y a veces se comía el
# "Victor" inicial (sin wake word el comando se descarta).
STT_PROMPT = get_key("STT_PROMPT",
    "Victor, ingeniero de carrera. Victor, cuánto combustible me queda. "
    "Victor, cómo están los neumáticos. Victor, modo gap. Victor, para los gaps. "
    "Victor, tengo daños. Victor, cuándo entro a boxes. Victor, en qué posición voy.")

# ── Calibración del pipeline de voz ──────────────────────────────────────────
# Defaults conservadores; teach_victor.py los reescribe con valores aprendidos.
DEFAULT_CALIBRATION = {
    "vad_threshold": 0.60,       # prob Silero para considerar voz
    "vad_end_threshold": 0.35,   # por debajo de esto cuenta como silencio
    "vad_confirm_frames": 6,     # frames de 32 ms consecutivos para disparar (~192 ms)
    "rms_prefilter": 250.0,      # RMS mínimo para molestar al VAD (ahorra CPU)
    "calibrated_at": None,
    "false_positive_rate": None,
    "sensitivity": None,
}


def load_calibration() -> dict:
    cal = dict(DEFAULT_CALIBRATION)
    try:
        cal.update(json.loads(CALIBRATION_PATH.read_text()))
    except Exception:
        pass
    return cal


def save_calibration(cal: dict) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2, ensure_ascii=False))
