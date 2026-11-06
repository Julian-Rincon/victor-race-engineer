#!/usr/bin/env python3
"""Victor doctor — prueba REAL de punta a punta, sin abrir Assetto Corsa.

Qué comprueba (cada paso imprime OK / FALLA / AVISO):
  1. Config: qué .env se cargaron y qué keys hay (nunca imprime las keys).
  2. Cerebro: ping real a cada backend con key (1 pregunta corta) — mide
     latencia, detecta modelos retirados (y muestra a cuál se autocuró).
  3. Voz: Piper sintetiza comandos reales del piloto ("Victor, cuánto
     combustible me queda", ...).
  4. Oído: cada comando pasa por Silero VAD (¿se detecta como voz?) y por el
     STT real (Groq si hay key, si no faster-whisper local en CPU).
  5. Cerebro de comandos: el texto transcrito entra a engineer._handle_voice
     con telemetría simulada de carrera y se verifica que Victor CONTESTE
     algo coherente (con LLM o con la respuesta local sin LLM).
  6. Nativo: wine, ac_shm_reader.exe, mic disponible.

Uso:
  python3 tools/victor_doctor.py            # todo, sin sonido
  python3 tools/victor_doctor.py --speak    # además reproduce las respuestas
  python3 tools/victor_doctor.py --no-llm   # no gasta llamadas a APIs
Código de salida 0 si no hubo FALLAs.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "victor"))

import numpy as np  # noqa: E402

import victor_config as cfg  # noqa: E402

# (frase, comprobación sobre la respuesta hablada)
CASES = [
    ("Victor, cuánto combustible me queda", lambda r: "litro" in r.lower() or "vuelta" in r.lower()),
    ("Victor, cómo están los neumáticos", lambda r: any(w in r.lower() for w in ("grado", "°c", "neum", "goma", "desgaste", "temperatura", "vida"))),
    ("Victor, a cuánto está el de adelante", lambda r: len(r) > 5),
    ("Victor, en qué posición voy", lambda r: any(w in r.lower() for w in ("cuarto", "4", "posición", "puesto", "lugar"))),
    ("Victor, tengo daños", lambda r: len(r) > 5),
    ("Victor, modo gap", lambda r: r.startswith("Gap mode activo")),
    ("Victor, para los gaps", lambda r: r == "Gap mode desactivado."),
    ("qué buena está esta canción", None),   # sin wake word → debe ignorarse
]

_fails = 0


def ok(msg):   print(f"  \033[32mOK\033[0m    {msg}")
def warn(msg): print(f"  \033[33mAVISO\033[0m {msg}")
def fail(msg):
    global _fails
    _fails += 1
    print(f"  \033[31mFALLA\033[0m {msg}")


def section(t): print(f"\n── {t} " + "─" * max(0, 60 - len(t)))


def check_config():
    section("1. Configuración")
    used = cfg.ENV_FILES_USED
    (ok if used else warn)(f".env cargados: {[str(p) for p in used] or 'ninguno'}")
    keys = {n: bool(getattr(cfg, n)) for n in
            ("GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "NVIDIA_API_KEY",
             "GEMINI_API_KEY", "MISTRAL_API_KEY", "DEEPGRAM_API_KEY")}
    present = [n.removesuffix("_API_KEY").lower() for n, v in keys.items() if v]
    unused = [n.removesuffix("_API_KEY").lower() for n, v in keys.items() if not v]
    if present:
        ok(f"keys: {', '.join(present)}" + (f" + LLM local {cfg.LOCAL_LLM_MODEL}" if cfg.LOCAL_LLM_MODEL else ""))
    # Proveedores sin key = no configurados a propósito, no es un problema.
    print(f"        no usados: {', '.join(unused)}")
    if not any(keys.values()):
        warn(f"sin keys: Victor funcionará en modo local (STT local + respuestas sin LLM). "
             f"Crea {cfg.USER_ENV_PATH} a partir de .env.example")
    return keys


def check_brain(skip: bool):
    section("2. Cerebro (LLM)")
    import victor_brain as vb
    brain = vb.VictorBrain()
    if cfg.LOCAL_LLM_MODEL and not skip:
        vb.ensure_local_llm()
    if skip:
        warn("omitido (--no-llm)")
        return brain
    any_ok = False
    for b in brain.backends:
        if not b.api_key:
            continue                                # no configurado a propósito
        others = [x for x in brain.backends if x is not b]
        saved = [(x, x.api_key) for x in others]
        for x in others:
            x.api_key = ""                          # aislar este backend
        t0 = time.monotonic()
        try:
            r = brain.complete("Eres un ingeniero de carreras. Responde en español, 5 palabras máximo.",
                               [{"role": "user", "content": "Di: radio check"}], max_tokens=40, timeout=20)
        finally:
            for x, k in saved:
                x.api_key = k
        dt = time.monotonic() - t0
        if r:
            any_ok = True
            ok(f"{b.name}/{b.model}: {dt:.2f}s → {r[:50]!r}")
        elif "cuota 0" in brain.status()[b.name]:
            warn(f"{b.name}: la cuenta no tiene cuota — activa el plan gratis en su consola")
        else:
            fail(f"{b.name}: sin respuesta ({brain.status()[b.name]})")
    if not any_ok:
        warn("ningún LLM respondió — Victor usará respuestas locales deterministas")
    return brain


def synth(phrases):
    section("3. Voz (Piper)")
    from piper import PiperVoice
    t0 = time.monotonic()
    voice = PiperVoice.load(str(cfg.PIPER_MODEL_PATH))
    ok(f"modelo {cfg.PIPER_MODEL_PATH.parent.name} cargado en {time.monotonic()-t0:.2f}s")
    tmp = Path(tempfile.mkdtemp(prefix="victor-doctor-"))
    out = []
    for i, text in enumerate(phrases):
        chunks = list(voice.synthesize(text))
        p = tmp / f"cmd{i}.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(chunks[0].sample_rate)
            for c in chunks:
                w.writeframes(c.audio_int16_bytes)
        out.append(p)
    ok(f"{len(out)} comandos sintetizados en {tmp}")
    return voice, out


def _to16k(path: Path) -> np.ndarray:
    from scipy.signal import resample_poly
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != 16000:
        pcm = resample_poly(pcm.astype(np.float32), 16000, sr).astype(np.int16)
    return pcm


def check_ears(wavs, phrases):
    section("4. Oído (Silero VAD + STT)")
    import victor_brain as vb
    from victor_ears import SileroVad, iter_vad_frames
    vad = SileroVad(cfg.SILERO_VAD_PATH)
    thr = cfg.load_calibration()["vad_threshold"]
    for p, text in zip(wavs, phrases):
        vad.reset()
        probs = [vad.speech_prob(f) for f in iter_vad_frames(_to16k(p))]  # ya vienen en float32
        mx = max(probs) if probs else 0.0
        (ok if mx >= thr else fail)(f"VAD {mx:.2f} (umbral {thr}) «{text}»")
    if not cfg.GROQ_API_KEY:
        vb.preload_local_stt()
    texts = []
    for p, text in zip(wavs, phrases):
        t0 = time.monotonic()
        got = vb.transcribe(str(p))
        dt = time.monotonic() - t0
        texts.append(got)
        (ok if got else fail)(f"STT {dt:.2f}s «{text}» → {got!r}")
    return texts


def check_commands(texts, phrases, speak: bool, voice):
    section("5. Comandos → respuesta de Victor")
    sys.modules.setdefault("pyaudio", mock.MagicMock())
    import engineer as eng

    t = eng.Telemetry()
    t.track, t.car, t.session, t.status = "ks_monza", "ks_ferrari_488_gt3", 2, 2
    t.fuel, t.fuel_laps, t.position, t.cars, t.lap = 31.5, 9.3, 4, 16, 6
    t.tyre_fl, t.tyre_fr, t.tyre_rl, t.tyre_rr = 86, 89, 81, 83
    t.tyre_wear_fl = t.tyre_wear_fr = t.tyre_wear_rl = t.tyre_wear_rr = 96
    t.dmg_front = 12
    eng._current = t

    spoken: list[str] = []
    eng._tts_pq.put = lambda item: spoken.append(item[1]) if item and item[1] else None
    eng._in_braking_zone = lambda: False
    eng._muted = False
    if speak:
        eng._piper = voice

    for (phrase, check), heard in zip(CASES, texts):
        spoken.clear()
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        with mock.patch.object(eng, "_transcribe", return_value=heard):
            t0 = time.monotonic()
            eng._handle_voice(str(tmp))
            dt = time.monotonic() - t0
        reply = spoken[0] if spoken else ""
        if check is None:
            (ok if not spoken else fail)(f"«{phrase}» ignorado sin wake word" if not spoken
                                         else f"«{phrase}» NO debió responder: {reply!r}")
            continue
        (ok if reply and check(reply) else fail)(f"{dt:.2f}s «{phrase}» → {reply!r}")
        if speak and reply:
            eng._speak_piper(reply)


def check_native():
    section("6. Nativo (Wine / SHM / mic)")
    (ok if shutil.which("wine") or list(Path.home().glob(".local/share/Steam/compatibilitytools.d/GE-Proton*"))
     else warn)("runtime wine/GE-Proton disponible")
    exe = _ROOT / "native/ac_shm_reader.exe"
    (ok if exe.exists() else fail)(f"{exe.relative_to(_ROOT)} " + ("presente" if exe.exists() else
                                   "falta — x86_64-w64-mingw32-gcc -O2 -o native/ac_shm_reader.exe native/ac_shm_reader.c"))
    (ok if shutil.which("paplay") else fail)("paplay (salida de voz)")
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        n = sum(1 for i in range(pa.get_device_count()) if pa.get_device_info_by_index(i)["maxInputChannels"] > 0)
        pa.terminate()
        (ok if n else fail)(f"{n} dispositivos de entrada de audio")
    except Exception as e:
        warn(f"pyaudio: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()

    check_config()
    check_brain(a.no_llm)
    phrases = [c[0] for c in CASES]
    voice, wavs = synth(phrases)
    texts = check_ears(wavs, phrases)
    if a.no_llm:
        import engineer as eng
        eng._groq = lambda *x, **k: ""
    check_native()
    check_commands(texts, phrases, a.speak, voice)
    print(f"\n{'✅ Todo OK' if not _fails else f'❌ {_fails} fallas'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
