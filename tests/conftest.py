"""Infra compartida de la suite.

Tras la reestructura del 2026-08-28 el código vive en src/victor/ y los tests en
tests/. Los módulos se importan planos (`import victor_brain`, `import engineer`),
así que solo hace falta poner src/victor/ en el path — ningún import cambió.

La fixture `eng` importa el módulo `engineer` una sola vez por sesión con las
dependencias de audio (piper/pyaudio) stubbeadas SOLO durante el import, y las
restaura enseguida para no pisar los tests que sí usan Piper real (test_v7_ears).
Antes esto vivía en la cabecera de cada banco standalone y contaminaba el proceso
si se corría junto a pytest — de ahí que hubiera que correrlos sueltos.

Los helpers para los tests que lanzan wine (test_shm_pipeline, test_e2e_pipeline)
viven en `tests/_wine_support.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "victor"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def eng():
    """El módulo `engineer` ya importado. piper/pyaudio se stubbean solo mientras
    dura el import y se restauran después (test_v7_ears usa Piper de verdad)."""
    keys = ("pyaudio", "piper", "piper.voice")
    saved = {k: sys.modules.get(k) for k in keys}

    piper_stub = mock.MagicMock()
    piper_stub.PiperVoice.load.return_value = mock.MagicMock()
    sys.modules["pyaudio"] = mock.MagicMock()
    sys.modules["piper"] = piper_stub
    sys.modules["piper.voice"] = piper_stub
    try:
        import engineer  # noqa: PLC0415
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return engineer


@pytest.fixture
def spoken(eng):
    """Captura lo que Victor mandaría a TTS. Lista de (prioridad, texto).

    Restaura `_tts_pq.put` al terminar para no dejar el módulo (scope sesión)
    con un put parcheado entre tests.
    """
    captured: list[tuple[int, str]] = []
    orig_put = eng._tts_pq.put

    def _put(item):
        if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
            captured.append(item)

    eng._tts_pq.put = _put
    try:
        yield captured
    finally:
        eng._tts_pq.put = orig_put
