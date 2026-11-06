"""Victor — audio ducking nativo (PipeWire/PulseAudio vía pactl).

Mientras Victor habla, baja el volumen SOLO del stream del juego (acs.exe bajo
Proton) y lo restaura exactamente (valor crudo por canal) al terminar. Idea de
Crew Chief V4.19 ("audio ducking"), hecha nativa para Linux: en Windows lo
hacen con la API de sesiones de audio; aquí cada app es un sink-input de
PipeWire que pactl puede ajustar sin tocar el volumen maestro.

Nunca toca EasyEffects (procesa el audio de los audífonos de Julian) ni el
propio paplay de Victor: solo streams cuyo nombre/binario coincide con la
lista de apps del juego.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Callable

DEFAULT_APPS = ("acs.exe", "acs_x86.exe", "acs_pro.exe", "assettocorsa", "assetto corsa")
_NEVER = ("easyeffects", "paplay", "victor", "pipewire", "wireplumber")

Runner = Callable[..., subprocess.CompletedProcess]


class Ducker:
    def __init__(self, level: float = 0.35, apps: tuple[str, ...] = DEFAULT_APPS,
                 runner: Runner = subprocess.run) -> None:
        self.level = max(0.0, min(1.0, level))
        self.apps = tuple(a.lower() for a in apps)
        self._run = runner
        self._lock = threading.Lock()
        self._saved: dict[int, list[int]] = {}

    def _pactl(self, *args: str) -> subprocess.CompletedProcess:
        return self._run(["pactl", *args], capture_output=True, text=True, timeout=2)

    def _game_streams(self) -> list[dict]:
        r = self._pactl("-f", "json", "list", "sink-inputs")
        if r.returncode != 0 or not r.stdout.strip():
            return []
        out = []
        for si in json.loads(r.stdout):
            props = si.get("properties", {})
            ident = " ".join(str(props.get(k, "")) for k in
                             ("application.name", "application.process.binary", "media.name")).lower()
            # La exclusión mira solo app/binario: media.name puede ser una ruta
            # cualquiera (p.ej. con "victor" dentro) y anulaba el ducking.
            owner = " ".join(str(props.get(k, "")) for k in
                             ("application.name", "application.process.binary")).lower()
            if any(n in owner for n in _NEVER):
                continue
            if any(a in ident for a in self.apps):
                out.append(si)
        return out

    def duck(self) -> int:
        """Baja los streams del juego. Devuelve cuántos se bajaron."""
        if self.level >= 1.0:
            return 0
        with self._lock:
            if self._saved:            # ya está agachado
                return len(self._saved)
            try:
                for si in self._game_streams():
                    chans = [int(c["value"]) for c in si.get("volume", {}).values()]
                    if not chans:
                        continue
                    idx = int(si["index"])
                    self._saved[idx] = chans
                    self._pactl("set-sink-input-volume", str(idx),
                                *[str(int(v * self.level)) for v in chans])
            except Exception as e:      # pactl ausente, JSON raro, timeout…
                print(f"[Duck] {e}")
            return len(self._saved)

    def restore(self) -> None:
        with self._lock:
            saved, self._saved = self._saved, {}
        for idx, chans in saved.items():
            try:
                self._pactl("set-sink-input-volume", str(idx), *[str(v) for v in chans])
            except Exception as e:
                print(f"[Duck] restore {idx}: {e}")

    @property
    def ducked(self) -> bool:
        return bool(self._saved)
