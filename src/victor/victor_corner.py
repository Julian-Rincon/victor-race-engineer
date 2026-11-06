"""Victor — detector de curva genérico (cualquier pista, sin mapas a mano).

Regla del piloto: en curva SOLO habla el spotter (coches a los lados); todo
lo demás espera a la recta. Las zonas de frenada mapeadas (BRAKING_ZONES)
solo existen para 14 circuitos y no cubren curvas sin frenada fuerte
(Blanchimont, Eau Rouge), así que se detecta la curva desde la telemetría:

- velocidad de giro (yaw rate) calculada con la posición mundial (wx, wz):
  el rumbo entre muestras separadas >= 3 m; en recta ~0 °/s, en curva rápida
  de 400 m de radio a 280 km/h ≈ 11 °/s.
- freno > 25 %: zona de frenada aunque la curva aún no empiece.

Histéresis: tras la última señal de curva hace falta HOLD_S de recta limpia
para volver a hablar (evita cortar a mitad de una enlazada).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

YAW_DEG_S = 7.0        # por encima = curva
BRAKE_ON = 0.25        # freno por encima = zona de frenada (0.21 = levantón leve, medido en Spa)
MIN_SPEED = 40.0       # km/h — por debajo (pits, parrilla) no se bloquea la voz
MIN_STEP_M = 3.0       # distancia mínima entre muestras para medir rumbo
HOLD_S = 1.2           # recta limpia necesaria tras una curva
STALE_S = 1.5          # sin datos recientes -> no bloquear


class CornerDetector:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._anchor: Optional[tuple[float, float, float]] = None   # x, z, t
        self._heading: Optional[float] = None
        self._yaw = 0.0
        self._last_corner_t = -1e9
        self._last_update = -1e9
        self._speed = 0.0
        self.raw = False           # señal de curva de ESTA muestra (para aprender el mapa)

    def update(self, wx: Optional[float], wz: Optional[float], speed: float, brake: float) -> None:
        now = self._clock()
        with self._lock:
            self._last_update = now
            self._speed = speed or 0.0
            if wx is not None and wz is not None:
                if self._anchor is None:
                    self._anchor = (wx, wz, now)
                else:
                    ax, az, at = self._anchor
                    dist = math.hypot(wx - ax, wz - az)
                    if dist / max(now - at, 1e-3) > 120.0:
                        # >430 km/h: teletransporte (reset a boxes/parrilla).
                        self._anchor, self._heading, self._yaw = (wx, wz, now), None, 0.0
                    elif dist >= MIN_STEP_M:
                        heading = math.atan2(wz - az, wx - ax)
                        dt = max(now - at, 1e-3)
                        if self._heading is not None:
                            d = (heading - self._heading + math.pi) % (2 * math.pi) - math.pi
                            raw = abs(math.degrees(d)) / dt
                            self._yaw = 0.5 * self._yaw + 0.5 * raw   # EMA suave
                        self._heading = heading
                        self._anchor = (wx, wz, now)
            self.raw = self._speed >= MIN_SPEED and (self._yaw > YAW_DEG_S or (brake or 0.0) > BRAKE_ON)
            if self.raw:
                self._last_corner_t = now

    def in_corner(self) -> bool:
        now = self._clock()
        with self._lock:
            if now - self._last_update > STALE_S or self._speed < MIN_SPEED:
                return False
            return now - self._last_corner_t < HOLD_S

    @property
    def yaw_rate(self) -> float:
        return self._yaw


# ─── Mapa de curvas por pista (mirar ADELANTE) ──────────────────────────────
#
# El detector solo sabe si AHORA estás girando. Un mensaje de 4 s que empieza
# al final de la recta termina en plena frenada (pasó en Spa: el aviso de
# vuelta salta en la línea de meta, 200 m antes de La Source). El mapa marca
# en qué tramos del spline hay curva para exigir recta suficiente ADELANTE
# antes de empezar a hablar.
#
# Fuentes, en este orden:
#   1. content/tracks/<pista>[/<layout>]/ai/fast_lane.ai — la línea de IA de
#      AC trae la geometría de TODA la pista desde la vuelta 0.
#   2. Lo que aprende el detector vuelta a vuelta (>= 2 vueltas distintas, para
#      que un trompo en recta no la marque como curva para siempre).

import struct
from pathlib import Path

MAP_BUCKETS = 2000          # ~3.5 m por casilla en Spa
CORNER_RADIUS_M = 350.0     # radio menor = curva (Kemmel/Blanchimont ya entran)
BRAKE_LEAD_M = 120.0        # la frenada empieza antes de la curva
LEARN_HITS = 2
CHARS_PER_S = 14.0          # Piper es_davefx, medido a ojo: ~14 caracteres/s
SPEECH_CAP_S = 6.0          # una frase nunca necesita más recta que esto


def load_ai_line(path: Path) -> Optional[list[tuple[float, float, float]]]:
    """Lee fast_lane.ai → [(x, z, distancia_acumulada)] o None."""
    try:
        b = Path(path).read_bytes()
        _ver, n, _lt, _sc = struct.unpack_from("<4i", b, 0)
        if n < 50 or 16 + n * 20 > len(b):
            return None
        out = []
        for i in range(n):
            x, _y, z, dist, _id = struct.unpack_from("<4fi", b, 16 + i * 20)
            out.append((x, z, dist))
        return out if out[-1][2] > 100 else None
    except (OSError, struct.error):
        return None


def corner_flags_from_line(pts: list[tuple[float, float, float]], buckets: int = MAP_BUCKETS,
                           radius_m: float = CORNER_RADIUS_M,
                           lead_m: float = BRAKE_LEAD_M) -> list[bool]:
    import bisect
    total = pts[-1][2]
    dists = [p[2] for p in pts]
    n = len(pts)

    def at(d: float) -> tuple[float, float]:
        i = min(bisect.bisect_left(dists, d % total), n - 1)
        return pts[i][0], pts[i][1]

    flags = [False] * buckets
    win = 15.0
    for k in range(buckets):
        d = (k + 0.5) / buckets * total
        (ax, az), (bx, bz), (cx, cz) = at(d - win), at(d), at(d + win)
        h1 = math.atan2(bz - az, bx - ax)
        h2 = math.atan2(cz - bz, cx - bx)
        dh = abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
        chord = math.hypot(cx - ax, cz - az)
        if dh > 1e-6 and chord / dh < radius_m:
            flags[k] = True
    lead = max(1, int(round(lead_m / total * buckets)))
    out = flags[:]
    for k in range(buckets):
        if flags[k] and not flags[k - 1]:          # entrada de curva
            for j in range(1, lead + 1):
                out[(k - j) % buckets] = True
    return out


class CornerMap:
    def __init__(self, buckets: int = MAP_BUCKETS) -> None:
        self.n = buckets
        self._lock = threading.Lock()
        self.key: Optional[tuple[str, str]] = None
        self.track_len = 0.0
        self._static = [False] * buckets
        self._hits = [0] * buckets
        self._lap_seen: list[int] = [-1] * buckets
        self.source = "vacío"

    def reset(self, key: tuple[str, str], ac_root: Optional[Path] = None) -> None:
        """Nueva pista: carga la línea de IA si existe y olvida lo aprendido."""
        track, layout = key
        pts = None
        if ac_root is not None and track:
            base = Path(ac_root) / "content" / "tracks" / track
            for p in ([base / layout / "ai" / "fast_lane.ai"] if layout else []) + [base / "ai" / "fast_lane.ai"]:
                pts = load_ai_line(p)
                if pts:
                    break
        with self._lock:
            self.key = key
            self._hits = [0] * self.n
            self._lap_seen = [-1] * self.n
            if pts:
                self._static = corner_flags_from_line(pts, self.n)
                self.track_len = pts[-1][2]
                self.source = "línea IA"
            else:
                self._static = [False] * self.n
                self.track_len = 0.0
                self.source = "aprendido"

    def _idx(self, spline: float) -> int:
        return int((spline % 1.0) * self.n) % self.n

    def learn(self, spline: float, lap: int, in_corner: bool) -> None:
        if not in_corner:
            return
        k = self._idx(spline)
        with self._lock:
            if self._lap_seen[k] != lap:
                self._lap_seen[k] = lap
                self._hits[k] += 1

    def _flag(self, k: int) -> bool:
        return self._static[k] or self._hits[k] >= LEARN_HITS

    def is_corner(self, spline: float) -> bool:
        with self._lock:
            return self._flag(self._idx(spline))

    def known(self) -> bool:
        with self._lock:
            return any(self._static) or any(h >= LEARN_HITS for h in self._hits)

    def clear_ahead_m(self, spline: float, track_len: float) -> float:
        """Metros de recta por delante hasta la próxima curva (inf = sin datos)."""
        L = track_len or self.track_len
        with self._lock:
            if L <= 0 or not (any(self._static) or any(h >= LEARN_HITS for h in self._hits)):
                return math.inf
            k0 = self._idx(spline)
            for j in range(self.n):
                if self._flag((k0 + j) % self.n):
                    return j / self.n * L
        return math.inf


def speech_seconds(text: str) -> float:
    return min(SPEECH_CAP_S, 0.4 + len(text or "") / CHARS_PER_S)
