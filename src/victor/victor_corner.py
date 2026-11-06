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
            if self._speed >= MIN_SPEED and (self._yaw > YAW_DEG_S or (brake or 0.0) > BRAKE_ON):
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
