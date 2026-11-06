#!/usr/bin/env bash
# Standalone launcher for AC Race Engineer (for testing without CM)
set -euo pipefail

# Raíz del repo (start.sh vive en la raíz). El código Python está en
# src/victor/, los .exe del lector SHM en native/.
ENGINEER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VICTOR_SRC="$ENGINEER_DIR/src/victor"
NATIVE_DIR="$ENGINEER_DIR/native"
PID_FILE="/tmp/ac-engineer.pid"
SHM_PID_FILE="/tmp/ac-engineer-shm.pid"
TRAY_PID_FILE="/tmp/ac-engineer-tray.pid"
# Proton/Steam paths for AC (Steam AppID 244210)
STEAM_ROOT="$HOME/.local/share/Steam"
AC_COMPAT_DATA="$STEAM_ROOT/steamapps/compatdata/244210"
CONFIG_INFO="$AC_COMPAT_DATA/config_info"

_resolve_proton_dir() {
  local proton_version=""
  if [[ -f "$CONFIG_INFO" ]]; then
    proton_version="$(head -1 "$CONFIG_INFO")"
  fi

  if [[ -n "$proton_version" && -x "$STEAM_ROOT/compatibilitytools.d/$proton_version/proton" ]]; then
    printf '%s\n' "$STEAM_ROOT/compatibilitytools.d/$proton_version"
  elif [[ -x "$STEAM_ROOT/compatibilitytools.d/GE-Proton10-34/proton" ]]; then
    printf '%s\n' "$STEAM_ROOT/compatibilitytools.d/GE-Proton10-34"
  elif [[ -x "$STEAM_ROOT/compatibilitytools.d/GE-Proton9-20/proton" ]]; then
    printf '%s\n' "$STEAM_ROOT/compatibilitytools.d/GE-Proton9-20"
  else
    return 1
  fi
}

PROTON_DIR="$(_resolve_proton_dir || true)"
AC_PROTON="${PROTON_DIR:+$PROTON_DIR/proton}"

_spawn_detached() {
  # Uso: _spawn_detached <pidfile> <cmd...>
  # El proceso escribe SU PROPIO PID (bash -c 'echo $$' + exec conserva el
  # PID) en vez de confiar en "$!". Antes se usaba `setsid --fork` + `$!`:
  # --fork hace que setsid bifurque y salga, así que "$!" era el PID del
  # setsid padre ya muerto — `status` decía "Detenido" con Victor vivo y
  # `stop` no lo mataba (quedaba huérfano con el micrófono abierto).
  # Verificado 2026-11-05: PID file 465969 vs engineer.py real 465971.
  local pidfile="$1"; shift
  rm -f "$pidfile"
  if command -v setsid >/dev/null 2>&1; then
    setsid --fork bash -c 'echo $$ > "$0"; exec "$@"' "$pidfile" "$@"
  else
    nohup bash -c 'echo $$ > "$0"; exec "$@"' "$pidfile" "$@" &
  fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -s "$pidfile" ]] && return 0; sleep 0.1; done
}

# Mata por PID file cuando existe (solo la instancia que ESTE script lanzó);
# recién si no hay PID file cae a pkill por nombre (huérfanos de antes de que
# existiera este tracking, o crashes que no llegaron a escribirlo). Sin esto,
# "stop" mataba por patrón de nombre sin importar de qué lanzamiento venía el
# proceso — con dos instancias de Content Manager corriendo a la vez (ej. Steam
# + una sesión de prueba vieja sin cerrar), cerrar una se llevaba puesto el
# daemon de Victor de la OTRA.
_stop_tracked() {
  local pidfile="$1" label="$2" pattern="$3"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null && echo "[Victor] $label detenido (PID $pid)." || true
    fi
    rm -f "$pidfile"
  elif pkill -f "$pattern" 2>/dev/null; then
    echo "[Victor] $label detenido (sin PID file — huérfano de una instancia anterior)."
  fi
}

case "${1:-tray}" in
  tray)
    echo "[Victor] Iniciando bandeja del sistema..."
    nohup python3 "$VICTOR_SRC/victor_tray.py" \
      > "/tmp/ac-engineer-tray.log" 2>&1 &
    echo $! > "$TRAY_PID_FILE"
    echo "[Victor] Tray PID $! — icono en la bandeja del sistema"
    ;;

  start)
    # Detiene solo LA instancia que este script lanzó antes (por PID file) —
    # no cualquier proceso con ese nombre en el sistema (ver _stop_tracked).
    _stop_tracked "$PID_FILE" "Daemon anterior" "engineer\.py"
    _stop_tracked "$SHM_PID_FILE" "SHM reader anterior" "ac_shm_reader\.exe"
    sleep 0.5
    rm -f "$NATIVE_DIR"/ac_telemetry_fast.json \
          "$NATIVE_DIR"/ac_telemetry_slow.json \
          "$NATIVE_DIR"/ac_telemetry_fast.json.tmp \
          "$NATIVE_DIR"/ac_telemetry_slow.json.tmp

    # Lanzar SHM reader como exe Windows via Wine (mismo prefix que AC)
    echo "[Victor] Iniciando SHM reader (ac_shm_reader.exe via Wine)..."
    AC_WINE="${PROTON_DIR:+$PROTON_DIR/files/bin/wine}"
    AC_WINEPREFIX="$AC_COMPAT_DATA/pfx"
    XAUTH_FILE=$(ls /run/user/$(id -u)/xauth_* 2>/dev/null | head -1)
    if [[ -f "$AC_WINE" && -f "$NATIVE_DIR/ac_shm_reader.exe" ]]; then
      WINEPREFIX="$AC_WINEPREFIX/" \
      WINEDEBUG=-all \
      WINENTSYNC=1 \
      WINEESYNC=1 \
      DISPLAY=:0 \
      WAYLAND_DISPLAY=wayland-0 \
      XDG_RUNTIME_DIR="/run/user/$(id -u)" \
      XAUTHORITY="${XAUTH_FILE:-/run/user/$(id -u)/.Xauthority}" \
      WINEDLLPATH="$PROTON_DIR/files/lib/vkd3d:$PROTON_DIR/files/lib/wine" \
      LD_LIBRARY_PATH="$PROTON_DIR/files/lib" \
      _spawn_detached "$SHM_PID_FILE" "$AC_WINE" "$NATIVE_DIR/ac_shm_reader.exe" \
        > "/tmp/ac-engineer-shm.log" 2>&1
      echo "[Victor] SHM reader PID $(cat "$SHM_PID_FILE" 2>/dev/null) — log en /tmp/ac-engineer-shm.log"
    else
      echo "[Victor] WARN: Wine de Proton o ac_shm_reader.exe no encontrado"
    fi

    echo "[Victor] Iniciando AI Race Engineer daemon..."
    _spawn_detached "$PID_FILE" python3 -u "$VICTOR_SRC/engineer.py" \
      > "/tmp/ac-engineer.log" 2>/tmp/ac-engineer-alsa.log
    echo "[Victor] Daemon PID $(cat "$PID_FILE" 2>/dev/null) — log en /tmp/ac-engineer.log"
    ;;

  stop)
    _stop_tracked "$PID_FILE" "Daemon" "engineer\.py"
    _stop_tracked "$SHM_PID_FILE" "SHM reader" "ac_shm_reader\.exe"
    ;;

  stop-tray)
    if [[ -f "$TRAY_PID_FILE" ]]; then
      PID=$(cat "$TRAY_PID_FILE")
      kill "$PID" 2>/dev/null && echo "[Victor] Tray detenido." || echo "[Victor] Tray ya terminado."
      rm -f "$TRAY_PID_FILE"
    fi
    ;;

  logs)
    tail -f /tmp/ac-engineer.log
    ;;

  shm-log)
    tail -f /tmp/ac-engineer-shm.log
    ;;

  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Daemon:  Corriendo (PID $(cat "$PID_FILE"))"
    else
      echo "Daemon:  Detenido"
    fi
    if [[ -f "$SHM_PID_FILE" ]] && kill -0 "$(cat "$SHM_PID_FILE")" 2>/dev/null; then
      echo "SHM:     Corriendo (PID $(cat "$SHM_PID_FILE"))"
    else
      echo "SHM:     Detenido"
    fi
    if [[ -f "$TRAY_PID_FILE" ]] && kill -0 "$(cat "$TRAY_PID_FILE")" 2>/dev/null; then
      echo "Tray:   Corriendo (PID $(cat "$TRAY_PID_FILE"))"
    else
      echo "Tray:   Detenido"
    fi
    ;;

  doctor)
    # Prueba real de punta a punta sin AC: keys, LLM, Piper, VAD, STT,
    # comandos de voz -> respuesta, wine/SHM/mic. Pasar --speak / --no-llm.
    shift
    exec python3 "$(dirname "$(readlink -f "$0")")/tools/victor_doctor.py" "$@"
    ;;

  *)
    echo "Uso: $0 {tray|start|stop|stop-tray|logs|status|doctor}"
    echo "  tray      — Lanza la app en bandeja del sistema (recomendado)"
    echo "  start     — Solo el daemon (sin GUI)"
    echo "  stop      — Detiene el daemon"
    echo "  stop-tray — Detiene la bandeja"
    echo "  logs      — Ver logs en tiempo real"
    echo "  status    — Estado actual"
    echo "  doctor    — Prueba real de todo el pipeline sin abrir AC (--speak, --no-llm)"
    exit 1
    ;;
esac
