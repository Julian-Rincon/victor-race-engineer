"""Helpers para los tests que lanzan wine/Proton con los .exe nativos
(test_shm_pipeline, test_e2e_pipeline).

Módulo plano y no de test — se importa con `from _wine_support import ...`
(pytest pone tests/ en sys.path). Va aparte de conftest.py a propósito: conftest
no debe importarse explícitamente.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DIR = REPO_ROOT / "native"

# Wine/Proton del prefix de AC (el mismo que usan los .exe nativos).
WINE_BIN = Path.home() / ".local/share/Steam/compatibilitytools.d/GE-Proton10-34/files/bin/wine"
AC_PREFIX = Path.home() / ".local/share/Steam/steamapps/compatdata/244210/pfx"


def wine_available() -> bool:
    return (WINE_BIN.exists()
            and (NATIVE_DIR / "ac_shm_reader.exe").exists()
            and (NATIVE_DIR / "shm_simulator.exe").exists())


def ac_is_running() -> bool:
    return subprocess.run(
        ["pgrep", "-f", "AssettoCorsa.exe|acs.exe"], capture_output=True
    ).returncode == 0


requires_wine = pytest.mark.skipif(
    not wine_available(),
    reason="wine/Proton GE-Proton10-34 o los .exe nativos no están disponibles",
)


def wine_env() -> dict:
    env = os.environ.copy()
    xauth = sorted(Path(f"/run/user/{os.getuid()}").glob("xauth_*"))
    proton = WINE_BIN.parents[1]  # .../GE-Proton10-34/files/bin/wine -> .../GE-Proton10-34/files
    env.update({
        "WINEPREFIX": str(AC_PREFIX) + "/",
        "WINEDEBUG": "-all",
        "WINENTSYNC": "1",
        "WINEESYNC": "1",
        "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
        "XAUTHORITY": str(xauth[0]) if xauth else "",
        "WINEDLLPATH": f"{proton}/lib/vkd3d:{proton}/lib/wine",
        "LD_LIBRARY_PATH": f"{proton}/lib",
    })
    env.setdefault("DISPLAY", ":0")
    return env
