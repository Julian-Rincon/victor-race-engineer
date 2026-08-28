"""Pone src/victor/ en sys.path para que la suite importe los módulos de Victor
(engineer, victor_*) sin instalar el paquete.

Tras la reestructura del 2026-08-28 el código vive en src/victor/ y los tests en
tests/. Los módulos se siguen importando planos (`import victor_brain`), así que
solo hace falta este ajuste de path — ningún import cambió.

Las suites viejas standalone (test_shm_pipeline.py, test_e2e_pipeline.py,
test_features.py, test_spotter_v6.py, test_v6_improvements.py) no cargan conftest
cuando se corren como `python tests/xxx.py`, así que llevan su propio
`sys.path.insert` en la cabecera.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "victor"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
