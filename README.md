<p align="center">
  <img src="assets/victor_logo.png" alt="Victor" width="220">
</p>

<h1 align="center">Victor — AI Race Engineer para Assetto Corsa</h1>

<p align="center">
  Un ingeniero de carrera con voz, que escucha telemetría real y responde en
  español, corriendo 100% nativo en Linux.
</p>

<p align="center">
  <a href="https://github.com/Julian-Rincon/victor-race-engineer/actions/workflows/ci.yml">
    <img src="https://github.com/Julian-Rincon/victor-race-engineer/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

---

## ¿Qué es esto?

Victor es un asistente de voz para [Assetto Corsa](https://assettocorsa.gg/)
pensado como alternativa a herramientas como [CrewChief](https://gitlab.com/mr_belowski/CrewChief)
o [Crew Chief V4](https://thecrewchief.org/), pero:

- **Nativo en Linux** — no necesita Windows Speech Recognition ni correr por
  fuera del prefix de Wine/Proton donde vive el juego.
- **Habla español** de verdad (no un locutor pregrabado con huecos rellenados
  a la fuerza) — usa un LLM en la nube para redactar cada respuesta en
  contexto.
- **Escucha y responde preguntas libres**, no solo comandos de una gramática
  fija — "¿cómo van los neumáticos?" y "dame el estado de las gomas"
  funcionan igual.
- **Lee la memoria compartida real de AC** (misma fuente que usa CrewChief),
  así que sabe combustible, temperaturas, daños, posición, sectores y rivales
  con la misma fidelidad.

No es un proyecto de una empresa — es una herramienta personal que comparto
por si le sirve a alguien más de la comunidad de sim racing en Linux.

## Qué hace

- **Spotter determinístico** (sin LLM, <150ms) — avisa coches al lado,
  "sigue ahí", "despejado", con la misma lógica de zonas que usa CrewChief.
- **Alertas proactivas** — combustible bajo (aviso <3.5 vueltas, crítico
  <1.8), temperatura de neumáticos (aviso >95°C, crítico >105°C), desgaste,
  bandera azul, rival que se acerca, ventana de pit.
- **Resumen de vuelta** — tiempo, consumo, ¿vuelta rápida personal?, delta
  contra el mejor tiempo por sectores.
- **Estrategia de pits** — cálculo de vueltas de combustible restantes y
  recomendación de cuándo entrar.
- **Memoria del piloto** — recuerda tu mejor vuelta y tus zonas débiles
  recurrentes por combinación pista+auto entre sesiones, y las menciona en
  el briefing de la siguiente.
- **Comandos de voz y consultas libres** — "cállate", "repite", "modo gap",
  o cualquier pregunta abierta sobre la carrera; todo pasa por un LLM con el
  contexto completo de la telemetría actual.
- **Wake word** — nada se ejecuta si no empezás diciendo "Victor" primero
  (tolerante a que la transcripción lo escuche mal). Evita que música o
  conversación de fondo dispare respuestas sin que nadie le hable.
- **Cerebro multi-backend con failover** — Groq → Cerebras → NVIDIA NIM →
  Gemini → Mistral → OpenRouter (solo modelos `:free`) → LLM local en CPU
  (Ollama, opcional), + un backend propio OpenAI-compatible; circuit breaker
  y salto automático al siguiente. Todos tienen capa gratuita.
- **LLM local sin tocar la VRAM** — Ollama con `num_gpu=0` como último
  recurso, solo para preguntas abiertas, setup, briefing y debrief (los datos
  concretos van por respuesta determinista: más rápida y exacta). Victor
  levanta `ollama serve` sin GPU si hace falta y lo apaga al salir.
- **Autocuración de modelos** — si un proveedor retira el modelo configurado
  (pasó con `llama-3.3-70b-versatile` y `gemini-2.0-flash` en 2026), Victor
  consulta `/models` y cambia solo a uno vivo. Key inválida o sin crédito
  (401/402/403) aparta ese backend toda la sesión; 429 respeta `Retry-After`.
- **Funciona sin red** — STT local con faster-whisper en CPU (0 VRAM, ~1 s por
  comando) si la nube no responde, y respuestas deterministas desde la
  telemetría (combustible, gaps, neumáticos, daños, posición, sectores) si no
  hay LLM: Victor nunca se queda callado.
- **Audio ducking nativo (PipeWire)** — mientras Victor habla baja solo el
  volumen de `acs.exe` y lo restaura exacto al terminar (no toca el volumen
  maestro, ni EasyEffects, ni otras apps).
- **Voz que se entiende** — números y símbolos se reescriben antes de Piper
  ("P4" → "cuarto", "104°C" → "104 grados", "+0.32s" → "más 0 coma 32
  segundos", "1:42.345" → "1 minuto 42 con 345").

## Cómo funciona (arquitectura)

```
Assetto Corsa (Wine/Proton)
   │  Shared Memory de Windows (acpmf_physics / graphics / static / crewchief)
   ▼
native/ac_shm_reader.exe   (C, compilado con MinGW, corre con el wine del prefix de AC)
   │  escribe JSON cada 100ms junto al propio .exe (native/)
   ▼
native/ac_telemetry_fast.json / native/ac_telemetry_slow.json
   │
   ▼
src/victor/engineer.py   (daemon Linux — nada de esto corre bajo Wine)
   ├─ victor_ears.py    → VAD real (Silero, ONNX) + gate de grabación
   ├─ victor_wake.py    → exige "Victor" antes de aceptar un comando
   ├─ victor_brain.py   → LLM multi-backend (Groq/Cerebras/NVIDIA/Gemini)
   └─ victor_memory.py  → memoria del piloto entre sesiones (JSON local)
```

## Estructura del repo

```
src/victor/   código del daemon (engineer.py) y módulos victor_*
tests/        suite pytest completa — unit + bancos SHM/e2e (marcados slow)
native/       fuentes C del lector SHM (ac_shm_reader.c, shm_simulator.c) y sus .exe
assets/       íconos y logo
data/ models/ voices/   assets no versionados (calibración, Silero VAD, voz Piper)
start.sh      launcher
```

La misma memoria compartida (`acpmf_*`) que usa CrewChief es la fuente de
datos — por eso Victor puede correr en paralelo a CrewChief sin pisarlo, o
reemplazarlo directamente.

## Requisitos

- Linux, con Assetto Corsa corriendo vía Steam + Proton (probado con
  GE-Proton, appid `244210`).
- Python 3.10+.
- `arecord` (paquete `alsa-utils`) para capturar el micrófono.
- Opcional: API key de alguno de estos proveedores (todos tienen capa
  gratuita): [Groq](https://console.groq.com/), [Cerebras](https://cloud.cerebras.ai/),
  [NVIDIA NIM](https://build.nvidia.com/), [Gemini](https://aistudio.google.com/),
  [Mistral](https://console.mistral.ai/), [OpenRouter](https://openrouter.ai/).
  Cuantos más configures, más resiliente es el failover.
- `mingw-w64-gcc` si querés recompilar `ac_shm_reader.exe` (el binario
  compilado no se versiona — ver más abajo).

## Instalación

```bash
git clone git@github.com:Julian-Rincon/victor-race-engineer.git
cd victor-race-engineer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Assets de voz y VAD (no van en el repo)

Dos modelos no están versionados (pesan varios MB y son de terceros,
descargables libremente):

1. **Voz Piper** (TTS en español) — cualquier voz de
   [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/es)
   sirve. Colocar el `.onnx` en `voices/<nombre>/model.onnx` (en la raíz del
   repo) y ajustar `PIPER_MODEL_PATH` en `src/victor/victor_config.py` si el
   nombre no coincide.
2. **Silero VAD v5** — descargar `silero_vad.onnx` de
   [snakers4/silero-vad](https://github.com/snakers4/silero-vad) y colocarlo
   en `models/silero_vad.onnx` (en la raíz del repo).

### Compilar el lector de memoria compartida

Las fuentes C y el binario viven en `native/`:

```bash
x86_64-w64-mingw32-gcc -O2 -o native/ac_shm_reader.exe native/ac_shm_reader.c
```

Para los tests del pipeline SHM sin AC también hace falta el simulador:

```bash
x86_64-w64-mingw32-gcc -O2 -o native/shm_simulator.exe native/shm_simulator.c
```

Este `.exe` corre con el **mismo Wine/Proton que usa Assetto Corsa** — lee la
memoria compartida de Windows que el juego expone y no se puede reemplazar
por un lector nativo de Linux. Escribe los JSON de telemetría en `native/`,
junto al propio binario.

### Configuración (API keys)

Copiá `.env.example` a `~/.config/victor/victor.env` (recomendado) o a
`.env` en la raíz (ignorado por git) y completá las keys que tengas. Orden de
búsqueda: entorno del proceso → `$VICTOR_ENV_FILE` → `<repo>/.env` →
`~/.config/victor/victor.env` → el `.env` legado de nexus-core. Se aceptan
nombres cortos (`GROQ_API_KEY`) o con prefijo `NEXUS_`.

Ninguna es obligatoria: sin keys Victor arranca en **modo local** (STT en CPU
+ respuestas sin LLM) y lo avisa en el log. `bash start.sh doctor` muestra qué
se cargó y prueba cada backend en vivo.

## Uso

```bash
bash start.sh start      # daemon + lector SHM, sin interfaz
bash start.sh tray        # lo mismo, más un ícono en la bandeja del sistema
bash start.sh stop        # detiene todo
bash start.sh status      # ¿qué está corriendo?
bash start.sh logs        # tail -f del log del daemon
bash start.sh shm-log     # tail -f del log del lector de memoria compartida
bash start.sh doctor      # prueba real de todo el pipeline sin abrir AC
```

`doctor` sintetiza con Piper comandos reales ("Victor, cuánto combustible me
queda", "Victor, modo gap", una frase sin wake word…), los pasa por Silero
VAD y el STT real, los manda al router de comandos con telemetría simulada de
carrera y verifica la respuesta. También hace ping a cada LLM con key y
revisa wine, el lector SHM y el micrófono. `--speak` reproduce las
respuestas; `--no-llm` no gasta llamadas a APIs.

Arrancalo **antes** de entrar a pista (no hace falta ningún widget dentro de
AC — el lector de memoria compartida se conecta solo apenas el juego expone
sus datos).

Una vez corriendo, hablale empezando con **"Victor"**:

> "Victor, ¿cómo vamos de combustible?"
> "Victor, cállate" / "Victor, repite" / "Victor, modo gap"

Las alertas proactivas (combustible, neumáticos, banderas, spotter) no
necesitan que le hables — hablan solas cuando corresponde.

## Tests

Toda la suite corre con `pytest` a secas:

```bash
python3 -m pytest                 # todo (incluye los bancos SHM/e2e bajo wine)
python3 -m pytest -m "not slow"    # rápido — omite los que lanzan wine/Proton
```

`tests/conftest.py` pone `src/victor/` en el path y expone la fixture `eng`
(importa `engineer` con las deps de audio stubbeadas). Los tests marcados
`slow` (`test_shm_pipeline.py`, `test_e2e_pipeline.py`) lanzan
`shm_simulator.exe` + `ac_shm_reader.exe` bajo el Wine de Proton para validar
el pipeline de memoria compartida sin AC; se saltan solos si Wine o los `.exe`
no están, o si AC está corriendo. Compilá los `.exe` (ver arriba) para
ejecutarlos.

### `test_v7_ears` — 10 skips esperados sin los modelos

`test_v7_ears.py` ejercita el VAD real: sintetiza voz con Piper y la pasa por
Silero VAD (ONNX), sin mocks. Necesita los dos assets de terceros que **no van
en el repo** (ver "Assets de voz y VAD" arriba): `voices/es_davefx_medium/model.onnx`
y `models/silero_vad.onnx`, más `onnxruntime` y `piper-tts` instalados.

Si esos archivos no están, los 10 tests se marcan `skipped` (no `failed`) — es
el comportamiento esperado, y es lo que pasa en CI (el runner no descarga
modelos). Para correrlos localmente:

```bash
pip install onnxruntime piper-tts          # además de pytest numpy scipy requests
# colocá los .onnx en voices/ y models/ (enlaces en "Assets de voz y VAD")
python3 -m pytest tests/test_v7_ears.py -v  # 10 passed
```

## Estado del proyecto

Desarrollo activo, uso personal diario en carreras propias. La arquitectura
(spotter, memoria compartida, estrategia de pits) está directamente
inspirada e investigada a partir de CrewChief V4 — gracias a ese proyecto por
años de código abierto de referencia en este espacio.

## Licencia

[MIT](LICENSE).
