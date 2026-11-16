# Benchmarks de Victor

Dos benchmarks, ambos reproducibles sin Assetto Corsa abierto:

| Benchmark | Qué mide | Necesita |
|---|---|---|
| `bench/run_metrics.py` | Coste del código de Victor: parser de telemetría, spotter, alertas, estrategia de pits, cadena de LLM y turno de voz con proveedores **simulados**, puente por archivo, `pytest` + cobertura | Solo Python (y `pytest-cov` para la cobertura) |
| `bench/llm/bench_llm.py` | El LLM local de verdad (Ollama, solo CPU): latencia, tok/s, RAM, forma de la respuesta y acierto en preguntas de combustible | Ollama con `gemma4:e2b` y `qwen3.5:4b` |

Todos los números de este archivo salen de los JSON en `bench/results/` y
`bench/llm/results/`. Nada está estimado a mano.

## Cómo correrlos

Desde la raíz del repo:

```bash
# 1. Métricas sin juego (~1 min). Escribe bench/results/<AAAA-MM-DD>/metrics.json
python3 bench/run_metrics.py --tests          # --tests añade pytest -q (+ cobertura si hay pytest-cov)
python3 bench/run_metrics.py --quick          # menos iteraciones, solo para probar que corre

# Cobertura sin tocar el Python del sistema:
python3 -m venv --system-site-packages /tmp/victor-cov && /tmp/victor-cov/bin/pip install pytest-cov
/tmp/victor-cov/bin/python bench/run_metrics.py --tests

# 2. LLM local (~6 min con los dos modelos). Escribe bench/llm/results/<AAAA-MM-DD>/
python3 bench/llm/bench_llm.py                # Ollama en 127.0.0.1:11434
OLLAMA_URL=http://127.0.0.1:11436 python3 bench/llm/bench_llm.py   # otro servidor
python3 bench/llm/rescore.py bench/llm/results/2026-11-16          # re-puntúa sin llamar a los modelos

# 3. Suite de pruebas
python3 -m pytest -q
```

No corras el benchmark de LLM con otra carga pesada de CPU encima: mide
inferencia en CPU y la contención se nota directamente en la latencia.

## Hardware y software

| | |
|---|---|
| CPU | 13th Gen Intel Core i7-13620H: 10 núcleos / 16 hilos, 4,9 GHz máx., 24 MiB de L3 (`lscpu`) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB (`nvidia-smi`), **no se usa**: el LLM corre con `num_gpu=0` porque en carrera la GPU es del juego |
| RAM | 31 GB |
| SO | Fedora 44, Linux 7.3.3 |
| Python | 3.14.10, pytest 9.1.1, pytest-cov 7.1.0 (coverage 7.17.3) |
| Ollama | 0.34.2 |

## Modelos (benchmark de LLM)

| Modelo | Parámetros | Cuantización | Papel |
|---|---|---|---|
| `gemma4:e2b` | 5,1B totales (~2B efectivos) | Q4_K_M | El que usa Victor |
| `qwen3.5:4b` | 4,2B densos | Q4_K_M | Candidato descartado |

Llamada idéntica a la de `VictorBrain._call_ollama`: `stream=false`,
`think=false`, `num_gpu=0`, `num_thread=6`, `num_predict=110`,
`temperature=0.3` y el prompt de sistema real de `engineer.py`. Son 12
mensajes de piloto × 2 repeticiones = 24 llamadas medidas por modelo, más 1
de calentamiento que no cuenta. La metodología completa y todas las respuestas
están en `bench/llm/results/2026-11-11/INFORME.md`.

## Resultados

### LLM local, solo CPU (`bench/llm/results/*/summary.json`)

| Métrica | gemma4:e2b · 11-11 | gemma4:e2b · 11-16 | qwen3.5:4b · 11-11 | qwen3.5:4b · 11-16 |
|---|---:|---:|---:|---:|
| Llamadas OK | 24/24 | 24/24 | 24/24 | 24/24 |
| Latencia p50 | 1,85 s | 1,72 s | 9,92 s | 17,16 s |
| Latencia p95 | 3,45 s | 2,77 s | 15,48 s | 20,74 s |
| Decodificación (mediana) | 18,4 tok/s | 20,9 tok/s | 10,6 tok/s | 11,4 tok/s |
| Acierto "llega / no llega" en combustible | 1/6 | 2/6 | 0/6 | 1/6 |
| Respuestas en español | 100 % | 100 % | 100 % | 100 % |
| RAM tras cargar | 3039 MB | 3214 MB | 3684 MB | 4095 MB |

La latencia por repetición explica el salto de qwen entre las dos fechas. La
rep 1 vuelve a evaluar ~1000 tokens de prompt; la rep 2 entra con caché de
prompt, y en carrera cada pregunta es nueva, así que la cifra realista es la
de la rep 1:

| | gemma4:e2b · 11-11 | gemma4:e2b · 11-16 | qwen3.5:4b · 11-11 | qwen3.5:4b · 11-16 |
|---|---:|---:|---:|---:|
| Rep 1 (sin caché), p50 | 1,99 s | 1,76 s | 13,35 s | 19,26 s |
| Rep 1, prefill mediano | 0,47 s | 0,35 s | 8,86 s | 14,41 s |
| Rep 2 (con caché), p50 | 1,59 s | 1,49 s | 4,40 s | 4,82 s |

gemma4:e2b es estable entre corridas. qwen3.5:4b no lo es: su distribución es
bimodal (rep 1 de 13 a 19 s, rep 2 de 4 a 5 s), así que su p50 sobre 24
llamadas cae entre los dos grupos, y su prefill sin caché varió de 8,9 a
14,4 s entre fechas. Hay dos diferencias de entorno conocidas: el 11-11 se usó
un `ollama serve` propio solo-CPU en el puerto 11436, y el 11-16 el servidor
por defecto en el 11434 (ambos con `num_gpu=0`), con el escritorio en uso.
La conclusión no cambia en ninguna de las dos corridas: gemma4:e2b responde
unas 5 a 10 veces más rápido en p50 (de 7 a 11 veces en la rep 1, sin caché).

### Código de Victor sin juego (`bench/results/2026-11-16/metrics.json`)

5000 iteraciones por micro-benchmark y 300 por los de red y puente.

| Camino | p50 | p95 | Nota |
|---|---:|---:|---|
| Parser de telemetría, paquete rápido (6 KB, 23 rivales) | 38 µs | 76 µs | ~23 000 paquetes/s; el lector real escribe a 10 Hz |
| Spotter, por frame (23 rivales) | 8,2 µs | 8,8 µs | |
| Alertas rápidas, por frame | 0,2 µs | 0,7 µs | |
| Alertas proactivas, por llamada | 2,7 µs | 40 µs | Se llaman cada ~6 s |
| Respuesta determinista de pits (`_local_reply("pit")`) | 2,8 µs | 2,9 µs | Sin LLM |
| Cadena de LLM, primer proveedor OK | 0,51 ms | 0,72 ms | Proveedor simulado en loopback |
| Cadena de LLM, failover tras HTTP 500 | 0,94 ms | 1,33 ms | |
| Cadena de LLM, los 6 proveedores caídos | 2,7 ms | 3,4 ms | Devuelve "" y Victor usa la respuesta local |
| Turno de voz, transcripción → respuesta encolada | 0,70 ms | 0,73 ms | STT y LLM simulados |
| Turno de voz con LLM caído (fallback determinista) | 3,0 ms | 3,2 ms | |
| Puente por archivo: `os.replace()` → `engineer._current` | 40,8 ms | 76,7 ms | Máx. 80,4 ms; 0 de 300 paquetes perdidos |

Más resultados del mismo JSON:

- **Spotter, retardo de anuncio (tiempo simulado a 10 Hz):** "Coche derecha" a
  los 300 ms del último frame despejado (200 ms desde el primer frame con
  solape) y "Despejado" a los 1000 ms. Es el debounce de diseño
  (`SPOTTER_CONFIRM_S = 0.15`, `SPOTTER_CLEAR_S = 0.8`) cuantizado a 10 Hz, no
  tiempo de CPU.
- **Combustible, camino determinista:** acierta 3/3 en los mismos casos del
  benchmark de LLM.
- **Circuit breaker:** el proveedor que falla queda pausado tras 3 fallos
  seguidos.
- **Pruebas:** `pytest -q` da 219 passed en ~21 s. La cobertura de
  `src/victor` es del 61,9 % (1525 de 2464 sentencias): `engineer.py` 61,0 %,
  `victor_brain.py` 63,9 %, el resto de módulos entre 93 % y 98 %, y
  `ac_shm_reader.py` y `victor_tray.py` en 0 % (no tienen pruebas en proceso).

## Limitaciones

- **El LLM local es malo con el combustible.** Cuando la pregunta trae los
  números en crudo y hay que razonar "¿llego?", gemma4:e2b acierta la
  conclusión en 1/6 y 2/6 casos, y qwen3.5:4b en 0/6 y 1/6. Son 6 llamadas por
  modelo y corrida: la muestra es mínima y las cifras son indicativas, no
  precisión estadística. Por eso las alertas de combustible, la ventana de pit
  y el fallback cuando no hay LLM son deterministas (3/3). Lo que no está
  medido es el acierto del LLM en carrera: ahí recibe las vueltas de
  combustible ya calculadas en el contexto (`engineer.py`, `~{fuel_laps}
  vueltas`), pero el veredicto de una pregunta libre lo sigue dando el modelo.
- **La puntuación de combustible y de idioma es heurística** (regex sobre la
  respuesta). Todas las respuestas quedan en `results.json` para revisarlas a
  mano.
- **Las métricas de la cadena de LLM y del turno de voz son el coste propio de
  Victor**, con un servidor HTTP local que responde al instante. No incluyen
  red, inferencia en Groq, Cerebras y demás, grabación, VAD, STT real ni la
  síntesis con Piper. No es una latencia de punta a punta "de voz a voz".
- **No hay mediciones con el juego en marcha.** El puente por archivo se mide
  con paquetes sintéticos escritos a 10 Hz en Linux, no con
  `ac_shm_reader.exe` bajo Wine/Proton mientras corre Assetto Corsa.
- **La telemetría es sintética:** las mismas claves que
  `native/ac_shm_reader.c` con una parrilla de 24 coches, no una grabación de
  una carrera real.
- **Una sola máquina.** Los tiempos de CPU dependen del equipo y del escalado
  de frecuencia (por ejemplo, el p95 del parser es el doble que su p50). No
  hay intervalos de confianza: son percentiles de una corrida.
- **La RAM del benchmark de LLM** es el RSS del árbol de procesos de Ollama. El
  11-16 el servidor arrancó con un modelo ya cargado, así que su "reposo"
  (4099 MB) no es comparable con el del 11-11. Las cifras "tras cargar" sí lo
  son.
- **El 0 % de cobertura** de `ac_shm_reader.py` y `victor_tray.py` es real: el
  lector se prueba como `.exe` bajo Wine (pruebas `slow`) y la bandeja no tiene
  pruebas.
