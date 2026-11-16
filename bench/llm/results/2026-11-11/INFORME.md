# Benchmark del LLM local de Victor — `gemma4:e2b` vs `qwen3.5:4b`

**Fecha:** 2026-11-11 22:26 · **Equipo:** 13th Gen Intel(R) Core(TM) i7-13620H · 31 GB RAM · 16 hilos
**Servidor:** Ollama propio solo-CPU en `127.0.0.1:11436`.
**Modelos:** `gemma4:e2b` (actual) y `qwen3.5:4b` (candidato), ambos ya descargados.

## 1. Metodología

Se reproduce **exactamente** la llamada de `src/victor/victor_brain.py::_call_ollama`
(payload idéntico, mismo prompt de sistema real de `engineer.py::_SYSTEM`, incluido el
bloque `CONTROLLER MODE` porque `CONTROLLER_MODE = True`):

```json
{
  "model": "<modelo>", "stream": false, "think": false, "keep_alive": "15m",
  "messages": [{"role": "system", "content": "<_SYSTEM de Victor>"}, {"role": "user", "content": "<mensaje>"}],
  "options": {"num_gpu": 0, "num_thread": 6, "num_predict": 110, "temperature": 0.3}
}
```

- **12 mensajes realistas de piloto** en español: 3 de combustible con datos concretos
  (verdad calculada por el script), 3 de estrategia/neumáticos/gaps, 1 charla corta,
  1 con errores de transcripción de voz, 1 petición en inglés, y 3 de pilotaje/tráfico/incidente.
- **2 repeticiones por mensaje y modelo** (24 llamadas medidas por modelo).
- **1 llamada de calentamiento por modelo**, no contada. El modelo anterior se descarga
  (`keep_alive: 0`) antes de medir el siguiente para que la RAM sea comparable.
- Cada mensaje se envía **independiente** (sin historial), como la mayoría de preguntas
  sueltas que Victor resuelve por radio.
- Métricas automáticas: latencia de pared, `eval_count/eval_duration` (tok/s), longitud,
  respuesta vacía, fuga de razonamiento (`thinking`, `Thinking Process`, `The user asks…`),
  idioma por heurística de vocabulario/acentos, brevedad y RSS del árbol de procesos del
  Ollama propio durante cada llamada.
- La corrección del combustible se puntúa por la **conclusión** (llega / no llega) con
  patrones de texto; el margen (litros/vueltas) se comprueba aparte. Todas las respuestas
  quedan transcritas abajo para revisión humana.

> **Espera:** el benchmark no arrancó hasta que `pgrep -f "[g]en_k.py"` y
> `pgrep -f "[m]etricas_sin_llm"` dejaron de devolver procesos (otra prueba usaba GPU/CPU).
> Además, al empezar, `qwen3.5:4b` aún se estaba descargando por parte de esa otra prueba:
> el script esperó a que su manifiesto y sus blobs estuvieran completos antes de medir.

### Un detalle importante: la segunda repetición no es gratis

Ollama 0.34 reutiliza el caché de prompt **solo cuando el prompt es idéntico**. La rep 1 de
cada pregunta vuelve a evaluar ~950-1100 tokens (prefill); la rep 2, con el mismo texto,
entra con caché (~0,16 s). En carrera **cada pregunta es nueva**, así que el número
realista es el de la **rep 1**; la tabla resumen da ambos para no maquillar nada.

### Verdad de los casos de combustible

| Caso | Litros | L/vuelta | Vueltas | Necesita | Resultado |
|---|---:|---:|---:|---:|---|
| fuel_1 | 18,4 | 2,3 | 9 | 20,7 L | NO LLEGA — faltan 2,3 L = 1,00 vueltas |
| fuel_2 | 22,0 | 2,6 | 8 | 20,8 L | LLEGA — sobran 1,2 L = 0,46 vueltas |
| fuel_3 | 9,6 | 2,1 | 5 | 10,5 L | NO LLEGA — faltan 0,9 L = 0,43 vueltas |

## 2. Entorno y modelos

| Modelo | Parámetros | Cuantización | Disco | RAM tras cargar | Decodificación |
|---|---|---:|---:|---:|---:|
| `gemma4:e2b` | 5,1B totales (~2B efectivos) | Q4_K_M | 7,16 GB | 3039 MB | 18,4 tok/s |
| `qwen3.5:4b` | 4,2B densos | Q4_K_M | 3,32 GB | 3684 MB | 10,6 tok/s |

- Ollama 0.34.2, `num_gpu=0`: **cero VRAM**; la GPU queda para Assetto Corsa.
  Arrancado con `OLLAMA_HOST=127.0.0.1:11436 CUDA_VISIBLE_DEVICES="" setsid nohup ollama serve`.
- `num_thread=6` como en producción. `num_predict=110`, `temperature=0.3`, `think=false`.
- RAM medida = RSS del `ollama serve` propio **más** su proceso hijo `llama-server`.

## 3. Tabla resumen

| Métrica | gemma4:e2b | qwen3.5:4b |
|---|---:|---:|
| Llamadas medidas (por modelo) | 24 | 24 |
| Latencia p50 — todas (s) | 1,85 | 9,92 |
| Latencia p95 — todas (s) | 3,45 | 15,48 |
| Latencia p50 — pregunta nueva (s) | 2,11 | 13,64 |
| Latencia p95 — pregunta nueva (s) | 3,45 | 15,48 |
| Latencia media (s) | 2,00 | 9,11 |
| Velocidad mediana (tok/s) | 18,4 | 10,6 |
| Acierto conclusión combustible | 1/6 (17%) | 0/6 (0%) |
| Respuestas vacías | 0% | 0% |
| Fuga de razonamiento / meta | 0% | 0% |
| Respondió en español | 100% | 100% |
| Breve (≤3 frases y ≤60 palabras) | 71% | 92% |
| Longitud mediana | 17 palabras / 98 car. | 30 palabras / 172 car. |
| RAM base tras cargar (MB) | 3039 | 3684 |
| RAM mediana por llamada (MB) | 3048 | 3837 |
| RAM pico (MB) | 3056 | 3837 |
| Calentamiento / carga + 1.ª respuesta (s) | 15,55 | 33,36 |

**Lectura rápida:** aciertos de combustible 1/6 (gemma4:e2b) frente a
0/6 (qwen3.5:4b). Ninguno de los dos es fiable para la cuenta de
combustible: qwen falla las tres (en una incluso se niega a calcular y pide datos que no
hacen falta para la cuenta), y además se inventa datos que no están en el mensaje en otras
preguntas (ver §7: tiempos de vuelta en la pregunta en inglés, proyecciones de ritmo en gaps).

### 3.1 Latencia: el dato que decide en carrera

| Métrica | gemma4:e2b | qwen3.5:4b |
|---|---:|---:|
| Rep 1 — prompt nuevo: p50 / p95 | 2,11 / 3,45 s | 13,64 / 15,48 s |
| Rep 2 — prompt idéntico (caché): p50 / p95 | 1,67 / 2,67 s | 4,54 / 7,19 s |
| Tiempo de prefill mediano, rep 1 (s) | 0,47 | 8,86 |
| Tiempo de prefill mediano, rep 2 (s) | 0,06 | 0,17 |
| Decodificación mediana (s) | 1,52 | 4,44 |
| Tiempo hasta el 1.er token, rep 1 (mediana, s) | 0,48 | 8,87 |

La rep 2 parece rápida en qwen porque el prompt es idéntico y entra por caché. La rep 1
—prompt nuevo, el caso real— tarda **~13,6 s de mediana** frente a **~2,1 s** de gemma.
En qwen, dos tercios de ese tiempo son *prefill* (8,9 s para ~950 tokens, ~107 tok/s);
gemma evalúa los mismos ~950 tokens en 0,5 s (~2.000 tok/s) y el resto es generación.

## 4. Calidad de forma (radio)

| Comportamiento | gemma4:e2b | qwen3.5:4b |
|---|---:|---:|
| Empieza la frase con la etiqueta «Normal.» | 15/24 | 0/24 |
| Cumple «máximo 2 frases» (≤2) | 0/24 | 17/24 |
| ≤3 frases | 17/24 | 23/24 |
| Respuesta más larga (palabras) | 34 | 62 |

Gemma arrastra la etiqueta literal **«Normal.»** al inicio en 15 de 24 respuestas (el
prompt de sistema lista los modos y el modelo los verbaliza); qwen no lo hace. A cambio,
qwen es más largo (30 palabras de mediana frente a 17) y llega a 62 palabras, lo que en
TTS son varios segundos más después de la latencia.

El conteo de frases no rompe por puntos decimales («3.2», «101.482»), pero sí cuenta la
etiqueta «Normal.» como frase: sin ella, gemma cumple «máximo 2 frases» en 17 de 24.

## 5. Latencia por categoría

| Categoría | gemma p50 / p95 (s) | qwen p50 / p95 (s) | gemma tok/s | qwen tok/s |
|---|---:|---:|---:|---:|
| combustible | 1,60 / 2,56 | 7,92 / 15,48 | 19,9 | 10,2 |
| neumáticos | 1,95 / 3,45 | 4,12 / 13,64 | 17,6 | 10,6 |
| estrategia | 1,29 / 1,85 | 4,54 / 12,45 | 18,2 | 10,7 |
| gaps | 3,22 / 4,45 | 3,91 / 15,89 | 18,1 | 10,7 |
| charla | 1,25 / 1,51 | 5,92 / 15,20 | 18,4 | 10,4 |
| transcripción | 1,87 / 2,67 | 6,50 / 12,39 | 18,4 | 10,6 |
| inglés | 1,51 / 1,60 | 4,25 / 13,66 | 18,0 | 10,6 |
| pilotaje | 1,67 / 2,19 | 3,50 / 9,92 | 18,5 | 10,8 |
| tráfico | 0,92 / 1,45 | 5,64 / 11,53 | 18,7 | 9,5 |
| incidente | 1,84 / 2,14 | 3,12 / 10,17 | 14,5 | 11,1 |

## 6. Combustible: aciertos y errores

| Caso | Datos del piloto | Cálculo correcto | gemma4:e2b | qwen3.5:4b |
|---|---|---|---|---|
| fuel_1 | «Julián: me quedan 18,4 litros, gasto 2,3 por vuelta y faltan 9 vueltas. ¿Llego al final?» | NO LLEGA: faltan 2,3 L (1,00 vueltas) | ❌ llega<br>❌ llega | ❌ llega<br>❌ llega |
| fuel_2 | «Tengo 22 litros, el consumo es 2,6 por vuelta y quedan 8 vueltas. ¿Llego o tengo que levantar?» | LLEGA: sobran 1,2 L (0,46 vueltas) | ❌ sin conclusión<br>✅ llega | ❌ no llega<br>❌ sin conclusión |
| fuel_3 | «Me quedan 9,6 litros, gasto 2,1 por vuelta y faltan 5 vueltas. ¿Aguanto hasta el final?» | NO LLEGA: faltan 0,9 L (0,43 vueltas) | ❌ llega<br>❌ llega | ❌ llega<br>❌ llega |

Criterio de acierto: la conclusión (llega / no llega) debe ser correcta. El margen
(litros o vueltas) se comprueba aparte y se ve en la tabla de detalle.

## 7. Todas las respuestas, lado a lado

### Combustible 1 — 18,4 L, 2,3 L/v, faltan 9 vueltas

**Piloto:** «Julián: me quedan 18,4 litros, gasto 2,3 por vuelta y faltan 9 vueltas. ¿Llego al final?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Con esos datos, si mantienes el ritmo actual, llegas al final. Gestiona bien el combustible.»<br>_1,60 s · 21,4 tok/s · 16 pal. · español · 3 frases_ | «Sí, Julián, con ese consumo tienes un margen de seguridad para llegar al final sin pitar.<br>Mantén la consistencia en los siguientes stints y no te arriesgues a forzar el motor.»<br>_15,48 s · 10,0 tok/s · 31 pal. · español · 2 frases_ |
| 2 | «Normal. Si mantienes ese ritmo de gasto, sí llegas al final. Gestiona bien las últimas vueltas.»<br>_1,30 s · 20,2 tok/s · 16 pal. · español · 3 frases_ | «Sí, Julián, con ese consumo te sobran casi dos vueltas de margen. Mantén la consistencia y no busques el límite ahora mismo.»<br>_3,42 s · 10,2 tok/s · 22 pal. · español · 2 frases_ |

### Combustible 2 — 22 L, 2,6 L/v, quedan 8 vueltas

**Piloto:** «Tengo 22 litros, el consumo es 2,6 por vuelta y quedan 8 vueltas. ¿Llego o tengo que levantar?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Con 22 litros y 8 vueltas restantes, el cálculo es estable. Sigue gestionando la gestión de gomas y el tráfico.»<br>_2,11 s · 19,7 tok/s · 21 pal. · español · 3 frases_ | «Con esos números no llegas a la meta con margen seguro, Julián. Levanta en esta vuelta para asegurar tu posición.»<br>_13,05 s · 10,3 tok/s · 20 pal. · español · 2 frases_ |
| 2 | «Normal. Con 22 litros y 8 vueltas, tienes suficiente para terminar la carrera. Sigue gestionando bien el consumo.»<br>_1,50 s · 20,2 tok/s · 18 pal. · español · 3 frases_ | «Con esos datos no puedo decirte si llegas, Julián. Necesito saber la posición actual en pista y los tiempos de pit de tu rival para calcular el margen real. Sin esos datos, solo puedo decirte que con 2.6 por vuelta te quedan exactamente 20.8 litros para las 8 vueltas restantes.»<br>_7,19 s · 10,0 tok/s · 50 pal. · español · 3 frases_ |

### Combustible 3 — 9,6 L, 2,1 L/v, faltan 5 vueltas

**Piloto:** «Me quedan 9,6 litros, gasto 2,1 por vuelta y faltan 5 vueltas. ¿Aguanto hasta el final?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Con esos 9.6 litros y ese consumo, tienes para las cinco vueltas restantes. Gestiona la entrada a las curvas con suavidad y mantén la línea estable.»<br>_2,56 s · 18,6 tok/s · 27 pal. · español · 3 frases_ | «Sí, Julián, con ese consumo te sobran casi dos vueltas de margen. Mantén la suavidad en los aceleradores para no gastar más, y si ves tráfico cerca de boxes, considera salir antes aunque tengas un poco de combustible extra.»<br>_15,15 s · 10,3 tok/s · 39 pal. · español · 2 frases_ |
| 2 | «Julián, con esos 9.6 litros y ese consumo, tienes para cinco vueltas más. Gestiona bien el combustible y el tráfico. Sigue el plan.»<br>_2,14 s · 17,9 tok/s · 23 pal. · español · 3 frases_ | «Sí, Julián, con ese consumo te sobra un poco para las cinco vueltas restantes sin apretar demasiado. Mantén la suavidad en los inputs, especialmente al salir de curva, y no busques límites extremos porque el stick no da tanta resolución como el volante. Si sientes que vas a llegar justo al final, puedes dejar una décima más de margen para ser seguro.»<br>_7,92 s · 10,6 tok/s · 62 pal. · español · 3 frases_ |

### Neumáticos — medias al 38%, rival a 1,8 s

**Piloto:** «Estoy en la vuelta 14, gomas medias con 38% de vida, el de atrás viene a 1,8 y cerrando. ¿Aguanto o entro a cambiar?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Julián, el de atrás viene a 1.8 y está cerrando. Mantén la línea, gestiona la energía. Aguanta hasta la siguiente, vamos a ver cómo se comporta la gestión de gomas.»<br>_3,45 s · 17,2 tok/s · 31 pal. · español · 4 frases_ | «Aguanta, Julián: tienes un margen de seguridad suficiente para mantener el ritmo sin perder posición. Si entras ahora, te quedas en la cola y pierdes tiempo valioso con gomas nuevas.»<br>_13,64 s · 10,7 tok/s · 30 pal. · español · 2 frases_ |
| 2 | «Normal. Julián, el de atrás está cerrando. Mantén la línea y gestiona la energía. Aguanta hasta la siguiente, revisa el tráfico.»<br>_1,95 s · 18,1 tok/s · 21 pal. · español · 4 frases_ | «Julián, aguantamos esta vuelta pero frena progresivo y temprano para no perder tracción en la salida. Si el rival entra a 2.0, cambiamos ahora mismo.»<br>_4,12 s · 10,6 tok/s · 25 pal. · español · 2 frases_ |

### Estrategia — safety car, 22 min, gomas nuevas

**Piloto:** «Quedan 22 minutos de carrera, safety car en pista y acabo de poner gomas nuevas. ¿Paro ahora o espero?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Espera, Julián. El Safety Car es la prioridad ahora. Deja que el tráfico se organice.»<br>_1,85 s · 17,6 tok/s · 15 pal. · español · 3 frases_ | «Espera a que salga el Safety Car, Julián — no pares ahora para no perder posición. Revisa la distancia al coche de atrás antes de decidir entrar.»<br>_12,45 s · 10,7 tok/s · 27 pal. · español · 2 frases_ |
| 2 | «Espera, Julián. El Safety Car es la prioridad ahora. Deja que el tráfico se organice.»<br>_1,29 s · 18,8 tok/s · 15 pal. · español · 3 frases_ | «Espera la señal del Safety Car, Julián — no pares hasta que se levante el coche.<br>Con las gomas nuevas y tiempo de carrera, es mejor aguantar para sacar más vueltas limpias.»<br>_4,54 s · 10,7 tok/s · 32 pal. · español · 2 frases_ |

### Gaps — contexto Monza, rivales a 3,2 / 5,7

**Piloto:** «[RACE][ks_monza][GT3] segundo lugar, a un puesto del liderato \| V18 \| 268km/h<br>Comb: 26.0L \| Neumáticos (M): FL92 FR94 RL88 RR91°C (max94) vida_gomas_min=38% (100%=nuevas)<br>Gaps: Pérez a 3.2s, Rossi a 5.7s detrás<br>Última: 101.482s \| Mejor: 101.204s \| ΔS: S1+120ms S2+95ms S3+63ms<br><br>¿Cómo van los gaps? El segundo está a 3,2 y el tercero a 5,7. Quedan 12 vueltas.»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Julián, los gaps están manejables. El de adelante está a tres punto dos, el tercero a cinco punto siete. Mantén la consistencia en la gestión de la energía.»<br>_4,45 s · 18,1 tok/s · 28 pal. · español · 3 frases_ | «Julián, mantén la consistencia: Pérez a 3,2 segundos y Rossi a 5,7. Con 12 vueltas quedan, necesitamos cerrar esos huecos sin apurar las gomas al máximo.»<br>_15,89 s · 10,7 tok/s · 26 pal. · español · 2 frases_ |
| 2 | «Vuelta 101.482. Gaps a Pérez a 3.2 segundos, a Rossi a 5.7 segundos. Quedan 12 vueltas. Mantén la consistencia, Julián. Sigue gestionando la energía.»<br>_3,22 s · 18,1 tok/s · 24 pal. · español · 5 frases_ | «Julián, mantén ese ritmo de 101 segundos; con dos vueltas más puedes cerrar esos tres segundos sobre Pérez sin tocar los límites. Consistente es la clave ahora.»<br>_3,91 s · 10,7 tok/s · 27 pal. · español · 2 frases_ |

### Charla — «¿Cómo vamos, Victor?»

**Piloto:** «[RACE][ks_monza][GT3] segundo lugar, a un puesto del liderato \| V18 \| 268km/h<br>Comb: 26.0L \| Neumáticos (M): FL92 FR94 RL88 RR91°C (max94) vida_gomas_min=38% (100%=nuevas)<br>Gaps: Pérez a 3.2s, Rossi a 5.7s detrás<br>Última: 101.482s \| Mejor: 101.204s \| ΔS: S1+120ms S2+95ms S3+63ms<br><br>¿Cómo vamos, Victor?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Julián, mantén la consistencia en la gestión de las gomas. El ritmo es bueno.»<br>_1,51 s · 18,0 tok/s · 15 pal. · español · 3 frases_ | «Vuelta 18, segundo lugar, Pérez a tres segundos y Rossi a cinco. Tienes el ritmo correcto pero las gomas están bajando rápido, mantén la suavidad en los inputs para no perder tracción. Box en esta vuelta si necesitas alargar más, o sigue así si quieres intentar por el liderato.»<br>_15,20 s · 10,2 tok/s · 49 pal. · español · 3 frases_ |
| 2 | «Normal. Julián, mantén la línea. El ritmo es bueno, no te ahogues.»<br>_1,25 s · 18,7 tok/s · 12 pal. · español · 3 frases_ | «Vuelta 18 en Monza, Julián. Consistente con ese ritmo, pero vigila los neumáticos: FL92 y FR94 están cerca del límite térmico. Pérez a 3.2 segundos, mantén esa distancia para no forzar la salida de curva.»<br>_5,92 s · 10,6 tok/s · 35 pal. · español · 3 frases_ |

### Transcripción con errores de voz

**Piloto:** «Oye Victo, ¿me pones el compuesto duro o sigo con el blando? Eh... el de atrá me está metiendo presión, creo que es el núme diez.»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Revisa la temperatura de los neumáticos. Si sientes presión, es mejor bajar un poco el valor.»<br>_1,87 s · 18,6 tok/s · 17 pal. · español · 3 frases_ | «Sigue con el blando, Julián — no cambies ahora, el de atrás te está empujando y necesitas tracción para salir. Mantén la línea de salida suave sin forzar el acelerador, deja que el coche respire antes de pensar en cambiar.»<br>_12,39 s · 10,5 tok/s · 40 pal. · español · 2 frases_ |
| 2 | «Normal. El compuesto duro te dará más agarre en las curvas lentas y la gestión de gomas será más predecible. Sigue con el blando si tienes confianza, pero si sientes presión, cambia a duro.»<br>_2,67 s · 18,1 tok/s · 34 pal. · español · 3 frases_ | «Sigue con el blando, Julián — no lo cambies ahora para evitar la pérdida de tracción en salida. El de adelante te empuja, pero si vas duro perderás control y te deslizarás en curva; mantén el stick suave y frena progresivo y temprano para conservar adherencia.»<br>_6,50 s · 10,6 tok/s · 46 pal. · español · 2 frases_ |

### Petición en inglés (debe responder en español)

**Piloto:** «Victor, what's the gap to the car ahead and should I push now or manage the tyres?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. ¿Qué distancia tienes con el coche de delante? Evalúa la gestión de gomas.»<br>_1,60 s · 17,4 tok/s · 14 pal. · español · 3 frases_ | «Vuelta 8, un cuarenta y dos seis. Mantén la consistencia, Julián — no empieces a forzar todavía. Las gomas están frescas pero el ritmo de los rivales está subiendo; espera a ver cómo reacciona el coche en la siguiente curva larga antes de decidir si pones presión o conservas para el final.»<br>_13,66 s · 10,6 tok/s · 52 pal. · español · 3 frases_ |
| 2 | «Normal. Necesito ver el dato del rival. Julián, mantén la línea y gestiona la entrada a la curva.»<br>_1,51 s · 18,7 tok/s · 18 pal. · español · 3 frases_ | «Vuelta 8, uno cuarenta y dos seis. Consistente, sigue así. Julián, revisa la temperatura de las gomas antes de decidir si empujar o guardar neumáticos.»<br>_4,25 s · 10,6 tok/s · 25 pal. · español · 3 frases_ |

### Pilotaje — morro en curva 1, mando

**Piloto:** «En la frenada de la curva 1 se me va el morro y pierdo como dos décimas por vuelta. ¿Qué hago?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «¡Hostia! Eso no es bueno. Julián, frena progresivo y temprano en la curva uno. Suavidad en el input.»<br>_2,19 s · 18,3 tok/s · 18 pal. · español · 4 frases_ | «Frena progresivo y temprano desde la entrada, mantén el stick estable para recuperar tracción sin tirar del volante. Revisa si hay tráfico cerca antes de salir a acelerar.»<br>_9,92 s · 10,8 tok/s · 28 pal. · español · 2 frases_ |
| 2 | «¡Hostia! Eso no es bueno. Julián, frena progresivo y temprano en la curva. Suavidad en el input.»<br>_1,67 s · 18,7 tok/s · 17 pal. · español · 4 frases_ | «Frena progresivo y temprano en la entrada, sube suave para recuperar adherencia sin perder tracción. Revisa si el stick está bien centrado antes de salir.»<br>_3,50 s · 10,8 tok/s · 25 pal. · español · 2 frases_ |

### Tráfico — LMP2 detrás

**Piloto:** «Tengo un LMP2 detrás y me saca dos segundos por vuelta, ¿le dejo pasar ahora o espero a la recta?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «Normal. Mantén la línea. Espera a la recta para gestionar el ritmo.»<br>_1,45 s · 18,5 tok/s · 12 pal. · español · 3 frases_ | «Espera a la recta principal para intentar recuperarle, pero mantén el stick muy suave en entrada de curva para no perder tracción. Si te queda menos de tres décimas en la salida, deja que pase y recupera posición en la siguiente vuelta.»<br>_11,53 s · 10,6 tok/s · 42 pal. · español · 2 frases_ |
| 2 | «Normal. Mantén la línea. Deja que el LMP2 se gestione.»<br>_0,92 s · 18,9 tok/s · 10 pal. · español · 3 frases_ | «Espera a la recta para dejarlo, Julián — con el stick suave no vas a ganar esa batalla.<br>Asegúrate de que el coche responde bien en entrada curva antes de soltar la posición.»<br>_5,64 s · 8,4 tok/s · 33 pal. · español · 2 frases_ |

### Incidente — toque en la chicane

**Piloto:** «¡Me han tocado por detrás en la chicane! El coche se va raro a la derecha. ¿Entro a boxes?»

| Rep | gemma4:e2b | qwen3.5:4b |
|---|---|---|
| 1 | «¡Hostia! Contacto trasero. Revisa la dirección ahora. Box en esta vuelta, Julián.»<br>_2,14 s · 14,2 tok/s · 12 pal. · español · 4 frases_ | «¡Coño! ¿Estás bien? Revisa el morro — daño frontal. Frena progresivo y suave para no perder tracción, luego dime cómo responde el coche. Box en esta vuelta o la siguiente.»<br>_10,17 s · 11,1 tok/s · 30 pal. · español · 5 frases_ |
| 2 | «¡Hostia! ¿Estás bien? Revisa el morro. ¿Cómo va la dirección? Box esta vuelta.»<br>_1,84 s · 14,8 tok/s · 13 pal. · español · 5 frases_ | «¡Joder, Julián! Daño lateral y pérdida de adherencia — no entres ahora, revisa el morro con calma antes de salir.»<br>_3,12 s · 11,2 tok/s · 20 pal. · español · 2 frases_ |


## 8. Recomendación

**Mantener `gemma4:e2b`. No cambiar el fallback local a `qwen3.5:4b`.**

### Por qué

1. **Latencia, que es lo que manda en carrera.** Con pregunta nueva (el caso real: cada
   mensaje de radio es un prompt distinto), gemma responde en **2,11 s de mediana y 3,45 s
   en el p95**; qwen tarda **13,64 s de mediana y 15,48 s en el p95**. Es ~6,5 veces más
   lento en el caso típico. Una respuesta a 13-15 s llega cuando Julián ya pasó la curva,
   el punto de frenada o la ventana de boxes: el mensaje deja de ser útil. Incluso con el
   prompt cacheado (rep 2), qwen tarda 4,54 s de mediana, más que el p95 de gemma (3,45 s).
2. **El coste de qwen está en el prefill, no en escribir.** qwen emplea ~8,9 s en evaluar
   los ~950 tokens del prompt de sistema y solo ~4,4 s en generar la respuesta. Eso es
   inherente a su tamaño/arquitectura en CPU y no se arregla subiendo `num_thread` (ya son
   6, como en producción) ni bajando `num_predict` (110 ya es corto).
3. **No mejora la precisión: la empeora.** En combustible, gemma acierta 1/6 y qwen 0/6.
   qwen falla los mismos casos que gemma (dice que llega con 18,4 L para 9 vueltas) y
   además responde con una seguridad peligrosa: con 9,6 L para 5 vueltas afirma que
   «te sobran casi dos vueltas». En una cuenta que decide una carrera, eso es peor que no
   contestar. Recuerda que Victor ya resuelve el combustible por la vía determinista
   (`allow_local=False`): el modelo local no es el sitio para aritmética.
4. **qwen se inventa datos.** En la pregunta en inglés (sin ningún dato de contexto)
   responde «Vuelta 8, un cuarenta y dos seis»; en gaps afirma «con dos vueltas más puedes
   cerrar esos tres segundos» cuando quedan 12; en neumáticos inventa «si el rival entra a
   2.0»; tras un toque **por detrás** dice «daño frontal». El prompt de sistema prohíbe
   explícitamente inventar nombres, gaps, tiempos o daños. Para un ingeniero de carrera,
   un dato inventado es más caro que una respuesta genérica: puede provocar una decisión
   equivocada (box, adelantamiento, gestión de gomas).
5. **RAM y forma.** qwen ocupa 3,84 GB de RAM frente a 3,05 GB de gemma (+0,8 GB en CPU,
   no en VRAM) en un equipo que ya comparte memoria con Assetto Corsa y Windows. qwen es
   más respetuoso con «máximo 2 frases» (17/24 frente a 0/24), pero es más largo en
   palabras (30 de mediana frente a 17, hasta 62), así que suma segundos de TTS **después**
   de la latencia. Gemma tiene un defecto claro y barato de corregir: verbaliza la etiqueta
   «Normal.» en 15/24 respuestas; se elimina con un `replace` al inicio o renombrando los
   modos del prompt, sin tocar el modelo.

### Qué haría en la práctica

- Dejar `LOCAL_LLM_MODEL=gemma4:e2b` como último recurso de la cadena.
- Mantener el combustible, gaps y tiempos por la vía determinista (ya está).
- Limpiar el «Normal.» de las respuestas de gemma (post-proceso de una línea).
- Revisar qwen3.5:4b solo si en el futuro se busca calidad para momentos **sin prisa**
  (briefing pre-carrera, debrief escrito) y siempre con datos inyectados y verificados;
  nunca para radio en vivo. Tal como está hoy, el intercambio es perder 11 segundos por
  pregunta a cambio de respuestas más largas, igual de malas en la cuenta de combustible y
  con más datos inventados.

**Veredicto: `gemma4:e2b` se queda.**
