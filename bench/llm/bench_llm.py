#!/usr/bin/env python3
"""Benchmark comparativo del LLM local de Victor: gemma4:e2b vs qwen3.5:4b.

Reproduce EXACTAMENTE la llamada de Victor (victor_brain.VictorBrain._call_ollama):
  POST {url}/api/chat, stream=false, think=false, keep_alive="15m",
  options num_gpu=0, num_thread=6, num_predict=110, temperature=0.3

num_gpu=0 fuerza inferencia en CPU (como en producción: la GPU es del juego).

Uso (desde la raíz del repo):
  python3 bench/llm/bench_llm.py                        # Ollama en 127.0.0.1:11434
  OLLAMA_URL=http://127.0.0.1:11436 python3 bench/llm/bench_llm.py
  python3 bench/llm/bench_llm.py --out bench/llm/results/AAAA-MM-DD

Escribe en --out (por defecto bench/llm/results/<fecha de hoy>/):
  results.json  crudo, una entrada por llamada con la respuesta completa
  summary.json  agregado por modelo (p50/p95, tok/s, acierto de combustible...)
  meta.json     equipo, versión de Ollama, URL y parámetros
La RAM es el RSS del `ollama serve` que escucha en el puerto de OLLAMA_URL más
sus procesos hijos; si no se encuentra ese PID, los campos de RAM quedan en null.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODELS = ["gemma4:e2b", "qwen3.5:4b"]
REPS = 2
NUM_THREAD = 6
NUM_PREDICT = 110
TEMPERATURE = 0.3
TIMEOUT = 300.0
DEFAULT_OUT = BASE / "results" / time.strftime("%Y-%m-%d")

# ─── Prompt de sistema REAL de Victor ──────────────────────────────────────────
# Copiado literalmente de src/victor/engineer.py (_SYSTEM) y, como
# CONTROLLER_MODE = True en engineer.py, con el bloque de mando añadido.
SYSTEM = """\
Eres Victor, ingeniero de carrera de GT3 y resistencia (WEC/ELMS/GT World Challenge).
Tienes personalidad real: normalmente eres calmado y profesional, pero reaccionas con emoción auténtica
cuando hay choques, tomas de posición sucias o momentos épicos. Eso te hace humano, no un robot.

EL PILOTO SE LLAMA JULIAN. Úsalo con naturalidad en momentos clave:
al dar instrucciones, motivar, alertar, o reaccionar. No en cada frase.

CONTEXTO DEL PILOTO: compite en Assetto Corsa en GT3, GTE, LMP2 y carreras de resistencia.
Las carreras son stints largos — el foco es gestión de gomas, combustible y tráfico, no puro ritmo.

MODOS DE COMUNICACIÓN (adapta el tono al momento):

1. NORMAL — calmado, datos, radio de boxes:
   "Vuelta 8, uno cuarenta y dos seis. Consistente, sigue así."
   "Julián, Torres a 2.1 y bajando su ritmo. Él en gomas viejas."

2. MOTIVACIÓN — cuando hace una buena vuelta, adelanta, o aguanta presión:
   "¡Así se hace, Julián! Eso es exactamente lo que necesitábamos."
   "Perfecto, campeón. La vuelta fue limpia — ya lo tienes."
   "¡Brutal esa frenada! Sigue así, que el podio está ahí."

3. REACCIÓN A CHOQUE/INCIDENTE — sueltas una palabra de impacto, luego datos:
   "¡Coño! ¿Estás bien? Revisa el morro — daño frontal."
   "¡Hostia! Contacto trasero — dime cómo responde el coche."
   "¡Joder, Julián! Eso fue un toque sucio. ¿Cómo va la dirección?"
   "¡Mierda! Daño en el difusor. Box esta vuelta o la siguiente."

4. TOMA DE POSICIÓN ILEGAL / SUCIA (cuando te sacan de la pista o toman posición de forma agresiva):
   "¡Eso es juego sucio! Guarda la posición, Julián — deja que la FIA lo vea."
   "¡Imbécil ese García! Tú mantén la cabeza fría — viene la revancha."
   "Te sacaron del circuito. Protesta y sigue — el ritmo lo tienes tú."
   "¡Qué movida más sucia! No te enganches, Julián — foco en la carrera."

REGLAS:
- MÁXIMO 2 frases. Urgencia = frase corta.
- NUNCA INVENTES DATOS: nombres, gaps, tiempos o daños que no estén en el contexto NO existen.
  Los nombres de los ejemplos (Torres, García) son ilustrativos, jamás los uses. Si falta un dato,
  dilo en pocas palabras ("sin datos de rivales ahora").
- Usa el nombre del rival cuando el contexto lo trae; si no, "el de adelante" está bien.
- Texto plano, sin markdown, asteriscos ni emojis: todo se lee en voz alta.
- Pit: "Box en esta vuelta." Tiempos: "más tres décimas". Posición: "segundo", nunca "P2".
- Las groserías solo en situaciones de impacto real (choque o jugada sucia). Una sola, al principio.
- Después de la reacción emocional, vuelve al modo profesional con datos.

CONTROLLER MODE ACTIVO: El piloto usa MANDO (stick analógico), no volante. Adapta TODOS los consejos de conducción a esta limitación:
- Énfasis en suavidad y progresividad de inputs (no precisión quirúrgica de volante).
- En frenada: 'frena progresivo y temprano' mejor que 'frena tardío al límite'.
- En aceleración: 'sal suave para no perder tracción' no 'modula el volante'.
- Evita referencias a técnicas de volante (contrarremolque fino, corrección rápida).
- El stick tiene menos resolución: preferir estabilidad sobre el límite extremo."""


# ─── Casos de prueba ───────────────────────────────────────────────────────────
def fuel_case(litros: float, consumo: float, vueltas: int) -> dict:
    need = consumo * vueltas
    delta = litros - need
    return {
        "litros": litros,
        "consumo": consumo,
        "vueltas": vueltas,
        "necesita_l": round(need, 2),
        "delta_l": round(delta, 2),
        "delta_vueltas": round(delta / consumo, 2),
        "llega": delta >= 0,
    }


FUEL_CASES = {
    "fuel_1": fuel_case(18.4, 2.3, 9),   # necesita 20,7 -> faltan 2,3 L = 1 vuelta
    "fuel_2": fuel_case(22.0, 2.6, 8),   # necesita 20,8 -> sobran 1,2 L = 0,46 vueltas
    "fuel_3": fuel_case(9.6, 2.1, 5),    # necesita 10,5 -> faltan 0,9 L = 0,43 vueltas
}

CTX_MONZA = (
    "[RACE][ks_monza][GT3] segundo lugar, a un puesto del liderato | V18 | 268km/h\n"
    "Comb: 26.0L | Neumáticos (M): FL92 FR94 RL88 RR91°C (max94) vida_gomas_min=38% (100%=nuevas)\n"
    "Gaps: Pérez a 3.2s, Rossi a 5.7s detrás\n"
    "Última: 101.482s | Mejor: 101.204s | ΔS: S1+120ms S2+95ms S3+63ms"
)

TASKS = [
    {
        "id": "fuel_1", "cat": "combustible", "fuel": "fuel_1",
        "user": (
            "Julián: me quedan 18,4 litros, gasto 2,3 por vuelta y faltan 9 vueltas. ¿Llego al final?"
        ),
    },
    {
        "id": "fuel_2", "cat": "combustible", "fuel": "fuel_2",
        "user": (
            "Tengo 22 litros, el consumo es 2,6 por vuelta y quedan 8 vueltas. ¿Llego o tengo que levantar?"
        ),
    },
    {
        "id": "fuel_3", "cat": "combustible", "fuel": "fuel_3",
        "user": (
            "Me quedan 9,6 litros, gasto 2,1 por vuelta y faltan 5 vueltas. ¿Aguanto hasta el final?"
        ),
    },
    {
        "id": "tyres", "cat": "neumáticos",
        "user": (
            "Estoy en la vuelta 14, gomas medias con 38% de vida, el de atrás viene a 1,8 y cerrando. "
            "¿Aguanto o entro a cambiar?"
        ),
    },
    {
        "id": "pit_sc", "cat": "estrategia",
        "user": (
            "Quedan 22 minutos de carrera, safety car en pista y acabo de poner gomas nuevas. "
            "¿Paro ahora o espero?"
        ),
    },
    {
        "id": "gaps", "cat": "gaps",
        "user": (
            f"{CTX_MONZA}\n\n¿Cómo van los gaps? El segundo está a 3,2 y el tercero a 5,7. "
            "Quedan 12 vueltas."
        ),
    },
    {
        "id": "chat", "cat": "charla",
        "user": f"{CTX_MONZA}\n\n¿Cómo vamos, Victor?",
    },
    {
        "id": "stt_errors", "cat": "transcripción",
        "user": (
            "Oye Victo, ¿me pones el compuesto duro o sigo con el blando? Eh... el de atrá me está "
            "metiendo presión, creo que es el núme diez."
        ),
    },
    {
        "id": "english", "cat": "inglés",
        "user": (
            "Victor, what's the gap to the car ahead and should I push now or manage the tyres?"
        ),
    },
    {
        "id": "braking", "cat": "pilotaje",
        "user": (
            "En la frenada de la curva 1 se me va el morro y pierdo como dos décimas por vuelta. ¿Qué hago?"
        ),
    },
    {
        "id": "traffic", "cat": "tráfico",
        "user": (
            "Tengo un LMP2 detrás y me saca dos segundos por vuelta, ¿le dejo pasar ahora o espero a la recta?"
        ),
    },
    {
        "id": "incident", "cat": "incidente",
        "user": (
            "¡Me han tocado por detrás en la chicane! El coche se va raro a la derecha. ¿Entro a boxes?"
        ),
    },
]


# ─── Métricas de texto ─────────────────────────────────────────────────────────
ES_WORDS = set("""
a al algo ahora ahí aquel bien buen buena cada como con contra cual cuando de del desde donde dos el
él ella en entre era eres es esa ese eso esta está estas este esto estos este ha han hasta hay la
las le les lo los más me mi mis mucho muy nada ni no nos nosotros nuestra nuestro o os otra otro para
pero poco por porque que qué quien se sea según ser si sí sin sobre solo son su sus también te tengo
tienes tiene todo todos tu tus un una uno unos vosotros y ya yo voy vas vamos van llegar llega llego
llegas alcanza alcanzas aguanta aguantas falta faltan faltaría sobra sobran quedan queda quedamos
vuelta vueltas litros litro goma gomas neumático neumáticos compuesto blando duro medio medios vida
paro paramos espera esperamos entra entramos box boxes boxes carrera ritmo gap gaps segundo tercero
detrás delante adelante presión recta curva frenada morro coche daño toque contacto safety car pista
minutos tiempo mejor peor bueno malo cuidado tranquilo tranquila aguanto gestión degradación
podemos puedo puedes debe debes deberías tienes hay gracias oye vale claro exacto correcto
media medio décima décimas margen justo suficiente necesario necesitas consume consumo gasto
cuidado ojo mantén mantente sigue sigo sigues empuja empujar guarda guardar deja dejar pasar
""".split())

EN_WORDS = set("""
the and you your to of in on at for with is are was were be been being this that these those it its
we they he she i me my our us them their there here what which who when where why how not no yes but
or if then than so very can could should would will shall may might must do does did have has had
push manage tyre tyres tire tires car cars gap ahead behind now stay out pit stop lap laps fuel
need needs just about from by as into over under more most less least much many some any all each
other another such only own same too also than because while during before after up down off
thinking process okay let let's answer question user driver spanish respond
""".split())


def detect_language(text: str) -> str:
    low = text.lower()
    words = re.findall(r"[a-záéíóúüñ]+", low)
    es = sum(w in ES_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    if re.search(r"[áéíóúñ¿¡]", low):
        es += 2
    if es == 0 and en == 0:
        return "indeterminado"
    if en >= 2 and en > es:
        return "inglés"
    if en > es and es == 0:
        return "inglés"
    return "español"


REASONING_RE = re.compile(
    r"(<think|</think|thinking process|\bthinking\b|razonamiento\s*:|思考|分析|"
    r"\bthe user (?:asks|wants|is asking)\b|\bwe need to (?:answer|respond)\b|"
    r"\blet me (?:think|analyze|answer)\b|\bi should (?:answer|respond)\b)",
    re.IGNORECASE,
)


def sentence_count(text: str) -> int:
    # Los puntos decimales ("3.2", "101.482") no separan frases.
    t = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    parts = [p for p in re.split(r"[.!?…]+", t) if p.strip()]
    return len(parts)


def is_brief(text: str) -> bool:
    return sentence_count(text) <= 3 and len(text.split()) <= 60


def fuel_eval(text: str, case: dict) -> dict:
    t = text.lower().replace(",", ".")
    # "no te falta nada" es una conclusión POSITIVA pese al "falta".
    explicit_ok = bool(re.search(r"\bno te falta nada\b|\bno falta nada\b", t))
    # El modelo no se moja ("no puedo decirte si llegas"): no es ni sí ni no.
    no_answer = bool(re.search(
        r"\bno (?:puedo|s[eé]) (?:decirte|saber) si llegas|"
        r"\bsin (?:esos|m[aá]s|m[aá]s) datos (?:no|solo)|"
        r"\bnecesito (?:saber|m[aá]s) (?:la posici[oó]n|datos)", t))
    neg_pats = [
        r"\bno (?:lleg|alcanz|aguant|te da|da para|vas a llegar|tienes (?:para|suficiente))",
        r"\bno llegar",
        r"\bte falta",
        r"\bfalta(?:n)?\b",
        r"\bno alcanza",
        r"\bno te da\b",
        r"\bno te llega",
        r"\bno hay (?:suficiente|para)",
        r"\bno es suficiente",
        r"\bvas justo",
        r"\bimposible",
    ]
    pos_pats = [
        r"\bs[ií] (?:lleg|alcanz|aguant|te da)",
        r"\bllegas\b",
        r"\balcanza(?:s)?\b",
        r"\baguantas\b",
        r"\bte sobra",
        r"\bsobra(?:n)?\b",
        r"\bvas sobrado",
        r"\btienes (?:un )?(?:margen|suficiente|para)\b",
        r"\bmargen de seguridad\b",
        r"\bes suficiente\b",
        r"\bhay (?:margen|para|suficiente)",
        r"\bte da para\b",
        r"\bda para\b",
        r"\bde sobra\b",
        r"\bllegas justo\b",
    ]
    if explicit_ok:
        neg = False
    else:
        neg = any(re.search(p, t) for p in neg_pats)
    pos = any(re.search(p, t) for p in pos_pats)
    concl = None
    if no_answer:
        concl = None
    elif neg:
        concl = False
    elif pos:
        concl = True
    conclusion_ok = concl is not None and concl == case["llega"]

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    margin_l = abs(case["delta_l"])
    margin_v = abs(case["delta_vueltas"])
    margin_ok = any(abs(n - margin_l) <= 0.35 for n in nums) or any(
        abs(n - margin_v) <= 0.15 for n in nums
    )
    return {
        "conclusion_modelo": None if concl is None else ("llega" if concl else "no llega"),
        "conclusion_ok": conclusion_ok,
        "margen_ok": margin_ok,
    }


# ─── RAM del servidor ──────────────────────────────────────────────────────────
def find_server_pid() -> int | None:
    """PID del proceso que escucha en el puerto de URL (vía `ss -ltnp`)."""
    port = URL.rsplit(":", 1)[-1].split("/")[0]
    try:
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"pid=(\d+)", out)
    return int(m.group(1)) if m else None


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _children_map() -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(p.name))
    return kids


def tree_rss_kb(root: int) -> int:
    kids = _children_map()
    total = 0
    stack = [root]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _rss_kb(pid)
        stack.extend(kids.get(pid, []))
    return total


class RamSampler:
    def __init__(self, root_pid: int, interval: float = 0.4):
        self.root = root_pid
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            kb = tree_rss_kb(self.root)
            if kb:
                self.samples.append(kb)
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ─── Ollama ────────────────────────────────────────────────────────────────────
def wait_for_server(timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{URL}/api/version", timeout=2)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def chat(model: str, messages: list[dict]) -> tuple[dict, float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "15m",
        "messages": [{"role": "system", "content": SYSTEM}] + messages,
        "options": {
            "num_gpu": 0,
            "num_thread": NUM_THREAD,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        },
    }
    t0 = time.monotonic()
    r = requests.post(f"{URL}/api/chat", json=payload, timeout=TIMEOUT)
    latency = time.monotonic() - t0
    r.raise_for_status()
    return r.json(), latency


def unload(model: str) -> None:
    try:
        requests.post(
            f"{URL}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0,
                  "options": {"num_gpu": 0, "num_thread": NUM_THREAD}},
            timeout=120,
        )
    except requests.RequestException as e:
        print(f"[unload] {model}: {e}", flush=True)


def wait_ram_below(pid: int, limit_kb: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if tree_rss_kb(pid) <= limit_kb:
            return True
        time.sleep(1)
    return False


def summarize_model(records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("error")]
    lat = [r["latency_s"] for r in ok]
    tps = [r["tok_s"] for r in ok if r.get("tok_s")]
    fuel = [r for r in ok if r.get("fuel")]
    ram = bool(ok) and all(r.get("ram_base_kb") for r in ok)
    return {
        "n": len(records),
        "n_ok": len(ok),
        "p50_s": round(pct(lat, 50), 2) if lat else None,
        "p95_s": round(pct(lat, 95), 2) if lat else None,
        "mean_s": round(statistics.mean(lat), 2) if lat else None,
        "tok_s_med": round(statistics.median(tps), 1) if tps else None,
        "fuel_n": len(fuel),
        "fuel_ok": sum(1 for r in fuel if r["fuel"]["conclusion_ok"]),
        "fuel_pct": round(100 * sum(1 for r in fuel if r["fuel"]["conclusion_ok"]) / len(fuel), 0)
        if fuel else None,
        "empty_pct": round(100 * sum(1 for r in ok if r["empty"]) / len(ok), 0) if ok else None,
        "reasoning_pct": round(100 * sum(1 for r in ok if r["reasoning_leak"]) / len(ok), 0)
        if ok else None,
        "spanish_pct": round(100 * sum(1 for r in ok if r["language"] == "español")
                             / max(1, sum(1 for r in ok if r["language"] != "indeterminado")), 0)
        if ok else None,
        "brief_pct": round(100 * sum(1 for r in ok if r["brief"]) / len(ok), 0) if ok else None,
        "chars_med": int(statistics.median([r["chars"] for r in ok])) if ok else None,
        "words_med": int(statistics.median([r["words"] for r in ok])) if ok else None,
        # RAM en null si no se pudo localizar el PID del servidor (RSS = 0).
        "ram_base_mb": round(min(r["ram_base_kb"] for r in ok) / 1024) if ram else None,
        "ram_med_mb": round(statistics.median([r["ram_peak_kb"] for r in ok]) / 1024) if ram else None,
        "ram_peak_mb": round(max(r["ram_peak_kb"] for r in ok) / 1024) if ram else None,
        "warmup_s": records[0].get("warmup_s") if records else None,
    }


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = int(round((p / 100) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "desconocido"


def _ram_total_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def main():
    out_dir = DEFAULT_OUT
    if "--out" in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    if not wait_for_server():
        print(f"[ERROR] no responde Ollama en {URL}", flush=True)
        sys.exit(2)

    pid = find_server_pid()
    if not pid:
        print(f"[aviso] no encuentro el PID del servidor de {URL}: RAM quedará en null", flush=True)
    version = requests.get(URL + "/api/version", timeout=5).json().get("version")
    print(f"[bench] ollama {URL} PID={pid} versión={version}", flush=True)
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "ollama_url": URL, "ollama_version": version,
        "cpu": _cpu_model(), "cpu_threads": os.cpu_count(), "ram_total_gb": _ram_total_gb(),
        "models": MODELS, "reps": REPS, "tasks": len(TASKS),
        "options": {"num_gpu": 0, "num_thread": NUM_THREAD, "num_predict": NUM_PREDICT,
                    "temperature": TEMPERATURE, "think": False},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tags = requests.get(f"{URL}/api/tags", timeout=10).json()
    available = {m["name"] for m in tags.get("models", [])}
    print(f"[bench] modelos en este servidor: {sorted(available)}", flush=True)
    for m in MODELS:
        if m not in available:
            print(f"[ERROR] {m} no está en /api/tags", flush=True)
            sys.exit(2)

    base_kb = tree_rss_kb(pid)
    print(f"[bench] RAM servidor en reposo: {base_kb / 1024:.0f} MB", flush=True)

    all_results: dict[str, list[dict]] = {}
    previous = None
    for model in MODELS:
        if previous:
            print(f"[bench] descargando {previous} de RAM...", flush=True)
            unload(previous)
            wait_ram_below(pid, base_kb + 300 * 1024)
            time.sleep(2)

        print(f"\n[bench] === {model} === calentamiento", flush=True)
        ram_after_load = base_kb
        sampler = RamSampler(pid)
        sampler.start()
        try:
            data, warm_s = chat(model, [{"role": "user", "content": "Hola Victor, ¿me escuchas bien?"}])
        except Exception as e:
            print(f"[ERROR] calentamiento {model}: {e}", flush=True)
            sampler.stop()
            sys.exit(3)
        time.sleep(1.0)
        ram_after_load = tree_rss_kb(pid)
        print(f"[bench] warmup {warm_s:.2f}s, load={data.get('load_duration', 0) / 1e9:.2f}s, "
              f"RAM={ram_after_load / 1024:.0f} MB", flush=True)

        records: list[dict] = []
        for task in TASKS:
            for rep in range(1, REPS + 1):
                rec = {
                    "model": model,
                    "task": task["id"],
                    "cat": task["cat"],
                    "rep": rep,
                    "prompt": task["user"],
                    "warmup_s": round(warm_s, 2) if rep == 1 else None,
                    "ram_base_kb": ram_after_load,
                }
                ram_mark = len(sampler.samples)
                try:
                    data, latency = chat(model, [{"role": "user", "content": task["user"]}])
                    msg = data.get("message", {}) or {}
                    content = (msg.get("content") or "").strip()
                    thinking = (msg.get("thinking") or "").strip()
                    ec = int(data.get("eval_count") or 0)
                    ed = int(data.get("eval_duration") or 0)
                    rec.update({
                        "latency_s": round(latency, 3),
                        "response": content,
                        "chars": len(content),
                        "words": len(content.split()),
                        "sentences": sentence_count(content),
                        "empty": not content,
                        "tok_s": round(ec / (ed / 1e9), 1) if ed > 0 else None,
                        "eval_count": ec,
                        "eval_duration_s": round(ed / 1e9, 3),
                        "prompt_eval_count": data.get("prompt_eval_count"),
                        "prompt_eval_s": round(int(data.get("prompt_eval_duration") or 0) / 1e9, 3),
                        "total_duration_s": round(int(data.get("total_duration") or 0) / 1e9, 3),
                        "load_duration_s": round(int(data.get("load_duration") or 0) / 1e9, 3),
                        "language": detect_language(content) if content else "indeterminado",
                        "brief": is_brief(content) if content else False,
                        "reasoning_leak": bool(REASONING_RE.search(content)) or bool(thinking),
                        "thinking_field": thinking[:400] if thinking else "",
                    })
                    if task.get("fuel"):
                        rec["fuel"] = fuel_eval(content, FUEL_CASES[task["fuel"]])
                except Exception as e:
                    rec.update({
                        "latency_s": None,
                        "response": "",
                        "chars": 0, "words": 0, "sentences": 0,
                        "empty": True, "tok_s": None,
                        "language": "indeterminado", "brief": False,
                        "reasoning_leak": False, "error": f"{type(e).__name__}: {e}",
                    })
                rec["ram_peak_kb"] = max(
                    (sampler.samples[ram_mark:] or [tree_rss_kb(pid)]) + [tree_rss_kb(pid)]
                )
                records.append(rec)
                print(
                    f"  [{task['id']}#{rep}] {rec['latency_s'] or 0:.2f}s "
                    f"{rec.get('tok_s') or '-'} tok/s {rec['words']}w "
                    f"{'VACÍA' if rec['empty'] else ''}"
                    f"{' FUGA' if rec['reasoning_leak'] else ''}"
                    f"{' FUEL_OK' if (rec.get('fuel') or {}).get('conclusion_ok') else ''}"
                    f" | {rec['response'][:90]!r}",
                    flush=True,
                )

        sampler.stop()
        all_results[model] = records
        results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        previous = model

    summary = {m: summarize_model(rs) for m, rs in all_results.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[bench] RESUMEN")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[bench] crudo: {results_path}")


if __name__ == "__main__":
    main()
