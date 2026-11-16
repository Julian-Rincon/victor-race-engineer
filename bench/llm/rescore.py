#!/usr/bin/env python3
"""Re-puntúa un results.json con los evaluadores actuales de bench_llm.py y
regenera su summary.json, sin volver a llamar a los modelos.

Uso (desde la raíz del repo):
  python3 bench/llm/rescore.py bench/llm/results/2026-11-11
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_llm as bench  # noqa: E402

if len(sys.argv) != 2:
    sys.exit(__doc__)
RUN = Path(sys.argv[1])
results = json.loads((RUN / "results.json").read_text(encoding="utf-8"))

changed = 0
for model, records in results.items():
    for r in records:
        text = r.get("response", "")
        old = (r.get("fuel") or {}).get("conclusion_ok")
        r["sentences"] = bench.sentence_count(text)
        r["brief"] = bench.is_brief(text)
        r["empty"] = not text.strip()
        r["language"] = bench.detect_language(text) if text else "indeterminado"
        r["reasoning_leak"] = bool(bench.REASONING_RE.search(text)) or bool(r.get("thinking_field"))
        task = next((t for t in bench.TASKS if t["id"] == r["task"]), None)
        if task and task.get("fuel"):
            new = bench.fuel_eval(text, bench.FUEL_CASES[task["fuel"]])
            r["fuel"] = new
            if old != new["conclusion_ok"]:
                changed += 1
                print(f"[cambio fuel] {model} {r['task']}#{r['rep']}: {old} -> "
                      f"{new['conclusion_ok']} ({new['conclusion_modelo']})")

summary = {m: bench.summarize_model(rs) for m, rs in results.items()}
(RUN / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
(RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{changed} registros de combustible cambiados")
print(json.dumps(summary, ensure_ascii=False, indent=2))
