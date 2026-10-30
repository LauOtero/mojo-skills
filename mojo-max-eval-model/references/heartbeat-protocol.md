# Protocolo de heartbeat / reporte de progreso

**Propósito**: JSONL estandarizado de progreso en stdout para que puedas monitorizar la vivacidad y detectar estancamientos mientras se ejecuta una evaluación. Los logs legibles por humanos van a stderr.

## Formato JSONL

Cada línea de heartbeat es un objeto JSON válido (delimitado por saltos de línea). Se emite a **stdout** cada `--heartbeat-interval` segundos y en las transiciones de fase (seed→full, tarea→tarea, done).

### Campos comunes (todos los heartbeats)

| Campo | Tipo | Descripción |
|---|---|---|
| `ts` | string (ISO8601 UTC) | Marca de tiempo de emisión |
| `phase` | string | Uno de: `preflight`, `seed`, `full`, `done`, `error` |
| `task` | string | Nombre de la tarea actual, o `""` para un error de configuración global |
| `completed` | int | Ejemplos completados en la fase actual |
| `total` | int\|null | Total de ejemplos, o `null` mientras corre una tarea agrupada |
| `accuracy` | float\|null | Precisión en curso (0.0-1.0) o `null` si aún no es calculable |
| `eta_sec` | int\|null | Segundos estimados hasta completar la fase |

### Campos específicos de fase

**`phase: "preflight"`** — Se están ejecutando las comprobaciones de capacidad del endpoint.

Los heartbeats de preflight usan un `task` vacío, con `completed` y `total` a `0`.

**`phase: "seed"`** — Muestra rápida de seed en ejecución

```jsonl
{"ts":"2026-07-07T14:23:12Z","phase":"seed","task":"gsm8k","completed":12,"total":24,"accuracy":0.583,"eta_sec":45}
{"ts":"2026-07-07T14:23:42Z","phase":"seed","task":"gsm8k","completed":24,"total":24,"accuracy":0.625,"eta_sec":0,"fast_fail":false,"floor":0.10}
```

- `fast_fail`: boolean — si el fast-fail se dispararía en este punto
- `floor`: float — el suelo de fast-fail para esta tarea

**`phase: "full"`** — Evaluación completa en ejecución

```jsonl
{"ts":"2026-07-07T14:23:43Z","phase":"full","task":"gsm8k","completed":0,"total":200,"accuracy":null,"eta_sec":420}
{"ts":"2026-07-07T14:27:12Z","phase":"full","task":"gsm8k","completed":50,"total":200,"accuracy":0.640,"eta_sec":350}
```

**`phase: "done"`** — Tarea completada

```jsonl
{"ts":"2026-07-07T14:35:00Z","phase":"done","task":"gsm8k","completed":200,"total":200,"accuracy":0.635,"summary_path":"eval-results/meta-llama_Meta-Llama-3-8B-Instruct/gsm8k_summary.json","status":"met","target":0.50}
```

- `summary_path`: ruta al JSON de resumen escrito
- `status`: `"completed" | "met" | "below-target" | "fast-fail" | "full-eval-skipped"`
- `target`: float o `null` — un `--target` explícito, si se proporcionó

El script de evaluación emite `seed`, `full`, `done` y `error`. No se auto-termina cuando un watcher externo considera que la ejecución está estancada.

## Contrato de monitorización

Al seguir stdout:

1. **Vivacidad** — Si no hay heartbeat durante `2 * --heartbeat-interval` segundos, trata la ejecución como estancada.
2. **Progreso** — Los heartbeats de evaluación directa pueden incluir precisión en curso. Los heartbeats de lm-eval reportan vivacidad; la precisión aparece cuando la tarea termina. Las tareas agrupadas también incluyen `limit_per_subtask` mientras su `total` agregado es desconocido.
3. **Fast-fail** — Con código de salida `3`, lee el JSON de resumen para `status: "fast-fail"`.
4. **Resultados** — Al completar, lee `summary_path` o `eval_summary.json`.

## Consumidor de ejemplo

```python
import json
import sys
import time

HEARTBEAT_INTERVAL = 30
last_heartbeat = time.time()

for line in sys.stdin:
    hb = json.loads(line)
    last_heartbeat = time.time()
    if hb["phase"] == "seed" and hb.get("fast_fail"):
        print(f"⚠️  {hb['task']} seed por debajo del suelo {hb['floor']:.0%}")
    elif hb["phase"] == "done":
        accuracy = hb.get("accuracy")
        formatted = f"{accuracy:.1%}" if accuracy is not None else "n/a"
        print(f"{hb['task']}: {formatted} ({hb.get('status', 'done')})")

if time.time() - last_heartbeat > 2 * HEARTBEAT_INTERVAL:
    print("La ejecución puede estar estancada — no hay heartbeat reciente")
```

## Notas de implementación

- Stdout se vacía después de cada línea de heartbeat.
- Los heartbeats se emiten en los límites de fase incluso si el intervalo no ha transcurrido.
- Las marcas de tiempo usan UTC (ISO8601 con sufijo `Z`).
