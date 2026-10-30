---
name: mojo-max-benchmark-model
description: >
  Mide el rendimiento de un modelo servido con MAX usando el comando
  `max benchmark`: throughput (tokens/s), latencia (TTFT, TPOT, ITL) y
  utilización de GPU/CPU generando carga contra un endpoint `max serve` en
  ejecución. Úsala siempre que el usuario quiera hacer benchmark, prueba de
  carga o medir el rendimiento de un modelo MAX, obtener cifras de
  tokens/s, TTFT o TPOT, ejecutar un sweep de concurrencia o de request-rate,
  comparar latencia frente a throughput, dimensionar un despliegue, o
  producir JSON de benchmark, aunque no diga "benchmark" explícitamente.
  También cuando una ejecución de `max benchmark` no conecta o devuelve
  cifras cero o erróneas.
compatibility: Requiere una instalación de MAX con pip o pixi y un endpoint `max serve` en ejecución. Las estadísticas de GPU (`--collect-gpu-stats`) necesitan que el benchmark se ejecute en la misma máquina NVIDIA o AMD que el servidor.
argument-hint: "[qué medir, por ejemplo 'latencia de una sola petición' o 'throughput máximo']"
---

# Benchmark de un modelo en MAX

`max benchmark` mide un servidor de modelos **en ejecución**. Es un generador de carga: envía peticiones de inferencia a un endpoint `max serve` activo, las cronometra y reporta throughput (tokens/s) y latencia (TTFT, TPOT, latencia inter-token). Un benchmark combina dos cosas: un servidor bajo prueba y una **carga de trabajo** que responde a la pregunta que te haces.

Cada ejecución reporta tanto throughput como latencia, así que no hay un "modo" que seleccionar. Decide primero qué quieres aprender y luego elige la carga de trabajo que lo mide. La latencia de un solo flujo y el throughput máximo provienen de cargas de trabajo distintas, por lo que **la carga que elijas es la medición**. Si aciertas con ella, el número sale limpio.

Usa esta skill cuando quieras una cifra de rendimiento de un modelo en MAX: tokens/s, TTFT/TPOT, un sweep de concurrencia o request-rate, un compromiso latencia-throughput o un dimensionamiento de despliegue.

**No la uses cuando aún no haya un servidor en ejecución.** `max benchmark` es un cliente; primero arranca un servidor con la skill `mojo-max-serve-model`. Para saber *dónde* se va el tiempo a nivel de kernel, usa `mojo-max-profile-model`. Para comprobar *si la salida es correcta*, trátalo como una tarea de paridad (`mojo-max-import-model` y luego `mojo-max-debug-model`), no como un benchmark.

Esta skill funciona en cualquier instalación de MAX (pip o pixi). Añade `pixi run` si trabajas en un proyecto pixi.

## Referencias

| Archivo | Léelo cuando |
|---------|--------------|
| [references/flags.md](references/flags.md) | Elijas cualquier flag o dataset más allá de los de esta página |
| [references/metrics.md](references/metrics.md) | Quieras convertir las cifras de throughput y latencia en una conclusión |
| [references/troubleshooting.md](references/troubleshooting.md) | Una ejecución no conecte, las peticiones fallen o los números parezcan incorrectos |

Lee la referencia de lo que estés haciendo, no todas de golpe.

## 1. Verifica el servidor y lee el nombre del modelo

```bash
curl -s http://localhost:8000/v1/health     # 200 = listo
curl -s http://localhost:8000/v1/models      # anota el nombre del modelo servido
```

Ambas comprobaciones importan antes de gastar una ejecución:

- El `--model` del benchmark debe coincidir **exactamente** con el `--served-model-name` del servidor, o todas las peticiones fallarán. Toma el valor de `/v1/models` en lugar de adivinarlo.
- Si ese nombre servido es un *alias* y no un ID de Hugging Face (no contiene `/`), `--tokenizer` tomará ese alias por defecto y no podrá resolverlo; la ejecución morirá con "not a valid model identifier". Pasa el ID real de Hugging Face como `--tokenizer`.

Si no hay nada sirviendo, arranca un servidor con `max serve` (para una arquitectura personalizada, usa la skill `mojo-max-serve-model`). Hay un ajuste del servidor que también importa: **`--max-batch-size` limita la concurrencia real**. Un sweep hasta `--max-concurrency 32` contra un servidor arrancado con `--max-batch-size 1` encola las peticiones en lugar de agruparlas, así que sube el batch size del servidor para que coincida con el sweep o los puntos de alta concurrencia no significarán nada. Ten en cuenta además que MAX Serve incorpora **in-flight batching** y **prefix caching**, que pueden alterar el rendimiento observado; un `--max-batch-size` demasiado bajo desaprovecha estas optimizaciones.

Espera a `Server ready` **y entonces** lanza el benchmark. Hacer benchmark durante la compilación o el warmup produce tiempos de primer token basura.

> **Nota de versión (v25.7)**: `max serve --do-penalties` se renombró a `--enable-penalties` y está habilitado por defecto. Para deshabilitarlo, usa `--no-enable-penalties`. Si tu servidor es anterior a v25.7, el flag antiguo sigue funcionando.

## 2. Elige la carga de trabajo para tu pregunta

La carga de trabajo **es** la medición. Hazla coincidir con lo que quieres saber:

| Qué quieres saber | Carga de trabajo |
|---|---|
| Latencia de una sola petición en el mejor caso (TTFT, TPOT) | `--max-concurrency 1`, forma `random` fija, `--num-prompts` pequeño |
| Throughput máximo y dónde se degrada la latencia | Sweep `--max-concurrency 1,2,4,8,16,32`, más prompts |
| Rendimiento con una mezcla realista | `--dataset-name sharegpt` (**por defecto**), concurrencia moderada |
| Comportamiento a una carga objetivo | Sweep `--request-rate 1,2,4,8` (peticiones/s) |
| Rendimiento multiturno o de agentes | `--num-chat-sessions`, `--fit-distributions`, `--agentic-tool-profiles` |

Un sweep responde a las dos primeras filas a la vez: el punto de concurrencia 1 es la cifra de latencia en el mejor caso, y el pico del sweep es la cifra de throughput. Recurre a una ejecución dedicada a concurrencia 1 cuando solo quieras la cifra de latencia y no quieras pagar el resto de la curva.

Los knobs clave son los siguientes (`references/flags.md` tiene el catálogo completo):

- **`--dataset-name`**: elige `random` (sintético, con forma definida por `--random-input-len` y `--random-output-len`), `sharegpt` (chat real, **por defecto**), `arxiv-summarization` (contexto largo), o los nuevos datasets multiturno/agénticos (`instruct-coder`, `agentic-code`, `nemotron-opencode`, `chat-judge`, etc.). `random` funciona mejor para micro-mediciones limpias y reproducibles.
- **`--max-concurrency`** y **`--request-rate`**: aceptan un valor único o un *sweep* separado por comas (`1,2,4,8`). Un sweep es cómo se encuentra el codo del throughput.
- **`--endpoint`**: usa `/v1/completions` para LMs base (no necesitan plantilla de chat) o `/v1/chat/completions` para modelos instruct y chat (deben tener plantilla de chat o las peticiones devolverán 400). **El valor por defecto es `/v1/chat/completions`**.
- **`--max-output-len`**: fija la longitud de decodificación, que domina la duración de la ejecución.
- **`--num-prompts`**: obligatorio para ejecuciones de un solo turno.
- **`--num-chat-sessions`**: número de sesiones de chat multiturno; obligatorio para `chat-judge` y para `--fit-distributions`.
- **`--seed`**: semilla aleatoria para el generador de carga. Por defecto `24301`; pasa `--seed none` para una semilla aleatoria. Útil para reproducibilidad.
- **`--backend`**: por defecto `modular`; también acepta `vllm`, `sglang`, `trtllm` y sus variantes `-chat` para comparar.

### Cargas de trabajo multiturno y agénticas

`max benchmark` puede simular conversaciones de varios turnos y bucles de agentes con herramientas:

- **`--fit-distributions`**: remodela cargas multiturno usando los flags `random_*` y `--delay-between-chat-turns`. Requiere `--num-chat-sessions` y un dataset compatible (`instruct-coder`, `agentic-code`, `nemotron-opencode`).
- **`--agentic-tool-profiles`**: añade un bucle de agente tras cada turno humano, sintetizado a partir de distribuciones por herramienta (no replay). Requiere `--fit-distributions` y `--agentic-rounds-per-turn`. Acepta un archivo YAML o un mapeo YAML/JSON en línea.
- **`--delay-between-chat-turns`**: retardo entre turnos de chat en milisegundos; acepta una constante o una distribución.

## 3. Ejecuta, guarda resultados y añade estadísticas

Para la latencia de una sola petición en el mejor caso, fija la concurrencia a 1 y mantén la forma fija:

```bash
pixi run max benchmark --backend modular --base-url http://localhost:8000 \
  --model <served-model-name> --endpoint /v1/completions \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --max-output-len 128 --num-prompts 32 --max-concurrency 1 \
  --result-filename results/latency.json
```

Para el throughput máximo y el codo de latencia, haz un sweep de concurrencia y envía más prompts:

```bash
pixi run max benchmark --backend modular --base-url http://localhost:8000 \
  --model <served-model-name> --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --max-output-len 128 --num-prompts 200 \
  --max-concurrency 1,2,4,8,16,32 \
  --result-filename results/throughput.json
```

Ten en cuenta estos puntos sobre el guardado y la instrumentación:

- **`--result-filename`**: escribe las métricas en JSON y crea los directorios necesarios. Úsalo siempre que quieras rastrear o comparar ejecuciones; sin él, MAX no guarda nada. `--metadata key=value` marca el JSON (por ejemplo, `--metadata tp=1 gpu=b200`). Un sweep también deja un `results-<N>-median.json` por paso bajo `--log-dir`.
- **Estadísticas de GPU**: `--collect-gpu-stats` está **habilitado por defecto**; recoge utilización de GPU y memoria pico. Funciona en NVIDIA y, desde v25.7, también en **AMD**. Solo funciona si el benchmark se ejecuta en la **misma máquina** que el servidor. Desactívalo con `--no-collect-gpu-stats`. También hay `--collect-cpu-stats` y `--collect-server-stats`, habilitados por defecto.
- **Sweep benchmarking**: desde la nightly v26.3, un sweep puede **vaciar la caché de prefijos entre ejecuciones** y **recopilar resultados JSON estructurados por ejecución**. Es la forma recomendada de generar una curva latencia-throughput.
- **LoRA**: desde v25.7, `max benchmark` puede ejecutar **benchmarking de LoRA** para modelos y módulos objetivo compatibles (`--lora`, `--lora-paths`, `--lora-uniform-traffic-ratio`, `--per-lora-traffic-ratio`, `--max-concurrent-lora-ops`).
- **Perfilado**: `--profile` captura un trace de Nsight Systems y muestra un resumen de kernels al terminar (el servidor debe estar corriendo bajo `nsys launch`). `--trace` es la alternativa de bajo nivel. Para un análisis más profundo, usa la skill `mojo-max-profile-model`.
- **Configuración YAML**: en lugar de pasar todos los flags, puedes usar `--workload-config` (opciones de carga de trabajo: `num-prompts`, `seed`, etc.) o `--config-file` (opciones generales del benchmark). Los flags de CLI sobrescriben los valores del archivo.

## 4. Lee las métricas

La ejecución imprime throughput y latencia, y un sweep imprime una fila por punto. Los números principales son:

- **Output token throughput (tok/s)**: la cifra principal de throughput.
- **TTFT (time to first token)**: capacidad de respuesta del prefill. Vigila p50 y p99.
- **TPOT e ITL (time per output token y latencia inter-token)**: velocidad de decodificación.
- **Utilización de GPU y memoria pico**: se reportan con `--collect-gpu-stats` (habilitado por defecto).
- **Estadísticas de CPU y servidor**: se reportan por defecto con `--collect-cpu-stats` y `--collect-server-stats`.

Para saber cómo convertir estas cifras en una conclusión y el compromiso latencia-throughput que revela un sweep, consulta `references/metrics.md`.

## Solución de problemas

Haz coincidir el síntoma con `references/troubleshooting.md`, que cubre fallos de conexión, desajustes de nombre de modelo, errores de alias de tokenizer, 400 por plantilla de chat, throughput plano por un tope de batch size, tiempos de primer token sesgados por warmup, problemas con LoRA y estadísticas de GPU. Confirma que `curl /v1/health` devuelve 200 antes de comprobar cualquier otra cosa.

## Cambios de versión que afectan a esta skill

| Cambio | Versión | Impacto |
|--------|---------|---------|
| Añadido el subcomando `max benchmark` | v25.6 | Origen de la herramienta |
| `--collect-gpu-stats` soporta AMD | v25.7 | Ya no es solo NVIDIA |
| Benchmarking de LoRA | v25.7 | Nuevos flags `--lora`, `--lora-paths`, etc. |
| `--do-penalties` → `--enable-penalties` | v25.7 | Afecta al servidor, no al cliente, pero puede cambiar los resultados |
| Sweep benchmarking con flush de caché de prefijos y JSON por ejecución | v26.3 (nightly) | Curvas latencia-throughput más fiables |
| `--collect-gpu-stats`, `--collect-cpu-stats` y `--collect-server-stats` por defecto | Actual | Ya no hay que activarlos explícitamente |
| `--seed` para reproducibilidad | Actual | Mismo workload entre ejecuciones |
| Cargas multiturno y agénticas | Actual | `--num-chat-sessions`, `--fit-distributions`, `--agentic-tool-profiles` |
| Perfilado con Nsight Systems | Actual | `--profile`, `--trace` |

