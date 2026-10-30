---
name: mojo-max-profile-model
description: >
  Perfila un modelo que se ejecuta en MAX para encontrar dónde pasa el tiempo y
  si la GPU está saturada. Úsala cuando el usuario pida "perfila mi modelo",
  "dónde pasa el tiempo mi modelo", "por qué es lenta la inferencia", "se está
  usando mi GPU", "cuánta GPU estoy usando", "obtén un desglose de kernels",
  "captura un trace nsys/rocprof/ncu de max serve", o quiera medir el rendimiento
  de inferencia de MAX. Funciona para cualquier modelo que MAX pueda ejecutar —
  arquitecturas integradas y personalizadas cargadas con --custom-architectures —
  desde una instalación pip o pixi (max generate, max serve o un script de
  Python) en GPUs NVIDIA o AMD. Decide de más barato a más caro: primero una
  comprobación de utilización de GPU, luego un desglose de kernels, y solo
  cuando un kernel domine, una inmersión profunda en ese kernel.
compatibility: Requiere una instalación de MAX con pip/pixi y una GPU (NVIDIA o AMD). Los pasos de desglose de kernels e inmersión profunda también necesitan el profiler del fabricante (nsys/ncu para NVIDIA, rocprofv3 para AMD).
argument-hint: "[model] [flags de serve/generate]"
---

# Perfila un modelo en MAX

Esta skill responde tres preguntas sobre cualquier modelo que puedas ejecutar en MAX, ya sea una arquitectura integrada o una personalizada cargada con `--custom-architectures`:

1. **¿Se está usando realmente mi GPU?** (utilización, memoria, relojes)
2. **¿Dónde pasa mi modelo la mayor parte del tiempo?** (qué kernels / fases)
3. **¿Por qué es lento *este* kernel?** (ocupación, stalls, roofline)

Funciona desde una instalación `pip` o `pixi` de MAX, manejando la CLI pública `max` o un script de Python.

## Cómo decidir qué ejecutar

Cada pregunta necesita una captura más profunda y costosa que la anterior. Trabaja de la comprobación más barata hacia la más invasiva, y deja que cada resultado decida si merece la pena profundizar — no captures más de lo que la pregunta necesita.

1. **Empieza por la comprobación de utilización**. Si la GPU está inactiva o poco cargada mientras corre la carga de trabajo, el cuello de botella está en el host (CPU, transferencias, batch pequeño) — detente y repórtalo. Un trace de kernels no te dirá nada que una GPU ocupada no te dijera.
2. **Si la GPU está ocupada, captura un desglose de kernels**. Es el caso común y normalmente la respuesta final: muestra qué kernels dominan el tiempo de GPU.
3. **Solo si un kernel domina y necesitas saber *por qué*, haz una inmersión profunda en ese kernel**. Sáltate este paso a menos que un desglose ya haya apuntado a un kernel concreto — reproduce el kernel muchas veces y es lento.

| Qué estás respondiendo | Herramienta | Coste |
|------------------------|-------------|-------|
| ¿Está ocupada la GPU? | `max.profiler.gpu` (Python puro, NVIDIA + AMD) | segundos, sin instalaciones extra |
| ¿Dónde va el tiempo? | `max ... --profile`, o `nsys` (NVIDIA) / `rocprofv3` (AMD) | un par de ejecuciones perfiladas |
| ¿Por qué es lento un kernel? | Nsight Compute (`ncu`), solo NVIDIA | una captura lenta por kernel |

Lee la referencia del paso en el que estás en lugar de cargarlas todas:

- **Comprobación de utilización** → [`references/utilization-api.md`](references/utilization-api.md) (y el script incluido `scripts/gpu_snapshot.py`)
- **Desglose de kernels, NVIDIA** → [`references/nsys.md`](references/nsys.md)
- **Desglose de kernels, AMD** → [`references/rocprofv3.md`](references/rocprofv3.md)
- **Inmersión profunda en un kernel (NVIDIA)** → [`references/ncu.md`](references/ncu.md)

### Modelos cargados con `--custom-architectures`

El profiling en sí funciona igual para una arquitectura personalizada que para una integrada. Lo que cambia es la carga: una arquitectura personalizada tiene prerequisitos y modos de fallo extra — compatibilidad de imports con tu versión instalada de MAX, trampas de `PYTHONPATH`, colisiones de nombre de arquitectura, modelos base sin plantilla de chat, y una limitación de `max serve` para modelos de embeddings personalizados. Eso impide que el modelo cargue *antes* de que cualquier profiling pueda ejecutarse. Si estás perfilando un modelo pasado vía `--custom-architectures`, lee [`references/custom-architectures.md`](references/custom-architectures.md) **primero**.

## Leer el resultado (qué te dice la mezcla de kernels)

El profiling responde dos cosas: *está saturada la GPU* (la comprobación de utilización) y *dónde va el tiempo* (el desglose de kernels). Relaciona lo que ves con un diagnóstico:

- **Decode dominado por GEMVs `gemv_split_k_*`, con atención `*_mha_*` y `rms_norm` debajo** — esta es la forma normal y sana de una generación de tokens de un LLM *denso* (el decode es GEMV memory-bound). Nada que perseguir.
- **Decode de MoE disperso dominado por matmul agrupado / block-scaled más kernels de routing** — el matmul agrupado por experto (`*grouped*` / `block_scaled_*` matmul) es el kernel dominante en lugar de un GEMM denso, junto con routing top-k (`topk_*`, `moe_create_indices`) y gather/scatter de expertos; en multi-GPU, también aparecen colectivos EP `dispatch`/`combine`. Esta es la forma sana de un modelo Mixture-of-Experts — el overhead de routing y gather es esperado, y solo merece la pena perseguirlo si rivaliza con el propio matmul.
- **Pesos de bajo bit (FP8 / FP4 / NVFP4) añaden kernels de preparación de cuantización** — junto al matmul verás kernels de cuantización dinámica de activaciones y de layout de escalas (`*quantize*`, `*block_scales*`) y variantes de matmul block-scaled en lugar de `gemv`/`gemm` planos. El tiempo en preparación de quant/dequant es normal para un modelo de bajo bit; señálalo solo si empequeñece al matmul al que alimenta. Los *prefijos* de kernel son específicos de la arquitectura de GPU (`sm100_*` en Blackwell, diferentes en Hopper / MI) — haz coincidir por la *familia* del kernel, no por el nombre exacto.
- **Una arquitectura personalizada que *no* alcanza el kernel fusionado esperado** (por ejemplo, atención mostrada como `elementwise`/`matmul` genérico en lugar de un kernel `*_mha_*`, o normas/RoPE sin fusionar) — una señal de cableado: puede que el grafo no esté construido como crees, incluso si los logits pasan. Merece una mirada durante el bring-up.
- **tok/s de decode patológicamente bajos + una mezcla de kernels con forma de prefill repitiéndose en cada paso** — clásico re-prefill O(n²) (sin caché KV o rota). Es un bug adyacente a la corrección, no solo lentitud.
- **`gpu_usage_percent` bajo mientras corre una carga de trabajo** — host-bound, overhead de launch/sync, o batch demasiado pequeño; la GPU no es el cuello de botella.
- **Razones de throttling presentes** — la GPU está limitada por reloj; los números no son representativos hasta que abordes el problema térmico o de energía.

## El algoritmo duradero

Las herramientas y el empaquetado de MAX cambian a menudo; esta forma es la parte que merece la pena preservar. Sea cual sea la captura que ejecutes, síguela:

1. **Detecta el entorno antes de comprometerte con una herramienta**. Ejecuta `nvidia-smi` o `rocm-smi` para conocer el fabricante — las herramientas de desglose de kernels e inmersión profunda difieren por fabricante. Confirma que el profiler está instalado (`which nsys` / `which rocprofv3` / `which ncu`); si falta, dile al usuario la línea de instalación exacta (ver la referencia) en lugar de fallar a mitad de camino.
2. **Confirma antes de cualquier ejecución de larga duración**. Perfilar `max serve` calienta una caché de compilación (la compilación en frío puede tardar minutos), lanza un servidor y ejecuta un benchmark. Antes de hacerlo, muestra al usuario el modelo, los flags y los comandos previstos, y espera confirmación. Un `max generate --profile` de una sola ejecución sobre un modelo pequeño es lo bastante barato como para saltarse esto.
3. **Calienta cachés primero, perfila después**. La primera ejecución de un modelo paga costes de compilación y carga de pesos que ahogan el tiempo real de kernel. Haz una ejecución de calentamiento sin perfilar, *luego* la ejecución perfilada, para que la captura refleje inferencia en estado estacionario — no compilación.
4. **Mantén la ventana de captura pequeña**. Una ventana de 10 segundos a concurrencia 1 es suficiente para ver la mezcla de kernels. Las capturas largas producen archivos de trace enormes que son lentos de abrir y no más informativos.
5. **Falla rápido**. Tras lanzar un servidor, comprueba en unos segundos que el proceso está vivo y que el log no tiene un error temprano (modelo no encontrado, OOM, flag incorrecto) antes de empezar a sondear `/health`. No esperes un timeout completo en un servidor que ya murió.
6. **Verifica el artefacto, luego reporta**. Confirma que el archivo de salida existe y no es trivial en tamaño antes de declarar éxito. Reporta la ruta del artefacto, cómo abrirlo, y los números principales (top kernels, o utilización pico de GPU).
7. **Limpia solo lo que tú lanzaste**. Mata el grupo de procesos del servidor / benchmark que lanzaste. Evita `pkill -f max` amplio en una máquina compartida — puedes detener la ejecución de otra persona.

## Nightly vs stable

Las funciones de profiling llegan a nightly antes que a stable, así que esta skill apunta a la build nightly de `max-serve`. Instálala con pixi:

```bash
pixi init quickstart -c https://conda.modular.com/max-nightly/ -c conda-forge
cd quickstart && pixi add max-serve
pixi run max --version          # espera una build *.dev
```

La API de utilización (`GPUDiagContext`, `BackgroundRecorder`) se distribuye en el paquete **conda** `max`, así que instala vía pixi o conda en lugar de un wheel `pip install max` plano, que puede no exponer `max.profiler.gpu`. Si un comando de abajo da 404 o un import falla, confirma que estás en una build nightly reciente.

> **Nota (MAX 26.7)**: `--model` está estandarizado en `max generate`, `max serve`, `max encode` y `max warm-cache` desde v26.3. Usa siempre `--model`, no `--model-path`. `max serve` no tiene flags CLI para los límites de `kernel_trace_*`; configúralos en la sección `profiling` de un `--config-file` (`ProfilingConfig`).

## Notas de instalación

- MAX en sí: un proyecto con el paquete nightly `max-serve` instalado vía pixi (ver [Nightly vs stable](#nightly-vs-stable) arriba). Todos los comandos `max` de abajo asumen que puedes ejecutar `pixi run max ...` en ese proyecto. El paquete `max-serve` incluye dependencias extra necesarias para servir un modelo con MAX, que no están incluidas en el paquete `max` principal.
- `nsys` y `ncu` vienen con el CUDA Toolkit. Si `which nsys` falla pero CUDA está presente, `export PATH=/usr/local/cuda/bin:$PATH`. En caso contrario, instala Nsight Systems / Nsight Compute desde NVIDIA, o `sudo apt-get install nsight-systems`.
- `rocprofv3` viene con ROCm (`/opt/rocm/bin`). Añádelo al PATH si hace falta.
- La API de utilización no necesita herramienta extra más allá de `max` — habla directamente con NVML / ROCm SMI a través de `max.profiler.gpu`.

## Rutas rápidas

Los comandos de menor fricción, cada uno ampliado en su archivo de referencia:

- **"¿Se está usando mi GPU?"** → ejecuta `scripts/gpu_snapshot.py` junto a un benchmark corto. Python puro, funciona en NVIDIA y AMD.
- **"Dame un desglose de kernels, rápido."** → `pixi run max generate --model <m> --prompt "hello" --num-warmups 1 --profile`. Imprime una tabla top-N de kernels de GPU y escribe un `.nsys-rep` (NVIDIA). Si no hay GPU o `nsys`, cae a un resumen de CPU.
- **"Perfila mi benchmark de servido."** → el flujo de endpoint de servido con `MODULAR_ENABLE_PROFILING=detailed` + `nsys launch` / `rocprofv3`.

## Referencias rápidas de API (MAX 26.7)

| API | Qué hace |
|-----|----------|
| `max.profiler.Tracer` | Gestor de spans anidados; emite rangos NVTX alrededor de bloques. |
| `max.profiler.traced` | Decorador que envuelve una función en un span de profiling. |
| `max.profiler.profiled_region()` | Context manager que delimita la región con `cuProfilerStart/Stop` y imprime un resumen `cProfile`. |
| `max.profiler.cuda_profiler_region` | Delimita una región para `nsys --capture-range=cudaProfilerApi` o `ncu --profile-from-start off`. |
| `max.profiler.maybe_reexec_under_nsys()` | Relanza el proceso bajo `nsys profile` y muestra un resumen top-N de kernels al salir. |
| `max.profiler.gpu.GPUDiagContext` | Context manager que da acceso unificado a NVML / ROCm SMI. |
| `max.profiler.gpu.BackgroundRecorder` | Muestrea métricas de GPU en segundo plano (`interval` por defecto 1.0 s). |
| `max.profiler.gpu.GPUStats` | Snapshot de memoria, utilización y relojes por GPU (`nv0`, `amd0`, …). |
| `max.profiler.cpu.CPUMetricsCollector` | Métricas de CPU por PIDs; `collect_pids_for_port(port)` obtiene los PIDs de un servidor. |
| `ProfilingConfig.gpu_profiling` | `off` / `on` / `detailed`; se configura en el `--config-file` de `max serve`. |
| `ProfilingConfig.kernel_trace_max_*` | Límites de captura de kernels por petición (sin flags CLI; solo `--config-file`). |
