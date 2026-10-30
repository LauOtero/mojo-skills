---
name: mojo-cpu-profiling
description: Perfila código Mojo y MAX en CPU. Cubre max.profiler.cpu (Python/MAX), max.profiler.profiling_range (spans en Mojo), --time-profile y herramientas externas como perf y VTune. Úsala para localizar cuellos de botella de CPU antes de optimizar kernels o algoritmos.
---

# Profiling de CPU en Mojo / MAX 26.7

El profiling de CPU en el ecosistema Mojo/MAX tiene tres frentes: instrumentación con `max.profiler`, profiling de tiempo del compilador y herramientas externas del sistema.

## `max.profiler.cpu` (Python / MAX)

`max.profiler.cpu` recoge métricas de utilización de CPU para un conjunto de PIDs. Es una API de Python, pensada para benchmarking y diagnóstico de MAX, no para Mojo puro.

```python
from max.profiler.cpu import CPUMetricsCollector, collect_pids_for_port

pids = collect_pids_for_port(8000)
collector = CPUMetricsCollector(pids)
metrics = collector.collect()
print(metrics.cpu_time)
```

Úsalo cuando quieras medir el consumo de CPU de un servidor MAX (`max serve`) o de un proceso de benchmarking.

## `profiling_range` (spans en Mojo)

`max.profiler.profiling_range` proporciona un context manager `Range` para instrumentar código Mojo (CPU y GPU) con spans con nombre, visibles en Nsight y en el visor de trazas basado en Kineto.

```mojo
from max.profiler import Range

def my_kernel():
    with Range("preprocess"):
        # código de CPU
    with Range("matmul"):
        # llamada a kernel
```

Los spans de Mojo se enrutan al profiler de MAX y aparecen correlacionados con los kernels de GPU, en lugar de una línea de tiempo de CPU vacía.

## `--time-profile`

`--time-profile` activa el profiling de tiempo en herramientas que lo soportan (por ejemplo, `max benchmark`). Requiere que el build tenga `MODULAR_ASYNCRT_MAX_PROFILING_LEVEL > 0`.

## Herramientas externas

| Herramienta | Uso |
|-------------|-----|
| `perf` (Linux) | `perf record -g ./mi_binario_mojo`; `perf report` |
| VTune (Intel) | Perfilado de CPU y memoria |
| `nsys` (NVIDIA) | Perfilado de CPU + GPU; `nsys profile ./binario` |
| `cProfile` (Python) | Si el código corre vía Python/MAX |

Para Mojo puro, compila con símbolos de depuración (`mojo build --debug`) y usa `perf` o VTune.

## Anti-patrones

- **No uses `max.profiler.cpu` desde Mojo puro**: es una API de Python; úsala desde un script de Python o desde MAX.
- **No perfile en builds de producción**: el overhead del profiling distorsiona los resultados.
- **No optimices sin medir**: el profiling debe preceder a cualquier cambio de algoritmo.
- **No confundas profiling de CPU con profiling de GPU**: `profile-model` cubre GPU; esta skill cubre CPU.

## Reglas prácticas

1. **Empieza con `perf` o VTune** para una vista de bajo nivel de dónde se va el tiempo.
2. **Usa `profiling_range`** para instrumentar secciones específicas de código Mojo.
3. **Usa `max.profiler.cpu`** para medir el consumo de CPU de un servidor MAX.
4. **Combina con `parallelism_level()`** para verificar que el número de workers coincide con lo esperado.
5. **Perfila con el mismo workload que usas en producción**; un microbenchmark puede ocultar cuellos de botella reales.
