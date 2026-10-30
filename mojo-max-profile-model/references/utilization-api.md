# Snapshot de utilización de GPU (`max.profiler.gpu`)

Úsalo para responder *"¿está la GPU realmente trabajando, o está inactiva / memory-bound / con throttling?"* Es la comprobación más barata: Python puro, sin instalar profiler, y funciona en NVIDIA (vía NVML) y AMD (vía ROCm SMI).

> [!NOTE]
> Requiere el paquete nightly `modular` instalado vía pixi o conda en lugar de un wheel pip plano, que puede no exponer `max.profiler.gpu`. Si el import falla con `ModuleNotFoundError: max.profiler.gpu`, estás en la build equivocada.

## Qué te da la API

`max.profiler.gpu` expone dos puntos de entrada principales:

- **`GPUDiagContext`**: un context manager. `get_stats()` devuelve un snapshot puntual: `{gpu_id: GPUStats}`. Los IDs de GPU llevan prefijo de fabricante (`nv0`, `nv1`, `amd0`, …).
- **`BackgroundRecorder(interval=1.0)`**: muestrea en un proceso en segundo plano mientras corre tu carga de trabajo. Tras salir del bloque `with`, `.stats` es una lista de series temporales de esos snapshots.

Campos de `GPUStats`:

- `memory` (`MemoryStats | None`): `used_bytes`, más total/free/reserved, o `None` cuando el driver no reporta información de memoria (como NVML en DGX Spark).
- `utilization` (`UtilizationStats`): `gpu_usage_percent` (0–100 de cómputo ocupado) y `memory_activity_percent` (actividad del controlador de memoria, o `None` si el fabricante no lo reporta).
- `clocks` (`ClockStats | None`): relojes actuales/máximos y razones de throttling. `HARDWARE_THROTTLE_REASONS` contiene las razones que indican que el hardware limitó la GPU (`hw_power_brake_slowdown`, `hw_slowdown`, `hw_thermal_slowdown`, `sw_power_cap`, `sw_thermal_slowdown`, `sync_boost`).

El número principal para "¿está ocupada mi GPU?" es `utilization.gpu_usage_percent`. Si está pegado cerca de 100, la GPU es el cuello de botella de cómputo; si es bajo mientras corre una carga, el cuello de botella está en otro sitio (CPU, transferencia host-dispositivo, batch pequeño, sincronización).

## Snapshot puntual

```python
from max.profiler.gpu import GPUDiagContext

with GPUDiagContext() as ctx:
    for gpu_id, s in ctx.get_stats().items():
        # used_bytes / 1e9 es GB aproximados (decimal); usa / 2**30 para GiB.
        if s.memory is not None:
            print(
                f"{gpu_id}: {s.utilization.gpu_usage_percent}% cómputo, "
                f"{s.memory.used_bytes / 1e9:.1f} GB usados"
            )
        else:  # el driver no reporta memoria, como NVML en DGX Spark
            print(f"{gpu_id}: {s.utilization.gpu_usage_percent}% cómputo")
```

Esto imprime una línea por GPU, por ejemplo `nv0: 87% cómputo, 14.2 GB usados`.

## Muestrear durante una carga de trabajo

Envuelve la carga de trabajo (por ejemplo, un bucle de generación de `InferenceSession`, o un `time.sleep()` que abarque un proceso de benchmark separado) en un `BackgroundRecorder`:

```python
from max.profiler.gpu import BackgroundRecorder

with BackgroundRecorder(interval=0.5) as rec:
    run_my_inference()  # o sleep mientras `max benchmark` corre en otra terminal

# rec.stats es una lista de snapshots {gpu_id: GPUStats}, uno por intervalo.
```

## Usa el script incluido

`scripts/gpu_snapshot.py` empaqueta el recorder + un resumen para que no tengas que reescribirlo cada vez. Graba durante una duración fija (15 s por defecto) e imprime, por GPU, la utilización de cómputo pico / media, la memoria pico usada y las razones de throttling vistas.

```bash
# Terminal A: arranca el modelo sirviendo (caliéntalo primero — ver
# ../SKILL.md "El algoritmo duradero").
pixi run max serve --model <model>

# Terminal B: lanza una carga corta y graba mientras corre.
pixi run python scripts/gpu_snapshot.py --duration 15 &
pixi run max benchmark --model <model> --backend modular \
  --endpoint /v1/chat/completions --dataset-name sharegpt \
  --num-prompts 50 --max-concurrency 1 --max-benchmark-duration-s 12
```

Para una carga de trabajo autónoma (no de servido), apunta el script a tu script y grabará durante su vida útil:

```bash
pixi run python scripts/gpu_snapshot.py --duration 30 --run "pixi run python my_program.py"
```

## Leer el resultado

- **`gpu_usage_percent` alto (80–100%)** en estado estacionario: la GPU es el cuello de botella. Pasa a un desglose de kernels para ver *qué kernels* dominan.
- **`gpu_usage_percent` bajo con ráfagas**: probablemente overhead de launch/sync, batch pequeño, o trabajo del host entre kernels. Una línea de tiempo de desglose de kernels muestra los huecos.
- **`memory.used_bytes` cerca del total**: estás cerca de OOM; la caché KV / el tamaño de batch pueden ser la restricción real.
- **Razones de throttling presentes** (`hw_thermal_slowdown`, `sw_power_cap`, …): la GPU está limitada por reloj, no por trabajo — los números no serán representativos.
