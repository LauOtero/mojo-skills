# Inmersión profunda en un solo kernel con Nsight Compute (`ncu`)

Úsalo solo **después** de que un desglose de kernels te haya dicho *qué* kernel es lento. `ncu` responde *por qué*: throughput de cómputo vs memoria, ocupación, razones de warp-stall, tasas de acierto de caché, posición en el roofline. Es solo para NVIDIA y caro — cada lanzamiento de kernel perfilado se reproduce muchas veces, así que apuntas a un kernel, no a una ejecución completa.

Confirma que `ncu` está en PATH (`which ncu`; viene con el CUDA Toolkit, a menudo en `/opt/nvidia/nsight-compute/<ver>/ncu` o `/usr/local/cuda/bin/ncu`). En una máquina compartida, fija los relojes para números reproducibles si tienes permiso, y elige una GPU inactiva (`nvidia-smi`).

Para perfilar un kernel, apunta `ncu` a una ejecución corta de `max generate` y filtra por nombre al kernel de interés.

## Apuntar a un kernel por nombre

Toma el nombre del kernel (o una subcadena distintiva) de la tabla top-N que produjo el desglose de kernels ([nsys.md](nsys.md) en NVIDIA), y usa `--kernel-name-base` / `-k` para filtrar, con `-c` para limitar cuántos lanzamientos se perfilan (mantenlo pequeño — cada uno es lento):

```bash
ncu --target-processes all \
  -k regex:"gemv_split_k" \
  -c 3 \
  --set full \
  -o gemv_profile \
  pixi run max generate \
    --model <model> --prompt "hello" --num-warmups 1 --max-new-tokens 8
```

- `-k regex:"…"`: solo perfila kernels cuyo nombre coincida. Sin esto, `ncu` intenta perfilar *todos* los kernels y la ejecución tarda una eternidad.
- `-c 3`: detente tras 3 lanzamientos coincidentes; un lanzamiento en estado estacionario suele bastar.
- `--set full`: recoge el conjunto completo de métricas (throughput, ocupación, stalls, memoria). Usa `--set basic` para una pasada más rápida y ligera.
- `-o gemv_profile`: escribe `gemv_profile.ncu-rep`.

Una generación corta (pocos `--max-new-tokens`) basta para alcanzar kernels de decode; no necesitas un servidor completo.

> **Nota (MAX 26.7)**: si el proceso usa `cuda_profiler_region` (de `max.profiler`), añade `--profile-from-start off` para que `ncu` grabe solo la región delimitada.

## Leer el resultado

Los nombres entre comillas y backticks de abajo son las etiquetas literales de sección y métrica que reporta `ncu`, para que puedas hacerlas coincidir directamente en la salida.

- **Terminal**: `ncu --import gemv_profile.ncu-rep --page details` (o `--page raw`) imprime las métricas. La sección `GPU Speed Of Light` te dice de inmediato si el kernel es compute-bound o memory-bound.
- **GUI**: abre `gemv_profile.ncu-rep` en la app de Nsight Compute (cópialo a tu portátil si perfilaste en remoto). Las vistas de roofline y de muestreo de warp-stalls son las más accionables.
- Lecturas típicas: `Achieved Occupancy` baja → configuración de lanzamiento / presión de registros o memoria compartida; `Memory Throughput` alto cerca del pico → memory-bound (el estado esperado para el GEMV de decode); la razón de stall dominante apunta al cuello de botella específico (`Stall Long Scoreboard` = latencia de memoria, etc.).

## Fallos comunes

- **La ejecución nunca termina** → olvidaste `-k` y `ncu` está perfilando todos los kernels, reproduciendo cada uno. Filtra por nombre y añade `-c`.
- **"No kernels were profiled"** → la regex de nombre no coincidió. Revisa la subcadena exacta de la tabla de desglose de kernels; los nombres son largos y con plantillas.
- **Errores de permisos / contadores** → NCU necesita acceso a los contadores de rendimiento de la GPU; en nodos bloqueados ejecútalo con permisos elevados o pide al admin que lo habilite.
