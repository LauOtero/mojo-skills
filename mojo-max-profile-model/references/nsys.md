# Desglose de kernels en NVIDIA con Nsight Systems (`nsys`)

Úsalo para responder *"¿dónde pasa el tiempo mi modelo?"* en GPUs NVIDIA. `nsys` captura la línea de tiempo completa — llamadas a la API CUDA, kernels de GPU, operaciones de memoria — y, con los marcadores de profiling de MAX activados, rangos NVTX que etiquetan las fases de prefill / decode / sampling.

Confirma primero que `nsys` está disponible: `which nsys`. Si falta pero CUDA está instalado, `export PATH=/usr/local/cuda/bin:$PATH`; en caso contrario `sudo apt-get install nsight-systems` o descárgalo de NVIDIA. Usa la versión 2024.4.2 o posterior (la que fijan las imágenes de benchmark de MAX); versiones más antiguas de `nsys` tienen problemas conocidos. Compruébalo con `nsys --version`.

## Activar los marcadores de MAX

MAX solo emite marcadores NVTX cuando se lo pides. Prefiere la variable de entorno `MODULAR_ENABLE_PROFILING=detailed`: no necesita cambios de código ni de comando y viaja con el proceso cuando envuelves `max serve` bajo `nsys`. Los valores son `off` (por defecto), `on` (marcadores de correlación de kernels) y `detailed` (añade NVTX a nivel de Python); usa `detailed`.

Dos alternativas sobrescriben la variable de entorno cuando se establecen: el flag CLI `max serve --gpu-profiling detailed`, y `InferenceSession.gpu_profiling("detailed")` en Python (llámalo antes de `load()`).

> **Nota (MAX 26.7)**: `max serve` no tiene flags CLI para los límites de `kernel_trace_*` (`max_passes`, `max_batch_links`, `max_spans`, `max_capture_bytes`). Configúralos en la sección `profiling` de un `--config-file` usando `ProfilingConfig`.

## Ruta rápida: `max generate --profile`

El desglose de menor fricción. `max generate` y `max benchmark` aceptan `--profile`: captura la ejecución cronometrada a un `.nsys-rep` e imprime una tabla top-N de kernels de GPU en la terminal. `cudaProfilerStart/Stop` delimita la captura, así que excluye automáticamente el calentamiento y la compilación.

```bash
pixi run max generate \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --prompt "Explain GPU profiling in one sentence." \
  --max-length 64 --num-warmups 1 \
  --profile
```

La salida termina con una tabla de kernels de GPU más una tabla de Python/CPU. Por ejemplo, en Llama-3.1-8B-GGUF:

```text
=== Top 15 GPU kernels (98.9% of GPU time, 71.35 ms total) ===
     %       total     calls  kernel
 33.6%    23.96 ms       672  gemv_split_k_bfloat16_bfloat16_bfloat16_256_f519d2c2
 20.3%    14.52 ms       651  gemv_split_k_bfloat16_bfloat16_bfloat16_128_e7d08336
 11.2%     8.02 ms       672  gemv_split_k_bfloat16_bfloat16_bfloat16_128_74024e6c
  4.7%     3.38 ms       672  sm100_mha_1q_depth128_bfloat16_bfloat16_nqh32_nkvh8_…
  3.8%     2.70 ms      1430  rms_norm_gpu_warp_tiling_bfloat16_…
  ...
Full profile saved to: /home/.../quickstart/max-profile.nsys-rep
Open with: nsys-ui /home/.../quickstart/max-profile.nsys-rep
```

Los GEMVs `gemv_split_k` dominando el decode, con atención (`sm100_mha`) y `rms_norm` debajo, es la mezcla en estado estacionario esperada para un LLM.

Opciones: `--profile-output PATH` (dónde va el `.nsys-rep`), `--profile-top-n N` (filas en cada tabla, por defecto 15). Sin `nsys` o sin GPU, `--profile` sigue imprimiendo un resumen `cProfile` solo de CPU.

> [!NOTE]
> Requiere la build nightly de `modular` (ver SKILL.md).

Esto suele bastar para responder "qué domina el tiempo de GPU". Usa los flujos manuales de abajo solo cuando necesites una ventana de trace personalizada o estés perfilando un servidor de larga duración.

## Perfilar un script autónomo

```bash
MODULAR_ENABLE_PROFILING=detailed \
  nsys profile --trace=cuda,osrt,nvtx \
  --cuda-memory-usage=true \
  --output=profile --force-overwrite=true \
  pixi run python my_program.py
```

Produce `profile.nsys-rep`.

## Perfilar un endpoint de servido

Usa `nsys launch` (arranca el servidor pero difiere la recolección) + `nsys start` / `nsys stop` alrededor del benchmark, siguiendo [el algoritmo duradero](../SKILL.md#el-algoritmo-duradero) (calienta cachés primero, mantén la ventana corta, falla rápido ante un servidor muerto).

1. Arranca el servidor bajo `nsys launch`. Comprueba primero el número de sockets con `lscpu | grep '^Socket(s):'`. En una máquina multi-socket (más de uno), fija NUMA (`numactl --cpunodebind=N --membind=N`) para números estables; en un nodo de un solo socket, ejecuta `nsys` sin él.

   ```bash
   MODULAR_ENABLE_PROFILING=detailed \
     nsys launch --trace=cuda,nvtx,osrt \
     --cuda-memory-usage=true --trace-fork-before-exec=true \
     pixi run max serve --model <model>
   ```

2. Espera a `🚀 Server ready on http://0.0.0.0:8000`. (Falla rápido: si el proceso muere o el log muestra un error temprano, detente y sácalo a la superficie.)

3. En una segunda terminal en el mismo entorno, empieza la recolección:

   ```bash
   nsys start --force-overwrite=true --output=server_profile \
     --session=$(nsys sessions list -p false | awk '{print $1}')
   ```

4. Ejecuta un benchmark corto a concurrencia 1 (lo bastante largo para abarcar la ventana):

   ```bash
   pixi run max benchmark --model <model> --backend modular \
     --endpoint /v1/chat/completions --dataset-name sharegpt \
     --num-prompts 50 --max-concurrency 1 --max-benchmark-duration-s 12
   ```

5. Detén la recolección:

   ```bash
   nsys stop --session=$(nsys sessions list -p false | awk '{print $1}')
   ```

Produce `server_profile.nsys-rep`.

### Confirmar que los marcadores llegaron

Antes de confiar en la línea de tiempo, verifica que los rangos NVTX de fase están realmente presentes — un perfil capturado sin `MODULAR_ENABLE_PROFILING=detailed` parece válido pero no tiene etiquetas de fase. Compruébalo desde la terminal:

```bash
nsys stats --report nvtx_pushpop_sum server_profile.nsys-rep
```

Una captura correctamente marcada lista rangos NVTX con nombres como `prefill`, `decode` y `sampling`. Un informe vacío (o "no NVTX data") significa que los marcadores no llegaron — vuelve a ejecutar con `MODULAR_ENABLE_PROFILING=detailed` establecido. En la GUI, los mismos marcadores aparecen como una pista de fase etiquetada sobre el flujo de kernels.

## Leer el resultado

- **El resumen de terminal** es lo más rápido: `nsys stats profile.nsys-rep` reimprime la tabla por kernel de cualquier `.nsys-rep`. Para un rollup legible y demangled (nombres sin el ruido de plantillas C++/Mojo — útil para agrupar por familia de kernels), usa `nsys stats --report cuda_gpu_kern_sum:base --timeunit msec profile.nsys-rep`.
- **GUI**: abre en la app de Nsight Systems — `nsys-ui profile.nsys-rep` en local, o `scp` el archivo a tu portátil y haz doble clic. La línea de tiempo muestra los rangos NVTX de fase sobre el flujo de kernels, así que puedes ver si domina el prefill o el decode y dónde están los huecos.
- Un nombre de kernel como `gemv_split_k_*` dominando el decode es esperado (el decode es GEMV-heavy). Si domina un único kernel elementwise/copy, es candidato a una inmersión profunda con `ncu`.

## Fallos comunes

- **El perfil no tiene NVTX / no tiene etiquetas de fase** → olvidaste `MODULAR_ENABLE_PROFILING=detailed` (o `--gpu-profiling detailed`).
- **"Process launch is not allowed in this state"** → una sesión `nsys` obsoleta. `nsys sessions list`; cancela/cierra las sesiones obsoletas antes de relanzar.
- **`.nsys-rep` vacío / diminuto** → el benchmark terminó antes de que se abriera la ventana, o la ventana nunca se abrió. Empieza el benchmark justo después de `nsys start`, y haz el benchmark más largo que la captura.
- **Los kernels aparecen como un blob de grafo opaco, o falta el tiempo por kernel** → la captura de grafo de dispositivo CUDA está activada (el valor por defecto para `max serve`), así que la línea de tiempo muestra *replays* de grafo en lugar de kernels individuales. Sirve/genera con `--no-device-graph-capture` para que `nsys` vea cada kernel (esto es también por lo que `custom-architectures.md` usa ese flag).
