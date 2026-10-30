# Desglose de kernels en AMD con `rocprofv3`

Úsalo para responder *"¿dónde pasa el tiempo mi modelo?"* en GPUs AMD (MI300, MI355 y similares). `rocprofv3` captura un trace de kernels más un trace de marcadores (equivalente a NVTX) y escribe un `.pftrace` de Perfetto que abres en <https://ui.perfetto.dev>.

Confirma primero la herramienta: `command -v rocprofv3` y `rocprofv3 --version`. Viene con ROCm bajo `/opt/rocm/bin` — añádelo al PATH si falta. Activa los marcadores de MAX con `MODULAR_ENABLE_PROFILING=detailed` (para que el trace de marcadores etiquete las fases de prefill/decode).

## La captura en tres pasadas

`rocprofv3 --collection-period` toma `delay:duration:rate` — los segundos a esperar antes de grabar, los segundos a grabar, y el multiplicador de muestreo. Abre la ventana de captura *después* de que el servidor esté caliente y sirviendo, así que primero necesitas el tiempo de arranque en caliente. De ahí las tres pasadas:

`R2` abajo es el tiempo de arranque en caliente: los segundos desde lanzar `max serve` hasta que `/health` devuelve 200. La pasada de medida existe solo para encontrarlo.

1. **Pasada de calentamiento** — arranca `max serve`, ejecuta un benchmark corto, apaga. Esto puebla la caché de compilación para que el siguiente arranque sea rápido.
2. **Pasada de medida** — arranca `max serve` de nuevo (sin profiler), cronometra cuánto tarda hasta que `/health` devuelve 200. Ese tiempo es `R2`. Ejecuta un benchmark corto, apaga.
3. **Pasada de profiling** — arranca `max serve` bajo `rocprofv3` con la ventana abriéndose a aproximadamente `R2 + 10` segundos para una captura de 10 segundos. El servidor perfilado y el benchmark corren como **dos procesos concurrentes**: arranca el servidor (bloquea la shell), luego maneja el benchmark desde una segunda shell.

   En la primera shell, arranca el servidor perfilado (`$((R2 + 10)):10:1` es la ventana `delay:duration:rate` — espera `R2 + 10` s, graba 10 s):

   ```bash
   MODULAR_ENABLE_PROFILING=detailed \
     rocprofv3 --kernel-trace --marker-trace \
     --collection-period $((R2 + 10)):10:1 \
     --output-format pftrace -d ./rocprof_out -- \
     pixi run max serve --model <model>
   ```

   En una **segunda shell**, empieza el benchmark ~2 s antes de que se abra la ventana y déjalo correr más allá del final de la ventana de 10 s:

   ```bash
   sleep $((R2 + 8))
   pixi run max benchmark --model <model> --backend modular \
     --endpoint /v1/chat/completions --dataset-name sharegpt \
     --num-prompts 100 --max-concurrency 1 --max-benchmark-duration-s 20
   ```

## Reglas básicas

- Lanza `max serve` **directamente** — no lo envuelvas en `rocprofv3 --attach`, y no uses ninguna indirección `--run_under`. Attach/run-under producen archivos `.pftrace` diminutos o vacíos aquí.
- Mantén la concurrencia del benchmark a 1 — un trace limpio de un solo flujo es mucho más fácil de leer que 600 peticiones concurrentes.
- Ejecuta las tres pasadas en orden; saltarte la pasada de calentamiento significa que la ventana de captura cae durante la compilación, no durante la inferencia.
- Limpia solo el grupo de procesos que lanzaste.

## Verificar y reportar

Después de la ejecución, encuentra el trace y comprueba que es real antes de declarar éxito:

```bash
find ./rocprof_out -name "*.pftrace" -size +1k -exec ls -lh {} \;
```

Un archivo Perfetto válido no es trivial en tamaño y empieza con el magic header de Perfetto. Reporta cada ruta y tamaño de `.pftrace`, y dile al usuario que lo abra en <https://ui.perfetto.dev> (arrastrar y soltar) — la pista de marcadores muestra las etiquetas de fase sobre la línea de tiempo de kernels.

## Fallos comunes

- **`.pftrace` diminuto / ausente** → usaste `--attach` o `--run_under`, o la ventana de recolección se abrió antes de que el servidor estuviera sirviendo. Vuelve a ejecutar el flujo de tres pasadas con `max serve` lanzado directamente y recalcula `R2`.
- **Sin pista de marcadores/fases** → no estableciste `MODULAR_ENABLE_PROFILING=detailed`, u omitiste `--marker-trace`.
