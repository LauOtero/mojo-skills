# Lectura de los resultados de `max benchmark`

Una ejecución de benchmark imprime las métricas de abajo, y un sweep de `--max-concurrency` o `--request-rate` imprime una fila por punto. Esta página te muestra cómo convertir esos números en una conclusión.

## Las métricas

Cada ejecución reporta las siguientes métricas:

- **Request throughput**: peticiones completadas por segundo. Ayuda sobre todo en cargas de trabajo de respuesta corta; para generación, importa más el output token throughput.
- **Input token throughput**: tokens de entrada procesados por segundo, que mide el trabajo de prefill.
- **Output token throughput (tok/s)**: tokens generados por segundo, agregados entre todas las peticiones concurrentes. Es la cifra principal de throughput.
- **TTFT (time to first token)**: el tiempo desde el inicio de la petición hasta el primer token, que es la latencia de *prefill*, o lo rápido que responde el modelo antes de empezar a emitir. Vigila p50 y p99, porque p99 es lo que sienten los usuarios de la cola.
- **TPOT (time per output token)**: el tiempo medio de decodificación por token después del primero. `1 / TPOT` aproxima los tok/s de decodificación por flujo.
- **ITL (inter-token latency)**: el tiempo entre tokens o fragmentos consecutivos. Se parece mucho a TPOT, y su p99 expone paradas y jitter en el streaming.
- **Utilización de GPU y memoria pico de GPU**: se reportan con `--collect-gpu-stats` (habilitado por defecto). La consola las imprime en la misma tabla de percentiles que la latencia y el throughput; el grupo `gpu_stats` del JSON guarda las listas por GPU. Una utilización baja con alta concurrencia significa que algo distinto del cómputo está haciendo cuello de botella, como el batching, la tokenización o el trabajo de host.
- **Estadísticas de CPU y servidor**: se reportan por defecto con `--collect-cpu-stats` y `--collect-server-stats`.

Las cargas de trabajo multiturno añaden una tasa de tokens cacheados por turno y una retención de caché KV por turno, que muestran la efectividad de la caché de prefijos entre turnos.

En el JSON de resultados, los campos agregados `output_throughput` y `total_token_throughput` pueden contener `null` para una ejecución de un solo flujo (concurrencia 1). Lee el `mean_output_throughput` por flujo en su lugar, o deriva el agregado de `total_output_tokens / duration`.

## Qué te dice un sweep sobre latencia y throughput

Las ejecuciones de un solo flujo (`--max-concurrency 1`) dan la latencia en el *mejor caso*, sin colas ni contención de batching. Cita ese número para "qué tan rápida es una petición", pero recuerda que deja la GPU infrautilizada.

A medida que sube la concurrencia, el servidor agrupa peticiones, así que el output token throughput sube mientras la latencia por petición se degrada. TTFT e ITL suben porque cada petición comparte la GPU y puede encolarse. Un sweep traza esa curva y te da tres lecturas útiles:

- **Throughput máximo**: el output tok/s más alto del sweep, y la concurrencia donde se estanca. Pasado el estancamiento, añades latencia sin ganar throughput.
- **El codo de latencia**: la concurrencia donde TTFT e ITL p99 empiezan a subir empinadamente. Para un SLA como "TTFT p99 por debajo de 500 ms", elige la concurrencia más alta que aún lo cumpla, lo que te da el mejor throughput dentro del presupuesto.
- **Throughput plano en todo el sweep**: una bandera roja. El `--max-batch-size` del servidor está por debajo del sweep, así que las peticiones se encolan en lugar de agruparse. Súbelo y vuelve a servir.

## Obtener números fiables

Sigue estas prácticas para que tus números se sostengan:

- **Calienta**: haz benchmark solo después de `Server ready`, y después de que hayan pasado unas cuantas peticiones. La primera petición paga costes de una sola vez, como el cebado de la caché, que distorsionan el TTFT.
- **Fija la semilla** (por defecto `24301`) y la forma de la carga de trabajo para que las ejecuciones sigan siendo comparables. Usa `--dataset-name random` con `--random-input-len` y `--random-output-len` explícitos para la comparación más limpia.
- **Envía suficientes prompts**. Muy pocos hacen que los promedios sean ruidosos. 32 prompts sirven para una comprobación rápida de latencia, y 200 o más dan números de throughput estables.
- **Guarda y marca la ejecución**: pasa `--result-filename run.json --metadata gpu=b200 tp=1 build=<nightly>` para que una ejecución se describa a sí misma y puedas compararla después.
- **Aísla la variable**. Cuando compares dos configuraciones (dtype, batch size o grado de TP), cambia una cosa a la vez y mantén la carga de trabajo idéntica.

## Guía rápida de decisión

Haz coincidir tu pregunta con una carga de trabajo:

- **"¿Qué tan rápida es una sola petición?"** Ejecuta `--max-concurrency 1` con una forma `random` fija y reporta TTFT p50/p99 y TPOT.
- **"¿Cuál es el máximo que puede empujar esta GPU?"** Haz un sweep `--max-concurrency 1,2,4,8,16,32` y reporta el output tok/s máximo y la concurrencia que necesita.
- **"¿Aguanta mi SLA bajo carga?"** Ejecuta el mismo sweep y lee la concurrencia más alta cuyo TTFT e ITL p99 aún cumplan el objetivo.
- **"¿Cómo se comporta con tráfico real?"** Ejecuta `--dataset-name sharegpt` con concurrencia moderada.
