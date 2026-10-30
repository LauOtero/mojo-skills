
# Solución de problemas de benchmark

`max benchmark` es un cliente que maneja un endpoint `max serve` activo, así que la mayoría de los fallos se reducen a dos cosas: el cliente no puede alcanzar el servidor, o la carga de trabajo no coincide con cómo arrancaste el servidor. Confirma que `curl http://localhost:8000/v1/health` devuelve 200 antes de perseguir cualquier otra cosa, porque un servidor caído o aún compilando explica la mayoría de los resultados cero o basura.

La siguiente tabla relaciona cada síntoma con su causa y su solución:

| Síntoma | Causa | Solución |
|---|---|---|
| `Connection refused` o no se alcanza el servidor | No hay nada sirviendo, o host/puerto incorrectos | Confirma que `curl /v1/health` devuelve 200; revisa `--host`, `--port` y `--base-url` contra el servidor en ejecución |
| Todas las peticiones fallan, 404, o modelo no encontrado | `--model` no coincide con el servidor | Lee el nombre servido de `curl /v1/models` y pásalo como `--model`; debe ser igual a `--served-model-name` |
| `<name> is not a valid model identifier` | El nombre servido es un alias en lugar de un ID de Hugging Face, así que `--tokenizer` tomó ese alias por defecto y no puede cargarlo | Pasa `--tokenizer <ID real de Hugging Face>` explícitamente |
| Las peticiones devuelven 400 en `/v1/chat/completions` | El modelo no tiene plantilla de chat | Usa `--endpoint /v1/completions` para LMs base, o sirve con `--chat-template` |
| El throughput se mantiene plano al subir la concurrencia | El `--max-batch-size` del servidor está por debajo del sweep, así que las peticiones se encolan en lugar de agruparse | Vuelve a servir con `--max-batch-size` igual o superior al máximo del sweep |
| Los tiempos de primer token salen muy altos o inestables | Hiciste benchmark durante la compilación o el warmup | Espera a `Server ready`, deja pasar unas cuantas peticiones y entonces lanza el benchmark |
| `--collect-gpu-stats` no reporta nada | El benchmark no está en la misma máquina que el servidor, o la GPU no es NVIDIA ni AMD | Ejecuta el benchmark en la máquina del servidor; las estadísticas de GPU funcionan en NVIDIA y, desde v25.7, también en AMD |
| Los resultados de LoRA no aparecen | El modelo no soporta LoRA, o los flags `--lora`/`--lora-paths` no son correctos | Verifica que el modelo y los módulos objetivo sean compatibles; consulta `references/flags.md` para los flags de LoRA |

## Lectura de una ejecución fallida

Tres patrones explican la mayoría de las ejecuciones malas:

- Un throughput cero o casi cero significa casi siempre que las peticiones nunca tuvieron éxito. Revisa el log del servidor y confirma `--model` y `--endpoint` primero, no los flags de carga de trabajo.
- Números que oscilan entre ejecuciones apuntan a warmup o a una máquina compartida y cargada. Fija la semilla (por defecto `24301`), calienta el servidor con unas cuantas peticiones y mantén la forma de la carga de trabajo idéntica entre ejecuciones. Consulta [flags.md](flags.md) y [metrics.md](metrics.md).
- Un sweep en el que todos los puntos parecen iguales significa que la concurrencia no está subiendo realmente, porque el `--max-batch-size` del servidor la limita. Súbelo y vuelve a servir.
