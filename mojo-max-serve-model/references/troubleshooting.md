# Fallos de arranque de `max serve`

Lee primero el log de servido. El error real suele estar unas líneas **por encima** del traceback final; el traceback a menudo es solo el marco exterior. Haz coincidir el síntoma de abajo.

| Síntoma (en el log o en la petición) | Causa | Solución |
|---|---|---|
| `... not found in registry` / ninguna arquitectura coincide | El `architectures[0]` del checkpoint no coincide con ningún `SupportedArchitecture.name` registrado | Para arquitectura personalizada: haz que `name=` en `arch.py` sea **exactamente** igual a `config.json::architectures[0]`. Para integrada: la arquitectura no está soportada, así que necesitas un paquete de arquitectura personalizada (tarea de bring-up). |
| `Refusing to override existing architecture for '<Name>'` | El `name=` de tu arquitectura personalizada coincide con el nombre de una arquitectura **integrada**. La documentación dice que la personalizada tiene precedencia, pero algunas versiones de MAX tienen un bug de orden de registro que rechaza la sobrescritura. | Da a tu `SupportedArchitecture` un `name` distinto (por ejemplo `LlamaCustomForCausalLM`) y establece el `config.json::architectures[0]` del checkpoint a ese mismo nombre (en una copia local de la config; enlaza simbólicamente los pesos, no re-descargues). Entonces la arquitectura personalizada es la única coincidencia. Si no puedes editar el paquete, esto es un bloqueante; repórtalo. |
| `TypeError: <Model>.__init__() got an unexpected keyword argument 'max_batch_size'` (u otro kwarg) | **Deriva de paquete**: el paquete se escribió para un MAX más antiguo y su firma `PipelineModel.__init__` está por detrás de la actual. MAX ahora pasa un argumento que el modelo no acepta. Ningún flag de servido evita esto (está en cada ruta de construcción). | Corrección de una línea en el código fuente: haz que el `__init__` del modelo reenvíe, `def __init__(self, pipeline_config, session, *args, **kwargs): super().__init__(pipeline_config, session, *args, **kwargs)`, en **cada** subclase de `PipelineModel` del paquete (una subclase que a su vez herede de otro modelo local necesita la corrección en ambas). Si no puedes editar el paquete, fija la versión de MAX contra la que fue verificado. |
| `AssertionError: Expected max_batch_size to be set` | Algunos modelos personalizados requieren un tamaño de batch explícito; MAX no estableció un valor por defecto. | Pasa `--max-batch-size 1` (súbelo para throughput). El inspector incluido añade esto por defecto. |
| `compatible weights cannot be found` / error de encoding | El `--quantization-encoding` (o el auto-detectado) no está en los `supported_encodings` de la arquitectura, o no coincide con lo que trae el checkpoint | Establece `--quantization-encoding` al formato real del checkpoint (revisa `config.json::torch_dtype` / `quantization_config`); confirma que está listado en `supported_encodings`. |
| Error de `trust_remote_code` al cargar | El repo de HF trae `modeling_*.py` / tokenizer personalizados que el loader debe ejecutar | Añade `--trust-remote-code` (solo para repos en los que confíes). |
| Falta / no hay adaptador de pesos para el formato | `weight_adapters` no tiene entrada para el formato de pesos del checkpoint | Añade una entrada `WeightsFormat.safetensors` (o `.gguf`) al `SupportedArchitecture`. |
| CUDA OOM / fallo de asignación al cargar | Los pesos + la caché KV exceden la memoria del dispositivo | Baja `--max-length`, baja `--max-batch-size`, sube o baja `--device-memory-utilization`, usa menos/más GPUs, o un checkpoint más pequeño/cuantizado. |
| `ModuleNotFoundError` al importar el paquete de arquitectura | La resolución de ruta/módulo de `--custom-architectures` es incorrecta | Usa la forma con dos puntos `IMPORT_PATH:MODULE_NAME` donde IMPORT_PATH es el directorio *padre* en `sys.path` y MODULE_NAME es el paquete (directorio con `__init__.py`). |
| `Address already in use` / fallo al enlazar el puerto | Otro servidor ocupa el puerto | Elige un `--port` libre, o mata el proceso `max serve` obsoleto. |
| Enrutamiento de GPU/dispositivo incorrecto bajo multiproceso | `--devices` y `CUDA_VISIBLE_DEVICES` establecidos ambos | Usa solo `--devices`; desestablece `CUDA_VISIBLE_DEVICES`. |
| El servidor "se cuelga" sin salida | El modelo grande aún está compilando | Busca heartbeats `Still compiling model (Ns elapsed)`; si el contador avanza, espera. Solo un contador *congelado* es un estancamiento real. |
| El servidor está en verde pero la salida es repetición / basura | El servido funciona; la corrección (paridad) del modelo es incorrecta | No es un problema de servido. Es un problema de bring-up/paridad; el grafo o los pesos no coinciden con la referencia. Usa `mojo-max-debug-model`. |
| Petición de `logprobs` rechazada con 400 | La configuración del runtime no puede honrar logprobs | Añade `--allow-unsupported-logprobs` para degradar a warning, o elimina la petición de logprobs. |
| `--rejection-sampling-strategy` no reconocido | El flag fue eliminado en v26.7 | Elimínalo; la decodificación especulativa se configura de otra forma. |
| Variables de entorno `MAX_SERVE_*` ignoradas | El prefijo cambió a `MODULAR_MAX_SERVE_*` en v26.3 | Renombra las variables con el prefijo `MODULAR_`. |

## Marcadores de disponibilidad y fallo que buscar en el log

- Listo: `Server ready`, `Uvicorn running`, `Application startup complete`
- Compilando (sé paciente): `Still compiling model (Ns elapsed)`
- Fallo: `Traceback`, `CUDA_ERROR`, `compatible weights cannot be found`, `not found in registry`

## Comprobaciones de salud

```bash
curl -s http://localhost:8000/v1/health    # 200 cuando está listo
curl -s http://localhost:8000/v1/models     # lista el nombre del modelo servido
```
