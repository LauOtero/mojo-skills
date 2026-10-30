# Trampas de configuración y registro

Esta página cubre las trampas de la Fase 1 que surgen mientras lees `config.json`, eliges dtypes, registras la arquitectura y resuelves rutas de importación de MAX. Describe estas trampas:

- `arch.py::name` debe coincidir exactamente con `architectures[0]`
- `default_encoding` debería coincidir con el dtype del checkpoint del Hub
- El estimador de memoria dimensiona los pesos a partir de los archivos del checkpoint
- `gelu` y `gelu_new` son GELUs diferentes
- Trampas de importación y de API de configuración

## `arch.py::name` debe coincidir exactamente con `architectures[0]`

MAX despacha arquitecturas personalizadas haciendo coincidir por string el nombre registrado contra `config.json::architectures[0]`. Si los nombres difieren en un carácter, MAX no selecciona tu registro. Usa una arquitectura integrada registrada con ese nombre si existe, y en caso contrario falla con `ValueError: No architecture found for <name>`.

Los donantes llamados `<arch>_modulev3` registran con sufijo `_ModuleV3` (`LlamaForCausalLM_ModuleV3`) porque conviven con una implementación más antigua del mismo modelo. Tu port registra el valor plano de `architectures[0]` (`olmo3` registra `Olmo3ForCausalLM`), así que no copies el `name=` del donante.

## `default_encoding` debería coincidir con el dtype del checkpoint del Hub

El campo `torch_dtype` de la config del Hub te dice en qué dtype están los pesos publicados. Establecer `default_encoding="float32"` para un checkpoint solo-BF16 fuerza un paso de conversión que puede no existir, o desperdicia memoria. Haz coincidir lo que publica el modelo.

## El estimador de memoria dimensiona los pesos a partir de los archivos del checkpoint

El `MemoryEstimator` de MAX dimensiona los pesos a partir de los tamaños de archivo del checkpoint en disco (`MAXModelConfig.weights_size()`), *antes* de que corra `weight_adapters.convert_safetensor_state_dict`. Si el checkpoint publicado trae tensores FP32 pero tu adaptador los castea a BF16 en tiempo de carga, el estimador sigue contando los bytes FP32. Un modelo BF16 de 25 GiB se pre-estima en ~50 GiB y dispara `--device-memory-utilization 0.5`, aunque los pesos BF16 quepan en el dispositivo.

**Solución:** sube `--device-memory-utilization` (0.7–0.9) para que el estimado del tamaño FP32 quepa. La fracción limita los pesos y la caché KV juntos. Súbela cuando el estimado de pesos no quepa, y bájala cuando el servidor falle al asignar la caché KV al arrancar.

## `gelu` y `gelu_new` son GELUs diferentes

`config.hidden_act` nombra la variante de GELU que usa HF. Búscala en `transformers/activations.py::ACT2FN`, y llama a la función MAX correspondiente:

| `hidden_act` de HF                              | Fórmula                                                    | MAX                              |
|----------------------------------------------|------------------------------------------------------------|----------------------------------|
| `gelu`                                       | `0.5 * x * (1 + erf(x / sqrt(2)))`                         | `F.gelu(x)`                      |
| `gelu_new`, `gelu_pytorch_tanh`, `gelu_fast` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))` | `F.gelu(x, approximate="tanh")`  |
| `quick_gelu`                                 | `x * sigmoid(1.702 * x)`                                   | `F.gelu(x, approximate="quick")` |

Las variantes erf y tanh difieren en menos de 0.001 por elemento, así que una comparación de una sola capa puede pasar con la incorrecta. Haz coincidir el nombre de `config.hidden_act`.

## Trampas de importación y de API de configuración

No mantengas una hoja de trucos paralela de la API. Cuando los imports o los campos pydantic fallen, copia del arch donante del que generaste el scaffold bajo `max/pipelines/architectures/<donor>/` (`arch.py`, `model.py`, `model_config.py`). Ese árbol es la fuente de verdad.

Estos errores son comunes en código copiado de ejemplos o blogs antiguos de MAX:

| Incorrecto                                              | Correcto                                                                                          |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `from max.pipelines.core import PipelineTask`           | `from max.pipelines.modeling.types import PipelineTask`                                           |
| `from max.driver import Tensor`                         | `Buffer`, `Device` de `max.driver`                                                              |
| `pipeline_config.model_config`                          | `pipeline_config.model`                                                                           |
| `pipeline_config.max_length`                            | `pipeline_config.model.max_length`                                                                |
| `pipeline_config.max_batch_size`                        | `pipeline_config.runtime.max_batch_size`                                                          |
| `KVCacheParams(..., cache_strategy=..., n_devices=...)` | Eliminados en MAX actual. Usa solo `kv_cache_config.to_params(...)`                               |
| `from max.nn import Linear, RMSNorm` en el port         | `from max.experimental.nn import Linear`, `from max.experimental.nn.norm import RMSNorm`          |
| `from max.nn import PagedCacheValues`                   | `from max.experimental.nn.common_layers.kv_cache import PagedCacheValues` (tiene `from_upstream()`) |
| `Linear(in_dim, out_dim, dtype=..., device=...)`        | `Linear(in_dim, out_dim, bias=False)`. `model.py` coloca el módulo con `.to()`                    |
| `ModuleList(*layers)`                                   | `ModuleList(layers)`. Toma un iterable y lanza `TypeError` con varios argumentos                  |

`SupportedArchitecture` toma strings planos para `default_encoding` y `supported_encodings` (`"bfloat16"`).

**Pesos en disco:** solo existen `WeightsFormat.safetensors` y `WeightsFormat.gguf` (`max/graph/weights/format.py`). No hay cargador `.bin` / shard PyTorch. Ver el preflight de pesos en [serve-and-iterate.md](serve-and-iterate.md).

**Encoding vs dispositivo** (`max/pipelines/modeling/config_enums.py`):

| Encoding                                   | Dispositivos      |
|--------------------------------------------|-------------------|
| `float32`                                  | `cpu`, `gpu`      |
| `float16`, `bfloat16`                      | solo `gpu`        |
| `float8_e4m3fn`, `float4_e2m1fnx2`, `float6_e2m3fn`, `gptq` | solo `gpu` |
| `q4_k`, `q4_0`, `q6_k`                     | solo `cpu`        |

`DeviceSpec` es solo `"cpu"` o `"gpu"` (Metal usa `gpu` en Apple Silicon).

