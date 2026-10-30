# Patrones de forma de distribución multi-GPU

La mayoría de los ports de decoder por encima de ~30B BF16 son multi-GPU. La tabla de donantes en [map-to-max.md](map-to-max.md) lista los archs por *forma de atención/MLP* (GQA vs MLA vs MoE). Esta referencia cubre la *forma de distribución* (single-GPU, tensor-parallel, DP+EP). Un port ModuleV3 distribuye colocando el módulo en un `DeviceMesh` y etiquetando los pesos con colocaciones, así que el mismo `forward()` sirve a una GPU o a muchas. Las decisiones son qué ejes de malla usa el port y sobre qué pesos se shardea.

## Regla de decisión (ejecutar en plan-and-veto)

Estima aproximadamente cuánta HBM necesitan los pesos, y luego divide entre la HBM de la GPU objetivo menos el colchón para caché KV + activaciones + overhead de compilación. El colchón es ~30–40% de la HBM en la práctica.

```text
weight_bytes ≈ total_params × bytes_per_param   (BF16 = 2, FP8 = 1, NVFP4 ≈ 0.5)
gpus_needed ≈ ceil(weight_bytes / (gpu_hbm × 0.6))
```

La tabla aplica la regla a una GPU con ~180 GB de HBM:

| Modelo                        | Bytes de pesos | gpus_needed | Forma de distribución              |
|-------------------------------|--------------|-------------|-----------------------------------|
| Llama-3-8B BF16               | ~16 GB       | 1           | Densa single-GPU                  |
| Llama-3-70B BF16              | ~140 GB      | 2–4         | TP multi-GPU                      |
| Mixtral-8x7B BF16             | ~95 GB       | 1–2         | MoE single-GPU (o TP pequeño)     |
| Qwen3-30B-A3B BF16            | ~62 GB       | 1           | MoE single-GPU (o TP por velocidad)  |
| MoE grande (~200B total) BF16 | ~400+ GB     | 4–8         | MoE multi-GPU (DP+EP recomendado) |
| DeepSeek-V3 BF16              | ~1.3 TB      | 8–16        | MoE multi-GPU + MLA               |

La regla falla cuando domina la caché KV (contexto extremadamente largo) y empuja un modelo a multi-GPU incluso si los pesos caben. Si `2 × max_position_embeddings × num_kv_heads × head_dim × bytes_per_element × num_layers × batch ≈ HBM cushion`, recalcula incluyendo la caché KV.

## Mapeo de donantes por forma de distribución

Elige el donante cuya forma de distribución coincida con la tuya, además de su forma de atención/MLP.

| Tu forma                     | Donante                                    | Malla                            | Cuándo usarlo                                                                             |
|------------------------------|--------------------------------------------|----------------------------------|-------------------------------------------------------------------------------------------|
| Densa single-GPU             | `llama3_modulev3`, `olmo3`                 | `.to(self.devices[0])`           | ≈30B BF16 típicamente, una GPU                                                            |
| MoE single-GPU               | `gpt_oss_modulev3`                         | `.to(self.devices[0])`           | MoE que cabe en una GPU                                                                   |
| TP multi-GPU (densa)         | `gemma3_modulev3`                          | 1-D `("tp",)`                    | 70B–200B densa en 2–8 GPUs                                                                |
| TP multi-GPU (MoE)           | `deepseekV3_modulev3`                      | 1-D `("tp",)`                    | MoE donde TP de atención + sharding uniforme de expertos funciona                        |
| **DP + EP multi-GPU** (MoE)  | `deepseekV3_modulev3`, `kimik2_5_modulev3` | 1-D `("dp",)` + `EPBatchManager` | MoE grande donde la atención data-parallel + MoE expert-parallel es el particionado correcto |
| MLA + MoE                    | `deepseekV3_modulev3`                      | `("tp",)` o `("dp",)`            | Familias con KV latente MLA                                                              |

Establece `multi_gpu_supported=True` en `arch.py` para un port que shardee.

## La malla

`model.py` construye la malla y construye el módulo dentro de `default_device(mesh)`, para que cada peso se cree en la malla. El nombre del eje le dice a las capas paralelas sobre qué eje de malla shardear: `"tp"` para tensor parallelism, `"dp"` para data parallelism (`max.experimental.nn.common_layers.mesh_axis`):

```python
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device


def _instantiate_module(self, model_config: MyConfig) -> MyModel:
    n_devices = len(self.devices)
    mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
    with default_device(mesh):
        return MyModel(model_config, self.kv_params)
```

`Module.to()` mueve un módulo a un solo dispositivo y lanza para una malla multi-dispositivo, así que un port shardeado no puede construirse en un dispositivo y moverse después.

`gemma3_modulev3/model.py` construye la malla aquí. `deepseekV3_modulev3/model.py` la construye en `_create_model_config()` y la guarda en la config, para que los módulos que necesitan la malla durante la construcción puedan leer `config.mesh`. Elige el nombre de eje `"dp"` cuando `data_parallel_degree > 1`.

## Capas shardeadas

Etiqueta cada peso con cómo se shardea. Una capa construida bajo `default_device(mesh)` crea cada shard en su dispositivo:

- **Column-parallel** (proyecciones Q/K/V, MLP gate y up, LM head): `ColumnParallelLinear(in_dim, out_dim, bias=False)`, o `col_parallel(Linear(...))` sobre una capa existente. Cada dispositivo guarda un slice de las features de salida.
- **Row-parallel** (salida de atención, MLP down): `RowParallelLinear(...)` o `row_parallel(Linear(...))`. Cada dispositivo guarda un slice de las features de entrada y produce una suma parcial.
- **Embedding vocab-parallel**: `VocabParallelEmbedding(vocab_size, dim=hidden_size)` de `max.experimental.nn.common_layers.embedding`.

Todas estas viven en `max.experimental.nn.common_layers.linear` a menos que se indique lo contrario. `gemma3_modulev3/layers/attention.py` es la referencia para un módulo de atención TP.

## Colectivos

El solver de sharding rastrea la colocación de cada tensor (`Replicated`, `Sharded`, `Partial`) e inserta el colectivo que necesita una op. En `gemma3_modulev3/layers/transformer_block.py`, la salida de atención row-parallel es una suma `Partial`, y la norma post-atención que la consume inserta el all-reduce. El `forward()` del bloque no tiene colectivo explícito.

Llama a un colectivo directamente cuando el port controla dónde ocurre la reducción:

- `F.allreduce_sum(t)`: `Partial` → `Replicated`.
- `F.reduce_scatter(t, scatter_axis=0)`: `Partial` → `Sharded`.
- `F.allgather(t, tensor_axis=-1)`: `Sharded` → `Replicated` (LM head atado en `gemma3_modulev3/gemma3.py`).
- `F.transfer_to(t, mapping)`: cualquier cambio de colocación.

`deepseekV3_modulev3/layers/transformer_block.py` alterna entre estos por modo de paralelismo (atención TP con MoE TP o EP, atención DP con MoE EP).

## Expert parallelism

EP enruta tokens a expertos en otros dispositivos a través de buffers de comunicación NVSHMEM. El modelo de pipeline lo configura en `_init_distributed_runtime()`. Ese método construye un `EPBatchManager` y guarda sus tipos de entrada en `self._modulev3_extra_input_types`, que la clase base añade a las entradas de compilación. También inicializa los buffers de comunicación. El módulo MoE recibe los buffers como entradas `forward()` extra. `deepseekV3_modulev3/model.py` es la referencia.

## Trabajo por dispositivo

Algunos pasos corren independientemente en cada dispositivo, como un gather sobre las filas propias de cada réplica de data parallel. Envuelve la función por dispositivo con `F.functional()`. Sin regla de sharding, se ejecuta en el shard de cada dispositivo. Para llamar a un helper que toma listas por dispositivo, pasa `[TensorValue(s) for s in t.local_shards]` y reconstruye el resultado con `Tensor.from_shard_values(values, mapping)`. `gather_last_tokens()` y `split_replicated_batch()` en `deepseekV3_modulev3/deepseekV3.py` muestran ambos.

## Trampas específicas de MoE multi-GPU

Estos modos de fallo son comunes en ports MoE multi-GPU:

1. **La cuantización selectiva necesita routing por capa.** ``QuantConfig`` lleva conjuntos ``attn_quantized_layers`` y ``mlp_quantized_layers``. Al construir cada bloque, consúltalos:

   ```python
   qc = config.quant_config
   attn_quant_config = (
       qc if qc is not None and layer_idx in qc.attn_quantized_layers else None
   )
   ```

   Sin esto, un modelo que cuantiza solo MoE (dejando la atención en bf16) construye parámetros de atención que esperan pesos FP8 y falla `compile()` en la comprobación de dtype.

2. **El dtype de dispatch difiere del dtype no cuantizado.** Para un modelo FP8, ``config.dtype == DType.float8_e4m3fn`` es el dtype de *dispatch*. Las secciones no cuantizadas (normas, sesgos, embeddings, atención si no está en ``attn_quantized_layers``, la puerta del router MoE) están en bf16 en disco. Sobrescribe `_module_default_dtype()` para devolver `DType.bfloat16` para que esos parámetros se construyan como bf16, como hace `deepseekV3_modulev3/model.py`.

3. **HF envuelve las configs multimodales.** Los tipos de config de visión o generación condicional anidan el backbone de texto bajo ``.text_config``. Los métodos del framework (``calculate_max_seq_len`` en la clase de config, ``get_kv_params`` en la clase del modelo de pipeline) leen la config *padre* por defecto. Sobrescríbelos para pasar ``.text_config``, como hace ``kimik2_5_modulev3/model.py::get_kv_params``.

4. **Desajuste de prefijo en la ignore-list de la config de cuantización.** Las configs compressed-tensors de HF guardan las entradas de ignore con el prefijo original (``model.language_model.layers.X.self_attn.q_proj``). El ``parse_quant_config`` de MAX las comprueba contra ``ignored_modules_prefix``, que por defecto es ``model.`` (``model.layers.X.self_attn.q_proj``). Para un modelo envuelto multimodal, pasa ``ignored_modules_prefix="model.language_model."``, como hace ``gemma4/model_config.py``.

5. **Tablas de RoPE en la malla.** Mueve `freqs_cis` a la malla antes de usarla (`rope.freqs_cis.cast(dtype).to(mesh)`, como hace `Gemma3TextModel.prepare_freq_cis()`). Para capas NoPE, construye una tabla identidad (cos=1, sin=0) junto a la real y selecciónala por capa con el campo que HF comprueba (`layer_types[i]` o `no_rope_layers[i]`, ver [divergences.md](divergences.md#17-nope--skip-rope-layers-via-identity-freqs_cis)). La tabla identidad debe coincidir con el diseño de `rope.freqs_cis`, incluido el número de filas ``max_seq_len * 2`` (el embedding rotatorio pre-asigna 2× para posiciones de decode más allá del prefill).

6. **Zombis de memoria de GPU tras `pkill max serve`.** Un worker de spawn multiprocessing puede sobrevivir a `kill -9` y retener HBM como proceso defunct (estado Z). Síntoma: intentos posteriores de servido ven menos HBM libre de la esperada en cada dispositivo.

