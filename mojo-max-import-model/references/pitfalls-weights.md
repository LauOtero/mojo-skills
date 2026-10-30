# Trampas del adaptador de pesos

Esta página cubre las trampas de la Fase 2 que surgen mientras escribes `weight_adapters.py`, mapeas claves de safetensors de HF a los nombres de parámetro del módulo raíz y manejas tensores BF16. Describe estas trampas:

- Los nombres de pesos siguen las rutas de atributo del módulo raíz
- Los embeddings atados conservan una sola copia del peso compartido
- Los tensores del checkpoint no consumidos cargan sin error
- Los desajustes de dtype lanzan a menos que `auto_cast` lo permita
- `numpy.from_dlpack` no soporta bfloat16
- El número de filas del embedding puede exceder `vocab_size`

## Los nombres de pesos siguen las rutas de atributo del módulo raíz

`compile(weights=...)` busca cada parámetro por su ruta de atributo desde el módulo raíz que `_instantiate_module()` devuelve. Cuando el raíz guarda su modelo de texto como `self.language_model`, cada peso del modelo de texto se nombra `language_model.<...>`, y el adaptador mapea `model.layers.` de HF a `language_model.layers.` (`olmo3/weight_adapters.py`). Imprime los nombres esperados desde el módulo construido perezosamente:

```python
from max.experimental import functional as F

with F.lazy():
    model = MyModel(config, kv_params)
expected = {name for name, _ in model.parameters}
```

HF trae `lm_head.weight` en el nivel superior. Mapea a donde tu módulo raíz guarde el head (`language_model.lm_head.weight` en `olmo3`).

## Los embeddings atados conservan una sola copia del peso compartido

Cuando `config.tie_word_embeddings=True`, no crees un parámetro `lm_head` y computa los logits del peso del embedding en `forward()`:

```python
if self.tie_word_embeddings:
    logits = h @ self.embed_tokens.weight.T
else:
    logits = self.lm_head(h)
```

El state dict contiene entonces un tensor, `embed_tokens.weight`. Si el checkpoint también trae `lm_head.weight` para un modelo atado, descártalo en el adaptador.

## Los tensores del checkpoint no consumidos cargan sin error

`compile(weights=...)` lanza `KeyError` para un parámetro sin tensor y `ValueError` para un desajuste de forma o dtype. Un tensor del checkpoint cuyo nombre no coincide con ningún parámetro se omite sin warning. Eso oculta estos bugs:

- Un typo en un renombrado envía un tensor real a un nombre que nada lee, mientras otro renombrado llena el parámetro previsto con un tensor de la misma forma.
- El checkpoint trae un componente que el módulo nunca construyó (pesos QK-norm, un experto compartido, sesgos de atención), así que el port computa sin él.

Ejecuta la auditoría de claves no consumidas en [state-dict-audit.md](state-dict-audit.md) antes de servir.

## Los desajustes de dtype lanzan a menos que `auto_cast` lo permita

El dtype de un parámetro viene del dtype por defecto del módulo (`_module_default_dtype()`, el dtype del encoding a menos que se sobrescriba). Un tensor del checkpoint en otro dtype falla `compile()` con `Loaded tensor (shape=..., dtype=...) not assignable to parameter`. Castea en el adaptador, o sobrescribe `_prepare_state_dict()` en `model.py`. `ModuleV3PipelineModelWithKVCache.load_model()` llama a `compile()` sin `auto_cast`, así que un port de pipeline necesita uno de estos casts. Cuando llamas a `compile()` tú mismo, `compile(auto_cast=True)` castea entre `float32` y `bfloat16`.

## `numpy.from_dlpack` no soporta bfloat16

`np.from_dlpack(weight_data.data)` se cae con `RuntimeError: Unsupported dtype in DLTensor` cuando el buffer subyacente es bfloat16 (o fp8, fp4, etc.). NumPy no tiene dtype bfloat16 nativo. Para manipular pesos BF16 en un adaptador de pesos (slicing, reshaping, repacking), ve el buffer como `uint16`, que tiene el mismo ancho de byte, transfórmalo en NumPy, y vuelve a ver el resultado como bfloat16 (`step3p5/weight_adapters.py`):

```python
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData

buf = Buffer.from_dlpack(weight_data.data)
if weight_data.dtype == DType.bfloat16:
    buf = buf.view(DType.uint16)
arr = buf.to_numpy()[:vocab_size].copy()
new_wd = WeightData(
    Buffer.from_numpy(arr).view(weight_data.dtype, list(arr.shape)),
    name,
    weight_data.dtype,
    Shape(arr.shape),
)
```

`WeightData.shape` es un `max.graph.Shape`. Envuelve las dimensiones en `Shape(...)`. Un reordenamiento de filas o columnas no necesita matemática de flotantes, así que la vista `uint16` lo lleva sin cambiar un bit. Para aritmética sobre los valores, castea primero con `weight_data.astype(DType.float32)`.

## El número de filas del embedding puede exceder `vocab_size`

Algunos modelos de HF dimensionan su `nn.Embedding` más grande que `config.vocab_size` para hacer sitio a tokens especiales de imagen/tile/audio que el LM head no predice. Mllama trae `nn.Embedding(vocab_size + 8, hidden_size)` con `lm_head: Linear(hidden_size, vocab_size)`. El embedding tiene 128264 filas, y `lm_head` tiene 128256. Un `Embedding` de MAX dimensionado a `vocab_size` falla la comprobación de forma con pesos de 128264 filas. El adaptador de pesos debe recortar las primeras `vocab_size` filas de `embed_tokens.weight`. Las filas extra son tokens especiales reservados que nunca aparecen en generación solo de texto, así que truncar es seguro para ese alcance. Usa el patrón de vista `uint16` de arriba, porque NumPy no tiene dtype BF16.

