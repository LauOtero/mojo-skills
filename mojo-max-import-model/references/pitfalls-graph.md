# Trampas de construcción del módulo

Esta página cubre las trampas de la Fase 2 que surgen mientras escribes `<slug>.py` y la ruta `forward()` de ModuleV3: llamadas `F.*`, constantes, cableado de RoPE, colocación residual y compartición de subgrafos. Describe estas trampas:

- El scaffold no es un port
- `F.constant` usa por defecto el acelerador y bfloat16
- El relleno de RoPE parcial es intercalado
- `F.sum` conserva la dimensión reducida
- Residual de pila vs bloque en arquitecturas recurrentes / de pesos compartidos
- Un subgrafo compartido asume firmas de capa uniformes

## El scaffold no es un port

`scaffold.py` subclasifica un donante (`llama3_modulev3`, `olmo3`, …). Hasta que implementes el módulo, `<slug>.py` ejecuta la **atención, el cableado del bloque y el MLP del donante**, no los de tu modelo. Servirlo y ejecutar la verificación de logits fallará. Eso es lo esperado, no un bug de tolerancia o del script. Completa [implement-graph.md](implement-graph.md) antes de cualquier actividad de verificación.

## `F.constant` usa por defecto el acelerador y bfloat16

`F.constant(value, dtype=None, device=None)` coloca la constante en un acelerador cuando hay uno presente, y un escalar Python sin `dtype` se convierte en `bfloat16` ahí. Un epsilon o escala creado de esa forma pierde precisión, y una constante que alimenta una entrada del lado CPU (un índice de capa de caché KV) aterriza en el dispositivo incorrecto. Pasa ambos argumentos:

```python
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

x = Tensor.ones([2, 4], dtype=DType.float32)

# Incorrecto: bfloat16 en la GPU
eps = F.constant(1e-6)

# Correcto
eps = F.constant(1e-6, DType.float32, device=x.device)
layer_idx = F.constant(0, DType.uint32, device=CPU())
```

En una malla, pasa la malla (`device=x.mesh`) para que la constante se replique en cada dispositivo.

## El relleno de RoPE parcial es intercalado

El `RotaryEmbedding` base produce pares de frecuencia intercalados: `[cos0, sin0, cos1, sin1, ...]`. Cuando `partial_rotary_factor < 1.0`, rellena la sección no rotada con pares identidad intercalados `[1, 0, 1, 0, ...]`, no split-half `[1, 1, ..., 0, 0, ...]`. El relleno incorrecto produce errores dependientes de la posición que crecen con la longitud de secuencia.

## `F.sum` conserva la dimensión reducida

En MAX, `F.sum(x, axis=-1)` sobre un tensor `[N, H, D]` devuelve `[N, H, 1]`, mientras que PyTorch devuelve `[N, H]`. `Tensor.sum()` y `F.mean()` se comportan igual. Haz squeeze después si necesitas rango 2.

## Residual de pila vs bloque en arquitecturas recurrentes / de pesos compartidos

Algunos modelos ejecutan la misma pila transformer varias veces por forward (bucles estilo HRM, pilas de pesos compartidos, algunos encoder-decoders). Estos modelos mezclan un estado de inyección en la pila **una vez antes del primer bloque**, no en cada bloque dentro del bucle.

Patrón de HF (correcto):

```python
def forward(self, x, injection, ...):
    x = x + injection
    for layer in self.layers:
        x = layer(x, ...)
    return self.final_norm(x)
```

Error común de port:

```python
for cycle in range(num_cycles):
    for layer in self.layers:
        z = layer(z + injection, ...)  # inyección añadida en cada bloque
```

Síntoma: las normas de estado oculto de MAX crecen superlinealmente con la profundidad mientras que HF crece linealmente. El texto greedy diverge desde el token 1–2 incluso cuando los pesos cargan limpiamente. Arreglo: mueve `z = z + injection` fuera del bucle interno de capas (una vez por invocación de la pila). Ver también [implement-graph.md](implement-graph.md#pilas-recurrentes--de-pesos-compartidos-mezcla-la-inyección-una-vez).

## Un subgrafo compartido asume firmas de capa uniformes

`as_subgraph(layer, name=...)` de `max.experimental.nn` traza un subgrafo por nombre y firma de entrada, y lo llama desde cada capa que use ese nombre, para que el compilador procese una vez un bloque repetido. Cada llamada lee sus propios pesos por nombre. El cuerpo viene de la **primera** llamada, incluidos los tamaños de pesos de esa capa y cualquier valor Python incrustado en el trazado. Una llamada posterior con tipos de entrada distintos obtiene un subgrafo separado. Una capa posterior cuyos pesos difieran de los de la primera capa rompe el cuerpo compartido:

- Distintos nombres de pesos hacen que `compile()` lance `ValueError` con `unable to look up weight by name: layers.1.<...>`.
- Distintas formas de pesos, o un valor Python por capa distinto, compilan y corren sin error, y la capa computa valores incorrectos.

Las arquitecturas con variables por capa chocan con esto cuando cada capa comparte un mismo nombre:

- Conteo de heads variable por capa (p. ej. ``num_attention_heads_per_layer = [48, 64, 48, 64, ...]``)
- Atención mixta completa/deslizante con distinto last-dim de `freqs_cis` (RoPE parcial 0.5 vs. RoPE completa 1.0)
- MLP denso/disperso mixto por capa (capa 0 densa + resto dispersas)

Nombra el subgrafo según la firma de la capa:

```python
for i, layer in enumerate(self.layers):
    kind = f"{heads_per[i]}_{layer_types[i]}_{mlp_types[i]}"
    h = as_subgraph(layer, name=kind)(h, ...)
```

`nemotron_h_modulev3/nemotron_h.py` nombra sus subgrafos por tipo de capa de la misma forma. Una capa que incrusta un valor por capa en el trazado (como un índice de capa de caché KV creado con `F.constant`) no puede compartir subgrafo con otras capas. Llama a esa capa directamente.
