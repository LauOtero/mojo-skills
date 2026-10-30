# Trampas de servido y verificación

Alcance: trampas de la Fase 3 que surgen mientras sirves el port, comparas contra un oráculo HF y ejecutas verificación de texto/logits.

Cubiertas:

- Prueba el decode a 16+ tokens, no solo a 1
- `trust_remote_code=True` con `.to("cuda")` puede producir NaN
- La comparación de texto es mejor con greedy
- La exportación `ARCHITECTURES = [arch]` es obligatoria para `--custom-architectures`
- Oráculo HF falso (referencia rota antes de comparar MAX)

## Prueba el decode a 16+ tokens, no solo a 1

`max_tokens=1` solo ejercita la ruta de prefill. Varios bugs comunes solo aparecen en el bucle de decode:

- `rope_theta` incorrecto → las salidas se degradan con la posición.
- Relleno de RoPE parcial incorrecto → crece con la posición.
- Reset del embedding posicional absoluto → cada paso usa la posición 0.
- Máscara de ventana deslizante incorrecta en decode → el comportamiento cambia tras el tamaño de ventana.

Ejecuta siempre un barrido de longitud antes de declarar terminado un port:

```bash
for n in 1 16 64 200 2000; do
  curl -s http://localhost:8000/v1/completions ... -d "{\"max_tokens\": $n, ...}"
done
```

Busca repetición, colapso de vocabulario (mismo token 100 veces), o tramos sin palabras (ids de token que decodifican a caracteres sueltos).

## `trust_remote_code=True` con `.to("cuda")` puede producir NaN

Algunos modelos personalizados de código remoto construyen cachés de embedding rotatorio en su constructor, en el dispositivo en que se construyeron. El patrón `from_pretrained(...).to("cuda")` deja esas cachés en CPU, y los forward posteriores producen silenciosamente logits todos NaN.

Carga siempre con `device_map="auto"` para modelos `trust_remote_code` en hosts CUDA:

```python
model = AutoModelForCausalLM.from_pretrained(
    "<HF_MODEL_ID>",
    trust_remote_code=True,
    device_map="auto",
    dtype="bfloat16",
)
```

## La comparación de texto es mejor con greedy

Para la comprobación final texto-vs-texto, ambas implementaciones deben usar los mismos ajustes de muestreo. Usa greedy (`temperature=0.0`, `top_p=1.0`) para la comparación. Cualquier temperatura distinta de cero introduce varianza que divergirá entre implementaciones incluso cuando los modelos sean idénticos bit a bit.

## La exportación `ARCHITECTURES = [arch]` es obligatoria para `--custom-architectures`

`pixi run max serve --custom-architectures <port_dir>` carga el paquete del slug: MAX añade `dirname(<port_dir>)` a `sys.path` e importa `basename(<port_dir>)`. Pasa la **carpeta del slug** (contiene `arch.py`), no su padre — pasar el padre importa el nombre de módulo incorrecto y el servido se cae con `AttributeError: module 'custom-arch' has no attribute 'ARCHITECTURES'`.

El `__init__.py` del slug debe exponer una lista de nivel superior:

```python
ARCHITECTURES = [my_arch]
```

## Oráculo HF falso (referencia rota antes de comparar MAX)

Un port puede parecer correcto contra una referencia de Hugging Face que en realidad nunca cargó bien. Para evitarlo:

- Antes de hacer paridad capa por capa, ejecuta una comprobación de cordura de coherencia sobre la referencia HF sola, con la **plantilla de prompt prevista de la model card** (lee la card — los PrefixLM necesitan tokens de prefijo condicional, los modelos instruct necesitan su plantilla de chat, etc.). Si HF mismo produce basura con el prompt de ejemplo de la model card, **no procedas a comparar contra el port** — el oráculo está roto.
- Contrasta el state-dict de HF cargado con las claves de safetensors en disco. Si la clase del modelo declara Q/K/V separados pero el safetensor solo tiene gqkv fusionado, debe haber una entrada correspondiente en `transformers.conversion_mapping._build_checkpoint_conversion_mapping` (o en el `_pre_load_state_dict_hook` del modelo). Si no hay, sube `transformers` o construye una referencia fuse-split manual y verifica que genera texto coherente de forma aislada antes de usarla como oráculo.
- El adaptador y el orden de slicing del port MAX son independientes de esto — deben coincidir con el *diseño en disco*, no con lo que HF eligió dividir. Confírmalo con una prueba empírica puntual: carga el peso fusionado en disco, pártelo `dim=0`, y verifica que cada trozo coincide con el `*_proj.weight` cargado correspondiente en HF.

Las puertas automatizadas pueden pasar mientras el texto greedy sigue divergiendo de HF en la plantilla de la model card — a menudo porque la referencia HF nunca cargó bien (atención con inicialización a cero de QKV no fusionado nunca mapeada). Eso es paridad de oráculo falso, no un port terminado. Una clase distinta de bug es inyectar estado dentro de cada bloque de una pila recurrente en lugar de una vez por invocación de la pila — ver [Stack-vs-block residual en arquitecturas recurrentes / de pesos compartidos](pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures).
