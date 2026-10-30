# Trampas universales (índice)

Encuentra tu síntoma abajo, luego carga el único archivo de categoría al que apunta.

## Configuración y registro → [pitfalls-config.md](pitfalls-config.md)

- [`arch.py::name` debe coincidir exactamente con `architectures[0]`](pitfalls-config.md#archpyname-must-match-architectures0-exactly)
- [`default_encoding` debería coincidir con el dtype del checkpoint del Hub](pitfalls-config.md#default_encoding-should-match-the-hub-checkpoint-dtype)
- [El estimador de memoria dimensiona los pesos a partir de los archivos del checkpoint](pitfalls-config.md#memory-estimator-sizes-weights-from-the-checkpoint-files)
- [`gelu` y `gelu_new` son GELUs diferentes](pitfalls-config.md#gelu-and-gelu_new-are-different-gelus)
- [Trampas de importación y de API de configuración](pitfalls-config.md#import-and-config-api-traps)

## Adaptador de pesos → [pitfalls-weights.md](pitfalls-weights.md)

- [Los nombres de pesos siguen las rutas de atributo del módulo raíz](pitfalls-weights.md#weight-names-follow-the-root-modules-attribute-paths)
- [Los embeddings atados conservan una sola copia del peso compartido](pitfalls-weights.md#tied-embeddings-keep-one-copy-of-the-shared-weight)
- [Los tensores del checkpoint no consumidos cargan sin error](pitfalls-weights.md#unconsumed-checkpoint-tensors-load-without-an-error)
- [Los desajustes de dtype lanzan a menos que `auto_cast` los permita](pitfalls-weights.md#dtype-mismatches-raise-unless-auto_cast-permits-them)
- [`numpy.from_dlpack` no soporta bfloat16](pitfalls-weights.md#numpyfrom_dlpack-does-not-support-bfloat16)
- [El número de filas del embedding puede exceder `vocab_size`](pitfalls-weights.md#embedding-row-count-may-exceed-vocab_size)

## Construcción del módulo → [pitfalls-graph.md](pitfalls-graph.md)

- [El scaffold no es un port](pitfalls-graph.md#scaffold-is-not-a-port)
- [`F.constant` usa por defecto el acelerador y bfloat16](pitfalls-graph.md#fconstant-defaults-to-the-accelerator-and-bfloat16)
- [El relleno de RoPE parcial es intercalado](pitfalls-graph.md#partial-rotary-padding-is-interleaved)
- [`F.sum` conserva la dimensión reducida](pitfalls-graph.md#fsum-keeps-the-reduced-dim)
- [Residual de pila vs. bloque en arquitecturas recurrentes / de pesos compartidos](pitfalls-graph.md#stack-vs-block-residual-in-recurrent--shared-weight-architectures)
- [Un subgrafo compartido asume firmas de capa uniformes](pitfalls-graph.md#a-shared-subgraph-assumes-uniform-layer-signatures)

## Servido y verificación → [pitfalls-serving.md](pitfalls-serving.md)

- [Prueba la decodificación a 16+ tokens, no solo a 1](pitfalls-serving.md#test-decode-at-16-tokens-not-just-1)
- [`trust_remote_code=True` con `.to("cuda")` puede producir NaN](pitfalls-serving.md#trust_remote_codetrue-with-tocuda-can-produce-nan)
- [La comparación de texto es mejor con greedy](pitfalls-serving.md#text-comparison-is-best-with-greedy)
- [La exportación `ARCHITECTURES = [arch]` es obligatoria para `--custom-architectures`](pitfalls-serving.md#architectures--arch-export-is-mandatory-for---custom-architectures)
- [Oráculo HF falso (referencia rota antes de comparar MAX)](pitfalls-serving.md#fake-hf-oracle-broken-reference-before-you-compare-max)