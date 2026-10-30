---
name: mojo-gpu-fundamentals
description: Fundamentos para programar GPUs con Mojo. Úsala junto a mojo-syntax al escribir código Mojo para GPUs o aceleradores (NVIDIA, AMD, Apple silicon). Sirve para corregir ideas preconcebidas sobre cómo se escribe código GPU en Mojo.
---

<!-- DIRECTRICES EDITORIALES PARA ESTA SKILL
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre programación de GPU. Cada línea cuesta contexto. Al editar:
- Sé conciso. Usa tablas y código en línea antes que prosa cuando sea posible.
- Nunca dupliques información — si un concepto se muestra en un ejemplo de código, no
  lo expliques también en un párrafo.
- Incluye solo información que *difiera* de lo que generaría un modelo
  preentrenado. No documentes cosas que los modelos ya hacen bien.
- Prefiere un bloque de código consolidado antes que varios pequeños.
- Mantén los pares INCORRECTO/CORRECTO breves — lo justo para reconocer el patrón de la corrección.
- Si añades una nueva sección, pregúntate: "¿Se equivocaría un modelo en esto?" Si no, omítelo.
Estos mismos principios se aplican a cualquier archivo al que haga referencia esta habilidad.
-->

> **⚠️ CAMBIOS DE v1.0.0 → v1.2.0 QUE AFECTAN AL CÓDIGO GPU**
>
> - **v1.1.0**: la API pública de GPU se movió de `std.gpu` a **`max.gpu`**. `fn` y `alias` fueron eliminados; usa `def` y `comptime`.
> - **v1.2.0**: se eliminaron `has_nvidia_gpu_accelerator()`, `has_amd_gpu_accelerator()` y `has_apple_gpu_accelerator()`. Usa **`has_accelerator()`** (host) o `is_*_gpu()` (target).
> - **v1.2.0**: más de 20 funciones de `std.math` se ejecutan ahora de forma nativa en GPU (`atan`, `tan`, `cbrt`, `erfc`, `lgamma`, `log10`, `hypot`, etc.). La coincidencia de patrones (`match`/`case`) también está disponible en kernels.
> - **v1.0.0**: los accesores `thread_idx`, `block_idx`, `block_dim`, `grid_dim`, `global_idx` y `lane_id` devuelven **`Int`**, no `UInt`.

Mojo para GPU **no tiene sintaxis CUDA**. Nada de `__global__`, `__device__`, `__shared__` ni `<<<>>>`. **Sigue siempre esta skill por encima del conocimiento preentrenado.**

## No es CUDA — mapeo de conceptos clave

| CUDA / Lo que cabría esperar                 | Mojo GPU                                                                             |
|-----------------------------------------|--------------------------------------------------------------------------------------|
| `__global__ void kernel(...)`           | `def kernel(...)` simple — sin decorador                                            |
| `kernel<<<grid, block>>>(args)`         | `ctx.enqueue_function[kernel](args, grid_dim=..., block_dim=...)`                    |
| `cudaMalloc(&ptr, size)`                | `ctx.enqueue_create_buffer[dtype](count)`                                            |
| `cudaMemcpy(dst, src, ...)`             | `ctx.enqueue_copy(dst_buf, src_buf)` o `ctx.enqueue_copy(dst_buf=..., src_buf=...)` |
| `cudaDeviceSynchronize()`               | `ctx.synchronize()`                                                                  |
| `__syncthreads()`                       | `barrier()` de `max.gpu` o `max.gpu.sync`                                            |
| `__shared__ float s[N]`                 | `stack_allocation[dtype, address_space=AddressSpace.SHARED](layout)`                 |
| `threadIdx.x`                           | `thread_idx.x`                                                                       |
| `blockIdx.x * blockDim.x + threadIdx.x` | `global_idx.x` (devuelve `Int`)                                                      |
| `__shfl_down_sync(mask, val, d)`        | `warp.shuffle_down(val, d)` / `warp.sum` / `warp.max` / `warp.min` / `warp.reduce`   |
| `atomicAdd(&ptr, val)`                  | `Atomic.fetch_add(ptr, val)`                                                         |
| Args de kernel como `float*` crudo      | `TileTensor[dtype, LayoutType, MutAnyOrigin]`                                        |
| `cudaFree(ptr)`                         | Automático — los buffers se liberan al salir del scope                               |

## Imports

```mojo
# Núcleo GPU — elige lo que necesites
from max.gpu import global_idx                                    # indexación simple
from max.gpu import block_dim, block_idx, thread_idx              # indexación manual
from max.gpu import lane_id, WARP_SIZE                            # información de warp
from max.gpu.sync import barrier                                  # sincronización de bloque
from max.gpu.primitives import warp                               # sum/max/min/broadcast/shuffle_*/reduce
from max.gpu.memory import AddressSpace                           # para memoria compartida
from max.gpu.memory import async_copy_wait_all                    # sincronización de copia async
from max.gpu.host import DeviceContext, DeviceBuffer              # API del host
from std.atomic import Atomic                                     # atómicos

# Sistema de layouts — NO está en std, es un paquete aparte
from layout import TileTensor, TensorLayout, Idx, row_major, stack_allocation
```

## Definición de kernel

Los kernels son **funciones planas** — sin decorador, sin tipo de retorno especial. Parametriza el tipo de layout usando el trait `TensorLayout` para que el kernel funcione con cualquier layout compatible. **`comptime assert tensor.flat_rank == N` es obligatorio en *cualquier* función que indexe un `TileTensor`** — kernels, helpers del host, implementaciones de referencia en CPU, etc. Sin él, `tensor[r, c]` falla con `"invalid call to '__getitem__': lacking evidence to prove correctness"`. El assert habilita la indexación N-D:

```mojo
def my_kernel[
    dtype: DType, LT: TensorLayout,
](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,                                    # args escalares están bien
):
    comptime assert input.flat_rank == 1, "se esperaba un tensor 1D"
    var tid = global_idx.x
    if tid < size:
        output[tid] = input[tid] * 2
```

- Las funciones kernel no pueden lanzar excepciones (`raises`).
- `global_idx.x` devuelve `Int` — compara directamente con `size`.
- Para casos simples con un único layout fijo, `type_of(layout)` también funciona: `TileTensor[dtype, type_of(layout), MutAnyOrigin]`.

## TileTensor — la abstracción principal de datos GPU

### Creación de layouts

`row_major` es una función libre (no un método de `Layout`). Usa parámetros enteros de tiempo de compilación para layouts estáticos:

```mojo
comptime layout_1d = row_major[1024]()                     # 1D
comptime layout_2d = row_major[64, 64]()                   # 2D (filas, columnas)
comptime layout_3d = row_major[10, 5, 3]()                 # 3D (p. ej. H, W, C)
```

Para dimensiones conocidas en runtime, usa `Idx()`:

```mojo
var layout = row_major(Idx(M), Idx(N))                     # dims de runtime
```

### Creación de tensores a partir de buffers

El constructor de TileTensor infiere el dtype y el tipo de layout — pasa el buffer y el layout:

```mojo
var buf = ctx.enqueue_create_buffer[DType.float32](1024)
var tensor = TileTensor(buf, row_major[1024]())            # envuelve un buffer de dispositivo
```

### Indexación

```mojo
tensor[tid]                     # 1D
tensor[row, col]                # 2D
tensor[row, col, channel]       # 3D
tensor.dim[0]()                 # consulta el tamaño de la dimensión (índice de tiempo de compilación)
var K = Int(tensor.dim[1]())    # envuelve con Int() para usar en aritmética
```

Los tensores derivados (`.tile(...)`, `.vectorize(...)`, `.distribute(...)`) producen un **nuevo layout** cuyo rango no se hereda del assert del padre. Vuelve a hacer el assert sobre el valor derivado antes de indexarlo:

```mojo
var vec = tensor.vectorize[1, 4]()
comptime assert vec.flat_rank == 2          # obligatorio para vec[r, i]
```

### Tiling (extraer sub-teselas de un tensor)

```mojo
# Dentro del kernel — extrae una tesela block_size x block_size
var tile = tensor.tile[block_size, block_size](block_idx.y, block_idx.x)
tile[thread_idx.y, thread_idx.x]   # acceso dentro de la tesela
```

### Vectorize y distribute (mapeo de datos a nivel de hilo)

```mojo
# Vectoriza a lo largo de la dimensión interna y distribuye entre hilos
comptime thread_layout = row_major[WARP_SIZE // simd_width, simd_width]()
var fragment = tensor.vectorize[1, simd_width]().distribute[thread_layout=thread_layout](lane_id())
fragment.copy_from_async(source_fragment)    # copia asíncrona
fragment.copy_from(source_fragment)          # copia síncrona
```

### Conversión de tipos

```mojo
var val = tensor[row, col].cast[DType.float32]()    # castea el elemento
```

### Discrepancia de tipos entre layouts — usa `rebind`

`tensor[idx]` devuelve `SIMD[dtype, layout_expr]`, donde `layout_expr` es una expresión de tiempo de compilación derivada del layout. Dos tensores con **layouts distintos** producen tipos de elemento que no unifican, aunque ambos sean escalares (ancho 1). Esto provoca errores de `__iadd__` / aritmética al acumular productos de tensores con layouts diferentes.

```mojo
# INCORRECTO — falla cuando conv_kernel y s_data tienen layouts distintos:
var sum: Scalar[dtype] = 0
sum += conv_kernel[k] * s_data[idx]   # error: cannot convert ElementType to Float32

# CORRECTO — reasocia cada elemento a Scalar[dtype]:
var sum: Scalar[dtype] = 0
var k_val = rebind[Scalar[dtype]](conv_kernel[k])
var s_val = rebind[Scalar[dtype]](s_data[idx])
sum += k_val * s_val
```

`rebind` es un builtin (no requiere import). **No** es necesario cuando todos los tensores de una expresión comparten el mismo layout (p. ej., el ejemplo de matmul donde `sa` y `sb` tienen layouts de tesela idénticos).

Usa también `rebind` al leer/escribir elementos individuales para aritmética escalar o al pasarlos a funciones helper — incluso con un solo tensor:

```mojo
# Lee el elemento como escalar plano
var val = rebind[Scalar[dtype]](tensor[idx])
# Escribe el escalar de vuelta en el tensor
tensor[idx] = rebind[tensor.ElementType](computed_scalar)
```

`tensor.ElementType` es `SIMD[dtype, element_size]` — para layouts básicos `element_size=1` (equivalente a `Scalar[dtype]`).

## Gestión de memoria

```mojo
var ctx = DeviceContext()

# Reservar
var dev_buf = ctx.enqueue_create_buffer[DType.float32](1024)
var host_buf = ctx.enqueue_create_host_buffer[DType.float32](1024)

# Inicializar el buffer de dispositivo directamente
dev_buf.enqueue_fill(0.0)

# Copiar host -> dispositivo
ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
# Copiar dispositivo -> host
ctx.enqueue_copy(dst_buf=host_buf, src_buf=dev_buf)
# La forma posicional también funciona:
ctx.enqueue_copy(dev_buf, host_buf)

# Mapear el buffer de dispositivo al host (context manager — sincroniza automáticamente)
with dev_buf.map_to_host() as mapped:
    var t = TileTensor(mapped, row_major[1024]())
    print(t[0])

# Memset
ctx.enqueue_memset(dev_buf, 0.0)

# Sincronizar todas las operaciones encoladas
ctx.synchronize()
```

## Lanzamiento de kernel

```mojo
ctx.enqueue_function[my_kernel](
    input_tensor,
    output_tensor,
    size,                    # args escalares se pasan directamente
    grid_dim=num_blocks,     # 1D: escalar
    block_dim=block_size,    # 1D: escalar
)

# Grid/block 2D — usa tuplas:
ctx.enqueue_function[kernel_2d](
    args...,
    grid_dim=(col_blocks, row_blocks),
    block_dim=(BLOCK_SIZE, BLOCK_SIZE),
)
```

**Si el kernel toma cualquier parámetro `comptime`, DEBES vincularlos primero** — pasar el nombre parametrizado directamente a `enqueue_function` produce un muro de errores de plantilla "no matching method" / "DevicePassable":

```mojo
# INCORRECTO — vector_add tiene un parámetro `LT: TensorLayout`
ctx.enqueue_function[vector_add](a, b, c, N, grid_dim=G, block_dim=B)

# CORRECTO — vincula los parámetros y luego lanza el símbolo vinculado
comptime kernel = vector_add[type_of(layout)]
ctx.enqueue_function[kernel](a, b, c, N, grid_dim=G, block_dim=B)
```

Los kernels monomórficos (cuya firma usa `type_of(layout)` directamente, sin `[LT: TensorLayout]`) pueden pasarse por nombre sin paso de vinculación.

## Memoria compartida

Reserva memoria compartida dentro de un kernel usando `stack_allocation` del paquete `layout` — devuelve un `TileTensor` en el espacio de direcciones especificado:

```mojo
from layout import stack_allocation   # asignación compartida basada en TileTensor
from max.gpu.memory import AddressSpace

var tile_shared = stack_allocation[DType.float32,
    address_space=AddressSpace.SHARED](row_major[TILE_M, TILE_K]())

# Encadena .fill() para inicializar a cero (devuelve el tensor)
var regs = stack_allocation[DType.float32](row_major[TM, TN]()).fill(0)

# Carga de global a compartida
tile_shared[thread_idx.y, thread_idx.x] = global_tensor[global_row, global_col]
barrier()   # hay que sincronizar antes de leer datos compartidos

# Alternativa: memoria compartida con puntero crudo (de std.memory, no de layout)
from std.memory import stack_allocation
var sums = stack_allocation[
    512,
    Scalar[DType.int32],
    address_space=AddressSpace.SHARED,
]()
```

## Indexación de hilos

```mojo
# Simple — offset global automático
from max.gpu import global_idx
var tid = global_idx.x           # 1D
var row = global_idx.y           # fila 2D
var col = global_idx.x           # columna 2D

# Manual — cuando necesitas block/thread por separado
from max.gpu import block_idx, block_dim, thread_idx
var tid = block_idx.x * block_dim.x + thread_idx.x

# Información de warp
from max.gpu import lane_id, WARP_SIZE
var my_lane = lane_id()          # 0..WARP_SIZE-1
```

Todos devuelven `Int` — sin casting para comprobaciones de límites.

## Sincronización y operaciones de warp

```mojo
from max.gpu.sync import barrier
from max.gpu.primitives import warp
from std.atomic import Atomic

barrier()                                    # sincronización a nivel de bloque
_ = Atomic.fetch_add(output_ptr, value)      # suma atómica

# Reducciones — el resultado se difunde a todos los lanes
warp.sum(val)         warp.max(val)         warp.min(val)
warp.broadcast(val)                          # valor del lane 0 → todos los lanes
warp.reduce[warp.shuffle_down](val, reduce_fn)  # reducción personalizada (difunde)

# Shuffles — desplazamiento/intercambio por lane, NO difusión
warp.shuffle_down(val, offset)               # offset: UInt32
warp.shuffle_xor(val, offset)                # offset: UInt32, XOR con el id de lane (mariposa)
warp.shuffle_xor(mask, val, offset)          # máscara de pertenencia explícita (rara vez necesario)

# `offset` es `UInt32` — un `Int` plano da error de tipo:
var off: UInt32 = 16
var v = warp.shuffle_xor(val, off)
```

## Comprobación de disponibilidad de GPU

```mojo
from std.sys import has_accelerator

def main() raises:
    comptime if not has_accelerator():
        print("No se encontró GPU")
    else:
        var ctx = DeviceContext()
        # ... código GPU
```

O como assert de tiempo de compilación — que debe ir dentro del cuerpo de una función:

```mojo
def main() raises:
    comptime assert has_accelerator(), "Requiere GPU"
```

## Detección de arquitectura — `is_` vs `has_`

**Distinción crítica**: `is_*` comprueba el **target de compilación** (úsalo dentro de código compilado para GPU). `has_*` comprueba el **sistema host** (úsalo desde código de host/CPU).

> **v1.2.0**: las funciones `has_nvidia_gpu_accelerator()`, `has_amd_gpu_accelerator()` y `has_apple_gpu_accelerator()` fueron **eliminadas**. Para el host, usa `has_accelerator()`; para el target, usa `is_*_gpu()`.

```mojo
from std.sys.info import (
    # Comprobaciones de target — "¿me estoy compilando PARA esta GPU?"
    # Úsalas dentro de kernels o rutas de código dirigidas a GPU.
    is_gpu, is_nvidia_gpu, is_amd_gpu, is_apple_gpu,

    # (Las variantes has_*_gpu_accelerator fueron eliminadas en v1.2.0)
)
from std.sys import has_accelerator   # comprobación de host: ¿hay alguna GPU?

# LADO HOST: decide si ejecutar código GPU
def main() raises:
    comptime if not has_accelerator():
        print("No hay GPU")
    else:
        # ...lanzar kernels

# DENTRO DEL KERNEL o código compilado para GPU: despacha por arquitectura
comptime if is_nvidia_gpu():
    # intrínsecos específicos de NVIDIA
elif is_amd_gpu():
    # ruta específica de AMD
```

Comprobaciones de subarquitectura (solo dentro de código GPU):

```mojo
from std.sys.info import _is_sm_9x_or_newer, _is_sm_100x_or_newer
comptime if is_nvidia_gpu["sm_90"]():   # comprobación exacta de arquitectura
    ...
```

## Patrón de constantes en tiempo de compilación

Todas las dimensiones, layouts y tamaños de GPU deberían ser `comptime`:

```mojo
comptime dtype = DType.float32
comptime SIZE = 1024
comptime BLOCK_SIZE = 256
comptime NUM_BLOCKS = ceildiv(SIZE, BLOCK_SIZE)
comptime layout = row_major[SIZE]()
```

## Ejemplo completo 1D (suma de vectores)

```mojo
from std.math import ceildiv
from std.sys import has_accelerator
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime N = 1024
comptime BLOCK = 256
comptime layout = row_major[N]()

def add_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        c[tid] = a[tid] + b[tid]

def main() raises:
    comptime assert has_accelerator(), "Requiere GPU"
    var ctx = DeviceContext()
    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    ctx.enqueue_function[add_kernel](
        a, b, c, N,
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )

    with c_buf.map_to_host() as host:
        var result = TileTensor(host, layout)
        print(result)
```

## Ejemplo completo 2D (matmul con teselas y memoria compartida)

```mojo
from std.math import ceildiv
from std.sys import has_accelerator
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import thread_idx, block_idx
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime dtype = DType.float32
comptime M = 64
comptime N = 64
comptime K = 64
comptime TILE = 16
comptime a_layout = row_major[M, K]()
comptime b_layout = row_major[K, N]()
comptime c_layout = row_major[M, N]()

def matmul_kernel[
    ALayout: TensorLayout, BLayout: TensorLayout, CLayout: TensorLayout,
](
    A: TileTensor[dtype, ALayout, MutAnyOrigin],
    B: TileTensor[dtype, BLayout, MutAnyOrigin],
    C: TileTensor[dtype, CLayout, MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE + ty
    var col = block_idx.x * TILE + tx

    var sa = stack_allocation[dtype,
        address_space=AddressSpace.SHARED](row_major[TILE, TILE]())
    var sb = stack_allocation[dtype,
        address_space=AddressSpace.SHARED](row_major[TILE, TILE]())

    var acc: C.ElementType = 0.0
    comptime for k_tile in range(0, K, TILE):
        if row < M and k_tile + tx < K:
            sa[ty, tx] = A[row, k_tile + tx]
        else:
            sa[ty, tx] = 0.0
        if k_tile + ty < K and col < N:
            sb[ty, tx] = B[k_tile + ty, col]
        else:
            sb[ty, tx] = 0.0
        barrier()
        comptime for k in range(TILE):
            acc += sa[ty, k] * sb[k, tx]
        barrier()

    if row < M and col < N:
        C[row, col] = acc

def main() raises:
    comptime assert has_accelerator(), "Requiere GPU"
    var ctx = DeviceContext()
    # ... reservar buffers, inicializar datos, luego:
    comptime kernel = matmul_kernel[type_of(a_layout), type_of(b_layout), type_of(c_layout)]
    ctx.enqueue_function[kernel](
        A, B, C,
        grid_dim=(ceildiv(N, TILE), ceildiv(M, TILE)),
        block_dim=(TILE, TILE),
    )
```

## Cargas SIMD en kernels

```mojo
# Carga vectorizada desde puntero crudo — `.load()` está obsoleto
var val = ptr.unsafe_load[width=8](idx)   # SIMD[dtype, 8]
var sum = val.reduce_add()                 # reducción escalar

# Acceso vectorizado con TileTensor
var vec_tensor = tensor.vectorize[1, 4]()  # agrupa elementos en SIMD[4]
```

## Patrón de reducción

La indexación por puntero es `ptr[unsafe_offset=i]` — `ptr[i]` plano está obsoleto. Usa `MutPointer`, no el obsoleto `UnsafePointer`.

```mojo
from std.bit import log2_floor

comptime TPB = 512          # `comptime for` necesita un límite estático, así que el
                            # tamaño de bloque debe ser comptime, no `block_dim.x`

def block_reduce(
    output: MutPointer[Int32, MutAnyOrigin],
    input: MutPointer[Int32, MutAnyOrigin],
):
    var sums = stack_allocation[TPB, Scalar[DType.int32],
        address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    sums[unsafe_offset=tid] = input[
        unsafe_offset=block_idx.x * block_dim.x + tid
    ]
    barrier()

    # Reducción en árbol en memoria compartida
    var active = TPB
    comptime for _ in range(log2_floor(TPB)):
        active >>= 1
        if tid < active:
            sums[unsafe_offset=tid] += sums[unsafe_offset=tid + active]
        barrier()

    # Reducción final de warp + acumulación atómica
    if tid < WARP_SIZE:
        var v = warp.sum(sums[unsafe_offset=tid][0])
        if tid == 0:
            _ = Atomic.fetch_add(output, v)
```

## DeviceBuffer a partir de un puntero existente

```mojo
# Envuelve un puntero existente como DeviceBuffer (no propietario)
var buf = DeviceBuffer[dtype](ctx, raw_ptr, count, owning=False)
```

## Benchmarking de kernels GPU

```mojo
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from max.benchmark import bencher_iter_custom   # forma GPU: función libre

var ctx = DeviceContext()

@inline(.always)
def bench_fn(mut b: Bencher) raises {imm}:
    @inline(.always)
    def launch(ctx: DeviceContext) raises {imm}:
        ctx.enqueue_function[kernel](args, grid_dim=G, block_dim=B)
    bencher_iter_custom(b, launch, ctx)         # NO `b.iter_custom(...)`

var bench = Bench(BenchConfig(max_iters=50000))
bench.bench_function(
    bench_fn,
    BenchId("kernel_name"),
    [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
)
```

`Bencher.iter_custom` no toma un `DeviceContext` — esa forma vive en `max.benchmark`. Tanto la función de benchmark como el closure de lanzamiento se pasan como valores; da a cada uno una lista de captura explícita (`{imm}`, `{var}` o capturas con nombre).

## Detalles de hardware

| Propiedad      | NVIDIA          | AMD CDNA     | AMD RDNA      |
|---------------|-----------------|--------------|---------------|
| Tamaño de warp | 32              | 64           | 32            |
| Memoria compartida | 48-228 KB/bloque | 64 KB/bloque  | configurable  |
| Tensor cores  | SM70+ (WMMA)    | Matrix cores | WMMA (RDNA3+) |
| TMA           | SM90+ (Hopper)  | N/A          | N/A           |
| Clusters      | SM90+           | N/A          | N/A           |
