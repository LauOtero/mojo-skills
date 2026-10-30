---
name: mojo-parallelize
description: Uso correcto de parallelize(), sync_parallelize() y parallelize_over_rows() en Mojo 1.2.0 para paralelismo de CPU. Cubre sintaxis de closures unificados, configuración del runtime y anti-patrones. No usa el API público de async (eliminado en v1.1.0; __async/__await inestables en v1.2.0).
---

<!-- PAUTAS EDITORIALES
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre paralelismo en Mojo.
Cada línea cuesta contexto. Al editar:
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

> **⚠️ ANTES DE ESCRIBIR CÓDIGO: CAMBIOS DE v1.0.0 → v1.2.0 QUE AFECTAN A `parallelize`**
>
> - **v1.1.0**: se eliminó el **API público de tareas asíncronas** (`create_task`, `Task`, `TaskGroup`, etc.). Los primitivos async que respaldan `parallelize()` viven ahora en el módulo privado `_asyncrt`, sin garantías de estabilidad.
> - **v1.1.0**: `fn` y `alias` fueron **eliminados**; usa `def` y `comptime`. Los closures pasaron a ser **unificados** y requieren lista de captura (`{imm x}`, `{mut y}`, `{var z}`).
> - **v1.2.0**: `async`/`await` se renombraron a **`__async`/`__await`** para marcar su inestabilidad. No los uses en producción; para CPU-bound, `parallelize` es la herramienta nativa.
> - **v1.2.0**: `parallelize` y `sync_parallelize` viven en **`std.algorithm`** para Mojo puro y en **`max.algorithm`** para el stack de MAX. `initialize_runtime` y `parallelism_level` viven en **`std.runtime`**.
>
> Por tanto: **no escribas `parallelize[func](n)` con `fn` ni con `alias`**. En Mojo 1.2 la forma correcta es `parallelize(func, n)` con un closure `def` (o una lambda con tipos explícitos). El resto de esta skill asume esa sintaxis.

`parallelize` distribuye `num_work_items` llamadas a un closure sobre el thread pool del runtime de Mojo. **Bloquea hasta que todas las tareas terminan**. Es la superficie pública y estable para paralelismo de CPU; el runtime subyacente (`_asyncrt`) es un detalle de implementación que no debes invocar directamente.

## Firma y sintaxis (Mojo 1.2.0)

```mojo
from std.algorithm import parallelize
# En MAX: from max.algorithm import parallelize

# Forma canónica: closure def + número de work items
parallelize(worker, num_work_items)
```

| Parámetro | Tipo | Semántica |
|-----------|------|-----------|
| `func` | `def(Int) raises -> None` | Se invoca como `func(0)` … `func(num_work_items-1)` en paralelo. Las closures **no-raising** se enlazan a esta firma; las excepciones que se lancen abortan el proceso, no se propagan al llamador. |
| `num_work_items` | `Int` | Número de tareas. Si es `0`, no se ejecuta nada. |

**INCORRECTO (Mojo ≤1.0)**: `parallelize[my_fn](n)` — `fn` y los closures como parámetros de compilación ya no existen.
**CORRECTO (Mojo 1.2)**: `parallelize(my_closure, n)` — el closure se pasa como argumento de runtime.

## Ejemplo mínimo

```mojo
from std.algorithm import parallelize
from std.sys import num_physical_cores

def process_chunk(i: Int) raises:
    # trabajo de la tarea i
    print("tarea", i)

def main() raises:
    comptime n = 16
    parallelize(process_chunk, n)
```

`num_physical_cores()` está en `std.sys`; úsalo para dimensionar `n`:
```mojo
parallelize(process_chunk, num_physical_cores() * 4)
```

## Captura de estado

Los closures unificados capturan el estado del ámbito que los rodea mediante una **lista de captura** explícita. La captura es por valor; para mutar estado compartido necesitas un tipo con semántica de referencia (`Pointer`, `ArcPointer`) y sincronización explícita.

```mojo
from std.algorithm import parallelize
from std.memory import Pointer

def main() raises:
    var data = List[Int](1, 2, 3, 4)
    var ptr = data.unsafe_ptr()

    def double(i: Int) {imm ptr}:
        ptr[i] *= 2

    parallelize(double, len(data))
```

| Captura | Significado |
|---------|-------------|
| `{imm ptr}` | Referencia inmutable al puntero; el closure puede leer a través de él. |
| `{mut counter}` | Referencia mutable; el closure puede modificar el valor. Requiere sincronización si varios workers lo tocan. |
| `{var x}` | Copia propia del valor; el closure puede mutar su copia sin afectar al original. |

**INCORRECTO**: `lambda i: ptr[i] *= 2` — las lambdas requieren tipos explícitos y no pueden tener cuerpo de sentencia.
**CORRECTO**: `def double(i: Int) {imm ptr}: ptr[i] *= 2` — closure con lista de captura.

## `sync_parallelize` y `parallelize_over_rows`

| Función | Módulo | Firma / Notas |
|---------|--------|---------------|
| `parallelize` | `std.algorithm` (Mojo) / `max.algorithm` (MAX) | `parallelize(func, num_work_items)`; bloquea hasta completar. |
| `sync_parallelize` | `std.algorithm` / `max.algorithm` | `sync_parallelize[FuncType: def(Int) raises -> None](func, num_work_items, ctx: Optional[DeviceContext] = None)`. Variante síncrona; las excepciones abortan. |
| `parallelize_over_rows` | `std.algorithm` / `max.algorithm` | Paraleliza sobre filas de una matriz; útil para operaciones por fila sin escribir el bucle de índices. |
| `parallelize_over_range` | — | **No existe** en la API pública de Mojo 1.2.0; usa `parallelize` sobre un rango de índices. |

`sync_parallelize` acepta un `Optional[DeviceContext]` para enrutar el trabajo a un contexto de dispositivo específico; en la mayoría de casos se omite y usa el contexto global.

## Configuración del runtime

El thread pool lo gestiona el runtime de Mojo y se inicializa automáticamente en `main()`. Para control explícito:

```mojo
from std.runtime import initialize_runtime, parallelism_level

def main() raises:
    let workers = parallelism_level()
    print("workers disponibles:", workers)
```

`initialize_runtime()` solo es necesario si **no hay un `main()` de Mojo** (por ejemplo, en una biblioteca compartida `--emit shared-lib` invocada desde C/C++ o Python). En ese caso, llámalo al inicio de cada función exportada con `@export`:

```mojo
from std.runtime import initialize_runtime
from std.algorithm import parallelize

@export("fill_squares")
def fill_squares(data: Pointer[Int64, MutUntrackedOrigin], len: Int) abi("C"):
    initialize_runtime()
    def fill(i: Int) {imm data}:
        data.unsafe_store(i, Int64(i * i))
    parallelize(fill, len)
```

`parallelism_level()` devuelve el número de worker threads disponibles (no el número de cores físicos). Úsalo para dimensionar `num_work_items` si quieres que coincida con la capacidad real del runtime.

## Anti-patrones

- **No uses `fn`, `alias` ni `parallelize[func](n)`**: esa sintaxis pertenece a versiones ≤1.0 y fue eliminada en v1.1.0.
- **No escribas `async`/`await` en código de producción**: las keywords son `__async`/`__await` y el soporte está inacabado. Para CPU-bound, usa `parallelize`.
- **No bloquees dentro del closure**: `parallelize` asume trabajo cooperativo. Si necesitas I/O o esperar un lock, hazlo fuera del closure o usa una cola externa.
- **No uses `print`/`printf` dentro del closure si la salida puede estar redirigida**: puede bloquear el worker. Usa `AsyncValue` o acumula resultados y imprime al final.
- **No paralelices tareas de menos de ~10 µs**: el overhead de encolado supera la ganancia.
- **No dependas de `runtime.asyncrt` ni `_asyncrt`**: el API público de async fue retirado y el módulo privado no tiene estabilidad.
- **No uses `lambda` sin tipos ni con cuerpo de sentencia**: las lambdas de Mojo 1.2 requieren `lambda (x: Int) -> T: expresión`. Para cuerpos con efectos secundarios, usa un `def` con lista de captura.
- **No uses `UnsafePointer`**: es un alias obsoleto de `Pointer`. Usa `Pointer` y `unsafe_ptr()`.

## Reglas prácticas

1. **Usa `parallelize` como límite de paralelismo**: encola `num_work_items` tareas y espera a que todas terminen.
2. **Elige `num_work_items` como múltiplo de `parallelism_level()` o `num_physical_cores()`** para equilibrar la carga.
3. **Mantén cada tarea por encima de ~10 µs**; por debajo, el overhead domina.
4. **Captura por valor con lista de captura**: para mutar estado compartido, usa `Pointer` o `ArcPointer` con sincronización explícita (`Atomic`, `BlockingScopedLock`).
5. **Combina con `vectorize` para SIMD dentro de cada tarea**, no en lugar de `parallelize`.
6. **No uses `parallelize` para I/O**: para I/O-bound, usa el interop de Python (`asyncio`) o un pool externo.
7. **Inicializa el runtime solo si no hay `main()`**: `initialize_runtime()` + `parallelism_level()` para consultar workers en bibliotecas compartidas.
8. **Usa `sync_parallelize` solo si necesitas pasar un `DeviceContext` explícito**; para el caso común, `parallelize` es suficiente.

## Resumen

- `parallelize(func, n)` es la API estable de paralelismo de CPU en Mojo 1.2.0; bloquea hasta completar las `n` tareas.
- El closure debe ser `def(Int) raises -> None`; las closures no-raising se enlazan. Las excepciones abortan.
- La sintaxis antigua `parallelize[func](n)` con `fn`/`alias` **ya no es válida** desde v1.1.0.
- `std.algorithm` es el módulo para Mojo puro; `max.algorithm` para MAX. `std.runtime` expone `initialize_runtime` y `parallelism_level`.
- `parallelize_over_rows` existe para matrices; `parallelize_over_range` no existe.
- `async`/`await` son `__async`/`__await` e inestables; para CPU-bound, `parallelize` es la herramienta correcta.
- Combina `parallelize` (entre cores) con `vectorize` (dentro de cada core) para máximo rendimiento.
