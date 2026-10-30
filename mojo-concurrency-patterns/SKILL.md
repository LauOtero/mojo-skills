---
name: mojo-concurrency-patterns
description: Patrones de concurrencia segura en Mojo 1.2 sin usar el API público de async (eliminado en v1.1.0) ni __async/__await (inestables). Cubre Atomic, Ordering, BlockingSpinLock, BlockingScopedLock, partición de datos y anti-patrones como bloquear dentro de parallelize.
---

<!-- PAUTAS EDITORIALES
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre concurrencia en Mojo.
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

> **⚠️ ANTES DE ESCRIBIR CÓDIGO: CAMBIOS DE v1.0.0 → v1.2.0**
>
> - **v1.1.0**: se eliminó el **API público de tareas asíncronas** (`create_task`, `Task`, `TaskGroup`). Los primitivos async que respaldan `parallelize()` viven en el módulo privado `_asyncrt`, sin estabilidad.
> - **v1.1.0**: `fn` y `alias` desaparecieron; usa `def` y `comptime`. Los closures son unificados (argumentos de runtime).
> - **v1.2.0**: `async`/`await` se renombraron a **`__async`/`__await`** para marcar su inestabilidad.
>
> **Regla**: para CPU-bound usa `parallelize` (ver skill `mojo-parallelize`); para sincronización usa `Atomic` y locks; **no uses `async`/`await` en producción**.

Mojo distingue **concurrencia** (varias tareas progresan, no necesariamente a la vez) de **paralelismo** (ejecución simultánea en varios cores). `parallelize` cubre el paralelismo de CPU; esta skill cubre las primitivas para coordinar el estado compartido cuando el paralelismo es inevitable.

## Primitivas de sincronización estables (v1.2)

| Primitiva | Módulo | Uso | Bloquea |
|-----------|--------|-----|---------|
| `Atomic[DType]` | `atomic` | Contadores, flags, referencias; `fetch_add`, `compare_exchange`, `load`, `store` | No (lock-free) |
| `Ordering` | `atomic` | Especifica el orden de memoria (`relaxed`, `acquire`, `release`, `acq_rel`, `seq_cst`) | — |
| `fence()` | `atomic` | Barrera de memoria independiente | — |
| `BlockingSpinLock` | `utils.lock` | Exclusión mutua básica; `acquire`/`release` manual | Sí (spin) |
| `BlockingScopedLock` | `utils.lock` | RAII sobre `BlockingSpinLock`; libera al salir del scope | Sí (spin) |

`Atomic` se implementa con instrucciones LLVM atómicas y garantiza el orden de memoria solicitado. `BlockingSpinLock` usa un entero para representar el propietario y espera en spin; `BlockingScopedLock` sigue RAII: adquiere en el constructor y libera en el destructor.

## Patrones

### 1. Contador atómico (lock-free)

```mojo
from atomic import Atomic, Ordering
from algorithm import parallelize

def main() raises:
    var counter = Atomic[DType.int64](0)

    def work(i: Int):
        _ = counter.fetch_add(1, Ordering.relaxed)

    parallelize(work, 1000)
    print(counter.load(Ordering.seq_cst))  # 1000
```

### 2. Lock con alcance (RAII)

Cuando la operación crítica es más compleja que una instrucción atómica (por ejemplo, actualizar una lista o un log), usa `BlockingScopedLock` para no olvidar el `release`.

```mojo
from utils.lock import BlockingSpinLock, BlockingScopedLock

def main() raises:
    var lock = BlockingSpinLock()
    var log = List[String]()

    def worker(id: Int):
        with BlockingScopedLock(lock):
            log.append("tarea " + String(id))

    # … parallelize(worker, n) …
```

**INCORRECTO**: adquirir `BlockingSpinLock` manualmente y olvidar `release()` en una excepción.
**CORRECTO**: `with BlockingScopedLock(lock):` garantiza la liberación.

### 3. Partición de datos (evitar estado compartido)

El patrón más seguro para `parallelize`: cada worker escribe en su **propio buffer**, y la reducción se hace secuencialmente al final. No hay contención ni necesidad de locks.

```mojo
from algorithm import parallelize
from memory import UnsafePointer

def main() raises:
    let n = 1024
    var partials = List[Float64](length=n)  # un slot por work item
    let ptr = partials.unsafe_ptr()

    def work(i: Int):
        ptr[i] = compute_chunk(i)

    parallelize(work, n)

    var total: Float64 = 0.0
    for i in range(n):
        total += partials[i]
```

### 4. Productor/consumidor sin `Channel`

La API pública de `Channel` **no forma parte de la superficie estable de Mojo 1.2**; las discusiones del repositorio la mencionan como parte del runtime aún no expuesto completamente. Para productor/consumidor en código de producción, usa una cola con `Atomic` y un buffer circular, o delega en `parallelize` con particiones.

```mojo
from atomic import Atomic, Ordering

# Esbozo: cola SPSC (single-producer, single-consumer) con índices atómicos.
struct SPSCQueue[T: AnyType]:
    var buffer: UnsafePointer[T]
    var capacity: Int
    var head: Atomic[DType.int64]  # productor
    var tail: Atomic[DType.int64]  # consumidor
    # … push/pop con acquire/release …
```

## Anti-patrones

- **No compartas estado mutable sin sincronización**: `parallelize` es seguro solo si los workers no mutan memoria compartida; en cuanto lo hacen, es una race condition.
- **No bloquees dentro de un closure de `parallelize`**: si necesitas I/O o esperar un lock, hazlo fuera o usa una cola externa; el runtime asume trabajo cooperativo.
- **No uses `async`/`await` en producción**: son `__async`/`__await` e inestables. Para CPU-bound, `parallelize`; para I/O, el interop de Python.
- **No uses `Atomic` para operaciones compuestas**: `Atomic` es para escalares; una secuencia leer-modificar-escribir sobre estructuras necesita un lock.
- **No abuses de `BlockingSpinLock` en secciones largas**: el spin desperdicia CPU. Mantén la sección crítica corta; si es larga, replantea el diseño para particionar datos.
- **No dependas de `Channel` ni de `runtime.asyncrt`**: no son API estable en 1.2; el API público de async se eliminó en v1.1.0.

## Reglas prácticas

1. **Primero intenta no compartir**: particiona los datos; cada worker escribe en su buffer y reduce al final.
2. **Si necesitas un contador, usa `Atomic` con el `Ordering` mínimo suficiente** (`relaxed` para estadísticas, `acquire`/`release` para sincronización).
3. **Si necesitas exclusión mutua, usa `BlockingScopedLock`**, no un spin lock manual.
4. **Mantén las secciones críticas por debajo de ~1 µs**; el spin lock no es adecuado para esperas largas.
5. **Combina con `parallelize`**: la concurrencia no reemplaza al paralelismo; `parallelize` distribuye trabajo y `Atomic`/locks coordinan el estado compartido residual.
6. **Verifica siempre con `Ordering.seq_cst` primero**; relaja el orden solo cuando midas y entiendas las dependencias.
