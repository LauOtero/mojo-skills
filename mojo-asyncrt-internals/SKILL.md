---
name: mojo-asyncrt-internals
description: Referencia interna de AsyncRT (runtime C++ privado de Mojo/MAX) en v1.2.0/26.7. Cubre ThreadPoolWorkQueue, AsyncValue, CPUDevice y su relación con parallelize. No es una API pública de programación asíncrona; el API async fue eliminado en v1.1.0 y __async/__await son inestables. Úsala solo para diagnóstico o comprensión arquitectónica.
---

<!-- PAUTAS EDITORIALES
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre AsyncRT.
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

> **⚠️ ADVERTENCIA DE ESTABILIDAD — LEER PRIMERO**
>
> AsyncRT **no es una API pública de programación asíncrona**. Es el runtime C++ que vive en `AsyncRT/` del repositorio `modular/modular` y que respalda internamente `parallelize()` y MAX.
>
> - **v1.1.0 (17 sep 2026)**: se eliminó el **API público de tareas asíncronas** (`create_task`, `create_raising_task`, `Task`, `TaskGroup`, etc.).
> - **v1.2.0 (22 oct 2026)**: `async`/`await` se renombraron a **`__async`/`__await`** para marcar el soporte async como inestable. La documentación de `std.runtime` lo dice explícitamente: «The async task primitives that back `parallelize()` and friends live in the private `_asyncrt` module. Mojo's async and coroutine support is unfinished and carries no stability guarantees, so it is deliberately not part of this package's public API. Do not build async patterns on it yet.»
> - La documentación de `max.runtime` lo confirma: «Despite the asyncrt name, nothing here is an async programming interface: there is no coroutine, task-spawning, or future API.»
>
> **Regla operativa**: no escribas código de producción contra `runtime.asyncrt` ni `_asyncrt`. Esta skill existe para **entender los mecanismos internos** que hacen posible `parallelize()`, no para invocarlos directamente. Para paralelismo de CPU, usa `parallelize` (ver skill `mojo-parallelize`).

AsyncRT es una tecnología de concurrencia de bajo nivel, de propósito general, basada en bibliotecas (*library-based design*). Se compone de dos capas:

| Capa | Contenido | Dependencias |
|------|-----------|--------------|
| `AsyncRT/Support` | Primitivas sin dependencias: `RCRef<>`, `Chain`, `Semaphore`, `LockFreeRingBuffer`, `ConcurrentMPMCQueue`, `Location`, `Diagnostic` | Solo C++ estándar |
| `AsyncRT/Runtime` | Runtime con «opinión»: `WorkQueue`, `AsyncValue`, `Allocator`, `CPUDevice`, `Algorithms` | Support + TCMalloc |

El diseño asume trabajo **cooperativo** (no bloqueante) y se enfoca en el rendimiento evitando el bloqueo. A diferencia de Intel TBB o Apple GCD, **no hay un thread pool global implícito**; múltiples instancias de AsyncRT pueden coexistir en el mismo proceso.

## Superficie Mojo actual (v1.2.0 / MAX 26.7)

El módulo `max.runtime.asyncrt` que expone Mojo es **deliberadamente mínimo** y no permite crear tareas ni futuros. Solo proporciona dos funciones:

| Función | Firma | Propósito |
|---------|-------|-----------|
| `parallelism_level` | `(ctx: Optional[DeviceContextBase[CompilationTarget()]]) -> Int` | Obtiene el nivel de paralelismo del `DeviceContext` (número de workers). Si `ctx` es `None` o la consulta falla, recurre al nivel global. |
| `task_id_for_device` | `(device_id: Int) -> Int` | Mapea un ID de dispositivo a un `desired_worker_id` de AsyncRT para afinidad de CPU; devuelve `-1` si no hay preferencia. |

El módulo `max.runtime.async_value` ofrece un **handle limitado** al tipo `AsyncValue` de AsyncRT:

| Tipo | Propósito |
|------|-----------|
| `AnyAsyncValueRef` | Handle owning y reference-counted a un `AsyncValue` de C++. Se usa como storage handle para mantener viva la memoria de un buffer o tensor (p. ej. dentro de `OwnedByteBuffer` y `OwnedTensor`). **No crea tareas.** |

Cualquier código que necesite paralelismo debe usar `parallelize()` (ver skill `mojo-parallelize`), que internamente se apoya en `TaskGroup` y `parallelism_level` del runtime.

## WorkQueue

`M::AsyncRT::WorkQueue` es una interfaz abstracta deliberadamente mínima para presentar trabajo y esperar a que un conjunto de `AsyncValue` se compute. Los work items **nunca deben bloquear**.

### Interfaz (C++)

```cpp
virtual void addTask(WorkItem &&work, int taskId = kDefaultTaskId) = 0; // nunca ejecuta inmediatamente
virtual void addLocalTask(WorkItem &&workItem) = 0; // difiere a la cola local del hilo actual
virtual void await(ArrayRef<AnyAsyncValueRef> values) = 0; // puede dormir o donar el hilo
virtual size_t getParallelismLevel() const = 0;
virtual void shutdown() = 0; // obligatorio antes de destruir
```

`kDefaultTaskId = -1` indica que la tarea debe ir a la cola **Global**. Un `taskId >= 0` la enruta a la cola de afinidad del worker correspondiente; en la práctica, solo `async_parallelize` de Mojo genera `taskId` no negativos.

### Jerarquía de colas (implementación real)

La documentación oficial habla de **tres Work Types**: *local*, *affinity-bound* y *global*. La implementación de `ThreadPoolWorkQueue` mantiene **cuatro listas** para gestionar la prioridad y el backpressure:

| Lista | Sincronización | Uso | Prioridad |
|-------|----------------|-----|-----------|
| `localTaskList` | Por worker, sin sync | `addLocalTask()` (continuaciones cortas) | 1ª |
| `affinityTaskList` | Ring buffer lock-free por worker; si se llena, usa `localSpillQueue` con mutex | `addTask` con `taskId >= 0` | 2ª |
| `taskList` | MPMC lock-free global | `taskId == kDefaultTaskId` | 3ª |
| `overflowTaskList` | Mutex | Cola global llena; se vacía antes de dormir | 4ª |

Cada worker procesa en orden **local → affinity → global → overflow**.

### `DelegateThreadPoolWorkQueue` (NUMA)

Existe un `DelegateThreadPoolWorkQueue` que delega a un conjunto de `WorkQueue` particionadas (típicamente una por nodo NUMA), emulando una cola unificada. Es el mecanismo que permite que el runtime sea NUMA-aware sin exponer múltiples colas al cliente.

### Tipos de hilo

| Tipo | `workerID` | Puede `await` | Se dona a sí mismo | Notas |
|------|------------|---------------|---------------------|-------|
| Worker | `0..N-1` | Sí | Sí | Hilo del pool; ejecuta `runItems` |
| Main (`mainWillDonate=true`) | `0` | Sí | Sí | Creado como worker 0; **debe** llamar a `shutdown()` |
| Foreign | — | Sí | No | Puede `addTask`/`await` pero no dona su ejecución |

### Derivación del número de hilos (`numThreads=0`)

1. Sistemas con P-cores/E-cores → número de **P-cores**.
2. Afinidad habilitada → número de **cores físicos**.
3. Sin afinidad → número de **cores lógicos** (incluye hyperthreading).
4. En contenedores (cgroup) → se limita a `millicores / 1000`.

Variables de entorno relevantes: `MODULAR_ENABLE_AFFINITY`, `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`, `runtime.worker_stack_size_mb`.

### Comportamiento en reposo (busy wait + semáforo)

Un worker sin tareas:

1. Hace **busy waiting con backoff exponencial** (por defecto `busyWaitTime = 1 ms`), comprobando nuevas tareas.
2. Vacía la `overflowTaskList`.
3. Se marca como suspendido en un `SuspendedThreadsBitvec` (un `uint64_t`; cada bit puede representar un grupo de workers para escalar más allá de 64 cores) y espera su semáforo per-worker.
4. `addTask` publica el semáforo correspondiente para despertarlo.

### Límites y configuración

| Parámetro | Valor | Fuente |
|-----------|-------|--------|
| `kMaxWorkers` | **1024** hilos | `ThreadPoolWorkQueue.cpp` |
| `kDefaultWorkerStackSizeBytes` | **8 MiB** (por defecto) | `ThreadPoolWorkQueue.cpp` |
| `kTaskListSlotsPerThread` | **1024** ranuras por lista | `ThreadPoolWorkQueue.cpp` |
| Estados de shutdown | `kReady`, `kShuttingDown`, `kShutdown` | — |

### Semántica de `await` (importante)

`await` **no bloquea**: el hilo llamante (worker, main o foreign) se **dona** para ejecutar work items mientras espera. Sin embargo, el retorno de `await` **no garantiza que todo el trabajo disparado por los valores esperados haya alcanzado el reposo**. La única garantía de que todas las computaciones en vuelo han terminado la proporciona `shutdown()`.

## AsyncValue

`AsyncValue` proporciona valores computados asíncronamente **sin operación bloqueante `get()`**. Los consumidores encolan un closure que se ejecuta con `andThenSync` cuando el valor está listo (Continuation-Passing Style).

### Estados

- **Unconstructed**: el valor se está preparando; los closures se encolan.
- **Unconstructed (con inline waiter)**: el primer closure se almacena en línea para ahorrar una allocation.
- **Value Available (`kAvailable`)**: valor listo; se notifican todos los closures.
- **Error (`kError`)**: la computación falló; se propaga `EncodedDiagnostic` con ubicación y mensaje.

### Type-erasing y acceso

`AsyncValue` es type-erased. `AnyAsyncValueRef` es un handle owning y reference-counted a un `AsyncValue` de C++. Usa `get<T>()` / `emplace<T>()` para acceder al tipo subyacente. `AsyncValueRef<T>` es la vista fuertemente tipada.

En Mojo, `max.runtime.async_value.AnyAsyncValueRef` es el único tipo expuesto; no hay una API para crear un `AsyncValue` desde Mojo. Se usa solo como storage handle para memoria de dispositivo.

### Paso de continuación

```cpp
void printWhenReady(AnyAsyncValueRef input) {
  input->andThenSync([]() { printf("listo!"); });
}

// Forma con parámetro para reducir el capture list
void addToTableWhenReady(AsyncValueRef<int32_t> input,
                         RCRef<TableOfValues> tablePtr) {
  input.andThenSync([tablePtr = std::move(tablePtr)]
                (const AsyncValueRef<int32_t> &input) {
    tablePtr->addValue(input.get());
  });
}
```

### Valores indirectos

Cuando el tipo no se conoce hasta después de la computación, usa `AsyncValue::createIndirect` y resuélvelo con `resolveIndirect`. Añade un nivel de indirección.

### `Chain` y dependencias de control

`Chain` es un tipo marcador de **tamaño cero** que convierte dependencias de control (orden de efectos secundarios) en dependencias de datos. `parallelForEachNChain` devuelve `AsyncValueRef<Chain>`; los consumidores se encadenan con `andThenSync` para serializar fases posteriores.

## CPUDevice

`CPUDevice` es el «god object» de nivel superior que organiza el thread pool (`ThreadPoolWorkQueue`), el `Allocator` y las políticas de afinidad NUMA.

### `CPUDeviceOptions`

| Campo | Valores | Notas |
|-------|---------|-------|
| `AllocatorType` | `kMalloc`, `kTCMalloc`, `kLeakChecker`, `kProfiler`, `kUseAfterFree` | Por defecto `kTCMalloc` |
| `WorkQueueType` | `kSingleThread`, `kThreadPool` | `kSingleThread` para contextos sin concurrencia |
| `numThreads` / `maxThreads` | 0 = derivar | Ver reglas de derivación |
| `mainWillDonate` | `true`/`false` | El hilo main se convierte en worker 0 |
| `profileFilename` | ruta | Para profiling |

### Allocator

El heap se abstrae con `Allocator`, que proporciona `malloc`/`free` con alineación y un `size_hint` en `dealloc`. La implementación por defecto usa **TCMalloc consciente de NUMA**: TCMalloc usa como máximo dos particiones, mapeando nodos NUMA alternativamente a particiones (`N % 2`).

Beneficios: profundidad de rastreo y estadísticas de fugas, uso NUMA-aware, y **huge pages** para reducir fallos de TLB.

**Regla crítica**: no uses `Allocator` para asignaciones pequeñas. La indirección de vtable lo hace inadecuado como allocator de grano fino (no lo metas en cada `std::string` o nodo de lista).

Logging opcional: `MODULAR_ALLOC_LOGGING` + nivel `DEBUG` + build con `--//AsyncRT:alloc_logging=true`.

### Afinidad y acceso thread-local

`AsyncRT::Globals::getCurrentCPUDeviceInTLS()` devuelve el `CompactCPUDevicePtr` del `CPUDevice` actual mediante TLS. Es fundamental para la afinidad de tareas y evita pasar manualmente un puntero global de dispositivo. Cada `AsyncValue` recuerda qué `CPUDevice` lo creó; `asyncVal->getRuntime()` devuelve su `CompactCPUDevicePtr`.

Al habilitar afinidad (`MODULAR_ENABLE_AFFINITY`), cada worker llama a `setThreadAffinity(cpuID)` al arrancar y fija la política de memoria para asignar preferentemente en el nodo NUMA de ese core. `task_id_for_device` delega en `DeviceAffinity.cpp`, que maneja `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`, el mapeo NUMA inferido GPU→CPU y el fallback round-robin.

## Relación con `parallelize`

`parallelize` es la **superficie pública y estable** que se asienta sobre AsyncRT. Cuando llamas a `parallelize(func, n)`, el runtime:

1. Crea `n` work items y los encola en el `ThreadPoolWorkQueue` (lista global o de afinidad según el caso).
2. Cada worker ejecuta `func(i)` en su `runItems` loop.
3. `parallelize` **bloquea** (o dona el hilo, si es `main`) hasta que todos los work items terminan.

No necesitas conocer los detalles de `ThreadPoolWorkQueue` para usar `parallelize`; esta sección existe solo para explicar **por qué** el runtime se comporta como lo hace (por ejemplo, por qué `parallelize` no debe contener bloqueos).

## Anti-patrones

- **No invoques `runtime.asyncrt` ni `_asyncrt` directamente**: el API público de async fue eliminado en v1.1.0 y el módulo privado no tiene estabilidad.
- **No uses `async`/`await` en producción**: son `__async`/`__await` e inestables; para CPU-bound, usa `parallelize`.
- **No bloquees dentro de work items**: si necesitas I/O o esperar un lock, hazlo fuera del `WorkQueue` y reencola un callback.
- **No uses `printf` dentro de work items**: puede bloquear si la salida está redirigida.
- **No asignes objetos pequeños a través de `Allocator`**: solo buffers grandes.
- **No asumas que `await` implica reposo global**: solo `shutdown()` garantiza que todo el trabajo en vuelo ha terminado.
- **No intentes crear tareas desde `max.runtime.asyncrt`**: solo expone `parallelism_level` y `task_id_for_device`; no hay API de creación de tareas.

## Reglas prácticas (solo para diagnóstico/comprensión)

1. **Usa `parallelize` para todo el paralelismo de CPU en producción**; esta skill es solo para entender qué hay debajo.
2. **Si ves contención en `taskList`**, considera usar `taskId` para enrutar a colas de afinidad (pero solo si controlas el runtime).
3. **Si el rendimiento en NUMA es pobre**, verifica que `MODULAR_ENABLE_AFFINITY` esté configurado correctamente, o usa `task_id_for_device` para obtener el `desired_worker_id` de un dispositivo.
4. **Recuerda los límites**: 1024 workers, 8 MiB de stack por defecto, 1024 slots por lista.
5. **Para producción en Mojo, no escribas código contra AsyncRT**; usa `parallelize` (skill `mojo-parallelize`) y las primitivas estables (`Atomic`, `BlockingScopedLock`).
6. **`parallelism_level(ctx)` es útil para dimensionar bucles**: úsalo para consultar cuántos workers tiene el contexto antes de particionar trabajo.

## Resumen

- AsyncRT es un runtime C++ **privado**; no es una API pública estable de programación asíncrona.
- `ThreadPoolWorkQueue` gestiona hasta 1024 workers con colas jerárquicas y `SuspendedThreadsBitvec`; `DelegateThreadPoolWorkQueue` añade particionado NUMA.
- `AsyncValue` es un futuro no bloqueante basado en CPS (`andThenSync`); en Mojo solo se expone `AnyAsyncValueRef` como handle de memoria.
- `CPUDevice` organiza el pool, el `Allocator` (TCMalloc NUMA) y la afinidad.
- `parallelize` se asienta sobre AsyncRT pero es la única superficie que debes usar en producción.
- El API público de async se eliminó en v1.1.0; las keywords `__async`/`__await` son inestables en v1.2.0. El módulo `max.runtime.asyncrt` solo expone `parallelism_level` y `task_id_for_device`.
