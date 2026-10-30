---
name: mojo-max-runtime-tuning
description: Ajuste de rendimiento del runtime de MAX en Mojo 1.2.0 / MAX 26.7. Cubre M_newRuntimeConfig/M_setNumThreads para controlar hilos, TCMalloc NUMA, afinidad con MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS, profiling, límites del ThreadPoolWorkQueue y parallelism_level() de std.runtime. No usa el API async eliminado en v1.1.0.
---

<!-- PAUTAS EDITORIALES
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre el runtime de MAX.
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

> **⚠️ ANTES DE ESCRIBIR CÓDIGO: CAMBIOS DE v1.0.0 → v1.2.0 QUE AFECTAN AL RUNTIME**
>
> - **v1.1.0**: se eliminó el **API público de tareas asíncronas** (`create_task`, `Task`, `TaskGroup`). Los primitivos que respaldan el runtime viven en el módulo privado `_asyncrt`, sin estabilidad.
> - **v1.1.0**: `fn` y `alias` desaparecieron; usa `def` y `comptime`. Los closures son unificados (argumentos de runtime).
> - **v1.2.0**: `async`/`await` se renombraron a **`__async`/`__await`**; no los uses para tuning.
>
> **Regla**: para ajustar el runtime de MAX, usa la **API C oficial** (`M_newRuntimeConfig`, `M_setNumThreads`) y las variables de entorno documentadas. Para consultar el número de workers desde Mojo, usa `parallelism_level()` de `std.runtime`. No intentes configurar hilos mediante APIs de Mojo que ya no son públicas.

El runtime de MAX organiza los recursos de CPU (hilos, memoria, afinidad NUMA) a través de `CPUDevice` y `ThreadPoolWorkQueue`. El ajuste consiste en controlar **cuántos hilos** se crean, **cómo se asignan** a los cores y **qué allocator** se usa. La API C de MAX expone un subconjunto estable de estas opciones.

## Configuración del número de hilos (API C)

La forma oficial de controlar el paralelismo es `M_RuntimeConfig` + `M_setNumThreads`.

```c
#include "max/c/context.h"

M_Status *status = M_newStatus();
M_RuntimeConfig *config = M_newRuntimeConfig();
// 0 = número de procesadores físicos en el primer socket (por defecto)
M_setNumThreads(config, 8);
M_RuntimeContext *context = M_newRuntimeContext(config, status);

// … usar el contexto …

M_freeRuntimeContext(context);
M_freeRuntimeConfig(config);
M_freeStatus(status);
```

| Aspecto | Detalle |
|---------|---------|
| **Valor por defecto** | `0`, interpretado como «número de procesadores físicos en el primer socket» |
| **Función de ajuste** | `M_setNumThreads(config, n)` |
| **Alcance** | Afecta al `ThreadPoolWorkQueue` del `CPUDevice` asociado al contexto |
| **Ciclo de vida** | Debes llamar a `M_freeRuntimeConfig()` cuando ya no lo necesites |

**INCORRECTO**: asumir que `numThreads=0` significa «todos los cores lógicos de la máquina». En sistemas multi-socket, el valor por defecto se limita al primer socket.
**CORRECTO**: si necesitas usar todos los sockets, pasa explícitamente el número total de cores físicos, o gestiona la afinidad con `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`.

## Consultar el paralelismo desde Mojo

`std.runtime` expone `parallelism_level()`, que devuelve el número de **worker threads** disponibles en el runtime. Es útil para dimensionar `parallelize()` o para verificar que el tuning surtió efecto:

```mojo
from std.runtime import parallelism_level

def main() raises:
    let workers = parallelism_level()
    print("workers disponibles:", workers)
```

| Función | Módulo | Retorno | Notas |
|---------|--------|---------|-------|
| `parallelism_level()` | `std.runtime` | `Int` | Número de worker threads del runtime |
| `initialize_runtime()` | `std.runtime` | — | Solo necesario en bibliotecas compartidas sin `main()`; con `main()` el runtime se inicializa solo |

## Afinidad y mapeo GPU→CPU

`MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS` permite especificar explícitamente qué cores de CPU se asignan a las tareas de un dispositivo (por ejemplo, una GPU). La implementación delega en `DeviceAffinity.cpp`, que también soporta **mapeo inferido por NUMA** y **round-robin** como fallback.

| Variable | Efecto |
|----------|--------|
| `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS` | Lista de IDs de CPU para tareas de dispositivo; sobrescribe la inferencia NUMA |
| `MODULAR_ENABLE_AFFINITY` | Habilita/deshabilita la afinidad de hilos a cores (si está disponible en el build) |

**Patrón práctico**: en sistemas con GPU y múltiples sockets, fija las tareas de la GPU al socket que tiene el bus PCIe más cercano mediante `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`. Sin esta variable, el runtime infiere el mapeo por NUMA, lo que puede no ser óptimo en topologías irregulares.

## Allocator: TCMalloc NUMA y límites

El runtime usa por defecto **TCMalloc consciente de NUMA** para las asignaciones internas (`TCMallocGlobals`). TCMalloc usa como máximo dos particiones y mapea nodos NUMA alternativamente a particiones (`N % 2`).

| `AllocatorType` (C++) | Uso | Notas |
|-----------------------|-----|-------|
| `kTCMalloc` | **Por defecto**; NUMA-aware | Adecuado para cargas de inferencia |
| `kMalloc` | Fallback estándar | Sin optimizaciones NUMA |
| `kLeakChecker` | Diagnóstico de fugas | No usar en producción |

**Regla**: TCMalloc está pensado para **buffers grandes** (tensores, activaciones). No lo uses para asignaciones pequeñas de grano fino; la indirección de vtable lo hace inadecuado.

## Límites del pool y tamaño de pila

| Parámetro | Valor por defecto | Cómo cambiarlo |
|-----------|-------------------|----------------|
| `kMaxWorkers` | **1024** hilos | No configurable desde la API pública; límite duro del `ThreadPoolWorkQueue` |
| Stack por worker | **8 MiB** | Variable de entorno `runtime.worker_stack_size_mb` |
| Slots por lista de tareas | **1024** | No configurable; límite interno (`kTaskListSlotsPerThread`) |
| Estados de shutdown | `kReady`, `kShuttingDown`, `kShutdown` | — |

Si una aplicación supera los 1024 workers solicitados, el runtime no creará más hilos; el exceso de trabajo se encola en la `overflowTaskList` (protegida por mutex) y se procesa cuando hay capacidad.

## Profiling y diagnóstico

| Variable / Opción | Efecto |
|-------------------|--------|
| `MODULAR_ASYNCRT_MAX_PROFILING_LEVEL` | Bitmask octal por categoría (AsyncRT, Mem, Mojo, Compiler). 0 = sin profiling |
| `--time-profile` | Activa profiling de tiempo; requiere que el build tenga `MODULAR_ASYNCRT_MAX_PROFILING_LEVEL > 0` |
| `MODULAR_ALLOC_LOGGING` | Logging de asignaciones (requiere build con `--//AsyncRT:alloc_logging=true`) |

## Anti-patrones

- **No uses `async`/`await` para tuning**: son `__async`/`__await` e inestables; el tuning se hace con la API C y variables de entorno.
- **No configures hilos con APIs de Mojo eliminadas**: el API público de async (`create_task`, `TaskGroup`) se retiró en v1.1.0; usa `M_setNumThreads`.
- **No asumas que `numThreads=0` usa toda la máquina**: se limita al primer socket; sé explícito en sistemas multi-socket.
- **No uses TCMalloc para objetos pequeños**: solo buffers grandes.
- **No superes `kMaxWorkers=1024`**: el trabajo extra va a la cola de overflow y puede degradar la latencia.
- **No fijes `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS` sin conocer la topología NUMA**: un mapeo incorrecto puede empeorar el rendimiento más que el round-robin por defecto.
- **No uses `parallelism_level()` de `std.runtime` para configurar hilos**: solo lo consulta; la configuración se hace con `M_setNumThreads` o variables de entorno.

## Reglas prácticas

1. **Empieza con `numThreads=0`** (primer socket) y mide; si la carga escala a más sockets, pasa el número total de cores físicos.
2. **En sistemas con GPU, fija `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`** al socket más cercano a la GPU.
3. **Usa `runtime.worker_stack_size_mb` solo si las tareas necesitan más de 8 MiB**; stacks más grandes consumen memoria y pueden reducir el número de workers efectivos.
4. **Activa profiling solo en builds de diagnóstico**: `MODULAR_ASYNCRT_MAX_PROFILING_LEVEL=0` en producción para minimizar el overhead.
5. **Verifica el número de workers con `parallelism_level()`** de `std.runtime`; si no coincide con lo esperado, revisa `M_setNumThreads` y `MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS`.
6. **No intentes cambiar `kMaxWorkers` ni `kTaskListSlotsPerThread`**: son límites de compilación del runtime C++.
