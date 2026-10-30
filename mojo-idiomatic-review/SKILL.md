---
name: mojo-idiomatic-review
description: Revisa código Mojo contra las reglas de idiomatic-mojo.md antes de dar cualquier cambio por finalizado. Úsalo al revisar diffs, refactorizar código Mojo, o al terminar una tarea que modifique archivos .mojo. Aplica junto con mojo-syntax. Cubre las prácticas de Mojo 1.2.0 (match/case, StringSpan inmutable, comptime, closures unificados).
---

# Revisión idiomática de Mojo

Compilar no es el listón para la revisión. Un diff que compila puede seguir violando las convenciones que los revisores de Mojo señalan de forma sistemática. Esta skill convierte esa revisión en un flujo repetible: escanear el diff, localizar los patrones marcados como INCORRECTO en `idiomatic-mojo.md`, corregirlos y reportar qué se cambió y qué se dejó con justificación.

## Cuándo usar esta skill

- Al terminar **cualquier** cambio en archivos `.mojo`.
- Al revisar un diff de código Mojo (propio o ajeno).
- Al refactorizar funciones Mojo existentes.
- Al migrar código de versiones anteriores de Mojo (v1.0.0, v1.1.0) a v1.2.0.

## Flujo de revisión

### 1. Carga el contexto

Carga `mojo-syntax` y su referencia `idiomatic-mojo.md`. Las secciones de `idiomatic-mojo.md` a las que se hace referencia abajo usan la numeración de ese archivo.

### 2. Escanea el diff en orden de prioridad

Recorre el diff de arriba abajo. Las primeras cinco secciones de `idiomatic-mojo.md` son las que los revisores señalan con más frecuencia en código escrito por agentes; revísalas siempre, aunque el diff sea pequeño.

| Prioridad | Sección de `idiomatic-mojo.md` | Qué buscar | Ejemplo de INCORRECTO |
|-----------|--------------------------------|------------|----------------------|
| **1** | §1 Helpers existentes | Matemáticas de índices, reducciones, reshapes o activaciones escritas a mano donde existe un helper | `(M + BM - 1) // BM` en lugar de `ceildiv(M, BM)` |
| **2** | §2 Casts redundantes | `Int(...)` sobre `thread_idx`, `block_idx`, `block_dim` o `grid_dim` (ya devuelven `Int`) | `Int(block_idx.x)` |
| **3** | §3 Parámetros innecesarios | Parámetros `rank`/`dtype`/`layout` que el tipo del argumento ya lleva | `def f[rank: Int, axis: Int](shape: IndexList[rank])` |
| **4** | §4 `IndexList` en lugar de tuplas | `Index(...)` o `IndexList[...]` creados solo para pasarlos | `t.load[width=1](IndexList[2](i, k))` |
| **5** | §5 Trabajo `comptime` | Bucles o ramas sobre valores conocidos en tiempo de compilación que no usan `comptime` | `for i in range(CHUNK):` en lugar de `comptime for i in range(CHUNK):` |
| 6 | §6 `rebind` innecesario | Cada `rebind` que puedas eliminar sin romper la compilación | `rebind[StrideType](StrideType())` |
| 7 | §7 Orígenes mutables en entradas | `MutAnyOrigin` en parámetros que nunca se escriben | `src: Pointer[Float32, MutAnyOrigin]` |
| 8 | §8 Punteros crudos en kernels | `out.ptr[row * N + col]` en lugar de `out[row, col]` | `ptr[i]` en lugar de `tensor[i]` |
| 9 | §9 Flujo de control y duplicación | `else` después de `return`, `min(max(...))` en lugar de `clamp`, acumuladores pelados, concatenación de strings | `if x: return 1 else: return 2` |
| 10 | §10 `match` sobre `elif` | Cadenas de 2+ `elif x == ...` sobre un mismo sujeto (v1.2.0) | `if dtype == .float32: ... elif dtype == .float16: ...` |
| 11 | §11 Nomenclatura | Fabricante/dispositivo en APIs genéricas, acrónimos en minúsculas, sombreado de builtins, locales sin `var` | `gpu_foo` en una API genérica |
| 12 | §12 Diseño de API | Centinelas en lugar de `Optional`, argumentos ignorados silenciosamente, funciones de bajo nivel exportadas | `def f(x: Int = -1)` como "sin valor" |
| 13 | §13 Invariantes y hardware | `32` o `128` codificados a mano, `comptime assert` ausentes, centinelas mágicos en tests | `var warp = 32` |
| 14 | §14 Puertas de fabricante en kernels genéricos | `gpu_constraints = ["//:nvidia_gpu"]` en un kernel que solo usa primitivas genéricas | Un kernel con `WARP_SIZE` limitado a NVIDIA |
| 15 | §15 Higiene | Hacks sin `TODO(<ticket>)`, comentarios que repiten el código, notas de agente en el fuente | `# (basado en TileTensor)` |

### 3. Corrige antes de reportar

Si encuentras un patrón marcado como INCORRECTO, corrígelo antes de reportar la tarea como completada. Si el patrón es ambiguo, consulta la sección correspondiente de `idiomatic-mojo.md` para ver la forma CORRECTA y los matices (por ejemplo, cuándo un `rebind` es legítimo).

### 4. Reporta al usuario

Reporta explícitamente:

- **Qué patrones corregiste**, con la sección de `idiomatic-mojo.md` a la que corresponden.
- **Qué patrones dejaste con justificación**, por ejemplo un `rebind` que sigue siendo necesario tras intentar eliminarlo, o un `comptime for` que se dejó como bucle de runtime porque el desenrollado no aportaba beneficio.
- **Qué partes del diff no pudiste verificar** (por ejemplo, si el diff es demasiado grande o si falta contexto sobre el hardware objetivo).

Un cambio Mojo **no está terminado** hasta que se ha escaneado el diff contra `idiomatic-mojo.md` y se ha reportado el resultado.

## Prohibiciones

- **No dar un cambio Mojo por finalizado sin haber escaneado el diff** contra `idiomatic-mojo.md`.
- **No justificar un `rebind` sin una frase** que explique por qué es necesario (qué tipos no unifican y por qué no se puede usar un `where` o `comptime assert`).
- **No dejar un `MutAnyOrigin` en un parámetro de solo lectura** sin una razón documentada.
- **No introducir `match` de runtime sobre parámetros**: para sujetos que son parámetros o valores `comptime`, usa `comptime match`.
- **No usar `StringSpan` como si fuera mutable**: desde v1.2.0, `StringSpan` es inmutable y `MutStringSpan`/`MutStringSlice` fueron eliminados.

## Referencias rápidas a cambios de v1.2.0

| Cambio | Implicación para la revisión |
|--------|------------------------------|
| Coincidencia de patrones (`match`/`case`) | Las cadenas `elif` sobre un mismo sujeto deberían ser `match` (§10). |
| `StringSpan` inmutable | Cualquier código que mute a través de un span debe migrarse; no existe `MutStringSpan` (§7, §11). |
| Funciones de `std.math` en GPU | Los kernels que llamaban a `libm` pueden usar ahora las versiones Mojo; verifica que no queden workarounds (§1). |
| `comptime` como reemplazo de `@parameter` | Cualquier `@parameter if`/`for` restante es un error de revisión (§5). |
| Closures unificados con lista de captura | Los closures pasados a `parallelize` o `vectorize` deben llevar `{imm}`, `{mut}` o `{var}` (§1, §9). |
