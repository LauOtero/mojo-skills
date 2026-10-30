---
name: mojo-python-to-mojo-migration
description: Convierte archivos Python a código Mojo 1.2.0 recreando TODAS las funcionalidades del origen. Revisa los archivos Mojo existentes para detectar stubs, funciones vacías o funcionalidades faltantes e implementa todo lo que falte. Se activa con frases como "convierte todo de origen a destino", "migra Python a Mojo", "port this folder to Mojo", "convierte la carpeta a Mojo".
license: MIT
---

# Migración Python → Mojo 1.2.0

Reescribe el 100% de las funcionalidades de los archivos `.py` de la carpeta origen a código Mojo 1.2.0 en la carpeta destino.

**Las rutas de origen y destino las proporciona el usuario al invocar la skill.** Si no las da, pídelas antes de empezar.

## ⛔ PROHIBICIONES ABSOLUTAS

Estas reglas son innegociables. Si las violas, la conversión está mal hecha.

1. **NO generes un script Python que haga la conversión.**
   Debes convertir los archivos TÚ MISMO, uno por uno, usando tus herramientas de escritura (`write`, `edit`, etc.). Un script de Python que emite texto no es una conversión: es un stub que no recrea ninguna funcionalidad.
2. **NO dejes stubs vacíos ni funciones con `pass` / `TODO`.**
   Cada función, método, clase y rama condicional del origen debe tener su implementación Mojo completa con la lógica real traducida.
3. **NO crees archivos con solo comentarios de referencia.**
   Un archivo que dice `# TODO: implementar` no es una migración. Es un marcador de posición. Si el origen tiene código, el destino debe tener código equivalente.
4. **NO uses `fn`.** Usa `def` siempre. `fn` está eliminado en Mojo 1.2.0.
5. **NO uses `alias`.** Usa `comptime`.
6. **NO uses `borrowed`, `inout` ni `owned`.** Las convenciones actuales son `imm` (implícito), `mut`, `out`, `deinit` y `var`. `borrowed`, `inout` y `owned` fueron eliminados.
7. **NO uses `__copyinit__` ni `__moveinit__`.** Usa `__init__(out self, *, copy: Self)` y `__init__(out self, *, deinit move: Self)`.
8. **NO copies archivos `.py` como `.mojo`.** Los `.py` se reescriben en Mojo. Solo los recursos no-código (`.cfg`, `.json`, `.yaml`, imágenes, etc.) se copian tal cual.
9. **NO escribas código async en producción.** `__async` y `__await` existen pero el soporte de async está inestable; el código de migración no debe usarlos. Para paralelismo de CPU, usa `parallelize` (ver `mojo-parallelize`).
10. **NO uses `MutStringSpan` ni `MutStringSlice`.** Fueron eliminados en Mojo 1.2.0. `StringSpan` es siempre inmutable; para mutar, usa `String` o un `Span[Byte]` mutable.
11. **NO uses `hash(ptr, len)`.** Usa `hash_bytes()`.

## Flujo obligatorio

### Paso 0 — Lee las habilidades oficiales de Mojo

Antes de escribir una sola línea de Mojo, invoca:
- `mojo-syntax` para sintaxis actual (incluye pattern matching, list destructuring, `__async`/`__await`, `where cond else "message"`, opt-out de traits `not`).
- `mojo-python-interop` si necesitas llamar a Python desde Mojo.
- `mojo-concurrency-patterns` si el código Python usa hilos, locks o colas.

Sin esto, generarás código con sintaxis obsoleta (`fn`, `alias`, `borrowed`, `inout`, `owned`).

### Paso 1 — Inventario completo del origen

Lista TODOS los archivos `.py` de la carpeta origen. Por cada archivo:
- Lee el contenido completo.
- Extrae: clases, funciones, métodos, decoradores, imports, constantes, variables globales, CLI, tests, efectos secundarios (I/O, red, BD).
- Documenta dependencias externas y librerías usadas.

### Paso 2 — Auditoría del destino

Lista TODOS los archivos existentes en la carpeta destino. Clasifica cada uno:
- **Completo**: implementación que cubre el origen funcionalmente.
- **Parcial**: existe pero le faltan funciones, ramas, validaciones o imports.
- **Vacío / Stub**: solo `pass`, `TODO`, firma sin cuerpo, o comentarios.
- **Ausente**: no existe archivo Mojo para un módulo del origen.

### Paso 3 — Matriz de cobertura

Genera una tabla con este formato:

| Python origen | Elemento (fn/clase/método) | Mojo destino | Estado | Funcionalidad faltante | Acción | Prioridad |
|---|---|---|---|---|---|---|

### Paso 4 — Conversión directa (archivo por archivo)

Para CADA archivo Python:

1. **Abre y lee** el archivo `.py` completo.
2. **Determina** el archivo `.mojo` de destino según el mapeo.
3. Si el archivo Mojo **no existe**: créalo con `write`.
4. Si el archivo Mojo **existe pero está vacío/parcial**:
   - Léelo, identifica qué falta.
   - Complétalo con `edit` o reescríbelo con `write`.
5. **Implementa TODAS las funciones, clases y métodos** con su lógica real traducida a Mojo. No dejes nada como `pass` o `TODO`.
6. Si una librería Python no existe en Mojo, usa interop con `from std.python import Python` o implementa una alternativa nativa. Documenta la decisión en un comentario.

**Mapeo de rutas:**
- `paquete/modulo.py` → `src/paquete/modulo.mojo`
- `paquete/__init__.py` → `src/paquete/__init__.mojo`
- `tests/test_x.py` → `test/test_x.mojo`
- Recursos (`.cfg`, `.json`, etc.) → misma ruta relativa

**Reglas de sintaxis actual (obligatorias):**
- `def` es el ÚNICO keyword de función. `fn` no existe.
- `def` es no-raising por defecto. Añade `raises` solo si lanza errores.
- `comptime` en lugar de `alias`.
- Constructores: `def __init__(out self, ...)`.
- Copy: `def __init__(out self, *, copy: Self)`.
- Move: `def __init__(out self, *, deinit move: Self)`.
- Destructor: `def __deinit__(deinit self)`.
- `@fieldwise_init` para constructor campo a campo.
- Tipos explícitos siempre que sea posible.
- **Convenciones de argumentos:** `imm` (implícito, solo lectura), `mut` (referencia mutable), `var` (copia propia), `deinit` (consume), `out` (salida no inicializada), `ref` (referencia con origen).
- **Closures:** `def nombre(args) {capturas} -> T`, con capturas `{imm x}`, `{mut y}`, `{var z}`. Las lambdas de una línea usan `lambda (x: Int) -> Int: ...`.
- **Pattern matching:** `match` para discriminaciones claras; `if case` y `while case` para iteraciones.
- **Destructuración de listas:** `var [first, *middle, last] = values`.
- **Opt-out de traits:** `struct Handle(not Movable, Writable):`.
- **Cláusulas where:** `def foo[sc: Int]() where sc > 1 else "scaling factor must be greater than 1":`.
- **StringSpan:** inmutable; para mutación, usa `String` o `Span[Byte]`. `StringSlice` sigue existiendo como vista no propietaria, pero no es mutable.
- **Hashes:** `hash_bytes()` en lugar de `hash(ptr, len)`.
- **Punteros:** `Pointer` es el tipo principal; `UnsafePointer` es un alias compatible. `alloc()` devuelve un `Allocation` que debe pasarse a `dealloc`.
- **Asincronía:** `__async def` y `__await expr` existen, pero el soporte es inestable; no los uses en código de producción.

Consulta `references/conversion-map.md` para patrones concretos.

**Ejemplo de lo que NO debes hacer:**
```mojo
struct Heater:
    def __init__(out self, config: Config):
        pass  # TODO: implementar
```

**Ejemplo de lo que DEBES hacer:**
```mojo
@fieldwise_init
struct Heater:
    var setting: Int

    def __init__(out self, config: Config):
        self.setting = config.value
```

Si necesitas que `Heater` sea convertible implícitamente desde `Config`, usa `@implicit` en un constructor adicional, no un tipo puntero inventado.

## Referencias

- [Mojo 1.2.0 release notes](https://mojolang.org/releases/v1.2.0/)
- [Mojo syntax skill](../mojo-syntax/SKILL.md)
- [Mojo Python interop skill](../mojo-python-interop/SKILL.md)
- [Mojo concurrency patterns skill](../mojo-concurrency-patterns/SKILL.md)
- [conversion-map.md](references/conversion-map.md)