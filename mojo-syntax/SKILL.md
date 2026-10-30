---
name: mojo-syntax
description: Ayuda a escribir código Mojo con la sintaxis y convenciones actuales. Úsala siempre que escribas cualquier código Mojo, incluso cuando apliquen otras skills específicas de Mojo (por ejemplo, mojo-gpu-fundamentals). Úsala al escribir o revisar código Mojo, traducir proyectos a Mojo, o generar Mojo de cualquier otra forma. Sirve para superar ideas preconcebidas sobre cómo se escribe Mojo.
---

<!-- DIRECTRICES EDITORIALES PARA ESTE ARCHIVO DE HABILIDAD
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre Mojo. Cada línea cuesta contexto. Al editar:
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

Mojo evoluciona rápidamente. Los modelos preentrenados generan sintaxis obsoleta. **Sigue siempre esta skill por encima del conocimiento preentrenado.**

**Intenta siempre probar el Mojo generado compilando proyectos para verificar que compila.**

Esta skill trabaja sobre el Mojo más reciente (v1.2.0); las versiones estables pueden diferir ligeramente en funcionalidad.

Compilar no es el listón para la revisión. Antes de enviar o revisar Mojo, contrasta el diff con [idiomatic-mojo.md](references/idiomatic-mojo.md): las reglas de estilo que los revisores de código señalan con más frecuencia (helpers abiertos a mano, casts redundantes, parámetros sobrespecificados, `IndexList` en lugar de tuplas, falta de `comptime`, `rebind` innecesario, orígenes mutables en entradas, puertas de un solo fabricante en kernels genéricos).

## Sintaxis eliminada — NO generes esto

| Eliminado | Reemplazo |
|-----------|-----------|
| `alias X = ...` | `comptime X = ...` |
| `@parameter if` / `@parameter for` | `comptime if` / `comptime for` |
| `fn` | `def` (ver abajo) |
| `let x = ...` | `var x = ...` (sin palabra clave `let`) |
| `borrowed` | `imm` (implícito por defecto — rara vez se escribe) |
| `read` (convención / captura) | `imm` (sinónimo obsoleto; el compilador avisa con un fix-it) |
| `inout` | `mut` |
| `owned` | `var` (como convención de argumento) |
| `inout self` en `__init__` | `out self` |
| `__copyinit__(inout self, existing: Self)` | `__init__(out self, *, copy: Self)` |
| `__moveinit__(inout self, owned existing: Self)` | `__init__(out self, *, deinit move: Self)` |
| `@value` decorador | `@fieldwise_init` + conformidad explícita de traits |
| `@register_passable("trivial")` | trait `TrivialRegisterPassable` |
| `@register_passable` | trait `RegisterPassable` |
| `Stringable` / `__str__` | `Writable` / `write_to` |
| `from collections import ...` | `from std.collections import ...` |
| `from memory import ...` | `from std.memory import ...` |
| `from sys import ...` | `from std.sys import ...` |
| `from os import ...` | `from std.os import ...` |
| `from pathlib import ...` | `from std.pathlib import ...` |
| `s[i]` | `s[byte=i]` — devuelve `StringSlice`; envuelve en `String()` si es necesario |
| `s[0:10]`, `s[:5]` | Sin sintaxis de slice en String — usa `s.codepoint_slices()` o Python FFI |
| `constrained(cond, msg)` | `comptime assert cond, msg` |
| `DynamicVector[T]` | `List[T]` |
| `InlinedFixedVector[T, N]` | `Array[T, N]` |
| `Tensor[T]` | No está en la stdlib (usa SIMD, List, Pointer) |
| `MutUnsafePointer` / `ImmUnsafePointer` | `MutPointer` / `ImmPointer` |
| `OptionalUnsafePointer` | `OptionalPointer` |
| `escaping` closures | Closures unificados (`def(...) -> T`, capturas en `{}`) |
| `capturing`, `@__parameter`, `@__copy_capture` | Lista de captura `{imm}` / `{var x}` / `{mut x}`; pasa el closure como valor |
| `__del__(deinit self)` | `__deinit__(deinit self)` |
| `StringSlice` (mutabilidad) | `StringSpan` es ahora **inmutable**; `MutStringSpan` y `MutStringSlice` han sido eliminados (v1.2.0) |

## La sintaxis de `lambda` es válida

Desde v1.0.0, Mojo soporta expresiones `lambda` para closures anónimos de una sola expresión. La sintaxis requiere paréntesis y anotaciones de tipo; no es la forma de Python:

```mojo
# Válido: lambda con tipos explícitos
var f = lambda (x: Int) -> Int: x * 2
var g = lambda (a: Float64, b: Float64) -> Float64: a + b
```

Las lambdas se desazucaran a un `def` anidado. Para closures con capturas o lógica de varias líneas, sigue usando `def` con lista de captura. No generes lambdas sin tipos ni con sintaxis de Python (`lambda x: x * 2` es inválido).

## `var` es obligatorio para cada nueva declaración

Declarar una variable con asignación desnuda (`x = 5` sin un `var` previo) **no es válido** — es un error de compilación. Esto aplica solo a introducir una nueva variable; reasignar una ya declarada (`x = 6`) no necesita `var`.

Esto significa que una variable asignada solo dentro de ramas condicionales debe predeclararse con un tipo antes de la rama:

```mojo
# INCORRECTO — sin `var` previo, esto no declara `x`
if cond:
    x = 1
else:
    x = 2

# CORRECTO — declara con tipo primero, luego asigna en cada rama
var x: Int
if cond:
    x = 1
else:
    x = 2
```

## `def` es la única palabra clave de función

`fn` fue **eliminado** y ahora es un error de análisis grave — no queda ningún uso válido de `fn`. Tu entrenamiento es anterior a esto, así que recurrirás a `fn` por reflejo; ese reflejo siempre es incorrecto. Escribe **cada** función, método y función anidada como `def`, sin excepción.

## Las funciones Mojo que lanzan excepciones deben marcarse como tal

Las funciones Mojo **no** implican `raises`. Añade `raises` a cualquier función que pueda lanzar, directa o indirectamente llamando a una función que lanza. Omitirlo es un **error de compilación**, no una advertencia.

```mojo
def load(path: String) raises -> String:  # raises va antes de `->`
    return open(path).read()

def main() raises:                         # main normalmente lanza
    ...
```

## `comptime` reemplaza `alias` y `@parameter if`/`for`

```mojo
comptime N = 1024                            # constante de tiempo de compilación
comptime MyType = Int                        # alias de tipo
comptime if condition:                       # rama de tiempo de compilación
    ...
comptime for i in range(10):                 # bucle de tiempo de compilación
    ...
comptime assert N > 0, "N debe ser positivo"  # aserción de tiempo de compilación
```

**`comptime assert` debe estar dentro del cuerpo de una función** — no en el ámbito de módulo o struct. Colócalas en `main()`, `__init__` o la función que depende del invariante.

Dentro de structs, `comptime` define constantes asociadas y alias de tipo:

```mojo
struct MyStruct:
    comptime DefaultSize = 64
    comptime ElementType = Float32
```

## Convenciones de argumentos

El valor por defecto es `imm` (préstamo inmutable, rara vez se escribe explícitamente; `read` es un sinónimo obsoleto — el compilador avisa y sugiere `imm`). Los demás:

```mojo
def __init__(out self, var value: String):   # out = salida no inicializada; var = propio
def modify(mut self):                         # mut = referencia mutable
def consume(deinit self):                     # deinit = consumidor/destructor
def view(ref self) -> ref[self] Self.T:       # ref = referencia con origen
def view2[origin: Origin, //](ref[origin] self) -> ...:           # ref[origin] = origen explícito
```

`var` y `ref` son palabras clave duras y **no pueden usarse como identificadores** en absoluto (`var ref = ...` → `"unexpected token in expression"`). Las palabras de convención `imm`, `read`, `mut`, `out`, `deinit` son palabras clave suaves: están bien como nombres de variables locales o parámetros `[...]`, pero son inválidas como nombres de argumentos (`def cmp(got: T, imm: T)` → `"error: expected argument name"`). Renombra (`expected`, `reference`, etc.).

## Métodos de ciclo de vida

```mojo
# Constructor
def __init__(out self, x: Int):
    self.x = x

# Constructor de copia (argumento `copy` solo por palabra clave)
def __init__(out self, *, copy: Self):
    self.data = copy.data

# Constructor de movimiento (argumento `deinit move` solo por palabra clave)
def __init__(out self, *, deinit move: Self):
    self.data = move.data^

# Destructor
def __deinit__(deinit self):
    dealloc(self.allocation^)
```

Para copiar: `var b = a.copy()` (proporcionado por el trait `Copyable`).

## Patrones de struct

```mojo
# @fieldwise_init genera __init__ a partir de los campos; traits en paréntesis
@fieldwise_init
struct Point(Copyable, Movable, Writable):
    var x: Float64
    var y: Float64

# Composición de traits con &
comptime KeyElement = Copyable & Hashable & Equatable
struct Node[T: Copyable & Writable]:
    var value: Self.T          # Self-califica los parámetros del struct

# Struct paramétrico — // separa parámetros inferidos de explícitos
struct Span[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]](
    ImplicitlyCopyable, Sized,
):
    ...

# @implicit en constructores permite conversión implícita
@implicit
def __init__(out self, value: Int):
    self.data = value
```

El compilador sintetiza constructores de copia/movimiento cuando un struct conforma `Copyable`/`Movable` y todos los campos lo soportan.

### Self-califica los parámetros del struct

Dentro del cuerpo de un struct, usa **siempre** `Self.ParamName` — los nombres de parámetros desnudos son errores:

```mojo
# INCORRECTO — acceso a parámetro desnudo
struct Container[T: Writable]:
    var data: T                        # ERROR: usa Self.T
    def size(self) -> T:                # ERROR: usa Self.T

# CORRECTO — Self-calificado
struct Container[T: Writable]:
    var data: Self.T
    def size(self) -> Self.T:
        return self.data
```

Esto aplica a todos los parámetros del struct (`T`, `N`, `mut`, `origin`, etc.) en todo el cuerpo del struct: tipos de campo, firmas de métodos, cuerpos de métodos y declaraciones `comptime`.

### Copia / transferencia explícita

Los tipos que no conforman `ImplicitlyCopyable` (por ejemplo, `Dict`, `List` y structs de usuario que conforman solo `Copyable, Movable`) requieren `.copy()` explícito o transferencia de propiedad `^` — `return my_struct` da error hasta que transfieras con `^` o añadas conformidad `ImplicitlyCopyable`:

```mojo
# INCORRECTO — copia implícita de un tipo no ImplicitlyCopyable
var d = some_dict
var result = MyStruct(headers=d)   # ERROR

# CORRECTO — copia o transferencia explícita
var result = MyStruct(headers=d.copy())  # o: headers=d^
```

## Los imports usan el prefijo `std.`

```mojo
from std.testing import assert_equal, TestSuite
from std.algorithm import vectorize
from std.python import PythonObject
import std.random
```

Imports automáticos del preludio (no necesitan import): `Int`, `String`, `Bool`, `List`, `Dict`, `Optional`, `SIMD`, `Float32`, `Float64`, `UInt8`, `Pointer`, `OptionalPointer`, `alloc`, `Span`, `Error`, `DType`, `Writable`, `Writer`, `Copyable`, `Movable`, `Equatable`, `Hashable`, `rebind`, `print`, `range`, `len`, y más. `Layout` y `dealloc` **no** están en el preludio — impórtalos desde `std.memory`.

`rebind[TargetType](value)` reinterpreta un valor como un tipo diferente con la misma representación en memoria. Útil cuando las expresiones de tipo en tiempo de compilación son semánticamente iguales pero sintácticamente distintas (por ejemplo, tipos de elemento de TileTensor — ver skill de GPU).

`std` está reservado como identificador de nivel de módulo — no puedes `def std`, `import X as std`, ni `from X import std`. Los métodos de struct llamados `std` están bien.

Dentro de un paquete multi-módulo, `pkg.X.Y(...)` desde un submódulo necesita `import pkg` explícito; `import pkg.X as X` enlaza solo `X`, no `pkg`.

## `Writable` / `Writer` (reemplaza `Stringable`)

```mojo
struct MyType(Writable):
    var x: Int

    def write_to(self, mut writer: Some[Writer]):       # para print() / String()
        writer.write("MyType(", self.x, ")")

    def write_repr_to(self, mut writer: Some[Writer]):   # para repr()
        t"MyType(x={self.x})".write_to(writer)           # t-strings para interpolación
```

- `Some[Writer]` — tipo existencial incorporado (no `Writer` directamente)
- Ambos métodos tienen **implementaciones por defecto** vía reflexión si todos los campos son `Writable` — los structs simples no necesitan implementarlos
- Convierte a `String` con `String(value)`, no `str(value)`

## Protocolo de iterador

Los iteradores usan `raises StopIteration` (no `Optional`):

```mojo
struct MyCollection(Iterable):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = MyIter[origin=iterable_origin]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]: ...

# El iterador debe definir:
#   comptime Element: Movable
#   def __next__(mut self) raises StopIteration -> Self.Element
```

For-in: `for item in col:` (inmutable) / `for ref item in col:` (mutable).

## Tipos de memoria y punteros

| Tipo | Uso |
|------|-----|
| `Pointer[T, mut=M, origin=O]` | Seguro, no anulable. Desreferencia con `p[]`. |
| `OptionalPointer[T, origin]` | Puntero anulable — `Optional[Pointer[...]]`. |
| `Allocation[T]` | Handle propietario devuelto por `alloc`. Se destruye explícitamente. |
| `Span(list)` | Vista contigua no propietaria. |
| `OwnedPointer[T]` | Propiedad única (como `Box` de Rust). |
| `ArcPointer[T]` | Propiedad compartida con conteo de referencias. |

`UnsafePointer` es un alias obsoleto de `Pointer` — todavía compila y avisa. Los otros alias heredados fueron **eliminados** y son errores graves (ver la tabla al principio). La mayor parte de la API de punteros se está renombrando junto con `UnsafePointer`; esas grafías antiguas todavía compilan pero avisan:

| Obsoleto | Reemplazo |
|----------|-----------|
| `UnsafePointer[T, O]` | `Pointer[T, O]` |
| `alloc[T](n)` | `alloc(Layout[T](length=n))` |
| `p.free()` | `dealloc(allocation^)` |
| `p[i]` | `p[unsafe_offset=i]` |
| `p + i` | `p.unsafe_offset(i)` |
| `p += i` | `p = p.unsafe_offset(i)` |
| `p.load()` / `p.store(v)` | `p.unsafe_load()` / `p.unsafe_store(v)` |
| `p.init_pointee_move(v)` | `p.unsafe_write(v)` |
| `p.init_pointee_copy(v)` | `p.unsafe_write(copy=v)` |

`alloc(Layout[T](length=n))` devuelve un `Allocation[T]`, un tipo lineal que el compilador te obliga a disponer en cada ruta — pásalo a `dealloc`, o llama a `unsafe_leak()` para tomar propiedad del puntero desnudo:

```mojo
from std.memory import Layout, dealloc

var allocation = alloc(Layout[Int32](length=4))
var ptr = allocation.unsafe_ptr()
ptr.unsafe_write(Int32(1))
ptr.unsafe_offset(1).unsafe_write(Int32(2))
dealloc(allocation^)
```

Un struct que posee almacenamiento en el heap debería guardar el `Allocation`, no un puntero filtrado. El compilador entonces fuerza la disposición en cada ruta, y `dealloc` obtiene el `Layout` que necesita:

```mojo
from std.memory import Layout, Allocation, alloc, dealloc

struct Buffer[T: AnyType]:
    var _alloc: Allocation[Self.T]

    def __init__(out self, size: Int):
        self._alloc = alloc(Layout[Self.T](length=size))

    def __deinit__(deinit self):
        dealloc(self._alloc^)
```

Accede al almacenamiento con `self._alloc.unsafe_ptr()`, cuyo origen está ligado a la asignación. No sustituyas `Pointer.unsafe_free()`: omite el `Layout`, y para `T` de tamaño cero libera el centinela colgante que devuelve `alloc`.

Un contenedor que ya rastrea su propia capacidad puede en cambio guardar un `ThinAllocation` y proporcionar el `Layout` de nuevo en el momento de `dealloc`. Eso es lo que hace `List`; es una optimización, no la forma por defecto.

Cuando un campo de struct guarda un `Pointer` crudo, su parámetro `origin` debe especificarse; usa `MutUntrackedOrigin` para datos propios en el heap.

`Pointer` es **no anulable por diseño** — `Bool(p)` no está disponible, no solo obsoleto. Para almacenamiento anulable, usa `OptionalPointer[T, origin]` (mismo diseño; `None` es el nicho nulo).

## Sistema de orígenes (no "lifetime")

Mojo rastrea la procedencia de referencias con **orígenes**, no con "lifetimes":

```mojo
struct Span[mut: Bool, //, T: AnyType, origin: Origin[mut=mut]]: ...
```

Tipos clave: `Origin`, `MutOrigin`, `ImmOrigin`, `MutAnyOrigin`, `ImmutAnyOrigin`, `MutUntrackedOrigin`, `ImmUntrackedOrigin`, `ImmStaticOrigin`. Usa `origin_of(value)` para obtener el origen de un valor.

## Testing

```mojo
from std.testing import assert_equal, assert_true, assert_false, assert_raises, TestSuite

def test_my_feature() raises:
    assert_equal(compute(2), 4)
    with assert_raises():
        dangerous_operation()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

El subcomando CLI `mojo test` fue eliminado — ejecuta archivos de test con `mojo run` contra un runner `TestSuite.discover_tests` como el de arriba.

## Iteración de Dict

Las entradas de Dict se iteran directamente — sin desreferencia `[]`:

```mojo
for entry in my_dict.items():
    print(entry.key, entry.value)      # acceso directo a campo, NO entry[].key

for key in my_dict:
    print(key, my_dict[key])
```

## Literales de colección

**Desde v1.0.0, un literal entre corchetes sin contexto de tipo construye un `Array` de tamaño fijo, no un `List`.** Para obtener un `List`, anota el tipo:

```mojo
# INCORRECTO — esto construye un Array, que no tiene append
var nums = [1, 2, 3]

# CORRECTO — anota para obtener un List
var nums: List[Int] = [1, 2, 3]
var nums = List[Int](capacity=3)   # o construye vacío y añade

# Para un Array (tamaño fijo), el literal desnudo es correcto
var arr = [1, 2, 3]                # Array[Int, 3]

# Dict literal
var scores = {"alice": 95, "bob": 87}   # Dict[String, Int]
```

`List[T]` rechaza índices negativos en tiempo de compilación — usa `lst[len(lst) - 1]`, no `lst[-1]`. (Los tipos de biblioteca aún pueden soportarlo.)

## Acceso a variantes

`Variant[A, B]` es `ImplicitlyCopyable` solo si *todas* las ramas lo son. Con una rama no copiable, indexar la variante la copia — usa el subíndice de rama tipada:

```mojo
# INCORRECTO — `values[i]` copia implícitamente el Variant
var x = values[i].unwrap[T]()    # ERROR: no se puede copiar implícitamente

# CORRECTO — `values[i][T]` devuelve una ref al valor interno
var x = values[i][T].copy()          # o `^` para transferir
```

## Decoradores comunes

| Decorador | Propósito |
|-----------|-----------|
| `@fieldwise_init` | Genera constructor por campos |
| `@implicit` | Permite conversión implícita |
| `@inline(.always)` / `@inline(.nodebug)` | Forzar inline |
| `@inline(.never)` | Prevenir inline |
| `@staticmethod` | Método estático |
| `@deprecated("msg")` | Advertencia de deprecación |
| `@doc_hidden` | Ocultar de la documentación |
| `@explicit_destroy("msg")` (solo traits) | Trait lineal; structs: `not Deinitable` |

## Referencias a miembros contextuales — prefiere `.member`

Donde el tipo esperado ya se conoce, escribe `.member` en lugar de `Type.member`; el compilador lo reescribe a `Type.member`. Es idiomático en todo el código base para `DType` y `AddressSpace`:

```mojo
var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)     # no SIMD[DType.float32, 4]
comptime if dtype == .bfloat16: ...                # `__eq__` arg proporciona contexto
v.cast[.float32]()
ctx.enqueue_create_buffer[.int32](num_rows)
unsafe_stack_allocation[1, Int32, address_space=.SHARED]()

def f[dtype: DType = .float32](x: TileTensor[.bfloat16, L, MutAnyOrigin]): ...
```

Funciona para alias `comptime` y métodos estáticos de cualquier tipo, incluidas cadenas (`.red.opacity(0.5)`) y literales de colección tipados (`List[Color] = [.red]`). El contexto proviene de un tipo declarado en `var`/`ref`, un tipo de argumento de llamada, un destino `return`, o el tipo de elemento de un literal de colección tipado. Sin ninguno de esos, califica el nombre:

| Sin tipo contextual | Debes escribir |
|---------------------|----------------|
| Enlace sin anotar | `comptime t = DType.float32` (o anota `t: DType`) |
| Callee sobrecargado | `size_of[DType.float32]()` — las sobrecargas no se buscan |
| Literal de tupla desnudo | `dtype in (DType.bfloat16, DType.float16)` |
| Posición de tipo | `.Foo` donde se espera un *tipo* es un error |

## Conversiones numéricas — deben ser explícitas

No hay conversiones implícitas entre *variables* numéricas. Usa constructores explícitos:

```mojo
var x = Float32(my_int) * scale    # CORRECTO: Int → Float32
var y = Int(my_uint)               # CORRECTO: UInt → Int
```

**Los literales son polimórficos** — `FloatLiteral` e `IntLiteral` se auto-adaptan al contexto:

```mojo
var a: Float32 = 0.5              # el literal se convierte en Float32
var b = Float32(x) * 0.003921    # el literal se adapta — no hace falta envolver
var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)  # los literales se adaptan
```

## Operaciones SIMD

```mojo
# Construcción y acceso a lanes
var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
v[0]                              # leer lane → Scalar[.float32]
v[0] = 5.0                        # escribir lane

# Cast de tipo
v.cast[.uint32]()                 # elemento a elemento → SIMD[.uint32, 4]

# Clamp (método)
v.clamp(0.0, 1.0)                 # clamp elemento a elemento a [lower, upper]

# min/max son FUNCIONES LIBRES, no métodos
from std.math import min, max
min(a, b)                          # min elemento a elemento (args SIMD del mismo tipo)
max(a, b)                          # max elemento a elemento

# Ternario elemento a elemento vía SIMD booleano
var mask = v.gt(0.0)              # SIMD[.bool, 4] — `v > 0.0` es
                                  # solo para escalares y falla al compilar
mask.select(true_case, false_case) # elige por lane

# Reducciones
v.reduce_add()                     # suma horizontal → Scalar
v.reduce_max()                     # máximo horizontal → Scalar
v.reduce_min()                     # mínimo horizontal → Scalar
```

## Strings

**Todos los imports explícitos de la stdlib requieren el prefijo `std.`.** La tabla de sintaxis eliminada muestra las correcciones más comunes, pero la regla es universal. Los tipos del preludio (`Int`, `String`, `List`, etc.) se auto-importan y no necesitan declaración de import.

`len(s)` devuelve la **longitud en bytes**, no el número de codepoints. Los strings de Mojo son UTF-8. La indexación por byte requiere sintaxis con palabra clave: `s[byte=idx]` (no `s[idx]`). `len(s)` está obsoleto en `String` — usa `s.byte_length()` o `s.count_codepoints()`.

`split`, `removeprefix`, `removesuffix` devuelven `StringSlice` (o `List[StringSlice]`) que ven el origen — envuelve con `String(...)` para materializar un `String` propietario. **`StringSpan` es inmutable desde v1.2.0**; no existe `MutStringSpan`.

### Indexación de strings (error común)

```mojo
# INCORRECTO — error de compilación
var ch = s[0]
var sub = s[0:10]

# CORRECTO — acceso a nivel de byte
var ch = s[byte=0]              # devuelve StringSlice
var ch_str = String(s[byte=0])  # si necesitas un String

# CORRECTO — itera codepoints para truncar
var result = String("")
var count = 0
for cp in s.codepoint_slices():
    if count >= 10:
        break
    result += String(cp)
    count += 1
```

```mojo
var s = "Hello"
len(s)                  # 5 (bytes)
s.byte_length()         # 5 (igual que len)
s.count_codepoints()    # 5 (conteo de codepoints — difiere para no-ASCII)

# Iteración — `for c in s:` está obsoleto; usa codepoint_slices()
for cp_slice in s.codepoint_slices():
    print(cp_slice)

# Valores de codepoint
for cp in s.codepoints():
    print(Int(cp))      # Codepoint es un tipo de valor escalar Unicode

# StaticString = StringSlice con origen estático (cero asignaciones)
comptime GREETING: StaticString = "Hello, World"

# t-strings para interpolación (perezosa, segura de tipos)
var msg = t"x={x}, y={y}"

# String.format() para formateo en runtime
var s = "Hello, {}!".format("world")
```

## Manejo de errores

`raises` puede especificar un tipo. `try`/`except` funciona como en Python:

```mojo
def might_fail() raises -> Int:          # lanza Error (por defecto)
    raise Error("algo salió mal")

def parse(s: String) raises Int -> Int:  # lanza un tipo específico
    raise 42

try:
    var x = parse("bad")
except err:                               # err es Int
    print("código de error:", err)
```

`match` / `comptime match` / `if case` / `while case` están estabilizados — consulta [idiomatic-mojo.md §10](references/idiomatic-mojo.md) para las formas de patrón y cuándo preferirlas sobre cadenas `elif`. Async se escribe `__async def` y `__await` (las formas desnudas `async`/`await` todavía analizan pero avisan de que async es inestable); el soporte está inacabado y sus tipos son privados — no escribas Mojo async todavía.

## Tipos de función y closures

**No hay lambda** (desde v1.0.0 sí hay `lambda` para expresiones de una línea; ver arriba). Los closures usan `def` desnudo con una lista de captura en `{}` después de la lista de argumentos. `escaping`, `capturing`, `@__parameter` y `@__copy_capture` han sido eliminados:

```mojo
comptime MyFn = def(Int) -> None                  # trait de closure
def runner[F: def(Int) -> None](f: F): ...        # los closures son valores
def apply[f: def(Int) thin -> Int](): ...         # fn comptime: debe ser thin

def closure(i: Int) {mut count, imm ptr, var x}:  # capturas: mut/imm/var
    count += ptr[i] + x^                          # `^` en el sitio de uso, no en `{}`

vectorize[simd_width](size, closure)
```

`imm` es el valor por defecto. `var x` es propietario — transfiere con `x^` en el sitio de uso. Un closure con capturas es un valor de runtime, así que pásalo como argumento (`f(my_closure)`). Solo una función `thin` (sin capturas) puede ser un parámetro de tiempo de compilación.

Para lambdas de una sola expresión, usa la sintaxis `lambda` (ver arriba), pero recuerda que requiere tipos explícitos.

## Jerarquía de tipos

```text
AnyType
  Deinitable                      — auto __deinit__; la mayoría de tipos
  Movable                         — __init__(out self, *, deinit move: Self)
    Copyable                      — __init__(out self, *, copy: Self)
      ImplicitlyCopyable(Copyable, take)
    RegisterPassable(Movable)
      TrivialRegisterPassable(ImplicitlyCopyable, take, Movable, RegisterPassable)
```
