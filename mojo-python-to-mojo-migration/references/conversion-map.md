# Mapa de conversión Python → Mojo 1.2.0

> `fn` eliminado → usa `def`. `alias` eliminado → usa `comptime`.
> `borrowed`/`inout`/`owned` eliminados → usa `imm`/`mut`/`var`/`deinit`.
> `__copyinit__`/`__moveinit__` fusionados en `__init__`.

## Funciones

| Python | Mojo 1.2.0 |
|---|---|
| `def foo(x, y): return x + y` | `def foo(x: Int, y: Int) -> Int: return x + y` |
| `def foo(x): raise ValueError(...)` | `def foo(x: Int) raises: raise Error("...")` |

**`def` es no-raising por defecto.** Añade `raises` solo si lanza errores.

## Convenciones de argumentos

| Python | Mojo 1.2.0 | Significado |
|--------|-----------|-------------|
| `def f(x)` (solo lectura) | `def f(x: T)` | `imm` implícito: referencia inmutable |
| `def f(self)` (mutación) | `def f(mut self)` | referencia mutable |
| `def f(x)` (toma propiedad) | `def f(var x: T)` | copia propia (mutable) |
| `def f(x)` (consume) | `def f(deinit x: T)` | consume el valor |
| `def f(self)` (constructor) | `def __init__(out self, ...)` | salida no inicializada |
| `def f(self)` (referencia) | `def f(ref self) -> ref[self] T` | referencia con origen |

## Structs y constructores

| Python | Mojo 1.2.0 |
|---|---|
| `class Foo: def __init__(self, x): self.x = x` | `struct Foo: var x: Int; def __init__(out self, x: Int): self.x = x` |
| `__copy__` | `def __init__(out self, *, copy: Self): ...` |
| `__deepcopy__` | Implementar manualmente |
| `__del__` | `def __deinit__(deinit self): ...` |
| Constructor campo a campo | `@fieldwise_init` sobre el struct |
| `__eq__` | `def __eq__(self, other: Self) -> Bool` + trait `Equatable` |
| `__hash__` | `def __hash__(self) -> UInt` + trait `Hashable` |
| `__len__` | `def __len__(self) -> Int` + trait `Sized` |
| `__str__` | `def write_to(self, mut writer: Some[Writer])` |
| `__repr__` | `def write_repr_to(self, mut writer: Some[Writer])` |

**`__copyinit__` y `__moveinit__` ya NO existen.**

## Constantes

| Python | Mojo 1.2.0 |
|---|---|
| `X = 5` | `comptime X = 5` |
| `alias X = 5` (obsoleto) | `comptime X = 5` |

## Excepciones

| Python | Mojo 1.2.0 |
|---|---|
| `try/except` | `try: ... except e: ...` |
| `raise ValueError("msg")` | `raise Error("msg")` |
| `raise CustomError(...)` | `def f() raises CustomError: raise CustomError(...)` |

## Pattern matching (nuevo en 1.2.0)

| Python | Mojo 1.2.0 |
|---|---|
| `if x == 1: ... elif x == 2: ...` | `match x: case 1: ... case 2: ...` |
| `if isinstance(x, Some): ...` | `if case .Some(v) = opt: ...` |
| `while opt is not None: ...` | `while case .Some(v) = opt: ...` |
| `a, *rest, b = values` | `var [a, *rest, b] = values` |

## Closures

| Python | Mojo 1.2.0 |
|---|---|
| `lambda x: x + 1` | `lambda (x: Int) -> Int: x + 1` |
| `def outer(): x = 1; def inner(): return x` | `def inner() {imm x} -> Int: return x` |
| `def inner(): nonlocal x; x += 1` | `def inner() {mut x}: x += 1` |
| `def inner(): return y` (copia propia) | `def inner() {var y}: ...` |

## Strings

| Python | Mojo 1.2.0 |
|---|---|
| `s[0]` | `s[byte=0]` (devuelve `StringSlice`) |
| `s[0:10]` | Sin slice en `String`; usa `s.codepoint_slices()` |
| `len(s)` | `s.byte_length()` o `s.count_codepoints()` |
| `s.split(",")` | Devuelve `List[StringSlice]`; envuelve con `String(...)` si necesitas `String` |
| `s += "x"` | `s += "x"` (String es mutable) |
| `StringSpan` mutable | **No existe.** `StringSpan` es inmutable; `MutStringSpan` y `MutStringSlice` fueron eliminados. |

## Punteros y memoria

| Python | Mojo 1.2.0 |
|---|---|
| `ctypes` / `cffi` | `Pointer[T]` (principal); `UnsafePointer[T]` es alias compatible |
| `malloc` / `free` | `alloc(Layout[T](length=n))` → `Allocation[T]`; `dealloc(allocation^)` |
| `None` como puntero nulo | `OptionalPointer[T]` |
| `id(obj)` | `Int(unsafe_ptr)` (solo si necesitas la dirección) |
| `hash(ptr, len)` (obsoleto) | `hash_bytes()` |

## Librerías

| Python | Estrategia |
|---|---|
| `numpy` | Interop o reescritura nativa con `SIMD` y `TileTensor` |
| `pandas` | Interop (no hay equivalente nativo) |
| `requests` | Interop o `std.net` (si está disponible) |
| `json` | `std.json` o interop |
| `re` | Interop |
| `flask`/`django` | Interop o rediseño |
| `threading`/`multiprocessing` | `parallelize` (CPU-bound) o interop para I/O |
| `asyncio` | No migrar a `__async`/`__await` (inestable); usa interop o `parallelize` |

## Interop Python

```mojo
from std.python import Python, PythonObject

def main() raises:
    var np = Python.import_module("numpy")
    var arr = np.array([1, 2, 3])
    var i = Int(py=arr[0])   # conversión explícita con `py=`
```

## Asincronía

`__async def` y `__await expr` existen en Mojo 1.2.0, pero el soporte async está **inestable** y no debe usarse en código de producción. Para paralelismo de CPU, usa `parallelize` (ver skill `mojo-parallelize`). Para I/O, usa interop con Python (`asyncio`) o un pool externo.

## Recursos adicionales

- [Mojo 1.2.0 release notes](https://mojolang.org/releases/v1.2.0/)
- [Mojo syntax skill](../../mojo-syntax/SKILL.md)
- [Mojo Python interop skill](../../mojo-python-interop/SKILL.md)
- [Mojo concurrency patterns skill](../../mojo-concurrency-patterns/SKILL.md)
