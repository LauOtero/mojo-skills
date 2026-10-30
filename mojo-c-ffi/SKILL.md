---
name: mojo-c-ffi
description: Llama a bibliotecas C desde Mojo 1.2.0 con std.ffi. Cubre external_call, CStringSpan, OwnedDLHandle, tipos C (c_int, c_char, c_size_t), funciones variádicas y carga dinámica de bibliotecas. Usa esta skill cuando necesites BLAS, CUDA, libc o cualquier API C nativa.
---

# C FFI en Mojo 1.2.0

`std.ffi` proporciona la interfaz para llamar a código C y cargar bibliotecas dinámicas. Mojo emite una llamada nativa directa, sin capa de traducción ni overhead de runtime: una llamada C desde Mojo corre tan rápido como C escrito a mano.

## Llamar a funciones de libc con `external_call`

```mojo
from std.ffi import c_int, external_call

def main():
    var n = external_call["abs", c_int](c_int(-42))
    print(t"valor absoluto: {n == 42}")  # True
```

`external_call[callee, return_type, *types](*args)` resuelve el símbolo en tiempo de compilación e infiere los tipos de los argumentos.

## Tipos C portables

| Alias | Tipo Mojo | Notas |
|-------|-----------|-------|
| `c_int` | `Int32` | 32 bits en todos los targets de Mojo |
| `c_char` | `Scalar[_c_char_dtype()]` | Signedness depende del ABI |
| `c_uchar` | `UInt8` | `unsigned char` |
| `c_long` | `Scalar[_c_long_dtype()]` | 64 bits en Linux/macOS, 32 en Windows |
| `c_size_t` | `UInt` | Tamaño de `size_t` |
| `c_float` / `c_double` | `Float32` / `Float64` | — |

Usa siempre los alias de `std.ffi`; no asumas que `Int` es `c_int`.

## Strings C: `CStringSpan`

`CStringSpan` representa una cadena terminada en `\0`. Convierte desde `String` con `as_c_string_span()`:

```mojo
from std.ffi import CStringSpan, c_char, external_call

def getenv_wrapper(name: CStringSpan) -> CStringSpan:
    return external_call["getenv", CStringSpan](name)
```

## Funciones variádicas

Para `printf`/`snprintf`, pasa `num_fixed_args` con el número de argumentos fijos antes de `...`:

```mojo
from std.ffi import c_char, c_int, c_size_t, external_call

var buf = Array[c_char, 64](uninitialized=True)
var written = external_call["snprintf", c_int, num_fixed_args=3](
    buf.unsafe_ptr(),
    c_size_t(64),
    "score: %d/%d".as_c_string_span(),
    c_int(7),
    c_int(10),
)
```

Sin `num_fixed_args`, Mojo trata todos los argumentos como fijos.

## Cargar bibliotecas dinámicas: `OwnedDLHandle`

```mojo
from std.ffi import OwnedDLHandle

def main() raises:
    var lib = OwnedDLHandle("libm.so")
    var sqrt = lib.get_function[Float64]("sqrt")
    print(sqrt(4.0))  # 2.0
```

`OwnedDLHandle` carga una `.so`/`.dylib`/`.dll` en runtime y permite obtener funciones por nombre.

## Anti-patrones

- **No uses `Int` donde la API C espera `c_int`**: usa los alias de `std.ffi`.
- **No pases `String` directamente a `external_call`**: convierte con `as_c_string_span()`.
- **No olvides `num_fixed_args` en variádicas**: el comportamiento es indefinido si lo omites.
- **No cargues la misma biblioteca repetidamente**: reutiliza el `OwnedDLHandle`.
- **No asumas que `c_long` es 64 bits**: en Windows es 32 bits.

## Reglas prácticas

1. **Usa `std.ffi` para todo C**: `external_call`, `CStringSpan`, `OwnedDLHandle`.
2. **Convierte tipos explícitamente**: `c_int(-42)`, `c_size_t(64)`.
3. **Para BLAS/LAPACK**, declara los símbolos con `external_call` y pasa `Pointer[Float64]` a los buffers.
4. **Para CUDA**, carga `libcudart.so` con `OwnedDLHandle` y obtén las funciones con `get_function`.
5. **Verifica el ABI**: los tipos C varían entre plataformas; usa los alias, no tipos fijos.

