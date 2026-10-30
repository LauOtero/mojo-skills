---
name: mojo-python-to-mojo-class-conversion
description: Convierte clases Python a structs y traits de Mojo 1.2.0. Ejecuta un pipeline de 7 fases con análisis de herencia, mapeo a traits, conversión de métodos especiales y validación idiomática. Usa py2mojo solo como borrador inicial (está desactualizado) y corrige su salida contra mojo-syntax. Úsalo al migrar clases Python a Mojo, al portar jerarquías de herencia, o al envolver objetos Python para consumo desde Mojo.
---

# Conversión de clases Python a Mojo 1.2.0

> **⚠️ ADVERTENCIA SOBRE `py2mojo`**
>
> `py2mojo` es una herramienta **de terceros, no oficial**, cuya última actualización data de 2024 y apunta a **Mojo 1.0**. Su salida contiene sintaxis que Mojo 1.2.0 **rechaza** (`fn`, `alias`, `@value`, `Stringable`, `let`, `inout`, `owned`, `__copyinit__`, `__moveinit__`, `@register_passable`, `DynamicVector`, `InlinedFixedVector`, `Tensor`, `UnsafePointer`).
>
> Úsalo **solo como borrador inicial** y trata su salida como código no válido hasta que pase el pipeline de corrección de esta skill y la revisión de `mojo-syntax`. No lo ejecutes sobre código de producción sin revisar el diff.

## Cuándo usar esta skill

- Al migrar una clase Python a un `struct` de Mojo.
- Al portar una jerarquía de herencia Python a traits y composición de Mojo.
- Al envolver un objeto Python opaco en un `struct` de Mojo que lo expone de forma tipada.
- Al convertir una dataclass, un `NamedTuple` o una clase con `__slots__` a un `struct` de Mojo.

**No la uses** para convertir funciones sueltas (usa `mojo-python-interop`), ni para migrar módulos completos (divide el trabajo por clase).

## Pipeline de 7 fases

### Fase 0 — Preprocesamiento con `py2mojo` (solo borrador)

Ejecuta `py2mojo` para obtener un borrador, pero **no confíes en su salida**:

```bash
pip install py2mojo
py2mojo mi_clase.py --convert-class-to-struct --extension mojo
```

Abre el `.mojo` generado y aplícale inmediatamente la tabla de correcciones de sintaxis de abajo. Si `py2mojo` falla o produce código irreconocible, salta directamente a la Fase 1 y escribe el struct a mano; el tiempo perdido depurando la salida de `py2mojo` suele ser mayor que escribirlo desde cero.

#### Correcciones obligatorias sobre la salida de `py2mojo`

| Si `py2mojo` genera… | Cámbialo por… |
|----------------------|---------------|
| `fn nombre(...)` | `def nombre(...)` |
| `alias X = ...` | `comptime X = ...` |
| `@value` | `@fieldwise_init` + conformidad explícita de traits |
| `Stringable` / `__str__` | `Writable` / `write_to` |
| `let x = ...` | `var x = ...` |
| `borrowed` | `imm` (implícito; rara vez se escribe) |
| `inout` | `mut` |
| `owned` | `var` (como convención de argumento) |
| `inout self` en `__init__` | `out self` |
| `__copyinit__(inout self, existing: Self)` | `__init__(out self, *, copy: Self)` |
| `__moveinit__(inout self, owned existing: Self)` | `__init__(out self, *, deinit move: Self)` |
| `@register_passable("trivial")` | trait `TrivialRegisterPassable` |
| `@register_passable` | trait `RegisterPassable` |
| `DynamicVector[T]` | `List[T]` |
| `InlinedFixedVector[T, N]` | `Array[T, N]` |
| `Tensor[T]` | `List`, `SIMD` o `Pointer` (según el uso) |
| `UnsafePointer[T]` | `Pointer[T]` |
| `from collections import ...` | `from std.collections import ...` |
| `from memory import ...` | `from std.memory import ...` |
| `from os import ...` | `from std.os import ...` |
| `from sys import ...` | `from std.sys import ...` |
| `from python import ...` | `from std.python import ...` |

### Fase 1 — Análisis de la clase Python

Antes de escribir Mojo, responde por escrito:

1. **¿Es una dataclass o un contenedor de datos?** → `@fieldwise_init` + traits `Copyable, Movable, Writable, Equatable`.
2. **¿Tiene herencia?** → Cada clase base se mapea a un trait (`trait Base(...)`) o a composición (un campo `var base: BaseStruct`). La herencia múltiple se mapea a múltiples traits, no a múltiples structs base.
3. **¿Usa `__slots__`?** → Todos los campos van declarados con `var` en el struct; no hay diccionario dinámico.
4. **¿Tiene métodos que mutan el estado?** → Esos métodos toman `mut self`; los que no mutan toman `self` (implícito `imm`).
5. **¿Usa `isinstance` o duck typing?** → En Mojo se reemplaza por traits y `match`/`case` sobre variantes, o por `PythonObject` si el tipo es realmente dinámico.
6. **¿Es un objeto Python opaco que no se puede reimplementar?** → Envuélvelo en un `struct` con un campo `var obj: PythonObject` y expón métodos tipados que deleguen en `obj`.

### Fase 2 — Mapeo de herencia a traits y composición

| Patrón Python | Patrón Mojo 1.2.0 |
|---------------|-------------------|
| `class A: ...` sin herencia | `struct A(...): ...` |
| `class B(A): ...` (herencia simple) | `trait A: ...` y `struct B(A): ...` si `A` es una interfaz; o `struct B: var a: A` si `A` tiene estado |
| `class C(A, B): ...` (herencia múltiple) | `struct C(A, B): ...` solo si `A` y `B` son traits; si tienen estado, usa composición (`var a: A`, `var b: B`) |
| `class D(ABC): @abstractmethod def f(self): ...` | `trait D: def f(self) -> ...` y `struct DImpl(D): def f(self) -> ...` |
| Mixin con estado | Composición: un campo del struct mixin, no herencia |

**Regla**: la herencia de implementación se convierte en composición; la herencia de interfaz se convierte en traits. No intentes replicar la herencia múltiple de Python con herencia múltiple de structs; Mojo no la soporta.

### Fase 3 — Conversión del constructor y los campos

```python
# Python
class Point:
    def __init__(self, x: float, y: float = 0.0):
        self.x = x
        self.y = y
```

```mojo
# Mojo 1.2.0
@fieldwise_init
struct Point(Copyable, Movable, Writable):
    var x: Float64
    var y: Float64

    def __init__(out self, x: Float64, y: Float64 = 0.0):
        self.x = x
        self.y = y
```

- `@fieldwise_init` genera el constructor por campos; solo escribe `__init__` manual si necesitas validación, valores por defecto no triviales o lógica adicional.
- Los campos se declaran con `var`; no hay `self.x = x` fuera de `__init__` para crear campos nuevos.
- Los argumentos por defecto son válidos en Mojo (`y: Float64 = 0.0`).
- Si la clase Python usa `__post_init__`, su lógica va en `__init__` después de asignar los campos.

### Fase 4 — Conversión de tipos y colecciones

| Python | Mojo 1.2.0 |
|--------|-----------|
| `list[T]` | `List[T]` |
| `tuple[T, ...]` (homogéneo) | `List[T]` o `Span[T]` |
| `tuple[A, B]` (heterogéneo) | `Tuple[A, B]` |
| `dict[K, V]` | `Dict[K, V]` |
| `set[T]` | `Set[T]` |
| `frozenset[T]` | `Set[T]` (no hay frozenset; usa `Set` con disciplina) |
| `Optional[T]` / `T \| None` | `Optional[T]` |
| `int` | `Int` (o `Int64`/`Int32` si necesitas ancho fijo) |
| `float` | `Float64` |
| `str` | `String` |
| `bytes` | `List[UInt8]` o `Span[UInt8]` |
| `bool` | `Bool` |
| `Any` | `PythonObject` si es realmente dinámico; si no, un trait |
| `Callable[[A], B]` | `def(A) -> B` (closure trait) |

### Fase 5 — Métodos especiales (`__str__`, `__eq__`, `__len__`, …)

| Python | Mojo 1.2.0 |
|--------|-----------|
| `__str__` | `write_to(self, mut writer: Some[Writer])` |
| `__repr__` | `write_repr_to(self, mut writer: Some[Writer])` |
| `__eq__` | `def __eq__(self, other: Self) -> Bool` + trait `Equatable` |
| `__hash__` | `def __hash__(self) -> UInt` + trait `Hashable` |
| `__len__` | `def __len__(self) -> Int` + trait `Sized` |
| `__iter__` | `def __iter__(ref self) -> Iterator` + trait `Iterable` |
| `__getitem__` | `def __getitem__(self, i: Int) -> T` (índice por `Int`; para slices, métodos explícitos) |
| `__call__` | `def __call__(self, ...) -> ...` |
| `__enter__` / `__exit__` | `def __enter__(self) -> Self` / `def __exit__(self, ...)` (context manager) |
| `__copy__` / `__deepcopy__` | `def __init__(out self, *, copy: Self)` (copia superficial); copia profunda como método explícito |
| `__del__` | `def __deinit__(deinit self)` |

### Fase 6 — Validación y revisión idiomática

1. **Compila**: `mojo build mi_clase.mojo`. Corrige todos los errores de sintaxis antes de continuar.
2. **Prueba**: escribe un test mínimo con `TestSuite` que instancie el struct, llame a cada método y compare con la salida de la clase Python original.
3. **Revisa contra `mojo-syntax` y `idiomatic-mojo.md`**: busca `fn`, `alias`, `let`, `@value`, `Stringable`, `inout`, `owned`, `UnsafePointer`, `rebind` innecesario, `MutAnyOrigin` en entradas, `IndexList` en lugar de tuplas, parámetros sobrespecificados.
4. **Comprueba la paridad**: si la clase Python original se puede ejecutar, compara la salida de ambos en los mismos casos de prueba.

### Fase 7 — Reporte

Reporta al usuario:

- Qué clases se convirtieron y a qué structs.
- Qué herencia se mapeó a traits y qué herencia se mapeó a composición.
- Qué métodos especiales se implementaron y cuáles se omitieron (con justificación).
- Qué partes de la salida de `py2mojo` se corrigieron manualmente.
- Qué tests se añadieron y si pasan.

## Matriz de decisión: struct nativo vs. wrapper de `PythonObject`

| Situación | Enfoque |
|-----------|---------|
| La clase es pura lógica, sin dependencias de Python | **Struct nativo** con campos tipados y traits. |
| La clase usa bibliotecas de Python que no tienen equivalente en Mojo (p. ej. `requests`, `numpy` para operaciones complejas) | **Wrapper de `PythonObject`**: un struct con `var obj: PythonObject` y métodos que llaman a `obj.metodo(...)`. |
| La clase es una dataclass simple | **Struct nativo** con `@fieldwise_init` y traits `Copyable, Movable, Writable`. |
| La clase tiene herencia múltiple con estado | **Composición**: un struct con campos para cada base con estado, y traits para las interfaces. |
| La clase usa `__getattr__` o atributos dinámicos | **Wrapper de `PythonObject`** o un `Dict[String, PythonObject]` si el conjunto de claves es conocido. |
| La clase es un `Enum` | **Struct con constantes `comptime`** y, si necesitas despacho exhaustivo, conformidad a `EnumLike` (ver `idiomatic-mojo.md` §10). |

## Anti-patrones

- **No ejecutes `py2mojo` y aceptes su salida sin revisarla**: genera sintaxis de Mojo 1.0 que Mojo 1.2.0 rechaza.
- **No uses `fn`, `alias`, `let`, `@value`, `Stringable`, `inout`, `owned`, `UnsafePointer`** en el código final.
- **No intentes replicar la herencia múltiple de Python con herencia múltiple de structs**: Mojo no la soporta; usa traits y composición.
- **No uses `PythonObject` para todo**: si la lógica se puede reimplementar en Mojo, un struct nativo es más rápido, tipado y testeable.
- **No omitas los tests de paridad**: una clase Python que funciona no garantiza que su struct Mojo equivalente se comporte igual en los bordes (valores por defecto, mutación, errores).
- **No conviertas una clase Python a un struct Mojo sin revisar `idiomatic-mojo.md`**: el código compila pero puede violar las convenciones de revisión.

## Referencias

- [Mojo 1.2.0 release notes](https://mojolang.org/releases/v1.2.0/)
- [mojo-syntax](../mojo-syntax/SKILL.md) — sintaxis actual y tabla de elementos eliminados
- [mojo-python-interop](../mojo-python-interop/SKILL.md) — `PythonObject`, `PythonModuleBuilder`, conversiones
- [idiomatic-mojo.md](../mojo-syntax/references/idiomatic-mojo.md) — reglas de revisión idiomática
- [py2mojo (GitHub)](https://github.com/msaelices/py2mojo) — herramienta de terceros, desactualizada; úsala solo como borrador
