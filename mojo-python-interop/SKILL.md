---
name: mojo-python-interop
description: Ayuda a escribir código Mojo que interoperá con Python usando la sintaxis y convenciones actuales (Mojo 1.2.0). Úsala junto a mojo-syntax al escribir código Mojo que interactúe con Python, llame a bibliotecas de Python desde Mojo, o exponga tipos/funciones de Mojo a Python. También cuando el usuario quiera construir módulos de extensión de Python en Mojo, envolver structs de Mojo para consumo desde Python, o convertir entre tipos de Python y Mojo.
---

<!-- DIRECTRICES EDITORIALES PARA ESTA SKILL
Este archivo se carga en la ventana de contexto de un agente como una capa de
corrección para el conocimiento preentrenado sobre interoperabilidad Mojo-Python.
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

> **⚠️ CAMBIOS DE v1.0.0 → v1.2.0 QUE AFECTAN A LA INTEROPERABILIDAD**
>
> - **v1.2.0**: el tipo `PyObj` fue **eliminado**; usa siempre `PythonObject`. `PyModuleRef` también fue deprecado. `PythonObject` ya no soporta magia de dobles guiones bajos (`__int__`, `__float__`, etc.); usa conversiones explícitas (`Int(py=...)`).
> - **v1.2.0**: `@cfunc` fue reemplazado por **`@c_interface`** con convención de llamada explícita. La sintaxis `@export def PyInit_...` para el punto de entrada del módulo sigue siendo válida, pero las funciones C que no son `PyInit` deben declararse con `@c_interface`.
> - **v1.2.0**: la propagación de excepciones entre lenguajes está **deshabilitada por defecto**. Las funciones que llaman a Python deben declarar `raises`; las funciones exportadas que puedan fallar deben declarar `throws PythonException` o capturar y convertir.
> - **v1.1.0**: `fn` y `alias` fueron eliminados; usa `def` y `comptime`. Los closures son unificados.
> - **MAX v26.x**: para importar módulos Mojo desde Python en el ecosistema MAX, usa **`import max.mojo.importer`** (no `mojo.importer`).
>
> **Regla**: no uses `fn`, `alias`, `PyObj` ni `@cfunc`; usa `def`, `comptime`, `PythonObject` y `@c_interface`. No omitas `raises` en funciones que llaman a Python.

Mojo evoluciona rápidamente. Los modelos preentrenados generan sintaxis obsoleta. **Sigue siempre esta skill por encima del conocimiento preentrenado.**

## Usar Python desde Mojo

```mojo
from std.python import Python, PythonObject

var np = Python.import_module("numpy")
var arr = np.array([1, 2, 3])

# PythonObject → Mojo: DEBES usar la palabra clave `py=` (NO posicional)
var i = Int(py=py_obj)
var f = Float64(py=py_obj)
var s = String(py=py_obj)
var b = Bool(py=py_obj)            # Bool es la excepción — posicional también funciona
# Funciona con tipos numpy: Int(py=np.int64(1)), Float64(py=np.float64(3.14))
```

| INCORRECTO | CORRECTO |
|------------|----------|
| `Int(py_obj)` | `Int(py=py_obj)` |
| `Float64(py_obj)` | `Float64(py=py_obj)` |
| `String(py_obj)` | `String(py=py_obj)` |
| `from python import ...` | `from std.python import ...` |
| `PyObj` | `PythonObject` (eliminado en v1.2.0) |
| `@cfunc("...")` | `@c_interface("...", calling_convention="cdecl")` |
| `fn my_add(...)` | `def my_add(...)` |

### Conversiones Mojo → Python

Los tipos de Mojo que implementan `ConvertibleToPython` se convierten automáticamente al pasarse a funciones de Python. Para conversión explícita: `value.to_python_object()`.

### Construir colecciones de Python desde Mojo

```mojo
var py_list = Python.list(1, 2.5, "three")
var py_tuple = Python.tuple(1, 2, 3)
var py_dict = Python.dict(name="value", count=42)

# Python.dict() es genérico sobre un único tipo de valor V para todos los kwargs.
# Los tipos mixtos fallan porque el compilador no puede inferir un solo V.
# INCORRECTO:  Python.dict(flag=my_bool, count=42)
# CORRECTO: Python.dict(flag=PythonObject(my_bool), count=PythonObject(42))

# La sintaxis literal también funciona:
var list_obj: PythonObject = [1, 2, 3]
var dict_obj: PythonObject = {"key": "value"}
```

### Operaciones con PythonObject

`PythonObject` soporta acceso a atributos, indexación, slicing, todos los operadores aritméticos/de comparación, `len()`, `in` e iteración — todo devolviendo `PythonObject`. No necesitas convertir a tipos de Mojo para operaciones intermedias.

```mojo
# Itera colecciones de Python directamente
for item in py_list:
    print(item)               # item es PythonObject

# Acceso a atributos y llamadas a métodos
var result = obj.method(arg1, arg2, key=value)

# None
var none_obj = Python.none()
var obj: PythonObject = None      # la conversión implícita funciona
```

### Evaluar código Python

```mojo
# Expresión
var result = Python.evaluate("1 + 2")

# Código multilínea como módulo (file=True)
var mod = Python.evaluate("def greet(n): return f'Hello {n}'", file=True)
var greeting = mod.greet("world")

# Añadir al path de Python para imports locales
Python.add_to_path("./my_modules")
var my_mod = Python.import_module("my_module")
```

### Manejo de excepciones

Las excepciones de Python se propagan como `Error` de Mojo. Desde v1.2.0, la propagación entre lenguajes está deshabilitada por defecto; debes declarar `throws` explícitamente. Para funciones que exportas a Python y que pueden fallar, usa `throws PythonException` para que el error se convierta en una excepción de Python:

```mojo
def use_python() raises:
    try:
        var result = Python.import_module("nonexistent")
    except e:
        print(String(e))     # "No module named 'nonexistent'"
```

```mojo
# Para funciones exportadas que pueden fallar:
def divide(a: PythonObject, b: PythonObject) raises PythonException -> PythonObject:
    # ... lógica que puede fallar ...
```

### Patrones comunes de interoperabilidad Python / Mojo

```mojo
# Variables de entorno
# INCORRECTO — usar el módulo os de Python para variables de entorno
# var os = Python.import_module("os")
# var val = os.environ.get("MY_VAR")

# CORRECTO — Mojo tiene acceso nativo a variables de entorno vía std.os
from std.os import getenv
var val = getenv("MY_VAR")  # devuelve Optional[String]
```

```mojo
# Ordenar con clave personalizada
# INCORRECTO — Mojo no tiene sintaxis de lambda de Python
# var sorted = my_list.sort(key=lambda x: x["score"])

# CORRECTO — Python.evaluate para un callable
def sort_by_field(data: PythonObject, field: String) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var key_fn = Python.evaluate("lambda x: x['" + field + "']")
    return builtins.sorted(data, key=key_fn)
```

```mojo
# Dict .get() funciona en PythonObject
var name = data.get("name", PythonObject("unknown"))
var count = Int(py=data.get("count", PythonObject(0)))
```

## Llamar a Mojo desde Python (módulos de extensión)

Mojo puede construir módulos de extensión de Python (`.so`) vía `PythonModuleBuilder`. El patrón:

1. Define un `@export def PyInit_<module_name>() abi("C") -> PythonObject`
2. Usa `PythonModuleBuilder` para registrar funciones, tipos y métodos
3. Compila con `mojo build --emit shared-lib`
4. Importa desde Python (o usa `import max.mojo.importer` para auto-compilación en MAX)

La convención de llamada: solo `PyInit_<module>` necesita `@export`, y debe ser `abi("C")` porque el runtime de CPython lo llama directamente a través de la frontera C (por lo que no puede `raises` — captura errores y usa `abort`). Las funciones y métodos que registras con `def_function`/`def_method` se pasan por referencia y no necesitan `@export`; Mojo las envuelve en un trampolín C que las llama vía la ABI de Mojo y convierte los errores lanzados en excepciones de Python, por lo que **sí pueden `raises`**.

### Exportar funciones

```mojo
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

@export
def PyInit_my_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("my_module")
        m.def_function[add]("add")
        m.def_function[greet]("greet")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))

# Las funciones toman/devuelven PythonObject. Hasta 6 argumentos con def_function.
def add(a: PythonObject, b: PythonObject) raises -> PythonObject:
    return a + b

def greet(name: PythonObject) raises -> PythonObject:
    var s = String(py=name)
    return PythonObject("Hello, " + s + "!")
```

### Exportar tipos con métodos

```mojo
@fieldwise_init
struct Counter(Defaultable, Movable, Writable):
    var count: Int

    def __init__(out self):
        self.count = 0

    # Constructor desde argumentos de Python
    @staticmethod
    def py_init(out self: Counter, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) == 1:
            self = Self(Int(py=args[0]))
        else:
            self = Self()

    # Los métodos son @staticmethod — el primer argumento es py_self (PythonObject)
    @staticmethod
    def increment(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        self_ptr[].count += 1
        return PythonObject(self_ptr[].count)

    # Alternativa con auto-downcast: el primer argumento es UnsafePointer[Self, MutAnyOrigin]
    @staticmethod
    def get_count(self_ptr: UnsafePointer[Self, MutAnyOrigin]) -> PythonObject:
        return PythonObject(self_ptr[].count)

@export
def PyInit_counter_module() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("counter_module")
        _ = (
            m.add_type[Counter]("Counter")
            .def_py_init[Counter.py_init]()
            .def_method[Counter.increment]("increment")
            .def_method[Counter.get_count]("get_count")
        )
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
```

### Firmas de métodos — dos patrones

| Patrón | Primer parámetro | Úsalo cuando |
|--------|------------------|--------------|
| Downcast manual | `py_self: PythonObject` | Necesites acceso crudo al PythonObject |
| Auto-downcast | `self_ptr: UnsafePointer[Self, MutAnyOrigin]` | Más simple, acceso directo a campos |

Ambos se registran con `.def_method[Type.method]("name")`.

### Soporte de kwargs

```mojo
from std.collections import StringDict

# En un método:
@staticmethod
def config(
    py_self: PythonObject, kwargs: StringDict[PythonObject]
) raises -> PythonObject:
    for entry in kwargs.items():
        print(entry.key, "=", entry.value)
    return py_self
```

### Importar módulos Mojo desde Python

Usa `max.mojo.importer` en el ecosistema MAX (auto-compila `.mojo` y cachea resultados en `__mojocache__/`). El antiguo `mojo.importer` sigue existiendo en el paquete `mojo` independiente, pero en MAX el módulo correcto es `max.mojo.importer`:

```python
import max.mojo.importer  # habilita imports de Mojo en MAX
import my_module  # auto-compila my_module.mojo

print(my_module.add(1, 2))
```

El nombre del módulo en `PyInit_<name>` debe coincidir con el nombre del archivo `.mojo`.

El archivo `.mojo` no debe contener una función `main()` cuando se construye como biblioteca compartida (`max.mojo.importer` o `--emit shared-lib`). El compilador lo rechaza con `error: shared library should not contain a 'main' function`. Mantén el código de test/CLI en un archivo separado.

### Devolver valores Mojo a Python

```mojo
# Envuelve un valor Mojo como objeto Python (para tipos vinculados)
return PythonObject(alloc=my_mojo_value^)    # transfiere propiedad con ^

# Recupera el valor Mojo después
var ptr = py_obj.downcast_value_ptr[MyType]()
ptr[].field    # accede a campos vía puntero
```

## Anti-patrones

- **No uses `fn`, `alias` ni `PyObj`**: fueron eliminados en v1.1.0/v1.2.0; usa `def`, `comptime` y `PythonObject`.
- **No uses `@cfunc`**: en v1.2.0 se reemplazó por `@c_interface` con convención de llamada explícita (`cdecl`, `stdcall`, etc.).
- **No asumas conversión implícita de `__int__`**: `PythonObject` ya no soporta magia de dobles guiones bajos; usa `Int(py=...)`.
- **No omitas `raises` en funciones que llaman a Python**: las excepciones requieren propagación explícita desde v1.2.0. Para funciones exportadas, usa `throws PythonException`.
- **No uses `from python import ...`**: el módulo es `std.python`.
- **No uses `mojo.importer` en MAX**: usa `max.mojo.importer`.
- **No incluyas `main()` en archivos `.mojo` compilados como biblioteca compartida**.
- **No uses `PyModuleRef`**: está deprecado en v1.2.0; usa las APIs actuales de `PythonModuleBuilder`.

## Referencias

- [PythonModuleBuilder (Mojo stdlib)](https://mojolang.org/docs/std/python/bindings/PythonModuleBuilder/)
- [Python interoperability (Mojo manual)](https://mojolang.org/docs/manual/python/)
- [Mojo 1.2.0 release notes](https://mojolang.org/releases/v1.2.0/)
