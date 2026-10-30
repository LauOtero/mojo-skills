---
name: mojo-serialization
description: Serialización y deserialización en Mojo 1.2.0. Cubre la ausencia de std.json oficial, el uso de EmberJSON (terceros) para JSON con reflexión, la interop con Python para JSON/YAML, y la serialización binaria con Writable/Writer y Span[Byte].
---

# Serialización en Mojo 1.2.0

**No existe un módulo `std.json` oficial en la stdlib de Mojo 1.2.0.** La propuesta de incluirlo sigue abierta; hasta entonces, las opciones son bibliotecas de terceros o interop con Python.

## JSON: opciones

| Enfoque | Cuándo usarlo |
|---------|---------------|
| **EmberJSON** (terceros) | JSON puro en Mojo, con reflexión y serialización estilo serde. |
| **Interop con Python** (`json`) | Prototipos, scripts, cuando ya tienes Python en el proyecto. |
| **Implementación manual** | Estructuras simples con `Writable`/`Reader`. |

### EmberJSON

```bash
pixi add emberjson
```

```mojo
from emberjson import parse, stringify

@fieldwise_init
struct Config(Copyable, Movable):
    var name: String
    var count: Int

def main() raises:
    var cfg = Config("demo", 3)
    var s = stringify(cfg)
    var back = parse[Config](s)
```

EmberJSON usa reflexión en tiempo de compilación y anotaciones para mapear structs a JSON; no envuelve simdjson ni C/C++/Rust.

### Interop con Python

```mojo
from std.python import Python, PythonObject

def to_json(value: PythonObject) raises -> String:
    var json = Python.import_module("json")
    return String(py=json.dumps(value))

def from_json(s: String) raises -> PythonObject:
    var json = Python.import_module("json")
    return json.loads(s)
```

## YAML

No hay un módulo YAML en la stdlib. Usa interop con Python (`yaml`) o una biblioteca de terceros.

## Serialización binaria

Para formatos binarios, implementa `Writable`/`Writer`:

```mojo
from std.io import Writer

struct Header(Writable):
    var magic: UInt32
    var version: UInt16

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.magic)
        writer.write(self.version)
```

Para leer, usa `Reader` y convierte los bytes explícitamente. Para buffers crudos, `Span[Byte]` o `List[UInt8]` con `Pointer` y `unsafe_load`.

## Anti-patrones

- **No intentes `import std.json`**: no existe en Mojo 1.2.0.
- **No uses `json` de Python para todo**: si el rendimiento importa, usa EmberJSON.
- **No serialices con `String` concatenado**: usa un `Writer` o una biblioteca.
- **No asumas que un struct es serializable por defecto**: necesita `Writable` o una biblioteca con reflexión.

## Reglas prácticas

1. **Para JSON en Mojo puro, usa EmberJSON**.
2. **Para prototipos o scripts, usa interop con Python**.
3. **Para binario, implementa `Writable`/`Writer`**.
4. **Para YAML, usa interop con Python**.
5. **Valida la entrada** antes de deserializar en producción.
