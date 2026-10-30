---
name: mojo-networking
description: Redes en Mojo 1.2.0. Cubre la ausencia de un cliente HTTP público en std.net, el uso de bibliotecas de terceros como flare (HTTP/1.1, HTTP/2, HTTP/3, WebSocket, TLS, TCP, UDP) y lightbug_http, y la interop con Python (requests, aiohttp) para prototipos.
---

# Redes en Mojo 1.2.0

**`std.net` no expone tipos públicos estables en Mojo 1.2.0**: el paquete contiene solo bloques internos; sus tipos públicos aún no están disponibles. Para redes en producción, usa una biblioteca de terceros o interop con Python.

## Opciones

| Enfoque | Alcance |
|---------|---------|
| **flare** | Stack completo: HTTP/1.1, HTTP/2, HTTP/3 (QUIC), WebSocket, TLS, TCP, UDP, Unix sockets, DNS. |
| **lightbug_http** | Framework HTTP simple y rápido para Mojo. |
| **Interop con Python** | `requests`, `aiohttp`, `httpx` para prototipos. |

### flare: servidor y cliente

```bash
pixi add flare --git https://github.com/ehsanmok/flare.git
```

```mojo
from flare.prelude import *

def hello(req: Request) -> Response:
    return ok("hello")

def main() raises:
    var r = Router()
    r.get("/", hello)
    var srv = HttpServer.bind(SocketAddr.localhost(8080))
    srv.serve(r^, num_workers=2)
```

Cliente:

```mojo
from flare.prelude import *

def main() raises:
    with HttpClient("https://example.com") as c:
        var r = c.get("/")
        print(r.status, r.text())
```

flare negocia HTTP/2 vía ALPN y permite optar a HTTP/3; el servidor es thread-per-core con reactor no bloqueante (`epoll`/`kqueue`, opt-in `io_uring`).

### Interop con Python

```mojo
from std.python import Python

def fetch(url: String) raises -> String:
    var requests = Python.import_module("requests")
    var resp = requests.get(url)
    return String(py=resp.text)
```

## Anti-patrones

- **No esperes un cliente HTTP en `std.net`**: no está expuesto públicamente en 1.2.0.
- **No uses `requests` en producción de alto rendimiento**: es Python puro; usa flare.
- **No implementes TLS desde cero**: usa flare o interop.
- **No bloquees el hilo principal en operaciones de red**: usa el modelo de flare (reactor no bloqueante) o `parallelize` para I/O fuera del hilo.

## Reglas prácticas

1. **Para servidores HTTP en Mojo, usa flare**.
2. **Para clientes HTTP en Mojo, usa flare** (`HttpClient`).
3. **Para prototipos, usa interop con Python** (`requests`).
4. **Para TCP/UDP crudo, usa flare** o `std.net` cuando sus tipos públicos estén disponibles.
5. **No asumas que `std.net` es estable**: verifica la versión antes de depender de él.

