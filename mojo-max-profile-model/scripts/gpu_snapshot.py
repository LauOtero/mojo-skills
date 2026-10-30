# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Registra la utilización/memoria de GPU mientras corre una carga de trabajo y
luego imprime un resumen.

La comprobación de utilización de GPU de la skill profile-model. Usa
max.profiler.gpu (NVML en NVIDIA, ROCm SMI en AMD), así que no necesita instalar
ningún profiler más allá del paquete `max`.

Dos modos:
  * --run "<cmd>"   : lanza el comando, graba durante su vida útil y resume.
  * (sin --run)     : graba durante --duration segundos mientras manejas la
                      carga desde otra terminal (p. ej. `max benchmark`).

Ejemplos:
  python gpu_snapshot.py --duration 15
  python gpu_snapshot.py --run "pixi run python my_program.py"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time


def _summarize(samples: list[dict]) -> None:
    """Imprime la utilización pico/media y la memoria pico por GPU."""
    if not samples:
        print("No se recogieron muestras — ¿fue la carga de trabajo demasiado corta?")
        return

    # Recoge la serie temporal por GPU.
    per_gpu: dict[str, dict] = {}
    for snapshot in samples:
        for gpu_id, s in snapshot.items():
            g = per_gpu.setdefault(
                gpu_id, {"util": [], "mem": [], "throttles": set()}
            )
            g["util"].append(s.utilization.gpu_usage_percent)
            if s.memory is not None:
                g["mem"].append(s.memory.used_bytes)
            clocks = getattr(s, "clocks", None)
            for reason in getattr(clocks, "throttle_reasons", None) or []:
                # gpu_idle es normal entre muestras; no alarmarse por ello.
                if reason != "gpu_idle":
                    g["throttles"].add(reason)

    print(f"\n=== Utilización de GPU sobre {len(samples)} muestras ===")
    print(f"{'gpu':<6}{'pico%':>7}{'media%':>7}{'mem pico (GB)':>15}  throttles")
    for gpu_id, g in sorted(per_gpu.items()):
        util = g["util"]
        peak = max(util)
        mean = sum(util) / len(util)
        peak_mem = f"{max(g['mem']) / 1e9:.1f}" if g["mem"] else "N/A"
        throttles = ", ".join(sorted(g["throttles"])) or "-"
        print(f"{gpu_id:<6}{peak:>7}{mean:>7.0f}{peak_mem:>15}  {throttles}")

    # Veredicto de una línea para orientar al lector.
    best = max(per_gpu.values(), key=lambda g: max(g["util"]))
    peak = max(best["util"])
    if peak >= 80:
        print(
            "\nVeredicto: la GPU está ocupada en el pico — probablemente "
            "compute-bound. Captura un desglose de kernels (nsys/rocprofv3) "
            "para ver qué kernels dominan."
        )
    elif peak >= 30:
        print(
            "\nVeredicto: uso moderado de GPU — los huecos sugieren overhead de "
            "launch/sync o batch pequeño. Una línea de tiempo de desglose de "
            "kernels mostrará los huecos de inactividad."
        )
    else:
        print(
            "\nVeredicto: GPU mayormente inactiva durante la ventana — el cuello "
            "de botella está probablemente en el host, en la transferencia, o la "
            "carga de trabajo aún no había arrancado. Comprueba que la carga "
            "estaba realmente corriendo."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Segundos a grabar (se ignora cuando --run termina antes). "
        "Por defecto 15.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Intervalo de muestreo en segundos. Por defecto 0.5.",
    )
    ap.add_argument(
        "--run",
        default=None,
        help="Comando opcional a lanzar; la grabación se detiene cuando sale "
        "o cuando transcurre --duration, lo que ocurra primero.",
    )
    args = ap.parse_args()

    try:
        from max.profiler.gpu import BackgroundRecorder
    except ImportError:
        print(
            "ERROR: no se pudo importar max.profiler.gpu. Instala el paquete "
            "`max` (pixi add max / pip install max) y ejecuta dentro de ese "
            "entorno.",
            file=sys.stderr,
        )
        return 1

    proc = None
    if args.run:
        proc = subprocess.Popen(shlex.split(args.run))

    with BackgroundRecorder(interval=args.interval) as rec:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                break
            time.sleep(
                min(args.interval, max(0.0, deadline - time.monotonic()))
            )

    if proc is not None and proc.poll() is None:
        proc.terminate()

    _summarize(rec.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
