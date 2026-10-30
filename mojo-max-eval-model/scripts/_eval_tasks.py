#!/usr/bin/env python3
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
"""
Implementaciones de evaluadores de tareas para la evaluación de precisión de datasets.

Cada tarea implementa:
- `evaluate_seed()` — muestra rápida de fast-fail
- `evaluate_full()` — evaluación completa
- Devuelve un dict de resultado estandarizado para el JSON de resumen

Usa endpoints nativos de MAX:
- Tareas de generación (gsm8k, aime) → `/v1/chat/completions`
- Tareas de loglikelihood (hellaswag, mmlu, arc_*, winogrande, truthfulqa) → `/v1/completions` + `echo=true&logprobs`
"""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

# ─── Configuraciones de tareas (refleja fast-fail-thresholds.md) ───

# Semilla fija para que la selección de muestras de seed/full sea reproducible.
SHUFFLE_SEED = 1234


@dataclass(frozen=True)
class TaskConfig:
    name: str
    type: str  # "generation" | "loglikelihood"
    hf_dataset: str
    hf_split: str
    hf_subset: str | None
    question_key: str
    answer_key: str
    seed_limit: int
    full_limit: int
    fast_fail_floor: float
    lm_eval_task: str  # nombre de tarea para la pasada completa de lm-eval
    is_canary: (
        bool  # cuando es True, una precisión de seed por debajo del suelo dispara el aborto fast-fail
    )
    num_choices: int | None = None  # para tareas MC de loglikelihood
    metric: str = (
        "exact_match"  # prefijo de la clave de métrica de lm-eval (exact_match/acc/acc_norm)
    )
    filter_key: str | None = None
    filter_contains: str | None = None
    # Las tareas lm_eval_only no tienen evaluador directo por HTTP: se saltan la
    # pasada de seed fast-fail y solo corren en la pasada completa (lm-eval). Se
    # usan para tareas de contexto largo y visión donde lm-eval construye la
    # petición (prompts multimodales/largos).
    lm_eval_only: bool = False
    lm_eval_metadata: str | None = (
        None  # reenviado literalmente a lm-eval --metadata
    )
    grouped: bool = False


def select_items(items: list[Any], limit: int) -> list[Any]:
    """Selecciona determinísticamente hasta `limit` elementos (barajado con semilla fija)."""
    ordered = list(items)
    random.Random(SHUFFLE_SEED).shuffle(ordered)
    return ordered[:limit]


TASK_CONFIGS: dict[str, TaskConfig] = {
    "gsm8k": TaskConfig(
        name="gsm8k",
        type="generation",
        hf_dataset="openai/gsm8k",
        hf_split="test",
        hf_subset="main",
        question_key="question",
        answer_key="answer",
        seed_limit=24,
        full_limit=200,
        fast_fail_floor=0.10,
        lm_eval_task="gsm8k_cot_llama",  # variante CoT — coincide con TEXT_TASK del smoke test de serve
        is_canary=True,
        metric="exact_match",
    ),
    "hellaswag": TaskConfig(
        name="hellaswag",
        type="loglikelihood",
        hf_dataset="hellaswag",
        hf_split="validation",
        hf_subset=None,
        question_key="ctx",
        answer_key="label",
        seed_limit=0,
        full_limit=200,
        fast_fail_floor=0.0,
        lm_eval_task="hellaswag",
        is_canary=False,
        num_choices=4,
        metric="acc_norm",
        lm_eval_only=True,
    ),
    "mmlu": TaskConfig(
        name="mmlu",
        type="loglikelihood",
        hf_dataset="cais/mmlu",
        hf_split="test",
        hf_subset="all",
        question_key="question",
        answer_key="answer",
        seed_limit=0,
        full_limit=200,
        fast_fail_floor=0.0,
        lm_eval_task="mmlu",
        is_canary=False,
        num_choices=4,
        metric="acc",
        lm_eval_only=True,
        grouped=True,
    ),
    "gpqa": TaskConfig(
        name="gpqa",
        type="loglikelihood",
        hf_dataset="Idavidrein/gpqa",
        hf_split="test",
        hf_subset="gpqa_diamond",
        question_key="Question",
        answer_key="Correct Answer",
        seed_limit=0,
        full_limit=100,
        fast_fail_floor=0.0,
        lm_eval_task="gpqa_diamond_zeroshot",
        is_canary=False,  # ~25% aleatorio; solo comprobación de paridad, no canario
        metric="acc",
        # Sin puntuación MC directa por HTTP sin juez aquí (el prompt carece de
        # opciones con letras), así que se ejecuta solo vía lm-eval en lugar de
        # enviar un seed roto.
        lm_eval_only=True,
    ),
    "aime": TaskConfig(
        name="aime",
        type="generation",
        hf_dataset="AI-MO/aimo-validation-aime",
        hf_split="train",
        hf_subset=None,
        question_key="problem",
        answer_key="answer",
        seed_limit=10,
        full_limit=30,
        fast_fail_floor=0.05,
        lm_eval_task="aime24",
        is_canary=True,
        metric="exact_match",
        filter_key="url",
        filter_contains="2024_",
    ),
    "arc_easy": TaskConfig(
        name="arc_easy",
        type="loglikelihood",
        hf_dataset="ai2_arc",
        hf_split="test",
        hf_subset="ARC-Easy",
        question_key="question",
        answer_key="answerKey",
        seed_limit=0,
        full_limit=200,
        fast_fail_floor=0.0,
        lm_eval_task="arc_easy",
        is_canary=False,
        num_choices=4,
        metric="acc",
        lm_eval_only=True,
    ),
    "arc_challenge": TaskConfig(
        name="arc_challenge",
        type="loglikelihood",
        hf_dataset="ai2_arc",
        hf_split="test",
        hf_subset="ARC-Challenge",
        question_key="question",
        answer_key="answerKey",
        seed_limit=0,
        full_limit=200,
        fast_fail_floor=0.0,
        lm_eval_task="arc_challenge",
        is_canary=False,
        num_choices=4,
        metric="acc",
        lm_eval_only=True,
    ),
    "winogrande": TaskConfig(
        name="winogrande",
        type="loglikelihood",
        hf_dataset="winogrande",
        hf_split="validation",
        hf_subset="winogrande_debiased",
        question_key="sentence",
        answer_key="answer",
        seed_limit=0,
        full_limit=200,
        fast_fail_floor=0.0,
        lm_eval_task="winogrande",
        is_canary=False,
        num_choices=2,
        metric="acc",
        lm_eval_only=True,
    ),
    "truthfulqa": TaskConfig(
        name="truthfulqa",
        type="loglikelihood",
        hf_dataset="truthful_qa",
        hf_split="validation",
        hf_subset="generation",
        question_key="question",
        answer_key="best_answer",
        seed_limit=0,
        full_limit=200,
        fast_fail_floor=0.0,
        lm_eval_task="truthfulqa_mc2",
        is_canary=False,
        num_choices=None,  # variable
        metric="acc",
        lm_eval_only=True,
    ),
    # ── Tareas solo lm-eval (sin evaluador directo por HTTP) ──
    # Contexto largo: refleja las tareas stock de babilong del smoke test de serve.
    # Apunta a qa1/qa3 con --tasks o a una tarea lm-eval diferente; la longitud de
    # secuencia se establece vía --lm-eval-metadata (por defecto abajo).
    "babilong": TaskConfig(
        name="babilong",
        type="long-context",
        hf_dataset="RMT-team/babilong",
        hf_split="test",
        hf_subset=None,
        question_key="input",
        answer_key="target",
        seed_limit=0,
        full_limit=100,
        fast_fail_floor=0.0,
        lm_eval_task="babilong_qa2",
        is_canary=False,
        metric="acc",
        lm_eval_only=True,
        lm_eval_metadata='{"max_seq_lengths": "16k"}',
    ),
}


# ─── Evaluador base ───


class TaskEvaluator(ABC):
    """Clase base para la lógica de evaluación específica de cada tarea."""

    def __init__(
        self,
        config: TaskConfig,
        base_url: str,
        model: str,
        tokenizer: Any,
        num_concurrent: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        apply_chat_template: bool = False,
    ):
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.num_concurrent = num_concurrent
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.apply_chat_template = apply_chat_template

        self._dataset = None

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            try:
                datasets = importlib.import_module("datasets")
            except ImportError as exc:
                raise RuntimeError(
                    "Falta la dependencia 'datasets'. Instala el entorno del "
                    "evaluador desde el pixi.toml incluido."
                ) from exc
            kwargs = {"split": self.config.hf_split}
            if self.config.hf_subset:
                kwargs["name"] = self.config.hf_subset
            self._dataset = datasets.load_dataset(
                self.config.hf_dataset, **kwargs
            )
            if self.config.filter_key and self.config.filter_contains:
                key = self.config.filter_key
                value = self.config.filter_contains
                self._dataset = self._dataset.filter(
                    lambda item: value in str(item.get(key, ""))
                )
        return self._dataset

    @abstractmethod
    def evaluate_item(self, item: dict, is_seed: bool) -> tuple[bool, dict]:
        """Evalúa un elemento. Devuelve (correcto, dict_de_detalles)."""

    def _post(self, endpoint: str, payload: dict, timeout: int = 120) -> dict:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"{url} devolvió HTTP {exc.code}: {body}"
            ) from exc

    def run_evaluation(
        self,
        limit: int,
        is_seed: bool,
        heartbeat_cb: Callable[[dict], None] | None = None,
        heartbeat_interval: int = 30,
    ) -> dict:
        """Ejecuta la evaluación sobre `limit` elementos con callbacks de heartbeat."""
        correct = 0
        completed = 0
        start_time = time.time()
        details_list = []
        state_lock = threading.Lock()
        stop_heartbeat = threading.Event()

        def heartbeat_data(force: bool = False) -> dict:
            with state_lock:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = int((limit - completed) / rate) if rate > 0 else None
                accuracy = correct / completed if completed > 0 else None
                return {
                    "phase": "seed" if is_seed else "full",
                    "task": self.config.name,
                    "completed": completed,
                    "total": limit,
                    "accuracy": accuracy,
                    "eta_sec": eta,
                    "force": force,
                }

        def heartbeat_loop() -> None:
            while not stop_heartbeat.wait(heartbeat_interval):
                if heartbeat_cb:
                    heartbeat_cb(heartbeat_data())

        heartbeat_thread = None
        if heartbeat_cb:
            heartbeat_thread = threading.Thread(
                target=heartbeat_loop,
                name=f"{self.config.name}-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()

        try:
            items = select_items(list(self.dataset), limit)
            total = len(items)
            if total == 0:
                raise RuntimeError(
                    f"El dataset para {self.config.name} no devolvió muestras"
                )

            with ThreadPoolExecutor(
                max_workers=self.num_concurrent
            ) as executor:
                futures = {
                    executor.submit(self.evaluate_item, item, is_seed): i
                    for i, item in enumerate(items)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        ok, details = future.result()
                        with state_lock:
                            correct += int(ok)
                            details_list.append(details)
                            completed += 1
                    except Exception as e:
                        with state_lock:
                            details_list.append({"error": str(e), "index": idx})
                            completed += 1
        finally:
            stop_heartbeat.set()
            if heartbeat_thread:
                heartbeat_thread.join()

        if heartbeat_cb:
            final_data = heartbeat_data(force=True)
            final_data["total"] = total
            heartbeat_cb(final_data)

        errors = [detail for detail in details_list if "error" in detail]
        if errors:
            first_error = errors[0]["error"]
            raise RuntimeError(
                f"{self.config.name}: {len(errors)}/{total} peticiones fallaron. "
                f"Primer error: {first_error}"
            )

        final_accuracy = correct / total
        return {
            "accuracy": final_accuracy,
            "correct": correct,
            "total": total,
            "details": details_list,
        }


# ─── Tareas de generación (GSM8K, AIME) ───


class GenerationEvaluator(TaskEvaluator):
    """Evalúa tareas de generación vía /v1/chat/completions."""

    def _extract_answer(self, text: str) -> str | None:
        """Extrae la respuesta final de la salida del modelo. Sobrescribir por tarea."""
        return text.strip()

    def _extract_gold(self, gold: str) -> str:
        """Extrae la respuesta comparable del gold del dataset. Sobrescribir por tarea."""
        return gold

    def _normalize_answer(self, ans: str) -> str:
        """Normaliza para comparación. Sobrescribir por tarea."""
        return ans.strip().lower()

    def score_output(self, output: str, gold: Any) -> tuple[bool, str | None]:
        """Puntúa la salida del modelo contra el gold del dataset. Devuelve (correcto, pred)."""
        pred = self._extract_answer(output)
        pred_norm = self._normalize_answer(pred or "")
        gold_norm = self._normalize_answer(self._extract_gold(str(gold)))
        return pred_norm == gold_norm, pred

    def evaluate_item(self, item: dict, is_seed: bool) -> tuple[bool, dict]:
        question = item[self.config.question_key]
        gold = item[self.config.answer_key]

        # Construir petición de chat completion
        messages = [{"role": "user", "content": question}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.reasoning_effort:
            payload["chat_template_kwargs"] = {
                "reasoning_effort": self.reasoning_effort
            }

        try:
            resp = self._post("/v1/chat/completions", payload)
            output = resp["choices"][0]["message"]["content"]

            ok, pred = self.score_output(output, gold)
            return ok, {
                "question": question[:100],
                "gold": gold,
                "pred": pred,
                "output_tail": output[-200:],
                "correct": ok,
            }
        except Exception as e:
            return False, {"error": str(e), "question": question[:100]}


class GSM8KEvaluator(GenerationEvaluator):
    def evaluate_item(self, item: dict, is_seed: bool) -> tuple[bool, dict]:
        enriched = dict(item)
        enriched[self.config.question_key] = (
            f"{item[self.config.question_key]}\n\n"
            "Solve the problem step by step. End with `#### <answer>`."
        )
        return super().evaluate_item(enriched, is_seed)

    def _extract_answer(self, text: str) -> str | None:
        # GSM8K: respuesta después de "#### " o último número
        if "####" in text:
            return text.split("####")[-1].strip()
        # Fallback: último número en el texto
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
        return nums[-1] if nums else None

    def _extract_gold(self, gold: str) -> str:
        # El gold de GSM8K es la solución completa terminando en "#### <respuesta>".
        return self._extract_answer(gold) or gold

    def _normalize_answer(self, ans: str) -> str:
        # Normalizar números: "18.0" -> "18", "7,000" -> "7000"
        ans = (ans or "").replace(",", "")
        if "." in ans:
            ans = ans.rstrip("0").rstrip(".")
        return ans.strip()


class AIMEEvaluator(GenerationEvaluator):
    def evaluate_item(self, item: dict, is_seed: bool) -> tuple[bool, dict]:
        enriched = dict(item)
        enriched[self.config.question_key] = (
            f"{item[self.config.question_key]}\n\n"
            "Return the final answer as an integer from 000 through 999."
        )
        return super().evaluate_item(enriched, is_seed)

    def _extract_answer(self, text: str) -> str | None:
        # AIME: respuesta entera 000-999
        nums = re.findall(r"\b\d{1,3}\b", text)
        return nums[-1] if nums else None

    def _normalize_answer(self, ans: str) -> str:
        return ans.strip().zfill(3)


# ─── Tareas de loglikelihood (HellaSwag, MMLU, ARC, Winogrande, TruthfulQA) ───


class LoglikelihoodEvaluator(TaskEvaluator):
    """Evalúa tareas de loglikelihood vía /v1/completions con echo+logprobs."""

    def _resolve_gold_idx(self, item: dict, choices: list[str]) -> int:
        """Mapea el campo gold del dataset a un índice de elección 0-based. Sobrescribir por tarea."""
        gold = item[self.config.answer_key]
        return (
            int(gold)
            if isinstance(gold, str) and gold.lstrip("-").isdigit()
            else gold
        )

    def _is_correct(
        self, pred_idx: int, item: dict, choices: list[str]
    ) -> bool:
        """Si el índice de elección predicho es correcto. Sobrescribir por tarea."""
        return pred_idx == self._resolve_gold_idx(item, choices)

    def _choice_logprobs(
        self, ctx: str, choice: str, logprobs: dict
    ) -> list[float]:
        """
        Devuelve los logprobs del prompt que pertenecen a `choice`.

        El endpoint de completion genera un token extra, que no debe incluirse
        en la puntuación de la continuación.
        """
        token_logprobs = logprobs.get("token_logprobs", [])
        offsets = logprobs.get("text_offset")
        prompt_end = len(ctx + choice)
        if offsets and len(offsets) == len(token_logprobs):
            selected = [
                value
                for value, offset in zip(token_logprobs, offsets, strict=False)
                if value is not None and len(ctx) <= offset < prompt_end
            ]
            if selected:
                return selected

        n = 0
        if self.tokenizer is not None:
            try:
                ids = self.tokenizer(choice, add_special_tokens=False)[
                    "input_ids"
                ]
                n = len(ids)
            except Exception:
                pass
        if n == 0:
            n = max(1, len(choice.split()))

        # El valor final pertenece al token generado.
        prompt_values = [
            value for value in token_logprobs[:-1] if value is not None
        ]
        return prompt_values[-n:]

    def _score_choice(self, ctx: str, choice: str) -> float:
        payload = {
            "model": self.model,
            "prompt": ctx + choice,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
        resp = self._post("/v1/completions", payload)
        lp = resp["choices"][0].get("logprobs")
        if not lp:
            raise RuntimeError(
                "La respuesta de completions no incluyó logprobs. Reinicia el "
                "servidor con --enable-echo y un runtime que soporte logprobs."
            )
        choice_logprobs = self._choice_logprobs(ctx, choice, lp)
        if not choice_logprobs:
            raise RuntimeError("No se pudieron identificar los logprobs de los tokens de elección")
        score = sum(choice_logprobs)
        if self.config.metric == "acc_norm":
            score /= len(choice_logprobs)
        return score

    def evaluate_item(self, item: dict, is_seed: bool) -> tuple[bool, dict]:
        ctx = item[self.config.question_key]

        choices = self._get_choices(item)
        if not choices:
            return False, {"error": "no se encontraron elecciones"}

        gold_idx = self._resolve_gold_idx(item, choices)
        scores = [self._score_choice(ctx, choice) for choice in choices]
        pred_idx = scores.index(max(scores)) if scores else 0
        ok = self._is_correct(pred_idx, item, choices)
        return ok, {
            "context": ctx[:100],
            "choices": choices,
            "scores": scores,
            "gold_idx": gold_idx,
            "pred_idx": pred_idx,
            "correct": ok,
        }

    @abstractmethod
    def _get_choices(self, item: dict) -> list[str]:
        """Extrae las cadenas de elección del elemento del dataset."""


class HellaSwagEvaluator(LoglikelihoodEvaluator):
    def _get_choices(self, item: dict) -> list[str]:
        # HellaSwag: los finales están en la lista "endings"
        return [f" {e}" for e in item.get("endings", [])]


class MMLUEvaluator(LoglikelihoodEvaluator):
    def _get_choices(self, item: dict) -> list[str]:
        # MMLU: elecciones en la lista "choices"
        return [f" {c}" for c in item.get("choices", [])]


class ARCEasyEvaluator(LoglikelihoodEvaluator):
    def _get_choices(self, item: dict) -> list[str]:
        choices = item.get("choices", {}).get("text", [])
        return [f" {c}" for c in choices]

    def _resolve_gold_idx(self, item: dict, choices: list[str]) -> int:
        # La answerKey de ARC es una etiqueta (letra "A".. o dígito "1"..); mapearla a su índice.
        labels = item.get("choices", {}).get("label", [])
        key = item.get("answerKey")
        if key in labels:
            return labels.index(key)
        return int(key) if isinstance(key, str) and key.isdigit() else key


class ARCChallengeEvaluator(ARCEasyEvaluator):
    pass


class WinograndeEvaluator(LoglikelihoodEvaluator):
    def _get_choices(self, item: dict) -> list[str]:
        # Winogrande: option1, option2
        return [f" {item.get('option1', '')}", f" {item.get('option2', '')}"]

    def _resolve_gold_idx(self, item: dict, choices: list[str]) -> int:
        # La "answer" de Winogrande es 1-indexada ("1"/"2"); convertir a 0-based.
        return int(item["answer"]) - 1


class TruthfulQAEvaluator(LoglikelihoodEvaluator):
    def _get_choices(self, item: dict) -> list[str]:
        # TruthfulQA: respuestas correctas primero, luego incorrectas.
        correct = item.get("correct_answers", [])
        incorrect = item.get("incorrect_answers", [])
        return [f" {c}" for c in (correct + incorrect)]

    def _resolve_gold_idx(self, item: dict, choices: list[str]) -> int:
        return 0  # primera respuesta correcta, solo para reporte

    def _is_correct(
        self, pred_idx: int, item: dict, choices: list[str]
    ) -> bool:
        # Correcto cuando el modelo prefiere cualquier respuesta del bloque correcto.
        n_correct = len(item.get("correct_answers", []))
        return pred_idx < n_correct


# ─── Fábrica ───

EVALUATOR_MAP = {
    "gsm8k": GSM8KEvaluator,
    "hellaswag": HellaSwagEvaluator,
    "mmlu": MMLUEvaluator,
    "aime": AIMEEvaluator,
    "arc_easy": ARCEasyEvaluator,
    "arc_challenge": ARCChallengeEvaluator,
    "winogrande": WinograndeEvaluator,
    "truthfulqa": TruthfulQAEvaluator,
}


def get_evaluator(config: TaskConfig, **kwargs) -> TaskEvaluator:
    """Instancia el evaluador apropiado para una tarea."""
    evaluator_cls = EVALUATOR_MAP.get(config.name)
    if not evaluator_cls:
        raise ValueError(f"No hay evaluador para la tarea: {config.name}")
    return evaluator_cls(config, **kwargs)


def get_task_config(task: str) -> TaskConfig:
    if task not in TASK_CONFIGS:
        raise ValueError(
            f"Tarea desconocida: {task}. Disponibles: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task]


def list_tasks() -> list[str]:
    return list(TASK_CONFIGS.keys())
