"""Runs the standard generation <-> summarization loop on BigCodeBench-Complete.

Each cycle follows the standard-loop protocol of the course baseline:

1. generate code for every active task (standard generation prompt);
2. test all of it in one batch with BigCodeBench's own test suites, on the Gradio evaluator;
3. summarize only the code that passed (standard summarization prompt); the summary, anchored
   to the task's ``code_prompt``, becomes the task's next prompt. Code that passes the final
   cycle is not summarized, since no further cycle uses the summary.

This is a standalone class rather than a subclass of :class:`EvaLoopEvaluator`, because
``evaluator.py`` imports EvalPlus at import time; the results file has the same structure, so
``evaloop analyze`` reads it unchanged.

Progress is checkpointed per phase (generated / tested / summarized) after every step, so a run
interrupted by an unreachable evaluator or a failing model API resumes where it stopped without
regenerating code it already has. The evaluator endpoint is never written to results, checkpoints
or logs.
"""

import datetime
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import evaloop.tasks.bigcodebench_summarization  # noqa: F401 - registers bigcodebench_summarization
from evaloop.datasets import bigcodebench as bcb
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.evaluation.gradio_tester import (
    ENDPOINT_ENV_VAR,
    EvaluatorUnavailableError,
    GradioBigCodeBenchTester,
    Sample,
    resolve_endpoint,
)
from evaloop.tasks.factory import TaskFactory

from .bigcodebench_config import STANDARD_PROTOCOL, BigCodeBenchConfig, api_system_prompt, check_protocol

CHECKPOINT_VERSION = 1
EXPERIMENT_TYPE = "bigcodebench"

# OpenAILLM reports API failures as responses starting with these strings instead of raising.
_GENERATION_ERROR_PREFIXES = ("Error generating response", "Error: Still received empty response")
# Model types whose ``max_tokens`` is the generation budget (EvaluationConfig sets it to the
# context length, which would exceed the protocol's max_new_tokens).
_API_MODEL_TYPES = ("openai", "anthropic")


class RunInterrupted(RuntimeError):
    """The run stopped with its progress saved; resume it from ``checkpoint_path``."""

    def __init__(self, message: str, checkpoint_path: Path):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path


class BigCodeBenchEvaluator:
    """Runs one model through the standard loop on BigCodeBench and writes its results file."""

    def __init__(
        self,
        config: Any,
        bcb_config: BigCodeBenchConfig,
        *,
        model: Any = None,
        tester: Any = None,
        loader: Optional[Callable[..., Mapping[str, Mapping[str, Any]]]] = None,
        hasher: Optional[Callable[..., str]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            config: The run's :class:`~evaloop.core.config.EvaluationConfig`.
            bcb_config: BigCodeBench settings.
            model: Ready-made model (tests); created from ``config`` otherwise.
            tester: Ready-made tester (tests); a :class:`GradioBigCodeBenchTester` otherwise.
            loader: Replaces :func:`~evaloop.datasets.bigcodebench.load_from_hub` (tests).
            hasher: Replaces the dataset hash computation (tests).
            clock: Clock used for the usage timings.
        """
        self.config = config
        self.bcb = bcb_config
        self._model = model
        self._tester = tester
        self._loader = loader
        self._hasher = hasher
        self._clock = clock
        self.logger = logging.getLogger(__name__)
        self.problems: Dict[str, bcb.BigCodeBenchProblem] = {}
        self.state: Dict[str, Any] = {}
        # Computed once: for Hugging Face names, EvaluationConfig may look up the model's config.
        self._model_config = config.get_model_config()
        self.provider = self._model_config["type"]
        self.system_prompt = self._system_prompt_to_send()
        self.checkpoint_path = Path(config.output_dir) / f"{config.experiment_name}.checkpoint.json"

    @property
    def results_path(self) -> Path:
        """Where the results file is written (next to the checkpoint)."""
        name = self.state.get("experiment_name", self.config.experiment_name)
        return self.checkpoint_path.parent / f"{name}_{EXPERIMENT_TYPE}_results.json"

    def run(self) -> List[Dict[str, Any]]:
        """Runs (or resumes) the experiment and saves the results.

        Returns:
            The results, a list with this model's result (the same shape ``EvaLoopEvaluator``
            returns).

        Raises:
            RunInterrupted: If the evaluator or the model API failed; progress is checkpointed.
            ValueError: If the settings break the protocol or a resume does not match its run.
        """
        protocol_settings = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens,
            "max_cycles": self.config.max_cycles,
        }
        checked_settings = dict(protocol_settings)
        if self.provider in _API_MODEL_TYPES:
            checked_settings["system_prompt"] = self.system_prompt
        check_protocol(checked_settings)

        if self.bcb.resume_from:
            self._resume(protocol_settings)
        else:
            self._start(protocol_settings)

        model = self._get_model()
        self.state["model_result"]["model"] = str(model)
        if not self.bcb.resume_from:
            # Records the system prompt the model object actually sends.
            self.state["model_result"]["protocol"]["system_prompt"] = getattr(model, "system_prompt", self.system_prompt)
        tasks = self._create_tasks()
        self._run_cycles(model, tasks)
        result = self._finalize()
        self.logger.info("BigCodeBench results saved to %s", self.results_path)
        return [result]

    # ------------------------------------------------------------------ setup

    def _start(self, protocol_settings: Dict[str, Any]) -> None:
        """Loads the tasks, checks the ground truth, and initializes a fresh run."""
        self.problems = bcb.load_problems(loader=self._loader)
        selected = bcb.sorted_task_ids(self.problems)

        # The pre-flight runs before the model is created: a dead endpoint fails in seconds.
        preflight = self._get_tester().preflight(selected)
        kept, excluded = bcb.apply_gt_exclusions(selected, preflight.failed_tasks)
        if excluded:
            self.logger.warning(
                "Excluding %d tasks whose canonical solution fails in the evaluator: %s", len(excluded), excluded
            )
        elif preflight.failed_tasks:
            self.logger.warning("Keeping %d tasks whose canonical solution fails", len(preflight.failed_tasks))
        if self.bcb.trial:
            kept = bcb.select_trial_tasks(kept, seed=self.bcb.trial_seed)
        bcb.check_min_tasks(kept, trial=self.bcb.trial)

        selection = bcb.TaskSelection(
            task_ids=tuple(kept),
            excluded_gt_failures=tuple(excluded),
            trial=self.bcb.trial,
            trial_seed=self.bcb.trial_seed if self.bcb.trial else None,
        )
        dataset = {
            "name": "bigcodebench",
            "split": "complete",
            **selection.to_metadata(),
            "gt_pass_rate": preflight.gt_pass_rate,
            "gt_failed_tasks": list(preflight.failed_tasks),
            "source": f"huggingface:{bcb.HF_DATASET}",
            "dataset_version": bcb.DATASET_VERSION,
            "dataset_hash": bcb.dataset_hash(hasher=self._hasher, loader=self._loader),
            "calibrated": True,
            "evaluator": {"backend": "gradio", "host": "colab"},
        }
        protocol = {
            **protocol_settings,
            "prompts": "assignment2-data standard loop",
            "anchor": "code_prompt",
            "system_prompt": self.system_prompt,
        }
        now = _now()
        self.state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "experiment_name": self.config.experiment_name,
            "complete": False,
            "active": [{"task_id": task_id, "prompt": self.problems[task_id].complete_prompt} for task_id in kept],
            "next_cycle": 1,
            "cycle_state": None,
            "model_result": {
                "model": None,
                "model_identifier": self.config.model,
                "provider": self.provider,
                "max_cycles": self.config.max_cycles,
                "run_id": uuid.uuid4().hex,
                "trial": self.bcb.trial,
                "dataset": dataset,
                "protocol": protocol,
                "usage": {"started_at": now, "finished_at": None, "per_cycle": [], "totals": {}},
                "prompt_results": [
                    {
                        "task_id": task_id,
                        "initial_prompt": self.problems[task_id].complete_prompt,
                        "cycles": [],
                        "successful_cycles": 0,
                        "max_cycles_reached": False,
                    }
                    for task_id in kept
                ],
            },
        }
        self.logger.info(
            "BigCodeBench %s: %d tasks (%d excluded for failing ground truth)",
            selection.configuration, len(kept), len(excluded),
        )
        self._save_checkpoint()

    def _resume(self, protocol_settings: Dict[str, Any]) -> None:
        """Restores a checkpoint after checking it belongs to the same run."""
        path = Path(self.bcb.resume_from)
        with open(path, "r", encoding="utf-8") as checkpoint:
            state = json.load(checkpoint)
        if state.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"{path} is not a BigCodeBench checkpoint of version {CHECKPOINT_VERSION}")
        if state.get("complete"):
            raise ValueError(f"The run in {path} is already complete")

        result = state["model_result"]
        dataset = result["dataset"]
        mismatches = []
        if result["model_identifier"] != self.config.model:
            mismatches.append(f"model ({result['model_identifier']!r} vs {self.config.model!r})")
        for name in STANDARD_PROTOCOL:
            if result["protocol"][name] != protocol_settings[name]:
                mismatches.append(f"{name} ({result['protocol'][name]!r} vs {protocol_settings[name]!r})")
        if result["protocol"].get("system_prompt") != self.system_prompt:
            mismatches.append(f"system prompt ({result['protocol'].get('system_prompt')!r} vs {self.system_prompt!r})")
        if dataset.get("dataset_version") != bcb.DATASET_VERSION:
            mismatches.append(f"dataset version ({dataset.get('dataset_version')!r} vs {bcb.DATASET_VERSION!r})")
        if dataset["dataset_hash"] != bcb.dataset_hash(hasher=self._hasher, loader=self._loader):
            mismatches.append("dataset hash")
        if mismatches:
            raise ValueError(f"Cannot resume {path}: it was run with different settings: " + "; ".join(mismatches))

        problems = bcb.load_problems(loader=self._loader)
        missing = [task_id for task_id in dataset["task_ids"] if task_id not in problems]
        if missing:
            raise ValueError(f"Cannot resume {path}: tasks no longer in the dataset: {missing[:10]}")
        self.problems = {task_id: problems[task_id] for task_id in dataset["task_ids"]}
        self.state = state
        self.checkpoint_path = path
        if self.bcb.trial != result["trial"]:
            self.logger.info("Resuming with the checkpoint's task set (%s)", dataset["configuration"])

        # Health check of the (new) endpoint before any further work.
        self._get_tester().preflight(dataset["task_ids"][:1])
        self.logger.info("Resuming %s at cycle %d", state["experiment_name"], state["next_cycle"])

    def _get_tester(self) -> Any:
        """Returns the injected tester or a Gradio tester."""
        if self._tester is None:
            self._tester = GradioBigCodeBenchTester(
                resolve_endpoint(self.bcb.gradio_endpoint),
                limits=self.bcb.limits(),
                task_libs={task_id: problem.libs for task_id, problem in self.problems.items()},
            )
        return self._tester

    def _system_prompt_to_send(self) -> Optional[str]:
        """The system prompt the model is sent: the standard one for API models (unless
        ``--system_prompt`` overrides it), and the configured one for local models."""
        if self.provider in _API_MODEL_TYPES:
            return api_system_prompt(self.config.system_prompt)
        return self._model_config.get("system_prompt")

    def _get_model(self) -> Any:
        """Returns the injected model or creates it from the run configuration."""
        if self._model is None:
            model_config = model_config_for_protocol(
                self._model_config, self.config.max_new_tokens, self.system_prompt
            )
            # The API clients are imported directly: ModelFactory also imports the vLLM and
            # transformers backends, which fail on machines without them.
            if model_config["type"] == "openai":
                from evaloop.models.closed_source import OpenAILLM

                self._model = OpenAILLM(model_config)
            elif model_config["type"] == "anthropic":
                from evaloop.models.anthropic_llm import AnthropicLLM

                self._model = AnthropicLLM(model_config)
            else:
                from evaloop.models.factory import ModelFactory

                self._model = ModelFactory().create_model(model_config)
        return self._model

    def _create_tasks(self) -> Dict[str, Any]:
        """Creates the generation and summarization tasks of the standard loop."""
        generation_config, summarization_config = bcb.task_configs(self.problems)
        factory = TaskFactory()
        return {
            "generation": factory.create_task(generation_config),
            "summarization": factory.create_task(summarization_config),
        }

    # ------------------------------------------------------------------ loop

    def _run_cycles(self, model: Any, tasks: Dict[str, Any]) -> None:
        """Runs cycles until every task failed or ``max_cycles`` is reached."""
        generator = BatchEvaluationCycle(model, [tasks["generation"]])
        summarizer = BatchEvaluationCycle(model, [tasks["summarization"]])
        max_cycles = self.state["model_result"]["max_cycles"]
        while self.state["active"] and self.state["next_cycle"] <= max_cycles:
            self._run_cycle(generator, summarizer, self.state["next_cycle"], max_cycles)

    def _run_cycle(self, generator: Any, summarizer: Any, cycle: int, max_cycles: int) -> None:
        """Runs (or continues) one cycle: generate, test, summarize, then record."""
        if self.state["cycle_state"] is None:
            self.state["cycle_state"] = {
                "cycle": cycle,
                "generated": {},
                "tested": {},
                "summarized": {},
                "usage": {
                    "generation_calls": 0, "summarization_calls": 0,
                    "generation_seconds": 0.0, "summarization_seconds": 0.0, "evaluation_seconds": 0.0,
                },
            }
        state = self.state["cycle_state"]
        usage = state["usage"]
        active = {entry["task_id"]: entry["prompt"] for entry in self.state["active"]}
        model = generator.model

        # 1. Generate code (only for tasks without a usable generation from an earlier attempt).
        pending = [task_id for task_id in active if not _usable(state["generated"].get(task_id))]
        if pending:
            started, truncated = self._clock(), _truncations(model)
            output = generator.run_single_cycle(
                [{"task_id": task_id, "prompt": active[task_id]} for task_id in pending], cycle
            )
            _count_truncations(usage, model, truncated)
            usage["generation_seconds"] += self._clock() - started
            usage["generation_calls"] += len(pending)
            for record in output["cycle_results"]:
                state["generated"][record["task_id"]] = {
                    "output": record.get("task_1_output"),
                    "raw_output": record.get("task_1_raw_output"),
                    "error": _generation_error(record),
                }
            self._handle_model_errors(state["generated"], pending, cycle, "code generation")

        # 2. Test every generated solution that has not been tested yet, in one batch.
        untested = [
            task_id for task_id in active
            if task_id not in state["tested"] and _usable(state["generated"].get(task_id))
        ]
        if untested:
            started = self._clock()
            try:
                outcomes = self._tester.test_batch(
                    [Sample(task_id, state["generated"][task_id]["output"]) for task_id in untested]
                )
            except EvaluatorUnavailableError as err:
                self._interrupt(cycle, f"the evaluator is unreachable: {err}")
            usage["evaluation_seconds"] += self._clock() - started
            for task_id in untested:
                state["tested"][task_id] = outcomes[task_id].to_dict()
        self._save_checkpoint()

        # 3. Summarize the passing code, except in the final cycle (no further cycle uses it).
        passing = [task_id for task_id in active if state["tested"][task_id]["success"]]
        to_summarize = [task_id for task_id in passing if not _usable(state["summarized"].get(task_id))]
        if to_summarize and cycle < max_cycles:
            started, truncated = self._clock(), _truncations(model)
            output = summarizer.run_single_cycle(
                [{"task_id": task_id, "prompt": state["generated"][task_id]["output"]} for task_id in to_summarize],
                cycle,
            )
            _count_truncations(usage, model, truncated)
            usage["summarization_seconds"] += self._clock() - started
            usage["summarization_calls"] += len(to_summarize)
            for record in output["cycle_results"]:
                state["summarized"][record["task_id"]] = {
                    "output": record.get("task_1_output"),
                    "raw_output": record.get("task_1_raw_output"),
                    "error": _generation_error(record),
                }
            self._handle_model_errors(state["summarized"], to_summarize, cycle, "summarization")

        # 4. Record the cycle and carry the anchored summaries into the next one.
        self._record_cycle(active, cycle, max_cycles)

    def _handle_model_errors(self, records: Dict[str, Any], task_ids: Sequence[str], cycle: int, phase: str) -> None:
        """Checkpoints, and stops the run on model API failures so a resume retries them."""
        self._save_checkpoint()
        failed = [task_id for task_id in task_ids if records[task_id]["error"]]
        if failed:
            self._interrupt(
                cycle,
                f"{phase} failed for {len(failed)} tasks ({failed[:5]}), e.g. {records[failed[0]]['error'][:200]!r}",
            )

    def _record_cycle(self, active: Dict[str, str], cycle: int, max_cycles: int) -> None:
        """Appends the cycle's records to the results and prepares the next cycle."""
        state = self.state["cycle_state"]
        prompt_results = {entry["task_id"]: entry for entry in self.state["model_result"]["prompt_results"]}
        next_active = []
        for task_id in active:
            generated = state["generated"][task_id]
            tested = state["tested"][task_id]
            summarized = state["summarized"].get(task_id)
            record = {
                "cycle": cycle,
                "task_id": task_id,
                "task_1_output": generated["output"],
                "task_1_raw_output": generated["raw_output"],
                "test_results": tested,
                "success": tested["success"],
            }
            if summarized:
                record["task_2_output"] = summarized["output"]
                record["task_2_raw_output"] = summarized["raw_output"]
            prompt_result = prompt_results[task_id]
            prompt_result["cycles"].append(record)
            if tested["success"]:
                prompt_result["successful_cycles"] = cycle
                if cycle < max_cycles:
                    next_active.append({"task_id": task_id, "prompt": summarized["output"]})

        passed = sum(state["tested"][task_id]["success"] for task_id in active)
        self.logger.info("Cycle %d: %d/%d tasks passed", cycle, passed, len(active))
        self.state["model_result"]["usage"]["per_cycle"].append({"cycle": cycle, **state["usage"]})
        self.state.update(active=next_active, next_cycle=cycle + 1, cycle_state=None)
        self._save_checkpoint()

    def _finalize(self) -> Dict[str, Any]:
        """Completes the results, writes the results file, and marks the checkpoint complete."""
        result = self.state["model_result"]
        max_cycles = result["max_cycles"]
        for prompt_result in result["prompt_results"]:
            prompt_result["max_cycles_reached"] = prompt_result["successful_cycles"] == max_cycles
        cycles = [prompt_result["successful_cycles"] for prompt_result in result["prompt_results"]]
        result["average_successful_cycles"] = sum(cycles) / len(cycles)
        usage = result["usage"]
        usage["finished_at"] = _now()
        totals: Dict[str, float] = {}
        for entry in usage["per_cycle"]:
            for key, value in entry.items():
                if key != "cycle":
                    totals[key] = totals.get(key, 0) + value
        usage["totals"] = totals

        _write_json(self.results_path, [result])
        self.state["complete"] = True
        self._save_checkpoint()
        return result

    def _save_checkpoint(self) -> None:
        """Writes the run state atomically."""
        _write_json(self.checkpoint_path, self.state)

    def _interrupt(self, cycle: int, reason: str) -> None:
        """Saves progress and stops the run with resume instructions."""
        self._save_checkpoint()
        raise RunInterrupted(
            f"Run interrupted in cycle {cycle}: {reason}. Progress is saved in {self.checkpoint_path}. "
            f"Resume with: evaloop evaluate --dataset bigcodebench --model {self.config.model!r} "
            f"--resume_from {self.checkpoint_path} (with {ENDPOINT_ENV_VAR} set to the evaluator's URL)",
            self.checkpoint_path,
        ) from None


def model_config_for_protocol(
    model_config: Dict[str, Any], max_new_tokens: int, system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Applies the protocol's generation budget and system prompt to API models.

    ``EvaluationConfig`` gives API models ``max_tokens = max_model_len`` (the context length),
    while the standard loop allows at most ``max_new_tokens`` generated tokens. It also passes
    API models no system prompt, so ``system_prompt`` (when given) is set explicitly.
    """
    adjusted = dict(model_config)
    if adjusted.get("type") in _API_MODEL_TYPES:
        adjusted["max_tokens"] = max_new_tokens
        if system_prompt is not None:
            adjusted["system_prompt"] = system_prompt
    return adjusted


def _truncations(model: Any) -> Optional[int]:
    """The model's count of responses cut off at the token budget, if it keeps one."""
    count = getattr(model, "truncated_responses", None)
    return count if isinstance(count, int) else None


def _count_truncations(usage: Dict[str, Any], model: Any, before: Optional[int]) -> None:
    """Adds the truncations since ``before`` to the cycle's usage (models that count them)."""
    after = _truncations(model)
    if before is not None and after is not None:
        usage["truncated_responses"] = usage.get("truncated_responses", 0) + after - before


def _generation_error(record: Mapping[str, Any]) -> Optional[str]:
    """Returns why a cycle runner record has no usable model output, or ``None``."""
    if record.get("task_1_error"):
        return str(record["task_1_error"])
    raw = record.get("task_1_raw_output")
    if not isinstance(raw, str):
        return "no model output"
    if raw.strip().startswith(_GENERATION_ERROR_PREFIXES):
        return raw.strip()
    return None


def _usable(record: Optional[Mapping[str, Any]]) -> bool:
    """Whether a generated or summarized record holds usable model output."""
    return bool(record) and not record["error"]


def _now() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    """Writes JSON atomically (temporary file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as output:
        json.dump(data, output, indent=2)
    os.replace(tmp_path, path)
