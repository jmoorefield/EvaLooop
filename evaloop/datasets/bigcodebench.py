"""BigCodeBench-Complete adapter for the BigCodeBench-Hard task set (148 tasks).

Tasks are read directly from the BigCodeBench-Hard dataset on the Hugging Face Hub, at the version
the ``bigcodebench`` package's own loader pins (:data:`DATASET_VERSION`), so they match what the
BigCodeBench evaluator loads without installing ``bigcodebench`` locally. ``datasets`` is imported
lazily: everything else in this module works without it, and tests inject a fake loader.

A trial run keeps the protocol but is limited to :data:`TRIAL_SIZE` tasks, drawn reproducibly from
the task set after ground-truth exclusions. Every non-trial run needs at least :data:`MIN_TASKS`
tasks.
"""

import ast
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .standard_prompts import GENERATION_TEMPLATE, SUMMARIZATION_TEMPLATE

ENTRY_POINT = "task_func"
TASK_ID_PREFIX = "BigCodeBench/"
# The evaluator's name for the task set, sent with every request.
SUBSET = "hard"
MIN_TASKS = 100
TRIAL_SIZE = 10
CONFIGURATION = "bigcodebench-hard-complete"

# The Hugging Face dataset, and the dataset version (one of its splits) that bigcodebench 0.2.5's
# loader pins as BIGCODEBENCH_VERSION. The Colab evaluator loads the same.
HF_DATASET = "bigcode/bigcodebench-hard"
DATASET_VERSION = "v0.1.4"

# A loader takes no arguments and returns the raw tasks keyed by task id, like
# :func:`load_from_hub`.
Loader = Callable[[], Mapping[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class BigCodeBenchProblem:
    """The fields of one BigCodeBench task that EvaLoop needs.

    Attributes:
        task_id: Task identifier, e.g. ``"BigCodeBench/13"``.
        complete_prompt: Cycle-1 prompt of the Complete split (imports, signature, docstring).
        code_prompt: Imports and ``def task_func(...):`` signature; the per-cycle anchor.
        libs: Libraries the task uses.
    """

    task_id: str
    complete_prompt: str
    code_prompt: str
    libs: Tuple[str, ...]


@dataclass(frozen=True)
class TaskSelection:
    """A run's task ids.

    Attributes:
        task_ids: Final task ids of the run, sorted by task number.
        excluded_gt_failures: Tasks dropped because their canonical solution fails in the
            evaluator's environment.
        trial: Whether this is a :data:`TRIAL_SIZE`-task trial run.
        trial_seed: Seed used to pick the trial tasks.
    """

    task_ids: Tuple[str, ...]
    excluded_gt_failures: Tuple[str, ...] = ()
    trial: bool = False
    trial_seed: Optional[int] = None

    @property
    def configuration(self) -> str:
        """Configuration name used in results and ``summary.csv``."""
        return configuration_name(self.trial)

    def to_metadata(self) -> Dict[str, Any]:
        """Returns the JSON-serializable selection record for the results file."""
        return {
            "configuration": self.configuration,
            "subset": SUBSET,
            "task_ids": list(self.task_ids),
            "excluded_gt_failures": list(self.excluded_gt_failures),
            "trial": self.trial,
            "trial_seed": self.trial_seed,
        }


def bare_task_id(task_id: str) -> str:
    """Strips the ``BigCodeBench/`` prefix (the evaluator's ``selective_evaluate`` wants bare ids)."""
    return task_id[len(TASK_ID_PREFIX):] if task_id.startswith(TASK_ID_PREFIX) else task_id


def full_task_id(task_id: str) -> str:
    """Adds the ``BigCodeBench/`` prefix if it is missing."""
    return task_id if task_id.startswith(TASK_ID_PREFIX) else TASK_ID_PREFIX + task_id


def task_sort_key(task_id: str) -> Tuple[int, Any]:
    """Sorts task ids by task number, falling back to the raw string."""
    bare = bare_task_id(task_id)
    return (0, int(bare)) if bare.isdigit() else (1, bare)


def load_problems(loader: Optional[Loader] = None) -> Dict[str, BigCodeBenchProblem]:
    """Loads and validates the BigCodeBench-Hard tasks.

    Args:
        loader: Replaces :func:`load_from_hub` (used by tests).

    Returns:
        Problems keyed by task id.

    Raises:
        ImportError: If no loader is given and ``datasets`` is not installed.
        ValueError: If any task violates the adapter's assumptions.
    """
    raw_tasks = (loader or load_from_hub)()

    problems: Dict[str, BigCodeBenchProblem] = {}
    errors: List[str] = []
    for key, task in raw_tasks.items():
        task_id = str(task.get("task_id", key))
        problem_errors = _validate_task(task_id, task)
        if problem_errors:
            errors.extend(f"{task_id}: {error}" for error in problem_errors)
            continue
        problems[task_id] = BigCodeBenchProblem(
            task_id=task_id,
            complete_prompt=task["complete_prompt"],
            code_prompt=task["code_prompt"],
            libs=_parse_libs(task.get("libs")),
        )
    if errors:
        shown = "\n  ".join(errors[:10])
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValueError(
            f"{len(errors)} BigCodeBench task problems violate the adapter's assumptions:\n  {shown}{more}"
        )
    if not problems:
        raise ValueError("The BigCodeBench-Hard dataset contains no tasks")
    return problems


def load_from_hub() -> Dict[str, Dict[str, Any]]:
    """Loads the raw BigCodeBench-Hard tasks from the Hugging Face Hub at :data:`DATASET_VERSION`.

    The first call downloads the dataset into the ``datasets`` cache; later calls read the cache.
    Only task text is read; nothing in it is executed.
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError(_MISSING_PACKAGE_MESSAGE) from err
    rows = load_dataset(HF_DATASET, split=DATASET_VERSION)
    return {row["task_id"]: dict(row) for row in rows}


def dataset_hash(hasher: Optional[Callable[[], str]] = None, loader: Optional[Loader] = None) -> str:
    """Returns an MD5 fingerprint of the raw tasks (every field, in task-id order).

    Args:
        hasher: Replaces the whole computation (used by tests).
        loader: Replaces :func:`load_from_hub`.
    """
    if hasher is not None:
        return hasher()
    tasks = (loader or load_from_hub)()
    canonical = json.dumps(
        [tasks[task_id] for task_id in sorted(tasks, key=task_sort_key)],
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def sorted_task_ids(task_ids: Iterable[str]) -> List[str]:
    """Returns the distinct task ids, sorted by task number."""
    return sorted(set(task_ids), key=task_sort_key)


def apply_gt_exclusions(
    task_ids: Sequence[str],
    failed_tasks: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Removes tasks whose canonical solution fails in the evaluator's environment.

    Args:
        task_ids: Selected task ids.
        failed_tasks: Task ids reported by the ground-truth pre-flight (with or without prefix).

    Returns:
        The kept task ids and the excluded ones, both in the input order.
    """
    failed = {full_task_id(task_id) for task_id in failed_tasks}
    excluded = [task_id for task_id in task_ids if task_id in failed]
    return [task_id for task_id in task_ids if task_id not in failed], excluded


def select_trial_tasks(task_ids: Sequence[str], seed: int = 0, size: int = TRIAL_SIZE) -> List[str]:
    """Picks the tasks of a trial run, reproducibly from ``seed``.

    Raises:
        ValueError: If fewer than ``size`` tasks are available.
    """
    ids = sorted(set(task_ids))
    if len(ids) < size:
        raise ValueError(f"A trial run needs {size} tasks, but only {len(ids)} are available")
    return sorted(random.Random(seed).sample(ids, size), key=task_sort_key)


def check_min_tasks(task_ids: Sequence[str], trial: bool = False, minimum: int = MIN_TASKS) -> None:
    """Refuses a non-trial run with fewer than ``minimum`` tasks.

    Raises:
        ValueError: If the task set is too small.
    """
    if not trial and len(task_ids) < minimum:
        raise ValueError(
            f"The run has {len(task_ids)} tasks after exclusions, fewer than the required {minimum}; "
            f"use --bcb_trial for a {TRIAL_SIZE}-task trial run"
        )


def configuration_name(trial: bool = False) -> str:
    """Returns the configuration name: ``bigcodebench-hard-complete``, with ``-trial10`` for a
    trial run."""
    return CONFIGURATION + (f"-trial{TRIAL_SIZE}" if trial else "")


def task_configs(problems: Mapping[str, BigCodeBenchProblem]) -> List[Dict[str, Any]]:
    """Builds the generation and summarization task configs of the standard loop.

    The summarization task gets the ``task_id -> code_prompt`` map it uses as the anchor.
    """
    return [
        {"type": "code_generation", "language": "python", "prompt_template": GENERATION_TEMPLATE},
        {
            "type": "bigcodebench_summarization",
            "prompt_template": SUMMARIZATION_TEMPLATE,
            "code_prompts": {task_id: problem.code_prompt for task_id, problem in problems.items()},
        },
    ]


_MISSING_PACKAGE_MESSAGE = (
    "Loading BigCodeBench needs the 'datasets' package (a core EvaLoop dependency): pip install -e ."
)


def _validate_task(task_id: str, task: Mapping[str, Any]) -> List[str]:
    """Returns the reasons a raw task cannot be used, empty if it is fine."""
    errors = []
    if not task_id.startswith(TASK_ID_PREFIX):
        errors.append(f"task id does not start with '{TASK_ID_PREFIX}'")
    if task.get("entry_point") != ENTRY_POINT:
        errors.append(f"entry_point is {task.get('entry_point')!r}, expected '{ENTRY_POINT}'")
    for field_name in ("complete_prompt", "code_prompt"):
        if not isinstance(task.get(field_name), str) or not task.get(field_name, "").strip():
            errors.append(f"{field_name} is missing or empty")
    if not errors:
        error = _code_prompt_error(task["code_prompt"])
        if error:
            errors.append(error)
    return errors


def _code_prompt_error(code_prompt: str) -> Optional[str]:
    """Checks that ``code_prompt`` ends with the (possibly multi-line) ``task_func`` signature.

    The anchor appends an indented docstring to ``code_prompt``, which only works if the last
    top-level statement is a ``def task_func(...):`` with no body yet.
    """
    stripped = code_prompt.rstrip()
    if not stripped.endswith(":"):
        return "code_prompt does not end with a function signature"
    try:
        module = ast.parse(stripped + "\n    pass\n")
    except SyntaxError as err:
        return f"code_prompt is not a valid signature stub ({err.msg})"
    last = module.body[-1] if module.body else None
    if not isinstance(last, (ast.FunctionDef, ast.AsyncFunctionDef)) or last.name != ENTRY_POINT:
        return f"code_prompt does not end with a '{ENTRY_POINT}' definition"
    if len(last.body) != 1 or not isinstance(last.body[0], ast.Pass):
        return f"code_prompt's '{ENTRY_POINT}' already has a body"
    return None


def _parse_libs(libs: Any) -> Tuple[str, ...]:
    """Normalizes the ``libs`` field, which the dataset stores as a list or its string form."""
    if libs is None:
        return ()
    if isinstance(libs, str):
        try:
            libs = ast.literal_eval(libs)
        except (ValueError, SyntaxError):
            return tuple(part.strip(" '\"") for part in libs.strip("[]").split(",") if part.strip(" '\""))
    if isinstance(libs, (list, tuple, set)):
        return tuple(str(lib) for lib in libs)
    return (str(libs),)
