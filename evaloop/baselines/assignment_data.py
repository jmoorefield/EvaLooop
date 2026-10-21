"""Read-only access to the course's MBPP Plus baseline in ``data/assignment2-data``.

The package holds the standard-loop records of 60 models on 378 MBPP Plus tasks (10 cycles at
most) and the leaderboard metrics of 107 models. It must never be copied: this module converts
records **in memory** into the EvaLoop results structure, so the analysis code treats baseline and
BigCodeBench runs alike, and writes nothing to disk. Refer to records by ``model_id``, ``task_id``
and ``cycle``.

Field mapping (package -> EvaLoop results):

* ``cycles[].code`` -> ``task_1_output``; ``cycles[].summary`` -> ``task_2_output``;
  ``cycles[].passed`` -> ``success`` (``base_passed`` / ``plus_passed`` -> ``test_results``).
* ``boundary_similarity`` is not used: the extension reports no judge-based metric.

Leaderboard metrics are averages over five runs for seven models, while ``loops/`` holds one run
(``exp_1``), so reported baseline numbers are always recomputed from ``loops/``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from evaloop.analysis import metrics

DEFAULT_BASELINE_DIR = Path("data") / "assignment2-data"
BASELINE_MAX_CYCLES = 10
BASELINE_CONFIGURATION = "standard-mbpp-plus"
# leaderboard.json rounds to 3 decimals.
LEADERBOARD_TOLERANCE = 0.0005 + 1e-9

PathLike = Union[str, Path]


class BaselineDataError(ValueError):
    """The data package is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class BaselineMetrics:
    """Judge-free metrics of one run.

    Attributes:
        n_tasks: Number of tasks.
        success_rate: pass@1 / n_tasks.
        asl_score: Unweighted ASL, ``sum_i i^2 * n_i / (n_tasks * max_cycles)`` (the package's
            ``aslScore``).
    """

    n_tasks: int
    success_rate: float
    asl_score: float


@dataclass(frozen=True)
class ReproductionCheck:
    """Recomputed metrics of a baseline model next to its leaderboard entry."""

    model_id: str
    recomputed: BaselineMetrics
    leaderboard_success_rate: Optional[float]
    leaderboard_asl_score: Optional[float]

    @property
    def matches(self) -> bool:
        """Whether both metrics agree with the leaderboard to its 3-decimal rounding."""
        if self.leaderboard_success_rate is None or self.leaderboard_asl_score is None:
            return False
        return (
            abs(self.recomputed.success_rate - self.leaderboard_success_rate) <= LEADERBOARD_TOLERANCE
            and abs(self.recomputed.asl_score - self.leaderboard_asl_score) <= LEADERBOARD_TOLERANCE
        )


def list_model_ids(baseline_dir: PathLike = DEFAULT_BASELINE_DIR) -> List[str]:
    """Returns the ids of the models with per-cycle records, sorted."""
    loops_dir = _require_dir(baseline_dir) / "loops"
    return sorted(path.stem for path in loops_dir.glob("*.json"))


def load_leaderboard(baseline_dir: PathLike = DEFAULT_BASELINE_DIR) -> Dict[str, Dict[str, Any]]:
    """Returns the leaderboard entries keyed by ``model_id``."""
    path = _require_dir(baseline_dir) / "leaderboard.json"
    data = _read_json(path)
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise BaselineDataError(f"{path} has no 'models' list")
    return {str(entry["model_id"]): entry for entry in models}


def load_loops(model_id: str, baseline_dir: PathLike = DEFAULT_BASELINE_DIR) -> Dict[str, Any]:
    """Loads and validates the per-cycle records of one model.

    Raises:
        BaselineDataError: If the model has no records or they are inconsistent.
    """
    path = _require_dir(baseline_dir) / "loops" / f"{model_id}.json"
    if not path.is_file():
        available = list_model_ids(baseline_dir)
        raise BaselineDataError(
            f"No baseline records for model_id '{model_id}' in {path.parent} "
            f"({len(available)} models available, e.g. {available[:5]})"
        )
    raw = _read_json(path)
    validate_loops(raw, source=str(path))
    return raw


def validate_loops(raw: Mapping[str, Any], source: str = "baseline records") -> None:
    """Checks that every task's ``successful_cycles`` agrees with its per-cycle outcomes.

    The standard loop stops at the first failing cycle, so a task that sustained ``s`` cycles has
    ``s`` passing cycles followed by one failing cycle (none if ``s`` is the maximum).

    Raises:
        BaselineDataError: Listing the inconsistent tasks.
    """
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise BaselineDataError(f"{source} has no 'tasks' list")
    problems = []
    for task in tasks:
        task_id = task.get("task_id")
        sustained = task.get("successful_cycles")
        cycles = task.get("cycles") or []
        if not isinstance(sustained, int) or not 0 <= sustained <= BASELINE_MAX_CYCLES:
            problems.append(f"{task_id}: successful_cycles={sustained!r}")
            continue
        outcomes = [bool(cycle.get("passed")) for cycle in cycles]
        expected = [True] * sustained + ([False] if sustained < BASELINE_MAX_CYCLES else [])
        numbers = [cycle.get("cycle") for cycle in cycles]
        if outcomes != expected or numbers != list(range(1, len(cycles) + 1)):
            problems.append(f"{task_id}: successful_cycles={sustained} but cycle outcomes {outcomes}")
        elif bool(task.get("max_cycles_reached")) != (sustained == BASELINE_MAX_CYCLES):
            problems.append(f"{task_id}: max_cycles_reached disagrees with successful_cycles={sustained}")
    if problems:
        shown = "\n  ".join(problems[:10])
        more = f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise BaselineDataError(f"{source} has {len(problems)} inconsistent tasks:\n  {shown}{more}")


def to_results(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Converts a model's package records into an EvaLoop model result (in memory only)."""
    prompt_results = []
    for task in raw["tasks"]:
        cycles = []
        for cycle in task.get("cycles") or []:
            record: Dict[str, Any] = {
                "cycle": cycle["cycle"],
                "task_id": task["task_id"],
                "task_1_output": cycle.get("code"),
                "success": bool(cycle.get("passed")),
            }
            if cycle.get("summary") is not None:
                record["task_2_output"] = cycle["summary"]
            test_results = {"success": bool(cycle.get("passed"))}
            for package_key, results_key in (("base_passed", "base_success"), ("plus_passed", "plus_success")):
                if package_key in cycle:
                    test_results[results_key] = cycle[package_key]
            record["test_results"] = test_results
            cycles.append(record)
        prompt_results.append({
            "task_id": task["task_id"],
            "initial_prompt": task.get("initial_prompt"),
            "cycles": cycles,
            "successful_cycles": task["successful_cycles"],
            "max_cycles_reached": bool(task.get("max_cycles_reached")),
        })
    return {
        "model": raw["model_id"],
        "model_name": raw.get("name"),
        "max_cycles": BASELINE_MAX_CYCLES,
        "dataset": {
            "name": "mbpp_plus",
            "configuration": BASELINE_CONFIGURATION,
            "source": "assignment2-data",
        },
        "prompt_results": prompt_results,
    }


def baseline_result(model_id: str, baseline_dir: PathLike = DEFAULT_BASELINE_DIR) -> Dict[str, Any]:
    """Loads, validates and converts one model's baseline records."""
    return to_results(load_loops(model_id, baseline_dir))


def compute_metrics(model_result: Mapping[str, Any]) -> BaselineMetrics:
    """Computes the judge-free headline metrics of a model result (baseline or BigCodeBench)."""
    cycles = [prompt_result["successful_cycles"] for prompt_result in model_result["prompt_results"]]
    max_cycles = int(model_result["max_cycles"])
    return BaselineMetrics(
        n_tasks=len(cycles),
        success_rate=metrics.pass_at_k(cycles, 1) / len(cycles),
        asl_score=metrics.asl_pow(cycles, max_cycles),
    )


def reproduction_check(
    model_id: str,
    baseline_dir: PathLike = DEFAULT_BASELINE_DIR,
    leaderboard: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ReproductionCheck:
    """Recomputes a baseline model's metrics and pairs them with its leaderboard entry."""
    if leaderboard is None:
        leaderboard = load_leaderboard(baseline_dir)
    recomputed = compute_metrics(baseline_result(model_id, baseline_dir))
    entry = leaderboard.get(model_id, {})
    return ReproductionCheck(
        model_id=model_id,
        recomputed=recomputed,
        leaderboard_success_rate=entry.get("successRate"),
        leaderboard_asl_score=entry.get("aslScore"),
    )


def check_all(baseline_dir: PathLike = DEFAULT_BASELINE_DIR) -> List[ReproductionCheck]:
    """Runs :func:`reproduction_check` for every model with per-cycle records."""
    leaderboard = load_leaderboard(baseline_dir)
    return [reproduction_check(model_id, baseline_dir, leaderboard) for model_id in list_model_ids(baseline_dir)]


def _require_dir(baseline_dir: PathLike) -> Path:
    """Returns the package directory, or explains how to provide it."""
    path = Path(baseline_dir)
    if not (path / "loops").is_dir() or not (path / "leaderboard.json").is_file():
        raise BaselineDataError(
            f"{path} is not the assignment2-data package (expected loops/ and leaderboard.json); "
            f"pass --baseline_dir with the unzipped package"
        )
    return path


def _read_json(path: Path) -> Any:
    """Reads a JSON file of the package."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as err:
        raise BaselineDataError(f"{path} is not valid JSON: {err}") from err
