"""Compares BigCodeBench runs with the same models' MBPP Plus baseline (``evaloop compare``).

Every number on both sides is computed by the same code from the per-task sustained loops: the
baseline from ``data/assignment2-data/loops/`` (recomputed, never the leaderboard's averages) and
the BigCodeBench side from the runs' results files. No LLM judge is involved.

Reports contain numbers and identifiers only, never prompts, code or summaries, so they can be
submitted without copying package records.

Metric conventions (``M`` = max cycles, ``n`` = tasks):

* ``success_rate`` = pass@1 / n
* ``asl`` = unweighted ASL, sum_i i^2 n_i / (n M) (the package's ``aslScore``)
* ``robustnessScore`` = the same sum normalised by pass@1 instead of n (the package's
  ``robustnessScore``: robustness on solvable tasks)
* ``pass_drop`` = (pass@M - pass@1) / n, a fraction, negative when tasks are lost. The sign matches
  the leaderboard's ``passDrop``, which is the unnormalised count pass@M - pass@1; dividing by n
  keeps it comparable between benchmarks of different sizes.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from evaloop.analysis import metrics
from evaloop.baselines import assignment_data as ad

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
COMPARED_METRICS = ("success_rate", "asl", "robustnessScore", "pass_drop")
RANKED_METRICS = ("asl", "success_rate", "robustnessScore")


@dataclass(frozen=True)
class RunMetrics:
    """Judge-free metrics of one run (baseline or BigCodeBench)."""

    n_tasks: int
    max_cycles: int
    success_rate: float
    asl: float
    robustnessScore: float
    pass_drop: float
    survival: Tuple[float, ...]  # share of tasks still passing at cycle k = 1..M
    histogram: Tuple[int, ...]  # tasks that sustained exactly i cycles, i = 0..M


@dataclass(frozen=True)
class Run:
    """A BigCodeBench results file paired with the package ``model_id`` of its baseline."""

    model_id: str
    path: Path
    result: Dict[str, Any]

    @property
    def configuration(self) -> str:
        return str(self.result["dataset"]["configuration"])

    @property
    def trial(self) -> bool:
        return bool(self.result.get("trial"))


def run_metrics(model_result: Mapping[str, Any]) -> RunMetrics:
    """Computes :class:`RunMetrics` from a model result's sustained loops."""
    cycles = [int(prompt_result["successful_cycles"]) for prompt_result in model_result["prompt_results"]]
    if not cycles:
        raise ValueError("Cannot compute metrics for a run without tasks")
    max_cycles = int(model_result["max_cycles"])
    n_tasks = len(cycles)
    return RunMetrics(
        n_tasks=n_tasks,
        max_cycles=max_cycles,
        success_rate=metrics.pass_at_k(cycles, 1) / n_tasks,
        asl=metrics.asl_pow(cycles, max_cycles),
        robustnessScore=metrics.asl_bias(cycles, max_cycles),
        pass_drop=-metrics.pass_drop(cycles, max_cycles),  # leaderboard sign: pass@M - pass@1
        survival=tuple(metrics.pass_at_k(cycles, k) / n_tasks for k in range(1, max_cycles + 1)),
        histogram=tuple(metrics.cycle_histogram(cycles, max_cycles)),
    )


def parse_run_args(runs: Union[str, Sequence[str]]) -> List[Tuple[str, Path]]:
    """Parses ``model_id=path`` entries (comma-separated, or a list as Fire may pass it).

    Raises:
        ValueError: If an entry is malformed or no entry is given.
    """
    items = runs.split(",") if isinstance(runs, str) else [str(item) for item in runs]
    parsed = []
    for item in (item.strip() for item in items):
        if not item:
            continue
        model_id, separator, path = item.partition("=")
        if not separator or not model_id.strip() or not path.strip():
            raise ValueError(f"Expected model_id=path in --runs, got {item!r}")
        parsed.append((model_id.strip(), Path(path.strip())))
    if not parsed:
        raise ValueError("--runs needs at least one model_id=path entry")
    return parsed


def load_runs(runs: Union[str, Sequence[str]], include_trial: bool = False) -> List[Run]:
    """Loads BigCodeBench results files and checks they may be reported together.

    Raises:
        ValueError: If a file is not a BigCodeBench result, is a trial run (unless included
            explicitly), or two runs share a (model, configuration).
    """
    loaded: List[Run] = []
    for model_id, path in parse_run_args(runs):
        run = Run(model_id=model_id, path=path, result=_load_result(path))
        if run.trial and not include_trial:
            raise ValueError(f"{path} is a trial run; pass --include_trial to use it (not reportable)")
        loaded.append(run)
    seen = set()
    for run in loaded:
        key = (run.model_id, run.configuration)
        if key in seen:
            raise ValueError(f"Two results files for model {run.model_id!r} and configuration {run.configuration!r}")
        seen.add(key)
    return loaded


def compare(runs: Sequence[Run], baseline_dir: PathLike = ad.DEFAULT_BASELINE_DIR) -> Dict[str, Any]:
    """Builds the comparison report of BigCodeBench runs against their MBPP Plus baselines.

    Raises:
        BaselineDataError: If a run's ``model_id`` has no baseline records.
    """
    leaderboard = ad.load_leaderboard(baseline_dir)
    loop_ids = ad.list_model_ids(baseline_dir)
    baselines = {model_id: run_metrics(ad.baseline_result(model_id, baseline_dir)) for model_id in loop_ids}
    for run in runs:
        if run.model_id not in baselines:
            ad.load_loops(run.model_id, baseline_dir)  # raises with the list of available ids

    loop_asl = {model_id: values.asl for model_id, values in baselines.items()}
    board_asl = {
        model_id: float(entry["aslScore"]) for model_id, entry in leaderboard.items() if entry.get("aslScore") is not None
    }

    models = []
    for run in runs:
        bcb = run_metrics(run.result)
        base = baselines[run.model_id]
        reproduction = ad.reproduction_check(run.model_id, baseline_dir, leaderboard)
        models.append({
            "model_id": run.model_id,
            "configuration": run.configuration,
            "run_id": run.result.get("run_id"),
            "model_identifier": run.result.get("model_identifier"),
            "trial": run.trial,
            "bigcodebench": _metrics_dict(bcb),
            "mbpp_plus": _metrics_dict(base),
            "change": {name: _change(getattr(base, name), getattr(bcb, name)) for name in COMPARED_METRICS},
            "baseline_rank": {
                "among_loop_models": _rank(run.model_id, loop_asl),
                "among_leaderboard_models": _rank(run.model_id, board_asl),
                "leaderboard_published_rank": leaderboard.get(run.model_id, {}).get("rank"),
                "leaderboard_matches_loops": reproduction.matches,
            },
        })

    return {
        "trial": any(run.trial for run in runs),
        "baseline_dir": str(baseline_dir),
        "models": models,
        "ranking": _ranking(models),
        "notes": [
            "asl is the unweighted ASL (the package's aslScore); no LLM judge is used.",
            "pass_drop = (pass@M - pass@1) / n; the leaderboard's passDrop has the same sign but is a task "
            "count (pass@M - pass@1), not divided by n.",
            "Baseline metrics are recomputed from loops/ (one run); leaderboard values may average several runs "
            "for some models (leaderboard_matches_loops = false).",
            "Leaderboard ranks here order models by aslScore; leaderboard_published_rank is the site's rank "
            "(by its judge-weighted ASL column).",
        ],
    }


def write_report(report: Mapping[str, Any], output_dir: PathLike) -> Tuple[Path, Path]:
    """Writes ``comparison.json`` and ``comparison.md`` (numbers and ids only)."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "comparison.json"
    md_path = directory / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: Mapping[str, Any]) -> str:
    """Renders the report as Markdown tables."""
    lines = []
    if report["trial"]:
        lines += ["> **TRIAL OUTPUT: not for reporting.**", ""]
    lines += ["# BigCodeBench vs. MBPP Plus baseline", ""]

    lines += ["## Metrics", "", "| model | configuration | benchmark | n_tasks | success_rate | asl | robustnessScore | pass_drop |",
              "|---|---|---|---|---|---|---|---|"]
    for model in report["models"]:
        for benchmark, label in (("mbpp_plus", "standard-mbpp-plus"), ("bigcodebench", model["configuration"])):
            values = model[benchmark]
            lines.append(
                f"| {model['model_id']} | {label} | {benchmark} | {values['n_tasks']} | {values['success_rate']:.3f} "
                f"| {values['asl']:.3f} | {values['robustnessScore']:.3f} | {values['pass_drop']:.3f} |"
            )

    lines += ["", "## Change from MBPP Plus to BigCodeBench", "", "| model | metric | absolute | relative |",
              "|---|---|---|---|"]
    for model in report["models"]:
        for name in COMPARED_METRICS:
            change = model["change"][name]
            relative = "n/a" if change["relative"] is None else f"{100 * change['relative']:+.1f}%"
            lines.append(f"| {model['model_id']} | {name} | {change['absolute']:+.3f} | {relative} |")

    lines += ["", "## Survival (share of tasks passing at cycle k)", ""]
    max_cycles = max(model["bigcodebench"]["max_cycles"] for model in report["models"])
    header = " | ".join(str(k) for k in range(1, max_cycles + 1))
    lines += [f"| model | benchmark | {header} |", "|---|---|" + "---|" * max_cycles]
    for model in report["models"]:
        for benchmark in ("mbpp_plus", "bigcodebench"):
            cells = " | ".join(f"{share:.3f}" for share in model[benchmark]["survival"])
            lines.append(f"| {model['model_id']} | {benchmark} | {cells} |")

    lines += ["", "## Baseline rank (MBPP Plus, by unweighted ASL)", "",
              "| model | among loop models | among leaderboard models | site rank | leaderboard = loops |",
              "|---|---|---|---|---|"]
    for model in report["models"]:
        rank = model["baseline_rank"]
        lines.append(
            f"| {model['model_id']} | {_format_rank(rank['among_loop_models'])} "
            f"| {_format_rank(rank['among_leaderboard_models'])} | {rank['leaderboard_published_rank']} "
            f"| {'yes' if rank['leaderboard_matches_loops'] else 'no (leaderboard may average runs)'} |"
        )

    ranking = report["ranking"]
    lines += ["", "## Ranking among the evaluated models", ""]
    for name in RANKED_METRICS:
        orders = ranking["orders"][name]
        changed = " **(order changed)**" if ranking["order_changed"][name] else ""
        lines.append(f"- **{name}**: MBPP Plus {' > '.join(orders['mbpp_plus'])}; "
                     f"BigCodeBench {' > '.join(orders['bigcodebench'])}{changed}")
    lines += ["", "## Notes", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def _ranking(models: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Orders of the evaluated models on each benchmark, and whether each order changed."""
    orders, changed = {}, {}
    for name in RANKED_METRICS:
        orders[name] = {
            benchmark: [m["model_id"] for m in sorted(models, key=lambda m: -m[benchmark][name])]
            for benchmark in ("mbpp_plus", "bigcodebench")
        }
        changed[name] = orders[name]["mbpp_plus"] != orders[name]["bigcodebench"]
    return {"orders": orders, "order_changed": changed}


def _rank(model_id: str, scores: Mapping[str, float]) -> Optional[Dict[str, Any]]:
    """Competition rank (1 = best) and percentile of a model among ``scores``.

    Percentile is the share of the other models it beats or ties, in percent (rank 1 -> 100).
    """
    if model_id not in scores:
        return None
    score = scores[model_id]
    others = [value for key, value in scores.items() if key != model_id]
    rank = 1 + sum(1 for value in others if value > score)
    percentile = 100.0 * sum(1 for value in others if value <= score) / len(others) if others else 100.0
    return {"rank": rank, "of": len(scores), "percentile": percentile}


def _format_rank(rank: Optional[Mapping[str, Any]]) -> str:
    return "n/a" if rank is None else f"{rank['rank']} of {rank['of']} (percentile {rank['percentile']:.0f})"


def _change(base: float, new: float) -> Dict[str, Optional[float]]:
    # Divide by |base| so the relative change has the absolute change's sign even when base is
    # negative (pass_drop).
    return {"absolute": new - base, "relative": (new - base) / abs(base) if base else None}


def _metrics_dict(values: RunMetrics) -> Dict[str, Any]:
    data = asdict(values)
    data["survival"] = list(values.survival)
    data["histogram"] = list(values.histogram)
    return data


def _load_result(path: Path) -> Dict[str, Any]:
    """Loads a BigCodeBench results file (a list with one model result, or the result itself)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"{path} holds {len(data)} model results; expected one per BigCodeBench results file")
        data = data[0]
    if not isinstance(data, dict) or "prompt_results" not in data or "max_cycles" not in data:
        raise ValueError(f"{path} is not an EvaLoop results file")
    if data.get("dataset", {}).get("name") != "bigcodebench" or "configuration" not in data.get("dataset", {}):
        raise ValueError(f"{path} is not a BigCodeBench results file (written by evaloop evaluate --dataset bigcodebench)")
    return data
