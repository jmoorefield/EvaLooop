"""Writes ``summary.csv`` (``evaloop summary``): one row per run, baseline rows included.

Header (exactly): ``model,configuration,n_tasks,success_rate,asl_score``

* ``model`` is the package ``model_id`` of the run's baseline.
* Each model gets one ``standard-mbpp-plus`` row, recomputed from ``loops/`` (and checked against
  ``leaderboard.json``), then one row per BigCodeBench run.
* ``success_rate`` = pass@1 / n_tasks; ``asl_score`` = unweighted ASL (the package's ``aslScore``).
  Both are written to 3 decimal places.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Union

from evaloop.baselines import assignment_data as ad

from .compare import Run, run_metrics

logger = logging.getLogger(__name__)

HEADER = ("model", "configuration", "n_tasks", "success_rate", "asl_score")
DEFAULT_OUT = Path("results") / "summary.csv"
DEFAULT_TRIAL_OUT = Path("results") / "summary_trial.csv"

PathLike = Union[str, Path]


def summary_rows(runs: Sequence[Run], baseline_dir: PathLike = ad.DEFAULT_BASELINE_DIR) -> List[Dict[str, str]]:
    """Builds the rows: per model (in order of first appearance) its baseline row, then its runs.

    Raises:
        BaselineDataError: If a run's ``model_id`` has no baseline records.
    """
    leaderboard = ad.load_leaderboard(baseline_dir)
    rows: List[Dict[str, str]] = []
    model_ids = list(dict.fromkeys(run.model_id for run in runs))
    for model_id in model_ids:
        check = ad.reproduction_check(model_id, baseline_dir, leaderboard)
        if not check.matches:
            logger.warning(
                "%s: recomputed baseline (success %.3f, ASL %.3f) differs from leaderboard.json (%s, %s); "
                "the leaderboard may average several runs. Reporting the recomputed values.",
                model_id, check.recomputed.success_rate, check.recomputed.asl_score,
                check.leaderboard_success_rate, check.leaderboard_asl_score,
            )
        rows.append(_row(model_id, ad.BASELINE_CONFIGURATION, check.recomputed.n_tasks,
                         check.recomputed.success_rate, check.recomputed.asl_score))
        for run in (run for run in runs if run.model_id == model_id):
            values = run_metrics(run.result)
            rows.append(_row(model_id, run.configuration, values.n_tasks, values.success_rate, values.asl))
    return rows


def write_summary(rows: Sequence[Dict[str, str]], out: PathLike) -> Path:
    """Writes the rows with the exact required header."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(model_id: str, configuration: str, n_tasks: int, success_rate: float, asl_score: float) -> Dict[str, str]:
    return {
        "model": model_id,
        "configuration": configuration,
        "n_tasks": str(n_tasks),
        "success_rate": f"{success_rate:.3f}",
        "asl_score": f"{asl_score:.3f}",
    }
