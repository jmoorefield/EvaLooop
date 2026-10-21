"""Draws failed loops for manual inspection (``evaloop inspect sample``).

The tool only selects and lays out loops. It proposes no categories and pre-fills no labels, so
the failure modes can be built inductively from the inspection itself. Outputs, in
``results/inspection/``:

* ``annotations.csv`` (submitted): one row per sampled loop, pre-filled with identifiers only,
  with free-text ``category`` and ``note`` columns to fill in by hand.
* ``sampling.json`` (submitted): seed, strata, population sizes, allocation, sampled loop ids and
  the representative cases, i.e. how the sample was drawn.
* ``worksheet.md`` (local only, never submitted): each loop's prompts, code, summaries and test
  outcome side by side. It contains data-package text for baseline loops.

A loop is ``(model_id, task_id, failing cycle)``, with failing cycle = ``successful_cycles + 1``.
Strata are model x failure-cycle group (first: cycle 1, early: 2-3, late: 4-10). Each stratum is
drawn with its own seed derived from the main seed, so one stratum's allocation does not change
another stratum's draw. Nothing here executes code.
"""

import csv
import difflib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from evaloop.baselines import assignment_data as ad

from .compare import Run

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
MIN_SAMPLE = 20
CYCLE_GROUPS = ("first", "early", "late")
SOURCES = ("bigcodebench", "baseline")
ANNOTATION_COLUMNS = (
    "source", "model_id", "task_id", "failing_cycle", "cycle_group", "category", "note",
)
DEFAULT_OUTPUT_DIR = Path("results") / "inspection"
_DETAILS_PREVIEW_CHARS = 1500


@dataclass(frozen=True)
class Loop:
    """One failed loop."""

    source: str
    model_id: str
    task_id: str
    failing_cycle: int

    @property
    def cycle_group(self) -> str:
        return cycle_group(self.failing_cycle)

    @property
    def key(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "task_id": self.task_id, "failing_cycle": self.failing_cycle}


def cycle_group(failing_cycle: int) -> str:
    """Failure-cycle group: ``first`` (1), ``early`` (2-3) or ``late`` (4 and later)."""
    if failing_cycle <= 1:
        return "first"
    return "early" if failing_cycle <= 3 else "late"


def failed_loops(source: str, model_id: str, model_result: Mapping[str, Any]) -> List[Loop]:
    """Returns a run's failed loops (tasks that did not sustain every cycle), sorted by task."""
    max_cycles = int(model_result["max_cycles"])
    loops = [
        Loop(source, model_id, str(prompt_result["task_id"]), int(prompt_result["successful_cycles"]) + 1)
        for prompt_result in model_result["prompt_results"]
        if int(prompt_result["successful_cycles"]) < max_cycles
    ]
    return sorted(loops, key=lambda loop: _task_key(loop.task_id))


def allocate(total: int, available: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    """Splits ``total`` places evenly over strata, capped at what each stratum has.

    Water-filling: strata are visited from smallest to largest, each taking an even share of the
    places still left (capped at its size), so a short stratum's places spread evenly over the
    others. Places left over by rounding go one at a time to the largest strata with spare
    loops. If every stratum is exhausted, fewer than ``total`` places are allocated.

    Args:
        total: Places to allocate.
        available: ``(stratum, number of loops)`` pairs in a fixed order (it breaks ties).
    """
    sizes = dict(available)
    order = [name for name, _ in available]
    counts = {name: 0 for name in order}
    left = total
    ascending = sorted(order, key=lambda name: sizes[name])
    for index, name in enumerate(ascending):
        take = min(sizes[name], left // (len(ascending) - index))
        counts[name] = take
        left -= take
    for name in sorted(order, key=lambda name: -sizes[name]):
        if left == 0:
            break
        if counts[name] < sizes[name]:
            counts[name] += 1
            left -= 1
    return counts


def draw(
    source: str,
    loops_by_model: Mapping[str, Sequence[Loop]],
    per_model: Mapping[str, int],
    seed: int,
) -> Tuple[List[Loop], Dict[str, Any]]:
    """Draws each model's places across its non-empty cycle groups.

    Returns:
        The sampled loops and a record of population sizes and allocation per stratum.
    """
    sampled: List[Loop] = []
    record: Dict[str, Any] = {}
    for model_id, loops in loops_by_model.items():
        groups = {group: [loop for loop in loops if loop.cycle_group == group] for group in CYCLE_GROUPS}
        population = {group: len(members) for group, members in groups.items()}
        allocation = allocate(per_model[model_id], [(group, population[group]) for group in CYCLE_GROUPS if population[group]])
        for group in CYCLE_GROUPS:
            take = allocation.get(group, 0)
            if take:
                rng = random.Random(f"{seed}:{source}:{model_id}:{group}")
                sampled.extend(sorted(rng.sample(groups[group], take), key=lambda loop: _task_key(loop.task_id)))
        record[model_id] = {
            "population": population,
            "allocation": {group: allocation.get(group, 0) for group in CYCLE_GROUPS},
        }
    return sampled, record


def sample(
    runs: Sequence[Run],
    n: int = MIN_SAMPLE,
    baseline_n: int = 10,
    seed: int = 1234,
    baseline_dir: PathLike = ad.DEFAULT_BASELINE_DIR,
) -> Dict[str, Any]:
    """Draws the inspection sample from BigCodeBench runs and their baselines.

    Raises:
        ValueError: If a model has several runs, ``n`` is below the minimum, or there are too few
            failed loops (both relaxed for trial runs).
        BaselineDataError: If a model has no baseline records.
    """
    model_ids = [run.model_id for run in runs]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("inspect sample takes one run per model; pass each model once in --runs")
    trial = any(run.trial for run in runs)
    if n < MIN_SAMPLE and not trial:
        raise ValueError(f"The assignment requires at least {MIN_SAMPLE} inspected loops; got --n {n}")
    if baseline_n < 0:
        raise ValueError(f"baseline_n must not be negative, got {baseline_n}")

    results = {run.model_id: run.result for run in runs}
    baselines = {model_id: ad.baseline_result(model_id, baseline_dir) for model_id in model_ids}
    bcb_loops = {model_id: failed_loops("bigcodebench", model_id, results[model_id]) for model_id in model_ids}
    base_loops = {model_id: failed_loops("baseline", model_id, baselines[model_id]) for model_id in model_ids}

    per_model = allocate(n, [(model_id, len(bcb_loops[model_id])) for model_id in model_ids])
    drawn = sum(per_model.values())
    if drawn < n:
        message = f"Only {drawn} failed loops exist across the runs; {n} were requested"
        if not trial:
            raise ValueError(message)
        logger.warning("%s (trial run: sampling all of them)", message)
    bcb_sample, bcb_record = draw("bigcodebench", bcb_loops, per_model, seed)

    base_per_model = {model_id: baseline_n for model_id in model_ids}
    base_sample, base_record = draw("baseline", base_loops, base_per_model, seed)
    for model_id, record in base_record.items():
        if sum(record["allocation"].values()) < baseline_n:
            logger.warning("%s: only %d baseline failed loops available", model_id, sum(record["allocation"].values()))

    representative = {
        model_id: _representative(model_id, results[model_id], [loop for loop in bcb_sample if loop.model_id == model_id], seed)
        for model_id in model_ids
    }
    return {
        "trial": trial,
        "seed": seed,
        "n": n,
        "baseline_n": baseline_n,
        "strata": {"groups": {"first": [1, 1], "early": [2, 3], "late": [4, None]}, "by": "model x cycle group"},
        "runs": [
            {"model_id": run.model_id, "path": str(run.path), "configuration": run.configuration,
             "run_id": run.result.get("run_id")}
            for run in runs
        ],
        "bigcodebench": {"per_model": per_model, "strata": bcb_record, "loops": [loop.key for loop in bcb_sample]},
        "baseline": {"per_model": base_per_model, "strata": base_record, "loops": [loop.key for loop in base_sample]},
        "representative_cases": representative,
        "_samples": {"bigcodebench": bcb_sample, "baseline": base_sample},
        "_results": {"bigcodebench": results, "baseline": baselines},
    }


def write_outputs(drawn: Mapping[str, Any], output_dir: PathLike = DEFAULT_OUTPUT_DIR, overwrite: bool = False) -> Dict[str, Path]:
    """Writes ``annotations.csv``, ``sampling.json`` and the local ``worksheet.md``.

    Raises:
        FileExistsError: If ``annotations.csv`` exists and ``overwrite`` is false (it holds
            hand-made annotations).
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    annotations = directory / "annotations.csv"
    if annotations.exists() and not overwrite:
        raise FileExistsError(
            f"{annotations} already exists and may hold your annotations; pass --overwrite to replace it"
        )
    samples = drawn["_samples"]
    rows = [_annotation_row(loop) for loop in samples["bigcodebench"] + samples["baseline"]]
    with open(annotations, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    sampling = directory / "sampling.json"
    public = {key: value for key, value in drawn.items() if not key.startswith("_")}
    sampling.write_text(json.dumps(public, indent=2), encoding="utf-8")

    worksheet = directory / "worksheet.md"
    worksheet.write_text(render_worksheet(drawn), encoding="utf-8")
    return {"annotations": annotations, "sampling": sampling, "worksheet": worksheet}


def render_worksheet(drawn: Mapping[str, Any]) -> str:
    """Lays out every sampled loop for reading (local file; contains package text)."""
    lines = [
        "<!-- LOCAL WORKING FILE: contains data-package and model text. Never commit or submit it. -->",
        "",
        "# Inspection worksheet",
        "",
        "> **Local working file. Never commit or submit it.** Baseline loops show data-package text.",
        "> Classify each loop in `annotations.csv`. Notes there must describe, not quote, package text.",
        "",
    ]
    if drawn["trial"]:
        lines[2:2] = ["**TRIAL SAMPLE: not for reporting.**", ""]
    representative = drawn["representative_cases"]
    for source in SOURCES:
        lines += [f"## {source}", ""]
        for loop in drawn["_samples"][source]:
            result = drawn["_results"][source][loop.model_id]
            prompt_result = next(entry for entry in result["prompt_results"] if str(entry["task_id"]) == loop.task_id)
            marker = ""
            if source == "bigcodebench" and representative.get(loop.model_id, {}).get("broke") == loop.key:
                marker = " (representative broken loop)"
            lines += _loop_section(loop, prompt_result, marker)
    for model_id, cases in representative.items():
        survived = cases.get("did_not_break")
        if survived:
            result = drawn["_results"]["bigcodebench"][model_id]
            prompt_result = next(entry for entry in result["prompt_results"] if str(entry["task_id"]) == survived["task_id"])
            lines += [f"## Representative loop that did not break: {model_id} / {survived['task_id']}", ""]
            lines += _block("Original prompt", prompt_result.get("initial_prompt"))
            final = prompt_result["cycles"][-1]
            lines += _block(f"Prompt fed into cycle {final['cycle']}", _prompt_fed(prompt_result, final["cycle"]))
            lines += _block(f"Code of cycle {final['cycle']}", final.get("task_1_output"))
    return "\n".join(lines) + "\n"


def word_diff(before: str, after: str) -> str:
    """Word-level diff: ``[-removed-]`` and ``{+added+}`` around changed words."""
    old, new = before.split(), after.split()
    parts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            parts.append(" ".join(old[i1:i2]))
        if tag in ("delete", "replace"):
            parts.append("[-" + " ".join(old[i1:i2]) + "-]")
        if tag in ("insert", "replace"):
            parts.append("{+" + " ".join(new[j1:j2]) + "+}")
    return " ".join(part for part in parts if part)


def _loop_section(loop: Loop, prompt_result: Mapping[str, Any], marker: str) -> List[str]:
    """Worksheet section of one loop: the last passing cycle and the failing cycle."""
    sustained = loop.failing_cycle - 1
    cycles = {int(cycle["cycle"]): cycle for cycle in prompt_result.get("cycles", [])}
    lines = [
        f"### {loop.source} / {loop.model_id} / {loop.task_id} / failing cycle {loop.failing_cycle} "
        f"({loop.cycle_group}){marker}",
        "",
        f"Sustained {sustained} cycle(s).",
        "",
    ]
    lines += _block("Original prompt", prompt_result.get("initial_prompt"))
    if sustained >= 1:
        last = cycles.get(sustained, {})
        lines += [f"#### Last passing cycle ({sustained})", ""]
        lines += _block("Prompt fed in", _prompt_fed(prompt_result, sustained))
        lines += _code_blocks(last)
        lines += _block("Summary (became the failing cycle's prompt)", last.get("task_2_output"))
    failing = cycles.get(loop.failing_cycle, {})
    lines += [f"#### Failing cycle ({loop.failing_cycle})", ""]
    fed = _prompt_fed(prompt_result, loop.failing_cycle)
    lines += _block("Prompt fed in", fed)
    if loop.failing_cycle > 1 and fed and prompt_result.get("initial_prompt"):
        lines += ["Word diff, original prompt -> prompt of the failing cycle:", "", "> " + word_diff(prompt_result["initial_prompt"], fed), ""]
    lines += _code_blocks(failing)
    lines += _test_outcome(failing)
    return lines


def _prompt_fed(prompt_result: Mapping[str, Any], cycle: int) -> Optional[str]:
    """The prompt that fed ``cycle``: the original prompt, else the previous cycle's summary."""
    if cycle <= 1:
        return prompt_result.get("initial_prompt")
    previous = next((entry for entry in prompt_result.get("cycles", []) if int(entry["cycle"]) == cycle - 1), {})
    return previous.get("task_2_output")


def _code_blocks(cycle: Mapping[str, Any]) -> List[str]:
    lines = _block("Extracted code", cycle.get("task_1_output"))
    raw = cycle.get("task_1_raw_output")
    if raw is not None and raw.strip() != (cycle.get("task_1_output") or "").strip():
        lines += _block("Raw model output (extraction changed it)", raw)
    return lines


def _test_outcome(cycle: Mapping[str, Any]) -> List[str]:
    results = cycle.get("test_results") or {}
    if "status" in results:  # BigCodeBench evaluator result
        lines = [f"Test status: `{results.get('status')}`", ""]
        details = results.get("details") or {}
        text = "\n".join(f"{name}: {trace}" for name, trace in details.items())
        return lines + _block("Test details (truncated)", text[-_DETAILS_PREVIEW_CHARS:] if text else None)
    flags = ", ".join(f"{key}={results[key]}" for key in ("base_success", "plus_success") if key in results)
    return [f"Test outcome: passed={cycle.get('success')}" + (f" ({flags})" if flags else ""), ""]


def _block(title: str, text: Optional[str]) -> List[str]:
    if not text:
        return [f"**{title}:** _(none)_", ""]
    return [f"**{title}:**", "", "````text", text.rstrip("\n"), "````", ""]


def _annotation_row(loop: Loop) -> Dict[str, Any]:
    row = {column: "" for column in ANNOTATION_COLUMNS}
    row.update(source=loop.source, model_id=loop.model_id, task_id=loop.task_id,
               failing_cycle=loop.failing_cycle, cycle_group=loop.cycle_group)
    return row


def _representative(model_id: str, result: Mapping[str, Any], sampled: Sequence[Loop], seed: int) -> Dict[str, Any]:
    """Seeded picks of a broken loop (from the sample) and a loop that survived every cycle.

    A broken loop is preferably one that failed after passing at least once, i.e. one the loop
    itself broke; a first-cycle failure is used only if no other exists.
    """
    rng = random.Random(f"{seed}:representative:{model_id}")
    broke_candidates = [loop for loop in sampled if loop.failing_cycle > 1] or list(sampled)
    max_cycles = int(result["max_cycles"])
    survivors = sorted(
        (str(entry["task_id"]) for entry in result["prompt_results"] if int(entry["successful_cycles"]) == max_cycles),
        key=_task_key,
    )
    return {
        "broke": rng.choice(broke_candidates).key if broke_candidates else None,
        "did_not_break": {"model_id": model_id, "task_id": rng.choice(survivors)} if survivors else None,
    }


def _task_key(task_id: str) -> Tuple[str, int, str]:
    prefix, _, number = task_id.rpartition("/")
    return (prefix, int(number), "") if number.isdigit() else (prefix, -1, task_id)
