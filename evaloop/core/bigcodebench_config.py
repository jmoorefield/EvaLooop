"""Configuration of BigCodeBench runs and the standard-loop protocol they must follow."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from evaloop.evaluation.gradio_tester import EvaluatorLimits

# Settings of the standard loop that produced the MBPP Plus baseline (see
# data/assignment2-data/prompts/task_prompts.md). Runs that deviate are refused.
STANDARD_PROTOCOL: Dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
    "max_cycles": 10,
}

# System prompt of the baseline's API runs: EvaluationConfig passes none to API models, so this
# repo's OpenAILLM fell back to its default. BigCodeBench runs send it explicitly to API models.
STANDARD_SYSTEM_PROMPT = "You are a helpful AI assistant."
# EvaluationConfig's (and the CLI's) default system prompt, which API models never received; a run
# that leaves it unchanged gets STANDARD_SYSTEM_PROMPT.
FRAMEWORK_DEFAULT_SYSTEM_PROMPT = "You are a helpful programming assistant."


@dataclass
class BigCodeBenchConfig:
    """BigCodeBench-specific settings of an ``evaloop evaluate`` run.

    Attributes:
        trial: Run the 10-task trial instead of the full task set.
        trial_seed: Seed used to pick the trial tasks.
        gradio_endpoint: Evaluator URL, for tests; runs read ``$EVALOOP_GRADIO_ENDPOINT``. Never
            recorded.
        parallel: Evaluator worker processes. The time and memory limits are fixed at the
            :class:`EvaluatorLimits` defaults, and calibrated mode is always on.
        resume_from: Checkpoint to resume.
    """

    trial: bool = False
    trial_seed: int = 0
    gradio_endpoint: Optional[str] = None
    parallel: int = EvaluatorLimits.parallel
    resume_from: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates the configuration."""
        if self.parallel < 1:
            raise ValueError(f"parallel must be at least 1, got {self.parallel}")

    def limits(self) -> EvaluatorLimits:
        """Returns the evaluator execution settings."""
        return EvaluatorLimits(parallel=self.parallel)


def api_system_prompt(requested: Optional[str]) -> str:
    """Returns the system prompt an API model is sent.

    Args:
        requested: The run's ``--system_prompt``. ``None`` or the framework default means the
            standard prompt; anything else is sent as given (a protocol deviation).
    """
    if requested is None or requested == FRAMEWORK_DEFAULT_SYSTEM_PROMPT:
        return STANDARD_SYSTEM_PROMPT
    return requested


def check_protocol(settings: Dict[str, Any]) -> None:
    """Refuses run settings that differ from :data:`STANDARD_PROTOCOL`.

    Args:
        settings: The run's ``temperature``, ``top_p``, ``max_new_tokens`` and ``max_cycles``,
            and, for API models, the ``system_prompt`` they are sent (compared with
            :data:`STANDARD_SYSTEM_PROMPT`).

    Raises:
        ValueError: If any setting deviates.
    """
    expected_settings = dict(STANDARD_PROTOCOL)
    if "system_prompt" in settings:
        expected_settings["system_prompt"] = STANDARD_SYSTEM_PROMPT
    deviations = [
        f"{name}={settings[name]!r} (standard: {expected!r})"
        for name, expected in expected_settings.items()
        if settings[name] != expected
    ]
    if deviations:
        raise ValueError(
            "These settings differ from the standard loop the baseline used: "
            + ", ".join(deviations)
            + ". Use the defaults."
        )
