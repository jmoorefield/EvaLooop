"""Summarization step of the standard loop on BigCodeBench, with the ``code_prompt`` anchor.

BigCodeBench prompts have no assert line to re-attach, so the anchor is the task's
``code_prompt`` (its imports and the ``def task_func(...):`` signature). Each summary becomes the
docstring of that stub, giving the next cycle's prompt the same shape as a Complete prompt and
keeping the function name, parameters and imports fixed across cycles::

    import itertools
    from random import shuffle

    def task_func(numbers=list(range(1, 3))):
        \"\"\"
        <summary, indented 4 spaces>
        \"\"\"

This module registers ``bigcodebench_summarization`` when it is imported; the BigCodeBench
evaluator imports it, so ``evaloop/tasks/__init__.py`` stays unchanged.
"""

from typing import Any, Dict

from evaloop.datasets.standard_prompts import SUMMARIZATION_TEMPLATE

from .base import BaseTask
from .factory import register_task

_DOCSTRING_INDENT = "    "


def sanitize_summary(summary: str) -> str:
    """Prepares a summary for embedding in a docstring, changing as little as possible.

    Like the standard loop, the summary is only stripped of surrounding whitespace; everything
    else (code fences, backslashes, formatting) is kept verbatim, because the model's own text is
    the only intended perturbation. The one addition is escaping triple double quotes: the summary
    sits inside a docstring here, and an unescaped ``\"\"\"`` would close it early.
    """
    return summary.strip().replace('"""', '\\"\\"\\"')


def build_anchored_prompt(code_prompt: str, summary: str) -> str:
    """Returns ``code_prompt`` followed by ``summary`` as the docstring of ``task_func``."""
    lines = sanitize_summary(summary).splitlines()
    body = "\n".join(_DOCSTRING_INDENT + line if line.strip() else "" for line in lines)
    docstring = f'{_DOCSTRING_INDENT}"""\n' + (body + "\n" if body else "") + f'{_DOCSTRING_INDENT}"""\n'
    return code_prompt.rstrip() + "\n" + docstring


@register_task("bigcodebench_summarization")
class BigCodeBenchSummarizationTask(BaseTask):
    """Summarizes code with the standard-loop prompt and anchors the summary to ``code_prompt``.

    Config keys:
        prompt_template: Summarization template with a ``{code}`` placeholder (defaults to the
            standard-loop template).
        code_prompts: ``task_id -> code_prompt`` map of the run's tasks.
    """

    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration."""
        super().__init__(task_config)
        self.code_prompts: Dict[str, str] = dict(task_config.get("code_prompts") or {})

    def create_prompt(self, input_text: str) -> str:
        """Create the standard-loop summarization prompt."""
        template = self.task_config.get("prompt_template", SUMMARIZATION_TEMPLATE)
        return template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        """The full stripped response is the summary, as in the standard loop."""
        return raw_response.strip()

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Builds the next cycle's prompt from the task's ``code_prompt`` and the summary.

        Raises:
            KeyError: If the task has no ``code_prompt`` in the config.
        """
        task_id = context.get("task_id")
        if task_id not in self.code_prompts:
            raise KeyError(f"No code_prompt configured for task {task_id!r}")
        return build_anchored_prompt(self.code_prompts[task_id], output)
