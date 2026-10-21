"""Prompt templates of the standard EvaLooop generation <-> summarization loop.

These are the exact templates that produced the course baseline (the per-cycle MBPP Plus records
of ``assignment2-data``); their source is ``data/assignment2-data/prompts/task_prompts.md``. They
are protocol definitions, reproduced here so the loop runs without the data package. Keep them
byte-identical to the source, or runs stop being comparable with the baseline.

The defaults in :meth:`evaloop.core.config.EvaluationConfig.get_task_configs` differ from these
(different line breaks, and a one-sentence summary); runs built on them are not the standard loop.
"""

# ``{description}`` is the current natural-language prompt: the task's original prompt in cycle
# 1, the anchored summary of the previous cycle afterwards.
GENERATION_TEMPLATE = (
    "Generate Python code for the following task:\n"
    "\n"
    "{description}\n"
    "\n"
    "Return only the code, without explanations or comments."
)

# ``{code}`` is the code generated in the current cycle; only code that passed the tests is
# summarized.
SUMMARIZATION_TEMPLATE = (
    "Summarize what the following code does in a detailed description:\n"
    "\n"
    "```\n"
    "{code}\n"
    "```\n"
    "\n"
    "Return only the summary, without explanations or comments."
)
