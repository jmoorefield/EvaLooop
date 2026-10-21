"""Benchmark adapters for evaluation loops beyond the built-in MBPP Plus path.

Modules here must import their heavy dependencies (e.g. ``bigcodebench``) lazily, so the package
stays importable with only the lightweight test dependencies installed.
"""

from .standard_prompts import GENERATION_TEMPLATE, SUMMARIZATION_TEMPLATE

__all__ = ["GENERATION_TEMPLATE", "SUMMARIZATION_TEMPLATE"]
