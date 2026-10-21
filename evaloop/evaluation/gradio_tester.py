"""Tests BigCodeBench solutions with BigCodeBench's own test suites on a remote Gradio evaluator.

The evaluator is the ``bigcode/bigcodebench-evaluator`` Gradio app (run on Colab by the notebook
in ``notebooks/``). It scores a whole JSONL file of ``{"task_id", "solution"}`` samples per
request, so each loop cycle sends one batch. Generated code never runs on the machine that runs
EvaLoop.

The ``/predict`` call mirrors ``bigcodebench.evaluate``. Two quirks of the app matter here:

* ``selective_evaluate`` takes bare task numbers (``"0,13"``); the app prepends ``BigCodeBench/``.
* Per-task results are ``results["eval"][task_id][0]`` with ``status`` in ``pass`` / ``fail`` /
  ``timeout`` and ``details`` mapping test names to tracebacks, or ``{"ALL": str(exception)}``
  when the code could not even be loaded (that message carries no exception type).

``gradio_client`` is imported lazily; tests inject a fake client factory. The endpoint URL is
treated as a secret: it is never logged above DEBUG and is redacted from error messages.
"""

import asyncio
import json
import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from evaloop.datasets.bigcodebench import SUBSET, bare_task_id, full_task_id, task_sort_key

logger = logging.getLogger(__name__)

# HTTP libraries that log each request's full URL at INFO, which would write the secret endpoint
# into logs/evaloop.log. They are raised to WARNING whenever a tester exists.
_URL_LOGGING_LIBRARIES = ("httpx", "httpx2", "gradio_client")

ENDPOINT_ENV_VAR = "EVALOOP_GRADIO_ENDPOINT"
BACKEND_NAME = "bigcodebench-gradio"
PASS_STATUS = "pass"
TIMEOUT_STATUS = "timeout"
MISSING_STATUS = "missing"

# Hints for manual inspection, derived by simple rules from the evaluator's output (not decisions):
#   environment         a library the task itself needs is missing, or a resource limit was hit
#   unavailable_import  the code imports a module that is not installed and the task does not need
#   timeout             the evaluator's time limit was exceeded
#   syntax_or_name      the code does not compile, or task_func is not defined
#   assertion           a test assertion failed
#   exception           the code raised some other exception
#   unknown             failed without recognisable details
#   missing             the evaluator returned no result for the task
ERROR_CLASSES = (
    "environment",
    "unavailable_import",
    "timeout",
    "syntax_or_name",
    "assertion",
    "exception",
    "unknown",
    MISSING_STATUS,
)

_MISSING_MODULE = re.compile(r"No module named '([^']+)'")
_RESOURCE = re.compile(r"\bMemoryError\b|Cannot allocate memory|Resource temporarily unavailable")
_SYNTAX = re.compile(r"\b(SyntaxError|IndentationError|TabError)\b|\(.+, line \d+\)\s*$", re.MULTILINE)
_UNDEFINED_ENTRY_POINT = re.compile(r"name 'task_func' is not defined")
_EXCEPTION_NAME = re.compile(r"\b[A-Z]\w*(Error|Exception)\b")

# Exception class names (anywhere in the MRO) that mean "try again": httpx transport errors and
# timeouts, and the cancellations gradio_client raises when a queued job is dropped.
_RETRYABLE_EXCEPTION_NAMES = frozenset({
    "TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "TransportError", "NetworkError", "ConnectError", "ReadError", "WriteError",
    "RemoteProtocolError", "CancelledError", "QueueError",
})


class EvaluatorError(RuntimeError):
    """The evaluator rejected a request or returned something unusable."""


class EvaluatorUnavailableError(EvaluatorError):
    """The evaluator stayed unreachable through every retry.

    Attributes:
        attempts: Number of attempts made.
    """

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class Sample:
    """One solution to test: the extracted code of a task in the current cycle."""

    task_id: str
    code: str


@dataclass(frozen=True)
class TestResult:
    """Outcome of one task's test suite.

    Attributes:
        success: Whether every test passed.
        status: ``pass``, ``fail``, ``timeout`` or ``missing``.
        details: Failing test names mapped to (truncated) tracebacks.
        error_class: One of :data:`ERROR_CLASSES` for failures, ``None`` on success.
        backend: Evaluator backend that produced the result.
    """

    __test__ = False  # not a pytest test class

    success: bool
    status: str
    details: Dict[str, str] = field(default_factory=dict)
    error_class: Optional[str] = None
    backend: str = BACKEND_NAME

    def to_dict(self) -> Dict[str, Any]:
        """Returns the ``test_results`` entry of a cycle record."""
        return {
            "success": self.success,
            "status": self.status,
            "details": dict(self.details),
            "error_class": self.error_class,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class PreflightResult:
    """Ground-truth check of the requested tasks in the evaluator's environment.

    Attributes:
        gt_pass_rate: Share of requested tasks whose canonical solution passes.
        failed_tasks: Requested tasks whose canonical solution fails, sorted by task number.
    """

    gt_pass_rate: float
    failed_tasks: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluatorLimits:
    """Execution settings passed to the evaluator.

    The defaults suit a free Colab CPU runtime (about 2 vCPUs and 12 GB RAM): two workers whose
    address-space and data limits (in MB) fit in memory together. BigCodeBench's own defaults
    (30 GB per process) assume a much larger machine.
    """

    parallel: int = 2
    min_time_limit: float = 1.0
    max_as_limit: int = 4 * 1024
    max_data_limit: int = 4 * 1024
    max_stack_limit: int = 10


@dataclass(frozen=True)
class RetryPolicy:
    """Retries with exponential backoff and jitter for transient evaluator failures.

    Attributes:
        attempts: Maximum number of attempts per request.
        request_timeout: Seconds before a single request times out.
        max_wait: Maximum seconds spent on one request across all attempts and back-off.
        base_delay: Back-off before the second attempt; doubles each time.
        max_delay: Cap on a single back-off.
    """

    attempts: int = 5
    request_timeout: float = 1800.0
    max_wait: float = 3600.0
    base_delay: float = 5.0
    max_delay: float = 300.0

    def __post_init__(self) -> None:
        """Validates the policy."""
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")
        if min(self.request_timeout, self.max_wait, self.base_delay, self.max_delay) < 0:
            raise ValueError("Retry timeouts and delays must not be negative")


ClientFactory = Callable[[str, float], Any]


def resolve_endpoint(endpoint: Optional[str] = None) -> str:
    """Returns ``endpoint``, falling back to ``$EVALOOP_GRADIO_ENDPOINT``.

    Raises:
        ValueError: If neither is set.
    """
    resolved = endpoint or os.environ.get(ENDPOINT_ENV_VAR)
    if not resolved:
        raise ValueError(
            f"BigCodeBench needs a Gradio evaluator endpoint: set {ENDPOINT_ENV_VAR} to its URL"
        )
    return resolved


def classify_failure(status: str, details: Mapping[str, str], task_libs: Iterable[str] = ()) -> str:
    """Derives an inspection hint (:data:`ERROR_CLASSES`) for a failed task.

    Args:
        status: Evaluator status (``fail``, ``timeout`` or ``missing``).
        details: Failing test names mapped to tracebacks or exception messages.
        task_libs: Libraries the task declares; a missing one points at the environment.
    """
    if status == MISSING_STATUS:
        return MISSING_STATUS
    if status == TIMEOUT_STATUS:
        return "timeout"
    text = "\n".join(str(value) for value in details.values())
    if _RESOURCE.search(text):
        return "environment"
    missing_module = _MISSING_MODULE.search(text)
    if missing_module:
        declared = {lib.split(".")[0] for lib in task_libs}
        return "environment" if missing_module.group(1).split(".")[0] in declared else "unavailable_import"
    if _SYNTAX.search(text) or _UNDEFINED_ENTRY_POINT.search(text):
        return "syntax_or_name"
    if "AssertionError" in text:
        return "assertion"
    if _EXCEPTION_NAME.search(text):
        return "exception"
    return "unknown"


def is_retryable(exc: BaseException) -> bool:
    """Whether an error is transient: connection problems, timeouts, HTTP 5xx, dropped jobs."""
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.CancelledError)):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _RETRYABLE_EXCEPTION_NAMES:
        return True
    if "HTTPStatusError" in names:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return isinstance(status_code, int) and status_code >= 500
    return False


class GradioBigCodeBenchTester:
    """Batch tester backed by a BigCodeBench Gradio evaluator."""

    backend = BACKEND_NAME

    def __init__(
        self,
        endpoint: str,
        *,
        limits: Optional[EvaluatorLimits] = None,
        retry: Optional[RetryPolicy] = None,
        task_libs: Optional[Mapping[str, Sequence[str]]] = None,
        details_max_chars: int = 2000,
        client_factory: Optional[ClientFactory] = None,
        file_wrapper: Optional[Callable[[str], Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Optional[random.Random] = None,
    ):
        """
        Args:
            endpoint: Evaluator URL (secret; see the module docstring).
            limits: Execution settings; defaults sized for a free Colab runtime.
            retry: Retry policy for transient failures.
            task_libs: ``task_id -> libs``; used to tell environment problems from model errors.
            details_max_chars: Truncation length of each traceback in the results.
            client_factory: ``(endpoint, request_timeout) -> client`` (tests pass a fake).
            file_wrapper: Wraps the samples path for upload (``gradio_client.handle_file``).
            sleep: Sleep function used for back-off (tests pass a fake).
            clock: Monotonic clock used for the retry budget (tests pass a fake).
            rng: Random source for back-off jitter.
        """
        if not endpoint:
            raise ValueError("A Gradio evaluator endpoint is required")
        for name in _URL_LOGGING_LIBRARIES:
            library_logger = logging.getLogger(name)
            library_logger.setLevel(max(library_logger.getEffectiveLevel(), logging.WARNING))
        if details_max_chars <= 0:
            raise ValueError(f"details_max_chars must be positive, got {details_max_chars}")
        self._endpoint = endpoint
        self.subset = SUBSET
        self.limits = limits or EvaluatorLimits()
        self.retry = retry or RetryPolicy()
        self.task_libs = {task_id: tuple(libs) for task_id, libs in (task_libs or {}).items()}
        self.details_max_chars = details_max_chars
        self._client_factory = client_factory or _default_client_factory
        self._file_wrapper = file_wrapper or _default_file_wrapper
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._client: Any = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(subset={self.subset!r}, endpoint=<redacted>)"

    def preflight(self, task_ids: Sequence[str]) -> PreflightResult:
        """Checks the canonical solutions of ``task_ids`` in the evaluator's environment.

        Runs before any generation: a wrong or dead endpoint fails here, and tasks whose canonical
        solution fails (usually a missing library) can be excluded from the run.

        Raises:
            ValueError: If ``task_ids`` is empty.
            EvaluatorError: If the evaluator rejects the request or answers malformed.
            EvaluatorUnavailableError: If the evaluator stays unreachable.
        """
        requested = sorted({full_task_id(task_id) for task_id in task_ids}, key=task_sort_key)
        if not requested:
            raise ValueError("The pre-flight needs at least one task")
        _, pass_at_k = self._evaluate([], requested, check_gt_only=True)
        failed_reported = pass_at_k.get("failed_tasks")
        if not isinstance(failed_reported, (list, tuple)):
            raise EvaluatorError("The evaluator's ground-truth check returned no 'failed_tasks' list")
        requested_set = set(requested)
        failed = sorted(
            {full_task_id(str(task_id)) for task_id in failed_reported} & requested_set, key=task_sort_key
        )
        gt_pass_rate = 1.0 - len(failed) / len(requested)
        logger.info(
            "Ground-truth pre-flight: %d/%d canonical solutions pass (%.1f%%)",
            len(requested) - len(failed), len(requested), 100 * gt_pass_rate,
        )
        return PreflightResult(gt_pass_rate=gt_pass_rate, failed_tasks=tuple(failed))

    def test_batch(self, samples: Sequence[Sample]) -> Dict[str, TestResult]:
        """Runs BigCodeBench's tests on one solution per task.

        Returns:
            Results keyed by task id; a task the evaluator did not report is ``missing``.

        Raises:
            ValueError: If a task appears more than once.
            EvaluatorError: If the evaluator rejects the request or answers malformed.
            EvaluatorUnavailableError: If the evaluator stays unreachable.
        """
        if not samples:
            return {}
        normalized = [Sample(full_task_id(sample.task_id), sample.code) for sample in samples]
        task_ids = [sample.task_id for sample in normalized]
        duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"Each task may appear once per batch; duplicated: {duplicates}")

        started = self._clock()
        results, _ = self._evaluate(normalized, task_ids, check_gt_only=False)
        eval_map = results.get("eval")
        if not isinstance(eval_map, Mapping):
            raise EvaluatorError("The evaluator's response has no 'eval' results")
        by_task = {full_task_id(str(task_id)): entries for task_id, entries in eval_map.items()}

        outcomes = {task_id: self._parse_entry(task_id, by_task.get(task_id)) for task_id in task_ids}
        missing = [task_id for task_id, outcome in outcomes.items() if outcome.status == MISSING_STATUS]
        if missing:
            logger.warning("The evaluator returned no result for %d tasks: %s", len(missing), missing[:10])
        passed = sum(outcome.success for outcome in outcomes.values())
        logger.info(
            "Tested %d solutions in %.1fs: %d passed, %d failed",
            len(outcomes), self._clock() - started, passed, len(outcomes) - passed,
        )
        return outcomes

    def _parse_entry(self, task_id: str, entries: Any) -> TestResult:
        """Turns the evaluator's entries for one task into a :class:`TestResult`."""
        if not entries or not isinstance(entries, (list, tuple)) or not isinstance(entries[0], Mapping):
            return TestResult(success=False, status=MISSING_STATUS, error_class=MISSING_STATUS)
        entry = entries[0]
        status = str(entry.get("status"))
        raw_details = entry.get("details") or {}
        if not isinstance(raw_details, Mapping):
            raw_details = {"ALL": str(raw_details)}
        details = {str(name): self._truncate(str(trace)) for name, trace in raw_details.items()}
        if status == PASS_STATUS:
            return TestResult(success=True, status=status, details=details)
        error_class = classify_failure(status, raw_details, self.task_libs.get(task_id, ()))
        return TestResult(success=False, status=status, details=details, error_class=error_class)

    def _truncate(self, text: str) -> str:
        """Shortens a traceback to ``details_max_chars``, keeping its end (where the error is)."""
        if len(text) <= self.details_max_chars:
            return text
        return "[...truncated...]\n" + text[-self.details_max_chars:]

    def _evaluate(
        self, samples: Sequence[Sample], task_ids: Sequence[str], check_gt_only: bool
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Writes the samples file, calls ``/predict`` (with retries) and decodes the response."""
        handle, path = tempfile.mkstemp(prefix="evaloop_bigcodebench_", suffix=".jsonl")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as samples_file:
                for sample in samples:
                    samples_file.write(json.dumps({"task_id": sample.task_id, "solution": sample.code}) + "\n")
            kwargs = {
                "split": "complete",
                "subset": self.subset,
                "samples": self._file_wrapper(path),
                "pass_k": "1",
                "parallel": self.limits.parallel,
                "min_time_limit": self.limits.min_time_limit,
                "max_as_limit": self.limits.max_as_limit,
                "max_data_limit": self.limits.max_data_limit,
                "max_stack_limit": self.limits.max_stack_limit,
                "calibrated": True,  # always on: the server prepends code_prompt to each solution
                "check_gt_only": check_gt_only,
                "no_gt": not check_gt_only,
                "selective_evaluate": ",".join(bare_task_id(task_id) for task_id in task_ids),
                "api_name": "/predict",
            }
            response = self._predict_with_retries(kwargs)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise EvaluatorError("The evaluator's response is not a (results, pass_at_k) pair")
        results, pass_at_k = (_decode_json(part) for part in response)
        if not isinstance(results, dict) or not isinstance(pass_at_k, dict):
            raise EvaluatorError("The evaluator's response does not hold two JSON objects")
        return results, pass_at_k

    def _predict_with_retries(self, kwargs: Dict[str, Any]) -> Any:
        """Calls ``/predict``, retrying transient failures per :attr:`retry`."""
        started = self._clock()
        attempt = 0
        while True:
            attempt += 1
            try:
                if self._client is None:
                    self._client = self._client_factory(self._endpoint, self.retry.request_timeout)
                return self._client.predict(**kwargs)
            except (Exception, asyncio.CancelledError) as exc:
                # Errors are re-raised "from None": httpx messages and chained tracebacks can
                # contain the endpoint URL, which must not end up in logs.
                message = self._redact(f"{type(exc).__name__}: {exc}")
                if not is_retryable(exc):
                    raise EvaluatorError(f"The evaluator rejected the request: {message}") from None
                self._client = None  # reconnect on the next attempt
                delay = self._backoff(attempt)
                elapsed = self._clock() - started
                if attempt >= self.retry.attempts or elapsed + delay > self.retry.max_wait:
                    raise EvaluatorUnavailableError(
                        f"The evaluator is unreachable after {attempt} attempts over {elapsed:.0f}s "
                        f"(last error: {message})",
                        attempts=attempt,
                    ) from None
                logger.warning(
                    "Evaluator request failed (attempt %d/%d: %s); retrying in %.0fs",
                    attempt, self.retry.attempts, message, delay,
                )
                self._sleep(delay)

    def _backoff(self, attempt: int) -> float:
        """Exponential back-off with "equal jitter": half fixed, half random."""
        delay = min(self.retry.max_delay, self.retry.base_delay * 2 ** (attempt - 1))
        return delay / 2 + self._rng.uniform(0, delay / 2)

    def _redact(self, text: str) -> str:
        """Removes the endpoint URL (and its host) from a message."""
        redacted = text.replace(self._endpoint.rstrip("/"), "<endpoint>")
        host = re.sub(r"^[a-z]+://", "", self._endpoint).split("/")[0]
        return redacted.replace(host, "<endpoint>") if host else redacted


def _decode_json(value: Any) -> Any:
    """Decodes a ``gr.JSON`` output, which gradio_client returns as an object, a JSON string or
    the path of a JSON file, depending on its version."""
    if isinstance(value, str):
        if os.path.isfile(value):
            with open(value, "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _default_client_factory(endpoint: str, request_timeout: float) -> Any:
    """Creates a ``gradio_client.Client``, with the request timeout where the version supports it."""
    try:
        from gradio_client import Client
    except ImportError as err:
        raise ImportError(
            "The BigCodeBench Gradio evaluator needs the 'gradio_client' package: "
            "pip install -e '.[bigcodebench]' (see the README's BigCodeBench section)"
        ) from err
    logger.debug("Connecting to the BigCodeBench evaluator at %s", endpoint)
    try:
        return Client(endpoint, verbose=False, httpx_kwargs={"timeout": request_timeout})
    except TypeError:  # older gradio_client without verbose/httpx_kwargs
        return Client(endpoint)


def _default_file_wrapper(path: str) -> Any:
    """Wraps a file for upload with ``gradio_client.handle_file`` (older versions take the path)."""
    try:
        from gradio_client import handle_file
    except ImportError:
        return path
    return handle_file(path)
