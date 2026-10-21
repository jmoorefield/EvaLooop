# BigCodeBench extension for EvaLoop

This extension adds a second benchmark with executable tests, BigCodeBench, to EvaLoop. It runs EvaLoop's standard generation ↔ summarization loop on BigCodeBench-Hard (Complete split), tests every generated solution with BigCodeBench's own test suites, and compares each model's robustness with its MBPP Plus baseline from the course data package.

- **Code:** the `extension` branch of the public fork, <https://github.com/jmoorefield/EvaLooop/tree/extension>
- **Upstream base commit of the patch:** `ce22fe2f80c8f36da85cc8a0cca49c2bcc8b1cb9` (upstream `dev`, "Merge pull request #3 from evalooop/feature/semantic-asl")

## Contents

1. [Extension summary](#1-what-the-extension-does)
2. [Tools, dependencies and models](#2-tools-dependencies-and-models)
3. [Changes to the existing framework](#3-changes-to-the-existing-framework)
4. [Setup](#4-setup)
5. [Running the experiment](#5-running-the-experiment)
6. [Reproducing each reported number](#6-reproducing-each-reported-number)
7. [Safety and data handling](#8-safety-and-data-handling)

## 1. Extension summary

Each task goes through up to 10 cycles of the standard loop, using the same prompts as the MBPP Plus baseline:

1. **Generate:** the model writes code for the task.
2. **Test:** the code runs against the task's BigCodeBench test suite.
3. **Summarize:** if it passes, the model summarizes its own code. The summary becomes the next cycle's prompt.

- **Prompts:** the generation and summarization templates are copied verbatim from the course package's `prompts/task_prompts.md`.
- **Anchor:** MBPP Plus keeps the task's assert line in every prompt. The BigCodeBench equivalent is the task's `code_prompt`, its imports and `def task_func(...)` signature. Each next prompt is that `code_prompt` followed by a docstring holding the summary.
- **Where code runs:** generated code is never run where EvaLoop runs. EvaLoop sends each cycle's code in one batch to a **BigCodeBench evaluator on a free Google Colab CPU runtime**, reached through a temporary Gradio share URL, and reads back the results.
- **Analysis:** `evaloop compare` and `evaloop summary` compute success rate and unweighted ASL for each run and for the same model's MBPP Plus baseline, using the same code for both. `evaloop inspect sample` draws a seeded sample of failed loops for manual classification.

## 2. Tools, dependencies and models

### Tools

| Tool | Used for |
|---|---|
| EvaLoop (this repo) | The loop, task loading, testing calls, checkpoints, analysis |
| Google Colab, free CPU runtime | Hosts the BigCodeBench evaluator ([notebooks/bigcodebench_evaluator.ipynb](notebooks/bigcodebench_evaluator.ipynb)) |
| Gradio share link | Connects EvaLoop to the evaluator |
| BigCodeBench evaluator app | Adapted from the [`bigcode/bigcodebench-evaluator`](https://huggingface.co/spaces/bigcode/bigcodebench-evaluator) Space (Apache-2.0, commit `8653fc7f`); the changes are listed in the app's header in the notebook |
| Hugging Face Hub | The BigCodeBench-Hard dataset, `bigcode/bigcodebench-hard`, split `v0.1.4` |
| Anthropic and OpenAI APIs | Model inference |

### Dependencies

**Where EvaLoop runs (Python ≥ 3.10):**

| Package | Version | Why |
|---|---|---|
| `evaloop` core dependencies | as in `pyproject.toml` | Includes `datasets` (task loading) and `openai` |
| `gradio_client` | `2.7.2` | Talks to the evaluator |
| `anthropic` | `1.11.0` | Claude client |

Both pins are in the `bigcodebench` extra. BigCodeBench itself is not installed here: the tasks are read from the Hugging Face Hub with `datasets`.

**In the Colab evaluator** (installed by the notebook into a separate Python 3.10 environment):

| Package | Version |
|---|---|
| Python | 3.10 (via `uv`), as in the official evaluator image |
| `bigcodebench` | `0.2.5` (its loader serves dataset `v0.1.4`) |
| BigCodeBench evaluation requirements | `Requirements/requirements-eval.txt` at tag `v0.2.5` |
| `gradio` | `6.29.1` (requires `gradio_client` 2.7.2) |
| `datasets` | `4.5.0` |
| `setuptools` | `69.5.1` |

The notebook saves the evaluator's full package list to `/content/evaluator-requirements.txt`.

### Models

| | Claude Sonnet 4.5 | GPT-4.1 mini |
|---|---|---|
| Package `model_id` (its baseline) | `claude-sonnet-4.5` | `gpt-4.1-mini` |
| Pinned snapshot (`--model`) | `claude-sonnet-4-5-20250929` | `gpt-4.1-mini-2025-04-14` |
| Served by | Anthropic Messages API (`evaloop/models/anthropic_llm.py`) | OpenAI Chat Completions API (existing `OpenAILLM`) |
| Temperature | 0.0, sent via `extra_body` (the `anthropic` 1.x SDK has no `temperature` argument) | 0.0 |
| top-p | not sent (API default 1.0) | 1.0 |
| Max new tokens | 1024 | 1024 |
| System prompt | `You are a helpful AI assistant.` | `You are a helpful AI assistant.` |

Both models are hosted APIs: no GPU is needed anywhere.


## 3. Changes to the existing framework

All new functionality is in new files. Three existing files were changed:

| File | Change | Purpose |
|---|---|---|
| `evaloop/cli.py` | `evaluate` accepts `--dataset bigcodebench` and dispatches to the new evaluator, with `--bcb_trial`, `--bcb_trial_seed`, `--bcb_parallel` and `--resume_from` flags (the evaluator URL comes from `$EVALOOP_GRADIO_ENDPOINT`). New commands `compare`, `summary` and `inspect sample`. | Commands to run the experiment and analyze the results |
| `pyproject.toml` | New optional extra `bigcodebench` (`gradio_client==2.7.2`, `anthropic==1.11.0`) | Packaging metadata |
| `README.md` | One line pointing to `extension-README.md` | Pointer leads readers of the fork to this file for instructions |

**New files:**
- `evaloop/datasets/`: the loader, trial selection and standard prompts.
- `evaloop/tasks/bigcodebench_summarization.py`: the summarization task that applies the anchor.
- `evaloop/core/bigcodebench_config.py` and `evaloop/core/bigcodebench_evaluator.py`: the run configuration and the loop runner.
- `evaloop/evaluation/gradio_tester.py`: the client for the evaluator.
- `evaloop/models/anthropic_llm.py`: the Claude client.
- `evaloop/baselines/`: the importer for the data package.
- `evaloop/analysis/compare.py`, `summary_csv.py` and `inspection.py`: the analysis commands.
- `notebooks/bigcodebench_evaluator.ipynb`: the Colab evaluator.

**Two holes in the existing framework are worked around in the new code instead of being fixed in place:**
1. `EvaluationConfig` passes API models no system prompt and a 4096-token budget. The BigCodeBench evaluator sets the protocol's system prompt and caps the budget at 1024 new tokens.
2. `OpenAILLM` returns API failures as response text instead of raising an error. The evaluator detects those strings and stops the run with a checkpoint, so an error message is never tested as code.

## 4. Setup

```bash
git clone https://github.com/jmoorefield/EvaLooop
cd EvaLooop
git checkout extension
pip install -e ".[bigcodebench]"          # Python 3.10 or newer
```

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

**The course data package** (`assignment2-data`, the MBPP Plus baseline) is not included in this repo or the submission, and must be obtained from the course. Unzip it to `data/assignment2-data/`, which is gitignored. Only `compare`, `summary` and `inspect sample` need it.

### 4.1 Run the evaluator on Colab

1. Open [notebooks/bigcodebench_evaluator.ipynb](notebooks/bigcodebench_evaluator.ipynb) in Colab on a **fresh free CPU runtime**. Don't mount Drive, don't add secrets, don't sign in to anything.
2. Run steps 1–5 in order. Installing takes about 25 minutes. Step 5 checks every Hard task's reference solution, which takes about 8 minutes, and then prints the share URL.
3. Check step 5's output: `gt_pass_rate` should be at least 0.95, with no missing modules listed. This is to ensure that there are minimal tasks who were excluded because their reference solution will not run in the evaluator. 
4. Start step 6, which keeps the session alive, and leave it running.
5. In your terminal, set the URL without writing it to any file:
   ```bash
   export EVALOOP_GRADIO_ENDPOINT='https://<id>.gradio.live'
   ```
6. When finished, disconnect and delete runtime. This stops the evaluator and makes the URL stop working.

## 5. Running the experiment

Run the commands from the repository root, one model after the other.

```bash
evaloop evaluate --dataset bigcodebench \
  --model claude-sonnet-4-5-20250929 --experiment_name bcb_hard_claude_sonnet_4_5
evaloop evaluate --dataset bigcodebench \
  --model gpt-4.1-mini-2025-04-14 --experiment_name bcb_hard_gpt_4_1_mini
```

All protocol settings are defaults: BigCodeBench-Hard, Complete split, temperature 0, top-p 1, 1024 new tokens, 10 cycles, calibrated mode, and tasks whose reference solution fails in the evaluator excluded before cycle 1. EvaLoop **refuses** any other protocol setting, so a results file always follows the standard loop.

**Main runs:**

Each run writes `results/<experiment_name>_bigcodebench_results.json`. Every results file records:
- the task IDs and the excluded ones;
- the dataset version and hash;
- the protocol, including the system prompt actually sent;
- the model snapshot and provider;
- per-cycle call counts and timings.

**If a run stops** (an unreachable evaluator, or an API error), its progress is in `results/<experiment_name>.checkpoint.json` and the run prints the resume command. Start a new evaluator if needed, set the new URL, and resume:

```bash
evaloop evaluate --dataset bigcodebench --model <same model> \
  --resume_from results/<experiment_name>.checkpoint.json
```

A resumed run is the same run. Code that was already generated is tested without being generated again.

**If you want to run a trial first:** (10 tasks, full protocol, never reported):

```bash
evaloop evaluate --dataset bigcodebench --bcb_trial \
  --model claude-sonnet-4-5-20250929 --experiment_name trial_claude_sonnet_4_5
evaloop evaluate --dataset bigcodebench --bcb_trial \
  --model gpt-4.1-mini-2025-04-14 --experiment_name trial_gpt_4_1_mini
```

## 6. Reproducing each reported number

**The run files** referred to below as `RUNS`:

```bash
RUNS="claude-sonnet-4.5=results/bcb_hard_claude_sonnet_4_5_bigcodebench_results.json,gpt-4.1-mini=results/bcb_hard_gpt_4_1_mini_bigcodebench_results.json"
```

### 6.1 `summary.csv`

```bash
evaloop summary --runs "$RUNS"            # writes results/summary.csv
```

| model | configuration | n_tasks | success_rate | asl_score | Produced by |
|---|---|---|---|---|---|
| claude-sonnet-4.5 | standard-mbpp-plus | 378 | 0.812 | 7.626 | `evaloop summary`, recomputed from `data/assignment2-data/loops/claude-sonnet-4.5.json` |
| claude-sonnet-4.5 | bigcodebench-hard-complete | 147 | 0.415 | 1.322 | `evaloop summary` on `results/bcb_hard_claude_sonnet_4_5_bigcodebench_results.json`, from the Claude main-run command in §4 |
| gpt-4.1-mini | standard-mbpp-plus | 378 | 0.778 | 7.291 | `evaloop summary`, recomputed from `data/assignment2-data/loops/gpt-4.1-mini.json` |
| gpt-4.1-mini | bigcodebench-hard-complete | 147 | 0.388 | 1.273 | `evaloop summary` on `results/bcb_hard_gpt_4_1_mini_bigcodebench_results.json`, from the GPT main-run command in §4 |

- **`success_rate`** = pass@1 / n_tasks.
- **`asl_score`** = unweighted ASL, Σᵢ i²·nᵢ / (n·10), where nᵢ is the number of tasks that sustained exactly i cycles. This equals the data package's `aslScore` and differs from the leaderboard's judge-weighted ASL column. No LLM judge is used anywhere in this extension.
- **Baseline rows** are recomputed from the package's per-task records, not copied from `leaderboard.json`. For both models they match the leaderboard's `successRate` and `aslScore` exactly.
- **`n_tasks`** for BigCodeBench is 148 minus the 1 task excluded for a failing reference solution (BigCodeBench/590), which is also listed in the results JSON file, in the field `excluded_gt_failures`.


### 6.2 Manual inspection of failed loops

Helper command to sample 20 failed loops for the assignment report. 

```bash
evaloop inspect sample --runs "$RUNS" --n 20 --baseline_n 10 --seed 1234
```

- **The sample:** 20 failed BigCodeBench loops (10 per model) are drawn with seed `1234`, stratified by model and by the cycle where the loop failed (first, cycles 2–3, cycles 4–10). So are 10 baseline failed loops per model from the same models' MBPP+ records.
- **The outputs:**
  - `results/inspection/sampling.json` records the seed, strata, allocation and sampled IDs.
  - `results/inspection/annotations.csv` holds one row per loop, with fields `(source, model_id, task_id, failing_cycle, cycle_group, category, note)`. It comes blank and I filled it in myself to help me do the report analysis. 
  - `results/inspection/worksheet.md` is a local file that lays out each loop for easier manual inspection.

## 7. Safety and data handling

- **Model-generated code runs only in the Colab evaluator,** under BigCodeBench's process isolation and resource limits, as an unprivileged user. It never runs where EvaLoop runs.
- **The share URL is unauthenticated:** anyone who has it can send code to the evaluator until the runtime stops. Keep it in an environment variable only, and delete the runtime when done. EvaLoop never writes the URL to results or checkpoints, nor to its log at the default INFO level (only `--log_level DEBUG` would log it).
- **The course data package is read at runtime only.** None of its records are copied into this repo or the submission. Submitted outputs contain numbers and `(model_id, task_id, cycle)` references only.
