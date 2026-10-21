#!/usr/bin/env python3
"""
Fire-based CLI interface for EvaLoop evaluation framework.
"""

import os
from typing import List, Optional, Dict, Any, Sequence, Union
from pathlib import Path

import fire

# The evaluator stack (torch, evalplus, ...) is imported lazily inside
# evaluate() so that --help, list_models, and validate_setup work even
# before the heavy dependencies are installed.
from .utils.logging_utils import setup_logging
from .utils.validation import SystemValidator
from .models.registry import ModelRegistry
from evaloop.analysis.metrics import DEFAULT_METRICS
from evaloop.analysis.similarity import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROMPT,
    JudgeConfig,
)


class EvaLoopCLI:
    """
    EvaLoop: LLM Robustness Evaluation Framework
    
    EvaLoop evaluates the robustness of Large Language Models through iterative 
    dual-task cycles. The framework measures how many cycles an LLM can execute 
    before generating code that fails functional testing.
    
    Available Commands:
        evaluate        Run code generation and summarization evaluation
        analyze         Analyze evaluation results and generate reports
        compare         Compare BigCodeBench runs with their MBPP Plus baseline
        inspect sample  Draw failed loops for manual inspection
        summary        Write summary.csv (one row per run, baselines included)
        list_models     List all available pre-configured models
        validate_setup  Validate EvaLoop setup and environment
    
    Quick Start Examples:
        # Basic evaluation with OpenAI model
        evaloop evaluate --model "gpt-4"
        
        # Large model with multi-GPU VLLM
        evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
                         --gpu_ids "0,1,2,3" --tensor_parallel_size 4
        
        # Custom experiment settings
        evaloop evaluate --model "deepseek-coder-33b" \\
                         --max_cycles 15 --seed 42 \\
                         --experiment_name "my_experiment"
        
        # Analyze results
        evaloop analyze --results_path "results/experiment_results.json"
    
    For detailed help on any command, use:
        evaloop COMMAND --help
        
    Environment Setup:
        export OPENAI_API_KEY="your-api-key"  # For OpenAI models
        
    More information: https://github.com/evalooop/EvaLooop
    """

    def __init__(self):
        """Initialize the EvaLoop CLI."""
        self.logger = None
        # Command group: `evaloop inspect sample ...`
        self.inspect = InspectCommands()

    def evaluate(
        self,
        model: str,
        dataset: str = "mbpp_plus",
        max_cycles: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
        output_dir: str = "results/",
        log_level: str = "INFO",
        experiment_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        max_new_tokens: int = 1024,
        trust_remote_code: bool = True,
        batch_size: int = 1,
        seed: Optional[int] = None,
        use_beam_search: bool = False,
        max_batch_size: int = 128,
        auto_batch_size: bool = True,
        preferred_batch_size: int = 64,
        is_chat_model: bool = True,
        system_prompt: str = "You are a helpful programming assistant.",
        bcb_trial: bool = False,
        bcb_trial_seed: int = 0,
        bcb_parallel: int = 2,
        resume_from: Optional[str] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run code generation and summarization evaluation experiment.
        
        This method performs iterative cycles of code generation and summarization:
        1. Start with a natural language prompt
        2. Generate code from the prompt
        3. Test the generated code for functionality
        4. Summarize the code back to natural language
        5. Repeat until code fails testing or max cycles reached

        Args:
            model: Single model name/path to evaluate.
                   Examples: "gpt-4", "Qwen/Qwen2.5-Coder-32B-Instruct"
            dataset: Dataset to use for evaluation: "mbpp_plus" or "bigcodebench" (the
                     standard loop on BigCodeBench-Hard, Complete split, tested by a BigCodeBench Gradio
                     evaluator whose URL is read from $EVALOOP_GRADIO_ENDPOINT; see
                     extension-README.md).
            max_cycles: Maximum number of evaluation cycles to run.
            temperature: Temperature for model generation (0.0 for greedy decoding).
            top_p: Top-p (nucleus) sampling parameter.
            output_dir: Directory to save results.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            experiment_name: Name for the experiment (auto-generated if None).
            openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var).
            gpu_ids: Comma-separated GPU IDs for VLLM models (e.g., "0,1,2,3").
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: GPU memory utilization ratio for VLLM.
            max_model_len: Maximum model context length.
            max_new_tokens: Maximum number of new tokens to generate.
            trust_remote_code: Whether to trust remote code for model loading.
            batch_size: Batch size for evaluation.
            seed: Random seed for reproducible generation.
            use_beam_search: Whether to use beam search for generation.
            max_batch_size: Maximum batch size for VLLM inference.
            auto_batch_size: Whether to automatically adjust batch size.
            preferred_batch_size: Preferred batch size when auto-batching.
            is_chat_model: Whether the model is a chat/instruction-tuned model.
            system_prompt: System prompt for chat models.
            bcb_trial: Run the 10-task trial (full protocol, not reportable) before the real run.
            bcb_trial_seed: Seed used to pick the trial tasks.
            bcb_parallel: Evaluator worker processes (default sized for free Colab).
            resume_from: BigCodeBench checkpoint to resume (after an interruption).

        Returns:
            Dictionary containing evaluation results with successful cycles for each task.
            
        Examples:
            # Basic evaluation with OpenAI model
            evaloop evaluate --model "gpt-4" --dataset "mbpp_plus"
            
            # Large model evaluation with VLLM and multi-GPU setup
            evaloop evaluate --model "Qwen/Qwen2.5-Coder-32B-Instruct" \\
                             --gpu_ids "0,1,2,3" --tensor_parallel_size 4 \\
                             --max_batch_size 128 --preferred_batch_size 64
                             
            # Custom experiment with specific generation settings
            evaloop evaluate --model "deepseek-coder-33b" \\
                             --max_cycles 15 --temperature 0.2 \\
                             --max_new_tokens 2048 --seed 42 \\
                             --system_prompt "You are an expert Python programmer." \\
                             --experiment_name "deepseek_robust_test"
        """
        
        if dataset not in ("mbpp_plus", "bigcodebench"):
            raise ValueError(
                f"Unsupported dataset: '{dataset}'. Supported: 'mbpp_plus', 'bigcodebench'."
            )

        bcb_config = None
        if dataset == "bigcodebench":
            from .core.bigcodebench_config import BigCodeBenchConfig

            bcb_config = BigCodeBenchConfig(
                trial=bcb_trial,
                trial_seed=bcb_trial_seed,
                parallel=bcb_parallel,
                resume_from=resume_from,
            )
        elif bcb_trial or resume_from:
            raise ValueError("BigCodeBench options (--bcb_trial, --resume_from) need --dataset bigcodebench")

        from .core.config import EvaluationConfig

        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info("Starting EvaLoop evaluation experiment")

        # Log model to evaluate
        self.logger.info(f"Model to evaluate: {model.strip()}")

        # Parse GPU IDs if provided
        gpu_id_list = None
        if gpu_ids:
            # Handle both string and tuple inputs from Fire
            if isinstance(gpu_ids, str):
                gpu_id_list = [int(gpu.strip()) for gpu in gpu_ids.split(",")]
            elif isinstance(gpu_ids, (list, tuple)):
                gpu_id_list = [int(gpu) for gpu in gpu_ids]
            else:
                gpu_id_list = [int(gpu_ids)]  # Single GPU ID

        # Setup API key
        # Anthropic API key loaded in models/anthropic_llm.py 
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # Create configuration
        config = EvaluationConfig(
            model=model.strip(),
            dataset=dataset,
            max_cycles=max_cycles,
            temperature=temperature,
            top_p=top_p,
            output_dir=output_dir,
            experiment_name=experiment_name,
            gpu_ids=gpu_id_list,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
            seed=seed,
            use_beam_search=use_beam_search,
            max_batch_size=max_batch_size,
            auto_batch_size=auto_batch_size,
            preferred_batch_size=preferred_batch_size,
            is_chat_model=is_chat_model,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key,
        )
        # Log model type detection results
        model_config = config.get_model_config()
        model_type = model_config["type"]
        self.logger.info(f"Model '{model}' → Using {model_type.upper()} inference engine")
        
        if model_type == "vllm":
            self.logger.info(f"  VLLM config: {model_config['tensor_parallel_size']} GPUs, "
                           f"batch_size={model_config['batch_config']['preferred_batch_size']}")
        elif model_type == "huggingface":
            self.logger.info(f"  HuggingFace config: device={model_config['device']}")
        elif model_type in ["openai", "anthropic"]:
            self.logger.info(f"  API config: model_id={model_config['model_id']}")

        # Create and run evaluator
        if bcb_config is not None:
            from .core.bigcodebench_evaluator import BigCodeBenchEvaluator

            BigCodeBenchEvaluator(config, bcb_config).run()
            # Return nothing: Fire would print the whole results dict to the terminal.
            # The evaluator has already logged where the results file was saved.
            self.logger.info("Evaluation completed successfully")
            return None
        else:
            from .core.evaluator import EvaLoopEvaluator

            evaluator = EvaLoopEvaluator(config)
            results = evaluator.run_code_generation_summarization()

        self.logger.info("Evaluation completed successfully")
        return results


    def analyze(
        self,
        results_path: str,
        metrics: str = ",".join(DEFAULT_METRICS),
        generate_plots: bool = True,
        output_dir: Optional[str] = None,
        max_cycles: Optional[int] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_api_key: Optional[str] = None,
        judge_api_key_file: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
        judge_temperature: float = 0.0,
        judge_system_prompt: Optional[str] = None,
        judge_prompt_file: Optional[str] = None,
        judge_delay: float = 0.0,
        similarity_path: Optional[str] = None,
        rejudge_failed: bool = True,
        log_level: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Analyze evaluation results and generate reports.

        Computes the EvaLooop robustness metrics and generates visualization plots. The
        headline metric ASL weights the quadratic loop score with the semantic similarity of
        each task's failure boundary, judged by an LLM. Judge scores are cached in a sidecar
        next to the results file, so reruns only query the judge for missing pairs.

        Args:
            results_path: Path to the evaluation results JSON file.
            metrics: Comma-separated metrics to compute (ASL,ASL_pow,ASL_bias,pass@1,pass_drop).
            generate_plots: Whether to generate visualization plots.
            output_dir: Directory to save analysis (defaults to results directory).
            max_cycles: Loop budget M of the experiment (defaults to the value recorded in the
                results file, then the value implied by tasks flagged max_cycles_reached, then 10).
            judge_model: Model id of the similarity judge.
            judge_api_key: API key of the judge endpoint (defaults to OPENAI_API_KEY).
            judge_api_key_file: File containing the judge API key (alternative to judge_api_key).
            judge_base_url: OpenAI-compatible endpoint of the judge (e.g. a local vLLM server).
            judge_max_tokens: Token budget of the judge answer (raise it for reasoning models).
            judge_temperature: Sampling temperature of the judge.
            judge_system_prompt: Optional system message for the judge (none by default).
            judge_prompt_file: File with a custom judge prompt template using the placeholders
                {prompt1}, {code1}, {prompt2} and {code2}.
            judge_delay: Seconds to wait between judge calls.
            similarity_path: Similarity sidecar to read and extend (defaults to
                <results stem>_<judge>_similarity_scores.json next to the results file).
            rejudge_failed: Whether to retry pairs whose cached judge call failed.
            log_level: Logging level for analysis output.

        Usage:
            evaloop analyze --results_path "results/my_experiment_results.json"
            evaloop analyze --results_path "results/exp.json" --judge_api_key_file api.key
            evaloop analyze --results_path "results/exp.json" --metrics "ASL_pow,pass@1"
            evaloop analyze --results_path "results/exp.json" --judge_model "Qwen/Qwen2.5-72B-Instruct" \\
                            --judge_base_url "http://localhost:8000/v1"
        """
        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info(f"Analyzing results from: {results_path}")

        # Parse metrics list (Fire passes "a,b" as a tuple and "a,pass@1" as a string)
        metrics_list = _split_list_arg(metrics)

        # Determine output directory
        if output_dir is None:
            output_dir = str(Path(results_path).parent / "analysis")

        # Import analyzer
        from evaloop.analysis.analyzer import ResultAnalyzer

        judge_config = JudgeConfig(
            model=str(judge_model),
            api_key=_resolve_api_key(judge_api_key, judge_api_key_file),
            base_url=judge_base_url,
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
            system_prompt=judge_system_prompt,
            prompt_template=Path(judge_prompt_file).read_text() if judge_prompt_file else DEFAULT_JUDGE_PROMPT,
            request_delay=judge_delay,
        )

        # Create analyzer and run analysis
        analyzer = ResultAnalyzer(
            results_path,
            output_dir,
            judge_config=judge_config,
            similarity_path=similarity_path,
            max_cycles=max_cycles,
            rejudge_failed=rejudge_failed,
        )
        analysis_results = analyzer.analyze(
            metrics=metrics_list,
            generate_plots=generate_plots,
        )

        self.logger.info(f"Analysis completed. Results saved to: {output_dir}")
        return analysis_results

    def compare(
        self,
        runs: Union[str, Sequence[str]],
        baseline_dir: str = "data/assignment2-data",
        output_dir: str = "results/compare",
        include_trial: bool = False,
        log_level: str = "INFO",
    ) -> Dict[str, str]:
        """
        Compare BigCodeBench runs with the same models' MBPP Plus baseline.

        Computes success rate, unweighted ASL, robustnessScore, pass_drop and survival curves for each
        run and for its baseline (recomputed from the data package's loops/), the changes between
        them, the models' baseline ranks, and the ranking among the evaluated models on each
        benchmark.
        Writes comparison.json and comparison.md (numbers and ids only) to output_dir and returns
        their paths.

        Args:
            runs: Comma-separated model_id=results_path entries; model_id is the package id of
                the run's baseline (e.g. "claude-sonnet-4.5=results/a.json,gpt-4.1-mini=results/b.json").
            baseline_dir: The unzipped assignment2-data package.
            output_dir: Where to write the report.
            include_trial: Accept trial runs (the report is then labelled as trial output).
            log_level: Logging level.

        Usage:
            evaloop compare --runs "claude-sonnet-4.5=results/x_bigcodebench_results.json,gpt-4.1-mini=results/y_bigcodebench_results.json"
        """
        self.logger = setup_logging(level=log_level)
        from evaloop.analysis import compare as comparison

        loaded = comparison.load_runs(runs, include_trial=include_trial)
        report = comparison.compare(loaded, baseline_dir)
        json_path, md_path = comparison.write_report(report, output_dir)
        self.logger.info(f"Comparison written to {json_path} and {md_path}")
        return {"json": str(json_path), "markdown": str(md_path)}

    def summary(
        self,
        runs: Union[str, Sequence[str]],
        baseline_dir: str = "data/assignment2-data",
        out: Optional[str] = None,
        include_trial: bool = False,
        log_level: str = "INFO",
    ) -> str:
        """
        Write summary.csv: one row per run, plus each model's standard-mbpp-plus baseline row.

        Header: model,configuration,n_tasks,success_rate,asl_score. asl_score is the unweighted
        ASL (the data package's aslScore); baseline rows are recomputed from loops/.

        Args:
            runs: Comma-separated model_id=results_path entries (see `evaloop compare`).
            baseline_dir: The unzipped assignment2-data package.
            out: Output path (default results/summary.csv, or results/summary_trial.csv when
                trial runs are included).
            include_trial: Accept trial runs (their configuration names end in -trial10).
            log_level: Logging level.

        Usage:
            evaloop summary --runs "claude-sonnet-4.5=results/x_bigcodebench_results.json,gpt-4.1-mini=results/y_bigcodebench_results.json"
        """
        self.logger = setup_logging(level=log_level)
        from evaloop.analysis import compare as comparison
        from evaloop.analysis import summary_csv

        loaded = comparison.load_runs(runs, include_trial=include_trial)
        if out is None:
            trial = any(run.trial for run in loaded)
            out = str(summary_csv.DEFAULT_TRIAL_OUT if trial else summary_csv.DEFAULT_OUT)
        path = summary_csv.write_summary(summary_csv.summary_rows(loaded, baseline_dir), out)
        self.logger.info(f"Summary written to {path}")
        return str(path)

    def list_models(self) -> List[str]:
        """
        List all available pre-configured models.
        
        Shows all models that can be used with the --model parameter.
        Includes both model names and full HuggingFace paths.
        
        Usage:
            evaloop list_models
        """
        registry = ModelRegistry()
        models = registry.list_available_models()

        print("Available Models:")
        print("================")
        for model in models:
            print(f"- {model}")

        return models

    def validate_setup(
        self,
        check_gpu: bool = True,
        check_api_keys: bool = True,
        log_level: str = "INFO",
    ) -> Dict[str, Any]:
        """
        Validate the EvaLoop setup and environment.
        
        Checks system requirements, dependencies, GPU availability, and API keys.
        Useful for troubleshooting installation and setup issues.
        
        Args:
            check_gpu: Whether to check GPU availability and CUDA setup.
            check_api_keys: Whether to validate API key configuration.
            log_level: Logging level for validation output.
        
        Usage:
            evaloop validate_setup
            evaloop validate_setup --check_gpu=False  # Skip GPU checks
        """
        # Setup logging
        self.logger = setup_logging(level=log_level)
        self.logger.info("Validating EvaLoop setup")


        validator = SystemValidator()
        results = validator.validate_all(
            check_gpu=check_gpu,
            check_api_keys=check_api_keys,
        )

        # Print results
        print("EvaLoop Setup Validation")
        print("========================")
        for check, status in results.items():
            status_symbol = "✓" if status["passed"] else "✗"
            print(f"{status_symbol} {check}: {status['message']}")

        return results


class InspectCommands:
    """Manual inspection of failed loops (`evaloop inspect sample`)."""

    def sample(
        self,
        runs: Union[str, Sequence[str]],
        n: int = 20,
        baseline_n: int = 10,
        seed: int = 1234,
        baseline_dir: str = "data/assignment2-data",
        output_dir: str = "results/inspection",
        overwrite: bool = False,
        include_trial: bool = False,
        log_level: str = "INFO",
    ) -> Dict[str, str]:
        """
        Draw a seeded sample of failed loops (and matching baseline loops) for manual inspection.

        Writes annotations.csv (pre-filled ids; fill in the categories by hand), sampling.json
        (how the sample was drawn) and worksheet.md (a LOCAL working file with each loop's
        prompts, code and summaries; never commit or submit it). Nothing is executed.

        Args:
            runs: Comma-separated model_id=results_path entries, one per model (see `evaloop compare`).
            n: Failed BigCodeBench loops to draw in total, split evenly across models (minimum 20).
            baseline_n: Failed baseline loops to draw per model.
            seed: Sampling seed (record it in the README).
            baseline_dir: The unzipped assignment2-data package.
            output_dir: Where to write the files.
            overwrite: Replace an existing annotations.csv (it may hold your annotations).
            include_trial: Accept trial runs (relaxes the 20-loop minimum; not reportable).
            log_level: Logging level.
        """
        logger = setup_logging(level=log_level)
        from evaloop.analysis import compare as comparison
        from evaloop.analysis import inspection

        loaded = comparison.load_runs(runs, include_trial=include_trial)
        drawn = inspection.sample(loaded, n=n, baseline_n=baseline_n, seed=seed, baseline_dir=baseline_dir)
        paths = inspection.write_outputs(drawn, output_dir, overwrite=overwrite)
        logger.info(f"Inspection sample written to {paths['annotations']} and {paths['sampling']}")
        logger.warning(f"{paths['worksheet']} contains data-package text: keep it local, never commit or submit it")
        return {name: str(path) for name, path in paths.items()}


def _split_list_arg(value: Union[str, Sequence[str]]) -> List[str]:
    """Normalize a comma-separated CLI argument that Fire may have parsed into a tuple."""
    items = value.split(",") if isinstance(value, str) else [str(item) for item in value]
    return [item.strip() for item in items if item.strip()]


def _resolve_api_key(api_key: Optional[str], api_key_file: Optional[str]) -> Optional[str]:
    """Return the API key given directly or read from a file (None falls back to the environment)."""
    if api_key and api_key_file:
        raise ValueError("Pass either an API key or an API key file, not both")
    if not api_key_file:
        return api_key
    key = Path(api_key_file).read_text().strip()
    if not key:
        raise ValueError(f"API key file '{api_key_file}' is empty")
    return key


def main():
    """Main entry point for the CLI."""
    import sys
    
    # Add version command support
    if len(sys.argv) > 1 and sys.argv[1] in ['--version', '-v', 'version']:
        from evaloop import __version__
        print(f"EvaLoop {__version__}")
        return
    
    try:
        fire.Fire(EvaLoopCLI)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nFor help, run: evaloop --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
