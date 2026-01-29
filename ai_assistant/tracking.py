"""
MLflow Tracking for AI Assistant.

This module provides observability and tracking capabilities for AI
operations using MLflow, enabling monitoring, cost tracking, and
experiment comparison.

Features:
- Automatic logging of all LLM calls
- Token usage and cost tracking
- Latency monitoring
- Prompt/response logging
- A/B testing between models
- Custom metrics and artifacts
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from functools import wraps
from datetime import datetime
from contextlib import contextmanager


@dataclass
class LLMCallMetrics:
    """
    Metrics for a single LLM call.

    Attributes:
        prompt: Input prompt (optionally truncated)
        response: LLM response (optionally truncated)
        model: Model name
        provider: Provider (gemini, claude)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: Response latency in milliseconds
        cost_usd: Estimated cost in USD
        timestamp: Call timestamp
        success: Whether the call succeeded
        error: Error message if failed
        metadata: Additional metadata
    """
    prompt: str
    response: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }


class AITracker:
    """
    MLflow-based tracker for AI operations.

    This class provides automatic tracking of LLM calls, including
    metrics, costs, and artifacts.

    Args:
        experiment_name: MLflow experiment name
        run_name: Optional run name
        tracking_uri: MLflow tracking URI
        log_prompts: Whether to log full prompts
        log_responses: Whether to log full responses
        max_text_length: Maximum text length to log
        auto_log: Whether to automatically start logging

    Example:
        >>> tracker = AITracker("ai_assistant_experiment")
        >>> with tracker.start_run("analysis_task"):
        ...     response = assistant.ask("What is Spark?")
        ...     tracker.log_call(prompt, response, model="claude")
    """

    # Token pricing (USD per 1K tokens)
    TOKEN_PRICING = {
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-2.0-flash-exp": {"input": 0.0001, "output": 0.0004},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    }

    def __init__(
        self,
        experiment_name: str = "ai_assistant",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        log_prompts: bool = True,
        log_responses: bool = True,
        max_text_length: int = 10000,
        auto_log: bool = False
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.max_text_length = max_text_length
        self.auto_log = auto_log

        self._mlflow = None
        self._active_run = None
        self._calls: List[LLMCallMetrics] = []

        # Aggregated metrics
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._total_calls = 0
        self._total_latency = 0.0

    def _get_mlflow(self):
        """Lazy load MLflow."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)

                # Set or create experiment
                mlflow.set_experiment(self.experiment_name)

            except ImportError:
                raise ImportError(
                    "mlflow package not installed. "
                    "Install with: pip install mlflow"
                )
        return self._mlflow

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start an MLflow run for tracking.

        Args:
            run_name: Name for the run
            tags: Optional tags for the run

        Yields:
            MLflow run object

        Example:
            >>> with tracker.start_run("data_analysis"):
            ...     response = assistant.ask("Analyze the data")
        """
        mlflow = self._get_mlflow()

        run_name = run_name or self.run_name or f"ai_run_{int(time.time())}"

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            self._active_run = run
            self._reset_aggregates()

            try:
                yield run
            finally:
                # Log final aggregated metrics
                self._log_aggregated_metrics()
                self._active_run = None

    def log_call(
        self,
        prompt: str,
        response: str,
        model: str,
        provider: str = "unknown",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMCallMetrics:
        """
        Log a single LLM call.

        Args:
            prompt: Input prompt
            response: LLM response
            model: Model name
            provider: Provider name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Response latency
            success: Whether call succeeded
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            LLMCallMetrics object
        """
        # Estimate tokens if not provided
        if input_tokens is None:
            input_tokens = len(prompt) // 4
        if output_tokens is None:
            output_tokens = len(response) // 4

        # Calculate cost
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        # Create metrics
        metrics = LLMCallMetrics(
            prompt=self._truncate(prompt) if self.log_prompts else "[not logged]",
            response=self._truncate(response) if self.log_responses else "[not logged]",
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms or 0.0,
            cost_usd=cost,
            success=success,
            error=error,
            metadata=metadata or {}
        )

        self._calls.append(metrics)

        # Update aggregates
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost
        self._total_calls += 1
        self._total_latency += (latency_ms or 0.0)

        # Log to MLflow if active run
        if self._active_run:
            self._log_call_to_mlflow(metrics)

        return metrics

    def _log_call_to_mlflow(self, metrics: LLMCallMetrics) -> None:
        """Log a call to MLflow."""
        mlflow = self._get_mlflow()

        # Log metrics
        step = self._total_calls
        mlflow.log_metric("input_tokens", metrics.input_tokens, step=step)
        mlflow.log_metric("output_tokens", metrics.output_tokens, step=step)
        mlflow.log_metric("latency_ms", metrics.latency_ms, step=step)
        mlflow.log_metric("cost_usd", metrics.cost_usd, step=step)
        mlflow.log_metric("success", 1 if metrics.success else 0, step=step)

        # Log cumulative metrics
        mlflow.log_metric("cumulative_input_tokens", self._total_input_tokens, step=step)
        mlflow.log_metric("cumulative_output_tokens", self._total_output_tokens, step=step)
        mlflow.log_metric("cumulative_cost_usd", self._total_cost, step=step)

    def _log_aggregated_metrics(self) -> None:
        """Log final aggregated metrics."""
        if not self._active_run:
            return

        mlflow = self._get_mlflow()

        # Summary metrics
        mlflow.log_metric("total_calls", self._total_calls)
        mlflow.log_metric("total_input_tokens", self._total_input_tokens)
        mlflow.log_metric("total_output_tokens", self._total_output_tokens)
        mlflow.log_metric("total_cost_usd", self._total_cost)

        if self._total_calls > 0:
            avg_latency = self._total_latency / self._total_calls
            mlflow.log_metric("avg_latency_ms", avg_latency)

            avg_cost = self._total_cost / self._total_calls
            mlflow.log_metric("avg_cost_per_call", avg_cost)

        # Log calls as artifact
        if self._calls:
            calls_data = [c.to_dict() for c in self._calls]
            calls_json = json.dumps(calls_data, indent=2, default=str)
            mlflow.log_text(calls_json, "llm_calls.json")

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for tokens."""
        pricing = self.TOKEN_PRICING.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def _truncate(self, text: str) -> str:
        """Truncate text to max length."""
        if len(text) <= self.max_text_length:
            return text
        return text[:self.max_text_length] + "...[truncated]"

    def _reset_aggregates(self) -> None:
        """Reset aggregated metrics."""
        self._calls.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._total_calls = 0
        self._total_latency = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked calls."""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": self._total_cost,
            "avg_latency_ms": (
                self._total_latency / self._total_calls
                if self._total_calls > 0 else 0
            ),
            "calls": [c.to_dict() for c in self._calls[-10:]]  # Last 10 calls
        }

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to active run."""
        if self._active_run:
            mlflow = self._get_mlflow()
            mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to active run."""
        if self._active_run:
            mlflow = self._get_mlflow()
            mlflow.log_artifact(local_path, artifact_path)

    def log_text(self, text: str, filename: str) -> None:
        """Log text as an artifact."""
        if self._active_run:
            mlflow = self._get_mlflow()
            mlflow.log_text(text, filename)


class TrackedAIClient:
    """
    Wrapper that adds tracking to any AI client.

    This wrapper automatically logs all LLM calls to the tracker.

    Args:
        client: The underlying AI client
        tracker: AITracker instance
        provider: Provider name for tracking

    Example:
        >>> tracker = AITracker("my_experiment")
        >>> tracked_client = TrackedAIClient(claude_client, tracker, "claude")
        >>>
        >>> with tracker.start_run("analysis"):
        ...     response = tracked_client.generate("Explain Spark")
    """

    def __init__(
        self,
        client: Any,
        tracker: AITracker,
        provider: str = "unknown"
    ):
        self.client = client
        self.tracker = tracker
        self.provider = provider

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response with tracking."""
        model_name = self._get_model_name()

        start_time = time.time()
        error = None
        response = ""

        try:
            response = self.client.generate(
                prompt,
                system_instruction=system_instruction,
                **kwargs
            )
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

            # Get token counts if available
            input_tokens = None
            output_tokens = None
            if hasattr(self.client, 'usage_stats'):
                stats = self.client.usage_stats
                input_tokens = stats.get('total_input_tokens', 0)
                output_tokens = stats.get('total_output_tokens', 0)

            self.tracker.log_call(
                prompt=prompt,
                response=response,
                model=model_name,
                provider=self.provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=error is None,
                error=error,
                metadata={"system_instruction": system_instruction is not None}
            )

        return response

    def chat(
        self,
        message: str,
        conversation_name: str = "default",
        system_instruction: Optional[str] = None
    ) -> str:
        """Chat with tracking."""
        model_name = self._get_model_name()

        start_time = time.time()
        error = None
        response = ""

        try:
            response = self.client.chat(
                message,
                conversation_name,
                system_instruction
            )
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

            self.tracker.log_call(
                prompt=f"[Chat:{conversation_name}] {message}",
                response=response,
                model=model_name,
                provider=self.provider,
                latency_ms=latency_ms,
                success=error is None,
                error=error,
                metadata={"conversation": conversation_name}
            )

        return response

    def _get_model_name(self) -> str:
        """Get model name from client."""
        if hasattr(self.client, 'model_config'):
            return self.client.model_config.name
        return "unknown"

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying client."""
        return getattr(self.client, name)


def track_llm_call(tracker: AITracker, provider: str = "unknown"):
    """
    Decorator to track LLM calls.

    Args:
        tracker: AITracker instance
        provider: Provider name

    Example:
        >>> @track_llm_call(tracker, "claude")
        ... def my_ai_function(prompt):
        ...     return client.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            prompt = args[0] if args else kwargs.get('prompt', '')
            error = None
            response = ""

            try:
                response = func(*args, **kwargs)
                return response
            except Exception as e:
                error = str(e)
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                tracker.log_call(
                    prompt=str(prompt),
                    response=str(response),
                    model="unknown",
                    provider=provider,
                    latency_ms=latency_ms,
                    success=error is None,
                    error=error
                )

        return wrapper
    return decorator


class ABExperiment:
    """
    A/B testing for model comparison.

    This class helps compare performance between different models
    or configurations.

    Args:
        tracker: AITracker instance
        experiment_name: Name for the A/B experiment

    Example:
        >>> ab = ABExperiment(tracker, "model_comparison")
        >>> ab.add_variant("claude", claude_client)
        >>> ab.add_variant("gemini", gemini_client)
        >>>
        >>> results = ab.run_comparison(
        ...     prompts=["Explain Spark", "What is Delta Lake?"],
        ...     evaluator=lambda r: len(r)  # Simple length metric
        ... )
    """

    def __init__(self, tracker: AITracker, experiment_name: str):
        self.tracker = tracker
        self.experiment_name = experiment_name
        self._variants: Dict[str, Any] = {}

    def add_variant(self, name: str, client: Any) -> None:
        """Add a variant (model/client) for comparison."""
        self._variants[name] = client

    def run_comparison(
        self,
        prompts: List[str],
        evaluator: Optional[Callable[[str], float]] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run A/B comparison across variants.

        Args:
            prompts: List of prompts to test
            evaluator: Function to score responses
            system_instruction: Optional system instruction

        Returns:
            Comparison results
        """
        results: Dict[str, Any] = {
            "experiment": self.experiment_name,
            "variants": {},
            "prompts": len(prompts)
        }

        for variant_name, client in self._variants.items():
            variant_results: Dict[str, Any] = {
                "responses": [],
                "latencies": [],
                "scores": [],
                "errors": 0
            }

            for prompt in prompts:
                start_time = time.time()
                try:
                    response = client.generate(
                        prompt,
                        system_instruction=system_instruction
                    )
                    latency = (time.time() - start_time) * 1000

                    variant_results["responses"].append(response)
                    variant_results["latencies"].append(latency)

                    if evaluator:
                        score = evaluator(response)
                        variant_results["scores"].append(score)

                except Exception as e:
                    variant_results["errors"] += 1
                    variant_results["responses"].append(f"Error: {e}")
                    variant_results["latencies"].append(0)

            # Calculate aggregates
            if variant_results["latencies"]:
                variant_results["avg_latency"] = (
                    sum(variant_results["latencies"]) /
                    len(variant_results["latencies"])
                )

            if variant_results["scores"]:
                variant_results["avg_score"] = (
                    sum(variant_results["scores"]) /
                    len(variant_results["scores"])
                )

            results["variants"][variant_name] = variant_results

        return results


def create_tracker(
    experiment_name: str = "ai_assistant",
    tracking_uri: Optional[str] = None,
    log_prompts: bool = True,
    log_responses: bool = True
) -> AITracker:
    """
    Factory function to create an AITracker.

    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI
        log_prompts: Whether to log prompts
        log_responses: Whether to log responses

    Returns:
        Configured AITracker

    Example:
        >>> tracker = create_tracker("my_experiment")
        >>> with tracker.start_run("task_1"):
        ...     # Your AI operations
        ...     pass
    """
    return AITracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        log_prompts=log_prompts,
        log_responses=log_responses
    )
