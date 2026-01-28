"""
Core AI Assistant module for Databricks.

This module provides the main AIAssistant class that provides a unified
interface for both Gemini and Claude models, with Databricks-specific
features and utilities.
"""

import os
from typing import Optional, List, Dict, Any, Union, Generator
from dataclasses import dataclass

from .config import AIConfig, ModelConfig, AVAILABLE_GEMINI_MODELS, AVAILABLE_CLAUDE_MODELS
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient, ClaudeCodeAssistant
from .exceptions import (
    AIAssistantError,
    APIKeyNotFoundError,
    DatabricksContextError
)


@dataclass
class DataFrameSchema:
    """Schema information for a DataFrame."""
    columns: List[Dict[str, str]]
    row_count: int
    sample_data: List[Dict]


class AIAssistant:
    """
    Unified AI Assistant for Databricks.

    This class provides a single interface for interacting with both
    Google Gemini and Anthropic Claude models, with special features
    for Databricks environments.

    Features:
    - Automatic API key management via Databricks Secrets
    - DataFrame analysis and code generation
    - Multi-turn conversations with memory
    - Cost tracking and usage monitoring
    - Streaming responses for interactive use

    Attributes:
        config: Configuration object
        gemini: GeminiClient instance (lazy loaded)
        claude: ClaudeClient instance (lazy loaded)

    Example:
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> response = assistant.ask("Explain Delta Lake")
        >>> print(response)
    """

    def __init__(
        self,
        secret_scope: Optional[str] = None,
        gemini_secret_key: str = "gemini-api-key",
        claude_secret_key: str = "claude-api-key",
        gemini_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        gemini_model: str = "gemini-1.5-pro",
        claude_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        dbutils=None
    ):
        """
        Initialize the AI Assistant.

        Args:
            secret_scope: Databricks secret scope name
            gemini_secret_key: Key name for Gemini API key in secret scope
            claude_secret_key: Key name for Claude API key in secret scope
            gemini_api_key: Direct Gemini API key (use secrets in production!)
            claude_api_key: Direct Claude API key (use secrets in production!)
            gemini_model: Default Gemini model to use
            claude_model: Default Claude model to use
            max_tokens: Maximum output tokens
            temperature: Response temperature (0-1)
            dbutils: Databricks utilities object (auto-detected if not provided)

        Example:
            >>> # Using Databricks secrets (recommended)
            >>> assistant = AIAssistant(secret_scope="ai-keys")

            >>> # Using direct keys (development only)
            >>> assistant = AIAssistant(
            ...     gemini_api_key="your-key",
            ...     claude_api_key="your-key"
            ... )
        """
        # Store dbutils reference
        self._dbutils = dbutils or self._get_dbutils()

        # Create configuration
        self.config = AIConfig(
            secret_scope=secret_scope,
            gemini_secret_key=gemini_secret_key,
            claude_secret_key=claude_secret_key,
            gemini_api_key=gemini_api_key,
            claude_api_key=claude_api_key,
            gemini_model=ModelConfig(
                name=gemini_model,
                max_tokens=max_tokens,
                temperature=temperature
            ),
            claude_model=ModelConfig(
                name=claude_model,
                max_tokens=max_tokens,
                temperature=temperature
            )
        )

        # Lazy-loaded clients
        self._gemini_client: Optional[GeminiClient] = None
        self._claude_client: Optional[ClaudeClient] = None
        self._claude_code_assistant: Optional[ClaudeCodeAssistant] = None

        # Default model preference
        self._default_model = "claude"

        # Usage tracking
        self._cost_tracking_enabled = False

    def _get_dbutils(self):
        """Attempt to get Databricks utilities object."""
        try:
            # Try to get dbutils from Databricks runtime
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            return spark._jvm.com.databricks.backend.daemon.driver.DBUtils(spark._sc)
        except Exception:
            pass

        try:
            # Alternative method for newer Databricks runtimes
            import IPython
            dbutils = IPython.get_ipython().user_ns.get("dbutils")
            return dbutils
        except Exception:
            pass

        return None

    @property
    def gemini(self) -> GeminiClient:
        """Get or create Gemini client (lazy loading)."""
        if self._gemini_client is None:
            api_key = self.config.get_gemini_key(self._dbutils)
            if not api_key:
                raise APIKeyNotFoundError(
                    "Gemini",
                    "Set GEMINI_API_KEY environment variable or configure Databricks secrets"
                )
            self._gemini_client = GeminiClient(
                api_key=api_key,
                model_config=self.config.gemini_model
            )
        return self._gemini_client

    @property
    def claude(self) -> ClaudeClient:
        """Get or create Claude client (lazy loading)."""
        if self._claude_client is None:
            api_key = self.config.get_claude_key(self._dbutils)
            if not api_key:
                raise APIKeyNotFoundError(
                    "Claude",
                    "Set ANTHROPIC_API_KEY environment variable or configure Databricks secrets"
                )
            self._claude_client = ClaudeClient(
                api_key=api_key,
                model_config=self.config.claude_model
            )
        return self._claude_client

    @property
    def code_assistant(self) -> ClaudeCodeAssistant:
        """Get or create Claude Code Assistant (lazy loading)."""
        if self._claude_code_assistant is None:
            api_key = self.config.get_claude_key(self._dbutils)
            if not api_key:
                raise APIKeyNotFoundError(
                    "Claude",
                    "Set ANTHROPIC_API_KEY environment variable or configure Databricks secrets"
                )
            self._claude_code_assistant = ClaudeCodeAssistant(
                api_key=api_key,
                model_config=self.config.claude_model
            )
        return self._claude_code_assistant

    def set_default_model(self, model: str) -> None:
        """
        Set the default model to use.

        Args:
            model: Either "gemini" or "claude"
        """
        if model.lower() not in ["gemini", "claude"]:
            raise ValueError("Model must be 'gemini' or 'claude'")
        self._default_model = model.lower()

    def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Ask a question to the AI model.

        This is the simplest way to get a response. Use ask_gemini()
        or ask_claude() for model-specific calls.

        Args:
            prompt: Your question or request
            model: "gemini" or "claude" (uses default if not specified)
            system_instruction: Optional system instruction
            **kwargs: Additional arguments passed to the model

        Returns:
            AI response as a string

        Example:
            >>> response = assistant.ask("What is Delta Lake?")
            >>> response = assistant.ask("Explain Spark", model="gemini")
        """
        model = model or self._default_model

        if model.lower() == "gemini":
            return self.ask_gemini(prompt, system_instruction, **kwargs)
        else:
            return self.ask_claude(prompt, system_instruction, **kwargs)

    def ask_gemini(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Ask Gemini a question.

        Args:
            prompt: Your question or request
            system_instruction: Optional system instruction
            **kwargs: Additional arguments (temperature, max_tokens)

        Returns:
            Gemini's response
        """
        return self.gemini.generate(
            prompt,
            system_instruction=system_instruction,
            **kwargs
        )

    def ask_claude(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Ask Claude a question.

        Args:
            prompt: Your question or request
            system_instruction: Optional system instruction
            **kwargs: Additional arguments (temperature, max_tokens)

        Returns:
            Claude's response
        """
        return self.claude.generate(
            prompt,
            system_instruction=system_instruction,
            **kwargs
        )

    def stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Get a streaming response from the AI model.

        Args:
            prompt: Your question or request
            model: "gemini" or "claude" (uses default if not specified)
            system_instruction: Optional system instruction

        Yields:
            Text chunks as they are generated

        Example:
            >>> for chunk in assistant.stream("Explain MapReduce"):
            ...     print(chunk, end="", flush=True)
        """
        model = model or self._default_model

        if model.lower() == "gemini":
            yield from self.gemini.generate_stream(prompt, system_instruction)
        else:
            yield from self.claude.generate_stream(prompt, system_instruction)

    def chat(
        self,
        message: str,
        conversation_name: str = "default",
        model: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Send a message in a multi-turn conversation.

        The conversation history is maintained, allowing for context-aware
        responses.

        Args:
            message: Your message
            conversation_name: Identifier for the conversation
            model: "gemini" or "claude" (uses default if not specified)
            system_instruction: System instruction (for new conversations)

        Returns:
            AI response

        Example:
            >>> assistant.chat("I have a Spark performance issue")
            >>> assistant.chat("The job takes 2 hours to run")
            >>> assistant.chat("What should I optimize first?")
        """
        model = model or self._default_model

        if model.lower() == "gemini":
            return self.gemini.chat(message, conversation_name, system_instruction)
        else:
            return self.claude.chat(message, conversation_name, system_instruction)

    def clear_conversation(
        self,
        conversation_name: str = "default",
        model: Optional[str] = None
    ) -> None:
        """Clear a conversation's history."""
        model = model or self._default_model

        if model.lower() == "gemini":
            self.gemini.clear_conversation(conversation_name)
        else:
            self.claude.clear_conversation(conversation_name)

    def generate_code(
        self,
        task: str,
        language: str = "python",
        model: Optional[str] = None,
        context: Optional[str] = None,
        include_tests: bool = False
    ) -> str:
        """
        Generate code for a specific task.

        Args:
            task: Description of what the code should do
            language: Programming language (default: python)
            model: "gemini" or "claude"
            context: Additional context about requirements
            include_tests: Whether to include unit tests

        Returns:
            Generated code

        Example:
            >>> code = assistant.generate_code(
            ...     "Create a function to read and merge Delta tables",
            ...     context="Using Unity Catalog"
            ... )
        """
        prompt = f"""Generate {language} code for the following task:

Task: {task}

{f'Context: {context}' if context else ''}

Requirements:
- Write production-ready, well-documented code
- Include necessary imports
- Handle edge cases and errors appropriately
- Follow best practices for {language}
{f'- Include comprehensive unit tests' if include_tests else ''}

Provide only the code with comments, no additional explanations."""

        system = """You are an expert software engineer specializing in:
- Apache Spark and PySpark
- Databricks platform and Delta Lake
- Data engineering best practices
- Clean, efficient code

Generate code that is ready for production use."""

        return self.ask(prompt, model=model, system_instruction=system)

    def analyze_dataframe(
        self,
        df,
        questions: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Analyze a Spark DataFrame using AI.

        Args:
            df: Spark DataFrame to analyze
            questions: Specific questions about the data
            model: "gemini" or "claude"

        Returns:
            Analysis results

        Example:
            >>> analysis = assistant.analyze_dataframe(
            ...     spark.table("sales"),
            ...     questions=["What are the data quality issues?"]
            ... )
        """
        # Extract schema and sample data
        schema_info = self._extract_schema_info(df)

        # Build analysis prompt
        prompt = f"""Analyze this Spark DataFrame:

Schema:
{schema_info['schema_string']}

Row count: {schema_info['row_count']}

Sample data (first 5 rows):
{schema_info['sample_data']}

{f"Please answer these questions:" if questions else "Provide a comprehensive analysis including:"}
{chr(10).join(f'- {q}' for q in (questions or [
    "Summary of the data structure",
    "Data quality observations",
    "Potential issues or anomalies",
    "Optimization recommendations",
    "Suggested transformations"
]))}"""

        system = """You are a data engineering expert analyzing Spark DataFrames.
Provide actionable insights and specific recommendations."""

        return self.ask(prompt, model=model, system_instruction=system)

    def optimize_query(
        self,
        query: str,
        model: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Get optimization suggestions for a Spark SQL query.

        Args:
            query: The SQL query to optimize
            model: "gemini" or "claude"
            context: Additional context (table sizes, current performance)

        Returns:
            Optimization suggestions and improved query

        Example:
            >>> optimized = assistant.optimize_query('''
            ...     SELECT * FROM orders
            ...     JOIN customers ON orders.customer_id = customers.id
            ... ''')
        """
        prompt = f"""Optimize this Spark SQL query:

```sql
{query}
```

{f'Context: {context}' if context else ''}

Provide:
1. Analysis of current query issues
2. Specific optimization recommendations
3. Optimized version of the query
4. Explanation of changes made"""

        system = """You are a Spark SQL optimization expert.
Focus on practical improvements that will significantly impact performance."""

        return self.ask(prompt, model=model, system_instruction=system)

    def explain_error(
        self,
        error_message: str,
        code: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Explain an error and suggest fixes.

        Args:
            error_message: The error message
            code: The code that caused the error (if available)
            model: "gemini" or "claude"

        Returns:
            Error explanation and suggested fixes

        Example:
            >>> fix = assistant.explain_error(
            ...     "AnalysisException: Table or view not found",
            ...     code=my_spark_code
            ... )
        """
        prompt = f"""Explain this error and provide solutions:

Error:
```
{error_message}
```

{f'Code that caused the error:' if code else ''}
{f'```python{chr(10)}{code}{chr(10)}```' if code else ''}

Provide:
1. Clear explanation of what went wrong
2. Common causes of this error
3. Step-by-step solution
4. Corrected code (if applicable)"""

        system = """You are a debugging expert for Spark and Databricks.
Provide clear, actionable solutions."""

        return self.ask(prompt, model=model, system_instruction=system)

    def _extract_schema_info(self, df) -> Dict[str, Any]:
        """Extract schema information from a Spark DataFrame."""
        try:
            # Get schema as string
            schema_string = df._jdf.schema().treeString()

            # Get row count
            row_count = df.count()

            # Get sample data
            sample_rows = df.limit(5).toPandas().to_string()

            return {
                "schema_string": schema_string,
                "row_count": row_count,
                "sample_data": sample_rows
            }
        except Exception as e:
            return {
                "schema_string": "Unable to extract schema",
                "row_count": "Unknown",
                "sample_data": f"Error: {str(e)}"
            }

    def enable_cost_tracking(self) -> None:
        """Enable cost tracking for all API calls."""
        self._cost_tracking_enabled = True

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get combined usage statistics from all clients.

        Returns:
            Dictionary with token usage and estimated costs
        """
        summary = {
            "gemini": {},
            "claude": {},
            "total_estimated_cost": 0.0
        }

        if self._gemini_client:
            gemini_stats = self._gemini_client.get_usage_stats()
            summary["gemini"] = gemini_stats
            summary["total_estimated_cost"] += gemini_stats.get("estimated_cost_usd", 0)

        if self._claude_client:
            claude_stats = self._claude_client.get_usage_stats()
            summary["claude"] = claude_stats
            summary["total_estimated_cost"] += claude_stats.get("estimated_cost_usd", 0)

        return summary

    def reset_usage_stats(self) -> None:
        """Reset usage statistics for all clients."""
        if self._gemini_client:
            self._gemini_client.reset_usage_stats()
        if self._claude_client:
            self._claude_client.reset_usage_stats()

    def is_gemini_available(self) -> bool:
        """Check if Gemini API key is configured."""
        try:
            key = self.config.get_gemini_key(self._dbutils)
            return key is not None and len(key) > 0
        except Exception:
            return False

    def is_claude_available(self) -> bool:
        """Check if Claude API key is configured."""
        try:
            key = self.config.get_claude_key(self._dbutils)
            return key is not None and len(key) > 0
        except Exception:
            return False

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models for each provider."""
        return {
            "gemini": AVAILABLE_GEMINI_MODELS,
            "claude": AVAILABLE_CLAUDE_MODELS
        }

    def __repr__(self) -> str:
        """String representation of the assistant."""
        gemini_status = "configured" if self.is_gemini_available() else "not configured"
        claude_status = "configured" if self.is_claude_available() else "not configured"

        return (
            f"AIAssistant(\n"
            f"  gemini={gemini_status}, model={self.config.gemini_model.name}\n"
            f"  claude={claude_status}, model={self.config.claude_model.name}\n"
            f"  default_model={self._default_model}\n"
            f")"
        )
