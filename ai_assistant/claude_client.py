"""
Anthropic Claude Client for Databricks.

This module provides a client for interacting with Anthropic's Claude AI models,
optimized for use within Databricks environments.
"""

import time
from typing import Optional, List, Dict, Any, Generator

from .config import ModelConfig, TOKEN_PRICING
from .exceptions import (
    APIKeyNotFoundError,
    ModelNotAvailableError,
    RateLimitError,
    TokenLimitError
)


class ClaudeClient:
    """
    Client for Anthropic Claude AI models.

    This client provides methods for text generation, code generation,
    and multi-turn conversations using Claude models.

    Attributes:
        client: The Anthropic client instance
        model_config: Configuration for the model
        usage_stats: Token usage statistics (if tracking enabled)

    Example:
        >>> client = ClaudeClient(api_key="your-key")
        >>> response = client.generate("Explain Apache Spark")
        >>> print(response)
    """

    def __init__(
        self,
        api_key: str,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key
            model_config: Model configuration (uses defaults if None)

        Raises:
            APIKeyNotFoundError: If api_key is None or empty
            ImportError: If anthropic package is not installed
        """
        if not api_key:
            raise APIKeyNotFoundError("Claude")

        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        # Initialize the client
        self.client = anthropic.Anthropic(api_key=api_key)

        # Set up model configuration
        self.model_config = model_config or ModelConfig(
            name="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95
        )

        # Usage tracking
        self.usage_stats: Dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0
        }

        # Conversation management - stores {"messages": [...], "system": "..."}
        self._conversations: Dict[str, Dict[str, Any]] = {}

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response for a single prompt.

        Args:
            prompt: The input prompt
            system_instruction: Optional system instruction to guide behavior
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text response

        Raises:
            RateLimitError: If rate limited by the API
            TokenLimitError: If input exceeds token limits

        Example:
            >>> response = client.generate(
            ...     "Write a PySpark UDF for data validation",
            ...     system_instruction="You are a Databricks expert"
            ... )
        """
        try:
            # Build the message
            message = self.client.messages.create(
                model=self.model_config.name,
                max_tokens=max_tokens or self.model_config.max_tokens,
                temperature=temperature if temperature is not None else self.model_config.temperature,
                system=system_instruction or "You are a helpful AI assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Update usage stats
            self._update_usage(message)

            # Extract text from response
            return self._extract_text(message)

        except Exception as e:
            return self._handle_error(e)

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response for real-time output.

        Args:
            prompt: The input prompt
            system_instruction: Optional system instruction

        Yields:
            Text chunks as they are generated

        Example:
            >>> for chunk in client.generate_stream("Explain Delta Lake"):
            ...     print(chunk, end="", flush=True)
        """
        try:
            with self.client.messages.stream(
                model=self.model_config.name,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                system=system_instruction or "You are a helpful AI assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"\n[Error: {self._handle_error(e)}]"

    def chat(
        self,
        message: str,
        conversation_name: str = "default",
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Send a message in a multi-turn conversation.

        Args:
            message: The user message
            conversation_name: Identifier for the conversation
            system_instruction: System instruction (used with each request)

        Returns:
            Model's response

        Example:
            >>> client.chat("I need help optimizing a query", "optimization")
            >>> client.chat("It uses multiple joins", "optimization")  # Has context
        """
        # Initialize conversation if it doesn't exist
        if conversation_name not in self._conversations:
            self._conversations[conversation_name] = {
                "messages": [],
                "system": system_instruction or "You are a helpful AI assistant specialized in data engineering and Databricks."
            }

        conversation = self._conversations[conversation_name]

        # Add user message to history
        conversation["messages"].append({
            "role": "user",
            "content": message
        })

        try:
            # Send request with full conversation history
            response = self.client.messages.create(
                model=self.model_config.name,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                system=conversation["system"],
                messages=conversation["messages"]
            )

            # Update usage stats
            self._update_usage(response)

            # Extract response text
            assistant_message = self._extract_text(response)

            # Add assistant response to history
            conversation["messages"].append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            # Remove failed user message from history
            conversation["messages"].pop()
            return self._handle_error(e)

    def clear_conversation(self, conversation_name: str = "default") -> None:
        """Clear a conversation's history."""
        if conversation_name in self._conversations:
            del self._conversations[conversation_name]

    def list_conversations(self) -> List[str]:
        """List all active conversation names."""
        return list(self._conversations.keys())

    def get_conversation_history(self, conversation_name: str = "default") -> List[Dict[str, str]]:
        """Get the message history for a conversation."""
        if conversation_name in self._conversations:
            return self._conversations[conversation_name]["messages"].copy()
        return []

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Note: This is an approximation. For exact counts, use the API's
        token counting endpoint if available.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated number of tokens
        """
        try:
            # Use Anthropic's token counting if available
            result = self.client.count_tokens(text)
            return result
        except Exception:
            # Rough estimate: ~4 characters per token for English text
            return len(text) // 4

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        stats = self.usage_stats.copy()

        # Calculate estimated cost
        pricing = TOKEN_PRICING.get(self.model_config.name, {})
        input_cost = (stats["total_input_tokens"] / 1000) * pricing.get("input", 0)
        output_cost = (stats["total_output_tokens"] / 1000) * pricing.get("output", 0)
        stats["estimated_cost_usd"] = input_cost + output_cost

        return stats

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0
        }

    def _extract_text(self, message) -> str:
        """Extract text content from a message response."""
        if hasattr(message, 'content') and message.content:
            # Handle list of content blocks
            if isinstance(message.content, list):
                texts = []
                for block in message.content:
                    if hasattr(block, 'text'):
                        texts.append(block.text)
                return "".join(texts)
            return str(message.content)
        return ""

    def _update_usage(self, message) -> None:
        """Update usage statistics from response."""
        try:
            if hasattr(message, 'usage'):
                self.usage_stats["total_input_tokens"] += message.usage.input_tokens
                self.usage_stats["total_output_tokens"] += message.usage.output_tokens
            self.usage_stats["total_requests"] += 1
        except Exception:
            pass

    def _handle_error(self, error: Exception) -> str:
        """Handle API errors and convert to appropriate exceptions."""
        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            raise RateLimitError("Claude")

        if "token" in error_str and ("limit" in error_str or "exceed" in error_str):
            raise TokenLimitError(0, self.model_config.max_tokens)

        if "not found" in error_str or "invalid model" in error_str:
            raise ModelNotAvailableError(
                self.model_config.name,
                "Claude",
                available_models=[
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-3-5-haiku-20241022"
                ]
            )

        if "authentication" in error_str or "api key" in error_str:
            raise APIKeyNotFoundError("Claude", "Invalid or expired API key")

        # Return error message for unknown errors
        return f"Error: {str(error)}"


class ClaudeCodeAssistant(ClaudeClient):
    """
    Specialized Claude client for code assistance tasks.

    This class extends ClaudeClient with additional methods optimized
    for code generation, review, and explanation tasks.

    Example:
        >>> assistant = ClaudeCodeAssistant(api_key="your-key")
        >>> code = assistant.generate_code("Create a Spark DataFrame reader")
        >>> review = assistant.review_code(existing_code)
    """

    def __init__(self, api_key: str, model_config: Optional[ModelConfig] = None):
        super().__init__(api_key, model_config)

        # Default system prompt for code tasks
        self.code_system_prompt = """You are an expert software engineer specializing in:
- Apache Spark and PySpark
- Databricks platform and Delta Lake
- Python best practices
- Data engineering patterns

When generating code:
1. Write clean, well-documented code
2. Include type hints where appropriate
3. Handle errors gracefully
4. Follow PEP 8 style guidelines
5. Optimize for Spark performance when applicable"""

    def generate_code(
        self,
        task: str,
        language: str = "python",
        context: Optional[str] = None,
        include_tests: bool = False
    ) -> str:
        """
        Generate code for a specific task.

        Args:
            task: Description of what the code should do
            language: Programming language (default: python)
            context: Additional context about the codebase or requirements
            include_tests: Whether to include unit tests

        Returns:
            Generated code as a string

        Example:
            >>> code = assistant.generate_code(
            ...     "Read a CSV file and convert to Delta format",
            ...     context="Using Databricks with Unity Catalog"
            ... )
        """
        prompt = f"""Generate {language} code for the following task:

Task: {task}

{f'Context: {context}' if context else ''}

Requirements:
- Write production-ready code
- Include clear comments
- Handle edge cases
{f'- Include unit tests' if include_tests else ''}

Provide only the code with necessary imports, no explanations outside of code comments."""

        return self.generate(prompt, system_instruction=self.code_system_prompt)

    def review_code(
        self,
        code: str,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Review code and provide feedback.

        Args:
            code: The code to review
            focus_areas: Specific areas to focus on (e.g., ["performance", "security"])

        Returns:
            Code review feedback

        Example:
            >>> feedback = assistant.review_code(
            ...     my_spark_code,
            ...     focus_areas=["performance", "error handling"]
            ... )
        """
        areas = focus_areas or ["correctness", "performance", "readability", "best practices"]

        prompt = f"""Review the following code and provide feedback:

```
{code}
```

Focus on these areas: {', '.join(areas)}

Provide:
1. Summary of what the code does
2. Issues found (if any)
3. Suggestions for improvement
4. Overall assessment"""

        return self.generate(prompt, system_instruction=self.code_system_prompt)

    def explain_code(self, code: str, detail_level: str = "medium") -> str:
        """
        Explain what a piece of code does.

        Args:
            code: The code to explain
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Explanation of the code

        Example:
            >>> explanation = assistant.explain_code(complex_spark_query)
        """
        detail_instructions = {
            "brief": "Provide a brief 2-3 sentence summary.",
            "medium": "Explain the main components and flow.",
            "detailed": "Provide a detailed line-by-line explanation."
        }

        prompt = f"""Explain the following code:

```
{code}
```

{detail_instructions.get(detail_level, detail_instructions['medium'])}"""

        return self.generate(prompt, system_instruction=self.code_system_prompt)

    def fix_code(self, code: str, error_message: Optional[str] = None) -> str:
        """
        Fix bugs in code.

        Args:
            code: The buggy code
            error_message: Error message if available

        Returns:
            Fixed code with explanation

        Example:
            >>> fixed = assistant.fix_code(buggy_code, "TypeError: ...")
        """
        prompt = f"""Fix the following code:

```
{code}
```

{f'Error message: {error_message}' if error_message else 'Identify and fix any bugs.'}

Provide:
1. The fixed code
2. Brief explanation of what was wrong and how you fixed it"""

        return self.generate(prompt, system_instruction=self.code_system_prompt)
