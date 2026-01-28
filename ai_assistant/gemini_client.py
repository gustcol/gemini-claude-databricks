"""
Google Gemini Client for Databricks.

This module provides a client for interacting with Google's Gemini AI models,
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


class GeminiClient:
    """
    Client for Google Gemini AI models.

    This client provides methods for text generation, code generation,
    and multi-turn conversations using Gemini models.

    Attributes:
        model: The Gemini model being used
        generation_config: Configuration for text generation
        usage_stats: Token usage statistics (if tracking enabled)

    Example:
        >>> client = GeminiClient(api_key="your-key")
        >>> response = client.generate("Explain machine learning")
        >>> print(response)
    """

    def __init__(
        self,
        api_key: str,
        model_config: Optional[ModelConfig] = None,
        safety_settings: Optional[List[Dict]] = None
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google AI API key
            model_config: Model configuration (uses defaults if None)
            safety_settings: Custom safety settings (uses defaults if None)

        Raises:
            APIKeyNotFoundError: If api_key is None or empty
            ImportError: If google-generativeai package is not installed
        """
        if not api_key:
            raise APIKeyNotFoundError("Gemini")

        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        # Configure the API
        genai.configure(api_key=api_key)

        # Set up model configuration
        self.model_config = model_config or ModelConfig(
            name="gemini-1.5-pro",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            top_k=40
        )

        # Create generation config
        self.generation_config = genai.GenerationConfig(
            max_output_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
        )

        # Set up safety settings (use permissive defaults for development)
        self.safety_settings = safety_settings or self._default_safety_settings()

        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_config.name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
        except Exception as e:
            raise ModelNotAvailableError(
                self.model_config.name,
                "Gemini",
                available_models=[
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.0-pro"
                ]
            )

        # Usage tracking
        self.usage_stats: Dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0
        }

        # Conversation management
        self._conversations: Dict[str, Any] = {}

    def _default_safety_settings(self) -> List[Dict]:
        """Get default safety settings (permissive for development)."""
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
        ]

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
            ...     "Write a PySpark function to read CSV files",
            ...     system_instruction="You are a Spark expert"
            ... )
        """
        # Build the model with system instruction if provided
        model = self.model
        if system_instruction:
            model = self._genai.GenerativeModel(
                model_name=self.model_config.name,
                generation_config=self._get_generation_config(temperature, max_tokens),
                safety_settings=self.safety_settings,
                system_instruction=system_instruction
            )

        try:
            response = model.generate_content(prompt)

            # Update usage stats
            self._update_usage(response)

            return response.text

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
        model = self.model
        if system_instruction:
            model = self._genai.GenerativeModel(
                model_name=self.model_config.name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=system_instruction
            )

        try:
            response = model.generate_content(prompt, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

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
            system_instruction: System instruction (only used for new conversations)

        Returns:
            Model's response

        Example:
            >>> client.chat("I need help with Spark", "spark_help")
            >>> client.chat("Specifically with joins", "spark_help")  # Has context
        """
        # Create conversation if it doesn't exist
        if conversation_name not in self._conversations:
            model = self.model
            if system_instruction:
                model = self._genai.GenerativeModel(
                    model_name=self.model_config.name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    system_instruction=system_instruction
                )
            self._conversations[conversation_name] = model.start_chat(history=[])

        chat = self._conversations[conversation_name]

        try:
            response = chat.send_message(message)
            self._update_usage(response)
            return response.text

        except Exception as e:
            return self._handle_error(e)

    def clear_conversation(self, conversation_name: str = "default") -> None:
        """Clear a conversation's history."""
        if conversation_name in self._conversations:
            del self._conversations[conversation_name]

    def list_conversations(self) -> List[str]:
        """List all active conversation names."""
        return list(self._conversations.keys())

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Rough estimate if counting fails
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

    def _get_generation_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Create generation config with optional overrides."""
        return self._genai.GenerationConfig(
            max_output_tokens=max_tokens or self.model_config.max_tokens,
            temperature=temperature or self.model_config.temperature,
            top_p=self.model_config.top_p,
            top_k=self.model_config.top_k,
        )

    def _update_usage(self, response) -> None:
        """Update usage statistics from response."""
        try:
            if hasattr(response, 'usage_metadata'):
                self.usage_stats["total_input_tokens"] += response.usage_metadata.prompt_token_count
                self.usage_stats["total_output_tokens"] += response.usage_metadata.candidates_token_count
            self.usage_stats["total_requests"] += 1
        except Exception:
            pass

    def _handle_error(self, error: Exception) -> str:
        """Handle API errors and convert to appropriate exceptions."""
        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            raise RateLimitError("Gemini")

        if "token" in error_str and ("limit" in error_str or "exceed" in error_str):
            raise TokenLimitError(0, self.model_config.max_tokens)

        if "not found" in error_str or "invalid" in error_str:
            raise ModelNotAvailableError(self.model_config.name, "Gemini")

        # Return error message for unknown errors
        return f"Error: {str(error)}"
