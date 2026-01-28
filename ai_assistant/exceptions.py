"""
Custom exceptions for the AI Assistant module.

This module defines all custom exceptions that may be raised during
AI assistant operations, providing clear error messages and handling guidance.
"""

from typing import Optional, List


class AIAssistantError(Exception):
    """
    Base exception for all AI Assistant errors.

    All custom exceptions in this module inherit from this class,
    allowing for broad exception catching when needed.

    Example:
        >>> try:
        ...     assistant.ask_gemini("query")
        ... except AIAssistantError as e:
        ...     print(f"AI operation failed: {e}")
    """
    pass


class APIKeyNotFoundError(AIAssistantError):
    """
    Raised when an API key cannot be found in secrets or environment.

    This typically occurs when:
    - The Databricks secret scope doesn't exist
    - The secret key name is incorrect
    - Environment variables are not set
    - Insufficient permissions to access secrets

    Resolution:
        1. Verify secret scope exists: `databricks secrets list-scopes`
        2. Verify secret key exists: `databricks secrets list --scope <scope>`
        3. Check permissions for the secret scope
        4. Or set environment variable: `os.environ["GEMINI_API_KEY"] = "..."`
    """

    def __init__(self, provider: str, message: Optional[str] = None):
        self.provider = provider
        self.message = message or f"API key for {provider} not found. Please configure it using Databricks Secrets or environment variables."
        super().__init__(self.message)


class ModelNotAvailableError(AIAssistantError):
    """
    Raised when the requested AI model is not available or invalid.

    This can happen when:
    - The model name is misspelled
    - The model has been deprecated
    - The model is not available in your region
    - Your API plan doesn't include access to this model

    Resolution:
        1. Check the model name spelling
        2. Verify model availability in the provider's documentation
        3. Try an alternative model from the supported list
    """

    def __init__(self, model: str, provider: str, available_models: Optional[List[str]] = None):
        self.model = model
        self.provider = provider
        self.available_models = available_models or []

        message = f"Model '{model}' is not available for {provider}."
        if self.available_models:
            message += f" Available models: {', '.join(self.available_models)}"

        super().__init__(message)


class RateLimitError(AIAssistantError):
    """
    Raised when API rate limits are exceeded.

    AI providers impose rate limits to ensure fair usage.
    When exceeded, you should wait before making more requests.

    Attributes:
        retry_after: Suggested wait time in seconds before retrying
        provider: The AI provider that rate limited the request

    Resolution:
        1. Implement exponential backoff
        2. Reduce request frequency
        3. Consider upgrading your API plan
        4. Use batch processing for multiple items
    """

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        self.provider = provider
        self.retry_after = retry_after

        message = f"Rate limit exceeded for {provider}."
        if retry_after:
            message += f" Retry after {retry_after} seconds."

        super().__init__(message)


class TokenLimitError(AIAssistantError):
    """
    Raised when the input or output exceeds the model's token limit.

    Each AI model has maximum token limits for:
    - Input tokens (context window)
    - Output tokens (response length)
    - Combined tokens (input + output)

    Attributes:
        token_count: Number of tokens in the request
        token_limit: Maximum allowed tokens
        token_type: 'input', 'output', or 'total'

    Resolution:
        1. Reduce input text length
        2. Summarize or chunk large documents
        3. Use a model with larger context window
        4. Reduce max_tokens parameter for output
    """

    def __init__(self, token_count: int, token_limit: int, token_type: str = "total"):
        self.token_count = token_count
        self.token_limit = token_limit
        self.token_type = token_type

        message = (
            f"Token limit exceeded: {token_count} {token_type} tokens "
            f"exceeds limit of {token_limit}. "
            f"Reduce your input size or use a model with larger context window."
        )

        super().__init__(message)


class ConversationNotFoundError(AIAssistantError):
    """
    Raised when attempting to access a conversation that doesn't exist.

    Resolution:
        1. Start a new conversation with `start_conversation(name)`
        2. Check conversation name spelling
        3. List active conversations with `list_conversations()`
    """

    def __init__(self, conversation_name: str):
        self.conversation_name = conversation_name
        message = f"Conversation '{conversation_name}' not found. Start it with start_conversation('{conversation_name}')"
        super().__init__(message)


class DatabricksContextError(AIAssistantError):
    """
    Raised when Databricks-specific operations fail.

    This can occur when:
    - Running outside of Databricks environment
    - SparkSession is not available
    - Secret scope access is denied

    Resolution:
        1. Ensure code runs in Databricks notebook or job
        2. Verify cluster has required permissions
        3. Use environment variables as fallback
    """

    def __init__(self, message: str):
        super().__init__(f"Databricks context error: {message}")
