"""
Configuration management for AI Assistant.

This module handles all configuration aspects including:
- Model settings
- API configurations
- Default parameters
- Environment detection
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration for a specific AI model.

    Attributes:
        name: Model identifier (e.g., 'gemini-1.5-pro')
        max_tokens: Maximum output tokens
        temperature: Response randomness (0-1)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (Gemini only)
    """
    name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = None  # Gemini-specific


@dataclass
class AIConfig:
    """
    Main configuration class for AI Assistant.

    This class manages all settings for both Gemini and Claude models,
    including API authentication and default parameters.

    Attributes:
        secret_scope: Databricks secret scope name
        gemini_secret_key: Key name for Gemini API key in secret scope
        claude_secret_key: Key name for Claude API key in secret scope
        gemini_model: Default Gemini model configuration
        claude_model: Default Claude model configuration
        enable_cost_tracking: Whether to track token usage and costs
        retry_attempts: Number of retry attempts for failed requests
        retry_delay: Base delay between retries in seconds

    Example:
        >>> config = AIConfig(
        ...     secret_scope="ai-keys",
        ...     gemini_secret_key="gemini-api-key",
        ...     claude_secret_key="claude-api-key"
        ... )
    """

    # Secret management
    secret_scope: Optional[str] = None
    gemini_secret_key: str = "gemini-api-key"
    claude_secret_key: str = "claude-api-key"

    # Direct API keys (use secrets in production!)
    gemini_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None

    # Model configurations
    gemini_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="gemini-1.5-pro",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
            top_k=40
        )
    )

    claude_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95
        )
    )

    # Operational settings
    enable_cost_tracking: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Databricks-specific
    use_mlflow_tracking: bool = False

    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")

        if not self.claude_api_key:
            self.claude_api_key = os.environ.get("ANTHROPIC_API_KEY")

    def get_gemini_key(self, dbutils=None) -> Optional[str]:
        """
        Retrieve Gemini API key from configured sources.

        Priority order:
        1. Direct API key in config
        2. Databricks secret scope
        3. Environment variable

        Args:
            dbutils: Databricks utilities object for secret access

        Returns:
            API key string or None if not found
        """
        # Try direct key first
        if self.gemini_api_key:
            return self.gemini_api_key

        # Try Databricks secrets
        if dbutils and self.secret_scope:
            try:
                return dbutils.secrets.get(
                    scope=self.secret_scope,
                    key=self.gemini_secret_key
                )
            except Exception:
                pass

        # Fall back to environment
        return os.environ.get("GEMINI_API_KEY")

    def get_claude_key(self, dbutils=None) -> Optional[str]:
        """
        Retrieve Claude API key from configured sources.

        Priority order:
        1. Direct API key in config
        2. Databricks secret scope
        3. Environment variable

        Args:
            dbutils: Databricks utilities object for secret access

        Returns:
            API key string or None if not found
        """
        # Try direct key first
        if self.claude_api_key:
            return self.claude_api_key

        # Try Databricks secrets
        if dbutils and self.secret_scope:
            try:
                return dbutils.secrets.get(
                    scope=self.secret_scope,
                    key=self.claude_secret_key
                )
            except Exception:
                pass

        # Fall back to environment
        return os.environ.get("ANTHROPIC_API_KEY")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            "secret_scope": self.secret_scope,
            "gemini_model": self.gemini_model.name,
            "claude_model": self.claude_model.name,
            "enable_cost_tracking": self.enable_cost_tracking,
            "retry_attempts": self.retry_attempts,
            "use_mlflow_tracking": self.use_mlflow_tracking,
        }


# Available models for reference
AVAILABLE_GEMINI_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",
    "gemini-2.0-flash-exp",
]

AVAILABLE_CLAUDE_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
]

# Token pricing (approximate, USD per 1K tokens)
TOKEN_PRICING = {
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
}
