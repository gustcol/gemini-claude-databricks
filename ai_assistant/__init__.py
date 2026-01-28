"""
AI Assistant for Databricks
============================

A unified interface for using Google Gemini and Anthropic Claude AI models
within Databricks notebooks and jobs.

Features:
    - Unified API for Gemini and Claude
    - Pipeline generation (DLT, ETL, Streaming)
    - Unity Catalog integration
    - Spark DataFrame processing
    - Code generation and review
    - Cost tracking and monitoring

Example:
    >>> from ai_assistant import AIAssistant
    >>> assistant = AIAssistant(secret_scope="ai-keys")
    >>> response = assistant.ask_gemini("Explain Delta Lake")
    >>> print(response)

Repository:
    https://github.com/gustcol/gemini-claude-databricks

License:
    MIT License
"""

from .core import AIAssistant
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient, ClaudeCodeAssistant
from .config import AIConfig, ModelConfig
from .pipelines import (
    PipelineGenerator,
    PipelineType,
    PipelineConfig,
    TableDefinition,
    create_dlt_template
)
from .unity_catalog import (
    UnityCatalogHelper,
    ColumnDefinition,
    TableDefinition as UCTableDefinition,
    PrivilegeType,
    SecurableType,
    get_uc_best_practices
)
from .exceptions import (
    AIAssistantError,
    APIKeyNotFoundError,
    ModelNotAvailableError,
    RateLimitError,
    TokenLimitError,
    ConversationNotFoundError,
    DatabricksContextError
)

__version__ = "0.1.0"
__author__ = "Guxxxta / Gustcol"
__email__ = "gustcol@gmail.com"
__repository__ = "https://github.com/gustcol/gemini-claude-databricks"

__all__ = [
    # Core
    "AIAssistant",
    "GeminiClient",
    "ClaudeClient",
    "ClaudeCodeAssistant",
    # Configuration
    "AIConfig",
    "ModelConfig",
    # Pipelines
    "PipelineGenerator",
    "PipelineType",
    "PipelineConfig",
    "TableDefinition",
    "create_dlt_template",
    # Unity Catalog
    "UnityCatalogHelper",
    "ColumnDefinition",
    "UCTableDefinition",
    "PrivilegeType",
    "SecurableType",
    "get_uc_best_practices",
    # Exceptions
    "AIAssistantError",
    "APIKeyNotFoundError",
    "ModelNotAvailableError",
    "RateLimitError",
    "TokenLimitError",
    "ConversationNotFoundError",
    "DatabricksContextError",
]
