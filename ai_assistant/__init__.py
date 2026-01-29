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
    - Semantic caching for cost optimization
    - RAG (Retrieval Augmented Generation)
    - Autonomous AI agents for data tasks
    - MLflow tracking and observability
    - Prompt templates library
    - Data quality expectation generation
    - IPython magic commands
    - Automatic documentation generation
    - dbt integration and conversion
    - Security guardrails (SQL validation, PII detection, rate limiting)

Example:
    >>> from ai_assistant import AIAssistant
    >>> assistant = AIAssistant(secret_scope="ai-keys")
    >>> response = assistant.ask_gemini("Explain Delta Lake")
    >>> print(response)

    # With caching
    >>> from ai_assistant import create_cache, CachedAIClient
    >>> cache = create_cache(max_size=1000)
    >>> cached = CachedAIClient(assistant.gemini, cache)
    >>> response = cached.generate("Query")  # Cached on second call

    # With guardrails
    >>> from ai_assistant import create_guardrails
    >>> safe_ai = create_guardrails(assistant.claude)
    >>> response = safe_ai.generate("Safe prompt", user_id="user_1")

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

# New modules
from .cache import (
    SemanticCache,
    CachedAIClient,
    EmbeddingProvider,
    SimpleHashEmbedding,
    create_cache
)
from .rag import (
    RAGPipeline,
    DocumentChunker,
    InMemoryVectorStore,
    UnityCatalogContextProvider,
    Document,
    Chunk,
    ChunkingStrategy,
    create_rag_pipeline
)
from .agents import (
    BaseAgent,
    ReActAgent,
    AgentExecutor,
    AgentMemory,
    DataAnalystAgent,
    DataEngineerAgent,
    Tool,
    ToolRegistry,
    ToolResult
)
from .tracking import (
    AITracker,
    TrackedAIClient,
    LLMCallMetrics,
    ABExperiment,
    create_tracker
)
from .prompts import (
    PromptTemplate,
    PromptVariable,
    PromptLibrary,
    SQL_OPTIMIZATION_PROMPT,
    DDL_GENERATION_PROMPT,
    PIPELINE_GENERATION_PROMPT,
    ERROR_EXPLANATION_PROMPT,
    CODE_REVIEW_PROMPT,
    DATA_ANALYSIS_PROMPT,
    create_template,
    get_data_engineering_prompts
)
from .data_quality import (
    DataQualityAnalyzer,
    DataExpectation,
    DataQualityReport,
    create_data_quality_analyzer
)
from .docs_generator import (
    DocsGenerator,
    DocSection,
    FunctionDoc,
    create_docs_generator
)
from .dbt_integration import (
    DBTIntegration,
    DBTModel,
    DBTProject,
    create_dbt_integration
)
from .guardrails import (
    AIGuardrails,
    SQLValidator,
    PIIDetector,
    RateLimiter,
    AuditLogger,
    create_guardrails
)

__version__ = "0.2.0"
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
    # Cache
    "SemanticCache",
    "CachedAIClient",
    "EmbeddingProvider",
    "SimpleHashEmbedding",
    "create_cache",
    # RAG
    "RAGPipeline",
    "DocumentChunker",
    "InMemoryVectorStore",
    "UnityCatalogContextProvider",
    "Document",
    "Chunk",
    "ChunkingStrategy",
    "create_rag_pipeline",
    # Agents
    "BaseAgent",
    "ReActAgent",
    "AgentExecutor",
    "AgentMemory",
    "DataAnalystAgent",
    "DataEngineerAgent",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    # Tracking
    "AITracker",
    "TrackedAIClient",
    "LLMCallMetrics",
    "ABExperiment",
    "create_tracker",
    # Prompts
    "PromptTemplate",
    "PromptVariable",
    "PromptLibrary",
    "SQL_OPTIMIZATION_PROMPT",
    "DDL_GENERATION_PROMPT",
    "PIPELINE_GENERATION_PROMPT",
    "ERROR_EXPLANATION_PROMPT",
    "CODE_REVIEW_PROMPT",
    "DATA_ANALYSIS_PROMPT",
    "create_template",
    "get_data_engineering_prompts",
    # Data Quality
    "DataQualityAnalyzer",
    "DataExpectation",
    "DataQualityReport",
    "create_data_quality_analyzer",
    # Docs Generator
    "DocsGenerator",
    "DocSection",
    "FunctionDoc",
    "create_docs_generator",
    # dbt Integration
    "DBTIntegration",
    "DBTModel",
    "DBTProject",
    "create_dbt_integration",
    # Guardrails
    "AIGuardrails",
    "SQLValidator",
    "PIIDetector",
    "RateLimiter",
    "AuditLogger",
    "create_guardrails",
]
