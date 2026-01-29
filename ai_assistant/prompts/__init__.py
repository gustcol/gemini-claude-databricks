"""
Prompt Templates Module for AI Assistant.

This module provides reusable, versioned prompt templates for
common AI operations in data engineering contexts.

Features:
- Type-safe prompt templates
- Variable substitution
- Few-shot examples
- Template versioning
- Domain-specific templates
"""

from .templates import (
    PromptTemplate,
    PromptVariable,
    PromptLibrary,
    create_template,
    render_template
)

from .data_engineering import (
    SQL_OPTIMIZATION_PROMPT,
    DDL_GENERATION_PROMPT,
    PIPELINE_GENERATION_PROMPT,
    ERROR_EXPLANATION_PROMPT,
    CODE_REVIEW_PROMPT,
    DATA_ANALYSIS_PROMPT,
    SCHEMA_DESIGN_PROMPT,
    DATA_QUALITY_PROMPT,
    get_data_engineering_prompts
)

__all__ = [
    # Core classes
    "PromptTemplate",
    "PromptVariable",
    "PromptLibrary",
    "create_template",
    "render_template",
    # Data engineering prompts
    "SQL_OPTIMIZATION_PROMPT",
    "DDL_GENERATION_PROMPT",
    "PIPELINE_GENERATION_PROMPT",
    "ERROR_EXPLANATION_PROMPT",
    "CODE_REVIEW_PROMPT",
    "DATA_ANALYSIS_PROMPT",
    "SCHEMA_DESIGN_PROMPT",
    "DATA_QUALITY_PROMPT",
    "get_data_engineering_prompts",
]
