"""
AI Agents Module for Databricks.

This module provides autonomous AI agents that can execute tasks
using tools and reasoning capabilities.

Available Agents:
- DataAnalystAgent: Analyzes data and generates insights
- DataEngineerAgent: Creates and modifies data pipelines
- SQLAgent: Executes and optimizes SQL queries
"""

from .base import BaseAgent, AgentExecutor, AgentResult, ReActAgent, AgentMemory
from .tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    SQLExecutorTool,
    TableInfoTool,
    DataProfilerTool,
    CodeGeneratorTool,
    FileReaderTool,
    PythonREPLTool
)
from .data_analyst import DataAnalystAgent
from .data_engineer import DataEngineerAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentExecutor",
    "AgentResult",
    "ReActAgent",
    "AgentMemory",
    # Tools
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "SQLExecutorTool",
    "TableInfoTool",
    "DataProfilerTool",
    "CodeGeneratorTool",
    "FileReaderTool",
    "PythonREPLTool",
    # Agents
    "DataAnalystAgent",
    "DataEngineerAgent",
]
