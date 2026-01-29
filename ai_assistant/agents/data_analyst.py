"""
Data Analyst Agent.

This module provides an AI agent specialized in data analysis tasks
within Databricks environments. The agent can explore data, run queries,
generate insights, and create visualizations.
"""

from typing import Optional, List, Dict, Any

from .base import ReActAgent, AgentMemory
from .tools import (
    Tool,
    SQLExecutorTool,
    TableInfoTool,
    DataProfilerTool,
    CodeGeneratorTool
)


class DataAnalystAgent(ReActAgent):
    """
    AI agent specialized in data analysis.

    This agent can:
    - Explore and understand data schemas
    - Write and execute SQL queries
    - Profile data for quality assessment
    - Generate statistical insights
    - Create data visualizations
    - Answer questions about data

    Args:
        ai_client: AI client for LLM calls
        spark: SparkSession for data access
        tools: Additional tools (SQL/profiler tools added automatically)
        catalog: Default Unity Catalog name
        schema: Default schema name
        verbose: Whether to print intermediate steps

    Example:
        >>> agent = DataAnalystAgent(
        ...     ai_client=assistant.claude,
        ...     spark=spark,
        ...     catalog="analytics",
        ...     verbose=True
        ... )
        >>> result = agent.run(
        ...     "What are the top 10 products by revenue?",
        ...     context={"table": "sales.transactions"}
        ... )
        >>> print(result.output)
    """

    ANALYST_SYSTEM_PROMPT = """You are an expert Data Analyst working with a Databricks lakehouse.

Your role is to help users understand their data by:
1. Exploring table schemas and understanding data structures
2. Writing efficient SQL queries to answer questions
3. Profiling data to assess quality and distributions
4. Generating statistical insights and summaries
5. Identifying patterns and anomalies
6. Providing clear, actionable recommendations

When analyzing data:
- Always start by understanding the table schema before writing queries
- Use appropriate SQL optimizations (filters, limits, aggregations)
- Explain your findings in business-friendly language
- Highlight any data quality issues you discover
- Suggest follow-up analyses when appropriate

For each step, respond in this format:

Thought: [Your reasoning about what analysis to perform]
Action: [The tool to use, or "Final Answer" if you're done]
Action Input: [Tool input as JSON, or your final answer with insights]

Available Tools:
{tool_descriptions}

When you have completed the analysis, use "Final Answer" as the Action and provide:
1. A clear answer to the user's question
2. Key insights discovered
3. Any data quality observations
4. Recommendations for further analysis if relevant"""

    def __init__(
        self,
        ai_client: Any,
        spark: Any,
        tools: Optional[List[Tool]] = None,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        verbose: bool = False,
        max_iterations: int = 15
    ):
        # Create default analyst tools
        default_tools = [
            SQLExecutorTool(spark),
            TableInfoTool(spark),
            DataProfilerTool(spark),
        ]

        # Add code generator if ai_client supports it
        if hasattr(ai_client, 'generate'):
            default_tools.append(CodeGeneratorTool(ai_client))

        # Combine with custom tools
        all_tools = default_tools + (tools or [])

        super().__init__(
            ai_client=ai_client,
            tools=all_tools,
            max_iterations=max_iterations,
            verbose=verbose
        )

        self.spark = spark
        self.catalog = catalog
        self.schema = schema

    def create_system_prompt(self) -> str:
        """Create analyst-specific system prompt."""
        prompt = self.ANALYST_SYSTEM_PROMPT.format(
            tool_descriptions=self.get_tool_descriptions()
        )

        # Add catalog/schema context
        if self.catalog:
            prompt += f"\n\nDefault catalog: {self.catalog}"
        if self.schema:
            prompt += f"\nDefault schema: {self.schema}"

        return prompt

    def analyze_table(
        self,
        table_name: str,
        questions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a table.

        Args:
            table_name: Table to analyze
            questions: Specific questions to answer

        Returns:
            Dictionary with analysis results
        """
        context = {
            "table": table_name,
            "analysis_type": "comprehensive"
        }

        if questions:
            context["questions"] = questions

        task = f"Perform a comprehensive analysis of the table {table_name}."
        if questions:
            task += f" Specifically answer: {'; '.join(questions)}"

        result = self.run(task, context)

        return {
            "table": table_name,
            "analysis": result.output,
            "steps": [s.to_dict() for s in result.steps],
            "success": result.success
        }

    def answer_question(
        self,
        question: str,
        tables: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Answer a data question.

        Args:
            question: The question to answer
            tables: Relevant tables to query
            context: Additional context

        Returns:
            Answer string
        """
        run_context = {}
        if tables:
            run_context["tables"] = tables
        if context:
            run_context["context"] = context

        result = self.run(question, run_context if run_context else None)
        return result.output or "Unable to answer the question."

    def profile_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Profile data in a table.

        Args:
            table_name: Table to profile
            columns: Specific columns to profile

        Returns:
            Profiling results
        """
        task = f"Profile the data in table {table_name}"
        if columns:
            task += f", focusing on columns: {', '.join(columns)}"

        result = self.run(task, {"table": table_name})

        # Try to extract profiling data from steps
        profile_data = None
        for step in result.steps:
            if step.action == "data_profiler" and step.observation:
                try:
                    import json
                    profile_data = json.loads(step.observation)
                except Exception:
                    pass

        return {
            "table": table_name,
            "profile": profile_data,
            "summary": result.output,
            "success": result.success
        }

    def compare_tables(
        self,
        table1: str,
        table2: str,
        comparison_type: str = "schema"
    ) -> str:
        """
        Compare two tables.

        Args:
            table1: First table name
            table2: Second table name
            comparison_type: Type of comparison (schema, data, statistics)

        Returns:
            Comparison results
        """
        task = f"Compare tables {table1} and {table2}. Focus on {comparison_type} comparison."

        result = self.run(task, {
            "table1": table1,
            "table2": table2,
            "comparison_type": comparison_type
        })

        return result.output or "Unable to complete comparison."


def create_data_analyst(
    ai_client: Any,
    spark: Any,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    verbose: bool = False
) -> DataAnalystAgent:
    """
    Factory function to create a Data Analyst agent.

    Args:
        ai_client: AI client for LLM calls
        spark: SparkSession
        catalog: Default catalog
        schema: Default schema
        verbose: Whether to print steps

    Returns:
        Configured DataAnalystAgent

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.agents import create_data_analyst
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> analyst = create_data_analyst(
        ...     assistant.claude,
        ...     spark,
        ...     catalog="analytics"
        ... )
        >>> result = analyst.answer_question("What's our monthly revenue trend?")
    """
    return DataAnalystAgent(
        ai_client=ai_client,
        spark=spark,
        catalog=catalog,
        schema=schema,
        verbose=verbose
    )
