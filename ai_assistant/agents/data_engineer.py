"""
Data Engineer Agent.

This module provides an AI agent specialized in data engineering tasks
within Databricks environments. The agent can create pipelines, optimize
queries, design schemas, and manage data infrastructure.
"""

from typing import Optional, List, Dict, Any

from .base import ReActAgent, AgentMemory
from .tools import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolStatus,
    SQLExecutorTool,
    TableInfoTool,
    DataProfilerTool,
    CodeGeneratorTool,
    FileReaderTool
)


class DDLGeneratorTool(Tool):
    """
    Generates DDL statements for Unity Catalog.

    Uses AI to create CREATE TABLE, ALTER TABLE, and other
    DDL statements based on requirements.
    """

    name = "ddl_generator"
    description = (
        "Generate DDL statements (CREATE TABLE, ALTER TABLE, etc.) "
        "for Unity Catalog based on natural language requirements."
    )
    parameters = [
        ToolParameter(
            name="requirement",
            description="Description of what DDL to generate",
            type="string",
            required=True
        ),
        ToolParameter(
            name="catalog",
            description="Target catalog name",
            type="string",
            required=False
        ),
        ToolParameter(
            name="schema",
            description="Target schema name",
            type="string",
            required=False
        )
    ]

    def __init__(self, ai_client: Any):
        super().__init__()
        self.ai_client = ai_client

    def run(self, **kwargs: Any) -> ToolResult:
        """Generate DDL statement."""
        requirement: str = kwargs.get("requirement", "")
        catalog: Optional[str] = kwargs.get("catalog")
        schema: Optional[str] = kwargs.get("schema")
        prompt = f"""Generate a DDL statement for Unity Catalog:

Requirement: {requirement}

{f'Target Catalog: {catalog}' if catalog else ''}
{f'Target Schema: {schema}' if schema else ''}

Guidelines:
- Use Delta Lake format
- Include appropriate TBLPROPERTIES for optimization
- Add table and column comments
- Use liquid clustering (CLUSTER BY) when appropriate
- Include governance tags (owner, pii, sensitivity)

Generate only the SQL DDL statement with comments."""

        try:
            ddl = self.ai_client.generate(
                prompt,
                system_instruction="You are a Unity Catalog DDL expert."
            )

            return ToolResult(ToolStatus.SUCCESS, output=ddl)
        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class PipelineGeneratorTool(Tool):
    """
    Generates pipeline code for DLT or PySpark.

    Creates complete pipeline code based on requirements.
    """

    name = "pipeline_generator"
    description = (
        "Generate data pipeline code (DLT or PySpark) "
        "based on natural language requirements."
    )
    parameters = [
        ToolParameter(
            name="requirement",
            description="Description of the pipeline to generate",
            type="string",
            required=True
        ),
        ToolParameter(
            name="pipeline_type",
            description="Type of pipeline to generate",
            type="string",
            required=False,
            default="dlt",
            enum=["dlt", "pyspark", "streaming"]
        ),
        ToolParameter(
            name="source_table",
            description="Source table or path",
            type="string",
            required=False
        ),
        ToolParameter(
            name="target_table",
            description="Target table",
            type="string",
            required=False
        )
    ]

    def __init__(self, ai_client: Any):
        super().__init__()
        self.ai_client = ai_client

    def run(self, **kwargs: Any) -> ToolResult:
        """Generate pipeline code."""
        requirement: str = kwargs.get("requirement", "")
        pipeline_type: str = kwargs.get("pipeline_type", "dlt")
        source_table: Optional[str] = kwargs.get("source_table")
        target_table: Optional[str] = kwargs.get("target_table")
        prompt = f"""Generate a {pipeline_type.upper()} pipeline:

Requirement: {requirement}

{f'Source: {source_table}' if source_table else ''}
{f'Target: {target_table}' if target_table else ''}

Guidelines:
- Include proper error handling
- Add data quality expectations/checks
- Use best practices for {pipeline_type}
- Include comprehensive comments
- Handle schema evolution

Generate only the Python code."""

        system = """You are an expert Databricks pipeline engineer.
Generate production-ready pipeline code."""

        try:
            code = self.ai_client.generate(prompt, system_instruction=system)
            return ToolResult(ToolStatus.SUCCESS, output=code)
        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class QueryOptimizerTool(Tool):
    """
    Analyzes and optimizes SQL queries.

    Provides optimization suggestions and rewrites queries
    for better performance.
    """

    name = "query_optimizer"
    description = (
        "Analyze a SQL query and provide optimization suggestions. "
        "Can also rewrite the query for better performance."
    )
    parameters = [
        ToolParameter(
            name="query",
            description="SQL query to optimize",
            type="string",
            required=True
        ),
        ToolParameter(
            name="context",
            description="Additional context (table sizes, indexes, etc.)",
            type="string",
            required=False
        )
    ]

    def __init__(self, ai_client: Any, spark: Any = None):
        super().__init__()
        self.ai_client = ai_client
        self.spark = spark

    def run(self, **kwargs: Any) -> ToolResult:
        """Optimize a SQL query."""
        query: str = kwargs.get("query", "")
        context: Optional[str] = kwargs.get("context")
        # Get query plan if spark available
        plan_info = ""
        if self.spark:
            try:
                df = self.spark.sql(f"EXPLAIN EXTENDED {query}")
                plan_info = df.collect()[0][0]
            except Exception:
                pass

        prompt = f"""Optimize this Spark SQL query:

```sql
{query}
```

{f'Query Plan:{chr(10)}{plan_info}' if plan_info else ''}
{f'Context: {context}' if context else ''}

Provide:
1. Issues identified in the current query
2. Optimization recommendations
3. Optimized version of the query
4. Expected performance improvement"""

        try:
            analysis = self.ai_client.generate(
                prompt,
                system_instruction="You are a Spark SQL optimization expert."
            )
            return ToolResult(ToolStatus.SUCCESS, output=analysis)
        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class DataEngineerAgent(ReActAgent):
    """
    AI agent specialized in data engineering tasks.

    This agent can:
    - Design and create table schemas
    - Generate data pipelines (DLT, PySpark)
    - Optimize SQL queries
    - Implement data quality checks
    - Migrate data between systems
    - Set up Unity Catalog governance

    Args:
        ai_client: AI client for LLM calls
        spark: SparkSession for data access
        tools: Additional tools
        catalog: Default Unity Catalog name
        schema: Default schema name
        verbose: Whether to print intermediate steps

    Example:
        >>> agent = DataEngineerAgent(
        ...     ai_client=assistant.claude,
        ...     spark=spark,
        ...     catalog="enterprise",
        ...     verbose=True
        ... )
        >>> result = agent.run(
        ...     "Create a medallion architecture for customer data"
        ... )
    """

    ENGINEER_SYSTEM_PROMPT = """You are an expert Data Engineer working with Databricks.

Your role is to help design and implement data infrastructure by:
1. Designing optimal table schemas for Unity Catalog
2. Creating efficient data pipelines (DLT, PySpark, Streaming)
3. Optimizing SQL queries and Spark jobs
4. Implementing data quality checks and expectations
5. Setting up data governance and security
6. Managing data migrations and transformations

When engineering solutions:
- Follow Unity Catalog best practices (three-level namespace)
- Use Delta Lake features (liquid clustering, Z-ORDER, etc.)
- Implement proper error handling and monitoring
- Consider data quality from the start
- Document all code and schemas
- Think about maintainability and scalability

For each step, respond in this format:

Thought: [Your reasoning about the engineering task]
Action: [The tool to use, or "Final Answer" if you're done]
Action Input: [Tool input as JSON, or your final deliverable]

Available Tools:
{tool_descriptions}

When you have completed the engineering task, use "Final Answer" as the Action and provide:
1. The solution (DDL, code, or recommendations)
2. Explanation of design decisions
3. Any prerequisites or dependencies
4. Next steps for implementation"""

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
        # Create default engineer tools
        default_tools = [
            SQLExecutorTool(
                spark,
                allowed_operations=["SELECT", "SHOW", "DESCRIBE", "EXPLAIN", "CREATE", "ALTER"]
            ),
            TableInfoTool(spark),
            DataProfilerTool(spark),
            DDLGeneratorTool(ai_client),
            PipelineGeneratorTool(ai_client),
            QueryOptimizerTool(ai_client, spark),
            CodeGeneratorTool(ai_client)
        ]

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
        """Create engineer-specific system prompt."""
        prompt = self.ENGINEER_SYSTEM_PROMPT.format(
            tool_descriptions=self.get_tool_descriptions()
        )

        if self.catalog:
            prompt += f"\n\nDefault catalog: {self.catalog}"
        if self.schema:
            prompt += f"\nDefault schema: {self.schema}"

        return prompt

    def create_table(
        self,
        description: str,
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a table based on description.

        Args:
            description: Natural language description of the table
            table_name: Optional specific table name

        Returns:
            Dictionary with DDL and execution result
        """
        task = f"Create a table: {description}"
        if table_name:
            task += f"\nTable name should be: {table_name}"

        context = {
            "catalog": self.catalog,
            "schema": self.schema
        }

        result = self.run(task, context)

        return {
            "description": description,
            "ddl": result.output,
            "steps": [s.to_dict() for s in result.steps],
            "success": result.success
        }

    def generate_pipeline(
        self,
        description: str,
        pipeline_type: str = "dlt",
        source: Optional[str] = None,
        target: Optional[str] = None
    ) -> str:
        """
        Generate a data pipeline.

        Args:
            description: Pipeline requirements
            pipeline_type: Type of pipeline (dlt, pyspark, streaming)
            source: Source table/path
            target: Target table

        Returns:
            Generated pipeline code
        """
        task = f"Generate a {pipeline_type} pipeline: {description}"

        context = {
            "pipeline_type": pipeline_type
        }
        if source:
            context["source"] = source
        if target:
            context["target"] = target

        result = self.run(task, context)
        return result.output or "Unable to generate pipeline."

    def optimize_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize a SQL query.

        Args:
            query: SQL query to optimize
            context: Additional context

        Returns:
            Optimization results
        """
        task = f"Optimize this SQL query:\n```sql\n{query}\n```"

        result = self.run(task, {"query": query, "context": context})

        return {
            "original_query": query,
            "analysis": result.output,
            "success": result.success
        }

    def migrate_table(
        self,
        source_table: str,
        target_catalog: str,
        target_schema: str,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate migration script for a table.

        Args:
            source_table: Source table (e.g., hive_metastore.db.table)
            target_catalog: Target Unity Catalog
            target_schema: Target schema
            options: Migration options

        Returns:
            Migration script
        """
        task = f"""Generate a migration script to move table {source_table}
to Unity Catalog {target_catalog}.{target_schema}

Include:
- Schema preservation
- Data migration (CTAS or DEEP CLONE)
- Governance tags
- Validation queries"""

        context = {
            "source": source_table,
            "target_catalog": target_catalog,
            "target_schema": target_schema,
            "options": options or {}
        }

        result = self.run(task, context)
        return result.output or "Unable to generate migration script."

    def implement_data_quality(
        self,
        table_name: str,
        requirements: Optional[List[str]] = None
    ) -> str:
        """
        Implement data quality checks for a table.

        Args:
            table_name: Table to add quality checks to
            requirements: Specific quality requirements

        Returns:
            Data quality implementation code
        """
        task = f"Implement data quality checks for table {table_name}"

        if requirements:
            task += f"\n\nRequirements:\n" + "\n".join(f"- {r}" for r in requirements)
        else:
            task += "\n\nAnalyze the table and implement appropriate quality checks."

        result = self.run(task, {"table": table_name})
        return result.output or "Unable to generate data quality checks."


def create_data_engineer(
    ai_client: Any,
    spark: Any,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    verbose: bool = False
) -> DataEngineerAgent:
    """
    Factory function to create a Data Engineer agent.

    Args:
        ai_client: AI client for LLM calls
        spark: SparkSession
        catalog: Default catalog
        schema: Default schema
        verbose: Whether to print steps

    Returns:
        Configured DataEngineerAgent

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.agents import create_data_engineer
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> engineer = create_data_engineer(
        ...     assistant.claude,
        ...     spark,
        ...     catalog="enterprise"
        ... )
        >>> code = engineer.generate_pipeline(
        ...     "ETL pipeline for customer orders",
        ...     pipeline_type="dlt"
        ... )
    """
    return DataEngineerAgent(
        ai_client=ai_client,
        spark=spark,
        catalog=catalog,
        schema=schema,
        verbose=verbose
    )
