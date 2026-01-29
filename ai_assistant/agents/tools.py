"""
Tools for AI Agents.

This module provides the tool infrastructure for agents,
including base classes and common tool implementations.

Tools allow agents to interact with external systems like
databases, file systems, and code execution environments.
"""

import json
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum


class ToolStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Attributes:
        status: Execution status
        output: Tool output (string or structured data)
        error: Error message if status is ERROR
        metadata: Additional execution metadata
    """
    status: ToolStatus
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        """String representation for agent consumption."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, indent=2, default=str)
        return f"Error: {self.error}"


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.

    Attributes:
        name: Parameter name
        description: Parameter description
        type: Parameter type (string, integer, boolean, etc.)
        required: Whether parameter is required
        default: Default value if not provided
        enum: List of allowed values
    """
    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-style function schema."""
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


class Tool(ABC):
    """
    Abstract base class for agent tools.

    Tools are callable units that agents can use to interact
    with external systems and perform actions.

    Attributes:
        name: Tool identifier
        description: Human-readable description
        parameters: List of tool parameters

    Example:
        >>> class MyTool(Tool):
        ...     name = "my_tool"
        ...     description = "Does something useful"
        ...
        ...     def run(self, **kwargs) -> ToolResult:
        ...         return ToolResult(ToolStatus.SUCCESS, "Done!")
    """

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: List[ToolParameter] = []

    def __init__(self, **config):
        """Initialize tool with configuration."""
        self.config = config

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        Run the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def validate_params(self, **kwargs) -> Optional[str]:
        """
        Validate input parameters.

        Returns:
            Error message if validation fails, None otherwise
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                if param.default is None:
                    return f"Missing required parameter: {param.name}"

            if param.name in kwargs and param.enum:
                if kwargs[param.name] not in param.enum:
                    return (
                        f"Invalid value for {param.name}. "
                        f"Must be one of: {param.enum}"
                    )

        return None

    def __call__(self, **kwargs) -> ToolResult:
        """Run tool with validation."""
        error = self.validate_params(**kwargs)
        if error:
            return ToolResult(ToolStatus.ERROR, error=error)

        try:
            return self.run(**kwargs)
        except Exception as e:
            return ToolResult(
                ToolStatus.ERROR,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={"traceback": traceback.format_exc()}
            )

    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class SQLExecutorTool(Tool):
    """
    Runs SQL queries against Spark/Databricks.

    This tool allows agents to run SQL queries and retrieve
    results from Unity Catalog tables.

    Args:
        spark: SparkSession instance
        max_rows: Maximum rows to return
        timeout_seconds: Query timeout

    Example:
        >>> tool = SQLExecutorTool(spark)
        >>> result = tool(query="SELECT * FROM my_table LIMIT 10")
    """

    name = "sql_executor"
    description = (
        "Run a SQL query against the Databricks lakehouse. "
        "Use this to query tables, run aggregations, and analyze data. "
        "Always include LIMIT clause to avoid returning too many rows."
    )
    parameters = [
        ToolParameter(
            name="query",
            description="The SQL query to run",
            type="string",
            required=True
        )
    ]

    def __init__(
        self,
        spark: Any,
        max_rows: int = 100,
        timeout_seconds: int = 300,
        allowed_operations: Optional[List[str]] = None
    ):
        super().__init__()
        self.spark = spark
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.allowed_operations = allowed_operations or [
            "SELECT", "SHOW", "DESCRIBE", "EXPLAIN"
        ]

    def run(self, **kwargs: Any) -> ToolResult:
        """Run SQL query and return results."""
        query: str = kwargs.get("query", "")
        # Validate query (basic safety check)
        query_upper = query.strip().upper()
        operation = query_upper.split()[0] if query_upper else ""

        if operation not in self.allowed_operations:
            return ToolResult(
                ToolStatus.PERMISSION_DENIED,
                error=f"Operation '{operation}' not allowed. "
                      f"Allowed: {self.allowed_operations}"
            )

        try:
            # Run query
            df = self.spark.sql(query)

            # Get results as pandas
            pdf = df.limit(self.max_rows).toPandas()

            # Format output
            output = {
                "columns": list(pdf.columns),
                "row_count": len(pdf),
                "total_rows": df.count() if len(pdf) == self.max_rows else len(pdf),
                "data": pdf.to_dict(orient="records"),
                "preview": pdf.to_string()
            }

            return ToolResult(
                ToolStatus.SUCCESS,
                output=output,
                metadata={"query": query}
            )

        except Exception as e:
            return ToolResult(
                ToolStatus.ERROR,
                error=str(e),
                metadata={"query": query}
            )


class TableInfoTool(Tool):
    """
    Retrieves information about tables in Unity Catalog.

    This tool provides schema information, statistics, and
    metadata about tables.

    Args:
        spark: SparkSession instance
    """

    name = "table_info"
    description = (
        "Get detailed information about a table including schema, "
        "statistics, and properties. Use this before querying to "
        "understand the table structure."
    )
    parameters = [
        ToolParameter(
            name="table_name",
            description="Full table name (catalog.schema.table)",
            type="string",
            required=True
        ),
        ToolParameter(
            name="include_sample",
            description="Whether to include sample data",
            type="boolean",
            required=False,
            default=True
        )
    ]

    def __init__(self, spark: Any):
        super().__init__()
        self.spark = spark

    def run(self, **kwargs: Any) -> ToolResult:
        """Get table information."""
        table_name: str = kwargs.get("table_name", "")
        include_sample: bool = kwargs.get("include_sample", True)
        try:
            info: Dict[str, Any] = {
                "table_name": table_name,
                "schema": [],
                "properties": {},
                "statistics": {}
            }

            # Get schema
            desc = self.spark.sql(f"DESCRIBE TABLE EXTENDED {table_name}")
            desc_rows = desc.collect()

            in_schema = True
            for row in desc_rows:
                col_name = row[0].strip() if row[0] else ""
                col_type = row[1] if row[1] else ""
                col_comment = row[2] if len(row) > 2 and row[2] else ""

                if col_name == "":
                    in_schema = False
                    continue

                if in_schema and not col_name.startswith("#"):
                    info["schema"].append({
                        "name": col_name,
                        "type": col_type,
                        "comment": col_comment
                    })
                elif col_name in ["Location", "Provider", "Owner", "Comment"]:
                    info["properties"][col_name.lower()] = col_type

            # Get row count
            try:
                count = self.spark.sql(
                    f"SELECT COUNT(*) as cnt FROM {table_name}"
                ).collect()[0][0]
                info["statistics"]["row_count"] = count
            except Exception:
                pass

            # Get sample data
            if include_sample:
                try:
                    sample = self.spark.sql(
                        f"SELECT * FROM {table_name} LIMIT 5"
                    ).toPandas()
                    info["sample_data"] = sample.to_dict(orient="records")
                except Exception as e:
                    info["sample_data"] = f"Error: {str(e)}"

            return ToolResult(ToolStatus.SUCCESS, output=info)

        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class DataProfilerTool(Tool):
    """
    Profiles data in a table or DataFrame.

    Generates statistics, distribution information, and
    data quality metrics.

    Args:
        spark: SparkSession instance
    """

    name = "data_profiler"
    description = (
        "Profile data in a table to understand distributions, "
        "null rates, unique values, and statistical summaries. "
        "Use this for data quality assessment and exploration."
    )
    parameters = [
        ToolParameter(
            name="table_name",
            description="Table to profile (catalog.schema.table)",
            type="string",
            required=True
        ),
        ToolParameter(
            name="columns",
            description="Specific columns to profile (comma-separated), or 'all'",
            type="string",
            required=False,
            default="all"
        ),
        ToolParameter(
            name="sample_size",
            description="Number of rows to sample for profiling",
            type="integer",
            required=False,
            default=10000
        )
    ]

    def __init__(self, spark: Any):
        super().__init__()
        self.spark = spark

    def run(self, **kwargs: Any) -> ToolResult:
        """Profile table data."""
        table_name: str = kwargs.get("table_name", "")
        columns: str = kwargs.get("columns", "all")
        sample_size: int = kwargs.get("sample_size", 10000)
        try:
            # Load data
            df = self.spark.sql(f"SELECT * FROM {table_name} LIMIT {sample_size}")

            # Determine columns to profile
            if columns == "all":
                cols_to_profile = df.columns
            else:
                cols_to_profile = [c.strip() for c in columns.split(",")]

            profile = {
                "table_name": table_name,
                "sample_size": sample_size,
                "columns": {}
            }

            # Get pandas for easier profiling
            pdf = df.toPandas()

            for col in cols_to_profile:
                if col not in pdf.columns:
                    continue

                col_profile = {
                    "dtype": str(pdf[col].dtype),
                    "null_count": int(pdf[col].isna().sum()),
                    "null_rate": float(pdf[col].isna().mean()),
                    "unique_count": int(pdf[col].nunique()),
                    "unique_rate": float(pdf[col].nunique() / len(pdf)) if len(pdf) > 0 else 0
                }

                # Numeric statistics
                if pdf[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    col_profile.update({
                        "min": float(pdf[col].min()) if not pdf[col].isna().all() else None,
                        "max": float(pdf[col].max()) if not pdf[col].isna().all() else None,
                        "mean": float(pdf[col].mean()) if not pdf[col].isna().all() else None,
                        "median": float(pdf[col].median()) if not pdf[col].isna().all() else None,
                        "std": float(pdf[col].std()) if not pdf[col].isna().all() else None
                    })

                # String statistics
                elif pdf[col].dtype == 'object':
                    non_null = pdf[col].dropna()
                    if len(non_null) > 0:
                        col_profile.update({
                            "min_length": int(non_null.str.len().min()),
                            "max_length": int(non_null.str.len().max()),
                            "avg_length": float(non_null.str.len().mean()),
                            "top_values": non_null.value_counts().head(5).to_dict()
                        })

                profile["columns"][col] = col_profile

            return ToolResult(ToolStatus.SUCCESS, output=profile)

        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class CodeGeneratorTool(Tool):
    """
    Generates code using AI.

    This tool uses the AI client to generate code based on
    natural language descriptions.

    Args:
        ai_client: AI client for generation
    """

    name = "code_generator"
    description = (
        "Generate Python/PySpark code for a given task. "
        "Describe what you need and get production-ready code."
    )
    parameters = [
        ToolParameter(
            name="task_description",
            description="Description of what the code should do",
            type="string",
            required=True
        ),
        ToolParameter(
            name="language",
            description="Programming language",
            type="string",
            required=False,
            default="python",
            enum=["python", "sql", "scala"]
        ),
        ToolParameter(
            name="context",
            description="Additional context (table schemas, requirements)",
            type="string",
            required=False
        )
    ]

    def __init__(self, ai_client: Any):
        super().__init__()
        self.ai_client = ai_client

    def run(self, **kwargs: Any) -> ToolResult:
        """Generate code."""
        task_description: str = kwargs.get("task_description", "")
        language: str = kwargs.get("language", "python")
        context: Optional[str] = kwargs.get("context")
        try:
            prompt = f"""Generate {language} code for the following task:

Task: {task_description}

{f'Context: {context}' if context else ''}

Requirements:
- Write clean, production-ready code
- Include necessary imports
- Add comments explaining the code
- Handle errors appropriately

Provide only the code, no explanations."""

            system = """You are an expert programmer specializing in:
- Apache Spark and PySpark
- Databricks and Delta Lake
- Data engineering best practices

Generate clean, efficient code."""

            code = self.ai_client.generate(
                prompt,
                system_instruction=system
            )

            # Extract code from response (handle markdown code blocks)
            if "```" in code:
                # Extract code between triple backticks
                pattern = r"```(?:\w+)?\n(.*?)```"
                matches = re.findall(pattern, code, re.DOTALL)
                if matches:
                    code = matches[0]

            return ToolResult(
                ToolStatus.SUCCESS,
                output=code.strip(),
                metadata={"language": language}
            )

        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class FileReaderTool(Tool):
    """
    Reads files from DBFS or cloud storage.

    Args:
        spark: SparkSession instance
        dbutils: Databricks utilities (optional)
    """

    name = "file_reader"
    description = (
        "Read content from files in DBFS, cloud storage, or local paths. "
        "Supports text, JSON, and CSV files."
    )
    parameters = [
        ToolParameter(
            name="path",
            description="File path (dbfs:/..., s3://..., /Workspace/...)",
            type="string",
            required=True
        ),
        ToolParameter(
            name="format",
            description="File format",
            type="string",
            required=False,
            default="text",
            enum=["text", "json", "csv"]
        ),
        ToolParameter(
            name="max_lines",
            description="Maximum lines to read (for text/csv)",
            type="integer",
            required=False,
            default=100
        )
    ]

    def __init__(self, spark: Any = None, dbutils: Any = None):
        super().__init__()
        self.spark = spark
        self.dbutils = dbutils

    def run(self, **kwargs: Any) -> ToolResult:
        """Read file content."""
        path: str = kwargs.get("path", "")
        format: str = kwargs.get("format", "text")
        max_lines: int = kwargs.get("max_lines", 100)
        try:
            if format == "text":
                if self.dbutils:
                    content = self.dbutils.fs.head(path, max_lines * 1000)
                    lines = content.split("\n")[:max_lines]
                    return ToolResult(
                        ToolStatus.SUCCESS,
                        output="\n".join(lines)
                    )
                else:
                    with open(path, 'r') as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= max_lines:
                                break
                            lines.append(line.rstrip())
                        return ToolResult(
                            ToolStatus.SUCCESS,
                            output="\n".join(lines)
                        )

            elif format == "json":
                if self.spark:
                    df = self.spark.read.json(path)
                    data = df.limit(max_lines).toPandas().to_dict(orient="records")
                    return ToolResult(ToolStatus.SUCCESS, output=data)
                else:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        return ToolResult(ToolStatus.SUCCESS, output=data)

            elif format == "csv":
                if self.spark:
                    df = self.spark.read.option("header", "true").csv(path)
                    data = df.limit(max_lines).toPandas().to_dict(orient="records")
                    return ToolResult(ToolStatus.SUCCESS, output=data)
                else:
                    import csv
                    with open(path, 'r') as f:
                        reader = csv.DictReader(f)
                        data = [row for i, row in enumerate(reader) if i < max_lines]
                        return ToolResult(ToolStatus.SUCCESS, output=data)

            else:
                return ToolResult(
                    ToolStatus.ERROR,
                    error=f"Unsupported format: {format}"
                )

        except Exception as e:
            return ToolResult(ToolStatus.ERROR, error=str(e))


class PythonREPLTool(Tool):
    """
    Runs Python code in a controlled environment.

    This tool runs Python code with safety checks.
    Only enabled for trusted environments.

    Args:
        spark: SparkSession (available as 'spark' in code)
        allowed_modules: List of allowed import modules
        timeout_seconds: Timeout
    """

    name = "python_repl"
    description = (
        "Run Python code and return the result. "
        "The code has access to 'spark' for Spark operations. "
        "Use print() for output or assign to 'result' variable."
    )
    parameters = [
        ToolParameter(
            name="code",
            description="Python code to run",
            type="string",
            required=True
        )
    ]

    def __init__(
        self,
        spark: Any = None,
        allowed_modules: Optional[List[str]] = None,
        timeout_seconds: int = 60
    ):
        super().__init__()
        self.spark = spark
        self.allowed_modules = allowed_modules or [
            "pandas", "numpy", "json", "re", "datetime",
            "collections", "itertools", "functools"
        ]
        self.timeout_seconds = timeout_seconds

    def run(self, **kwargs: Any) -> ToolResult:
        """Run Python code."""
        code: str = kwargs.get("code", "")
        try:
            # Basic safety patterns to reject
            dangerous_patterns = [
                r'\bos\.system\b',
                r'\bsubprocess\b',
                r'\b__import__\b',
                r'\bopen\s*\([^)]*["\']w',  # write mode
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return ToolResult(
                        ToolStatus.PERMISSION_DENIED,
                        error="Potentially unsafe code pattern detected"
                    )

            # Create controlled environment
            run_globals = {
                "spark": self.spark,
                "result": None,
                "__builtins__": __builtins__
            }

            # Capture output
            import io
            import sys
            output_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = output_buffer

            try:
                # Run the code
                compiled = compile(code, '<string>', 'exec')
                exec(compiled, run_globals)  # noqa: S102
            finally:
                sys.stdout = old_stdout

            # Get output
            printed_output = output_buffer.getvalue()
            result = run_globals.get("result")

            if result is not None:
                output = result
            elif printed_output:
                output = printed_output
            else:
                output = "Code ran successfully (no output)"

            return ToolResult(
                ToolStatus.SUCCESS,
                output=output,
                metadata={"code": code}
            )

        except Exception as e:
            return ToolResult(
                ToolStatus.ERROR,
                error=f"{type(e).__name__}: {str(e)}",
                metadata={"traceback": traceback.format_exc()}
            )


class ToolRegistry:
    """
    Registry for managing available tools.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(SQLExecutorTool(spark))
        >>> tool = registry.get("sql_executor")
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def run_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Run a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                ToolStatus.ERROR,
                error=f"Tool not found: {tool_name}"
            )
        return tool(**kwargs)


def create_databricks_tools(
    spark: Any,
    ai_client: Any = None,
    dbutils: Any = None,
    enable_code_runner: bool = False
) -> ToolRegistry:
    """
    Factory function to create common Databricks tools.

    Args:
        spark: SparkSession
        ai_client: AI client for code generation
        dbutils: Databricks utilities
        enable_code_runner: Whether to enable Python runner tool

    Returns:
        ToolRegistry with common tools
    """
    registry = ToolRegistry()

    # SQL tools
    registry.register(SQLExecutorTool(spark))
    registry.register(TableInfoTool(spark))
    registry.register(DataProfilerTool(spark))

    # File tools
    registry.register(FileReaderTool(spark, dbutils))

    # AI-powered tools
    if ai_client:
        registry.register(CodeGeneratorTool(ai_client))

    # Code runner (use with caution)
    if enable_code_runner:
        registry.register(PythonREPLTool(spark))

    return registry
