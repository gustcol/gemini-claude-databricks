"""
Pipeline Generator for Databricks.

This module provides AI-powered pipeline generation capabilities,
including Delta Live Tables (DLT), Workflows, and ETL pipelines.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class PipelineType(Enum):
    """Types of pipelines that can be generated."""
    DLT = "delta_live_tables"
    WORKFLOW = "databricks_workflow"
    ETL = "etl_pipeline"
    STREAMING = "streaming_pipeline"
    MEDALLION = "medallion_architecture"


@dataclass
class TableDefinition:
    """Definition of a table in a pipeline."""
    name: str
    source: str
    transformations: List[str] = field(default_factory=list)
    schema: Optional[Dict[str, str]] = None
    partition_by: Optional[List[str]] = None
    cluster_by: Optional[List[str]] = None
    expectations: Optional[Dict[str, str]] = None
    comment: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""
    name: str
    pipeline_type: PipelineType
    tables: List[TableDefinition] = field(default_factory=list)
    target_catalog: Optional[str] = None
    target_schema: Optional[str] = None
    continuous: bool = False
    development: bool = True
    photon: bool = True
    serverless: bool = False
    notifications: Optional[Dict[str, Any]] = None


class PipelineGenerator:
    """
    AI-powered pipeline generator for Databricks.

    This class uses AI models to generate complete pipeline code
    based on natural language descriptions or configurations.

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.pipelines import PipelineGenerator
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> generator = PipelineGenerator(assistant)
        >>>
        >>> code = generator.generate_dlt_pipeline(
        ...     description="Create a medallion architecture for sales data",
        ...     source_table="raw.sales_events",
        ...     target_catalog="analytics"
        ... )
    """

    def __init__(self, assistant: Any):
        """
        Initialize the pipeline generator.

        Args:
            assistant: AIAssistant instance for AI-powered generation
        """
        self.assistant = assistant
        self._system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for pipeline generation."""
        return """You are an expert Databricks pipeline engineer specializing in:
- Delta Live Tables (DLT) pipeline development
- Databricks Workflows and Jobs
- ETL/ELT pipeline design
- Medallion architecture (Bronze/Silver/Gold)
- Unity Catalog integration
- Data quality with expectations
- Streaming pipelines with Auto Loader

When generating pipeline code:
1. Use best practices for Delta Lake and Unity Catalog
2. Include proper data quality expectations
3. Add comprehensive comments
4. Use efficient transformations
5. Handle schema evolution
6. Implement proper error handling
7. Follow naming conventions for Unity Catalog"""

    def generate_dlt_pipeline(
        self,
        description: str,
        source_table: Optional[str] = None,
        source_path: Optional[str] = None,
        source_format: str = "delta",
        target_catalog: str = "main",
        target_schema: str = "default",
        include_expectations: bool = True,
        include_streaming: bool = False,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a Delta Live Tables pipeline.

        Args:
            description: Natural language description of the pipeline
            source_table: Source table name (Unity Catalog format)
            source_path: Source file path (alternative to table)
            source_format: Format of source data (delta, parquet, json, csv)
            target_catalog: Target Unity Catalog name
            target_schema: Target schema name
            include_expectations: Whether to include data quality expectations
            include_streaming: Whether to generate streaming tables
            model: AI model to use ("gemini" or "claude")

        Returns:
            Generated DLT pipeline code

        Example:
            >>> code = generator.generate_dlt_pipeline(
            ...     description="Process customer orders with data quality checks",
            ...     source_table="bronze.raw_orders",
            ...     target_catalog="analytics",
            ...     target_schema="gold"
            ... )
        """
        prompt = f"""Generate a complete Delta Live Tables (DLT) pipeline in Python:

## Requirements:
{description}

## Source Configuration:
- Source Table: {source_table or 'Not specified'}
- Source Path: {source_path or 'Not specified'}
- Source Format: {source_format}

## Target Configuration:
- Target Catalog: {target_catalog}
- Target Schema: {target_schema}

## Features:
- Include data quality expectations: {include_expectations}
- Use streaming tables: {include_streaming}

## Instructions:
1. Import dlt module
2. Create bronze layer (raw ingestion)
3. Create silver layer (cleaned, validated)
4. Create gold layer (aggregated, business-ready)
5. Add @dlt.table decorators with proper configuration
6. Include @dlt.expect or @dlt.expect_or_drop for data quality
7. Use Unity Catalog three-level namespace
8. Add comprehensive docstrings and comments

Generate only the Python code, no explanations."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_workflow(
        self,
        description: str,
        tasks: Optional[List[Dict[str, Any]]] = None,
        schedule: Optional[str] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a Databricks Workflow definition.

        Args:
            description: Natural language description of the workflow
            tasks: List of task definitions
            schedule: Cron schedule expression
            cluster_config: Cluster configuration
            model: AI model to use

        Returns:
            Generated workflow JSON configuration

        Example:
            >>> workflow = generator.generate_workflow(
            ...     description="Daily ETL pipeline with data validation",
            ...     schedule="0 0 6 * * ?",
            ...     tasks=[
            ...         {"name": "extract", "notebook": "/ETL/extract"},
            ...         {"name": "transform", "depends_on": ["extract"]}
            ...     ]
            ... )
        """
        tasks_str = ""
        if tasks:
            tasks_str = f"Tasks: {tasks}"

        prompt = f"""Generate a Databricks Workflow (Job) definition in JSON format:

## Requirements:
{description}

## Configuration:
- Schedule: {schedule or 'Manual trigger'}
- {tasks_str if tasks else 'Auto-generate appropriate tasks'}
- Cluster: {cluster_config or 'Use serverless compute'}

## Instructions:
1. Create a complete workflow JSON definition
2. Include task dependencies
3. Add error handling and retries
4. Configure notifications if appropriate
5. Use job clusters for efficiency
6. Add tags for organization

Generate the JSON configuration with comments explaining each section."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_etl_pipeline(
        self,
        description: str,
        source_config: Dict[str, Any],
        target_config: Dict[str, Any],
        transformations: Optional[List[str]] = None,
        schedule: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate an ETL pipeline using PySpark.

        Args:
            description: Natural language description
            source_config: Source configuration (table, path, format)
            target_config: Target configuration (catalog, schema, table)
            transformations: List of transformations to apply
            schedule: Optional schedule for the pipeline
            model: AI model to use

        Returns:
            Generated ETL pipeline code

        Example:
            >>> etl_code = generator.generate_etl_pipeline(
            ...     description="Load and transform customer data",
            ...     source_config={"path": "/data/customers", "format": "json"},
            ...     target_config={"catalog": "prod", "schema": "customers", "table": "dim_customer"}
            ... )
        """
        prompt = f"""Generate a production-ready ETL pipeline in PySpark:

## Requirements:
{description}

## Source Configuration:
{source_config}

## Target Configuration:
{target_config}

## Transformations:
{transformations or 'Determine appropriate transformations from description'}

## Instructions:
1. Create a complete ETL class with extract, transform, load methods
2. Use Delta Lake for target tables
3. Implement MERGE for upserts
4. Add comprehensive logging
5. Include error handling and retry logic
6. Add data quality checks
7. Use Unity Catalog three-level namespace
8. Include audit columns (load_timestamp, source_file, etc.)
9. Make the code modular and testable

Generate only the Python code."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_medallion_architecture(
        self,
        description: str,
        source_path: str,
        source_format: str,
        catalog: str,
        include_dlt: bool = True,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a complete Medallion Architecture pipeline.

        Args:
            description: Description of the data domain
            source_path: Path to raw data
            source_format: Format of source data
            catalog: Unity Catalog name
            include_dlt: Whether to use DLT (vs plain PySpark)
            model: AI model to use

        Returns:
            Generated medallion architecture code

        Example:
            >>> code = generator.generate_medallion_architecture(
            ...     description="E-commerce sales data pipeline",
            ...     source_path="/mnt/landing/sales/",
            ...     source_format="json",
            ...     catalog="ecommerce"
            ... )
        """
        pipeline_type = "Delta Live Tables" if include_dlt else "PySpark with Delta Lake"

        prompt = f"""Generate a complete Medallion Architecture ({pipeline_type}):

## Data Domain:
{description}

## Source:
- Path: {source_path}
- Format: {source_format}

## Target Catalog: {catalog}

## Architecture:
### Bronze Layer (Raw):
- Ingest raw data as-is
- Add metadata columns (ingestion_time, source_file)
- Use Auto Loader for incremental processing

### Silver Layer (Cleaned):
- Data type standardization
- Null handling
- Deduplication
- Data quality expectations
- Business key identification

### Gold Layer (Business):
- Aggregations and metrics
- Dimension tables (SCD Type 2 if appropriate)
- Fact tables
- Business-ready views

## Instructions:
1. Create complete {pipeline_type} code
2. Include all three layers
3. Add comprehensive data quality checks
4. Use proper naming conventions
5. Include comments explaining each transformation
6. Handle schema evolution
7. Optimize for query performance

Generate only the Python code."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_streaming_pipeline(
        self,
        description: str,
        source_config: Dict[str, Any],
        target_config: Dict[str, Any],
        watermark_column: Optional[str] = None,
        watermark_delay: str = "10 minutes",
        checkpoint_location: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a Structured Streaming pipeline.

        Args:
            description: Natural language description
            source_config: Streaming source configuration
            target_config: Target configuration
            watermark_column: Column for watermarking
            watermark_delay: Watermark delay threshold
            checkpoint_location: Checkpoint location
            model: AI model to use

        Returns:
            Generated streaming pipeline code

        Example:
            >>> streaming_code = generator.generate_streaming_pipeline(
            ...     description="Real-time clickstream processing",
            ...     source_config={"format": "kafka", "topic": "clicks"},
            ...     target_config={"catalog": "realtime", "table": "click_events"}
            ... )
        """
        prompt = f"""Generate a Spark Structured Streaming pipeline:

## Requirements:
{description}

## Source Configuration:
{source_config}

## Target Configuration:
{target_config}

## Streaming Configuration:
- Watermark Column: {watermark_column or 'event_time'}
- Watermark Delay: {watermark_delay}
- Checkpoint Location: {checkpoint_location or '/checkpoints/<table_name>'}

## Instructions:
1. Create a complete streaming pipeline
2. Configure appropriate trigger (processingTime or availableNow)
3. Add watermarking for late data handling
4. Include state management if needed
5. Use foreachBatch for complex writes
6. Add proper error handling
7. Configure checkpointing
8. Implement graceful shutdown
9. Add monitoring and metrics

Generate only the Python code."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def analyze_pipeline(
        self,
        code: str,
        model: Optional[str] = None
    ) -> str:
        """
        Analyze an existing pipeline and provide recommendations.

        Args:
            code: Pipeline code to analyze
            model: AI model to use

        Returns:
            Analysis and recommendations

        Example:
            >>> analysis = generator.analyze_pipeline(existing_code)
            >>> print(analysis)
        """
        prompt = f"""Analyze this Databricks pipeline code and provide recommendations:

```python
{code}
```

Provide analysis covering:
1. **Architecture Assessment**: Is the design optimal?
2. **Performance Issues**: Identify bottlenecks
3. **Data Quality**: Are there adequate checks?
4. **Error Handling**: Is it robust?
5. **Best Practices**: What improvements are recommended?
6. **Unity Catalog**: Is it properly integrated?
7. **Cost Optimization**: Suggestions for reducing costs
8. **Monitoring**: Are metrics and logging adequate?

For each issue found, provide:
- Description of the issue
- Impact (High/Medium/Low)
- Recommended fix with code example"""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def convert_pipeline(
        self,
        code: str,
        source_format: str,
        target_format: str,
        model: Optional[str] = None
    ) -> str:
        """
        Convert a pipeline between different formats.

        Args:
            code: Original pipeline code
            source_format: Source format (e.g., "spark", "dlt", "airflow")
            target_format: Target format
            model: AI model to use

        Returns:
            Converted pipeline code

        Example:
            >>> dlt_code = generator.convert_pipeline(
            ...     spark_code,
            ...     source_format="spark",
            ...     target_format="dlt"
            ... )
        """
        prompt = f"""Convert this pipeline from {source_format} to {target_format}:

```
{code}
```

Requirements:
1. Preserve all business logic
2. Use {target_format} best practices
3. Maintain data quality checks
4. Keep equivalent functionality
5. Add comments explaining the conversion

Generate only the converted code."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )


def create_dlt_template(
    pipeline_name: str,
    catalog: str,
    schema: str,
    include_streaming: bool = False
) -> str:
    """
    Create a basic DLT pipeline template.

    Args:
        pipeline_name: Name of the pipeline
        catalog: Target Unity Catalog
        schema: Target schema
        include_streaming: Whether to include streaming tables

    Returns:
        DLT pipeline template code

    Example:
        >>> template = create_dlt_template("sales_pipeline", "analytics", "sales")
        >>> print(template)
    """
    streaming_decorator = "@dlt.table" if not include_streaming else "@dlt.table(table_properties={'pipelines.autoOptimize.managed': 'true'})"

    template = f'''"""
Delta Live Tables Pipeline: {pipeline_name}

Target: {catalog}.{schema}

This pipeline implements a medallion architecture with:
- Bronze: Raw data ingestion
- Silver: Cleaned and validated data
- Gold: Business-ready aggregations
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *


# =============================================================================
# BRONZE LAYER - Raw Data Ingestion
# =============================================================================

{streaming_decorator}
def bronze_{pipeline_name}():
    """
    Bronze layer: Ingest raw data.

    This table contains raw, unprocessed data with metadata columns
    for tracking data lineage.
    """
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"/checkpoints/{pipeline_name}/schema")
        .load("/path/to/source/data")
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


# =============================================================================
# SILVER LAYER - Cleaned Data
# =============================================================================

@dlt.table(
    comment="Cleaned and validated data",
    table_properties={{
        "quality": "silver",
        "pipelines.autoOptimize.zOrderCols": "id"
    }}
)
@dlt.expect_or_drop("valid_id", "id IS NOT NULL")
@dlt.expect("valid_timestamp", "event_timestamp IS NOT NULL")
def silver_{pipeline_name}():
    """
    Silver layer: Clean and validate data.

    Transformations:
    - Remove duplicates
    - Standardize data types
    - Apply data quality rules
    """
    return (
        dlt.read_stream("bronze_{pipeline_name}")
        .dropDuplicates(["id"])
        .withColumn("processed_timestamp", F.current_timestamp())
    )


# =============================================================================
# GOLD LAYER - Business Aggregations
# =============================================================================

@dlt.table(
    comment="Business-ready aggregated metrics",
    table_properties={{
        "quality": "gold"
    }}
)
def gold_{pipeline_name}_summary():
    """
    Gold layer: Business aggregations.

    Contains pre-computed metrics for analytics and reporting.
    """
    return (
        dlt.read("silver_{pipeline_name}")
        .groupBy("category")
        .agg(
            F.count("*").alias("total_count"),
            F.sum("amount").alias("total_amount"),
            F.avg("amount").alias("avg_amount")
        )
    )
'''

    return template
