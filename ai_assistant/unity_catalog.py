"""
Unity Catalog Integration for AI Assistant.

This module provides AI-powered utilities for working with
Unity Catalog in Databricks, including schema generation,
data governance, and catalog management.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class PrivilegeType(Enum):
    """Unity Catalog privilege types."""
    SELECT = "SELECT"
    MODIFY = "MODIFY"
    CREATE_TABLE = "CREATE TABLE"
    CREATE_VIEW = "CREATE VIEW"
    CREATE_FUNCTION = "CREATE FUNCTION"
    CREATE_MODEL = "CREATE MODEL"
    USAGE = "USAGE"
    ALL_PRIVILEGES = "ALL PRIVILEGES"


class SecurableType(Enum):
    """Unity Catalog securable object types."""
    CATALOG = "CATALOG"
    SCHEMA = "SCHEMA"
    TABLE = "TABLE"
    VIEW = "VIEW"
    FUNCTION = "FUNCTION"
    MODEL = "MODEL"
    VOLUME = "VOLUME"
    EXTERNAL_LOCATION = "EXTERNAL LOCATION"


@dataclass
class ColumnDefinition:
    """Definition of a table column."""
    name: str
    data_type: str
    nullable: bool = True
    comment: Optional[str] = None
    default: Optional[str] = None
    mask: Optional[str] = None  # Column mask function


@dataclass
class TableDefinition:
    """Definition of a Unity Catalog table."""
    catalog: str
    schema: str
    name: str
    columns: List[ColumnDefinition] = field(default_factory=list)
    partition_columns: Optional[List[str]] = None
    cluster_columns: Optional[List[str]] = None
    comment: Optional[str] = None
    properties: Optional[Dict[str, str]] = None
    tags: Optional[Dict[str, str]] = None
    row_filter: Optional[str] = None  # Row-level security

    @property
    def full_name(self) -> str:
        """Get full three-level namespace."""
        return f"{self.catalog}.{self.schema}.{self.name}"


class UnityCatalogHelper:
    """
    AI-powered Unity Catalog helper.

    This class provides utilities for working with Unity Catalog,
    including schema generation, governance setup, and best practices.

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.unity_catalog import UnityCatalogHelper
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> uc_helper = UnityCatalogHelper(assistant)
        >>>
        >>> ddl = uc_helper.generate_table_ddl(
        ...     description="Customer dimension table",
        ...     catalog="analytics",
        ...     schema="dimensions"
        ... )
    """

    def __init__(self, assistant: Any, spark: Any = None):
        """
        Initialize the Unity Catalog helper.

        Args:
            assistant: AIAssistant instance
            spark: SparkSession (optional, for executing commands)
        """
        self.assistant = assistant
        self.spark = spark
        self._system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for Unity Catalog operations."""
        return """You are an expert in Databricks Unity Catalog with deep knowledge of:
- Catalog, schema, and table management
- Data governance and access control
- Row-level and column-level security
- Data lineage and auditing
- Best practices for data organization
- Delta Lake table optimization
- Liquid clustering and Z-ordering
- Table properties and tags

When generating code or DDL:
1. Use three-level namespace (catalog.schema.table)
2. Include proper comments and documentation
3. Apply appropriate security policies
4. Follow naming conventions
5. Optimize for query performance
6. Include data governance tags"""

    def generate_table_ddl(
        self,
        description: str,
        catalog: str,
        schema: str,
        table_name: Optional[str] = None,
        sample_data: Optional[str] = None,
        include_governance: bool = True,
        model: Optional[str] = None
    ) -> str:
        """
        Generate CREATE TABLE DDL from description.

        Args:
            description: Natural language description of the table
            catalog: Target catalog name
            schema: Target schema name
            table_name: Table name (generated if not provided)
            sample_data: Sample data for schema inference
            include_governance: Include governance tags and policies
            model: AI model to use

        Returns:
            Generated DDL statement

        Example:
            >>> ddl = uc_helper.generate_table_ddl(
            ...     description="Customer master data with PII fields",
            ...     catalog="enterprise",
            ...     schema="master_data",
            ...     include_governance=True
            ... )
        """
        prompt = f"""Generate a CREATE TABLE DDL statement for Unity Catalog:

## Requirements:
{description}

## Target Location:
- Catalog: {catalog}
- Schema: {schema}
- Table Name: {table_name or 'Generate appropriate name'}

## Sample Data (if available):
{sample_data or 'No sample data provided'}

## Include Governance: {include_governance}

## Instructions:
1. Generate complete CREATE TABLE statement
2. Use appropriate Delta Lake data types
3. Add column comments
4. Include table comment
5. Add TBLPROPERTIES for optimization
6. {'Include governance tags (pii, sensitivity, owner)' if include_governance else 'Skip governance tags'}
7. Consider partitioning strategy
8. Use liquid clustering if appropriate (CLUSTER BY)

Generate only the DDL statement with comments."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_schema_ddl(
        self,
        description: str,
        catalog: str,
        schema_name: str,
        tables: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate complete schema with multiple tables.

        Args:
            description: Description of the schema/domain
            catalog: Target catalog
            schema_name: Schema name
            tables: List of table descriptions to create
            model: AI model to use

        Returns:
            Complete DDL for schema and tables

        Example:
            >>> ddl = uc_helper.generate_schema_ddl(
            ...     description="E-commerce data domain",
            ...     catalog="ecommerce",
            ...     schema_name="sales",
            ...     tables=["customers", "orders", "products"]
            ... )
        """
        prompt = f"""Generate complete DDL for a Unity Catalog schema:

## Domain Description:
{description}

## Target:
- Catalog: {catalog}
- Schema: {schema_name}

## Tables to Create:
{tables or 'Generate appropriate tables based on domain description'}

## Instructions:
1. Create schema with comment
2. Generate all related tables
3. Include foreign key relationships (as comments - UC doesn't enforce FK)
4. Add proper indexes via CLUSTER BY
5. Include governance tags
6. Follow star schema or normalized design as appropriate
7. Add table and column comments

Generate complete DDL statements."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_access_policy(
        self,
        description: str,
        securable: str,
        securable_type: SecurableType,
        model: Optional[str] = None
    ) -> str:
        """
        Generate access control GRANT/REVOKE statements.

        Args:
            description: Description of access requirements
            securable: Full name of the securable object
            securable_type: Type of securable
            model: AI model to use

        Returns:
            GRANT/REVOKE statements

        Example:
            >>> grants = uc_helper.generate_access_policy(
            ...     description="Data scientists need read access, data engineers need write",
            ...     securable="analytics.sales.orders",
            ...     securable_type=SecurableType.TABLE
            ... )
        """
        prompt = f"""Generate Unity Catalog access control statements:

## Requirements:
{description}

## Securable Object:
- Name: {securable}
- Type: {securable_type.value}

## Instructions:
1. Generate GRANT statements for appropriate roles
2. Include any necessary REVOKE statements
3. Consider principle of least privilege
4. Include ownership transfer if needed
5. Add comments explaining each permission

Use Unity Catalog GRANT syntax."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_row_level_security(
        self,
        table: str,
        description: str,
        model: Optional[str] = None
    ) -> str:
        """
        Generate row-level security (row filter) for a table.

        Args:
            table: Full table name (catalog.schema.table)
            description: Description of access rules
            model: AI model to use

        Returns:
            Row filter function and ALTER TABLE statement

        Example:
            >>> rls = uc_helper.generate_row_level_security(
            ...     table="sales.orders.transactions",
            ...     description="Users can only see their own region's data"
            ... )
        """
        prompt = f"""Generate row-level security for Unity Catalog:

## Table: {table}

## Access Rules:
{description}

## Instructions:
1. Create a SQL UDF function for the row filter
2. Generate ALTER TABLE statement to apply the filter
3. Include examples of how the filter works
4. Consider performance implications
5. Add documentation

Generate the complete solution with SQL statements."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_column_mask(
        self,
        table: str,
        column: str,
        description: str,
        model: Optional[str] = None
    ) -> str:
        """
        Generate column masking function for sensitive data.

        Args:
            table: Full table name
            column: Column to mask
            description: Masking requirements
            model: AI model to use

        Returns:
            Column mask function and ALTER TABLE statement

        Example:
            >>> mask = uc_helper.generate_column_mask(
            ...     table="hr.employees.personal_info",
            ...     column="ssn",
            ...     description="Mask SSN showing only last 4 digits for non-HR users"
            ... )
        """
        prompt = f"""Generate column masking for Unity Catalog:

## Table: {table}
## Column: {column}

## Masking Requirements:
{description}

## Instructions:
1. Create a SQL UDF function for the mask
2. Handle different user groups appropriately
3. Generate ALTER TABLE statement to apply the mask
4. Include test examples
5. Document the masking logic

Generate the complete solution."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_data_lineage_query(
        self,
        table: str,
        direction: str = "both",
        model: Optional[str] = None
    ) -> str:
        """
        Generate queries to explore data lineage.

        Args:
            table: Table to analyze
            direction: "upstream", "downstream", or "both"
            model: AI model to use

        Returns:
            Lineage exploration queries

        Example:
            >>> lineage = uc_helper.generate_data_lineage_query(
            ...     table="gold.sales.daily_metrics",
            ...     direction="upstream"
            ... )
        """
        prompt = f"""Generate Unity Catalog lineage exploration queries:

## Table: {table}
## Direction: {direction}

## Instructions:
1. Query system tables for lineage information
2. Include table lineage (TABLE_LINEAGE)
3. Include column lineage (COLUMN_LINEAGE)
4. Show notebook/job lineage
5. Format output for readability
6. Add explanatory comments

Use system.access.table_lineage and related system tables."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def generate_audit_queries(
        self,
        catalog: Optional[str] = None,
        event_type: Optional[str] = None,
        time_range: str = "7 days",
        model: Optional[str] = None
    ) -> str:
        """
        Generate audit log analysis queries.

        Args:
            catalog: Filter by catalog (optional)
            event_type: Filter by event type (optional)
            time_range: Time range to analyze
            model: AI model to use

        Returns:
            Audit analysis queries

        Example:
            >>> audit = uc_helper.generate_audit_queries(
            ...     catalog="production",
            ...     event_type="TABLE_ACCESS",
            ...     time_range="30 days"
            ... )
        """
        prompt = f"""Generate Unity Catalog audit log queries:

## Filters:
- Catalog: {catalog or 'All catalogs'}
- Event Type: {event_type or 'All events'}
- Time Range: {time_range}

## Required Analysis:
1. Access patterns by user
2. Most accessed tables
3. Failed access attempts
4. Schema changes
5. Permission changes
6. Data exports

Generate queries using system.access.audit table."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def analyze_table(
        self,
        table: str,
        model: Optional[str] = None
    ) -> str:
        """
        Analyze a table and provide optimization recommendations.

        Args:
            table: Full table name
            model: AI model to use

        Returns:
            Analysis and recommendations

        Example:
            >>> analysis = uc_helper.analyze_table("prod.sales.transactions")
        """
        # Get table info if spark is available
        table_info = ""
        if self.spark:
            try:
                desc = self.spark.sql(f"DESCRIBE EXTENDED {table}").collect()
                table_info = "\n".join([f"{row[0]}: {row[1]}" for row in desc])
            except Exception as e:
                table_info = f"Could not retrieve table info: {e}"

        prompt = f"""Analyze this Unity Catalog table and provide recommendations:

## Table: {table}

## Table Information:
{table_info or 'No table information available - provide general best practices'}

## Provide Analysis For:
1. Schema design
2. Partitioning strategy
3. Clustering/Z-ordering
4. Table properties
5. Governance tags
6. Access patterns
7. Performance optimization
8. Storage optimization

Include specific actionable recommendations with SQL examples."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )

    def migrate_table_to_uc(
        self,
        source_table: str,
        target_catalog: str,
        target_schema: str,
        include_data: bool = True,
        model: Optional[str] = None
    ) -> str:
        """
        Generate migration script from Hive metastore to Unity Catalog.

        Args:
            source_table: Source table (hive_metastore.schema.table)
            target_catalog: Target Unity Catalog
            target_schema: Target schema
            include_data: Whether to migrate data
            model: AI model to use

        Returns:
            Migration script

        Example:
            >>> migration = uc_helper.migrate_table_to_uc(
            ...     source_table="hive_metastore.legacy.customers",
            ...     target_catalog="enterprise",
            ...     target_schema="master_data"
            ... )
        """
        prompt = f"""Generate Unity Catalog migration script:

## Source Table: {source_table}
## Target: {target_catalog}.{target_schema}
## Include Data: {include_data}

## Instructions:
1. Create target table with equivalent schema
2. Add proper governance tags
3. {'Include CTAS or DEEP CLONE for data migration' if include_data else 'Create empty table only'}
4. Grant appropriate permissions
5. Validate data integrity
6. Include rollback procedure
7. Add documentation

Generate complete migration script with validation steps."""

        return self.assistant.ask(
            prompt,
            model=model,
            system_instruction=self._system_prompt
        )


def get_uc_best_practices() -> str:
    """
    Get Unity Catalog best practices documentation.

    Returns:
        Best practices guide as a string
    """
    return """
# Unity Catalog Best Practices

## Naming Conventions
- Catalogs: Use business domains (analytics, marketing, finance)
- Schemas: Use functional areas (raw, staging, curated, reporting)
- Tables: Use descriptive names (dim_customer, fact_sales, agg_daily_revenue)

## Governance
1. Tag all tables with ownership and sensitivity level
2. Implement row-level security for multi-tenant data
3. Use column masking for PII fields
4. Enable audit logging

## Performance
1. Use liquid clustering (CLUSTER BY) instead of partitioning for most cases
2. Set OPTIMIZE WRITE and AUTO COMPACT properties
3. Use ZORDER only when queries have predictable filter patterns

## Security
1. Follow principle of least privilege
2. Use groups instead of individual users
3. Implement column masks for sensitive data
4. Regular access reviews

## Organization
- One catalog per environment (dev, staging, prod) or per domain
- Consistent schema naming across catalogs
- Use managed tables unless external location is required

## Example Table Creation:
```sql
CREATE TABLE catalog.schema.table_name (
    id BIGINT NOT NULL COMMENT 'Primary key',
    name STRING COMMENT 'Customer name',
    email STRING COMMENT 'Email address - PII',
    created_at TIMESTAMP COMMENT 'Record creation time'
)
USING DELTA
CLUSTER BY (id)
COMMENT 'Customer master data'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'owner' = 'data-team',
    'pii' = 'true',
    'sensitivity' = 'high'
);
```
"""
