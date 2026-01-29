"""
Data Engineering Prompt Templates.

This module provides pre-built prompt templates for common
data engineering tasks in Databricks environments.
"""

from .templates import (
    PromptTemplate,
    PromptVariable,
    VariableType,
    FewShotExample,
    PromptLibrary
)


# =============================================================================
# SQL Optimization Template
# =============================================================================

SQL_OPTIMIZATION_PROMPT = PromptTemplate(
    name="sql_optimization",
    description="Optimize Spark SQL queries for better performance",
    template="""Optimize the following Spark SQL query for better performance:

## Query:
```sql
{query}
```

{context_section}

## Optimization Goals:
{goals}

## Instructions:
1. Analyze the query for performance issues
2. Identify bottlenecks (joins, aggregations, scans)
3. Suggest specific optimizations
4. Provide the optimized query
5. Explain the expected performance improvement

## Output Format:
### Issues Found
[List issues]

### Optimized Query
```sql
[Your optimized query]
```

### Explanation
[Explain changes and expected improvements]""",
    variables=[
        PromptVariable(
            name="query",
            description="The SQL query to optimize",
            var_type=VariableType.SQL,
            required=True
        ),
        PromptVariable(
            name="context_section",
            description="Additional context (table sizes, indexes, etc.)",
            var_type=VariableType.STRING,
            required=False,
            default=""
        ),
        PromptVariable(
            name="goals",
            description="Optimization goals",
            var_type=VariableType.STRING,
            required=False,
            default="- Reduce execution time\n- Minimize data shuffling\n- Optimize memory usage"
        )
    ],
    system_instruction="""You are a Spark SQL optimization expert.
Focus on practical optimizations that significantly impact performance.
Consider: predicate pushdown, partition pruning, broadcast joins, and caching.""",
    tags=["sql", "optimization", "spark"],
    version="1.0.0"
)


# =============================================================================
# DDL Generation Template
# =============================================================================

DDL_GENERATION_PROMPT = PromptTemplate(
    name="ddl_generation",
    description="Generate CREATE TABLE DDL for Unity Catalog",
    template="""Generate a CREATE TABLE DDL statement for Unity Catalog:

## Requirements:
{requirements}

## Target Location:
- Catalog: {catalog}
- Schema: {schema}
- Table Name: {table_name}

## Data Types (if known):
{schema_hints}

## Guidelines:
- Use Delta Lake format
- Include appropriate TBLPROPERTIES for optimization
- Add table and column comments
- Use liquid clustering (CLUSTER BY) when appropriate
- Include governance tags (owner, pii, sensitivity)
- Consider partitioning strategy

## Output:
Provide only the DDL statement with inline comments.""",
    variables=[
        PromptVariable(
            name="requirements",
            description="Natural language description of the table",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="catalog",
            description="Unity Catalog name",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="schema",
            description="Schema name",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="table_name",
            description="Table name",
            var_type=VariableType.STRING,
            required=False,
            default="[generate appropriate name]"
        ),
        PromptVariable(
            name="schema_hints",
            description="Known column types or sample data",
            var_type=VariableType.STRING,
            required=False,
            default="Not specified - infer from requirements"
        )
    ],
    system_instruction="""You are a Unity Catalog DDL expert.
Create well-designed tables following Databricks best practices.
Always include proper documentation and governance tags.""",
    examples=[
        FewShotExample(
            input_vars={
                "requirements": "Customer master data with PII fields",
                "catalog": "enterprise",
                "schema": "master_data",
                "table_name": "customers"
            },
            expected_output="""CREATE TABLE enterprise.master_data.customers (
    customer_id BIGINT NOT NULL COMMENT 'Unique customer identifier',
    email STRING COMMENT 'Customer email - PII',
    full_name STRING COMMENT 'Customer full name - PII',
    created_at TIMESTAMP COMMENT 'Account creation timestamp',
    updated_at TIMESTAMP COMMENT 'Last update timestamp'
)
USING DELTA
CLUSTER BY (customer_id)
COMMENT 'Customer master data table'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'owner' = 'data-team',
    'pii' = 'true',
    'sensitivity' = 'high'
);""",
            explanation="Includes PII tags, liquid clustering, and optimization properties"
        )
    ],
    tags=["ddl", "unity_catalog", "schema"],
    version="1.0.0"
)


# =============================================================================
# Pipeline Generation Template
# =============================================================================

PIPELINE_GENERATION_PROMPT = PromptTemplate(
    name="pipeline_generation",
    description="Generate data pipeline code (DLT or PySpark)",
    template="""Generate a {pipeline_type} data pipeline:

## Requirements:
{requirements}

## Source:
{source}

## Target:
{target}

## Features to Include:
{features}

## Guidelines:
- Follow medallion architecture (bronze/silver/gold) where appropriate
- Include data quality expectations
- Add comprehensive comments
- Handle schema evolution
- Implement proper error handling
- Use Unity Catalog three-level namespace

## Output:
Provide complete, production-ready Python code.""",
    variables=[
        PromptVariable(
            name="pipeline_type",
            description="Type of pipeline (DLT, PySpark, Streaming)",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="requirements",
            description="Pipeline requirements description",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="source",
            description="Source table/path",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="target",
            description="Target catalog.schema",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="features",
            description="Features to include",
            var_type=VariableType.STRING,
            required=False,
            default="- Data quality checks\n- Incremental processing\n- Audit columns"
        )
    ],
    system_instruction="""You are a Databricks pipeline expert.
Generate production-ready pipeline code following best practices.
Use proper typing, error handling, and documentation.""",
    tags=["pipeline", "dlt", "etl"],
    version="1.0.0"
)


# =============================================================================
# Error Explanation Template
# =============================================================================

ERROR_EXPLANATION_PROMPT = PromptTemplate(
    name="error_explanation",
    description="Explain errors and provide solutions",
    template="""Explain the following error and provide solutions:

## Error Message:
```
{error_message}
```

## Code (if available):
```{language}
{code}
```

## Context:
{context}

## Please provide:
1. **Clear Explanation**: What went wrong in simple terms
2. **Root Cause**: Technical reason for the error
3. **Solution Steps**: Step-by-step fix
4. **Corrected Code**: Fixed version (if applicable)
5. **Prevention**: How to avoid this in the future""",
    variables=[
        PromptVariable(
            name="error_message",
            description="The error message",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="code",
            description="The code that caused the error",
            var_type=VariableType.CODE,
            required=False,
            default=""
        ),
        PromptVariable(
            name="language",
            description="Programming language",
            var_type=VariableType.STRING,
            required=False,
            default="python"
        ),
        PromptVariable(
            name="context",
            description="Additional context",
            var_type=VariableType.STRING,
            required=False,
            default="No additional context provided"
        )
    ],
    system_instruction="""You are a debugging expert for Spark and Databricks.
Provide clear, actionable solutions. Be specific about fixes.""",
    tags=["debugging", "error", "troubleshooting"],
    version="1.0.0"
)


# =============================================================================
# Code Review Template
# =============================================================================

CODE_REVIEW_PROMPT = PromptTemplate(
    name="code_review",
    description="Review code and provide feedback",
    template="""Review the following code and provide detailed feedback:

## Code:
```{language}
{code}
```

## Review Focus Areas:
{focus_areas}

## Please provide:
1. **Summary**: What does this code do?
2. **Issues Found**: List any problems (with severity)
3. **Suggestions**: Improvements and best practices
4. **Security**: Any security concerns?
5. **Performance**: Any performance issues?
6. **Overall Assessment**: Ready for production?""",
    variables=[
        PromptVariable(
            name="code",
            description="Code to review",
            var_type=VariableType.CODE,
            required=True
        ),
        PromptVariable(
            name="language",
            description="Programming language",
            var_type=VariableType.STRING,
            required=False,
            default="python"
        ),
        PromptVariable(
            name="focus_areas",
            description="Specific areas to focus on",
            var_type=VariableType.STRING,
            required=False,
            default="- Correctness\n- Performance\n- Readability\n- Best practices\n- Error handling"
        )
    ],
    system_instruction="""You are a senior software engineer reviewing code.
Be thorough but constructive. Provide specific, actionable feedback.""",
    tags=["code_review", "quality"],
    version="1.0.0"
)


# =============================================================================
# Data Analysis Template
# =============================================================================

DATA_ANALYSIS_PROMPT = PromptTemplate(
    name="data_analysis",
    description="Analyze data and generate insights",
    template="""Analyze the following data and provide insights:

## Table/Data:
{data_description}

## Schema:
{schema}

## Sample Data:
{sample_data}

## Analysis Questions:
{questions}

## Please provide:
1. **Data Overview**: Summary of the data structure
2. **Key Findings**: Important insights discovered
3. **Data Quality**: Any issues or anomalies
4. **Recommendations**: Suggested actions
5. **SQL Queries**: Useful queries for further analysis""",
    variables=[
        PromptVariable(
            name="data_description",
            description="Description of the data",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="schema",
            description="Table schema",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="sample_data",
            description="Sample data rows",
            var_type=VariableType.STRING,
            required=False,
            default="No sample data provided"
        ),
        PromptVariable(
            name="questions",
            description="Specific questions to answer",
            var_type=VariableType.STRING,
            required=False,
            default="Provide a comprehensive analysis"
        )
    ],
    system_instruction="""You are a data analyst expert.
Provide clear, actionable insights. Use data to support conclusions.""",
    tags=["analysis", "insights"],
    version="1.0.0"
)


# =============================================================================
# Schema Design Template
# =============================================================================

SCHEMA_DESIGN_PROMPT = PromptTemplate(
    name="schema_design",
    description="Design optimal schema for a use case",
    template="""Design an optimal schema for the following use case:

## Use Case:
{use_case}

## Requirements:
{requirements}

## Query Patterns:
{query_patterns}

## Constraints:
{constraints}

## Please provide:
1. **Data Model**: Recommended tables and relationships
2. **Schema DDL**: Complete DDL statements
3. **Indexing Strategy**: Clustering and partitioning
4. **Trade-offs**: Design decisions and trade-offs
5. **Sample Queries**: Common query patterns""",
    variables=[
        PromptVariable(
            name="use_case",
            description="Business use case description",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="requirements",
            description="Specific requirements",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="query_patterns",
            description="Expected query patterns",
            var_type=VariableType.STRING,
            required=False,
            default="Not specified"
        ),
        PromptVariable(
            name="constraints",
            description="Design constraints",
            var_type=VariableType.STRING,
            required=False,
            default="None specified"
        )
    ],
    system_instruction="""You are a data modeling expert.
Design efficient, scalable schemas following best practices.""",
    tags=["schema", "modeling", "design"],
    version="1.0.0"
)


# =============================================================================
# Data Quality Template
# =============================================================================

DATA_QUALITY_PROMPT = PromptTemplate(
    name="data_quality",
    description="Generate data quality checks and expectations",
    template="""Generate data quality checks for the following table:

## Table:
{table_name}

## Schema:
{schema}

## Business Rules:
{business_rules}

## Quality Requirements:
{quality_requirements}

## Please provide:
1. **DLT Expectations**: @dlt.expect decorators
2. **SQL Checks**: Data quality SQL queries
3. **Great Expectations**: GE expectation suite (if applicable)
4. **Monitoring Queries**: Queries for ongoing monitoring
5. **Alert Thresholds**: Suggested alert thresholds""",
    variables=[
        PromptVariable(
            name="table_name",
            description="Table name",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="schema",
            description="Table schema",
            var_type=VariableType.STRING,
            required=True
        ),
        PromptVariable(
            name="business_rules",
            description="Business rules to validate",
            var_type=VariableType.STRING,
            required=False,
            default="Infer from schema"
        ),
        PromptVariable(
            name="quality_requirements",
            description="Quality requirements",
            var_type=VariableType.STRING,
            required=False,
            default="- No nulls in key columns\n- Referential integrity\n- Valid data ranges"
        )
    ],
    system_instruction="""You are a data quality expert.
Generate comprehensive, practical quality checks.""",
    tags=["data_quality", "testing", "validation"],
    version="1.0.0"
)


def get_data_engineering_prompts() -> PromptLibrary:
    """
    Get a PromptLibrary with all data engineering templates.

    Returns:
        PromptLibrary with pre-loaded templates

    Example:
        >>> library = get_data_engineering_prompts()
        >>> prompt = library.render("sql_optimization", query="SELECT * FROM t")
    """
    library = PromptLibrary()

    library.add(SQL_OPTIMIZATION_PROMPT)
    library.add(DDL_GENERATION_PROMPT)
    library.add(PIPELINE_GENERATION_PROMPT)
    library.add(ERROR_EXPLANATION_PROMPT)
    library.add(CODE_REVIEW_PROMPT)
    library.add(DATA_ANALYSIS_PROMPT)
    library.add(SCHEMA_DESIGN_PROMPT)
    library.add(DATA_QUALITY_PROMPT)

    return library
