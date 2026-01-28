"""
Code Generation Example
========================

This example demonstrates how to use the AI Assistant for generating,
reviewing, explaining, and fixing code - particularly useful for
PySpark and Databricks development.

Prerequisites:
    - API keys configured
    - Required packages installed
"""

import os
os.environ["ANTHROPIC_API_KEY"] = "your-claude-api-key"

from ai_assistant import AIAssistant
from ai_assistant.claude_client import ClaudeCodeAssistant

# =============================================================================
# Initialize
# =============================================================================

# Option 1: Use the main assistant
assistant = AIAssistant()

# Option 2: Use the specialized code assistant (Claude only)
# code_assistant = ClaudeCodeAssistant(api_key=os.environ["ANTHROPIC_API_KEY"])

# =============================================================================
# Code Generation
# =============================================================================

print("=" * 60)
print("CODE GENERATION")
print("=" * 60)

# Generate a PySpark function
print("\n1. Generating a Delta Lake reader function...")
code = assistant.generate_code(
    task="Create a function that reads a Delta table, applies schema validation, and returns a cleaned DataFrame",
    language="python",
    context="Using Databricks with Unity Catalog. The function should handle missing columns gracefully."
)
print(code)

# Generate with tests
print("\n" + "-" * 40)
print("2. Generating code with unit tests...")
code_with_tests = assistant.generate_code(
    task="Create a data quality check function that validates: non-null IDs, valid dates, and numeric ranges",
    language="python",
    context="For PySpark DataFrames",
    include_tests=True
)
print(code_with_tests)

# =============================================================================
# Using generate_code Method
# =============================================================================

print("\n" + "=" * 60)
print("ADVANCED CODE GENERATION")
print("=" * 60)

# Generate a complete ETL pipeline
print("\n3. Generating an ETL pipeline...")
etl_code = assistant.generate_code(
    task="""Create a complete ETL pipeline that:
    1. Reads JSON files from a landing zone
    2. Transforms the data (flattens nested structures, adds audit columns)
    3. Applies data quality checks
    4. Writes to a Delta table with MERGE for upserts
    5. Handles failures gracefully with logging""",
    language="python",
    context="Databricks environment with Unity Catalog. Use medallion architecture (bronze/silver/gold).",
    model="claude"
)
print(etl_code)

# =============================================================================
# Code Review
# =============================================================================

print("\n" + "=" * 60)
print("CODE REVIEW")
print("=" * 60)

# Sample code to review
sample_code = '''
def process_data(spark, table_name):
    df = spark.read.table(table_name)
    df = df.filter(df.status == "active")
    df = df.select("*")
    result = df.collect()
    for row in result:
        print(row)
    return result
'''

print("4. Reviewing sample code...")
print("\nCode to review:")
print(sample_code)

# Use the code assistant for review
code_assistant = assistant.code_assistant
review = code_assistant.review_code(
    code=sample_code,
    focus_areas=["performance", "best practices", "error handling"]
)
print("\nReview feedback:")
print(review)

# =============================================================================
# Code Explanation
# =============================================================================

print("\n" + "=" * 60)
print("CODE EXPLANATION")
print("=" * 60)

complex_code = '''
from delta.tables import DeltaTable

def merge_updates(spark, source_df, target_table, merge_keys):
    delta_table = DeltaTable.forName(spark, target_table)

    merge_condition = " AND ".join([
        f"target.{key} = source.{key}" for key in merge_keys
    ])

    update_set = {col: f"source.{col}" for col in source_df.columns}

    (delta_table.alias("target")
        .merge(source_df.alias("source"), merge_condition)
        .whenMatchedUpdate(set=update_set)
        .whenNotMatchedInsertAll()
        .execute())
'''

print("5. Explaining complex code...")
print("\nCode to explain:")
print(complex_code)

explanation = code_assistant.explain_code(complex_code, detail_level="detailed")
print("\nDetailed explanation:")
print(explanation)

# Brief explanation
brief = code_assistant.explain_code(complex_code, detail_level="brief")
print("\nBrief explanation:")
print(brief)

# =============================================================================
# Code Fixing
# =============================================================================

print("\n" + "=" * 60)
print("CODE FIXING")
print("=" * 60)

buggy_code = '''
def calculate_metrics(df):
    # This has several bugs
    total = df.agg({"amount": "sum"}).first()[0]
    avg = df.agg({"amount": "avg"}).first()[0]

    # Division by zero risk
    ratio = total / df.count()

    # Incorrect column reference
    filtered = df.filter(df.status = "active")

    return {"total": total, "average": avg, "ratio": ratio}
'''

print("6. Fixing buggy code...")
print("\nBuggy code:")
print(buggy_code)

error_message = "SyntaxError: invalid syntax at line 'filtered = df.filter(df.status = \"active\")'"

fixed = code_assistant.fix_code(buggy_code, error_message)
print("\nFixed code and explanation:")
print(fixed)

# =============================================================================
# SQL Query Generation
# =============================================================================

print("\n" + "=" * 60)
print("SQL QUERY GENERATION")
print("=" * 60)

print("7. Generating Spark SQL queries...")

sql_code = assistant.generate_code(
    task="""Generate a Spark SQL query that:
    1. Joins orders with customers and products tables
    2. Calculates total revenue by customer segment and product category
    3. Includes year-over-year growth comparison
    4. Filters to only include results from the last 2 years
    5. Orders by revenue descending""",
    language="sql",
    context="Databricks SQL with Delta tables"
)
print(sql_code)

# =============================================================================
# Query Optimization
# =============================================================================

print("\n" + "=" * 60)
print("QUERY OPTIMIZATION")
print("=" * 60)

slow_query = '''
SELECT
    c.customer_name,
    p.product_name,
    SUM(o.quantity * o.unit_price) as total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE o.order_date >= '2023-01-01'
GROUP BY c.customer_name, p.product_name
ORDER BY total_revenue DESC
'''

print("8. Optimizing a slow query...")
print("\nOriginal query:")
print(slow_query)

optimized = assistant.optimize_query(
    query=slow_query,
    context="""
    - orders table: 500 million rows, partitioned by order_date
    - customers table: 10 million rows
    - products table: 100,000 rows
    - Currently takes 45 minutes to run
    """
)
print("\nOptimization suggestions:")
print(optimized)

# =============================================================================
# Error Explanation
# =============================================================================

print("\n" + "=" * 60)
print("ERROR EXPLANATION")
print("=" * 60)

error_code = '''
df = spark.read.format("delta").load("/data/sales")
df.write.format("delta").mode("append").saveAsTable("catalog.schema.sales")
'''

error_msg = '''
AnalysisException: Cannot create table 'catalog.schema.sales' with
schema inference from Delta table when columns do not match.
Existing columns: [id, date, amount, customer_id]
Incoming columns: [id, date, amount, customer_id, region]
'''

print("9. Explaining an error...")
print("\nCode:")
print(error_code)
print("\nError message:")
print(error_msg)

explanation = assistant.explain_error(error_msg, error_code)
print("\nError explanation and solution:")
print(explanation)

# =============================================================================
# Generate Documentation
# =============================================================================

print("\n" + "=" * 60)
print("DOCUMENTATION GENERATION")
print("=" * 60)

code_to_document = '''
class DataPipeline:
    def __init__(self, spark, config):
        self.spark = spark
        self.config = config
        self.metrics = {}

    def extract(self, source_path, format="parquet"):
        return self.spark.read.format(format).load(source_path)

    def transform(self, df, transformations):
        for t in transformations:
            df = t(df)
        return df

    def load(self, df, target_table, mode="append"):
        df.write.format("delta").mode(mode).saveAsTable(target_table)
'''

print("10. Generating documentation...")
print("\nCode to document:")
print(code_to_document)

doc_prompt = f"""Generate comprehensive docstrings for this code:

```python
{code_to_document}
```

Include:
- Class docstring with examples
- Method docstrings with Args, Returns, Raises
- Type hints
Follow Google-style docstring format."""

documentation = assistant.ask_claude(doc_prompt)
print("\nGenerated documentation:")
print(documentation)

print("\n" + "=" * 60)
print("Code generation examples completed!")
print("=" * 60)
