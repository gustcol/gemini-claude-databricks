"""
Unit tests for dbt Integration module.

Tests dbt model generation, DLT-to-dbt conversion,
and schema.yml generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai_assistant.dbt_integration import (
    DBTModel,
    DBTProject,
    DBTIntegration,
    create_dbt_integration
)


class TestDBTModel:
    """Tests for DBTModel dataclass."""

    def test_model_creation(self):
        """Test creating a dbt model."""
        model = DBTModel(
            name="stg_users",
            description="Staging model for users",
            sql="SELECT * FROM {{ source('raw', 'users') }}",
            config={"materialized": "view"},
            columns=[
                {"name": "id", "description": "User ID", "tests": ["not_null", "unique"]}
            ]
        )

        assert model.name == "stg_users"
        assert model.description == "Staging model for users"
        assert "source" in model.sql

    def test_model_defaults(self):
        """Test default values."""
        model = DBTModel(
            name="test_model",
            description="Test",
            sql="SELECT 1"
        )

        assert model.config == {}
        assert model.columns == []
        assert model.tests == []

    def test_to_sql_file(self):
        """Test generating SQL file content."""
        model = DBTModel(
            name="dim_customers",
            description="Customer dimension table",
            sql="SELECT id, name FROM customers",
            config={"materialized": "table"}
        )

        sql_content = model.to_sql_file()

        assert "dim_customers" in sql_content
        assert "Customer dimension" in sql_content
        assert "materialized" in sql_content
        assert "SELECT id, name" in sql_content

    def test_to_sql_file_no_config(self):
        """Test SQL file without config."""
        model = DBTModel(
            name="simple_model",
            description="Simple model",
            sql="SELECT 1 as value"
        )

        sql_content = model.to_sql_file()

        assert "simple_model" in sql_content
        assert "config" not in sql_content.lower() or "{% config" not in sql_content

    def test_to_schema_entry(self):
        """Test generating schema.yml entry."""
        model = DBTModel(
            name="fact_orders",
            description="Order facts",
            sql="SELECT * FROM orders",
            columns=[
                {"name": "order_id", "description": "Order ID", "tests": ["unique"]}
            ],
            tests=["unique_combination"]
        )

        entry = model.to_schema_entry()

        assert entry["name"] == "fact_orders"
        assert entry["description"] == "Order facts"
        assert "columns" in entry
        assert "tests" in entry


class TestDBTProject:
    """Tests for DBTProject dataclass."""

    def test_project_creation(self):
        """Test creating a dbt project."""
        project = DBTProject(
            name="analytics",
            models=[
                DBTModel(name="stg_users", description="Users", sql="SELECT 1"),
                DBTModel(name="dim_users", description="User dim", sql="SELECT 2")
            ],
            sources=[
                {"name": "raw", "tables": [{"name": "users"}]}
            ]
        )

        assert project.name == "analytics"
        assert len(project.models) == 2
        assert len(project.sources) == 1

    def test_generate_schema_yml(self):
        """Test generating schema.yml content."""
        project = DBTProject(
            name="test_project",
            models=[
                DBTModel(
                    name="model_a",
                    description="Model A",
                    sql="SELECT 1",
                    columns=[{"name": "id", "description": "ID"}]
                )
            ]
        )

        schema_yml = project.generate_schema_yml()

        assert "version: 2" in schema_yml
        assert "model_a" in schema_yml
        assert "Model A" in schema_yml

    def test_generate_schema_yml_with_sources(self):
        """Test schema.yml with sources."""
        project = DBTProject(
            name="test_project",
            models=[],
            sources=[
                {
                    "name": "raw_data",
                    "database": "raw",
                    "schema": "public",
                    "tables": [{"name": "users"}]
                }
            ]
        )

        schema_yml = project.generate_schema_yml()

        assert "sources" in schema_yml
        assert "raw_data" in schema_yml


class TestDBTIntegration:
    """Tests for DBTIntegration class."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="""
```json
{
    "name": "daily_sales_summary",
    "description": "Aggregates daily sales metrics",
    "sql": "SELECT date, SUM(amount) as total FROM {{ ref('stg_sales') }} GROUP BY date",
    "columns": [
        {"name": "date", "description": "Sale date", "tests": ["not_null"]},
        {"name": "total", "description": "Total sales amount"}
    ]
}
```
""")
        return client

    @pytest.fixture
    def integration(self, mock_ai_client):
        """Create a dbt integration."""
        return DBTIntegration(
            ai_client=mock_ai_client,
            project_name="test_project"
        )

    def test_integration_initialization(self, integration):
        """Test integration initialization."""
        assert integration.project_name == "test_project"

    def test_generate_model(self, integration, mock_ai_client):
        """Test generating a dbt model."""
        model = integration.generate_model(
            description="Create a daily sales summary",
            source_table="raw.sales",
            materialization="table"
        )

        assert isinstance(model, DBTModel)
        assert model.name is not None
        assert model.sql is not None
        mock_ai_client.generate.assert_called_once()

    def test_generate_model_with_tests(self, integration, mock_ai_client):
        """Test model generation with tests."""
        model = integration.generate_model(
            description="User metrics model",
            source_table="analytics.users",
            include_tests=True
        )

        assert isinstance(model, DBTModel)

    def test_generate_model_fallback(self, integration, mock_ai_client):
        """Test fallback when generation fails."""
        mock_ai_client.generate.return_value = "Invalid response"

        model = integration.generate_model(
            description="Test model",
            source_table="test.table"
        )

        assert isinstance(model, DBTModel)
        assert model.name == "generated_model"

    def test_convert_dlt_to_dbt(self, integration, mock_ai_client):
        """Test converting DLT to dbt."""
        mock_ai_client.generate.return_value = """
```json
[
    {
        "name": "bronze_events",
        "description": "Raw events from source",
        "sql": "SELECT * FROM {{ source('raw', 'events') }}",
        "columns": []
    },
    {
        "name": "silver_events",
        "description": "Cleaned events",
        "sql": "SELECT * FROM {{ ref('bronze_events') }} WHERE valid = true",
        "columns": []
    }
]
```
"""

        dlt_code = """
import dlt

@dlt.table
def bronze_events():
    return spark.read.format("json").load("/raw/events")

@dlt.table
def silver_events():
    return dlt.read("bronze_events").filter("valid = true")
"""

        models = integration.convert_dlt_to_dbt(dlt_code)

        assert isinstance(models, list)
        assert len(models) >= 1
        mock_ai_client.generate.assert_called()

    def test_convert_dbt_to_dlt(self, integration, mock_ai_client):
        """Test converting dbt to DLT."""
        mock_ai_client.generate.return_value = """
```python
import dlt

@dlt.table(
    name="daily_sales",
    comment="Daily sales aggregation"
)
def daily_sales():
    df = spark.read.table("raw.sales")
    return df.groupBy("date").agg(sum("amount").alias("total"))
```
"""

        dbt_model = DBTModel(
            name="daily_sales",
            description="Daily sales aggregation",
            sql="SELECT date, SUM(amount) as total FROM {{ source('raw', 'sales') }} GROUP BY date",
            columns=[
                {"name": "date", "description": "Date"},
                {"name": "total", "description": "Total"}
            ]
        )

        dlt_code = integration.convert_dbt_to_dlt(dbt_model)

        assert "@dlt" in dlt_code or "dlt" in dlt_code.lower()

    def test_generate_source_definition(self, integration):
        """Test generating source definition."""
        source = integration.generate_source_definition(
            database="raw_data",
            schema="public",
            tables=["users", "orders", "products"]
        )

        assert source["name"] == "raw_data_public"
        assert source["database"] == "raw_data"
        assert source["schema"] == "public"
        assert len(source["tables"]) == 3

    def test_generate_tests(self, integration, mock_ai_client):
        """Test generating tests."""
        mock_ai_client.generate.return_value = """
columns:
  - name: id
    tests:
      - not_null
      - unique
  - name: email
    tests:
      - not_null
      - unique
"""

        model = DBTModel(
            name="users",
            description="Users model",
            sql="SELECT * FROM raw.users",
            columns=[
                {"name": "id", "description": "User ID"},
                {"name": "email", "description": "Email"}
            ]
        )

        tests = integration.generate_tests(model)

        assert isinstance(tests, str)
        mock_ai_client.generate.assert_called()

    def test_generate_documentation(self, integration, mock_ai_client):
        """Test generating documentation."""
        mock_ai_client.generate.return_value = """
# daily_sales

## Overview
This model aggregates daily sales data.

## Business Context
Used for daily reporting and trend analysis.

## Columns
- date: The date of sales
- total: Sum of all sales for the day

## Example Queries
```sql
SELECT * FROM {{ ref('daily_sales') }} WHERE date >= '2024-01-01'
```
"""

        model = DBTModel(
            name="daily_sales",
            description="Daily sales",
            sql="SELECT date, SUM(amount) as total FROM sales GROUP BY date"
        )

        docs = integration.generate_documentation(model)

        assert isinstance(docs, str)
        assert "daily_sales" in docs

    def test_generate_staging_model(self, integration, mock_ai_client):
        """Test generating staging model."""
        mock_ai_client.generate.return_value = """
```json
{
    "name": "stg_raw__users",
    "sql": "SELECT id, LOWER(email) as email, created_at FROM {{ source('raw', 'users') }}",
    "columns": [
        {"name": "id", "description": "User ID"},
        {"name": "email", "description": "Normalized email"},
        {"name": "created_at", "description": "Creation timestamp"}
    ]
}
```
"""

        model = integration.generate_staging_model(
            source_name="raw",
            source_table="users",
            columns=[
                {"name": "id", "type": "int"},
                {"name": "email", "type": "string"},
                {"name": "created_at", "type": "timestamp"}
            ]
        )

        assert isinstance(model, DBTModel)
        assert "stg" in model.name.lower()
        assert model.config.get("materialized") == "view"


class TestDBTIntegrationEdgeCases:
    """Edge case tests for DBTIntegration."""

    @pytest.fixture
    def integration(self):
        """Create integration with mock client."""
        client = Mock()
        client.generate = Mock(return_value="")
        return DBTIntegration(ai_client=client)

    def test_empty_dlt_code(self, integration):
        """Test converting empty DLT code."""
        models = integration.convert_dlt_to_dbt("")

        assert models == []

    def test_invalid_json_response(self, integration):
        """Test handling invalid JSON response."""
        integration.ai_client.generate.return_value = "Not valid JSON"

        model = integration.generate_model(
            description="Test",
            source_table="test.table"
        )

        # Should return fallback model
        assert isinstance(model, DBTModel)


class TestCreateDBTIntegration:
    """Tests for factory function."""

    def test_create_dbt_integration(self):
        """Test creating a dbt integration."""
        mock_client = Mock()

        integration = create_dbt_integration(
            ai_client=mock_client,
            project_name="my_dbt_project"
        )

        assert isinstance(integration, DBTIntegration)
        assert integration.project_name == "my_dbt_project"

    def test_create_dbt_integration_default_name(self):
        """Test default project name."""
        mock_client = Mock()

        integration = create_dbt_integration(ai_client=mock_client)

        assert integration.project_name == "dbt_project"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
