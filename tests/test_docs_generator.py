"""
Unit tests for Documentation Generator module.

Tests AI-powered documentation generation for code,
schemas, notebooks, and data pipelines.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai_assistant.docs_generator import (
    DocSection,
    FunctionDoc,
    DocsGenerator,
    create_docs_generator
)


class TestDocSection:
    """Tests for DocSection dataclass."""

    def test_section_creation(self):
        """Test creating a section."""
        section = DocSection(
            title="Overview",
            content="This is the overview section.",
            level=1
        )

        assert section.title == "Overview"
        assert section.content == "This is the overview section."
        assert section.level == 1

    def test_section_with_subsections(self):
        """Test section with subsections."""
        subsection = DocSection(
            title="Details",
            content="Detail content",
            level=2
        )

        section = DocSection(
            title="Main",
            content="Main content",
            level=1,
            subsections=[subsection]
        )

        assert len(section.subsections) == 1
        assert section.subsections[0].title == "Details"

    def test_to_markdown(self):
        """Test converting to markdown."""
        section = DocSection(
            title="Test",
            content="Test content",
            level=2
        )

        md = section.to_markdown()

        assert "## Test" in md
        assert "Test content" in md

    def test_nested_markdown(self):
        """Test nested sections to markdown."""
        subsection = DocSection(
            title="Sub",
            content="Sub content",
            level=2
        )

        section = DocSection(
            title="Main",
            content="Main content",
            level=1,
            subsections=[subsection]
        )

        md = section.to_markdown()

        assert "# Main" in md
        assert "## Sub" in md


class TestFunctionDoc:
    """Tests for FunctionDoc dataclass."""

    def test_function_doc_creation(self):
        """Test creating function documentation."""
        doc = FunctionDoc(
            name="calculate_total",
            description="Calculate the total amount.",
            parameters=[
                {"name": "items", "type": "list", "description": "List of items"},
                {"name": "tax_rate", "type": "float", "description": "Tax rate"}
            ],
            returns="float: The total amount",
            raises=["ValueError: If items is empty"],
            examples=["calculate_total([10, 20], 0.1)"]
        )

        assert doc.name == "calculate_total"
        assert len(doc.parameters) == 2
        assert len(doc.raises) == 1
        assert len(doc.examples) == 1

    def test_to_google_style(self):
        """Test generating Google-style docstring."""
        doc = FunctionDoc(
            name="test_func",
            description="Test function.",
            parameters=[
                {"name": "x", "description": "Input value"}
            ],
            returns="The result"
        )

        docstring = doc.to_docstring(style="google")

        assert '"""Test function.' in docstring
        assert "Args:" in docstring
        assert "x:" in docstring
        assert "Returns:" in docstring

    def test_to_numpy_style(self):
        """Test generating NumPy-style docstring."""
        doc = FunctionDoc(
            name="test_func",
            description="Test function.",
            parameters=[
                {"name": "x", "type": "int", "description": "Input value"}
            ],
            returns="The result"
        )

        docstring = doc.to_docstring(style="numpy")

        assert '"""Test function.' in docstring
        assert "Parameters" in docstring
        assert "----------" in docstring
        assert "Returns" in docstring

    def test_docstring_with_raises(self):
        """Test docstring with raises section."""
        doc = FunctionDoc(
            name="risky_func",
            description="A risky function.",
            parameters=[],
            returns="",
            raises=["ValueError: On invalid input", "TypeError: On wrong type"]
        )

        docstring = doc.to_docstring()

        assert "Raises:" in docstring
        assert "ValueError" in docstring
        assert "TypeError" in docstring

    def test_docstring_with_examples(self):
        """Test docstring with examples."""
        doc = FunctionDoc(
            name="example_func",
            description="Example function.",
            parameters=[],
            returns="",
            examples=["example_func()", "example_func(arg=1)"]
        )

        docstring = doc.to_docstring()

        assert "Example:" in docstring
        assert ">>>" in docstring


class TestDocsGenerator:
    """Tests for DocsGenerator class."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="""
```json
{
    "name": "test_function",
    "description": "A test function that does something useful.",
    "parameters": [
        {"name": "x", "type": "int", "description": "Input value"}
    ],
    "returns": "The processed result",
    "raises": [],
    "examples": ["test_function(5)"]
}
```
""")
        return client

    @pytest.fixture
    def generator(self, mock_ai_client):
        """Create a docs generator."""
        return DocsGenerator(ai_client=mock_ai_client, style="google")

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.style == "google"

    def test_generate_function_docs(self, generator, mock_ai_client):
        """Test generating function documentation."""
        code = """
def process_data(data: list, threshold: float = 0.5) -> dict:
    result = {}
    for item in data:
        if item > threshold:
            result[item] = True
    return result
"""

        doc = generator.generate_function_docs(code)

        assert isinstance(doc, FunctionDoc)
        assert doc.name is not None
        mock_ai_client.generate.assert_called_once()

    def test_generate_class_docs(self, generator, mock_ai_client):
        """Test generating class documentation."""
        mock_ai_client.generate.return_value = '''
"""
DataProcessor class for handling data transformations.

Attributes:
    config: Configuration dictionary
    logger: Logger instance

Example:
    >>> processor = DataProcessor(config={})
    >>> processor.process(data)
"""
'''

        code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        return data
"""

        docs = generator.generate_class_docs(code)

        assert isinstance(docs, str)
        assert '"""' in docs

    def test_generate_module_docs(self, generator, mock_ai_client):
        """Test generating module documentation."""
        mock_ai_client.generate.return_value = """
This module provides utilities for data processing.

Features:
- Data validation
- Data transformation
- Data export
"""

        code = """
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def save_data(df, path):
    df.to_csv(path)
"""

        docs = generator.generate_module_docs(code, "data_utils")

        assert isinstance(docs, str)
        mock_ai_client.generate.assert_called()

    def test_generate_schema_docs(self, generator, mock_ai_client):
        """Test generating schema documentation."""
        mock_ai_client.generate.return_value = """
# users

## Description
The users table stores information about registered users.

## Columns
- id (int): Unique user identifier
- email (string): User's email address
- created_at (timestamp): Account creation timestamp

## Example Queries
```sql
SELECT * FROM users WHERE created_at > '2024-01-01'
```
"""

        schema = [
            {"name": "id", "type": "int"},
            {"name": "email", "type": "string"},
            {"name": "created_at", "type": "timestamp"}
        ]

        docs = generator.generate_schema_docs("catalog.schema.users", schema)

        assert isinstance(docs, str)
        mock_ai_client.generate.assert_called()

    def test_generate_data_dictionary(self, generator, mock_ai_client):
        """Test generating data dictionary."""
        mock_ai_client.generate.return_value = """
# Data Dictionary

## users
User account information.

| Column | Type | Description |
|--------|------|-------------|
| id | int | Primary key |
| email | string | User email |

## orders
Customer orders.

| Column | Type | Description |
|--------|------|-------------|
| id | int | Order ID |
| user_id | int | Foreign key to users |
"""

        tables = [
            {
                "name": "users",
                "columns": [
                    {"name": "id", "type": "int"},
                    {"name": "email", "type": "string"}
                ]
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "type": "int"},
                    {"name": "user_id", "type": "int"}
                ]
            }
        ]

        docs = generator.generate_data_dictionary(tables)

        assert isinstance(docs, str)
        assert "users" in docs.lower()

    def test_generate_notebook_readme(self, generator, mock_ai_client):
        """Test generating notebook README."""
        mock_ai_client.generate.return_value = """
# Data Analysis Notebook

## Overview
This notebook performs exploratory data analysis.

## Prerequisites
- Databricks cluster with ML runtime
- Access to data lake

## Sections
1. Data Loading
2. Data Cleaning
3. Analysis
4. Visualization
"""

        notebook_content = """
# Data Analysis

## Load Data
df = spark.read.table("analytics.sales")

## Clean Data
df_clean = df.dropna()

## Analysis
df_clean.groupBy("category").count().show()
"""

        docs = generator.generate_notebook_readme(
            notebook_content,
            "data_analysis.py"
        )

        assert isinstance(docs, str)

    def test_generate_pipeline_docs(self, generator, mock_ai_client):
        """Test generating pipeline documentation."""
        mock_ai_client.generate.return_value = """
# Sales Pipeline

## Overview
ETL pipeline for sales data.

## Architecture
```mermaid
graph LR
    A[Raw Data] --> B[Bronze]
    B --> C[Silver]
    C --> D[Gold]
```

## Data Flow
1. Ingest raw sales from source
2. Clean and validate
3. Aggregate for reporting
"""

        pipeline_code = """
@dlt.table
def bronze_sales():
    return spark.read.format("parquet").load("/raw/sales")

@dlt.table
def silver_sales():
    return dlt.read("bronze_sales").dropna()
"""

        docs = generator.generate_pipeline_docs(pipeline_code, "sales_pipeline")

        assert isinstance(docs, str)

    def test_add_docstrings_to_code(self, generator, mock_ai_client):
        """Test adding docstrings to code."""
        mock_ai_client.generate.return_value = '''
```python
def calculate(x: int, y: int) -> int:
    """
    Calculate the sum of two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y
```
'''

        code = """
def calculate(x: int, y: int) -> int:
    return x + y
"""

        result = generator.add_docstrings_to_code(code)

        assert isinstance(result, str)
        # Should contain docstring
        assert '"""' in result or "docstring" in result.lower()


class TestDocsGeneratorErrorHandling:
    """Tests for error handling in DocsGenerator."""

    @pytest.fixture
    def failing_generator(self):
        """Create generator with failing AI client."""
        client = Mock()
        client.generate = Mock(side_effect=Exception("API Error"))
        return DocsGenerator(ai_client=client)

    def test_generate_function_docs_fallback(self, failing_generator):
        """Test fallback when generation fails."""
        doc = failing_generator.generate_function_docs("def test(): pass")

        assert isinstance(doc, FunctionDoc)
        assert doc.name == "unknown"

    def test_generate_class_docs_fallback(self, failing_generator):
        """Test fallback for class docs."""
        docs = failing_generator.generate_class_docs("class Test: pass")

        assert "failed" in docs.lower()

    def test_generate_schema_docs_fallback(self, failing_generator):
        """Test fallback for schema docs."""
        docs = failing_generator.generate_schema_docs(
            "test_table",
            [{"name": "id", "type": "int"}]
        )

        assert "failed" in docs.lower()


class TestCreateDocsGenerator:
    """Tests for factory function."""

    def test_create_docs_generator(self):
        """Test creating a docs generator."""
        mock_client = Mock()

        generator = create_docs_generator(mock_client, style="numpy")

        assert isinstance(generator, DocsGenerator)
        assert generator.style == "numpy"

    def test_create_docs_generator_default_style(self):
        """Test default style."""
        mock_client = Mock()

        generator = create_docs_generator(mock_client)

        assert generator.style == "google"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
