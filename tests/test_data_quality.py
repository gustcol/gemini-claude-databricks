"""
Unit tests for Data Quality module.

Tests the data quality analyzer including expectation generation,
DLT expectations, and Great Expectations integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai_assistant.data_quality import (
    DataExpectation,
    DataQualityReport,
    DataQualityAnalyzer,
    create_data_quality_analyzer
)


class TestDataExpectation:
    """Tests for DataExpectation dataclass."""

    def test_expectation_creation(self):
        """Test creating an expectation."""
        exp = DataExpectation(
            column="email",
            expectation_type="not_null",
            parameters={},
            description="Email should not be null",
            severity="error"
        )

        assert exp.column == "email"
        assert exp.expectation_type == "not_null"
        assert exp.severity == "error"

    def test_expectation_with_parameters(self):
        """Test expectation with parameters."""
        exp = DataExpectation(
            column="age",
            expectation_type="between",
            parameters={"min": 0, "max": 150},
            description="Age should be between 0 and 150"
        )

        assert exp.parameters["min"] == 0
        assert exp.parameters["max"] == 150

    def test_expectation_defaults(self):
        """Test default values."""
        exp = DataExpectation(
            column="id",
            expectation_type="unique",
            parameters={},
            description="ID should be unique"
        )

        assert exp.severity == "error"  # Default severity


class TestDataQualityReport:
    """Tests for DataQualityReport dataclass."""

    def test_report_creation(self):
        """Test creating a report."""
        expectations = [
            DataExpectation(
                column="id",
                expectation_type="not_null",
                parameters={},
                description="ID not null"
            ),
            DataExpectation(
                column="email",
                expectation_type="unique",
                parameters={},
                description="Email unique"
            )
        ]

        report = DataQualityReport(
            table_name="users",
            expectations=expectations,
            summary="2 expectations generated"
        )

        assert report.table_name == "users"
        assert len(report.expectations) == 2
        assert report.summary is not None

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        expectations = [
            DataExpectation(
                column="id",
                expectation_type="not_null",
                parameters={},
                description="Test"
            )
        ]

        report = DataQualityReport(
            table_name="test_table",
            expectations=expectations,
            summary="Test report"
        )

        data = report.to_dict()

        assert data["table_name"] == "test_table"
        assert len(data["expectations"]) == 1


class TestDataQualityAnalyzer:
    """Tests for DataQualityAnalyzer class."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="""
Based on the schema, here are the data quality expectations:

1. Column: id
   - Type: not_null
   - Description: ID should never be null

2. Column: email
   - Type: regex_match
   - Pattern: ^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$
   - Description: Email should be valid format

```json
[
    {"column": "id", "type": "not_null", "description": "ID should not be null"},
    {"column": "email", "type": "unique", "description": "Email should be unique"}
]
```
""")
        return client

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        spark = MagicMock()

        # Mock schema info
        mock_schema = MagicMock()
        mock_schema.collect.return_value = [
            MagicMock(col_name="id", data_type="int"),
            MagicMock(col_name="email", data_type="string"),
            MagicMock(col_name="created_at", data_type="timestamp")
        ]
        spark.sql.return_value = mock_schema

        return spark

    @pytest.fixture
    def analyzer(self, mock_ai_client, mock_spark):
        """Create a data quality analyzer."""
        return DataQualityAnalyzer(
            ai_client=mock_ai_client,
            spark=mock_spark
        )

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None

    def test_analyze_table(self, analyzer, mock_ai_client):
        """Test analyzing a table."""
        report = analyzer.analyze_table("catalog.schema.users")

        assert report is not None
        assert isinstance(report, DataQualityReport)
        mock_ai_client.generate.assert_called()

    def test_generate_expectations(self, analyzer, mock_ai_client):
        """Test generating expectations."""
        schema = [
            {"name": "id", "type": "int"},
            {"name": "email", "type": "string"}
        ]

        expectations = analyzer.generate_expectations(schema)

        assert isinstance(expectations, list)
        mock_ai_client.generate.assert_called()

    def test_generate_dlt_expectations(self, analyzer, mock_ai_client):
        """Test generating DLT expectations."""
        mock_ai_client.generate.return_value = """
@dlt.expect_or_drop("valid_id", "id IS NOT NULL")
@dlt.expect("valid_email", "email LIKE '%@%.%'")
"""

        expectations = [
            DataExpectation(
                column="id",
                expectation_type="not_null",
                parameters={},
                description="ID not null"
            )
        ]

        dlt_code = analyzer.to_dlt_expectations(expectations)

        assert "expect" in dlt_code.lower()

    def test_generate_great_expectations(self, analyzer):
        """Test generating Great Expectations suite."""
        expectations = [
            DataExpectation(
                column="id",
                expectation_type="not_null",
                parameters={},
                description="ID not null"
            ),
            DataExpectation(
                column="email",
                expectation_type="unique",
                parameters={},
                description="Email unique"
            )
        ]

        ge_suite = analyzer.to_great_expectations(expectations)

        assert isinstance(ge_suite, dict)
        assert "expectations" in ge_suite

    def test_analyze_with_sample_data(self, analyzer, mock_ai_client, mock_spark):
        """Test analysis with sample data."""
        # Mock sample data
        mock_df = MagicMock()
        mock_df.limit.return_value.collect.return_value = [
            {"id": 1, "email": "test@test.com"},
            {"id": 2, "email": "user@example.com"}
        ]
        mock_spark.table.return_value = mock_df

        report = analyzer.analyze_table(
            "catalog.schema.users",
            include_sample_data=True
        )

        assert report is not None

    def test_common_patterns_detection(self, analyzer, mock_ai_client):
        """Test detection of common patterns."""
        mock_ai_client.generate.return_value = """
Based on column names, detected patterns:
- email: Should be valid email format
- phone: Should match phone number pattern
- created_at: Should not be in the future

```json
[
    {"column": "email", "type": "regex_match", "pattern": "email_pattern"},
    {"column": "phone", "type": "regex_match", "pattern": "phone_pattern"},
    {"column": "created_at", "type": "max_value", "max": "current_timestamp()"}
]
```
"""

        schema = [
            {"name": "email", "type": "string"},
            {"name": "phone", "type": "string"},
            {"name": "created_at", "type": "timestamp"}
        ]

        expectations = analyzer.generate_expectations(schema)

        # Should have detected patterns
        mock_ai_client.generate.assert_called()


class TestDataQualityAnalyzerEdgeCases:
    """Edge case tests for DataQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mock client."""
        client = Mock()
        client.generate = Mock(return_value="[]")
        spark = MagicMock()
        return DataQualityAnalyzer(ai_client=client, spark=spark)

    def test_empty_schema(self, analyzer):
        """Test with empty schema."""
        expectations = analyzer.generate_expectations([])

        assert expectations == []

    def test_invalid_response(self, analyzer):
        """Test handling invalid AI response."""
        analyzer.ai_client.generate.return_value = "Invalid JSON response"

        # Should handle gracefully
        schema = [{"name": "id", "type": "int"}]
        expectations = analyzer.generate_expectations(schema)

        # Should return empty or partial results, not crash
        assert isinstance(expectations, list)


class TestCreateDataQualityAnalyzer:
    """Tests for factory function."""

    def test_create_analyzer(self):
        """Test creating an analyzer."""
        mock_client = Mock()
        mock_spark = MagicMock()

        analyzer = create_data_quality_analyzer(
            ai_client=mock_client,
            spark=mock_spark
        )

        assert isinstance(analyzer, DataQualityAnalyzer)


class TestDLTExpectationGeneration:
    """Tests for DLT expectation code generation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        client = Mock()
        spark = MagicMock()
        return DataQualityAnalyzer(ai_client=client, spark=spark)

    def test_not_null_expectation(self, analyzer):
        """Test not_null expectation generation."""
        expectations = [
            DataExpectation(
                column="id",
                expectation_type="not_null",
                parameters={},
                description="ID not null"
            )
        ]

        dlt_code = analyzer.to_dlt_expectations(expectations)

        assert "id" in dlt_code.lower()
        assert "not null" in dlt_code.lower() or "is not null" in dlt_code.lower()

    def test_unique_expectation(self, analyzer):
        """Test unique expectation generation."""
        expectations = [
            DataExpectation(
                column="email",
                expectation_type="unique",
                parameters={},
                description="Email unique"
            )
        ]

        dlt_code = analyzer.to_dlt_expectations(expectations)

        assert "email" in dlt_code.lower()

    def test_range_expectation(self, analyzer):
        """Test range expectation generation."""
        expectations = [
            DataExpectation(
                column="age",
                expectation_type="between",
                parameters={"min": 0, "max": 150},
                description="Age between 0 and 150"
            )
        ]

        dlt_code = analyzer.to_dlt_expectations(expectations)

        assert "age" in dlt_code.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
