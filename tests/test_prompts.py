"""
Unit tests for Prompt Templates module.

Tests the prompt template system including variables,
templates, and the prompt library.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from ai_assistant.prompts import (
    PromptVariable,
    PromptTemplate,
    PromptLibrary,
    SQL_OPTIMIZATION_PROMPT,
    DDL_GENERATION_PROMPT,
    PIPELINE_GENERATION_PROMPT,
    ERROR_EXPLANATION_PROMPT,
    CODE_REVIEW_PROMPT,
    DATA_ANALYSIS_PROMPT,
    create_template,
    get_data_engineering_prompts
)


class TestPromptVariable:
    """Tests for PromptVariable dataclass."""

    def test_variable_creation(self):
        """Test creating a variable."""
        var = PromptVariable(
            name="table_name",
            description="Name of the table",
            required=True,
            default=None
        )

        assert var.name == "table_name"
        assert var.description == "Name of the table"
        assert var.required is True
        assert var.default is None

    def test_variable_with_default(self):
        """Test variable with default value."""
        var = PromptVariable(
            name="limit",
            description="Row limit",
            required=False,
            default=100
        )

        assert var.default == 100
        assert var.required is False

    def test_variable_with_examples(self):
        """Test variable with examples."""
        var = PromptVariable(
            name="query",
            description="SQL query",
            required=True,
            examples=["SELECT * FROM t", "SELECT id FROM users"]
        )

        assert len(var.examples) == 2


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    @pytest.fixture
    def simple_template(self):
        """Create a simple template."""
        return PromptTemplate(
            name="test_template",
            description="A test template",
            template="Hello, {name}! Welcome to {place}.",
            variables=[
                PromptVariable(name="name", description="User name", required=True),
                PromptVariable(name="place", description="Location", required=True)
            ]
        )

    @pytest.fixture
    def template_with_defaults(self):
        """Create a template with default values."""
        return PromptTemplate(
            name="greeting",
            description="Greeting template",
            template="Hello, {name}! Today is {day}.",
            variables=[
                PromptVariable(name="name", description="User name", required=True),
                PromptVariable(name="day", description="Day", required=False, default="Monday")
            ]
        )

    def test_template_creation(self, simple_template):
        """Test template creation."""
        assert simple_template.name == "test_template"
        assert len(simple_template.variables) == 2

    def test_format_template(self, simple_template):
        """Test formatting template."""
        result = simple_template.format(name="Alice", place="Databricks")

        assert "Alice" in result
        assert "Databricks" in result

    def test_format_with_defaults(self, template_with_defaults):
        """Test formatting with default values."""
        result = template_with_defaults.format(name="Bob")

        assert "Bob" in result
        assert "Monday" in result

    def test_format_missing_required(self, simple_template):
        """Test formatting with missing required variable."""
        with pytest.raises(ValueError):
            simple_template.format(name="Alice")  # Missing 'place'

    def test_validate_variables(self, simple_template):
        """Test variable validation."""
        # Valid
        assert simple_template.validate(name="Test", place="Here") is True

        # Missing required
        assert simple_template.validate(name="Test") is False

    def test_get_variable_names(self, simple_template):
        """Test getting variable names."""
        names = simple_template.get_variable_names()

        assert "name" in names
        assert "place" in names

    def test_template_with_system_instruction(self):
        """Test template with system instruction."""
        template = PromptTemplate(
            name="sql_help",
            description="SQL help template",
            template="Help with: {query}",
            system_instruction="You are a SQL expert.",
            variables=[
                PromptVariable(name="query", description="SQL query", required=True)
            ]
        )

        assert template.system_instruction == "You are a SQL expert."

    def test_template_versioning(self):
        """Test template versioning."""
        template = PromptTemplate(
            name="versioned",
            description="Versioned template",
            template="Version {version}",
            version="1.0.0",
            variables=[
                PromptVariable(name="version", description="Version", required=True)
            ]
        )

        assert template.version == "1.0.0"


class TestPromptLibrary:
    """Tests for PromptLibrary class."""

    @pytest.fixture
    def library(self):
        """Create a prompt library."""
        return PromptLibrary()

    @pytest.fixture
    def sample_template(self):
        """Create a sample template."""
        return PromptTemplate(
            name="sample",
            description="Sample template",
            template="Sample: {text}",
            variables=[
                PromptVariable(name="text", description="Text", required=True)
            ]
        )

    def test_library_initialization(self, library):
        """Test library initialization."""
        assert library is not None

    def test_register_template(self, library, sample_template):
        """Test registering a template."""
        library.register(sample_template)

        assert "sample" in library._templates

    def test_get_template(self, library, sample_template):
        """Test getting a template."""
        library.register(sample_template)

        template = library.get("sample")
        assert template == sample_template

    def test_get_nonexistent_template(self, library):
        """Test getting nonexistent template."""
        template = library.get("nonexistent")
        assert template is None

    def test_list_templates(self, library, sample_template):
        """Test listing templates."""
        library.register(sample_template)

        templates = library.list()

        assert "sample" in templates

    def test_remove_template(self, library, sample_template):
        """Test removing a template."""
        library.register(sample_template)
        library.remove("sample")

        assert library.get("sample") is None

    def test_search_templates(self, library):
        """Test searching templates."""
        template1 = PromptTemplate(
            name="sql_optimize",
            description="Optimize SQL queries",
            template="Optimize: {query}",
            variables=[PromptVariable(name="query", description="Query", required=True)],
            tags=["sql", "optimization"]
        )
        template2 = PromptTemplate(
            name="python_help",
            description="Python coding help",
            template="Help: {code}",
            variables=[PromptVariable(name="code", description="Code", required=True)],
            tags=["python", "coding"]
        )

        library.register(template1)
        library.register(template2)

        sql_templates = library.search(tag="sql")
        assert len(sql_templates) == 1
        assert sql_templates[0].name == "sql_optimize"


class TestBuiltInTemplates:
    """Tests for built-in templates."""

    def test_sql_optimization_template(self):
        """Test SQL optimization template."""
        assert SQL_OPTIMIZATION_PROMPT is not None
        assert SQL_OPTIMIZATION_PROMPT.name is not None

        # Should have query variable
        var_names = SQL_OPTIMIZATION_PROMPT.get_variable_names()
        assert "query" in var_names

    def test_ddl_generation_template(self):
        """Test DDL generation template."""
        assert DDL_GENERATION_PROMPT is not None

    def test_pipeline_generation_template(self):
        """Test pipeline generation template."""
        assert PIPELINE_GENERATION_PROMPT is not None

    def test_error_explanation_template(self):
        """Test error explanation template."""
        assert ERROR_EXPLANATION_PROMPT is not None

        var_names = ERROR_EXPLANATION_PROMPT.get_variable_names()
        assert "error_message" in var_names

    def test_code_review_template(self):
        """Test code review template."""
        assert CODE_REVIEW_PROMPT is not None

    def test_data_analysis_template(self):
        """Test data analysis template."""
        assert DATA_ANALYSIS_PROMPT is not None


class TestTemplateHelperFunctions:
    """Tests for helper functions."""

    def test_create_template_function(self):
        """Test create_template function."""
        # Should create a template
        template = create_template(
            name="test_template",
            description="Test description",
            template_text="Hello {name}!",
            variables=[
                PromptVariable(name="name", description="Name", required=True)
            ]
        )

        assert template is not None
        assert template.name == "test_template"

    def test_get_data_engineering_prompts_function(self):
        """Test get_data_engineering_prompts function."""
        prompts = get_data_engineering_prompts()

        assert isinstance(prompts, dict)
        # Should have some built-in prompts
        assert len(prompts) > 0


class TestPromptTemplateIntegration:
    """Integration tests for prompt templates."""

    def test_full_workflow(self):
        """Test complete workflow with templates."""
        # Create library
        library = PromptLibrary()

        # Create template
        template = PromptTemplate(
            name="analyze_table",
            description="Analyze a database table",
            template="""
Analyze the following table:
Table: {table_name}
Schema: {schema}

Provide:
1. Data quality assessment
2. Potential issues
3. Recommendations
""",
            system_instruction="You are a data quality expert.",
            variables=[
                PromptVariable(name="table_name", description="Table name", required=True),
                PromptVariable(name="schema", description="Table schema", required=True)
            ],
            tags=["data-quality", "analysis"]
        )

        # Register
        library.register(template)

        # Retrieve and format
        retrieved = library.get("analyze_table")
        assert retrieved is not None

        formatted = retrieved.format(
            table_name="users",
            schema="id INT, name STRING, email STRING"
        )

        assert "users" in formatted
        assert "id INT" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
