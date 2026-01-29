"""
dbt Integration for AI Assistant.

This module provides AI-powered integration with dbt (data build tool),
including model generation, conversion between DLT and dbt, and
documentation generation.

Features:
- Generate dbt models from descriptions
- Convert DLT pipelines to dbt
- Generate dbt documentation
- Create dbt tests
- Schema.yml generation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re
import yaml


@dataclass
class DBTModel:
    """
    Represents a dbt model.

    Attributes:
        name: Model name
        description: Model description
        sql: SQL code
        config: Model configuration
        columns: Column definitions
        tests: Model-level tests
    """
    name: str
    description: str
    sql: str
    config: Dict[str, Any] = field(default_factory=dict)
    columns: List[Dict[str, Any]] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)

    def to_sql_file(self) -> str:
        """Generate dbt model SQL file content."""
        config_str = ""
        if self.config:
            config_items = [f"    {k}='{v}'" if isinstance(v, str) else f"    {k}={v}"
                          for k, v in self.config.items()]
            config_str = f"""{{% config(
{chr(10).join(config_items)}
) %}}

"""

        return f"""-- {self.name}
-- {self.description}

{config_str}{self.sql}
"""

    def to_schema_entry(self) -> Dict[str, Any]:
        """Generate schema.yml entry for this model."""
        entry = {
            "name": self.name,
            "description": self.description,
        }

        if self.columns:
            entry["columns"] = self.columns

        if self.tests:
            entry["tests"] = self.tests

        return entry


@dataclass
class DBTProject:
    """
    Represents a dbt project structure.

    Attributes:
        name: Project name
        models: List of models
        sources: List of source definitions
        macros: List of macros
    """
    name: str
    models: List[DBTModel] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    macros: List[Dict[str, str]] = field(default_factory=list)

    def generate_schema_yml(self) -> str:
        """Generate schema.yml content."""
        schema = {
            "version": 2,
            "models": [model.to_schema_entry() for model in self.models]
        }

        if self.sources:
            schema["sources"] = self.sources

        return yaml.dump(schema, default_flow_style=False, sort_keys=False)


class DBTIntegration:
    """
    AI-powered dbt integration.

    Provides tools for generating dbt models, converting pipelines,
    and creating documentation.

    Args:
        ai_client: AI client for generation
        project_name: dbt project name

    Example:
        >>> integration = DBTIntegration(assistant.claude, "my_project")
        >>> model = integration.generate_model(
        ...     "Create a model for daily sales aggregation",
        ...     source_table="raw.sales"
        ... )
        >>> print(model.to_sql_file())
    """

    def __init__(self, ai_client: Any, project_name: str = "dbt_project"):
        self.ai_client = ai_client
        self.project_name = project_name

    def generate_model(
        self,
        description: str,
        source_table: Optional[str] = None,
        materialization: str = "table",
        include_tests: bool = True
    ) -> DBTModel:
        """
        Generate a dbt model from description.

        Args:
            description: Natural language description
            source_table: Source table for the model
            materialization: Materialization strategy
            include_tests: Whether to generate tests

        Returns:
            DBTModel with generated content
        """
        prompt = f"""Generate a dbt model for:

Description: {description}
{f'Source Table: {source_table}' if source_table else ''}
Materialization: {materialization}

Generate the model with:
1. Proper dbt SQL (use Jinja/ref/source as appropriate)
2. Clear CTE structure
3. Column documentation

Return in this JSON format:
{{
    "name": "model_name",
    "description": "Model description",
    "sql": "SELECT ... FROM ...",
    "columns": [
        {{"name": "col", "description": "desc", "tests": ["not_null"]}}
    ]
}}

Use snake_case for model name. Include appropriate dbt functions."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a dbt expert. Generate clean, well-structured dbt models."
            )

            # Parse JSON
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return DBTModel(
                    name=data.get("name", "generated_model"),
                    description=data.get("description", description),
                    sql=data.get("sql", ""),
                    config={"materialized": materialization},
                    columns=data.get("columns", []),
                    tests=data.get("tests", [])
                )

        except Exception as e:
            pass

        # Fallback
        return DBTModel(
            name="generated_model",
            description=description,
            sql=f"-- Generation failed\nSELECT * FROM {source_table or 'source_table'}",
            config={"materialized": materialization}
        )

    def convert_dlt_to_dbt(self, dlt_code: str) -> List[DBTModel]:
        """
        Convert DLT pipeline code to dbt models.

        Args:
            dlt_code: DLT pipeline Python code

        Returns:
            List of DBTModel objects
        """
        prompt = f"""Convert this Delta Live Tables pipeline to dbt models:

```python
{dlt_code}
```

For each @dlt.table, create a separate dbt model.
Maintain the same logic and transformations.
Use appropriate dbt functions (ref, source, etc.).

Return as JSON array:
[
    {{
        "name": "model_name",
        "description": "description",
        "sql": "dbt SQL code",
        "columns": [...]
    }}
]"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are an expert in both DLT and dbt. Convert accurately."
            )

            # Parse JSON
            import json
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                models_data = json.loads(json_match.group())

                return [
                    DBTModel(
                        name=m.get("name", "model"),
                        description=m.get("description", ""),
                        sql=m.get("sql", ""),
                        columns=m.get("columns", [])
                    )
                    for m in models_data
                ]

        except Exception:
            pass

        return []

    def convert_dbt_to_dlt(self, dbt_model: DBTModel) -> str:
        """
        Convert a dbt model to DLT code.

        Args:
            dbt_model: DBTModel to convert

        Returns:
            DLT Python code
        """
        prompt = f"""Convert this dbt model to Delta Live Tables:

Model: {dbt_model.name}
Description: {dbt_model.description}

SQL:
```sql
{dbt_model.sql}
```

Generate DLT Python code using:
- @dlt.table decorator
- Proper expectations from column tests
- PySpark or Spark SQL as appropriate"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are an expert in both dbt and DLT. Generate clean DLT code."
            )

            # Extract code
            if "```python" in response:
                match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
                if match:
                    return match.group(1)

            return response

        except Exception as e:
            return f"# Conversion failed: {e}"

    def generate_source_definition(
        self,
        database: str,
        schema: str,
        tables: List[str]
    ) -> Dict[str, Any]:
        """
        Generate dbt source definition.

        Args:
            database: Database/catalog name
            schema: Schema name
            tables: List of table names

        Returns:
            Source definition dictionary
        """
        return {
            "name": f"{database}_{schema}",
            "database": database,
            "schema": schema,
            "tables": [{"name": table} for table in tables]
        }

    def generate_tests(
        self,
        model: DBTModel,
        test_types: Optional[List[str]] = None
    ) -> str:
        """
        Generate dbt tests for a model.

        Args:
            model: Model to test
            test_types: Types of tests to generate

        Returns:
            Test SQL or schema.yml test definitions
        """
        test_types = test_types or ["not_null", "unique", "relationships"]

        prompt = f"""Generate dbt tests for this model:

Model: {model.name}
Description: {model.description}
Columns: {[c.get('name') for c in model.columns]}

Generate tests including:
- {chr(10).join(f'- {t}' for t in test_types)}

Return as YAML for schema.yml:
columns:
  - name: column_name
    tests:
      - test_type"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a dbt testing expert."
            )
            return response

        except Exception as e:
            return f"# Test generation failed: {e}"

    def generate_documentation(
        self,
        model: DBTModel
    ) -> str:
        """
        Generate dbt documentation for a model.

        Args:
            model: Model to document

        Returns:
            Documentation markdown
        """
        prompt = f"""Generate comprehensive dbt documentation for:

Model: {model.name}
Description: {model.description}

SQL:
```sql
{model.sql}
```

Generate documentation including:
1. Model overview
2. Business context
3. Column descriptions
4. Example queries
5. Dependencies
6. Refresh schedule recommendation"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a dbt documentation expert."
            )
            return response

        except Exception as e:
            return f"# Documentation failed: {e}"

    def generate_staging_model(
        self,
        source_name: str,
        source_table: str,
        columns: Optional[List[Dict[str, str]]] = None
    ) -> DBTModel:
        """
        Generate a staging model following dbt best practices.

        Args:
            source_name: dbt source name
            source_table: Source table name
            columns: Column definitions

        Returns:
            DBTModel for staging layer
        """
        col_str = ""
        if columns:
            col_str = f"Columns: {columns}"

        prompt = f"""Generate a dbt staging model for:

Source: {source_name}.{source_table}
{col_str}

Follow dbt staging best practices:
1. Use source() function
2. Rename columns to snake_case
3. Cast data types explicitly
4. Add surrogate key if needed
5. Use consistent naming (stg_source__table)

Return JSON:
{{
    "name": "stg_...",
    "sql": "...",
    "columns": [...]
}}"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a dbt expert following dbt best practices."
            )

            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                return DBTModel(
                    name=data.get("name", f"stg_{source_name}__{source_table}"),
                    description=f"Staging model for {source_name}.{source_table}",
                    sql=data.get("sql", f"SELECT * FROM {{{{ source('{source_name}', '{source_table}') }}}}"),
                    config={"materialized": "view"},
                    columns=data.get("columns", [])
                )

        except Exception:
            pass

        # Fallback
        return DBTModel(
            name=f"stg_{source_name}__{source_table}",
            description=f"Staging model for {source_name}.{source_table}",
            sql=f"SELECT * FROM {{{{ source('{source_name}', '{source_table}') }}}}",
            config={"materialized": "view"}
        )


def create_dbt_integration(
    ai_client: Any,
    project_name: str = "dbt_project"
) -> DBTIntegration:
    """
    Factory function to create a DBTIntegration.

    Args:
        ai_client: AI client for generation
        project_name: dbt project name

    Returns:
        Configured DBTIntegration

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.dbt_integration import create_dbt_integration
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> dbt = create_dbt_integration(assistant.claude)
        >>> model = dbt.generate_model("Daily sales summary")
    """
    return DBTIntegration(ai_client, project_name)
