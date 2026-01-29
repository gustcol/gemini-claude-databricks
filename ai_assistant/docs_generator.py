"""
Documentation Generator for AI Assistant.

This module provides AI-powered documentation generation capabilities
for code, schemas, notebooks, and data pipelines.

Features:
- Automatic docstring generation
- Schema documentation
- Notebook README generation
- Data dictionary creation
- API documentation
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DocSection:
    """
    A section of generated documentation.

    Attributes:
        title: Section title
        content: Section content
        level: Heading level (1-6)
        subsections: Child sections
    """
    title: str
    content: str
    level: int = 1
    subsections: List["DocSection"] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert section to markdown."""
        md = f"{'#' * self.level} {self.title}\n\n{self.content}\n\n"

        for sub in self.subsections:
            md += sub.to_markdown()

        return md


@dataclass
class FunctionDoc:
    """Documentation for a function."""
    name: str
    description: str
    parameters: List[Dict[str, str]]
    returns: str
    raises: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_docstring(self, style: str = "google") -> str:
        """Generate docstring in specified style."""
        if style == "google":
            return self._to_google_style()
        elif style == "numpy":
            return self._to_numpy_style()
        else:
            return self._to_google_style()

    def _to_google_style(self) -> str:
        """Generate Google-style docstring."""
        lines = [f'"""{self.description}']

        if self.parameters:
            lines.append("")
            lines.append("Args:")
            for param in self.parameters:
                lines.append(f"    {param['name']}: {param['description']}")

        if self.returns:
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {self.returns}")

        if self.raises:
            lines.append("")
            lines.append("Raises:")
            for exc in self.raises:
                lines.append(f"    {exc}")

        if self.examples:
            lines.append("")
            lines.append("Example:")
            for example in self.examples:
                lines.append(f"    >>> {example}")

        lines.append('"""')
        return "\n".join(lines)

    def _to_numpy_style(self) -> str:
        """Generate NumPy-style docstring."""
        lines = [f'"""{self.description}']

        if self.parameters:
            lines.append("")
            lines.append("Parameters")
            lines.append("----------")
            for param in self.parameters:
                lines.append(f"{param['name']} : {param.get('type', 'Any')}")
                lines.append(f"    {param['description']}")

        if self.returns:
            lines.append("")
            lines.append("Returns")
            lines.append("-------")
            lines.append(f"    {self.returns}")

        lines.append('"""')
        return "\n".join(lines)


class DocsGenerator:
    """
    AI-powered documentation generator.

    Generates documentation for code, schemas, and notebooks
    using AI to create clear, comprehensive documentation.

    Args:
        ai_client: AI client for generation
        style: Documentation style (google, numpy)

    Example:
        >>> generator = DocsGenerator(assistant.claude)
        >>> docs = generator.generate_function_docs(my_function_code)
        >>> print(docs.to_docstring())
    """

    def __init__(self, ai_client: Any, style: str = "google"):
        self.ai_client = ai_client
        self.style = style

    def generate_function_docs(self, code: str) -> FunctionDoc:
        """
        Generate documentation for a function.

        Args:
            code: Function source code

        Returns:
            FunctionDoc with generated documentation
        """
        prompt = f"""Generate documentation for this Python function:

```python
{code}
```

Provide documentation in this JSON format:
{{
    "name": "function_name",
    "description": "Clear description of what the function does",
    "parameters": [
        {{"name": "param1", "type": "type", "description": "description"}}
    ],
    "returns": "What the function returns",
    "raises": ["Exception: when..."],
    "examples": ["example usage"]
}}

Be thorough but concise. Return only valid JSON."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a technical documentation expert."
            )

            # Parse JSON from response
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return FunctionDoc(
                    name=data.get("name", "unknown"),
                    description=data.get("description", ""),
                    parameters=data.get("parameters", []),
                    returns=data.get("returns", ""),
                    raises=data.get("raises", []),
                    examples=data.get("examples", [])
                )

        except Exception as e:
            pass

        # Fallback
        return FunctionDoc(
            name="unknown",
            description="Documentation generation failed",
            parameters=[],
            returns=""
        )

    def generate_class_docs(self, code: str) -> str:
        """
        Generate documentation for a class.

        Args:
            code: Class source code

        Returns:
            Generated class docstring
        """
        prompt = f"""Generate comprehensive documentation for this Python class:

```python
{code}
```

Include:
1. Class description
2. Attributes
3. Public methods (brief descriptions)
4. Example usage

Use {self.style} docstring style."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a technical documentation expert. Generate clear, comprehensive docstrings."
            )

            # Extract docstring from response
            if '"""' in response:
                match = re.search(r'""".*?"""', response, re.DOTALL)
                if match:
                    return match.group()

            return response

        except Exception as e:
            return f'"""Documentation generation failed: {e}"""'

    def generate_module_docs(self, code: str, module_name: str = "module") -> str:
        """
        Generate documentation for a module.

        Args:
            code: Module source code
            module_name: Name of the module

        Returns:
            Generated module documentation
        """
        prompt = f"""Generate module-level documentation for this Python module:

Module name: {module_name}

```python
{code}
```

Generate a module docstring that includes:
1. Module description
2. Main features
3. Key classes/functions
4. Usage examples"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a technical documentation expert."
            )
            return response

        except Exception as e:
            return f"Documentation generation failed: {e}"

    def generate_schema_docs(
        self,
        table_name: str,
        schema: List[Dict[str, str]],
        sample_data: Optional[str] = None
    ) -> str:
        """
        Generate documentation for a table schema.

        Args:
            table_name: Full table name
            schema: List of column definitions
            sample_data: Optional sample data

        Returns:
            Generated schema documentation in markdown
        """
        schema_str = "\n".join([
            f"- {col['name']}: {col['type']}"
            for col in schema
        ])

        prompt = f"""Generate comprehensive documentation for this table:

Table: {table_name}

Schema:
{schema_str}

{f'Sample Data:{chr(10)}{sample_data}' if sample_data else ''}

Generate markdown documentation including:
1. Table description
2. Column descriptions (infer from names/types)
3. Data types explanation
4. Potential use cases
5. Example queries"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a data documentation expert."
            )
            return response

        except Exception as e:
            return f"# {table_name}\n\nDocumentation generation failed: {e}"

    def generate_data_dictionary(
        self,
        tables: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a data dictionary for multiple tables.

        Args:
            tables: List of table definitions with schema info

        Returns:
            Complete data dictionary in markdown
        """
        tables_str = ""
        for table in tables:
            tables_str += f"\n## {table['name']}\n"
            for col in table.get('columns', []):
                tables_str += f"- {col['name']}: {col['type']}\n"

        prompt = f"""Generate a comprehensive data dictionary for these tables:

{tables_str}

Include for each table:
1. Business description
2. Column details (description, constraints, examples)
3. Relationships to other tables
4. Data quality notes
5. Common query patterns

Format as a well-organized markdown document."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a data governance expert creating a data dictionary."
            )
            return response

        except Exception as e:
            return f"# Data Dictionary\n\nGeneration failed: {e}"

    def generate_notebook_readme(
        self,
        notebook_content: str,
        notebook_name: str
    ) -> str:
        """
        Generate a README for a notebook.

        Args:
            notebook_content: Notebook cells content
            notebook_name: Name of the notebook

        Returns:
            Generated README in markdown
        """
        prompt = f"""Generate a README for this Databricks notebook:

Notebook: {notebook_name}

Content:
{notebook_content[:5000]}  # Truncate for prompt size

Generate a README with:
1. Title and description
2. Prerequisites
3. Setup instructions
4. Main sections overview
5. Input/output description
6. Example usage
7. Dependencies"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a technical writer creating notebook documentation."
            )
            return response

        except Exception as e:
            return f"# {notebook_name}\n\nREADME generation failed: {e}"

    def generate_pipeline_docs(
        self,
        pipeline_code: str,
        pipeline_name: str
    ) -> str:
        """
        Generate documentation for a data pipeline.

        Args:
            pipeline_code: Pipeline source code
            pipeline_name: Name of the pipeline

        Returns:
            Generated pipeline documentation
        """
        prompt = f"""Generate comprehensive documentation for this data pipeline:

Pipeline: {pipeline_name}

```python
{pipeline_code}
```

Generate documentation including:
1. Pipeline overview
2. Architecture diagram (in mermaid format)
3. Data flow description
4. Source and target tables
5. Transformations applied
6. Data quality checks
7. Dependencies
8. Scheduling requirements
9. Monitoring and alerting"""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a data engineering documentation expert."
            )
            return response

        except Exception as e:
            return f"# {pipeline_name}\n\nDocumentation generation failed: {e}"

    def add_docstrings_to_code(self, code: str) -> str:
        """
        Add docstrings to all functions/classes in code.

        Args:
            code: Python source code

        Returns:
            Code with added docstrings
        """
        prompt = f"""Add comprehensive docstrings to all functions and classes in this code.
Use {self.style} style. Keep the original code intact, only add docstrings.

```python
{code}
```

Return the complete code with docstrings added."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a Python documentation expert. Add clear, helpful docstrings."
            )

            # Extract code from response
            if "```python" in response:
                match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
                if match:
                    return match.group(1)

            return response

        except Exception as e:
            return f"# Docstring generation failed: {e}\n{code}"


def create_docs_generator(
    ai_client: Any,
    style: str = "google"
) -> DocsGenerator:
    """
    Factory function to create a DocsGenerator.

    Args:
        ai_client: AI client for generation
        style: Documentation style (google, numpy)

    Returns:
        Configured DocsGenerator

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.docs_generator import create_docs_generator
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> docs_gen = create_docs_generator(assistant.claude)
        >>> docs = docs_gen.generate_function_docs(my_code)
    """
    return DocsGenerator(ai_client, style)
