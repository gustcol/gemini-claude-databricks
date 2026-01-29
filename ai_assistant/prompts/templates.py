"""
Prompt Template Infrastructure.

This module provides the core infrastructure for creating,
managing, and rendering prompt templates.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class VariableType(Enum):
    """Types of template variables."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    CODE = "code"
    SQL = "sql"


@dataclass
class PromptVariable:
    """
    Definition of a template variable.

    Attributes:
        name: Variable name (used in template as {name})
        description: Description of the variable
        var_type: Type of the variable
        required: Whether the variable is required
        default: Default value if not provided
        examples: Example values for documentation
    """
    name: str
    description: str
    var_type: VariableType = VariableType.STRING
    required: bool = True
    default: Any = None
    examples: List[str] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate a value against this variable's type."""
        if value is None:
            return not self.required

        type_checks = {
            VariableType.STRING: lambda v: isinstance(v, str),
            VariableType.INTEGER: lambda v: isinstance(v, int),
            VariableType.FLOAT: lambda v: isinstance(v, (int, float)),
            VariableType.BOOLEAN: lambda v: isinstance(v, bool),
            VariableType.LIST: lambda v: isinstance(v, list),
            VariableType.DICT: lambda v: isinstance(v, dict),
            VariableType.CODE: lambda v: isinstance(v, str),
            VariableType.SQL: lambda v: isinstance(v, str),
        }

        return type_checks.get(self.var_type, lambda v: True)(value)


@dataclass
class FewShotExample:
    """
    A few-shot example for a prompt template.

    Attributes:
        input_vars: Input variable values
        expected_output: Expected/example output
        explanation: Optional explanation
    """
    input_vars: Dict[str, Any]
    expected_output: str
    explanation: Optional[str] = None


@dataclass
class PromptTemplate:
    """
    A reusable prompt template.

    Attributes:
        name: Template name
        description: Template description
        template: The prompt template string
        variables: List of template variables
        system_instruction: Optional system instruction
        examples: Few-shot examples
        version: Template version
        tags: Tags for categorization
        metadata: Additional metadata

    Example:
        >>> template = PromptTemplate(
        ...     name="sql_optimizer",
        ...     description="Optimize SQL queries",
        ...     template="Optimize this SQL:\\n{query}\\nContext: {context}",
        ...     variables=[
        ...         PromptVariable("query", "SQL query", VariableType.SQL),
        ...         PromptVariable("context", "Context", required=False)
        ...     ]
        ... )
        >>> prompt = template.render(query="SELECT * FROM t", context="Large table")
    """
    name: str
    description: str
    template: str
    variables: List[PromptVariable] = field(default_factory=list)
    system_instruction: Optional[str] = None
    examples: List[FewShotExample] = field(default_factory=list)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If required variables are missing or invalid
        """
        # Validate and apply defaults
        render_vars = {}

        for var in self.variables:
            value = kwargs.get(var.name, var.default)

            if value is None and var.required:
                raise ValueError(f"Missing required variable: {var.name}")

            if value is not None and not var.validate(value):
                raise ValueError(
                    f"Invalid type for {var.name}: expected {var.var_type.value}"
                )

            render_vars[var.name] = value if value is not None else ""

        # Add few-shot examples if available
        prompt = self.template

        if self.examples:
            examples_text = self._format_examples()
            prompt = f"{examples_text}\n\n{prompt}"

        # Render template
        return prompt.format(**render_vars)

    def _format_examples(self) -> str:
        """Format few-shot examples for inclusion in prompt."""
        if not self.examples:
            return ""

        parts = ["Here are some examples:\n"]

        for i, example in enumerate(self.examples, 1):
            parts.append(f"Example {i}:")

            # Format input
            for var_name, var_value in example.input_vars.items():
                parts.append(f"  {var_name}: {var_value}")

            # Format output
            parts.append(f"  Output: {example.expected_output}")

            if example.explanation:
                parts.append(f"  Explanation: {example.explanation}")

            parts.append("")

        return "\n".join(parts)

    def get_variable_names(self) -> List[str]:
        """Get list of variable names."""
        return [v.name for v in self.variables]

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "variables": [
                {
                    "name": v.name,
                    "description": v.description,
                    "type": v.var_type.value,
                    "required": v.required,
                    "default": v.default
                }
                for v in self.variables
            ],
            "system_instruction": self.system_instruction,
            "version": self.version,
            "tags": self.tags
        }


class PromptLibrary:
    """
    Library for managing prompt templates.

    Provides storage, retrieval, and versioning of templates.

    Example:
        >>> library = PromptLibrary()
        >>> library.add(sql_template)
        >>> template = library.get("sql_optimizer")
        >>> prompt = library.render("sql_optimizer", query="SELECT * FROM t")
    """

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._versions: Dict[str, Dict[str, PromptTemplate]] = {}

    def add(self, template: PromptTemplate) -> None:
        """
        Add a template to the library.

        Args:
            template: Template to add
        """
        self._templates[template.name] = template

        # Store version history
        if template.name not in self._versions:
            self._versions[template.name] = {}
        self._versions[template.name][template.version] = template

    def get(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """
        Get a template by name.

        Args:
            name: Template name
            version: Specific version (latest if not specified)

        Returns:
            PromptTemplate or None if not found
        """
        if version:
            return self._versions.get(name, {}).get(version)
        return self._templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        """
        Render a template by name.

        Args:
            name: Template name
            **kwargs: Variable values

        Returns:
            Rendered prompt

        Raises:
            KeyError: If template not found
        """
        template = self.get(name)
        if template is None:
            raise KeyError(f"Template not found: {name}")
        return template.render(**kwargs)

    def list_templates(self, tag: Optional[str] = None) -> List[str]:
        """
        List available templates.

        Args:
            tag: Filter by tag

        Returns:
            List of template names
        """
        if tag:
            return [
                name for name, template in self._templates.items()
                if tag in template.tags
            ]
        return list(self._templates.keys())

    def get_versions(self, name: str) -> List[str]:
        """Get available versions for a template."""
        return list(self._versions.get(name, {}).keys())

    def remove(self, name: str) -> bool:
        """Remove a template from the library."""
        if name in self._templates:
            del self._templates[name]
            if name in self._versions:
                del self._versions[name]
            return True
        return False

    def export_all(self) -> Dict[str, Any]:
        """Export all templates as a dictionary."""
        return {
            name: template.to_dict()
            for name, template in self._templates.items()
        }


def create_template(
    name: str,
    template: str,
    description: str = "",
    variables: Optional[List[Dict[str, Any]]] = None,
    system_instruction: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None
) -> PromptTemplate:
    """
    Factory function to create a prompt template.

    Args:
        name: Template name
        template: Template string
        description: Template description
        variables: List of variable definitions
        system_instruction: System instruction
        examples: Few-shot examples

    Returns:
        PromptTemplate instance

    Example:
        >>> template = create_template(
        ...     name="summarizer",
        ...     template="Summarize this text:\\n{text}",
        ...     variables=[{"name": "text", "description": "Text to summarize"}]
        ... )
    """
    # Parse variables
    prompt_vars = []
    if variables:
        for var in variables:
            prompt_vars.append(PromptVariable(
                name=var["name"],
                description=var.get("description", ""),
                var_type=VariableType(var.get("type", "string")),
                required=var.get("required", True),
                default=var.get("default"),
                examples=var.get("examples", [])
            ))

    # Parse examples
    few_shot_examples = []
    if examples:
        for ex in examples:
            few_shot_examples.append(FewShotExample(
                input_vars=ex.get("input", {}),
                expected_output=ex.get("output", ""),
                explanation=ex.get("explanation")
            ))

    return PromptTemplate(
        name=name,
        description=description,
        template=template,
        variables=prompt_vars,
        system_instruction=system_instruction,
        examples=few_shot_examples
    )


def render_template(template: str, **kwargs) -> str:
    """
    Simple template rendering without PromptTemplate class.

    Args:
        template: Template string with {variable} placeholders
        **kwargs: Variable values

    Returns:
        Rendered string

    Example:
        >>> render_template("Hello {name}!", name="World")
        'Hello World!'
    """
    return template.format(**kwargs)


# Global default library
_default_library = PromptLibrary()


def get_default_library() -> PromptLibrary:
    """Get the default prompt library."""
    return _default_library
