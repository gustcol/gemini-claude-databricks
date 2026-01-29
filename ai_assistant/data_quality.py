"""
Data Quality Module for AI Assistant.

This module provides AI-powered data quality capabilities including
automatic expectation generation, anomaly detection, and data contract
creation.

Features:
- Automatic Great Expectations generation
- DLT expectation generation
- Anomaly detection with AI
- Data contract suggestions
- Quality scoring and reporting
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json


class ExpectationType(Enum):
    """Types of data quality expectations."""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    IN_SET = "in_set"
    IN_RANGE = "in_range"
    REGEX_MATCH = "regex_match"
    REFERENTIAL = "referential"
    CUSTOM = "custom"


class Severity(Enum):
    """Severity levels for data quality issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DataExpectation:
    """
    A data quality expectation.

    Attributes:
        name: Expectation name
        column: Column to check (or None for table-level)
        expectation_type: Type of expectation
        parameters: Expectation parameters
        severity: Issue severity if expectation fails
        description: Human-readable description
        dlt_syntax: DLT expectation decorator syntax
        sql_check: SQL query to validate
    """
    name: str
    column: Optional[str]
    expectation_type: ExpectationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.HIGH
    description: str = ""
    dlt_syntax: str = ""
    sql_check: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "column": self.column,
            "type": self.expectation_type.value,
            "parameters": self.parameters,
            "severity": self.severity.value,
            "description": self.description,
            "dlt_syntax": self.dlt_syntax,
            "sql_check": self.sql_check
        }


@dataclass
class DataQualityReport:
    """
    Report of data quality analysis.

    Attributes:
        table_name: Table analyzed
        expectations: Generated expectations
        anomalies: Detected anomalies
        score: Quality score (0-100)
        recommendations: AI recommendations
        metadata: Additional metadata
    """
    table_name: str
    expectations: List[DataExpectation] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 100.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_name": self.table_name,
            "expectations": [e.to_dict() for e in self.expectations],
            "anomalies": self.anomalies,
            "score": self.score,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }

    def to_dlt_code(self) -> str:
        """Generate DLT expectation code."""
        lines = [
            "# Data Quality Expectations",
            f"# Table: {self.table_name}",
            ""
        ]

        for exp in self.expectations:
            if exp.dlt_syntax:
                lines.append(exp.dlt_syntax)

        return "\n".join(lines)


class DataQualityAnalyzer:
    """
    AI-powered data quality analyzer.

    Analyzes tables and generates appropriate data quality
    expectations using AI.

    Args:
        ai_client: AI client for analysis
        spark: SparkSession for data access

    Example:
        >>> analyzer = DataQualityAnalyzer(assistant.claude, spark)
        >>> report = analyzer.analyze_table("catalog.schema.table")
        >>> print(report.to_dlt_code())
    """

    def __init__(self, ai_client: Any, spark: Any = None):
        self.ai_client = ai_client
        self.spark = spark

    def analyze_table(
        self,
        table_name: str,
        sample_size: int = 1000,
        include_ai_analysis: bool = True
    ) -> DataQualityReport:
        """
        Analyze a table and generate quality expectations.

        Args:
            table_name: Full table name (catalog.schema.table)
            sample_size: Number of rows to sample
            include_ai_analysis: Whether to use AI for analysis

        Returns:
            DataQualityReport with expectations
        """
        report = DataQualityReport(table_name=table_name)

        # Get schema info
        schema_info = self._get_schema_info(table_name)
        profile = self._profile_data(table_name, sample_size)

        # Generate rule-based expectations
        expectations = self._generate_rule_based_expectations(
            schema_info, profile
        )
        report.expectations.extend(expectations)

        # Use AI for additional analysis
        if include_ai_analysis and self.ai_client:
            ai_expectations = self._generate_ai_expectations(
                table_name, schema_info, profile
            )
            report.expectations.extend(ai_expectations)

            # Get AI recommendations
            report.recommendations = self._get_ai_recommendations(
                table_name, schema_info, profile
            )

        # Detect anomalies
        report.anomalies = self._detect_anomalies(profile)

        # Calculate quality score
        report.score = self._calculate_score(profile, report.anomalies)

        report.metadata = {
            "sample_size": sample_size,
            "schema": schema_info,
            "profile_summary": self._summarize_profile(profile)
        }

        return report

    def _get_schema_info(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information."""
        if not self.spark:
            return {"columns": [], "error": "No Spark session"}

        try:
            desc = self.spark.sql(f"DESCRIBE TABLE {table_name}")
            columns = []

            for row in desc.collect():
                col_name = row[0]
                if col_name and not col_name.startswith("#"):
                    columns.append({
                        "name": col_name,
                        "type": row[1] if row[1] else "",
                        "comment": row[2] if len(row) > 2 and row[2] else ""
                    })

            return {"columns": columns}

        except Exception as e:
            return {"columns": [], "error": str(e)}

    def _profile_data(
        self,
        table_name: str,
        sample_size: int
    ) -> Dict[str, Any]:
        """Profile data in the table."""
        if not self.spark:
            return {}

        try:
            df = self.spark.sql(
                f"SELECT * FROM {table_name} LIMIT {sample_size}"
            )
            pdf = df.toPandas()

            profile = {"columns": {}}

            for col in pdf.columns:
                col_profile = {
                    "dtype": str(pdf[col].dtype),
                    "null_count": int(pdf[col].isna().sum()),
                    "null_rate": float(pdf[col].isna().mean()),
                    "unique_count": int(pdf[col].nunique()),
                    "total_count": len(pdf)
                }

                # Numeric stats
                if pdf[col].dtype in ['int64', 'float64']:
                    col_profile.update({
                        "min": float(pdf[col].min()) if not pdf[col].isna().all() else None,
                        "max": float(pdf[col].max()) if not pdf[col].isna().all() else None,
                        "mean": float(pdf[col].mean()) if not pdf[col].isna().all() else None
                    })

                # String stats
                elif pdf[col].dtype == 'object':
                    non_null = pdf[col].dropna()
                    if len(non_null) > 0:
                        col_profile.update({
                            "min_length": int(non_null.str.len().min()),
                            "max_length": int(non_null.str.len().max()),
                            "sample_values": non_null.head(5).tolist()
                        })

                profile["columns"][col] = col_profile

            return profile

        except Exception as e:
            return {"error": str(e)}

    def _generate_rule_based_expectations(
        self,
        schema_info: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> List[DataExpectation]:
        """Generate expectations based on rules."""
        expectations = []

        for col_info in schema_info.get("columns", []):
            col_name = col_info["name"]
            col_type = col_info["type"].upper()
            col_profile = profile.get("columns", {}).get(col_name, {})

            # Not null for likely key columns
            if any(k in col_name.lower() for k in ["id", "key", "_pk"]):
                expectations.append(DataExpectation(
                    name=f"{col_name}_not_null",
                    column=col_name,
                    expectation_type=ExpectationType.NOT_NULL,
                    severity=Severity.CRITICAL,
                    description=f"{col_name} should not be null (likely key column)",
                    dlt_syntax=f'@dlt.expect_or_drop("{col_name}_not_null", "{col_name} IS NOT NULL")',
                    sql_check=f"SELECT COUNT(*) FROM {{table}} WHERE {col_name} IS NULL"
                ))

            # Unique for ID columns
            if col_name.lower().endswith("_id") or col_name.lower() == "id":
                unique_rate = col_profile.get("unique_count", 0) / max(col_profile.get("total_count", 1), 1)
                if unique_rate > 0.99:  # Likely unique
                    expectations.append(DataExpectation(
                        name=f"{col_name}_unique",
                        column=col_name,
                        expectation_type=ExpectationType.UNIQUE,
                        severity=Severity.HIGH,
                        description=f"{col_name} should be unique",
                        dlt_syntax=f'@dlt.expect("{col_name}_unique", "COUNT(DISTINCT {col_name}) = COUNT(*)")',
                        sql_check=f"SELECT {col_name}, COUNT(*) FROM {{table}} GROUP BY {col_name} HAVING COUNT(*) > 1"
                    ))

            # Range checks for numeric columns
            if "INT" in col_type or "DOUBLE" in col_type or "DECIMAL" in col_type:
                min_val = col_profile.get("min")
                max_val = col_profile.get("max")

                if min_val is not None and min_val >= 0:
                    expectations.append(DataExpectation(
                        name=f"{col_name}_positive",
                        column=col_name,
                        expectation_type=ExpectationType.IN_RANGE,
                        parameters={"min": 0},
                        severity=Severity.MEDIUM,
                        description=f"{col_name} should be non-negative",
                        dlt_syntax=f'@dlt.expect("{col_name}_positive", "{col_name} >= 0")',
                        sql_check=f"SELECT COUNT(*) FROM {{table}} WHERE {col_name} < 0"
                    ))

        return expectations

    def _generate_ai_expectations(
        self,
        table_name: str,
        schema_info: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> List[DataExpectation]:
        """Use AI to generate additional expectations."""
        prompt = f"""Analyze this table and suggest data quality expectations:

Table: {table_name}

Schema:
{json.dumps(schema_info, indent=2)}

Data Profile:
{json.dumps(self._summarize_profile(profile), indent=2)}

Generate additional data quality expectations in JSON format.
For each expectation include:
- name: Expectation name
- column: Column name (or null for table-level)
- type: Type (not_null, unique, in_set, in_range, regex_match, referential, custom)
- parameters: Any parameters needed
- severity: critical, high, medium, low, info
- description: Human-readable description
- sql_check: SQL query to validate

Return only valid JSON array."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a data quality expert. Generate practical, useful data quality expectations."
            )

            # Parse JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                expectations_data = json.loads(json_match.group())

                expectations = []
                for exp_data in expectations_data[:10]:  # Limit to 10
                    expectations.append(DataExpectation(
                        name=exp_data.get("name", "unnamed"),
                        column=exp_data.get("column"),
                        expectation_type=ExpectationType(exp_data.get("type", "custom")),
                        parameters=exp_data.get("parameters", {}),
                        severity=Severity(exp_data.get("severity", "medium")),
                        description=exp_data.get("description", ""),
                        sql_check=exp_data.get("sql_check", "")
                    ))

                return expectations

        except Exception:
            pass

        return []

    def _get_ai_recommendations(
        self,
        table_name: str,
        schema_info: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> List[str]:
        """Get AI recommendations for data quality improvement."""
        prompt = f"""Based on this table analysis, provide data quality recommendations:

Table: {table_name}
Schema: {json.dumps(schema_info, indent=2)}
Profile: {json.dumps(self._summarize_profile(profile), indent=2)}

Provide 3-5 specific, actionable recommendations for improving data quality.
Return as a simple list of recommendations."""

        try:
            response = self.ai_client.generate(
                prompt,
                system_instruction="You are a data quality expert. Provide practical recommendations."
            )

            # Parse recommendations from response
            lines = response.strip().split("\n")
            recommendations = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Clean up the line
                    clean_line = line.lstrip("0123456789.-) ").strip()
                    if clean_line:
                        recommendations.append(clean_line)

            return recommendations[:5]

        except Exception:
            return ["Unable to generate AI recommendations"]

    def _detect_anomalies(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data profile."""
        anomalies = []

        for col_name, col_profile in profile.get("columns", {}).items():
            # High null rate
            null_rate = col_profile.get("null_rate", 0)
            if null_rate > 0.5:
                anomalies.append({
                    "type": "high_null_rate",
                    "column": col_name,
                    "value": null_rate,
                    "description": f"Column {col_name} has {null_rate:.1%} null values"
                })

            # Low uniqueness for potential ID column
            if "id" in col_name.lower():
                unique_rate = col_profile.get("unique_count", 0) / max(col_profile.get("total_count", 1), 1)
                if unique_rate < 0.9:
                    anomalies.append({
                        "type": "low_uniqueness",
                        "column": col_name,
                        "value": unique_rate,
                        "description": f"ID column {col_name} has only {unique_rate:.1%} unique values"
                    })

        return anomalies

    def _calculate_score(
        self,
        profile: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall data quality score."""
        score = 100.0

        # Deduct for anomalies
        for anomaly in anomalies:
            if anomaly["type"] == "high_null_rate":
                score -= 10
            elif anomaly["type"] == "low_uniqueness":
                score -= 15

        # Deduct for high overall null rates
        total_null_rate = 0
        num_cols = 0
        for col_profile in profile.get("columns", {}).values():
            total_null_rate += col_profile.get("null_rate", 0)
            num_cols += 1

        if num_cols > 0:
            avg_null_rate = total_null_rate / num_cols
            score -= avg_null_rate * 20

        return max(0, min(100, score))

    def _summarize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the profile for AI analysis."""
        summary = {}

        for col_name, col_profile in profile.get("columns", {}).items():
            summary[col_name] = {
                "dtype": col_profile.get("dtype"),
                "null_rate": col_profile.get("null_rate"),
                "unique_count": col_profile.get("unique_count")
            }

            if "min" in col_profile:
                summary[col_name]["min"] = col_profile["min"]
                summary[col_name]["max"] = col_profile["max"]

        return summary

    def generate_great_expectations(
        self,
        table_name: str,
        expectations: Optional[List[DataExpectation]] = None
    ) -> str:
        """
        Generate Great Expectations suite code.

        Args:
            table_name: Table name
            expectations: Expectations to include (or analyze table)

        Returns:
            Great Expectations Python code
        """
        if expectations is None:
            report = self.analyze_table(table_name)
            expectations = report.expectations

        code = f'''"""
Great Expectations Suite for {table_name}
Auto-generated by AI Assistant
"""

import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

# Create context
context = gx.get_context()

# Create expectation suite
suite = context.add_expectation_suite(
    expectation_suite_name="{table_name.replace('.', '_')}_suite"
)

# Add expectations
'''

        for exp in expectations:
            if exp.expectation_type == ExpectationType.NOT_NULL:
                code += f'''
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={{"column": "{exp.column}"}}
    )
)
'''
            elif exp.expectation_type == ExpectationType.UNIQUE:
                code += f'''
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_unique",
        kwargs={{"column": "{exp.column}"}}
    )
)
'''
            elif exp.expectation_type == ExpectationType.IN_RANGE:
                min_val = exp.parameters.get("min", "None")
                max_val = exp.parameters.get("max", "None")
                code += f'''
suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={{"column": "{exp.column}", "min_value": {min_val}, "max_value": {max_val}}}
    )
)
'''

        code += '''
# Save suite
context.save_expectation_suite(suite)
print(f"Created expectation suite with {len(suite.expectations)} expectations")
'''

        return code


def create_data_quality_analyzer(
    ai_client: Any,
    spark: Any = None
) -> DataQualityAnalyzer:
    """
    Factory function to create a DataQualityAnalyzer.

    Args:
        ai_client: AI client for analysis
        spark: SparkSession

    Returns:
        Configured DataQualityAnalyzer

    Example:
        >>> from ai_assistant import AIAssistant
        >>> from ai_assistant.data_quality import create_data_quality_analyzer
        >>>
        >>> assistant = AIAssistant(secret_scope="ai-keys")
        >>> analyzer = create_data_quality_analyzer(assistant.claude, spark)
        >>> report = analyzer.analyze_table("catalog.schema.table")
    """
    return DataQualityAnalyzer(ai_client, spark)
