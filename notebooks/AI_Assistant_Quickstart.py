# Databricks notebook source
# MAGIC %md
# MAGIC # AI Assistant Quickstart for Databricks
# MAGIC
# MAGIC This notebook demonstrates how to use Gemini and Claude AI models within Databricks
# MAGIC for development acceleration, code generation, and data analysis.
# MAGIC
# MAGIC ## Setup Requirements
# MAGIC
# MAGIC 1. **API Keys**: Configure your API keys using Databricks Secrets
# MAGIC 2. **Libraries**: Install required packages using the cell below

# COMMAND ----------

# MAGIC %pip install google-generativeai anthropic --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure API Keys
# MAGIC
# MAGIC ### Option A: Using Databricks Secrets (Recommended)
# MAGIC
# MAGIC Run these commands in your terminal (Databricks CLI):
# MAGIC ```bash
# MAGIC databricks secrets create-scope --scope ai-keys
# MAGIC databricks secrets put --scope ai-keys --key gemini-api-key
# MAGIC databricks secrets put --scope ai-keys --key claude-api-key
# MAGIC ```

# COMMAND ----------

# Configuration - Update these values
SECRET_SCOPE = "ai-keys"          # Your Databricks secret scope
GEMINI_SECRET_KEY = "gemini-api-key"
CLAUDE_SECRET_KEY = "claude-api-key"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize the AI Assistant

# COMMAND ----------

# Copy the ai_assistant module or install from package
# For this quickstart, we'll define a simplified version inline

import os
from typing import Optional, Generator

class DatabricksAIAssistant:
    """
    AI Assistant for Databricks - Unified interface for Gemini and Claude.

    This class provides easy access to both AI models with automatic
    API key management via Databricks Secrets.
    """

    def __init__(
        self,
        secret_scope: str = None,
        gemini_secret_key: str = "gemini-api-key",
        claude_secret_key: str = "claude-api-key"
    ):
        """Initialize with Databricks secret scope."""
        self.secret_scope = secret_scope
        self.gemini_secret_key = gemini_secret_key
        self.claude_secret_key = claude_secret_key

        self._gemini_client = None
        self._claude_client = None
        self._gemini_key = None
        self._claude_key = None

        # Load API keys
        self._load_keys()

    def _load_keys(self):
        """Load API keys from Databricks secrets or environment."""
        if self.secret_scope:
            try:
                self._gemini_key = dbutils.secrets.get(
                    scope=self.secret_scope,
                    key=self.gemini_secret_key
                )
            except Exception:
                pass

            try:
                self._claude_key = dbutils.secrets.get(
                    scope=self.secret_scope,
                    key=self.claude_secret_key
                )
            except Exception:
                pass

        # Fallback to environment
        if not self._gemini_key:
            self._gemini_key = os.environ.get("GEMINI_API_KEY")
        if not self._claude_key:
            self._claude_key = os.environ.get("ANTHROPIC_API_KEY")

    @property
    def gemini(self):
        """Get Gemini client."""
        if self._gemini_client is None and self._gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=self._gemini_key)
            self._gemini_client = genai.GenerativeModel('gemini-1.5-pro')
        return self._gemini_client

    @property
    def claude(self):
        """Get Claude client."""
        if self._claude_client is None and self._claude_key:
            import anthropic
            self._claude_client = anthropic.Anthropic(api_key=self._claude_key)
        return self._claude_client

    def ask_gemini(self, prompt: str, system_instruction: str = None) -> str:
        """Ask Gemini a question."""
        if not self.gemini:
            return "Gemini API key not configured"

        try:
            if system_instruction:
                import google.generativeai as genai
                model = genai.GenerativeModel(
                    'gemini-1.5-pro',
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt)
            else:
                response = self.gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def ask_claude(
        self,
        prompt: str,
        system_instruction: str = "You are a helpful AI assistant.",
        max_tokens: int = 4096
    ) -> str:
        """Ask Claude a question."""
        if not self.claude:
            return "Claude API key not configured"

        try:
            message = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_instruction,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

    def ask(
        self,
        prompt: str,
        model: str = "claude",
        system_instruction: str = None
    ) -> str:
        """Ask either model."""
        if model.lower() == "gemini":
            return self.ask_gemini(prompt, system_instruction)
        return self.ask_claude(prompt, system_instruction or "You are a helpful AI assistant.")

    def stream_claude(self, prompt: str, system_instruction: str = None) -> Generator:
        """Stream response from Claude."""
        if not self.claude:
            yield "Claude API key not configured"
            return

        try:
            with self.claude.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_instruction or "You are a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"Error: {str(e)}"

    def generate_code(self, task: str, language: str = "python") -> str:
        """Generate code for a specific task."""
        system = """You are an expert software engineer specializing in:
- Apache Spark and PySpark
- Databricks platform and Delta Lake
- Python best practices and data engineering

Generate clean, production-ready code with comments."""

        prompt = f"""Generate {language} code for:
{task}

Include necessary imports and error handling."""

        return self.ask_claude(prompt, system)

    def analyze_dataframe(self, df) -> str:
        """Analyze a Spark DataFrame."""
        # Get schema and sample
        schema = df._jdf.schema().treeString()
        row_count = df.count()
        sample = df.limit(5).toPandas().to_string()

        prompt = f"""Analyze this Spark DataFrame:

Schema:
{schema}

Row count: {row_count}

Sample data:
{sample}

Provide:
1. Data structure summary
2. Data quality observations
3. Optimization recommendations"""

        return self.ask_claude(prompt)

    def optimize_query(self, query: str, context: str = None) -> str:
        """Get optimization suggestions for a SQL query."""
        prompt = f"""Optimize this Spark SQL query:

```sql
{query}
```

{f'Context: {context}' if context else ''}

Provide specific optimization recommendations and an improved query."""

        return self.ask_claude(prompt)

    def explain_error(self, error: str, code: str = None) -> str:
        """Explain an error and suggest fixes."""
        prompt = f"""Explain this error and provide solutions:

Error: {error}

{f'Code: {code}' if code else ''}

Provide:
1. Error explanation
2. Common causes
3. Step-by-step fix"""

        return self.ask_claude(prompt)

    def status(self) -> dict:
        """Check configuration status."""
        return {
            "gemini_configured": self._gemini_key is not None,
            "claude_configured": self._claude_key is not None,
            "secret_scope": self.secret_scope
        }

# COMMAND ----------

# Initialize the assistant
assistant = DatabricksAIAssistant(secret_scope=SECRET_SCOPE)

# Check status
print("Configuration Status:")
print(assistant.status())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Basic Usage Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ask Questions

# COMMAND ----------

# Ask Claude (default)
response = assistant.ask("What are the best practices for partitioning Delta tables?")
print(response)

# COMMAND ----------

# Ask Gemini specifically
response = assistant.ask_gemini("Explain the difference between Delta Lake and Apache Iceberg")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Code

# COMMAND ----------

# Generate PySpark code
code = assistant.generate_code("""
Create a function that:
1. Reads a Delta table
2. Removes duplicates based on a key column
3. Adds audit columns (load_timestamp, source_file)
4. Returns the cleaned DataFrame
""")
print(code)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze DataFrames

# COMMAND ----------

# Create sample DataFrame
sample_data = [
    (1, "Alice", 1000.0, "2024-01-15"),
    (2, "Bob", None, "2024-01-16"),
    (3, "Charlie", 2500.0, "2024-01-17"),
    (4, "Diana", 1500.0, "invalid-date"),
    (5, None, 3000.0, "2024-01-19"),
]

df = spark.createDataFrame(
    sample_data,
    ["id", "name", "amount", "date"]
)

# Analyze with AI
analysis = assistant.analyze_dataframe(df)
print(analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimize Queries

# COMMAND ----------

slow_query = """
SELECT
    customer_id,
    SUM(amount) as total,
    COUNT(*) as transactions
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY customer_id
HAVING COUNT(*) > 10
ORDER BY total DESC
LIMIT 100
"""

optimization = assistant.optimize_query(
    slow_query,
    context="transactions table has 1 billion rows, partitioned by date"
)
print(optimization)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explain Errors

# COMMAND ----------

error_msg = """
Py4JJavaError: An error occurred while calling o123.save.
: org.apache.spark.SparkException: Job aborted due to stage failure:
Task 0 in stage 5.0 failed 4 times, most recent failure:
Lost task 0.3 in stage 5.0 (TID 123) (10.0.0.5 executor 2):
java.lang.OutOfMemoryError: Java heap space
"""

explanation = assistant.explain_error(error_msg)
print(explanation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Advanced: Streaming Responses

# COMMAND ----------

# Stream a long response
print("Streaming response:")
for chunk in assistant.stream_claude("Write a detailed guide for implementing CDC (Change Data Capture) in Databricks"):
    print(chunk, end="", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Interactive Development Helper

# COMMAND ----------

def dev_help(query: str, model: str = "claude"):
    """Quick helper for development questions."""
    system = """You are a Databricks development expert.
    Provide concise, actionable answers focused on practical solutions.
    Include code examples when relevant."""

    return assistant.ask(query, model=model, system_instruction=system)

# COMMAND ----------

# Example usage
print(dev_help("How do I enable CDF (Change Data Feed) on an existing Delta table?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Code Review Helper

# COMMAND ----------

def review_code(code: str):
    """Review code and provide feedback."""
    system = """You are a senior code reviewer.
    Focus on: correctness, performance, security, and best practices.
    Be constructive and provide specific suggestions."""

    prompt = f"""Review this code:

```python
{code}
```

Provide:
1. Summary
2. Issues found
3. Improvement suggestions"""

    return assistant.ask_claude(prompt, system)

# COMMAND ----------

# Example code review
my_code = """
def process_orders(spark):
    df = spark.read.table("orders")
    df = df.filter("status = 'active'")
    df = df.collect()
    results = []
    for row in df:
        results.append(row.asDict())
    return results
"""

print(review_code(my_code))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tips & Best Practices
# MAGIC
# MAGIC 1. **Use Secrets**: Never hardcode API keys in notebooks
# MAGIC 2. **Be Specific**: More detailed prompts get better results
# MAGIC 3. **Context Matters**: Provide relevant context about your data and environment
# MAGIC 4. **Iterate**: Use follow-up questions for refinement
# MAGIC 5. **Verify**: Always review and test generated code before production use
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC - Install the full `ai_assistant` package for more features
# MAGIC - Set up conversation memory for complex multi-turn interactions
# MAGIC - Explore batch processing for large-scale AI operations
