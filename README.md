# Gemini & Claude AI Integration for Databricks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Databricks](https://img.shields.io/badge/Databricks-Ready-orange.svg)](https://databricks.com/)
[![GitHub](https://img.shields.io/badge/GitHub-gustcol%2Fgemini--claude--databricks-blue.svg)](https://github.com/gustcol/gemini-claude-databricks)

A comprehensive Python library that enables seamless integration of **Google's Gemini** and **Anthropic's Claude** AI models within Databricks notebooks and jobs. This library simplifies AI-assisted development workflows directly in your Databricks environment.

**Repository:** [https://github.com/gustcol/gemini-claude-databricks](https://github.com/gustcol/gemini-claude-databricks)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core Components](#core-components)
- [Usage Examples](#usage-examples)
- [Spark Integration](#spark-integration)
- [API Reference](#api-reference)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)
- [Pipeline Generation](#pipeline-generation)
- [Unity Catalog Integration](#unity-catalog-integration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This library provides a unified interface to interact with multiple AI providers (Google Gemini and Anthropic Claude) within Databricks, enabling developers to:

- Generate code automatically for Spark/PySpark tasks
- Analyze DataFrames with AI-powered insights
- Optimize SQL queries with intelligent suggestions
- Debug errors with contextual explanations
- Process data at scale using AI-powered Spark UDFs

```mermaid
graph LR
    A[👨‍💻 Developer] --> B[AI Assistant]
    B --> C{Choose Provider}
    C -->|Gemini| D[Google Gemini API]
    C -->|Claude| E[Anthropic Claude API]
    D --> F[🎯 AI Response]
    E --> F
    F --> G[Databricks Notebook]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#f3e5f5
    style G fill:#e1f5fe
```

---

## Architecture

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Databricks["☁️ Databricks Environment"]
        subgraph Notebook["📓 Notebook / Job"]
            User["User Code"]
            Assistant["AIAssistant"]
        end

        subgraph Security["🔐 Security Layer"]
            Secrets["Databricks Secrets"]
            ENV["Environment Variables"]
        end

        subgraph Data["📊 Data Layer"]
            Delta["Delta Tables"]
            Spark["Spark DataFrames"]
        end
    end

    subgraph External["🌐 External APIs"]
        Gemini["Google Gemini API"]
        Claude["Anthropic Claude API"]
    end

    User --> Assistant
    Assistant --> Secrets
    Assistant --> ENV
    Secrets --> Assistant
    ENV --> Assistant
    Assistant <--> Gemini
    Assistant <--> Claude
    Assistant --> Spark
    Spark --> Delta

    style Databricks fill:#e3f2fd
    style External fill:#fff8e1
    style Security fill:#ffebee
    style Data fill:#e8f5e9
```

### Component Architecture

```mermaid
classDiagram
    class AIAssistant {
        +config: AIConfig
        +gemini: GeminiClient
        +claude: ClaudeClient
        +ask(prompt, model)
        +ask_gemini(prompt)
        +ask_claude(prompt)
        +stream(prompt, model)
        +chat(message, conversation)
        +generate_code(task, language)
        +analyze_dataframe(df)
        +optimize_query(query)
        +explain_error(error)
    }

    class GeminiClient {
        +model: GenerativeModel
        +generation_config: Config
        +generate(prompt)
        +generate_stream(prompt)
        +chat(message, conversation)
        +count_tokens(text)
    }

    class ClaudeClient {
        +client: Anthropic
        +model_config: ModelConfig
        +generate(prompt)
        +generate_stream(prompt)
        +chat(message, conversation)
    }

    class ClaudeCodeAssistant {
        +generate_code(task)
        +review_code(code)
        +explain_code(code)
        +fix_code(code, error)
    }

    class AIConfig {
        +secret_scope: str
        +gemini_model: ModelConfig
        +claude_model: ModelConfig
        +get_gemini_key()
        +get_claude_key()
    }

    AIAssistant --> GeminiClient
    AIAssistant --> ClaudeClient
    AIAssistant --> AIConfig
    ClaudeCodeAssistant --|> ClaudeClient
```

### Request Flow

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant A as AIAssistant
    participant C as Config
    participant S as Databricks Secrets
    participant G as Gemini/Claude API

    U->>A: ask("Explain Delta Lake")
    A->>C: get_api_key()
    C->>S: secrets.get(scope, key)
    S-->>C: API Key
    C-->>A: API Key
    A->>G: POST /generate
    Note over G: Process request<br/>Generate response
    G-->>A: Response text
    A-->>U: "Delta Lake is..."
```

---

## Features

### Feature Overview

```mermaid
mindmap
  root((AI Assistant))
    Unified Interface
      Single API
      Model Switching
      Consistent Methods
    Databricks Native
      Secret Management
      Spark Integration
      Delta Lake Support
    AI Capabilities
      Code Generation
      Query Optimization
      Error Explanation
      DataFrame Analysis
    Developer Experience
      Streaming Responses
      Conversation Memory
      Cost Tracking
```

### Detailed Features

| Feature | Description | Gemini | Claude |
|---------|-------------|:------:|:------:|
| **Text Generation** | Generate responses to prompts | ✅ | ✅ |
| **Streaming** | Real-time response streaming | ✅ | ✅ |
| **Conversations** | Multi-turn chat with memory | ✅ | ✅ |
| **Code Generation** | Generate code for specific tasks | ✅ | ✅ |
| **Code Review** | Review and improve code | ✅ | ✅ |
| **Code Explanation** | Explain complex code | ✅ | ✅ |
| **Error Analysis** | Debug errors with context | ✅ | ✅ |
| **DataFrame Analysis** | AI-powered data insights | ✅ | ✅ |
| **Query Optimization** | SQL optimization suggestions | ✅ | ✅ |
| **Batch Processing** | Process data at scale with UDFs | ✅ | ✅ |
| **Cost Tracking** | Monitor token usage and costs | ✅ | ✅ |

---

## Installation

### Installation Flow

```mermaid
flowchart LR
    A[Start] --> B{Installation Method}
    B -->|PyPI| C["pip install gemini-claude-databricks"]
    B -->|Databricks| D["%pip install google-generativeai anthropic"]
    B -->|Source| E["git clone & pip install -e ."]
    C --> F[Configure Secrets]
    D --> F
    E --> F
    F --> G[Ready to Use]

    style A fill:#c8e6c9
    style G fill:#c8e6c9
```

### Option 1: Install from PyPI (when published)
```python
%pip install gemini-claude-databricks
```

### Option 2: Install directly in Databricks notebook
```python
%pip install google-generativeai anthropic

# Copy the ai_assistant module to your workspace or DBFS
```

### Option 3: Clone and install locally
```bash
git clone https://github.com/gustcol/gemini-claude-databricks.git
cd gemini-claude-databricks
pip install -e .
```

### Dependencies

```mermaid
graph TD
    A[gemini-claude-databricks] --> B[google-generativeai >= 0.3.0]
    A --> C[anthropic >= 0.18.0]
    A -.->|optional| D[pyspark >= 3.4.0]
    A -.->|dev| E[pytest >= 7.0.0]
    A -.->|dev| F[black >= 23.0.0]

    style A fill:#fff3e0
    style B fill:#e8f5e9
    style C fill:#fce4ec
```

---

## Quick Start

### Setup Process

```mermaid
flowchart TD
    A[🚀 Start] --> B[1️⃣ Install Dependencies]
    B --> C[2️⃣ Configure API Keys]
    C --> D{Choose Method}
    D -->|Recommended| E[Databricks Secrets]
    D -->|Development| F[Environment Variables]
    E --> G[3️⃣ Initialize Assistant]
    F --> G
    G --> H[4️⃣ Start Using AI]
    H --> I[✅ Ready!]

    style A fill:#c8e6c9
    style I fill:#c8e6c9
```

### Step 1: Configure Secrets in Databricks

```bash
# Using Databricks CLI
databricks secrets create-scope --scope ai-keys

# Add your API keys
databricks secrets put --scope ai-keys --key gemini-api-key
databricks secrets put --scope ai-keys --key claude-api-key
```

### Step 2: Initialize and Use

```python
from ai_assistant import AIAssistant

# Initialize with Databricks secret scope
assistant = AIAssistant(
    secret_scope="ai-keys",
    gemini_secret_key="gemini-api-key",
    claude_secret_key="claude-api-key"
)

# Use Gemini
response = assistant.ask_gemini("Explain PySpark DataFrame operations")
print(response)

# Use Claude
response = assistant.ask_claude("Write a function to optimize Spark queries")
print(response)

# Use default model (Claude)
response = assistant.ask("What is Delta Lake?")
print(response)
```

---

## Configuration

### Configuration Hierarchy

```mermaid
flowchart TD
    A[API Key Resolution] --> B{Direct API Key?}
    B -->|Yes| C[Use Direct Key]
    B -->|No| D{Databricks Secrets?}
    D -->|Yes| E[Use Secret Scope]
    D -->|No| F{Environment Variable?}
    F -->|Yes| G[Use Env Variable]
    F -->|No| H[❌ APIKeyNotFoundError]

    C --> I[✅ Key Resolved]
    E --> I
    G --> I

    style C fill:#c8e6c9
    style E fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#ffcdd2
    style I fill:#c8e6c9
```

### Configuration Options

```python
from ai_assistant import AIAssistant

assistant = AIAssistant(
    # Secret Management
    secret_scope="ai-keys",           # Databricks secret scope name
    gemini_secret_key="gemini-key",   # Key name in secret scope
    claude_secret_key="claude-key",   # Key name in secret scope

    # Model Selection
    gemini_model="gemini-1.5-pro",    # Gemini model to use
    claude_model="claude-sonnet-4-20250514",  # Claude model to use

    # Generation Parameters
    max_tokens=4096,                  # Maximum output tokens
    temperature=0.7,                  # Creativity (0.0 - 1.0)
)
```

### Environment Variables (Alternative)

```python
import os
os.environ["GEMINI_API_KEY"] = "your-gemini-key"
os.environ["ANTHROPIC_API_KEY"] = "your-claude-key"

assistant = AIAssistant()  # Auto-detects from environment
```

### Supported Models

```mermaid
graph LR
    subgraph Gemini["Google Gemini Models"]
        G1["gemini-1.5-pro<br/>(Default)"]
        G2["gemini-1.5-flash<br/>(Fast)"]
        G3["gemini-1.5-flash-8b<br/>(Lightweight)"]
        G4["gemini-2.0-flash-exp<br/>(Experimental)"]
    end

    subgraph Claude["Anthropic Claude Models"]
        C1["claude-sonnet-4-20250514<br/>(Default)"]
        C2["claude-opus-4-20250514<br/>(Most Capable)"]
        C3["claude-3-5-haiku<br/>(Fast)"]
    end

    style G1 fill:#e8f5e9
    style C1 fill:#fce4ec
```

---

## Core Components

### Module Structure

```mermaid
graph TB
    subgraph Package["ai_assistant"]
        Init["__init__.py<br/>Package exports"]
        Core["core.py<br/>AIAssistant class"]
        Gemini["gemini_client.py<br/>GeminiClient class"]
        Claude["claude_client.py<br/>ClaudeClient class"]
        Config["config.py<br/>Configuration classes"]
        Exceptions["exceptions.py<br/>Custom exceptions"]
        Spark["spark_utils.py<br/>Spark utilities"]
    end

    Init --> Core
    Core --> Gemini
    Core --> Claude
    Core --> Config
    Gemini --> Exceptions
    Claude --> Exceptions
    Core --> Spark

    style Package fill:#e3f2fd
```

### Class Relationships

```mermaid
erDiagram
    AIAssistant ||--o| GeminiClient : "lazy loads"
    AIAssistant ||--o| ClaudeClient : "lazy loads"
    AIAssistant ||--|| AIConfig : "uses"
    GeminiClient ||--|| ModelConfig : "configured by"
    ClaudeClient ||--|| ModelConfig : "configured by"
    ClaudeCodeAssistant ||--|| ClaudeClient : "extends"

    AIAssistant {
        AIConfig config
        GeminiClient gemini
        ClaudeClient claude
        string default_model
    }

    AIConfig {
        string secret_scope
        ModelConfig gemini_model
        ModelConfig claude_model
    }

    ModelConfig {
        string name
        int max_tokens
        float temperature
    }
```

---

## Usage Examples

### Basic Conversation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as Assistant
    participant AI as AI Model

    U->>A: ask("What is Spark?")
    A->>AI: Generate response
    AI-->>A: Response text
    A-->>U: "Apache Spark is..."

    U->>A: chat("Tell me more", "spark_topic")
    Note over A: Stores in conversation memory
    A->>AI: Generate with context
    AI-->>A: Contextual response
    A-->>U: "Building on that..."

    U->>A: chat("How about performance?", "spark_topic")
    Note over A: Retrieves conversation history
    A->>AI: Generate with full context
    AI-->>A: Informed response
    A-->>U: "For performance..."
```

### Code Generation Example

```python
# Generate PySpark code
code = assistant.generate_code(
    task="Create a function that reads a Delta table and performs aggregations",
    language="python",
    context="Using Databricks with Unity Catalog",
    include_tests=True
)
print(code)
```

### DataFrame Analysis

```python
# Load a DataFrame
df = spark.read.table("sales.transactions")

# Get AI-powered analysis
analysis = assistant.analyze_dataframe(
    df,
    questions=[
        "What are the data quality issues?",
        "Suggest optimization strategies",
        "Recommend partitioning scheme"
    ]
)
print(analysis)
```

### Query Optimization

```python
slow_query = """
SELECT customer_id, SUM(amount) as total
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE order_date >= '2024-01-01'
GROUP BY customer_id
"""

optimized = assistant.optimize_query(
    query=slow_query,
    context="orders: 500M rows, customers: 10M rows"
)
print(optimized)
```

### Error Debugging

```python
error_message = """
AnalysisException: Table or view not found: sales.transactions
"""

explanation = assistant.explain_error(
    error_message=error_message,
    code="df = spark.read.table('sales.transactions')"
)
print(explanation)
```

---

## Spark Integration

### Batch Processing Architecture

```mermaid
flowchart LR
    subgraph Input["📥 Input"]
        DF1["Spark DataFrame"]
    end

    subgraph Processing["⚙️ AI Processing"]
        UDF["AI UDF"]
        Batch["Batch Controller"]
        Rate["Rate Limiter"]
    end

    subgraph API["🌐 AI API"]
        Gemini["Gemini"]
        Claude["Claude"]
    end

    subgraph Output["📤 Output"]
        DF2["Enhanced DataFrame"]
    end

    DF1 --> UDF
    UDF --> Batch
    Batch --> Rate
    Rate --> Gemini
    Rate --> Claude
    Gemini --> DF2
    Claude --> DF2

    style Input fill:#e3f2fd
    style Processing fill:#fff3e0
    style API fill:#f3e5f5
    style Output fill:#e8f5e9
```

### Process DataFrame with AI

```python
from ai_assistant.spark_utils import process_with_ai

# Process text column with AI
results_df = process_with_ai(
    spark_df=input_df,
    prompt_column="product_description",
    model="gemini",
    output_column="ai_analysis",
    batch_size=10,
    rate_limit_delay=0.5,
    system_instruction="Analyze sentiment and extract key features"
)

results_df.show()
```

### Create Custom AI UDF

```python
from ai_assistant.spark_utils import create_ai_udf
from pyspark.sql.functions import col

# Create a sentiment analysis UDF
sentiment_udf = create_ai_udf(
    model="claude",
    api_key=api_key,
    system_instruction="Classify sentiment as: positive, negative, or neutral",
    max_tokens=50
)

# Apply to DataFrame
df_with_sentiment = df.withColumn(
    "sentiment",
    sentiment_udf(col("review_text"))
)
```

### Cost Estimation

```python
from ai_assistant.spark_utils import estimate_processing_cost

# Estimate before processing
estimate = estimate_processing_cost(
    row_count=100000,
    avg_input_tokens=150,
    avg_output_tokens=100,
    model="gemini-1.5-flash"
)

print(f"Estimated cost: ${estimate['total_cost']:.2f}")
print(f"Total tokens: {estimate['total_input_tokens'] + estimate['total_output_tokens']:,}")
```

---

## API Reference

### Core Methods

```mermaid
graph TD
    subgraph AIAssistant["AIAssistant Methods"]
        A1["ask(prompt, model)"]
        A2["ask_gemini(prompt)"]
        A3["ask_claude(prompt)"]
        A4["stream(prompt, model)"]
        A5["chat(message, conversation)"]
        A6["generate_code(task)"]
        A7["analyze_dataframe(df)"]
        A8["optimize_query(sql)"]
        A9["explain_error(error)"]
    end

    subgraph Returns["Return Types"]
        R1["str"]
        R2["Generator[str]"]
        R3["Dict"]
    end

    A1 --> R1
    A2 --> R1
    A3 --> R1
    A4 --> R2
    A5 --> R1
    A6 --> R1
    A7 --> R1
    A8 --> R1
    A9 --> R1
```

### Method Signatures

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `ask()` | `prompt: str, model: str = None` | `str` | Ask any model |
| `ask_gemini()` | `prompt: str, system_instruction: str = None` | `str` | Ask Gemini |
| `ask_claude()` | `prompt: str, system_instruction: str = None` | `str` | Ask Claude |
| `stream()` | `prompt: str, model: str = None` | `Generator` | Stream response |
| `chat()` | `message: str, conversation_name: str` | `str` | Multi-turn chat |
| `generate_code()` | `task: str, language: str = "python"` | `str` | Generate code |
| `analyze_dataframe()` | `df: DataFrame, questions: List[str]` | `str` | Analyze data |
| `optimize_query()` | `query: str, context: str = None` | `str` | Optimize SQL |
| `explain_error()` | `error_message: str, code: str = None` | `str` | Debug errors |

See [API Documentation](docs/api.md) for complete reference.

---

## Security Best Practices

### Security Flow

```mermaid
flowchart TD
    subgraph DO["✅ Best Practices"]
        D1["Use Databricks Secrets"]
        D2["Limit scope permissions"]
        D3["Audit API usage"]
        D4["Sanitize sensitive data"]
        D5["Use environment isolation"]
    end

    subgraph DONT["❌ Avoid"]
        N1["Hardcoding API keys"]
        N2["Committing secrets to git"]
        N3["Sending PII to AI models"]
        N4["Sharing secret scopes broadly"]
    end

    style DO fill:#c8e6c9
    style DONT fill:#ffcdd2
```

### Security Checklist

- [ ] **Never hardcode API keys** - Always use Databricks Secrets or environment variables
- [ ] **Limit secret scope access** - Grant minimal permissions to secret scopes
- [ ] **Audit usage** - Monitor API calls and costs regularly
- [ ] **Data privacy** - Be mindful of sensitive data sent to AI models
- [ ] **Use service principals** - For production workloads
- [ ] **Rotate keys regularly** - Update API keys periodically

---

## Troubleshooting

### Error Resolution Flow

```mermaid
flowchart TD
    A[Error Occurred] --> B{Error Type?}

    B -->|APIKeyNotFoundError| C[Check Secret Configuration]
    C --> C1[Verify scope exists]
    C1 --> C2[Verify key name]
    C2 --> C3[Check permissions]

    B -->|RateLimitError| D[Handle Rate Limiting]
    D --> D1[Implement backoff]
    D1 --> D2[Reduce request frequency]
    D2 --> D3[Use batch processing]

    B -->|TokenLimitError| E[Manage Token Limits]
    E --> E1[Reduce input size]
    E1 --> E2[Use summarization]
    E2 --> E3[Choose larger model]

    B -->|ModelNotAvailableError| F[Check Model Config]
    F --> F1[Verify model name]
    F1 --> F2[Check API access]

    style A fill:#ffcdd2
    style C3 fill:#c8e6c9
    style D3 fill:#c8e6c9
    style E3 fill:#c8e6c9
    style F2 fill:#c8e6c9
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `APIKeyNotFoundError` | API key not configured | Configure secrets or environment variables |
| `RateLimitError` | Too many requests | Implement exponential backoff |
| `TokenLimitError` | Input/output too large | Reduce text size or use larger model |
| `ModelNotAvailableError` | Invalid model name | Check supported models list |
| `DatabricksContextError` | Not in Databricks env | Use environment variables instead |

---

## Project Structure

```
gemini-claude-databricks/
├── 📁 ai_assistant/
│   ├── 📄 __init__.py          # Package exports
│   ├── 📄 core.py              # Main AIAssistant class
│   ├── 📄 gemini_client.py     # Google Gemini client
│   ├── 📄 claude_client.py     # Anthropic Claude client
│   ├── 📄 config.py            # Configuration management
│   ├── 📄 exceptions.py        # Custom exceptions
│   └── 📄 spark_utils.py       # Spark integration
├── 📁 examples/
│   ├── 📄 01_basic_usage.py    # Basic examples
│   └── 📄 02_code_generation.py # Code generation
├── 📁 notebooks/
│   └── 📄 AI_Assistant_Quickstart.py  # Databricks notebook
├── 📁 docs/
│   └── 📄 api.md               # API documentation
├── 📁 tests/
│   └── 📄 test_ai_assistant.py # Unit tests
├── 📄 README.md                # This file
├── 📄 requirements.txt         # Dependencies
├── 📄 setup.py                 # Package setup
└── 📄 .gitignore              # Git ignore rules
```

---

## Token Pricing Reference

```mermaid
xychart-beta
    title "Cost per 1M Tokens (USD)"
    x-axis ["Gemini Flash", "Gemini Pro", "Claude Haiku", "Claude Sonnet", "Claude Opus"]
    y-axis "Cost ($)" 0 --> 80
    bar [0.075, 1.25, 0.80, 3.00, 15.00]
```

| Model | Input (per 1K) | Output (per 1K) | Best For |
|-------|----------------|-----------------|----------|
| Gemini 1.5 Flash | $0.000075 | $0.0003 | High-volume, simple tasks |
| Gemini 1.5 Pro | $0.00125 | $0.005 | Complex reasoning |
| Claude Haiku | $0.0008 | $0.004 | Fast, cost-effective |
| Claude Sonnet | $0.003 | $0.015 | Balanced performance |
| Claude Opus | $0.015 | $0.075 | Maximum capability |

---

## Pipeline Generation

The library includes powerful AI-driven pipeline generation capabilities for creating DLT pipelines, ETL workflows, and medallion architectures.

### Pipeline Architecture

```mermaid
flowchart TB
    subgraph Input["📝 User Input"]
        Desc["Natural Language Description"]
        Config["Configuration Parameters"]
    end

    subgraph Generator["🤖 AI Pipeline Generator"]
        DLT["DLT Generator"]
        ETL["ETL Generator"]
        Streaming["Streaming Generator"]
        Medallion["Medallion Generator"]
    end

    subgraph Output["📤 Generated Code"]
        Bronze["Bronze Layer"]
        Silver["Silver Layer"]
        Gold["Gold Layer"]
        Jobs["Workflow Jobs"]
    end

    Desc --> Generator
    Config --> Generator
    DLT --> Bronze
    DLT --> Silver
    DLT --> Gold
    ETL --> Jobs
    Streaming --> Bronze
    Medallion --> Bronze
    Medallion --> Silver
    Medallion --> Gold

    style Input fill:#e3f2fd
    style Generator fill:#fff3e0
    style Output fill:#e8f5e9
```

### Generate Delta Live Tables Pipeline

```python
from ai_assistant import AIAssistant
from ai_assistant.pipelines import PipelineGenerator

assistant = AIAssistant(secret_scope="ai-keys")
generator = PipelineGenerator(assistant)

# Generate a complete DLT pipeline
dlt_code = generator.generate_dlt_pipeline(
    description="Process customer orders with data quality validation",
    source_table="bronze.raw_orders",
    target_catalog="analytics",
    target_schema="gold",
    include_expectations=True,
    include_streaming=True
)

print(dlt_code)
```

### Generate Medallion Architecture

```python
# Generate complete medallion architecture
medallion_code = generator.generate_medallion_architecture(
    description="E-commerce sales data pipeline with customer analytics",
    source_path="/mnt/landing/sales/",
    source_format="json",
    catalog="ecommerce",
    include_dlt=True
)

print(medallion_code)
```

### Generate ETL Pipeline

```python
# Generate traditional ETL pipeline
etl_code = generator.generate_etl_pipeline(
    description="Daily customer data synchronization",
    source_config={
        "path": "/data/customers",
        "format": "parquet"
    },
    target_config={
        "catalog": "prod",
        "schema": "dimensions",
        "table": "dim_customer"
    },
    transformations=["deduplicate", "validate_email", "standardize_phone"]
)

print(etl_code)
```

### Generate Streaming Pipeline

```python
# Generate Structured Streaming pipeline
streaming_code = generator.generate_streaming_pipeline(
    description="Real-time clickstream processing",
    source_config={
        "format": "kafka",
        "topic": "user_clicks",
        "bootstrap_servers": "kafka:9092"
    },
    target_config={
        "catalog": "realtime",
        "schema": "events",
        "table": "click_events"
    },
    watermark_column="event_time",
    watermark_delay="10 minutes"
)

print(streaming_code)
```

### Generate Databricks Workflow

```python
# Generate workflow/job definition
workflow = generator.generate_workflow(
    description="Daily ETL pipeline with data validation and alerting",
    schedule="0 0 6 * * ?",  # Daily at 6 AM
    tasks=[
        {"name": "extract", "notebook": "/ETL/extract"},
        {"name": "transform", "notebook": "/ETL/transform", "depends_on": ["extract"]},
        {"name": "validate", "notebook": "/ETL/validate", "depends_on": ["transform"]},
        {"name": "load", "notebook": "/ETL/load", "depends_on": ["validate"]}
    ]
)

print(workflow)
```

### Analyze Existing Pipeline

```python
# Analyze and get recommendations
analysis = generator.analyze_pipeline(existing_code)
print(analysis)

# Convert between formats
dlt_code = generator.convert_pipeline(
    spark_code,
    source_format="spark",
    target_format="dlt"
)
```

---

## Unity Catalog Integration

Full support for Unity Catalog operations including schema generation, governance, and data lineage.

### Unity Catalog Architecture

```mermaid
flowchart TB
    subgraph UC["🏛️ Unity Catalog"]
        subgraph Catalog["📚 Catalog"]
            Schema1["Schema: raw"]
            Schema2["Schema: curated"]
            Schema3["Schema: reporting"]
        end

        subgraph Governance["🔐 Governance"]
            RLS["Row-Level Security"]
            ColMask["Column Masking"]
            Tags["Data Tags"]
        end

        subgraph Lineage["🔗 Lineage"]
            TableLin["Table Lineage"]
            ColLin["Column Lineage"]
        end
    end

    subgraph Helper["🤖 UC Helper"]
        DDL["Generate DDL"]
        Policy["Generate Policies"]
        Migrate["Migration Scripts"]
        Audit["Audit Queries"]
    end

    Helper --> UC

    style UC fill:#e3f2fd
    style Governance fill:#ffebee
    style Helper fill:#fff3e0
```

### Generate Table DDL

```python
from ai_assistant import AIAssistant
from ai_assistant.unity_catalog import UnityCatalogHelper

assistant = AIAssistant(secret_scope="ai-keys")
uc_helper = UnityCatalogHelper(assistant)

# Generate table DDL from description
ddl = uc_helper.generate_table_ddl(
    description="Customer master data table with PII fields including email, phone, and address",
    catalog="enterprise",
    schema="master_data",
    table_name="dim_customer",
    include_governance=True
)

print(ddl)
```

### Generate Complete Schema

```python
# Generate entire schema with multiple tables
schema_ddl = uc_helper.generate_schema_ddl(
    description="E-commerce data domain for order processing",
    catalog="ecommerce",
    schema_name="orders",
    tables=["customers", "orders", "order_items", "products", "payments"]
)

print(schema_ddl)
```

### Row-Level Security

```python
# Generate row-level security policy
rls_policy = uc_helper.generate_row_level_security(
    table="sales.transactions.orders",
    description="Sales reps can only see orders from their assigned region. Managers see all regions."
)

print(rls_policy)
```

### Column Masking

```python
# Generate column masking for PII
mask = uc_helper.generate_column_mask(
    table="hr.employees.personal_info",
    column="ssn",
    description="Mask SSN showing only last 4 digits. HR team sees full SSN."
)

print(mask)
```

### Access Policies

```python
from ai_assistant.unity_catalog import SecurableType

# Generate access control statements
grants = uc_helper.generate_access_policy(
    description="Data scientists need SELECT access, Data engineers need ALL PRIVILEGES",
    securable="analytics.sales.transactions",
    securable_type=SecurableType.TABLE
)

print(grants)
```

### Data Lineage Exploration

```python
# Generate lineage queries
lineage_queries = uc_helper.generate_data_lineage_query(
    table="gold.sales.daily_metrics",
    direction="upstream"  # or "downstream" or "both"
)

print(lineage_queries)
```

### Audit Log Analysis

```python
# Generate audit queries
audit_queries = uc_helper.generate_audit_queries(
    catalog="production",
    event_type="TABLE_ACCESS",
    time_range="30 days"
)

print(audit_queries)
```

### Table Migration to Unity Catalog

```python
# Generate migration script from Hive metastore
migration_script = uc_helper.migrate_table_to_uc(
    source_table="hive_metastore.legacy.customers",
    target_catalog="enterprise",
    target_schema="master_data",
    include_data=True
)

print(migration_script)
```

### Table Analysis

```python
# Analyze table and get recommendations
analysis = uc_helper.analyze_table("prod.sales.transactions")
print(analysis)
```

### Best Practices

```python
from ai_assistant.unity_catalog import get_uc_best_practices

# Get Unity Catalog best practices guide
best_practices = get_uc_best_practices()
print(best_practices)
```

---

## Contributing

Contributions are welcome! Please follow these steps:

```mermaid
gitGraph
    commit id: "Fork repo"
    branch feature
    commit id: "Create branch"
    commit id: "Make changes"
    commit id: "Add tests"
    commit id: "Update docs"
    checkout main
    merge feature id: "Create PR"
    commit id: "Review & merge"
```

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Google Generative AI](https://ai.google.dev/) for Gemini API
- [Anthropic](https://www.anthropic.com/) for Claude API
- [Databricks](https://databricks.com/) for the amazing platform

---

## Author

**Guxxxta / Gustcol**
- Email: gustcol@gmail.com
- GitHub: [@gustcol](https://github.com/gustcol)

---

<p align="center">
  Made with ❤️ for the Databricks community
</p>
