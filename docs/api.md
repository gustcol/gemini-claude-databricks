# API Reference

This document provides a comprehensive reference for the AI Assistant for Databricks library.

## Table of Contents

- [AIAssistant](#aiassistant)
- [GeminiClient](#geminiclient)
- [ClaudeClient](#claudeclient)
- [ClaudeCodeAssistant](#claudecodeassistant)
- [Spark Utilities](#spark-utilities)
- [Configuration](#configuration)
- [Exceptions](#exceptions)

---

## AIAssistant

The main unified interface for interacting with both Gemini and Claude models.

### Constructor

```python
AIAssistant(
    secret_scope: str = None,
    gemini_secret_key: str = "gemini-api-key",
    claude_secret_key: str = "claude-api-key",
    gemini_api_key: str = None,
    claude_api_key: str = None,
    gemini_model: str = "gemini-1.5-pro",
    claude_model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    dbutils = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `secret_scope` | str | None | Databricks secret scope name |
| `gemini_secret_key` | str | "gemini-api-key" | Key name for Gemini API key in secret scope |
| `claude_secret_key` | str | "claude-api-key" | Key name for Claude API key in secret scope |
| `gemini_api_key` | str | None | Direct Gemini API key (use secrets in production!) |
| `claude_api_key` | str | None | Direct Claude API key (use secrets in production!) |
| `gemini_model` | str | "gemini-1.5-pro" | Default Gemini model |
| `claude_model` | str | "claude-sonnet-4-20250514" | Default Claude model |
| `max_tokens` | int | 4096 | Maximum output tokens |
| `temperature` | float | 0.7 | Response temperature (0-1) |
| `dbutils` | object | None | Databricks utilities (auto-detected if not provided) |

### Methods

#### ask()

```python
ask(
    prompt: str,
    model: str = None,
    system_instruction: str = None,
    **kwargs
) -> str
```

Ask a question to the AI model.

**Parameters:**
- `prompt`: Your question or request
- `model`: "gemini" or "claude" (uses default if not specified)
- `system_instruction`: Optional system instruction
- `**kwargs`: Additional arguments passed to the model

**Returns:** AI response as a string

**Example:**
```python
response = assistant.ask("What is Delta Lake?")
response = assistant.ask("Explain Spark", model="gemini")
```

---

#### ask_gemini()

```python
ask_gemini(
    prompt: str,
    system_instruction: str = None,
    **kwargs
) -> str
```

Ask Gemini a question directly.

**Parameters:**
- `prompt`: Your question or request
- `system_instruction`: Optional system instruction
- `**kwargs`: Additional arguments (temperature, max_tokens)

**Returns:** Gemini's response

---

#### ask_claude()

```python
ask_claude(
    prompt: str,
    system_instruction: str = None,
    **kwargs
) -> str
```

Ask Claude a question directly.

**Parameters:**
- `prompt`: Your question or request
- `system_instruction`: Optional system instruction
- `**kwargs`: Additional arguments (temperature, max_tokens)

**Returns:** Claude's response

---

#### stream()

```python
stream(
    prompt: str,
    model: str = None,
    system_instruction: str = None
) -> Generator[str, None, None]
```

Get a streaming response from the AI model.

**Parameters:**
- `prompt`: Your question or request
- `model`: "gemini" or "claude"
- `system_instruction`: Optional system instruction

**Yields:** Text chunks as they are generated

**Example:**
```python
for chunk in assistant.stream("Explain MapReduce"):
    print(chunk, end="", flush=True)
```

---

#### chat()

```python
chat(
    message: str,
    conversation_name: str = "default",
    model: str = None,
    system_instruction: str = None
) -> str
```

Send a message in a multi-turn conversation.

**Parameters:**
- `message`: Your message
- `conversation_name`: Identifier for the conversation
- `model`: "gemini" or "claude"
- `system_instruction`: System instruction (for new conversations)

**Returns:** AI response

**Example:**
```python
assistant.chat("I have a performance issue")
assistant.chat("The job uses 100 executors")  # Has context
```

---

#### generate_code()

```python
generate_code(
    task: str,
    language: str = "python",
    model: str = None,
    context: str = None,
    include_tests: bool = False
) -> str
```

Generate code for a specific task.

**Parameters:**
- `task`: Description of what the code should do
- `language`: Programming language (default: python)
- `model`: "gemini" or "claude"
- `context`: Additional context about requirements
- `include_tests`: Whether to include unit tests

**Returns:** Generated code

---

#### analyze_dataframe()

```python
analyze_dataframe(
    df,
    questions: List[str] = None,
    model: str = None
) -> str
```

Analyze a Spark DataFrame using AI.

**Parameters:**
- `df`: Spark DataFrame to analyze
- `questions`: Specific questions about the data
- `model`: "gemini" or "claude"

**Returns:** Analysis results

---

#### optimize_query()

```python
optimize_query(
    query: str,
    model: str = None,
    context: str = None
) -> str
```

Get optimization suggestions for a Spark SQL query.

**Parameters:**
- `query`: The SQL query to optimize
- `model`: "gemini" or "claude"
- `context`: Additional context (table sizes, current performance)

**Returns:** Optimization suggestions and improved query

---

#### explain_error()

```python
explain_error(
    error_message: str,
    code: str = None,
    model: str = None
) -> str
```

Explain an error and suggest fixes.

**Parameters:**
- `error_message`: The error message
- `code`: The code that caused the error (if available)
- `model`: "gemini" or "claude"

**Returns:** Error explanation and suggested fixes

---

## GeminiClient

Low-level client for Google Gemini models.

### Constructor

```python
GeminiClient(
    api_key: str,
    model_config: ModelConfig = None,
    safety_settings: List[Dict] = None
)
```

### Methods

- `generate(prompt, system_instruction, temperature, max_tokens) -> str`
- `generate_stream(prompt, system_instruction) -> Generator`
- `chat(message, conversation_name, system_instruction) -> str`
- `clear_conversation(conversation_name) -> None`
- `list_conversations() -> List[str]`
- `count_tokens(text) -> int`
- `get_usage_stats() -> Dict`
- `reset_usage_stats() -> None`

---

## ClaudeClient

Low-level client for Anthropic Claude models.

### Constructor

```python
ClaudeClient(
    api_key: str,
    model_config: ModelConfig = None
)
```

### Methods

- `generate(prompt, system_instruction, temperature, max_tokens) -> str`
- `generate_stream(prompt, system_instruction) -> Generator`
- `chat(message, conversation_name, system_instruction) -> str`
- `clear_conversation(conversation_name) -> None`
- `list_conversations() -> List[str]`
- `get_conversation_history(conversation_name) -> List[Dict]`
- `count_tokens(text) -> int`
- `get_usage_stats() -> Dict`
- `reset_usage_stats() -> None`

---

## ClaudeCodeAssistant

Specialized Claude client for code assistance tasks.

### Constructor

```python
ClaudeCodeAssistant(
    api_key: str,
    model_config: ModelConfig = None
)
```

### Methods

#### generate_code()

```python
generate_code(
    task: str,
    language: str = "python",
    context: str = None,
    include_tests: bool = False
) -> str
```

#### review_code()

```python
review_code(
    code: str,
    focus_areas: List[str] = None
) -> str
```

#### explain_code()

```python
explain_code(
    code: str,
    detail_level: str = "medium"
) -> str
```

#### fix_code()

```python
fix_code(
    code: str,
    error_message: str = None
) -> str
```

---

## Spark Utilities

### process_with_ai()

```python
process_with_ai(
    spark_df,
    prompt_column: str,
    model: str = "gemini",
    output_column: str = "ai_response",
    api_key: str = None,
    secret_scope: str = None,
    secret_key: str = None,
    batch_size: int = 10,
    max_retries: int = 3,
    rate_limit_delay: float = 0.5,
    system_instruction: str = None
)
```

Process DataFrame rows using AI models.

**Returns:** DataFrame with added AI response column

---

### create_ai_udf()

```python
create_ai_udf(
    model: str = "gemini",
    api_key: str = None,
    system_instruction: str = None,
    max_tokens: int = 1024
) -> Callable
```

Create a Spark UDF for AI text processing.

**Returns:** A Spark UDF function

---

### estimate_processing_cost()

```python
estimate_processing_cost(
    row_count: int,
    avg_input_tokens: int = 100,
    avg_output_tokens: int = 200,
    model: str = "gemini-1.5-flash"
) -> dict
```

Estimate the cost of processing a DataFrame with AI.

**Returns:** Dictionary with cost estimates

---

## Configuration

### AIConfig

```python
@dataclass
class AIConfig:
    secret_scope: str = None
    gemini_secret_key: str = "gemini-api-key"
    claude_secret_key: str = "claude-api-key"
    gemini_api_key: str = None
    claude_api_key: str = None
    gemini_model: ModelConfig = ...
    claude_model: ModelConfig = ...
    enable_cost_tracking: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0
    use_mlflow_tracking: bool = False
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = None  # Gemini only
```

### Available Models

**Gemini:**
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-1.5-flash-8b`
- `gemini-1.0-pro`

**Claude:**
- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `claude-3-5-haiku-20241022`

---

## Exceptions

### AIAssistantError

Base exception for all AI Assistant errors.

### APIKeyNotFoundError

Raised when an API key cannot be found.

```python
APIKeyNotFoundError(provider: str, message: str = None)
```

### ModelNotAvailableError

Raised when the requested model is not available.

```python
ModelNotAvailableError(model: str, provider: str, available_models: list = None)
```

### RateLimitError

Raised when API rate limits are exceeded.

```python
RateLimitError(provider: str, retry_after: int = None)
```

### TokenLimitError

Raised when token limits are exceeded.

```python
TokenLimitError(token_count: int, token_limit: int, token_type: str = "total")
```

### ConversationNotFoundError

Raised when accessing a non-existent conversation.

```python
ConversationNotFoundError(conversation_name: str)
```

### DatabricksContextError

Raised when Databricks-specific operations fail.

```python
DatabricksContextError(message: str)
```
