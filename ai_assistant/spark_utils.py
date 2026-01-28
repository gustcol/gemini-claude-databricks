"""
Spark utilities for AI-powered data processing.

This module provides utilities for integrating AI models with Spark
for scalable data processing, including batch AI calls and UDFs.
"""

import time
from typing import Optional, List, Callable, Any
from functools import wraps


def process_with_ai(
    spark_df,
    prompt_column: str,
    model: str = "gemini",
    output_column: str = "ai_response",
    api_key: Optional[str] = None,
    secret_scope: Optional[str] = None,
    secret_key: Optional[str] = None,
    batch_size: int = 10,
    max_retries: int = 3,
    rate_limit_delay: float = 0.5,
    system_instruction: Optional[str] = None
):
    """
    Process DataFrame rows using AI models.

    This function applies AI processing to each row in a Spark DataFrame,
    handling batching and rate limiting automatically.

    Args:
        spark_df: Input Spark DataFrame
        prompt_column: Column containing text to process
        model: "gemini" or "claude"
        output_column: Name for the output column
        api_key: Direct API key (use secrets in production!)
        secret_scope: Databricks secret scope name
        secret_key: Secret key name
        batch_size: Number of rows to process per batch
        max_retries: Maximum retry attempts for failed requests
        rate_limit_delay: Delay between requests in seconds
        system_instruction: System instruction for the AI

    Returns:
        DataFrame with added AI response column

    Example:
        >>> result_df = process_with_ai(
        ...     input_df,
        ...     prompt_column="product_description",
        ...     model="claude",
        ...     output_column="sentiment",
        ...     system_instruction="Classify sentiment as positive, negative, or neutral"
        ... )

    Note:
        For large datasets, consider using Spark's mapPartitions with
        the AI clients directly for better control over parallelism.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType

    spark = SparkSession.builder.getOrCreate()

    # Get API key
    if not api_key:
        if secret_scope and secret_key:
            try:
                dbutils = spark._jvm.com.databricks.backend.daemon.driver.DBUtils(spark._sc)
                api_key = dbutils.secrets.get(scope=secret_scope, key=secret_key)
            except Exception:
                # Try alternative method
                import IPython
                dbutils = IPython.get_ipython().user_ns.get("dbutils")
                if dbutils:
                    api_key = dbutils.secrets.get(scope=secret_scope, key=secret_key)

    if not api_key:
        import os
        if model.lower() == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            f"API key for {model} not found. "
            f"Provide api_key, configure secrets, or set environment variable."
        )

    # Create AI processing function
    def process_text(text: str) -> Optional[str]:
        """Process a single text item with AI."""
        if not text:
            return None

        for attempt in range(max_retries):
            try:
                if model.lower() == "gemini":
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model_instance = genai.GenerativeModel(
                        model_name="gemini-1.5-flash",
                        system_instruction=system_instruction
                    )
                    response = model_instance.generate_content(text)
                    return response.text
                else:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=1024,
                        system=system_instruction or "You are a helpful assistant.",
                        messages=[{"role": "user", "content": text}]
                    )
                    return message.content[0].text

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(rate_limit_delay * (attempt + 1))
                else:
                    return f"Error: {str(e)}"

        return None

    # Register UDF
    process_udf = udf(process_text, StringType())

    # Apply UDF with rate limiting via repartitioning
    # Repartition to control parallelism and avoid rate limits
    num_partitions = max(1, spark_df.count() // batch_size)
    result_df = (
        spark_df
        .repartition(num_partitions)
        .withColumn(output_column, process_udf(col(prompt_column)))
    )

    return result_df


def create_ai_udf(
    model: str = "gemini",
    api_key: Optional[str] = None,
    system_instruction: Optional[str] = None,
    max_tokens: int = 1024
) -> Callable:
    """
    Create a Spark UDF for AI text processing.

    This allows more flexibility in how you apply AI processing
    within Spark transformations.

    Args:
        model: "gemini" or "claude"
        api_key: API key for the model
        system_instruction: System instruction
        max_tokens: Maximum output tokens

    Returns:
        A Spark UDF function

    Example:
        >>> ai_summarize = create_ai_udf(
        ...     model="claude",
        ...     system_instruction="Summarize the text in one sentence"
        ... )
        >>> df.withColumn("summary", ai_summarize(col("long_text")))
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    def ai_process(text: str) -> Optional[str]:
        if not text:
            return None

        try:
            if model.lower() == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_instruction
                )
                response = model_instance.generate_content(text)
                return response.text
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=max_tokens,
                    system=system_instruction or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": text}]
                )
                return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

    return udf(ai_process, StringType())


def batch_process_partition(
    partition_iterator,
    model: str,
    api_key: str,
    prompt_extractor: Callable,
    system_instruction: Optional[str] = None,
    batch_size: int = 10
):
    """
    Process a Spark partition with batched AI calls.

    Use this with mapPartitions for efficient batch processing.

    Args:
        partition_iterator: Iterator over partition rows
        model: "gemini" or "claude"
        api_key: API key
        prompt_extractor: Function to extract prompt from row
        system_instruction: System instruction
        batch_size: Batch size for processing

    Yields:
        Processed rows with AI responses

    Example:
        >>> def extract_prompt(row):
        ...     return f"Analyze: {row.text}"
        >>>
        >>> processed_rdd = df.rdd.mapPartitions(
        ...     lambda partition: batch_process_partition(
        ...         partition,
        ...         model="claude",
        ...         api_key=api_key,
        ...         prompt_extractor=extract_prompt
        ...     )
        ... )
    """
    # Collect rows into batches
    batch = []
    for row in partition_iterator:
        batch.append(row)

        if len(batch) >= batch_size:
            # Process batch
            for processed_row in _process_batch(
                batch, model, api_key, prompt_extractor, system_instruction
            ):
                yield processed_row
            batch = []

    # Process remaining rows
    if batch:
        for processed_row in _process_batch(
            batch, model, api_key, prompt_extractor, system_instruction
        ):
            yield processed_row


def _process_batch(
    batch: List[Any],
    model: str,
    api_key: str,
    prompt_extractor: Callable[[Any], str],
    system_instruction: Optional[str]
) -> Any:
    """Process a batch of rows."""
    for row in batch:
        prompt = prompt_extractor(row)

        try:
            if model.lower() == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_instruction
                )
                response = model_instance.generate_content(prompt)
                ai_response = response.text
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1024,
                    system=system_instruction or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_response = message.content[0].text

            # Return row with response appended
            yield (*row, ai_response)

        except Exception as e:
            yield (*row, f"Error: {str(e)}")

        # Small delay to avoid rate limits
        time.sleep(0.1)


class AIStreamProcessor:
    """
    Streaming processor for real-time AI inference.

    This class provides utilities for processing streaming data
    with AI models in Databricks Structured Streaming.

    Example:
        >>> processor = AIStreamProcessor(api_key="your-key")
        >>> processed_stream = processor.process_stream(
        ...     input_stream,
        ...     prompt_column="message",
        ...     output_column="response"
        ... )
    """

    def __init__(
        self,
        model: str = "gemini",
        api_key: Optional[str] = None,
        system_instruction: Optional[str] = None
    ):
        """
        Initialize the stream processor.

        Args:
            model: "gemini" or "claude"
            api_key: API key
            system_instruction: System instruction
        """
        self.model = model
        self.api_key = api_key
        self.system_instruction = system_instruction

    def process_stream(
        self,
        streaming_df,
        prompt_column: str,
        output_column: str = "ai_response"
    ):
        """
        Add AI processing to a streaming DataFrame.

        Args:
            streaming_df: Input streaming DataFrame
            prompt_column: Column with text to process
            output_column: Output column name

        Returns:
            Streaming DataFrame with AI responses
        """
        ai_udf = create_ai_udf(
            model=self.model,
            api_key=self.api_key,
            system_instruction=self.system_instruction
        )

        from pyspark.sql.functions import col
        return streaming_df.withColumn(output_column, ai_udf(col(prompt_column)))


def estimate_processing_cost(
    row_count: int,
    avg_input_tokens: int = 100,
    avg_output_tokens: int = 200,
    model: str = "gemini-1.5-flash"
) -> dict:
    """
    Estimate the cost of processing a DataFrame with AI.

    Args:
        row_count: Number of rows to process
        avg_input_tokens: Average input tokens per row
        avg_output_tokens: Average output tokens per row
        model: Model to use

    Returns:
        Dictionary with cost estimates

    Example:
        >>> estimate = estimate_processing_cost(
        ...     row_count=10000,
        ...     avg_input_tokens=150,
        ...     model="claude-sonnet-4-20250514"
        ... )
        >>> print(f"Estimated cost: ${estimate['total_cost']:.2f}")
    """
    from .config import TOKEN_PRICING

    pricing = TOKEN_PRICING.get(model, {"input": 0.001, "output": 0.002})

    total_input_tokens = row_count * avg_input_tokens
    total_output_tokens = row_count * avg_output_tokens

    input_cost = (total_input_tokens / 1000) * pricing["input"]
    output_cost = (total_output_tokens / 1000) * pricing["output"]

    return {
        "row_count": row_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model
    }
