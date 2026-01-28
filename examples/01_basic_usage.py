"""
Basic Usage Example
===================

This example demonstrates the fundamental features of the AI Assistant
for Databricks, including simple queries, model selection, and
conversation management.

Prerequisites:
    - API keys configured (via secrets or environment variables)
    - Required packages installed: google-generativeai, anthropic
"""

# =============================================================================
# Setup
# =============================================================================

# Option 1: Using Databricks Secrets (Recommended for production)
# from ai_assistant import AIAssistant
# assistant = AIAssistant(secret_scope="ai-keys")

# Option 2: Using environment variables (Development)
import os
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-claude-api-key"

from ai_assistant import AIAssistant

# Initialize the assistant
assistant = AIAssistant()

# Check which models are available
print("Available models:")
print(assistant.get_available_models())
print()

# =============================================================================
# Simple Questions
# =============================================================================

print("=" * 60)
print("SIMPLE QUESTIONS")
print("=" * 60)

# Ask using the default model (Claude)
response = assistant.ask("What is Delta Lake and why is it useful?")
print("\nDefault model response:")
print(response)

# Explicitly use Gemini
response = assistant.ask_gemini("Explain the difference between DataFrame and Dataset in Spark")
print("\nGemini response:")
print(response)

# Explicitly use Claude
response = assistant.ask_claude("What are the best practices for Spark job optimization?")
print("\nClaude response:")
print(response)

# =============================================================================
# Changing Default Model
# =============================================================================

print("\n" + "=" * 60)
print("CHANGING DEFAULT MODEL")
print("=" * 60)

# Set Gemini as default
assistant.set_default_model("gemini")
response = assistant.ask("What is Apache Iceberg?")
print("\nUsing Gemini as default:")
print(response[:200] + "...")

# Switch back to Claude
assistant.set_default_model("claude")

# =============================================================================
# Custom System Instructions
# =============================================================================

print("\n" + "=" * 60)
print("CUSTOM SYSTEM INSTRUCTIONS")
print("=" * 60)

# Provide a specific persona or behavior
response = assistant.ask(
    "How should I structure my data pipeline?",
    system_instruction="""You are a senior data architect at a Fortune 500 company.
    Provide advice based on enterprise-grade requirements including:
    - Scalability to petabytes of data
    - 99.99% uptime requirements
    - Strict data governance and compliance
    Be concise and focus on actionable recommendations."""
)
print("\nWith custom system instruction:")
print(response)

# =============================================================================
# Temperature Control
# =============================================================================

print("\n" + "=" * 60)
print("TEMPERATURE CONTROL")
print("=" * 60)

# Low temperature for factual, deterministic responses
response = assistant.ask_claude(
    "List the ACID properties of Delta Lake",
    temperature=0.1
)
print("\nLow temperature (factual):")
print(response)

# Higher temperature for creative responses
response = assistant.ask_claude(
    "Suggest creative ways to visualize data pipeline performance",
    temperature=0.9
)
print("\nHigh temperature (creative):")
print(response)

# =============================================================================
# Multi-turn Conversations
# =============================================================================

print("\n" + "=" * 60)
print("MULTI-TURN CONVERSATIONS")
print("=" * 60)

# Start a conversation about a specific topic
print("Starting conversation about Spark optimization...")

response1 = assistant.chat(
    "I have a Spark job that's running slowly",
    conversation_name="spark_help"
)
print(f"\nUser: I have a Spark job that's running slowly")
print(f"AI: {response1[:300]}...")

response2 = assistant.chat(
    "It's doing a join between two large tables, about 1TB each",
    conversation_name="spark_help"
)
print(f"\nUser: It's doing a join between two large tables, about 1TB each")
print(f"AI: {response2[:300]}...")

response3 = assistant.chat(
    "The join key is customer_id. What specific changes should I make?",
    conversation_name="spark_help"
)
print(f"\nUser: The join key is customer_id. What specific changes should I make?")
print(f"AI: {response3[:400]}...")

# List active conversations
print(f"\nActive conversations: {assistant.gemini.list_conversations()}")

# Clear the conversation when done
assistant.clear_conversation("spark_help")
print("Conversation cleared.")

# =============================================================================
# Streaming Responses
# =============================================================================

print("\n" + "=" * 60)
print("STREAMING RESPONSES")
print("=" * 60)

print("Streaming response from Claude:")
for chunk in assistant.stream("Explain partitioning in Delta Lake"):
    print(chunk, end="", flush=True)
print("\n")

# =============================================================================
# Usage Statistics
# =============================================================================

print("\n" + "=" * 60)
print("USAGE STATISTICS")
print("=" * 60)

# Get usage summary
usage = assistant.get_usage_summary()
print("\nUsage Summary:")
print(f"  Gemini: {usage.get('gemini', {})}")
print(f"  Claude: {usage.get('claude', {})}")
print(f"  Total estimated cost: ${usage.get('total_estimated_cost', 0):.4f}")

# Reset stats if needed
assistant.reset_usage_stats()
print("\nUsage stats have been reset.")

# =============================================================================
# Error Handling
# =============================================================================

print("\n" + "=" * 60)
print("ERROR HANDLING")
print("=" * 60)

from ai_assistant import (
    AIAssistantError,
    APIKeyNotFoundError,
    RateLimitError,
    TokenLimitError
)

try:
    # This might fail if API key is invalid
    response = assistant.ask("Test query")
    print("Query successful!")
except APIKeyNotFoundError as e:
    print(f"API key error: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except TokenLimitError as e:
    print(f"Token limit exceeded: {e}")
except AIAssistantError as e:
    print(f"General AI error: {e}")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
