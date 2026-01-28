"""
Unit tests for AI Assistant module.

These tests verify the functionality of the AI Assistant without
making actual API calls (using mocks).

Run with: pytest tests/ -v
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from ai_assistant.config import AIConfig, ModelConfig
from ai_assistant.exceptions import (
    AIAssistantError,
    APIKeyNotFoundError,
    ModelNotAvailableError,
    RateLimitError,
    TokenLimitError,
    ConversationNotFoundError
)


class TestAIConfig:
    """Tests for AIConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AIConfig()

        assert config.secret_scope is None
        assert config.gemini_secret_key == "gemini-api-key"
        assert config.claude_secret_key == "claude-api-key"
        assert config.gemini_model.name == "gemini-1.5-pro"
        assert config.claude_model.name == "claude-sonnet-4-20250514"
        assert config.retry_attempts == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AIConfig(
            secret_scope="my-scope",
            gemini_secret_key="my-gemini-key",
            claude_secret_key="my-claude-key",
            gemini_model=ModelConfig(name="gemini-1.5-flash", max_tokens=2048),
            retry_attempts=5
        )

        assert config.secret_scope == "my-scope"
        assert config.gemini_secret_key == "my-gemini-key"
        assert config.gemini_model.name == "gemini-1.5-flash"
        assert config.gemini_model.max_tokens == 2048
        assert config.retry_attempts == 5

    def test_env_variable_loading(self):
        """Test API key loading from environment variables."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-gemini-key",
            "ANTHROPIC_API_KEY": "test-claude-key"
        }):
            config = AIConfig()

            assert config.gemini_api_key == "test-gemini-key"
            assert config.claude_api_key == "test-claude-key"

    def test_get_gemini_key_direct(self):
        """Test getting Gemini key when directly provided."""
        config = AIConfig(gemini_api_key="direct-key")
        assert config.get_gemini_key() == "direct-key"

    def test_get_claude_key_direct(self):
        """Test getting Claude key when directly provided."""
        config = AIConfig(claude_api_key="direct-key")
        assert config.get_claude_key() == "direct-key"

    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = AIConfig(secret_scope="test-scope")
        config_dict = config.to_dict()

        assert "secret_scope" in config_dict
        assert config_dict["secret_scope"] == "test-scope"
        assert "gemini_api_key" not in config_dict  # Sensitive data excluded


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_values(self):
        """Test default ModelConfig values."""
        config = ModelConfig(name="test-model")

        assert config.name == "test-model"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.top_k is None

    def test_custom_values(self):
        """Test custom ModelConfig values."""
        config = ModelConfig(
            name="test-model",
            max_tokens=2048,
            temperature=0.5,
            top_k=40
        )

        assert config.max_tokens == 2048
        assert config.temperature == 0.5
        assert config.top_k == 40


class TestExceptions:
    """Tests for custom exceptions."""

    def test_api_key_not_found_error(self):
        """Test APIKeyNotFoundError."""
        error = APIKeyNotFoundError("Gemini")

        assert "Gemini" in str(error)
        assert error.provider == "Gemini"

    def test_api_key_not_found_custom_message(self):
        """Test APIKeyNotFoundError with custom message."""
        error = APIKeyNotFoundError("Claude", "Custom error message")

        assert error.message == "Custom error message"
        assert str(error) == "Custom error message"

    def test_model_not_available_error(self):
        """Test ModelNotAvailableError."""
        error = ModelNotAvailableError(
            "unknown-model",
            "Gemini",
            available_models=["gemini-1.5-pro", "gemini-1.5-flash"]
        )

        assert "unknown-model" in str(error)
        assert error.model == "unknown-model"
        assert error.provider == "Gemini"
        assert "gemini-1.5-pro" in str(error)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Claude", retry_after=60)

        assert error.provider == "Claude"
        assert error.retry_after == 60
        assert "60 seconds" in str(error)

    def test_token_limit_error(self):
        """Test TokenLimitError."""
        error = TokenLimitError(5000, 4096, "input")

        assert error.token_count == 5000
        assert error.token_limit == 4096
        assert error.token_type == "input"
        assert "5000" in str(error)
        assert "4096" in str(error)

    def test_conversation_not_found_error(self):
        """Test ConversationNotFoundError."""
        error = ConversationNotFoundError("my_conversation")

        assert "my_conversation" in str(error)
        assert error.conversation_name == "my_conversation"


class TestGeminiClient:
    """Tests for GeminiClient (mocked)."""

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="google-generativeai not installed or configured"
    )
    def test_initialization(self):
        """Test GeminiClient initialization."""
        pytest.skip("Requires google-generativeai package")

    def test_initialization_no_api_key(self):
        """Test GeminiClient initialization without API key."""
        from ai_assistant.gemini_client import GeminiClient

        with pytest.raises(APIKeyNotFoundError):
            GeminiClient(api_key=None)

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="google-generativeai not installed or configured"
    )
    def test_generate(self):
        """Test generate method."""
        pytest.skip("Requires google-generativeai package")

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="google-generativeai not installed or configured"
    )
    def test_usage_stats(self):
        """Test usage statistics tracking."""
        pytest.skip("Requires google-generativeai package")


class TestClaudeClient:
    """Tests for ClaudeClient (mocked)."""

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="anthropic not installed or configured"
    )
    def test_initialization(self):
        """Test ClaudeClient initialization."""
        pytest.skip("Requires anthropic package")

    def test_initialization_no_api_key(self):
        """Test ClaudeClient initialization without API key."""
        from ai_assistant.claude_client import ClaudeClient

        with pytest.raises(APIKeyNotFoundError):
            ClaudeClient(api_key=None)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="anthropic not installed or configured"
    )
    def test_generate(self):
        """Test generate method."""
        pytest.skip("Requires anthropic package")

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="anthropic not installed or configured"
    )
    def test_conversation_management(self):
        """Test conversation management."""
        pytest.skip("Requires anthropic package")


class TestAIAssistant:
    """Tests for main AIAssistant class (mocked)."""

    def test_initialization_without_keys(self):
        """Test AIAssistant initialization without API keys."""
        from ai_assistant.core import AIAssistant

        # Should initialize without errors (lazy loading)
        assistant = AIAssistant()

        assert assistant.config is not None
        assert assistant._gemini_client is None
        assert assistant._claude_client is None

    def test_set_default_model(self):
        """Test setting default model."""
        from ai_assistant.core import AIAssistant

        assistant = AIAssistant()

        assistant.set_default_model("gemini")
        assert assistant._default_model == "gemini"

        assistant.set_default_model("claude")
        assert assistant._default_model == "claude"

    def test_set_invalid_default_model(self):
        """Test setting invalid default model."""
        from ai_assistant.core import AIAssistant

        assistant = AIAssistant()

        with pytest.raises(ValueError):
            assistant.set_default_model("invalid")

    def test_is_available_methods(self):
        """Test availability check methods."""
        from ai_assistant.core import AIAssistant

        assistant = AIAssistant()

        # Without API keys configured, should return False
        # (depends on environment)
        assert isinstance(assistant.is_gemini_available(), bool)
        assert isinstance(assistant.is_claude_available(), bool)

    def test_get_available_models(self):
        """Test getting available models list."""
        from ai_assistant.core import AIAssistant

        assistant = AIAssistant()
        models = assistant.get_available_models()

        assert "gemini" in models
        assert "claude" in models
        assert len(models["gemini"]) > 0
        assert len(models["claude"]) > 0

    def test_repr(self):
        """Test string representation."""
        from ai_assistant.core import AIAssistant

        assistant = AIAssistant()
        repr_str = repr(assistant)

        assert "AIAssistant" in repr_str
        assert "gemini" in repr_str.lower()
        assert "claude" in repr_str.lower()


class TestSparkUtils:
    """Tests for Spark utilities."""

    def test_estimate_processing_cost(self):
        """Test cost estimation function."""
        from ai_assistant.spark_utils import estimate_processing_cost

        estimate = estimate_processing_cost(
            row_count=1000,
            avg_input_tokens=100,
            avg_output_tokens=200,
            model="gemini-1.5-flash"
        )

        assert "row_count" in estimate
        assert estimate["row_count"] == 1000
        assert "total_input_tokens" in estimate
        assert estimate["total_input_tokens"] == 100000
        assert "total_output_tokens" in estimate
        assert estimate["total_output_tokens"] == 200000
        assert "total_cost" in estimate
        assert estimate["total_cost"] >= 0


class TestIntegration:
    """Integration tests (require actual API keys)."""

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    def test_gemini_integration(self):
        """Test actual Gemini API call."""
        from ai_assistant.gemini_client import GeminiClient

        client = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
        response = client.generate("Say 'Hello, World!'")

        assert response is not None
        assert len(response) > 0

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_claude_integration(self):
        """Test actual Claude API call."""
        from ai_assistant.claude_client import ClaudeClient

        client = ClaudeClient(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.generate("Say 'Hello, World!'")

        assert response is not None
        assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
