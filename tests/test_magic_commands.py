"""
Unit tests for Magic Commands module.

Tests IPython magic commands for AI Assistant integration
in notebook environments.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO


class TestAIMagics:
    """Tests for AIMagics class."""

    @pytest.fixture
    def mock_shell(self):
        """Create a mock IPython shell."""
        shell = MagicMock()
        shell.user_ns = {}
        return shell

    @pytest.fixture
    def mock_assistant(self):
        """Create a mock AI assistant."""
        assistant = Mock()
        assistant.ask = Mock(return_value="AI response")
        assistant.stream = Mock(return_value=iter(["chunk1", "chunk2"]))
        assistant.explain_error = Mock(return_value="Error explanation")
        return assistant

    @pytest.fixture
    def magics(self, mock_shell, mock_assistant):
        """Create AIMagics instance."""
        from ai_assistant.magic_commands import AIMagics

        mock_shell.user_ns['ai_assistant'] = mock_assistant
        return AIMagics(mock_shell, mock_assistant)

    def test_magics_initialization(self, magics, mock_assistant):
        """Test magics initialization."""
        assert magics._assistant == mock_assistant
        assert magics._last_response is None

    def test_ai_magic_basic(self, magics, mock_assistant, capsys):
        """Test basic %ai magic."""
        magics.ai("What is Spark?")

        mock_assistant.ask.assert_called_once()
        assert "Spark" in str(mock_assistant.ask.call_args)

    def test_ai_magic_with_model(self, magics, mock_assistant, capsys):
        """Test %ai with model specification."""
        magics.ai("-m gemini What is Delta Lake?")

        call_args = mock_assistant.ask.call_args
        assert call_args.kwargs.get('model') == 'gemini'

    def test_ai_magic_streaming(self, magics, mock_assistant, capsys):
        """Test %ai with streaming."""
        magics.ai("-s What is streaming?")

        mock_assistant.stream.assert_called_once()

    def test_ai_magic_empty_prompt(self, magics, capsys):
        """Test %ai with empty prompt."""
        magics.ai("")

        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_ai_cell_magic(self, magics, mock_assistant, capsys):
        """Test %%ai_cell magic."""
        magics.ai_cell("", "Multi-line\nprompt here")

        mock_assistant.ask.assert_called_once()
        assert "Multi-line" in str(mock_assistant.ask.call_args)

    def test_ai_cell_with_system(self, magics, mock_assistant, capsys):
        """Test %%ai_cell with system instruction."""
        magics.ai_cell('-s "You are a SQL expert"', "Optimize this query")

        call_args = mock_assistant.ask.call_args
        assert "SQL expert" in str(call_args)

    def test_ai_explain_magic(self, magics, mock_assistant, mock_shell, capsys):
        """Test %ai_explain magic."""
        mock_shell.user_ns['my_code'] = "def hello(): print('hi')"

        magics.ai_explain("-v my_code")

        mock_assistant.ask.assert_called_once()
        assert "hello" in str(mock_assistant.ask.call_args)

    def test_ai_explain_variable_not_found(self, magics, capsys):
        """Test %ai_explain with nonexistent variable."""
        magics.ai_explain("-v nonexistent_var")

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_ai_explain_detail_levels(self, magics, mock_assistant, mock_shell, capsys):
        """Test %ai_explain with different detail levels."""
        mock_shell.user_ns['code'] = "x = 1"

        magics.ai_explain("-v code -d brief")
        call1 = mock_assistant.ask.call_args

        mock_assistant.reset_mock()

        magics.ai_explain("-v code -d detailed")
        call2 = mock_assistant.ask.call_args

        # Different prompts for different detail levels
        assert str(call1) != str(call2)

    def test_ai_optimize_magic(self, magics, mock_assistant, capsys):
        """Test %%ai_optimize magic."""
        sql_query = "SELECT * FROM users WHERE active = true"

        magics.ai_optimize("", sql_query)

        mock_assistant.ask.assert_called_once()
        assert "Optimize" in str(mock_assistant.ask.call_args)
        assert "users" in str(mock_assistant.ask.call_args)

    def test_ai_generate_magic(self, magics, mock_assistant, capsys):
        """Test %%ai_generate magic."""
        magics.ai_generate("", "Create a function to validate emails")

        mock_assistant.ask.assert_called_once()
        assert "email" in str(mock_assistant.ask.call_args)

    def test_ai_generate_with_language(self, magics, mock_assistant, capsys):
        """Test %%ai_generate with language option."""
        magics.ai_generate("-l sql", "Create a query to find top users")

        call_args = mock_assistant.ask.call_args
        assert "sql" in str(call_args).lower()

    def test_ai_generate_with_tests(self, magics, mock_assistant, capsys):
        """Test %%ai_generate with tests option."""
        magics.ai_generate("-t", "Create a factorial function")

        call_args = mock_assistant.ask.call_args
        assert "test" in str(call_args).lower()

    def test_ai_generate_output_variable(self, magics, mock_assistant, mock_shell, capsys):
        """Test %%ai_generate with output variable."""
        magics.ai_generate("-o result", "Create hello function")

        assert "result" in mock_shell.user_ns

    def test_ai_last_magic(self, magics):
        """Test %ai_last magic."""
        magics._last_response = "Previous response"

        result = magics.ai_last("")

        assert result == "Previous response"

    def test_ai_last_no_response(self, magics, capsys):
        """Test %ai_last with no previous response."""
        magics._last_response = None

        result = magics.ai_last("")

        assert result is None
        captured = capsys.readouterr()
        assert "No previous" in captured.out

    def test_ai_fix_magic(self, magics, mock_assistant, mock_shell, capsys):
        """Test %ai_fix magic."""
        mock_shell.user_ns['error'] = "NameError: name 'x' is not defined"

        magics.ai_fix("-v error")

        mock_assistant.explain_error.assert_called_once()

    def test_ai_fix_with_code(self, magics, mock_assistant, mock_shell, capsys):
        """Test %ai_fix with code variable."""
        mock_shell.user_ns['error'] = "TypeError: unsupported operand"
        mock_shell.user_ns['code'] = "result = 'a' + 1"

        magics.ai_fix("-v error -c code")

        call_args = mock_assistant.explain_error.call_args
        assert call_args.kwargs.get('code') is not None

    def test_ai_help_magic(self, magics, capsys):
        """Test %ai_help magic."""
        magics.ai_help("")

        captured = capsys.readouterr()
        assert "AI Assistant" in captured.out
        assert "%ai" in captured.out
        assert "%%ai_cell" in captured.out


class TestMagicsWithoutAssistant:
    """Tests for magics when assistant is not available."""

    @pytest.fixture
    def mock_shell(self):
        """Create a mock shell without assistant."""
        shell = MagicMock()
        shell.user_ns = {}
        return shell

    def test_assistant_lazy_creation(self, mock_shell):
        """Test lazy assistant creation."""
        from ai_assistant.magic_commands import AIMagics

        magics = AIMagics(mock_shell, None)

        # Accessing assistant should try to create one
        with patch('ai_assistant.magic_commands.AIAssistant') as MockAI:
            MockAI.side_effect = Exception("No API key")
            _ = magics.assistant

        # Should have tried to create assistant


class TestLoadExtension:
    """Tests for extension loading."""

    def test_load_extension(self):
        """Test loading the extension."""
        from ai_assistant.magic_commands import load_ipython_extension

        mock_ipython = MagicMock()
        mock_ipython.user_ns = {}

        load_ipython_extension(mock_ipython)

        mock_ipython.register_magics.assert_called_once()

    def test_unload_extension(self):
        """Test unloading the extension."""
        from ai_assistant.magic_commands import unload_ipython_extension

        mock_ipython = MagicMock()

        # Should not raise
        unload_ipython_extension(mock_ipython)


class TestMagicsErrorHandling:
    """Tests for error handling in magics."""

    @pytest.fixture
    def magics_with_failing_assistant(self, mock_shell):
        """Create magics with a failing assistant."""
        from ai_assistant.magic_commands import AIMagics

        assistant = Mock()
        assistant.ask = Mock(side_effect=Exception("API Error"))
        mock_shell = MagicMock()
        mock_shell.user_ns = {'ai_assistant': assistant}

        return AIMagics(mock_shell, assistant)

    @pytest.fixture
    def mock_shell(self):
        """Create mock shell."""
        shell = MagicMock()
        shell.user_ns = {}
        return shell

    def test_ai_magic_error_handling(self, magics_with_failing_assistant, capsys):
        """Test error handling in %ai magic."""
        magics_with_failing_assistant.ai("Test prompt")

        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_ai_cell_error_handling(self, magics_with_failing_assistant, capsys):
        """Test error handling in %%ai_cell magic."""
        magics_with_failing_assistant.ai_cell("", "Test prompt")

        captured = capsys.readouterr()
        assert "Error" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
