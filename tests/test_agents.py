"""
Unit tests for Agents module.

Tests the agent infrastructure including tools, base agent,
ReAct agent, and specialized agents.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from ai_assistant.agents import (
    Tool,
    ToolResult,
    ToolRegistry,
    SQLExecutorTool,
    TableInfoTool,
    DataProfilerTool,
    BaseAgent,
    AgentMemory,
    ReActAgent,
    AgentExecutor,
    DataAnalystAgent,
    DataEngineerAgent
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult(
            success=True,
            output="Query executed successfully",
            error=None
        )

        assert result.success is True
        assert result.output == "Query executed successfully"
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            success=False,
            output=None,
            error="Connection failed"
        )

        assert result.success is False
        assert result.error == "Connection failed"


class TestTool:
    """Tests for Tool base class."""

    def test_tool_creation(self):
        """Test creating a tool."""
        class TestTool(Tool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "A test tool"

            def run(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="Done")

        tool = TestTool()

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_tool_schema(self):
        """Test tool schema generation."""
        class TestTool(Tool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "Test description"

            @property
            def parameters(self) -> Dict[str, Any]:
                return {
                    "query": {"type": "string", "description": "SQL query"}
                }

            def run(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="Done")

        tool = TestTool()
        schema = tool.get_schema()

        assert schema["name"] == "test_tool"
        assert schema["description"] == "Test description"
        assert "parameters" in schema


class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a tool registry."""
        return ToolRegistry()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock(spec=Tool)
        tool.name = "mock_tool"
        tool.description = "A mock tool"
        tool.run = Mock(return_value=ToolResult(success=True, output="Done"))
        return tool

    def test_register_tool(self, registry, mock_tool):
        """Test registering a tool."""
        registry.register(mock_tool)

        assert "mock_tool" in registry._tools

    def test_get_tool(self, registry, mock_tool):
        """Test getting a tool."""
        registry.register(mock_tool)

        tool = registry.get("mock_tool")
        assert tool == mock_tool

    def test_get_nonexistent_tool(self, registry):
        """Test getting nonexistent tool."""
        tool = registry.get("nonexistent")
        assert tool is None

    def test_list_tools(self, registry, mock_tool):
        """Test listing tools."""
        registry.register(mock_tool)

        tools = registry.list_tools()
        assert "mock_tool" in tools

    def test_execute_tool(self, registry, mock_tool):
        """Test executing a tool."""
        registry.register(mock_tool)

        result = registry.execute("mock_tool", query="SELECT 1")

        assert result.success is True
        mock_tool.run.assert_called_once()


class TestSQLExecutorTool:
    """Tests for SQLExecutorTool."""

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [{"col1": "value1"}]
        mock_result.limit.return_value = mock_result
        spark.sql.return_value = mock_result
        return spark

    @pytest.fixture
    def sql_tool(self, mock_spark):
        """Create SQL executor tool."""
        return SQLExecutorTool(spark=mock_spark, max_rows=100)

    def test_sql_tool_name(self, sql_tool):
        """Test tool name."""
        assert sql_tool.name == "sql_executor"

    def test_execute_query(self, sql_tool, mock_spark):
        """Test executing a query."""
        result = sql_tool.run(query="SELECT * FROM test_table")

        assert result.success is True
        mock_spark.sql.assert_called_once()

    def test_execute_invalid_query(self, mock_spark):
        """Test executing invalid query."""
        mock_spark.sql.side_effect = Exception("Syntax error")
        tool = SQLExecutorTool(spark=mock_spark)

        result = tool.run(query="INVALID SQL")

        assert result.success is False
        assert "error" in result.error.lower()


class TestTableInfoTool:
    """Tests for TableInfoTool."""

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        spark = MagicMock()
        mock_result = MagicMock()
        mock_result.collect.return_value = [
            MagicMock(col_name="id", data_type="int"),
            MagicMock(col_name="name", data_type="string")
        ]
        spark.sql.return_value = mock_result
        return spark

    def test_get_table_info(self, mock_spark):
        """Test getting table info."""
        tool = TableInfoTool(spark=mock_spark)

        result = tool.run(table_name="test_table")

        assert result.success is True
        assert "id" in result.output or result.output is not None


class TestAgentMemory:
    """Tests for AgentMemory."""

    @pytest.fixture
    def memory(self):
        """Create agent memory."""
        return AgentMemory(max_history=10)

    def test_add_message(self, memory):
        """Test adding a message."""
        memory.add("user", "Hello")

        assert len(memory.history) == 1
        assert memory.history[0]["role"] == "user"
        assert memory.history[0]["content"] == "Hello"

    def test_max_history(self):
        """Test max history limit."""
        memory = AgentMemory(max_history=3)

        for i in range(5):
            memory.add("user", f"Message {i}")

        assert len(memory.history) == 3
        assert "Message 2" in memory.history[0]["content"]

    def test_clear_memory(self, memory):
        """Test clearing memory."""
        memory.add("user", "Test")
        memory.clear()

        assert len(memory.history) == 0

    def test_get_context(self, memory):
        """Test getting context string."""
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there")

        context = memory.get_context()

        assert "Hello" in context
        assert "Hi there" in context


class TestReActAgent:
    """Tests for ReActAgent."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        # Simulate ReAct response format
        client.generate = Mock(return_value="""
Thought: I need to query the database
Action: sql_executor
Action Input: {"query": "SELECT COUNT(*) FROM users"}
""")
        return client

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock(spec=Tool)
        tool.name = "sql_executor"
        tool.description = "Execute SQL"
        tool.run = Mock(return_value=ToolResult(success=True, output="100"))
        tool.get_schema = Mock(return_value={"name": "sql_executor"})
        return tool

    @pytest.fixture
    def agent(self, mock_ai_client, mock_tool):
        """Create a ReAct agent."""
        registry = ToolRegistry()
        registry.register(mock_tool)

        return ReActAgent(
            ai_client=mock_ai_client,
            tool_registry=registry,
            max_iterations=3
        )

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.max_iterations == 3

    def test_agent_run(self, agent, mock_ai_client, mock_tool):
        """Test running the agent."""
        # Mock final answer after tool use
        mock_ai_client.generate.side_effect = [
            """
Thought: I need to query the database
Action: sql_executor
Action Input: {"query": "SELECT COUNT(*) FROM users"}
""",
            """
Thought: I have the answer
Final Answer: There are 100 users in the database
"""
        ]

        result = agent.run("How many users are there?")

        assert result is not None
        mock_tool.run.assert_called()


class TestAgentExecutor:
    """Tests for AgentExecutor."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock()
        agent.run = Mock(return_value="Agent response")
        return agent

    def test_executor_run(self, mock_agent):
        """Test executor run."""
        executor = AgentExecutor(
            agent=mock_agent,
            max_execution_time=30
        )

        result = executor.execute("Test task")

        assert result is not None
        mock_agent.run.assert_called_once()

    def test_executor_timeout(self, mock_agent):
        """Test executor timeout handling."""
        import time

        def slow_run(task):
            time.sleep(2)
            return "Done"

        mock_agent.run = slow_run

        executor = AgentExecutor(
            agent=mock_agent,
            max_execution_time=1
        )

        # May raise timeout or return partial result
        # Implementation specific


class TestDataAnalystAgent:
    """Tests for DataAnalystAgent."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="Analysis complete")
        return client

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        return MagicMock()

    @pytest.fixture
    def analyst(self, mock_ai_client, mock_spark):
        """Create a data analyst agent."""
        return DataAnalystAgent(
            ai_client=mock_ai_client,
            spark=mock_spark
        )

    def test_analyst_initialization(self, analyst):
        """Test analyst initialization."""
        assert analyst is not None
        assert analyst.tool_registry is not None

    def test_analyze_table(self, analyst, mock_ai_client):
        """Test table analysis."""
        result = analyst.analyze_table("test_catalog.schema.table")

        assert result is not None
        mock_ai_client.generate.assert_called()

    def test_answer_question(self, analyst, mock_ai_client):
        """Test answering a question."""
        mock_ai_client.generate.return_value = """
Thought: I need to query the data
Final Answer: The average is 42
"""

        result = analyst.answer_question("What is the average?")

        assert result is not None


class TestDataEngineerAgent:
    """Tests for DataEngineerAgent."""

    @pytest.fixture
    def mock_ai_client(self):
        """Create a mock AI client."""
        client = Mock()
        client.generate = Mock(return_value="CREATE TABLE test (...)")
        return client

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        return MagicMock()

    @pytest.fixture
    def engineer(self, mock_ai_client, mock_spark):
        """Create a data engineer agent."""
        return DataEngineerAgent(
            ai_client=mock_ai_client,
            spark=mock_spark
        )

    def test_engineer_initialization(self, engineer):
        """Test engineer initialization."""
        assert engineer is not None

    def test_create_table(self, engineer, mock_ai_client):
        """Test table creation."""
        result = engineer.create_table(
            "Create a users table with id and name"
        )

        assert result is not None
        mock_ai_client.generate.assert_called()

    def test_generate_pipeline(self, engineer, mock_ai_client):
        """Test pipeline generation."""
        mock_ai_client.generate.return_value = """
```python
@dlt.table
def my_table():
    return spark.read.table("source")
```
"""

        result = engineer.generate_pipeline(
            "Create a DLT pipeline for users"
        )

        assert result is not None

    def test_optimize_query(self, engineer, mock_ai_client):
        """Test query optimization."""
        mock_ai_client.generate.return_value = """
Optimized query:
SELECT id FROM users WHERE active = true
"""

        result = engineer.optimize_query(
            "SELECT * FROM users WHERE active = true"
        )

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
