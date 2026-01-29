"""
Base Agent Infrastructure.

This module provides the foundational classes for building
AI agents that can reason, plan, and execute tasks using tools.

The agent architecture follows a ReAct-style pattern:
1. Reason about the task
2. Decide on an action (tool call)
3. Execute the action
4. Observe the result
5. Repeat until task is complete
"""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum

from .tools import Tool, ToolResult, ToolStatus, ToolRegistry


class AgentStatus(Enum):
    """Status of an agent execution."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AgentStep:
    """
    Represents a single step in agent execution.

    Attributes:
        thought: Agent's reasoning
        action: Tool name to call
        action_input: Tool input parameters
        observation: Tool result
        timestamp: When the step occurred
    """
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "timestamp": self.timestamp
        }


@dataclass
class AgentResult:
    """
    Final result of an agent execution.

    Attributes:
        status: Execution status
        output: Final output/answer
        steps: List of execution steps
        total_time: Total execution time in seconds
        error: Error message if failed
        metadata: Additional result metadata
    """
    status: AgentStatus
    output: Optional[str] = None
    steps: List[AgentStep] = field(default_factory=list)
    total_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == AgentStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "output": self.output,
            "steps": [s.to_dict() for s in self.steps],
            "total_time": self.total_time,
            "error": self.error,
            "metadata": self.metadata
        }


class AgentMemory:
    """
    Memory management for agents.

    Stores conversation history, context, and working memory
    for multi-turn agent interactions.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._messages: List[Dict[str, str]] = []
        self._working_memory: Dict[str, Any] = {}
        self._facts: List[str] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history."""
        self._messages.append({"role": role, "content": content})
        # Trim history if needed
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get message history."""
        return self._messages.copy()

    def set_context(self, key: str, value: Any) -> None:
        """Set a working memory value."""
        self._working_memory[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a working memory value."""
        return self._working_memory.get(key, default)

    def add_fact(self, fact: str) -> None:
        """Add a learned fact."""
        if fact not in self._facts:
            self._facts.append(fact)

    def get_facts(self) -> List[str]:
        """Get all learned facts."""
        return self._facts.copy()

    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()
        self._working_memory.clear()
        self._facts.clear()

    def get_summary(self) -> str:
        """Get a summary of memory contents."""
        parts = []

        if self._facts:
            parts.append("Known Facts:")
            parts.extend([f"- {fact}" for fact in self._facts[:10]])

        if self._working_memory:
            parts.append("\nContext:")
            for key, value in list(self._working_memory.items())[:5]:
                parts.append(f"- {key}: {str(value)[:100]}")

        return "\n".join(parts) if parts else "No memory stored"


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Agents combine LLM reasoning with tool use to accomplish
    complex tasks autonomously.

    Args:
        ai_client: AI client for LLM calls
        tools: List of available tools
        max_iterations: Maximum reasoning iterations
        verbose: Whether to print intermediate steps

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def create_system_prompt(self) -> str:
        ...         return "You are a helpful assistant."
        ...
        ...     def parse_response(self, response: str) -> AgentStep:
        ...         # Parse LLM response into thought/action
        ...         pass
    """

    def __init__(
        self,
        ai_client: Any,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        memory: Optional[AgentMemory] = None
    ):
        self.ai_client = ai_client
        self.tools = tools or []
        self.tool_registry = ToolRegistry()
        for tool in self.tools:
            self.tool_registry.register(tool)

        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory = memory or AgentMemory()

    @abstractmethod
    def create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        pass

    @abstractmethod
    def parse_response(self, response: str) -> AgentStep:
        """
        Parse LLM response into an AgentStep.

        Should extract thought, action, and action_input from
        the LLM's response.
        """
        pass

    @abstractmethod
    def format_tool_result(self, step: AgentStep) -> str:
        """Format a tool result for the next LLM call."""
        pass

    @abstractmethod
    def is_finished(self, step: AgentStep) -> bool:
        """Check if the agent has finished its task."""
        pass

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the prompt."""
        descriptions = []
        for tool in self.tools:
            params = []
            for param in tool.parameters:
                param_str = f"  - {param.name}: {param.description}"
                if not param.required:
                    param_str += f" (optional, default: {param.default})"
                params.append(param_str)

            desc = f"""Tool: {tool.name}
Description: {tool.description}
Parameters:
{chr(10).join(params)}"""
            descriptions.append(desc)

        return "\n\n".join(descriptions)

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: The task description
            context: Optional additional context

        Returns:
            AgentResult with execution details
        """
        start_time = time.time()
        steps: List[AgentStep] = []

        # Store context in memory
        if context:
            for key, value in context.items():
                self.memory.set_context(key, value)

        # Build initial prompt
        system_prompt = self.create_system_prompt()
        user_prompt = self._build_initial_prompt(task, context)

        self.memory.add_message("user", task)

        if self.verbose:
            print(f"Task: {task}")
            print("-" * 50)

        # Reasoning loop
        for iteration in range(self.max_iterations):
            try:
                # Get LLM response
                response = self.ai_client.generate(
                    user_prompt,
                    system_instruction=system_prompt
                )

                if self.verbose:
                    print(f"\n[Iteration {iteration + 1}]")
                    print(f"Response: {response[:500]}...")

                # Parse response
                step = self.parse_response(response)
                steps.append(step)

                if self.verbose:
                    print(f"Thought: {step.thought}")
                    if step.action:
                        print(f"Action: {step.action}")
                        print(f"Input: {step.action_input}")

                # Check if finished
                if self.is_finished(step):
                    self.memory.add_message("assistant", step.thought)
                    return AgentResult(
                        status=AgentStatus.COMPLETED,
                        output=step.thought,
                        steps=steps,
                        total_time=time.time() - start_time
                    )

                # Execute tool if action specified
                if step.action:
                    tool_result = self._execute_tool(step.action, step.action_input or {})
                    step.observation = str(tool_result)

                    if self.verbose:
                        print(f"Observation: {step.observation[:500]}...")

                    # Build next prompt
                    user_prompt = self.format_tool_result(step)
                else:
                    # No action, might be done or need more reasoning
                    user_prompt = "Please continue with the task or provide a final answer."

            except Exception as e:
                return AgentResult(
                    status=AgentStatus.FAILED,
                    steps=steps,
                    total_time=time.time() - start_time,
                    error=str(e)
                )

        # Max iterations reached
        return AgentResult(
            status=AgentStatus.MAX_ITERATIONS,
            output="Maximum iterations reached without completing the task.",
            steps=steps,
            total_time=time.time() - start_time
        )

    def _build_initial_prompt(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the initial user prompt."""
        prompt_parts = [f"Task: {task}"]

        if context:
            prompt_parts.append("\nContext:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")

        # Add memory summary if available
        memory_summary = self.memory.get_summary()
        if memory_summary != "No memory stored":
            prompt_parts.append(f"\nMemory:\n{memory_summary}")

        return "\n".join(prompt_parts)

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool and return the result."""
        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return ToolResult(
                ToolStatus.ERROR,
                error=f"Tool '{tool_name}' not found. Available tools: {self.tool_registry.list_tools()}"
            )

        return tool(**tool_input)


class ReActAgent(BaseAgent):
    """
    ReAct-style agent implementation.

    Uses the Reason-Act pattern where the LLM explicitly
    states its reasoning before taking actions.

    Example:
        >>> agent = ReActAgent(
        ...     ai_client=claude_client,
        ...     tools=[SQLExecutorTool(spark)],
        ...     verbose=True
        ... )
        >>> result = agent.run("Find the top 5 customers by revenue")
    """

    REACT_TEMPLATE = """You are an AI assistant that solves tasks using a combination of reasoning and tools.

For each step, you must respond in the following format:

Thought: [Your reasoning about what to do next]
Action: [The tool to use, or "Final Answer" if you're done]
Action Input: [The input for the tool as valid JSON, or your final answer]

Available Tools:
{tool_descriptions}

Rules:
1. Always start with a Thought explaining your reasoning
2. Use tools to gather information and perform actions
3. When you have enough information, use "Final Answer" as the Action
4. Be concise but thorough in your reasoning
5. If a tool fails, reason about why and try a different approach

Remember: You must ALWAYS respond with Thought/Action/Action Input format."""

    def create_system_prompt(self) -> str:
        """Create ReAct system prompt."""
        return self.REACT_TEMPLATE.format(
            tool_descriptions=self.get_tool_descriptions()
        )

    def parse_response(self, response: str) -> AgentStep:
        """Parse ReAct-formatted response."""
        # Extract Thought
        thought_match = re.search(
            r'Thought:\s*(.+?)(?=\nAction:|$)',
            response,
            re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract Action
        action_match = re.search(
            r'Action:\s*(.+?)(?=\nAction Input:|$)',
            response,
            re.DOTALL
        )
        action = action_match.group(1).strip() if action_match else None

        # Extract Action Input
        action_input = None
        input_match = re.search(
            r'Action Input:\s*(.+?)(?=\n\n|$)',
            response,
            re.DOTALL
        )
        if input_match:
            input_str = input_match.group(1).strip()
            # Try to parse as JSON
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # If not JSON, treat as simple string
                action_input = {"input": input_str}

        return AgentStep(
            thought=thought,
            action=action,
            action_input=action_input
        )

    def format_tool_result(self, step: AgentStep) -> str:
        """Format tool result for next iteration."""
        return f"""Observation: {step.observation}

Based on this observation, continue with your next thought and action."""

    def is_finished(self, step: AgentStep) -> bool:
        """Check if agent is finished (Final Answer action)."""
        if step.action:
            return step.action.lower() == "final answer"
        return False


class AgentExecutor:
    """
    High-level executor for running agents with additional features.

    Provides timeout handling, logging, and result formatting.

    Args:
        agent: The agent to execute
        timeout_seconds: Maximum execution time
        on_step: Callback for each step
        on_complete: Callback when done

    Example:
        >>> executor = AgentExecutor(agent, timeout_seconds=300)
        >>> result = executor.run("Analyze sales data")
    """

    def __init__(
        self,
        agent: BaseAgent,
        timeout_seconds: int = 600,
        on_step: Optional[Callable[[AgentStep], None]] = None,
        on_complete: Optional[Callable[[AgentResult], None]] = None
    ):
        self.agent = agent
        self.timeout_seconds = timeout_seconds
        self.on_step = on_step
        self.on_complete = on_complete

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Run the agent with execution management.

        Args:
            task: Task description
            context: Optional context

        Returns:
            AgentResult
        """
        start_time = time.time()

        # Simple timeout check (not true timeout, just periodic check)
        original_max_iterations = self.agent.max_iterations

        try:
            result = self.agent.run(task, context)

            # Call step callbacks
            if self.on_step:
                for step in result.steps:
                    self.on_step(step)

            # Call completion callback
            if self.on_complete:
                self.on_complete(result)

            return result

        except Exception as e:
            result = AgentResult(
                status=AgentStatus.FAILED,
                error=str(e),
                total_time=time.time() - start_time
            )

            if self.on_complete:
                self.on_complete(result)

            return result

        finally:
            self.agent.max_iterations = original_max_iterations

    def run_interactive(
        self,
        initial_task: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Run agent interactively with user feedback.

        Allows user to provide additional input between iterations.

        Args:
            initial_task: Initial task
            context: Optional context

        Yields:
            AgentStep for each step
        """
        # Store context
        if context:
            for key, value in context.items():
                self.agent.memory.set_context(key, value)

        current_task = initial_task

        while True:
            result = self.agent.run(current_task, context)

            for step in result.steps:
                yield step

            if result.status == AgentStatus.COMPLETED:
                return

            # Could add user input here for interactive mode
            break


def create_react_agent(
    ai_client: Any,
    tools: List[Tool],
    verbose: bool = False
) -> ReActAgent:
    """
    Factory function to create a ReAct agent.

    Args:
        ai_client: AI client for LLM calls
        tools: List of tools to use
        verbose: Whether to print steps

    Returns:
        Configured ReActAgent
    """
    return ReActAgent(
        ai_client=ai_client,
        tools=tools,
        verbose=verbose
    )
