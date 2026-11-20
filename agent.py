"""
ReAct Tool Agent
A clean implementation of the ReAct (Reason + Act) pattern for AI agents.
Demonstrates: tool calling, agent loop, short-term memory, structured logging.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Tool system
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: Dict[str, str]  # name -> description

    def run(self, **kwargs) -> ToolResult:
        try:
            result = self.func(**kwargs)
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    allowed = set("0123456789+-*/(). %")
    if not all(c in allowed for c in expression):
        raise ValueError("Expression contains invalid characters")
    return str(eval(expression, {"__builtins__": {}}, {}))


def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current time (simplified)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def word_count(text: str) -> str:
    """Count words in the given text."""
    words = text.strip().split()
    return f"Word count: {len(words)}"


def reverse_text(text: str) -> str:
    """Reverse the provided text."""
    return text[::-1]


DEFAULT_TOOLS = [
    Tool(
        name="calculator",
        description="Evaluate a basic mathematical expression (e.g. '2 + 3 * 4')",
        func=calculator,
        parameters={"expression": "Mathematical expression to evaluate"},
    ),
    Tool(
        name="get_current_time",
        description="Get the current UTC time",
        func=get_current_time,
        parameters={"timezone_name": "Timezone name (currently only UTC supported)"},
    ),
    Tool(
        name="word_count",
        description="Count the number of words in a text string",
        func=word_count,
        parameters={"text": "The text to count words in"},
    ),
    Tool(
        name="reverse_text",
        description="Reverse a string of text",
        func=reverse_text,
        parameters={"text": "The text to reverse"},
    ),
]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@dataclass
class Message:
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[ToolResult] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ShortTermMemory:
    def __init__(self, max_messages: int = 20):
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_context(self) -> str:
        lines = []
        for m in self.messages:
            if m.role == "tool":
                lines.append(f"[Tool Result] {m.content}")
            else:
                lines.append(f"[{m.role.upper()}] {m.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.messages.clear()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@dataclass
class AgentStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


class ReActAgent:
    """
    Simple ReAct agent.
    In a real system this would call an LLM. Here we use a rule-based
    "reasoner" so the project runs without API keys and still demonstrates
    the full agent loop, tool calling, and memory.
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        max_steps: int = 6,
        verbose: bool = True,
    ):
        self.tools = {t.name: t for t in (tools or DEFAULT_TOOLS)}
        self.max_steps = max_steps
        self.verbose = verbose
        self.memory = ShortTermMemory()
        self.run_id = str(uuid4())[:8]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.run_id}] {msg}")

    def _get_tools_description(self) -> str:
        lines = []
        for name, tool in self.tools.items():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            lines.append(f"- {name}({params}): {tool.description}")
        return "\n".join(lines)

    def _decide(self, user_input: str) -> AgentStep:
        """
        Rule-based decision maker (stands in for an LLM).
        In production you would replace this method with an LLM call
        that returns structured JSON: {"thought": "...", "action": "...", "action_input": {...}}
        """
        lower = user_input.lower()

        # Calculator
        if any(op in user_input for op in ["+", "-", "*", "/"]) and any(c.isdigit() for c in user_input):
            # Extract a simple expression
            expr = "".join(c for c in user_input if c in "0123456789+-*/(). ")
            return AgentStep(
                thought="The user is asking for a calculation. I should use the calculator tool.",
                action="calculator",
                action_input={"expression": expr.strip()},
            )

        # Time
        if "time" in lower or "date" in lower or "what day" in lower:
            return AgentStep(
                thought="The user is asking about the current time. I should use the get_current_time tool.",
                action="get_current_time",
                action_input={"timezone_name": "UTC"},
            )

        # Word count
        if "how many words" in lower or "word count" in lower:
            # naive extraction
            text = user_input
            for prefix in ["how many words in", "word count of", "count the words in"]:
                if prefix in lower:
                    text = user_input[lower.find(prefix) + len(prefix) :].strip(" ?:\"'")
                    break
            return AgentStep(
                thought="The user wants a word count. I will use the word_count tool.",
                action="word_count",
                action_input={"text": text},
            )

        # Reverse
        if "reverse" in lower:
            text = user_input
            for prefix in ["reverse this", "reverse the text", "reverse"]:
                if prefix in lower:
                    text = user_input[lower.find(prefix) + len(prefix) :].strip(" ?:\"'")
                    break
            return AgentStep(
                thought="The user wants text reversed. I will use the reverse_text tool.",
                action="reverse_text",
                action_input={"text": text},
            )

        # Fallback – no tool needed
        return AgentStep(
            thought="I can answer this directly without tools.",
            action=None,
            action_input=None,
        )

    def run(self, user_input: str) -> str:
        self._log(f"User: {user_input}")
        self.memory.add(Message(role="user", content=user_input))

        final_answer = None

        for step_num in range(1, self.max_steps + 1):
            self._log(f"--- Step {step_num} ---")

            step = self._decide(user_input if step_num == 1 else f"Continue from previous observation")

            self._log(f"Thought: {step.thought}")

            if step.action is None:
                # Direct answer
                final_answer = (
                    "I understood your request. "
                    "In a full LLM-powered version I would generate a natural language answer here. "
                    f"(Tools available: {', '.join(self.tools.keys())})"
                )
                break

            if step.action not in self.tools:
                self._log(f"Unknown tool: {step.action}")
                final_answer = f"I tried to use an unknown tool: {step.action}"
                break

            tool = self.tools[step.action]
            self._log(f"Action: {step.action}({step.action_input})")

            result = tool.run(**(step.action_input or {}))
            observation = result.output if result.success else f"Error: {result.error}"
            self._log(f"Observation: {observation}")

            self.memory.add(
                Message(
                    role="tool",
                    content=observation,
                    tool_call={"name": step.action, "input": step.action_input},
                    tool_result=result,
                )
            )

            # In a real ReAct loop the LLM would see the observation and decide next step.
            # Here we treat a successful tool call as the final answer for simplicity.
            final_answer = observation
            break

        if final_answer is None:
            final_answer = "I reached the maximum number of steps without a final answer."

        self.memory.add(Message(role="assistant", content=final_answer))
        self._log(f"Final Answer: {final_answer}")
        return final_answer


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ReAct Tool Agent Demo")
    print("Type 'quit' or 'exit' to stop. Type 'tools' to list available tools.")
    print("=" * 60)

    agent = ReActAgent(verbose=True)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "tools":
            print(agent._get_tools_description())
            continue

        answer = agent.run(user_input)
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
