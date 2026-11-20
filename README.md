# ReAct Tool Agent

A clean, educational implementation of the **ReAct (Reason + Act)** agent pattern.

This project demonstrates core skills expected of an **AI Agent Engineer**:

- Agent loop (Thought → Action → Observation)
- Tool definition & calling
- Short-term conversational memory
- Structured step logging / observability
- Separation of decision logic vs tool execution
- Easy path to swap the rule-based reasoner for a real LLM

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # currently no hard dependencies

python agent.py
```

### Example interactions

```
You: What is 25 * 4 + 10?
→ Uses calculator tool

You: What time is it?
→ Uses get_current_time tool

You: How many words in "AI agents are the future of software"?
→ Uses word_count tool

You: Reverse this text: hello world
→ Uses reverse_text tool
```

## Architecture

```
User Input
    ↓
ReActAgent.run()
    ↓
┌─────────────────────────────┐
│  1. Decide (Thought)        │  ← currently rule-based; replace with LLM
│  2. Select Tool + Args      │
│  3. Execute Tool            │
│  4. Observe result          │
│  5. Store in memory         │
│  6. Return final answer     │
└─────────────────────────────┘
```

## Key Components

| Component          | File       | Purpose                                      |
|--------------------|------------|----------------------------------------------|
| `Tool`             | agent.py   | Tool schema + safe execution                 |
| `ShortTermMemory`  | agent.py   | Rolling conversation history                 |
| `ReActAgent`       | agent.py   | Main agent loop + decision logic             |
| `AgentStep`        | agent.py   | Structured representation of one reasoning step |

## Extending to a real LLM

Replace the `_decide()` method with an LLM call that returns structured output:

```python
{
  "thought": "I need to calculate this expression",
  "action": "calculator",
  "action_input": {"expression": "25 * 4 + 10"}
}
```

Recommended libraries for production:
- OpenAI / Anthropic function calling or tool use
- LangChain / LlamaIndex agents
- Instructor or Pydantic for structured output

## Skills Demonstrated

- Designing agent control loops
- Tool abstraction and registration
- Memory management
- Observability of agent steps
- Clean separation of concerns for testability and replacement of the reasoner

## License

MIT
