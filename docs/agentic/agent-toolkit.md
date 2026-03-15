# NeMo Agent Toolkit (NAT)

> Framework-agnostic toolkit for building, evaluating, and optimizing AI agents.

## Canonical URLs

| Resource | URL |
|---|---|
| Documentation | `https://docs.nvidia.com/nemo/agent-toolkit/latest/` |
| GitHub | `https://github.com/NVIDIA/nemo-agent-toolkit` |
| PyPI | `pip install nvidia-nat` |

## Overview

NeMo Agent Toolkit (NAT) provides a framework-agnostic way to build AI agents that work with popular orchestration frameworks. It adds profiling, evaluation, hyperparameter optimization, and RL fine-tuning on top of any agent framework.

## Supported Frameworks

| Framework | Integration |
|---|---|
| LangChain | Native adapter |
| LlamaIndex | Native adapter |
| CrewAI | Native adapter |
| Semantic Kernel | Native adapter |
| Google ADK | Native adapter |

## Agent Types

| Type | Description |
|---|---|
| **ReAct** | Reason + Act loop — think, then use tools |
| **Reasoning** | Extended chain-of-thought before acting |
| **ReWOO** | Reason Without Observation — plan all tool calls upfront |
| **Router** | Route to specialized sub-agents based on query type |
| **Tool Calling** | Direct function/tool calling (no reasoning loop) |
| **Sequential Executor** | Execute a fixed sequence of steps |

## Key Features

### Protocol Support
- **MCP (Model Context Protocol)** — Connect to any MCP-compatible tool server
- **A2A (Agent-to-Agent)** — Inter-agent communication protocol

### Profiling
Built-in profiling from agent-level down to individual tokens:
- Latency breakdowns (tool calls, LLM inference, orchestration)
- Token usage tracking
- Cost estimation

### Evaluation
- Task completion metrics
- Tool use accuracy
- Response quality scoring
- Custom evaluation metrics

### Optimization
- Hyperparameter search for agent configurations
- RL fine-tuning of agent behavior
- Agent Performance Primitives (APP) for parallel execution

### NVIDIA Dynamo Integration
High-throughput inference orchestration for production deployments.

## Quick Start

```python
from nvidia_nat import Agent, Tool

# Define tools
@Tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return search_engine.query(query)

# Create agent
agent = Agent(
    type="react",
    tools=[search_docs],
    llm="nvidia/nemotron-4-340b-instruct",
)

# Run
result = agent.run("Find docs about NeMo training")
```

## Web UI and API Server

NAT includes a built-in web interface and API server for interactive agent development and testing.

```bash
nat serve --port 8080
# Open http://localhost:8080
```
