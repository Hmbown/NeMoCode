# Agentic AI Services

> Frameworks and tools for building AI agents with NVIDIA technology.

## Overview

NVIDIA provides a full stack for building agentic AI applications — from safety guardrails to agent orchestration to reference architectures.

## Components

| Component | Doc | Description |
|---|---|---|
| [NeMo Guardrails](guardrails.md) | `docs/agentic/guardrails.md` | Programmable safety rails for LLM applications |
| [Agent Toolkit](agent-toolkit.md) | `docs/agentic/agent-toolkit.md` | Framework-agnostic agent building toolkit |
| [Examples](examples.md) | `docs/agentic/examples.md` | Reference architectures for RAG, agents, and workflows |

## Canonical URLs

| Resource | URL |
|---|---|
| NeMo Guardrails Docs | `https://docs.nvidia.com/nemo/guardrails/index.html` |
| Agent Toolkit Docs | `https://docs.nvidia.com/nemo/agent-toolkit/latest/` |
| Generative AI Examples | `https://github.com/NVIDIA/GenerativeAIExamples` |

## Architecture Pattern

```
User Request
    ↓
┌─────────────────────────┐
│   NeMo Guardrails       │  ← Input rails (jailbreak, topic, safety)
│   (Colang DSL)          │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│   Agent Toolkit         │  ← Agent orchestration (ReAct, Router, etc.)
│   (NAT Framework)       │
│   ┌───────────────┐     │
│   │ Tools / MCP   │     │  ← External tool access
│   │ A2A Protocol  │     │  ← Agent-to-agent communication
│   └───────────────┘     │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│   NIM / LLM Backend     │  ← Inference via NIM or cloud API
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│   NeMo Guardrails       │  ← Output rails (content safety, factuality)
└─────────────────────────┘
           ↓
      Response
```
