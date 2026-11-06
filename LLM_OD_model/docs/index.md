# LLM Opinion Dynamics Simulator

Welcome to the **LLM Opinion Dynamics Simulator** documentation. 

This project simulates opinion dynamics and debates within a network of social agents. Unlike traditional agent-based models (ABMs) that use simplified mathematical transition rules, this simulator leverages **Large Language Models (LLMs)** to model cognitive agents who discuss topics, exchange arguments, and dynamically update their beliefs based on conversational interactions.

---

## Core Concept: The Ship of Theseus

The primary experiment in this repository investigates the classic philosophical thought experiment: **The Ship of Theseus**.
Agents are placed in a network where they discuss the statement:

> *"If every part of a ship is replaced over time, it remains the same ship."*

Agents hold initial beliefs ranging from `0` ("strongly disagree") to `6` ("fully agree"). During the simulation, agents are paired to debate this statement:
1. One agent acts as the **Opponent** and presents arguments in favor of their own opinion.
2. The other acts as the **Discussant**, listens to the opponent's argument, and decides whether to **Accept**, **Reject**, or **Ignore** the argument, shifting their belief accordingly.

---

## System Architecture

The simulation is built modularly with three main components:

```
  +------------------+      Loads Graph      +--------------------+
  |  Network Loader  | --------------------> |  Network of Agents |
  +------------------+                       +--------------------+
                                                        |
                                                        | Running debates
                                                        v
  +------------------+   Updates Statuses    +--------------------+
  |    Simulator     | <-------------------- |      Monitors      |
  +------------------+                       +--------------------+
```

*   **Agents**: Individual profiles representing human-like agents with a specific opinion, profile description, and preferred LLM backend.
*   **Network**: A representation of the social network topology (loaded from GraphML or Edge List format) defining who can converse with whom.
*   **Monitors**: Classes governing the interaction rules. They select who interacts and orchestrate the debate flow (e.g., `MonitorOpinionDistribution`, `MonitorBoundedConfidence`).
*   **Simulator**: The orchestration engine that runs multiple iterations of interactions and logs the history to structured JSONL files.

---

## Acceleration via vLLM

Running LLM simulations sequentially is extremely slow. For a network of 100 agents over 100 iterations, a sequential run executes **10,000 multi-turn debates**, taking up to **22 hours** on local hardware. 

To overcome this, this repository includes a **batched vLLM interaction engine** allowing you to execute all agent interactions within a simulation step concurrently, reducing the runtime to just **a few minutes**.
