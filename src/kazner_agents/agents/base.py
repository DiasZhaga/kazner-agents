"""Agent specification and the common agent interface."""

from __future__ import annotations

from dataclasses import dataclass

from kazner_agents.context import RunContext
from kazner_agents.messages import Message


@dataclass(frozen=True)
class AgentSpec:
    """One row of the agent table (DESIGN.md section 4); docs/agents.md is generated from it."""

    name: str
    role: str
    inputs: str
    outputs: str
    tools: tuple[str, ...]
    done_when: str
    stage: str  # "A2" = implemented now, "A4" = planned for Assignment 4


class Agent:
    """An agent receives one payload and returns its output payloads.

    Agents never call each other: they return payloads, and the orchestrator decides where each
    payload goes (hub-and-spoke).
    """

    spec: AgentSpec

    @property
    def name(self) -> str:
        return self.spec.name

    def handle(self, payload: Message, ctx: RunContext) -> list[Message]:
        raise NotImplementedError
