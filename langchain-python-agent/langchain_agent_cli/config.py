from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    base_url: str
    model: str | None
    timeout_seconds: float = 60.0
    max_tool_turns: int = 10

    @classmethod
    def from_env(cls) -> "AgentConfig":
        # Keep the same fixed local configuration as the original project.
        return cls(
            api_key="sk-24b86a67cd48417092412b5a1da99252",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            timeout_seconds=60.0,
            max_tool_turns=10,
        )
