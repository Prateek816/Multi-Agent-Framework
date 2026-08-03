from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Env var resolution — ${VAR_NAME} anywhere in the raw YAML dict, resolved
# BEFORE Pydantic validation runs. This is why api_key: ${GEMINI_API_KEY}
# in the example config just works with no special field type needed.
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            resolved = os.environ.get(var_name)
            if resolved is None:
                raise ValueError(
                    f"Config references environment variable '${{{var_name}}}' "
                    f"but it is not set."
                )
            return resolved

        return _ENV_VAR_PATTERN.sub(replace, value)

    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]

    return value


# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: str            # "gemini" | "openai" | "groq" — LLMRegistry maps this to a client class
    model: str
    api_key: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)   # temperature, max_tokens, etc., provider-specific


# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------

class MCPServerConfig(BaseModel):
    name: str
    url: str


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """
    One entry under `agents:`. Note `agents:` in YAML is a MAPPING
    (key = agent name), not a list — `name` gets filled in by Config's
    validator from the dict key, so it never has to be typed twice.
    """

    name: str = ""              # filled in by Config._inject_agent_names, not set directly in YAML
    llm: str                    # key into `llms:`
    tools: list[str] = Field(default_factory=list)

    prompt: Optional[str] = None
    prompt_file: Optional[str] = None

    memory: Optional[str] = None

    skills_directory: Optional[str] = None
    knowledge_directory: Optional[str] = None

    peers: list[str] = Field(default_factory=list)   # which OTHER agent names this one may call via A2A
    streaming_on: bool = False
    expose_a2a: bool = False    # reachable from OUTSIDE the process; triggers AgentExecutor wrapping

    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_prompt_source(self) -> "AgentConfig":
        if self.prompt and self.prompt_file:
            raise ValueError(
                f"Agent '{self.name}': set only one of `prompt` or `prompt_file`, not both."
            )
        return self

    def resolve_prompt(self) -> Optional[str]:
        if self.prompt is not None:
            return self.prompt
        if self.prompt_file is not None:
            path = Path(self.prompt_file)
            if not path.exists():
                raise FileNotFoundError(
                    f"Agent '{self.name}': prompt_file '{self.prompt_file}' does not exist."
                )
            return path.read_text(encoding="utf-8")
        return None

# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    llms: dict[str, LLMConfig]
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    agents: dict[str, AgentConfig]

    @model_validator(mode="after")
    def _inject_agent_names(self) -> "Config":
        for name, agent_cfg in self.agents.items():
            agent_cfg.name = name
        return self

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "Config":
        # Fail fast at load time, not deep inside a run — same principle
        # as tool-name validation: catch typos before Framework even starts.
        agent_names = set(self.agents)

        """add external cross references validation for agents and workflows"""
        for agent_name, agent_cfg in self.agents.items():
            if agent_cfg.llm not in self.llms:
                raise ValueError(
                    f"Agent '{agent_name}' references unknown llm '{agent_cfg.llm}'. "
                    f"Known llms: {sorted(self.llms)}"
                )
            for peer in agent_cfg.peers:
                if peer not in agent_names:
                    raise ValueError(
                        f"Agent '{agent_name}' lists unknown peer '{peer}'. "
                        f"Known agents: {sorted(agent_names)}"
                    )
                if peer == agent_name:
                    raise ValueError(f"Agent '{agent_name}' cannot list itself as a peer.")

        return self

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """
    Single entrypoint for turning a YAML file into a validated Config object.
    Everything downstream (LLMRegistry, MCPManager, AgentFactory, Orchestrator, ...)
    consumes a Config instance — none of them read YAML or touch env vars directly.
    """

    @staticmethod
    def load(path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Config file '{path}' did not parse to a mapping at the top level.")

        resolved = _resolve_env_vars(raw)

        try:
            return Config.model_validate(resolved)
        except Exception as e:
            raise ValueError(f"Invalid config in '{path}':\n{e}") from e