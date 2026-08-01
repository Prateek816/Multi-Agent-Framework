"""
framework/agents/factory.py

Something has to turn a validated `AgentConfig` into a live, constructed
`BaseAgent` -- resolving `llm: "gemini"` into an actual client, `tools:
[...]` into actual `Tool` objects, etc. Without a dedicated factory, that
resolution logic would leak into whichever code creates agents (likely the
Orchestrator), giving it a second, unrelated reason to change.

`AgentFactory` deliberately does NOT register agents anywhere. `build()`
and `build_all()` return constructed agents; they don't add them to
`AgentRegistry`. That's the caller's job, so this module has no side
effects on global state -- agent construction and agent registration stay
two separate concerns.

Known inconsistency: `AgentConfig` is imported below from
`framework.config.schema`, but as of this writing it actually lives in
`config_loader.py`. Left as-is per the source design doc; update the
import once `AgentConfig`'s real home is settled.
"""

from typing import Protocol

from framework.agent.base import BaseAgent
from framework.config.schema import AgentConfig


class LLMRegistry(Protocol):
    """Typing contract only. Resolves a config-level LLM name to a live client."""

    def get(self, name: str) -> object:
        """Return the constructed LLM client registered under `name`."""
        ...


class ToolRegistry(Protocol):
    """Typing contract only. Resolves config-level tool names to live Tool objects."""

    def resolve_many(self, names: list[str]) -> list[object]:
        """Return the `Tool` objects registered under each of `names`, in order."""
        ...


class MemoryManager(Protocol):
    """Typing contract only. Hands out a memory scope for a given agent."""

    def scope_for(self, agent_name: str) -> object:
        """Return the memory object/scope this agent should use."""
        ...


class PluginLoader(Protocol):
    """Typing contract only. Resolves a config-level agent type to an agent class."""

    def resolve(self, agent_type: str) -> type[BaseAgent]:
        """Return the `BaseAgent` subclass registered under `agent_type`."""
        ...


class AgentFactory:
    """
    Turns validated `AgentConfig` objects into live `BaseAgent` instances.

    Pure constructor injection: this class builds nothing itself at
    __init__ time, it just holds the four resolvers it needs at `build()`
    time.
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        plugin_loader: PluginLoader,
    ) -> None:
        self.llm_registry = llm_registry
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.plugin_loader = plugin_loader

    def build(self, agent_config: AgentConfig) -> BaseAgent:
        """
        Resolve `llm`, `tools`, and `memory` for a single agent config, look
        up the agent class via the plugin loader, and construct it through
        `from_config` (not `__init__` directly -- see base.py for why: it
        lets a custom agent subclass override how it pulls extra fields out
        of `agent_config.extra` without every other agent needing to
        implement construction logic itself).
        """
        llm = self.llm_registry.get(agent_config.llm)
        tools = self.tool_registry.resolve_many(agent_config.tools)
        memory = self.memory_manager.scope_for(agent_config.name)
        agent_cls = self.plugin_loader.resolve(agent_config.type)

        return agent_cls.from_config(
            agent_config,
            llm=llm,
            tools=tools,
            memory=memory,
        )

    def build_all(self, agent_configs: list[AgentConfig]) -> dict[str, BaseAgent]:
        """
        Startup convenience wrapper: build every config and return a
        name -> agent mapping. Raises on duplicate names, fail-fast, same
        as everywhere else in the framework. Does not register anything --
        the caller decides what to do with the returned agents.
        """
        built: dict[str, BaseAgent] = {}

        for agent_config in agent_configs:
            if agent_config.name in built:
                raise ValueError(
                    f"Duplicate agent name '{agent_config.name}' encountered "
                    f"while building agents"
                )
            built[agent_config.name] = self.build(agent_config)

        return built