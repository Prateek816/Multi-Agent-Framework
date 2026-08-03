"""
framework/agents/factory.py

Something has to turn a validated `AgentConfig` into a live, constructed
`BaseAgent` -- but as of `base.py`, that resolution work (llm client,
memory manager, RAG, skills) happens *inside* `BaseAgent.__init__`,
driven entirely off the fields on `AgentConfig` itself. `AgentFactory`
no longer owns any of that resolution; the only thing it still supplies
that `AgentConfig` doesn't already carry is which `BaseAgent` subclass to
instantiate for a given `agent_config.type`.

`AgentFactory` deliberately does NOT register agents anywhere. `build()`
and `build_all()` return constructed agents; they don't add them to
`AgentRegistry`. That's the caller's job, so this module has no side
effects on global state -- agent construction and agent registration stay
two separate concerns.
"""

from typing import Protocol

from framework.agent.base import BaseAgent
from config_loader import AgentConfig


class PluginLoader(Protocol):
    """Typing contract only. Resolves a config-level agent type to an agent class."""

    def resolve(self, agent_type: str) -> type[BaseAgent]:
        """Return the `BaseAgent` subclass registered under `agent_type`."""
        ...


class AgentFactory:
    """
    Turns validated `AgentConfig` objects into live `BaseAgent` instances.

    `BaseAgent.__init__` builds its own llm client, memory manager, RAG,
    and skill registry straight from the `AgentConfig` it's handed, so the
    only thing this factory still resolves externally is the agent
    *class* itself, via `plugin_loader`. That's the one dependency it
    holds.
    """

    def __init__(
        self,
        plugin_loader: PluginLoader,
    ) -> None:
        self.plugin_loader = plugin_loader

    def build(self, agent_config: AgentConfig) -> BaseAgent:
        """
        Look up the agent class registered for `agent_config.type` via the
        plugin loader, then construct it through `from_config` (not
        `__init__` directly -- see base.py for why: it lets a custom agent
        subclass override how it pulls extra fields out of
        `agent_config.extra` without every other agent needing to
        implement construction logic itself). `from_config` takes only the
        config; `BaseAgent.__init__` handles the rest of the resolution
        (llm, memory, RAG, skills) from `agent_config`'s own fields.
        """
        agent_cls = self.plugin_loader.resolve(agent_config.type)
        return agent_cls.from_config(agent_config)

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