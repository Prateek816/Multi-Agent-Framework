"""
framework/agents/base.py

Framework-native base class for agents.

This replaces an earlier draft where `BaseAgent` inherited directly from the
A2A SDK's `AgentExecutor`. That forced every agent -- even ones that never
leave the process -- to carry A2A transport machinery (`RequestContext`,
`EventQueue`) just to be constructed or called internally, and gave `run()`
only a bare `str` return, with no room for tool-call metadata, state
updates, or usage stats the Orchestrator needs.

This module is intentionally a leaf: it has no A2A dependency and no
internal framework imports, so nothing forces an A2A SDK import
transitively through it. A future `A2AExecutorAdapter` is meant to wrap a
`BaseAgent` instance from the outside, translating A2A's types at the
boundary -- this file stays ignorant of that entirely.

`run()` delegates the model <-> tool loop to `create_agent`, which builds
a prebuilt (LangGraph-style) tool-calling agent. That means this class no
longer hand-rolls its own invoke/dispatch loop -- there is nothing here
equivalent to a `_chat_impl` / `_run_tool_loop` pair anymore.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

import logging

from core.llm import get_llm, LLMConfig
from core.memory.manager import MemoryManager
from core.RAG.rag import KnowledgeRAG
from core.skills import SkillRegistry

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from config_loader import AgentConfig
from langchain_agent.agent import create_agent

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Framework-native replacement for A2A's `RequestContext`."""

    input: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Shared workflow / prior-step state visible to this task.
    context: Dict[str, Any] = field(default_factory=dict)
    # Free-form metadata: run_id, calling agent, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Framework-native replacement for a bare `str` return from `run()`."""

    output: Any
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class every framework agent extends.

    No A2A dependency of any kind lives here. `AgentFactory` injects
    everything an agent needs through `__init__` (or, more commonly,
    through `from_config`).
    """

    def __init__(
        self,
        config:AgentConfig,
    ) -> None:
        self.agent_name = config.name
        self.llm = get_llm(LLMConfig(config.llm))
        self.memory = MemoryManager(MemoryConfig = config.memory)
        self.RAG = KnowledgeRAG(knowledge_dir = config.knowledge_directory)
        self.skills = SkillRegistry(skills_dirs= config.skills_directory)

    @classmethod
    def from_config(
        cls,
        config:AgentConfig
    ) -> "BaseAgent":
        """
        Construction path used by `AgentFactory`.
        """
        return cls(
            config = config
        )

    def run(self, task: Task) -> AgentResult:
        """Execute the agent against a task and return a structured result.

        `create_agent` owns the model <-> tool loop (binding tools,
        invoking the LLM, dispatching tool calls, feeding results back)
        internally, so this method's job is just: assemble the pieces
        (system prompt, tools, messages), hand them to the agent, and
        translate its output into an `AgentResult`.
        """
        if not getattr(self, "_system_prompt", None):
            self._init_system_prompt()

        tools = self._build_tools()

        agent = create_agent(
            llm=self.llm,
            tools=tools,
            system_prompt=self._system_prompt,
        )

        messages = self._build_messages(task)

        try:
            result = agent.invoke({"messages": messages})
        except Exception as exc:
            logger.exception(
                "Agent '%s' failed on task %s", self.agent_name, task.id
            )
            return AgentResult(output=None, error=str(exc))

        out_messages: List[BaseMessage] = result.get("messages", [])
        final = out_messages[-1] if out_messages else None
        output = (final.content if final else "") or ""

        return AgentResult(
            output=output,
            tool_calls=self._extract_tool_calls(out_messages),
            usage=self._extract_usage(out_messages),
        )

    def _init_system_prompt(self) -> None:
        """Build the system prompt from indentity layers + skills + memory"""

        parts: list[str] = []
        skill_catalog = self.skills.build_catalog()

        if skill_catalog:
            parts.append(f"## Available Skills\n\n{skill_catalog}")
            logger.info(
                "Skills injected into prompt: %d chars", len(skill_catalog)
            )
        else:
            logger.info("No skills to inject (dirs=%s)", self.skills.skills_dirs)

        if self.RAG:
            parts.append(
                "You have access to a knowledge base. Use the retrieve_knowledge tool "
                "to search it when the user's question may be answered by stored documents."
            )

        boot_ctx = self.memory.boot_context(max_chars=3000)
        if boot_ctx:
            parts.append(f"## What You Remember\n\n{boot_ctx}")

        self._system_prompt = "\n\n---\n\n".join(parts)

        logger.debug("System prompt: %d chars", len(self._system_prompt))

    def _build_tools(self) -> list:
        """Build the LangChain tool list with runtime bindings."""
        from langchain_core.tools import StructuredTool
        
        tools = []
        
        # Memory tools — bound to this agent's memory manager
        memory_defs = [
            ("remember", "Store a fact in long-term memory."),
            ("recall", "Search long-term memory for relevant facts."),
            ("memory_get", "Read a specific memory file by path."),
            ("memory_list_files", "List all memory files."),
            ("forget", "Remove a memory entry by key."),
            ("update_index", "Update the memory INDEX.md file."),
        ]
        for name, desc in memory_defs:
            handler = self._make_memory_handler(name)
            tools.append(StructuredTool.from_function(
                func=handler, name=name, description=desc,
            ))

        # Skill tools — bound to this agent's skill registry
        skill_defs = [
            ("use_skill", "Load and use a skill by name."),
            ("list_skill_resources", "List resource files for a skill."),
        ]
        for name, desc in skill_defs:
            handler = self._make_skill_handler(name)
            tools.append(StructuredTool.from_function(
                func=handler, name=name, description=desc,
            ))

        # Knowledge retrieval tool
        if self.RAG:
            def retrieve_knowledge(query: str) -> str:
                """Search the knowledge base for relevant documents."""
                assert self.RAG is not None
                results = self.RAG.retrieve(query, top_k=5)
                if not results:
                    return "No relevant documents found."
                return "\n\n".join(
                    f"[{r.get('source', 'unknown')}]\n{r['content']}" for r in results
                )

            tools.append(StructuredTool.from_function(
                func=retrieve_knowledge, name="retrieve_knowledge",
                description="Search the knowledge base for relevant documents.",
            ))

        # MCP tools
        mcp_provider = self._get_mcp_provider()
        if mcp_provider:
            tools.extend(mcp_provider.build_tools())

        return tools

    def _make_memory_handler(self, tool_name: str):
        """Create a runtime handler for a memory tool."""
        def handle_remember(content: str, key: str = "") -> str:
            return self.memory.remember(content, key or None)

        def handle_recall(query: str = "*") -> str:
            return self.memory.recall(query)

        def handle_memory_get(path: str) -> str:
            return self.memory.memory_get(path)

        def handle_memory_list_files() -> str:
            files = self.memory.list_files()
            return "\n".join(files) if files else "No memory files."

        def handle_forget(key: str) -> str:
            return self.memory.forget(key)

        def handle_update_index(content: str) -> str:
            self.memory.write_index(content)
            return "INDEX.md updated."

        handlers = {
            "lc_remember": handle_remember,
            "lc_recall": handle_recall,
            "lc_memory_get": handle_memory_get,
            "lc_memory_list_files": handle_memory_list_files,
            "lc_forget": handle_forget,
            "lc_update_index": handle_update_index,
        }
        return handlers.get(tool_name, lambda **_: f"Unknown memory tool: {tool_name}")

    def _make_skill_handler(self, tool_name: str):
        """Create a runtime handler for a skill tool."""
        def handle_use_skill(skill_name: str) -> str:
            skill = self.skills.load_skill(skill_name)
            if not skill:
                return f"Skill '{skill_name}' not found."
            return f"## {skill.name}\n\n{skill.instructions}"

        def handle_list_skill_resources(skill_name: str) -> str:
            resources = self.skills.list_resources(skill_name)
            if not resources:
                return f"No resources found for '{skill_name}'."
            return "\n".join(resources)

        handlers = {
            "use_skill": handle_use_skill,
            "list_skill_resources": handle_list_skill_resources,
        }
        return handlers.get(tool_name, lambda **_: f"Unknown skill tool: {tool_name}")

    def _build_messages(self, task: Task) -> list[BaseMessage]:
        """Build the conversation messages for this task.

        The system prompt is *not* included here — it's passed to
        `create_agent` directly as `system_prompt`, since the agent owns
        prepending it on every internal loop iteration. History comes from
        `task.context`, not instance state, since a `BaseAgent` is meant to
        be reusable across tasks/runs rather than keep its own chat log.
        """
        msgs: list[BaseMessage] = []

        history = task.context.get("messages", [])
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))

        # Current input — may be a plain string or multimodal content list
        msgs.append(HumanMessage(content=task.input))

        return msgs

    @staticmethod
    def _extract_tool_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Pull every tool call issued during the agent's run out of its
        final message list, in order."""
        calls: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                calls.extend(m.tool_calls)
        return calls

    @staticmethod
    def _extract_usage(messages: List[BaseMessage]) -> Dict[str, Any]:
        """Sum per-message `usage_metadata` (if the provider populates it)
        across every model call the agent made during its run."""
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        seen = False
        for m in messages:
            meta = getattr(m, "usage_metadata", None)
            if not meta:
                continue
            seen = True
            for k in totals:
                totals[k] += meta.get(k, 0) or 0
        return totals if seen else {}

    def _get_mcp_provider(self):
        """Lazy-initialize and return the MCP tool provider, or None."""
        if hasattr(self, '_mcp_provider'):
            return self._mcp_provider

        mcp_enabled = bool(_cfg.get("mcp", "servers"))
        if not mcp_enabled:
            self._mcp_provider = None
            return None

        try:
            from core.mcp.integration import MCPToolProvider
            provider = MCPToolProvider()
            provider.initialize()
            self._mcp_provider = provider
            if self.verbose:
                logger.info(
                    "[Agent] MCP initialized with %d server(s)",
                    len(provider._connected_servers),
                )
            return provider
        except Exception as exc:
            logger.warning("[Agent] MCP initialization failed: %s", exc)
            self._mcp_provider = None
            return None