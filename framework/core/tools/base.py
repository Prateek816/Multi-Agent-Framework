"""
framework/tools/base.py

Built-in tools and MCP-discovered tools need a single, uniform shape.
Without this, `ToolRegistry` would have to special-case "is this a local
Python function or a remote MCP tool call" everywhere it's used -- every
agent, every registry lookup.

This is a leaf module: everything else in the framework depends on it, it
depends on nothing internal.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# A tool's underlying implementation may be a plain sync function or an
# async function -- callers never need to know which (see Tool.call below).
ToolFunc = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass
class Tool:
    """Uniform representation of a callable tool, regardless of origin."""

    name: str
    description: str
    func: ToolFunc
    input_schema: dict[str, Any] = field(default_factory=dict)
    # Purely for debugging/logging, never affects behavior. Conventionally
    # "builtin:core" for framework-shipped tools or "mcp:<server>" for
    # tools discovered from a real MCP server.
    source: str = "builtin"

    async def call(self, **kwargs: Any) -> Any:
        """
        Invoke the underlying function and await it if needed.

        This is the one place that knows how to handle both sync and async
        tool implementations, so every caller (agents, registry, adapters)
        can just `await tool.call(**kwargs)` without caring which kind of
        function backs the tool.
        """
        result = self.func(**kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result