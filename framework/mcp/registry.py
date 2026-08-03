"""
Connects to all configured MCP servers and builds a flat, name-keyed
registry of the tools they expose. Agents resolve their `tools:` list
against this registry at build time.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from framework.config_loader import MCPServerConfig


def _to_client_config(cfg: MCPServerConfig) -> dict:
    transport = "streamable_http" if cfg.transport == "http" else cfg.transport
    return {"transport": transport, "url": cfg.url}


class MCPToolRegistry:
    def __init__(self, mcp_servers: List[MCPServerConfig]):
        self._server_configs = mcp_servers
        self._client: Optional[MultiServerMCPClient] = None
        self._tools_by_name: Dict[str, BaseTool] = {}
        self._tool_source: Dict[str, str] = {}  # tool_name -> owning server name

    async def discover(self) -> None:
        """Connect to every configured MCP server and index their tools by name."""
        if not self._server_configs:
            return

        servers = {cfg.name: _to_client_config(cfg) for cfg in self._server_configs}
        self._client = MultiServerMCPClient(servers)

        all_tools = await self._client.get_tools()

        for tool in all_tools:
            server_name = None
            metadata = getattr(tool, "metadata", None) or {}
            server_name = metadata.get("server_name", "unknown")

            if tool.name in self._tools_by_name:
                existing_server = self._tool_source.get(tool.name, "unknown")
                raise ValueError(
                    f"Tool name collision: '{tool.name}' is exposed by both "
                    f"'{existing_server}' and '{server_name}'. Rename the tool "
                    f"on one of the MCP servers, or add namespacing."
                )

            self._tools_by_name[tool.name] = tool
            self._tool_source[tool.name] = server_name

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools_by_name.get(name)

    def all_names(self) -> List[str]:
        return list(self._tools_by_name.keys())