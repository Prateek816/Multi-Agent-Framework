"""
Standalone MCP server exposing filesystem tools.

Run directly:

    # stdio transport (default)
    python -m framework.core.tools.builtin.filesystem_server

    # streamable-http transport
    python -m framework.core.tools.builtin.filesystem_server --transport http --host 0.0.0.0 --port 8001

Or import the app directly:

    from framework.core.tools.builtin.filesystem_server import mcp as filesystem_mcp
"""

from __future__ import annotations

import argparse

from framework.mcp.builtin.filesystem import mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the filesystem MCP tool server standalone."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help=(
            "Transport to serve over. 'stdio' (default) is for local "
            "subprocess use by an MCP client. 'http'/'sse' start a "
            "network-reachable server."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind when using the http/sse transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind when using the http/sse transport (default: 8001).",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()