"""
Builtin filesystem MCP server.

Exposes file and directory operations (read, write, append, list, create,
delete, copy, move, stat) as MCP tools, built with FastMCP.

This is a straight MCP port of the original LangChain `Tool` wrappers: same
operations, same semantics, but with typed parameters instead of a JSON
string blob (so MCP clients get a real, validated input schema per tool)
and MCP-native error signaling via `ToolError` instead of `"Error: ..."`
strings (so clients see a proper `isError` result instead of having to
parse text).

Run standalone (stdio transport, e.g. for Claude Desktop / Claude Code):

    python -m framework.core.tools.builtin.filesystem_tools

Mount into a larger app:

    from framework.core.tools.builtin.filesystem_tools import mcp as filesystem_mcp
    main_mcp.mount(filesystem_mcp, prefix="fs")

Optional sandboxing:

    Set the FS_TOOL_ROOT environment variable to an absolute directory path
    to confine every operation to that directory tree. Any path that
    resolves outside of it is rejected. Leave it unset for unrestricted
    access (matches the original tools' behavior). Sandboxing is strongly
    recommended if this server is ever exposed to an untrusted model or
    a multi-tenant application.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP(
    name="filesystem-tools",
    instructions=(
        "Tools for reading, writing, and managing files and directories on "
        "the local filesystem. Paths may be relative (resolved against the "
        "server process's working directory) or absolute."
    ),
)

# If set, every resolved path must live inside this directory tree.
_ROOT = os.environ.get("FS_TOOL_ROOT")


def _resolve(path: str) -> str:
    """Resolve `path` to an absolute path, enforcing FS_TOOL_ROOT if configured.

    When FS_TOOL_ROOT is set, relative paths are resolved against that root
    (not the process's current working directory), and any path -- relative
    or absolute -- that would resolve outside the root is rejected. This
    keeps the sandbox usable (you don't have to chdir into the root first)
    while still blocking traversal like "../../etc/passwd".
    """
    if _ROOT:
        root = os.path.abspath(_ROOT)
        abs_path = (
            os.path.abspath(path)
            if os.path.isabs(path)
            else os.path.abspath(os.path.join(root, path))
        )
        if os.path.commonpath([abs_path, root]) != root:
            raise ToolError(
                f"Path '{path}' resolves outside the allowed root '{root}'"
            )
        return abs_path
    return os.path.abspath(path)


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read and return the full text contents of a file."""
    path = _resolve(file_path)
    if not os.path.exists(path):
        raise ToolError(f"File not found: {file_path}")
    if os.path.isdir(path):
        raise ToolError(f"Path is a directory, not a file: {file_path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise ToolError(f"Error reading file '{file_path}': {e}") from e


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """Write `content` to a file, overwriting it if it already exists.
    Parent directories are created automatically."""
    path = _resolve(file_path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        raise ToolError(f"Error writing file '{file_path}': {e}") from e


@mcp.tool()
def append_file(file_path: str, content: str) -> str:
    """Append `content` to the end of a file, creating the file (and any
    missing parent directories) if it doesn't already exist."""
    path = _resolve(file_path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully appended {len(content)} characters to {file_path}"
    except Exception as e:
        raise ToolError(f"Error appending to file '{file_path}': {e}") from e


@mcp.tool()
def list_directory(directory_path: str) -> list[str]:
    """List the names of files and subdirectories directly inside `directory_path`."""
    path = _resolve(directory_path)
    if not os.path.exists(path):
        raise ToolError(f"Directory not found: {directory_path}")
    if not os.path.isdir(path):
        raise ToolError(f"Path is not a directory: {directory_path}")
    try:
        return sorted(os.listdir(path))
    except Exception as e:
        raise ToolError(f"Error listing directory '{directory_path}': {e}") from e


@mcp.tool()
def create_directory(directory_path: str) -> str:
    """Create a directory, including any missing parent directories."""
    path = _resolve(directory_path)
    try:
        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory: {directory_path}"
    except Exception as e:
        raise ToolError(f"Error creating directory '{directory_path}': {e}") from e


@mcp.tool()
def delete_file(file_path: str) -> str:
    """Delete a single file."""
    path = _resolve(file_path)
    if not os.path.exists(path):
        raise ToolError(f"File not found: {file_path}")
    if os.path.isdir(path):
        raise ToolError(
            f"Path is a directory, use delete_directory instead: {file_path}"
        )
    try:
        os.remove(path)
        return f"Successfully deleted file: {file_path}"
    except Exception as e:
        raise ToolError(f"Error deleting file '{file_path}': {e}") from e


@mcp.tool()
def delete_directory(directory_path: str, recursive: bool = False) -> str:
    """Delete a directory. By default only an empty directory can be
    removed; pass recursive=true to delete a directory and everything
    inside it."""
    path = _resolve(directory_path)
    if not os.path.exists(path):
        raise ToolError(f"Directory not found: {directory_path}")
    if not os.path.isdir(path):
        raise ToolError(
            f"Path is not a directory, use delete_file instead: {directory_path}"
        )

    if recursive:
        try:
            shutil.rmtree(path)
            return f"Successfully deleted directory (recursive): {directory_path}"
        except Exception as e:
            raise ToolError(f"Error deleting directory '{directory_path}': {e}") from e

    try:
        os.rmdir(path)
        return f"Successfully deleted directory: {directory_path}"
    except OSError:
        raise ToolError(
            f"Directory not empty: {directory_path}. "
            "Pass recursive=true to delete it along with its contents."
        )


@mcp.tool()
def copy_file(source: str, destination: str) -> str:
    """Copy a file or directory to a new location, leaving the original in place."""
    src = _resolve(source)
    dst = _resolve(destination)
    if not os.path.exists(src):
        raise ToolError(f"Source not found: {source}")
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return f"Successfully copied {source} to {destination}"
    except Exception as e:
        raise ToolError(f"Error copying '{source}' to '{destination}': {e}") from e


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory."""
    src = _resolve(source)
    dst = _resolve(destination)
    if not os.path.exists(src):
        raise ToolError(f"Source not found: {source}")
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return f"Successfully moved {source} to {destination}"
    except Exception as e:
        raise ToolError(f"Error moving '{source}' to '{destination}': {e}") from e


@mcp.tool()
def get_file_info(path: str) -> dict:
    """Get metadata about a file or directory: whether it's a file or
    directory, its size in bytes, and its last-modified time (UTC, ISO 8601)."""
    abs_path = _resolve(path)
    if not os.path.exists(abs_path):
        raise ToolError(f"Path not found: {path}")
    try:
        stat = os.stat(abs_path)
        return {
            "path": path,
            "is_directory": os.path.isdir(abs_path),
            "is_file": os.path.isfile(abs_path),
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error getting file info for '{path}': {e}") from e


if __name__ == "__main__":
    mcp.run()