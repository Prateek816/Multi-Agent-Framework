"""
Filesystem management tools for LangChain agents.
Provides basic file and directory operations as LangChain tools.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from langchain_core.tools import Tool
from typing import Dict, Any


def read_file(input_str: str) -> str:
    """
    Read the contents of a file.
    Input: JSON string with key "file_path"
    Output: File contents as string or error message
    """
    try:
        data = json.loads(input_str)
        file_path = data.get("file_path")
        if not file_path:
            return "Error: Missing 'file_path' in input"

        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"file_path\": \"path/to/file\"}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(input_str: str) -> str:
    """
    Write content to a file. Overwrites the file if it already exists.
    Input: JSON string with keys "file_path" and "content"
    Output: Success message or error
    """
    try:
        data = json.loads(input_str)
        file_path = data.get("file_path")
        content = data.get("content")

        if file_path is None:
            return "Error: Missing 'file_path' in input"
        if content is None:
            return "Error: Missing 'content' in input"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"file_path\": \"path/to/file\", \"content\": \"content\"}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def append_file(input_str: str) -> str:
    """
    Append content to a file. Creates the file (and parent directories) if it
    doesn't already exist, otherwise adds to the end of the existing content.
    Input: JSON string with keys "file_path" and "content"
    Output: Success message or error
    """
    try:
        data = json.loads(input_str)
        file_path = data.get("file_path")
        content = data.get("content")

        if file_path is None:
            return "Error: Missing 'file_path' in input"
        if content is None:
            return "Error: Missing 'content' in input"

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully appended to {file_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"file_path\": \"path/to/file\", \"content\": \"content\"}"
    except Exception as e:
        return f"Error appending to file: {str(e)}"


def list_directory(input_str: str) -> str:
    """
    List contents of a directory.
    Input: JSON string with key "directory_path"
    Output: JSON array of filenames or error message
    """
    try:
        data = json.loads(input_str)
        directory_path = data.get("directory_path")
        if not directory_path:
            return "Error: Missing 'directory_path' in input"

        if not os.path.exists(directory_path):
            return f"Error: Directory not found: {directory_path}"

        if not os.path.isdir(directory_path):
            return f"Error: Path is not a directory: {directory_path}"

        items = os.listdir(directory_path)
        return json.dumps(items)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"directory_path\": \"path/to/dir\"}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def create_directory(input_str: str) -> str:
    """
    Create a directory.
    Input: JSON string with key "directory_path"
    Output: Success message or error
    """
    try:
        data = json.loads(input_str)
        directory_path = data.get("directory_path")
        if not directory_path:
            return "Error: Missing 'directory_path' in input"

        os.makedirs(directory_path, exist_ok=True)
        return f"Successfully created directory: {directory_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"directory_path\": \"path/to/dir\"}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"


def delete_file(input_str: str) -> str:
    """
    Delete a file.
    Input: JSON string with key "file_path"
    Output: Success message or error
    """
    try:
        data = json.loads(input_str)
        file_path = data.get("file_path")
        if not file_path:
            return "Error: Missing 'file_path' in input"

        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        if os.path.isdir(file_path):
            return f"Error: Path is a directory, use delete_directory instead: {file_path}"

        os.remove(file_path)
        return f"Successfully deleted file: {file_path}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"file_path\": \"path/to/file\"}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"


def delete_directory(input_str: str) -> str:
    """
    Delete a directory.
    Input: JSON string with key "directory_path" and optional "recursive" (bool, default False)
    Output: Success message or error

    By default this only removes empty directories, to avoid an agent
    accidentally wiping out a directory tree. Pass "recursive": true to
    delete a directory and everything inside it.
    """
    try:
        data = json.loads(input_str)
        directory_path = data.get("directory_path")
        recursive = data.get("recursive", False)

        if not directory_path:
            return "Error: Missing 'directory_path' in input"

        if not os.path.exists(directory_path):
            return f"Error: Directory not found: {directory_path}"

        if not os.path.isdir(directory_path):
            return f"Error: Path is not a directory, use delete_file instead: {directory_path}"

        if recursive:
            shutil.rmtree(directory_path)
            return f"Successfully deleted directory (recursive): {directory_path}"
        else:
            try:
                os.rmdir(directory_path)
                return f"Successfully deleted directory: {directory_path}"
            except OSError:
                return (
                    f"Error: Directory not empty: {directory_path}. "
                    "Pass \"recursive\": true to delete it along with its contents."
                )
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"directory_path\": \"path/to/dir\", \"recursive\": false}"
    except Exception as e:
        return f"Error deleting directory: {str(e)}"


def copy_file(input_str: str) -> str:
    """
    Copy a file or directory to a new location, leaving the original in place.
    Input: JSON string with keys "source" and "destination"
    Output: Success message or error
    """
    try:
        data = json.loads(input_str)
        source = data.get("source")
        destination = data.get("destination")

        if source is None:
            return "Error: Missing 'source' in input"
        if destination is None:
            return "Error: Missing 'destination' in input"

        if not os.path.exists(source):
            return f"Error: Source not found: {source}"

        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        return f"Successfully copied {source} to {destination}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"source\": \"path/to/source\", \"destination\": \"path/to/dest\"}"
    except Exception as e:
        return f"Error copying file: {str(e)}"


def move_file(input_str: str) -> str:
    """
    Move or rename a file or directory.
    Input: JSON string with keys "source" and "destination"
    Output: Success message or error
    """
    try:
        data = json.loads(input_str)
        source = data.get("source")
        destination = data.get("destination")

        if source is None:
            return "Error: Missing 'source' in input"
        if destination is None:
            return "Error: Missing 'destination' in input"

        if not os.path.exists(source):
            return f"Error: Source not found: {source}"

        # Create destination directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        shutil.move(source, destination)
        return f"Successfully moved {source} to {destination}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"source\": \"path/to/source\", \"destination\": \"path/to/dest\"}"
    except Exception as e:
        return f"Error moving file: {str(e)}"


def get_file_info(input_str: str) -> str:
    """
    Get metadata about a file or directory: type, size, and last-modified time.
    Input: JSON string with key "path"
    Output: JSON object with info, or error message
    """
    try:
        data = json.loads(input_str)
        path = data.get("path")
        if not path:
            return "Error: Missing 'path' in input"

        if not os.path.exists(path):
            return f"Error: Path not found: {path}"

        stat = os.stat(path)
        info = {
            "path": path,
            "is_directory": os.path.isdir(path),
            "is_file": os.path.isfile(path),
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        return json.dumps(info)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Expected format: {\"path\": \"path/to/file_or_dir\"}"
    except Exception as e:
        return f"Error getting file info: {str(e)}"


# Create LangChain tools
read_file_tool = Tool(
    name="read_file",
    func=read_file,
    description="Read the contents of a file. Input: JSON string with 'file_path' key. Output: File contents or error message."
)

write_file_tool = Tool(
    name="write_file",
    func=write_file,
    description="Write content to a file, overwriting it if it exists. Input: JSON string with 'file_path' and 'content' keys. Output: Success message or error."
)

append_file_tool = Tool(
    name="append_file",
    func=append_file,
    description="Append content to the end of a file, creating it if it doesn't exist. Input: JSON string with 'file_path' and 'content' keys. Output: Success message or error."
)

list_directory_tool = Tool(
    name="list_directory",
    func=list_directory,
    description="List contents of a directory. Input: JSON string with 'directory_path' key. Output: JSON array of filenames or error message."
)

create_directory_tool = Tool(
    name="create_directory",
    func=create_directory,
    description="Create a directory. Input: JSON string with 'directory_path' key. Output: Success message or error."
)

delete_file_tool = Tool(
    name="delete_file",
    func=delete_file,
    description="Delete a file. Input: JSON string with 'file_path' key. Output: Success message or error."
)

delete_directory_tool = Tool(
    name="delete_directory",
    func=delete_directory,
    description="Delete a directory. Input: JSON string with 'directory_path' key and optional 'recursive' bool (default false; required to delete non-empty directories). Output: Success message or error."
)

copy_file_tool = Tool(
    name="copy_file",
    func=copy_file,
    description="Copy a file or directory to a new location, keeping the original. Input: JSON string with 'source' and 'destination' keys. Output: Success message or error."
)

move_file_tool = Tool(
    name="move_file",
    func=move_file,
    description="Move or rename a file or directory. Input: JSON string with 'source' and 'destination' keys. Output: Success message or error."
)

get_file_info_tool = Tool(
    name="get_file_info",
    func=get_file_info,
    description="Get metadata about a file or directory (type, size in bytes, last modified time). Input: JSON string with 'path' key. Output: JSON object with info or error message."
)

# Export all tools
__all__ = [
    "read_file_tool",
    "write_file_tool",
    "append_file_tool",
    "list_directory_tool",
    "create_directory_tool",
    "delete_file_tool",
    "delete_directory_tool",
    "copy_file_tool",
    "move_file_tool",
    "get_file_info_tool",
]