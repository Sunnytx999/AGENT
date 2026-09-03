from __future__ import annotations

from ..model.types import ToolDefinition


MAX_READ_LINES = 200

LOCAL_TOOLS: list[ToolDefinition] = [
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "Get the current date and time from the local computer.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the local computer. "
                "Use offset and limit to read large files in sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to the current directory.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "First line to return, starting at 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_READ_LINES,
                        "description": f"Maximum lines to return (up to {MAX_READ_LINES}).",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List local files using an absolute or relative glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": 'Glob pattern such as "*.py" or "**/*.py".',
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search local UTF-8 text files with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regular expression.",
                    },
                    "glob": {
                        "type": "string",
                        "description": 'File glob filter, such as "**/*.py".',
                    },
                    "case_sensitive": {"type": "boolean"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or completely overwrite a UTF-8 text file. "
                "Parent directories are created automatically. Prefer edit_file "
                "when changing only part of an existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to the current directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 text content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a UTF-8 text file by exact string replacement. "
                "The old text must occur exactly once unless replace_all is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to the current directory.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact existing text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring one unique match.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
]
