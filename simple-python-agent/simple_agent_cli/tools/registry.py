from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from .definitions import MAX_READ_LINES


ToolResult = dict[str, object]
ToolHandler = Callable[[dict[str, object]], ToolResult]

MAX_LIST_RESULTS = 100
MAX_SEARCH_RESULTS = 100


def get_local_time(_arguments: dict[str, object]) -> ToolResult:
    now = datetime.now().astimezone()
    return {
        "ok": True,
        "result": {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": now.tzname() or "",
            "utc_offset": now.strftime("%z"),
            "unix_timestamp": int(now.timestamp()),
        },
    }


def read_file(arguments: dict[str, object]) -> ToolResult:
    path = Path(str(arguments.get("path") or "")).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")

    offset = int(arguments.get("offset") or 1)
    limit = int(arguments.get("limit") or MAX_READ_LINES)
    if offset < 1 or not 1 <= limit <= MAX_READ_LINES:
        raise ValueError("offset must be positive and limit must be between 1 and 200.")

    lines = path.read_text(encoding="utf-8").splitlines()
    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = [f"{number}: {line}" for number, line in enumerate(selected, offset)]
    return {
        "ok": True,
        "result": {
            "path": str(path),
            "content": "\n".join(numbered),
            "start_line": offset,
            "returned_lines": len(selected),
            "total_lines": len(lines),
            "truncated": offset - 1 + len(selected) < len(lines),
        },
    }


def _glob_root(pattern: str) -> tuple[Path, str]:
    path_pattern = Path(pattern)
    anchor = path_pattern.anchor
    root = Path(anchor) if anchor else Path.cwd().resolve()
    relative_pattern = pattern[len(anchor) :] if anchor else pattern
    return root, relative_pattern


def list_files(arguments: dict[str, object]) -> ToolResult:
    root, pattern = _glob_root(str(arguments.get("pattern") or "**/*"))
    matches = sorted(
        str(path.resolve())
        for path in root.glob(pattern)
        if path.is_file()
    )
    return {
        "ok": True,
        "result": {
            "files": matches[:MAX_LIST_RESULTS],
            "count": min(len(matches), MAX_LIST_RESULTS),
            "truncated": len(matches) > MAX_LIST_RESULTS,
        },
    }


def search_text(arguments: dict[str, object]) -> ToolResult:
    raw_pattern = str(arguments.get("pattern") or "")
    if not raw_pattern:
        raise ValueError("pattern is required.")

    flags = 0 if arguments.get("case_sensitive", True) else re.IGNORECASE
    try:
        regex = re.compile(raw_pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc

    root, glob_pattern = _glob_root(str(arguments.get("glob") or "**/*"))
    matches: list[dict[str, object]] = []
    for path in root.glob(glob_pattern):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(
                    {
                        "path": str(path.resolve()),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return {"ok": True, "result": {"matches": matches, "truncated": True}}

    return {"ok": True, "result": {"matches": matches, "truncated": False}}


def write_file(arguments: dict[str, object]) -> ToolResult:
    raw_path = str(arguments.get("path") or "")
    if not raw_path:
        raise ValueError("path is required.")
    if "content" not in arguments or not isinstance(arguments["content"], str):
        raise ValueError("content must be a string.")

    path = Path(raw_path).expanduser().resolve()
    existed = path.exists()
    if existed and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    content = arguments["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return {
        "ok": True,
        "result": {
            "path": str(path),
            "operation": "updated" if existed else "created",
            "characters_written": len(content),
        },
    }


def edit_file(arguments: dict[str, object]) -> ToolResult:
    path = Path(str(arguments.get("path") or "")).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")

    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        raise ValueError("old_string and new_string must be strings.")
    if old_string == new_string:
        raise ValueError("old_string and new_string are identical.")
    if not old_string:
        raise ValueError("old_string must not be empty.")

    content = path.read_text(encoding="utf-8")
    match_count = content.count(old_string)
    if match_count == 0:
        raise ValueError("old_string was not found in the file.")

    replace_all = arguments.get("replace_all", False)
    if not isinstance(replace_all, bool):
        raise ValueError("replace_all must be a boolean.")
    if match_count > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {match_count} times; provide more context "
            "or set replace_all to true."
        )

    replacements = match_count if replace_all else 1
    updated = content.replace(old_string, new_string, -1 if replace_all else 1)
    path.write_text(updated, encoding="utf-8", newline="")
    return {
        "ok": True,
        "result": {
            "path": str(path),
            "replacements": replacements,
        },
    }


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_local_time": get_local_time,
    "read_file": read_file,
    "list_files": list_files,
    "search_text": search_text,
    "write_file": write_file,
    "edit_file": edit_file,
}
