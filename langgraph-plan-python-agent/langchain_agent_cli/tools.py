from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain.tools import tool

MAX_READ_LINES = 500
MAX_RESULTS = 200


def _glob_root(pattern: str) -> tuple[Path, str]:
    path_pattern = Path(pattern)
    anchor = path_pattern.anchor
    root = Path(anchor) if anchor else Path.cwd().resolve()
    return root, pattern[len(anchor):] if anchor else pattern


@tool(
    description="Get the current local date, time, timezone, UTC offset, and Unix timestamp."
)
def get_local_time() -> dict[str, object]:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": now.tzname() or "",
        "utc_offset": now.strftime("%z"),
        "unix_timestamp": int(now.timestamp()),
    }


@tool(
    description=f"Read numbered lines from a UTF-8 file. Paths may be absolute or relative. At most {MAX_READ_LINES} lines are returned per call; use offset and limit to page through longer files."
)
def read_file(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "error": f"File does not exist: {resolved}",
        }
    if offset < 1 or not 1 <= limit <= MAX_READ_LINES:
        return {
            "path": str(resolved),
            "error": f"offset must be positive and limit must be between 1 and {MAX_READ_LINES}",
        }
    lines = resolved.read_text(encoding="utf-8").splitlines()
    selected = lines[offset - 1:offset - 1 + limit]
    return {
        "path": str(resolved),
        "content": "\n".join(f"{number}: {line}" for number, line in enumerate(selected, offset)),
        "start_line": offset,
        "returned_lines": len(selected),
        "total_lines": len(lines),
        "truncated": offset - 1 + len(selected) < len(lines),
    }


@tool(
    description=f"List local files matching a glob such as *.py or **/*.py. Returns at most {MAX_RESULTS} files; use a narrower pattern when truncated is true."
)
def list_files(pattern: str = "**/*") -> dict[str, object]:
    root, relative_pattern = _glob_root(pattern)
    matches = sorted(str(path.resolve()) for path in root.glob(relative_pattern) if path.is_file())
    return {
        "files": matches[:MAX_RESULTS],
        "count": min(len(matches), MAX_RESULTS),
        "truncated": len(matches) > MAX_RESULTS,
    }


@tool(
    description=f"Search UTF-8 text files with a Python regular expression. Returns at most {MAX_RESULTS} matches; narrow glob or regex when truncated is true."
)
def search_text(pattern: str, glob: str = "**/*", case_sensitive: bool = True) -> dict[str, object]:
    if not pattern:
        raise ValueError("pattern is required")
    try:
        regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc
    root, relative_pattern = _glob_root(glob)
    matches: list[dict[str, object]] = []
    for path in root.glob(relative_pattern):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append({"path": str(path.resolve()), "line": line_number, "text": line[:500]})
                if len(matches) >= MAX_RESULTS:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


@tool(
    description="Create or completely overwrite a UTF-8 text file, creating parent directories when needed."
)
def write_file(path: str, content: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    existed = resolved.exists()
    if existed and not resolved.is_file():
        raise ValueError(f"Path is not a file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8", newline="")
    return {"path": str(resolved), "operation": "updated" if existed else "created", "characters_written": len(content)}


@tool(
    description="Edit a UTF-8 text file by exact string replacement; the match must be unique unless replace_all is true."
)
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"File does not exist: {resolved}")
    if not old_string or old_string == new_string:
        raise ValueError("old_string must be non-empty and different from new_string")
    content = resolved.read_text(encoding="utf-8")
    count = content.count(old_string)
    if count == 0:
        raise ValueError("old_string was not found in the file")
    if count > 1 and not replace_all:
        raise ValueError(f"old_string occurs {count} times; provide more context or set replace_all to true")
    replacements = count if replace_all else 1
    resolved.write_text(content.replace(old_string, new_string, -1 if replace_all else 1), encoding="utf-8", newline="")
    return {"path": str(resolved), "replacements": replacements}


TOOLS = [get_local_time, read_file, list_files, search_text, write_file, edit_file]

READ_ONLY_TOOLS = [get_local_time, read_file, list_files, search_text]


def make_main_file_tools(
    can_write: Callable[[str], bool],
    on_write: Callable[[str], None],
) -> list[object]:
    @tool(
        "write_file",
        description=(
            "Create or completely overwrite a UTF-8 text file. In Plan Mode, "
            "only the active plan.md file may be written."
        ),
    )
    def guarded_write_file(path: str, content: str) -> dict[str, object]:
        if not can_write(path):
            raise PermissionError(
                "Plan Mode is read-only except for the active plan.md file."
            )
        result = write_file.invoke({"path": path, "content": content})
        on_write(str(result["path"]))
        return result

    @tool(
        "edit_file",
        description=(
            "Edit a UTF-8 text file by exact string replacement. In Plan Mode, "
            "only the active plan.md file may be edited."
        ),
    )
    def guarded_edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, object]:
        if not can_write(path):
            raise PermissionError(
                "Plan Mode is read-only except for the active plan.md file."
            )
        result = edit_file.invoke(
            {
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            }
        )
        on_write(str(result["path"]))
        return result

    return [*READ_ONLY_TOOLS, guarded_write_file, guarded_edit_file]
