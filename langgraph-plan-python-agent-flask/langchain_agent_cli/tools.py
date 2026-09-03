from __future__ import annotations

from typing import Annotated, Callable

from langchain.tools import tool

from .cpp_bridge import native_tools

MAX_READ_LINES = 500
MAX_RESULTS = 200


def _raise_native_error(result: dict[str, object]) -> dict[str, object]:
    error = result.get("error")
    if error:
        raise ValueError(str(error))
    return result


@tool(
    description="Get the current local date, time, timezone, UTC offset, and Unix timestamp."
)
def get_local_time() -> dict[str, object]:
    return _raise_native_error(native_tools().get_local_time())


@tool(
    description=f"Read numbered lines from a UTF-8 file using an absolute path. At most {MAX_READ_LINES} lines are returned per call; use offset and limit to page through longer files."
)
def read_file(
    path: Annotated[
        str,
        "Absolute path of the UTF-8 file to read. Relative paths are rejected.",
    ],
    offset: Annotated[
        int,
        "One-based line number at which reading starts.",
    ] = 1,
    limit: Annotated[
        int,
        f"Maximum number of lines to return; must be between 1 and {MAX_READ_LINES}.",
    ] = MAX_READ_LINES,
) -> dict[str, object]:
    # read_file historically reports ordinary path/range errors in its result.
    return native_tools().read_file(path, offset, limit)


@tool(
    description=f"List local files matching an absolute glob. Returns at most {MAX_RESULTS} files; use a narrower pattern when truncated is true."
)
def list_files(
    pattern: Annotated[
        str,
        "Absolute glob pattern, for example E:\\project\\**\\*.py. Relative glob patterns are rejected.",
    ],
) -> dict[str, object]:
    return _raise_native_error(native_tools().list_files(pattern, MAX_RESULTS))


@tool(
    description=f"Search UTF-8 text files with a regular expression. Returns at most {MAX_RESULTS} matches; narrow glob or regex when truncated is true."
)
def search_text(
    pattern: Annotated[
        str,
        "Regular expression to search for in matching UTF-8 text files.",
    ],
    glob: Annotated[
        str,
        "Absolute glob selecting files to search, for example E:\\project\\**\\*.py. Relative globs are rejected.",
    ],
    case_sensitive: Annotated[
        bool,
        "Whether regular-expression matching is case-sensitive.",
    ] = True,
) -> dict[str, object]:
    return _raise_native_error(
        native_tools().search_text(pattern, glob, case_sensitive, MAX_RESULTS)
    )


@tool(
    description="Create or completely overwrite a UTF-8 text file, creating parent directories when needed."
)
def write_file(
    path: Annotated[
        str,
        "Absolute path of the UTF-8 file to create or overwrite. Relative paths are rejected.",
    ],
    content: Annotated[
        str,
        "Complete UTF-8 content to write to the file.",
    ],
) -> dict[str, object]:
    return _raise_native_error(native_tools().write_file(path, content))


@tool(
    description="Edit a UTF-8 text file by exact string replacement; the match must be unique unless replace_all is true."
)
def edit_file(
    path: Annotated[
        str,
        "Absolute path of the UTF-8 file to edit. Relative paths are rejected.",
    ],
    old_string: Annotated[
        str,
        "Exact existing text to replace; include enough surrounding context to make it unique.",
    ],
    new_string: Annotated[
        str,
        "Replacement text that will take the place of old_string.",
    ],
    replace_all: Annotated[
        bool,
        "Replace every occurrence when true; otherwise old_string must occur exactly once.",
    ] = False,
) -> dict[str, object]:
    return _raise_native_error(
        native_tools().edit_file(path, old_string, new_string, replace_all)
    )


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
    def guarded_write_file(
        path: Annotated[
            str,
            "Absolute path of the UTF-8 file to create or overwrite. Relative paths are rejected.",
        ],
        content: Annotated[
            str,
            "Complete UTF-8 content to write to the file.",
        ],
    ) -> dict[str, object]:
        if not can_write(path):
            raise PermissionError(
                "Write denied. An absolute path is required, and Plan Mode may only write the active plan.md file."
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
        path: Annotated[
            str,
            "Absolute path of the UTF-8 file to edit. Relative paths are rejected.",
        ],
        old_string: Annotated[
            str,
            "Exact existing text to replace; include enough surrounding context to make it unique.",
        ],
        new_string: Annotated[
            str,
            "Replacement text that will take the place of old_string.",
        ],
        replace_all: Annotated[
            bool,
            "Replace every occurrence when true; otherwise old_string must occur exactly once.",
        ] = False,
    ) -> dict[str, object]:
        if not can_write(path):
            raise PermissionError(
                "Edit denied. An absolute path is required, and Plan Mode may only edit the active plan.md file."
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
