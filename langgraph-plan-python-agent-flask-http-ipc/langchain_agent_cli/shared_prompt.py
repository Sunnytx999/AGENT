SHARED_FILE_INSPECTION_PROMPT = """

## Local File Inspection Policy

Use local codebase tools only when the user's request genuinely depends on existing
project files.

1. Do not inspect local files for greetings, casual conversation, general conceptual
   questions, or requests that can be answered without codebase-specific evidence.
2. Inspect local files when the user asks to modify, debug, review, explain, plan
   changes to, or otherwise work with an existing project, or when the user explicitly
   says relevant local materials are available.
3. When the relevant file path is unknown, use `list_files` with a focused absolute
   glob rooted in the active working directory and based on the expected directory,
   filename, or extension. If only a symbol, function, class, or text fragment is
   known, use `search_text` with a focused absolute glob instead.
4. Review the files returned by `list_files` before performing deeper inspection.
5. If the result contains no files plausibly related to the request, stop exploring.
   Do not make speculative `read_file` or repeated broad `search_text` calls merely
   to find something to inspect.
6. When no relevant files are found, clearly report that the current workspace does
   not appear to contain the required material and ask the user to provide the file,
   path, or additional context when necessary.
7. If relevant files are present, inspect only the smallest useful set with
   `read_file` or focused `search_text` calls. Avoid reading unrelated files or the
   entire repository.
8. Do not repeat equivalent `list_files`, `search_text`, or `read_file` calls after
   sufficient evidence has already been collected.
"""


def prompt_with_working_directory(base_prompt: str, working_directory: str) -> str:
    """Bind an agent prompt to one explicit filesystem working directory."""
    return (
        f"{base_prompt}\n\n"
        "## Working Directory\n\n"
        f"Your active working directory is: `{working_directory}`\n\n"
        "Treat this directory as the root of the current task. Every path or glob "
        "passed to a local file tool must be an absolute path. Resolve intended "
        "relative locations against the active working directory yourself before "
        "calling a tool. File tools do not infer, prepend, or query a working "
        "directory. Do not confuse the working directory with the directory that "
        "contains the Python application or native library.\n"
    )
