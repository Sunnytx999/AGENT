from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any


class DirectNativeError(RuntimeError):
    pass


def default_library_path() -> Path:
    native_dir = Path(__file__).resolve().parent.parent / "native" / "bin"
    if sys.platform == "win32":
        filename = "agent_tools_http_ipc.dll"
    elif sys.platform == "darwin":
        filename = "libagent_tools_http_ipc.dylib"
    else:
        filename = "libagent_tools_http_ipc.so"
    return native_dir / filename


class DirectNativeTools:
    """Direct ctypes bridge used only by the parent native-host process."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        configured = library_path or os.getenv("SIMPLE_AGENT_TOOLS_LIBRARY")
        self.path = Path(configured).expanduser().resolve() if configured else default_library_path()
        if not self.path.is_file():
            raise DirectNativeError(
                f"Native tool library was not found: {self.path}. Build it with native\\build.ps1."
            )
        try:
            self._library = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise DirectNativeError(f"Unable to load native tool library {self.path}: {exc}") from exc
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        text = ctypes.c_char_p
        output = ctypes.POINTER(ctypes.c_char)
        size = ctypes.POINTER(ctypes.c_size_t)
        library = self._library
        library.agent_get_local_time.argtypes = [output, ctypes.c_size_t, size]
        library.agent_read_file.argtypes = [text, ctypes.c_int, ctypes.c_int, output, ctypes.c_size_t, size]
        library.agent_list_files.argtypes = [text, ctypes.c_int, output, ctypes.c_size_t, size]
        library.agent_search_text.argtypes = [text, text, ctypes.c_int, ctypes.c_int, output, ctypes.c_size_t, size]
        library.agent_write_file.argtypes = [text, text, output, ctypes.c_size_t, size]
        library.agent_edit_file.argtypes = [text, text, text, ctypes.c_int, output, ctypes.c_size_t, size]
        library.agent_start_process.argtypes = [text, text, text, output, ctypes.c_size_t, size]
        for name in (
            "agent_get_local_time", "agent_read_file", "agent_list_files",
            "agent_search_text", "agent_write_file", "agent_edit_file",
            "agent_start_process",
        ):
            getattr(library, name).restype = ctypes.c_int

    @staticmethod
    def _utf8(value: str) -> bytes:
        return value.encode("utf-8")

    def _call(self, name: str, *arguments: object) -> dict[str, Any]:
        function = getattr(self._library, name)
        required = ctypes.c_size_t()
        status = function(*arguments, None, 0, ctypes.byref(required))
        if status not in (0, 1) or required.value < 2:
            raise DirectNativeError(f"{name} failed to report its output size (status={status}).")
        buffer = ctypes.create_string_buffer(required.value)
        status = function(*arguments, buffer, len(buffer), ctypes.byref(required))
        if status != 0:
            raise DirectNativeError(f"{name} failed (status={status}).")
        try:
            result = json.loads(buffer.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectNativeError(f"{name} returned invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise DirectNativeError(f"{name} returned a non-object JSON value.")
        return result

    def get_local_time(self) -> dict[str, Any]:
        return self._call("agent_get_local_time")

    def read_file(self, path: str, offset: int, limit: int) -> dict[str, Any]:
        return self._call("agent_read_file", self._utf8(path), offset, limit)

    def list_files(self, pattern: str, max_results: int) -> dict[str, Any]:
        return self._call("agent_list_files", self._utf8(pattern), max_results)

    def search_text(self, pattern: str, glob: str, case_sensitive: bool, max_results: int) -> dict[str, Any]:
        return self._call("agent_search_text", self._utf8(pattern), self._utf8(glob), int(case_sensitive), max_results)

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        return self._call("agent_write_file", self._utf8(path), self._utf8(content))

    def edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool) -> dict[str, Any]:
        return self._call("agent_edit_file", self._utf8(path), self._utf8(old_string), self._utf8(new_string), int(replace_all))

    def start_process(self, executable: str, script: str, working_directory: str) -> dict[str, Any]:
        return self._call("agent_start_process", self._utf8(executable), self._utf8(script), self._utf8(working_directory))
