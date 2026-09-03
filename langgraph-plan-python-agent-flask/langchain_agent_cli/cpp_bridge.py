from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any


class NativeToolError(RuntimeError):
    pass


def _default_library_path() -> Path:
    native_dir = Path(__file__).resolve().parent.parent / "native" / "bin"
    if sys.platform == "win32":
        filename = "agent_tools_v2.dll"
    elif sys.platform == "darwin":
        filename = "libagent_tools_v2.dylib"
    else:
        filename = "libagent_tools_v2.so"
    return native_dir / filename


class NativeTools:
    def __init__(self, library_path: str | Path | None = None) -> None:
        configured = library_path or os.getenv("SIMPLE_AGENT_TOOLS_LIBRARY")
        self.path = Path(configured).expanduser().resolve() if configured else _default_library_path()
        if not self.path.is_file():
            raise NativeToolError(
                f"Native tool library was not found: {self.path}. "
                "Build it with native\\build.ps1."
            )
        try:
            self._library = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise NativeToolError(f"Unable to load native tool library {self.path}: {exc}") from exc
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        char_pointer = ctypes.c_char_p
        output_pointer = ctypes.POINTER(ctypes.c_char)
        size_pointer = ctypes.POINTER(ctypes.c_size_t)

        self._library.agent_get_local_time.argtypes = [output_pointer, ctypes.c_size_t, size_pointer]
        self._library.agent_read_file.argtypes = [char_pointer, ctypes.c_int, ctypes.c_int, output_pointer, ctypes.c_size_t, size_pointer]
        self._library.agent_list_files.argtypes = [char_pointer, ctypes.c_int, output_pointer, ctypes.c_size_t, size_pointer]
        self._library.agent_search_text.argtypes = [char_pointer, char_pointer, ctypes.c_int, ctypes.c_int, output_pointer, ctypes.c_size_t, size_pointer]
        self._library.agent_write_file.argtypes = [char_pointer, char_pointer, output_pointer, ctypes.c_size_t, size_pointer]
        self._library.agent_edit_file.argtypes = [char_pointer, char_pointer, char_pointer, ctypes.c_int, output_pointer, ctypes.c_size_t, size_pointer]
        for name in (
            "agent_get_local_time",
            "agent_read_file",
            "agent_list_files",
            "agent_search_text",
            "agent_write_file",
            "agent_edit_file",
        ):
            getattr(self._library, name).restype = ctypes.c_int

    @staticmethod
    def _utf8(value: str) -> bytes:
        return value.encode("utf-8")

    def _call(self, function_name: str, *arguments: object) -> dict[str, Any]:
        function = getattr(self._library, function_name)
        required = ctypes.c_size_t()
        status = function(*arguments, None, 0, ctypes.byref(required))
        if status not in (0, 1) or required.value < 2:
            raise NativeToolError(f"{function_name} failed to report its output size (status={status}).")
        buffer = ctypes.create_string_buffer(required.value)
        status = function(*arguments, buffer, len(buffer), ctypes.byref(required))
        if status != 0:
            raise NativeToolError(f"{function_name} failed (status={status}).")
        try:
            result = json.loads(buffer.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NativeToolError(f"{function_name} returned invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise NativeToolError(f"{function_name} returned a non-object JSON value.")
        return result

    def get_local_time(self) -> dict[str, Any]:
        return self._call("agent_get_local_time")

    def read_file(self, path: str, offset: int, limit: int) -> dict[str, Any]:
        return self._call("agent_read_file", self._utf8(path), offset, limit)

    def list_files(self, pattern: str, max_results: int) -> dict[str, Any]:
        return self._call("agent_list_files", self._utf8(pattern), max_results)

    def search_text(self, pattern: str, glob: str, case_sensitive: bool, max_results: int) -> dict[str, Any]:
        return self._call(
            "agent_search_text",
            self._utf8(pattern),
            self._utf8(glob),
            int(case_sensitive),
            max_results,
        )

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        return self._call("agent_write_file", self._utf8(path), self._utf8(content))

    def edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool) -> dict[str, Any]:
        return self._call(
            "agent_edit_file",
            self._utf8(path),
            self._utf8(old_string),
            self._utf8(new_string),
            int(replace_all),
        )


_instance: NativeTools | None = None
_instance_lock = Lock()


def native_tools() -> NativeTools:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NativeTools()
    return _instance
