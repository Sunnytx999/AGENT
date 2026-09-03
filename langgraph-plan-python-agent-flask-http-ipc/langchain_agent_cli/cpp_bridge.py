from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class NativeToolError(RuntimeError):
    pass


class NativeTools:
    """HTTP/JSON client used by the Flask child; it never loads the DLL."""

    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("SIMPLE_AGENT_TOOL_URL", "")).rstrip("/")
        self.token = token or os.getenv("SIMPLE_AGENT_TOOL_TOKEN", "")
        if not self.base_url or not self.token:
            raise NativeToolError(
                "Native tool HTTP endpoint is not configured. "
                "Start the project with: python native_host.py"
            )

    def _call(self, method: str, params: dict[str, object]) -> dict[str, Any]:
        body = json.dumps(
            {"method": method, "params": params}, ensure_ascii=False
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/tools/call",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise NativeToolError(f"Native tool HTTP error {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NativeToolError(f"Native tool service request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise NativeToolError("Native tool service returned a non-object JSON value.")
        if not payload.get("ok"):
            raise NativeToolError(str(payload.get("error") or "Unknown native tool service error"))
        result = payload.get("result")
        if not isinstance(result, dict):
            raise NativeToolError("Native tool service returned an invalid result.")
        return result

    def get_local_time(self) -> dict[str, Any]:
        return self._call("get_local_time", {})

    def read_file(self, path: str, offset: int, limit: int) -> dict[str, Any]:
        return self._call("read_file", {"path": path, "offset": offset, "limit": limit})

    def list_files(self, pattern: str, max_results: int) -> dict[str, Any]:
        return self._call("list_files", {"pattern": pattern, "max_results": max_results})

    def search_text(
        self,
        pattern: str,
        glob: str,
        case_sensitive: bool,
        max_results: int,
    ) -> dict[str, Any]:
        return self._call(
            "search_text",
            {
                "pattern": pattern,
                "glob": glob,
                "case_sensitive": case_sensitive,
                "max_results": max_results,
            },
        )

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        return self._call("write_file", {"path": path, "content": content})

    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> dict[str, Any]:
        return self._call(
            "edit_file",
            {
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            },
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
