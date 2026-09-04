from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from threading import Thread

import langchain_agent_cli.cpp_bridge as http_bridge
from direct_native import DirectNativeTools
from langchain_agent_cli.tools import (
    edit_file,
    get_local_time,
    list_files,
    make_main_file_tools,
    read_file,
    search_text,
    write_file,
)
from native_host import create_tool_server


class NativeToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.direct = DirectNativeTools()
        cls.server = create_tool_server(cls.direct, "test-token")
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.old_url = os.environ.get("SIMPLE_AGENT_TOOL_URL")
        cls.old_token = os.environ.get("SIMPLE_AGENT_TOOL_TOKEN")
        os.environ["SIMPLE_AGENT_TOOL_URL"] = f"http://{host}:{port}"
        os.environ["SIMPLE_AGENT_TOOL_TOKEN"] = "test-token"
        http_bridge._instance = None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        environment = os.environ
        if cls.old_url is None:
            environment.pop("SIMPLE_AGENT_TOOL_URL", None)
        else:
            environment["SIMPLE_AGENT_TOOL_URL"] = cls.old_url
        if cls.old_token is None:
            environment.pop("SIMPLE_AGENT_TOOL_TOKEN", None)
        else:
            environment["SIMPLE_AGENT_TOOL_TOKEN"] = cls.old_token
        http_bridge._instance = None

    def test_library_loads_and_returns_local_time(self) -> None:
        library = self.direct
        self.assertTrue(library.path.is_file())
        result = get_local_time.invoke({})
        self.assertIn("iso", result)
        self.assertIn("unix_timestamp", result)

    def test_file_tools_round_trip_through_http_and_dll(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "资料" / "示例.txt"
            text_glob = str(root / "**" / "*.txt")

            created = write_file.invoke(
                {"path": str(target), "content": "第一行\nhello native\n"}
            )
            self.assertEqual(created["operation"], "created")
            self.assertEqual(created["characters_written"], 17)

            read = read_file.invoke(
                {"path": str(target), "offset": 1, "limit": 10}
            )
            self.assertEqual(read["returned_lines"], 2)
            self.assertIn("1: 第一行", read["content"])

            listed = list_files.invoke({"pattern": text_glob})
            self.assertEqual(listed["count"], 1)
            self.assertEqual(Path(listed["files"][0]), target)

            searched = search_text.invoke(
                {
                    "pattern": "hello\\s+native",
                    "glob": text_glob,
                    "case_sensitive": True,
                }
            )
            self.assertEqual(searched["matches"][0]["line"], 2)

            edited = edit_file.invoke(
                {
                    "path": str(target),
                    "old_string": "native",
                    "new_string": "DLL",
                }
            )
            self.assertEqual(edited["replacements"], 1)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "第一行\nhello DLL\n",
            )

    def test_relative_paths_and_globs_are_rejected(self) -> None:
        read_result = read_file.invoke({"path": "relative.txt"})
        self.assertIn("absolute path is required", read_result["error"])
        with self.assertRaises(ValueError):
            write_file.invoke({"path": "relative.txt", "content": "blocked"})
        with self.assertRaises(ValueError):
            list_files.invoke({"pattern": "**/*.py"})
        with self.assertRaises(ValueError):
            search_text.invoke({"pattern": "x", "glob": "**/*.py"})

    def test_read_file_keeps_nonexistent_absolute_path_as_tool_result(self) -> None:
        missing = Path(tempfile.gettempdir()).resolve() / "definitely-missing-file.txt"
        result = read_file.invoke({"path": str(missing)})
        self.assertIn("error", result)

    def test_edit_validation_becomes_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "value.txt"
            path.write_text("same same", encoding="utf-8")
            with self.assertRaises(ValueError):
                edit_file.invoke(
                    {
                        "path": str(path),
                        "old_string": "same",
                        "new_string": "changed",
                    }
                )

    def test_tool_schema_exposes_absolute_path_descriptions(self) -> None:
        write_schema = write_file.args_schema.model_json_schema()
        list_schema = list_files.args_schema.model_json_schema()
        self.assertIn("Absolute path", write_schema["properties"]["path"]["description"])
        self.assertIn("Absolute glob", list_schema["properties"]["pattern"]["description"])
        self.assertIn("pattern", list_schema["required"])

    def test_main_agent_write_guard_stays_in_python(self) -> None:
        writes: list[str] = []
        tools = make_main_file_tools(
            can_write=lambda path: Path(path).name == "plan.md",
            on_write=writes.append,
        )
        by_name = {tool.name: tool for tool in tools}
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary).resolve() / "plan.md"
            denied_path = Path(temporary).resolve() / "source.py"
            by_name["write_file"].invoke(
                {"path": str(plan_path), "content": "# Plan"}
            )
            self.assertEqual(plan_path.read_text(encoding="utf-8"), "# Plan")
            self.assertEqual(writes, [str(plan_path)])
            with self.assertRaises(PermissionError):
                by_name["write_file"].invoke(
                    {"path": str(denied_path), "content": "blocked"}
                )
            self.assertFalse(denied_path.exists())


if __name__ == "__main__":
    unittest.main()
