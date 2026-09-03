1. Create `E:\persona-chatbot` with directories `src\persona_chatbot\personas` and `tests`.
2. Create `pyproject.toml` declaring dependencies `langchain>=1.0,<2.0`, `langchain-openai>=1.0,<2.0`, `langgraph>=1.0,<2.0`, `python-dotenv>=1.0,<2.0`, optional `pytest` dev dependency, and console script `persona-chatbot = "persona_chatbot.cli:main"`.
3. Create `.env.example` with `OPENAI_API_KEY=`, `OPENAI_BASE_URL=https://api.deepseek.com`, and `OPENAI_MODEL=deepseek-chat`.
4. Create `README.md` documenting install, environment setup, run commands, available personas, and chat commands.
5. Create `src\persona_chatbot\__init__.py` exporting the public API version.
6. Create `src\persona_chatbot\config.py` with frozen dataclass `ChatConfig(api_key, base_url, model, timeout_seconds=60.0, max_retries=3)` and `get_config()` that calls `dotenv.load_dotenv()` then reads `os.getenv`.
7. Create `src\persona_chatbot\state.py` defining `AgentState(TypedDict)` with `messages: Annotated[list[AnyMessage], add_messages]` and `persona_id: str`.
8. Create `src\persona_chatbot\personas\base.py` defining frozen dataclass `Persona(id, display_name, system_prompt, style_hints, opening_line)`.
9. In `base.py`, add `PERSONAS: dict[str, Persona] = {}`, `register_persona(persona)`, `get_persona(persona_id)`, `list_personas()`, and `DEFAULT_PERSONA_ID = "pirate"`.
10. Create `src\persona_chatbot\personas\pirate.py` with `PIRATE = Persona(id="pirate", display_name="海盗船长", system_prompt=..., style_hints=..., opening_line=...)` using pirate-speak Chinese prompts.
11. Create `src\persona_chatbot\personas\witch.py` with `WITCH = Persona(id="witch", display_name="森林女巫", system_prompt=..., style_hints=..., opening_line=...)` using witch-speak Chinese prompts.
12. Update `src\persona_chatbot\personas\__init__.py` to import `PIRATE` and `WITCH`, register both, and re-export `Persona`, `get_persona`, `list_personas`, and `DEFAULT_PERSONA_ID`.
13. Create `src\persona_chatbot\llm.py` with `create_chat_model(config: ChatConfig)` returning `ChatOpenAI(model=..., api_key=..., base_url=..., timeout=..., max_retries=...)`.
14. Create `src\persona_chatbot\memory.py` with `create_memory() -> MemorySaver` and `build_thread_id(persona_id: str) -> str` returning `persona:{persona_id}`.
15. Create `src\persona_chatbot\graph.py` with `build_chat_graph(llm, memory)` that constructs `StateGraph(AgentState)` with `START -> chatbot -> END`.
16. In `graph.py`, implement `chatbot_node(state)` to resolve the persona via `get_persona(state["persona_id"])`, prepend `SystemMessage(system_prompt + "\n" + style_hints)` to `state["messages"]`, call `llm.invoke`, and return `{"messages": [response]}`.
17. In `build_chat_graph`, compile the graph with `checkpointer=memory` and return the compiled graph.
18. Create `src\persona_chatbot\commands.py` with `Command` dataclass and `parse_command(text)` returning `None` for non-`/` input, otherwise `Command(name, arg)`.
19. In `commands.py`, add `run_command(command, chatbot)` handling `/help`, `/personas`, `/pirate`, and `/witch`; switch commands call `chatbot.switch_persona` and return that persona's opening line.
20. In `commands.py`, make unknown slash commands return a help/error message and never pass command text to the LLM or memory.
21. Create `src\persona_chatbot\chatbot.py` with class `PersonaChatbot` whose `__init__` builds config, model, memory, and compiled graph, and sets `current_persona_id = DEFAULT_PERSONA_ID`.
22. In `chatbot.py`, implement `switch_persona(persona_id)` validating via `get_persona`, updating `current_persona_id`, and returning the `Persona`.
23. In `chatbot.py`, implement `respond(text)` that invokes the graph with `{"messages": [HumanMessage(content=text)], "persona_id": self.current_persona_id}` and `config={"configurable": {"thread_id": build_thread_id(self.current_persona_id)}}`.
24. In `chatbot.py`, extract the final `AIMessage` from the returned `result["messages"]` with a `_message_text(message)` helper that handles both string and list content.
25. Create `src\persona_chatbot\cli.py` with `main()` using argparse options `-p/--print` for one-shot input and `--persona` for the initial persona.
26. In `cli.py`, implement `run_interactive()` that prints welcome text and available personas, then loops over `input()`.
27. In `run_interactive`, for each input call `parse_command`; if a command, execute it and break on `/exit`; otherwise call `chatbot.respond` and print the reply.
28. Create `src\persona_chatbot\__main__.py` calling `main()` and root `main.py` re-exporting `main` for `python -m persona_chatbot`.
29. Create `tests\conftest.py` with `FakeChatModel` whose `invoke(messages)` records messages and returns `AIMessage(content="fake reply")`.
30. Create `tests\test_personas.py` asserting unique persona ids, non-empty prompts and opening lines, and `get_persona` raising `KeyError` for unknown ids.
31. Create `tests\test_commands.py` asserting `/pirate` and `/witch` return switch commands, `/help` is handled, and ordinary text returns `None`.
32. Create `tests\test_memory.py` asserting `build_thread_id("pirate") == "persona:pirate"` and `create_memory()` returns a `MemorySaver`.
33. Create `tests\test_graph.py` using `FakeChatModel` and `MemorySaver` to assert the first message sent to the model for `persona_id="pirate"` is a `SystemMessage` containing the pirate prompt.
34. Add a `test_graph.py` case invoking `witch` after `pirate` and asserting the fake model receives different system prompts and that `persona_id` flows through state correctly.
35. Create `tests\test_chatbot.py` with a stub graph recording `invoke(value, config)` to assert `respond` sends a `HumanMessage`, the active `persona_id`, and `thread_id=persona:pirate`; then switch to `witch` and assert both values change.
36. Run `python -m pip install -e .[dev]` in `E:\persona-chatbot`.
37. Run `python -m pytest` in `E:\persona-chatbot` and fix failures until all tests pass.
38. Run `python -m persona_chatbot` and manually verify `/pirate`, `/witch`, normal chat, and that each persona retains separate memory after switching back and forth.
39. Run `python main.py -p "你好" --persona witch` to verify one-shot CLI mode with initial persona selection.
40. Review `config.py`, `.env.example`, and `README.md` to ensure no hardcoded API keys remain and all secrets are loaded from environment variables.