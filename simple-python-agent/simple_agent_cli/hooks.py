from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .messages import APIMessage, Message, MessageList


BeforeModelHook = Callable[[MessageList], None]
AfterModelHook = Callable[[Message], None]
BeforeToolHook = Callable[[object], None]
AfterToolHook = Callable[[APIMessage], None]


def _ignore(_value: object) -> None:
    pass


@dataclass
class QueryHooks:
    """查询循环扩展点；后续权限、日志和追踪可以从这里接入。"""

    before_model: BeforeModelHook = _ignore
    after_model: AfterModelHook = _ignore
    before_tool: BeforeToolHook = _ignore
    after_tool: AfterToolHook = _ignore
