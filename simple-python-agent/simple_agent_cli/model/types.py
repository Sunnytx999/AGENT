from __future__ import annotations

from typing import TypeAlias

from ..messages import Message


ModelMessage: TypeAlias = Message
ToolDefinition: TypeAlias = dict[str, object]
