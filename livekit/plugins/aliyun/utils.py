from __future__ import annotations

import os
from collections.abc import Awaitable
from typing import Any, Callable, Union

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from livekit.agents import llm

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv(
            "ZHIPU_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
        )
    return base_url


def to_fnc_ctx(fnc_ctx: list[llm.Tool]) -> list[ChatCompletionToolParam]:
    return llm.ToolContext(fnc_ctx).parse_function_tools("openai", strict=True)  # type: ignore[return-value]


def to_chat_ctx(
    chat_ctx: llm.ChatContext, cache_key: Any
) -> list[ChatCompletionMessageParam]:
    del cache_key
    messages, _ = chat_ctx.to_provider_format(format="openai")
    return messages  # type: ignore[return-value]
