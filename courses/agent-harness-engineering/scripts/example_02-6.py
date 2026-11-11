# -*- coding: utf-8 -*-
"""
import asyncio

مقتطف من الفصل 12 — «إطار التقييم والاختبار للوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch12_evaluation_harness.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

type ToolPayload = dict[str, Any]
type LiveToolCallable = Callable[[str, ToolPayload], Awaitable[ToolPayload]]


class CassetteMode(StrEnum):
    RECORD = "record"
    REPLAY_STRICT = "replay_strict"


class CassetteMissError(RuntimeError):
    """يُرفع عندما يحاول الوكيل في وضع REPLAY_STRICT استدعاء أداة أو معاملات غير مسجلة."""


class ToolCallCassetteEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    signature_hash: str = Field(min_length=64, max_length=64)
    tool_name: str
    arguments: ToolPayload
    recorded_response: ToolPayload


class DeterministicToolCassette:
    """
    حاضنة تسجيل وإعادة تشغيل حتمية لاستدعاءات الأدوات (VCR Pattern).
    تضمن عزل اختبارات CI/CD عن الشبكة الخارجية وكشف أي انحراف في سلوك الوكيل.
    """

    def __init__(self, mode: CassetteMode = CassetteMode.REPLAY_STRICT) -> None:
        self._mode = mode
        self._entries: dict[str, ToolCallCassetteEntry] = {}

    @staticmethod
    def compute_canonical_hash(tool_name: str, arguments: ToolPayload) -> str:
        """
        حساب بصمة SHA-256 معيارية ومستقلة عن ترتيب مفاتيح القاموس (Canonical JSON).
        """
        canonical_payload = json.dumps(
            {"tool": tool_name, "args": arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    async def execute_or_replay(
        self,
        tool_name: str,
        arguments: ToolPayload,
        live_executor: LiveToolCallable,
    ) -> ToolPayload:
        sig_hash = self.compute_canonical_hash(tool_name, arguments)

        match self._mode:
            case CassetteMode.RECORD:
                response = await live_executor(tool_name, arguments)
                entry = ToolCallCassetteEntry(
                    signature_hash=sig_hash,
                    tool_name=tool_name,
                    arguments=arguments,
                    recorded_response=response,
                )
                self._entries[sig_hash] = entry
                return response

            case CassetteMode.REPLAY_STRICT:
                if sig_hash not in self._entries:
                    raise CassetteMissError(
                        f"انحراف في مسار الوكيل (Cassette Miss)! لم يتم العثور على استدعاء مسجل "
                        f"للأداة '{tool_name}' بالمعاملات {arguments} (البصمة: {sig_hash[:12]}...)."
                    )
                return self._entries[sig_hash].recorded_response

    def dump_json(self) -> str:
        adapter = TypeAdapter(list[ToolCallCassetteEntry])
        return adapter.dump_json(list(self._entries.values()), indent=2).decode("utf-8")

    def load_json(self, raw_json: str) -> None:
        adapter = TypeAdapter(list[ToolCallCassetteEntry])
        loaded = adapter.validate_json(raw_json)
        self._entries = {entry.signature_hash: entry for entry in loaded}
