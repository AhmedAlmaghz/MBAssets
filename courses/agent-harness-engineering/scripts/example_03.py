# -*- coding: utf-8 -*-
"""
from typing import Any, Literal, Self

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseAgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ToolCallItem(BaseAgentEvent):
    """نموذج صارم لكل أداة يطلبها الوكيل"""

    name: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",  # يمنع أي محاولة لحقن رموز غريبة في اسم الأداة
        description="اسم الأداة بالأحرف الإنجليزية الصغيرة والشرطة السفلية فقط",
    )
    args: dict[str, Any] = Field(default_factory=dict)


class ExecuteToolDecision(BaseAgentEvent):
    decision_type: Literal["execute_tools"] = "execute_tools"
    thought: str = Field(min_length=5)
    tools_to_call: list[ToolCallItem] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def prevent_duplicate_parallel_calls(self) -> Self:
        """يمنع الوكيل من طلب تنفيذ نفس الأداة بنفس المعطيات مرتين في نفس اللحظة"""
        seen_signatures: set[str] = set()
        for item in self.tools_to_call:
            # تكوين بصمة فريدة للأداة ومدخلاتها المرتبة
            signature = f"{item.name}:{sorted(item.args.items())}"
            if signature in seen_signatures:
                raise ValueError(
                    f"Duplicate parallel tool call detected for '{item.name}' with identical args. "
                    "Call each unique tool/argument combination only once per batch."
                )
            seen_signatures.add(signature)
        return self
