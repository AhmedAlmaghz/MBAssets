# -*- coding: utf-8 -*-
"""
from typing import Protocol

مقتطف من الفصل 4 — «هندسة الأدوات واستدعاء الدوال».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch04_tool_registry.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from typing import Protocol


# 1. تعريف عقد برمجي (Protocol) يضمن أن أي سياق يمرر للـ Harness يمتلك خاصية user_scopes
class HasSecurityScopes(Protocol):
    @property
    def user_scopes(self) -> set[str]:
        ...


# 2. تقييد الـ Generic Type في الـ Harness ليقبل فقط السياقات التي تحقق HasSecurityScopes
class EnterpriseToolHarness[ContextT: HasSecurityScopes]:
    # ... (داخل المزخرف @tool نضيف المعامل required_scopes: set[str] | None = None)
    # ونحفظه داخل ToolSpec كـ required_scopes: frozenset[str] = field(default_factory=frozenset)

    def get_llm_schemas(self, trusted_context: ContextT) -> list[dict[str, object]]:
        """المستوى الأول (حاجز الرؤية): إخفاء الأدوات التي لا يملك المستخدم صلاحياتها عن الـ LLM تماماً"""
        authorized_schemas: list[dict[str, object]] = []
        for spec in self._tools.values():
            if spec.required_scopes.issubset(trusted_context.user_scopes):
                authorized_schemas.append(spec.to_llm_json_schema())
        return authorized_schemas

    async def execute_single(
        self,
        request: ToolCallRequest,
        trusted_context: ContextT,
    ) -> ToolExecutionOutput:
        """المستوى الثاني (الدفاع في العمق - Defense-in-Depth): التحقق الصارم قبل التنفيذ الفعلي"""
        spec = self._tools.get(request.tool_name)
        if spec is None:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Error: Tool '{request.tool_name}' is not available.",
                is_error=True,
            )

        # فحص الصلاحيات وقت التشغيل لمنع استدعاء الأدوات المخفية عبر التخمين أو الحقن!
        missing_scopes = spec.required_scopes - trusted_context.user_scopes
        if missing_scopes:
            return ToolExecutionOutput(
                call_id=request.call_id,
                tool_name=request.tool_name,
                content=f"Authorization Denied: Missing required scopes {sorted(missing_scopes)} to run '{spec.name}'.",
                is_error=True,
            )

        # ... استكمال التحقق من المعطيات والتنفيذ كالمعتاد
