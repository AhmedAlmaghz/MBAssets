# -*- coding: utf-8 -*-
"""
from collections.abc import Awaitable, Callable

مقتطف من الفصل 9 — «التدخل البشري ونقاط الحفظ وإدارة الجلسات الطويلة».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch09_durable_hitl_checkpoints.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel

# تعريف نوع دالة السياسة الديناميكية: تستلم (معطيات_الأداة، حالة_النطاق) وتعيد (هل_تتطلب_موافقة، سبب_المخاطرة)
type DynamicApprovalPolicy[StateT: BaseModel] = Callable[
    [dict[str, Any], StateT], tuple[bool, str]
]


@dataclass(slots=True)
class SmartGatedTool[StateT: BaseModel]:
    name: str
    func: Callable[..., Awaitable[str]]
    requires_approval: bool = False
    risk_description: str = "أداة حساسة تتطلب مراجعة المشرف."
    approval_policy: DynamicApprovalPolicy[StateT] | None = None

    def evaluate_gate(self, tool_args: dict[str, Any], domain_state: StateT) -> tuple[bool, str]:
        """تقييم بوابة الأمان ديناميكياً وقت التشغيل بناءً على القيم الفعلية للمعطيات"""
        if self.approval_policy is not None:
            return self.approval_policy(tool_args, domain_state)
        return self.requires_approval, self.risk_description


# مثال لسياسة مالية ديناميكية: المبالغ <= 100$ تمر تلقائياً، وما زاد يتجمد للموافقة البشرية!
def refund_risk_policy(args: dict[str, Any], state: BaseModel) -> tuple[bool, str]:
    amount = float(args.get("amount_usd", 0.0))
    if amount > 100.0:
        return True, f"المبلغ المطلوب (${amount:.2f}) يتجاوز سقف الموافقة التلقائية ($100.00)."
    return False, "ضمن سقف الموافقة التلقائية الآمن."
