# -*- coding: utf-8 -*-
"""
from dataclasses import dataclass

مقتطف من الفصل 5 — «هندسة الذاكرة والسياق».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch05_memory_and_context.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

from dataclasses import dataclass
import math
import re
import time


@dataclass(slots=True, frozen=True)
class SecureMemoryRecord:
    record_id: str
    tenant_id: str
    category: str
    content: str
    embedding: tuple[float, ...]
    trust_level: float = 1.0  # من 0.1 (مصدر خارجي غير موثوق) إلى 1.0 (موثق)
    created_at_ts: float = 0.0


class HardenedSemanticMemoryStore:
    # أنماط الهجمات المعروفة لتسميم الذاكرة (Memory Poisoning Signatures)
    _INJECTION_PATTERNS = re.compile(
        r"(ignore\s+previous\s+instructions|system\s+override|new\s+admin\s+iban|forget\s+all\s+rules)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._records: list[SecureMemoryRecord] = []

    def remember(
        self,
        *,
        tenant_id: str,
        category: str,
        content: str,
        trust_level: float = 1.0,
    ) -> str:
        # 1. جدار الحماية ضد تسميم الذاكرة (Memory Firewall)
        if self._INJECTION_PATTERNS.search(content):
            raise ValueError(
                "🚨 Memory Firewall Blocked Potential Injection: المحتوى يتضمن أنماط توجيه خبيثة!"
            )

        clamped_trust = max(0.1, min(1.0, trust_level))
        rec_id = f"mem_{len(self._records) + 1:03d}"
        # نفترض وجود دالة حساب المتجهات كما في الفصل الخامس
        emb = (1.0, 0.0)
        self._records.append(
            SecureMemoryRecord(
                record_id=rec_id,
                tenant_id=tenant_id,
                category=category,
                content=content,
                embedding=emb,
                trust_level=clamped_trust,
                created_at_ts=time.time(),
            )
        )
        return rec_id

    @staticmethod
    def _compute_final_rank_score(
        hybrid_similarity: float,
        trust_level: float,
        created_at_ts: float,
        half_life_days: float = 30.0,
    ) -> float:
        """دمج التشابه الدلالي مع درجة الموثوقية وتلاشي الأهمية بمرور الزمن (Exponential Time Decay)"""
        age_seconds = max(0.0, time.time() - created_at_ts)
        age_days = age_seconds / 86_400.0
        time_decay_factor = math.exp(-0.693 * (age_days / half_life_days))
        return hybrid_similarity * trust_level * time_decay_factor
