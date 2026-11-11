#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 5: هندسة الذاكرة والسياق
==============================================================================

البرنامج الكامل للفصل 5 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
هندسة الذاكرة والسياق — Memory & Context Engineering.

التشغيل:
    python ch05_memory_and_context.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum, auto
import math
import re
from typing import Any, Protocol


# =====================================================================
# 1. هياكل الرسائل والوحدات الذرية (Atomic Conversation Units)
# =====================================================================
class MessageRole(StrEnum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()


@dataclass(slots=True, frozen=True)
class ToolCallRef:
    """إشارة لطلب استدعاء أداة داخل رسالة المساعد"""

    call_id: str
    tool_name: str
    arguments_json: str


@dataclass(slots=True)
class ContextMessage:
    """رسالة داخل نافذة السياق مع حساب تقريبي سريع لعدد التوكنز"""

    role: MessageRole
    content: str
    tool_calls: tuple[ToolCallRef, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_masked: bool = False

    @property
    def estimated_tokens(self) -> int:
        """تقدير هندسي سريع للتوكنز (حوالي 3.5 حرف لكل توكن + عبء الهيكلة)"""
        base_chars = len(self.content) + sum(
            len(tc.tool_name) + len(tc.arguments_json) for tc in self.tool_calls
        )
        return max(4, (base_chars // 3) + 6)

    def to_api_dict(self) -> dict[str, Any]:
        """تحويل الرسالة إلى القاموس القياسي المتوافق مع واجهات LLMs"""
        payload: dict[str, Any] = {"role": str(self.role), "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.tool_name, "arguments": tc.arguments_json},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            payload["name"] = self.tool_name
        return payload


@dataclass(slots=True)
class AtomicTurnGroup:
    """وحدة ذرية غير قابلة للتجزئة!

    تضم إما رسالة مستخدم، أو دورة تفكير مساعد كاملة مع جميع نتائج أدواتها (Tool Results).
    """

    turn_id: int
    messages: list[ContextMessage] = field(default_factory=list)
    pinned: bool = False  # إذا كانت True، يُمنع الـ Harness من حذف هذه الدورة أبداً (مثل هدف المستخدم الأول)

    @property
    def total_tokens(self) -> int:
        return sum(m.estimated_tokens for m in self.messages)


# =====================================================================
# 2. الذاكرة الدلالية طويلة المدى المعزولة (Multi-Tenant Semantic Store)
# =====================================================================
@dataclass(slots=True, frozen=True)
class MemoryRecord:
    """سجل ذاكرة طويل المدى معزول بمعرف المستأجر (Tenant Isolation)"""

    record_id: str
    tenant_id: str
    category: str  # مثل: 'user_preference', 'episodic_lesson', 'domain_fact'
    content: str
    embedding: tuple[float, ...]


class SemanticMemoryStore:
    """مخزن ذاكرة طويل المدى يدعم البحث الهجين (المتجهي + الكلمات المفتاحية) مع عزل المستأجرين"""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []

    @staticmethod
    def _compute_lightweight_embedding(text: str, dims: int = 16) -> tuple[float, ...]:
        """توليد متجه رقمي حتمي لأغراض المحاكاة المستقلة (في الإنتاج يستبدل بـ Embedding API)"""
        vec = [0.0] * dims
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            idx = hash(tok) % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(round(v / norm, 4) for v in vec)

    @staticmethod
    def _cosine_similarity(v1: tuple[float, ...], v2: tuple[float, ...]) -> float:
        return sum(a * b for a, b in zip(v1, v2, strict=False))

    def remember(self, *, tenant_id: str, category: str, content: str) -> str:
        rec_id = f"mem_{len(self._records) + 1:03d}"
        emb = self._compute_lightweight_embedding(content)
        self._records.append(
            MemoryRecord(
                record_id=rec_id,
                tenant_id=tenant_id,
                category=category,
                content=content,
                embedding=emb,
            )
        )
        return rec_id

    def recall(self, *, tenant_id: str, query: str, top_k: int = 3) -> list[MemoryRecord]:
        """استرجاع الذكريات الأكثر صلة مع فرض عزل صارم للمستأجر (Zero Cross-Tenant Leak)"""
        query_emb = self._compute_lightweight_embedding(query)
        query_terms = set(re.findall(r"\w+", query.lower()))

        scored: list[tuple[float, MemoryRecord]] = []
        for rec in self._records:
            if rec.tenant_id != tenant_id:
                continue  # حاجز أمني صارم: لا تقرأ أبداً ذاكرة مستخدم أو مؤسسة أخرى!

            vec_score = self._cosine_similarity(query_emb, rec.embedding)
            rec_terms = set(re.findall(r"\w+", rec.content.lower()))
            keyword_overlap = len(query_terms & rec_terms) / max(1, len(query_terms))
            hybrid_score = (0.6 * vec_score) + (0.4 * keyword_overlap)
            scored.append((hybrid_score, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for score, rec in scored[:top_k] if score > 0.05]


# =====================================================================
# 3. قلب النظام: ContextMemoryHarness (إدارة الميزانية، الطمس، والتلخيص)
# =====================================================================
class StateSummarizer(Protocol):
    """عقد برمجي لضغط الدورات القديمة في ملخص حالة مكثف"""

    async def summarize_turns(
        self, previous_summary: str, evicted_turns: list[AtomicTurnGroup]
    ) -> str:
        ...


class ContextMemoryHarness:
    """إطار التحكم بهندسة السياق والذاكرة العاملة.

    يحمي أزواج الأدوات الذرية، يطمس المخرجات القديمة الضخمة، ويلخص التاريخ عند تجاوز الميزانية،
    مع الحفاظ على ثبات البادئة (Prefix KV-Cache Optimization).
    """

    def __init__(
        self,
        *,
        static_system_prompt: str,
        max_context_tokens: int = 1_200,
        keep_recent_unmasked_turns: int = 2,
        mask_threshold_chars: int = 200,
        summarizer: StateSummarizer,
        long_term_store: SemanticMemoryStore,
    ) -> None:
        self.static_system_prompt = static_system_prompt.strip()
        self.max_context_tokens = max_context_tokens
        self.keep_recent_unmasked_turns = keep_recent_unmasked_turns
        self.mask_threshold_chars = mask_threshold_chars
        self.summarizer = summarizer
        self.long_term_store = long_term_store

        self._turn_counter = 0
        self._turns: list[AtomicTurnGroup] = []
        self._rolling_summary: str = ""

    def add_user_goal(self, goal_text: str, *, pin: bool = True) -> None:
        """إضافة هدف المستخدم كدورة ذرية مثبتة (Pinned) لا يمسحها التقليم أبداً"""
        self._turn_counter += 1
        msg = ContextMessage(role=MessageRole.USER, content=goal_text)
        self._turns.append(
            AtomicTurnGroup(turn_id=self._turn_counter, messages=[msg], pinned=pin)
        )

    def add_agent_tool_cycle(
        self,
        *,
        assistant_thought: str,
        tool_calls: list[ToolCallRef],
        tool_results: list[tuple[str, str, str]],  # (call_id, tool_name, result_content)
    ) -> None:
        """تسجيل دورة عمل أداة كاملة (طلب المساعد + جميع نتائج الأدوات) كوحدة ذرية واحدة!

        هذا التصميم يجعل ظهور خطأ 'Orphaned Tool Result' مستحيلاً رياضياً.
        """
        self._turn_counter += 1
        group_msgs: list[ContextMessage] = [
            ContextMessage(
                role=MessageRole.ASSISTANT,
                content=assistant_thought,
                tool_calls=tuple(tool_calls),
            )
        ]
        for call_id, t_name, res_text in tool_results:
            group_msgs.append(
                ContextMessage(
                    role=MessageRole.TOOL,
                    content=res_text,
                    tool_call_id=call_id,
                    tool_name=t_name,
                )
            )
        self._turns.append(AtomicTurnGroup(turn_id=self._turn_counter, messages=group_msgs))

    def _apply_stage1_observation_masking(self) -> int:
        """المرحلة الأولى للضغط (مجانية وفورية):

        طمس مخرجات الأدوات الضخمة في الدورات القديمة مع إبقاء آخر N دورات بكامل تفاصيلها.
        """
        tokens_saved = 0
        if len(self._turns) <= self.keep_recent_unmasked_turns:
            return 0

        cutoff_index = len(self._turns) - self.keep_recent_unmasked_turns
        for turn in self._turns[:cutoff_index]:
            for msg in turn.messages:
                if (
                    msg.role == MessageRole.TOOL
                    and not msg.is_masked
                    and len(msg.content) > self.mask_threshold_chars
                ):
                    before_tokens = msg.estimated_tokens
                    preview = msg.content[:80].replace("\n", " ")
                    msg.content = (
                        f"[🗜️ COMPACTED TOOL OUTPUT from `{msg.tool_name}` "
                        f"( Turn #{turn.turn_id} ): '{preview}...' — "
                        f"Full data omitted to save context window.]"
                    )
                    msg.is_masked = True
                    tokens_saved += before_tokens - msg.estimated_tokens
        return tokens_saved

    def current_working_tokens(self) -> int:
        base_sys = (len(self.static_system_prompt) // 3) + (len(self._rolling_summary) // 3)
        return base_sys + sum(t.total_tokens for t in self._turns)

    async def compact_if_needed(self) -> dict[str, int]:
        """إدارة ميزانية السياق على مرحلتين: الطمس أولاً، ثم التلخيص المتدحرج للوحدات الذرية"""
        stats = {"masked_tokens_saved": 0, "evicted_turns": 0}

        if self.current_working_tokens() <= self.max_context_tokens:
            return stats

        # المرحلة 1: طمس المخرجات القديمة الضخمة
        stats["masked_tokens_saved"] = self._apply_stage1_observation_masking()

        # المرحلة 2: إذا ظل السياق فوق الميزانية، نرحّل أقدم الدورات غير المثبتة (Non-Pinned) للتلخيص
        evicted: list[AtomicTurnGroup] = []
        while self.current_working_tokens() > self.max_context_tokens:
            # البحث عن أقدم دورة غير مثبتة، مع ضمان عدم حذف آخر دورة حديثة أبداً
            candidate_idx = next(
                (
                    i
                    for i, t in enumerate(self._turns[:-1])
                    if not t.pinned
                ),
                None,
            )
            if candidate_idx is None:
                break
            evicted.append(self._turns.pop(candidate_idx))

        if evicted:
            stats["evicted_turns"] = len(evicted)
            self._rolling_summary = await self.summarizer.summarize_turns(
                self._rolling_summary, evicted
            )

        return stats

    async def build_kv_cache_friendly_messages(
        self, *, tenant_id: str, active_query: str
    ) -> list[dict[str, Any]]:
        """تجميع الموجه النهائي بترتيب يحافظ على ثبات البادئة (Prefix KV-Cache) بنسبة 100%"""
        await self.compact_if_needed()

        # 1. الرسالة الأولى: ثابتة 100% دائماً لتحقيق أعلى نسبة KV-Cache Hit!
        compiled: list[dict[str, Any]] = [
            {"role": str(MessageRole.SYSTEM), "content": self.static_system_prompt}
        ]

        # 2. استرجاع الذكريات طويلة المدى الخاصة بالمستأجر الحالي + الملخص المتدحرج
        memories = self.long_term_store.recall(tenant_id=tenant_id, query=active_query, top_k=2)
        context_blocks: list[str] = []

        if memories:
            mem_lines = "\n".join(f"  - [{m.category}] {m.content}" for m in memories)
            context_blocks.append(f"<retrieved_long_term_memory>\n{mem_lines}\n</retrieved_long_term_memory>")

        if self._rolling_summary:
            context_blocks.append(
                f"<compressed_history_state>\n  {self._rolling_summary}\n</compressed_history_state>"
            )

        if context_blocks:
            compiled.append(
                {
                    "role": str(MessageRole.SYSTEM),
                    "content": "\n".join(context_blocks),
                }
            )

        # 3. الذيل الديناميكي: الدورات الذرية النشطة (مضمونة الترابط بين المساعد والأدوات)
        for turn in self._turns:
            for msg in turn.messages:
                compiled.append(msg.to_api_dict())

        return compiled


# =====================================================================
# 4. محاكاة تشغيلية كاملة (Runnable Simulation)
# =====================================================================
class MockFastSummarizer:
    """محاكاة لنموذج سريع (مثل GPT-4o-mini أو Haiku) يلخص الدورات المرحّلة"""

    async def summarize_turns(
        self, previous_summary: str, evicted_turns: list[AtomicTurnGroup]
    ) -> str:
        await asyncio.sleep(0.02)
        turn_ids = [f"Turn #{t.turn_id}" for t in evicted_turns]
        new_notes = f"تم فحص السجلات الأولية وقاعدة البيانات في الدورات ({', '.join(turn_ids)})."
        return f"{previous_summary} | {new_notes}".strip(" |")


async def main() -> None:
    store = SemanticMemoryStore()
    # تخزين ذكريات لمستأجرين مختلفين لاختبار العزل والدقة
    store.remember(
        tenant_id="ORG-ALPHA",
        category="infra_fact",
        content="خادم قاعدة البيانات الأساسي هو db-primary-eu-west ويعمل بمنفذ 5432.",
    )
    store.remember(
        tenant_id="ORG-BETA",
        category="infra_fact",
        content="سر خطير خاص بشركة بيتا: الخادم في طوكيو.",
    )

    memory_harness = ContextMemoryHarness(
        static_system_prompt="أنت وكيل تشخيص أعطال سحابية صارم. التزم بالحقائق المسترجعة.",
        max_context_tokens=420,  # ميزانية صغيرة متعمدة لمشاهدة الضغط التلقائي أثناء العمل!
        keep_recent_unmasked_turns=1,
        mask_threshold_chars=120,
        summarizer=MockFastSummarizer(),
        long_term_store=store,
    )

    # 1. إضافة هدف المستخدم الأول (مثبت Pinned = True فلا يمسح أبداً!)
    memory_harness.add_user_goal("ابحث عن سبب بطء قاعدة البيانات الأساسية وعالجه.")

    # 2. محاكاة 3 دورات عمل للوكيل مع مخرجات أدوات ضخمة
    for step in range(1, 4):
        bulky_tool_output = (
            f"RAW_DIAGNOSTIC_DUMP_STEP_{step}: "
            + ("SELECT * FROM heavy_table WHERE latency > 900ms; " * 8)
        )
        memory_harness.add_agent_tool_cycle(
            assistant_thought=f"تفكير الخطوة {step}: سأفحص مؤشرات الأداء.",
            tool_calls=[
                ToolCallRef(
                    call_id=f"call_{step}",
                    tool_name=" inspect_db_node ",
                    arguments_json=f'{{"step": {step}}}',
                )
            ],
            tool_results=[(f"call_{step}", "inspect_db_node", bulky_tool_output)],
        )
        print(
            f"📥 بعد إضافة الدورة الذرية #{step + 1} | التوكنز قبل الضغط: {memory_harness.current_working_tokens()}"
        )
        stats = await memory_harness.compact_if_needed()
        print(
            f"   🗜️ إحصائيات الضغط الفوري: {stats} | التوكنز بعد الضغط: {memory_harness.current_working_tokens()}"
        )

    # 3. بناء الموجه النهائي الصديق للـ KV-Cache
    final_messages = await memory_harness.build_kv_cache_friendly_messages(
        tenant_id="ORG-ALPHA",
        active_query="أين يقع خادم قاعدة البيانات الأساسي؟",
    )

    print("\n✨ هيكل السياق النهائي المرسل للـ LLM (لاحظ ثبات البادئة، والملخص، وسلامة أزواج الأدوات):")
    for idx, m in enumerate(final_messages, 1):
        role = m["role"].upper()
        preview = m["content"][:110].replace("\n", " ")
        tc_info = f" [ToolCalls: {len(m['tool_calls'])}]" if "tool_calls" in m else ""
        tid_info = f" [ToolCallID: {m['tool_call_id']}]" if "tool_call_id" in m else ""
        print(f"  {idx:02d}. [{role}]{tc_info}{tid_info}: {preview}...")


if __name__ == "__main__":
    asyncio.run(main())
