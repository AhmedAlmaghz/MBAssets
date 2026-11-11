#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 6: حلقة التنفيذ وآلات الحالة الموجهة
==============================================================================

البرنامج الكامل للفصل 6 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
حلقة التنفيذ وآلات الحالة الموجهة — The Execution Loop, ReAct & State Graph Machines.

التشغيل:
    python ch06_execution_loop_state_graph.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import hashlib
import time
from pydantic import BaseModel, ConfigDict, Field

# =====================================================================
# 1. تعريف الأنواع الأساسية لمحرك آلات الحالة (Python 3.14+ Type Aliases)
# =====================================================================
type NodeId = str
type NodeHandler[StateT: BaseModel] = Callable[[StateT], Awaitable[StateT]]
type EdgeRouter[StateT: BaseModel] = Callable[[StateT], NodeId | Awaitable[NodeId]]

END_NODE: NodeId = "__END__"


class GraphStopStatus(StrEnum):
    """الحالات النهائية الحتمية لتوقف محرك الرسم البياني"""

    REACHED_END = auto()          # وصل إلى عقدة النهاية الطبيعية بنجاح
    NODE_QUOTA_EXCEEDED = auto()  # تجاوز الحد الأقصى لزيارة عقدة معينة (Ping-Pong Defense)
    MAX_TRANSITIONS_HIT = auto()  # تجاوز الحد الأقصى لإجمالي الانتقالات في الرسم البياني
    TIMEOUT = auto()              # انتهاء المهلة الزمنية الشاملة


@dataclass(slots=True, frozen=True)
class TransitionAuditEntry:
    """سجل تدقيق غير قابل للتغيير لكل خطوة انتقال داخل الرسم البياني (Observability)"""

    step_index: int
    from_node: NodeId
    to_node: NodeId
    duration_ms: float
    state_hash_after: str
    note: str = ""


@dataclass(slots=True)
class NodeSpec[StateT: BaseModel]:
    """مواصفات العقدة التنفيذية وقيود زيارتها"""

    node_id: NodeId
    handler: NodeHandler[StateT]
    max_visits: int = 5
    quota_fallback_node: NodeId | None = None  # مسار الهروب الآمن إذا علقت العقدة في حلقة!


@dataclass(slots=True)
class GraphExecutionResult[StateT: BaseModel]:
    """المخرج النهائي الشامل لتشغيل آلة الحالة"""

    status: GraphStopStatus
    final_state: StateT
    total_steps: int
    total_elapsed_ms: float
    node_visit_counts: dict[NodeId, int]
    audit_trail: list[TransitionAuditEntry] = field(default_factory=list)


# =====================================================================
# 2. قلب النظام: StateGraphHarness (محرك الرسوم البيانية الموجهة للوكلاء)
# =====================================================================
class StateGraphHarness[StateT: BaseModel]:
    """إطار تحكم مبني على آلات الحالة الموجهة بالحلقات (Directed Cyclic State Machine).

    يجمع بين مرونة قرارات الوكيل وصرامة مسارات العمل الهندسية.
    """

    def __init__(
        self,
        *,
        max_total_transitions: int = 20,
        execution_timeout_sec: float = 60.0,
    ) -> None:
        self.max_total_transitions = max_total_transitions
        self.execution_timeout_sec = execution_timeout_sec

        self._nodes: dict[NodeId, NodeSpec[StateT]] = {}
        self._direct_edges: dict[NodeId, NodeId] = {}
        self._conditional_edges: dict[NodeId, tuple[EdgeRouter[StateT], set[NodeId]]] = {}
        self._entry_node: NodeId | None = None
        self._is_compiled: bool = False

    def add_node(
        self,
        node_id: NodeId,
        handler: NodeHandler[StateT],
        *,
        max_visits: int = 3,
        quota_fallback_node: NodeId | None = None,
    ) -> None:
        """إضافة عقدة تنفيذية مع تحديد سقف زياراتها لمنع حلقات Ping-Pong"""
        if node_id == END_NODE:
            raise ValueError(f"لا يمكن تسمية عقدة بالاسم المحجوز '{END_NODE}'.")
        self._nodes[node_id] = NodeSpec(
            node_id=node_id,
            handler=handler,
            max_visits=max_visits,
            quota_fallback_node=quota_fallback_node,
        )
        self._is_compiled = False

    def set_entry_point(self, node_id: NodeId) -> None:
        """تحديد العقدة الأولى التي يبدأ منها التنفيذ"""
        self._entry_node = node_id
        self._is_compiled = False

    def add_edge(self, from_node: NodeId, to_node: NodeId) -> None:
        """إضافة حافة انتقال حتمية مباشرة (from_node -> to_node)"""
        if from_node in self._conditional_edges:
            raise ValueError(f"العقدة '{from_node}' تمتلك بالفعل حافة شرطية!")
        self._direct_edges[from_node] = to_node
        self._is_compiled = False

    def add_conditional_edges(
        self,
        from_node: NodeId,
        router: EdgeRouter[StateT],
        allowed_targets: set[NodeId],
    ) -> None:
        """إضافة حافة شرطية محمية بقائمة بيضاء (Whitelist) للعقد المسموح الانتقال إليها فقط!"""
        if from_node in self._direct_edges:
            raise ValueError(f"العقدة '{from_node}' تمتلك بالفعل حافة مباشرة!")
        if not allowed_targets:
            raise ValueError("يجب تحديد عقدة هدف واحدة على الأقل في allowed_targets.")
        self._conditional_edges[from_node] = (router, set(allowed_targets))
        self._is_compiled = False

    def compile(self) -> None:
        """التحقق الهندسي الصارم من سلامة الرسم البياني قبل السماح بتشغيله"""
        if not self._entry_node or self._entry_node not in self._nodes:
            raise RuntimeError(f"نقطة البداية '{self._entry_node}' غير معرفة في العقد!")

        valid_destinations = set(self._nodes.keys()) | {END_NODE}

        for node_id, spec in self._nodes.items():
            # 1. التأكد من أن كل عقدة لها مخرج (إما حافة مباشرة أو شرطية)
            has_direct = node_id in self._direct_edges
            has_cond = node_id in self._conditional_edges
            if not (has_direct or has_cond):
                raise RuntimeError(
                    f"خطأ معماري في الرسم البياني: العقدة '{node_id}' طريق مسدود (Dead-End) بلا حواف خارجة!"
                )

            # 2. التحقق من صحة عقدة الهروب الاحتياطية إن وجدت
            if spec.quota_fallback_node and spec.quota_fallback_node not in valid_destinations:
                raise RuntimeError(
                    f"عقدة الهروب '{spec.quota_fallback_node}' للعقدة '{node_id}' غير موجودة!"
                )

        # 3. التحقق من أن جميع الحواف المباشرة والشرطية تشير إلى عقد موجودة فعلاً
        for src, dst in self._direct_edges.items():
            if dst not in valid_destinations:
                raise RuntimeError(f"الحافة من '{src}' تشير إلى عقدة غير موجودة '{dst}'!")

        for src, (_, targets) in self._conditional_edges.items():
            invalid = targets - valid_destinations
            if invalid:
                raise RuntimeError(f"الحافة الشرطية من '{src}' تتضمن أهدافاً غير موجودة: {invalid}")

        self._is_compiled = True

    @staticmethod
    def _compute_state_hash(state: BaseModel) -> str:
        """حساب بصمة سريعة للحالة لتتبع متى وكيف تغيرت البيانات بين العقد"""
        raw = state.model_dump_json()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]

    async def _resolve_next_node(self, current_node: NodeId, state: StateT) -> NodeId:
        """تحديد العقدة التالية والتحقق من عدم اختراق القائمة البيضاء للانتقالات"""
        if current_node in self._direct_edges:
            return self._direct_edges[current_node]

        router, allowed_targets = self._conditional_edges[current_node]
        result = router(state)
        next_node = await result if asyncio.iscoroutine(result) else result

        if next_node not in allowed_targets:
            raise RuntimeError(
                f"🚨 Security / Graph Violation: حاول الموجه في العقدة '{current_node}' القفز إلى "
                f"'{next_node}' وهو خارج القائمة البيضاء المسموحة: {allowed_targets}!"
            )
        return next_node

    async def run(self, initial_state: StateT) -> GraphExecutionResult[StateT]:
        """تشغيل آلة الحالة تحت رقابة صارمة على الزيارات والوقت والانتقالات"""
        if not self._is_compiled:
            self.compile()

        assert self._entry_node is not None
        t_start = time.perf_counter()
        current_node: NodeId = self._entry_node
        state: StateT = initial_state.model_copy(deep=True)

        visit_counts: dict[NodeId, int] = {n: 0 for n in self._nodes}
        audit_trail: list[TransitionAuditEntry] = []
        steps = 0

        try:
            async with asyncio.timeout(self.execution_timeout_sec):
                while current_node != END_NODE:
                    if steps >= self.max_total_transitions:
                        return GraphExecutionResult(
                            status=GraphStopStatus.MAX_TRANSITIONS_HIT,
                            final_state=state,
                            total_steps=steps,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            node_visit_counts=visit_counts,
                            audit_trail=audit_trail,
                        )

                    spec = self._nodes[current_node]

                    # فحص ميزانية زيارة العقدة (حاجز الحماية ضد Ping-Pong Loops)
                    if visit_counts[current_node] >= spec.max_visits:
                        if spec.quota_fallback_node:
                            fallback = spec.quota_fallback_node
                            audit_trail.append(
                                TransitionAuditEntry(
                                    step_index=steps + 1,
                                    from_node=current_node,
                                    to_node=fallback,
                                    duration_ms=0.0,
                                    state_hash_after=self._compute_state_hash(state),
                                    note=f"Quota ({spec.max_visits}) hit! Rerouted to fallback '{fallback}'.",
                                )
                            )
                            current_node = fallback
                            continue

                        return GraphExecutionResult(
                            status=GraphStopStatus.NODE_QUOTA_EXCEEDED,
                            final_state=state,
                            total_steps=steps,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            node_visit_counts=visit_counts,
                            audit_trail=audit_trail,
                        )

                    # تنفيذ العقدة الحالية وقياس أدائها
                    steps += 1
                    visit_counts[current_node] += 1
                    node_t0 = time.perf_counter()

                    updated_state = await spec.handler(state)
                    # التحقق الصارم من أن العقدة أعادت كائن حالة سليم
                    state = updated_state

                    next_node = await self._resolve_next_node(current_node, state)
                    node_ms = round((time.perf_counter() - node_t0) * 1000, 2)

                    audit_trail.append(
                        TransitionAuditEntry(
                            step_index=steps,
                            from_node=current_node,
                            to_node=next_node,
                            duration_ms=node_ms,
                            state_hash_after=self._compute_state_hash(state),
                            note=f"Visit #{visit_counts[current_node]}/{spec.max_visits}",
                        )
                    )
                    current_node = next_node

                return GraphExecutionResult(
                    status=GraphStopStatus.REACHED_END,
                    final_state=state,
                    total_steps=steps,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    node_visit_counts=visit_counts,
                    audit_trail=audit_trail,
                )

        except TimeoutError:
            return GraphExecutionResult(
                status=GraphStopStatus.TIMEOUT,
                final_state=state,
                total_steps=steps,
                total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                node_visit_counts=visit_counts,
                audit_trail=audit_trail,
            )


# =====================================================================
# 3. تطبيق واقعي: وكيل هندسة البرمجيات ذاتي الإصلاح (Self-Healing SWE Agent)
# =====================================================================
class CodePatchState(BaseModel):
    """الحالة المشتركة المهيكلة التي تنتقل عبر عقد الرسم البياني"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    bug_report: str
    plan: str = ""
    proposed_code: str = ""
    patch_iteration: int = 0
    tests_passed: bool = False
    test_error_log: str = ""
    security_approved: bool = False
    final_resolution: str = ""
    diagnostic_notes: list[str] = Field(default_factory=list)


# --- تعريف العقد التنفيذية (Nodes) ---
async def node_plan_fix(state: CodePatchState) -> CodePatchState:
    await asyncio.sleep(0.02)
    state.plan = "1. إضافة فحص القسمة على صفر. 2. التحقق عبر اختبارات الوحدة."
    state.diagnostic_notes.append("تم وضع خطة الإصلاح الهندسية.")
    return state


async def node_write_patch(state: CodePatchState) -> CodePatchState:
    await asyncio.sleep(0.03)
    state.patch_iteration += 1

    # محاكاة واقعية: في المحاولة الأولى يكتب الوكيل كوداً فيه خطأ بسيط، وفي المحاولة الثانية يصلحه!
    if state.patch_iteration == 1:
        state.proposed_code = "def calc_ratio(a, b):\n    return a / b  # نسي معالجة الصفر!"
        state.diagnostic_notes.append("المحاولة #1: تم توليد الكود الأولي.")
    else:
        state.proposed_code = (
            "def calc_ratio(a: float, b: float) -> float:\n"
            "    if b == 0:\n        return 0.0\n    return a / b"
        )
        state.diagnostic_notes.append(
            f"المحاولة #{state.patch_iteration}: تم إصلاح الكود بناءً على سجل خطأ الاختبار."
        )
    return state


async def node_run_sandbox_tests(state: CodePatchState) -> CodePatchState:
    await asyncio.sleep(0.03)
    if "if b == 0:" not in state.proposed_code:
        state.tests_passed = False
        state.test_error_log = "ZeroDivisionError: float division by zero on test_zero_denominator"
        state.diagnostic_notes.append("❌ فشل الاختبار في صندوق الرمل!")
    else:
        state.tests_passed = True
        state.test_error_log = ""
        state.diagnostic_notes.append("✅ نجحت جميع اختبارات الوحدة في صندوق الرمل!")
    return state


async def node_security_gate(state: CodePatchState) -> CodePatchState:
    await asyncio.sleep(0.02)
    # فحص حتمي: التأكد من عدم وجود دوال خطيرة مثل eval أو exec
    state.security_approved = "eval(" not in state.proposed_code
    state.final_resolution = "PR Ready: تم إصلاح الثغرة واجتياز الاختبارات والفحص الأمني."
    return state


async def node_escalate_to_human(state: CodePatchState) -> CodePatchState:
    """عقدة الهروب الآمن (Fallback Node) إذا فشل الوكيل في إصلاح الكود بعد استنفاد محاولاته"""
    state.final_resolution = (
        f"⚠️ Escalated to Senior Engineer: استنفد الوكيل محاولات الإصلاح. آخر خطأ: {state.test_error_log}"
    )
    return state


# --- تعريف الموجه الشرطي (Conditional Router) ---
def route_after_tests(state: CodePatchState) -> NodeId:
    """الـ Harness يقرر المسار حتمياً بناءً على نتيجة الاختبار (لا يمكن للوكيل تزويرها!)"""
    if state.tests_passed:
        return "security_gate"
    return "write_patch"  # حلقة عودة (Cycle) لإصلاح الكود!


async def main() -> None:
    # 1. إنشاء وبناء الرسم البياني للحالة
    graph = StateGraphHarness[CodePatchState](max_total_transitions=15)

    graph.add_node("plan_fix", node_plan_fix, max_visits=1)
    graph.add_node(
        "write_patch",
        node_write_patch,
        max_visits=3,  # نسمح بـ 3 محاولات إصلاح كحد أقصى، وإلا ننتقل لعقدة التصعيد البشري!
        quota_fallback_node="escalate_human",
    )
    graph.add_node("run_tests", node_run_sandbox_tests, max_visits=3)
    graph.add_node("security_gate", node_security_gate, max_visits=1)
    graph.add_node("escalate_human", node_escalate_to_human, max_visits=1)

    # 2. ربط الحواف الحتمية والشرطية
    graph.set_entry_point("plan_fix")
    graph.add_edge("plan_fix", "write_patch")
    graph.add_edge("write_patch", "run_tests")

    # حافة شرطية محمية بقائمة بيضاء صارمة
    graph.add_conditional_edges(
        "run_tests",
        router=route_after_tests,
        allowed_targets={"write_patch", "security_gate"},
    )

    graph.add_edge("security_gate", END_NODE)
    graph.add_edge("escalate_human", END_NODE)

    # 3. تشغيل آلة الحالة
    initial = CodePatchState(bug_report="Crash in calc_ratio when denominator is 0.")
    result = await graph.run(initial)

    print(f"🏁 حالة انتهاء الرسم البياني : {result.status.upper()}")
    print(f"⏱️ إجمالي الخطوات والزمن   : {result.total_steps} خطوات في {result.total_elapsed_ms} ms")
    print(f"📊 عداد زيارات العقد       : {result.node_visit_counts}")
    print(f"💡 القرار النهائي           : {result.final_state.final_resolution}\n")

    print("📜 سجل التدقيق الزمني للانتقالات (State Graph Audit Trail):")
    for entry in result.audit_trail:
        print(
            f"   Step {entry.step_index:02d}: [{entry.from_node}] ──► [{entry.to_node}] "
            f"| ({entry.duration_ms}ms) | StateHash: {entry.state_hash_after} | {entry.note}"
        )


if __name__ == "__main__":
    asyncio.run(main())
