#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 9: التدخل البشري ونقاط الحفظ وإدارة الجلسات الطويلة
==============================================================================

البرنامج الكامل للفصل 9 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
التدخل البشري ونقاط الحفظ وإدارة الجلسات الطويلة — Human-in-the-Loop, Checkpointing & Durable Execution.

التشغيل:
    python ch09_durable_hitl_checkpoints.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum, auto
import hashlib
import hmac
import json
import sqlite3
from typing import Any
import uuid
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. حالات التنفيذ المستدام وقرارات التدخل البشري (HITL Contracts)
# =====================================================================
class RunStatus(StrEnum):
    RUNNING = auto()
    SUSPENDED_FOR_APPROVAL = auto()  # مجمد على القرص بانتظار قرار بشري (0% RAM)
    COMPLETED = auto()
    FAILED = auto()


class HumanDecisionType(StrEnum):
    APPROVE = auto()              # موافقة على تنفيذ الأداة كما طلبها الوكيل
    EDIT_AND_APPROVE = auto()     # تعديل معطيات الأداة من قِبل الإنسان ثم التنفيذ
    REJECT_WITH_FEEDBACK = auto() # رفض تنفيذ الأداة وإرسال توجيه للوكيل لتغيير خطته


@dataclass(slots=True, frozen=True)
class PendingApprovalTicket:
    """التذكرة التي يصدرها الـ Harness لإرسالها إلى Slack أو لوحة تحكم المشرفين"""

    ticket_id: str
    thread_id: str
    checkpoint_id: str
    tool_name: str
    proposed_args: dict[str, Any]
    agent_reasoning: str
    risk_reason: str


@dataclass(slots=True, frozen=True)
class HumanApprovalInput:
    """القرار القادم من المشرف البشري لاستئناف الوكيل المجمد"""

    decision: HumanDecisionType
    reviewer_id: str
    modified_args: dict[str, Any] | None = None
    feedback_message: str = ""


# =====================================================================
# 2. نموذج حالة الجلسة ونقطة الحفظ الموقعة رقمياً (Signed Checkpoint)
# =====================================================================
class DurableAgentState[CustomStateT: BaseModel](BaseModel):
    """الحالة الشاملة القابلة للتسلسل (Serializable) التي تُحفظ في قاعدة البيانات"""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    step_number: int = 0
    status: RunStatus = RunStatus.RUNNING
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    domain_state: CustomStateT
    pending_tool_name: str | None = None
    pending_tool_args: dict[str, Any] | None = None
    pending_thought: str | None = None
    final_output: str | None = None


@dataclass(slots=True, frozen=True)
class CheckpointRecord:
    """سجل نقطة حفظ غير قابل للتغيير داخل سلسلة الزمن (Lineage Graph)"""

    checkpoint_id: str
    thread_id: str
    parent_checkpoint_id: str | None
    step_number: int
    status: RunStatus
    state_json: str
    hmac_signature: str


class SQLiteCheckpointStore:
    """مخزن نقاط حفظ مستدام (Durable Store) محمي بتوقيع HMAC-SHA256 وقفل تفاؤلي"""

    def __init__(self, db_path: str = ":memory:", secret_key: str = "harness-secret-key-314") -> None:
        self._conn = sqlite3.connect(db_path)
        self._secret = secret_key.encode("utf-8")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                step_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                state_json TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                is_consumed INTEGER DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def _sign(self, thread_id: str, step: int, state_json: str) -> str:
        payload = f"{thread_id}:{step}:{state_json}".encode("utf-8")
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def save_checkpoint[CustomStateT: BaseModel](
        self,
        state: DurableAgentState[CustomStateT],
        parent_checkpoint_id: str | None = None,
    ) -> CheckpointRecord:
        cp_id = f"cp_{uuid.uuid4().hex[:10]}"
        state_json = state.model_dump_json()
        sig = self._sign(state.thread_id, state.step_number, state_json)

        self._conn.execute(
            """
            INSERT INTO checkpoints
            (checkpoint_id, thread_id, parent_checkpoint_id, step_number, status, state_json, hmac_signature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cp_id,
                state.thread_id,
                parent_checkpoint_id,
                state.step_number,
                str(state.status),
                state_json,
                sig,
            ),
        )
        self._conn.commit()
        return CheckpointRecord(
            checkpoint_id=cp_id,
            thread_id=state.thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
            step_number=state.step_number,
            status=state.status,
            state_json=state_json,
            hmac_signature=sig,
        )

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointRecord:
        row = self._conn.execute(
            """
            SELECT checkpoint_id, thread_id, parent_checkpoint_id, step_number, status, state_json, hmac_signature
            FROM checkpoints WHERE checkpoint_id = ?
            """,
            (checkpoint_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"نقطة الحفظ '{checkpoint_id}' غير موجودة!")

        cp = CheckpointRecord(
            checkpoint_id=row[0],
            thread_id=row[1],
            parent_checkpoint_id=row[2],
            step_number=row[3],
            status=RunStatus(row[4]),
            state_json=row[5],
            hmac_signature=row[6],
        )
        # التحقق الجنائي من عدم تعرض الحالة للتلاعب في قاعدة البيانات!
        expected_sig = self._sign(cp.thread_id, cp.step_number, cp.state_json)
        if not hmac.compare_digest(cp.hmac_signature, expected_sig):
            raise SecurityError(
                f"🚨 تم اكتشاف تلاعب ببيانات نقطة الحفظ '{checkpoint_id}'! التوقيع الرقمي غير مطابق."
            )
        return cp

    def get_latest_checkpoint(self, thread_id: str) -> CheckpointRecord:
        row = self._conn.execute(
            """
            SELECT checkpoint_id FROM checkpoints
            WHERE thread_id = ? ORDER BY rowid DESC LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"لا توجد جلسة محفوظة بالمعرف '{thread_id}'.")
        return self.load_checkpoint(row[0])

    def consume_suspended_checkpoint_atomically(self, checkpoint_id: str) -> None:
        """قفل تفاؤلي ذري (Optimistic Concurrency Lock):

        يمنع مديرين بشريين من استئناف نفس التذكرة المعلقة مرتين في نفس اللحظة!
        """
        cursor = self._conn.execute(
            """
            UPDATE checkpoints SET is_consumed = 1
            WHERE checkpoint_id = ? AND status = ? AND is_consumed = 0
            """,
            (checkpoint_id, str(RunStatus.SUSPENDED_FOR_APPROVAL)),
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            raise RuntimeError(
                f"⚠️ Race Condition Prevented: التذكرة '{checkpoint_id}' تمت معالجتها واستئنافها مسبقاً!"
            )


class SecurityError(RuntimeError):
    pass


# =====================================================================
# 3. قلب النظام: DurableAgentHarness (التجميد، الاستئناف، والسفر عبر الزمن)
# =====================================================================
@dataclass(slots=True)
class GatedTool:
    name: str
    func: Callable[..., Awaitable[str]]
    requires_approval: bool = False
    risk_description: str = ""


@dataclass(slots=True)
class HarnessStepOutcome[CustomStateT: BaseModel]:
    """نتيجة تشغيل أو استئناف الوكيل: إما اكتمل، أو تجمد بانتظار موافقة بشرية"""

    status: RunStatus
    thread_id: str
    latest_checkpoint_id: str
    state: DurableAgentState[CustomStateT]
    approval_ticket: PendingApprovalTicket | None = None


class DurableAgentHarness[CustomStateT: BaseModel]:
    """إطار التحكم المستدام: يدعم التوقف الصفري (Zero-Memory HitL) والسفر عبر الزمن."""

    def __init__(
        self,
        *,
        store: SQLiteCheckpointStore,
        state_model: type[DurableAgentState[CustomStateT]],
        planner_fn: Callable[[DurableAgentState[CustomStateT]], Awaitable[dict[str, Any]]],
        tools: list[GatedTool],
        max_steps: int = 10,
    ) -> None:
        self.store = store
        self.state_model = state_model
        self.planner_fn = planner_fn
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    async def start_run(
        self, thread_id: str, user_goal: str, initial_domain_state: CustomStateT
    ) -> HarnessStepOutcome[CustomStateT]:
        """بدء جلسة عمل جديدة وحفظ نقطة الحفظ الابتدائية (Genesis Checkpoint)"""
        state = self.state_model(
            thread_id=thread_id,
            step_number=0,
            status=RunStatus.RUNNING,
            conversation_history=[{"role": "user", "content": user_goal}],
            domain_state=initial_domain_state,
        )
        cp = self.store.save_checkpoint(state, parent_checkpoint_id=None)
        return await self._drive_loop(state, current_cp_id=cp.checkpoint_id)

    async def _drive_loop(
        self, state: DurableAgentState[CustomStateT], current_cp_id: str
    ) -> HarnessStepOutcome[CustomStateT]:
        """حلقة التنفيذ المستدامة: تحفظ نقطة حفظ بعد كل خطوة، وتتجمد فوراً عند الأدوات الحساسة"""
        while state.step_number < self.max_steps:
            state.step_number += 1
            decision = await self.planner_fn(state)

            # الحالة 1: الوكيل أنجز المهمة
            if "final_answer" in decision:
                state.status = RunStatus.COMPLETED
                state.final_output = str(decision["final_answer"])
                state.conversation_history.append(
                    {"role": "assistant", "content": state.final_output}
                )
                cp = self.store.save_checkpoint(state, parent_checkpoint_id=current_cp_id)
                return HarnessStepOutcome(
                    status=RunStatus.COMPLETED,
                    thread_id=state.thread_id,
                    latest_checkpoint_id=cp.checkpoint_id,
                    state=state,
                )

            # الحالة 2: الوكيل يطلب تشغيل أداة
            tool_name = str(decision["tool_name"])
            tool_args = dict(decision.get("tool_args", {}))
            thought = str(decision.get("thought", ""))
            tool_spec = self.tools[tool_name]

            # 🛡️ فحص بوابة التدخل البشري (Human-in-the-Loop Gate)!
            if tool_spec.requires_approval:
                state.status = RunStatus.SUSPENDED_FOR_APPROVAL
                state.pending_tool_name = tool_name
                state.pending_tool_args = tool_args
                state.pending_thought = thought

                # حفظ الحالة المجمدة في قاعدة البيانات والخروج من الذاكرة فوراً!
                cp = self.store.save_checkpoint(state, parent_checkpoint_id=current_cp_id)
                ticket = PendingApprovalTicket(
                    ticket_id=f"tkt_{cp.checkpoint_id}",
                    thread_id=state.thread_id,
                    checkpoint_id=cp.checkpoint_id,
                    tool_name=tool_name,
                    proposed_args=tool_args,
                    agent_reasoning=thought,
                    risk_reason=tool_spec.risk_description,
                )
                return HarnessStepOutcome(
                    status=RunStatus.SUSPENDED_FOR_APPROVAL,
                    thread_id=state.thread_id,
                    latest_checkpoint_id=cp.checkpoint_id,
                    state=state,
                    approval_ticket=ticket,
                )

            # إذا كانت الأداة عادية (لا تتطلب موافقة)، ننفذها فوراً ونحفظ Checkpoint
            obs = await tool_spec.func(**tool_args)
            state.conversation_history.append(
                {"role": "assistant", "content": f"Thought: {thought} | Tool: {tool_name}"}
            )
            state.conversation_history.append(
                {"role": "tool", "content": obs}
            )
            cp = self.store.save_checkpoint(state, parent_checkpoint_id=current_cp_id)
            current_cp_id = cp.checkpoint_id

        state.status = RunStatus.FAILED
        cp = self.store.save_checkpoint(state, parent_checkpoint_id=current_cp_id)
        return HarnessStepOutcome(
            status=RunStatus.FAILED,
            thread_id=state.thread_id,
            latest_checkpoint_id=cp.checkpoint_id,
            state=state,
        )

    async def resume_from_human(
        self, thread_id: str, human_input: HumanApprovalInput
    ) -> HarnessStepOutcome[CustomStateT]:
        """إيقاظ الوكيل من القرص الصلب بعد ساعات أو أيام وتطبيق قرار المشرف البشري"""
        latest_cp = self.store.get_latest_checkpoint(thread_id)
        if latest_cp.status != RunStatus.SUSPENDED_FOR_APPROVAL:
            raise RuntimeError(f"الجلسة '{thread_id}' ليست في حالة انتظار موافقة!")

        # قفل ذري لمنع الاستئناف المزدوج لنفس التذكرة
        self.store.consume_suspended_checkpoint_atomically(latest_cp.checkpoint_id)

        # إعادة إحياء كائن الحالة المهيكل من JSON
        state = self.state_model.model_validate_json(latest_cp.state_json)
        assert state.pending_tool_name is not None
        assert state.pending_tool_args is not None

        tool_spec = self.tools[state.pending_tool_name]
        thought = state.pending_thought or ""

        # معالجة الأنماط الثلاثة لقرار الإنسان عبر Structural Pattern Matching
        match human_input.decision:
            case HumanDecisionType.APPROVE:
                obs = await tool_spec.func(**state.pending_tool_args)
                audit_msg = (
                    f"[✅ HUMAN APPROVED by {human_input.reviewer_id}]: Tool output -> {obs}"
                )

            case HumanDecisionType.EDIT_AND_APPROVE:
                safe_args = human_input.modified_args or state.pending_tool_args
                obs = await tool_spec.func(**safe_args)
                audit_msg = (
                    f"[✏️ HUMAN MODIFIED ARGS & APPROVED by {human_input.reviewer_id} "
                    f"(New Args: {safe_args})]: Tool output -> {obs}"
                )

            case HumanDecisionType.REJECT_WITH_FEEDBACK:
                # لا ننفذ الأداة أبداً! نحول رفض الإنسان وتوجيهه إلى ملاحظة يقرؤها الوكيل ليغير خطته
                audit_msg = (
                    f"[🛑 HUMAN REJECTED TOOL EXECUTION by {human_input.reviewer_id}]: "
                    f"Tool '{tool_spec.name}' was NOT executed. Supervisor instruction: "
                    f"'{human_input.feedback_message}'. Adjust your plan accordingly."
                )

        # تنظيف حقول التعليق واستكمال رحلة الوكيل
        state.conversation_history.append(
            {"role": "assistant", "content": f"Thought: {thought} | Requested: {tool_spec.name}"}
        )
        state.conversation_history.append({"role": "tool", "content": audit_msg})
        state.pending_tool_name = None
        state.pending_tool_args = None
        state.pending_thought = None
        state.status = RunStatus.RUNNING

        cp = self.store.save_checkpoint(state, parent_checkpoint_id=latest_cp.checkpoint_id)
        return await self._drive_loop(state, current_cp_id=cp.checkpoint_id)

    async def fork_time_travel(
        self,
        source_checkpoint_id: str,
        *,
        new_thread_id: str,
        domain_state_patch: dict[str, Any] | None = None,
        injected_instruction: str | None = None,
    ) -> HarnessStepOutcome[CustomStateT]:
        """السفر عبر الزمن (Time-Travel Forking):

        العودة لأي نقطة حفظ تاريخية، تعديل الحالة أو حقن توجيه جديد، والانطلاق في مسار زمني موازٍ!
        """
        historical_cp = self.store.load_checkpoint(source_checkpoint_id)
        forked_state = self.state_model.model_validate_json(historical_cp.state_json)

        forked_state.thread_id = new_thread_id
        forked_state.status = RunStatus.RUNNING
        forked_state.pending_tool_name = None
        forked_state.pending_tool_args = None

        if domain_state_patch:
            updated_domain = forked_state.domain_state.model_copy(update=domain_state_patch)
            forked_state.domain_state = updated_domain

        if injected_instruction:
            forked_state.conversation_history.append(
                {"role": "user", "content": f"[⚡ TIME-TRAVEL OVERRIDE]: {injected_instruction}"}
            )

        # حفظ نقطة بداية الفرع الجديد مع ربطها بنقطة الحفظ التاريخية كأب (Parent Lineage)
        fork_cp = self.store.save_checkpoint(
            forked_state, parent_checkpoint_id=historical_cp.checkpoint_id
        )
        return await self._drive_loop(forked_state, current_cp_id=fork_cp.checkpoint_id)


# =====================================================================
# 4. محاكاة كاملة: التجميد، رفض المدير، تعديل المبلغ، ثم السفر عبر الزمن!
# =====================================================================
class RefundWorkflowState(BaseModel):
    customer_id: str
    requested_refund_usd: float
    refund_completed: bool = False


async def simulated_finops_planner(
    state: DurableAgentState[RefundWorkflowState],
) -> dict[str, Any]:
    """عقل الوكيل المالي الذي يتفاعل مع تاريخ المحادثة وتوجيهات المشرف البشري"""
    await asyncio.sleep(0.02)
    last_msg = state.conversation_history[-1]["content"]

    # الخطوة 1: فحص سجل العميل أولاً (أداة آمنة لا تتطلب موافقة)
    if state.step_number == 1:
        return {
            "thought": "سأتحقق أولاً من حالة اشتراك العميل وسجله المالي.",
            "tool_name": "check_customer_tier",
            "tool_args": {"customer_id": state.domain_state.customer_id},
        }

    # إذا تم رفض الأداة سابقاً من المدير، يقرأ الوكيل ملاحظة المدير ويعدل خطته!
    if "HUMAN REJECTED" in last_msg:
        return {
            "thought": "رفض المدير الاسترداد الكامل وطلب خصم رسوم المعالجة ليصبح 1200 دولار.",
            "tool_name": "execute_wire_refund",
            "tool_args": {"customer_id": state.domain_state.customer_id, "amount_usd": 1200.0},
        }

    # إذا تمت الموافقة وتنفيذ التحويل بنجاح، ينهي الوكيل المهمة
    if "WIRE_TRANSFER_SUCCESS" in last_msg:
        state.domain_state.refund_completed = True
        return {
            "final_answer": f"تم إتمام عملية الاسترداد للعميل {state.domain_state.customer_id} بنجاح. التفاصيل: {last_msg}"
        }

    # الخطوة 2: طلب تنفيذ التحويل المالي الكبير (أداة حساسة ستؤدي لتجميد الوكيل!)
    return {
        "thought": "العميل من الفئة الذهبية. سأطلب تحويل مبلغ الاسترداد المطلوب.",
        "tool_name": "execute_wire_refund",
        "tool_args": {
            "customer_id": state.domain_state.customer_id,
            "amount_usd": state.domain_state.requested_refund_usd,
        },
    }


async def tool_check_tier(customer_id: str) -> str:
    return json.dumps({"customer_id": customer_id, "tier": "ENTERPRISE_GOLD", "sla_breached": True})


async def tool_wire_refund(customer_id: str, amount_usd: float) -> str:
    return f"WIRE_TRANSFER_SUCCESS: Transferred ${amount_usd:.2f} to '{customer_id}'."


async def main() -> None:
    store = SQLiteCheckpointStore(db_path=":memory:")
    harness = DurableAgentHarness[RefundWorkflowState](
        store=store,
        state_model=DurableAgentState[RefundWorkflowState],
        planner_fn=simulated_finops_planner,
        tools=[
            GatedTool(name="check_customer_tier", func=tool_check_tier, requires_approval=False),
            GatedTool(
                name="execute_wire_refund",
                func=tool_wire_refund,
                requires_approval=True,
                risk_description="تحويل مالي خارجي يتجاوز سقف الـ 1000 دولار.",
            ),
        ],
    )

    print("1️⃣ بدء تشغيل الوكيل (سينفذ أداة الفحص الآمنة ثم يتجمد تلقائياً عند أداة التحويل المالي):")
    outcome1 = await harness.start_run(
        thread_id="thread_fin_01",
        user_goal="عالج طلب الاسترداد للعميل CUST-88 بمبلغ 1500 دولار.",
        initial_domain_state=RefundWorkflowState(customer_id="CUST-88", requested_refund_usd=1500.0),
    )
    print(f"   ⏸️ حالة الوكيل الآن : {outcome1.status.upper()} (Checkpoint: {outcome1.latest_checkpoint_id})")
    print(f"   🎫 تذكرة الموافقة   : الأداة={outcome1.approval_ticket.tool_name} | المعطيات={outcome1.approval_ticket.proposed_args}\n")  # type: ignore[union-attr]

    # حفظ معرف نقطة الحفظ الأولى لاستخدامه لاحقاً في السفر عبر الزمن!
    suspended_cp_id = outcome1.latest_checkpoint_id

    print("2️⃣ بعد 4 ساعات: المدير المالي يرفض مبلغ 1500$ ويكتب توجيهاً للوكيل (REJECT_WITH_FEEDBACK):")
    outcome2 = await harness.resume_from_human(
        thread_id="thread_fin_01",
        human_input=HumanApprovalInput(
            decision=HumanDecisionType.REJECT_WITH_FEEDBACK,
            reviewer_id="CFO_SARAH",
            feedback_message="لا تحول 1500 دولار؛ اخصم 300 دولار رسوم استخدام وأعد المحاولة بـ 1200 دولار.",
        ),
    )
    print(f"   ⏸️ الوكيل قرأ التوجيه، عدّل خطته، وتجمد مجدداً للموافقة على المبلغ الجديد:")
    print(f"   🎫 التذكرة الجديدة  : المعطيات المقترحة من الوكيل={outcome2.approval_ticket.proposed_args}\n")  # type: ignore[union-attr]

    print("3️⃣ المدير المالي يقرر تعديل المبلغ بنفسه إلى 1250$ والموافقة الفورية (EDIT_AND_APPROVE):")
    outcome3 = await harness.resume_from_human(
        thread_id="thread_fin_01",
        human_input=HumanApprovalInput(
            decision=HumanDecisionType.EDIT_AND_APPROVE,
            reviewer_id="CFO_SARAH",
            modified_args={"customer_id": "CUST-88", "amount_usd": 1250.0},
        ),
    )
    print(f"   🏁 الحالة النهائية  : {outcome3.status.upper()}")
    print(f"   💡 المخرج النهائي   : {outcome3.state.final_output}\n")

    print("4️⃣ السفر عبر الزمن (Time-Travel Fork): العودة لنقطة الحفظ الأولى وتفريع مسار موازٍ (thread_fin_02_vip):")
    fork_outcome = await harness.fork_time_travel(
        source_checkpoint_id=suspended_cp_id,
        new_thread_id="thread_fin_02_vip",
        domain_state_patch={"requested_refund_usd": 900.0},
        injected_instruction="العميل تنازل عن جزء من المبلغ وأصبح المطلوب 900 دولار فقط.",
    )
    print(f"   ⚡ تم تفريع المسار الموازي بنجاح! المعطيات في المسار الجديد: {fork_outcome.approval_ticket.proposed_args}")  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
