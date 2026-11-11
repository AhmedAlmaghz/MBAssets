# -*- coding: utf-8 -*-
"""
import asyncio

مقتطف من الفصل 15 — «المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch15_financial_swarm.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

import asyncio
import hashlib
import hmac
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. الاستثناءات ونماذج الحالة المزودة بـ TTL و Idempotency
# =====================================================================

class StaleProposalExpiredError(RuntimeError):
    """يُرفع عندما يحاول مسؤول الامتثال اعتماد مقترح مالي تجاوز صلاحيته الزمنية (TTL)."""


class WorkflowStatus(StrEnum):
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"
    EXPIRED_STALE_PROPOSAL = "EXPIRED_STALE_PROPOSAL"


class FinancialWorkflowCheckpointTTL(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    ticker: str
    proposed_notional_usd: float = Field(gt=0.0)
    status: WorkflowStatus
    created_at_epoch: float
    expires_at_epoch: float
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    last_idempotency_key: str | None = None
    audit_trail: list[str]


class HumanDecisionWithIdempotency(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(
        min_length=8,
        description="مفتاح عدم تكرار فريد يولده العميل (UUID) لحماية إعادة المحاولة الشبكية",
    )
    decision_type: Literal["APPROVE", "REJECT"]
    compliance_notes: str = Field(min_length=5)


# =====================================================================
# 2. حارس الصلاحية الزمنية وعدم التكرار (TTL & Idempotency Guard Harness)
# =====================================================================

class ApprovalTTLAndIdempotencyHarness:
    """
    يدير نقاط الحفظ الموقعة بـ HMAC-SHA256، ويبطل المقترحات منتهية الصلاحية (TTL)،
    ويضمن سلامة إعادة المحاولة الشبكية عبر Idempotency Keys.
    """

    def __init__(self, secret_key: str, default_ttl_sec: float = 900.0) -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._default_ttl_sec = default_ttl_sec
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ttl_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_sig TEXT NOT NULL
                )
                """
            )

    def _sign(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes, payload_json.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _save(self, ckpt: FinancialWorkflowCheckpointTTL) -> None:
        payload = ckpt.model_dump_json()
        sig = self._sign(payload)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO ttl_workflows (workflow_id, tenant_id, status, payload_json, hmac_sig)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_sig = excluded.hmac_sig
                """,
                (ckpt.workflow_id, ckpt.tenant_id, ckpt.status.value, payload, sig),
            )

    def _load(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpointTTL:
        row = self._conn.execute(
            "SELECT payload_json, hmac_sig FROM ttl_workflows WHERE workflow_id = ? AND tenant_id = ?",
            (workflow_id, tenant_id),
        ).fetchone()
        if not row:
            raise KeyError(f"Workflow '{workflow_id}' not found.")
        payload, sig = row
        if not hmac.compare_digest(sig, self._sign(payload)):
            raise RuntimeError("HMAC Signature Tampering Detected!")
        return FinancialWorkflowCheckpointTTL.model_validate_json(payload)

    def create_suspended_proposal(
        self,
        tenant_id: str,
        requester_user_id: str,
        ticker: str,
        notional_usd: float,
        ttl_override_sec: float | None = None,
    ) -> FinancialWorkflowCheckpointTTL:
        now = time.time()
        ttl = ttl_override_sec if ttl_override_sec is not None else self._default_ttl_sec
        wf_id = f"WF-TTL-{uuid.uuid4().hex[:6].upper()}"

        ckpt = FinancialWorkflowCheckpointTTL(
            workflow_id=wf_id,
            tenant_id=tenant_id,
            requester_user_id=requester_user_id,
            ticker=ticker,
            proposed_notional_usd=notional_usd,
            status=WorkflowStatus.SUSPENDED_FOR_HITL,
            created_at_epoch=now,
            expires_at_epoch=now + ttl,
            audit_trail=[
                f"تم تعليق المقترح ({ticker} - ${notional_usd:,.2f}) بصلاحية زمنية قدرها {ttl:.1f} ثانية."
            ],
        )
        self._save(ckpt)
        return ckpt

    def process_human_decision(
        self,
        tenant_id: str,
        workflow_id: str,
        approver_user_id: str,
        decision: HumanDecisionWithIdempotency,
    ) -> FinancialWorkflowCheckpointTTL:
        current = self._load(workflow_id, tenant_id)

        # 1. فحص مفتاح عدم التكرار (Idempotency Replay Check)
        # إذا كان هذا الطلب إعادة إرسال شبكية لنفس القرار المنفذ سابقاً، نعيد نفس اللقطة بأمان!
        if (
            current.last_idempotency_key is not None
            and current.last_idempotency_key == decision.idempotency_key
        ):
            return current

        # إذا كان سير العمل قد عولج بمفتاح مختلف، نمنع التنفيذ المزدوج
        if current.status != WorkflowStatus.SUSPENDED_FOR_HITL:
            raise RuntimeError(
                f"لا يمكن تعديل سير العمل '{workflow_id}' لأن حالته النهائية هي: {current.status}"
            )

        # 2. فحص الصلاحية الزمنية للمقترح المالي (TTL Stale Proposal Check)
        now = time.time()
        if now > current.expires_at_epoch:
            delay_sec = now - current.expires_at_epoch
            expired_ckpt = FinancialWorkflowCheckpointTTL(
                workflow_id=current.workflow_id,
                tenant_id=current.tenant_id,
                requester_user_id=current.requester_user_id,
                ticker=current.ticker,
                proposed_notional_usd=current.proposed_notional_usd,
                status=WorkflowStatus.EXPIRED_STALE_PROPOSAL,
                created_at_epoch=current.created_at_epoch,
                expires_at_epoch=current.expires_at_epoch,
                audit_trail=[
                    *current.audit_trail,
                    f"⏰ انتهت الصلاحية الزمنية للمقترح قبل {delay_sec:.2f} ثانية. تم إبطال الصفقة تلقائياً.",
                ],
            )
            self._save(expired_ckpt)
            raise StaleProposalExpiredError(
                f"انتهت صلاحية مقترح السهم '{current.ticker}'! "
                "تغيرت أسعار السوق؛ يرجى إعادة تشغيل سرب البحث المالي."
            )

        # 3. تنفيذ الاعتماد أو الرفض ضمن النافذة الزمنية الصالحة
        new_status = (
            WorkflowStatus.APPROVED_AND_EXECUTED
            if decision.decision_type == "APPROVE"
            else WorkflowStatus.REJECTED_BY_COMPLIANCE
        )
        exec_usd = (
            current.proposed_notional_usd
            if decision.decision_type == "APPROVE"
            else None
        )

        updated = FinancialWorkflowCheckpointTTL(
            workflow_id=current.workflow_id,
            tenant_id=current.tenant_id,
            requester_user_id=current.requester_user_id,
            ticker=current.ticker,
            proposed_notional_usd=current.proposed_notional_usd,
            status=new_status,
            created_at_epoch=current.created_at_epoch,
            expires_at_epoch=current.expires_at_epoch,
            executed_notional_usd=exec_usd,
            approver_user_id=approver_user_id,
            last_idempotency_key=decision.idempotency_key,
            audit_trail=[
                *current.audit_trail,
                f"✅ تم تنفيذ القرار ({decision.decision_type}) بواسطة '{approver_user_id}' "
                f"[Idempotency-Key: {decision.idempotency_key}].",
            ],
        )
        self._save(updated)
        return updated


async def verify_ttl_and_idempotency() -> None:
    guard = ApprovalTTLAndIdempotencyHarness(secret_key="master-secret-key-2026")

    print("=" * 80)
    print("🧪 اختبار التحدي الختامي: TTL Expiration & Idempotent Network Replay")
    print("=" * 80)

    # 1. اختبار مقترح تنتهي صلاحيته خلال 0.05 ثانية (محاكاة تأخر المسؤول)
    stale_wf = guard.create_suspended_proposal(
        tenant_id="TENANT-01",
        requester_user_id="analyst_omar",
        ticker="AAPL",
        notional_usd=150_000.0,
        ttl_override_sec=0.05,
    )
    await asyncio.sleep(0.08)  # تجاوز النافذة الزمنية المسموحة!

    try:
        guard.process_human_decision(
            tenant_id="TENANT-01",
            workflow_id=stale_wf.workflow_id,
            approver_user_id="officer_sara",
            decision=HumanDecisionWithIdempotency(
                idempotency_key="IDEMP-KEY-001",
                decision_type="APPROVE",
                compliance_notes="موافقة متأخرة بعد تغير السوق.",
            ),
        )
    except StaleProposalExpiredError as exc:
        print(f"⏰ [نجاح حارس TTL]: {exc}")

    # 2. اختبار إعادة المحاولة الشبكية بنفس مفتاح Idempotency-Key
    valid_wf = guard.create_suspended_proposal(
        tenant_id="TENANT-01",
        requester_user_id="analyst_omar",
        ticker="MSFT",
        notional_usd=200_000.0,
        ttl_override_sec=60.0,
    )
    decision_payload = HumanDecisionWithIdempotency(
        idempotency_key="IDEMP-KEY-999-SAFE",
        decision_type="APPROVE",
        compliance_notes="اعتماد ضمن الوقت المسموح.",
    )
    res1 = guard.process_human_decision(
        "TENANT-01", valid_wf.workflow_id, "officer_sara", decision_payload
    )
    # إرسال نفس الطلب مرة ثانية (محاكاة انقطاع الشبكة وإعادة الإرسال من المتصفح)
    res2 = guard.process_human_decision(
        "TENANT-01", valid_wf.workflow_id, "officer_sara", decision_payload
    )
    print(
        f"🔁 [نجاح Idempotency Replay]: تطابق الاستجابتين بدون تكرار التنفيذ -> "
        f"Status={res2.status}, AuditEntries={len(res1.audit_trail) == len(res2.audit_trail)}"
    )


if __name__ == "__main__":
    asyncio.run(verify_ttl_and_idempotency())
