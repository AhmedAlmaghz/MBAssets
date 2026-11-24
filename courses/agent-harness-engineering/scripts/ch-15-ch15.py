#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
AI Agent Engineering — Capstone II — Production Multi-Agent Financial System

Book slug : agent-harness-engineering
Chapter   : ch-15-ch15 (lesson 15)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-15-ch15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
AI Agent Engineering — Capstone II — Production Multi-Agent Financial System

Book slug : agent-harness-engineering
Chapter   : ch-15-ch15 (lesson 15)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-15-ch15.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
AI Agent Engineering — Capstone II — Production Multi-Agent Financial System

Book slug : agent-harness-engineering
Chapter   : ch-15-ch15 (lesson 15)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [002] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [009] code/ch15_financial_swarm/ch15_financial_swarm.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 15: المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 15 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء — Capstone II — Production Multi-Agent Financial System.

التشغيل:
    python ch15_financial_swarm.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [010] code/ch15_financial_swarm/snippets/anti_pattern_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!

مقتطف من الفصل 15 — «المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch15_financial_swarm.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [011] code/ch15_financial_swarm/snippets/example_02.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [013] dist/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [014] dist/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 15: المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 15 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء — Capstone II — Production Multi-Agent Financial System.

التشغيل:
    python ch15_financial_swarm.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!

مقتطف من الفصل 15 — «المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch15_financial_swarm.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
AI Agent Engineering — Capstone II — Production Multi-Agent Financial System

Book slug : agent-harness-engineering
Chapter   : ch-15-ch15 (lesson 15)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [002] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch15_financial_swarm/ch15_financial_swarm.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [009] code/ch15_financial_swarm/ch15_financial_swarm.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 15: المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 15 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء — Capstone II — Production Multi-Agent Financial System.

التشغيل:
    python ch15_financial_swarm.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [010] code/ch15_financial_swarm/snippets/anti_pattern_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!

مقتطف من الفصل 15 — «المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch15_financial_swarm.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [011] code/ch15_financial_swarm/snippets/example_02.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [013] dist/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [014] dist/md/ch15.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# ⚙️ إعداد البيئة — يعمل على Google Colab و Kaggle ومحلياً
from __future__ import annotations   # ينتقل تلقائياً إلى كل خلايا الدفتر التالية

import sys, subprocess, types, typing, warnings

if sys.version_info < (3, 12):
    raise RuntimeError(
        "هذا الدفتر يتطلب Python 3.12 أو أحدث (الكتاب يستهدف 3.14+). "
        f"الإصدار الحالي: {sys.version.split()[0]}"
    )

try:
    import pydantic  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic>=2.9"], check=True)

# طبقة توافق: annotationlib متاحة أصلاً في Python 3.14 فقط
if sys.version_info < (3, 14):
    _al = types.ModuleType("annotationlib")

    class _Format:
        VALUE, FORWARDREF, STRING = 1, 2, 3

    _al.Format = _Format
    _al.get_annotations = lambda obj, **kw: typing.get_type_hints(obj, include_extras=True)
    sys.modules.setdefault("annotationlib", _al)

warnings.filterwarnings("ignore", category=DeprecationWarning)
print("✅ البيئة جاهزة — Python", sys.version.split()[0])

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 15: المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 15 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء — Capstone II — Production Multi-Agent Financial System.

التشغيل:
    python ch15_financial_swarm.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!

مقتطف من الفصل 15 — «المشروع التخرجي الثاني: منظومة مالية متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch15_financial_swarm.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import hmac
import json
import sqlite3
import time
import uuid
from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. سياق الأمان، الصلاحيات (RBAC)، ومبدأ العيون الأربع (Four-Eyes Guard)
# =====================================================================

class PermissionDeniedError(PermissionError):
    """يُرفع عند غياب الصلاحية المطلوبة أو خرق مبدأ الفصل بين المهام (Four-Eyes)."""


class TamperedWorkflowStateError(SecurityError := RuntimeError):
    """يُرفع عند فشل التحقق من توقيع HMAC-SHA256 للقطة سير العمل المعلقة."""


class WorkflowAlreadyProcessedError(RuntimeError):
    """يُرفع عند محاولة استئناف أو تنفيذ صفقة معلقة أكثر من مرة (Double-Execution Guard)."""


class FinancialSecurityContext(BaseModel):
    """سياق الأمان المحقون من طبقة المصادقة (JWT / API Gateway) خارج سيطرة النموذج."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str = Field(min_length=3)
    user_id: str = Field(min_length=3)
    role: str
    scopes: frozenset[str]

    def require_scope(self, required_scope: str) -> None:
        if required_scope not in self.scopes:
            raise PermissionDeniedError(
                f"رفض وصول (RBAC): المستخدم '{self.user_id}' (الدور: {self.role}) "
                f"لا يملك الصلاحية الإلزامية '{required_scope}'."
            )


# =====================================================================
# 2. نماذج تقارير الوكلاء الفرعيين ومقترح الصفقة المهيكل (Domain Schemas)
# =====================================================================

class TradeAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class WorkflowStatus(StrEnum):
    COMPLETED_AUTOMATICALLY = "COMPLETED_AUTOMATICALLY"
    SUSPENDED_FOR_HITL = "SUSPENDED_FOR_HITL"
    APPROVED_AND_EXECUTED = "APPROVED_AND_EXECUTED"
    REJECTED_BY_COMPLIANCE = "REJECTED_BY_COMPLIANCE"


class SpecialistAgentFinding(BaseModel):
    """تقرير معزول ومهيكل يعيده كل وكيل متخصص في مرحلة Map-Reduce."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_id: str
    domain: Literal["quant_metrics", "sec_filings", "macro_compliance"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    key_metrics: dict[str, Any]
    executive_summary: str
    latency_ms: float = Field(ge=0.0)


class TradeOrderProposal(BaseModel):
    """مقترح الأمر المالي الناتج عن دمج أبحاث الوكلاء المتخصصين."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1, max_length=10)
    action: TradeAction
    notional_usd: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    synthesized_rationale: str
    risk_warnings: list[str]


class HumanComplianceDecision(BaseModel):
    """قرار مسؤول الامتثال البشري لاستئناف سير العمل المعلق."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: Literal["APPROVE", "MODIFY_AND_APPROVE", "REJECT"]
    override_notional_usd: float | None = Field(default=None, gt=0.0)
    compliance_notes: str = Field(min_length=10)


class FinancialWorkflowCheckpoint(BaseModel):
    """اللقطة الكاملة لسير العمل التي تُحفظ وتُوقع بـ HMAC-SHA256 في SQLite."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    tenant_id: str
    requester_user_id: str
    status: WorkflowStatus
    specialist_findings: list[SpecialistAgentFinding]
    original_proposal: TradeOrderProposal
    executed_notional_usd: float | None = None
    approver_user_id: str | None = None
    compliance_notes: str | None = None
    audit_trail: list[str]


# =====================================================================
# 3. سرب الوكلاء الماليين المتوازي (Parallel Financial Research Swarm)
# =====================================================================

class ParallelFinancialResearchSwarm:
    """
    يشغّل 3 وكلاء متخصصين بالتوازي المهيكل عبر asyncio.TaskGroup،
    مع عزل سياق كل وكيل (Context Firewall) لمنع تلوث الانتباه.
    """

    async def _run_quant_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.03)  # محاكاة استعلام محرك البيانات الكمية
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_quant_v2",
            domain="quant_metrics",
            sentiment_score=0.78,
            key_metrics={
                "pe_forward": 31.4,
                "sharpe_ratio_1y": 2.15,
                "daily_var_99_pct": "3.2%",
            },
            executive_summary=(
                f"الزخم الكمي للسهم {ticker} إيجابي بقوة مع نسبة شارب 2.15 ومخاطر قيمة معرضة للخطر ضمن الحدود."
            ),
            latency_ms=elapsed,
        )

    async def _run_sec_filings_specialist(self, ticker: str) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.04)  # محاكاة تحليل تقارير 10-K و 10-Q
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_sec_edgar_v2",
            domain="sec_filings",
            sentiment_score=0.85,
            key_metrics={
                "revenue_yoy_growth": "+68%",
                "free_cash_flow_billions": 27.5,
                "debt_to_equity": 0.41,
            },
            executive_summary=(
                f"إفصاحات 10-K للسهم {ticker} تُظهر تدفقات نقدية حرة قياسية بقيمة 27.5 مليار دولار ومديونية منخفضة."
            ),
            latency_ms=elapsed,
        )

    async def _run_macro_compliance_specialist(
        self, ticker: str, requested_notional_usd: float
    ) -> SpecialistAgentFinding:
        t0 = time.perf_counter()
        await asyncio.sleep(0.025)  # محاكاة فحص قوائم العقوبات والتركز القطاعي
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return SpecialistAgentFinding(
            specialist_id="agent_risk_compliance_v2",
            domain="macro_compliance",
            sentiment_score=0.60,
            key_metrics={
                "sanctions_clear": True,
                "sector_concentration_post_trade": "14.2%",
                "requested_notional_usd": requested_notional_usd,
            },
            executive_summary=(
                f"السهم {ticker} خالٍ من القيود التنظيمية، لكن حجم الصفقة المطلوب "
                f"(${requested_notional_usd:,.2f}) يرفع التركز القطاعي إلى 14.2%."
            ),
            latency_ms=elapsed,
        )

    async def gather_parallel_research(
        self, ticker: str, requested_notional_usd: float, timeout_sec: float = 5.0
    ) -> list[SpecialistAgentFinding]:
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                t_quant = tg.create_task(self._run_quant_specialist(ticker))
                t_sec = tg.create_task(self._run_sec_filings_specialist(ticker))
                t_risk = tg.create_task(
                    self._run_macro_compliance_specialist(
                        ticker, requested_notional_usd
                    )
                )

        return [t_quant.result(), t_sec.result(), t_risk.result()]


# =====================================================================
# 4. مخزن نقاط الحفظ المتين والموقع بـ HMAC-SHA256 (Durable SQLite Store)
# =====================================================================

class SQLiteDurableWorkflowStore:
    """
    مخزن متين يحفظ حالة سير العمل المالي مع توقيع HMAC-SHA256،
    ويدعم القفل الذري التفاؤلي (Atomic Compare-And-Swap) لمنع التنفيذ المزدوج.
    """

    def __init__(self, secret_key: str, db_uri: str = ":memory:") -> None:
        self._secret_bytes = secret_key.encode("utf-8")
        self._conn = sqlite3.connect(db_uri, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _compute_hmac(self, payload_json: str) -> str:
        return hmac.new(
            self._secret_bytes,
            payload_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def save_checkpoint(self, checkpoint: FinancialWorkflowCheckpoint) -> None:
        payload_json = checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO financial_workflows
                    (workflow_id, tenant_id, status, payload_json, hmac_signature, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    hmac_signature = excluded.hmac_signature,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.workflow_id,
                    checkpoint.tenant_id,
                    checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                ),
            )

    def load_verified_checkpoint(
        self, workflow_id: str, tenant_id: str
    ) -> FinancialWorkflowCheckpoint:
        cur = self._conn.execute(
            """
            SELECT payload_json, hmac_signature
            FROM financial_workflows
            WHERE workflow_id = ? AND tenant_id = ?
            """,
            (workflow_id, tenant_id),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(
                f"لم يتم العثور على سير العمل '{workflow_id}' للمستأجر '{tenant_id}'."
            )

        payload_json, stored_sig = row
        expected_sig = self._compute_hmac(payload_json)
        if not hmac.compare_digest(stored_sig, expected_sig):
            raise TamperedWorkflowStateError(
                f"تنبيه أمني حرج: فشل التحقق من توقيع HMAC-SHA256 لسير العمل '{workflow_id}'!"
            )

        return FinancialWorkflowCheckpoint.model_validate_json(payload_json)

    def transition_suspended_atomically(
        self,
        workflow_id: str,
        tenant_id: str,
        updated_checkpoint: FinancialWorkflowCheckpoint,
    ) -> None:
        """
        عملية Compare-And-Swap ذرية: لا تحدث الحالة إلا إذا كانت لا تزال SUSPENDED_FOR_HITL.
        """
        payload_json = updated_checkpoint.model_dump_json()
        signature = self._compute_hmac(payload_json)
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE financial_workflows
                SET status = ?, payload_json = ?, hmac_signature = ?, updated_at = ?
                WHERE workflow_id = ? AND tenant_id = ? AND status = ?
                """,
                (
                    updated_checkpoint.status.value,
                    payload_json,
                    signature,
                    time.time(),
                    workflow_id,
                    tenant_id,
                    WorkflowStatus.SUSPENDED_FOR_HITL.value,
                ),
            )
            if cur.rowcount == 0:
                raise WorkflowAlreadyProcessedError(
                    f"تمت معالجة سير العمل '{workflow_id}' مسبقاً! تم إحباط التنفيذ المزدوج (Idempotency Guard)."
                )


# =====================================================================
# 5. إطار التحكم المالي المؤسسي (The Institutional Financial Agent Harness)
# =====================================================================

class InstitutionalFinancialHarness:
    """
    المحرك المركزي للمشروع التخرجي الثاني:
    ينسق البحث المتوازي، وتوليد المقترح، وفحص سقف المخاطر الآلي،
    والتجميد المتين لبوابات الاعتماد البشري (HITL) مع فرض مبدأ العيون الأربع.
    """

    def __init__(
        self,
        store: SQLiteDurableWorkflowStore,
        auto_execution_limit_usd: float = 50_000.0,
    ) -> None:
        self._store = store
        self._swarm = ParallelFinancialResearchSwarm()
        self._auto_limit_usd = auto_execution_limit_usd

    def _synthesize_proposal(
        self,
        ticker: str,
        requested_notional_usd: float,
        findings: list[SpecialistAgentFinding],
    ) -> TradeOrderProposal:
        avg_sentiment = sum(f.sentiment_score for f in findings) / len(findings)
        action = TradeAction.BUY if avg_sentiment >= 0.50 else TradeAction.HOLD
        confidence = round(min(1.0, max(0.0, (avg_sentiment + 1.0) / 2.0)), 4)

        summaries = " | ".join(
            f"[{f.domain}] {f.executive_summary}" for f in findings
        )
        warnings: list[str] = []
        if requested_notional_usd > self._auto_limit_usd:
            warnings.append(
                f"حجم الصفقة (${requested_notional_usd:,.2f}) يتجاوز سقف التنفيذ الآلي "
                f"(${self._auto_limit_usd:,.2f}) ويتطلب اعتماد مسؤول امتثال مستقل (HITL)."
            )

        return TradeOrderProposal(
            ticker=ticker,
            action=action,
            notional_usd=requested_notional_usd,
            confidence=confidence,
            synthesized_rationale=summaries,
            risk_warnings=warnings,
        )

    async def start_research_and_propose(
        self,
        ctx: FinancialSecurityContext,
        ticker: str,
        requested_notional_usd: float,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحيات RBAC لبدء البحث واقتراح الصفقة
        ctx.require_scope("research:run")
        ctx.require_scope("portfolio:propose")

        workflow_id = f"WF-FIN-{uuid.uuid4().hex[:8].upper()}"
        audit_trail = [
            f"[{workflow_id}] بدأ المحلل '{ctx.user_id}' طلب تحليل للسهم {ticker} بقيمة ${requested_notional_usd:,.2f}."
        ]

        # 2. تشغيل سرب الوكلاء المتخصصين بالتوازي (Map-Reduce Fan-Out)
        findings = await self._swarm.gather_parallel_research(
            ticker=ticker, requested_notional_usd=requested_notional_usd
        )
        audit_trail.append(
            f"اكتمل البحث المتوازي لـ {len(findings)} وكلاء متخصصين: "
            f"{[f.specialist_id for f in findings]}."
        )

        # 3. تركيب مذكرة الاستثمار ومقترح الأمر المالي
        proposal = self._synthesize_proposal(
            ticker=ticker,
            requested_notional_usd=requested_notional_usd,
            findings=findings,
        )

        # 4. بوابة سياسة المخاطر: تنفيذ آلي أم تجميد لاعتماد بشري (HITL)؟
        if requested_notional_usd <= self._auto_limit_usd:
            status = WorkflowStatus.COMPLETED_AUTOMATICALLY
            executed_usd: float | None = requested_notional_usd
            audit_trail.append(
                f"✅ الصفقة ضمن السقف الآلي (<= ${self._auto_limit_usd:,.2f}). تم التنفيذ الفوري."
            )
        else:
            status = WorkflowStatus.SUSPENDED_FOR_HITL
            executed_usd = None
            audit_trail.append(
                f"⏸️ تم تجميد سير العمل (Zero-Memory HITL Suspension) وتوقيعه بـ HMAC-SHA256 "
                f"بانتظار موافقة مسؤول الامتثال."
            )

        checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            requester_user_id=ctx.user_id,
            status=status,
            specialist_findings=findings,
            original_proposal=proposal,
            executed_notional_usd=executed_usd,
            audit_trail=audit_trail,
        )
        self._store.save_checkpoint(checkpoint)
        return checkpoint

    async def resume_with_human_decision(
        self,
        ctx: FinancialSecurityContext,
        workflow_id: str,
        decision: HumanComplianceDecision,
    ) -> FinancialWorkflowCheckpoint:
        # 1. التحقق من صلاحية RBAC لاعتماد الصفقات الحرجة
        ctx.require_scope("compliance:approve_trade")

        # 2. تحميل اللقطة والتحقق من سلامة توقيع HMAC-SHA256
        existing = self._store.load_verified_checkpoint(
            workflow_id=workflow_id, tenant_id=ctx.tenant_id
        )

        # 3. فرض مبدأ العيون الأربع (Four-Eyes Principle / Separation of Duties)
        if ctx.user_id == existing.requester_user_id:
            raise PermissionDeniedError(
                f"خرق مبدأ الفصل بين المهام (Four-Eyes Principle): لا يجوز للمستخدم "
                f"'{ctx.user_id}' اعتماد صفقة قام هو نفسه باقتراحها!"
            )

        updated_trail = list(existing.audit_trail)

        match decision.decision_type:
            case "APPROVE":
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd: float | None = existing.original_proposal.notional_usd
                updated_trail.append(
                    f"✅ اعتمد مسؤول الامتثال '{ctx.user_id}' الصفقة بالقيمة الأصلية "
                    f"(${final_usd:,.2f}). ملاحظات: {decision.compliance_notes}"
                )

            case "MODIFY_AND_APPROVE":
                if decision.override_notional_usd is None:
                    raise ValueError(
                        "يجب تحديد override_notional_usd عند اختيار MODIFY_AND_APPROVE."
                    )
                new_status = WorkflowStatus.APPROVED_AND_EXECUTED
                final_usd = decision.override_notional_usd
                updated_trail.append(
                    f"✏️ عدّل مسؤول الامتثال '{ctx.user_id}' قيمة الصفقة من "
                    f"${existing.original_proposal.notional_usd:,.2f} إلى ${final_usd:,.2f} "
                    f"واعتمد التنفيذ. ملاحظات: {decision.compliance_notes}"
                )

            case "REJECT":
                new_status = WorkflowStatus.REJECTED_BY_COMPLIANCE
                final_usd = None
                updated_trail.append(
                    f"❌ رفض مسؤول الامتثال '{ctx.user_id}' الصفقة. السبب: {decision.compliance_notes}"
                )

        updated_checkpoint = FinancialWorkflowCheckpoint(
            workflow_id=existing.workflow_id,
            tenant_id=existing.tenant_id,
            requester_user_id=existing.requester_user_id,
            status=new_status,
            specialist_findings=existing.specialist_findings,
            original_proposal=existing.original_proposal,
            executed_notional_usd=final_usd,
            approver_user_id=ctx.user_id,
            compliance_notes=decision.compliance_notes,
            audit_trail=updated_trail,
        )

        # 4. حفظ الانتقال ذرياً عبر Compare-And-Swap لمنع التنفيذ المزدوج
        self._store.transition_suspended_atomically(
            workflow_id=workflow_id,
            tenant_id=ctx.tenant_id,
            updated_checkpoint=updated_checkpoint,
        )
        return updated_checkpoint


# =====================================================================
# 6. محاكاة الإنتاج الشاملة لدورة حياة المنظومة المالية (End-to-End Run)
# =====================================================================

async def main() -> None:
    store = SQLiteDurableWorkflowStore(
        secret_key="enterprise-vault-hmac-key-2026-ultra-secret"
    )
    harness = InstitutionalFinancialHarness(
        store=store, auto_execution_limit_usd=50_000.0
    )

    # تعريف هويات المستخدمين (RBAC Security Contexts)
    analyst_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",
        role="senior_quant_analyst",
        scopes=frozenset({"research:run", "portfolio:propose"}),
    )

    analyst_with_elevated_scope_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="analyst_omar",  # نفس المستخدم المقترح!
        role="lead_analyst",
        scopes=frozenset({"research:run", "portfolio:propose", "compliance:approve_trade"}),
    )

    compliance_officer_ctx = FinancialSecurityContext(
        tenant_id="TENANT-CAPITAL-01",
        user_id="officer_sara",
        role="chief_compliance_officer",
        scopes=frozenset({"compliance:approve_trade", "audit:read"}),
    )

    print("=" * 82)
    print("🏦 المشروع التخرجي الثاني: منظومة الوكلاء الماليين المؤسسية (Multi-Agent FinSystem)")
    print("=" * 82)

    # -----------------------------------------------------------------
    # الخطوة 1: المحلل يطلق بحثاً متوازياً ويقترح صفقة بقيمة 250,000 دولار
    # -----------------------------------------------------------------
    print("\n🚀 [الخطوة 1]: إطلاق سرب البحث المتوازي للسهم NVDA بقيمة $250,000...")
    suspended_ckpt = await harness.start_research_and_propose(
        ctx=analyst_ctx,
        ticker="NVDA",
        requested_notional_usd=250_000.0,
    )
    print(f"   • رقم سير العمل (Workflow ID) : {suspended_ckpt.workflow_id}")
    print(f"   • الحالة الحالية (HTTP 202)  : {suspended_ckpt.status}")
    print(
        f"   • التوصية المركبة            : {suspended_ckpt.original_proposal.action} "
        f"(Confidence: {suspended_ckpt.original_proposal.confidence:.0%})"
    )
    for f in suspended_ckpt.specialist_findings:
        print(
            f"     - [{f.specialist_id} | {f.latency_ms}ms]: {f.executive_summary}"
        )

    # -----------------------------------------------------------------
    # الخطوة 2: اختبار حماية RBAC ومبدأ العيون الأربع (Four-Eyes Guard)
    # -----------------------------------------------------------------
    print("\n🛡️ [الخطوة 2]: اختبار محاولة المحلل اعتماد صفقته الخاصة (Self-Approval Attack)...")
    try:
        await harness.resume_with_human_decision(
            ctx=analyst_with_elevated_scope_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="أوافق على صفقتي الخاصة فوراً.",
            ),
        )
    except PermissionDeniedError as exc:
        print(f"   ✅ تم صد المحاولة بنجاح: {exc}")

    # -----------------------------------------------------------------
    # الخطوة 3: مسؤولة الامتثال المستقلة تعتمد الصفقة مع تعديل سقف المخاطر
    # -----------------------------------------------------------------
    print("\n✍️ [الخطوة 3]: مسؤولة الامتثال (officer_sara) تراجع اللقطة وتعتمدها بتعديل السقف...")
    final_ckpt = await harness.resume_with_human_decision(
        ctx=compliance_officer_ctx,
        workflow_id=suspended_ckpt.workflow_id,
        decision=HumanComplianceDecision(
            decision_type="MODIFY_AND_APPROVE",
            override_notional_usd=180_000.0,
            compliance_notes="المؤشرات الأساسية ممتازة، تم تخفيض الحجم إلى 180 ألف دولار لضبط التركز القطاعي.",
        ),
    )
    print(f"   • الحالة النهائية            : {final_ckpt.status}")
    print(f"   • المبلغ المنفذ فعلياً       : ${final_ckpt.executed_notional_usd:,.2f}")
    print(f"   • المعتمد الرسمي             : {final_ckpt.approver_user_id}")

    # -----------------------------------------------------------------
    # الخطوة 4: اختبار منع التنفيذ المزدوج (Atomic CAS Double-Execution Guard)
    # -----------------------------------------------------------------
    print("\n🔒 [الخطوة 4]: اختبار إرسال نقرة موافقة مكررة لنفس الصفقة (Replay / Double-Click)...")
    try:
        await harness.resume_with_human_decision(
            ctx=compliance_officer_ctx,
            workflow_id=suspended_ckpt.workflow_id,
            decision=HumanComplianceDecision(
                decision_type="APPROVE",
                compliance_notes="نقرة مزدوجة بالخطأ من المتصفح.",
            ),
        )
    except WorkflowAlreadyProcessedError as exc:
        print(f"   ✅ القفل الذري منع التنفيذ المزدوج: {exc}")

    print("\n" + "=" * 82)
    print("📜 سجل التدقيق المؤسسي الكامل والموثق (Immutable Workflow Audit Trail):")
    print("=" * 82)
    for idx, log_entry in enumerate(final_ckpt.audit_trail, start=1):
        print(f"   {idx}. {log_entry}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ❌ نمط ضار: تجميد عامل FastAPI/Uvicorn في الذاكرة لساعات بانتظار موافقة المدير!
@app.post("/run-agent")
async def run_agent_endpoint():
    proposal = await agent.propose_trade()
    approved = await wait_for_slack_webhook_or_human()  # كارثة! سيقطع البروكسي الاتصال بـ 504 Timeout

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/agent-harness-engineering/notebooks/ch-15-ch15.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
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
