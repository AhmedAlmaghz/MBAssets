# -*- coding: utf-8 -*-
"""
import asyncio

مقتطف من الفصل 11 — «تنسيق الأنظمة متعددة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch11_multi_agent_orchestration.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

import asyncio
from typing import Protocol
from pydantic import BaseModel, ConfigDict, Field


# 1. تعريف بطاقة الاقتراع المهيكلة والجامدة (Immutable Ballot Schema)
class VoteBallot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    voter_agent_id: str
    approved: bool = Field(description="هل يوافق الوكيل على تمرير المقترح؟")
    confidence: float = Field(
        ge=0.0, le=1.0, description="درجة ثقة الوكيل في حكمه من 0.0 إلى 1.0"
    )
    rationale: str = Field(
        min_length=10, description="التبرير الهندسي المفصل للقرار"
    )
    has_veto: bool = Field(
        default=False,
        description="حق نقض مطلق يُفعّل فقط عند رصد خطر أمني أو قانوني كارثي",
    )


# 2. تعريف تقرير نتيجة الإجماع النهائي
class ConsensusVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    vetoed_by: list[str]
    weighted_approval_score: float
    quorum_threshold: float
    ballots: list[VoteBallot]
    summary_reason: str


# واجهة لمحرك استدعاء وكلاء التحقق (Verifier Provider)
class VerifierGatewayProtocol(Protocol):
    async def evaluate_proposal(
        self, agent_id: str, proposal: str
    ) -> VoteBallot: ...


class ConsensusOrchestratorMixin:
    """
    إضافة (Mixin) تُدمج داخل MultiAgentOrchestrationHarness
    لتنفيذ بروتوكول التصويت الموزون المتوازي مع حق الفيتو.
    """

    def __init__(self, verifier_gateway: VerifierGatewayProtocol) -> None:
        self._verifier_gateway = verifier_gateway

    async def run_consensus_vote(
        self,
        proposal_to_verify: str,
        verifier_agent_ids: list[str],
        quorum_threshold: float = 0.67,
        timeout_sec: float = 15.0,
    ) -> ConsensusVerdict:
        if not verifier_agent_ids:
            raise ValueError("يجب تحديد وكيل تحقق واحد على الأقل للتصويت.")

        tasks: list[asyncio.Task[VoteBallot]] = []

        # تشغيل جميع وكلاء التحقق بالتوازي المهيكل مع سقف زمني صارم
        async with asyncio.timeout(timeout_sec):
            async with asyncio.TaskGroup() as tg:
                for agent_id in verifier_agent_ids:
                    tasks.append(
                        tg.create_task(
                            self._verifier_gateway.evaluate_proposal(
                                agent_id=agent_id,
                                proposal=proposal_to_verify,
                            )
                        )
                    )

        ballots = [t.result() for t in tasks]

        # فحص القاعدة الأولى: حق النقض المطلق (Absolute Veto Power)
        veto_agents = [b.voter_agent_id for b in ballots if b.has_veto]
        if veto_agents:
            return ConsensusVerdict(
                passed=False,
                vetoed_by=veto_agents,
                weighted_approval_score=0.0,
                quorum_threshold=quorum_threshold,
                ballots=ballots,
                summary_reason=(
                    f"تم إجهاض المقترح فوراً لاستخدام حق الفيتو (Veto) من قِبل: "
                    f"{', '.join(veto_agents)}"
                ),
            )

        # فحص القاعدة الثانية: حساب الإجماع الموزون بدرجة الثقة (Confidence-Weighted Quorum)
        total_confidence = sum(b.confidence for b in ballots)
        if total_confidence == 0.0:
            return ConsensusVerdict(
                passed=False,
                vetoed_by=[],
                weighted_approval_score=0.0,
                quorum_threshold=quorum_threshold,
                ballots=ballots,
                summary_reason="تم رفض المقترح لأن مجموع ثقة المصوتين يساوي صفراً.",
            )

        approved_confidence = sum(
            b.confidence for b in ballots if b.approved
        )
        weighted_score = round(approved_confidence / total_confidence, 4)
        passed = weighted_score >= quorum_threshold

        return ConsensusVerdict(
            passed=passed,
            vetoed_by=[],
            weighted_approval_score=weighted_score,
            quorum_threshold=quorum_threshold,
            ballots=ballots,
            summary_reason=(
                f"النتيجة الموزونة: {weighted_score:.2%} "
                f"(النصاب المطلوب: {quorum_threshold:.2%}) -> "
                f"{'مقبول ✅' if passed else 'مرفوض ❌'}"
            ),
        )
