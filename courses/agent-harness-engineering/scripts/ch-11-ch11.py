#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تنسيق الأنظمة متعددة الوكلاء
AI Agent Engineering — Multi-Agent Orchestration Harness

Book slug : agent-harness-engineering
Chapter   : ch-11-ch11 (lesson 11)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-11-ch11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تنسيق الأنظمة متعددة الوكلاء
AI Agent Engineering — Multi-Agent Orchestration Harness

Book slug : agent-harness-engineering
Chapter   : ch-11-ch11 (lesson 11)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/agent-harness-engineering/code/ch-11-ch11.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تنسيق الأنظمة متعددة الوكلاء
AI Agent Engineering — Multi-Agent Orchestration Harness

Book slug : agent-harness-engineering
Chapter   : ch-11-ch11 (lesson 11)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-1
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
# [002] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 11: تنسيق الأنظمة متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 11 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
تنسيق الأنظمة متعددة الوكلاء — Multi-Agent Orchestration Harness.

التشغيل:
    python ch11_multi_agent_orchestration.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch11_multi_agent_orchestration/snippets/anti_pattern_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!

مقتطف من الفصل 11 — «تنسيق الأنظمة متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch11_multi_agent_orchestration.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [009] code/ch11_multi_agent_orchestration/snippets/example_02.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] dist/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [011] dist/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-1
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
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 11: تنسيق الأنظمة متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 11 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
تنسيق الأنظمة متعددة الوكلاء — Multi-Agent Orchestration Harness.

التشغيل:
    python ch11_multi_agent_orchestration.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!

مقتطف من الفصل 11 — «تنسيق الأنظمة متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch11_multi_agent_orchestration.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
هندسة وكلاء الذكاء الاصطناعي — تنسيق الأنظمة متعددة الوكلاء
AI Agent Engineering — Multi-Agent Orchestration Harness

Book slug : agent-harness-engineering
Chapter   : ch-11-ch11 (lesson 11)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-1
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
# [002] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ────────────────────────────────────────────────────────────────────────
# [003] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]

# ────────────────────────────────────────────────────────────────────────
# [004] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]

# ────────────────────────────────────────────────────────────────────────
# [005] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )

# ────────────────────────────────────────────────────────────────────────
# [006] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [007] code/ch11_multi_agent_orchestration/ch11_multi_agent_orchestration.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 11: تنسيق الأنظمة متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 11 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
تنسيق الأنظمة متعددة الوكلاء — Multi-Agent Orchestration Harness.

التشغيل:
    python ch11_multi_agent_orchestration.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [008] code/ch11_multi_agent_orchestration/snippets/anti_pattern_01.py
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!

مقتطف من الفصل 11 — «تنسيق الأنظمة متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch11_multi_agent_orchestration.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [009] code/ch11_multi_agent_orchestration/snippets/example_02.py
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [010] dist/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [011] dist/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [012] dist/md/ch11.md
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-2
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
# [004] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")

# في الدفتر لا نستخدم asyncio.run: الحلقة قائمة بالفعل، لذا ننتظر التنفيذ مباشرة.
await main()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 11: تنسيق الأنظمة متعددة الوكلاء
==============================================================================

البرنامج الكامل للفصل 11 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
تنسيق الأنظمة متعددة الوكلاء — Multi-Agent Orchestration Harness.

التشغيل:
    python ch11_multi_agent_orchestration.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!

مقتطف من الفصل 11 — «تنسيق الأنظمة متعددة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch11_multi_agent_orchestration.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import time
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# =====================================================================
# 1. عقود التواصل المهيكلة بين الوكلاء (Inter-Agent Handoff & Delegation Contracts)
# =====================================================================
class BaseMultiAgentAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ScopedHandoffPacket(BaseMultiAgentAction):
    """حقيبة التسليم المعزولة (Context Firewall Packet):

    تمنع تمرير تاريخ المحادثة الملوث؛ وتمرر فقط الحقائق المستخلصة والمهمة المحددة للوكيل المستلم!
    """

    task_directive: str = Field(min_length=5, description="المهمة المحددة المطلوبة من الوكيل المستلم")
    extracted_facts: list[str] = Field(
        default_factory=list, description="الحقائق المؤكدة فقط دون ضوضاء الأدوات الخام"
    )
    priority: Literal["normal", "high", "critical"] = "normal"


class HandoffToSpecialist(BaseMultiAgentAction):
    """قرار بتسليم القيادة بالكامل لوكيل متخصص آخر (Swarm Handoff Pattern)"""

    action_type: Literal["handoff"] = "handoff"
    target_agent_id: str
    reason: str
    packet: ScopedHandoffPacket


class WorkerSubTask(BaseMultiAgentAction):
    """مهمة فرعية مخصصة لوكيل عامل ضمن موجة تنفيذ متوازية"""

    worker_agent_id: str
    sub_task_prompt: str
    token_budget_share: float = Field(default=0.25, gt=0.0, le=1.0)


class ParallelDelegateToWorkers(BaseMultiAgentAction):
    """قرار بتفويض عدة مهام لعدة وكلاء متخصصين بالتوازي ثم استلام تقاريرهم (Supervisor Map-Reduce)"""

    action_type: Literal["parallel_delegate"] = "parallel_delegate"
    coordination_thought: str
    sub_tasks: list[WorkerSubTask] = Field(min_length=1, max_length=6)


class CompleteTeamMission(BaseMultiAgentAction):
    """قرار بحسم المهمة وتقديم المخرج النهائي الموحد للفريق"""

    action_type: Literal["complete"] = "complete"
    executive_summary: str
    key_findings: list[str]


# اتحاد مميز فائق السرعة لقرارات التنسيق بين الوكلاء
type OrchestrationDecision = Annotated[
    Annotated[HandoffToSpecialist, Tag("handoff")]
    | Annotated[ParallelDelegateToWorkers, Tag("parallel_delegate")]
    | Annotated[CompleteTeamMission, Tag("complete")],
    Discriminator("action_type"),
]


# =====================================================================
# 2. تعريف الوكيل المتخصص المعزول (Specialist Agent Node)
# =====================================================================
@dataclass(slots=True, frozen=True)
class WorkerReport:
    """التقرير المهيكل الذي يعيده الوكيل العامل بعد انتهاء مهمته المعزولة"""

    worker_id: str
    findings: str
    tokens_consumed: int
    latency_ms: float
    succeeded: bool = True


@dataclass(slots=True)
class SpecialistAgentSpec:
    """مواصفات الوكيل المتخصص وحدود صلاحياته في التواصل"""

    agent_id: str
    role_title: str
    system_prompt: str
    allowed_Tools: tuple[str, ...]
    # القائمة البيضاء للوكلاء المسموح لهذا الوكيل بتسليم القيادة لهم أو تفويضهم!
    allowed_targets: frozenset[str]
    brain_fn: Callable[[str, list[str], int], Awaitable[ tuple[OrchestrationDecision | str, int] ]]


# =====================================================================
# 3. قلب النظام: MultiAgentOrchestrationHarness
# =====================================================================
class TeamStopReason(StrEnum):
    MISSION_ACCOMPLISHED = auto()
    MAX_HANDOFF_HOPS_EXCEEDED = auto()
    HANDOFF_CYCLE_DETECTED = auto()
    TEAM_BUDGET_EXHAUSTED = auto()
    UNAUTHORIZED_TARGET_BLOCKED = auto()


@dataclass(slots=True)
class HandoffTraceEvent:
    """سجل تدقيق لكل عملية تسليم أو تفويض متوازٍ بين أعضاء الفريق"""

    step: int
    source_agent: str
    event_type: str  # 'HANDOFF' | 'PARALLEL_FANOUT' | 'COMPLETE'
    target_agents: tuple[str, ...]
    tokens_used: int
    elapsed_ms: float
    note: str = ""


@dataclass(slots=True)
class TeamExecutionResult:
    stop_reason: TeamStopReason
    final_summary: str
    findings: list[str]
    total_team_tokens: int
    total_hops: int
    total_elapsed_ms: float
    trace: list[HandoffTraceEvent] = field(default_factory=list)


class MultiAgentOrchestrationHarness:
    """إطار التحكم والتنسيق للأنظمة متعددة الوكلاء.

    يدمج بين نمط التسليم اللامركزي (Swarm Handoffs) ونمط التفويض المتوازي (Supervisor Fan-Out)،
    مع فرض جدار عزل السياق (Context Firewall) وتقسيم الميزانية ومنع الحلقات الدائرية.
    """

    def __init__(
        self,
        *,
        entry_agent_id: str,
        max_handoff_hops: int = 8,
        team_token_budget: int = 12_000,
        per_worker_timeout_sec: float = 15.0,
    ) -> None:
        self.entry_agent_id = entry_agent_id
        self.max_handoff_hops = max_handoff_hops
        self.team_token_budget = team_token_budget
        self.per_worker_timeout_sec = per_worker_timeout_sec
        self._agents: dict[str, SpecialistAgentSpec] = {}

    def register_specialist(self, spec: SpecialistAgentSpec) -> None:
        self._agents[spec.agent_id] = spec

    async def _run_isolated_worker(
        self,
        worker_spec: SpecialistAgentSpec,
        sub_task_prompt: str,
        allocated_budget: int,
    ) -> WorkerReport:
        """تشغيل وكيل عامل في نافذة سياق معزولة ونظيفة تماماً (Zero Context Pollution)"""
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.per_worker_timeout_sec):
                # العامل يستلم مهمته الفرعية فقط وميزانيته المحددة، ويعيد تقريراً نصياً مركزاً
                output, tokens = await worker_spec.brain_fn(
                    sub_task_prompt, [], allocated_budget
                )
            text_res = output if isinstance(output, str) else str(output)
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=text_res,
                tokens_consumed=tokens,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=True,
            )
        except Exception as exc:
            return WorkerReport(
                worker_id=worker_spec.agent_id,
                findings=f"WORKER_ERROR({type(exc).__name__}): {exc}",
                tokens_consumed=0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                succeeded=False,
            )

    async def run_team(self, user_mission: str) -> TeamExecutionResult:
        """تشغيل فريق الوكلاء تحت رقابة الـ Harness"""
        if self.entry_agent_id not in self._agents:
            raise RuntimeError(f"وكيل البداية '{self.entry_agent_id}' غير مسجل!")

        t_start = time.perf_counter()
        active_agent_id = self.entry_agent_id
        active_directive = user_mission
        accumulated_facts: list[str] = []

        total_tokens = 0
        hops = 0
        trace: list[HandoffTraceEvent] = []
        # تتبع مسار التسليم المتتالي لكشف حلقات Ping-Pong بين الوكلاء
        handoff_chain: list[str] = [active_agent_id]

        while hops < self.max_handoff_hops:
            hops += 1
            remaining_budget = self.team_token_budget - total_tokens
            if remaining_budget <= 0:
                return TeamExecutionResult(
                    stop_reason=TeamStopReason.TEAM_BUDGET_EXHAUSTED,
                    final_summary="توقف الفريق لاستنفاد ميزانية التوكنز المجمعة.",
                    findings=accumulated_facts,
                    total_team_tokens=total_tokens,
                    total_hops=hops,
                    total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    trace=trace,
                )

            agent_spec = self._agents[active_agent_id]
            step_t0 = time.perf_counter()

            # استدعاء عقل الوكيل النشط بسياقه المعزول والحقائق المنقحة فقط!
            decision_or_text, tokens = await agent_spec.brain_fn(
                active_directive, accumulated_facts, remaining_budget
            )
            total_tokens += tokens

            if isinstance(decision_or_text, str):
                # إذا أعاد نصاً مباشراً نعتبره إتماماً للمهمة
                decision: OrchestrationDecision = CompleteTeamMission(
                    executive_summary=decision_or_text,
                    key_findings=accumulated_facts,
                )
            else:
                decision = decision_or_text

            match decision:
                # ─── الحالة 1: حسم المهمة وتقديم التقرير النهائي ───
                case CompleteTeamMission(executive_summary=summary, key_findings=findings):
                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="COMPLETE",
                            target_agents=(),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note="Mission accomplished.",
                        )
                    )
                    return TeamExecutionResult(
                        stop_reason=TeamStopReason.MISSION_ACCOMPLISHED,
                        final_summary=summary,
                        findings=findings,
                        total_team_tokens=total_tokens,
                        total_hops=hops,
                        total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                        trace=trace,
                    )

                # ─── الحالة 2: تسليم القيادة لوكيل متخصص آخر (Swarm Handoff) ───
                case HandoffToSpecialist(
                    target_agent_id=target_id, reason=reason, packet=packet
                ):
                    # فحص القائمة البيضاء: هل يحق لهذا الوكيل تسليم القيادة للهدف؟
                    if target_id not in agent_spec.allowed_targets or target_id not in self._agents:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر أمني: الوكيل '{active_agent_id}' حاول التسليم لهدف غير مصرح به '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # كشف حلقات الدوران المفرغ (مثلاً A -> B -> A بدون إضافة حقائق جديدة)
                    if handoff_chain.count(target_id) >= 2:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.HANDOFF_CYCLE_DETECTED,
                            final_summary=f"تم إيقاف الفريق لاكتشاف حلقة تسليم دائرية نحو '{target_id}'.",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="HANDOFF",
                            target_agents=(target_id,),
                            tokens_used=tokens,
                            elapsed_ms=round((time.perf_counter() - step_t0) * 1000, 2),
                            note=f"Reason: {reason} | Priority: {packet.priority}",
                        )
                    )

                    # تطبيق Context Firewall: الوكيل الجديد يستلم فقط التوجيه والحقائق المنقحة!
                    active_agent_id = target_id
                    active_directive = packet.task_directive
                    accumulated_facts = list(packet.extracted_facts)
                    handoff_chain.append(target_id)

                # ─── الحالة 3: تفويض متوازٍ لعدة وكلاء عمال (Supervisor Map-Reduce Fan-Out) ───
                case ParallelDelegateToWorkers(
                    coordination_thought=thought, sub_tasks=sub_tasks
                ):
                    target_ids = tuple(st.worker_agent_id for st in sub_tasks)
                    unauthorized = set(target_ids) - agent_spec.allowed_targets
                    if unauthorized:
                        return TeamExecutionResult(
                            stop_reason=TeamStopReason.UNAUTHORIZED_TARGET_BLOCKED,
                            final_summary=f"🚨 حظر تفويض غير مصرح به للوكلاء: {unauthorized}",
                            findings=accumulated_facts,
                            total_team_tokens=total_tokens,
                            total_hops=hops,
                            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
                            trace=trace,
                        )

                    # تشغيل جميع الوكلاء العمال بالتوازي عبر asyncio.TaskGroup مع تقسيم الميزانية!
                    fanout_t0 = time.perf_counter()
                    worker_tasks: list[asyncio.Task[WorkerReport]] = []
                    avail_for_workers = max(500, self.team_token_budget - total_tokens)

                    async with asyncio.TaskGroup() as tg:
                        for st in sub_tasks:
                            w_spec = self._agents[st.worker_agent_id]
                            sub_budget = int(avail_for_workers * st.token_budget_share)
                            worker_tasks.append(
                                tg.create_task(
                                    self._run_isolated_worker(
                                        w_spec, st.sub_task_prompt, sub_budget
                                    )
                                )
                            )

                    reports = [wt.result() for wt in worker_tasks]
                    workers_tokens = sum(r.tokens_consumed for r in reports)
                    total_tokens += workers_tokens

                    # دمج التقارير المهيكلة في قائمة الحقائق ليبني عليها المشرف قراره في الدورة التالية
                    for rep in reports:
                        accumulated_facts.append(
                            f"[Report from {rep.worker_id}]: {rep.findings}"
                        )

                    trace.append(
                        HandoffTraceEvent(
                            step=hops,
                            source_agent=active_agent_id,
                            event_type="PARALLEL_FANOUT",
                            target_agents=target_ids,
                            tokens_used=tokens + workers_tokens,
                            elapsed_ms=round((time.perf_counter() - fanout_t0) * 1000, 2),
                            note=f"Thought: {thought} | Ran {len(reports)} workers in parallel.",
                        )
                    )
                    # يبقى المشرف هو الوكيل النشط ليقرأ تقارير العمال ويصيغ القرار النهائي
                    active_directive = "Synthesize the worker reports and finalize the mission."

        return TeamExecutionResult(
            stop_reason=TeamStopReason.MAX_HANDOFF_HOPS_EXCEEDED,
            final_summary="تجاوز الفريق الحد الأقصى لخطوات التنسيق والتسليم.",
            findings=accumulated_facts,
            total_team_tokens=total_tokens,
            total_hops=hops,
            total_elapsed_ms=round((time.perf_counter() - t_start) * 1000, 2),
            trace=trace,
        )


# =====================================================================
# 4. محاكاة تشغيلية واقعية: فريق الاستجابة للحوادث السحابية والأمنية (Incident Response Team)
# =====================================================================
async def triage_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """1. وكيل الفرز الأولي: يحلل البلاغ ويسلم القيادة لقائد التحقيق الجنائي والتقني"""
    await asyncio.sleep(0.02)
    return (
        HandoffToSpecialist(
            target_agent_id="incident_commander",
            reason="البلاغ يتضمن اشتباهاً باختراق أمني مع بطء شديد في قاعدة البيانات.",
            packet=ScopedHandoffPacket(
                task_directive="قُد التحقيق المتوازي (أمنياً وتقنياً) للخادم prod-eu-01 وأصدر خطة الاحتواء.",
                extracted_facts=[
                    "الخادم المتأثر: prod-eu-01",
                    "الأعراض: ارتفاع حركة البيانات الصادرة + استهلاك CPU 99% في PostgreSQL.",
                ],
                priority="critical",
            ),
        ),
        280,
    )


async def commander_agent_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """2. قائد الحادثة (Supervisor): يفوض المهمة بالتوازي لوكيل الأمن ووكيل قواعد البيانات، ثم يدمج تقاريرهما!"""
    await asyncio.sleep(0.03)

    # إذا لم تكن تقارير العمال قد وصلت بعد، أطلق الموجة المتوازية (Parallel Fan-Out)!
    if not any(f.startswith("[Report from") for f in facts):
        return (
            ParallelDelegateToWorkers(
                coordination_thought="سأكلف وكيل الأمن الجنائي ووكيل قواعد البيانات بالعمل بالتوازي لكسب الوقت.",
                sub_tasks=[
                    WorkerSubTask(
                        worker_agent_id="sec_forensics_worker",
                        sub_task_prompt="افحص اتصالات الشبكة المشبوهة على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                    WorkerSubTask(
                        worker_agent_id="db_sre_worker",
                        sub_task_prompt="افحص الاستعلامات المعلقة والأقفال في PostgreSQL على prod-eu-01.",
                        token_budget_share=0.4,
                    ),
                ],
            ),
            410,
        )

    # بعد وصول تقارير العاملين المتوازيين، يصيغ القائد القرار النهائي الشامل
    return (
        CompleteTeamMission(
            executive_summary=(
                "تم احتواء الحادثة بنجاح: تبين وجود استعلام استخراج بيانات غير مصرح به عبر جلسة مسروقة. "
                "تم عزل الـ IP المهاجم وقتل عملية قاعدة البيانات المستنزفة للمعالج."
            ),
            key_findings=facts,
        ),
        350,
    )


async def sec_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """3. وكيل الأمن الجنائي (يعمل في سياق معزول بالتوازي)"""
    await asyncio.sleep(0.05)
    return "تم رصد اتصال مشبوه نحو IP خارجي 198.51.100.42 عبر مفتاح API قديم، وتم حظره في جدار الحماية.", 520


async def db_worker_brain(
    directive: str, facts: list[str], budget: int
) -> tuple[OrchestrationDecision | str, int]:
    """4. وكيل هندسة قواعد البيانات (يعمل في سياق معزول بالتوازي في نفس اللحظة!)"""
    await asyncio.sleep(0.05)
    return "تم رصد استعلام FULL TABLE SCAN معلق منذ 12 دقيقة (PID 8841)، وتم إنهاؤه عبر pg_terminate_backend.", 490


async def main() -> None:
    orchestrator = MultiAgentOrchestrationHarness(
        entry_agent_id="triage_frontline",
        max_handoff_hops=6,
        team_token_budget=10_000,
    )

    # تسجيل أعضاء الفريق مع تحديد القوائم البيضاء للتواصل (Graph Topology)
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="triage_frontline",
            role_title="وكيل الفرز الأولي",
            system_prompt="صنف البلاغات وسلمها للقائد المختص بحقيبة معلومات نقية.",
            allowed_Tools=("read_ticket",),
            allowed_targets=frozenset({"incident_commander"}),
            brain_fn=triage_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="incident_commander",
            role_title="قائد الاستجابة للحوادث (Supervisor)",
            system_prompt="نسق العمل المتوازي بين خبراء الأمن وقواعد البيانات.",
            allowed_Tools=(),
            allowed_targets=frozenset({"sec_forensics_worker", "db_sre_worker"}),
            brain_fn=commander_agent_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="sec_forensics_worker",
            role_title="خبير التحقيق الجنائي السيبراني",
            system_prompt="افحص حركة الشبكة والجدار الناري.",
            allowed_Tools=("inspect_netflow", "block_ip"),
            allowed_targets=frozenset(),
            brain_fn=sec_worker_brain,
        )
    )
    orchestrator.register_specialist(
        SpecialistAgentSpec(
            agent_id="db_sre_worker",
            role_title="خبير موثوقية قواعد البيانات",
            system_prompt="شخص أعطال PostgreSQL وأنهِ العمليات العالقة.",
            allowed_Tools=("pg_stat_activity", "pg_terminate"),
            allowed_targets=frozenset(),
            brain_fn=db_worker_brain,
        )
    )

    result = await orchestrator.run_team(
        "إنذار طوارئ: الخادم prod-eu-01 لا يستجيب وهناك حركة بيانات صادرة غير طبيعية!"
    )

    print(f"🏁 حالة مهمة الفريق     : {result.stop_reason.upper()}")
    print(f"🪙 إجمالي توكنز الفريق  : {result.total_team_tokens} / 10000")
    print(f"⏱️ الزمن الكلي للتنسيق : {result.total_elapsed_ms} ms\n")

    print("📡 مسار التنسيق والتفويض المتوازي (Multi-Agent Trace):")
    for ev in result.trace:
        targets_str = ", ".join(ev.target_agents) if ev.target_agents else "FINAL_USER"
        print(
            f"   Hop #{ev.step}: [{ev.source_agent}] ──({ev.event_type})──► [{targets_str}] "
            f"| ({ev.elapsed_ms}ms, {ev.tokens_used} tok) | {ev.note}"
        )

    print("\n📋 الأدلة والتقارير المجمعة من العمال المعزولين:")
    for idx, finding in enumerate(result.findings, 1):
        print(f"   {idx}. {finding}")

    print(f"\n💎 الملخص التنفيذي النهائي:\n   {result.final_summary}")


if __name__ == "__main__":
    asyncio.run(main())

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ ANTI-PATTERN: غرفة الدردشة الفوضوية! كل وكيل يقرأ ضوضاء وهلوسات الوكلاء الآخرين!
for agent in [planner, researcher, coder, reviewer]:
    reply = await agent.run(shared_messages)  # تضخم تربيعي للتوكنز وتشتت كامل للانتباه!
    shared_messages.append(reply)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/agent-harness-engineering/notebooks/ch-11-ch11.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
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
