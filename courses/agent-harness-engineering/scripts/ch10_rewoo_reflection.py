#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 10: أنماط التفكير والتخطيط المتقدم
==============================================================================

البرنامج الكامل للفصل 10 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
أنماط التفكير والتخطيط المتقدم — Advanced Reasoning: Plan-and-Solve, ReWOO & Self-Reflection.

التشغيل:
    python ch10_rewoo_reflection.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
import re
import time
from typing import Any, Protocol
from pydantic import BaseModel, ConfigDict, Field, model_validator


# =====================================================================
# الجزء الأول: محرك التخطيط الشبكي الرمزي المتوازي (ReWOO Symbolic DAG Harness)
# =====================================================================
class SymbolicPlanStep(BaseModel):
    """خطوة واحدة داخل خطة ReWOO الرمزية"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str = Field(
        pattern=r"^#E[1-9][0-9]*$",
        description="المعرف الرمزي للخطوة مثل #E1 أو #E2",
    )
    purpose: str = Field(min_length=5, description="الهدف من هذه الخطوة")
    tool_name: str = Field(min_length=2)
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="معطيات الأداة؛ يمكن أن تحتوي قيمها على إشارات رمزية لخطوات سابقة مثل '#E1'",
    )

    @property
    def dependencies(self) -> set[str]:
        """استخراج جميع المتغيرات الرمزية (#E1, #E2) التي تعتمد عليها هذه الخطوة تلقائياً!"""
        deps: set[str] = set()
        pattern = re.compile(r"#E[1-9][0-9]*")
        for val in self.tool_args.values():
            if isinstance(val, str):
                deps.update(pattern.findall(val))
        return deps


class SymbolicExecutionPlan(BaseModel):
    """المخطط الشبكي الكامل (DAG) الذي يولده المخطط (Planner LLM) في استدعاء واحد"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_overview: str
    steps: list[SymbolicPlanStep] = Field(min_length=1, max_length=15)

    @model_validator(mode="after")
    def validate_dag_integrity(self) -> SymbolicExecutionPlan:
        """التحقق الرياضي الصارم من أن الخطة تشكل DAG سليم خالٍ من الحلقات أو المراجع المجهولة"""
        defined_ids = {s.step_id for s in self.steps}
        if len(defined_ids) != len(self.steps):
            raise ValueError("Duplicate step_id detected in SymbolicExecutionPlan!")

        sorter: TopologicalSorter[str] = TopologicalSorter()
        for step in self.steps:
            unknown_deps = step.dependencies - defined_ids
            if unknown_deps:
                raise ValueError(
                    f"الخطوة '{step.step_id}' تشير إلى متغيرات رمزية غير معرفة: {unknown_deps}"
                )
            if step.step_id in step.dependencies:
                raise ValueError(f"الخطوة '{step.step_id}' لا يمكن أن تعتمد على نفسها!")
            sorter.add(step.step_id, *step.dependencies)

        try:
            sorter.prepare()
        except CycleError as cyc_err:
            raise ValueError(f"تحتوي الخطة على اعتمادية دائرية مستحيلة التنفيذ: {cyc_err}") from cyc_err

        return self


@dataclass(slots=True)
class ReWOOExecutionResult:
    """النتيجة الشاملة لتنفيذ خطة ReWOO مع مقاييس التوازي"""

    final_answer: str
    total_steps_executed: int
    parallel_waves_count: int
    llm_calls_count: int
    evidence_map: dict[str, str]
    wave_log: list[list[str]] = field(default_factory=list)
    elapsed_ms: float = 0.0


class ReWOODAGHarness:
    """محرك ReWOO الاحترافي:

    1. يطلب الخطة الرمزية من الـ Planner (استدعاء LLM أول).
    2. يحل الاعتماديات طوبولوجياً عبر TopologicalSorter وينفذ الموجات المتوازية بـ TaskGroup.
    3. يعوض المتغيرات الرمزية (#E1) حتمياً في بايثون.
    4. يجمع الأدلة ويقدمها للـ Solver (استدعاء LLM ثانٍ وأخير!).
    """

    _VAR_PATTERN = re.compile(r"#E[1-9][0-9]*")

    def __init__(
        self,
        *,
        planner_fn: Callable[[str], Awaitable[SymbolicExecutionPlan]],
        solver_fn: Callable[[str, dict[str, str]], Awaitable[str]],
        tools: dict[str, Callable[..., Awaitable[str]]],
    ) -> None:
        self.planner_fn = planner_fn
        self.solver_fn = solver_fn
        self.tools = tools

    @classmethod
    def _interpolate_args(
        cls, raw_args: dict[str, Any], evidence: dict[str, str]
    ) -> dict[str, Any]:
        """استبدال الرموز (#E1, #E2) داخل معطيات الأداة بقيمها الحقيقية المنتجة من الموجات السابقة"""
        resolved: dict[str, Any] = {}
        for key, val in raw_args.items():
            if isinstance(val, str):
                resolved[key] = cls._VAR_PATTERN.sub(
                    lambda m: evidence.get(m.group(0), m.group(0)), val
                )
            else:
                resolved[key] = val
        return resolved

    async def _execute_step(
        self, step: SymbolicPlanStep, evidence: dict[str, str]
    ) -> tuple[str, str]:
        """تشغيل خطوة واحدة بعد تعويض متغيراتها الرمزية"""
        tool_func = self.tools.get(step.tool_name)
        if tool_func is None:
            return step.step_id, f"ERROR: Tool '{step.tool_name}' not registered."

        concrete_args = self._interpolate_args(step.tool_args, evidence)
        try:
            output = await tool_func(**concrete_args)
            return step.step_id, str(output)
        except Exception as exc:
            return step.step_id, f"ERROR({type(exc).__name__}): {exc}"

    async def run(self, user_goal: str) -> ReWOOExecutionResult:
        t0 = time.perf_counter()

        # 1. استدعاء الـ Planner لتوليد الـ Symbolic DAG (الاستدعاء رقم 1 للـ LLM)
        plan = await self.planner_fn(user_goal)
        step_lookup = {s.step_id: s for s in plan.steps}

        # 2. بناء المجدول الطوبولوجي لاكتشاف الموجات المتوازية (Parallel Waves)
        sorter: TopologicalSorter[str] = TopologicalSorter()
        for step in plan.steps:
            sorter.add(step.step_id, *step.dependencies)
        sorter.prepare()

        evidence: dict[str, str] = {}
        wave_log: list[list[str]] = []

        # 3. حلقة التنفيذ الطوبولوجي الموزع بدون أي تدخل من الـ LLM!
        while sorter.is_active():
            ready_step_ids = list(sorter.get_ready())
            if not ready_step_ids:
                break

            wave_log.append(ready_step_ids)
            tasks: list[asyncio.Task[tuple[str, str]]] = []

            # تشغيل جميع الخطوات الجاهزة في هذه الموجة بالتوازي التام!
            async with asyncio.TaskGroup() as tg:
                for sid in ready_step_ids:
                    tasks.append(
                        tg.create_task(self._execute_step(step_lookup[sid], evidence))
                    )

            for task in tasks:
                sid, result_text = task.result()
                evidence[sid] = result_text
                sorter.done(sid)

        # 4. استدعاء الـ Solver لصياغة الإجابة النهائية من جدول الأدلة (الاستدعاء رقم 2 للـ LLM)
        final_answer = await self.solver_fn(user_goal, evidence)

        return ReWOOExecutionResult(
            final_answer=final_answer,
            total_steps_executed=len(plan.steps),
            parallel_waves_count=len(wave_log),
            llm_calls_count=2,  # دائماً استدعاءان فقط مهما بلغ عدد الأدوات!
            evidence_map=evidence,
            wave_log=wave_log,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


# =====================================================================
# الجزء الثاني: محرك التأمل والنقد الذاتي الموجه بالمعايير (Rubric Reflection Harness)
# =====================================================================
class RubricEvaluation(BaseModel):
    """عقد التقييم الصارم الذي يملؤه الناقد (Critic) في كل دورة مراجعة"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    security_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    actionable_defects: list[str] = Field(
        default_factory=list,
        description="قائمة محددة ودقيقة بالعيوب التي يجب على المولد إصلاحها",
    )

    @property
    def weighted_score(self) -> float:
        """حساب الدرجة الموزونة (الأمان له الوزن الأعلى 40%)"""
        return round(
            (0.4 * self.security_score)
            + (0.35 * self.accuracy_score)
            + (0.25 * self.completeness_score),
            3,
        )


@dataclass(slots=True)
class ReflectionResult[DraftT]:
    best_output: DraftT
    final_score: float
    iterations_run: int
    stop_reason: str
    score_progression: list[float] = field(default_factory=list)


class RubricReflectionHarness[DraftT]:
    """إطار التحكم بالتأمل والتحسين الذاتي (Generator-Critic Loop).

    يضمن إيقاف الحلقة عند بلوغ درجة الإتقان أو عند توقف التحسن (Diminishing Returns).
    """

    def __init__(
        self,
        *,
        generator_fn: Callable[[str, DraftT | None, RubricEvaluation | None], Awaitable[DraftT]],
        critic_fn: Callable[[str, DraftT], Awaitable[RubricEvaluation]],
        target_score: float = 0.90,
        max_iterations: int = 4,
        min_improvement_delta: float = 0.03,
    ) -> None:
        self.generator_fn = generator_fn
        self.critic_fn = critic_fn
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.min_improvement_delta = min_improvement_delta

    async def generate_with_reflection(self, task_spec: str) -> ReflectionResult[DraftT]:
        best_draft: DraftT | None = None
        best_score: float = -1.0
        current_draft: DraftT | None = None
        last_eval: RubricEvaluation | None = None
        progression: list[float] = []

        for iteration in range(1, self.max_iterations + 1):
            # 1. توليد المسودة الأولى أو تحسين المسودة السابقة بناءً على نقد الناقد
            current_draft = await self.generator_fn(task_spec, current_draft, last_eval)

            # 2. تقييم المسودة عبر الناقد المهيكل
            evaluation = await self.critic_fn(task_spec, current_draft)
            score = evaluation.weighted_score
            progression.append(score)

            # الاحتفاظ دائماً بأفضل نسخة تم الوصول إليها
            prev_best = best_score
            if score > best_score:
                best_score = score
                best_draft = current_draft

            # شرط التوقف 1: الوصول لدرجة الإتقان المستهدفة وخلو العمل من العيوب
            if score >= self.target_score and not evaluation.actionable_defects:
                assert best_draft is not None
                return ReflectionResult(
                    best_output=best_draft,
                    final_score=best_score,
                    iterations_run=iteration,
                    stop_reason="TARGET_SCORE_ACHIEVED",
                    score_progression=progression,
                )

            # شرط التوقف 2: كاشف تناقص العوائد (Plateau / Diminishing Returns Breaker)
            if iteration > 1 and (score - prev_best) < self.min_improvement_delta:
                assert best_draft is not None
                return ReflectionResult(
                    best_output=best_draft,
                    final_score=best_score,
                    iterations_run=iteration,
                    stop_reason="DIMINISHING_RETURNS_PLATEAU",
                    score_progression=progression,
                )

            last_eval = evaluation

        assert best_draft is not None
        return ReflectionResult(
            best_output=best_draft,
            final_score=best_score,
            iterations_run=self.max_iterations,
            stop_reason="MAX_ITERATIONS_REACHED",
            score_progression=progression,
        )


# =====================================================================
# 3. محاكاة تشغيلية مزدوجة: اختبار ReWOO المتوازي + اختبار حلقة النقد الذاتي
# =====================================================================
async def mock_rewoo_planner(goal: str) -> SymbolicExecutionPlan:
    await asyncio.sleep(0.03)
    return SymbolicExecutionPlan(
        strategy_overview="جلب بيانات الخادم وقاعدة البيانات بالتوازي، ثم فحص توافق الشهادات والتكوين.",
        steps=[
            # الموجة 1: خطوتان مستقلتان ستعملان معاً بالتوازي!
            SymbolicPlanStep(
                step_id="#E1",
                purpose="الحصول على عنوان IP للخادم الرئيسي",
                tool_name="resolve_host_ip",
                tool_args={"hostname": "api.enterprise.internal"},
            ),
            SymbolicPlanStep(
                step_id="#E2",
                purpose="جلب إصدار قاعدة البيانات النشطة",
                tool_name="get_db_version",
                tool_args={"cluster": "eu-central-1"},
            ),
            # الموجة 2: تعتمد على نتيجة #E1!
            SymbolicPlanStep(
                step_id="#E3",
                purpose="فحص المنافذ المفتوحة وشهادة TLS للعنوان المستخرج في #E1",
                tool_name="scan_tls_status",
                tool_args={"ip_address": "#E1"},
            ),
            # الموجة 3: تدمج نتيجة #E2 و #E3 معاً!
            SymbolicPlanStep(
                step_id="#E4",
                purpose="التحقق من توافق التشفير بين الخادم وقاعدة البيانات",
                tool_name="verify_compliance",
                tool_args={"tls_info": "#E3", "db_ver": "#E2"},
            ),
        ],
    )


async def mock_rewoo_solver(goal: str, evidence: dict[str, str]) -> str:
    await asyncio.sleep(0.02)
    return f"التقرير النهائي الموثق: الخادم ({evidence['#E1']}) متوافق تماماً -> [{evidence['#E4']}]."


async def main() -> None:
    print("─── 🚀 أولاً: تشغيل محرك ReWOODAGHarness (التخطيط الشبكي الرمزي المتوازي) ───")

    async def tool_resolve_ip(hostname: str) -> str:
        await asyncio.sleep(0.05)
        return "10.24.88.105"

    async def tool_db_ver(cluster: str) -> str:
        await asyncio.sleep(0.05)
        return "PostgreSQL-17.2-TLSv1.3"

    async def tool_scan_tls(ip_address: str) -> str:
        await asyncio.sleep(0.04)
        return f"TLSv1.3-Valid-On({ip_address})"

    async def tool_verify(tls_info: str, db_ver: str) -> str:
        await asyncio.sleep(0.03)
        return f"COMPLIANT({tls_info} <-> {db_ver})"

    rewoo = ReWOODAGHarness(
        planner_fn=mock_rewoo_planner,
        solver_fn=mock_rewoo_solver,
        tools={
            "resolve_host_ip": tool_resolve_ip,
            "get_db_version": tool_db_ver,
            "scan_tls_status": tool_scan_tls,
            "verify_compliance": tool_verify,
        },
    )

    res_rewoo = await rewoo.run("تحقق من التوافق الأمني بين خادم الـ API وقاعدة البيانات.")
    print(f"⚡ عدد الأدوات المنفذة  : {res_rewoo.total_steps_executed} أدوات مترابطة")
    print(f"🌊 الموجات المتوازية    : {res_rewoo.parallel_waves_count} موجات -> {res_rewoo.wave_log}")
    print(f"🧠 استدعاءات الـ LLM     : {res_rewoo.llm_calls_count} فقط (بدلاً من 5 في نمط ReAct!)")
    print("🗂️ جدول الأدلة الرمزية  :")
    for k, v in res_rewoo.evidence_map.items():
        print(f"   {k} => {v}")
    print(f"✅ النتيجة النهائية     : {res_rewoo.final_answer}\n")

    print("─── 🔍 ثانياً: تشغيل محرك RubricReflectionHarness (التأمل والنقد الذاتي المهيكل) ───")

    async def mock_code_generator(
        spec: str, prev_draft: str | None, critique: RubricEvaluation | None
    ) -> str:
        await asyncio.sleep(0.02)
        if prev_draft is None:
            # المسودة 1: تفتقر إلى تشفير كلمة المرور والتحقق من المدخلات
            return "def login(u, p): return db.query(f'SELECT * FROM users WHERE u={u} AND p={p}')"
        # المسودة 2: بعد قراءة نقد الناقد، يصلح ثغرة SQL Injection ويضيف التجزئة الآمنة
        return "def login(u: str, p_hash: str) -> User | None: return db.execute( SAFE_STMT, (u, p_hash) )"

    async def mock_security_critic(spec: str, draft: str) -> RubricEvaluation:
        await asyncio.sleep(0.02)
        if "f'SELECT" in draft:
            return RubricEvaluation(
                security_score=0.1,
                accuracy_score=0.6,
                completeness_score=0.5,
                actionable_defects=[
                    "Critical SQL Injection via f-string interpolation.",
                    "Missing type annotations and password hashing.",
                ],
            )
        return RubricEvaluation(
            security_score=0.98,
            accuracy_score=0.95,
            completeness_score=0.92,
            actionable_defects=[],
        )

    reflector = RubricReflectionHarness[str](
        generator_fn=mock_code_generator,
        critic_fn=mock_security_critic,
        target_score=0.90,
    )
    res_ref = await reflector.generate_with_reflection("اكتب دالة تسجيل دخول آمنة.")
    print(f"📈 تطور درجة الجودة    : {res_ref.score_progression}")
    print(f"🛑 سبب التوقف الذكي    : {res_ref.stop_reason} (في {res_ref.iterations_run} دورات)")
    print(f"💎 الكود النهائي المعتمد : {res_ref.best_output}")


if __name__ == "__main__":
    asyncio.run(main())
