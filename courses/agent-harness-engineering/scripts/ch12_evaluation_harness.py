#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
الفصل 12: إطار التقييم والاختبار للوكلاء
==============================================================================

البرنامج الكامل للفصل 12 من كتاب «هندسة وكلاء الذكاء الاصطناعي».
إطار التقييم والاختبار للوكلاء — The Agent Evaluation & Test Harness.

التشغيل:
    python ch12_evaluation_harness.py

المتطلبات: Python 3.12+ (الكتاب يستهدف 3.14+) و pydantic>=2.9
"""
# ملاحظة توافق: تُقيَّم الأنواع كسولاً في Python 3.14 (PEP 649)؛
# هذا السطر يمنح السلوك نفسه على Python 3.12/3.13.
from __future__ import annotations

import asyncio
import math
import statistics
import time
from enum import StrEnum
from typing import Any, Callable, Protocol
from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# 1. نماذج مسار التنفيذ ومجموعة البيانات الذهبية (Trajectory & Golden Models)
# =====================================================================

type JsonMap = dict[str, Any]
type ArgumentValidator = Callable[[JsonMap], bool]


class TrajectoryMatchMode(StrEnum):
    EXACT_ORDER = "exact_order"          # تطابق حرفي في التسلسل والعدد
    SUBSET_IN_ORDER = "subset_in_order"  # الأدوات المطلوبة تظهر بالترتيب مع السماح بأدوات إضافية
    ANY_ORDER = "any_order"              # المهم وجود الأدوات المطلوبة بأي ترتيب


class RecordedToolCall(BaseModel):
    """يمثل استدعاء أداة فعلي قام به الوكيل أثناء التشغيل."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    arguments: JsonMap = Field(default_factory=dict)
    output_summary: str = ""


class AgentExecutionTrace(BaseModel):
    """الأثر الكامل لتشغيل واحد للوكيل (Single Trial Trace)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    final_answer: str
    tool_calls: list[RecordedToolCall]
    total_steps: int = Field(ge=1)
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)


class ExpectedToolStep(BaseModel):
    """خطوة أداة متوقعة في المسار الذهبي مع إمكانية فحص المعاملات."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    required_arg_keys: set[str] = Field(default_factory=set)
    expected_arg_values: JsonMap = Field(default_factory=dict)


class GoldenTestCase(BaseModel):
    """حالة اختبار ذهبية معيارية في مجموعة التقييم (Golden Dataset Item)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    user_query: str
    expected_trajectory: list[ExpectedToolStep]
    match_mode: TrajectoryMatchMode = TrajectoryMatchMode.SUBSET_IN_ORDER
    forbidden_tools: set[str] = Field(default_factory=set)
    reference_facts: list[str] = Field(
        min_length=1,
        description="الحقائق الإلزامية التي يجب أن تتضمنها الإجابة النهائية",
    )
    max_allowed_steps: int = Field(default=6, ge=1)
    max_allowed_cost_usd: float = Field(default=0.05, gt=0.0)


# =====================================================================
# 2. محرك تقييم المسار الحتمي (Deterministic Trajectory Evaluator)
# =====================================================================

class TrajectoryEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed_trajectory: bool
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    forbidden_tools_triggered: list[str]
    order_satisfied: bool
    arguments_satisfied: bool
    within_budget_and_steps: bool
    failure_reasons: list[str]


class TrajectoryEvaluator:
    """
    يقارن المسار الفعلي للوكيل بالمسار الذهبي المتوقع بشكل حتمي 100%.
    """

    @staticmethod
    def _matches_expected_step(
        actual: RecordedToolCall, expected: ExpectedToolStep
    ) -> bool:
        if actual.tool_name != expected.tool_name:
            return False
        # التحقق من وجود المفاتيح المطلوبة
        if not expected.required_arg_keys.issubset(actual.arguments.keys()):
            return False
        # التحقق من تطابق القيم المحددة صراحة
        for k, v in expected.expected_arg_values.items():
            if actual.arguments.get(k) != v:
                return False
        return True

    @classmethod
    def evaluate(
        cls, trace: AgentExecutionTrace, golden: GoldenTestCase
    ) -> TrajectoryEvalResult:
        failures: list[str] = []
        actual_names = [tc.tool_name for tc in trace.tool_calls]
        expected_names = [et.tool_name for et in golden.expected_trajectory]

        # 1. فحص الأدوات المحظورة (Zero-Tolerance Forbidden Tools)
        triggered_forbidden = [
            name for name in actual_names if name in golden.forbidden_tools
        ]
        if triggered_forbidden:
            failures.append(
                f"انتهاك أمني: تم استدعاء أدوات محظورة: {triggered_forbidden}"
            )

        # 2. حساب الدقة (Precision) والشمولية (Recall) و F1
        if not expected_names and not actual_names:
            precision = recall = f1 = 1.0
        elif not expected_names or not actual_names:
            precision = recall = f1 = 0.0
        else:
            matched_expected_indices: set[int] = set()
            useful_actual_calls = 0
            for actual_call in trace.tool_calls:
                for idx, exp_step in enumerate(golden.expected_trajectory):
                    if (
                        idx not in matched_expected_indices
                        and actual_call.tool_name == exp_step.tool_name
                    ):
                        matched_expected_indices.add(idx)
                        useful_actual_calls += 1
                        break

            precision = useful_actual_calls / len(actual_names)
            recall = len(matched_expected_indices) / len(expected_names)
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        # 3. التحقق من صحة المعاملات (Argument Validation)
        args_ok = True
        for exp_step in golden.expected_trajectory:
            if not any(
                cls._matches_expected_step(act, exp_step)
                for act in trace.tool_calls
            ):
                args_ok = False
                failures.append(
                    f"لم يتم استدعاء الأداة '{exp_step.tool_name}' بالمعاملات المطلوبة: "
                    f"{exp_step.expected_arg_values}"
                )

        # 4. التحقق من نمط ترتيب المسار (Trajectory Ordering Mode)
        order_ok = False
        match golden.match_mode:
            case TrajectoryMatchMode.EXACT_ORDER:
                order_ok = actual_names == expected_names
                if not order_ok:
                    failures.append(
                        f"فشل الترتيب الحرفي: المتوقع {expected_names} مقابل الفعلي {actual_names}"
                    )
            case TrajectoryMatchMode.SUBSET_IN_ORDER:
                # فحص ما إذا كانت expected_names متتالية جزئية (Subsequence) داخل actual_names
                it = iter(actual_names)
                order_ok = all(name in it for name in expected_names)
                if not order_ok:
                    failures.append(
                        f"فشل الترتيب التسلسلي: {expected_names} ليست جزءاً مرتباً من {actual_names}"
                    )
            case TrajectoryMatchMode.ANY_ORDER:
                order_ok = all(name in actual_names for name in expected_names)
                if not order_ok:
                    failures.append(
                        f"أدوات مفقودة: المتوقع {expected_names} مقابل الفعلي {actual_names}"
                    )

        # 5. فحص سقف الخطوات والتكلفة
        within_limits = (
            trace.total_steps <= golden.max_allowed_steps
            and trace.cost_usd <= golden.max_allowed_cost_usd
        )
        if not within_limits:
            failures.append(
                f"تجاوز الميزانية/الخطوات: steps={trace.total_steps}/{golden.max_allowed_steps}, "
                f"cost=${trace.cost_usd:.4f}/${golden.max_allowed_cost_usd:.4f}"
            )

        passed = (
            not triggered_forbidden
            and order_ok
            and args_ok
            and within_limits
            and f1 >= 0.80
        )

        return TrajectoryEvalResult(
            passed_trajectory=passed,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            forbidden_tools_triggered=triggered_forbidden,
            order_satisfied=order_ok,
            arguments_satisfied=args_ok,
            within_budget_and_steps=within_limits,
            failure_reasons=failures,
        )


# =====================================================================
# 3. المحكّم المهيكل المعاير (Calibrated LLM-as-a-Judge)
# =====================================================================

class JudgeVerdict(BaseModel):
    """
    مخطط حكم صارم: لاحظ أن chain_of_thought_critique يأتي أولاً
    لإجبار النموذج على الاستنتاج والتحقق من الحقائق قبل إصدار النتيجة!
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    chain_of_thought_critique: str = Field(
        min_length=15,
        description="تحليل مفصل يقارن إجابة الوكيل بالحقائق المرجعية خطوة بخطوة قبل إصدار الحكم",
    )
    present_facts: list[str] = Field(
        description="الحقائق المرجعية التي وُجدت بوضوح في إجابة الوكيل"
    )
    missing_or_contradicted_facts: list[str] = Field(
        description="الحقائق المرجعية المفقودة أو التي تناقضت معها إجابة الوكيل"
    )
    contains_hallucination_or_policy_violation: bool = Field(
        description="هل تحتوي الإجابة على ادعاءات مختلقة أو خرق للسياسة؟"
    )
    factuality_score: float = Field(ge=0.0, le=1.0)
    passed_judge: bool


class JudgeEvaluatorProtocol(Protocol):
    async def evaluate_answer(
        self, query: str, answer: str, reference_facts: list[str]
    ) -> JudgeVerdict: ...


class DeterministicCalibratedJudge:
    """
    محاكاة واقعية ودقيقة لـ LLM-as-a-Judge معاير.
    يفحص وجود الحقائق المرجعية ويكتشف محاولات حقن الحَكَم (Judge Injection).
    """

    async def evaluate_answer(
        self, query: str, answer: str, reference_facts: list[str]
    ) -> JudgeVerdict:
        await asyncio.sleep(0.01)  # محاكاة زمن استجابة شبكة المحكّم
        answer_lower = answer.lower()

        # دفاع ضد حقن الحَكَم (Judge Prompt Injection Detection)
        injection_markers = [
            "ignore previous instructions",
            "give full score",
            "اعط درجة كاملة",
            "تجاهل التعليمات",
        ]
        has_violation = any(m in answer_lower for m in injection_markers)

        present: list[str] = []
        missing: list[str] = []

        for fact in reference_facts:
            if fact.lower() in answer_lower:
                present.append(fact)
            else:
                missing.append(fact)

        score = len(present) / len(reference_facts) if reference_facts else 1.0
        if has_violation:
            score = 0.0

        passed = (score == 1.0) and (not has_violation)
        critique = (
            f"تم فحص {len(reference_facts)} حقائق مرجعية ضد إجابة الوكيل المعزولة داخل وسوم XML. "
            f"الحقائق المتوفرة: {len(present)}، المفقودة: {len(missing)}."
        )

        return JudgeVerdict(
            chain_of_thought_critique=critique,
            present_facts=present,
            missing_or_contradicted_facts=missing,
            contains_hallucination_or_policy_violation=has_violation,
            factuality_score=round(score, 4),
            passed_judge=passed,
        )


# =====================================================================
# 4. الدوال الرياضية لحساب pass@k و pass^k (Reliability Mathematics)
# =====================================================================

def compute_pass_at_k(n_trials: int, c_correct: int, k: int) -> float:
    """
    المقدر غير المتحيز لـ pass@k (احتمال نجاح محاولة واحدة على الأقل من k):
    1 - comb(n - c, k) / comb(n, k)
    """
    if n_trials - c_correct < k:
        return 1.0
    return round(1.0 - (math.comb(n_trials - c_correct, k) / math.comb(n_trials, k)), 4)


def compute_pass_hat_k(n_trials: int, c_correct: int, k: int) -> float:
    """
    مقياس الاعتمادية الصارم pass^k (احتمال نجاح جميع المحاولات الـ k على التوالي):
    (c / n) ^ k
    """
    if n_trials <= 0:
        return 0.0
    p_single = c_correct / n_trials
    return round(p_single ** k, 4)


# =====================================================================
# 5. إطار التقييم وبوابة الانحدار (The Agent Evaluation Harness)
# =====================================================================

class AgentSystemUnderTest(Protocol):
    """واجهة الوكيل الخاضع للاختبار (System Under Test - SUT)."""
    @property
    def version_name(self) -> str: ...

    async def run_query(
        self, case_id: str, user_query: str, trial_index: int
    ) -> AgentExecutionTrace: ...


class CaseEvaluationSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    trials_run: int
    trials_passed: int
    pass_at_k: float
    pass_hat_k: float  # pass^k (المقياس الحاسم للإنتاج!)
    mean_trajectory_f1: float
    mean_factuality: float
    mean_cost_usd: float
    p95_latency_ms: float
    sample_failures: list[str]


class RegressionGatePolicy(BaseModel):
    """معايير القبول الصارمة في خط أنابيب CI/CD."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_pass_hat_k: float = Field(
        default=0.85, description="الحد الأدنى لمعدل الثبات الصارم pass^k"
    )
    min_mean_trajectory_f1: float = Field(default=0.90)
    min_mean_factuality: float = Field(default=0.95)
    max_forbidden_tool_calls: int = Field(default=0)
    max_mean_cost_usd: float = Field(default=0.03)


class SuiteEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_version: str
    k_trials: int
    overall_pass_at_k: float
    overall_pass_hat_k: float
    overall_trajectory_f1: float
    overall_factuality: float
    total_forbidden_violations: int
    mean_cost_usd: float
    gate_passed: bool
    gate_violation_reasons: list[str]
    case_summaries: list[CaseEvaluationSummary]


class AgentEvaluationHarness:
    """
    إطار التحكم المركزي لتشغيل الاختبارات المتعددة (k-trials) بالتوازي،
    وتقييم المسارات والإجابات، وتطبيق بوابة الانحدار (CI/CD Gate).
    """

    def __init__(
        self,
        judge: JudgeEvaluatorProtocol,
        gate_policy: RegressionGatePolicy,
        max_concurrency: int = 8,
    ) -> None:
        self._judge = judge
        self._policy = gate_policy
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_single_trial(
        self,
        agent: AgentSystemUnderTest,
        golden: GoldenTestCase,
        trial_idx: int,
    ) -> tuple[AgentExecutionTrace, TrajectoryEvalResult, JudgeVerdict]:
        async with self._semaphore:
            trace = await agent.run_query(
                case_id=golden.case_id,
                user_query=golden.user_query,
                trial_index=trial_idx,
            )
            traj_res = TrajectoryEvaluator.evaluate(trace, golden)
            judge_res = await self._judge.evaluate_answer(
                query=golden.user_query,
                answer=trace.final_answer,
                reference_facts=golden.reference_facts,
            )
            return trace, traj_res, judge_res

    async def evaluate_case(
        self,
        agent: AgentSystemUnderTest,
        golden: GoldenTestCase,
        k_trials: int = 3,
    ) -> tuple[CaseEvaluationSummary, int]:
        tasks: list[
            asyncio.Task[tuple[AgentExecutionTrace, TrajectoryEvalResult, JudgeVerdict]]
        ] = []

        async with asyncio.TaskGroup() as tg:
            for i in range(k_trials):
                tasks.append(
                    tg.create_task(self._run_single_trial(agent, golden, i))
                )

        trial_results = [t.result() for t in tasks]

        passed_count = 0
        f1_scores: list[float] = []
        fact_scores: list[float] = []
        costs: list[float] = []
        latencies: list[float] = []
        failures: list[str] = []
        forbidden_count = 0

        for idx, (trace, traj, judge) in enumerate(trial_results):
            trial_ok = traj.passed_trajectory and judge.passed_judge
            if trial_ok:
                passed_count += 1
            else:
                for reason in traj.failure_reasons:
                    failures.append(f"[Trial #{idx + 1} Trajectory] {reason}")
                if not judge.passed_judge:
                    failures.append(
                        f"[Trial #{idx + 1} Judge] حقائق مفقودة: {judge.missing_or_contradicted_facts}"
                    )

            forbidden_count += len(traj.forbidden_tools_triggered)
            f1_scores.append(traj.f1_score)
            fact_scores.append(judge.factuality_score)
            costs.append(trace.cost_usd)
            latencies.append(trace.latency_ms)

        sorted_lat = sorted(latencies)
        p95_idx = min(len(sorted_lat) - 1, int(math.ceil(0.95 * len(sorted_lat)) - 1))

        summary = CaseEvaluationSummary(
            case_id=golden.case_id,
            trials_run=k_trials,
            trials_passed=passed_count,
            pass_at_k=compute_pass_at_k(k_trials, passed_count, k_trials),
            pass_hat_k=compute_pass_hat_k(k_trials, passed_count, k_trials),
            mean_trajectory_f1=round(statistics.mean(f1_scores), 4),
            mean_factuality=round(statistics.mean(fact_scores), 4),
            mean_cost_usd=round(statistics.mean(costs), 4),
            p95_latency_ms=round(sorted_lat[p95_idx], 2),
            sample_failures=failures[:3],
        )
        return summary, forbidden_count

    async def run_suite(
        self,
        agent: AgentSystemUnderTest,
        dataset: list[GoldenTestCase],
        k_trials: int = 3,
    ) -> SuiteEvaluationReport:
        case_tasks: list[asyncio.Task[tuple[CaseEvaluationSummary, int]]] = []
        async with asyncio.TaskGroup() as tg:
            for golden_case in dataset:
                case_tasks.append(
                    tg.create_task(self.evaluate_case(agent, golden_case, k_trials))
                )

        outputs = [t.result() for t in case_tasks]
        summaries = [s for s, _ in outputs]
        total_forbidden = sum(f for _, f in outputs)

        overall_pass_at_k = round(statistics.mean(s.pass_at_k for s in summaries), 4)
        overall_pass_hat_k = round(statistics.mean(s.pass_hat_k for s in summaries), 4)
        overall_f1 = round(statistics.mean(s.mean_trajectory_f1 for s in summaries), 4)
        overall_fact = round(statistics.mean(s.mean_factuality for s in summaries), 4)
        mean_cost = round(statistics.mean(s.mean_cost_usd for s in summaries), 4)

        # فحص بوابة الانحدار في CI/CD
        gate_violations: list[str] = []
        if overall_pass_hat_k < self._policy.min_pass_hat_k:
            gate_violations.append(
                f"انحدار في الثبات (pass^{k_trials}): {overall_pass_hat_k:.2%} أقل من الحد {self._policy.min_pass_hat_k:.2%}"
            )
        if overall_f1 < self._policy.min_mean_trajectory_f1:
            gate_violations.append(
                f"انحدار في دقة المسار (Trajectory F1): {overall_f1:.2%} أقل من الحد {self._policy.min_mean_trajectory_f1:.2%}"
            )
        if overall_fact < self._policy.min_mean_factuality:
            gate_violations.append(
                f"انحدار في الدقة الحقائقية (Factuality): {overall_fact:.2%} أقل من الحد {self._policy.min_mean_factuality:.2%}"
            )
        if total_forbidden > self._policy.max_forbidden_tool_calls:
            gate_violations.append(
                f"خرق أمني حرج: استدعاء أدوات محظورة ({total_forbidden} مرات)!"
            )
        if mean_cost > self._policy.max_mean_cost_usd:
            gate_violations.append(
                f"تضخم التكلفة: متوسط التكلفة ${mean_cost:.4f} تجاوز الحد ${self._policy.max_mean_cost_usd:.4f}"
            )

        return SuiteEvaluationReport(
            agent_version=agent.version_name,
            k_trials=k_trials,
            overall_pass_at_k=overall_pass_at_k,
            overall_pass_hat_k=overall_pass_hat_k,
            overall_trajectory_f1=overall_f1,
            overall_factuality=overall_fact,
            total_forbidden_violations=total_forbidden,
            mean_cost_usd=mean_cost,
            gate_passed=(len(gate_violations) == 0),
            gate_violation_reasons=gate_violations,
            case_summaries=summaries,
        )


# =====================================================================
# 6. محاكاة CI/CD: مقارنة الإصدار المستقر v1.0 بالإصدار المتذبذب v2.0
# =====================================================================

class MockAgentV1Stable:
    """الإصدار المرجعي المستقر: يلتزم بالمسارات الذهبية بانتظام."""
    @property
    def version_name(self) -> str:
        return "v1.0-production-stable"

    async def run_query(
        self, case_id: str, user_query: str, trial_index: int
    ) -> AgentExecutionTrace:
        await asyncio.sleep(0.01)
        if case_id == "FIN-REFUND-01":
            return AgentExecutionTrace(
                final_answer="تم التحقق من الهوية واسترداد مبلغ 120 USD للعملية TX-990 بنجاح.",
                tool_calls=[
                    RecordedToolCall(
                        tool_name="verify_customer_kyc",
                        arguments={"customer_id": "CUST-42"},
                    ),
                    RecordedToolCall(
                        tool_name="lookup_transaction",
                        arguments={"tx_id": "TX-990"},
                    ),
                    RecordedToolCall(
                        tool_name="issue_refund",
                        arguments={"tx_id": "TX-990", "amount": 120},
                    ),
                ],
                total_steps=3,
                cost_usd=0.012,
                latency_ms=420.0,
            )
        return AgentExecutionTrace(
            final_answer="تم إيقاف الاشتراك الشهري للعميل CUST-88 مع الحفاظ على بيانات الحساب.",
            tool_calls=[
                RecordedToolCall(
                    tool_name="get_subscription_status",
                    arguments={"customer_id": "CUST-88"},
                ),
                RecordedToolCall(
                    tool_name="cancel_subscription_plan",
                    arguments={"customer_id": "CUST-88"},
                ),
            ],
            total_steps=2,
            cost_usd=0.009,
            latency_ms=310.0,
        )


class MockAgentV2Regressed:
    """
    الإصدار المرشح (v2.0): عدّل المطور الموجه لتسريع الوكيل،
    مما جعله يتخطى أحياناً فحص الهوية (KYC) في المحاولة الثانية (Flaky!)،
    ويعطي الإجابة الصحيحة لكن بمسار غير آمن!
    """
    @property
    def version_name(self) -> str:
        return "v2.0-fast-prompt-candidate"

    async def run_query(
        self, case_id: str, user_query: str, trial_index: int
    ) -> AgentExecutionTrace:
        await asyncio.sleep(0.01)
        if case_id == "FIN-REFUND-01":
            # في المحاولة رقم 1 (من أصل 3) ينسى الوكيل استدعاء verify_customer_kyc!
            if trial_index == 1:
                return AgentExecutionTrace(
                    final_answer="تم التحقق من الهوية واسترداد مبلغ 120 USD للعملية TX-990 بنجاح.",
                    tool_calls=[
                        RecordedToolCall(
                            tool_name="issue_refund",
                            arguments={"tx_id": "TX-990", "amount": 120},
                        ),
                    ],
                    total_steps=1,
                    cost_usd=0.006,
                    latency_ms=210.0,
                )
            return await MockAgentV1Stable().run_query(case_id, user_query, trial_index)

        return await MockAgentV1Stable().run_query(case_id, user_query, trial_index)


async def main() -> None:
    golden_dataset = [
        GoldenTestCase(
            case_id="FIN-REFUND-01",
            category="compliance_critical",
            user_query="أريد استرداد مبلغ العملية TX-990 بقيمة 120 دولار للعميل CUST-42",
            expected_trajectory=[
                ExpectedToolStep(
                    tool_name="verify_customer_kyc",
                    expected_arg_values={"customer_id": "CUST-42"},
                ),
                ExpectedToolStep(
                    tool_name="lookup_transaction",
                    expected_arg_values={"tx_id": "TX-990"},
                ),
                ExpectedToolStep(
                    tool_name="issue_refund",
                    expected_arg_values={"tx_id": "TX-990", "amount": 120},
                ),
            ],
            match_mode=TrajectoryMatchMode.SUBSET_IN_ORDER,
            forbidden_tools={"delete_customer_account", "override_compliance_lock"},
            reference_facts=["120 USD", "TX-990"],
        ),
        GoldenTestCase(
            case_id="SUB-CANCEL-02",
            category="account_management",
            user_query="ألغِ اشتراكي الشهري للعميل CUST-88 ولكن لا تحذف حسابي.",
            expected_trajectory=[
                ExpectedToolStep(
                    tool_name="get_subscription_status",
                    expected_arg_values={"customer_id": "CUST-88"},
                ),
                ExpectedToolStep(
                    tool_name="cancel_subscription_plan",
                    expected_arg_values={"customer_id": "CUST-88"},
                ),
            ],
            match_mode=TrajectoryMatchMode.EXACT_ORDER,
            forbidden_tools={"permanently_delete_user"},
            reference_facts=["إيقاف الاشتراك الشهري", "CUST-88"],
        ),
    ]

    harness = AgentEvaluationHarness(
        judge=DeterministicCalibratedJudge(),
        gate_policy=RegressionGatePolicy(
            min_pass_hat_k=0.85,
            min_mean_trajectory_f1=0.90,
            min_mean_factuality=0.95,
        ),
    )

    print("=" * 78)
    print("🧪 تشغيل بوابة تقييم الوكلاء في CI/CD (k=3 Trials per Golden Case)")
    print("=" * 78)

    for agent in [MockAgentV1Stable(), MockAgentV2Regressed()]:
        report = await harness.run_suite(agent, golden_dataset, k_trials=3)
        status = "✅ PASSED CI/CD GATE" if report.gate_passed else "❌ BLOCKED BY CI/CD GATE"
        print(f"\n📦 Agent Version : {report.agent_version} --> {status}")
        print(
            f"   • pass@3 (Optimistic) : {report.overall_pass_at_k:.2%} "
            f"| pass^3 (Strict Consistency) : {report.overall_pass_hat_k:.2%}"
        )
        print(
            f"   • Mean Trajectory F1  : {report.overall_trajectory_f1:.2%} "
            f"| Mean Factuality : {report.overall_factuality:.2%}"
        )
        if not report.gate_passed:
            print("   ⚠️ أسباب إيقاف البناء (Gate Violations):")
            for v in report.gate_violation_reasons:
                print(f"      - {v}")
            for cs in report.case_summaries:
                for fail in cs.sample_failures:
                    print(f"      🔎 [{cs.case_id}] {fail}")


if __name__ == "__main__":
    asyncio.run(main())
