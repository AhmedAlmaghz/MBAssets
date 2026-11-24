#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الوكلاء المستقلون المتقدمون (Agentic AI)
Large Language Model Engineering — Agentic

Book slug : llm-engineering-69-chapters
Chapter   : ch-67-chapter-67 (lesson 67)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-67-chapter-67.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الوكلاء المستقلون المتقدمون (Agentic AI)
Large Language Model Engineering — Agentic

Book slug : llm-engineering-69-chapters
Chapter   : ch-67-chapter-67 (lesson 67)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-67-chapter-67.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الوكلاء المستقلون المتقدمون (Agentic AI)
Large Language Model Engineering — Agentic

Book slug : llm-engineering-69-chapters
Chapter   : ch-67-chapter-67 (lesson 67)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_67.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 67: الفصل 67: الوكلاء المستقلون المتقدمون (Agentic AI)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agentic() -> None:
    states = ['plan', 'retrieve', 'verify', 'answer']
    allowed_transitions = {('plan', 'retrieve'), ('retrieve', 'verify'), ('verify', 'answer')}
    current, steps = ('plan', [])
    for proposed in ['retrieve', 'verify', 'answer']:
        if (current, proposed) not in allowed_transitions:
            raise RuntimeError('invalid state transition')
        steps.append(proposed)
        current = proposed
        if len(steps) >= 4:
            break
    print('state sequence:', ['plan', *steps])
    print('bounded steps:', len(steps), '| completed:', current == 'answer')

if __name__ == "__main__":
    print("Chapter 67 lab: Execute a bounded agent state machine")
    run_agentic()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
agent_core.py
بنية أساسية كاملة لوكيل يدعم نمطي ReAct وPlan-and-Execute،
مع واجهة أدوات (Tools) قابلة للتوسعة، وقابلة للتشغيل الفعلي.

ملاحظة: دالة call_llm هنا محاكاة توضيحية صريحة تحاكي سلوك نموذج لغوي
عبر قواعد منطقية مبسّطة؛ في الإنتاج الحقيقي تُستبدل باستدعاء
فعلي لـ API (OpenAI، Anthropic، Gemini) مع function calling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import json
import re


# =========================================================
# 1. تعريف الأدوات (Tools)
# =========================================================

@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., str]

    def execute(self, **kwargs) -> str:
        try:
            return self.function(**kwargs)
        except Exception as e:
            return f"ERROR: فشل تنفيذ الأداة '{self.name}': {str(e)}"


class ToolRegistry:
    """سجل مركزي للأدوات المتاحة للوكيل"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe_all(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )


# =========================================================
# 2. حالة الوكيل والذاكرة العاملة
# =========================================================

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: str
    tool_args: dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: str | None = None


@dataclass
class AgentState:
    """الذاكرة العاملة (scratchpad): كل ما يحتاجه الوكيل لتذكّر تقدّمه"""
    goal: str
    plan: list[PlanStep] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    max_replanning_attempts: int = 3
    replanning_count: int = 0

    def log(self, message: str) -> None:
        self.execution_log.append(message)

    def get_completed_results_summary(self) -> str:
        completed = [s for s in self.plan if s.status == StepStatus.COMPLETED]
        return "\n".join(f"[{s.step_id}] {s.description} → {s.result}" for s in completed)


# =========================================================
# 3. محاكاة استدعاء النموذج اللغوي (Simulated LLM)
# =========================================================

def simulated_planning_llm(goal: str, available_tools_description: str) -> list[dict]:
    """
    محاكاة تعليمية: في الواقع، هذا الاستدعاء يُرسل prompt حقيقي للنموذج
    يطلب منه إرجاع خطة بصيغة JSON منظمة (structured output / function calling).
    هنا نحاكي منطق تخطيط بسيط قائم على كلمات مفتاحية لأغراض العرض التعليمي.
    """
    plan = []
    step_id = 1

    if "رحلة" in goal or "طيران" in goal:
        plan.append({
            "step_id": step_id, "description": "البحث عن رحلات طيران متاحة",
            "tool_name": "search_flights", "tool_args": {"query": goal}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "التحقق من سياسة الإلغاء لأرخص رحلة",
            "tool_name": "check_cancellation_policy", "tool_args": {}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "تجميع التوصية النهائية",
            "tool_name": "summarize_recommendation", "tool_args": {}
        })
    else:
        plan.append({
            "step_id": step_id, "description": "البحث العام عن المعلومة المطلوبة",
            "tool_name": "web_search", "tool_args": {"query": goal}
        })

    return plan


def simulated_replanning_llm(
    original_goal: str, failed_step: PlanStep, state: AgentState
) -> list[dict]:
    """
    محاكاة إعادة التخطيط: عند فشل خطوة، نولّد خطة بديلة بناءً على
    ما تم إنجازه فعلًا + سبب الفشل. محاكاة تعليمية مبسّطة.
    """
    remaining_step_id = failed_step.step_id
    return [{
        "step_id": remaining_step_id,
        "description": f"محاولة بديلة بعد فشل: {failed_step.description}",
        "tool_name": "search_flights",
        "tool_args": {"query": original_goal, "fallback_mode": True},
    }]


# =========================================================
# 4. وكيل Plan-and-Execute الكامل
# =========================================================

class PlanAndExecuteAgent:
    def __init__(self, tool_registry: ToolRegistry, max_steps: int = 10):
        self.tools = tool_registry
        self.max_steps = max_steps

    def create_initial_plan(self, goal: str) -> list[PlanStep]:
        raw_plan = simulated_planning_llm(goal, self.tools.describe_all())
        return [
            PlanStep(
                step_id=p["step_id"],
                description=p["description"],
                tool_name=p["tool_name"],
                tool_args=p["tool_args"],
            )
            for p in raw_plan
        ]

    def execute_step(self, step: PlanStep, state: AgentState) -> bool:
        """ينفذ خطوة واحدة، ويرجع True عند النجاح"""
        step.status = StepStatus.IN_PROGRESS
        tool = self.tools.get(step.tool_name)

        if tool is None:
            step.status = StepStatus.FAILED
            step.result = f"أداة غير معروفة: {step.tool_name}"
            state.log(f"❌ خطوة {step.step_id} فشلت: أداة غير موجودة")
            return False

        result = tool.execute(**step.tool_args)

        if result.startswith("ERROR"):
            step.status = StepStatus.FAILED
            step.result = result
            state.log(f"❌ خطوة {step.step_id} فشلت: {result}")
            return False

        step.status = StepStatus.COMPLETED
        step.result = result
        state.log(f"✅ خطوة {step.step_id} اكتملت: {result[:60]}")
        return True

    def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        state.plan = self.create_initial_plan(goal)
        state.log(f"🎯 الهدف: {goal}")
        state.log(f"📋 الخطة الأولية: {len(state.plan)} خطوات")

        steps_executed = 0
        step_index = 0

        while step_index < len(state.plan) and steps_executed < self.max_steps:
            current_step = state.plan[step_index]
            success = self.execute_step(current_step, state)
            steps_executed += 1

            if not success:
                if state.replanning_count >= state.max_replanning_attempts:
                    state.log("🛑 تجاوزنا الحد الأقصى لمحاولات إعادة التخطيط. "
                               "إيقاف وطلب تدخل بشري.")
                    break

                state.replanning_count += 1
                state.log(f"🔄 إعادة تخطيط (محاولة {state.replanning_count})")

                new_raw_steps = simulated_replanning_llm(goal, current_step, state)
                new_steps = [
                    PlanStep(
                        step_id=s["step_id"], description=s["description"],
                        tool_name=s["tool_name"], tool_args=s["tool_args"],
                    )
                    for s in new_raw_steps
                ]
                state.plan[step_index:step_index + 1] = new_steps
                continue

            step_index += 1

        all_completed = all(s.status == StepStatus.COMPLETED for s in state.plan)
        state.log(f"\n{'✅ المهمة اكتملت بنجاح' if all_completed else '⚠️ المهمة لم تكتمل بالكامل'}")
        return state

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
mock_tools.py — أدوات محاكاة واقعية الشكل لاختبار الوكيل
(محاكاة تعليمية لـ API حجوزات طيران حقيقي)
"""
import random


def search_flights_impl(query: str, fallback_mode: bool = False) -> str:
    if fallback_mode:
        return json.dumps({
            "flights": [{"airline": "EgyptAir", "price": 850, "stops": 1}]
        }, ensure_ascii=False)

    # محاكاة فشل عشوائي 30% من الوقت لاختبار إعادة التخطيط
    if random.random() < 0.3:
        raise ConnectionError("انتهت مهلة الاتصال بمحرك الحجوزات")

    return json.dumps({
        "flights": [
            {"airline": "Qatar Airways", "price": 720, "stops": 0},
            {"airline": "Emirates", "price": 680, "stops": 1},
        ]
    }, ensure_ascii=False)


def check_cancellation_policy_impl() -> str:
    return "سياسة الإلغاء: استرداد كامل حتى 48 ساعة قبل الرحلة."


def summarize_recommendation_impl() -> str:
    return "التوصية: رحلة Emirates بسعر 680$ مع توقف واحد، أفضل قيمة مقابل السعر."


def web_search_impl(query: str) -> str:
    return f"نتائج بحث محاكاة عن: {query}"


import json  # نقلت للأعلى منطقيًا لكن أُبقيت هنا للوضوح التعليمي فقط

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
rewoo_pattern.py
محاكاة تعليمية لنمط ReWOO: تخطيط رمزي واحد → تنفيذ أدوات → حل نهائي واحد.
يوفّر استدعاءات LLM مقارنة بـ ReAct (استدعاءان ثابتان بدل n استدعاء).
"""
import re
from dataclasses import dataclass


@dataclass
class SymbolicStep:
    variable_name: str      # مثل "#E1"
    tool_name: str
    raw_args: str            # قد تحتوي على مراجع لمتغيرات سابقة مثل "#E1"


def simulated_rewoo_planner(goal: str) -> list[SymbolicStep]:
    """استدعاء LLM واحد ينتج خطة رمزية كاملة بدون تنفيذ فعلي لأي أداة بعد"""
    return [
        SymbolicStep("#E1", "search_flights", goal),
        SymbolicStep("#E2", "check_cancellation_policy", "#E1"),
        SymbolicStep("#E3", "summarize_recommendation", "#E1, #E2"),
    ]


def resolve_and_execute(
    steps: list[SymbolicStep], tool_registry: "ToolRegistry"
) -> dict[str, str]:
    """المرحلة الثانية (Worker): تنفيذ الأدوات فعليًا، استبدال المتغيرات بالقيم الحقيقية"""
    resolved_values: dict[str, str] = {}

    for step in steps:
        resolved_args = step.raw_args
        for var_name, value in resolved_values.items():
            resolved_args = resolved_args.replace(var_name, value)

        tool = tool_registry.get(step.tool_name)
        result = tool.execute(query=resolved_args) if step.tool_name == "search_flights" \
            else tool.execute()

        resolved_values[step.variable_name] = result

    return resolved_values


def simulated_rewoo_solver(goal: str, resolved_values: dict[str, str]) -> str:
    """استدعاء LLM أخير: يأخذ كل النتائج المجمّعة وينتج إجابة نهائية متماسكة"""
    all_evidence = "\n".join(f"{k}: {v}" for k, v in resolved_values.items())
    return f"بناءً على البحث الكامل حول '{goal}':\n{all_evidence}\n\n✅ التوصية جاهزة أعلاه."


# تشغيل المثال الكامل
tool_registry = ToolRegistry()
tool_registry.register(Tool("search_flights", "بحث رحلات", search_flights_impl))
tool_registry.register(Tool("check_cancellation_policy", "سياسة الإلغاء", check_cancellation_policy_impl))
tool_registry.register(Tool("summarize_recommendation", "تلخيص", summarize_recommendation_impl))

goal = "أريد رحلة من القاهرة إلى طوكيو"
plan = simulated_rewoo_planner(goal)                  # استدعاء LLM رقم 1 فقط
evidence = resolve_and_execute(plan, tool_registry)   # بدون أي استدعاء LLM إضافي
final_answer = simulated_rewoo_solver(goal, evidence) # استدعاء LLM رقم 2 فقط

print(final_answer)
print(f"\n📊 عدد استدعاءات LLM المستخدمة: 2 (بدل {len(plan) + 1} في ReAct التقليدي)")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
reflection_agent.py
تطبيق كامل لنمط Reflexion: توليد محاولة → تقييم → نقد ذاتي → محاولة محسّنة.
"""
from dataclasses import dataclass, field


@dataclass
class Attempt:
    attempt_number: int
    output: str
    passed_evaluation: bool
    critique: str | None = None


@dataclass
class ReflexionState:
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)


def simulated_generator_llm(task: str, lessons: list[str]) -> str:
    """
    محاكاة: توليد محاولة حل للمهمة، مع الاستفادة من الدروس السابقة
    إن وُجدت (تُضاف كجزء من الـ prompt في التطبيق الحقيقي).
    """
    if not lessons:
        return "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"

    if "بطيء" in " ".join(lessons):
        return (
            "def is_prime(n):\n"
            "    if n < 2:\n        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n"
            "    return True"
        )
    return "def is_prime(n): return n > 1"


def simulated_evaluator(output: str) -> tuple[bool, str | None]:
    """
    محاكاة تقييم: تحقق هل الحل يحتوي على تحسين الجذر التربيعي
    (معيار بسيط لأغراض العرض التعليمي فقط؛ في الواقع هذا يكون
    اختبار وحدات فعلي + تحليل أداء حقيقي).
    """
    if "** 0.5" in output or "sqrt" in output:
        return True, None
    return False, (
        "الحل يفحص كل الأعداد حتى n بدلاً من التوقف عند الجذر التربيعي، "
        "مما يجعله بطيء جدًا للأعداد الكبيرة (تعقيد O(n) بدل O(√n))."
    )


class ReflexionAgent:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def run(self, task: str) -> ReflexionState:
        state = ReflexionState(task=task)

        for attempt_num in range(1, self.max_attempts + 1):
            output = simulated_generator_llm(task, state.lessons_learned)
            passed, critique = simulated_evaluator(output)

            attempt = Attempt(
                attempt_number=attempt_num,
                output=output,
                passed_evaluation=passed,
                critique=critique,
            )
            state.attempts.append(attempt)

            print(f"\n--- المحاولة {attempt_num} ---")
            print(f"الناتج:\n{output}")
            print(f"النتيجة: {'✅ نجح' if passed else '❌ يحتاج تحسين'}")

            if passed:
                print("🎉 تم الوصول لحل مُرضٍ، إيقاف الحلقة.")
                break

            if critique:
                state.lessons_learned.append(critique)
                print(f"📝 درس مُستخلص: {critique}")

        return state


# تشغيل المثال
agent = ReflexionAgent(max_attempts=3)
final_state = agent.run("اكتب دالة بايثون فعّالة للتحقق من كون العدد أوليًا")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
multi_agent_orchestrator.py
نمط Supervisor: وكيل مدير يحلل المهمة، يوزّعها على وكلاء متخصصين،
يجمع النتائج، ويقرر إن كانت هناك حاجة لجولة تنسيق إضافية.
"""
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(Enum):
    RESEARCHER = "researcher"
    FINANCIAL_ANALYST = "financial_analyst"
    WRITER = "writer"
    SUPERVISOR = "supervisor"


@dataclass
class SubTask:
    assigned_role: AgentRole
    instruction: str
    result: str | None = None


class SpecialistAgent:
    """وكيل متخصص بدور واحد ضيق ومحدد جيدًا"""

    def __init__(self, role: AgentRole, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt

    def execute(self, instruction: str, shared_context: str = "") -> str:
        """
        محاكاة تنفيذ المهمة الفرعية. في الواقع هذا استدعاء LLM حقيقي
        مع self.system_prompt كتوجيه ثابت للدور المتخصص.
        """
        if self.role == AgentRole.RESEARCHER:
            return (f"[بحث] تم جمع بيانات عن: {instruction}. "
                    f"الشركة تعمل في قطاع التكنولوجيا، إيرادات 2024: 4.2 مليار دولار.")
        elif self.role == AgentRole.FINANCIAL_ANALYST:
            return (f"[تحليل مالي] بناءً على السياق: {shared_context[:50]}... "
                    f"هامش الربح 18%، نمو سنوي 12%، أعلى من متوسط القطاع (8%).")
        elif self.role == AgentRole.WRITER:
            return (f"[تقرير نهائي] بناءً على كل التحليلات المُجمَّعة:\n"
                    f"الشركة تُظهر أداءً ماليًا قويًا يفوق متوسط القطاع، "
                    f"نوصي بمتابعة الاستثمار مع مراقبة ربع سنوية.")
        return "مهمة غير معروفة."


class SupervisorAgent:
    """الوكيل المنسّق: يحلل الهدف الكلي، يبني خطة توزيع المهام، يجمع النتائج"""

    def __init__(self, specialists: dict[AgentRole, SpecialistAgent]):
        self.specialists = specialists

    def decompose_goal(self, goal: str) -> list[SubTask]:
        """
        محاكاة: في الواقع، استدعاء LLM يحلل الهدف الكلي
        ويقرر أي الأدوار المتخصصة مطلوبة وبأي ترتيب.
        """
        return [
            SubTask(AgentRole.RESEARCHER, f"اجمع بيانات أساسية عن: {goal}"),
            SubTask(AgentRole.FINANCIAL_ANALYST, "حلل الأداء المالي بناءً على البيانات المجمّعة"),
            SubTask(AgentRole.WRITER, "اكتب تقريرًا استثماريًا نهائيًا متماسكًا"),
        ]

    def run(self, goal: str) -> str:
        subtasks = self.decompose_goal(goal)
        shared_context = ""

        print(f"🎯 الهدف الكلي: {goal}")
        print(f"📋 تم توزيع المهمة على {len(subtasks)} وكلاء متخصصين\n")

        for task in subtasks:
            specialist = self.specialists[task.assigned_role]
            print(f"▶️  تفعيل الوكيل: {task.assigned_role.value}")

            result = specialist.execute(task.instruction, shared_context)
            task.result = result
            shared_context += f"\n{result}"

            print(f"   الناتج: {result}\n")

        final_report = subtasks[-1].result
        return final_report


# تشغيل المثال الكامل
specialists = {
    AgentRole.RESEARCHER: SpecialistAgent(
        AgentRole.RESEARCHER, "أنت باحث متخصص في جمع البيانات الأولية بدقة."
    ),
    AgentRole.FINANCIAL_ANALYST: SpecialistAgent(
        AgentRole.FINANCIAL_ANALYST, "أنت محلل مالي متخصص في تقييم الأداء المالي للشركات."
    ),
    AgentRole.WRITER: SpecialistAgent(
        AgentRole.WRITER, "أنت كاتب تقارير استثمارية محترف وموجز."
    ),
}

supervisor = SupervisorAgent(specialists)
final_output = supervisor.run("شركة تقنية ناشئة في قطاع الذكاء الاصطناعي")

print("=" * 60)
print("التقرير النهائي:")
print(final_output)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
research_agent_system.py
نظام متكامل End-to-End: وكيل بحثي مستقل يجمع
Plan-and-Execute + أدوات + Reflexion + تقرير نهائي منظم.

جميع استدعاءات "النموذج اللغوي" والأدوات هنا محاكاة تعليمية واضحة؛
البنية والمنطق قابلان للتوصيل المباشر بأي LLM API حقيقي.
"""
from dataclasses import dataclass, field
from enum import Enum
import json


# ---------- طبقة الأدوات ----------

def search_papers_tool(topic: str) -> str:
    """محاكاة بحث في قاعدة بيانات أوراق بحثية (مثل arXiv API)"""
    fake_db = {
        "ring attention": [
            "Liu et al. 2023 - Ring Attention with Blockwise Transformers",
            "Brandon et al. 2023 - Striped Attention",
        ],
        "test-time compute": [
            "Snell et al. 2024 - Scaling LLM Test-Time Compute Optimally",
            "DeepSeek-AI 2025 - DeepSeek-R1",
        ],
    }
    results = fake_db.get(topic.lower(), ["لا توجد نتائج مباشرة لهذا الموضوع"])
    return json.dumps(results, ensure_ascii=False)


def extract_key_findings_tool(papers_json: str) -> str:
    """محاكاة استخلاص النقاط الرئيسية من قائمة أوراق بحثية"""
    papers = json.loads(papers_json)
    if not papers or papers[0] == "لا توجد نتائج مباشرة لهذا الموضوع":
        return "لا توجد نتائج كافية لاستخلاص نتائج رئيسية."
    return f"تم استخلاص {len(papers)} نتيجة رئيسية من الأوراق المُسترجعة بنجاح."


def draft_summary_tool(findings: str, topic: str, include_citations: bool = False) -> str:
    """محاكاة صياغة ملخص نهائي، مع خيار تضمين الاستشهادات أو عدمه"""
    base = f"ملخص حول '{topic}': {findings}"
    if include_citations:
        base += " [تتضمن النسخة هذه استشهادات مرجعية كاملة لكل نتيجة.]"
    return base


# ---------- طبقة التقييم الذاتي (Reflexion) ----------

@dataclass
class QualityCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def evaluate_summary_quality(summary: str) -> QualityCheckResult:
    """
    محاكاة تقييم جودة: نتحقق من معيارين بسيطين
    (في الواقع، هذا قد يكون استدعاء LLM-as-judge كما في فصل التقييم).
    """
    issues = []
    if "استشهادات" not in summary:
        issues.append("الملخص لا يتضمن استشهادات مرجعية، وهو مطلوب لملخص أكاديمي.")
    if len(summary) < 30:
        issues.append("الملخص قصير جدًا وغير كافٍ.")

    return QualityCheckResult(passed=len(issues) == 0, issues=issues)


# ---------- الوكيل المتكامل ----------

class AutonomousResearchAgent:
    def __init__(self, max_reflection_attempts: int = 2):
        self.max_reflection_attempts = max_reflection_attempts
        self.execution_trace: list[str] = []

    def _log(self, message: str):
        self.execution_trace.append(message)
        print(message)

    def run(self, topic: str) -> str:
        self._log(f"🎯 بدء مهمة البحث حول: '{topic}'")

        # ----- المرحلة 1: التخطيط (مُبسَّط لثلاث خطوات ثابتة لهذا المثال) -----
        self._log("📋 الخطة: (1) بحث الأوراق (2) استخلاص النتائج (3) صياغة ملخص")

        # ----- المرحلة 2: التنفيذ -----
        papers_json = search_papers_tool(topic)
        self._log(f"🔍 نتائج البحث: {papers_json}")

        findings = extract_key_findings_tool(papers_json)
        self._log(f"🧩 النتائج المُستخلصة: {findings}")

        include_citations = False
        attempt = 0
        final_summary = ""

        # ----- المرحلة 3: التنفيذ + المراجعة الذاتية (Reflexion Loop) -----
        while attempt <= self.max_reflection_attempts:
            summary = draft_summary_tool(findings, topic, include_citations=include_citations)
            self._log(f"\n--- محاولة صياغة رقم {attempt + 1} ---")
            self._log(f"📝 المسودة: {summary}")

            quality = evaluate_summary_quality(summary)

            if quality.passed:
                self._log("✅ اجتاز الملخص فحص الجودة.")
                final_summary = summary
                break

            self._log(f"⚠️ مشاكل اكتُشفت: {quality.issues}")

            # تطبيق "الدرس المستفاد" على المحاولة القادمة
            if any("استشهادات" in issue for issue in quality.issues):
                include_citations = True
                self._log("📝 درس مُستخلص: سيُضاف دعم استشهادات في المحاولة القادمة.")

            attempt += 1
            final_summary = summary  # الاحتفاظ بآخر مسودة حتى لو لم تجتز الفحص بالكامل

        if attempt > self.max_reflection_attempts:
            self._log("🛑 تجاوزنا الحد الأقصى لمحاولات التحسين الذاتي. "
                       "تسليم أفضل نسخة متاحة مع تنبيه بوجود قيود.")

        self._log(f"\n🎉 التسليم النهائي:\n{final_summary}")
        return final_summary


# تشغيل النظام الكامل
agent = AutonomousResearchAgent(max_reflection_attempts=2)
result = agent.run("test-time compute")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 67: الفصل 67: الوكلاء المستقلون المتقدمون (Agentic AI)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agentic() -> None:
    states = ['plan', 'retrieve', 'verify', 'answer']
    allowed_transitions = {('plan', 'retrieve'), ('retrieve', 'verify'), ('verify', 'answer')}
    current, steps = ('plan', [])
    for proposed in ['retrieve', 'verify', 'answer']:
        if (current, proposed) not in allowed_transitions:
            raise RuntimeError('invalid state transition')
        steps.append(proposed)
        current = proposed
        if len(steps) >= 4:
            break
    print('state sequence:', ['plan', *steps])
    print('bounded steps:', len(steps), '| completed:', current == 'answer')

if __name__ == "__main__":
    print("Chapter 67 lab: Execute a bounded agent state machine")
    run_agentic()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
agent_core.py
بنية أساسية كاملة لوكيل يدعم نمطي ReAct وPlan-and-Execute،
مع واجهة أدوات (Tools) قابلة للتوسعة، وقابلة للتشغيل الفعلي.

ملاحظة: دالة call_llm هنا محاكاة توضيحية صريحة تحاكي سلوك نموذج لغوي
عبر قواعد منطقية مبسّطة؛ في الإنتاج الحقيقي تُستبدل باستدعاء
فعلي لـ API (OpenAI، Anthropic، Gemini) مع function calling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import json
import re


# =========================================================
# 1. تعريف الأدوات (Tools)
# =========================================================

@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., str]

    def execute(self, **kwargs) -> str:
        try:
            return self.function(**kwargs)
        except Exception as e:
            return f"ERROR: فشل تنفيذ الأداة '{self.name}': {str(e)}"


class ToolRegistry:
    """سجل مركزي للأدوات المتاحة للوكيل"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe_all(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )


# =========================================================
# 2. حالة الوكيل والذاكرة العاملة
# =========================================================

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: str
    tool_args: dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: str | None = None


@dataclass
class AgentState:
    """الذاكرة العاملة (scratchpad): كل ما يحتاجه الوكيل لتذكّر تقدّمه"""
    goal: str
    plan: list[PlanStep] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    max_replanning_attempts: int = 3
    replanning_count: int = 0

    def log(self, message: str) -> None:
        self.execution_log.append(message)

    def get_completed_results_summary(self) -> str:
        completed = [s for s in self.plan if s.status == StepStatus.COMPLETED]
        return "\n".join(f"[{s.step_id}] {s.description} → {s.result}" for s in completed)


# =========================================================
# 3. محاكاة استدعاء النموذج اللغوي (Simulated LLM)
# =========================================================

def simulated_planning_llm(goal: str, available_tools_description: str) -> list[dict]:
    """
    محاكاة تعليمية: في الواقع، هذا الاستدعاء يُرسل prompt حقيقي للنموذج
    يطلب منه إرجاع خطة بصيغة JSON منظمة (structured output / function calling).
    هنا نحاكي منطق تخطيط بسيط قائم على كلمات مفتاحية لأغراض العرض التعليمي.
    """
    plan = []
    step_id = 1

    if "رحلة" in goal or "طيران" in goal:
        plan.append({
            "step_id": step_id, "description": "البحث عن رحلات طيران متاحة",
            "tool_name": "search_flights", "tool_args": {"query": goal}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "التحقق من سياسة الإلغاء لأرخص رحلة",
            "tool_name": "check_cancellation_policy", "tool_args": {}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "تجميع التوصية النهائية",
            "tool_name": "summarize_recommendation", "tool_args": {}
        })
    else:
        plan.append({
            "step_id": step_id, "description": "البحث العام عن المعلومة المطلوبة",
            "tool_name": "web_search", "tool_args": {"query": goal}
        })

    return plan


def simulated_replanning_llm(
    original_goal: str, failed_step: PlanStep, state: AgentState
) -> list[dict]:
    """
    محاكاة إعادة التخطيط: عند فشل خطوة، نولّد خطة بديلة بناءً على
    ما تم إنجازه فعلًا + سبب الفشل. محاكاة تعليمية مبسّطة.
    """
    remaining_step_id = failed_step.step_id
    return [{
        "step_id": remaining_step_id,
        "description": f"محاولة بديلة بعد فشل: {failed_step.description}",
        "tool_name": "search_flights",
        "tool_args": {"query": original_goal, "fallback_mode": True},
    }]


# =========================================================
# 4. وكيل Plan-and-Execute الكامل
# =========================================================

class PlanAndExecuteAgent:
    def __init__(self, tool_registry: ToolRegistry, max_steps: int = 10):
        self.tools = tool_registry
        self.max_steps = max_steps

    def create_initial_plan(self, goal: str) -> list[PlanStep]:
        raw_plan = simulated_planning_llm(goal, self.tools.describe_all())
        return [
            PlanStep(
                step_id=p["step_id"],
                description=p["description"],
                tool_name=p["tool_name"],
                tool_args=p["tool_args"],
            )
            for p in raw_plan
        ]

    def execute_step(self, step: PlanStep, state: AgentState) -> bool:
        """ينفذ خطوة واحدة، ويرجع True عند النجاح"""
        step.status = StepStatus.IN_PROGRESS
        tool = self.tools.get(step.tool_name)

        if tool is None:
            step.status = StepStatus.FAILED
            step.result = f"أداة غير معروفة: {step.tool_name}"
            state.log(f"❌ خطوة {step.step_id} فشلت: أداة غير موجودة")
            return False

        result = tool.execute(**step.tool_args)

        if result.startswith("ERROR"):
            step.status = StepStatus.FAILED
            step.result = result
            state.log(f"❌ خطوة {step.step_id} فشلت: {result}")
            return False

        step.status = StepStatus.COMPLETED
        step.result = result
        state.log(f"✅ خطوة {step.step_id} اكتملت: {result[:60]}")
        return True

    def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        state.plan = self.create_initial_plan(goal)
        state.log(f"🎯 الهدف: {goal}")
        state.log(f"📋 الخطة الأولية: {len(state.plan)} خطوات")

        steps_executed = 0
        step_index = 0

        while step_index < len(state.plan) and steps_executed < self.max_steps:
            current_step = state.plan[step_index]
            success = self.execute_step(current_step, state)
            steps_executed += 1

            if not success:
                if state.replanning_count >= state.max_replanning_attempts:
                    state.log("🛑 تجاوزنا الحد الأقصى لمحاولات إعادة التخطيط. "
                               "إيقاف وطلب تدخل بشري.")
                    break

                state.replanning_count += 1
                state.log(f"🔄 إعادة تخطيط (محاولة {state.replanning_count})")

                new_raw_steps = simulated_replanning_llm(goal, current_step, state)
                new_steps = [
                    PlanStep(
                        step_id=s["step_id"], description=s["description"],
                        tool_name=s["tool_name"], tool_args=s["tool_args"],
                    )
                    for s in new_raw_steps
                ]
                state.plan[step_index:step_index + 1] = new_steps
                continue

            step_index += 1

        all_completed = all(s.status == StepStatus.COMPLETED for s in state.plan)
        state.log(f"\n{'✅ المهمة اكتملت بنجاح' if all_completed else '⚠️ المهمة لم تكتمل بالكامل'}")
        return state

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
mock_tools.py — أدوات محاكاة واقعية الشكل لاختبار الوكيل
(محاكاة تعليمية لـ API حجوزات طيران حقيقي)
"""
import random


def search_flights_impl(query: str, fallback_mode: bool = False) -> str:
    if fallback_mode:
        return json.dumps({
            "flights": [{"airline": "EgyptAir", "price": 850, "stops": 1}]
        }, ensure_ascii=False)

    # محاكاة فشل عشوائي 30% من الوقت لاختبار إعادة التخطيط
    if random.random() < 0.3:
        raise ConnectionError("انتهت مهلة الاتصال بمحرك الحجوزات")

    return json.dumps({
        "flights": [
            {"airline": "Qatar Airways", "price": 720, "stops": 0},
            {"airline": "Emirates", "price": 680, "stops": 1},
        ]
    }, ensure_ascii=False)


def check_cancellation_policy_impl() -> str:
    return "سياسة الإلغاء: استرداد كامل حتى 48 ساعة قبل الرحلة."


def summarize_recommendation_impl() -> str:
    return "التوصية: رحلة Emirates بسعر 680$ مع توقف واحد، أفضل قيمة مقابل السعر."


def web_search_impl(query: str) -> str:
    return f"نتائج بحث محاكاة عن: {query}"


import json  # نقلت للأعلى منطقيًا لكن أُبقيت هنا للوضوح التعليمي فقط

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
rewoo_pattern.py
محاكاة تعليمية لنمط ReWOO: تخطيط رمزي واحد → تنفيذ أدوات → حل نهائي واحد.
يوفّر استدعاءات LLM مقارنة بـ ReAct (استدعاءان ثابتان بدل n استدعاء).
"""
import re
from dataclasses import dataclass


@dataclass
class SymbolicStep:
    variable_name: str      # مثل "#E1"
    tool_name: str
    raw_args: str            # قد تحتوي على مراجع لمتغيرات سابقة مثل "#E1"


def simulated_rewoo_planner(goal: str) -> list[SymbolicStep]:
    """استدعاء LLM واحد ينتج خطة رمزية كاملة بدون تنفيذ فعلي لأي أداة بعد"""
    return [
        SymbolicStep("#E1", "search_flights", goal),
        SymbolicStep("#E2", "check_cancellation_policy", "#E1"),
        SymbolicStep("#E3", "summarize_recommendation", "#E1, #E2"),
    ]


def resolve_and_execute(
    steps: list[SymbolicStep], tool_registry: "ToolRegistry"
) -> dict[str, str]:
    """المرحلة الثانية (Worker): تنفيذ الأدوات فعليًا، استبدال المتغيرات بالقيم الحقيقية"""
    resolved_values: dict[str, str] = {}

    for step in steps:
        resolved_args = step.raw_args
        for var_name, value in resolved_values.items():
            resolved_args = resolved_args.replace(var_name, value)

        tool = tool_registry.get(step.tool_name)
        result = tool.execute(query=resolved_args) if step.tool_name == "search_flights" \
            else tool.execute()

        resolved_values[step.variable_name] = result

    return resolved_values


def simulated_rewoo_solver(goal: str, resolved_values: dict[str, str]) -> str:
    """استدعاء LLM أخير: يأخذ كل النتائج المجمّعة وينتج إجابة نهائية متماسكة"""
    all_evidence = "\n".join(f"{k}: {v}" for k, v in resolved_values.items())
    return f"بناءً على البحث الكامل حول '{goal}':\n{all_evidence}\n\n✅ التوصية جاهزة أعلاه."


# تشغيل المثال الكامل
tool_registry = ToolRegistry()
tool_registry.register(Tool("search_flights", "بحث رحلات", search_flights_impl))
tool_registry.register(Tool("check_cancellation_policy", "سياسة الإلغاء", check_cancellation_policy_impl))
tool_registry.register(Tool("summarize_recommendation", "تلخيص", summarize_recommendation_impl))

goal = "أريد رحلة من القاهرة إلى طوكيو"
plan = simulated_rewoo_planner(goal)                  # استدعاء LLM رقم 1 فقط
evidence = resolve_and_execute(plan, tool_registry)   # بدون أي استدعاء LLM إضافي
final_answer = simulated_rewoo_solver(goal, evidence) # استدعاء LLM رقم 2 فقط

print(final_answer)
print(f"\n📊 عدد استدعاءات LLM المستخدمة: 2 (بدل {len(plan) + 1} في ReAct التقليدي)")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
reflection_agent.py
تطبيق كامل لنمط Reflexion: توليد محاولة → تقييم → نقد ذاتي → محاولة محسّنة.
"""
from dataclasses import dataclass, field


@dataclass
class Attempt:
    attempt_number: int
    output: str
    passed_evaluation: bool
    critique: str | None = None


@dataclass
class ReflexionState:
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)


def simulated_generator_llm(task: str, lessons: list[str]) -> str:
    """
    محاكاة: توليد محاولة حل للمهمة، مع الاستفادة من الدروس السابقة
    إن وُجدت (تُضاف كجزء من الـ prompt في التطبيق الحقيقي).
    """
    if not lessons:
        return "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"

    if "بطيء" in " ".join(lessons):
        return (
            "def is_prime(n):\n"
            "    if n < 2:\n        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n"
            "    return True"
        )
    return "def is_prime(n): return n > 1"


def simulated_evaluator(output: str) -> tuple[bool, str | None]:
    """
    محاكاة تقييم: تحقق هل الحل يحتوي على تحسين الجذر التربيعي
    (معيار بسيط لأغراض العرض التعليمي فقط؛ في الواقع هذا يكون
    اختبار وحدات فعلي + تحليل أداء حقيقي).
    """
    if "** 0.5" in output or "sqrt" in output:
        return True, None
    return False, (
        "الحل يفحص كل الأعداد حتى n بدلاً من التوقف عند الجذر التربيعي، "
        "مما يجعله بطيء جدًا للأعداد الكبيرة (تعقيد O(n) بدل O(√n))."
    )


class ReflexionAgent:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def run(self, task: str) -> ReflexionState:
        state = ReflexionState(task=task)

        for attempt_num in range(1, self.max_attempts + 1):
            output = simulated_generator_llm(task, state.lessons_learned)
            passed, critique = simulated_evaluator(output)

            attempt = Attempt(
                attempt_number=attempt_num,
                output=output,
                passed_evaluation=passed,
                critique=critique,
            )
            state.attempts.append(attempt)

            print(f"\n--- المحاولة {attempt_num} ---")
            print(f"الناتج:\n{output}")
            print(f"النتيجة: {'✅ نجح' if passed else '❌ يحتاج تحسين'}")

            if passed:
                print("🎉 تم الوصول لحل مُرضٍ، إيقاف الحلقة.")
                break

            if critique:
                state.lessons_learned.append(critique)
                print(f"📝 درس مُستخلص: {critique}")

        return state


# تشغيل المثال
agent = ReflexionAgent(max_attempts=3)
final_state = agent.run("اكتب دالة بايثون فعّالة للتحقق من كون العدد أوليًا")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
multi_agent_orchestrator.py
نمط Supervisor: وكيل مدير يحلل المهمة، يوزّعها على وكلاء متخصصين،
يجمع النتائج، ويقرر إن كانت هناك حاجة لجولة تنسيق إضافية.
"""
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(Enum):
    RESEARCHER = "researcher"
    FINANCIAL_ANALYST = "financial_analyst"
    WRITER = "writer"
    SUPERVISOR = "supervisor"


@dataclass
class SubTask:
    assigned_role: AgentRole
    instruction: str
    result: str | None = None


class SpecialistAgent:
    """وكيل متخصص بدور واحد ضيق ومحدد جيدًا"""

    def __init__(self, role: AgentRole, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt

    def execute(self, instruction: str, shared_context: str = "") -> str:
        """
        محاكاة تنفيذ المهمة الفرعية. في الواقع هذا استدعاء LLM حقيقي
        مع self.system_prompt كتوجيه ثابت للدور المتخصص.
        """
        if self.role == AgentRole.RESEARCHER:
            return (f"[بحث] تم جمع بيانات عن: {instruction}. "
                    f"الشركة تعمل في قطاع التكنولوجيا، إيرادات 2024: 4.2 مليار دولار.")
        elif self.role == AgentRole.FINANCIAL_ANALYST:
            return (f"[تحليل مالي] بناءً على السياق: {shared_context[:50]}... "
                    f"هامش الربح 18%، نمو سنوي 12%، أعلى من متوسط القطاع (8%).")
        elif self.role == AgentRole.WRITER:
            return (f"[تقرير نهائي] بناءً على كل التحليلات المُجمَّعة:\n"
                    f"الشركة تُظهر أداءً ماليًا قويًا يفوق متوسط القطاع، "
                    f"نوصي بمتابعة الاستثمار مع مراقبة ربع سنوية.")
        return "مهمة غير معروفة."


class SupervisorAgent:
    """الوكيل المنسّق: يحلل الهدف الكلي، يبني خطة توزيع المهام، يجمع النتائج"""

    def __init__(self, specialists: dict[AgentRole, SpecialistAgent]):
        self.specialists = specialists

    def decompose_goal(self, goal: str) -> list[SubTask]:
        """
        محاكاة: في الواقع، استدعاء LLM يحلل الهدف الكلي
        ويقرر أي الأدوار المتخصصة مطلوبة وبأي ترتيب.
        """
        return [
            SubTask(AgentRole.RESEARCHER, f"اجمع بيانات أساسية عن: {goal}"),
            SubTask(AgentRole.FINANCIAL_ANALYST, "حلل الأداء المالي بناءً على البيانات المجمّعة"),
            SubTask(AgentRole.WRITER, "اكتب تقريرًا استثماريًا نهائيًا متماسكًا"),
        ]

    def run(self, goal: str) -> str:
        subtasks = self.decompose_goal(goal)
        shared_context = ""

        print(f"🎯 الهدف الكلي: {goal}")
        print(f"📋 تم توزيع المهمة على {len(subtasks)} وكلاء متخصصين\n")

        for task in subtasks:
            specialist = self.specialists[task.assigned_role]
            print(f"▶️  تفعيل الوكيل: {task.assigned_role.value}")

            result = specialist.execute(task.instruction, shared_context)
            task.result = result
            shared_context += f"\n{result}"

            print(f"   الناتج: {result}\n")

        final_report = subtasks[-1].result
        return final_report


# تشغيل المثال الكامل
specialists = {
    AgentRole.RESEARCHER: SpecialistAgent(
        AgentRole.RESEARCHER, "أنت باحث متخصص في جمع البيانات الأولية بدقة."
    ),
    AgentRole.FINANCIAL_ANALYST: SpecialistAgent(
        AgentRole.FINANCIAL_ANALYST, "أنت محلل مالي متخصص في تقييم الأداء المالي للشركات."
    ),
    AgentRole.WRITER: SpecialistAgent(
        AgentRole.WRITER, "أنت كاتب تقارير استثمارية محترف وموجز."
    ),
}

supervisor = SupervisorAgent(specialists)
final_output = supervisor.run("شركة تقنية ناشئة في قطاع الذكاء الاصطناعي")

print("=" * 60)
print("التقرير النهائي:")
print(final_output)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
research_agent_system.py
نظام متكامل End-to-End: وكيل بحثي مستقل يجمع
Plan-and-Execute + أدوات + Reflexion + تقرير نهائي منظم.

جميع استدعاءات "النموذج اللغوي" والأدوات هنا محاكاة تعليمية واضحة؛
البنية والمنطق قابلان للتوصيل المباشر بأي LLM API حقيقي.
"""
from dataclasses import dataclass, field
from enum import Enum
import json


# ---------- طبقة الأدوات ----------

def search_papers_tool(topic: str) -> str:
    """محاكاة بحث في قاعدة بيانات أوراق بحثية (مثل arXiv API)"""
    fake_db = {
        "ring attention": [
            "Liu et al. 2023 - Ring Attention with Blockwise Transformers",
            "Brandon et al. 2023 - Striped Attention",
        ],
        "test-time compute": [
            "Snell et al. 2024 - Scaling LLM Test-Time Compute Optimally",
            "DeepSeek-AI 2025 - DeepSeek-R1",
        ],
    }
    results = fake_db.get(topic.lower(), ["لا توجد نتائج مباشرة لهذا الموضوع"])
    return json.dumps(results, ensure_ascii=False)


def extract_key_findings_tool(papers_json: str) -> str:
    """محاكاة استخلاص النقاط الرئيسية من قائمة أوراق بحثية"""
    papers = json.loads(papers_json)
    if not papers or papers[0] == "لا توجد نتائج مباشرة لهذا الموضوع":
        return "لا توجد نتائج كافية لاستخلاص نتائج رئيسية."
    return f"تم استخلاص {len(papers)} نتيجة رئيسية من الأوراق المُسترجعة بنجاح."


def draft_summary_tool(findings: str, topic: str, include_citations: bool = False) -> str:
    """محاكاة صياغة ملخص نهائي، مع خيار تضمين الاستشهادات أو عدمه"""
    base = f"ملخص حول '{topic}': {findings}"
    if include_citations:
        base += " [تتضمن النسخة هذه استشهادات مرجعية كاملة لكل نتيجة.]"
    return base


# ---------- طبقة التقييم الذاتي (Reflexion) ----------

@dataclass
class QualityCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def evaluate_summary_quality(summary: str) -> QualityCheckResult:
    """
    محاكاة تقييم جودة: نتحقق من معيارين بسيطين
    (في الواقع، هذا قد يكون استدعاء LLM-as-judge كما في فصل التقييم).
    """
    issues = []
    if "استشهادات" not in summary:
        issues.append("الملخص لا يتضمن استشهادات مرجعية، وهو مطلوب لملخص أكاديمي.")
    if len(summary) < 30:
        issues.append("الملخص قصير جدًا وغير كافٍ.")

    return QualityCheckResult(passed=len(issues) == 0, issues=issues)


# ---------- الوكيل المتكامل ----------

class AutonomousResearchAgent:
    def __init__(self, max_reflection_attempts: int = 2):
        self.max_reflection_attempts = max_reflection_attempts
        self.execution_trace: list[str] = []

    def _log(self, message: str):
        self.execution_trace.append(message)
        print(message)

    def run(self, topic: str) -> str:
        self._log(f"🎯 بدء مهمة البحث حول: '{topic}'")

        # ----- المرحلة 1: التخطيط (مُبسَّط لثلاث خطوات ثابتة لهذا المثال) -----
        self._log("📋 الخطة: (1) بحث الأوراق (2) استخلاص النتائج (3) صياغة ملخص")

        # ----- المرحلة 2: التنفيذ -----
        papers_json = search_papers_tool(topic)
        self._log(f"🔍 نتائج البحث: {papers_json}")

        findings = extract_key_findings_tool(papers_json)
        self._log(f"🧩 النتائج المُستخلصة: {findings}")

        include_citations = False
        attempt = 0
        final_summary = ""

        # ----- المرحلة 3: التنفيذ + المراجعة الذاتية (Reflexion Loop) -----
        while attempt <= self.max_reflection_attempts:
            summary = draft_summary_tool(findings, topic, include_citations=include_citations)
            self._log(f"\n--- محاولة صياغة رقم {attempt + 1} ---")
            self._log(f"📝 المسودة: {summary}")

            quality = evaluate_summary_quality(summary)

            if quality.passed:
                self._log("✅ اجتاز الملخص فحص الجودة.")
                final_summary = summary
                break

            self._log(f"⚠️ مشاكل اكتُشفت: {quality.issues}")

            # تطبيق "الدرس المستفاد" على المحاولة القادمة
            if any("استشهادات" in issue for issue in quality.issues):
                include_citations = True
                self._log("📝 درس مُستخلص: سيُضاف دعم استشهادات في المحاولة القادمة.")

            attempt += 1
            final_summary = summary  # الاحتفاظ بآخر مسودة حتى لو لم تجتز الفحص بالكامل

        if attempt > self.max_reflection_attempts:
            self._log("🛑 تجاوزنا الحد الأقصى لمحاولات التحسين الذاتي. "
                       "تسليم أفضل نسخة متاحة مع تنبيه بوجود قيود.")

        self._log(f"\n🎉 التسليم النهائي:\n{final_summary}")
        return final_summary


# تشغيل النظام الكامل
agent = AutonomousResearchAgent(max_reflection_attempts=2)
result = agent.run("test-time compute")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الوكلاء المستقلون المتقدمون (Agentic AI)
Large Language Model Engineering — Agentic

Book slug : llm-engineering-69-chapters
Chapter   : ch-67-chapter-67 (lesson 67)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_67.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 67: الفصل 67: الوكلاء المستقلون المتقدمون (Agentic AI)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agentic() -> None:
    states = ['plan', 'retrieve', 'verify', 'answer']
    allowed_transitions = {('plan', 'retrieve'), ('retrieve', 'verify'), ('verify', 'answer')}
    current, steps = ('plan', [])
    for proposed in ['retrieve', 'verify', 'answer']:
        if (current, proposed) not in allowed_transitions:
            raise RuntimeError('invalid state transition')
        steps.append(proposed)
        current = proposed
        if len(steps) >= 4:
            break
    print('state sequence:', ['plan', *steps])
    print('bounded steps:', len(steps), '| completed:', current == 'answer')

if __name__ == "__main__":
    print("Chapter 67 lab: Execute a bounded agent state machine")
    run_agentic()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
agent_core.py
بنية أساسية كاملة لوكيل يدعم نمطي ReAct وPlan-and-Execute،
مع واجهة أدوات (Tools) قابلة للتوسعة، وقابلة للتشغيل الفعلي.

ملاحظة: دالة call_llm هنا محاكاة توضيحية صريحة تحاكي سلوك نموذج لغوي
عبر قواعد منطقية مبسّطة؛ في الإنتاج الحقيقي تُستبدل باستدعاء
فعلي لـ API (OpenAI، Anthropic، Gemini) مع function calling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import json
import re


# =========================================================
# 1. تعريف الأدوات (Tools)
# =========================================================

@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., str]

    def execute(self, **kwargs) -> str:
        try:
            return self.function(**kwargs)
        except Exception as e:
            return f"ERROR: فشل تنفيذ الأداة '{self.name}': {str(e)}"


class ToolRegistry:
    """سجل مركزي للأدوات المتاحة للوكيل"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe_all(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )


# =========================================================
# 2. حالة الوكيل والذاكرة العاملة
# =========================================================

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: str
    tool_args: dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: str | None = None


@dataclass
class AgentState:
    """الذاكرة العاملة (scratchpad): كل ما يحتاجه الوكيل لتذكّر تقدّمه"""
    goal: str
    plan: list[PlanStep] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    max_replanning_attempts: int = 3
    replanning_count: int = 0

    def log(self, message: str) -> None:
        self.execution_log.append(message)

    def get_completed_results_summary(self) -> str:
        completed = [s for s in self.plan if s.status == StepStatus.COMPLETED]
        return "\n".join(f"[{s.step_id}] {s.description} → {s.result}" for s in completed)


# =========================================================
# 3. محاكاة استدعاء النموذج اللغوي (Simulated LLM)
# =========================================================

def simulated_planning_llm(goal: str, available_tools_description: str) -> list[dict]:
    """
    محاكاة تعليمية: في الواقع، هذا الاستدعاء يُرسل prompt حقيقي للنموذج
    يطلب منه إرجاع خطة بصيغة JSON منظمة (structured output / function calling).
    هنا نحاكي منطق تخطيط بسيط قائم على كلمات مفتاحية لأغراض العرض التعليمي.
    """
    plan = []
    step_id = 1

    if "رحلة" in goal or "طيران" in goal:
        plan.append({
            "step_id": step_id, "description": "البحث عن رحلات طيران متاحة",
            "tool_name": "search_flights", "tool_args": {"query": goal}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "التحقق من سياسة الإلغاء لأرخص رحلة",
            "tool_name": "check_cancellation_policy", "tool_args": {}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "تجميع التوصية النهائية",
            "tool_name": "summarize_recommendation", "tool_args": {}
        })
    else:
        plan.append({
            "step_id": step_id, "description": "البحث العام عن المعلومة المطلوبة",
            "tool_name": "web_search", "tool_args": {"query": goal}
        })

    return plan


def simulated_replanning_llm(
    original_goal: str, failed_step: PlanStep, state: AgentState
) -> list[dict]:
    """
    محاكاة إعادة التخطيط: عند فشل خطوة، نولّد خطة بديلة بناءً على
    ما تم إنجازه فعلًا + سبب الفشل. محاكاة تعليمية مبسّطة.
    """
    remaining_step_id = failed_step.step_id
    return [{
        "step_id": remaining_step_id,
        "description": f"محاولة بديلة بعد فشل: {failed_step.description}",
        "tool_name": "search_flights",
        "tool_args": {"query": original_goal, "fallback_mode": True},
    }]


# =========================================================
# 4. وكيل Plan-and-Execute الكامل
# =========================================================

class PlanAndExecuteAgent:
    def __init__(self, tool_registry: ToolRegistry, max_steps: int = 10):
        self.tools = tool_registry
        self.max_steps = max_steps

    def create_initial_plan(self, goal: str) -> list[PlanStep]:
        raw_plan = simulated_planning_llm(goal, self.tools.describe_all())
        return [
            PlanStep(
                step_id=p["step_id"],
                description=p["description"],
                tool_name=p["tool_name"],
                tool_args=p["tool_args"],
            )
            for p in raw_plan
        ]

    def execute_step(self, step: PlanStep, state: AgentState) -> bool:
        """ينفذ خطوة واحدة، ويرجع True عند النجاح"""
        step.status = StepStatus.IN_PROGRESS
        tool = self.tools.get(step.tool_name)

        if tool is None:
            step.status = StepStatus.FAILED
            step.result = f"أداة غير معروفة: {step.tool_name}"
            state.log(f"❌ خطوة {step.step_id} فشلت: أداة غير موجودة")
            return False

        result = tool.execute(**step.tool_args)

        if result.startswith("ERROR"):
            step.status = StepStatus.FAILED
            step.result = result
            state.log(f"❌ خطوة {step.step_id} فشلت: {result}")
            return False

        step.status = StepStatus.COMPLETED
        step.result = result
        state.log(f"✅ خطوة {step.step_id} اكتملت: {result[:60]}")
        return True

    def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        state.plan = self.create_initial_plan(goal)
        state.log(f"🎯 الهدف: {goal}")
        state.log(f"📋 الخطة الأولية: {len(state.plan)} خطوات")

        steps_executed = 0
        step_index = 0

        while step_index < len(state.plan) and steps_executed < self.max_steps:
            current_step = state.plan[step_index]
            success = self.execute_step(current_step, state)
            steps_executed += 1

            if not success:
                if state.replanning_count >= state.max_replanning_attempts:
                    state.log("🛑 تجاوزنا الحد الأقصى لمحاولات إعادة التخطيط. "
                               "إيقاف وطلب تدخل بشري.")
                    break

                state.replanning_count += 1
                state.log(f"🔄 إعادة تخطيط (محاولة {state.replanning_count})")

                new_raw_steps = simulated_replanning_llm(goal, current_step, state)
                new_steps = [
                    PlanStep(
                        step_id=s["step_id"], description=s["description"],
                        tool_name=s["tool_name"], tool_args=s["tool_args"],
                    )
                    for s in new_raw_steps
                ]
                state.plan[step_index:step_index + 1] = new_steps
                continue

            step_index += 1

        all_completed = all(s.status == StepStatus.COMPLETED for s in state.plan)
        state.log(f"\n{'✅ المهمة اكتملت بنجاح' if all_completed else '⚠️ المهمة لم تكتمل بالكامل'}")
        return state

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
mock_tools.py — أدوات محاكاة واقعية الشكل لاختبار الوكيل
(محاكاة تعليمية لـ API حجوزات طيران حقيقي)
"""
import random


def search_flights_impl(query: str, fallback_mode: bool = False) -> str:
    if fallback_mode:
        return json.dumps({
            "flights": [{"airline": "EgyptAir", "price": 850, "stops": 1}]
        }, ensure_ascii=False)

    # محاكاة فشل عشوائي 30% من الوقت لاختبار إعادة التخطيط
    if random.random() < 0.3:
        raise ConnectionError("انتهت مهلة الاتصال بمحرك الحجوزات")

    return json.dumps({
        "flights": [
            {"airline": "Qatar Airways", "price": 720, "stops": 0},
            {"airline": "Emirates", "price": 680, "stops": 1},
        ]
    }, ensure_ascii=False)


def check_cancellation_policy_impl() -> str:
    return "سياسة الإلغاء: استرداد كامل حتى 48 ساعة قبل الرحلة."


def summarize_recommendation_impl() -> str:
    return "التوصية: رحلة Emirates بسعر 680$ مع توقف واحد، أفضل قيمة مقابل السعر."


def web_search_impl(query: str) -> str:
    return f"نتائج بحث محاكاة عن: {query}"


import json  # نقلت للأعلى منطقيًا لكن أُبقيت هنا للوضوح التعليمي فقط

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
rewoo_pattern.py
محاكاة تعليمية لنمط ReWOO: تخطيط رمزي واحد → تنفيذ أدوات → حل نهائي واحد.
يوفّر استدعاءات LLM مقارنة بـ ReAct (استدعاءان ثابتان بدل n استدعاء).
"""
import re
from dataclasses import dataclass


@dataclass
class SymbolicStep:
    variable_name: str      # مثل "#E1"
    tool_name: str
    raw_args: str            # قد تحتوي على مراجع لمتغيرات سابقة مثل "#E1"


def simulated_rewoo_planner(goal: str) -> list[SymbolicStep]:
    """استدعاء LLM واحد ينتج خطة رمزية كاملة بدون تنفيذ فعلي لأي أداة بعد"""
    return [
        SymbolicStep("#E1", "search_flights", goal),
        SymbolicStep("#E2", "check_cancellation_policy", "#E1"),
        SymbolicStep("#E3", "summarize_recommendation", "#E1, #E2"),
    ]


def resolve_and_execute(
    steps: list[SymbolicStep], tool_registry: "ToolRegistry"
) -> dict[str, str]:
    """المرحلة الثانية (Worker): تنفيذ الأدوات فعليًا، استبدال المتغيرات بالقيم الحقيقية"""
    resolved_values: dict[str, str] = {}

    for step in steps:
        resolved_args = step.raw_args
        for var_name, value in resolved_values.items():
            resolved_args = resolved_args.replace(var_name, value)

        tool = tool_registry.get(step.tool_name)
        result = tool.execute(query=resolved_args) if step.tool_name == "search_flights" \
            else tool.execute()

        resolved_values[step.variable_name] = result

    return resolved_values


def simulated_rewoo_solver(goal: str, resolved_values: dict[str, str]) -> str:
    """استدعاء LLM أخير: يأخذ كل النتائج المجمّعة وينتج إجابة نهائية متماسكة"""
    all_evidence = "\n".join(f"{k}: {v}" for k, v in resolved_values.items())
    return f"بناءً على البحث الكامل حول '{goal}':\n{all_evidence}\n\n✅ التوصية جاهزة أعلاه."


# تشغيل المثال الكامل
tool_registry = ToolRegistry()
tool_registry.register(Tool("search_flights", "بحث رحلات", search_flights_impl))
tool_registry.register(Tool("check_cancellation_policy", "سياسة الإلغاء", check_cancellation_policy_impl))
tool_registry.register(Tool("summarize_recommendation", "تلخيص", summarize_recommendation_impl))

goal = "أريد رحلة من القاهرة إلى طوكيو"
plan = simulated_rewoo_planner(goal)                  # استدعاء LLM رقم 1 فقط
evidence = resolve_and_execute(plan, tool_registry)   # بدون أي استدعاء LLM إضافي
final_answer = simulated_rewoo_solver(goal, evidence) # استدعاء LLM رقم 2 فقط

print(final_answer)
print(f"\n📊 عدد استدعاءات LLM المستخدمة: 2 (بدل {len(plan) + 1} في ReAct التقليدي)")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
reflection_agent.py
تطبيق كامل لنمط Reflexion: توليد محاولة → تقييم → نقد ذاتي → محاولة محسّنة.
"""
from dataclasses import dataclass, field


@dataclass
class Attempt:
    attempt_number: int
    output: str
    passed_evaluation: bool
    critique: str | None = None


@dataclass
class ReflexionState:
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)


def simulated_generator_llm(task: str, lessons: list[str]) -> str:
    """
    محاكاة: توليد محاولة حل للمهمة، مع الاستفادة من الدروس السابقة
    إن وُجدت (تُضاف كجزء من الـ prompt في التطبيق الحقيقي).
    """
    if not lessons:
        return "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"

    if "بطيء" in " ".join(lessons):
        return (
            "def is_prime(n):\n"
            "    if n < 2:\n        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n"
            "    return True"
        )
    return "def is_prime(n): return n > 1"


def simulated_evaluator(output: str) -> tuple[bool, str | None]:
    """
    محاكاة تقييم: تحقق هل الحل يحتوي على تحسين الجذر التربيعي
    (معيار بسيط لأغراض العرض التعليمي فقط؛ في الواقع هذا يكون
    اختبار وحدات فعلي + تحليل أداء حقيقي).
    """
    if "** 0.5" in output or "sqrt" in output:
        return True, None
    return False, (
        "الحل يفحص كل الأعداد حتى n بدلاً من التوقف عند الجذر التربيعي، "
        "مما يجعله بطيء جدًا للأعداد الكبيرة (تعقيد O(n) بدل O(√n))."
    )


class ReflexionAgent:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def run(self, task: str) -> ReflexionState:
        state = ReflexionState(task=task)

        for attempt_num in range(1, self.max_attempts + 1):
            output = simulated_generator_llm(task, state.lessons_learned)
            passed, critique = simulated_evaluator(output)

            attempt = Attempt(
                attempt_number=attempt_num,
                output=output,
                passed_evaluation=passed,
                critique=critique,
            )
            state.attempts.append(attempt)

            print(f"\n--- المحاولة {attempt_num} ---")
            print(f"الناتج:\n{output}")
            print(f"النتيجة: {'✅ نجح' if passed else '❌ يحتاج تحسين'}")

            if passed:
                print("🎉 تم الوصول لحل مُرضٍ، إيقاف الحلقة.")
                break

            if critique:
                state.lessons_learned.append(critique)
                print(f"📝 درس مُستخلص: {critique}")

        return state


# تشغيل المثال
agent = ReflexionAgent(max_attempts=3)
final_state = agent.run("اكتب دالة بايثون فعّالة للتحقق من كون العدد أوليًا")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
multi_agent_orchestrator.py
نمط Supervisor: وكيل مدير يحلل المهمة، يوزّعها على وكلاء متخصصين،
يجمع النتائج، ويقرر إن كانت هناك حاجة لجولة تنسيق إضافية.
"""
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(Enum):
    RESEARCHER = "researcher"
    FINANCIAL_ANALYST = "financial_analyst"
    WRITER = "writer"
    SUPERVISOR = "supervisor"


@dataclass
class SubTask:
    assigned_role: AgentRole
    instruction: str
    result: str | None = None


class SpecialistAgent:
    """وكيل متخصص بدور واحد ضيق ومحدد جيدًا"""

    def __init__(self, role: AgentRole, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt

    def execute(self, instruction: str, shared_context: str = "") -> str:
        """
        محاكاة تنفيذ المهمة الفرعية. في الواقع هذا استدعاء LLM حقيقي
        مع self.system_prompt كتوجيه ثابت للدور المتخصص.
        """
        if self.role == AgentRole.RESEARCHER:
            return (f"[بحث] تم جمع بيانات عن: {instruction}. "
                    f"الشركة تعمل في قطاع التكنولوجيا، إيرادات 2024: 4.2 مليار دولار.")
        elif self.role == AgentRole.FINANCIAL_ANALYST:
            return (f"[تحليل مالي] بناءً على السياق: {shared_context[:50]}... "
                    f"هامش الربح 18%، نمو سنوي 12%، أعلى من متوسط القطاع (8%).")
        elif self.role == AgentRole.WRITER:
            return (f"[تقرير نهائي] بناءً على كل التحليلات المُجمَّعة:\n"
                    f"الشركة تُظهر أداءً ماليًا قويًا يفوق متوسط القطاع، "
                    f"نوصي بمتابعة الاستثمار مع مراقبة ربع سنوية.")
        return "مهمة غير معروفة."


class SupervisorAgent:
    """الوكيل المنسّق: يحلل الهدف الكلي، يبني خطة توزيع المهام، يجمع النتائج"""

    def __init__(self, specialists: dict[AgentRole, SpecialistAgent]):
        self.specialists = specialists

    def decompose_goal(self, goal: str) -> list[SubTask]:
        """
        محاكاة: في الواقع، استدعاء LLM يحلل الهدف الكلي
        ويقرر أي الأدوار المتخصصة مطلوبة وبأي ترتيب.
        """
        return [
            SubTask(AgentRole.RESEARCHER, f"اجمع بيانات أساسية عن: {goal}"),
            SubTask(AgentRole.FINANCIAL_ANALYST, "حلل الأداء المالي بناءً على البيانات المجمّعة"),
            SubTask(AgentRole.WRITER, "اكتب تقريرًا استثماريًا نهائيًا متماسكًا"),
        ]

    def run(self, goal: str) -> str:
        subtasks = self.decompose_goal(goal)
        shared_context = ""

        print(f"🎯 الهدف الكلي: {goal}")
        print(f"📋 تم توزيع المهمة على {len(subtasks)} وكلاء متخصصين\n")

        for task in subtasks:
            specialist = self.specialists[task.assigned_role]
            print(f"▶️  تفعيل الوكيل: {task.assigned_role.value}")

            result = specialist.execute(task.instruction, shared_context)
            task.result = result
            shared_context += f"\n{result}"

            print(f"   الناتج: {result}\n")

        final_report = subtasks[-1].result
        return final_report


# تشغيل المثال الكامل
specialists = {
    AgentRole.RESEARCHER: SpecialistAgent(
        AgentRole.RESEARCHER, "أنت باحث متخصص في جمع البيانات الأولية بدقة."
    ),
    AgentRole.FINANCIAL_ANALYST: SpecialistAgent(
        AgentRole.FINANCIAL_ANALYST, "أنت محلل مالي متخصص في تقييم الأداء المالي للشركات."
    ),
    AgentRole.WRITER: SpecialistAgent(
        AgentRole.WRITER, "أنت كاتب تقارير استثمارية محترف وموجز."
    ),
}

supervisor = SupervisorAgent(specialists)
final_output = supervisor.run("شركة تقنية ناشئة في قطاع الذكاء الاصطناعي")

print("=" * 60)
print("التقرير النهائي:")
print(final_output)

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-12-frontier-trends/chapter-67.md
# ────────────────────────────────────────────────────────────────────────
"""
research_agent_system.py
نظام متكامل End-to-End: وكيل بحثي مستقل يجمع
Plan-and-Execute + أدوات + Reflexion + تقرير نهائي منظم.

جميع استدعاءات "النموذج اللغوي" والأدوات هنا محاكاة تعليمية واضحة؛
البنية والمنطق قابلان للتوصيل المباشر بأي LLM API حقيقي.
"""
from dataclasses import dataclass, field
from enum import Enum
import json


# ---------- طبقة الأدوات ----------

def search_papers_tool(topic: str) -> str:
    """محاكاة بحث في قاعدة بيانات أوراق بحثية (مثل arXiv API)"""
    fake_db = {
        "ring attention": [
            "Liu et al. 2023 - Ring Attention with Blockwise Transformers",
            "Brandon et al. 2023 - Striped Attention",
        ],
        "test-time compute": [
            "Snell et al. 2024 - Scaling LLM Test-Time Compute Optimally",
            "DeepSeek-AI 2025 - DeepSeek-R1",
        ],
    }
    results = fake_db.get(topic.lower(), ["لا توجد نتائج مباشرة لهذا الموضوع"])
    return json.dumps(results, ensure_ascii=False)


def extract_key_findings_tool(papers_json: str) -> str:
    """محاكاة استخلاص النقاط الرئيسية من قائمة أوراق بحثية"""
    papers = json.loads(papers_json)
    if not papers or papers[0] == "لا توجد نتائج مباشرة لهذا الموضوع":
        return "لا توجد نتائج كافية لاستخلاص نتائج رئيسية."
    return f"تم استخلاص {len(papers)} نتيجة رئيسية من الأوراق المُسترجعة بنجاح."


def draft_summary_tool(findings: str, topic: str, include_citations: bool = False) -> str:
    """محاكاة صياغة ملخص نهائي، مع خيار تضمين الاستشهادات أو عدمه"""
    base = f"ملخص حول '{topic}': {findings}"
    if include_citations:
        base += " [تتضمن النسخة هذه استشهادات مرجعية كاملة لكل نتيجة.]"
    return base


# ---------- طبقة التقييم الذاتي (Reflexion) ----------

@dataclass
class QualityCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def evaluate_summary_quality(summary: str) -> QualityCheckResult:
    """
    محاكاة تقييم جودة: نتحقق من معيارين بسيطين
    (في الواقع، هذا قد يكون استدعاء LLM-as-judge كما في فصل التقييم).
    """
    issues = []
    if "استشهادات" not in summary:
        issues.append("الملخص لا يتضمن استشهادات مرجعية، وهو مطلوب لملخص أكاديمي.")
    if len(summary) < 30:
        issues.append("الملخص قصير جدًا وغير كافٍ.")

    return QualityCheckResult(passed=len(issues) == 0, issues=issues)


# ---------- الوكيل المتكامل ----------

class AutonomousResearchAgent:
    def __init__(self, max_reflection_attempts: int = 2):
        self.max_reflection_attempts = max_reflection_attempts
        self.execution_trace: list[str] = []

    def _log(self, message: str):
        self.execution_trace.append(message)
        print(message)

    def run(self, topic: str) -> str:
        self._log(f"🎯 بدء مهمة البحث حول: '{topic}'")

        # ----- المرحلة 1: التخطيط (مُبسَّط لثلاث خطوات ثابتة لهذا المثال) -----
        self._log("📋 الخطة: (1) بحث الأوراق (2) استخلاص النتائج (3) صياغة ملخص")

        # ----- المرحلة 2: التنفيذ -----
        papers_json = search_papers_tool(topic)
        self._log(f"🔍 نتائج البحث: {papers_json}")

        findings = extract_key_findings_tool(papers_json)
        self._log(f"🧩 النتائج المُستخلصة: {findings}")

        include_citations = False
        attempt = 0
        final_summary = ""

        # ----- المرحلة 3: التنفيذ + المراجعة الذاتية (Reflexion Loop) -----
        while attempt <= self.max_reflection_attempts:
            summary = draft_summary_tool(findings, topic, include_citations=include_citations)
            self._log(f"\n--- محاولة صياغة رقم {attempt + 1} ---")
            self._log(f"📝 المسودة: {summary}")

            quality = evaluate_summary_quality(summary)

            if quality.passed:
                self._log("✅ اجتاز الملخص فحص الجودة.")
                final_summary = summary
                break

            self._log(f"⚠️ مشاكل اكتُشفت: {quality.issues}")

            # تطبيق "الدرس المستفاد" على المحاولة القادمة
            if any("استشهادات" in issue for issue in quality.issues):
                include_citations = True
                self._log("📝 درس مُستخلص: سيُضاف دعم استشهادات في المحاولة القادمة.")

            attempt += 1
            final_summary = summary  # الاحتفاظ بآخر مسودة حتى لو لم تجتز الفحص بالكامل

        if attempt > self.max_reflection_attempts:
            self._log("🛑 تجاوزنا الحد الأقصى لمحاولات التحسين الذاتي. "
                       "تسليم أفضل نسخة متاحة مع تنبيه بوجود قيود.")

        self._log(f"\n🎉 التسليم النهائي:\n{final_summary}")
        return final_summary


# تشغيل النظام الكامل
agent = AutonomousResearchAgent(max_reflection_attempts=2)
result = agent.run("test-time compute")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 67: الفصل 67: الوكلاء المستقلون المتقدمون (Agentic AI)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_agentic() -> None:
    states = ['plan', 'retrieve', 'verify', 'answer']
    allowed_transitions = {('plan', 'retrieve'), ('retrieve', 'verify'), ('verify', 'answer')}
    current, steps = ('plan', [])
    for proposed in ['retrieve', 'verify', 'answer']:
        if (current, proposed) not in allowed_transitions:
            raise RuntimeError('invalid state transition')
        steps.append(proposed)
        current = proposed
        if len(steps) >= 4:
            break
    print('state sequence:', ['plan', *steps])
    print('bounded steps:', len(steps), '| completed:', current == 'answer')

if __name__ == "__main__":
    print("Chapter 67 lab: Execute a bounded agent state machine")
    run_agentic()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
agent_core.py
بنية أساسية كاملة لوكيل يدعم نمطي ReAct وPlan-and-Execute،
مع واجهة أدوات (Tools) قابلة للتوسعة، وقابلة للتشغيل الفعلي.

ملاحظة: دالة call_llm هنا محاكاة توضيحية صريحة تحاكي سلوك نموذج لغوي
عبر قواعد منطقية مبسّطة؛ في الإنتاج الحقيقي تُستبدل باستدعاء
فعلي لـ API (OpenAI، Anthropic، Gemini) مع function calling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import json
import re


# =========================================================
# 1. تعريف الأدوات (Tools)
# =========================================================

@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., str]

    def execute(self, **kwargs) -> str:
        try:
            return self.function(**kwargs)
        except Exception as e:
            return f"ERROR: فشل تنفيذ الأداة '{self.name}': {str(e)}"


class ToolRegistry:
    """سجل مركزي للأدوات المتاحة للوكيل"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe_all(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )


# =========================================================
# 2. حالة الوكيل والذاكرة العاملة
# =========================================================

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: str
    tool_args: dict[str, Any]
    status: StepStatus = StepStatus.PENDING
    result: str | None = None


@dataclass
class AgentState:
    """الذاكرة العاملة (scratchpad): كل ما يحتاجه الوكيل لتذكّر تقدّمه"""
    goal: str
    plan: list[PlanStep] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    max_replanning_attempts: int = 3
    replanning_count: int = 0

    def log(self, message: str) -> None:
        self.execution_log.append(message)

    def get_completed_results_summary(self) -> str:
        completed = [s for s in self.plan if s.status == StepStatus.COMPLETED]
        return "\n".join(f"[{s.step_id}] {s.description} → {s.result}" for s in completed)


# =========================================================
# 3. محاكاة استدعاء النموذج اللغوي (Simulated LLM)
# =========================================================

def simulated_planning_llm(goal: str, available_tools_description: str) -> list[dict]:
    """
    محاكاة تعليمية: في الواقع، هذا الاستدعاء يُرسل prompt حقيقي للنموذج
    يطلب منه إرجاع خطة بصيغة JSON منظمة (structured output / function calling).
    هنا نحاكي منطق تخطيط بسيط قائم على كلمات مفتاحية لأغراض العرض التعليمي.
    """
    plan = []
    step_id = 1

    if "رحلة" in goal or "طيران" in goal:
        plan.append({
            "step_id": step_id, "description": "البحث عن رحلات طيران متاحة",
            "tool_name": "search_flights", "tool_args": {"query": goal}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "التحقق من سياسة الإلغاء لأرخص رحلة",
            "tool_name": "check_cancellation_policy", "tool_args": {}
        })
        step_id += 1
        plan.append({
            "step_id": step_id, "description": "تجميع التوصية النهائية",
            "tool_name": "summarize_recommendation", "tool_args": {}
        })
    else:
        plan.append({
            "step_id": step_id, "description": "البحث العام عن المعلومة المطلوبة",
            "tool_name": "web_search", "tool_args": {"query": goal}
        })

    return plan


def simulated_replanning_llm(
    original_goal: str, failed_step: PlanStep, state: AgentState
) -> list[dict]:
    """
    محاكاة إعادة التخطيط: عند فشل خطوة، نولّد خطة بديلة بناءً على
    ما تم إنجازه فعلًا + سبب الفشل. محاكاة تعليمية مبسّطة.
    """
    remaining_step_id = failed_step.step_id
    return [{
        "step_id": remaining_step_id,
        "description": f"محاولة بديلة بعد فشل: {failed_step.description}",
        "tool_name": "search_flights",
        "tool_args": {"query": original_goal, "fallback_mode": True},
    }]


# =========================================================
# 4. وكيل Plan-and-Execute الكامل
# =========================================================

class PlanAndExecuteAgent:
    def __init__(self, tool_registry: ToolRegistry, max_steps: int = 10):
        self.tools = tool_registry
        self.max_steps = max_steps

    def create_initial_plan(self, goal: str) -> list[PlanStep]:
        raw_plan = simulated_planning_llm(goal, self.tools.describe_all())
        return [
            PlanStep(
                step_id=p["step_id"],
                description=p["description"],
                tool_name=p["tool_name"],
                tool_args=p["tool_args"],
            )
            for p in raw_plan
        ]

    def execute_step(self, step: PlanStep, state: AgentState) -> bool:
        """ينفذ خطوة واحدة، ويرجع True عند النجاح"""
        step.status = StepStatus.IN_PROGRESS
        tool = self.tools.get(step.tool_name)

        if tool is None:
            step.status = StepStatus.FAILED
            step.result = f"أداة غير معروفة: {step.tool_name}"
            state.log(f"❌ خطوة {step.step_id} فشلت: أداة غير موجودة")
            return False

        result = tool.execute(**step.tool_args)

        if result.startswith("ERROR"):
            step.status = StepStatus.FAILED
            step.result = result
            state.log(f"❌ خطوة {step.step_id} فشلت: {result}")
            return False

        step.status = StepStatus.COMPLETED
        step.result = result
        state.log(f"✅ خطوة {step.step_id} اكتملت: {result[:60]}")
        return True

    def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        state.plan = self.create_initial_plan(goal)
        state.log(f"🎯 الهدف: {goal}")
        state.log(f"📋 الخطة الأولية: {len(state.plan)} خطوات")

        steps_executed = 0
        step_index = 0

        while step_index < len(state.plan) and steps_executed < self.max_steps:
            current_step = state.plan[step_index]
            success = self.execute_step(current_step, state)
            steps_executed += 1

            if not success:
                if state.replanning_count >= state.max_replanning_attempts:
                    state.log("🛑 تجاوزنا الحد الأقصى لمحاولات إعادة التخطيط. "
                               "إيقاف وطلب تدخل بشري.")
                    break

                state.replanning_count += 1
                state.log(f"🔄 إعادة تخطيط (محاولة {state.replanning_count})")

                new_raw_steps = simulated_replanning_llm(goal, current_step, state)
                new_steps = [
                    PlanStep(
                        step_id=s["step_id"], description=s["description"],
                        tool_name=s["tool_name"], tool_args=s["tool_args"],
                    )
                    for s in new_raw_steps
                ]
                state.plan[step_index:step_index + 1] = new_steps
                continue

            step_index += 1

        all_completed = all(s.status == StepStatus.COMPLETED for s in state.plan)
        state.log(f"\n{'✅ المهمة اكتملت بنجاح' if all_completed else '⚠️ المهمة لم تكتمل بالكامل'}")
        return state

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
mock_tools.py — أدوات محاكاة واقعية الشكل لاختبار الوكيل
(محاكاة تعليمية لـ API حجوزات طيران حقيقي)
"""
import random


def search_flights_impl(query: str, fallback_mode: bool = False) -> str:
    if fallback_mode:
        return json.dumps({
            "flights": [{"airline": "EgyptAir", "price": 850, "stops": 1}]
        }, ensure_ascii=False)

    # محاكاة فشل عشوائي 30% من الوقت لاختبار إعادة التخطيط
    if random.random() < 0.3:
        raise ConnectionError("انتهت مهلة الاتصال بمحرك الحجوزات")

    return json.dumps({
        "flights": [
            {"airline": "Qatar Airways", "price": 720, "stops": 0},
            {"airline": "Emirates", "price": 680, "stops": 1},
        ]
    }, ensure_ascii=False)


def check_cancellation_policy_impl() -> str:
    return "سياسة الإلغاء: استرداد كامل حتى 48 ساعة قبل الرحلة."


def summarize_recommendation_impl() -> str:
    return "التوصية: رحلة Emirates بسعر 680$ مع توقف واحد، أفضل قيمة مقابل السعر."


def web_search_impl(query: str) -> str:
    return f"نتائج بحث محاكاة عن: {query}"


import json  # نقلت للأعلى منطقيًا لكن أُبقيت هنا للوضوح التعليمي فقط

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
rewoo_pattern.py
محاكاة تعليمية لنمط ReWOO: تخطيط رمزي واحد → تنفيذ أدوات → حل نهائي واحد.
يوفّر استدعاءات LLM مقارنة بـ ReAct (استدعاءان ثابتان بدل n استدعاء).
"""
import re
from dataclasses import dataclass


@dataclass
class SymbolicStep:
    variable_name: str      # مثل "#E1"
    tool_name: str
    raw_args: str            # قد تحتوي على مراجع لمتغيرات سابقة مثل "#E1"


def simulated_rewoo_planner(goal: str) -> list[SymbolicStep]:
    """استدعاء LLM واحد ينتج خطة رمزية كاملة بدون تنفيذ فعلي لأي أداة بعد"""
    return [
        SymbolicStep("#E1", "search_flights", goal),
        SymbolicStep("#E2", "check_cancellation_policy", "#E1"),
        SymbolicStep("#E3", "summarize_recommendation", "#E1, #E2"),
    ]


def resolve_and_execute(
    steps: list[SymbolicStep], tool_registry: "ToolRegistry"
) -> dict[str, str]:
    """المرحلة الثانية (Worker): تنفيذ الأدوات فعليًا، استبدال المتغيرات بالقيم الحقيقية"""
    resolved_values: dict[str, str] = {}

    for step in steps:
        resolved_args = step.raw_args
        for var_name, value in resolved_values.items():
            resolved_args = resolved_args.replace(var_name, value)

        tool = tool_registry.get(step.tool_name)
        result = tool.execute(query=resolved_args) if step.tool_name == "search_flights" \
            else tool.execute()

        resolved_values[step.variable_name] = result

    return resolved_values


def simulated_rewoo_solver(goal: str, resolved_values: dict[str, str]) -> str:
    """استدعاء LLM أخير: يأخذ كل النتائج المجمّعة وينتج إجابة نهائية متماسكة"""
    all_evidence = "\n".join(f"{k}: {v}" for k, v in resolved_values.items())
    return f"بناءً على البحث الكامل حول '{goal}':\n{all_evidence}\n\n✅ التوصية جاهزة أعلاه."


# تشغيل المثال الكامل
tool_registry = ToolRegistry()
tool_registry.register(Tool("search_flights", "بحث رحلات", search_flights_impl))
tool_registry.register(Tool("check_cancellation_policy", "سياسة الإلغاء", check_cancellation_policy_impl))
tool_registry.register(Tool("summarize_recommendation", "تلخيص", summarize_recommendation_impl))

goal = "أريد رحلة من القاهرة إلى طوكيو"
plan = simulated_rewoo_planner(goal)                  # استدعاء LLM رقم 1 فقط
evidence = resolve_and_execute(plan, tool_registry)   # بدون أي استدعاء LLM إضافي
final_answer = simulated_rewoo_solver(goal, evidence) # استدعاء LLM رقم 2 فقط

print(final_answer)
print(f"\n📊 عدد استدعاءات LLM المستخدمة: 2 (بدل {len(plan) + 1} في ReAct التقليدي)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
reflection_agent.py
تطبيق كامل لنمط Reflexion: توليد محاولة → تقييم → نقد ذاتي → محاولة محسّنة.
"""
from dataclasses import dataclass, field


@dataclass
class Attempt:
    attempt_number: int
    output: str
    passed_evaluation: bool
    critique: str | None = None


@dataclass
class ReflexionState:
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)


def simulated_generator_llm(task: str, lessons: list[str]) -> str:
    """
    محاكاة: توليد محاولة حل للمهمة، مع الاستفادة من الدروس السابقة
    إن وُجدت (تُضاف كجزء من الـ prompt في التطبيق الحقيقي).
    """
    if not lessons:
        return "def is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"

    if "بطيء" in " ".join(lessons):
        return (
            "def is_prime(n):\n"
            "    if n < 2:\n        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n            return False\n"
            "    return True"
        )
    return "def is_prime(n): return n > 1"


def simulated_evaluator(output: str) -> tuple[bool, str | None]:
    """
    محاكاة تقييم: تحقق هل الحل يحتوي على تحسين الجذر التربيعي
    (معيار بسيط لأغراض العرض التعليمي فقط؛ في الواقع هذا يكون
    اختبار وحدات فعلي + تحليل أداء حقيقي).
    """
    if "** 0.5" in output or "sqrt" in output:
        return True, None
    return False, (
        "الحل يفحص كل الأعداد حتى n بدلاً من التوقف عند الجذر التربيعي، "
        "مما يجعله بطيء جدًا للأعداد الكبيرة (تعقيد O(n) بدل O(√n))."
    )


class ReflexionAgent:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def run(self, task: str) -> ReflexionState:
        state = ReflexionState(task=task)

        for attempt_num in range(1, self.max_attempts + 1):
            output = simulated_generator_llm(task, state.lessons_learned)
            passed, critique = simulated_evaluator(output)

            attempt = Attempt(
                attempt_number=attempt_num,
                output=output,
                passed_evaluation=passed,
                critique=critique,
            )
            state.attempts.append(attempt)

            print(f"\n--- المحاولة {attempt_num} ---")
            print(f"الناتج:\n{output}")
            print(f"النتيجة: {'✅ نجح' if passed else '❌ يحتاج تحسين'}")

            if passed:
                print("🎉 تم الوصول لحل مُرضٍ، إيقاف الحلقة.")
                break

            if critique:
                state.lessons_learned.append(critique)
                print(f"📝 درس مُستخلص: {critique}")

        return state


# تشغيل المثال
agent = ReflexionAgent(max_attempts=3)
final_state = agent.run("اكتب دالة بايثون فعّالة للتحقق من كون العدد أوليًا")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
multi_agent_orchestrator.py
نمط Supervisor: وكيل مدير يحلل المهمة، يوزّعها على وكلاء متخصصين،
يجمع النتائج، ويقرر إن كانت هناك حاجة لجولة تنسيق إضافية.
"""
from dataclasses import dataclass, field
from enum import Enum


class AgentRole(Enum):
    RESEARCHER = "researcher"
    FINANCIAL_ANALYST = "financial_analyst"
    WRITER = "writer"
    SUPERVISOR = "supervisor"


@dataclass
class SubTask:
    assigned_role: AgentRole
    instruction: str
    result: str | None = None


class SpecialistAgent:
    """وكيل متخصص بدور واحد ضيق ومحدد جيدًا"""

    def __init__(self, role: AgentRole, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt

    def execute(self, instruction: str, shared_context: str = "") -> str:
        """
        محاكاة تنفيذ المهمة الفرعية. في الواقع هذا استدعاء LLM حقيقي
        مع self.system_prompt كتوجيه ثابت للدور المتخصص.
        """
        if self.role == AgentRole.RESEARCHER:
            return (f"[بحث] تم جمع بيانات عن: {instruction}. "
                    f"الشركة تعمل في قطاع التكنولوجيا، إيرادات 2024: 4.2 مليار دولار.")
        elif self.role == AgentRole.FINANCIAL_ANALYST:
            return (f"[تحليل مالي] بناءً على السياق: {shared_context[:50]}... "
                    f"هامش الربح 18%، نمو سنوي 12%، أعلى من متوسط القطاع (8%).")
        elif self.role == AgentRole.WRITER:
            return (f"[تقرير نهائي] بناءً على كل التحليلات المُجمَّعة:\n"
                    f"الشركة تُظهر أداءً ماليًا قويًا يفوق متوسط القطاع، "
                    f"نوصي بمتابعة الاستثمار مع مراقبة ربع سنوية.")
        return "مهمة غير معروفة."


class SupervisorAgent:
    """الوكيل المنسّق: يحلل الهدف الكلي، يبني خطة توزيع المهام، يجمع النتائج"""

    def __init__(self, specialists: dict[AgentRole, SpecialistAgent]):
        self.specialists = specialists

    def decompose_goal(self, goal: str) -> list[SubTask]:
        """
        محاكاة: في الواقع، استدعاء LLM يحلل الهدف الكلي
        ويقرر أي الأدوار المتخصصة مطلوبة وبأي ترتيب.
        """
        return [
            SubTask(AgentRole.RESEARCHER, f"اجمع بيانات أساسية عن: {goal}"),
            SubTask(AgentRole.FINANCIAL_ANALYST, "حلل الأداء المالي بناءً على البيانات المجمّعة"),
            SubTask(AgentRole.WRITER, "اكتب تقريرًا استثماريًا نهائيًا متماسكًا"),
        ]

    def run(self, goal: str) -> str:
        subtasks = self.decompose_goal(goal)
        shared_context = ""

        print(f"🎯 الهدف الكلي: {goal}")
        print(f"📋 تم توزيع المهمة على {len(subtasks)} وكلاء متخصصين\n")

        for task in subtasks:
            specialist = self.specialists[task.assigned_role]
            print(f"▶️  تفعيل الوكيل: {task.assigned_role.value}")

            result = specialist.execute(task.instruction, shared_context)
            task.result = result
            shared_context += f"\n{result}"

            print(f"   الناتج: {result}\n")

        final_report = subtasks[-1].result
        return final_report


# تشغيل المثال الكامل
specialists = {
    AgentRole.RESEARCHER: SpecialistAgent(
        AgentRole.RESEARCHER, "أنت باحث متخصص في جمع البيانات الأولية بدقة."
    ),
    AgentRole.FINANCIAL_ANALYST: SpecialistAgent(
        AgentRole.FINANCIAL_ANALYST, "أنت محلل مالي متخصص في تقييم الأداء المالي للشركات."
    ),
    AgentRole.WRITER: SpecialistAgent(
        AgentRole.WRITER, "أنت كاتب تقارير استثمارية محترف وموجز."
    ),
}

supervisor = SupervisorAgent(specialists)
final_output = supervisor.run("شركة تقنية ناشئة في قطاع الذكاء الاصطناعي")

print("=" * 60)
print("التقرير النهائي:")
print(final_output)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-67-chapter-67.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
research_agent_system.py
نظام متكامل End-to-End: وكيل بحثي مستقل يجمع
Plan-and-Execute + أدوات + Reflexion + تقرير نهائي منظم.

جميع استدعاءات "النموذج اللغوي" والأدوات هنا محاكاة تعليمية واضحة؛
البنية والمنطق قابلان للتوصيل المباشر بأي LLM API حقيقي.
"""
from dataclasses import dataclass, field
from enum import Enum
import json


# ---------- طبقة الأدوات ----------

def search_papers_tool(topic: str) -> str:
    """محاكاة بحث في قاعدة بيانات أوراق بحثية (مثل arXiv API)"""
    fake_db = {
        "ring attention": [
            "Liu et al. 2023 - Ring Attention with Blockwise Transformers",
            "Brandon et al. 2023 - Striped Attention",
        ],
        "test-time compute": [
            "Snell et al. 2024 - Scaling LLM Test-Time Compute Optimally",
            "DeepSeek-AI 2025 - DeepSeek-R1",
        ],
    }
    results = fake_db.get(topic.lower(), ["لا توجد نتائج مباشرة لهذا الموضوع"])
    return json.dumps(results, ensure_ascii=False)


def extract_key_findings_tool(papers_json: str) -> str:
    """محاكاة استخلاص النقاط الرئيسية من قائمة أوراق بحثية"""
    papers = json.loads(papers_json)
    if not papers or papers[0] == "لا توجد نتائج مباشرة لهذا الموضوع":
        return "لا توجد نتائج كافية لاستخلاص نتائج رئيسية."
    return f"تم استخلاص {len(papers)} نتيجة رئيسية من الأوراق المُسترجعة بنجاح."


def draft_summary_tool(findings: str, topic: str, include_citations: bool = False) -> str:
    """محاكاة صياغة ملخص نهائي، مع خيار تضمين الاستشهادات أو عدمه"""
    base = f"ملخص حول '{topic}': {findings}"
    if include_citations:
        base += " [تتضمن النسخة هذه استشهادات مرجعية كاملة لكل نتيجة.]"
    return base


# ---------- طبقة التقييم الذاتي (Reflexion) ----------

@dataclass
class QualityCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def evaluate_summary_quality(summary: str) -> QualityCheckResult:
    """
    محاكاة تقييم جودة: نتحقق من معيارين بسيطين
    (في الواقع، هذا قد يكون استدعاء LLM-as-judge كما في فصل التقييم).
    """
    issues = []
    if "استشهادات" not in summary:
        issues.append("الملخص لا يتضمن استشهادات مرجعية، وهو مطلوب لملخص أكاديمي.")
    if len(summary) < 30:
        issues.append("الملخص قصير جدًا وغير كافٍ.")

    return QualityCheckResult(passed=len(issues) == 0, issues=issues)


# ---------- الوكيل المتكامل ----------

class AutonomousResearchAgent:
    def __init__(self, max_reflection_attempts: int = 2):
        self.max_reflection_attempts = max_reflection_attempts
        self.execution_trace: list[str] = []

    def _log(self, message: str):
        self.execution_trace.append(message)
        print(message)

    def run(self, topic: str) -> str:
        self._log(f"🎯 بدء مهمة البحث حول: '{topic}'")

        # ----- المرحلة 1: التخطيط (مُبسَّط لثلاث خطوات ثابتة لهذا المثال) -----
        self._log("📋 الخطة: (1) بحث الأوراق (2) استخلاص النتائج (3) صياغة ملخص")

        # ----- المرحلة 2: التنفيذ -----
        papers_json = search_papers_tool(topic)
        self._log(f"🔍 نتائج البحث: {papers_json}")

        findings = extract_key_findings_tool(papers_json)
        self._log(f"🧩 النتائج المُستخلصة: {findings}")

        include_citations = False
        attempt = 0
        final_summary = ""

        # ----- المرحلة 3: التنفيذ + المراجعة الذاتية (Reflexion Loop) -----
        while attempt <= self.max_reflection_attempts:
            summary = draft_summary_tool(findings, topic, include_citations=include_citations)
            self._log(f"\n--- محاولة صياغة رقم {attempt + 1} ---")
            self._log(f"📝 المسودة: {summary}")

            quality = evaluate_summary_quality(summary)

            if quality.passed:
                self._log("✅ اجتاز الملخص فحص الجودة.")
                final_summary = summary
                break

            self._log(f"⚠️ مشاكل اكتُشفت: {quality.issues}")

            # تطبيق "الدرس المستفاد" على المحاولة القادمة
            if any("استشهادات" in issue for issue in quality.issues):
                include_citations = True
                self._log("📝 درس مُستخلص: سيُضاف دعم استشهادات في المحاولة القادمة.")

            attempt += 1
            final_summary = summary  # الاحتفاظ بآخر مسودة حتى لو لم تجتز الفحص بالكامل

        if attempt > self.max_reflection_attempts:
            self._log("🛑 تجاوزنا الحد الأقصى لمحاولات التحسين الذاتي. "
                       "تسليم أفضل نسخة متاحة مع تنبيه بوجود قيود.")

        self._log(f"\n🎉 التسليم النهائي:\n{final_summary}")
        return final_summary


# تشغيل النظام الكامل
agent = AutonomousResearchAgent(max_reflection_attempts=2)
result = agent.run("test-time compute")
