#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أنظمة Multi-Agent المتعاونة
Large Language Model Engineering — Multi-Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-56-chapter-56 (lesson 56)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-56-chapter-56.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أنظمة Multi-Agent المتعاونة
Large Language Model Engineering — Multi-Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-56-chapter-56 (lesson 56)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-56-chapter-56.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أنظمة Multi-Agent المتعاونة
Large Language Model Engineering — Multi-Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-56-chapter-56 (lesson 56)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_56.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 56: الفصل 56: أنظمة Multi-Agent المتعاونة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multi_agent() -> None:
    recommendations = {'planner': 'retrieve', 'critic': 'retrieve', 'verifier': 'abstain'}
    counts = Counter(recommendations.values())
    winner, votes = counts.most_common(1)[0]
    print('agent votes:', recommendations)
    print('consensus:', winner, f'({votes}/{len(recommendations)})')
    print('escalate when disagreement exceeds the configured threshold.')

if __name__ == "__main__":
    print("Chapter 56 lab: Combine independent multi-agent recommendations")
    run_multi_agent()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypedDict, Annotated
import operator

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(system_prompt: str, user_prompt: str,
             model: str = "gpt-4o-mini", temperature: float = 0.7) -> Dict:
    """نداء موحّد لأي نموذج لغوي، يُرجع النص + عدد التوكنز لتتبّع التكلفة بدقة"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return {
        "text": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
    }


@dataclass
class CostTracker:
    """يتتبع تكلفة كل وكيل على حدة — ضروري لتشخيص انفجار التكلفة لاحقًا"""
    total_tokens: int = 0
    total_calls: int = 0
    history: List[Dict] = field(default_factory=list)

    def log(self, agent_name: str, tokens: int):
        self.total_tokens += tokens
        self.total_calls += 1
        self.history.append({"agent": agent_name, "tokens": tokens})

    def report(self):
        print(f"\n📊 إجمالي النداءات: {self.total_calls} | إجمالي التوكنز: {self.total_tokens}")
        by_agent: Dict[str, int] = {}
        for h in self.history:
            by_agent[h["agent"]] = by_agent.get(h["agent"], 0) + h["tokens"]
        for agent, tok in sorted(by_agent.items(), key=lambda x: -x[1]):
            pct = tok / self.total_tokens * 100 if self.total_tokens else 0
            print(f"  - {agent:<15}: {tok:>6} توكن ({pct:5.1f}%)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
class DebateAgent:
    """وكيل يشارك في نقاش متعدد الجولات، يحتفظ بذاكرة مواقفه السابقة"""

    def __init__(self, name: str, persona: str, model: str, cost_tracker: CostTracker):
        self.name = name
        self.persona = persona
        self.model = model
        self.cost_tracker = cost_tracker
        self.memory: List[str] = []

    def respond(self, question: str, other_responses: List[str], round_num: int) -> str:
        system_prompt = f"""أنت {self.name}. شخصيتك ومنهجك: {self.persona}
أنت تشارك في نقاش علمي لحل مسألة بأعلى دقة ممكنة.

قواعد صارمة:
1. اقرأ إجابات الوكلاء الآخرين بعناية وابحث عن أخطاء منطقية أو حسابية فعلية.
2. لا تغيّر رأيك إلا إذا رأيت دليلًا أو منطقًا مقنعًا — لا تتقارب لمجرد المجاملة.
3. إن وجدت خطأً، اذكره صراحة بالأرقام أو الخطوات.
4. اختصر: لا تتجاوز 150 كلمة."""

        if round_num == 0:
            user_prompt = f"السؤال: {question}\n\nقدّم إجابتك الأولية مع التفكير خطوة بخطوة، واذكر درجة ثقتك (0-1)."
        else:
            others_text = "\n\n".join(f"[رأي آخر {i+1}]: {r}" for i, r in enumerate(other_responses))
            user_prompt = f"""السؤال: {question}

موقفك في الجولة السابقة: {self.memory[-1] if self.memory else 'لا يوجد'}

آراء الوكلاء الآخرين في الجولة السابقة:
{others_text}

راجع موقفك بدقة: هل تتمسك بإجابتك أم تعدّلها؟ قدّم إجابتك النهائية المحدّثة مع درجة ثقتك الجديدة."""

        result = call_llm(system_prompt, user_prompt, model=self.model)
        self.cost_tracker.log(self.name, result["tokens"])
        self.memory.append(result["text"])
        return result["text"]


class JudgeAgent:
    """حَكَم محايد يستخرج الإجابة الأصح بناءً على قوة الحجة لا على الأغلبية فقط"""

    def __init__(self, model: str, cost_tracker: CostTracker):
        self.model = model
        self.cost_tracker = cost_tracker

    def adjudicate(self, question: str, final_responses: Dict[str, str]) -> Dict:
        system_prompt = """أنت حَكَم خبير محايد تمامًا. اقرأ آراء عدة خبراء متنافسين
واستخرج الإجابة الأصح بناءً على قوة الأدلة والمنطق الرياضي/الحقائقي، وليس الأغلبية العددية.
أعد الناتج بصيغة JSON فقط دون أي نص إضافي."""

        responses_text = "\n\n".join(f"رأي {name}:\n{resp}" for name, resp in final_responses.items())
        user_prompt = f"""السؤال الأصلي: {question}

{responses_text}

أعد JSON بالشكل التالي حصرًا:
{{"final_answer": "...", "confidence": 0.0, "reasoning": "...", "agreement_level": "high/medium/low"}}"""

        result = call_llm(system_prompt, user_prompt, model=self.model, temperature=0.0)
        self.cost_tracker.log("Judge", result["tokens"])
        try:
            return json.loads(result["text"])
        except json.JSONDecodeError:
            return {"final_answer": result["text"], "confidence": 0.5,
                     "reasoning": "تعذر تحليل JSON", "agreement_level": "unknown"}


class DebateOrchestrator:
    """المنسّق: يدير الجولات، يفحص التقارب، ويستدعي الحَكَم عند الحاجة"""

    def __init__(self, agents: List[DebateAgent], judge: JudgeAgent,
                 max_rounds: int = 3, convergence_threshold: float = 0.90):
        self.agents = agents
        self.judge = judge
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self._embedder = None  # تحميل كسول (lazy loading)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embedder

    def _has_converged(self, responses: List[str]) -> bool:
        """يقيس التشابه الدلالي بين إجابات الوكلاء — تقارب حقيقي وليس تطابق نصي"""
        if len(responses) < 2:
            return False
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = self.embedder.encode(responses)
        sim_matrix = cosine_similarity(embeddings)
        n = len(responses)
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # استبعاد القطر (تشابه الوكيل مع نفسه = 1)
        return avg_sim >= self.convergence_threshold

    def run(self, question: str) -> Dict:
        previous_round: List[str] = []
        round_num = 0

        for round_num in range(self.max_rounds):
            current_round: List[str] = []
            for i, agent in enumerate(self.agents):
                others = [r for j, r in enumerate(previous_round) if j != i] if previous_round else []
                response = agent.respond(question, other_responses=others, round_num=round_num)
                current_round.append(response)

            previous_round = current_round
            print(f"--- الجولة {round_num + 1}/{self.max_rounds} اكتملت ---")

            if self._has_converged(current_round):
                print(f"✅ تقارب الوكلاء دلاليًا عند الجولة {round_num + 1}")
                break

        final_map = {agent.name: resp for agent, resp in zip(self.agents, previous_round)}
        verdict = self.judge.adjudicate(question, final_map)

        return {"rounds_used": round_num + 1, "final_responses": final_map, "verdict": verdict}


# ───────────── تشغيل تجريبي: مسألة استدلال منطقي مفخّخة ─────────────
if __name__ == "__main__":
    cost_tracker = CostTracker()

    # ملاحظة مهمة: نستخدم نماذج/شخصيات مختلفة لتفادي Echo Chamber
    agents = [
        DebateAgent("المحلل_الرياضي", "تحلل المسائل خطوة بخطوة بدقة حسابية صارمة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المدقق_المنطقي", "تبحث عن مغالطات منطقية وافتراضات خاطئة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المحامي_الناقد", "تتحدى كل إجابة وتطلب دليلًا دامغًا قبل القبول", "gpt-4o-mini", cost_tracker),
    ]
    judge = JudgeAgent("gpt-4o", cost_tracker)
    orchestrator = DebateOrchestrator(agents, judge, max_rounds=3, convergence_threshold=0.90)

    question = """متجر يبيع قميصًا بسعر 100 ريال. في عرض خاص، يُخفّض السعر 20% ثم يُرفع
السعر الناتج 20% مرة أخرى. هل السعر النهائي يساوي السعر الأصلي (100 ريال)؟ برّر إجابتك رياضيًا."""

    result = orchestrator.run(question)
    print("\n=== الحكم النهائي ===")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    cost_tracker.report()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END


class SupervisorState(TypedDict):
    task: str
    subtasks: List[Dict]
    results: Annotated[List[Dict], operator.add]
    final_report: str


SPECIALIST_PERSONAS = {
    "researcher": "خبير أبحاث يجمع الحقائق والبيانات بدقة ويذكر الافتراضات بوضوح.",
    "coder": "مهندس برمجيات يكتب كودًا نظيفًا وقابلًا للتشغيل فعليًا مع تعليقات.",
    "critic": "مراجع صارم يبحث عن الثغرات المنطقية والأخطاء التقنية قبل النشر.",
}


def make_supervisor_node(cost_tracker: CostTracker):
    def supervisor_node(state: SupervisorState) -> Dict:
        system_prompt = """أنت مدير مشروع تدير فريقًا من 3 متخصصين:
- researcher: يجمع المعلومات والحقائق
- coder: يكتب كودًا وينفّذ حسابات
- critic: يراجع وينتقد العمل المنجز

قسّم المهمة إلى مهام فرعية واضحة ومحددة (2-4 مهام)، وحدد المتخصص المناسب لكل منها.
أعد الناتج بصيغة JSON فقط: {"subtasks": [{"agent": "researcher|coder|critic", "instruction": "..."}]}"""

        result = call_llm(system_prompt, f"المهمة: {state['task']}", model="gpt-4o", temperature=0.3)
        cost_tracker.log("Supervisor", result["tokens"])

        try:
            subtasks = json.loads(result["text"])["subtasks"]
        except (json.JSONDecodeError, KeyError):
            subtasks = [{"agent": "researcher", "instruction": state["task"]}]

        return {"subtasks": subtasks}
    return supervisor_node


def make_specialist_node(cost_tracker: CostTracker):
    def specialist_node(state: SupervisorState) -> Dict:
        results = []
        for subtask in state["subtasks"]:
            role = subtask["agent"]
            persona = SPECIALIST_PERSONAS.get(role, "متخصص عام")
            system_prompt = f"أنت {role}. {persona}"
            result = call_llm(system_prompt, subtask["instruction"], model="gpt-4o-mini")
            cost_tracker.log(role, result["tokens"])
            results.append({"agent": role, "instruction": subtask["instruction"], "output": result["text"]})
        return {"results": results}
    return specialist_node


def make_aggregator_node(cost_tracker: CostTracker):
    def aggregator_node(state: SupervisorState) -> Dict:
        system_prompt = "أنت كاتب تقارير محترف تجمّع مخرجات فريق متخصصين في تقرير نهائي متماسك وخالٍ من التكرار."
        combined = "\n\n".join(f"[{r['agent']}] {r['output']}" for r in state["results"])
        prompt = f"المهمة الأصلية: {state['task']}\n\nمخرجات الفريق:\n{combined}\n\nاكتب تقريرًا نهائيًا موحدًا."
        result = call_llm(system_prompt, prompt, model="gpt-4o-mini")
        cost_tracker.log("Aggregator", result["tokens"])
        return {"final_report": result["text"]}
    return aggregator_node


def build_hierarchical_system(cost_tracker: CostTracker):
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", make_supervisor_node(cost_tracker))
    graph.add_node("specialists", make_specialist_node(cost_tracker))
    graph.add_node("aggregator", make_aggregator_node(cost_tracker))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "specialists")
    graph.add_edge("specialists", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ───────────── تشغيل تجريبي ─────────────
if __name__ == "__main__":
    cost_tracker_h = CostTracker()
    app = build_hierarchical_system(cost_tracker_h)

    final_state = app.invoke({
        "task": "حلّل أداء خوارزمية فرز سريع (Quick Sort) من ناحية التعقيد الزمني، "
                "اكتب تطبيقًا بسيطًا بلغة Python، ثم انتقد التطبيق واقترح تحسينات.",
        "subtasks": [], "results": [], "final_report": "",
    })

    print(final_state["final_report"])
    cost_tracker_h.report()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
from math import comb


class ConflictResolver:
    """يحل التعارض بين قرارات الوكلاء بطريقتين: تصويت مرجّح + رياضيات Condorcet"""

    @staticmethod
    def weighted_vote(proposals: List[Dict]) -> Dict:
        """proposals: [{"agent": "a1", "answer": "X", "confidence": 0.8}, ...]"""
        scores: Dict[str, float] = {}
        for p in proposals:
            scores[p["answer"]] = scores.get(p["answer"], 0.0) + p["confidence"]

        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        return {
            "winning_answer": winner,
            "vote_share": scores[winner] / total if total else 0,
            "all_scores": scores,
        }

    @staticmethod
    def condorcet_probability(n_agents: int, individual_accuracy: float) -> float:
        """احتمال صحة قرار الأغلبية — يفترض استقلالية الأخطاء (شرط حرج غالبًا مكسور فعليًا)"""
        if individual_accuracy <= 0.5:
            return individual_accuracy
        majority_needed = n_agents // 2 + 1
        return sum(
            comb(n_agents, k) * (individual_accuracy ** k) * ((1 - individual_accuracy) ** (n_agents - k))
            for k in range(majority_needed, n_agents + 1)
        )


# توضيح عملي: كم يضيف كل وكيل إضافي من دقة؟ (بافتراض استقلالية مثالية - نادرة في الواقع)
for n in [1, 3, 5, 7, 9]:
    p = ConflictResolver.condorcet_probability(n, individual_accuracy=0.65)
    print(f"N={n} وكلاء (دقة فردية 65%) → دقة الأغلبية: {p*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
def coordination_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Debate الكامل (Full Mesh): كل وكيل يقرأ رسائل كل الآخرين في كل جولة"""
    return rounds * n_agents * (n_agents - 1) * avg_tokens


def hierarchical_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Hierarchical (Star): المدير فقط يتواصل مع كل وكيل"""
    return rounds * n_agents * avg_tokens


print(f"{'N':>3} | {'Debate (Mesh)':>15} | {'Hierarchical (Star)':>20} | {'نسبة الانفجار':>15}")
print("-" * 65)
for n in [2, 3, 5, 8, 10, 15]:
    mesh = coordination_cost(n, rounds=3)
    star = hierarchical_cost(n, rounds=3)
    print(f"{n:>3} | {mesh:>15,} | {star:>20,} | {mesh/star:>14.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    {"question": "...", "correct_answer": "96 ريال (ليس 100)"},
    # ... 19 سؤالًا آخر بنفس الطراز (نسب مئوية متتالية، احتمالات شرطية، مغالطات شائعة)
]

def run_single_agent(question: str, cost_tracker: CostTracker) -> str:
    result = call_llm("حل المسألة بدقة خطوة بخطوة.", question, model="gpt-4o-mini")
    cost_tracker.log("SingleAgent", result["tokens"])
    return result["text"]

def evaluate_benchmark():
    single_tracker, debate_tracker = CostTracker(), CostTracker()
    single_correct, debate_correct = 0, 0

    for item in BENCHMARK_QUESTIONS:
        # شغّل single-agent وقيّم الإجابة (يدويًا أو عبر LLM-as-judge)
        # شغّل DebateOrchestrator وقيّم verdict["final_answer"]
        pass

    # اطبع جدول مقارنة نهائي: الدقة، التكلفة، نسبة الكفاءة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 56: الفصل 56: أنظمة Multi-Agent المتعاونة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multi_agent() -> None:
    recommendations = {'planner': 'retrieve', 'critic': 'retrieve', 'verifier': 'abstain'}
    counts = Counter(recommendations.values())
    winner, votes = counts.most_common(1)[0]
    print('agent votes:', recommendations)
    print('consensus:', winner, f'({votes}/{len(recommendations)})')
    print('escalate when disagreement exceeds the configured threshold.')

if __name__ == "__main__":
    print("Chapter 56 lab: Combine independent multi-agent recommendations")
    run_multi_agent()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypedDict, Annotated
import operator

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(system_prompt: str, user_prompt: str,
             model: str = "gpt-4o-mini", temperature: float = 0.7) -> Dict:
    """نداء موحّد لأي نموذج لغوي، يُرجع النص + عدد التوكنز لتتبّع التكلفة بدقة"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return {
        "text": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
    }


@dataclass
class CostTracker:
    """يتتبع تكلفة كل وكيل على حدة — ضروري لتشخيص انفجار التكلفة لاحقًا"""
    total_tokens: int = 0
    total_calls: int = 0
    history: List[Dict] = field(default_factory=list)

    def log(self, agent_name: str, tokens: int):
        self.total_tokens += tokens
        self.total_calls += 1
        self.history.append({"agent": agent_name, "tokens": tokens})

    def report(self):
        print(f"\n📊 إجمالي النداءات: {self.total_calls} | إجمالي التوكنز: {self.total_tokens}")
        by_agent: Dict[str, int] = {}
        for h in self.history:
            by_agent[h["agent"]] = by_agent.get(h["agent"], 0) + h["tokens"]
        for agent, tok in sorted(by_agent.items(), key=lambda x: -x[1]):
            pct = tok / self.total_tokens * 100 if self.total_tokens else 0
            print(f"  - {agent:<15}: {tok:>6} توكن ({pct:5.1f}%)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class DebateAgent:
    """وكيل يشارك في نقاش متعدد الجولات، يحتفظ بذاكرة مواقفه السابقة"""

    def __init__(self, name: str, persona: str, model: str, cost_tracker: CostTracker):
        self.name = name
        self.persona = persona
        self.model = model
        self.cost_tracker = cost_tracker
        self.memory: List[str] = []

    def respond(self, question: str, other_responses: List[str], round_num: int) -> str:
        system_prompt = f"""أنت {self.name}. شخصيتك ومنهجك: {self.persona}
أنت تشارك في نقاش علمي لحل مسألة بأعلى دقة ممكنة.

قواعد صارمة:
1. اقرأ إجابات الوكلاء الآخرين بعناية وابحث عن أخطاء منطقية أو حسابية فعلية.
2. لا تغيّر رأيك إلا إذا رأيت دليلًا أو منطقًا مقنعًا — لا تتقارب لمجرد المجاملة.
3. إن وجدت خطأً، اذكره صراحة بالأرقام أو الخطوات.
4. اختصر: لا تتجاوز 150 كلمة."""

        if round_num == 0:
            user_prompt = f"السؤال: {question}\n\nقدّم إجابتك الأولية مع التفكير خطوة بخطوة، واذكر درجة ثقتك (0-1)."
        else:
            others_text = "\n\n".join(f"[رأي آخر {i+1}]: {r}" for i, r in enumerate(other_responses))
            user_prompt = f"""السؤال: {question}

موقفك في الجولة السابقة: {self.memory[-1] if self.memory else 'لا يوجد'}

آراء الوكلاء الآخرين في الجولة السابقة:
{others_text}

راجع موقفك بدقة: هل تتمسك بإجابتك أم تعدّلها؟ قدّم إجابتك النهائية المحدّثة مع درجة ثقتك الجديدة."""

        result = call_llm(system_prompt, user_prompt, model=self.model)
        self.cost_tracker.log(self.name, result["tokens"])
        self.memory.append(result["text"])
        return result["text"]


class JudgeAgent:
    """حَكَم محايد يستخرج الإجابة الأصح بناءً على قوة الحجة لا على الأغلبية فقط"""

    def __init__(self, model: str, cost_tracker: CostTracker):
        self.model = model
        self.cost_tracker = cost_tracker

    def adjudicate(self, question: str, final_responses: Dict[str, str]) -> Dict:
        system_prompt = """أنت حَكَم خبير محايد تمامًا. اقرأ آراء عدة خبراء متنافسين
واستخرج الإجابة الأصح بناءً على قوة الأدلة والمنطق الرياضي/الحقائقي، وليس الأغلبية العددية.
أعد الناتج بصيغة JSON فقط دون أي نص إضافي."""

        responses_text = "\n\n".join(f"رأي {name}:\n{resp}" for name, resp in final_responses.items())
        user_prompt = f"""السؤال الأصلي: {question}

{responses_text}

أعد JSON بالشكل التالي حصرًا:
{{"final_answer": "...", "confidence": 0.0, "reasoning": "...", "agreement_level": "high/medium/low"}}"""

        result = call_llm(system_prompt, user_prompt, model=self.model, temperature=0.0)
        self.cost_tracker.log("Judge", result["tokens"])
        try:
            return json.loads(result["text"])
        except json.JSONDecodeError:
            return {"final_answer": result["text"], "confidence": 0.5,
                     "reasoning": "تعذر تحليل JSON", "agreement_level": "unknown"}


class DebateOrchestrator:
    """المنسّق: يدير الجولات، يفحص التقارب، ويستدعي الحَكَم عند الحاجة"""

    def __init__(self, agents: List[DebateAgent], judge: JudgeAgent,
                 max_rounds: int = 3, convergence_threshold: float = 0.90):
        self.agents = agents
        self.judge = judge
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self._embedder = None  # تحميل كسول (lazy loading)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embedder

    def _has_converged(self, responses: List[str]) -> bool:
        """يقيس التشابه الدلالي بين إجابات الوكلاء — تقارب حقيقي وليس تطابق نصي"""
        if len(responses) < 2:
            return False
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = self.embedder.encode(responses)
        sim_matrix = cosine_similarity(embeddings)
        n = len(responses)
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # استبعاد القطر (تشابه الوكيل مع نفسه = 1)
        return avg_sim >= self.convergence_threshold

    def run(self, question: str) -> Dict:
        previous_round: List[str] = []
        round_num = 0

        for round_num in range(self.max_rounds):
            current_round: List[str] = []
            for i, agent in enumerate(self.agents):
                others = [r for j, r in enumerate(previous_round) if j != i] if previous_round else []
                response = agent.respond(question, other_responses=others, round_num=round_num)
                current_round.append(response)

            previous_round = current_round
            print(f"--- الجولة {round_num + 1}/{self.max_rounds} اكتملت ---")

            if self._has_converged(current_round):
                print(f"✅ تقارب الوكلاء دلاليًا عند الجولة {round_num + 1}")
                break

        final_map = {agent.name: resp for agent, resp in zip(self.agents, previous_round)}
        verdict = self.judge.adjudicate(question, final_map)

        return {"rounds_used": round_num + 1, "final_responses": final_map, "verdict": verdict}


# ───────────── تشغيل تجريبي: مسألة استدلال منطقي مفخّخة ─────────────
if __name__ == "__main__":
    cost_tracker = CostTracker()

    # ملاحظة مهمة: نستخدم نماذج/شخصيات مختلفة لتفادي Echo Chamber
    agents = [
        DebateAgent("المحلل_الرياضي", "تحلل المسائل خطوة بخطوة بدقة حسابية صارمة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المدقق_المنطقي", "تبحث عن مغالطات منطقية وافتراضات خاطئة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المحامي_الناقد", "تتحدى كل إجابة وتطلب دليلًا دامغًا قبل القبول", "gpt-4o-mini", cost_tracker),
    ]
    judge = JudgeAgent("gpt-4o", cost_tracker)
    orchestrator = DebateOrchestrator(agents, judge, max_rounds=3, convergence_threshold=0.90)

    question = """متجر يبيع قميصًا بسعر 100 ريال. في عرض خاص، يُخفّض السعر 20% ثم يُرفع
السعر الناتج 20% مرة أخرى. هل السعر النهائي يساوي السعر الأصلي (100 ريال)؟ برّر إجابتك رياضيًا."""

    result = orchestrator.run(question)
    print("\n=== الحكم النهائي ===")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    cost_tracker.report()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END


class SupervisorState(TypedDict):
    task: str
    subtasks: List[Dict]
    results: Annotated[List[Dict], operator.add]
    final_report: str


SPECIALIST_PERSONAS = {
    "researcher": "خبير أبحاث يجمع الحقائق والبيانات بدقة ويذكر الافتراضات بوضوح.",
    "coder": "مهندس برمجيات يكتب كودًا نظيفًا وقابلًا للتشغيل فعليًا مع تعليقات.",
    "critic": "مراجع صارم يبحث عن الثغرات المنطقية والأخطاء التقنية قبل النشر.",
}


def make_supervisor_node(cost_tracker: CostTracker):
    def supervisor_node(state: SupervisorState) -> Dict:
        system_prompt = """أنت مدير مشروع تدير فريقًا من 3 متخصصين:
- researcher: يجمع المعلومات والحقائق
- coder: يكتب كودًا وينفّذ حسابات
- critic: يراجع وينتقد العمل المنجز

قسّم المهمة إلى مهام فرعية واضحة ومحددة (2-4 مهام)، وحدد المتخصص المناسب لكل منها.
أعد الناتج بصيغة JSON فقط: {"subtasks": [{"agent": "researcher|coder|critic", "instruction": "..."}]}"""

        result = call_llm(system_prompt, f"المهمة: {state['task']}", model="gpt-4o", temperature=0.3)
        cost_tracker.log("Supervisor", result["tokens"])

        try:
            subtasks = json.loads(result["text"])["subtasks"]
        except (json.JSONDecodeError, KeyError):
            subtasks = [{"agent": "researcher", "instruction": state["task"]}]

        return {"subtasks": subtasks}
    return supervisor_node


def make_specialist_node(cost_tracker: CostTracker):
    def specialist_node(state: SupervisorState) -> Dict:
        results = []
        for subtask in state["subtasks"]:
            role = subtask["agent"]
            persona = SPECIALIST_PERSONAS.get(role, "متخصص عام")
            system_prompt = f"أنت {role}. {persona}"
            result = call_llm(system_prompt, subtask["instruction"], model="gpt-4o-mini")
            cost_tracker.log(role, result["tokens"])
            results.append({"agent": role, "instruction": subtask["instruction"], "output": result["text"]})
        return {"results": results}
    return specialist_node


def make_aggregator_node(cost_tracker: CostTracker):
    def aggregator_node(state: SupervisorState) -> Dict:
        system_prompt = "أنت كاتب تقارير محترف تجمّع مخرجات فريق متخصصين في تقرير نهائي متماسك وخالٍ من التكرار."
        combined = "\n\n".join(f"[{r['agent']}] {r['output']}" for r in state["results"])
        prompt = f"المهمة الأصلية: {state['task']}\n\nمخرجات الفريق:\n{combined}\n\nاكتب تقريرًا نهائيًا موحدًا."
        result = call_llm(system_prompt, prompt, model="gpt-4o-mini")
        cost_tracker.log("Aggregator", result["tokens"])
        return {"final_report": result["text"]}
    return aggregator_node


def build_hierarchical_system(cost_tracker: CostTracker):
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", make_supervisor_node(cost_tracker))
    graph.add_node("specialists", make_specialist_node(cost_tracker))
    graph.add_node("aggregator", make_aggregator_node(cost_tracker))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "specialists")
    graph.add_edge("specialists", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ───────────── تشغيل تجريبي ─────────────
if __name__ == "__main__":
    cost_tracker_h = CostTracker()
    app = build_hierarchical_system(cost_tracker_h)

    final_state = app.invoke({
        "task": "حلّل أداء خوارزمية فرز سريع (Quick Sort) من ناحية التعقيد الزمني، "
                "اكتب تطبيقًا بسيطًا بلغة Python، ثم انتقد التطبيق واقترح تحسينات.",
        "subtasks": [], "results": [], "final_report": "",
    })

    print(final_state["final_report"])
    cost_tracker_h.report()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from math import comb


class ConflictResolver:
    """يحل التعارض بين قرارات الوكلاء بطريقتين: تصويت مرجّح + رياضيات Condorcet"""

    @staticmethod
    def weighted_vote(proposals: List[Dict]) -> Dict:
        """proposals: [{"agent": "a1", "answer": "X", "confidence": 0.8}, ...]"""
        scores: Dict[str, float] = {}
        for p in proposals:
            scores[p["answer"]] = scores.get(p["answer"], 0.0) + p["confidence"]

        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        return {
            "winning_answer": winner,
            "vote_share": scores[winner] / total if total else 0,
            "all_scores": scores,
        }

    @staticmethod
    def condorcet_probability(n_agents: int, individual_accuracy: float) -> float:
        """احتمال صحة قرار الأغلبية — يفترض استقلالية الأخطاء (شرط حرج غالبًا مكسور فعليًا)"""
        if individual_accuracy <= 0.5:
            return individual_accuracy
        majority_needed = n_agents // 2 + 1
        return sum(
            comb(n_agents, k) * (individual_accuracy ** k) * ((1 - individual_accuracy) ** (n_agents - k))
            for k in range(majority_needed, n_agents + 1)
        )


# توضيح عملي: كم يضيف كل وكيل إضافي من دقة؟ (بافتراض استقلالية مثالية - نادرة في الواقع)
for n in [1, 3, 5, 7, 9]:
    p = ConflictResolver.condorcet_probability(n, individual_accuracy=0.65)
    print(f"N={n} وكلاء (دقة فردية 65%) → دقة الأغلبية: {p*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def coordination_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Debate الكامل (Full Mesh): كل وكيل يقرأ رسائل كل الآخرين في كل جولة"""
    return rounds * n_agents * (n_agents - 1) * avg_tokens


def hierarchical_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Hierarchical (Star): المدير فقط يتواصل مع كل وكيل"""
    return rounds * n_agents * avg_tokens


print(f"{'N':>3} | {'Debate (Mesh)':>15} | {'Hierarchical (Star)':>20} | {'نسبة الانفجار':>15}")
print("-" * 65)
for n in [2, 3, 5, 8, 10, 15]:
    mesh = coordination_cost(n, rounds=3)
    star = hierarchical_cost(n, rounds=3)
    print(f"{n:>3} | {mesh:>15,} | {star:>20,} | {mesh/star:>14.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    {"question": "...", "correct_answer": "96 ريال (ليس 100)"},
    # ... 19 سؤالًا آخر بنفس الطراز (نسب مئوية متتالية، احتمالات شرطية، مغالطات شائعة)
]

def run_single_agent(question: str, cost_tracker: CostTracker) -> str:
    result = call_llm("حل المسألة بدقة خطوة بخطوة.", question, model="gpt-4o-mini")
    cost_tracker.log("SingleAgent", result["tokens"])
    return result["text"]

def evaluate_benchmark():
    single_tracker, debate_tracker = CostTracker(), CostTracker()
    single_correct, debate_correct = 0, 0

    for item in BENCHMARK_QUESTIONS:
        # شغّل single-agent وقيّم الإجابة (يدويًا أو عبر LLM-as-judge)
        # شغّل DebateOrchestrator وقيّم verdict["final_answer"]
        pass

    # اطبع جدول مقارنة نهائي: الدقة، التكلفة، نسبة الكفاءة

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أنظمة Multi-Agent المتعاونة
Large Language Model Engineering — Multi-Agent

Book slug : llm-engineering-69-chapters
Chapter   : ch-56-chapter-56 (lesson 56)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_56.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 56: الفصل 56: أنظمة Multi-Agent المتعاونة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multi_agent() -> None:
    recommendations = {'planner': 'retrieve', 'critic': 'retrieve', 'verifier': 'abstain'}
    counts = Counter(recommendations.values())
    winner, votes = counts.most_common(1)[0]
    print('agent votes:', recommendations)
    print('consensus:', winner, f'({votes}/{len(recommendations)})')
    print('escalate when disagreement exceeds the configured threshold.')

if __name__ == "__main__":
    print("Chapter 56 lab: Combine independent multi-agent recommendations")
    run_multi_agent()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypedDict, Annotated
import operator

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(system_prompt: str, user_prompt: str,
             model: str = "gpt-4o-mini", temperature: float = 0.7) -> Dict:
    """نداء موحّد لأي نموذج لغوي، يُرجع النص + عدد التوكنز لتتبّع التكلفة بدقة"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return {
        "text": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
    }


@dataclass
class CostTracker:
    """يتتبع تكلفة كل وكيل على حدة — ضروري لتشخيص انفجار التكلفة لاحقًا"""
    total_tokens: int = 0
    total_calls: int = 0
    history: List[Dict] = field(default_factory=list)

    def log(self, agent_name: str, tokens: int):
        self.total_tokens += tokens
        self.total_calls += 1
        self.history.append({"agent": agent_name, "tokens": tokens})

    def report(self):
        print(f"\n📊 إجمالي النداءات: {self.total_calls} | إجمالي التوكنز: {self.total_tokens}")
        by_agent: Dict[str, int] = {}
        for h in self.history:
            by_agent[h["agent"]] = by_agent.get(h["agent"], 0) + h["tokens"]
        for agent, tok in sorted(by_agent.items(), key=lambda x: -x[1]):
            pct = tok / self.total_tokens * 100 if self.total_tokens else 0
            print(f"  - {agent:<15}: {tok:>6} توكن ({pct:5.1f}%)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
class DebateAgent:
    """وكيل يشارك في نقاش متعدد الجولات، يحتفظ بذاكرة مواقفه السابقة"""

    def __init__(self, name: str, persona: str, model: str, cost_tracker: CostTracker):
        self.name = name
        self.persona = persona
        self.model = model
        self.cost_tracker = cost_tracker
        self.memory: List[str] = []

    def respond(self, question: str, other_responses: List[str], round_num: int) -> str:
        system_prompt = f"""أنت {self.name}. شخصيتك ومنهجك: {self.persona}
أنت تشارك في نقاش علمي لحل مسألة بأعلى دقة ممكنة.

قواعد صارمة:
1. اقرأ إجابات الوكلاء الآخرين بعناية وابحث عن أخطاء منطقية أو حسابية فعلية.
2. لا تغيّر رأيك إلا إذا رأيت دليلًا أو منطقًا مقنعًا — لا تتقارب لمجرد المجاملة.
3. إن وجدت خطأً، اذكره صراحة بالأرقام أو الخطوات.
4. اختصر: لا تتجاوز 150 كلمة."""

        if round_num == 0:
            user_prompt = f"السؤال: {question}\n\nقدّم إجابتك الأولية مع التفكير خطوة بخطوة، واذكر درجة ثقتك (0-1)."
        else:
            others_text = "\n\n".join(f"[رأي آخر {i+1}]: {r}" for i, r in enumerate(other_responses))
            user_prompt = f"""السؤال: {question}

موقفك في الجولة السابقة: {self.memory[-1] if self.memory else 'لا يوجد'}

آراء الوكلاء الآخرين في الجولة السابقة:
{others_text}

راجع موقفك بدقة: هل تتمسك بإجابتك أم تعدّلها؟ قدّم إجابتك النهائية المحدّثة مع درجة ثقتك الجديدة."""

        result = call_llm(system_prompt, user_prompt, model=self.model)
        self.cost_tracker.log(self.name, result["tokens"])
        self.memory.append(result["text"])
        return result["text"]


class JudgeAgent:
    """حَكَم محايد يستخرج الإجابة الأصح بناءً على قوة الحجة لا على الأغلبية فقط"""

    def __init__(self, model: str, cost_tracker: CostTracker):
        self.model = model
        self.cost_tracker = cost_tracker

    def adjudicate(self, question: str, final_responses: Dict[str, str]) -> Dict:
        system_prompt = """أنت حَكَم خبير محايد تمامًا. اقرأ آراء عدة خبراء متنافسين
واستخرج الإجابة الأصح بناءً على قوة الأدلة والمنطق الرياضي/الحقائقي، وليس الأغلبية العددية.
أعد الناتج بصيغة JSON فقط دون أي نص إضافي."""

        responses_text = "\n\n".join(f"رأي {name}:\n{resp}" for name, resp in final_responses.items())
        user_prompt = f"""السؤال الأصلي: {question}

{responses_text}

أعد JSON بالشكل التالي حصرًا:
{{"final_answer": "...", "confidence": 0.0, "reasoning": "...", "agreement_level": "high/medium/low"}}"""

        result = call_llm(system_prompt, user_prompt, model=self.model, temperature=0.0)
        self.cost_tracker.log("Judge", result["tokens"])
        try:
            return json.loads(result["text"])
        except json.JSONDecodeError:
            return {"final_answer": result["text"], "confidence": 0.5,
                     "reasoning": "تعذر تحليل JSON", "agreement_level": "unknown"}


class DebateOrchestrator:
    """المنسّق: يدير الجولات، يفحص التقارب، ويستدعي الحَكَم عند الحاجة"""

    def __init__(self, agents: List[DebateAgent], judge: JudgeAgent,
                 max_rounds: int = 3, convergence_threshold: float = 0.90):
        self.agents = agents
        self.judge = judge
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self._embedder = None  # تحميل كسول (lazy loading)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embedder

    def _has_converged(self, responses: List[str]) -> bool:
        """يقيس التشابه الدلالي بين إجابات الوكلاء — تقارب حقيقي وليس تطابق نصي"""
        if len(responses) < 2:
            return False
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = self.embedder.encode(responses)
        sim_matrix = cosine_similarity(embeddings)
        n = len(responses)
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # استبعاد القطر (تشابه الوكيل مع نفسه = 1)
        return avg_sim >= self.convergence_threshold

    def run(self, question: str) -> Dict:
        previous_round: List[str] = []
        round_num = 0

        for round_num in range(self.max_rounds):
            current_round: List[str] = []
            for i, agent in enumerate(self.agents):
                others = [r for j, r in enumerate(previous_round) if j != i] if previous_round else []
                response = agent.respond(question, other_responses=others, round_num=round_num)
                current_round.append(response)

            previous_round = current_round
            print(f"--- الجولة {round_num + 1}/{self.max_rounds} اكتملت ---")

            if self._has_converged(current_round):
                print(f"✅ تقارب الوكلاء دلاليًا عند الجولة {round_num + 1}")
                break

        final_map = {agent.name: resp for agent, resp in zip(self.agents, previous_round)}
        verdict = self.judge.adjudicate(question, final_map)

        return {"rounds_used": round_num + 1, "final_responses": final_map, "verdict": verdict}


# ───────────── تشغيل تجريبي: مسألة استدلال منطقي مفخّخة ─────────────
if __name__ == "__main__":
    cost_tracker = CostTracker()

    # ملاحظة مهمة: نستخدم نماذج/شخصيات مختلفة لتفادي Echo Chamber
    agents = [
        DebateAgent("المحلل_الرياضي", "تحلل المسائل خطوة بخطوة بدقة حسابية صارمة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المدقق_المنطقي", "تبحث عن مغالطات منطقية وافتراضات خاطئة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المحامي_الناقد", "تتحدى كل إجابة وتطلب دليلًا دامغًا قبل القبول", "gpt-4o-mini", cost_tracker),
    ]
    judge = JudgeAgent("gpt-4o", cost_tracker)
    orchestrator = DebateOrchestrator(agents, judge, max_rounds=3, convergence_threshold=0.90)

    question = """متجر يبيع قميصًا بسعر 100 ريال. في عرض خاص، يُخفّض السعر 20% ثم يُرفع
السعر الناتج 20% مرة أخرى. هل السعر النهائي يساوي السعر الأصلي (100 ريال)؟ برّر إجابتك رياضيًا."""

    result = orchestrator.run(question)
    print("\n=== الحكم النهائي ===")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    cost_tracker.report()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END


class SupervisorState(TypedDict):
    task: str
    subtasks: List[Dict]
    results: Annotated[List[Dict], operator.add]
    final_report: str


SPECIALIST_PERSONAS = {
    "researcher": "خبير أبحاث يجمع الحقائق والبيانات بدقة ويذكر الافتراضات بوضوح.",
    "coder": "مهندس برمجيات يكتب كودًا نظيفًا وقابلًا للتشغيل فعليًا مع تعليقات.",
    "critic": "مراجع صارم يبحث عن الثغرات المنطقية والأخطاء التقنية قبل النشر.",
}


def make_supervisor_node(cost_tracker: CostTracker):
    def supervisor_node(state: SupervisorState) -> Dict:
        system_prompt = """أنت مدير مشروع تدير فريقًا من 3 متخصصين:
- researcher: يجمع المعلومات والحقائق
- coder: يكتب كودًا وينفّذ حسابات
- critic: يراجع وينتقد العمل المنجز

قسّم المهمة إلى مهام فرعية واضحة ومحددة (2-4 مهام)، وحدد المتخصص المناسب لكل منها.
أعد الناتج بصيغة JSON فقط: {"subtasks": [{"agent": "researcher|coder|critic", "instruction": "..."}]}"""

        result = call_llm(system_prompt, f"المهمة: {state['task']}", model="gpt-4o", temperature=0.3)
        cost_tracker.log("Supervisor", result["tokens"])

        try:
            subtasks = json.loads(result["text"])["subtasks"]
        except (json.JSONDecodeError, KeyError):
            subtasks = [{"agent": "researcher", "instruction": state["task"]}]

        return {"subtasks": subtasks}
    return supervisor_node


def make_specialist_node(cost_tracker: CostTracker):
    def specialist_node(state: SupervisorState) -> Dict:
        results = []
        for subtask in state["subtasks"]:
            role = subtask["agent"]
            persona = SPECIALIST_PERSONAS.get(role, "متخصص عام")
            system_prompt = f"أنت {role}. {persona}"
            result = call_llm(system_prompt, subtask["instruction"], model="gpt-4o-mini")
            cost_tracker.log(role, result["tokens"])
            results.append({"agent": role, "instruction": subtask["instruction"], "output": result["text"]})
        return {"results": results}
    return specialist_node


def make_aggregator_node(cost_tracker: CostTracker):
    def aggregator_node(state: SupervisorState) -> Dict:
        system_prompt = "أنت كاتب تقارير محترف تجمّع مخرجات فريق متخصصين في تقرير نهائي متماسك وخالٍ من التكرار."
        combined = "\n\n".join(f"[{r['agent']}] {r['output']}" for r in state["results"])
        prompt = f"المهمة الأصلية: {state['task']}\n\nمخرجات الفريق:\n{combined}\n\nاكتب تقريرًا نهائيًا موحدًا."
        result = call_llm(system_prompt, prompt, model="gpt-4o-mini")
        cost_tracker.log("Aggregator", result["tokens"])
        return {"final_report": result["text"]}
    return aggregator_node


def build_hierarchical_system(cost_tracker: CostTracker):
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", make_supervisor_node(cost_tracker))
    graph.add_node("specialists", make_specialist_node(cost_tracker))
    graph.add_node("aggregator", make_aggregator_node(cost_tracker))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "specialists")
    graph.add_edge("specialists", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ───────────── تشغيل تجريبي ─────────────
if __name__ == "__main__":
    cost_tracker_h = CostTracker()
    app = build_hierarchical_system(cost_tracker_h)

    final_state = app.invoke({
        "task": "حلّل أداء خوارزمية فرز سريع (Quick Sort) من ناحية التعقيد الزمني، "
                "اكتب تطبيقًا بسيطًا بلغة Python، ثم انتقد التطبيق واقترح تحسينات.",
        "subtasks": [], "results": [], "final_report": "",
    })

    print(final_state["final_report"])
    cost_tracker_h.report()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
from math import comb


class ConflictResolver:
    """يحل التعارض بين قرارات الوكلاء بطريقتين: تصويت مرجّح + رياضيات Condorcet"""

    @staticmethod
    def weighted_vote(proposals: List[Dict]) -> Dict:
        """proposals: [{"agent": "a1", "answer": "X", "confidence": 0.8}, ...]"""
        scores: Dict[str, float] = {}
        for p in proposals:
            scores[p["answer"]] = scores.get(p["answer"], 0.0) + p["confidence"]

        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        return {
            "winning_answer": winner,
            "vote_share": scores[winner] / total if total else 0,
            "all_scores": scores,
        }

    @staticmethod
    def condorcet_probability(n_agents: int, individual_accuracy: float) -> float:
        """احتمال صحة قرار الأغلبية — يفترض استقلالية الأخطاء (شرط حرج غالبًا مكسور فعليًا)"""
        if individual_accuracy <= 0.5:
            return individual_accuracy
        majority_needed = n_agents // 2 + 1
        return sum(
            comb(n_agents, k) * (individual_accuracy ** k) * ((1 - individual_accuracy) ** (n_agents - k))
            for k in range(majority_needed, n_agents + 1)
        )


# توضيح عملي: كم يضيف كل وكيل إضافي من دقة؟ (بافتراض استقلالية مثالية - نادرة في الواقع)
for n in [1, 3, 5, 7, 9]:
    p = ConflictResolver.condorcet_probability(n, individual_accuracy=0.65)
    print(f"N={n} وكلاء (دقة فردية 65%) → دقة الأغلبية: {p*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
def coordination_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Debate الكامل (Full Mesh): كل وكيل يقرأ رسائل كل الآخرين في كل جولة"""
    return rounds * n_agents * (n_agents - 1) * avg_tokens


def hierarchical_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Hierarchical (Star): المدير فقط يتواصل مع كل وكيل"""
    return rounds * n_agents * avg_tokens


print(f"{'N':>3} | {'Debate (Mesh)':>15} | {'Hierarchical (Star)':>20} | {'نسبة الانفجار':>15}")
print("-" * 65)
for n in [2, 3, 5, 8, 10, 15]:
    mesh = coordination_cost(n, rounds=3)
    star = hierarchical_cost(n, rounds=3)
    print(f"{n:>3} | {mesh:>15,} | {star:>20,} | {mesh/star:>14.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-10-applications-agents/chapter-56.md
# ────────────────────────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    {"question": "...", "correct_answer": "96 ريال (ليس 100)"},
    # ... 19 سؤالًا آخر بنفس الطراز (نسب مئوية متتالية، احتمالات شرطية، مغالطات شائعة)
]

def run_single_agent(question: str, cost_tracker: CostTracker) -> str:
    result = call_llm("حل المسألة بدقة خطوة بخطوة.", question, model="gpt-4o-mini")
    cost_tracker.log("SingleAgent", result["tokens"])
    return result["text"]

def evaluate_benchmark():
    single_tracker, debate_tracker = CostTracker(), CostTracker()
    single_correct, debate_correct = 0, 0

    for item in BENCHMARK_QUESTIONS:
        # شغّل single-agent وقيّم الإجابة (يدويًا أو عبر LLM-as-judge)
        # شغّل DebateOrchestrator وقيّم verdict["final_answer"]
        pass

    # اطبع جدول مقارنة نهائي: الدقة، التكلفة، نسبة الكفاءة

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 56: الفصل 56: أنظمة Multi-Agent المتعاونة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_multi_agent() -> None:
    recommendations = {'planner': 'retrieve', 'critic': 'retrieve', 'verifier': 'abstain'}
    counts = Counter(recommendations.values())
    winner, votes = counts.most_common(1)[0]
    print('agent votes:', recommendations)
    print('consensus:', winner, f'({votes}/{len(recommendations)})')
    print('escalate when disagreement exceeds the configured threshold.')

if __name__ == "__main__":
    print("Chapter 56 lab: Combine independent multi-agent recommendations")
    run_multi_agent()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TypedDict, Annotated
import operator

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(system_prompt: str, user_prompt: str,
             model: str = "gpt-4o-mini", temperature: float = 0.7) -> Dict:
    """نداء موحّد لأي نموذج لغوي، يُرجع النص + عدد التوكنز لتتبّع التكلفة بدقة"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return {
        "text": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
    }


@dataclass
class CostTracker:
    """يتتبع تكلفة كل وكيل على حدة — ضروري لتشخيص انفجار التكلفة لاحقًا"""
    total_tokens: int = 0
    total_calls: int = 0
    history: List[Dict] = field(default_factory=list)

    def log(self, agent_name: str, tokens: int):
        self.total_tokens += tokens
        self.total_calls += 1
        self.history.append({"agent": agent_name, "tokens": tokens})

    def report(self):
        print(f"\n📊 إجمالي النداءات: {self.total_calls} | إجمالي التوكنز: {self.total_tokens}")
        by_agent: Dict[str, int] = {}
        for h in self.history:
            by_agent[h["agent"]] = by_agent.get(h["agent"], 0) + h["tokens"]
        for agent, tok in sorted(by_agent.items(), key=lambda x: -x[1]):
            pct = tok / self.total_tokens * 100 if self.total_tokens else 0
            print(f"  - {agent:<15}: {tok:>6} توكن ({pct:5.1f}%)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class DebateAgent:
    """وكيل يشارك في نقاش متعدد الجولات، يحتفظ بذاكرة مواقفه السابقة"""

    def __init__(self, name: str, persona: str, model: str, cost_tracker: CostTracker):
        self.name = name
        self.persona = persona
        self.model = model
        self.cost_tracker = cost_tracker
        self.memory: List[str] = []

    def respond(self, question: str, other_responses: List[str], round_num: int) -> str:
        system_prompt = f"""أنت {self.name}. شخصيتك ومنهجك: {self.persona}
أنت تشارك في نقاش علمي لحل مسألة بأعلى دقة ممكنة.

قواعد صارمة:
1. اقرأ إجابات الوكلاء الآخرين بعناية وابحث عن أخطاء منطقية أو حسابية فعلية.
2. لا تغيّر رأيك إلا إذا رأيت دليلًا أو منطقًا مقنعًا — لا تتقارب لمجرد المجاملة.
3. إن وجدت خطأً، اذكره صراحة بالأرقام أو الخطوات.
4. اختصر: لا تتجاوز 150 كلمة."""

        if round_num == 0:
            user_prompt = f"السؤال: {question}\n\nقدّم إجابتك الأولية مع التفكير خطوة بخطوة، واذكر درجة ثقتك (0-1)."
        else:
            others_text = "\n\n".join(f"[رأي آخر {i+1}]: {r}" for i, r in enumerate(other_responses))
            user_prompt = f"""السؤال: {question}

موقفك في الجولة السابقة: {self.memory[-1] if self.memory else 'لا يوجد'}

آراء الوكلاء الآخرين في الجولة السابقة:
{others_text}

راجع موقفك بدقة: هل تتمسك بإجابتك أم تعدّلها؟ قدّم إجابتك النهائية المحدّثة مع درجة ثقتك الجديدة."""

        result = call_llm(system_prompt, user_prompt, model=self.model)
        self.cost_tracker.log(self.name, result["tokens"])
        self.memory.append(result["text"])
        return result["text"]


class JudgeAgent:
    """حَكَم محايد يستخرج الإجابة الأصح بناءً على قوة الحجة لا على الأغلبية فقط"""

    def __init__(self, model: str, cost_tracker: CostTracker):
        self.model = model
        self.cost_tracker = cost_tracker

    def adjudicate(self, question: str, final_responses: Dict[str, str]) -> Dict:
        system_prompt = """أنت حَكَم خبير محايد تمامًا. اقرأ آراء عدة خبراء متنافسين
واستخرج الإجابة الأصح بناءً على قوة الأدلة والمنطق الرياضي/الحقائقي، وليس الأغلبية العددية.
أعد الناتج بصيغة JSON فقط دون أي نص إضافي."""

        responses_text = "\n\n".join(f"رأي {name}:\n{resp}" for name, resp in final_responses.items())
        user_prompt = f"""السؤال الأصلي: {question}

{responses_text}

أعد JSON بالشكل التالي حصرًا:
{{"final_answer": "...", "confidence": 0.0, "reasoning": "...", "agreement_level": "high/medium/low"}}"""

        result = call_llm(system_prompt, user_prompt, model=self.model, temperature=0.0)
        self.cost_tracker.log("Judge", result["tokens"])
        try:
            return json.loads(result["text"])
        except json.JSONDecodeError:
            return {"final_answer": result["text"], "confidence": 0.5,
                     "reasoning": "تعذر تحليل JSON", "agreement_level": "unknown"}


class DebateOrchestrator:
    """المنسّق: يدير الجولات، يفحص التقارب، ويستدعي الحَكَم عند الحاجة"""

    def __init__(self, agents: List[DebateAgent], judge: JudgeAgent,
                 max_rounds: int = 3, convergence_threshold: float = 0.90):
        self.agents = agents
        self.judge = judge
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self._embedder = None  # تحميل كسول (lazy loading)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embedder

    def _has_converged(self, responses: List[str]) -> bool:
        """يقيس التشابه الدلالي بين إجابات الوكلاء — تقارب حقيقي وليس تطابق نصي"""
        if len(responses) < 2:
            return False
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = self.embedder.encode(responses)
        sim_matrix = cosine_similarity(embeddings)
        n = len(responses)
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))  # استبعاد القطر (تشابه الوكيل مع نفسه = 1)
        return avg_sim >= self.convergence_threshold

    def run(self, question: str) -> Dict:
        previous_round: List[str] = []
        round_num = 0

        for round_num in range(self.max_rounds):
            current_round: List[str] = []
            for i, agent in enumerate(self.agents):
                others = [r for j, r in enumerate(previous_round) if j != i] if previous_round else []
                response = agent.respond(question, other_responses=others, round_num=round_num)
                current_round.append(response)

            previous_round = current_round
            print(f"--- الجولة {round_num + 1}/{self.max_rounds} اكتملت ---")

            if self._has_converged(current_round):
                print(f"✅ تقارب الوكلاء دلاليًا عند الجولة {round_num + 1}")
                break

        final_map = {agent.name: resp for agent, resp in zip(self.agents, previous_round)}
        verdict = self.judge.adjudicate(question, final_map)

        return {"rounds_used": round_num + 1, "final_responses": final_map, "verdict": verdict}


# ───────────── تشغيل تجريبي: مسألة استدلال منطقي مفخّخة ─────────────
if __name__ == "__main__":
    cost_tracker = CostTracker()

    # ملاحظة مهمة: نستخدم نماذج/شخصيات مختلفة لتفادي Echo Chamber
    agents = [
        DebateAgent("المحلل_الرياضي", "تحلل المسائل خطوة بخطوة بدقة حسابية صارمة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المدقق_المنطقي", "تبحث عن مغالطات منطقية وافتراضات خاطئة", "gpt-4o-mini", cost_tracker),
        DebateAgent("المحامي_الناقد", "تتحدى كل إجابة وتطلب دليلًا دامغًا قبل القبول", "gpt-4o-mini", cost_tracker),
    ]
    judge = JudgeAgent("gpt-4o", cost_tracker)
    orchestrator = DebateOrchestrator(agents, judge, max_rounds=3, convergence_threshold=0.90)

    question = """متجر يبيع قميصًا بسعر 100 ريال. في عرض خاص، يُخفّض السعر 20% ثم يُرفع
السعر الناتج 20% مرة أخرى. هل السعر النهائي يساوي السعر الأصلي (100 ريال)؟ برّر إجابتك رياضيًا."""

    result = orchestrator.run(question)
    print("\n=== الحكم النهائي ===")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    cost_tracker.report()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END


class SupervisorState(TypedDict):
    task: str
    subtasks: List[Dict]
    results: Annotated[List[Dict], operator.add]
    final_report: str


SPECIALIST_PERSONAS = {
    "researcher": "خبير أبحاث يجمع الحقائق والبيانات بدقة ويذكر الافتراضات بوضوح.",
    "coder": "مهندس برمجيات يكتب كودًا نظيفًا وقابلًا للتشغيل فعليًا مع تعليقات.",
    "critic": "مراجع صارم يبحث عن الثغرات المنطقية والأخطاء التقنية قبل النشر.",
}


def make_supervisor_node(cost_tracker: CostTracker):
    def supervisor_node(state: SupervisorState) -> Dict:
        system_prompt = """أنت مدير مشروع تدير فريقًا من 3 متخصصين:
- researcher: يجمع المعلومات والحقائق
- coder: يكتب كودًا وينفّذ حسابات
- critic: يراجع وينتقد العمل المنجز

قسّم المهمة إلى مهام فرعية واضحة ومحددة (2-4 مهام)، وحدد المتخصص المناسب لكل منها.
أعد الناتج بصيغة JSON فقط: {"subtasks": [{"agent": "researcher|coder|critic", "instruction": "..."}]}"""

        result = call_llm(system_prompt, f"المهمة: {state['task']}", model="gpt-4o", temperature=0.3)
        cost_tracker.log("Supervisor", result["tokens"])

        try:
            subtasks = json.loads(result["text"])["subtasks"]
        except (json.JSONDecodeError, KeyError):
            subtasks = [{"agent": "researcher", "instruction": state["task"]}]

        return {"subtasks": subtasks}
    return supervisor_node


def make_specialist_node(cost_tracker: CostTracker):
    def specialist_node(state: SupervisorState) -> Dict:
        results = []
        for subtask in state["subtasks"]:
            role = subtask["agent"]
            persona = SPECIALIST_PERSONAS.get(role, "متخصص عام")
            system_prompt = f"أنت {role}. {persona}"
            result = call_llm(system_prompt, subtask["instruction"], model="gpt-4o-mini")
            cost_tracker.log(role, result["tokens"])
            results.append({"agent": role, "instruction": subtask["instruction"], "output": result["text"]})
        return {"results": results}
    return specialist_node


def make_aggregator_node(cost_tracker: CostTracker):
    def aggregator_node(state: SupervisorState) -> Dict:
        system_prompt = "أنت كاتب تقارير محترف تجمّع مخرجات فريق متخصصين في تقرير نهائي متماسك وخالٍ من التكرار."
        combined = "\n\n".join(f"[{r['agent']}] {r['output']}" for r in state["results"])
        prompt = f"المهمة الأصلية: {state['task']}\n\nمخرجات الفريق:\n{combined}\n\nاكتب تقريرًا نهائيًا موحدًا."
        result = call_llm(system_prompt, prompt, model="gpt-4o-mini")
        cost_tracker.log("Aggregator", result["tokens"])
        return {"final_report": result["text"]}
    return aggregator_node


def build_hierarchical_system(cost_tracker: CostTracker):
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", make_supervisor_node(cost_tracker))
    graph.add_node("specialists", make_specialist_node(cost_tracker))
    graph.add_node("aggregator", make_aggregator_node(cost_tracker))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "specialists")
    graph.add_edge("specialists", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


# ───────────── تشغيل تجريبي ─────────────
if __name__ == "__main__":
    cost_tracker_h = CostTracker()
    app = build_hierarchical_system(cost_tracker_h)

    final_state = app.invoke({
        "task": "حلّل أداء خوارزمية فرز سريع (Quick Sort) من ناحية التعقيد الزمني، "
                "اكتب تطبيقًا بسيطًا بلغة Python، ثم انتقد التطبيق واقترح تحسينات.",
        "subtasks": [], "results": [], "final_report": "",
    })

    print(final_state["final_report"])
    cost_tracker_h.report()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from math import comb


class ConflictResolver:
    """يحل التعارض بين قرارات الوكلاء بطريقتين: تصويت مرجّح + رياضيات Condorcet"""

    @staticmethod
    def weighted_vote(proposals: List[Dict]) -> Dict:
        """proposals: [{"agent": "a1", "answer": "X", "confidence": 0.8}, ...]"""
        scores: Dict[str, float] = {}
        for p in proposals:
            scores[p["answer"]] = scores.get(p["answer"], 0.0) + p["confidence"]

        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        return {
            "winning_answer": winner,
            "vote_share": scores[winner] / total if total else 0,
            "all_scores": scores,
        }

    @staticmethod
    def condorcet_probability(n_agents: int, individual_accuracy: float) -> float:
        """احتمال صحة قرار الأغلبية — يفترض استقلالية الأخطاء (شرط حرج غالبًا مكسور فعليًا)"""
        if individual_accuracy <= 0.5:
            return individual_accuracy
        majority_needed = n_agents // 2 + 1
        return sum(
            comb(n_agents, k) * (individual_accuracy ** k) * ((1 - individual_accuracy) ** (n_agents - k))
            for k in range(majority_needed, n_agents + 1)
        )


# توضيح عملي: كم يضيف كل وكيل إضافي من دقة؟ (بافتراض استقلالية مثالية - نادرة في الواقع)
for n in [1, 3, 5, 7, 9]:
    p = ConflictResolver.condorcet_probability(n, individual_accuracy=0.65)
    print(f"N={n} وكلاء (دقة فردية 65%) → دقة الأغلبية: {p*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def coordination_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Debate الكامل (Full Mesh): كل وكيل يقرأ رسائل كل الآخرين في كل جولة"""
    return rounds * n_agents * (n_agents - 1) * avg_tokens


def hierarchical_cost(n_agents: int, rounds: int, avg_tokens: int = 300) -> int:
    """Hierarchical (Star): المدير فقط يتواصل مع كل وكيل"""
    return rounds * n_agents * avg_tokens


print(f"{'N':>3} | {'Debate (Mesh)':>15} | {'Hierarchical (Star)':>20} | {'نسبة الانفجار':>15}")
print("-" * 65)
for n in [2, 3, 5, 8, 10, 15]:
    mesh = coordination_cost(n, rounds=3)
    star = hierarchical_cost(n, rounds=3)
    print(f"{n:>3} | {mesh:>15,} | {star:>20,} | {mesh/star:>14.1f}x")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-56-chapter-56.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    {"question": "...", "correct_answer": "96 ريال (ليس 100)"},
    # ... 19 سؤالًا آخر بنفس الطراز (نسب مئوية متتالية، احتمالات شرطية، مغالطات شائعة)
]

def run_single_agent(question: str, cost_tracker: CostTracker) -> str:
    result = call_llm("حل المسألة بدقة خطوة بخطوة.", question, model="gpt-4o-mini")
    cost_tracker.log("SingleAgent", result["tokens"])
    return result["text"]

def evaluate_benchmark():
    single_tracker, debate_tracker = CostTracker(), CostTracker()
    single_correct, debate_correct = 0, 0

    for item in BENCHMARK_QUESTIONS:
        # شغّل single-agent وقيّم الإجابة (يدويًا أو عبر LLM-as-judge)
        # شغّل DebateOrchestrator وقيّم verdict["final_answer"]
        pass

    # اطبع جدول مقارنة نهائي: الدقة، التكلفة، نسبة الكفاءة
