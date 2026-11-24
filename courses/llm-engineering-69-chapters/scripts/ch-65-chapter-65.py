#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"
Large Language Model Engineering — Test-Time

Book slug : llm-engineering-69-chapters
Chapter   : ch-65-chapter-65 (lesson 65)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-65-chapter-65.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"
Large Language Model Engineering — Test-Time

Book slug : llm-engineering-69-chapters
Chapter   : ch-65-chapter-65 (lesson 65)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-65-chapter-65.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"
Large Language Model Engineering — Test-Time

Book slug : llm-engineering-69-chapters
Chapter   : ch-65-chapter-65 (lesson 65)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_65.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 65: الفصل 65: نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_test_time() -> None:
    candidates = ['A', 'B', 'A', 'A', 'C', 'B', 'A']
    counts = Counter(candidates)
    winner, votes = counts.most_common(1)[0]
    print('candidates generated:', len(candidates))
    print('self-consistency vote:', dict(counts))
    print('selected:', winner, f'({votes}/{len(candidates)})')
    print('increase test-time budget only when expected quality gain justifies cost.')

if __name__ == "__main__":
    print("Chapter 65 lab: Allocate a test-time compute budget to voting")
    run_test_time()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-65.md
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_rl_core.py
تطبيق تعليمي مُبسّط لجوهر آلية التدريب بـ RL لنماذج الاستدلال
على نمط DeepSeek-R1: مكافأة قائمة على قواعد تحقق (Rule-Based Reward)
بدل الاعتماد على Reward Model منفصل عرضة لـ Reward Hacking.

ملاحظة: هذا تبسيط تعليمي لتوضيح المبدأ الرياضي والخوارزمي؛
التطبيق الصناعي الفعلي يتطلب GRPO/PPO كاملين على نطاق موزّع (راجع الفصل 23 و35).
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ReasoningSample:
    """عينة تدريب: سؤال + الإجابة الصحيحة المعروفة (للتحقق الآلي)"""
    question: str
    ground_truth_answer: str


class RuleBasedRewardModel:
    """
    بدل تدريب Reward Model منفصل (عرضة للخداع - Reward Hacking)،
    نستخدم قواعد تحقق صريحة وموضوعية:
    1. مكافأة الدقة (Accuracy Reward): هل الإجابة النهائية صحيحة رياضيًا/منطقيًا؟
    2. مكافأة التنسيق (Format Reward): هل اتّبع النموذج بنية <think>...</think><answer>...</answer>؟
    هذا بالضبط جوهر ما استخدمته DeepSeek-R1-Zero لتجنّب تعقيد Reward Models المُتعلَّمة.
    """

    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def compute_format_reward(self, generated_text: str) -> float:
        """هل التزم النموذج بالبنية الهيكلية المطلوبة؟"""
        has_think = bool(self.THINK_PATTERN.search(generated_text))
        has_answer = bool(self.ANSWER_PATTERN.search(generated_text))
        if has_think and has_answer:
            return 1.0
        elif has_answer:  # على الأقل أعطى إجابة، حتى لو بدون تفكير ظاهر
            return 0.3
        return 0.0

    def compute_accuracy_reward(self, generated_text: str, ground_truth: str) -> float:
        """هل الإجابة المُستخرجة من <answer> مطابقة للحل الصحيح؟"""
        match = self.ANSWER_PATTERN.search(generated_text)
        if not match:
            return 0.0

        extracted_answer = match.group(1).strip()
        # تطبيع بسيط للمقارنة (في التطبيق الفعلي: تحقق رياضي/رمزي أعمق عبر SymPy مثلاً)
        normalized_extracted = re.sub(r"[^\d.\-]", "", extracted_answer)
        normalized_truth = re.sub(r"[^\d.\-]", "", ground_truth)

        return 1.0 if normalized_extracted == normalized_truth else 0.0

    def compute_total_reward(self, generated_text: str, ground_truth: str) -> dict[str, float]:
        format_r = self.compute_format_reward(generated_text)
        accuracy_r = self.compute_accuracy_reward(generated_text, ground_truth)

        # الدقة أهم بكثير من التنسيق — هذا يعكس أولوية R1: النتيجة الصحيحة أولاً
        total = 0.8 * accuracy_r + 0.2 * format_r

        return {
            "format_reward": format_r,
            "accuracy_reward": accuracy_r,
            "total_reward": total,
        }


class GRPOTrainer:
    """
    تطبيق مُبسّط لـ GRPO (Group Relative Policy Optimization) —
    الخوارزمية التي استخدمتها DeepSeek-R1 بدلًا من PPO التقليدي.

    الفكرة الجوهرية لـ GRPO: بدل تدريب Value Model منفصل (كما في PPO)
    لتقدير "ميزة" (Advantage) كل إجابة، نُولّد مجموعة (Group) من الإجابات
    لنفس السؤال، ونحسب الميزة بالمقارنة النسبية داخل المجموعة نفسها:

        A_i = (r_i - mean(r_group)) / std(r_group)

    هذا يُلغي الحاجة الكاملة لـ Value Model منفصل — توفير ضخم في الذاكرة والحوسبة!
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_model: RuleBasedRewardModel,
        group_size: int = 8,
        kl_coefficient: float = 0.04,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.group_size = group_size
        self.kl_coef = kl_coefficient
        self.reference_model = None  # نسخة مجمّدة من النموذج الأصلي لحساب KL Penalty

    def set_reference_model(self, reference_model: AutoModelForCausalLM) -> None:
        """نسخة مرجعية مُجمَّدة (لا تُحدَّث) تمنع النموذج من الانحراف الجذري أثناء RL"""
        self.reference_model = reference_model
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def _build_reasoning_prompt(self, question: str) -> str:
        """نفس تنسيق الـ Prompt المُستخدم في R1-Zero لتشجيع التفكير الصريح"""
        return (
            f"حل المسألة التالية خطوة بخطوة. ضع تفكيرك الكامل داخل "
            f"<think></think> ثم ضع الإجابة النهائية فقط داخل <answer></answer>.\n\n"
            f"المسألة: {question}\n"
        )

    @torch.no_grad()
    def generate_group(self, sample: ReasoningSample) -> list[tuple[str, Tensor]]:
        """
        توليد مجموعة من G إجابات مختلفة لنفس السؤال (عبر temperature sampling)
        هذا بديل GRPO لتوليد "عينات متنوعة للمقارنة النسبية" بدل Value Model
        """
        prompt = self._build_reasoning_prompt(sample.question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,          # مساحة كافية لسلاسل تفكير طويلة
            num_return_sequences=self.group_size,
            do_sample=True,
            temperature=0.8,               # تنوع كافٍ بين أفراد المجموعة
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_texts = [
            self.tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]
        return list(zip(generated_texts, outputs.sequences))

    def compute_group_advantages(
        self, sample: ReasoningSample, generated_texts: list[str]
    ) -> Tensor:
        """
        جوهر GRPO: حساب الميزة النسبية داخل المجموعة بدل استخدام Value Model
        A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
        """
        rewards = torch.tensor([
            self.reward_model.compute_total_reward(text, sample.ground_truth_answer)["total_reward"]
            for text in generated_texts
        ])

        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8  # تجنّب القسمة على صفر

        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def compute_kl_penalty(self, sequences: Tensor, policy_logits: Tensor) -> Tensor:
        """
        عقوبة KL Divergence بين النموذج الحالي والنموذج المرجعي المُجمَّد —
        تمنع "نسيان كارثي" أو انحراف غير آمن أثناء تحسين RL (راجع الفصل 35)
        """
        if self.reference_model is None:
            return torch.tensor(0.0)

        with torch.no_grad():
            ref_logits = self.reference_model(sequences).logits

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        kl_div = F.kl_div(policy_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
        return kl_div

    def training_step(self, sample: ReasoningSample) -> dict[str, float]:
        """خطوة تدريب واحدة كاملة بمنطق GRPO"""
        # 1. توليد مجموعة من الإجابات المتنوعة
        generated_pairs = self.generate_group(sample)
        generated_texts = [text for text, _ in generated_pairs]

        # 2. حساب الميزة النسبية داخل المجموعة (بدل Value Model)
        advantages = self.compute_group_advantages(sample, generated_texts)

        # 3. حساب دالة الخسارة: -log(policy) * advantage + KL penalty
        #    (تبسيط تعليمي؛ GRPO الفعلي يستخدم أيضًا clipping مشابه لـ PPO)
        total_loss = torch.tensor(0.0, requires_grad=True)
        for (text, sequence), advantage in zip(generated_pairs, advantages):
            logits = self.model(sequence.unsqueeze(0)).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # الخسارة السياسية: نُشجّع التسلسلات ذات الميزة الموجبة
            policy_loss = -log_probs.mean() * advantage
            kl_penalty = self.compute_kl_penalty(sequence.unsqueeze(0), logits)

            total_loss = total_loss + policy_loss + self.kl_coef * kl_penalty

        avg_loss = total_loss / len(generated_pairs)
        avg_loss.backward()

        return {
            "loss": avg_loss.item(),
            "mean_reward": advantages.mean().item(),
            "max_reward_in_group": advantages.max().item(),
        }


# ───────────────────────── مثال تطبيقي (مُبسّط) ─────────────────────────
if __name__ == "__main__":
    reward_model = RuleBasedRewardModel()

    # اختبار منطق المكافأة مباشرة قبل أي تدريب فعلي
    good_response = """<think>
    لدي 17 تفاحة. أعطيت 3 أشخاص × 4 تفاحات = 12 تفاحة. تبقى 17-12=5.
    اشتريت ضعف المتبقي = 2×5=10. المجموع النهائي = 5+10=15.
    </think>
    <answer>15</answer>"""

    bad_response_no_format = "الإجابة هي 15 تقريبًا"

    bad_response_wrong_answer = """<think>حسابات خاطئة هنا</think><answer>99</answer>"""

    for name, response in [
        ("إجابة صحيحة ومنسّقة", good_response),
        ("بدون تنسيق صحيح", bad_response_no_format),
        ("منسّقة لكن إجابة خاطئة", bad_response_wrong_answer),
    ]:
        rewards = reward_model.compute_total_reward(response, ground_truth="15")
        print(f"\n{name}:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-65.md
# ────────────────────────────────────────────────────────────────────────
"""
test_time_search.py
تطبيق Best-of-N مع Process Reward Model (PRM) مُبسّط —
بديل/مُكمّل لـ RL الكامل، يُستخدم وقت الاستدلال مباشرة دون الحاجة لإعادة تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class ReasoningStep:
    text: str
    step_score: float   # تقييم PRM لجودة هذه الخطوة تحديدًا


@dataclass
class ReasoningPath:
    steps: list[ReasoningStep]
    final_answer: str

    @property
    def aggregate_score(self) -> float:
        """تجميع تقييمات الخطوات — نستخدم الحد الأدنى (Min) كما في أبحاث PRM
        لأن سلسلة التفكير تنكسر عند أضعف حلقة فيها، وليس بمتوسط كل الخطوات"""
        if not self.steps:
            return 0.0
        return min(step.step_score for step in self.steps)


class ProcessRewardModel:
    """
    نموذج تقييم خطوة بخطوة — يُجيب على سؤال: 'هل هذه الخطوة التفكيرية
    المحددة صحيحة منطقيًا، بناءً على ما سبقها؟' وليس فقط 'هل النتيجة النهائية صحيحة؟'
    هذا أدق بكثير من Reward Model يُقيّم فقط الإجابة الأخيرة (Outcome Reward Model).
    """

    def __init__(self, model_name: str = "prm-scorer-checkpoint") -> None:
        # في الإنتاج الفعلي: نموذج مُدرَّب خصيصًا لتقييم الخطوات
        # (مثال معروف: Math-Shepherd من Microsoft، أو PRM800K من OpenAI)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # بديل توضيحي بسيط
        self.model = None  # Placeholder — نُحاكي السلوك للتبسيط التعليمي

    def score_step(self, previous_steps: list[str], current_step: str) -> float:
        """
        في التطبيق الفعلي: نمرر السياق الكامل (الخطوات السابقة + الحالية)
        لنموذج مُدرَّب على بيانات مُعلَّمة بشريًا (أو ذاتيًا عبر MCTS)
        يُنتج احتمالية أن هذه الخطوة 'صحيحة منطقيًا'.
        هنا نُحاكي heuristically لأغراض الشرح.
        """
        # محاكاة: خطوات تحتوي حسابات رقمية واضحة تحصل على تقييم أعلى
        has_explicit_calculation = any(c.isdigit() for c in current_step)
        has_contradiction_markers = "لكن" in current_step and "خطأ" in current_step
        base_score = 0.85 if has_explicit_calculation else 0.6
        if has_contradiction_markers:
            base_score -= 0.3
        return max(0.0, min(1.0, base_score))


class BestOfNSearcher:
    """
    استراتيجية Test-Time Compute بسيطة وفعالة جدًا عمليًا:
    1. وَلِّد N مسار تفكير كامل ومستقل لنفس السؤال
    2. قيّم كل خطوة في كل مسار عبر PRM
    3. اختر المسار ذا أعلى تقييم إجمالي (وليس فقط أول إجابة تظهر)

    هذا يُجسّد عمليًا مبدأ 'Test-Time Compute Scaling': كلما زاد N،
    زادت دقة النتيجة النهائية — بتكلفة حوسبة إضافية خطية تقريبًا.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prm: ProcessRewardModel,
        n_samples: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.n_samples = n_samples

    def _parse_steps(self, generated_text: str) -> list[str]:
        """تقسيم النص المُولَّد إلى خطوات منفصلة (بافتراض فواصل أسطر أو ترقيم)"""
        raw_steps = [s.strip() for s in generated_text.split("\n") if s.strip()]
        return raw_steps

    def _extract_final_answer(self, generated_text: str) -> str:
        """استخراج الإجابة النهائية من آخر سطر عادة"""
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    @torch.no_grad()
    def search(self, question: str) -> ReasoningPath:
        """البحث الكامل: توليد N مسارات، تقييمها، واختيار الأفضل"""
        prompt = f"حل المسألة التالية خطوة بخطوة:\n{question}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=self.n_samples,
            do_sample=True,
            temperature=0.7,
        )

        candidate_paths: list[ReasoningPath] = []

        for seq in outputs:
            generated_text = self.tokenizer.decode(
                seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            raw_steps = self._parse_steps(generated_text)

            scored_steps: list[ReasoningStep] = []
            previous_context: list[str] = []
            for step_text in raw_steps:
                score = self.prm.score_step(previous_context, step_text)
                scored_steps.append(ReasoningStep(text=step_text, step_score=score))
                previous_context.append(step_text)

            path = ReasoningPath(
                steps=scored_steps,
                final_answer=self._extract_final_answer(generated_text),
            )
            candidate_paths.append(path)

        # 🎯 اختيار المسار الأفضل بناءً على أضعف حلقة فيه (Min Aggregation)
        best_path = max(candidate_paths, key=lambda p: p.aggregate_score)
        return best_path


# ───────────────────────── مثال تطبيقي توضيحي ─────────────────────────
if __name__ == "__main__":
    prm = ProcessRewardModel()

    # محاكاة تقييم مسارين متنافسين دون الحاجة لنموذج فعلي ضخم هنا
    path_good = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("3 أشخاص × 4 = 12 تفاحة أُعطيت", 0.9),
            ReasoningStep("17 - 12 = 5 تفاحات متبقية", 0.95),
            ReasoningStep("اشتريت ضعف 5 = 10", 0.9),
            ReasoningStep("5 + 10 = 15 تفاحة إجمالًا", 0.92),
        ],
        final_answer="15",
    )

    path_flawed = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("أعطيت الكثير من التفاح للناس", 0.4),  # خطوة غامضة بلا حساب دقيق
            ReasoningStep("لذلك الإجابة تقريبًا 20", 0.2),         # خطأ واضح، تقييم منخفض
        ],
        final_answer="20",
    )

    print(f"تقييم المسار الجيد (Min Aggregation): {path_good.aggregate_score:.2f}")
    print(f"تقييم المسار المعيب (Min Aggregation): {path_flawed.aggregate_score:.2f}")
    print(f"\n✅ المسار المُختار: المسار ذو التقييم الأعلى → '{path_good.final_answer}'")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-65.md
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_strategy_benchmark.py
مقارنة تجريبية شاملة بين استراتيجيات الاستدلال المختلفة:
الدقة المحققة مقابل تكلفة الحوسبة (Tokens المُستهلكة) لكل استراتيجية.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReasoningStrategy(Enum):
    DIRECT_ANSWER = "direct"                 # بدون تفكير، إجابة فورية
    CHAIN_OF_THOUGHT = "cot"                 # تفكير واحد خطوة بخطوة
    BEST_OF_N = "best_of_n"                   # عدة مسارات + اختيار الأفضل عبر PRM
    LONG_REASONING_RL = "long_reasoning_rl"   # نموذج مُدرَّب بـ RL على نمط o1/R1


@dataclass
class BenchmarkResult:
    strategy: ReasoningStrategy
    accuracy: float
    avg_tokens_used: int
    avg_latency_seconds: float
    estimated_cost_usd_per_1000_queries: float


class ReasoningBenchmark:
    """
    محاكاة بيانات مرجعية (مبنية على اتجاهات عامة موثقة في الأدبيات البحثية
    مثل ورقة o1 technical report وDeepSeek-R1 paper) لتوضيح العلاقة
    بين استثمار الحوسبة وقت الاستدلال والدقة المحققة.
    """

    # بيانات تقريبية ملهمة من الاتجاهات المنشورة في الأبحاث (AIME, MATH benchmarks)
    SIMULATED_RESULTS: dict[ReasoningStrategy, dict] = {
        ReasoningStrategy.DIRECT_ANSWER: {
            "accuracy": 0.42, "avg_tokens": 50, "avg_latency": 0.8,
        },
        ReasoningStrategy.CHAIN_OF_THOUGHT: {
            "accuracy": 0.58, "avg_tokens": 300, "avg_latency": 2.1,
        },
        ReasoningStrategy.BEST_OF_N: {
            "accuracy": 0.71, "avg_tokens": 300 * 8,  # N=8 مسارات متوازية
            "avg_latency": 3.5,  # يمكن التوازي، فالزمن لا يتضاعف خطيًا بالضرورة
        },
        ReasoningStrategy.LONG_REASONING_RL: {
            "accuracy": 0.83, "avg_tokens": 4500,  # سلاسل تفكير طويلة جدًا (كما في o1/R1)
            "avg_latency": 12.0,
        },
    }

    PRICE_PER_1K_TOKENS_USD = 0.01  # افتراضي توضيحي

    def run_comparison(self) -> list[BenchmarkResult]:
        results = []
        for strategy, data in self.SIMULATED_RESULTS.items():
            cost_per_query = (data["avg_tokens"] / 1000) * self.PRICE_PER_1K_TOKENS_USD
            results.append(BenchmarkResult(
                strategy=strategy,
                accuracy=data["accuracy"],
                avg_tokens_used=data["avg_tokens"],
                avg_latency_seconds=data["avg_latency"],
                estimated_cost_usd_per_1000_queries=round(cost_per_query * 1000, 2),
            ))
        return results

    def recommend_strategy(
        self, accuracy_requirement: float, max_latency_seconds: float, max_cost_per_1k_usd: float
    ) -> ReasoningStrategy | None:
        """
        🎯 القرار الهندسي الحقيقي: بناءً على متطلبات تطبيقك
        (SLA زمني، ميزانية، دقة مطلوبة)، أي استراتيجية تختار؟
        """
        results = self.run_comparison()
        eligible = [
            r for r in results
            if r.accuracy >= accuracy_requirement
            and r.avg_latency_seconds <= max_latency_seconds
            and r.estimated_cost_usd_per_1000_queries <= max_cost_per_1k_usd
        ]
        if not eligible:
            return None
        # من بين المؤهلين، اختر الأرخص (لأنهم جميعًا يحققون المتطلبات بالفعل)
        cheapest = min(eligible, key=lambda r: r.estimated_cost_usd_per_1000_queries)
        return cheapest.strategy


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    benchmark = ReasoningBenchmark()
    results = benchmark.run_comparison()

    print("📊 مقارنة شاملة بين استراتيجيات الاستدلال:\n")
    print(f"{'الاستراتيجية':<25}{'الدقة':<10}{'Tokens':<10}{'الزمن(ث)':<12}{'التكلفة/1000':<15}")
    print("-" * 75)
    for r in results:
        print(
            f"{r.strategy.value:<25}{r.accuracy:<10.2%}{r.avg_tokens_used:<10}"
            f"{r.avg_latency_seconds:<12.1f}${r.estimated_cost_usd_per_1000_queries:<15.2f}"
        )

    print("\n🎯 سيناريو 1: تطبيق دردشة فورية (SLA صارم، دقة متوسطة كافية)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.5, max_latency_seconds=3.0, max_cost_per_1k_usd=10.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

    print("\n🎯 سيناريو 2: حل مسائل أولمبياد رياضية (دقة قصوى، زمن غير حرج)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.8, max_latency_seconds=20.0, max_cost_per_1k_usd=100.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 65: الفصل 65: نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_test_time() -> None:
    candidates = ['A', 'B', 'A', 'A', 'C', 'B', 'A']
    counts = Counter(candidates)
    winner, votes = counts.most_common(1)[0]
    print('candidates generated:', len(candidates))
    print('self-consistency vote:', dict(counts))
    print('selected:', winner, f'({votes}/{len(candidates)})')
    print('increase test-time budget only when expected quality gain justifies cost.')

if __name__ == "__main__":
    print("Chapter 65 lab: Allocate a test-time compute budget to voting")
    run_test_time()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_rl_core.py
تطبيق تعليمي مُبسّط لجوهر آلية التدريب بـ RL لنماذج الاستدلال
على نمط DeepSeek-R1: مكافأة قائمة على قواعد تحقق (Rule-Based Reward)
بدل الاعتماد على Reward Model منفصل عرضة لـ Reward Hacking.

ملاحظة: هذا تبسيط تعليمي لتوضيح المبدأ الرياضي والخوارزمي؛
التطبيق الصناعي الفعلي يتطلب GRPO/PPO كاملين على نطاق موزّع (راجع الفصل 23 و35).
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ReasoningSample:
    """عينة تدريب: سؤال + الإجابة الصحيحة المعروفة (للتحقق الآلي)"""
    question: str
    ground_truth_answer: str


class RuleBasedRewardModel:
    """
    بدل تدريب Reward Model منفصل (عرضة للخداع - Reward Hacking)،
    نستخدم قواعد تحقق صريحة وموضوعية:
    1. مكافأة الدقة (Accuracy Reward): هل الإجابة النهائية صحيحة رياضيًا/منطقيًا؟
    2. مكافأة التنسيق (Format Reward): هل اتّبع النموذج بنية <think>...</think><answer>...</answer>؟
    هذا بالضبط جوهر ما استخدمته DeepSeek-R1-Zero لتجنّب تعقيد Reward Models المُتعلَّمة.
    """

    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def compute_format_reward(self, generated_text: str) -> float:
        """هل التزم النموذج بالبنية الهيكلية المطلوبة؟"""
        has_think = bool(self.THINK_PATTERN.search(generated_text))
        has_answer = bool(self.ANSWER_PATTERN.search(generated_text))
        if has_think and has_answer:
            return 1.0
        elif has_answer:  # على الأقل أعطى إجابة، حتى لو بدون تفكير ظاهر
            return 0.3
        return 0.0

    def compute_accuracy_reward(self, generated_text: str, ground_truth: str) -> float:
        """هل الإجابة المُستخرجة من <answer> مطابقة للحل الصحيح؟"""
        match = self.ANSWER_PATTERN.search(generated_text)
        if not match:
            return 0.0

        extracted_answer = match.group(1).strip()
        # تطبيع بسيط للمقارنة (في التطبيق الفعلي: تحقق رياضي/رمزي أعمق عبر SymPy مثلاً)
        normalized_extracted = re.sub(r"[^\d.\-]", "", extracted_answer)
        normalized_truth = re.sub(r"[^\d.\-]", "", ground_truth)

        return 1.0 if normalized_extracted == normalized_truth else 0.0

    def compute_total_reward(self, generated_text: str, ground_truth: str) -> dict[str, float]:
        format_r = self.compute_format_reward(generated_text)
        accuracy_r = self.compute_accuracy_reward(generated_text, ground_truth)

        # الدقة أهم بكثير من التنسيق — هذا يعكس أولوية R1: النتيجة الصحيحة أولاً
        total = 0.8 * accuracy_r + 0.2 * format_r

        return {
            "format_reward": format_r,
            "accuracy_reward": accuracy_r,
            "total_reward": total,
        }


class GRPOTrainer:
    """
    تطبيق مُبسّط لـ GRPO (Group Relative Policy Optimization) —
    الخوارزمية التي استخدمتها DeepSeek-R1 بدلًا من PPO التقليدي.

    الفكرة الجوهرية لـ GRPO: بدل تدريب Value Model منفصل (كما في PPO)
    لتقدير "ميزة" (Advantage) كل إجابة، نُولّد مجموعة (Group) من الإجابات
    لنفس السؤال، ونحسب الميزة بالمقارنة النسبية داخل المجموعة نفسها:

        A_i = (r_i - mean(r_group)) / std(r_group)

    هذا يُلغي الحاجة الكاملة لـ Value Model منفصل — توفير ضخم في الذاكرة والحوسبة!
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_model: RuleBasedRewardModel,
        group_size: int = 8,
        kl_coefficient: float = 0.04,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.group_size = group_size
        self.kl_coef = kl_coefficient
        self.reference_model = None  # نسخة مجمّدة من النموذج الأصلي لحساب KL Penalty

    def set_reference_model(self, reference_model: AutoModelForCausalLM) -> None:
        """نسخة مرجعية مُجمَّدة (لا تُحدَّث) تمنع النموذج من الانحراف الجذري أثناء RL"""
        self.reference_model = reference_model
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def _build_reasoning_prompt(self, question: str) -> str:
        """نفس تنسيق الـ Prompt المُستخدم في R1-Zero لتشجيع التفكير الصريح"""
        return (
            f"حل المسألة التالية خطوة بخطوة. ضع تفكيرك الكامل داخل "
            f"<think></think> ثم ضع الإجابة النهائية فقط داخل <answer></answer>.\n\n"
            f"المسألة: {question}\n"
        )

    @torch.no_grad()
    def generate_group(self, sample: ReasoningSample) -> list[tuple[str, Tensor]]:
        """
        توليد مجموعة من G إجابات مختلفة لنفس السؤال (عبر temperature sampling)
        هذا بديل GRPO لتوليد "عينات متنوعة للمقارنة النسبية" بدل Value Model
        """
        prompt = self._build_reasoning_prompt(sample.question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,          # مساحة كافية لسلاسل تفكير طويلة
            num_return_sequences=self.group_size,
            do_sample=True,
            temperature=0.8,               # تنوع كافٍ بين أفراد المجموعة
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_texts = [
            self.tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]
        return list(zip(generated_texts, outputs.sequences))

    def compute_group_advantages(
        self, sample: ReasoningSample, generated_texts: list[str]
    ) -> Tensor:
        """
        جوهر GRPO: حساب الميزة النسبية داخل المجموعة بدل استخدام Value Model
        A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
        """
        rewards = torch.tensor([
            self.reward_model.compute_total_reward(text, sample.ground_truth_answer)["total_reward"]
            for text in generated_texts
        ])

        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8  # تجنّب القسمة على صفر

        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def compute_kl_penalty(self, sequences: Tensor, policy_logits: Tensor) -> Tensor:
        """
        عقوبة KL Divergence بين النموذج الحالي والنموذج المرجعي المُجمَّد —
        تمنع "نسيان كارثي" أو انحراف غير آمن أثناء تحسين RL (راجع الفصل 35)
        """
        if self.reference_model is None:
            return torch.tensor(0.0)

        with torch.no_grad():
            ref_logits = self.reference_model(sequences).logits

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        kl_div = F.kl_div(policy_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
        return kl_div

    def training_step(self, sample: ReasoningSample) -> dict[str, float]:
        """خطوة تدريب واحدة كاملة بمنطق GRPO"""
        # 1. توليد مجموعة من الإجابات المتنوعة
        generated_pairs = self.generate_group(sample)
        generated_texts = [text for text, _ in generated_pairs]

        # 2. حساب الميزة النسبية داخل المجموعة (بدل Value Model)
        advantages = self.compute_group_advantages(sample, generated_texts)

        # 3. حساب دالة الخسارة: -log(policy) * advantage + KL penalty
        #    (تبسيط تعليمي؛ GRPO الفعلي يستخدم أيضًا clipping مشابه لـ PPO)
        total_loss = torch.tensor(0.0, requires_grad=True)
        for (text, sequence), advantage in zip(generated_pairs, advantages):
            logits = self.model(sequence.unsqueeze(0)).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # الخسارة السياسية: نُشجّع التسلسلات ذات الميزة الموجبة
            policy_loss = -log_probs.mean() * advantage
            kl_penalty = self.compute_kl_penalty(sequence.unsqueeze(0), logits)

            total_loss = total_loss + policy_loss + self.kl_coef * kl_penalty

        avg_loss = total_loss / len(generated_pairs)
        avg_loss.backward()

        return {
            "loss": avg_loss.item(),
            "mean_reward": advantages.mean().item(),
            "max_reward_in_group": advantages.max().item(),
        }


# ───────────────────────── مثال تطبيقي (مُبسّط) ─────────────────────────
if __name__ == "__main__":
    reward_model = RuleBasedRewardModel()

    # اختبار منطق المكافأة مباشرة قبل أي تدريب فعلي
    good_response = """<think>
    لدي 17 تفاحة. أعطيت 3 أشخاص × 4 تفاحات = 12 تفاحة. تبقى 17-12=5.
    اشتريت ضعف المتبقي = 2×5=10. المجموع النهائي = 5+10=15.
    </think>
    <answer>15</answer>"""

    bad_response_no_format = "الإجابة هي 15 تقريبًا"

    bad_response_wrong_answer = """<think>حسابات خاطئة هنا</think><answer>99</answer>"""

    for name, response in [
        ("إجابة صحيحة ومنسّقة", good_response),
        ("بدون تنسيق صحيح", bad_response_no_format),
        ("منسّقة لكن إجابة خاطئة", bad_response_wrong_answer),
    ]:
        rewards = reward_model.compute_total_reward(response, ground_truth="15")
        print(f"\n{name}:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
test_time_search.py
تطبيق Best-of-N مع Process Reward Model (PRM) مُبسّط —
بديل/مُكمّل لـ RL الكامل، يُستخدم وقت الاستدلال مباشرة دون الحاجة لإعادة تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class ReasoningStep:
    text: str
    step_score: float   # تقييم PRM لجودة هذه الخطوة تحديدًا


@dataclass
class ReasoningPath:
    steps: list[ReasoningStep]
    final_answer: str

    @property
    def aggregate_score(self) -> float:
        """تجميع تقييمات الخطوات — نستخدم الحد الأدنى (Min) كما في أبحاث PRM
        لأن سلسلة التفكير تنكسر عند أضعف حلقة فيها، وليس بمتوسط كل الخطوات"""
        if not self.steps:
            return 0.0
        return min(step.step_score for step in self.steps)


class ProcessRewardModel:
    """
    نموذج تقييم خطوة بخطوة — يُجيب على سؤال: 'هل هذه الخطوة التفكيرية
    المحددة صحيحة منطقيًا، بناءً على ما سبقها؟' وليس فقط 'هل النتيجة النهائية صحيحة؟'
    هذا أدق بكثير من Reward Model يُقيّم فقط الإجابة الأخيرة (Outcome Reward Model).
    """

    def __init__(self, model_name: str = "prm-scorer-checkpoint") -> None:
        # في الإنتاج الفعلي: نموذج مُدرَّب خصيصًا لتقييم الخطوات
        # (مثال معروف: Math-Shepherd من Microsoft، أو PRM800K من OpenAI)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # بديل توضيحي بسيط
        self.model = None  # Placeholder — نُحاكي السلوك للتبسيط التعليمي

    def score_step(self, previous_steps: list[str], current_step: str) -> float:
        """
        في التطبيق الفعلي: نمرر السياق الكامل (الخطوات السابقة + الحالية)
        لنموذج مُدرَّب على بيانات مُعلَّمة بشريًا (أو ذاتيًا عبر MCTS)
        يُنتج احتمالية أن هذه الخطوة 'صحيحة منطقيًا'.
        هنا نُحاكي heuristically لأغراض الشرح.
        """
        # محاكاة: خطوات تحتوي حسابات رقمية واضحة تحصل على تقييم أعلى
        has_explicit_calculation = any(c.isdigit() for c in current_step)
        has_contradiction_markers = "لكن" in current_step and "خطأ" in current_step
        base_score = 0.85 if has_explicit_calculation else 0.6
        if has_contradiction_markers:
            base_score -= 0.3
        return max(0.0, min(1.0, base_score))


class BestOfNSearcher:
    """
    استراتيجية Test-Time Compute بسيطة وفعالة جدًا عمليًا:
    1. وَلِّد N مسار تفكير كامل ومستقل لنفس السؤال
    2. قيّم كل خطوة في كل مسار عبر PRM
    3. اختر المسار ذا أعلى تقييم إجمالي (وليس فقط أول إجابة تظهر)

    هذا يُجسّد عمليًا مبدأ 'Test-Time Compute Scaling': كلما زاد N،
    زادت دقة النتيجة النهائية — بتكلفة حوسبة إضافية خطية تقريبًا.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prm: ProcessRewardModel,
        n_samples: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.n_samples = n_samples

    def _parse_steps(self, generated_text: str) -> list[str]:
        """تقسيم النص المُولَّد إلى خطوات منفصلة (بافتراض فواصل أسطر أو ترقيم)"""
        raw_steps = [s.strip() for s in generated_text.split("\n") if s.strip()]
        return raw_steps

    def _extract_final_answer(self, generated_text: str) -> str:
        """استخراج الإجابة النهائية من آخر سطر عادة"""
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    @torch.no_grad()
    def search(self, question: str) -> ReasoningPath:
        """البحث الكامل: توليد N مسارات، تقييمها، واختيار الأفضل"""
        prompt = f"حل المسألة التالية خطوة بخطوة:\n{question}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=self.n_samples,
            do_sample=True,
            temperature=0.7,
        )

        candidate_paths: list[ReasoningPath] = []

        for seq in outputs:
            generated_text = self.tokenizer.decode(
                seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            raw_steps = self._parse_steps(generated_text)

            scored_steps: list[ReasoningStep] = []
            previous_context: list[str] = []
            for step_text in raw_steps:
                score = self.prm.score_step(previous_context, step_text)
                scored_steps.append(ReasoningStep(text=step_text, step_score=score))
                previous_context.append(step_text)

            path = ReasoningPath(
                steps=scored_steps,
                final_answer=self._extract_final_answer(generated_text),
            )
            candidate_paths.append(path)

        # 🎯 اختيار المسار الأفضل بناءً على أضعف حلقة فيه (Min Aggregation)
        best_path = max(candidate_paths, key=lambda p: p.aggregate_score)
        return best_path


# ───────────────────────── مثال تطبيقي توضيحي ─────────────────────────
if __name__ == "__main__":
    prm = ProcessRewardModel()

    # محاكاة تقييم مسارين متنافسين دون الحاجة لنموذج فعلي ضخم هنا
    path_good = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("3 أشخاص × 4 = 12 تفاحة أُعطيت", 0.9),
            ReasoningStep("17 - 12 = 5 تفاحات متبقية", 0.95),
            ReasoningStep("اشتريت ضعف 5 = 10", 0.9),
            ReasoningStep("5 + 10 = 15 تفاحة إجمالًا", 0.92),
        ],
        final_answer="15",
    )

    path_flawed = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("أعطيت الكثير من التفاح للناس", 0.4),  # خطوة غامضة بلا حساب دقيق
            ReasoningStep("لذلك الإجابة تقريبًا 20", 0.2),         # خطأ واضح، تقييم منخفض
        ],
        final_answer="20",
    )

    print(f"تقييم المسار الجيد (Min Aggregation): {path_good.aggregate_score:.2f}")
    print(f"تقييم المسار المعيب (Min Aggregation): {path_flawed.aggregate_score:.2f}")
    print(f"\n✅ المسار المُختار: المسار ذو التقييم الأعلى → '{path_good.final_answer}'")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_strategy_benchmark.py
مقارنة تجريبية شاملة بين استراتيجيات الاستدلال المختلفة:
الدقة المحققة مقابل تكلفة الحوسبة (Tokens المُستهلكة) لكل استراتيجية.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReasoningStrategy(Enum):
    DIRECT_ANSWER = "direct"                 # بدون تفكير، إجابة فورية
    CHAIN_OF_THOUGHT = "cot"                 # تفكير واحد خطوة بخطوة
    BEST_OF_N = "best_of_n"                   # عدة مسارات + اختيار الأفضل عبر PRM
    LONG_REASONING_RL = "long_reasoning_rl"   # نموذج مُدرَّب بـ RL على نمط o1/R1


@dataclass
class BenchmarkResult:
    strategy: ReasoningStrategy
    accuracy: float
    avg_tokens_used: int
    avg_latency_seconds: float
    estimated_cost_usd_per_1000_queries: float


class ReasoningBenchmark:
    """
    محاكاة بيانات مرجعية (مبنية على اتجاهات عامة موثقة في الأدبيات البحثية
    مثل ورقة o1 technical report وDeepSeek-R1 paper) لتوضيح العلاقة
    بين استثمار الحوسبة وقت الاستدلال والدقة المحققة.
    """

    # بيانات تقريبية ملهمة من الاتجاهات المنشورة في الأبحاث (AIME, MATH benchmarks)
    SIMULATED_RESULTS: dict[ReasoningStrategy, dict] = {
        ReasoningStrategy.DIRECT_ANSWER: {
            "accuracy": 0.42, "avg_tokens": 50, "avg_latency": 0.8,
        },
        ReasoningStrategy.CHAIN_OF_THOUGHT: {
            "accuracy": 0.58, "avg_tokens": 300, "avg_latency": 2.1,
        },
        ReasoningStrategy.BEST_OF_N: {
            "accuracy": 0.71, "avg_tokens": 300 * 8,  # N=8 مسارات متوازية
            "avg_latency": 3.5,  # يمكن التوازي، فالزمن لا يتضاعف خطيًا بالضرورة
        },
        ReasoningStrategy.LONG_REASONING_RL: {
            "accuracy": 0.83, "avg_tokens": 4500,  # سلاسل تفكير طويلة جدًا (كما في o1/R1)
            "avg_latency": 12.0,
        },
    }

    PRICE_PER_1K_TOKENS_USD = 0.01  # افتراضي توضيحي

    def run_comparison(self) -> list[BenchmarkResult]:
        results = []
        for strategy, data in self.SIMULATED_RESULTS.items():
            cost_per_query = (data["avg_tokens"] / 1000) * self.PRICE_PER_1K_TOKENS_USD
            results.append(BenchmarkResult(
                strategy=strategy,
                accuracy=data["accuracy"],
                avg_tokens_used=data["avg_tokens"],
                avg_latency_seconds=data["avg_latency"],
                estimated_cost_usd_per_1000_queries=round(cost_per_query * 1000, 2),
            ))
        return results

    def recommend_strategy(
        self, accuracy_requirement: float, max_latency_seconds: float, max_cost_per_1k_usd: float
    ) -> ReasoningStrategy | None:
        """
        🎯 القرار الهندسي الحقيقي: بناءً على متطلبات تطبيقك
        (SLA زمني، ميزانية، دقة مطلوبة)، أي استراتيجية تختار؟
        """
        results = self.run_comparison()
        eligible = [
            r for r in results
            if r.accuracy >= accuracy_requirement
            and r.avg_latency_seconds <= max_latency_seconds
            and r.estimated_cost_usd_per_1000_queries <= max_cost_per_1k_usd
        ]
        if not eligible:
            return None
        # من بين المؤهلين، اختر الأرخص (لأنهم جميعًا يحققون المتطلبات بالفعل)
        cheapest = min(eligible, key=lambda r: r.estimated_cost_usd_per_1000_queries)
        return cheapest.strategy


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    benchmark = ReasoningBenchmark()
    results = benchmark.run_comparison()

    print("📊 مقارنة شاملة بين استراتيجيات الاستدلال:\n")
    print(f"{'الاستراتيجية':<25}{'الدقة':<10}{'Tokens':<10}{'الزمن(ث)':<12}{'التكلفة/1000':<15}")
    print("-" * 75)
    for r in results:
        print(
            f"{r.strategy.value:<25}{r.accuracy:<10.2%}{r.avg_tokens_used:<10}"
            f"{r.avg_latency_seconds:<12.1f}${r.estimated_cost_usd_per_1000_queries:<15.2f}"
        )

    print("\n🎯 سيناريو 1: تطبيق دردشة فورية (SLA صارم، دقة متوسطة كافية)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.5, max_latency_seconds=3.0, max_cost_per_1k_usd=10.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

    print("\n🎯 سيناريو 2: حل مسائل أولمبياد رياضية (دقة قصوى، زمن غير حرج)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.8, max_latency_seconds=20.0, max_cost_per_1k_usd=100.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"
Large Language Model Engineering — Test-Time

Book slug : llm-engineering-69-chapters
Chapter   : ch-65-chapter-65 (lesson 65)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_65.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 65: الفصل 65: نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_test_time() -> None:
    candidates = ['A', 'B', 'A', 'A', 'C', 'B', 'A']
    counts = Counter(candidates)
    winner, votes = counts.most_common(1)[0]
    print('candidates generated:', len(candidates))
    print('self-consistency vote:', dict(counts))
    print('selected:', winner, f'({votes}/{len(candidates)})')
    print('increase test-time budget only when expected quality gain justifies cost.')

if __name__ == "__main__":
    print("Chapter 65 lab: Allocate a test-time compute budget to voting")
    run_test_time()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-65.md
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_rl_core.py
تطبيق تعليمي مُبسّط لجوهر آلية التدريب بـ RL لنماذج الاستدلال
على نمط DeepSeek-R1: مكافأة قائمة على قواعد تحقق (Rule-Based Reward)
بدل الاعتماد على Reward Model منفصل عرضة لـ Reward Hacking.

ملاحظة: هذا تبسيط تعليمي لتوضيح المبدأ الرياضي والخوارزمي؛
التطبيق الصناعي الفعلي يتطلب GRPO/PPO كاملين على نطاق موزّع (راجع الفصل 23 و35).
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ReasoningSample:
    """عينة تدريب: سؤال + الإجابة الصحيحة المعروفة (للتحقق الآلي)"""
    question: str
    ground_truth_answer: str


class RuleBasedRewardModel:
    """
    بدل تدريب Reward Model منفصل (عرضة للخداع - Reward Hacking)،
    نستخدم قواعد تحقق صريحة وموضوعية:
    1. مكافأة الدقة (Accuracy Reward): هل الإجابة النهائية صحيحة رياضيًا/منطقيًا؟
    2. مكافأة التنسيق (Format Reward): هل اتّبع النموذج بنية <think>...</think><answer>...</answer>؟
    هذا بالضبط جوهر ما استخدمته DeepSeek-R1-Zero لتجنّب تعقيد Reward Models المُتعلَّمة.
    """

    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def compute_format_reward(self, generated_text: str) -> float:
        """هل التزم النموذج بالبنية الهيكلية المطلوبة؟"""
        has_think = bool(self.THINK_PATTERN.search(generated_text))
        has_answer = bool(self.ANSWER_PATTERN.search(generated_text))
        if has_think and has_answer:
            return 1.0
        elif has_answer:  # على الأقل أعطى إجابة، حتى لو بدون تفكير ظاهر
            return 0.3
        return 0.0

    def compute_accuracy_reward(self, generated_text: str, ground_truth: str) -> float:
        """هل الإجابة المُستخرجة من <answer> مطابقة للحل الصحيح؟"""
        match = self.ANSWER_PATTERN.search(generated_text)
        if not match:
            return 0.0

        extracted_answer = match.group(1).strip()
        # تطبيع بسيط للمقارنة (في التطبيق الفعلي: تحقق رياضي/رمزي أعمق عبر SymPy مثلاً)
        normalized_extracted = re.sub(r"[^\d.\-]", "", extracted_answer)
        normalized_truth = re.sub(r"[^\d.\-]", "", ground_truth)

        return 1.0 if normalized_extracted == normalized_truth else 0.0

    def compute_total_reward(self, generated_text: str, ground_truth: str) -> dict[str, float]:
        format_r = self.compute_format_reward(generated_text)
        accuracy_r = self.compute_accuracy_reward(generated_text, ground_truth)

        # الدقة أهم بكثير من التنسيق — هذا يعكس أولوية R1: النتيجة الصحيحة أولاً
        total = 0.8 * accuracy_r + 0.2 * format_r

        return {
            "format_reward": format_r,
            "accuracy_reward": accuracy_r,
            "total_reward": total,
        }


class GRPOTrainer:
    """
    تطبيق مُبسّط لـ GRPO (Group Relative Policy Optimization) —
    الخوارزمية التي استخدمتها DeepSeek-R1 بدلًا من PPO التقليدي.

    الفكرة الجوهرية لـ GRPO: بدل تدريب Value Model منفصل (كما في PPO)
    لتقدير "ميزة" (Advantage) كل إجابة، نُولّد مجموعة (Group) من الإجابات
    لنفس السؤال، ونحسب الميزة بالمقارنة النسبية داخل المجموعة نفسها:

        A_i = (r_i - mean(r_group)) / std(r_group)

    هذا يُلغي الحاجة الكاملة لـ Value Model منفصل — توفير ضخم في الذاكرة والحوسبة!
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_model: RuleBasedRewardModel,
        group_size: int = 8,
        kl_coefficient: float = 0.04,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.group_size = group_size
        self.kl_coef = kl_coefficient
        self.reference_model = None  # نسخة مجمّدة من النموذج الأصلي لحساب KL Penalty

    def set_reference_model(self, reference_model: AutoModelForCausalLM) -> None:
        """نسخة مرجعية مُجمَّدة (لا تُحدَّث) تمنع النموذج من الانحراف الجذري أثناء RL"""
        self.reference_model = reference_model
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def _build_reasoning_prompt(self, question: str) -> str:
        """نفس تنسيق الـ Prompt المُستخدم في R1-Zero لتشجيع التفكير الصريح"""
        return (
            f"حل المسألة التالية خطوة بخطوة. ضع تفكيرك الكامل داخل "
            f"<think></think> ثم ضع الإجابة النهائية فقط داخل <answer></answer>.\n\n"
            f"المسألة: {question}\n"
        )

    @torch.no_grad()
    def generate_group(self, sample: ReasoningSample) -> list[tuple[str, Tensor]]:
        """
        توليد مجموعة من G إجابات مختلفة لنفس السؤال (عبر temperature sampling)
        هذا بديل GRPO لتوليد "عينات متنوعة للمقارنة النسبية" بدل Value Model
        """
        prompt = self._build_reasoning_prompt(sample.question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,          # مساحة كافية لسلاسل تفكير طويلة
            num_return_sequences=self.group_size,
            do_sample=True,
            temperature=0.8,               # تنوع كافٍ بين أفراد المجموعة
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_texts = [
            self.tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]
        return list(zip(generated_texts, outputs.sequences))

    def compute_group_advantages(
        self, sample: ReasoningSample, generated_texts: list[str]
    ) -> Tensor:
        """
        جوهر GRPO: حساب الميزة النسبية داخل المجموعة بدل استخدام Value Model
        A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
        """
        rewards = torch.tensor([
            self.reward_model.compute_total_reward(text, sample.ground_truth_answer)["total_reward"]
            for text in generated_texts
        ])

        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8  # تجنّب القسمة على صفر

        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def compute_kl_penalty(self, sequences: Tensor, policy_logits: Tensor) -> Tensor:
        """
        عقوبة KL Divergence بين النموذج الحالي والنموذج المرجعي المُجمَّد —
        تمنع "نسيان كارثي" أو انحراف غير آمن أثناء تحسين RL (راجع الفصل 35)
        """
        if self.reference_model is None:
            return torch.tensor(0.0)

        with torch.no_grad():
            ref_logits = self.reference_model(sequences).logits

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        kl_div = F.kl_div(policy_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
        return kl_div

    def training_step(self, sample: ReasoningSample) -> dict[str, float]:
        """خطوة تدريب واحدة كاملة بمنطق GRPO"""
        # 1. توليد مجموعة من الإجابات المتنوعة
        generated_pairs = self.generate_group(sample)
        generated_texts = [text for text, _ in generated_pairs]

        # 2. حساب الميزة النسبية داخل المجموعة (بدل Value Model)
        advantages = self.compute_group_advantages(sample, generated_texts)

        # 3. حساب دالة الخسارة: -log(policy) * advantage + KL penalty
        #    (تبسيط تعليمي؛ GRPO الفعلي يستخدم أيضًا clipping مشابه لـ PPO)
        total_loss = torch.tensor(0.0, requires_grad=True)
        for (text, sequence), advantage in zip(generated_pairs, advantages):
            logits = self.model(sequence.unsqueeze(0)).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # الخسارة السياسية: نُشجّع التسلسلات ذات الميزة الموجبة
            policy_loss = -log_probs.mean() * advantage
            kl_penalty = self.compute_kl_penalty(sequence.unsqueeze(0), logits)

            total_loss = total_loss + policy_loss + self.kl_coef * kl_penalty

        avg_loss = total_loss / len(generated_pairs)
        avg_loss.backward()

        return {
            "loss": avg_loss.item(),
            "mean_reward": advantages.mean().item(),
            "max_reward_in_group": advantages.max().item(),
        }


# ───────────────────────── مثال تطبيقي (مُبسّط) ─────────────────────────
if __name__ == "__main__":
    reward_model = RuleBasedRewardModel()

    # اختبار منطق المكافأة مباشرة قبل أي تدريب فعلي
    good_response = """<think>
    لدي 17 تفاحة. أعطيت 3 أشخاص × 4 تفاحات = 12 تفاحة. تبقى 17-12=5.
    اشتريت ضعف المتبقي = 2×5=10. المجموع النهائي = 5+10=15.
    </think>
    <answer>15</answer>"""

    bad_response_no_format = "الإجابة هي 15 تقريبًا"

    bad_response_wrong_answer = """<think>حسابات خاطئة هنا</think><answer>99</answer>"""

    for name, response in [
        ("إجابة صحيحة ومنسّقة", good_response),
        ("بدون تنسيق صحيح", bad_response_no_format),
        ("منسّقة لكن إجابة خاطئة", bad_response_wrong_answer),
    ]:
        rewards = reward_model.compute_total_reward(response, ground_truth="15")
        print(f"\n{name}:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-65.md
# ────────────────────────────────────────────────────────────────────────
"""
test_time_search.py
تطبيق Best-of-N مع Process Reward Model (PRM) مُبسّط —
بديل/مُكمّل لـ RL الكامل، يُستخدم وقت الاستدلال مباشرة دون الحاجة لإعادة تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class ReasoningStep:
    text: str
    step_score: float   # تقييم PRM لجودة هذه الخطوة تحديدًا


@dataclass
class ReasoningPath:
    steps: list[ReasoningStep]
    final_answer: str

    @property
    def aggregate_score(self) -> float:
        """تجميع تقييمات الخطوات — نستخدم الحد الأدنى (Min) كما في أبحاث PRM
        لأن سلسلة التفكير تنكسر عند أضعف حلقة فيها، وليس بمتوسط كل الخطوات"""
        if not self.steps:
            return 0.0
        return min(step.step_score for step in self.steps)


class ProcessRewardModel:
    """
    نموذج تقييم خطوة بخطوة — يُجيب على سؤال: 'هل هذه الخطوة التفكيرية
    المحددة صحيحة منطقيًا، بناءً على ما سبقها؟' وليس فقط 'هل النتيجة النهائية صحيحة؟'
    هذا أدق بكثير من Reward Model يُقيّم فقط الإجابة الأخيرة (Outcome Reward Model).
    """

    def __init__(self, model_name: str = "prm-scorer-checkpoint") -> None:
        # في الإنتاج الفعلي: نموذج مُدرَّب خصيصًا لتقييم الخطوات
        # (مثال معروف: Math-Shepherd من Microsoft، أو PRM800K من OpenAI)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # بديل توضيحي بسيط
        self.model = None  # Placeholder — نُحاكي السلوك للتبسيط التعليمي

    def score_step(self, previous_steps: list[str], current_step: str) -> float:
        """
        في التطبيق الفعلي: نمرر السياق الكامل (الخطوات السابقة + الحالية)
        لنموذج مُدرَّب على بيانات مُعلَّمة بشريًا (أو ذاتيًا عبر MCTS)
        يُنتج احتمالية أن هذه الخطوة 'صحيحة منطقيًا'.
        هنا نُحاكي heuristically لأغراض الشرح.
        """
        # محاكاة: خطوات تحتوي حسابات رقمية واضحة تحصل على تقييم أعلى
        has_explicit_calculation = any(c.isdigit() for c in current_step)
        has_contradiction_markers = "لكن" in current_step and "خطأ" in current_step
        base_score = 0.85 if has_explicit_calculation else 0.6
        if has_contradiction_markers:
            base_score -= 0.3
        return max(0.0, min(1.0, base_score))


class BestOfNSearcher:
    """
    استراتيجية Test-Time Compute بسيطة وفعالة جدًا عمليًا:
    1. وَلِّد N مسار تفكير كامل ومستقل لنفس السؤال
    2. قيّم كل خطوة في كل مسار عبر PRM
    3. اختر المسار ذا أعلى تقييم إجمالي (وليس فقط أول إجابة تظهر)

    هذا يُجسّد عمليًا مبدأ 'Test-Time Compute Scaling': كلما زاد N،
    زادت دقة النتيجة النهائية — بتكلفة حوسبة إضافية خطية تقريبًا.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prm: ProcessRewardModel,
        n_samples: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.n_samples = n_samples

    def _parse_steps(self, generated_text: str) -> list[str]:
        """تقسيم النص المُولَّد إلى خطوات منفصلة (بافتراض فواصل أسطر أو ترقيم)"""
        raw_steps = [s.strip() for s in generated_text.split("\n") if s.strip()]
        return raw_steps

    def _extract_final_answer(self, generated_text: str) -> str:
        """استخراج الإجابة النهائية من آخر سطر عادة"""
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    @torch.no_grad()
    def search(self, question: str) -> ReasoningPath:
        """البحث الكامل: توليد N مسارات، تقييمها، واختيار الأفضل"""
        prompt = f"حل المسألة التالية خطوة بخطوة:\n{question}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=self.n_samples,
            do_sample=True,
            temperature=0.7,
        )

        candidate_paths: list[ReasoningPath] = []

        for seq in outputs:
            generated_text = self.tokenizer.decode(
                seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            raw_steps = self._parse_steps(generated_text)

            scored_steps: list[ReasoningStep] = []
            previous_context: list[str] = []
            for step_text in raw_steps:
                score = self.prm.score_step(previous_context, step_text)
                scored_steps.append(ReasoningStep(text=step_text, step_score=score))
                previous_context.append(step_text)

            path = ReasoningPath(
                steps=scored_steps,
                final_answer=self._extract_final_answer(generated_text),
            )
            candidate_paths.append(path)

        # 🎯 اختيار المسار الأفضل بناءً على أضعف حلقة فيه (Min Aggregation)
        best_path = max(candidate_paths, key=lambda p: p.aggregate_score)
        return best_path


# ───────────────────────── مثال تطبيقي توضيحي ─────────────────────────
if __name__ == "__main__":
    prm = ProcessRewardModel()

    # محاكاة تقييم مسارين متنافسين دون الحاجة لنموذج فعلي ضخم هنا
    path_good = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("3 أشخاص × 4 = 12 تفاحة أُعطيت", 0.9),
            ReasoningStep("17 - 12 = 5 تفاحات متبقية", 0.95),
            ReasoningStep("اشتريت ضعف 5 = 10", 0.9),
            ReasoningStep("5 + 10 = 15 تفاحة إجمالًا", 0.92),
        ],
        final_answer="15",
    )

    path_flawed = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("أعطيت الكثير من التفاح للناس", 0.4),  # خطوة غامضة بلا حساب دقيق
            ReasoningStep("لذلك الإجابة تقريبًا 20", 0.2),         # خطأ واضح، تقييم منخفض
        ],
        final_answer="20",
    )

    print(f"تقييم المسار الجيد (Min Aggregation): {path_good.aggregate_score:.2f}")
    print(f"تقييم المسار المعيب (Min Aggregation): {path_flawed.aggregate_score:.2f}")
    print(f"\n✅ المسار المُختار: المسار ذو التقييم الأعلى → '{path_good.final_answer}'")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-65.md
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_strategy_benchmark.py
مقارنة تجريبية شاملة بين استراتيجيات الاستدلال المختلفة:
الدقة المحققة مقابل تكلفة الحوسبة (Tokens المُستهلكة) لكل استراتيجية.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReasoningStrategy(Enum):
    DIRECT_ANSWER = "direct"                 # بدون تفكير، إجابة فورية
    CHAIN_OF_THOUGHT = "cot"                 # تفكير واحد خطوة بخطوة
    BEST_OF_N = "best_of_n"                   # عدة مسارات + اختيار الأفضل عبر PRM
    LONG_REASONING_RL = "long_reasoning_rl"   # نموذج مُدرَّب بـ RL على نمط o1/R1


@dataclass
class BenchmarkResult:
    strategy: ReasoningStrategy
    accuracy: float
    avg_tokens_used: int
    avg_latency_seconds: float
    estimated_cost_usd_per_1000_queries: float


class ReasoningBenchmark:
    """
    محاكاة بيانات مرجعية (مبنية على اتجاهات عامة موثقة في الأدبيات البحثية
    مثل ورقة o1 technical report وDeepSeek-R1 paper) لتوضيح العلاقة
    بين استثمار الحوسبة وقت الاستدلال والدقة المحققة.
    """

    # بيانات تقريبية ملهمة من الاتجاهات المنشورة في الأبحاث (AIME, MATH benchmarks)
    SIMULATED_RESULTS: dict[ReasoningStrategy, dict] = {
        ReasoningStrategy.DIRECT_ANSWER: {
            "accuracy": 0.42, "avg_tokens": 50, "avg_latency": 0.8,
        },
        ReasoningStrategy.CHAIN_OF_THOUGHT: {
            "accuracy": 0.58, "avg_tokens": 300, "avg_latency": 2.1,
        },
        ReasoningStrategy.BEST_OF_N: {
            "accuracy": 0.71, "avg_tokens": 300 * 8,  # N=8 مسارات متوازية
            "avg_latency": 3.5,  # يمكن التوازي، فالزمن لا يتضاعف خطيًا بالضرورة
        },
        ReasoningStrategy.LONG_REASONING_RL: {
            "accuracy": 0.83, "avg_tokens": 4500,  # سلاسل تفكير طويلة جدًا (كما في o1/R1)
            "avg_latency": 12.0,
        },
    }

    PRICE_PER_1K_TOKENS_USD = 0.01  # افتراضي توضيحي

    def run_comparison(self) -> list[BenchmarkResult]:
        results = []
        for strategy, data in self.SIMULATED_RESULTS.items():
            cost_per_query = (data["avg_tokens"] / 1000) * self.PRICE_PER_1K_TOKENS_USD
            results.append(BenchmarkResult(
                strategy=strategy,
                accuracy=data["accuracy"],
                avg_tokens_used=data["avg_tokens"],
                avg_latency_seconds=data["avg_latency"],
                estimated_cost_usd_per_1000_queries=round(cost_per_query * 1000, 2),
            ))
        return results

    def recommend_strategy(
        self, accuracy_requirement: float, max_latency_seconds: float, max_cost_per_1k_usd: float
    ) -> ReasoningStrategy | None:
        """
        🎯 القرار الهندسي الحقيقي: بناءً على متطلبات تطبيقك
        (SLA زمني، ميزانية، دقة مطلوبة)، أي استراتيجية تختار؟
        """
        results = self.run_comparison()
        eligible = [
            r for r in results
            if r.accuracy >= accuracy_requirement
            and r.avg_latency_seconds <= max_latency_seconds
            and r.estimated_cost_usd_per_1000_queries <= max_cost_per_1k_usd
        ]
        if not eligible:
            return None
        # من بين المؤهلين، اختر الأرخص (لأنهم جميعًا يحققون المتطلبات بالفعل)
        cheapest = min(eligible, key=lambda r: r.estimated_cost_usd_per_1000_queries)
        return cheapest.strategy


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    benchmark = ReasoningBenchmark()
    results = benchmark.run_comparison()

    print("📊 مقارنة شاملة بين استراتيجيات الاستدلال:\n")
    print(f"{'الاستراتيجية':<25}{'الدقة':<10}{'Tokens':<10}{'الزمن(ث)':<12}{'التكلفة/1000':<15}")
    print("-" * 75)
    for r in results:
        print(
            f"{r.strategy.value:<25}{r.accuracy:<10.2%}{r.avg_tokens_used:<10}"
            f"{r.avg_latency_seconds:<12.1f}${r.estimated_cost_usd_per_1000_queries:<15.2f}"
        )

    print("\n🎯 سيناريو 1: تطبيق دردشة فورية (SLA صارم، دقة متوسطة كافية)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.5, max_latency_seconds=3.0, max_cost_per_1k_usd=10.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

    print("\n🎯 سيناريو 2: حل مسائل أولمبياد رياضية (دقة قصوى، زمن غير حرج)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.8, max_latency_seconds=20.0, max_cost_per_1k_usd=100.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 65: الفصل 65: نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_test_time() -> None:
    candidates = ['A', 'B', 'A', 'A', 'C', 'B', 'A']
    counts = Counter(candidates)
    winner, votes = counts.most_common(1)[0]
    print('candidates generated:', len(candidates))
    print('self-consistency vote:', dict(counts))
    print('selected:', winner, f'({votes}/{len(candidates)})')
    print('increase test-time budget only when expected quality gain justifies cost.')

if __name__ == "__main__":
    print("Chapter 65 lab: Allocate a test-time compute budget to voting")
    run_test_time()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_rl_core.py
تطبيق تعليمي مُبسّط لجوهر آلية التدريب بـ RL لنماذج الاستدلال
على نمط DeepSeek-R1: مكافأة قائمة على قواعد تحقق (Rule-Based Reward)
بدل الاعتماد على Reward Model منفصل عرضة لـ Reward Hacking.

ملاحظة: هذا تبسيط تعليمي لتوضيح المبدأ الرياضي والخوارزمي؛
التطبيق الصناعي الفعلي يتطلب GRPO/PPO كاملين على نطاق موزّع (راجع الفصل 23 و35).
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ReasoningSample:
    """عينة تدريب: سؤال + الإجابة الصحيحة المعروفة (للتحقق الآلي)"""
    question: str
    ground_truth_answer: str


class RuleBasedRewardModel:
    """
    بدل تدريب Reward Model منفصل (عرضة للخداع - Reward Hacking)،
    نستخدم قواعد تحقق صريحة وموضوعية:
    1. مكافأة الدقة (Accuracy Reward): هل الإجابة النهائية صحيحة رياضيًا/منطقيًا؟
    2. مكافأة التنسيق (Format Reward): هل اتّبع النموذج بنية <think>...</think><answer>...</answer>؟
    هذا بالضبط جوهر ما استخدمته DeepSeek-R1-Zero لتجنّب تعقيد Reward Models المُتعلَّمة.
    """

    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def compute_format_reward(self, generated_text: str) -> float:
        """هل التزم النموذج بالبنية الهيكلية المطلوبة؟"""
        has_think = bool(self.THINK_PATTERN.search(generated_text))
        has_answer = bool(self.ANSWER_PATTERN.search(generated_text))
        if has_think and has_answer:
            return 1.0
        elif has_answer:  # على الأقل أعطى إجابة، حتى لو بدون تفكير ظاهر
            return 0.3
        return 0.0

    def compute_accuracy_reward(self, generated_text: str, ground_truth: str) -> float:
        """هل الإجابة المُستخرجة من <answer> مطابقة للحل الصحيح؟"""
        match = self.ANSWER_PATTERN.search(generated_text)
        if not match:
            return 0.0

        extracted_answer = match.group(1).strip()
        # تطبيع بسيط للمقارنة (في التطبيق الفعلي: تحقق رياضي/رمزي أعمق عبر SymPy مثلاً)
        normalized_extracted = re.sub(r"[^\d.\-]", "", extracted_answer)
        normalized_truth = re.sub(r"[^\d.\-]", "", ground_truth)

        return 1.0 if normalized_extracted == normalized_truth else 0.0

    def compute_total_reward(self, generated_text: str, ground_truth: str) -> dict[str, float]:
        format_r = self.compute_format_reward(generated_text)
        accuracy_r = self.compute_accuracy_reward(generated_text, ground_truth)

        # الدقة أهم بكثير من التنسيق — هذا يعكس أولوية R1: النتيجة الصحيحة أولاً
        total = 0.8 * accuracy_r + 0.2 * format_r

        return {
            "format_reward": format_r,
            "accuracy_reward": accuracy_r,
            "total_reward": total,
        }


class GRPOTrainer:
    """
    تطبيق مُبسّط لـ GRPO (Group Relative Policy Optimization) —
    الخوارزمية التي استخدمتها DeepSeek-R1 بدلًا من PPO التقليدي.

    الفكرة الجوهرية لـ GRPO: بدل تدريب Value Model منفصل (كما في PPO)
    لتقدير "ميزة" (Advantage) كل إجابة، نُولّد مجموعة (Group) من الإجابات
    لنفس السؤال، ونحسب الميزة بالمقارنة النسبية داخل المجموعة نفسها:

        A_i = (r_i - mean(r_group)) / std(r_group)

    هذا يُلغي الحاجة الكاملة لـ Value Model منفصل — توفير ضخم في الذاكرة والحوسبة!
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_model: RuleBasedRewardModel,
        group_size: int = 8,
        kl_coefficient: float = 0.04,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.group_size = group_size
        self.kl_coef = kl_coefficient
        self.reference_model = None  # نسخة مجمّدة من النموذج الأصلي لحساب KL Penalty

    def set_reference_model(self, reference_model: AutoModelForCausalLM) -> None:
        """نسخة مرجعية مُجمَّدة (لا تُحدَّث) تمنع النموذج من الانحراف الجذري أثناء RL"""
        self.reference_model = reference_model
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def _build_reasoning_prompt(self, question: str) -> str:
        """نفس تنسيق الـ Prompt المُستخدم في R1-Zero لتشجيع التفكير الصريح"""
        return (
            f"حل المسألة التالية خطوة بخطوة. ضع تفكيرك الكامل داخل "
            f"<think></think> ثم ضع الإجابة النهائية فقط داخل <answer></answer>.\n\n"
            f"المسألة: {question}\n"
        )

    @torch.no_grad()
    def generate_group(self, sample: ReasoningSample) -> list[tuple[str, Tensor]]:
        """
        توليد مجموعة من G إجابات مختلفة لنفس السؤال (عبر temperature sampling)
        هذا بديل GRPO لتوليد "عينات متنوعة للمقارنة النسبية" بدل Value Model
        """
        prompt = self._build_reasoning_prompt(sample.question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,          # مساحة كافية لسلاسل تفكير طويلة
            num_return_sequences=self.group_size,
            do_sample=True,
            temperature=0.8,               # تنوع كافٍ بين أفراد المجموعة
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_texts = [
            self.tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for seq in outputs.sequences
        ]
        return list(zip(generated_texts, outputs.sequences))

    def compute_group_advantages(
        self, sample: ReasoningSample, generated_texts: list[str]
    ) -> Tensor:
        """
        جوهر GRPO: حساب الميزة النسبية داخل المجموعة بدل استخدام Value Model
        A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
        """
        rewards = torch.tensor([
            self.reward_model.compute_total_reward(text, sample.ground_truth_answer)["total_reward"]
            for text in generated_texts
        ])

        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8  # تجنّب القسمة على صفر

        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def compute_kl_penalty(self, sequences: Tensor, policy_logits: Tensor) -> Tensor:
        """
        عقوبة KL Divergence بين النموذج الحالي والنموذج المرجعي المُجمَّد —
        تمنع "نسيان كارثي" أو انحراف غير آمن أثناء تحسين RL (راجع الفصل 35)
        """
        if self.reference_model is None:
            return torch.tensor(0.0)

        with torch.no_grad():
            ref_logits = self.reference_model(sequences).logits

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        kl_div = F.kl_div(policy_log_probs, ref_log_probs, reduction="batchmean", log_target=True)
        return kl_div

    def training_step(self, sample: ReasoningSample) -> dict[str, float]:
        """خطوة تدريب واحدة كاملة بمنطق GRPO"""
        # 1. توليد مجموعة من الإجابات المتنوعة
        generated_pairs = self.generate_group(sample)
        generated_texts = [text for text, _ in generated_pairs]

        # 2. حساب الميزة النسبية داخل المجموعة (بدل Value Model)
        advantages = self.compute_group_advantages(sample, generated_texts)

        # 3. حساب دالة الخسارة: -log(policy) * advantage + KL penalty
        #    (تبسيط تعليمي؛ GRPO الفعلي يستخدم أيضًا clipping مشابه لـ PPO)
        total_loss = torch.tensor(0.0, requires_grad=True)
        for (text, sequence), advantage in zip(generated_pairs, advantages):
            logits = self.model(sequence.unsqueeze(0)).logits
            log_probs = F.log_softmax(logits, dim=-1)

            # الخسارة السياسية: نُشجّع التسلسلات ذات الميزة الموجبة
            policy_loss = -log_probs.mean() * advantage
            kl_penalty = self.compute_kl_penalty(sequence.unsqueeze(0), logits)

            total_loss = total_loss + policy_loss + self.kl_coef * kl_penalty

        avg_loss = total_loss / len(generated_pairs)
        avg_loss.backward()

        return {
            "loss": avg_loss.item(),
            "mean_reward": advantages.mean().item(),
            "max_reward_in_group": advantages.max().item(),
        }


# ───────────────────────── مثال تطبيقي (مُبسّط) ─────────────────────────
if __name__ == "__main__":
    reward_model = RuleBasedRewardModel()

    # اختبار منطق المكافأة مباشرة قبل أي تدريب فعلي
    good_response = """<think>
    لدي 17 تفاحة. أعطيت 3 أشخاص × 4 تفاحات = 12 تفاحة. تبقى 17-12=5.
    اشتريت ضعف المتبقي = 2×5=10. المجموع النهائي = 5+10=15.
    </think>
    <answer>15</answer>"""

    bad_response_no_format = "الإجابة هي 15 تقريبًا"

    bad_response_wrong_answer = """<think>حسابات خاطئة هنا</think><answer>99</answer>"""

    for name, response in [
        ("إجابة صحيحة ومنسّقة", good_response),
        ("بدون تنسيق صحيح", bad_response_no_format),
        ("منسّقة لكن إجابة خاطئة", bad_response_wrong_answer),
    ]:
        rewards = reward_model.compute_total_reward(response, ground_truth="15")
        print(f"\n{name}:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
test_time_search.py
تطبيق Best-of-N مع Process Reward Model (PRM) مُبسّط —
بديل/مُكمّل لـ RL الكامل، يُستخدم وقت الاستدلال مباشرة دون الحاجة لإعادة تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class ReasoningStep:
    text: str
    step_score: float   # تقييم PRM لجودة هذه الخطوة تحديدًا


@dataclass
class ReasoningPath:
    steps: list[ReasoningStep]
    final_answer: str

    @property
    def aggregate_score(self) -> float:
        """تجميع تقييمات الخطوات — نستخدم الحد الأدنى (Min) كما في أبحاث PRM
        لأن سلسلة التفكير تنكسر عند أضعف حلقة فيها، وليس بمتوسط كل الخطوات"""
        if not self.steps:
            return 0.0
        return min(step.step_score for step in self.steps)


class ProcessRewardModel:
    """
    نموذج تقييم خطوة بخطوة — يُجيب على سؤال: 'هل هذه الخطوة التفكيرية
    المحددة صحيحة منطقيًا، بناءً على ما سبقها؟' وليس فقط 'هل النتيجة النهائية صحيحة؟'
    هذا أدق بكثير من Reward Model يُقيّم فقط الإجابة الأخيرة (Outcome Reward Model).
    """

    def __init__(self, model_name: str = "prm-scorer-checkpoint") -> None:
        # في الإنتاج الفعلي: نموذج مُدرَّب خصيصًا لتقييم الخطوات
        # (مثال معروف: Math-Shepherd من Microsoft، أو PRM800K من OpenAI)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # بديل توضيحي بسيط
        self.model = None  # Placeholder — نُحاكي السلوك للتبسيط التعليمي

    def score_step(self, previous_steps: list[str], current_step: str) -> float:
        """
        في التطبيق الفعلي: نمرر السياق الكامل (الخطوات السابقة + الحالية)
        لنموذج مُدرَّب على بيانات مُعلَّمة بشريًا (أو ذاتيًا عبر MCTS)
        يُنتج احتمالية أن هذه الخطوة 'صحيحة منطقيًا'.
        هنا نُحاكي heuristically لأغراض الشرح.
        """
        # محاكاة: خطوات تحتوي حسابات رقمية واضحة تحصل على تقييم أعلى
        has_explicit_calculation = any(c.isdigit() for c in current_step)
        has_contradiction_markers = "لكن" in current_step and "خطأ" in current_step
        base_score = 0.85 if has_explicit_calculation else 0.6
        if has_contradiction_markers:
            base_score -= 0.3
        return max(0.0, min(1.0, base_score))


class BestOfNSearcher:
    """
    استراتيجية Test-Time Compute بسيطة وفعالة جدًا عمليًا:
    1. وَلِّد N مسار تفكير كامل ومستقل لنفس السؤال
    2. قيّم كل خطوة في كل مسار عبر PRM
    3. اختر المسار ذا أعلى تقييم إجمالي (وليس فقط أول إجابة تظهر)

    هذا يُجسّد عمليًا مبدأ 'Test-Time Compute Scaling': كلما زاد N،
    زادت دقة النتيجة النهائية — بتكلفة حوسبة إضافية خطية تقريبًا.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prm: ProcessRewardModel,
        n_samples: int = 8,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.n_samples = n_samples

    def _parse_steps(self, generated_text: str) -> list[str]:
        """تقسيم النص المُولَّد إلى خطوات منفصلة (بافتراض فواصل أسطر أو ترقيم)"""
        raw_steps = [s.strip() for s in generated_text.split("\n") if s.strip()]
        return raw_steps

    def _extract_final_answer(self, generated_text: str) -> str:
        """استخراج الإجابة النهائية من آخر سطر عادة"""
        lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
        return lines[-1] if lines else ""

    @torch.no_grad()
    def search(self, question: str) -> ReasoningPath:
        """البحث الكامل: توليد N مسارات، تقييمها، واختيار الأفضل"""
        prompt = f"حل المسألة التالية خطوة بخطوة:\n{question}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=self.n_samples,
            do_sample=True,
            temperature=0.7,
        )

        candidate_paths: list[ReasoningPath] = []

        for seq in outputs:
            generated_text = self.tokenizer.decode(
                seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            raw_steps = self._parse_steps(generated_text)

            scored_steps: list[ReasoningStep] = []
            previous_context: list[str] = []
            for step_text in raw_steps:
                score = self.prm.score_step(previous_context, step_text)
                scored_steps.append(ReasoningStep(text=step_text, step_score=score))
                previous_context.append(step_text)

            path = ReasoningPath(
                steps=scored_steps,
                final_answer=self._extract_final_answer(generated_text),
            )
            candidate_paths.append(path)

        # 🎯 اختيار المسار الأفضل بناءً على أضعف حلقة فيه (Min Aggregation)
        best_path = max(candidate_paths, key=lambda p: p.aggregate_score)
        return best_path


# ───────────────────────── مثال تطبيقي توضيحي ─────────────────────────
if __name__ == "__main__":
    prm = ProcessRewardModel()

    # محاكاة تقييم مسارين متنافسين دون الحاجة لنموذج فعلي ضخم هنا
    path_good = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("3 أشخاص × 4 = 12 تفاحة أُعطيت", 0.9),
            ReasoningStep("17 - 12 = 5 تفاحات متبقية", 0.95),
            ReasoningStep("اشتريت ضعف 5 = 10", 0.9),
            ReasoningStep("5 + 10 = 15 تفاحة إجمالًا", 0.92),
        ],
        final_answer="15",
    )

    path_flawed = ReasoningPath(
        steps=[
            ReasoningStep("لدي 17 تفاحة", 0.9),
            ReasoningStep("أعطيت الكثير من التفاح للناس", 0.4),  # خطوة غامضة بلا حساب دقيق
            ReasoningStep("لذلك الإجابة تقريبًا 20", 0.2),         # خطأ واضح، تقييم منخفض
        ],
        final_answer="20",
    )

    print(f"تقييم المسار الجيد (Min Aggregation): {path_good.aggregate_score:.2f}")
    print(f"تقييم المسار المعيب (Min Aggregation): {path_flawed.aggregate_score:.2f}")
    print(f"\n✅ المسار المُختار: المسار ذو التقييم الأعلى → '{path_good.final_answer}'")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-65-chapter-65.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
reasoning_strategy_benchmark.py
مقارنة تجريبية شاملة بين استراتيجيات الاستدلال المختلفة:
الدقة المحققة مقابل تكلفة الحوسبة (Tokens المُستهلكة) لكل استراتيجية.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReasoningStrategy(Enum):
    DIRECT_ANSWER = "direct"                 # بدون تفكير، إجابة فورية
    CHAIN_OF_THOUGHT = "cot"                 # تفكير واحد خطوة بخطوة
    BEST_OF_N = "best_of_n"                   # عدة مسارات + اختيار الأفضل عبر PRM
    LONG_REASONING_RL = "long_reasoning_rl"   # نموذج مُدرَّب بـ RL على نمط o1/R1


@dataclass
class BenchmarkResult:
    strategy: ReasoningStrategy
    accuracy: float
    avg_tokens_used: int
    avg_latency_seconds: float
    estimated_cost_usd_per_1000_queries: float


class ReasoningBenchmark:
    """
    محاكاة بيانات مرجعية (مبنية على اتجاهات عامة موثقة في الأدبيات البحثية
    مثل ورقة o1 technical report وDeepSeek-R1 paper) لتوضيح العلاقة
    بين استثمار الحوسبة وقت الاستدلال والدقة المحققة.
    """

    # بيانات تقريبية ملهمة من الاتجاهات المنشورة في الأبحاث (AIME, MATH benchmarks)
    SIMULATED_RESULTS: dict[ReasoningStrategy, dict] = {
        ReasoningStrategy.DIRECT_ANSWER: {
            "accuracy": 0.42, "avg_tokens": 50, "avg_latency": 0.8,
        },
        ReasoningStrategy.CHAIN_OF_THOUGHT: {
            "accuracy": 0.58, "avg_tokens": 300, "avg_latency": 2.1,
        },
        ReasoningStrategy.BEST_OF_N: {
            "accuracy": 0.71, "avg_tokens": 300 * 8,  # N=8 مسارات متوازية
            "avg_latency": 3.5,  # يمكن التوازي، فالزمن لا يتضاعف خطيًا بالضرورة
        },
        ReasoningStrategy.LONG_REASONING_RL: {
            "accuracy": 0.83, "avg_tokens": 4500,  # سلاسل تفكير طويلة جدًا (كما في o1/R1)
            "avg_latency": 12.0,
        },
    }

    PRICE_PER_1K_TOKENS_USD = 0.01  # افتراضي توضيحي

    def run_comparison(self) -> list[BenchmarkResult]:
        results = []
        for strategy, data in self.SIMULATED_RESULTS.items():
            cost_per_query = (data["avg_tokens"] / 1000) * self.PRICE_PER_1K_TOKENS_USD
            results.append(BenchmarkResult(
                strategy=strategy,
                accuracy=data["accuracy"],
                avg_tokens_used=data["avg_tokens"],
                avg_latency_seconds=data["avg_latency"],
                estimated_cost_usd_per_1000_queries=round(cost_per_query * 1000, 2),
            ))
        return results

    def recommend_strategy(
        self, accuracy_requirement: float, max_latency_seconds: float, max_cost_per_1k_usd: float
    ) -> ReasoningStrategy | None:
        """
        🎯 القرار الهندسي الحقيقي: بناءً على متطلبات تطبيقك
        (SLA زمني، ميزانية، دقة مطلوبة)، أي استراتيجية تختار؟
        """
        results = self.run_comparison()
        eligible = [
            r for r in results
            if r.accuracy >= accuracy_requirement
            and r.avg_latency_seconds <= max_latency_seconds
            and r.estimated_cost_usd_per_1000_queries <= max_cost_per_1k_usd
        ]
        if not eligible:
            return None
        # من بين المؤهلين، اختر الأرخص (لأنهم جميعًا يحققون المتطلبات بالفعل)
        cheapest = min(eligible, key=lambda r: r.estimated_cost_usd_per_1000_queries)
        return cheapest.strategy


# ───────────────────────── مثال تطبيقي ─────────────────────────
if __name__ == "__main__":
    benchmark = ReasoningBenchmark()
    results = benchmark.run_comparison()

    print("📊 مقارنة شاملة بين استراتيجيات الاستدلال:\n")
    print(f"{'الاستراتيجية':<25}{'الدقة':<10}{'Tokens':<10}{'الزمن(ث)':<12}{'التكلفة/1000':<15}")
    print("-" * 75)
    for r in results:
        print(
            f"{r.strategy.value:<25}{r.accuracy:<10.2%}{r.avg_tokens_used:<10}"
            f"{r.avg_latency_seconds:<12.1f}${r.estimated_cost_usd_per_1000_queries:<15.2f}"
        )

    print("\n🎯 سيناريو 1: تطبيق دردشة فورية (SLA صارم، دقة متوسطة كافية)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.5, max_latency_seconds=3.0, max_cost_per_1k_usd=10.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")

    print("\n🎯 سيناريو 2: حل مسائل أولمبياد رياضية (دقة قصوى، زمن غير حرج)")
    choice = benchmark.recommend_strategy(
        accuracy_requirement=0.8, max_latency_seconds=20.0, max_cost_per_1k_usd=100.0
    )
    print(f"   التوصية: {choice.value if choice else 'لا يوجد خيار مناسب'}")
