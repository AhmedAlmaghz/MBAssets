#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)
Large Language Model Engineering — LLM-as-a-Judge

Book slug : llm-engineering-69-chapters
Chapter   : ch-47-chapter-47 (lesson 47)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-47-chapter-47.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)
Large Language Model Engineering — LLM-as-a-Judge

Book slug : llm-engineering-69-chapters
Chapter   : ch-47-chapter-47 (lesson 47)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-47-chapter-47.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)
Large Language Model Engineering — LLM-as-a-Judge

Book slug : llm-engineering-69-chapters
Chapter   : ch-47-chapter-47 (lesson 47)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_47.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 47: الفصل 47: التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_judge() -> None:
    judge_a, judge_b = ([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1])
    observed = sum((a == b for a, b in zip(judge_a, judge_b))) / len(judge_a)
    p_a = sum(judge_a) / len(judge_a)
    p_b = sum(judge_b) / len(judge_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (observed - expected) / (1 - expected)
    print('raw agreement:', round(observed, 3))
    print("Cohen's kappa:", round(kappa, 3))
    print('resolve disagreement with a blinded human audit.')

if __name__ == "__main__":
    print("Chapter 47 lab: Measure agreement between independent model judges")
    run_judge()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-47.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام LLM-as-a-Judge متكامل: Single-Answer Grading + Pairwise مع Position Swapping
+ قياس اتفاق مع تقييم بشري (Cohen's Kappa) + حساب تصنيف Elo
يستخدم: openai (أو أي واجهة متوافقة)، numpy
"""

import re
import json
import itertools
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from openai import OpenAI

client = OpenAI()  # يتطلب OPENAI_API_KEY في متغيرات البيئة
JUDGE_MODEL = "gpt-4o"


# ============================================================
# الجزء 1: Single-Answer Grading مع Rubric صريح
# ============================================================

GRADING_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا، مهمتك تقييم جودة إجابة مساعد ذكاء اصطناعي.

قيّم الإجابة بناءً على المعايير التالية:
- الصحة الواقعية (0-3 نقاط): هل المعلومات صحيحة وخالية من الهلوسة؟
- الوضوح والتنظيم (0-3 نقاط): هل الإجابة منظّمة وسهلة الفهم؟
- اكتمال الإجابة (0-2 نقطة): هل تغطي كل جوانب السؤال؟
- الأسلوب والنبرة (0-2 نقطة): هل الأسلوب مناسب للسياق؟

فكّر خطوة بخطوة في كل معيار على حدة أولًا، ثم اختم بالضبط بالصيغة التالية:
[[SCORE]]: X

حيث X رقم صحيح من 0 إلى 10.

[السؤال]:
{question}

[الإجابة المُقيَّمة]:
{answer}

[تحليلك]:"""


def grade_single_answer(question: str, answer: str, judge_model: str = JUDGE_MODEL) -> dict:
    """يُقيّم إجابة واحدة بمعزل عن أي منافس، عبر Rubric صريح وCoT قبل الدرجة."""
    prompt = GRADING_PROMPT_TEMPLATE.format(question=question, answer=answer)

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # حسم عالٍ مطلوب لحَكَم متّسق (انظر فصل 38)
    )
    judgment_text = response.choices[0].message.content

    match = re.search(r"\[\[SCORE\]\]:\s*(\d+)", judgment_text)
    score = int(match.group(1)) if match else None

    return {"score": score, "reasoning": judgment_text}


# ============================================================
# الجزء 2: Pairwise Comparison مع Position Swapping
# ============================================================

PAIRWISE_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا. قارن بين إجابتين لنفس السؤال وحدد أيهما أفضل.

قيّم بناءً على: الصحة الواقعية، الوضوح، الاكتمال، والأسلوب.
تجاهل تمامًا طول الإجابة أو ترتيب عرضها كمعيار — ركّز فقط على الجودة الفعلية للمحتوى.

فكّر خطوة بخطوة، ثم اختم بالضبط بالصيغة التالية:
[[VERDICT]]: A
أو
[[VERDICT]]: B
أو
[[VERDICT]]: TIE

[السؤال]:
{question}

[الإجابة A]:
{answer_a}

[الإجابة B]:
{answer_b}

[تحليلك]:"""


def _single_pairwise_call(question: str, answer_a: str, answer_b: str,
                            judge_model: str = JUDGE_MODEL) -> str:
    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    match = re.search(r"\[\[VERDICT\]\]:\s*(A|B|TIE)", text)
    return match.group(1) if match else "TIE"


def pairwise_compare_with_swapping(question: str, answer_a: str, answer_b: str,
                                     judge_model: str = JUDGE_MODEL) -> dict:
    """
    يُجري المقارنة مرتين (بالترتيب الأصلي والمعكوس) لإزالة تحيّز الموضع
    (Position Bias)، ويُعلن "تعادل" إن لم يتفق الحَكَم مع نفسه.
    """
    verdict_forward = _single_pairwise_call(question, answer_a, answer_b, judge_model)

    # في الاستدعاء المعكوس، "A" و"B" في الـ prompt تشيران لـ (answer_b, answer_a)
    verdict_backward_raw = _single_pairwise_call(question, answer_b, answer_a, judge_model)
    # نعيد التفسير لمرجعية answer_a/answer_b الأصلية
    swap_map = {"A": "B", "B": "A", "TIE": "TIE"}
    verdict_backward = swap_map[verdict_backward_raw]

    if verdict_forward == verdict_backward and verdict_forward != "TIE":
        final_verdict = verdict_forward
        is_consistent = True
    else:
        final_verdict = "TIE"
        is_consistent = (verdict_forward == verdict_backward)  # تعادل حقيقي أم تناقض؟

    return {
        "verdict_forward": verdict_forward,
        "verdict_backward": verdict_backward,
        "final_verdict": final_verdict,
        "position_bias_detected": not is_consistent,
    }


# ============================================================
# الجزء 3: قياس اتفاق الحَكَم الآلي مع تقييم بشري (Cohen's Kappa)
# ============================================================

def cohens_kappa(human_labels: list[str], judge_labels: list[str]) -> float:
    """
    يحسب Cohen's Kappa بين تصنيفين (بشري وآلي) على نفس العينات،
    طبقًا للصيغة الرياضية المُشتقة في القسم 🟡 أعلاه.
    """
    assert len(human_labels) == len(judge_labels)
    n = len(human_labels)
    categories = sorted(set(human_labels) | set(judge_labels))

    # الاتفاق الملاحَظ p_o
    p_o = sum(h == j for h, j in zip(human_labels, judge_labels)) / n

    # الاتفاق المتوقَّع بالصدفة p_e
    human_dist = {c: human_labels.count(c) / n for c in categories}
    judge_dist = {c: judge_labels.count(c) / n for c in categories}
    p_e = sum(human_dist[c] * judge_dist[c] for c in categories)

    if p_e == 1.0:
        return 1.0  # تفادي القسمة على صفر في حالة اتفاق تام مصادفة
    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(kappa: float) -> str:
    """تفسير قياسي من Landis & Koch (1977)."""
    if kappa < 0:
        return "أسوأ من الصدفة ⚠️"
    elif kappa < 0.20:
        return "ضعيف جدًا"
    elif kappa < 0.40:
        return "ضعيف"
    elif kappa < 0.60:
        return "معتدل"
    elif kappa < 0.80:
        return "جوهري (مقبول للإنتاج)"
    return "شبه تام"


# ============================================================
# الجزء 4: حساب تصنيف Elo من نتائج مقارنات ثنائية متعددة
# ============================================================

@dataclass
class EloRatingSystem:
    """نظام تصنيف Elo لترتيب عدة نماذج من نتائج مقارنات ثنائية متكررة."""

    k_factor: float = 32.0
    initial_rating: float = 1000.0
    ratings: dict = field(default_factory=lambda: defaultdict(lambda: 1000.0))

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, model_a: str, model_b: str, outcome: str) -> None:
        """outcome: 'A' أو 'B' أو 'TIE' حسب نتيجة المقارنة."""
        r_a, r_b = self.ratings[model_a], self.ratings[model_b]
        e_a = self.expected_score(r_a, r_b)
        e_b = 1.0 - e_a

        score_map = {"A": (1.0, 0.0), "B": (0.0, 1.0), "TIE": (0.5, 0.5)}
        s_a, s_b = score_map[outcome]

        self.ratings[model_a] = r_a + self.k_factor * (s_a - e_a)
        self.ratings[model_b] = r_b + self.k_factor * (s_b - e_b)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ============================================================
# الجزء 5: تشغيل السيناريو الكامل
# ============================================================

if __name__ == "__main__":
    question = "اشرح لماذا تتجمد المياه عند صفر درجة مئوية بلغة بسيطة."

    answer_concise = (
        "لأن جزيئات الماء تتباطأ حركتها عند هذه الدرجة وتترتب في شكل بلوري ثابت هو الجليد."
    )
    answer_verbose = (
        "هذا سؤال رائع! دعني أشرح لك بالتفصيل. الماء يتكون من جزيئات H2O، وهذه "
        "الجزيئات في الحالة السائلة تتحرك بحرية نسبية وتتصادم باستمرار فيما بينها "
        "بفعل الطاقة الحرارية. عندما تنخفض درجة الحرارة إلى الصفر المئوي، تقل "
        "الطاقة الحركية لهذه الجزيئات بشكل كبير جدًا، مما يسمح لقوى الترابط "
        "الهيدروجيني بينها بأن تُثبّتها في شبكة بلورية منتظمة الشكل السداسي، "
        "وهذا التحوّل في الترتيب الجزيئي هو ما نسميه التجمّد أو التحوّل لحالة الجليد الصلبة."
    )

    print("=" * 70)
    print("القسم 1: Single-Answer Grading لكل إجابة بمفردها")
    print("=" * 70)
    for name, ans in [("موجزة", answer_concise), ("مُطوَّلة", answer_verbose)]:
        result = grade_single_answer(question, ans)
        print(f"\n[{name}] الدرجة: {result['score']}/10")

    print("\n" + "=" * 70)
    print("القسم 2: Pairwise Comparison مع Position Swapping")
    print("=" * 70)
    pairwise_result = pairwise_compare_with_swapping(question, answer_concise, answer_verbose)
    print(f"الحكم بالترتيب الأصلي (A=موجزة, B=مُطوَّلة)  : {pairwise_result['verdict_forward']}")
    print(f"الحكم بالترتيب المعكوس (بعد إعادة التفسير)   : {pairwise_result['verdict_backward']}")
    print(f"الحكم النهائي                                : {pairwise_result['final_verdict']}")
    print(f"تم اكتشاف تحيّز موضع؟                        : "
          f"{'نعم ⚠️' if pairwise_result['position_bias_detected'] else 'لا ✅'}")

    print("\n" + "=" * 70)
    print("القسم 3: معايرة الحَكَم الآلي مقابل تقييمات بشرية (محاكاة)")
    print("=" * 70)

    # محاكاة: 10 مقارنات قيّمها بشر، وقيّمها الحَكَم الآلي أيضًا
    human_verdicts = ["A", "A", "B", "TIE", "A", "B", "B", "A", "TIE", "A"]
    judge_verdicts = ["A", "A", "B", "A", "A", "B", "TIE", "A", "TIE", "B"]

    kappa = cohens_kappa(human_verdicts, judge_verdicts)
    print(f"Cohen's Kappa بين الحَكَم الآلي والبشري: {kappa:.3f}")
    print(f"التفسير: {interpret_kappa(kappa)}")

    print("\n" + "=" * 70)
    print("القسم 4: بناء تصنيف Elo من نتائج مقارنات متعددة النماذج")
    print("=" * 70)

    elo_system = EloRatingSystem(k_factor=32.0)

    # محاكاة: نتائج مقارنات ثنائية عشوائية بين 4 نماذج عبر 200 مواجهة
    rng = np.random.default_rng(7)
    models = ["Model_GPT", "Model_Claude", "Model_Llama", "Model_Mistral"]
    # قوة "حقيقية" خفية لكل نموذج (لمحاكاة نتائج واقعية متسقة إحصائيًا)
    true_strength = {"Model_GPT": 1.15, "Model_Claude": 1.05, "Model_Llama": 0.95, "Model_Mistral": 0.85}

    for _ in range(200):
        model_a, model_b = rng.choice(models, size=2, replace=False)
        p_a_wins = true_strength[model_a] / (true_strength[model_a] + true_strength[model_b])
        outcome = "A" if rng.random() < p_a_wins else "B"
        elo_system.update(model_a, model_b, outcome)

    print("التصنيف النهائي (Elo) بعد 200 مواجهة:")
    for rank, (model, rating) in enumerate(elo_system.leaderboard(), 1):
        print(f"  {rank}. {model:20s}: {rating:.1f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-47-chapter-47.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 47: الفصل 47: التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_judge() -> None:
    judge_a, judge_b = ([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1])
    observed = sum((a == b for a, b in zip(judge_a, judge_b))) / len(judge_a)
    p_a = sum(judge_a) / len(judge_a)
    p_b = sum(judge_b) / len(judge_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (observed - expected) / (1 - expected)
    print('raw agreement:', round(observed, 3))
    print("Cohen's kappa:", round(kappa, 3))
    print('resolve disagreement with a blinded human audit.')

if __name__ == "__main__":
    print("Chapter 47 lab: Measure agreement between independent model judges")
    run_judge()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-47-chapter-47.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
نظام LLM-as-a-Judge متكامل: Single-Answer Grading + Pairwise مع Position Swapping
+ قياس اتفاق مع تقييم بشري (Cohen's Kappa) + حساب تصنيف Elo
يستخدم: openai (أو أي واجهة متوافقة)، numpy
"""

import re
import json
import itertools
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from openai import OpenAI

client = OpenAI()  # يتطلب OPENAI_API_KEY في متغيرات البيئة
JUDGE_MODEL = "gpt-4o"


# ============================================================
# الجزء 1: Single-Answer Grading مع Rubric صريح
# ============================================================

GRADING_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا، مهمتك تقييم جودة إجابة مساعد ذكاء اصطناعي.

قيّم الإجابة بناءً على المعايير التالية:
- الصحة الواقعية (0-3 نقاط): هل المعلومات صحيحة وخالية من الهلوسة؟
- الوضوح والتنظيم (0-3 نقاط): هل الإجابة منظّمة وسهلة الفهم؟
- اكتمال الإجابة (0-2 نقطة): هل تغطي كل جوانب السؤال؟
- الأسلوب والنبرة (0-2 نقطة): هل الأسلوب مناسب للسياق؟

فكّر خطوة بخطوة في كل معيار على حدة أولًا، ثم اختم بالضبط بالصيغة التالية:
[[SCORE]]: X

حيث X رقم صحيح من 0 إلى 10.

[السؤال]:
{question}

[الإجابة المُقيَّمة]:
{answer}

[تحليلك]:"""


def grade_single_answer(question: str, answer: str, judge_model: str = JUDGE_MODEL) -> dict:
    """يُقيّم إجابة واحدة بمعزل عن أي منافس، عبر Rubric صريح وCoT قبل الدرجة."""
    prompt = GRADING_PROMPT_TEMPLATE.format(question=question, answer=answer)

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # حسم عالٍ مطلوب لحَكَم متّسق (انظر فصل 38)
    )
    judgment_text = response.choices[0].message.content

    match = re.search(r"\[\[SCORE\]\]:\s*(\d+)", judgment_text)
    score = int(match.group(1)) if match else None

    return {"score": score, "reasoning": judgment_text}


# ============================================================
# الجزء 2: Pairwise Comparison مع Position Swapping
# ============================================================

PAIRWISE_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا. قارن بين إجابتين لنفس السؤال وحدد أيهما أفضل.

قيّم بناءً على: الصحة الواقعية، الوضوح، الاكتمال، والأسلوب.
تجاهل تمامًا طول الإجابة أو ترتيب عرضها كمعيار — ركّز فقط على الجودة الفعلية للمحتوى.

فكّر خطوة بخطوة، ثم اختم بالضبط بالصيغة التالية:
[[VERDICT]]: A
أو
[[VERDICT]]: B
أو
[[VERDICT]]: TIE

[السؤال]:
{question}

[الإجابة A]:
{answer_a}

[الإجابة B]:
{answer_b}

[تحليلك]:"""


def _single_pairwise_call(question: str, answer_a: str, answer_b: str,
                            judge_model: str = JUDGE_MODEL) -> str:
    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    match = re.search(r"\[\[VERDICT\]\]:\s*(A|B|TIE)", text)
    return match.group(1) if match else "TIE"


def pairwise_compare_with_swapping(question: str, answer_a: str, answer_b: str,
                                     judge_model: str = JUDGE_MODEL) -> dict:
    """
    يُجري المقارنة مرتين (بالترتيب الأصلي والمعكوس) لإزالة تحيّز الموضع
    (Position Bias)، ويُعلن "تعادل" إن لم يتفق الحَكَم مع نفسه.
    """
    verdict_forward = _single_pairwise_call(question, answer_a, answer_b, judge_model)

    # في الاستدعاء المعكوس، "A" و"B" في الـ prompt تشيران لـ (answer_b, answer_a)
    verdict_backward_raw = _single_pairwise_call(question, answer_b, answer_a, judge_model)
    # نعيد التفسير لمرجعية answer_a/answer_b الأصلية
    swap_map = {"A": "B", "B": "A", "TIE": "TIE"}
    verdict_backward = swap_map[verdict_backward_raw]

    if verdict_forward == verdict_backward and verdict_forward != "TIE":
        final_verdict = verdict_forward
        is_consistent = True
    else:
        final_verdict = "TIE"
        is_consistent = (verdict_forward == verdict_backward)  # تعادل حقيقي أم تناقض؟

    return {
        "verdict_forward": verdict_forward,
        "verdict_backward": verdict_backward,
        "final_verdict": final_verdict,
        "position_bias_detected": not is_consistent,
    }


# ============================================================
# الجزء 3: قياس اتفاق الحَكَم الآلي مع تقييم بشري (Cohen's Kappa)
# ============================================================

def cohens_kappa(human_labels: list[str], judge_labels: list[str]) -> float:
    """
    يحسب Cohen's Kappa بين تصنيفين (بشري وآلي) على نفس العينات،
    طبقًا للصيغة الرياضية المُشتقة في القسم 🟡 أعلاه.
    """
    assert len(human_labels) == len(judge_labels)
    n = len(human_labels)
    categories = sorted(set(human_labels) | set(judge_labels))

    # الاتفاق الملاحَظ p_o
    p_o = sum(h == j for h, j in zip(human_labels, judge_labels)) / n

    # الاتفاق المتوقَّع بالصدفة p_e
    human_dist = {c: human_labels.count(c) / n for c in categories}
    judge_dist = {c: judge_labels.count(c) / n for c in categories}
    p_e = sum(human_dist[c] * judge_dist[c] for c in categories)

    if p_e == 1.0:
        return 1.0  # تفادي القسمة على صفر في حالة اتفاق تام مصادفة
    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(kappa: float) -> str:
    """تفسير قياسي من Landis & Koch (1977)."""
    if kappa < 0:
        return "أسوأ من الصدفة ⚠️"
    elif kappa < 0.20:
        return "ضعيف جدًا"
    elif kappa < 0.40:
        return "ضعيف"
    elif kappa < 0.60:
        return "معتدل"
    elif kappa < 0.80:
        return "جوهري (مقبول للإنتاج)"
    return "شبه تام"


# ============================================================
# الجزء 4: حساب تصنيف Elo من نتائج مقارنات ثنائية متعددة
# ============================================================

@dataclass
class EloRatingSystem:
    """نظام تصنيف Elo لترتيب عدة نماذج من نتائج مقارنات ثنائية متكررة."""

    k_factor: float = 32.0
    initial_rating: float = 1000.0
    ratings: dict = field(default_factory=lambda: defaultdict(lambda: 1000.0))

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, model_a: str, model_b: str, outcome: str) -> None:
        """outcome: 'A' أو 'B' أو 'TIE' حسب نتيجة المقارنة."""
        r_a, r_b = self.ratings[model_a], self.ratings[model_b]
        e_a = self.expected_score(r_a, r_b)
        e_b = 1.0 - e_a

        score_map = {"A": (1.0, 0.0), "B": (0.0, 1.0), "TIE": (0.5, 0.5)}
        s_a, s_b = score_map[outcome]

        self.ratings[model_a] = r_a + self.k_factor * (s_a - e_a)
        self.ratings[model_b] = r_b + self.k_factor * (s_b - e_b)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ============================================================
# الجزء 5: تشغيل السيناريو الكامل
# ============================================================

if __name__ == "__main__":
    question = "اشرح لماذا تتجمد المياه عند صفر درجة مئوية بلغة بسيطة."

    answer_concise = (
        "لأن جزيئات الماء تتباطأ حركتها عند هذه الدرجة وتترتب في شكل بلوري ثابت هو الجليد."
    )
    answer_verbose = (
        "هذا سؤال رائع! دعني أشرح لك بالتفصيل. الماء يتكون من جزيئات H2O، وهذه "
        "الجزيئات في الحالة السائلة تتحرك بحرية نسبية وتتصادم باستمرار فيما بينها "
        "بفعل الطاقة الحرارية. عندما تنخفض درجة الحرارة إلى الصفر المئوي، تقل "
        "الطاقة الحركية لهذه الجزيئات بشكل كبير جدًا، مما يسمح لقوى الترابط "
        "الهيدروجيني بينها بأن تُثبّتها في شبكة بلورية منتظمة الشكل السداسي، "
        "وهذا التحوّل في الترتيب الجزيئي هو ما نسميه التجمّد أو التحوّل لحالة الجليد الصلبة."
    )

    print("=" * 70)
    print("القسم 1: Single-Answer Grading لكل إجابة بمفردها")
    print("=" * 70)
    for name, ans in [("موجزة", answer_concise), ("مُطوَّلة", answer_verbose)]:
        result = grade_single_answer(question, ans)
        print(f"\n[{name}] الدرجة: {result['score']}/10")

    print("\n" + "=" * 70)
    print("القسم 2: Pairwise Comparison مع Position Swapping")
    print("=" * 70)
    pairwise_result = pairwise_compare_with_swapping(question, answer_concise, answer_verbose)
    print(f"الحكم بالترتيب الأصلي (A=موجزة, B=مُطوَّلة)  : {pairwise_result['verdict_forward']}")
    print(f"الحكم بالترتيب المعكوس (بعد إعادة التفسير)   : {pairwise_result['verdict_backward']}")
    print(f"الحكم النهائي                                : {pairwise_result['final_verdict']}")
    print(f"تم اكتشاف تحيّز موضع؟                        : "
          f"{'نعم ⚠️' if pairwise_result['position_bias_detected'] else 'لا ✅'}")

    print("\n" + "=" * 70)
    print("القسم 3: معايرة الحَكَم الآلي مقابل تقييمات بشرية (محاكاة)")
    print("=" * 70)

    # محاكاة: 10 مقارنات قيّمها بشر، وقيّمها الحَكَم الآلي أيضًا
    human_verdicts = ["A", "A", "B", "TIE", "A", "B", "B", "A", "TIE", "A"]
    judge_verdicts = ["A", "A", "B", "A", "A", "B", "TIE", "A", "TIE", "B"]

    kappa = cohens_kappa(human_verdicts, judge_verdicts)
    print(f"Cohen's Kappa بين الحَكَم الآلي والبشري: {kappa:.3f}")
    print(f"التفسير: {interpret_kappa(kappa)}")

    print("\n" + "=" * 70)
    print("القسم 4: بناء تصنيف Elo من نتائج مقارنات متعددة النماذج")
    print("=" * 70)

    elo_system = EloRatingSystem(k_factor=32.0)

    # محاكاة: نتائج مقارنات ثنائية عشوائية بين 4 نماذج عبر 200 مواجهة
    rng = np.random.default_rng(7)
    models = ["Model_GPT", "Model_Claude", "Model_Llama", "Model_Mistral"]
    # قوة "حقيقية" خفية لكل نموذج (لمحاكاة نتائج واقعية متسقة إحصائيًا)
    true_strength = {"Model_GPT": 1.15, "Model_Claude": 1.05, "Model_Llama": 0.95, "Model_Mistral": 0.85}

    for _ in range(200):
        model_a, model_b = rng.choice(models, size=2, replace=False)
        p_a_wins = true_strength[model_a] / (true_strength[model_a] + true_strength[model_b])
        outcome = "A" if rng.random() < p_a_wins else "B"
        elo_system.update(model_a, model_b, outcome)

    print("التصنيف النهائي (Elo) بعد 200 مواجهة:")
    for rank, (model, rating) in enumerate(elo_system.leaderboard(), 1):
        print(f"  {rank}. {model:20s}: {rating:.1f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-47-chapter-47.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)
Large Language Model Engineering — LLM-as-a-Judge

Book slug : llm-engineering-69-chapters
Chapter   : ch-47-chapter-47 (lesson 47)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_47.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 47: الفصل 47: التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_judge() -> None:
    judge_a, judge_b = ([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1])
    observed = sum((a == b for a, b in zip(judge_a, judge_b))) / len(judge_a)
    p_a = sum(judge_a) / len(judge_a)
    p_b = sum(judge_b) / len(judge_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (observed - expected) / (1 - expected)
    print('raw agreement:', round(observed, 3))
    print("Cohen's kappa:", round(kappa, 3))
    print('resolve disagreement with a blinded human audit.')

if __name__ == "__main__":
    print("Chapter 47 lab: Measure agreement between independent model judges")
    run_judge()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-47.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام LLM-as-a-Judge متكامل: Single-Answer Grading + Pairwise مع Position Swapping
+ قياس اتفاق مع تقييم بشري (Cohen's Kappa) + حساب تصنيف Elo
يستخدم: openai (أو أي واجهة متوافقة)، numpy
"""

import re
import json
import itertools
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from openai import OpenAI

client = OpenAI()  # يتطلب OPENAI_API_KEY في متغيرات البيئة
JUDGE_MODEL = "gpt-4o"


# ============================================================
# الجزء 1: Single-Answer Grading مع Rubric صريح
# ============================================================

GRADING_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا، مهمتك تقييم جودة إجابة مساعد ذكاء اصطناعي.

قيّم الإجابة بناءً على المعايير التالية:
- الصحة الواقعية (0-3 نقاط): هل المعلومات صحيحة وخالية من الهلوسة؟
- الوضوح والتنظيم (0-3 نقاط): هل الإجابة منظّمة وسهلة الفهم؟
- اكتمال الإجابة (0-2 نقطة): هل تغطي كل جوانب السؤال؟
- الأسلوب والنبرة (0-2 نقطة): هل الأسلوب مناسب للسياق؟

فكّر خطوة بخطوة في كل معيار على حدة أولًا، ثم اختم بالضبط بالصيغة التالية:
[[SCORE]]: X

حيث X رقم صحيح من 0 إلى 10.

[السؤال]:
{question}

[الإجابة المُقيَّمة]:
{answer}

[تحليلك]:"""


def grade_single_answer(question: str, answer: str, judge_model: str = JUDGE_MODEL) -> dict:
    """يُقيّم إجابة واحدة بمعزل عن أي منافس، عبر Rubric صريح وCoT قبل الدرجة."""
    prompt = GRADING_PROMPT_TEMPLATE.format(question=question, answer=answer)

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # حسم عالٍ مطلوب لحَكَم متّسق (انظر فصل 38)
    )
    judgment_text = response.choices[0].message.content

    match = re.search(r"\[\[SCORE\]\]:\s*(\d+)", judgment_text)
    score = int(match.group(1)) if match else None

    return {"score": score, "reasoning": judgment_text}


# ============================================================
# الجزء 2: Pairwise Comparison مع Position Swapping
# ============================================================

PAIRWISE_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا. قارن بين إجابتين لنفس السؤال وحدد أيهما أفضل.

قيّم بناءً على: الصحة الواقعية، الوضوح، الاكتمال، والأسلوب.
تجاهل تمامًا طول الإجابة أو ترتيب عرضها كمعيار — ركّز فقط على الجودة الفعلية للمحتوى.

فكّر خطوة بخطوة، ثم اختم بالضبط بالصيغة التالية:
[[VERDICT]]: A
أو
[[VERDICT]]: B
أو
[[VERDICT]]: TIE

[السؤال]:
{question}

[الإجابة A]:
{answer_a}

[الإجابة B]:
{answer_b}

[تحليلك]:"""


def _single_pairwise_call(question: str, answer_a: str, answer_b: str,
                            judge_model: str = JUDGE_MODEL) -> str:
    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    match = re.search(r"\[\[VERDICT\]\]:\s*(A|B|TIE)", text)
    return match.group(1) if match else "TIE"


def pairwise_compare_with_swapping(question: str, answer_a: str, answer_b: str,
                                     judge_model: str = JUDGE_MODEL) -> dict:
    """
    يُجري المقارنة مرتين (بالترتيب الأصلي والمعكوس) لإزالة تحيّز الموضع
    (Position Bias)، ويُعلن "تعادل" إن لم يتفق الحَكَم مع نفسه.
    """
    verdict_forward = _single_pairwise_call(question, answer_a, answer_b, judge_model)

    # في الاستدعاء المعكوس، "A" و"B" في الـ prompt تشيران لـ (answer_b, answer_a)
    verdict_backward_raw = _single_pairwise_call(question, answer_b, answer_a, judge_model)
    # نعيد التفسير لمرجعية answer_a/answer_b الأصلية
    swap_map = {"A": "B", "B": "A", "TIE": "TIE"}
    verdict_backward = swap_map[verdict_backward_raw]

    if verdict_forward == verdict_backward and verdict_forward != "TIE":
        final_verdict = verdict_forward
        is_consistent = True
    else:
        final_verdict = "TIE"
        is_consistent = (verdict_forward == verdict_backward)  # تعادل حقيقي أم تناقض؟

    return {
        "verdict_forward": verdict_forward,
        "verdict_backward": verdict_backward,
        "final_verdict": final_verdict,
        "position_bias_detected": not is_consistent,
    }


# ============================================================
# الجزء 3: قياس اتفاق الحَكَم الآلي مع تقييم بشري (Cohen's Kappa)
# ============================================================

def cohens_kappa(human_labels: list[str], judge_labels: list[str]) -> float:
    """
    يحسب Cohen's Kappa بين تصنيفين (بشري وآلي) على نفس العينات،
    طبقًا للصيغة الرياضية المُشتقة في القسم 🟡 أعلاه.
    """
    assert len(human_labels) == len(judge_labels)
    n = len(human_labels)
    categories = sorted(set(human_labels) | set(judge_labels))

    # الاتفاق الملاحَظ p_o
    p_o = sum(h == j for h, j in zip(human_labels, judge_labels)) / n

    # الاتفاق المتوقَّع بالصدفة p_e
    human_dist = {c: human_labels.count(c) / n for c in categories}
    judge_dist = {c: judge_labels.count(c) / n for c in categories}
    p_e = sum(human_dist[c] * judge_dist[c] for c in categories)

    if p_e == 1.0:
        return 1.0  # تفادي القسمة على صفر في حالة اتفاق تام مصادفة
    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(kappa: float) -> str:
    """تفسير قياسي من Landis & Koch (1977)."""
    if kappa < 0:
        return "أسوأ من الصدفة ⚠️"
    elif kappa < 0.20:
        return "ضعيف جدًا"
    elif kappa < 0.40:
        return "ضعيف"
    elif kappa < 0.60:
        return "معتدل"
    elif kappa < 0.80:
        return "جوهري (مقبول للإنتاج)"
    return "شبه تام"


# ============================================================
# الجزء 4: حساب تصنيف Elo من نتائج مقارنات ثنائية متعددة
# ============================================================

@dataclass
class EloRatingSystem:
    """نظام تصنيف Elo لترتيب عدة نماذج من نتائج مقارنات ثنائية متكررة."""

    k_factor: float = 32.0
    initial_rating: float = 1000.0
    ratings: dict = field(default_factory=lambda: defaultdict(lambda: 1000.0))

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, model_a: str, model_b: str, outcome: str) -> None:
        """outcome: 'A' أو 'B' أو 'TIE' حسب نتيجة المقارنة."""
        r_a, r_b = self.ratings[model_a], self.ratings[model_b]
        e_a = self.expected_score(r_a, r_b)
        e_b = 1.0 - e_a

        score_map = {"A": (1.0, 0.0), "B": (0.0, 1.0), "TIE": (0.5, 0.5)}
        s_a, s_b = score_map[outcome]

        self.ratings[model_a] = r_a + self.k_factor * (s_a - e_a)
        self.ratings[model_b] = r_b + self.k_factor * (s_b - e_b)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ============================================================
# الجزء 5: تشغيل السيناريو الكامل
# ============================================================

if __name__ == "__main__":
    question = "اشرح لماذا تتجمد المياه عند صفر درجة مئوية بلغة بسيطة."

    answer_concise = (
        "لأن جزيئات الماء تتباطأ حركتها عند هذه الدرجة وتترتب في شكل بلوري ثابت هو الجليد."
    )
    answer_verbose = (
        "هذا سؤال رائع! دعني أشرح لك بالتفصيل. الماء يتكون من جزيئات H2O، وهذه "
        "الجزيئات في الحالة السائلة تتحرك بحرية نسبية وتتصادم باستمرار فيما بينها "
        "بفعل الطاقة الحرارية. عندما تنخفض درجة الحرارة إلى الصفر المئوي، تقل "
        "الطاقة الحركية لهذه الجزيئات بشكل كبير جدًا، مما يسمح لقوى الترابط "
        "الهيدروجيني بينها بأن تُثبّتها في شبكة بلورية منتظمة الشكل السداسي، "
        "وهذا التحوّل في الترتيب الجزيئي هو ما نسميه التجمّد أو التحوّل لحالة الجليد الصلبة."
    )

    print("=" * 70)
    print("القسم 1: Single-Answer Grading لكل إجابة بمفردها")
    print("=" * 70)
    for name, ans in [("موجزة", answer_concise), ("مُطوَّلة", answer_verbose)]:
        result = grade_single_answer(question, ans)
        print(f"\n[{name}] الدرجة: {result['score']}/10")

    print("\n" + "=" * 70)
    print("القسم 2: Pairwise Comparison مع Position Swapping")
    print("=" * 70)
    pairwise_result = pairwise_compare_with_swapping(question, answer_concise, answer_verbose)
    print(f"الحكم بالترتيب الأصلي (A=موجزة, B=مُطوَّلة)  : {pairwise_result['verdict_forward']}")
    print(f"الحكم بالترتيب المعكوس (بعد إعادة التفسير)   : {pairwise_result['verdict_backward']}")
    print(f"الحكم النهائي                                : {pairwise_result['final_verdict']}")
    print(f"تم اكتشاف تحيّز موضع؟                        : "
          f"{'نعم ⚠️' if pairwise_result['position_bias_detected'] else 'لا ✅'}")

    print("\n" + "=" * 70)
    print("القسم 3: معايرة الحَكَم الآلي مقابل تقييمات بشرية (محاكاة)")
    print("=" * 70)

    # محاكاة: 10 مقارنات قيّمها بشر، وقيّمها الحَكَم الآلي أيضًا
    human_verdicts = ["A", "A", "B", "TIE", "A", "B", "B", "A", "TIE", "A"]
    judge_verdicts = ["A", "A", "B", "A", "A", "B", "TIE", "A", "TIE", "B"]

    kappa = cohens_kappa(human_verdicts, judge_verdicts)
    print(f"Cohen's Kappa بين الحَكَم الآلي والبشري: {kappa:.3f}")
    print(f"التفسير: {interpret_kappa(kappa)}")

    print("\n" + "=" * 70)
    print("القسم 4: بناء تصنيف Elo من نتائج مقارنات متعددة النماذج")
    print("=" * 70)

    elo_system = EloRatingSystem(k_factor=32.0)

    # محاكاة: نتائج مقارنات ثنائية عشوائية بين 4 نماذج عبر 200 مواجهة
    rng = np.random.default_rng(7)
    models = ["Model_GPT", "Model_Claude", "Model_Llama", "Model_Mistral"]
    # قوة "حقيقية" خفية لكل نموذج (لمحاكاة نتائج واقعية متسقة إحصائيًا)
    true_strength = {"Model_GPT": 1.15, "Model_Claude": 1.05, "Model_Llama": 0.95, "Model_Mistral": 0.85}

    for _ in range(200):
        model_a, model_b = rng.choice(models, size=2, replace=False)
        p_a_wins = true_strength[model_a] / (true_strength[model_a] + true_strength[model_b])
        outcome = "A" if rng.random() < p_a_wins else "B"
        elo_system.update(model_a, model_b, outcome)

    print("التصنيف النهائي (Elo) بعد 200 مواجهة:")
    for rank, (model, rating) in enumerate(elo_system.leaderboard(), 1):
        print(f"  {rank}. {model:20s}: {rating:.1f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-47-chapter-47.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 47: الفصل 47: التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_judge() -> None:
    judge_a, judge_b = ([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1])
    observed = sum((a == b for a, b in zip(judge_a, judge_b))) / len(judge_a)
    p_a = sum(judge_a) / len(judge_a)
    p_b = sum(judge_b) / len(judge_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (observed - expected) / (1 - expected)
    print('raw agreement:', round(observed, 3))
    print("Cohen's kappa:", round(kappa, 3))
    print('resolve disagreement with a blinded human audit.')

if __name__ == "__main__":
    print("Chapter 47 lab: Measure agreement between independent model judges")
    run_judge()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-47-chapter-47.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
نظام LLM-as-a-Judge متكامل: Single-Answer Grading + Pairwise مع Position Swapping
+ قياس اتفاق مع تقييم بشري (Cohen's Kappa) + حساب تصنيف Elo
يستخدم: openai (أو أي واجهة متوافقة)، numpy
"""

import re
import json
import itertools
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from openai import OpenAI

client = OpenAI()  # يتطلب OPENAI_API_KEY في متغيرات البيئة
JUDGE_MODEL = "gpt-4o"


# ============================================================
# الجزء 1: Single-Answer Grading مع Rubric صريح
# ============================================================

GRADING_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا، مهمتك تقييم جودة إجابة مساعد ذكاء اصطناعي.

قيّم الإجابة بناءً على المعايير التالية:
- الصحة الواقعية (0-3 نقاط): هل المعلومات صحيحة وخالية من الهلوسة؟
- الوضوح والتنظيم (0-3 نقاط): هل الإجابة منظّمة وسهلة الفهم؟
- اكتمال الإجابة (0-2 نقطة): هل تغطي كل جوانب السؤال؟
- الأسلوب والنبرة (0-2 نقطة): هل الأسلوب مناسب للسياق؟

فكّر خطوة بخطوة في كل معيار على حدة أولًا، ثم اختم بالضبط بالصيغة التالية:
[[SCORE]]: X

حيث X رقم صحيح من 0 إلى 10.

[السؤال]:
{question}

[الإجابة المُقيَّمة]:
{answer}

[تحليلك]:"""


def grade_single_answer(question: str, answer: str, judge_model: str = JUDGE_MODEL) -> dict:
    """يُقيّم إجابة واحدة بمعزل عن أي منافس، عبر Rubric صريح وCoT قبل الدرجة."""
    prompt = GRADING_PROMPT_TEMPLATE.format(question=question, answer=answer)

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # حسم عالٍ مطلوب لحَكَم متّسق (انظر فصل 38)
    )
    judgment_text = response.choices[0].message.content

    match = re.search(r"\[\[SCORE\]\]:\s*(\d+)", judgment_text)
    score = int(match.group(1)) if match else None

    return {"score": score, "reasoning": judgment_text}


# ============================================================
# الجزء 2: Pairwise Comparison مع Position Swapping
# ============================================================

PAIRWISE_PROMPT_TEMPLATE = """أنت حَكَم خبير محايد تمامًا. قارن بين إجابتين لنفس السؤال وحدد أيهما أفضل.

قيّم بناءً على: الصحة الواقعية، الوضوح، الاكتمال، والأسلوب.
تجاهل تمامًا طول الإجابة أو ترتيب عرضها كمعيار — ركّز فقط على الجودة الفعلية للمحتوى.

فكّر خطوة بخطوة، ثم اختم بالضبط بالصيغة التالية:
[[VERDICT]]: A
أو
[[VERDICT]]: B
أو
[[VERDICT]]: TIE

[السؤال]:
{question}

[الإجابة A]:
{answer_a}

[الإجابة B]:
{answer_b}

[تحليلك]:"""


def _single_pairwise_call(question: str, answer_a: str, answer_b: str,
                            judge_model: str = JUDGE_MODEL) -> str:
    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    match = re.search(r"\[\[VERDICT\]\]:\s*(A|B|TIE)", text)
    return match.group(1) if match else "TIE"


def pairwise_compare_with_swapping(question: str, answer_a: str, answer_b: str,
                                     judge_model: str = JUDGE_MODEL) -> dict:
    """
    يُجري المقارنة مرتين (بالترتيب الأصلي والمعكوس) لإزالة تحيّز الموضع
    (Position Bias)، ويُعلن "تعادل" إن لم يتفق الحَكَم مع نفسه.
    """
    verdict_forward = _single_pairwise_call(question, answer_a, answer_b, judge_model)

    # في الاستدعاء المعكوس، "A" و"B" في الـ prompt تشيران لـ (answer_b, answer_a)
    verdict_backward_raw = _single_pairwise_call(question, answer_b, answer_a, judge_model)
    # نعيد التفسير لمرجعية answer_a/answer_b الأصلية
    swap_map = {"A": "B", "B": "A", "TIE": "TIE"}
    verdict_backward = swap_map[verdict_backward_raw]

    if verdict_forward == verdict_backward and verdict_forward != "TIE":
        final_verdict = verdict_forward
        is_consistent = True
    else:
        final_verdict = "TIE"
        is_consistent = (verdict_forward == verdict_backward)  # تعادل حقيقي أم تناقض؟

    return {
        "verdict_forward": verdict_forward,
        "verdict_backward": verdict_backward,
        "final_verdict": final_verdict,
        "position_bias_detected": not is_consistent,
    }


# ============================================================
# الجزء 3: قياس اتفاق الحَكَم الآلي مع تقييم بشري (Cohen's Kappa)
# ============================================================

def cohens_kappa(human_labels: list[str], judge_labels: list[str]) -> float:
    """
    يحسب Cohen's Kappa بين تصنيفين (بشري وآلي) على نفس العينات،
    طبقًا للصيغة الرياضية المُشتقة في القسم 🟡 أعلاه.
    """
    assert len(human_labels) == len(judge_labels)
    n = len(human_labels)
    categories = sorted(set(human_labels) | set(judge_labels))

    # الاتفاق الملاحَظ p_o
    p_o = sum(h == j for h, j in zip(human_labels, judge_labels)) / n

    # الاتفاق المتوقَّع بالصدفة p_e
    human_dist = {c: human_labels.count(c) / n for c in categories}
    judge_dist = {c: judge_labels.count(c) / n for c in categories}
    p_e = sum(human_dist[c] * judge_dist[c] for c in categories)

    if p_e == 1.0:
        return 1.0  # تفادي القسمة على صفر في حالة اتفاق تام مصادفة
    return (p_o - p_e) / (1 - p_e)


def interpret_kappa(kappa: float) -> str:
    """تفسير قياسي من Landis & Koch (1977)."""
    if kappa < 0:
        return "أسوأ من الصدفة ⚠️"
    elif kappa < 0.20:
        return "ضعيف جدًا"
    elif kappa < 0.40:
        return "ضعيف"
    elif kappa < 0.60:
        return "معتدل"
    elif kappa < 0.80:
        return "جوهري (مقبول للإنتاج)"
    return "شبه تام"


# ============================================================
# الجزء 4: حساب تصنيف Elo من نتائج مقارنات ثنائية متعددة
# ============================================================

@dataclass
class EloRatingSystem:
    """نظام تصنيف Elo لترتيب عدة نماذج من نتائج مقارنات ثنائية متكررة."""

    k_factor: float = 32.0
    initial_rating: float = 1000.0
    ratings: dict = field(default_factory=lambda: defaultdict(lambda: 1000.0))

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, model_a: str, model_b: str, outcome: str) -> None:
        """outcome: 'A' أو 'B' أو 'TIE' حسب نتيجة المقارنة."""
        r_a, r_b = self.ratings[model_a], self.ratings[model_b]
        e_a = self.expected_score(r_a, r_b)
        e_b = 1.0 - e_a

        score_map = {"A": (1.0, 0.0), "B": (0.0, 1.0), "TIE": (0.5, 0.5)}
        s_a, s_b = score_map[outcome]

        self.ratings[model_a] = r_a + self.k_factor * (s_a - e_a)
        self.ratings[model_b] = r_b + self.k_factor * (s_b - e_b)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ============================================================
# الجزء 5: تشغيل السيناريو الكامل
# ============================================================

if __name__ == "__main__":
    question = "اشرح لماذا تتجمد المياه عند صفر درجة مئوية بلغة بسيطة."

    answer_concise = (
        "لأن جزيئات الماء تتباطأ حركتها عند هذه الدرجة وتترتب في شكل بلوري ثابت هو الجليد."
    )
    answer_verbose = (
        "هذا سؤال رائع! دعني أشرح لك بالتفصيل. الماء يتكون من جزيئات H2O، وهذه "
        "الجزيئات في الحالة السائلة تتحرك بحرية نسبية وتتصادم باستمرار فيما بينها "
        "بفعل الطاقة الحرارية. عندما تنخفض درجة الحرارة إلى الصفر المئوي، تقل "
        "الطاقة الحركية لهذه الجزيئات بشكل كبير جدًا، مما يسمح لقوى الترابط "
        "الهيدروجيني بينها بأن تُثبّتها في شبكة بلورية منتظمة الشكل السداسي، "
        "وهذا التحوّل في الترتيب الجزيئي هو ما نسميه التجمّد أو التحوّل لحالة الجليد الصلبة."
    )

    print("=" * 70)
    print("القسم 1: Single-Answer Grading لكل إجابة بمفردها")
    print("=" * 70)
    for name, ans in [("موجزة", answer_concise), ("مُطوَّلة", answer_verbose)]:
        result = grade_single_answer(question, ans)
        print(f"\n[{name}] الدرجة: {result['score']}/10")

    print("\n" + "=" * 70)
    print("القسم 2: Pairwise Comparison مع Position Swapping")
    print("=" * 70)
    pairwise_result = pairwise_compare_with_swapping(question, answer_concise, answer_verbose)
    print(f"الحكم بالترتيب الأصلي (A=موجزة, B=مُطوَّلة)  : {pairwise_result['verdict_forward']}")
    print(f"الحكم بالترتيب المعكوس (بعد إعادة التفسير)   : {pairwise_result['verdict_backward']}")
    print(f"الحكم النهائي                                : {pairwise_result['final_verdict']}")
    print(f"تم اكتشاف تحيّز موضع؟                        : "
          f"{'نعم ⚠️' if pairwise_result['position_bias_detected'] else 'لا ✅'}")

    print("\n" + "=" * 70)
    print("القسم 3: معايرة الحَكَم الآلي مقابل تقييمات بشرية (محاكاة)")
    print("=" * 70)

    # محاكاة: 10 مقارنات قيّمها بشر، وقيّمها الحَكَم الآلي أيضًا
    human_verdicts = ["A", "A", "B", "TIE", "A", "B", "B", "A", "TIE", "A"]
    judge_verdicts = ["A", "A", "B", "A", "A", "B", "TIE", "A", "TIE", "B"]

    kappa = cohens_kappa(human_verdicts, judge_verdicts)
    print(f"Cohen's Kappa بين الحَكَم الآلي والبشري: {kappa:.3f}")
    print(f"التفسير: {interpret_kappa(kappa)}")

    print("\n" + "=" * 70)
    print("القسم 4: بناء تصنيف Elo من نتائج مقارنات متعددة النماذج")
    print("=" * 70)

    elo_system = EloRatingSystem(k_factor=32.0)

    # محاكاة: نتائج مقارنات ثنائية عشوائية بين 4 نماذج عبر 200 مواجهة
    rng = np.random.default_rng(7)
    models = ["Model_GPT", "Model_Claude", "Model_Llama", "Model_Mistral"]
    # قوة "حقيقية" خفية لكل نموذج (لمحاكاة نتائج واقعية متسقة إحصائيًا)
    true_strength = {"Model_GPT": 1.15, "Model_Claude": 1.05, "Model_Llama": 0.95, "Model_Mistral": 0.85}

    for _ in range(200):
        model_a, model_b = rng.choice(models, size=2, replace=False)
        p_a_wins = true_strength[model_a] / (true_strength[model_a] + true_strength[model_b])
        outcome = "A" if rng.random() < p_a_wins else "B"
        elo_system.update(model_a, model_b, outcome)

    print("التصنيف النهائي (Elo) بعد 200 مواجهة:")
    for rank, (model, rating) in enumerate(elo_system.leaderboard(), 1):
        print(f"  {rank}. {model:20s}: {rating:.1f}")
