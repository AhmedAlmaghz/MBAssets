#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق
Large Language Model Engineering — Chapter 20 — Build An Llm

Book slug : llm-engineering-69-chapters
Chapter   : ch-20-chapter-20 (lesson 20)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-20-chapter-20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق
Large Language Model Engineering — Chapter 20 — Build An Llm

Book slug : llm-engineering-69-chapters
Chapter   : ch-20-chapter-20 (lesson 20)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-20-chapter-20.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق
Large Language Model Engineering — Chapter 20 — Build An Llm

Book slug : llm-engineering-69-chapters
Chapter   : ch-20-chapter-20 (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 20: الفصل العشرون: خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_data_pipeline() -> None:
    raw = ['  A useful note. ', 'A useful note.', 'BUY NOW!!!', 'The model learns.', '']
    cleaned = []
    seen = set()
    for text in raw:
        text = re.sub('\\s+', ' ', text).strip().lower()
        text = re.sub('[!]{2,}', '!', text)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    print('raw / retained:', len(raw), '/', len(cleaned))
    print('deduplicated documents:', cleaned)
    print('split by document before tokenization to prevent leakage.')

if __name__ == "__main__":
    print("Chapter 20 lab: A reproducible text-data cleaning pipeline")
    run_data_pipeline()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter20_data_pipeline.py
خط أنابيب بيانات مصغّر يطبّق: تصفية بالقواعد، كشف اللغة المبسّط،
وإزالة التكرار شبه المتطابق عبر MinHash، على نصوص عربية تجريبية.

Python 3.13+
المتطلبات: pip install datasketch
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH


# =========================================================================
# 1) بيانات تجريبية: مزيج من نصوص نظيفة، فاسدة، ومكررة عمدًا
# =========================================================================
SAMPLE_DOCUMENTS: list[str] = [
    # وثيقة نظيفة 1
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرار.",

    # وثيقة نظيفة 2 (مختلفة تمامًا)
    "تُعتبر اللغة العربية من أقدم اللغات السامية المستخدمة حتى اليوم، "
    "وتتميز بثراء مفرداتها وتنوع لهجاتها عبر الوطن العربي.",

    # وثيقة شبه مكررة من الوثيقة 1 (تغيير بسيط فقط)
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرارات.",

    # وثيقة قصيرة جدًا (يجب استبعادها بفلتر الطول)
    "مرحبا بك",

    # وثيقة بها تكرار داخلي مفرط (نمط Spam)
    "اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن أفضل سعر",

    # وثيقة بنسبة عالية من الرموز غير الأبجدية (ضجيج/ترميز معطوب)
    "###$$$ 123456 !!! @@@ %%% ***  ### $$$ 999999 !!!",

    # وثيقة مختلطة اللغة بشكل كبير جدًا (إنجليزية بالكامل تقريبًا، لمجموعة عربية)
    "This is a completely English document that should not belong "
    "to an Arabic-focused training corpus at all.",

    # وثيقة نظيفة 3
    "التحول الرقمي أصبح ضرورة ملحة لكل المؤسسات الحديثة، حيث يُساهم "
    "في تحسين الكفاءة التشغيلية وتعزيز تجربة العملاء بشكل ملحوظ جدًا.",
]


# =========================================================================
# 2) الطبقة الأولى: فلاتر قائمة على القواعد (Rule-Based Heuristics)
# =========================================================================
@dataclass
class QualityFilterConfig:
    min_words: int = 5
    max_words: int = 100_000
    max_non_alpha_ratio: float = 0.3
    max_repeated_word_ratio: float = 0.3


def count_words(text: str) -> int:
    return len(text.split())


def non_alpha_character_ratio(text: str, arabic_pattern: str = r"[\u0600-\u06FF\s]") -> float:
    """نسبة الرموز التي ليست أحرفًا عربية ولا مسافات."""
    total = len(text)
    if total == 0:
        return 1.0
    non_alpha = len(re.sub(arabic_pattern, "", text))
    return non_alpha / total


def repeated_word_ratio(text: str) -> float:
    """نسبة أكبر كلمة متكررة إلى إجمالي الكلمات (كشف نمط Spam)."""
    words = text.split()
    if not words:
        return 1.0
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words)


def passes_quality_filters(
    text: str, config: QualityFilterConfig
) -> tuple[bool, str]:
    """تُعيد (مقبول؟, سبب الرفض إن وُجد)."""
    n_words = count_words(text)
    if n_words < config.min_words:
        return False, f"قصير جدًا ({n_words} كلمة)"
    if n_words > config.max_words:
        return False, f"طويل جدًا ({n_words} كلمة)"

    non_alpha_ratio = non_alpha_character_ratio(text)
    if non_alpha_ratio > config.max_non_alpha_ratio:
        return False, f"نسبة رموز غير أبجدية عالية ({non_alpha_ratio:.2%})"

    rep_ratio = repeated_word_ratio(text)
    if rep_ratio > config.max_repeated_word_ratio:
        return False, f"تكرار داخلي مفرط ({rep_ratio:.2%})"

    return True, ""


# =========================================================================
# 3) الطبقة الثانية: كشف اللغة المبسّط (heuristic بديل عن fastText للتوضيح)
# =========================================================================
def arabic_character_ratio(text: str) -> float:
    """نسبة الأحرف العربية الفعلية إلى إجمالي الأحرف غير الفراغية."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    non_space_chars = len(re.sub(r"\s", "", text))
    if non_space_chars == 0:
        return 0.0
    return arabic_chars / non_space_chars


def is_arabic_document(text: str, threshold: float = 0.5) -> bool:
    return arabic_character_ratio(text) >= threshold


# =========================================================================
# 4) الطبقة الثالثة: إزالة التكرار شبه المتطابق عبر MinHash + LSH
# =========================================================================
def get_shingles(text: str, k: int = 4) -> set[str]:
    """تقسيم النص إلى shingles (n-grams على مستوى الكلمات)."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate_near_duplicates(
    documents: list[str], threshold: float = 0.8, num_perm: int = 128
) -> list[int]:
    """
    تُعيد قائمة بمؤشرات (indices) الوثائق التي يجب الاحتفاظ بها
    بعد إزالة التكرار شبه المتطابق.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    keep_indices: list[int] = []
    seen_duplicate_of: dict[int, int] = {}

    for idx, doc in enumerate(documents):
        mh = build_minhash(doc, num_perm=num_perm)
        minhashes[idx] = mh

        # ابحث عن وثائق مشابهة موجودة مسبقًا في LSH
        similar = lsh.query(mh)
        if similar:
            # وُجد تطابق شبه كامل مع وثيقة سابقة -> نعتبرها مكررة، نتجاهلها
            seen_duplicate_of[idx] = similar[0]
            continue

        # لا يوجد تطابق -> وثيقة جديدة فريدة، نُضيفها لـ LSH ونحتفظ بها
        lsh.insert(str(idx), mh)
        keep_indices.append(idx)

    return keep_indices


# =========================================================================
# 5) خط الأنابيب الكامل (End-to-End)
# =========================================================================
def run_full_pipeline(documents: list[str]) -> list[str]:
    print(f"عدد الوثائق الأصلي: {len(documents)}\n")

    # --- المرحلة 1: تصفية الجودة بالقواعد ---
    config = QualityFilterConfig()
    stage1_passed: list[str] = []
    for i, doc in enumerate(documents):
        ok, reason = passes_quality_filters(doc, config)
        status = "✅ قُبلت" if ok else f"❌ رُفضت ({reason})"
        print(f"[جودة] وثيقة {i}: {status}")
        if ok:
            stage1_passed.append(doc)

    print(f"\nبعد تصفية الجودة: {len(stage1_passed)} / {len(documents)} وثيقة\n")

    # --- المرحلة 2: كشف اللغة ---
    stage2_passed: list[str] = []
    for i, doc in enumerate(stage1_passed):
        is_ar = is_arabic_document(doc)
        status = "✅ عربية" if is_ar else "❌ ليست عربية بما يكفي"
        print(f"[لغة] وثيقة {i}: {status}")
        if is_ar:
            stage2_passed.append(doc)

    print(f"\nبعد فلترة اللغة: {len(stage2_passed)} / {len(stage1_passed)} وثيقة\n")

    # --- المرحلة 3: إزالة التكرار شبه المتطابق ---
    keep_idx = deduplicate_near_duplicates(stage2_passed, threshold=0.7)
    final_docs = [stage2_passed[i] for i in keep_idx]

    print(f"[تكرار] المؤشرات المحتفظ بها بعد إزالة التكرار: {keep_idx}")
    print(f"\nبعد إزالة التكرار: {len(final_docs)} / {len(stage2_passed)} وثيقة\n")

    print("=" * 60)
    print(f"الخلاصة النهائية: من {len(documents)} وثيقة خام، "
          f"نجت {len(final_docs)} وثيقة فقط "
          f"({len(final_docs) / len(documents):.1%})")
    print("=" * 60)

    return final_docs


if __name__ == "__main__":
    clean_corpus = run_full_pipeline(SAMPLE_DOCUMENTS)
    print("\nالوثائق النهائية النظيفة:")
    for doc in clean_corpus:
        print(f"  - {doc[:60]}...")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عتبات طول الكلمة المصمَّمة للإنجليزية لا تناسب العربية
# (العربية لغة معقدة صرفيًا، الكلمة الواحدة قد تحمل معنى جملة كاملة بالإنجليزية)
def wrong_avg_word_length_filter(text: str) -> bool:
    avg_len = sum(len(w) for w in text.split()) / len(text.split())
    return 3 <= avg_len <= 10  # عتبات مأخوذة من دراسات إنجليزية مباشرة

# ✅ صحيح: احسب إحصائيات اللغة العربية فعليًا من عيّنة موثوقة (مثل ويكيبيديا العربية)
# وابنِ عتباتك الخاصة بناءً عليها، لا تفترض انتقال العتبات بين اللغات

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الفروقات في التشكيل، الهمزات، أو المسافات الزائدة
# تجعل MinHash يفشل في اكتشاف تكرار واضح للعين البشرية
text_a = "الذكاء الإصطناعي رائع"
text_b = "الذكاء الاصطناعي رائع"  # فرق بسيط في الهمزة

# ✅ صحيح: طبّق تطبيعًا نصيًا (توحيد الهمزات، إزالة التشكيل، توحيد المسافات)
# قبل حساب shingles و MinHash
import re

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)       # توحيد الألفات
    text = re.sub(r"ى", "ي", text)             # توحيد الألف المقصورة
    text = re.sub(r"ة", "ه", text)             # توحيد التاء المربوطة (اختياري حسب الحاجة)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"\s+", " ", text).strip()   # توحيد المسافات
    return text

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: فلاتر صارمة جدًا تحذف كل شيء ليس "فصحى معيارية مثالية"
# (تحذف أمثال شعبية، اقتباسات بلهجات محلية، شعرًا بأساليب غير تقليدية)

# ✅ صحيح: ميّز بين "منخفض الجودة فعليًا" (Spam، نص معطوب) و"متنوع أسلوبيًا"
# (لهجات، شعر، أمثال) - الأخير قيّم جدًا ويجب الاحتفاظ به بنسبة مدروسة
# وليس حذفه بالكامل، للحفاظ على تنوع النموذج اللغوي

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق سلسلة طويلة من الفلاتر "لأنها تبدو منطقية نظريًا"
# دون التحقق من أنها فعلًا تُحسّن أداء نموذج مُدرَّب على البيانات الناتجة

# ✅ صحيح (منهجية FineWeb): درّب نموذج صغير (حتى لو 100-500 مليون معامل)
# على عيّنة من البيانات قبل وبعد كل فلتر، وقارن الأداء على معيار قياسي
# قبل اعتماد الفلتر نهائيًا على مليارات الوثائق

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_real_arabic_pipeline.py
تطبيق خط أنابيب البيانات الكامل على عيّنة حقيقية من ويكيبيديا العربية
أو مجموعة بيانات عربية مفتوحة من Hugging Face Datasets.

المطلوب إكماله:
1. حمّل عيّنة (10,000 وثيقة مثلًا) من "wikimedia/wikipedia" (تهيئة "20231101.ar")
   عبر مكتبة `datasets` من Hugging Face.
2. طبّق المرحلة 1 (فلاتر الجودة بالقواعد) من هذا الفصل، مع تعديل العتبات
   لتناسب طبيعة نصوص ويكيبيديا (مقالات أطول عادة من نصوص الويب العامة).
3. طبّق تطبيعًا نصيًا عربيًا (توحيد الهمزات، إزالة التشكيل) قبل أي مقارنة.
4. طبّق إزالة التكرار شبه المتطابق بـ MinHash+LSH على كامل العيّنة.
5. احسب واطبع إحصائيات نهائية: نسبة البيانات الناجية من كل مرحلة،
   متوسط طول الوثيقة قبل وبعد، وعدد التكرارات المكتشفة.
6. (تحدٍّ إضافي) درّب نموذج Word2Vec بسيط (من الفصل 7!) على النسخة
   الخام مقابل النسخة المُصفّاة، وقارن جودة أقرب الجيران لكلمات مُختارة
   كدليل عملي بسيط على أثر التصفية.
"""
from __future__ import annotations

from datasets import load_dataset


def load_sample_arabic_wikipedia(n_samples: int = 10_000) -> list[str]:
    # TODO: استخدم load_dataset("wikimedia/wikipedia", "20231101.ar", streaming=True)
    # وخذ أول n_samples وثيقة من حقل "text"
    raise NotImplementedError


def main() -> None:
    raw_docs = load_sample_arabic_wikipedia(10_000)
    # TODO: طبّق كامل خط الأنابيب من هذا الفصل خطوة بخطوة
    # TODO: اطبع تقرير إحصائي نهائي شامل
    pass


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 20: الفصل العشرون: خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_data_pipeline() -> None:
    raw = ['  A useful note. ', 'A useful note.', 'BUY NOW!!!', 'The model learns.', '']
    cleaned = []
    seen = set()
    for text in raw:
        text = re.sub('\\s+', ' ', text).strip().lower()
        text = re.sub('[!]{2,}', '!', text)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    print('raw / retained:', len(raw), '/', len(cleaned))
    print('deduplicated documents:', cleaned)
    print('split by document before tokenization to prevent leakage.')

if __name__ == "__main__":
    print("Chapter 20 lab: A reproducible text-data cleaning pipeline")
    run_data_pipeline()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter20_data_pipeline.py
خط أنابيب بيانات مصغّر يطبّق: تصفية بالقواعد، كشف اللغة المبسّط،
وإزالة التكرار شبه المتطابق عبر MinHash، على نصوص عربية تجريبية.

Python 3.13+
المتطلبات: pip install datasketch
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH


# =========================================================================
# 1) بيانات تجريبية: مزيج من نصوص نظيفة، فاسدة، ومكررة عمدًا
# =========================================================================
SAMPLE_DOCUMENTS: list[str] = [
    # وثيقة نظيفة 1
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرار.",

    # وثيقة نظيفة 2 (مختلفة تمامًا)
    "تُعتبر اللغة العربية من أقدم اللغات السامية المستخدمة حتى اليوم، "
    "وتتميز بثراء مفرداتها وتنوع لهجاتها عبر الوطن العربي.",

    # وثيقة شبه مكررة من الوثيقة 1 (تغيير بسيط فقط)
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرارات.",

    # وثيقة قصيرة جدًا (يجب استبعادها بفلتر الطول)
    "مرحبا بك",

    # وثيقة بها تكرار داخلي مفرط (نمط Spam)
    "اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن أفضل سعر",

    # وثيقة بنسبة عالية من الرموز غير الأبجدية (ضجيج/ترميز معطوب)
    "###$$$ 123456 !!! @@@ %%% ***  ### $$$ 999999 !!!",

    # وثيقة مختلطة اللغة بشكل كبير جدًا (إنجليزية بالكامل تقريبًا، لمجموعة عربية)
    "This is a completely English document that should not belong "
    "to an Arabic-focused training corpus at all.",

    # وثيقة نظيفة 3
    "التحول الرقمي أصبح ضرورة ملحة لكل المؤسسات الحديثة، حيث يُساهم "
    "في تحسين الكفاءة التشغيلية وتعزيز تجربة العملاء بشكل ملحوظ جدًا.",
]


# =========================================================================
# 2) الطبقة الأولى: فلاتر قائمة على القواعد (Rule-Based Heuristics)
# =========================================================================
@dataclass
class QualityFilterConfig:
    min_words: int = 5
    max_words: int = 100_000
    max_non_alpha_ratio: float = 0.3
    max_repeated_word_ratio: float = 0.3


def count_words(text: str) -> int:
    return len(text.split())


def non_alpha_character_ratio(text: str, arabic_pattern: str = r"[\u0600-\u06FF\s]") -> float:
    """نسبة الرموز التي ليست أحرفًا عربية ولا مسافات."""
    total = len(text)
    if total == 0:
        return 1.0
    non_alpha = len(re.sub(arabic_pattern, "", text))
    return non_alpha / total


def repeated_word_ratio(text: str) -> float:
    """نسبة أكبر كلمة متكررة إلى إجمالي الكلمات (كشف نمط Spam)."""
    words = text.split()
    if not words:
        return 1.0
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words)


def passes_quality_filters(
    text: str, config: QualityFilterConfig
) -> tuple[bool, str]:
    """تُعيد (مقبول؟, سبب الرفض إن وُجد)."""
    n_words = count_words(text)
    if n_words < config.min_words:
        return False, f"قصير جدًا ({n_words} كلمة)"
    if n_words > config.max_words:
        return False, f"طويل جدًا ({n_words} كلمة)"

    non_alpha_ratio = non_alpha_character_ratio(text)
    if non_alpha_ratio > config.max_non_alpha_ratio:
        return False, f"نسبة رموز غير أبجدية عالية ({non_alpha_ratio:.2%})"

    rep_ratio = repeated_word_ratio(text)
    if rep_ratio > config.max_repeated_word_ratio:
        return False, f"تكرار داخلي مفرط ({rep_ratio:.2%})"

    return True, ""


# =========================================================================
# 3) الطبقة الثانية: كشف اللغة المبسّط (heuristic بديل عن fastText للتوضيح)
# =========================================================================
def arabic_character_ratio(text: str) -> float:
    """نسبة الأحرف العربية الفعلية إلى إجمالي الأحرف غير الفراغية."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    non_space_chars = len(re.sub(r"\s", "", text))
    if non_space_chars == 0:
        return 0.0
    return arabic_chars / non_space_chars


def is_arabic_document(text: str, threshold: float = 0.5) -> bool:
    return arabic_character_ratio(text) >= threshold


# =========================================================================
# 4) الطبقة الثالثة: إزالة التكرار شبه المتطابق عبر MinHash + LSH
# =========================================================================
def get_shingles(text: str, k: int = 4) -> set[str]:
    """تقسيم النص إلى shingles (n-grams على مستوى الكلمات)."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate_near_duplicates(
    documents: list[str], threshold: float = 0.8, num_perm: int = 128
) -> list[int]:
    """
    تُعيد قائمة بمؤشرات (indices) الوثائق التي يجب الاحتفاظ بها
    بعد إزالة التكرار شبه المتطابق.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    keep_indices: list[int] = []
    seen_duplicate_of: dict[int, int] = {}

    for idx, doc in enumerate(documents):
        mh = build_minhash(doc, num_perm=num_perm)
        minhashes[idx] = mh

        # ابحث عن وثائق مشابهة موجودة مسبقًا في LSH
        similar = lsh.query(mh)
        if similar:
            # وُجد تطابق شبه كامل مع وثيقة سابقة -> نعتبرها مكررة، نتجاهلها
            seen_duplicate_of[idx] = similar[0]
            continue

        # لا يوجد تطابق -> وثيقة جديدة فريدة، نُضيفها لـ LSH ونحتفظ بها
        lsh.insert(str(idx), mh)
        keep_indices.append(idx)

    return keep_indices


# =========================================================================
# 5) خط الأنابيب الكامل (End-to-End)
# =========================================================================
def run_full_pipeline(documents: list[str]) -> list[str]:
    print(f"عدد الوثائق الأصلي: {len(documents)}\n")

    # --- المرحلة 1: تصفية الجودة بالقواعد ---
    config = QualityFilterConfig()
    stage1_passed: list[str] = []
    for i, doc in enumerate(documents):
        ok, reason = passes_quality_filters(doc, config)
        status = "✅ قُبلت" if ok else f"❌ رُفضت ({reason})"
        print(f"[جودة] وثيقة {i}: {status}")
        if ok:
            stage1_passed.append(doc)

    print(f"\nبعد تصفية الجودة: {len(stage1_passed)} / {len(documents)} وثيقة\n")

    # --- المرحلة 2: كشف اللغة ---
    stage2_passed: list[str] = []
    for i, doc in enumerate(stage1_passed):
        is_ar = is_arabic_document(doc)
        status = "✅ عربية" if is_ar else "❌ ليست عربية بما يكفي"
        print(f"[لغة] وثيقة {i}: {status}")
        if is_ar:
            stage2_passed.append(doc)

    print(f"\nبعد فلترة اللغة: {len(stage2_passed)} / {len(stage1_passed)} وثيقة\n")

    # --- المرحلة 3: إزالة التكرار شبه المتطابق ---
    keep_idx = deduplicate_near_duplicates(stage2_passed, threshold=0.7)
    final_docs = [stage2_passed[i] for i in keep_idx]

    print(f"[تكرار] المؤشرات المحتفظ بها بعد إزالة التكرار: {keep_idx}")
    print(f"\nبعد إزالة التكرار: {len(final_docs)} / {len(stage2_passed)} وثيقة\n")

    print("=" * 60)
    print(f"الخلاصة النهائية: من {len(documents)} وثيقة خام، "
          f"نجت {len(final_docs)} وثيقة فقط "
          f"({len(final_docs) / len(documents):.1%})")
    print("=" * 60)

    return final_docs


if __name__ == "__main__":
    clean_corpus = run_full_pipeline(SAMPLE_DOCUMENTS)
    print("\nالوثائق النهائية النظيفة:")
    for doc in clean_corpus:
        print(f"  - {doc[:60]}...")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عتبات طول الكلمة المصمَّمة للإنجليزية لا تناسب العربية
# (العربية لغة معقدة صرفيًا، الكلمة الواحدة قد تحمل معنى جملة كاملة بالإنجليزية)
def wrong_avg_word_length_filter(text: str) -> bool:
    avg_len = sum(len(w) for w in text.split()) / len(text.split())
    return 3 <= avg_len <= 10  # عتبات مأخوذة من دراسات إنجليزية مباشرة

# ✅ صحيح: احسب إحصائيات اللغة العربية فعليًا من عيّنة موثوقة (مثل ويكيبيديا العربية)
# وابنِ عتباتك الخاصة بناءً عليها، لا تفترض انتقال العتبات بين اللغات

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الفروقات في التشكيل، الهمزات، أو المسافات الزائدة
# تجعل MinHash يفشل في اكتشاف تكرار واضح للعين البشرية
text_a = "الذكاء الإصطناعي رائع"
text_b = "الذكاء الاصطناعي رائع"  # فرق بسيط في الهمزة

# ✅ صحيح: طبّق تطبيعًا نصيًا (توحيد الهمزات، إزالة التشكيل، توحيد المسافات)
# قبل حساب shingles و MinHash
import re

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)       # توحيد الألفات
    text = re.sub(r"ى", "ي", text)             # توحيد الألف المقصورة
    text = re.sub(r"ة", "ه", text)             # توحيد التاء المربوطة (اختياري حسب الحاجة)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"\s+", " ", text).strip()   # توحيد المسافات
    return text

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: فلاتر صارمة جدًا تحذف كل شيء ليس "فصحى معيارية مثالية"
# (تحذف أمثال شعبية، اقتباسات بلهجات محلية، شعرًا بأساليب غير تقليدية)

# ✅ صحيح: ميّز بين "منخفض الجودة فعليًا" (Spam، نص معطوب) و"متنوع أسلوبيًا"
# (لهجات، شعر، أمثال) - الأخير قيّم جدًا ويجب الاحتفاظ به بنسبة مدروسة
# وليس حذفه بالكامل، للحفاظ على تنوع النموذج اللغوي

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق سلسلة طويلة من الفلاتر "لأنها تبدو منطقية نظريًا"
# دون التحقق من أنها فعلًا تُحسّن أداء نموذج مُدرَّب على البيانات الناتجة

# ✅ صحيح (منهجية FineWeb): درّب نموذج صغير (حتى لو 100-500 مليون معامل)
# على عيّنة من البيانات قبل وبعد كل فلتر، وقارن الأداء على معيار قياسي
# قبل اعتماد الفلتر نهائيًا على مليارات الوثائق

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_real_arabic_pipeline.py
تطبيق خط أنابيب البيانات الكامل على عيّنة حقيقية من ويكيبيديا العربية
أو مجموعة بيانات عربية مفتوحة من Hugging Face Datasets.

المطلوب إكماله:
1. حمّل عيّنة (10,000 وثيقة مثلًا) من "wikimedia/wikipedia" (تهيئة "20231101.ar")
   عبر مكتبة `datasets` من Hugging Face.
2. طبّق المرحلة 1 (فلاتر الجودة بالقواعد) من هذا الفصل، مع تعديل العتبات
   لتناسب طبيعة نصوص ويكيبيديا (مقالات أطول عادة من نصوص الويب العامة).
3. طبّق تطبيعًا نصيًا عربيًا (توحيد الهمزات، إزالة التشكيل) قبل أي مقارنة.
4. طبّق إزالة التكرار شبه المتطابق بـ MinHash+LSH على كامل العيّنة.
5. احسب واطبع إحصائيات نهائية: نسبة البيانات الناجية من كل مرحلة،
   متوسط طول الوثيقة قبل وبعد، وعدد التكرارات المكتشفة.
6. (تحدٍّ إضافي) درّب نموذج Word2Vec بسيط (من الفصل 7!) على النسخة
   الخام مقابل النسخة المُصفّاة، وقارن جودة أقرب الجيران لكلمات مُختارة
   كدليل عملي بسيط على أثر التصفية.
"""
from __future__ import annotations

from datasets import load_dataset


def load_sample_arabic_wikipedia(n_samples: int = 10_000) -> list[str]:
    # TODO: استخدم load_dataset("wikimedia/wikipedia", "20231101.ar", streaming=True)
    # وخذ أول n_samples وثيقة من حقل "text"
    raise NotImplementedError


def main() -> None:
    raw_docs = load_sample_arabic_wikipedia(10_000)
    # TODO: طبّق كامل خط الأنابيب من هذا الفصل خطوة بخطوة
    # TODO: اطبع تقرير إحصائي نهائي شامل
    pass


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق
Large Language Model Engineering — Chapter 20 — Build An Llm

Book slug : llm-engineering-69-chapters
Chapter   : ch-20-chapter-20 (lesson 20)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_20.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 20: الفصل العشرون: خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_data_pipeline() -> None:
    raw = ['  A useful note. ', 'A useful note.', 'BUY NOW!!!', 'The model learns.', '']
    cleaned = []
    seen = set()
    for text in raw:
        text = re.sub('\\s+', ' ', text).strip().lower()
        text = re.sub('[!]{2,}', '!', text)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    print('raw / retained:', len(raw), '/', len(cleaned))
    print('deduplicated documents:', cleaned)
    print('split by document before tokenization to prevent leakage.')

if __name__ == "__main__":
    print("Chapter 20 lab: A reproducible text-data cleaning pipeline")
    run_data_pipeline()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter20_data_pipeline.py
خط أنابيب بيانات مصغّر يطبّق: تصفية بالقواعد، كشف اللغة المبسّط،
وإزالة التكرار شبه المتطابق عبر MinHash، على نصوص عربية تجريبية.

Python 3.13+
المتطلبات: pip install datasketch
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH


# =========================================================================
# 1) بيانات تجريبية: مزيج من نصوص نظيفة، فاسدة، ومكررة عمدًا
# =========================================================================
SAMPLE_DOCUMENTS: list[str] = [
    # وثيقة نظيفة 1
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرار.",

    # وثيقة نظيفة 2 (مختلفة تمامًا)
    "تُعتبر اللغة العربية من أقدم اللغات السامية المستخدمة حتى اليوم، "
    "وتتميز بثراء مفرداتها وتنوع لهجاتها عبر الوطن العربي.",

    # وثيقة شبه مكررة من الوثيقة 1 (تغيير بسيط فقط)
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرارات.",

    # وثيقة قصيرة جدًا (يجب استبعادها بفلتر الطول)
    "مرحبا بك",

    # وثيقة بها تكرار داخلي مفرط (نمط Spam)
    "اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن أفضل سعر",

    # وثيقة بنسبة عالية من الرموز غير الأبجدية (ضجيج/ترميز معطوب)
    "###$$$ 123456 !!! @@@ %%% ***  ### $$$ 999999 !!!",

    # وثيقة مختلطة اللغة بشكل كبير جدًا (إنجليزية بالكامل تقريبًا، لمجموعة عربية)
    "This is a completely English document that should not belong "
    "to an Arabic-focused training corpus at all.",

    # وثيقة نظيفة 3
    "التحول الرقمي أصبح ضرورة ملحة لكل المؤسسات الحديثة، حيث يُساهم "
    "في تحسين الكفاءة التشغيلية وتعزيز تجربة العملاء بشكل ملحوظ جدًا.",
]


# =========================================================================
# 2) الطبقة الأولى: فلاتر قائمة على القواعد (Rule-Based Heuristics)
# =========================================================================
@dataclass
class QualityFilterConfig:
    min_words: int = 5
    max_words: int = 100_000
    max_non_alpha_ratio: float = 0.3
    max_repeated_word_ratio: float = 0.3


def count_words(text: str) -> int:
    return len(text.split())


def non_alpha_character_ratio(text: str, arabic_pattern: str = r"[\u0600-\u06FF\s]") -> float:
    """نسبة الرموز التي ليست أحرفًا عربية ولا مسافات."""
    total = len(text)
    if total == 0:
        return 1.0
    non_alpha = len(re.sub(arabic_pattern, "", text))
    return non_alpha / total


def repeated_word_ratio(text: str) -> float:
    """نسبة أكبر كلمة متكررة إلى إجمالي الكلمات (كشف نمط Spam)."""
    words = text.split()
    if not words:
        return 1.0
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words)


def passes_quality_filters(
    text: str, config: QualityFilterConfig
) -> tuple[bool, str]:
    """تُعيد (مقبول؟, سبب الرفض إن وُجد)."""
    n_words = count_words(text)
    if n_words < config.min_words:
        return False, f"قصير جدًا ({n_words} كلمة)"
    if n_words > config.max_words:
        return False, f"طويل جدًا ({n_words} كلمة)"

    non_alpha_ratio = non_alpha_character_ratio(text)
    if non_alpha_ratio > config.max_non_alpha_ratio:
        return False, f"نسبة رموز غير أبجدية عالية ({non_alpha_ratio:.2%})"

    rep_ratio = repeated_word_ratio(text)
    if rep_ratio > config.max_repeated_word_ratio:
        return False, f"تكرار داخلي مفرط ({rep_ratio:.2%})"

    return True, ""


# =========================================================================
# 3) الطبقة الثانية: كشف اللغة المبسّط (heuristic بديل عن fastText للتوضيح)
# =========================================================================
def arabic_character_ratio(text: str) -> float:
    """نسبة الأحرف العربية الفعلية إلى إجمالي الأحرف غير الفراغية."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    non_space_chars = len(re.sub(r"\s", "", text))
    if non_space_chars == 0:
        return 0.0
    return arabic_chars / non_space_chars


def is_arabic_document(text: str, threshold: float = 0.5) -> bool:
    return arabic_character_ratio(text) >= threshold


# =========================================================================
# 4) الطبقة الثالثة: إزالة التكرار شبه المتطابق عبر MinHash + LSH
# =========================================================================
def get_shingles(text: str, k: int = 4) -> set[str]:
    """تقسيم النص إلى shingles (n-grams على مستوى الكلمات)."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate_near_duplicates(
    documents: list[str], threshold: float = 0.8, num_perm: int = 128
) -> list[int]:
    """
    تُعيد قائمة بمؤشرات (indices) الوثائق التي يجب الاحتفاظ بها
    بعد إزالة التكرار شبه المتطابق.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    keep_indices: list[int] = []
    seen_duplicate_of: dict[int, int] = {}

    for idx, doc in enumerate(documents):
        mh = build_minhash(doc, num_perm=num_perm)
        minhashes[idx] = mh

        # ابحث عن وثائق مشابهة موجودة مسبقًا في LSH
        similar = lsh.query(mh)
        if similar:
            # وُجد تطابق شبه كامل مع وثيقة سابقة -> نعتبرها مكررة، نتجاهلها
            seen_duplicate_of[idx] = similar[0]
            continue

        # لا يوجد تطابق -> وثيقة جديدة فريدة، نُضيفها لـ LSH ونحتفظ بها
        lsh.insert(str(idx), mh)
        keep_indices.append(idx)

    return keep_indices


# =========================================================================
# 5) خط الأنابيب الكامل (End-to-End)
# =========================================================================
def run_full_pipeline(documents: list[str]) -> list[str]:
    print(f"عدد الوثائق الأصلي: {len(documents)}\n")

    # --- المرحلة 1: تصفية الجودة بالقواعد ---
    config = QualityFilterConfig()
    stage1_passed: list[str] = []
    for i, doc in enumerate(documents):
        ok, reason = passes_quality_filters(doc, config)
        status = "✅ قُبلت" if ok else f"❌ رُفضت ({reason})"
        print(f"[جودة] وثيقة {i}: {status}")
        if ok:
            stage1_passed.append(doc)

    print(f"\nبعد تصفية الجودة: {len(stage1_passed)} / {len(documents)} وثيقة\n")

    # --- المرحلة 2: كشف اللغة ---
    stage2_passed: list[str] = []
    for i, doc in enumerate(stage1_passed):
        is_ar = is_arabic_document(doc)
        status = "✅ عربية" if is_ar else "❌ ليست عربية بما يكفي"
        print(f"[لغة] وثيقة {i}: {status}")
        if is_ar:
            stage2_passed.append(doc)

    print(f"\nبعد فلترة اللغة: {len(stage2_passed)} / {len(stage1_passed)} وثيقة\n")

    # --- المرحلة 3: إزالة التكرار شبه المتطابق ---
    keep_idx = deduplicate_near_duplicates(stage2_passed, threshold=0.7)
    final_docs = [stage2_passed[i] for i in keep_idx]

    print(f"[تكرار] المؤشرات المحتفظ بها بعد إزالة التكرار: {keep_idx}")
    print(f"\nبعد إزالة التكرار: {len(final_docs)} / {len(stage2_passed)} وثيقة\n")

    print("=" * 60)
    print(f"الخلاصة النهائية: من {len(documents)} وثيقة خام، "
          f"نجت {len(final_docs)} وثيقة فقط "
          f"({len(final_docs) / len(documents):.1%})")
    print("=" * 60)

    return final_docs


if __name__ == "__main__":
    clean_corpus = run_full_pipeline(SAMPLE_DOCUMENTS)
    print("\nالوثائق النهائية النظيفة:")
    for doc in clean_corpus:
        print(f"  - {doc[:60]}...")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عتبات طول الكلمة المصمَّمة للإنجليزية لا تناسب العربية
# (العربية لغة معقدة صرفيًا، الكلمة الواحدة قد تحمل معنى جملة كاملة بالإنجليزية)
def wrong_avg_word_length_filter(text: str) -> bool:
    avg_len = sum(len(w) for w in text.split()) / len(text.split())
    return 3 <= avg_len <= 10  # عتبات مأخوذة من دراسات إنجليزية مباشرة

# ✅ صحيح: احسب إحصائيات اللغة العربية فعليًا من عيّنة موثوقة (مثل ويكيبيديا العربية)
# وابنِ عتباتك الخاصة بناءً عليها، لا تفترض انتقال العتبات بين اللغات

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الفروقات في التشكيل، الهمزات، أو المسافات الزائدة
# تجعل MinHash يفشل في اكتشاف تكرار واضح للعين البشرية
text_a = "الذكاء الإصطناعي رائع"
text_b = "الذكاء الاصطناعي رائع"  # فرق بسيط في الهمزة

# ✅ صحيح: طبّق تطبيعًا نصيًا (توحيد الهمزات، إزالة التشكيل، توحيد المسافات)
# قبل حساب shingles و MinHash
import re

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)       # توحيد الألفات
    text = re.sub(r"ى", "ي", text)             # توحيد الألف المقصورة
    text = re.sub(r"ة", "ه", text)             # توحيد التاء المربوطة (اختياري حسب الحاجة)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"\s+", " ", text).strip()   # توحيد المسافات
    return text

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: فلاتر صارمة جدًا تحذف كل شيء ليس "فصحى معيارية مثالية"
# (تحذف أمثال شعبية، اقتباسات بلهجات محلية، شعرًا بأساليب غير تقليدية)

# ✅ صحيح: ميّز بين "منخفض الجودة فعليًا" (Spam، نص معطوب) و"متنوع أسلوبيًا"
# (لهجات، شعر، أمثال) - الأخير قيّم جدًا ويجب الاحتفاظ به بنسبة مدروسة
# وليس حذفه بالكامل، للحفاظ على تنوع النموذج اللغوي

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق سلسلة طويلة من الفلاتر "لأنها تبدو منطقية نظريًا"
# دون التحقق من أنها فعلًا تُحسّن أداء نموذج مُدرَّب على البيانات الناتجة

# ✅ صحيح (منهجية FineWeb): درّب نموذج صغير (حتى لو 100-500 مليون معامل)
# على عيّنة من البيانات قبل وبعد كل فلتر، وقارن الأداء على معيار قياسي
# قبل اعتماد الفلتر نهائيًا على مليارات الوثائق

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-20.md
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_real_arabic_pipeline.py
تطبيق خط أنابيب البيانات الكامل على عيّنة حقيقية من ويكيبيديا العربية
أو مجموعة بيانات عربية مفتوحة من Hugging Face Datasets.

المطلوب إكماله:
1. حمّل عيّنة (10,000 وثيقة مثلًا) من "wikimedia/wikipedia" (تهيئة "20231101.ar")
   عبر مكتبة `datasets` من Hugging Face.
2. طبّق المرحلة 1 (فلاتر الجودة بالقواعد) من هذا الفصل، مع تعديل العتبات
   لتناسب طبيعة نصوص ويكيبيديا (مقالات أطول عادة من نصوص الويب العامة).
3. طبّق تطبيعًا نصيًا عربيًا (توحيد الهمزات، إزالة التشكيل) قبل أي مقارنة.
4. طبّق إزالة التكرار شبه المتطابق بـ MinHash+LSH على كامل العيّنة.
5. احسب واطبع إحصائيات نهائية: نسبة البيانات الناجية من كل مرحلة،
   متوسط طول الوثيقة قبل وبعد، وعدد التكرارات المكتشفة.
6. (تحدٍّ إضافي) درّب نموذج Word2Vec بسيط (من الفصل 7!) على النسخة
   الخام مقابل النسخة المُصفّاة، وقارن جودة أقرب الجيران لكلمات مُختارة
   كدليل عملي بسيط على أثر التصفية.
"""
from __future__ import annotations

from datasets import load_dataset


def load_sample_arabic_wikipedia(n_samples: int = 10_000) -> list[str]:
    # TODO: استخدم load_dataset("wikimedia/wikipedia", "20231101.ar", streaming=True)
    # وخذ أول n_samples وثيقة من حقل "text"
    raise NotImplementedError


def main() -> None:
    raw_docs = load_sample_arabic_wikipedia(10_000)
    # TODO: طبّق كامل خط الأنابيب من هذا الفصل خطوة بخطوة
    # TODO: اطبع تقرير إحصائي نهائي شامل
    pass


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 20: الفصل العشرون: خط أنابيب البيانات — من فوضى الإنترنت إلى وقود تدريب نموذج لغوي عملاق."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_data_pipeline() -> None:
    raw = ['  A useful note. ', 'A useful note.', 'BUY NOW!!!', 'The model learns.', '']
    cleaned = []
    seen = set()
    for text in raw:
        text = re.sub('\\s+', ' ', text).strip().lower()
        text = re.sub('[!]{2,}', '!', text)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    print('raw / retained:', len(raw), '/', len(cleaned))
    print('deduplicated documents:', cleaned)
    print('split by document before tokenization to prevent leakage.')

if __name__ == "__main__":
    print("Chapter 20 lab: A reproducible text-data cleaning pipeline")
    run_data_pipeline()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter20_data_pipeline.py
خط أنابيب بيانات مصغّر يطبّق: تصفية بالقواعد، كشف اللغة المبسّط،
وإزالة التكرار شبه المتطابق عبر MinHash، على نصوص عربية تجريبية.

Python 3.13+
المتطلبات: pip install datasketch
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH


# =========================================================================
# 1) بيانات تجريبية: مزيج من نصوص نظيفة، فاسدة، ومكررة عمدًا
# =========================================================================
SAMPLE_DOCUMENTS: list[str] = [
    # وثيقة نظيفة 1
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرار.",

    # وثيقة نظيفة 2 (مختلفة تمامًا)
    "تُعتبر اللغة العربية من أقدم اللغات السامية المستخدمة حتى اليوم، "
    "وتتميز بثراء مفرداتها وتنوع لهجاتها عبر الوطن العربي.",

    # وثيقة شبه مكررة من الوثيقة 1 (تغيير بسيط فقط)
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم ببناء أنظمة قادرة على "
    "محاكاة بعض جوانب الذكاء البشري، مثل التعلم من البيانات واتخاذ القرارات.",

    # وثيقة قصيرة جدًا (يجب استبعادها بفلتر الطول)
    "مرحبا بك",

    # وثيقة بها تكرار داخلي مفرط (نمط Spam)
    "اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن اشترِ الآن أفضل سعر",

    # وثيقة بنسبة عالية من الرموز غير الأبجدية (ضجيج/ترميز معطوب)
    "###$$$ 123456 !!! @@@ %%% ***  ### $$$ 999999 !!!",

    # وثيقة مختلطة اللغة بشكل كبير جدًا (إنجليزية بالكامل تقريبًا، لمجموعة عربية)
    "This is a completely English document that should not belong "
    "to an Arabic-focused training corpus at all.",

    # وثيقة نظيفة 3
    "التحول الرقمي أصبح ضرورة ملحة لكل المؤسسات الحديثة، حيث يُساهم "
    "في تحسين الكفاءة التشغيلية وتعزيز تجربة العملاء بشكل ملحوظ جدًا.",
]


# =========================================================================
# 2) الطبقة الأولى: فلاتر قائمة على القواعد (Rule-Based Heuristics)
# =========================================================================
@dataclass
class QualityFilterConfig:
    min_words: int = 5
    max_words: int = 100_000
    max_non_alpha_ratio: float = 0.3
    max_repeated_word_ratio: float = 0.3


def count_words(text: str) -> int:
    return len(text.split())


def non_alpha_character_ratio(text: str, arabic_pattern: str = r"[\u0600-\u06FF\s]") -> float:
    """نسبة الرموز التي ليست أحرفًا عربية ولا مسافات."""
    total = len(text)
    if total == 0:
        return 1.0
    non_alpha = len(re.sub(arabic_pattern, "", text))
    return non_alpha / total


def repeated_word_ratio(text: str) -> float:
    """نسبة أكبر كلمة متكررة إلى إجمالي الكلمات (كشف نمط Spam)."""
    words = text.split()
    if not words:
        return 1.0
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words)


def passes_quality_filters(
    text: str, config: QualityFilterConfig
) -> tuple[bool, str]:
    """تُعيد (مقبول؟, سبب الرفض إن وُجد)."""
    n_words = count_words(text)
    if n_words < config.min_words:
        return False, f"قصير جدًا ({n_words} كلمة)"
    if n_words > config.max_words:
        return False, f"طويل جدًا ({n_words} كلمة)"

    non_alpha_ratio = non_alpha_character_ratio(text)
    if non_alpha_ratio > config.max_non_alpha_ratio:
        return False, f"نسبة رموز غير أبجدية عالية ({non_alpha_ratio:.2%})"

    rep_ratio = repeated_word_ratio(text)
    if rep_ratio > config.max_repeated_word_ratio:
        return False, f"تكرار داخلي مفرط ({rep_ratio:.2%})"

    return True, ""


# =========================================================================
# 3) الطبقة الثانية: كشف اللغة المبسّط (heuristic بديل عن fastText للتوضيح)
# =========================================================================
def arabic_character_ratio(text: str) -> float:
    """نسبة الأحرف العربية الفعلية إلى إجمالي الأحرف غير الفراغية."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    non_space_chars = len(re.sub(r"\s", "", text))
    if non_space_chars == 0:
        return 0.0
    return arabic_chars / non_space_chars


def is_arabic_document(text: str, threshold: float = 0.5) -> bool:
    return arabic_character_ratio(text) >= threshold


# =========================================================================
# 4) الطبقة الثالثة: إزالة التكرار شبه المتطابق عبر MinHash + LSH
# =========================================================================
def get_shingles(text: str, k: int = 4) -> set[str]:
    """تقسيم النص إلى shingles (n-grams على مستوى الكلمات)."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text):
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate_near_duplicates(
    documents: list[str], threshold: float = 0.8, num_perm: int = 128
) -> list[int]:
    """
    تُعيد قائمة بمؤشرات (indices) الوثائق التي يجب الاحتفاظ بها
    بعد إزالة التكرار شبه المتطابق.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    keep_indices: list[int] = []
    seen_duplicate_of: dict[int, int] = {}

    for idx, doc in enumerate(documents):
        mh = build_minhash(doc, num_perm=num_perm)
        minhashes[idx] = mh

        # ابحث عن وثائق مشابهة موجودة مسبقًا في LSH
        similar = lsh.query(mh)
        if similar:
            # وُجد تطابق شبه كامل مع وثيقة سابقة -> نعتبرها مكررة، نتجاهلها
            seen_duplicate_of[idx] = similar[0]
            continue

        # لا يوجد تطابق -> وثيقة جديدة فريدة، نُضيفها لـ LSH ونحتفظ بها
        lsh.insert(str(idx), mh)
        keep_indices.append(idx)

    return keep_indices


# =========================================================================
# 5) خط الأنابيب الكامل (End-to-End)
# =========================================================================
def run_full_pipeline(documents: list[str]) -> list[str]:
    print(f"عدد الوثائق الأصلي: {len(documents)}\n")

    # --- المرحلة 1: تصفية الجودة بالقواعد ---
    config = QualityFilterConfig()
    stage1_passed: list[str] = []
    for i, doc in enumerate(documents):
        ok, reason = passes_quality_filters(doc, config)
        status = "✅ قُبلت" if ok else f"❌ رُفضت ({reason})"
        print(f"[جودة] وثيقة {i}: {status}")
        if ok:
            stage1_passed.append(doc)

    print(f"\nبعد تصفية الجودة: {len(stage1_passed)} / {len(documents)} وثيقة\n")

    # --- المرحلة 2: كشف اللغة ---
    stage2_passed: list[str] = []
    for i, doc in enumerate(stage1_passed):
        is_ar = is_arabic_document(doc)
        status = "✅ عربية" if is_ar else "❌ ليست عربية بما يكفي"
        print(f"[لغة] وثيقة {i}: {status}")
        if is_ar:
            stage2_passed.append(doc)

    print(f"\nبعد فلترة اللغة: {len(stage2_passed)} / {len(stage1_passed)} وثيقة\n")

    # --- المرحلة 3: إزالة التكرار شبه المتطابق ---
    keep_idx = deduplicate_near_duplicates(stage2_passed, threshold=0.7)
    final_docs = [stage2_passed[i] for i in keep_idx]

    print(f"[تكرار] المؤشرات المحتفظ بها بعد إزالة التكرار: {keep_idx}")
    print(f"\nبعد إزالة التكرار: {len(final_docs)} / {len(stage2_passed)} وثيقة\n")

    print("=" * 60)
    print(f"الخلاصة النهائية: من {len(documents)} وثيقة خام، "
          f"نجت {len(final_docs)} وثيقة فقط "
          f"({len(final_docs) / len(documents):.1%})")
    print("=" * 60)

    return final_docs


if __name__ == "__main__":
    clean_corpus = run_full_pipeline(SAMPLE_DOCUMENTS)
    print("\nالوثائق النهائية النظيفة:")
    for doc in clean_corpus:
        print(f"  - {doc[:60]}...")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع: عتبات طول الكلمة المصمَّمة للإنجليزية لا تناسب العربية
# (العربية لغة معقدة صرفيًا، الكلمة الواحدة قد تحمل معنى جملة كاملة بالإنجليزية)
def wrong_avg_word_length_filter(text: str) -> bool:
    avg_len = sum(len(w) for w in text.split()) / len(text.split())
    return 3 <= avg_len <= 10  # عتبات مأخوذة من دراسات إنجليزية مباشرة

# ✅ صحيح: احسب إحصائيات اللغة العربية فعليًا من عيّنة موثوقة (مثل ويكيبيديا العربية)
# وابنِ عتباتك الخاصة بناءً عليها، لا تفترض انتقال العتبات بين اللغات

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الفروقات في التشكيل، الهمزات، أو المسافات الزائدة
# تجعل MinHash يفشل في اكتشاف تكرار واضح للعين البشرية
text_a = "الذكاء الإصطناعي رائع"
text_b = "الذكاء الاصطناعي رائع"  # فرق بسيط في الهمزة

# ✅ صحيح: طبّق تطبيعًا نصيًا (توحيد الهمزات، إزالة التشكيل، توحيد المسافات)
# قبل حساب shingles و MinHash
import re

def normalize_arabic(text: str) -> str:
    text = re.sub(r"[إأآا]", "ا", text)       # توحيد الألفات
    text = re.sub(r"ى", "ي", text)             # توحيد الألف المقصورة
    text = re.sub(r"ة", "ه", text)             # توحيد التاء المربوطة (اختياري حسب الحاجة)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"\s+", " ", text).strip()   # توحيد المسافات
    return text

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: فلاتر صارمة جدًا تحذف كل شيء ليس "فصحى معيارية مثالية"
# (تحذف أمثال شعبية، اقتباسات بلهجات محلية، شعرًا بأساليب غير تقليدية)

# ✅ صحيح: ميّز بين "منخفض الجودة فعليًا" (Spam، نص معطوب) و"متنوع أسلوبيًا"
# (لهجات، شعر، أمثال) - الأخير قيّم جدًا ويجب الاحتفاظ به بنسبة مدروسة
# وليس حذفه بالكامل، للحفاظ على تنوع النموذج اللغوي

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تطبيق سلسلة طويلة من الفلاتر "لأنها تبدو منطقية نظريًا"
# دون التحقق من أنها فعلًا تُحسّن أداء نموذج مُدرَّب على البيانات الناتجة

# ✅ صحيح (منهجية FineWeb): درّب نموذج صغير (حتى لو 100-500 مليون معامل)
# على عيّنة من البيانات قبل وبعد كل فلتر، وقارن الأداء على معيار قياسي
# قبل اعتماد الفلتر نهائيًا على مليارات الوثائق

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-20-chapter-20.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mini_project_real_arabic_pipeline.py
تطبيق خط أنابيب البيانات الكامل على عيّنة حقيقية من ويكيبيديا العربية
أو مجموعة بيانات عربية مفتوحة من Hugging Face Datasets.

المطلوب إكماله:
1. حمّل عيّنة (10,000 وثيقة مثلًا) من "wikimedia/wikipedia" (تهيئة "20231101.ar")
   عبر مكتبة `datasets` من Hugging Face.
2. طبّق المرحلة 1 (فلاتر الجودة بالقواعد) من هذا الفصل، مع تعديل العتبات
   لتناسب طبيعة نصوص ويكيبيديا (مقالات أطول عادة من نصوص الويب العامة).
3. طبّق تطبيعًا نصيًا عربيًا (توحيد الهمزات، إزالة التشكيل) قبل أي مقارنة.
4. طبّق إزالة التكرار شبه المتطابق بـ MinHash+LSH على كامل العيّنة.
5. احسب واطبع إحصائيات نهائية: نسبة البيانات الناجية من كل مرحلة،
   متوسط طول الوثيقة قبل وبعد، وعدد التكرارات المكتشفة.
6. (تحدٍّ إضافي) درّب نموذج Word2Vec بسيط (من الفصل 7!) على النسخة
   الخام مقابل النسخة المُصفّاة، وقارن جودة أقرب الجيران لكلمات مُختارة
   كدليل عملي بسيط على أثر التصفية.
"""
from __future__ import annotations

from datasets import load_dataset


def load_sample_arabic_wikipedia(n_samples: int = 10_000) -> list[str]:
    # TODO: استخدم load_dataset("wikimedia/wikipedia", "20231101.ar", streaming=True)
    # وخذ أول n_samples وثيقة من حقل "text"
    raise NotImplementedError


def main() -> None:
    raw_docs = load_sample_arabic_wikipedia(10_000)
    # TODO: طبّق كامل خط الأنابيب من هذا الفصل خطوة بخطوة
    # TODO: اطبع تقرير إحصائي نهائي شامل
    pass


if __name__ == "__main__":
    main()
