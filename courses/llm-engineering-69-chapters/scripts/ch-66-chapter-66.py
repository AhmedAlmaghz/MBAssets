#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — السياق الطويل جدًا (Very Long Context)
Large Language Model Engineering — Very

Book slug : llm-engineering-69-chapters
Chapter   : ch-66-chapter-66 (lesson 66)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-66-chapter-66.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — السياق الطويل جدًا (Very Long Context)
Large Language Model Engineering — Very

Book slug : llm-engineering-69-chapters
Chapter   : ch-66-chapter-66 (lesson 66)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-66-chapter-66.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — السياق الطويل جدًا (Very Long Context)
Large Language Model Engineering — Very

Book slug : llm-engineering-69-chapters
Chapter   : ch-66-chapter-66 (lesson 66)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_66.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 66: الفصل 66: السياق الطويل جدًا (Very Long Context)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_long_context() -> None:
    document = ' '.join((f'section{i} filler' for i in range(40))) + ' attention caches keys and values'
    size, overlap = (12, 3)
    words = document.split()
    chunks = [words[start:start + size] for start in range(0, len(words), size - overlap)]
    query = set('attention keys values'.split())
    ranked = sorted(((len(query & set(chunk)), index, chunk) for index, chunk in enumerate(chunks)), reverse=True)
    print('tokens / chunks:', len(words), len(chunks))
    print('best chunk index / overlap:', ranked[0][1], overlap)
    print('retrieved evidence:', ' '.join(ranked[0][2]))

if __name__ == "__main__":
    print("Chapter 66 lab: Retrieve relevant passages from long-context chunks")
    run_long_context()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
# حساب تقريبي لحجم مصفوفة الانتباه لطبقة واحدة، رأس واحد
n = 1_000_000
bytes_per_element = 2  # FP16

attention_matrix_elements = n * n
attention_matrix_bytes = attention_matrix_elements * bytes_per_element
attention_matrix_gb = attention_matrix_bytes / (1024**3)

print(f"عدد عناصر مصفوفة الانتباه: {attention_matrix_elements:,}")
print(f"الحجم بالذاكرة لطبقة واحدة رأس واحد: {attention_matrix_gb:,.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """حساب الترددات الأساسية لـ RoPE القياسي"""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))

def ntk_aware_scaled_frequencies(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,   # مثال: تمديد من 8K إلى 64K
) -> np.ndarray:
    """
    NTK-aware scaling: نرفع الـ base نفسه بدلاً من تمديد المواقع خطيًا.
    هذا يحافظ على دقة الترددات العالية (العلاقات القريبة)
    بينما يمدد الترددات المنخفضة (العلاقات البعيدة).
    """
    adjusted_base = base * (scale_factor ** (dim / (dim - 2)))
    i = np.arange(0, dim, 2)
    return 1.0 / (adjusted_base ** (i / dim))

dim = 128
original_freqs = rope_frequencies(dim)
scaled_freqs = ntk_aware_scaled_frequencies(dim, scale_factor=8.0)

print("أول 5 ترددات أصلية :", original_freqs[:5])
print("أول 5 ترددات بعد NTK scaling:", scaled_freqs[:5])
print("آخر 5 ترددات أصلية :", original_freqs[-5:])
print("آخر 5 ترددات بعد NTK scaling:", scaled_freqs[-5:])

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class OnlineSoftmaxAccumulator:
    """
    محاكاة تعليمية مبسّطة لآلية دمج نتائج الانتباه الجزئية
    (نفس الفكرة المستخدمة داخل FlashAttention وRing Attention).

    بدلاً من حساب softmax على كل التسلسل دفعة واحدة،
    ندمج الكتل الجزئية تدريجيًا مع الحفاظ على الصحة الرياضية الكاملة.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.running_max = -np.inf          # أعلى قيمة score رأيناها حتى الآن
        self.running_sum = 0.0              # مجموع exp(score - max) المُصحّح
        self.running_output = np.zeros(dim) # مجموع V المرجّح، مُصحّح

    def update(self, scores_block: np.ndarray, values_block: np.ndarray):
        """
        دمج كتلة جديدة من scores (QK^T لشريحة واردة) وvalues المقابلة لها.
        scores_block: شكل (block_len,)
        values_block: شكل (block_len, dim)
        """
        block_max = scores_block.max()
        new_max = max(self.running_max, block_max)

        # إعادة معايرة (rescale) المجاميع القديمة بالنسبة للـ max الجديد
        correction_old = np.exp(self.running_max - new_max) if self.running_max != -np.inf else 0.0
        correction_new = np.exp(scores_block - new_max)

        self.running_output = (
            self.running_output * correction_old
            + (correction_new[:, None] * values_block).sum(axis=0)
        )
        self.running_sum = self.running_sum * correction_old + correction_new.sum()
        self.running_max = new_max

    def finalize(self) -> np.ndarray:
        """الحصول على ناتج الانتباه النهائي بعد دمج كل الكتل"""
        return self.running_output / self.running_sum


# ---------- تجربة: مقارنة الدمج التدريجي بالحساب المباشر ----------
np.random.seed(42)
total_len, dim = 12, 4
full_scores = np.random.randn(total_len)
full_values = np.random.randn(total_len, dim)

# الطريقة المباشرة (كأن كل شيء في جهاز واحد)
direct_softmax = np.exp(full_scores - full_scores.max())
direct_softmax /= direct_softmax.sum()
direct_output = (direct_softmax[:, None] * full_values).sum(axis=0)

# الطريقة التراكمية (محاكاة وصول الكتل من "أجهزة" مختلفة عبر الحلقة)
accumulator = OnlineSoftmaxAccumulator(dim=dim)
block_size = 4
for start in range(0, total_len, block_size):
    block_scores = full_scores[start:start + block_size]
    block_values = full_values[start:start + block_size]
    accumulator.update(block_scores, block_values)

ring_style_output = accumulator.finalize()

print("الناتج المباشر      :", np.round(direct_output, 6))
print("الناتج بالدمج التراكمي:", np.round(ring_style_output, 6))
print("الفرق الأقصى         :", np.max(np.abs(direct_output - ring_style_output)))

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
# مثال تعليمي: حساب تكلفة استخدام Context Caching مقابل عدم استخدامه
# (محاكاة تبسيطية لتوضيح الفكرة الاقتصادية، وليست أرقام تسعير رسمية)

def cost_without_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
) -> float:
    """في كل استعلام، يُعاد إرسال ومعالجة السياق الكامل من الصفر"""
    total_input_tokens = context_tokens * num_queries
    return (total_input_tokens / 1_000_000) * price_per_million_input_tokens


def cost_with_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
    cached_price_discount: float = 0.25,  # مثال: السياق المخزن أرخص بنسبة 75%
    cache_storage_price_per_hour: float = 1.0,
    hours_cached: float = 1.0,
) -> float:
    """السياق يُعالج مرة واحدة ويُعاد استخدامه؛ فقط استعلامات جديدة قصيرة تُضاف"""
    one_time_full_cost = (context_tokens / 1_000_000) * price_per_million_input_tokens
    cached_reuse_cost = (
        (context_tokens / 1_000_000)
        * price_per_million_input_tokens
        * cached_price_discount
        * (num_queries - 1)
    )
    storage_cost = cache_storage_price_per_hour * hours_cached
    return one_time_full_cost + cached_reuse_cost + storage_cost


context_tokens = 1_000_000   # كتاب كامل كسياق ثابت
num_queries = 20             # 20 سؤالًا مختلفًا عن نفس الكتاب
price = 1.25                 # دولار لكل مليون توكن إدخال (رقم توضيحي)

no_cache = cost_without_caching(context_tokens, num_queries, price)
with_cache = cost_with_caching(context_tokens, num_queries, price)

print(f"التكلفة بدون Context Caching: ${no_cache:,.2f}")
print(f"التكلفة مع Context Caching  : ${with_cache:,.2f}")
print(f"نسبة التوفير               : {(1 - with_cache/no_cache) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
import random
import string
from dataclasses import dataclass
from typing import Callable


@dataclass
class NeedleTestResult:
    context_length: int
    needle_depth_percent: int
    found_correctly: bool
    model_response: str


class NeedleInHaystackBenchmark:
    """
    تطبيق مبسّط ولكن كامل وقابل للتشغيل لمنهجية Needle-in-a-Haystack.
    يستخدم نصًا حشويًا اصطناعيًا بدلاً من استدعاء نموذج حقيقي،
    مع واجهة (query_function) تسمح بتوصيله بأي LLM فعلي لاحقًا.
    """

    def __init__(self, filler_corpus: str):
        self.filler_corpus = filler_corpus

    def _generate_unique_needle(self) -> tuple[str, str]:
        """توليد حقيقة فريدة غير قابلة للتخمين من السياق"""
        secret_code = "".join(random.choices(string.digits, k=6))
        needle_sentence = (
            f"ملاحظة سرّية هامة: الرمز التعريفي الخاص بالمشروع "
            f"في هذا الاجتماع هو {secret_code}. يُرجى عدم مشاركته."
        )
        return needle_sentence, secret_code

    def _build_context(
        self, target_length_tokens: int, needle: str, depth_percent: int
    ) -> str:
        """بناء سياق بطول معين مع إدخال الإبرة في عمق نسبي محدد"""
        words = self.filler_corpus.split()
        # تكرار النص الحشو حتى الوصول للطول المطلوب (تقريب: كلمة ≈ توكن)
        repeats = (target_length_tokens // len(words)) + 1
        full_filler = (words * repeats)[:target_length_tokens]

        insert_position = int(len(full_filler) * (depth_percent / 100))
        needle_words = needle.split()

        context_words = (
            full_filler[:insert_position]
            + needle_words
            + full_filler[insert_position:]
        )
        return " ".join(context_words)

    def run_single_test(
        self,
        context_length: int,
        depth_percent: int,
        query_function: Callable[[str, str], str],
    ) -> NeedleTestResult:
        """
        query_function: دالة تأخذ (context, question) وترجع إجابة النموذج.
        في الإنتاج الحقيقي، هذه تستدعي Gemini/Claude/GPT API فعليًا.
        """
        needle, secret_code = self._generate_unique_needle()
        context = self._build_context(context_length, needle, depth_percent)
        question = "ما هو الرمز التعريفي السري المذكور في النص؟ أجب بالرقم فقط."

        response = query_function(context, question)
        found_correctly = secret_code in response

        return NeedleTestResult(
            context_length=context_length,
            needle_depth_percent=depth_percent,
            found_correctly=found_correctly,
            model_response=response,
        )

    def run_full_benchmark(
        self,
        context_lengths: list[int],
        depth_percentages: list[int],
        query_function: Callable[[str, str], str],
    ) -> list[NeedleTestResult]:
        """تشغيل كل التركيبات الممكنة من الطول × العمق"""
        results = []
        for length in context_lengths:
            for depth in depth_percentages:
                result = self.run_single_test(length, depth, query_function)
                results.append(result)
        return results

    def render_heatmap_data(
        self, results: list[NeedleTestResult]
    ) -> dict[tuple[int, int], bool]:
        """تجهيز بيانات جاهزة للرسم كخريطة حرارية"""
        return {
            (r.context_length, r.needle_depth_percent): r.found_correctly
            for r in results
        }


# ---------- محاكاة نموذج "ضعيف" لتوضيح ظاهرة Lost in the Middle ----------
def simulated_weak_model(context: str, question: str) -> str:
    """
    محاكاة تعليمية: نموذج يفشل في استرجاع المعلومات من منتصف
    السياق الطويل، لكنه ينجح في البداية والنهاية.
    هذا تبسيط تعليمي بحت وليس استدعاءً لنموذج حقيقي.
    """
    words = context.split()
    needle_index = next(
        (i for i, w in enumerate(words) if w.isdigit() and len(w) == 6), None
    )
    if needle_index is None:
        return "لم أجد معلومة كهذه."

    relative_position = needle_index / len(words)
    # محاكاة: الفشل يزداد كلما اقتربت الإبرة من المنتصف (0.5)
    distance_from_middle = abs(relative_position - 0.5)
    success_probability = min(1.0, distance_from_middle * 2.2)

    if random.random() < success_probability:
        return words[needle_index]
    return "عذرًا، لم أتمكن من تحديد الرمز بدقة."


# تشغيل التجربة
filler_text = (
    "في عالم هندسة النماذج اللغوية الكبيرة تتطور الأبحاث باستمرار "
    "ويعمل الباحثون على تحسين الكفاءة والدقة عبر تقنيات متعددة "
) * 50

benchmark = NeedleInHaystackBenchmark(filler_corpus=filler_text)

results = benchmark.run_full_benchmark(
    context_lengths=[500, 2000, 8000],
    depth_percentages=[0, 25, 50, 75, 100],
    query_function=simulated_weak_model,
)

print(f"{'طول السياق':<12}{'العمق %':<10}{'النتيجة':<10}")
print("-" * 32)
for r in results:
    status = "✅ نجح" if r.found_correctly else "❌ فشل"
    print(f"{r.context_length:<12}{r.needle_depth_percent:<10}{status:<10}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام متكامل لمقارنة استراتيجيات التعامل مع السياق الطويل جدًا.
يحاكي بيئة واقعية: وثيقة قانونية طويلة، وسؤال يتطلب ربط معلومتين متباعدتين.

ملاحظة: هذا المثال يستخدم دوال محاكاة (simulate_llm_call) بدلًا من
استدعاء API حقيقي، مع توضيح صريح أن هذا تبسيط تعليمي.
في الإنتاج الفعلي، استبدل simulate_llm_call باستدعاء حقيقي
لـ Gemini/Claude/GPT عبر SDK الرسمي.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    RAG_TRADITIONAL = "rag_traditional"
    FULL_CONTEXT = "full_context"
    HYBRID_EXPANDED_RAG = "hybrid_expanded_rag"


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy: Strategy
    answer: str
    correct: bool
    latency_seconds: float
    estimated_cost_usd: float
    tokens_processed: int


class LongDocumentSystem:
    """نظام محاكاة لمعالجة وثيقة طويلة بثلاث استراتيجيات مختلفة"""

    PRICE_PER_MILLION_TOKENS = 1.25  # دولار / مليون توكن، رقم توضيحي

    def __init__(self, full_document: str, chunk_size: int = 500):
        self.full_document = full_document
        self.chunk_size = chunk_size
        self.chunks = self._split_into_chunks()

    def _split_into_chunks(self) -> list[DocumentChunk]:
        words = self.full_document.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            chunks.append(DocumentChunk(chunk_id=len(chunks), text=chunk_text))
        return chunks

    def _simulate_embedding_retrieval(
        self, query: str, top_k: int = 3
    ) -> list[DocumentChunk]:
        """
        محاكاة استرجاع دلالي: في الواقع نستخدم نموذج تضمين حقيقي
        (مثل text-embedding-3 أو Gemini Embedding) + فهرس متجهي (FAISS/Pinecone).
        هنا نحاكي بإرجاع كتل عشوائية مع تحيّز بسيط نحو الكتل ذات الكلمات المشتركة.
        """
        query_words = set(query.split())
        scored_chunks = []
        for chunk in self.chunks:
            overlap = len(query_words & set(chunk.text.split()))
            # إضافة ضوضاء لمحاكاة عدم كمال الاسترجاع الدلالي الحقيقي
            noise = random.uniform(0, 2)
            scored_chunks.append((overlap + noise, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def _simulate_llm_call(
        self, context: str, question: str, ground_truth_fact: str
    ) -> tuple[str, float]:
        """
        محاكاة استدعاء LLM: نتحقق هل الحقيقة المطلوبة موجودة فعليًا
        داخل السياق المُمرَّر، ونعيد إجابة بناءً على ذلك، مع زمن استجابة
        يتناسب تقريبيًا مع طول السياق (محاكاة واقعية لتأخر المعالجة).
        """
        processing_delay = len(context.split()) / 50_000  # محاكاة تناسب تقريبي
        time.sleep(min(processing_delay, 0.05))  # محدود لأغراض العرض فقط

        if ground_truth_fact in context:
            return f"وفقًا للوثيقة: {ground_truth_fact}", processing_delay
        return "لم أجد معلومات كافية في السياق المتاح للإجابة بدقة.", processing_delay

    def answer_with_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3
    ) -> StrategyResult:
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        context = "\n".join(c.text for c in retrieved_chunks)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.RAG_TRADITIONAL,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_full_context(
        self, question: str, ground_truth_fact: str
    ) -> StrategyResult:
        start = time.perf_counter()
        context = self.full_document

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.FULL_CONTEXT,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_hybrid_expanded_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3, expansion: int = 2
    ) -> StrategyResult:
        """استرجاع الكتل الأعلى تطابقًا + الكتل المجاورة لها (قبل وبعد)"""
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        expanded_ids = set()
        for chunk in retrieved_chunks:
            for offset in range(-expansion, expansion + 1):
                expanded_ids.add(chunk.chunk_id + offset)

        valid_ids = sorted(
            cid for cid in expanded_ids if 0 <= cid < len(self.chunks)
        )
        context = "\n".join(self.chunks[cid].text for cid in valid_ids)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.HYBRID_EXPANDED_RAG,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )


def build_synthetic_legal_document() -> tuple[str, str]:
    """
    بناء وثيقة قانونية اصطناعية طويلة تحتوي على حقيقة واحدة محددة
    مدفونة في منتصفها تقريبًا — تحاكي سيناريو واقعي.
    """
    filler_paragraph = (
        "تنص هذه الفقرة على الشروط العامة للتعاقد بين الطرفين "
        "وتشمل الالتزامات المتبادلة وآليات فض النزاعات المحتملة. "
    )
    critical_fact = (
        "البند الاستثنائي رقم 47 ينص على أن مدة الإشعار المسبق "
        "لإنهاء العقد هي 90 يومًا وليس 30 يومًا كما في البنود الأخرى."
    )

    paragraphs = [filler_paragraph] * 300
    insertion_point = len(paragraphs) // 2
    paragraphs.insert(insertion_point, critical_fact)

    return " ".join(paragraphs), critical_fact


# ---------- تشغيل المقارنة الكاملة ----------
document, critical_fact = build_synthetic_legal_document()
system = LongDocumentSystem(full_document=document, chunk_size=200)

question = "ما هي مدة الإشعار المسبق المطلوبة لإنهاء العقد وفقًا للبند الاستثنائي؟"

rag_result = system.answer_with_rag(question, critical_fact)
full_context_result = system.answer_with_full_context(question, critical_fact)
hybrid_result = system.answer_with_hybrid_expanded_rag(question, critical_fact)

print("=" * 70)
print(f"عدد كتل الوثيقة الكلي: {len(system.chunks)}")
print("=" * 70)

for result in [rag_result, hybrid_result, full_context_result]:
    print(f"\n📌 الاستراتيجية: {result.strategy.value}")
    print(f"   صحيح؟            : {'✅ نعم' if result.correct else '❌ لا'}")
    print(f"   الإجابة           : {result.answer[:80]}...")
    print(f"   التوكنز المُعالجة  : {result.tokens_processed:,}")
    print(f"   التكلفة التقديرية : ${result.estimated_cost_usd:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 66: الفصل 66: السياق الطويل جدًا (Very Long Context)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_long_context() -> None:
    document = ' '.join((f'section{i} filler' for i in range(40))) + ' attention caches keys and values'
    size, overlap = (12, 3)
    words = document.split()
    chunks = [words[start:start + size] for start in range(0, len(words), size - overlap)]
    query = set('attention keys values'.split())
    ranked = sorted(((len(query & set(chunk)), index, chunk) for index, chunk in enumerate(chunks)), reverse=True)
    print('tokens / chunks:', len(words), len(chunks))
    print('best chunk index / overlap:', ranked[0][1], overlap)
    print('retrieved evidence:', ' '.join(ranked[0][2]))

if __name__ == "__main__":
    print("Chapter 66 lab: Retrieve relevant passages from long-context chunks")
    run_long_context()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# حساب تقريبي لحجم مصفوفة الانتباه لطبقة واحدة، رأس واحد
n = 1_000_000
bytes_per_element = 2  # FP16

attention_matrix_elements = n * n
attention_matrix_bytes = attention_matrix_elements * bytes_per_element
attention_matrix_gb = attention_matrix_bytes / (1024**3)

print(f"عدد عناصر مصفوفة الانتباه: {attention_matrix_elements:,}")
print(f"الحجم بالذاكرة لطبقة واحدة رأس واحد: {attention_matrix_gb:,.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """حساب الترددات الأساسية لـ RoPE القياسي"""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))

def ntk_aware_scaled_frequencies(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,   # مثال: تمديد من 8K إلى 64K
) -> np.ndarray:
    """
    NTK-aware scaling: نرفع الـ base نفسه بدلاً من تمديد المواقع خطيًا.
    هذا يحافظ على دقة الترددات العالية (العلاقات القريبة)
    بينما يمدد الترددات المنخفضة (العلاقات البعيدة).
    """
    adjusted_base = base * (scale_factor ** (dim / (dim - 2)))
    i = np.arange(0, dim, 2)
    return 1.0 / (adjusted_base ** (i / dim))

dim = 128
original_freqs = rope_frequencies(dim)
scaled_freqs = ntk_aware_scaled_frequencies(dim, scale_factor=8.0)

print("أول 5 ترددات أصلية :", original_freqs[:5])
print("أول 5 ترددات بعد NTK scaling:", scaled_freqs[:5])
print("آخر 5 ترددات أصلية :", original_freqs[-5:])
print("آخر 5 ترددات بعد NTK scaling:", scaled_freqs[-5:])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class OnlineSoftmaxAccumulator:
    """
    محاكاة تعليمية مبسّطة لآلية دمج نتائج الانتباه الجزئية
    (نفس الفكرة المستخدمة داخل FlashAttention وRing Attention).

    بدلاً من حساب softmax على كل التسلسل دفعة واحدة،
    ندمج الكتل الجزئية تدريجيًا مع الحفاظ على الصحة الرياضية الكاملة.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.running_max = -np.inf          # أعلى قيمة score رأيناها حتى الآن
        self.running_sum = 0.0              # مجموع exp(score - max) المُصحّح
        self.running_output = np.zeros(dim) # مجموع V المرجّح، مُصحّح

    def update(self, scores_block: np.ndarray, values_block: np.ndarray):
        """
        دمج كتلة جديدة من scores (QK^T لشريحة واردة) وvalues المقابلة لها.
        scores_block: شكل (block_len,)
        values_block: شكل (block_len, dim)
        """
        block_max = scores_block.max()
        new_max = max(self.running_max, block_max)

        # إعادة معايرة (rescale) المجاميع القديمة بالنسبة للـ max الجديد
        correction_old = np.exp(self.running_max - new_max) if self.running_max != -np.inf else 0.0
        correction_new = np.exp(scores_block - new_max)

        self.running_output = (
            self.running_output * correction_old
            + (correction_new[:, None] * values_block).sum(axis=0)
        )
        self.running_sum = self.running_sum * correction_old + correction_new.sum()
        self.running_max = new_max

    def finalize(self) -> np.ndarray:
        """الحصول على ناتج الانتباه النهائي بعد دمج كل الكتل"""
        return self.running_output / self.running_sum


# ---------- تجربة: مقارنة الدمج التدريجي بالحساب المباشر ----------
np.random.seed(42)
total_len, dim = 12, 4
full_scores = np.random.randn(total_len)
full_values = np.random.randn(total_len, dim)

# الطريقة المباشرة (كأن كل شيء في جهاز واحد)
direct_softmax = np.exp(full_scores - full_scores.max())
direct_softmax /= direct_softmax.sum()
direct_output = (direct_softmax[:, None] * full_values).sum(axis=0)

# الطريقة التراكمية (محاكاة وصول الكتل من "أجهزة" مختلفة عبر الحلقة)
accumulator = OnlineSoftmaxAccumulator(dim=dim)
block_size = 4
for start in range(0, total_len, block_size):
    block_scores = full_scores[start:start + block_size]
    block_values = full_values[start:start + block_size]
    accumulator.update(block_scores, block_values)

ring_style_output = accumulator.finalize()

print("الناتج المباشر      :", np.round(direct_output, 6))
print("الناتج بالدمج التراكمي:", np.round(ring_style_output, 6))
print("الفرق الأقصى         :", np.max(np.abs(direct_output - ring_style_output)))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# مثال تعليمي: حساب تكلفة استخدام Context Caching مقابل عدم استخدامه
# (محاكاة تبسيطية لتوضيح الفكرة الاقتصادية، وليست أرقام تسعير رسمية)

def cost_without_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
) -> float:
    """في كل استعلام، يُعاد إرسال ومعالجة السياق الكامل من الصفر"""
    total_input_tokens = context_tokens * num_queries
    return (total_input_tokens / 1_000_000) * price_per_million_input_tokens


def cost_with_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
    cached_price_discount: float = 0.25,  # مثال: السياق المخزن أرخص بنسبة 75%
    cache_storage_price_per_hour: float = 1.0,
    hours_cached: float = 1.0,
) -> float:
    """السياق يُعالج مرة واحدة ويُعاد استخدامه؛ فقط استعلامات جديدة قصيرة تُضاف"""
    one_time_full_cost = (context_tokens / 1_000_000) * price_per_million_input_tokens
    cached_reuse_cost = (
        (context_tokens / 1_000_000)
        * price_per_million_input_tokens
        * cached_price_discount
        * (num_queries - 1)
    )
    storage_cost = cache_storage_price_per_hour * hours_cached
    return one_time_full_cost + cached_reuse_cost + storage_cost


context_tokens = 1_000_000   # كتاب كامل كسياق ثابت
num_queries = 20             # 20 سؤالًا مختلفًا عن نفس الكتاب
price = 1.25                 # دولار لكل مليون توكن إدخال (رقم توضيحي)

no_cache = cost_without_caching(context_tokens, num_queries, price)
with_cache = cost_with_caching(context_tokens, num_queries, price)

print(f"التكلفة بدون Context Caching: ${no_cache:,.2f}")
print(f"التكلفة مع Context Caching  : ${with_cache:,.2f}")
print(f"نسبة التوفير               : {(1 - with_cache/no_cache) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import random
import string
from dataclasses import dataclass
from typing import Callable


@dataclass
class NeedleTestResult:
    context_length: int
    needle_depth_percent: int
    found_correctly: bool
    model_response: str


class NeedleInHaystackBenchmark:
    """
    تطبيق مبسّط ولكن كامل وقابل للتشغيل لمنهجية Needle-in-a-Haystack.
    يستخدم نصًا حشويًا اصطناعيًا بدلاً من استدعاء نموذج حقيقي،
    مع واجهة (query_function) تسمح بتوصيله بأي LLM فعلي لاحقًا.
    """

    def __init__(self, filler_corpus: str):
        self.filler_corpus = filler_corpus

    def _generate_unique_needle(self) -> tuple[str, str]:
        """توليد حقيقة فريدة غير قابلة للتخمين من السياق"""
        secret_code = "".join(random.choices(string.digits, k=6))
        needle_sentence = (
            f"ملاحظة سرّية هامة: الرمز التعريفي الخاص بالمشروع "
            f"في هذا الاجتماع هو {secret_code}. يُرجى عدم مشاركته."
        )
        return needle_sentence, secret_code

    def _build_context(
        self, target_length_tokens: int, needle: str, depth_percent: int
    ) -> str:
        """بناء سياق بطول معين مع إدخال الإبرة في عمق نسبي محدد"""
        words = self.filler_corpus.split()
        # تكرار النص الحشو حتى الوصول للطول المطلوب (تقريب: كلمة ≈ توكن)
        repeats = (target_length_tokens // len(words)) + 1
        full_filler = (words * repeats)[:target_length_tokens]

        insert_position = int(len(full_filler) * (depth_percent / 100))
        needle_words = needle.split()

        context_words = (
            full_filler[:insert_position]
            + needle_words
            + full_filler[insert_position:]
        )
        return " ".join(context_words)

    def run_single_test(
        self,
        context_length: int,
        depth_percent: int,
        query_function: Callable[[str, str], str],
    ) -> NeedleTestResult:
        """
        query_function: دالة تأخذ (context, question) وترجع إجابة النموذج.
        في الإنتاج الحقيقي، هذه تستدعي Gemini/Claude/GPT API فعليًا.
        """
        needle, secret_code = self._generate_unique_needle()
        context = self._build_context(context_length, needle, depth_percent)
        question = "ما هو الرمز التعريفي السري المذكور في النص؟ أجب بالرقم فقط."

        response = query_function(context, question)
        found_correctly = secret_code in response

        return NeedleTestResult(
            context_length=context_length,
            needle_depth_percent=depth_percent,
            found_correctly=found_correctly,
            model_response=response,
        )

    def run_full_benchmark(
        self,
        context_lengths: list[int],
        depth_percentages: list[int],
        query_function: Callable[[str, str], str],
    ) -> list[NeedleTestResult]:
        """تشغيل كل التركيبات الممكنة من الطول × العمق"""
        results = []
        for length in context_lengths:
            for depth in depth_percentages:
                result = self.run_single_test(length, depth, query_function)
                results.append(result)
        return results

    def render_heatmap_data(
        self, results: list[NeedleTestResult]
    ) -> dict[tuple[int, int], bool]:
        """تجهيز بيانات جاهزة للرسم كخريطة حرارية"""
        return {
            (r.context_length, r.needle_depth_percent): r.found_correctly
            for r in results
        }


# ---------- محاكاة نموذج "ضعيف" لتوضيح ظاهرة Lost in the Middle ----------
def simulated_weak_model(context: str, question: str) -> str:
    """
    محاكاة تعليمية: نموذج يفشل في استرجاع المعلومات من منتصف
    السياق الطويل، لكنه ينجح في البداية والنهاية.
    هذا تبسيط تعليمي بحت وليس استدعاءً لنموذج حقيقي.
    """
    words = context.split()
    needle_index = next(
        (i for i, w in enumerate(words) if w.isdigit() and len(w) == 6), None
    )
    if needle_index is None:
        return "لم أجد معلومة كهذه."

    relative_position = needle_index / len(words)
    # محاكاة: الفشل يزداد كلما اقتربت الإبرة من المنتصف (0.5)
    distance_from_middle = abs(relative_position - 0.5)
    success_probability = min(1.0, distance_from_middle * 2.2)

    if random.random() < success_probability:
        return words[needle_index]
    return "عذرًا، لم أتمكن من تحديد الرمز بدقة."


# تشغيل التجربة
filler_text = (
    "في عالم هندسة النماذج اللغوية الكبيرة تتطور الأبحاث باستمرار "
    "ويعمل الباحثون على تحسين الكفاءة والدقة عبر تقنيات متعددة "
) * 50

benchmark = NeedleInHaystackBenchmark(filler_corpus=filler_text)

results = benchmark.run_full_benchmark(
    context_lengths=[500, 2000, 8000],
    depth_percentages=[0, 25, 50, 75, 100],
    query_function=simulated_weak_model,
)

print(f"{'طول السياق':<12}{'العمق %':<10}{'النتيجة':<10}")
print("-" * 32)
for r in results:
    status = "✅ نجح" if r.found_correctly else "❌ فشل"
    print(f"{r.context_length:<12}{r.needle_depth_percent:<10}{status:<10}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
نظام متكامل لمقارنة استراتيجيات التعامل مع السياق الطويل جدًا.
يحاكي بيئة واقعية: وثيقة قانونية طويلة، وسؤال يتطلب ربط معلومتين متباعدتين.

ملاحظة: هذا المثال يستخدم دوال محاكاة (simulate_llm_call) بدلًا من
استدعاء API حقيقي، مع توضيح صريح أن هذا تبسيط تعليمي.
في الإنتاج الفعلي، استبدل simulate_llm_call باستدعاء حقيقي
لـ Gemini/Claude/GPT عبر SDK الرسمي.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    RAG_TRADITIONAL = "rag_traditional"
    FULL_CONTEXT = "full_context"
    HYBRID_EXPANDED_RAG = "hybrid_expanded_rag"


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy: Strategy
    answer: str
    correct: bool
    latency_seconds: float
    estimated_cost_usd: float
    tokens_processed: int


class LongDocumentSystem:
    """نظام محاكاة لمعالجة وثيقة طويلة بثلاث استراتيجيات مختلفة"""

    PRICE_PER_MILLION_TOKENS = 1.25  # دولار / مليون توكن، رقم توضيحي

    def __init__(self, full_document: str, chunk_size: int = 500):
        self.full_document = full_document
        self.chunk_size = chunk_size
        self.chunks = self._split_into_chunks()

    def _split_into_chunks(self) -> list[DocumentChunk]:
        words = self.full_document.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            chunks.append(DocumentChunk(chunk_id=len(chunks), text=chunk_text))
        return chunks

    def _simulate_embedding_retrieval(
        self, query: str, top_k: int = 3
    ) -> list[DocumentChunk]:
        """
        محاكاة استرجاع دلالي: في الواقع نستخدم نموذج تضمين حقيقي
        (مثل text-embedding-3 أو Gemini Embedding) + فهرس متجهي (FAISS/Pinecone).
        هنا نحاكي بإرجاع كتل عشوائية مع تحيّز بسيط نحو الكتل ذات الكلمات المشتركة.
        """
        query_words = set(query.split())
        scored_chunks = []
        for chunk in self.chunks:
            overlap = len(query_words & set(chunk.text.split()))
            # إضافة ضوضاء لمحاكاة عدم كمال الاسترجاع الدلالي الحقيقي
            noise = random.uniform(0, 2)
            scored_chunks.append((overlap + noise, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def _simulate_llm_call(
        self, context: str, question: str, ground_truth_fact: str
    ) -> tuple[str, float]:
        """
        محاكاة استدعاء LLM: نتحقق هل الحقيقة المطلوبة موجودة فعليًا
        داخل السياق المُمرَّر، ونعيد إجابة بناءً على ذلك، مع زمن استجابة
        يتناسب تقريبيًا مع طول السياق (محاكاة واقعية لتأخر المعالجة).
        """
        processing_delay = len(context.split()) / 50_000  # محاكاة تناسب تقريبي
        time.sleep(min(processing_delay, 0.05))  # محدود لأغراض العرض فقط

        if ground_truth_fact in context:
            return f"وفقًا للوثيقة: {ground_truth_fact}", processing_delay
        return "لم أجد معلومات كافية في السياق المتاح للإجابة بدقة.", processing_delay

    def answer_with_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3
    ) -> StrategyResult:
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        context = "\n".join(c.text for c in retrieved_chunks)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.RAG_TRADITIONAL,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_full_context(
        self, question: str, ground_truth_fact: str
    ) -> StrategyResult:
        start = time.perf_counter()
        context = self.full_document

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.FULL_CONTEXT,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_hybrid_expanded_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3, expansion: int = 2
    ) -> StrategyResult:
        """استرجاع الكتل الأعلى تطابقًا + الكتل المجاورة لها (قبل وبعد)"""
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        expanded_ids = set()
        for chunk in retrieved_chunks:
            for offset in range(-expansion, expansion + 1):
                expanded_ids.add(chunk.chunk_id + offset)

        valid_ids = sorted(
            cid for cid in expanded_ids if 0 <= cid < len(self.chunks)
        )
        context = "\n".join(self.chunks[cid].text for cid in valid_ids)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.HYBRID_EXPANDED_RAG,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )


def build_synthetic_legal_document() -> tuple[str, str]:
    """
    بناء وثيقة قانونية اصطناعية طويلة تحتوي على حقيقة واحدة محددة
    مدفونة في منتصفها تقريبًا — تحاكي سيناريو واقعي.
    """
    filler_paragraph = (
        "تنص هذه الفقرة على الشروط العامة للتعاقد بين الطرفين "
        "وتشمل الالتزامات المتبادلة وآليات فض النزاعات المحتملة. "
    )
    critical_fact = (
        "البند الاستثنائي رقم 47 ينص على أن مدة الإشعار المسبق "
        "لإنهاء العقد هي 90 يومًا وليس 30 يومًا كما في البنود الأخرى."
    )

    paragraphs = [filler_paragraph] * 300
    insertion_point = len(paragraphs) // 2
    paragraphs.insert(insertion_point, critical_fact)

    return " ".join(paragraphs), critical_fact


# ---------- تشغيل المقارنة الكاملة ----------
document, critical_fact = build_synthetic_legal_document()
system = LongDocumentSystem(full_document=document, chunk_size=200)

question = "ما هي مدة الإشعار المسبق المطلوبة لإنهاء العقد وفقًا للبند الاستثنائي؟"

rag_result = system.answer_with_rag(question, critical_fact)
full_context_result = system.answer_with_full_context(question, critical_fact)
hybrid_result = system.answer_with_hybrid_expanded_rag(question, critical_fact)

print("=" * 70)
print(f"عدد كتل الوثيقة الكلي: {len(system.chunks)}")
print("=" * 70)

for result in [rag_result, hybrid_result, full_context_result]:
    print(f"\n📌 الاستراتيجية: {result.strategy.value}")
    print(f"   صحيح؟            : {'✅ نعم' if result.correct else '❌ لا'}")
    print(f"   الإجابة           : {result.answer[:80]}...")
    print(f"   التوكنز المُعالجة  : {result.tokens_processed:,}")
    print(f"   التكلفة التقديرية : ${result.estimated_cost_usd:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — السياق الطويل جدًا (Very Long Context)
Large Language Model Engineering — Very

Book slug : llm-engineering-69-chapters
Chapter   : ch-66-chapter-66 (lesson 66)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_66.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 66: الفصل 66: السياق الطويل جدًا (Very Long Context)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_long_context() -> None:
    document = ' '.join((f'section{i} filler' for i in range(40))) + ' attention caches keys and values'
    size, overlap = (12, 3)
    words = document.split()
    chunks = [words[start:start + size] for start in range(0, len(words), size - overlap)]
    query = set('attention keys values'.split())
    ranked = sorted(((len(query & set(chunk)), index, chunk) for index, chunk in enumerate(chunks)), reverse=True)
    print('tokens / chunks:', len(words), len(chunks))
    print('best chunk index / overlap:', ranked[0][1], overlap)
    print('retrieved evidence:', ' '.join(ranked[0][2]))

if __name__ == "__main__":
    print("Chapter 66 lab: Retrieve relevant passages from long-context chunks")
    run_long_context()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
# حساب تقريبي لحجم مصفوفة الانتباه لطبقة واحدة، رأس واحد
n = 1_000_000
bytes_per_element = 2  # FP16

attention_matrix_elements = n * n
attention_matrix_bytes = attention_matrix_elements * bytes_per_element
attention_matrix_gb = attention_matrix_bytes / (1024**3)

print(f"عدد عناصر مصفوفة الانتباه: {attention_matrix_elements:,}")
print(f"الحجم بالذاكرة لطبقة واحدة رأس واحد: {attention_matrix_gb:,.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """حساب الترددات الأساسية لـ RoPE القياسي"""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))

def ntk_aware_scaled_frequencies(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,   # مثال: تمديد من 8K إلى 64K
) -> np.ndarray:
    """
    NTK-aware scaling: نرفع الـ base نفسه بدلاً من تمديد المواقع خطيًا.
    هذا يحافظ على دقة الترددات العالية (العلاقات القريبة)
    بينما يمدد الترددات المنخفضة (العلاقات البعيدة).
    """
    adjusted_base = base * (scale_factor ** (dim / (dim - 2)))
    i = np.arange(0, dim, 2)
    return 1.0 / (adjusted_base ** (i / dim))

dim = 128
original_freqs = rope_frequencies(dim)
scaled_freqs = ntk_aware_scaled_frequencies(dim, scale_factor=8.0)

print("أول 5 ترددات أصلية :", original_freqs[:5])
print("أول 5 ترددات بعد NTK scaling:", scaled_freqs[:5])
print("آخر 5 ترددات أصلية :", original_freqs[-5:])
print("آخر 5 ترددات بعد NTK scaling:", scaled_freqs[-5:])

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class OnlineSoftmaxAccumulator:
    """
    محاكاة تعليمية مبسّطة لآلية دمج نتائج الانتباه الجزئية
    (نفس الفكرة المستخدمة داخل FlashAttention وRing Attention).

    بدلاً من حساب softmax على كل التسلسل دفعة واحدة،
    ندمج الكتل الجزئية تدريجيًا مع الحفاظ على الصحة الرياضية الكاملة.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.running_max = -np.inf          # أعلى قيمة score رأيناها حتى الآن
        self.running_sum = 0.0              # مجموع exp(score - max) المُصحّح
        self.running_output = np.zeros(dim) # مجموع V المرجّح، مُصحّح

    def update(self, scores_block: np.ndarray, values_block: np.ndarray):
        """
        دمج كتلة جديدة من scores (QK^T لشريحة واردة) وvalues المقابلة لها.
        scores_block: شكل (block_len,)
        values_block: شكل (block_len, dim)
        """
        block_max = scores_block.max()
        new_max = max(self.running_max, block_max)

        # إعادة معايرة (rescale) المجاميع القديمة بالنسبة للـ max الجديد
        correction_old = np.exp(self.running_max - new_max) if self.running_max != -np.inf else 0.0
        correction_new = np.exp(scores_block - new_max)

        self.running_output = (
            self.running_output * correction_old
            + (correction_new[:, None] * values_block).sum(axis=0)
        )
        self.running_sum = self.running_sum * correction_old + correction_new.sum()
        self.running_max = new_max

    def finalize(self) -> np.ndarray:
        """الحصول على ناتج الانتباه النهائي بعد دمج كل الكتل"""
        return self.running_output / self.running_sum


# ---------- تجربة: مقارنة الدمج التدريجي بالحساب المباشر ----------
np.random.seed(42)
total_len, dim = 12, 4
full_scores = np.random.randn(total_len)
full_values = np.random.randn(total_len, dim)

# الطريقة المباشرة (كأن كل شيء في جهاز واحد)
direct_softmax = np.exp(full_scores - full_scores.max())
direct_softmax /= direct_softmax.sum()
direct_output = (direct_softmax[:, None] * full_values).sum(axis=0)

# الطريقة التراكمية (محاكاة وصول الكتل من "أجهزة" مختلفة عبر الحلقة)
accumulator = OnlineSoftmaxAccumulator(dim=dim)
block_size = 4
for start in range(0, total_len, block_size):
    block_scores = full_scores[start:start + block_size]
    block_values = full_values[start:start + block_size]
    accumulator.update(block_scores, block_values)

ring_style_output = accumulator.finalize()

print("الناتج المباشر      :", np.round(direct_output, 6))
print("الناتج بالدمج التراكمي:", np.round(ring_style_output, 6))
print("الفرق الأقصى         :", np.max(np.abs(direct_output - ring_style_output)))

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
# مثال تعليمي: حساب تكلفة استخدام Context Caching مقابل عدم استخدامه
# (محاكاة تبسيطية لتوضيح الفكرة الاقتصادية، وليست أرقام تسعير رسمية)

def cost_without_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
) -> float:
    """في كل استعلام، يُعاد إرسال ومعالجة السياق الكامل من الصفر"""
    total_input_tokens = context_tokens * num_queries
    return (total_input_tokens / 1_000_000) * price_per_million_input_tokens


def cost_with_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
    cached_price_discount: float = 0.25,  # مثال: السياق المخزن أرخص بنسبة 75%
    cache_storage_price_per_hour: float = 1.0,
    hours_cached: float = 1.0,
) -> float:
    """السياق يُعالج مرة واحدة ويُعاد استخدامه؛ فقط استعلامات جديدة قصيرة تُضاف"""
    one_time_full_cost = (context_tokens / 1_000_000) * price_per_million_input_tokens
    cached_reuse_cost = (
        (context_tokens / 1_000_000)
        * price_per_million_input_tokens
        * cached_price_discount
        * (num_queries - 1)
    )
    storage_cost = cache_storage_price_per_hour * hours_cached
    return one_time_full_cost + cached_reuse_cost + storage_cost


context_tokens = 1_000_000   # كتاب كامل كسياق ثابت
num_queries = 20             # 20 سؤالًا مختلفًا عن نفس الكتاب
price = 1.25                 # دولار لكل مليون توكن إدخال (رقم توضيحي)

no_cache = cost_without_caching(context_tokens, num_queries, price)
with_cache = cost_with_caching(context_tokens, num_queries, price)

print(f"التكلفة بدون Context Caching: ${no_cache:,.2f}")
print(f"التكلفة مع Context Caching  : ${with_cache:,.2f}")
print(f"نسبة التوفير               : {(1 - with_cache/no_cache) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
import random
import string
from dataclasses import dataclass
from typing import Callable


@dataclass
class NeedleTestResult:
    context_length: int
    needle_depth_percent: int
    found_correctly: bool
    model_response: str


class NeedleInHaystackBenchmark:
    """
    تطبيق مبسّط ولكن كامل وقابل للتشغيل لمنهجية Needle-in-a-Haystack.
    يستخدم نصًا حشويًا اصطناعيًا بدلاً من استدعاء نموذج حقيقي،
    مع واجهة (query_function) تسمح بتوصيله بأي LLM فعلي لاحقًا.
    """

    def __init__(self, filler_corpus: str):
        self.filler_corpus = filler_corpus

    def _generate_unique_needle(self) -> tuple[str, str]:
        """توليد حقيقة فريدة غير قابلة للتخمين من السياق"""
        secret_code = "".join(random.choices(string.digits, k=6))
        needle_sentence = (
            f"ملاحظة سرّية هامة: الرمز التعريفي الخاص بالمشروع "
            f"في هذا الاجتماع هو {secret_code}. يُرجى عدم مشاركته."
        )
        return needle_sentence, secret_code

    def _build_context(
        self, target_length_tokens: int, needle: str, depth_percent: int
    ) -> str:
        """بناء سياق بطول معين مع إدخال الإبرة في عمق نسبي محدد"""
        words = self.filler_corpus.split()
        # تكرار النص الحشو حتى الوصول للطول المطلوب (تقريب: كلمة ≈ توكن)
        repeats = (target_length_tokens // len(words)) + 1
        full_filler = (words * repeats)[:target_length_tokens]

        insert_position = int(len(full_filler) * (depth_percent / 100))
        needle_words = needle.split()

        context_words = (
            full_filler[:insert_position]
            + needle_words
            + full_filler[insert_position:]
        )
        return " ".join(context_words)

    def run_single_test(
        self,
        context_length: int,
        depth_percent: int,
        query_function: Callable[[str, str], str],
    ) -> NeedleTestResult:
        """
        query_function: دالة تأخذ (context, question) وترجع إجابة النموذج.
        في الإنتاج الحقيقي، هذه تستدعي Gemini/Claude/GPT API فعليًا.
        """
        needle, secret_code = self._generate_unique_needle()
        context = self._build_context(context_length, needle, depth_percent)
        question = "ما هو الرمز التعريفي السري المذكور في النص؟ أجب بالرقم فقط."

        response = query_function(context, question)
        found_correctly = secret_code in response

        return NeedleTestResult(
            context_length=context_length,
            needle_depth_percent=depth_percent,
            found_correctly=found_correctly,
            model_response=response,
        )

    def run_full_benchmark(
        self,
        context_lengths: list[int],
        depth_percentages: list[int],
        query_function: Callable[[str, str], str],
    ) -> list[NeedleTestResult]:
        """تشغيل كل التركيبات الممكنة من الطول × العمق"""
        results = []
        for length in context_lengths:
            for depth in depth_percentages:
                result = self.run_single_test(length, depth, query_function)
                results.append(result)
        return results

    def render_heatmap_data(
        self, results: list[NeedleTestResult]
    ) -> dict[tuple[int, int], bool]:
        """تجهيز بيانات جاهزة للرسم كخريطة حرارية"""
        return {
            (r.context_length, r.needle_depth_percent): r.found_correctly
            for r in results
        }


# ---------- محاكاة نموذج "ضعيف" لتوضيح ظاهرة Lost in the Middle ----------
def simulated_weak_model(context: str, question: str) -> str:
    """
    محاكاة تعليمية: نموذج يفشل في استرجاع المعلومات من منتصف
    السياق الطويل، لكنه ينجح في البداية والنهاية.
    هذا تبسيط تعليمي بحت وليس استدعاءً لنموذج حقيقي.
    """
    words = context.split()
    needle_index = next(
        (i for i, w in enumerate(words) if w.isdigit() and len(w) == 6), None
    )
    if needle_index is None:
        return "لم أجد معلومة كهذه."

    relative_position = needle_index / len(words)
    # محاكاة: الفشل يزداد كلما اقتربت الإبرة من المنتصف (0.5)
    distance_from_middle = abs(relative_position - 0.5)
    success_probability = min(1.0, distance_from_middle * 2.2)

    if random.random() < success_probability:
        return words[needle_index]
    return "عذرًا، لم أتمكن من تحديد الرمز بدقة."


# تشغيل التجربة
filler_text = (
    "في عالم هندسة النماذج اللغوية الكبيرة تتطور الأبحاث باستمرار "
    "ويعمل الباحثون على تحسين الكفاءة والدقة عبر تقنيات متعددة "
) * 50

benchmark = NeedleInHaystackBenchmark(filler_corpus=filler_text)

results = benchmark.run_full_benchmark(
    context_lengths=[500, 2000, 8000],
    depth_percentages=[0, 25, 50, 75, 100],
    query_function=simulated_weak_model,
)

print(f"{'طول السياق':<12}{'العمق %':<10}{'النتيجة':<10}")
print("-" * 32)
for r in results:
    status = "✅ نجح" if r.found_correctly else "❌ فشل"
    print(f"{r.context_length:<12}{r.needle_depth_percent:<10}{status:<10}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-12-frontier-trends/chapter-66.md
# ────────────────────────────────────────────────────────────────────────
"""
نظام متكامل لمقارنة استراتيجيات التعامل مع السياق الطويل جدًا.
يحاكي بيئة واقعية: وثيقة قانونية طويلة، وسؤال يتطلب ربط معلومتين متباعدتين.

ملاحظة: هذا المثال يستخدم دوال محاكاة (simulate_llm_call) بدلًا من
استدعاء API حقيقي، مع توضيح صريح أن هذا تبسيط تعليمي.
في الإنتاج الفعلي، استبدل simulate_llm_call باستدعاء حقيقي
لـ Gemini/Claude/GPT عبر SDK الرسمي.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    RAG_TRADITIONAL = "rag_traditional"
    FULL_CONTEXT = "full_context"
    HYBRID_EXPANDED_RAG = "hybrid_expanded_rag"


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy: Strategy
    answer: str
    correct: bool
    latency_seconds: float
    estimated_cost_usd: float
    tokens_processed: int


class LongDocumentSystem:
    """نظام محاكاة لمعالجة وثيقة طويلة بثلاث استراتيجيات مختلفة"""

    PRICE_PER_MILLION_TOKENS = 1.25  # دولار / مليون توكن، رقم توضيحي

    def __init__(self, full_document: str, chunk_size: int = 500):
        self.full_document = full_document
        self.chunk_size = chunk_size
        self.chunks = self._split_into_chunks()

    def _split_into_chunks(self) -> list[DocumentChunk]:
        words = self.full_document.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            chunks.append(DocumentChunk(chunk_id=len(chunks), text=chunk_text))
        return chunks

    def _simulate_embedding_retrieval(
        self, query: str, top_k: int = 3
    ) -> list[DocumentChunk]:
        """
        محاكاة استرجاع دلالي: في الواقع نستخدم نموذج تضمين حقيقي
        (مثل text-embedding-3 أو Gemini Embedding) + فهرس متجهي (FAISS/Pinecone).
        هنا نحاكي بإرجاع كتل عشوائية مع تحيّز بسيط نحو الكتل ذات الكلمات المشتركة.
        """
        query_words = set(query.split())
        scored_chunks = []
        for chunk in self.chunks:
            overlap = len(query_words & set(chunk.text.split()))
            # إضافة ضوضاء لمحاكاة عدم كمال الاسترجاع الدلالي الحقيقي
            noise = random.uniform(0, 2)
            scored_chunks.append((overlap + noise, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def _simulate_llm_call(
        self, context: str, question: str, ground_truth_fact: str
    ) -> tuple[str, float]:
        """
        محاكاة استدعاء LLM: نتحقق هل الحقيقة المطلوبة موجودة فعليًا
        داخل السياق المُمرَّر، ونعيد إجابة بناءً على ذلك، مع زمن استجابة
        يتناسب تقريبيًا مع طول السياق (محاكاة واقعية لتأخر المعالجة).
        """
        processing_delay = len(context.split()) / 50_000  # محاكاة تناسب تقريبي
        time.sleep(min(processing_delay, 0.05))  # محدود لأغراض العرض فقط

        if ground_truth_fact in context:
            return f"وفقًا للوثيقة: {ground_truth_fact}", processing_delay
        return "لم أجد معلومات كافية في السياق المتاح للإجابة بدقة.", processing_delay

    def answer_with_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3
    ) -> StrategyResult:
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        context = "\n".join(c.text for c in retrieved_chunks)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.RAG_TRADITIONAL,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_full_context(
        self, question: str, ground_truth_fact: str
    ) -> StrategyResult:
        start = time.perf_counter()
        context = self.full_document

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.FULL_CONTEXT,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_hybrid_expanded_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3, expansion: int = 2
    ) -> StrategyResult:
        """استرجاع الكتل الأعلى تطابقًا + الكتل المجاورة لها (قبل وبعد)"""
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        expanded_ids = set()
        for chunk in retrieved_chunks:
            for offset in range(-expansion, expansion + 1):
                expanded_ids.add(chunk.chunk_id + offset)

        valid_ids = sorted(
            cid for cid in expanded_ids if 0 <= cid < len(self.chunks)
        )
        context = "\n".join(self.chunks[cid].text for cid in valid_ids)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.HYBRID_EXPANDED_RAG,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )


def build_synthetic_legal_document() -> tuple[str, str]:
    """
    بناء وثيقة قانونية اصطناعية طويلة تحتوي على حقيقة واحدة محددة
    مدفونة في منتصفها تقريبًا — تحاكي سيناريو واقعي.
    """
    filler_paragraph = (
        "تنص هذه الفقرة على الشروط العامة للتعاقد بين الطرفين "
        "وتشمل الالتزامات المتبادلة وآليات فض النزاعات المحتملة. "
    )
    critical_fact = (
        "البند الاستثنائي رقم 47 ينص على أن مدة الإشعار المسبق "
        "لإنهاء العقد هي 90 يومًا وليس 30 يومًا كما في البنود الأخرى."
    )

    paragraphs = [filler_paragraph] * 300
    insertion_point = len(paragraphs) // 2
    paragraphs.insert(insertion_point, critical_fact)

    return " ".join(paragraphs), critical_fact


# ---------- تشغيل المقارنة الكاملة ----------
document, critical_fact = build_synthetic_legal_document()
system = LongDocumentSystem(full_document=document, chunk_size=200)

question = "ما هي مدة الإشعار المسبق المطلوبة لإنهاء العقد وفقًا للبند الاستثنائي؟"

rag_result = system.answer_with_rag(question, critical_fact)
full_context_result = system.answer_with_full_context(question, critical_fact)
hybrid_result = system.answer_with_hybrid_expanded_rag(question, critical_fact)

print("=" * 70)
print(f"عدد كتل الوثيقة الكلي: {len(system.chunks)}")
print("=" * 70)

for result in [rag_result, hybrid_result, full_context_result]:
    print(f"\n📌 الاستراتيجية: {result.strategy.value}")
    print(f"   صحيح؟            : {'✅ نعم' if result.correct else '❌ لا'}")
    print(f"   الإجابة           : {result.answer[:80]}...")
    print(f"   التوكنز المُعالجة  : {result.tokens_processed:,}")
    print(f"   التكلفة التقديرية : ${result.estimated_cost_usd:.6f}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 66: الفصل 66: السياق الطويل جدًا (Very Long Context)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_long_context() -> None:
    document = ' '.join((f'section{i} filler' for i in range(40))) + ' attention caches keys and values'
    size, overlap = (12, 3)
    words = document.split()
    chunks = [words[start:start + size] for start in range(0, len(words), size - overlap)]
    query = set('attention keys values'.split())
    ranked = sorted(((len(query & set(chunk)), index, chunk) for index, chunk in enumerate(chunks)), reverse=True)
    print('tokens / chunks:', len(words), len(chunks))
    print('best chunk index / overlap:', ranked[0][1], overlap)
    print('retrieved evidence:', ' '.join(ranked[0][2]))

if __name__ == "__main__":
    print("Chapter 66 lab: Retrieve relevant passages from long-context chunks")
    run_long_context()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# حساب تقريبي لحجم مصفوفة الانتباه لطبقة واحدة، رأس واحد
n = 1_000_000
bytes_per_element = 2  # FP16

attention_matrix_elements = n * n
attention_matrix_bytes = attention_matrix_elements * bytes_per_element
attention_matrix_gb = attention_matrix_bytes / (1024**3)

print(f"عدد عناصر مصفوفة الانتباه: {attention_matrix_elements:,}")
print(f"الحجم بالذاكرة لطبقة واحدة رأس واحد: {attention_matrix_gb:,.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import numpy as np

def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """حساب الترددات الأساسية لـ RoPE القياسي"""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))

def ntk_aware_scaled_frequencies(
    dim: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,   # مثال: تمديد من 8K إلى 64K
) -> np.ndarray:
    """
    NTK-aware scaling: نرفع الـ base نفسه بدلاً من تمديد المواقع خطيًا.
    هذا يحافظ على دقة الترددات العالية (العلاقات القريبة)
    بينما يمدد الترددات المنخفضة (العلاقات البعيدة).
    """
    adjusted_base = base * (scale_factor ** (dim / (dim - 2)))
    i = np.arange(0, dim, 2)
    return 1.0 / (adjusted_base ** (i / dim))

dim = 128
original_freqs = rope_frequencies(dim)
scaled_freqs = ntk_aware_scaled_frequencies(dim, scale_factor=8.0)

print("أول 5 ترددات أصلية :", original_freqs[:5])
print("أول 5 ترددات بعد NTK scaling:", scaled_freqs[:5])
print("آخر 5 ترددات أصلية :", original_freqs[-5:])
print("آخر 5 ترددات بعد NTK scaling:", scaled_freqs[-5:])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import numpy as np

class OnlineSoftmaxAccumulator:
    """
    محاكاة تعليمية مبسّطة لآلية دمج نتائج الانتباه الجزئية
    (نفس الفكرة المستخدمة داخل FlashAttention وRing Attention).

    بدلاً من حساب softmax على كل التسلسل دفعة واحدة،
    ندمج الكتل الجزئية تدريجيًا مع الحفاظ على الصحة الرياضية الكاملة.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.running_max = -np.inf          # أعلى قيمة score رأيناها حتى الآن
        self.running_sum = 0.0              # مجموع exp(score - max) المُصحّح
        self.running_output = np.zeros(dim) # مجموع V المرجّح، مُصحّح

    def update(self, scores_block: np.ndarray, values_block: np.ndarray):
        """
        دمج كتلة جديدة من scores (QK^T لشريحة واردة) وvalues المقابلة لها.
        scores_block: شكل (block_len,)
        values_block: شكل (block_len, dim)
        """
        block_max = scores_block.max()
        new_max = max(self.running_max, block_max)

        # إعادة معايرة (rescale) المجاميع القديمة بالنسبة للـ max الجديد
        correction_old = np.exp(self.running_max - new_max) if self.running_max != -np.inf else 0.0
        correction_new = np.exp(scores_block - new_max)

        self.running_output = (
            self.running_output * correction_old
            + (correction_new[:, None] * values_block).sum(axis=0)
        )
        self.running_sum = self.running_sum * correction_old + correction_new.sum()
        self.running_max = new_max

    def finalize(self) -> np.ndarray:
        """الحصول على ناتج الانتباه النهائي بعد دمج كل الكتل"""
        return self.running_output / self.running_sum


# ---------- تجربة: مقارنة الدمج التدريجي بالحساب المباشر ----------
np.random.seed(42)
total_len, dim = 12, 4
full_scores = np.random.randn(total_len)
full_values = np.random.randn(total_len, dim)

# الطريقة المباشرة (كأن كل شيء في جهاز واحد)
direct_softmax = np.exp(full_scores - full_scores.max())
direct_softmax /= direct_softmax.sum()
direct_output = (direct_softmax[:, None] * full_values).sum(axis=0)

# الطريقة التراكمية (محاكاة وصول الكتل من "أجهزة" مختلفة عبر الحلقة)
accumulator = OnlineSoftmaxAccumulator(dim=dim)
block_size = 4
for start in range(0, total_len, block_size):
    block_scores = full_scores[start:start + block_size]
    block_values = full_values[start:start + block_size]
    accumulator.update(block_scores, block_values)

ring_style_output = accumulator.finalize()

print("الناتج المباشر      :", np.round(direct_output, 6))
print("الناتج بالدمج التراكمي:", np.round(ring_style_output, 6))
print("الفرق الأقصى         :", np.max(np.abs(direct_output - ring_style_output)))

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# مثال تعليمي: حساب تكلفة استخدام Context Caching مقابل عدم استخدامه
# (محاكاة تبسيطية لتوضيح الفكرة الاقتصادية، وليست أرقام تسعير رسمية)

def cost_without_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
) -> float:
    """في كل استعلام، يُعاد إرسال ومعالجة السياق الكامل من الصفر"""
    total_input_tokens = context_tokens * num_queries
    return (total_input_tokens / 1_000_000) * price_per_million_input_tokens


def cost_with_caching(
    context_tokens: int,
    num_queries: int,
    price_per_million_input_tokens: float,
    cached_price_discount: float = 0.25,  # مثال: السياق المخزن أرخص بنسبة 75%
    cache_storage_price_per_hour: float = 1.0,
    hours_cached: float = 1.0,
) -> float:
    """السياق يُعالج مرة واحدة ويُعاد استخدامه؛ فقط استعلامات جديدة قصيرة تُضاف"""
    one_time_full_cost = (context_tokens / 1_000_000) * price_per_million_input_tokens
    cached_reuse_cost = (
        (context_tokens / 1_000_000)
        * price_per_million_input_tokens
        * cached_price_discount
        * (num_queries - 1)
    )
    storage_cost = cache_storage_price_per_hour * hours_cached
    return one_time_full_cost + cached_reuse_cost + storage_cost


context_tokens = 1_000_000   # كتاب كامل كسياق ثابت
num_queries = 20             # 20 سؤالًا مختلفًا عن نفس الكتاب
price = 1.25                 # دولار لكل مليون توكن إدخال (رقم توضيحي)

no_cache = cost_without_caching(context_tokens, num_queries, price)
with_cache = cost_with_caching(context_tokens, num_queries, price)

print(f"التكلفة بدون Context Caching: ${no_cache:,.2f}")
print(f"التكلفة مع Context Caching  : ${with_cache:,.2f}")
print(f"نسبة التوفير               : {(1 - with_cache/no_cache) * 100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import random
import string
from dataclasses import dataclass
from typing import Callable


@dataclass
class NeedleTestResult:
    context_length: int
    needle_depth_percent: int
    found_correctly: bool
    model_response: str


class NeedleInHaystackBenchmark:
    """
    تطبيق مبسّط ولكن كامل وقابل للتشغيل لمنهجية Needle-in-a-Haystack.
    يستخدم نصًا حشويًا اصطناعيًا بدلاً من استدعاء نموذج حقيقي،
    مع واجهة (query_function) تسمح بتوصيله بأي LLM فعلي لاحقًا.
    """

    def __init__(self, filler_corpus: str):
        self.filler_corpus = filler_corpus

    def _generate_unique_needle(self) -> tuple[str, str]:
        """توليد حقيقة فريدة غير قابلة للتخمين من السياق"""
        secret_code = "".join(random.choices(string.digits, k=6))
        needle_sentence = (
            f"ملاحظة سرّية هامة: الرمز التعريفي الخاص بالمشروع "
            f"في هذا الاجتماع هو {secret_code}. يُرجى عدم مشاركته."
        )
        return needle_sentence, secret_code

    def _build_context(
        self, target_length_tokens: int, needle: str, depth_percent: int
    ) -> str:
        """بناء سياق بطول معين مع إدخال الإبرة في عمق نسبي محدد"""
        words = self.filler_corpus.split()
        # تكرار النص الحشو حتى الوصول للطول المطلوب (تقريب: كلمة ≈ توكن)
        repeats = (target_length_tokens // len(words)) + 1
        full_filler = (words * repeats)[:target_length_tokens]

        insert_position = int(len(full_filler) * (depth_percent / 100))
        needle_words = needle.split()

        context_words = (
            full_filler[:insert_position]
            + needle_words
            + full_filler[insert_position:]
        )
        return " ".join(context_words)

    def run_single_test(
        self,
        context_length: int,
        depth_percent: int,
        query_function: Callable[[str, str], str],
    ) -> NeedleTestResult:
        """
        query_function: دالة تأخذ (context, question) وترجع إجابة النموذج.
        في الإنتاج الحقيقي، هذه تستدعي Gemini/Claude/GPT API فعليًا.
        """
        needle, secret_code = self._generate_unique_needle()
        context = self._build_context(context_length, needle, depth_percent)
        question = "ما هو الرمز التعريفي السري المذكور في النص؟ أجب بالرقم فقط."

        response = query_function(context, question)
        found_correctly = secret_code in response

        return NeedleTestResult(
            context_length=context_length,
            needle_depth_percent=depth_percent,
            found_correctly=found_correctly,
            model_response=response,
        )

    def run_full_benchmark(
        self,
        context_lengths: list[int],
        depth_percentages: list[int],
        query_function: Callable[[str, str], str],
    ) -> list[NeedleTestResult]:
        """تشغيل كل التركيبات الممكنة من الطول × العمق"""
        results = []
        for length in context_lengths:
            for depth in depth_percentages:
                result = self.run_single_test(length, depth, query_function)
                results.append(result)
        return results

    def render_heatmap_data(
        self, results: list[NeedleTestResult]
    ) -> dict[tuple[int, int], bool]:
        """تجهيز بيانات جاهزة للرسم كخريطة حرارية"""
        return {
            (r.context_length, r.needle_depth_percent): r.found_correctly
            for r in results
        }


# ---------- محاكاة نموذج "ضعيف" لتوضيح ظاهرة Lost in the Middle ----------
def simulated_weak_model(context: str, question: str) -> str:
    """
    محاكاة تعليمية: نموذج يفشل في استرجاع المعلومات من منتصف
    السياق الطويل، لكنه ينجح في البداية والنهاية.
    هذا تبسيط تعليمي بحت وليس استدعاءً لنموذج حقيقي.
    """
    words = context.split()
    needle_index = next(
        (i for i, w in enumerate(words) if w.isdigit() and len(w) == 6), None
    )
    if needle_index is None:
        return "لم أجد معلومة كهذه."

    relative_position = needle_index / len(words)
    # محاكاة: الفشل يزداد كلما اقتربت الإبرة من المنتصف (0.5)
    distance_from_middle = abs(relative_position - 0.5)
    success_probability = min(1.0, distance_from_middle * 2.2)

    if random.random() < success_probability:
        return words[needle_index]
    return "عذرًا، لم أتمكن من تحديد الرمز بدقة."


# تشغيل التجربة
filler_text = (
    "في عالم هندسة النماذج اللغوية الكبيرة تتطور الأبحاث باستمرار "
    "ويعمل الباحثون على تحسين الكفاءة والدقة عبر تقنيات متعددة "
) * 50

benchmark = NeedleInHaystackBenchmark(filler_corpus=filler_text)

results = benchmark.run_full_benchmark(
    context_lengths=[500, 2000, 8000],
    depth_percentages=[0, 25, 50, 75, 100],
    query_function=simulated_weak_model,
)

print(f"{'طول السياق':<12}{'العمق %':<10}{'النتيجة':<10}")
print("-" * 32)
for r in results:
    status = "✅ نجح" if r.found_correctly else "❌ فشل"
    print(f"{r.context_length:<12}{r.needle_depth_percent:<10}{status:<10}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-66-chapter-66.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
نظام متكامل لمقارنة استراتيجيات التعامل مع السياق الطويل جدًا.
يحاكي بيئة واقعية: وثيقة قانونية طويلة، وسؤال يتطلب ربط معلومتين متباعدتين.

ملاحظة: هذا المثال يستخدم دوال محاكاة (simulate_llm_call) بدلًا من
استدعاء API حقيقي، مع توضيح صريح أن هذا تبسيط تعليمي.
في الإنتاج الفعلي، استبدل simulate_llm_call باستدعاء حقيقي
لـ Gemini/Claude/GPT عبر SDK الرسمي.
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    RAG_TRADITIONAL = "rag_traditional"
    FULL_CONTEXT = "full_context"
    HYBRID_EXPANDED_RAG = "hybrid_expanded_rag"


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy: Strategy
    answer: str
    correct: bool
    latency_seconds: float
    estimated_cost_usd: float
    tokens_processed: int


class LongDocumentSystem:
    """نظام محاكاة لمعالجة وثيقة طويلة بثلاث استراتيجيات مختلفة"""

    PRICE_PER_MILLION_TOKENS = 1.25  # دولار / مليون توكن، رقم توضيحي

    def __init__(self, full_document: str, chunk_size: int = 500):
        self.full_document = full_document
        self.chunk_size = chunk_size
        self.chunks = self._split_into_chunks()

    def _split_into_chunks(self) -> list[DocumentChunk]:
        words = self.full_document.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk_text = " ".join(words[i:i + self.chunk_size])
            chunks.append(DocumentChunk(chunk_id=len(chunks), text=chunk_text))
        return chunks

    def _simulate_embedding_retrieval(
        self, query: str, top_k: int = 3
    ) -> list[DocumentChunk]:
        """
        محاكاة استرجاع دلالي: في الواقع نستخدم نموذج تضمين حقيقي
        (مثل text-embedding-3 أو Gemini Embedding) + فهرس متجهي (FAISS/Pinecone).
        هنا نحاكي بإرجاع كتل عشوائية مع تحيّز بسيط نحو الكتل ذات الكلمات المشتركة.
        """
        query_words = set(query.split())
        scored_chunks = []
        for chunk in self.chunks:
            overlap = len(query_words & set(chunk.text.split()))
            # إضافة ضوضاء لمحاكاة عدم كمال الاسترجاع الدلالي الحقيقي
            noise = random.uniform(0, 2)
            scored_chunks.append((overlap + noise, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def _simulate_llm_call(
        self, context: str, question: str, ground_truth_fact: str
    ) -> tuple[str, float]:
        """
        محاكاة استدعاء LLM: نتحقق هل الحقيقة المطلوبة موجودة فعليًا
        داخل السياق المُمرَّر، ونعيد إجابة بناءً على ذلك، مع زمن استجابة
        يتناسب تقريبيًا مع طول السياق (محاكاة واقعية لتأخر المعالجة).
        """
        processing_delay = len(context.split()) / 50_000  # محاكاة تناسب تقريبي
        time.sleep(min(processing_delay, 0.05))  # محدود لأغراض العرض فقط

        if ground_truth_fact in context:
            return f"وفقًا للوثيقة: {ground_truth_fact}", processing_delay
        return "لم أجد معلومات كافية في السياق المتاح للإجابة بدقة.", processing_delay

    def answer_with_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3
    ) -> StrategyResult:
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        context = "\n".join(c.text for c in retrieved_chunks)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.RAG_TRADITIONAL,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_full_context(
        self, question: str, ground_truth_fact: str
    ) -> StrategyResult:
        start = time.perf_counter()
        context = self.full_document

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.FULL_CONTEXT,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )

    def answer_with_hybrid_expanded_rag(
        self, question: str, ground_truth_fact: str, top_k: int = 3, expansion: int = 2
    ) -> StrategyResult:
        """استرجاع الكتل الأعلى تطابقًا + الكتل المجاورة لها (قبل وبعد)"""
        start = time.perf_counter()
        retrieved_chunks = self._simulate_embedding_retrieval(question, top_k=top_k)
        expanded_ids = set()
        for chunk in retrieved_chunks:
            for offset in range(-expansion, expansion + 1):
                expanded_ids.add(chunk.chunk_id + offset)

        valid_ids = sorted(
            cid for cid in expanded_ids if 0 <= cid < len(self.chunks)
        )
        context = "\n".join(self.chunks[cid].text for cid in valid_ids)

        answer, _ = self._simulate_llm_call(context, question, ground_truth_fact)
        latency = time.perf_counter() - start

        tokens = len(context.split())
        cost = (tokens / 1_000_000) * self.PRICE_PER_MILLION_TOKENS

        return StrategyResult(
            strategy=Strategy.HYBRID_EXPANDED_RAG,
            answer=answer,
            correct=ground_truth_fact in answer,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            tokens_processed=tokens,
        )


def build_synthetic_legal_document() -> tuple[str, str]:
    """
    بناء وثيقة قانونية اصطناعية طويلة تحتوي على حقيقة واحدة محددة
    مدفونة في منتصفها تقريبًا — تحاكي سيناريو واقعي.
    """
    filler_paragraph = (
        "تنص هذه الفقرة على الشروط العامة للتعاقد بين الطرفين "
        "وتشمل الالتزامات المتبادلة وآليات فض النزاعات المحتملة. "
    )
    critical_fact = (
        "البند الاستثنائي رقم 47 ينص على أن مدة الإشعار المسبق "
        "لإنهاء العقد هي 90 يومًا وليس 30 يومًا كما في البنود الأخرى."
    )

    paragraphs = [filler_paragraph] * 300
    insertion_point = len(paragraphs) // 2
    paragraphs.insert(insertion_point, critical_fact)

    return " ".join(paragraphs), critical_fact


# ---------- تشغيل المقارنة الكاملة ----------
document, critical_fact = build_synthetic_legal_document()
system = LongDocumentSystem(full_document=document, chunk_size=200)

question = "ما هي مدة الإشعار المسبق المطلوبة لإنهاء العقد وفقًا للبند الاستثنائي؟"

rag_result = system.answer_with_rag(question, critical_fact)
full_context_result = system.answer_with_full_context(question, critical_fact)
hybrid_result = system.answer_with_hybrid_expanded_rag(question, critical_fact)

print("=" * 70)
print(f"عدد كتل الوثيقة الكلي: {len(system.chunks)}")
print("=" * 70)

for result in [rag_result, hybrid_result, full_context_result]:
    print(f"\n📌 الاستراتيجية: {result.strategy.value}")
    print(f"   صحيح؟            : {'✅ نعم' if result.correct else '❌ لا'}")
    print(f"   الإجابة           : {result.answer[:80]}...")
    print(f"   التوكنز المُعالجة  : {result.tokens_processed:,}")
    print(f"   التكلفة التقديرية : ${result.estimated_cost_usd:.6f}")
