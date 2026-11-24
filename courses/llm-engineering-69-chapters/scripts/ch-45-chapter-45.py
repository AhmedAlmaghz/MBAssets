#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها
Large Language Model Engineering — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-45-chapter-45 (lesson 45)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-45-chapter-45.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها
Large Language Model Engineering — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-45-chapter-45 (lesson 45)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-45-chapter-45.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها
Large Language Model Engineering — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-45-chapter-45 (lesson 45)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_45.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 45: الفصل 45: أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_metrics() -> None:
    actual, predicted = ([1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    tp = sum((a == p == 1 for a, p in zip(actual, predicted)))
    fp = sum((a == 0 and p == 1 for a, p in zip(actual, predicted)))
    fn = sum((a == 1 and p == 0 for a, p in zip(actual, predicted)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision / recall / F1:', round(precision, 3), round(recall, 3), round(2 * precision * recall / (precision + recall), 3))
    print('accuracy:', round(sum((a == p for a, p in zip(actual, predicted))) / len(actual), 3))

if __name__ == "__main__":
    print("Chapter 45 lab: Calculate classification metrics and uncertainty")
    run_metrics()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-45.md
# ────────────────────────────────────────────────────────────────────────
"""
EvalHarness متكامل: Perplexity + BLEU + ROUGE-L + BERTScore + Paired Bootstrap + Cohen's d
سيناريو واقعي: مقارنة نموذجين لتلخيص تذاكر دعم العملاء.
المكتبات: torch, transformers, numpy
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# الجزء 1: Perplexity بنافذة منزلقة متداخلة
# ============================================================

class PerplexityEvaluator:
    """يحسب PPL على نص طويل عبر نافذة منزلقة، مع تفادي ازدواج حساب الرموز المتداخلة."""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.max_length = self.model.config.n_positions

    @torch.no_grad()
    def compute(self, text: str, stride: int = 512) -> float:
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        seq_len = ids.size(1)

        nll_sum, n_tokens, prev_end = 0.0, 0, 0
        for begin in range(0, seq_len, stride):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end

            input_ids = ids[:, begin:end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # لا نُعيد حساب الرموز المحسوبة سابقًا

            loss = self.model(input_ids, labels=target_ids).loss
            n_valid = (target_ids != -100).sum().item()
            if n_valid > 0:
                nll_sum += loss.item() * n_valid
                n_tokens += n_valid

            prev_end = end
            if end == seq_len:
                break

        return math.exp(nll_sum / n_tokens)


# ============================================================
# الجزء 2: BLEU وROUGE-L من الصفر
# ============================================================

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
        total = sum(c_ng.values())
        if total == 0:
            precisions.append(1e-10)
            continue
        overlap = sum(min(cnt, r_ng.get(g, 0)) for g, cnt in c_ng.items())
        precisions.append(max(overlap / total, 1e-10))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    c, r = len(cand), len(ref)
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)
    return bp * geo_mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    recall, precision = lcs / len(ref), lcs / len(cand)
    if recall == 0 and precision == 0:
        return 0.0
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


# ============================================================
# الجزء 3: BERTScore (مُبسَّط، عبر تضمينات نموذج محلي لتوضيح الآلية)
# ============================================================

def compute_bertscore_batch(candidates: list[str], references: list[str],
                             lang: str = "en") -> list[float]:
    """
    يستخدم مكتبة bert-score الرسمية (Zhang et al., 2020) إن كانت متوفرة،
    وإلا يعود لتقريب مبسّط عبر تشابه متوسط التضمينات (لأغراض العرض فقط).
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, f1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
        return f1.tolist()
    except ImportError:
        print("⚠️ bert-score غير مثبتة (pip install bert-score)؛ استخدام تقريب مبسّط بديل.")
        from transformers import AutoTokenizer as AT, AutoModel as AM
        tok = AT.from_pretrained("bert-base-uncased")
        mdl = AM.from_pretrained("bert-base-uncased").to(device).eval()

        def embed(text: str) -> torch.Tensor:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = mdl(**inputs).last_hidden_state.mean(dim=1)
            return F.normalize(out, dim=-1)

        scores = []
        for c, r in zip(candidates, references):
            sim = (embed(c) @ embed(r).T).item()
            scores.append(sim)
        return scores


# ============================================================
# الجزء 4: الصرامة الإحصائية — Paired Bootstrap + Cohen's d
# ============================================================

def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                            n_bootstrap: int = 10_000, confidence: float = 0.95,
                            seed: int = 42) -> dict:
    """
    اختبار Bootstrap مُزدوَج: يفترض أن scores_a[i] وscores_b[i] قُيِّما
    على نفس العيّنة i بالضبط (مقارنة مزدوجة، أقوى إحصائيًا من المقارنة المستقلة).
    """
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_b) - np.array(scores_a)
    n = len(diffs)

    observed_mean_diff = diffs.mean()
    bootstrap_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = diffs[idx].mean()

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    is_significant = not (lower <= 0 <= upper)

    cohens_d = observed_mean_diff / (diffs.std(ddof=1) + 1e-12)

    return {
        "mean_diff": observed_mean_diff,
        "ci_95": (lower, upper),
        "is_significant": is_significant,
        "cohens_d": cohens_d,
        "effect_size_label": _interpret_cohens_d(cohens_d),
    }


def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "ضئيل جدًا (قد لا يستحق التعقيد الهندسي للنشر)"
    elif d < 0.5:
        return "صغير"
    elif d < 0.8:
        return "متوسط"
    return "كبير"


# ============================================================
# الجزء 5: السيناريو الكامل — تلخيص تذاكر دعم العملاء
# ============================================================

if __name__ == "__main__":
    # تذاكر دعم عملاء حقيقية (مُصغّرة) + ملخصات مرجعية كتبها موظفو الدعم
    reference_summaries = [
        "Customer requested a refund due to a damaged product received last week.",
        "User cannot log in after resetting their password twice.",
        "Subscription renewal failed because the credit card expired.",
        "Customer asked how to export their data before closing the account.",
        "Shipping was delayed by five days and the customer wants compensation.",
    ]

    # محاكاة ناتج "النموذج الأساسي" (Model A) — أقل دقة، صياغة أبسط
    model_a_outputs = [
        "Customer wants money back, product was broken.",
        "Login problem after password reset.",
        "Card expired so renewal did not work.",
        "Customer wants data before deleting account.",
        "Package was late, customer unhappy.",
    ]

    # محاكاة ناتج "النموذج المُحسَّن" (Model B) — أقرب دلاليًا وتفصيلًا للمرجع
    model_b_outputs = [
        "Customer requested a refund for a damaged item delivered last week.",
        "User is unable to log in after resetting their password two times.",
        "Subscription renewal failed due to an expired credit card.",
        "Customer wants to export their data before account closure.",
        "Delivery was delayed by five days; customer is requesting compensation.",
    ]

    print("=" * 75)
    print("القسم 1: المقاييس اللفظية والدلالية لكل عيّنة")
    print("=" * 75)

    bleu_a = [compute_bleu(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    bleu_b = [compute_bleu(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    rouge_a = [compute_rouge_l(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    rouge_b = [compute_rouge_l(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    bert_a = compute_bertscore_batch(model_a_outputs, reference_summaries)
    bert_b = compute_bertscore_batch(model_b_outputs, reference_summaries)

    print(f"{'عينة':<6}{'BLEU-A':<10}{'BLEU-B':<10}{'ROUGE-A':<10}{'ROUGE-B':<10}"
          f"{'BERT-A':<10}{'BERT-B':<10}")
    for i in range(len(reference_summaries)):
        print(f"{i+1:<6}{bleu_a[i]:<10.3f}{bleu_b[i]:<10.3f}{rouge_a[i]:<10.3f}"
              f"{rouge_b[i]:<10.3f}{bert_a[i]:<10.3f}{bert_b[i]:<10.3f}")

    print(f"\nمتوسط BLEU    : A={np.mean(bleu_a):.3f} | B={np.mean(bleu_b):.3f}")
    print(f"متوسط ROUGE-L : A={np.mean(rouge_a):.3f} | B={np.mean(rouge_b):.3f}")
    print(f"متوسط BERTScore: A={np.mean(bert_a):.3f} | B={np.mean(bert_b):.3f}")

    print("\n" + "=" * 75)
    print("القسم 2: هل الفرق حقيقي إحصائيًا؟ (Paired Bootstrap على BERTScore)")
    print("=" * 75)

    # ملاحظة منهجية: 5 عينات فقط للتوضيح؛ في الواقع نحتاج مئات العينات
    # لنتائج Bootstrap موثوقة إحصائيًا (انظر قسم الأخطاء الشائعة أدناه)
    result = paired_bootstrap_test(bert_a, bert_b, n_bootstrap=10_000)

    print(f"الفرق المُلاحَظ (B - A)   : {result['mean_diff']:.4f}")
    print(f"فاصل الثقة 95%           : [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"ذو دلالة إحصائية؟       : {'نعم ✅' if result['is_significant'] else 'لا ❌ (عينة صغيرة جدًا هنا)'}")
    print(f"Cohen's d (حجم الأثر)   : {result['cohens_d']:.3f} ({result['effect_size_label']})")

    print("\n" + "=" * 75)
    print("القسم 3: Perplexity كمقياس جوهري مستقل عن المهمة")
    print("=" * 75)

    ppl_eval = PerplexityEvaluator("gpt2")
    sample_text = (
        "The customer service team resolved the issue within 24 hours and "
        "issued a full refund after confirming the product defect through photos."
    )
    ppl = ppl_eval.compute(sample_text)
    print(f"Perplexity (GPT-2) على نص دعم العملاء: {ppl:.2f}")
    print("تذكير: PPL تقيس الطلاقة الإحصائية فقط، وليست مؤشرًا على دقّة التلخيص"
          " أو فائدته — لهذا احتجنا الأقسام 1 و2 أعلاه لتقييم المهمة الفعلية.")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-45-chapter-45.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 45: الفصل 45: أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_metrics() -> None:
    actual, predicted = ([1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    tp = sum((a == p == 1 for a, p in zip(actual, predicted)))
    fp = sum((a == 0 and p == 1 for a, p in zip(actual, predicted)))
    fn = sum((a == 1 and p == 0 for a, p in zip(actual, predicted)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision / recall / F1:', round(precision, 3), round(recall, 3), round(2 * precision * recall / (precision + recall), 3))
    print('accuracy:', round(sum((a == p for a, p in zip(actual, predicted))) / len(actual), 3))

if __name__ == "__main__":
    print("Chapter 45 lab: Calculate classification metrics and uncertainty")
    run_metrics()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-45-chapter-45.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
EvalHarness متكامل: Perplexity + BLEU + ROUGE-L + BERTScore + Paired Bootstrap + Cohen's d
سيناريو واقعي: مقارنة نموذجين لتلخيص تذاكر دعم العملاء.
المكتبات: torch, transformers, numpy
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# الجزء 1: Perplexity بنافذة منزلقة متداخلة
# ============================================================

class PerplexityEvaluator:
    """يحسب PPL على نص طويل عبر نافذة منزلقة، مع تفادي ازدواج حساب الرموز المتداخلة."""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.max_length = self.model.config.n_positions

    @torch.no_grad()
    def compute(self, text: str, stride: int = 512) -> float:
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        seq_len = ids.size(1)

        nll_sum, n_tokens, prev_end = 0.0, 0, 0
        for begin in range(0, seq_len, stride):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end

            input_ids = ids[:, begin:end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # لا نُعيد حساب الرموز المحسوبة سابقًا

            loss = self.model(input_ids, labels=target_ids).loss
            n_valid = (target_ids != -100).sum().item()
            if n_valid > 0:
                nll_sum += loss.item() * n_valid
                n_tokens += n_valid

            prev_end = end
            if end == seq_len:
                break

        return math.exp(nll_sum / n_tokens)


# ============================================================
# الجزء 2: BLEU وROUGE-L من الصفر
# ============================================================

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
        total = sum(c_ng.values())
        if total == 0:
            precisions.append(1e-10)
            continue
        overlap = sum(min(cnt, r_ng.get(g, 0)) for g, cnt in c_ng.items())
        precisions.append(max(overlap / total, 1e-10))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    c, r = len(cand), len(ref)
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)
    return bp * geo_mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    recall, precision = lcs / len(ref), lcs / len(cand)
    if recall == 0 and precision == 0:
        return 0.0
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


# ============================================================
# الجزء 3: BERTScore (مُبسَّط، عبر تضمينات نموذج محلي لتوضيح الآلية)
# ============================================================

def compute_bertscore_batch(candidates: list[str], references: list[str],
                             lang: str = "en") -> list[float]:
    """
    يستخدم مكتبة bert-score الرسمية (Zhang et al., 2020) إن كانت متوفرة،
    وإلا يعود لتقريب مبسّط عبر تشابه متوسط التضمينات (لأغراض العرض فقط).
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, f1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
        return f1.tolist()
    except ImportError:
        print("⚠️ bert-score غير مثبتة (pip install bert-score)؛ استخدام تقريب مبسّط بديل.")
        from transformers import AutoTokenizer as AT, AutoModel as AM
        tok = AT.from_pretrained("bert-base-uncased")
        mdl = AM.from_pretrained("bert-base-uncased").to(device).eval()

        def embed(text: str) -> torch.Tensor:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = mdl(**inputs).last_hidden_state.mean(dim=1)
            return F.normalize(out, dim=-1)

        scores = []
        for c, r in zip(candidates, references):
            sim = (embed(c) @ embed(r).T).item()
            scores.append(sim)
        return scores


# ============================================================
# الجزء 4: الصرامة الإحصائية — Paired Bootstrap + Cohen's d
# ============================================================

def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                            n_bootstrap: int = 10_000, confidence: float = 0.95,
                            seed: int = 42) -> dict:
    """
    اختبار Bootstrap مُزدوَج: يفترض أن scores_a[i] وscores_b[i] قُيِّما
    على نفس العيّنة i بالضبط (مقارنة مزدوجة، أقوى إحصائيًا من المقارنة المستقلة).
    """
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_b) - np.array(scores_a)
    n = len(diffs)

    observed_mean_diff = diffs.mean()
    bootstrap_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = diffs[idx].mean()

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    is_significant = not (lower <= 0 <= upper)

    cohens_d = observed_mean_diff / (diffs.std(ddof=1) + 1e-12)

    return {
        "mean_diff": observed_mean_diff,
        "ci_95": (lower, upper),
        "is_significant": is_significant,
        "cohens_d": cohens_d,
        "effect_size_label": _interpret_cohens_d(cohens_d),
    }


def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "ضئيل جدًا (قد لا يستحق التعقيد الهندسي للنشر)"
    elif d < 0.5:
        return "صغير"
    elif d < 0.8:
        return "متوسط"
    return "كبير"


# ============================================================
# الجزء 5: السيناريو الكامل — تلخيص تذاكر دعم العملاء
# ============================================================

if __name__ == "__main__":
    # تذاكر دعم عملاء حقيقية (مُصغّرة) + ملخصات مرجعية كتبها موظفو الدعم
    reference_summaries = [
        "Customer requested a refund due to a damaged product received last week.",
        "User cannot log in after resetting their password twice.",
        "Subscription renewal failed because the credit card expired.",
        "Customer asked how to export their data before closing the account.",
        "Shipping was delayed by five days and the customer wants compensation.",
    ]

    # محاكاة ناتج "النموذج الأساسي" (Model A) — أقل دقة، صياغة أبسط
    model_a_outputs = [
        "Customer wants money back, product was broken.",
        "Login problem after password reset.",
        "Card expired so renewal did not work.",
        "Customer wants data before deleting account.",
        "Package was late, customer unhappy.",
    ]

    # محاكاة ناتج "النموذج المُحسَّن" (Model B) — أقرب دلاليًا وتفصيلًا للمرجع
    model_b_outputs = [
        "Customer requested a refund for a damaged item delivered last week.",
        "User is unable to log in after resetting their password two times.",
        "Subscription renewal failed due to an expired credit card.",
        "Customer wants to export their data before account closure.",
        "Delivery was delayed by five days; customer is requesting compensation.",
    ]

    print("=" * 75)
    print("القسم 1: المقاييس اللفظية والدلالية لكل عيّنة")
    print("=" * 75)

    bleu_a = [compute_bleu(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    bleu_b = [compute_bleu(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    rouge_a = [compute_rouge_l(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    rouge_b = [compute_rouge_l(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    bert_a = compute_bertscore_batch(model_a_outputs, reference_summaries)
    bert_b = compute_bertscore_batch(model_b_outputs, reference_summaries)

    print(f"{'عينة':<6}{'BLEU-A':<10}{'BLEU-B':<10}{'ROUGE-A':<10}{'ROUGE-B':<10}"
          f"{'BERT-A':<10}{'BERT-B':<10}")
    for i in range(len(reference_summaries)):
        print(f"{i+1:<6}{bleu_a[i]:<10.3f}{bleu_b[i]:<10.3f}{rouge_a[i]:<10.3f}"
              f"{rouge_b[i]:<10.3f}{bert_a[i]:<10.3f}{bert_b[i]:<10.3f}")

    print(f"\nمتوسط BLEU    : A={np.mean(bleu_a):.3f} | B={np.mean(bleu_b):.3f}")
    print(f"متوسط ROUGE-L : A={np.mean(rouge_a):.3f} | B={np.mean(rouge_b):.3f}")
    print(f"متوسط BERTScore: A={np.mean(bert_a):.3f} | B={np.mean(bert_b):.3f}")

    print("\n" + "=" * 75)
    print("القسم 2: هل الفرق حقيقي إحصائيًا؟ (Paired Bootstrap على BERTScore)")
    print("=" * 75)

    # ملاحظة منهجية: 5 عينات فقط للتوضيح؛ في الواقع نحتاج مئات العينات
    # لنتائج Bootstrap موثوقة إحصائيًا (انظر قسم الأخطاء الشائعة أدناه)
    result = paired_bootstrap_test(bert_a, bert_b, n_bootstrap=10_000)

    print(f"الفرق المُلاحَظ (B - A)   : {result['mean_diff']:.4f}")
    print(f"فاصل الثقة 95%           : [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"ذو دلالة إحصائية؟       : {'نعم ✅' if result['is_significant'] else 'لا ❌ (عينة صغيرة جدًا هنا)'}")
    print(f"Cohen's d (حجم الأثر)   : {result['cohens_d']:.3f} ({result['effect_size_label']})")

    print("\n" + "=" * 75)
    print("القسم 3: Perplexity كمقياس جوهري مستقل عن المهمة")
    print("=" * 75)

    ppl_eval = PerplexityEvaluator("gpt2")
    sample_text = (
        "The customer service team resolved the issue within 24 hours and "
        "issued a full refund after confirming the product defect through photos."
    )
    ppl = ppl_eval.compute(sample_text)
    print(f"Perplexity (GPT-2) على نص دعم العملاء: {ppl:.2f}")
    print("تذكير: PPL تقيس الطلاقة الإحصائية فقط، وليست مؤشرًا على دقّة التلخيص"
          " أو فائدته — لهذا احتجنا الأقسام 1 و2 أعلاه لتقييم المهمة الفعلية.")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-45-chapter-45.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها
Large Language Model Engineering — Foundations

Book slug : llm-engineering-69-chapters
Chapter   : ch-45-chapter-45 (lesson 45)
Code files: 2 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_45.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 45: الفصل 45: أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_metrics() -> None:
    actual, predicted = ([1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    tp = sum((a == p == 1 for a, p in zip(actual, predicted)))
    fp = sum((a == 0 and p == 1 for a, p in zip(actual, predicted)))
    fn = sum((a == 1 and p == 0 for a, p in zip(actual, predicted)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision / recall / F1:', round(precision, 3), round(recall, 3), round(2 * precision * recall / (precision + recall), 3))
    print('accuracy:', round(sum((a == p for a, p in zip(actual, predicted))) / len(actual), 3))

if __name__ == "__main__":
    print("Chapter 45 lab: Calculate classification metrics and uncertainty")
    run_metrics()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-09-evaluation-testing/chapter-45.md
# ────────────────────────────────────────────────────────────────────────
"""
EvalHarness متكامل: Perplexity + BLEU + ROUGE-L + BERTScore + Paired Bootstrap + Cohen's d
سيناريو واقعي: مقارنة نموذجين لتلخيص تذاكر دعم العملاء.
المكتبات: torch, transformers, numpy
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# الجزء 1: Perplexity بنافذة منزلقة متداخلة
# ============================================================

class PerplexityEvaluator:
    """يحسب PPL على نص طويل عبر نافذة منزلقة، مع تفادي ازدواج حساب الرموز المتداخلة."""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.max_length = self.model.config.n_positions

    @torch.no_grad()
    def compute(self, text: str, stride: int = 512) -> float:
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        seq_len = ids.size(1)

        nll_sum, n_tokens, prev_end = 0.0, 0, 0
        for begin in range(0, seq_len, stride):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end

            input_ids = ids[:, begin:end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # لا نُعيد حساب الرموز المحسوبة سابقًا

            loss = self.model(input_ids, labels=target_ids).loss
            n_valid = (target_ids != -100).sum().item()
            if n_valid > 0:
                nll_sum += loss.item() * n_valid
                n_tokens += n_valid

            prev_end = end
            if end == seq_len:
                break

        return math.exp(nll_sum / n_tokens)


# ============================================================
# الجزء 2: BLEU وROUGE-L من الصفر
# ============================================================

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
        total = sum(c_ng.values())
        if total == 0:
            precisions.append(1e-10)
            continue
        overlap = sum(min(cnt, r_ng.get(g, 0)) for g, cnt in c_ng.items())
        precisions.append(max(overlap / total, 1e-10))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    c, r = len(cand), len(ref)
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)
    return bp * geo_mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    recall, precision = lcs / len(ref), lcs / len(cand)
    if recall == 0 and precision == 0:
        return 0.0
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


# ============================================================
# الجزء 3: BERTScore (مُبسَّط، عبر تضمينات نموذج محلي لتوضيح الآلية)
# ============================================================

def compute_bertscore_batch(candidates: list[str], references: list[str],
                             lang: str = "en") -> list[float]:
    """
    يستخدم مكتبة bert-score الرسمية (Zhang et al., 2020) إن كانت متوفرة،
    وإلا يعود لتقريب مبسّط عبر تشابه متوسط التضمينات (لأغراض العرض فقط).
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, f1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
        return f1.tolist()
    except ImportError:
        print("⚠️ bert-score غير مثبتة (pip install bert-score)؛ استخدام تقريب مبسّط بديل.")
        from transformers import AutoTokenizer as AT, AutoModel as AM
        tok = AT.from_pretrained("bert-base-uncased")
        mdl = AM.from_pretrained("bert-base-uncased").to(device).eval()

        def embed(text: str) -> torch.Tensor:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = mdl(**inputs).last_hidden_state.mean(dim=1)
            return F.normalize(out, dim=-1)

        scores = []
        for c, r in zip(candidates, references):
            sim = (embed(c) @ embed(r).T).item()
            scores.append(sim)
        return scores


# ============================================================
# الجزء 4: الصرامة الإحصائية — Paired Bootstrap + Cohen's d
# ============================================================

def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                            n_bootstrap: int = 10_000, confidence: float = 0.95,
                            seed: int = 42) -> dict:
    """
    اختبار Bootstrap مُزدوَج: يفترض أن scores_a[i] وscores_b[i] قُيِّما
    على نفس العيّنة i بالضبط (مقارنة مزدوجة، أقوى إحصائيًا من المقارنة المستقلة).
    """
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_b) - np.array(scores_a)
    n = len(diffs)

    observed_mean_diff = diffs.mean()
    bootstrap_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = diffs[idx].mean()

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    is_significant = not (lower <= 0 <= upper)

    cohens_d = observed_mean_diff / (diffs.std(ddof=1) + 1e-12)

    return {
        "mean_diff": observed_mean_diff,
        "ci_95": (lower, upper),
        "is_significant": is_significant,
        "cohens_d": cohens_d,
        "effect_size_label": _interpret_cohens_d(cohens_d),
    }


def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "ضئيل جدًا (قد لا يستحق التعقيد الهندسي للنشر)"
    elif d < 0.5:
        return "صغير"
    elif d < 0.8:
        return "متوسط"
    return "كبير"


# ============================================================
# الجزء 5: السيناريو الكامل — تلخيص تذاكر دعم العملاء
# ============================================================

if __name__ == "__main__":
    # تذاكر دعم عملاء حقيقية (مُصغّرة) + ملخصات مرجعية كتبها موظفو الدعم
    reference_summaries = [
        "Customer requested a refund due to a damaged product received last week.",
        "User cannot log in after resetting their password twice.",
        "Subscription renewal failed because the credit card expired.",
        "Customer asked how to export their data before closing the account.",
        "Shipping was delayed by five days and the customer wants compensation.",
    ]

    # محاكاة ناتج "النموذج الأساسي" (Model A) — أقل دقة، صياغة أبسط
    model_a_outputs = [
        "Customer wants money back, product was broken.",
        "Login problem after password reset.",
        "Card expired so renewal did not work.",
        "Customer wants data before deleting account.",
        "Package was late, customer unhappy.",
    ]

    # محاكاة ناتج "النموذج المُحسَّن" (Model B) — أقرب دلاليًا وتفصيلًا للمرجع
    model_b_outputs = [
        "Customer requested a refund for a damaged item delivered last week.",
        "User is unable to log in after resetting their password two times.",
        "Subscription renewal failed due to an expired credit card.",
        "Customer wants to export their data before account closure.",
        "Delivery was delayed by five days; customer is requesting compensation.",
    ]

    print("=" * 75)
    print("القسم 1: المقاييس اللفظية والدلالية لكل عيّنة")
    print("=" * 75)

    bleu_a = [compute_bleu(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    bleu_b = [compute_bleu(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    rouge_a = [compute_rouge_l(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    rouge_b = [compute_rouge_l(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    bert_a = compute_bertscore_batch(model_a_outputs, reference_summaries)
    bert_b = compute_bertscore_batch(model_b_outputs, reference_summaries)

    print(f"{'عينة':<6}{'BLEU-A':<10}{'BLEU-B':<10}{'ROUGE-A':<10}{'ROUGE-B':<10}"
          f"{'BERT-A':<10}{'BERT-B':<10}")
    for i in range(len(reference_summaries)):
        print(f"{i+1:<6}{bleu_a[i]:<10.3f}{bleu_b[i]:<10.3f}{rouge_a[i]:<10.3f}"
              f"{rouge_b[i]:<10.3f}{bert_a[i]:<10.3f}{bert_b[i]:<10.3f}")

    print(f"\nمتوسط BLEU    : A={np.mean(bleu_a):.3f} | B={np.mean(bleu_b):.3f}")
    print(f"متوسط ROUGE-L : A={np.mean(rouge_a):.3f} | B={np.mean(rouge_b):.3f}")
    print(f"متوسط BERTScore: A={np.mean(bert_a):.3f} | B={np.mean(bert_b):.3f}")

    print("\n" + "=" * 75)
    print("القسم 2: هل الفرق حقيقي إحصائيًا؟ (Paired Bootstrap على BERTScore)")
    print("=" * 75)

    # ملاحظة منهجية: 5 عينات فقط للتوضيح؛ في الواقع نحتاج مئات العينات
    # لنتائج Bootstrap موثوقة إحصائيًا (انظر قسم الأخطاء الشائعة أدناه)
    result = paired_bootstrap_test(bert_a, bert_b, n_bootstrap=10_000)

    print(f"الفرق المُلاحَظ (B - A)   : {result['mean_diff']:.4f}")
    print(f"فاصل الثقة 95%           : [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"ذو دلالة إحصائية؟       : {'نعم ✅' if result['is_significant'] else 'لا ❌ (عينة صغيرة جدًا هنا)'}")
    print(f"Cohen's d (حجم الأثر)   : {result['cohens_d']:.3f} ({result['effect_size_label']})")

    print("\n" + "=" * 75)
    print("القسم 3: Perplexity كمقياس جوهري مستقل عن المهمة")
    print("=" * 75)

    ppl_eval = PerplexityEvaluator("gpt2")
    sample_text = (
        "The customer service team resolved the issue within 24 hours and "
        "issued a full refund after confirming the product defect through photos."
    )
    ppl = ppl_eval.compute(sample_text)
    print(f"Perplexity (GPT-2) على نص دعم العملاء: {ppl:.2f}")
    print("تذكير: PPL تقيس الطلاقة الإحصائية فقط، وليست مؤشرًا على دقّة التلخيص"
          " أو فائدته — لهذا احتجنا الأقسام 1 و2 أعلاه لتقييم المهمة الفعلية.")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-45-chapter-45.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 45: الفصل 45: أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_metrics() -> None:
    actual, predicted = ([1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    tp = sum((a == p == 1 for a, p in zip(actual, predicted)))
    fp = sum((a == 0 and p == 1 for a, p in zip(actual, predicted)))
    fn = sum((a == 1 and p == 0 for a, p in zip(actual, predicted)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision / recall / F1:', round(precision, 3), round(recall, 3), round(2 * precision * recall / (precision + recall), 3))
    print('accuracy:', round(sum((a == p for a, p in zip(actual, predicted))) / len(actual), 3))

if __name__ == "__main__":
    print("Chapter 45 lab: Calculate classification metrics and uncertainty")
    run_metrics()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-45-chapter-45.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
EvalHarness متكامل: Perplexity + BLEU + ROUGE-L + BERTScore + Paired Bootstrap + Cohen's d
سيناريو واقعي: مقارنة نموذجين لتلخيص تذاكر دعم العملاء.
المكتبات: torch, transformers, numpy
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# الجزء 1: Perplexity بنافذة منزلقة متداخلة
# ============================================================

class PerplexityEvaluator:
    """يحسب PPL على نص طويل عبر نافذة منزلقة، مع تفادي ازدواج حساب الرموز المتداخلة."""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.max_length = self.model.config.n_positions

    @torch.no_grad()
    def compute(self, text: str, stride: int = 512) -> float:
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        seq_len = ids.size(1)

        nll_sum, n_tokens, prev_end = 0.0, 0, 0
        for begin in range(0, seq_len, stride):
            end = min(begin + self.max_length, seq_len)
            trg_len = end - prev_end

            input_ids = ids[:, begin:end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # لا نُعيد حساب الرموز المحسوبة سابقًا

            loss = self.model(input_ids, labels=target_ids).loss
            n_valid = (target_ids != -100).sum().item()
            if n_valid > 0:
                nll_sum += loss.item() * n_valid
                n_tokens += n_valid

            prev_end = end
            if end == seq_len:
                break

        return math.exp(nll_sum / n_tokens)


# ============================================================
# الجزء 2: BLEU وROUGE-L من الصفر
# ============================================================

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
        total = sum(c_ng.values())
        if total == 0:
            precisions.append(1e-10)
            continue
        overlap = sum(min(cnt, r_ng.get(g, 0)) for g, cnt in c_ng.items())
        precisions.append(max(overlap / total, 1e-10))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    c, r = len(cand), len(ref)
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)
    return bp * geo_mean


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    cand, ref = candidate.lower().split(), reference.lower().split()
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    recall, precision = lcs / len(ref), lcs / len(cand)
    if recall == 0 and precision == 0:
        return 0.0
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


# ============================================================
# الجزء 3: BERTScore (مُبسَّط، عبر تضمينات نموذج محلي لتوضيح الآلية)
# ============================================================

def compute_bertscore_batch(candidates: list[str], references: list[str],
                             lang: str = "en") -> list[float]:
    """
    يستخدم مكتبة bert-score الرسمية (Zhang et al., 2020) إن كانت متوفرة،
    وإلا يعود لتقريب مبسّط عبر تشابه متوسط التضمينات (لأغراض العرض فقط).
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, f1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
        return f1.tolist()
    except ImportError:
        print("⚠️ bert-score غير مثبتة (pip install bert-score)؛ استخدام تقريب مبسّط بديل.")
        from transformers import AutoTokenizer as AT, AutoModel as AM
        tok = AT.from_pretrained("bert-base-uncased")
        mdl = AM.from_pretrained("bert-base-uncased").to(device).eval()

        def embed(text: str) -> torch.Tensor:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = mdl(**inputs).last_hidden_state.mean(dim=1)
            return F.normalize(out, dim=-1)

        scores = []
        for c, r in zip(candidates, references):
            sim = (embed(c) @ embed(r).T).item()
            scores.append(sim)
        return scores


# ============================================================
# الجزء 4: الصرامة الإحصائية — Paired Bootstrap + Cohen's d
# ============================================================

def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                            n_bootstrap: int = 10_000, confidence: float = 0.95,
                            seed: int = 42) -> dict:
    """
    اختبار Bootstrap مُزدوَج: يفترض أن scores_a[i] وscores_b[i] قُيِّما
    على نفس العيّنة i بالضبط (مقارنة مزدوجة، أقوى إحصائيًا من المقارنة المستقلة).
    """
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_b) - np.array(scores_a)
    n = len(diffs)

    observed_mean_diff = diffs.mean()
    bootstrap_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means[b] = diffs[idx].mean()

    alpha = 1 - confidence
    lower, upper = np.percentile(bootstrap_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    is_significant = not (lower <= 0 <= upper)

    cohens_d = observed_mean_diff / (diffs.std(ddof=1) + 1e-12)

    return {
        "mean_diff": observed_mean_diff,
        "ci_95": (lower, upper),
        "is_significant": is_significant,
        "cohens_d": cohens_d,
        "effect_size_label": _interpret_cohens_d(cohens_d),
    }


def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "ضئيل جدًا (قد لا يستحق التعقيد الهندسي للنشر)"
    elif d < 0.5:
        return "صغير"
    elif d < 0.8:
        return "متوسط"
    return "كبير"


# ============================================================
# الجزء 5: السيناريو الكامل — تلخيص تذاكر دعم العملاء
# ============================================================

if __name__ == "__main__":
    # تذاكر دعم عملاء حقيقية (مُصغّرة) + ملخصات مرجعية كتبها موظفو الدعم
    reference_summaries = [
        "Customer requested a refund due to a damaged product received last week.",
        "User cannot log in after resetting their password twice.",
        "Subscription renewal failed because the credit card expired.",
        "Customer asked how to export their data before closing the account.",
        "Shipping was delayed by five days and the customer wants compensation.",
    ]

    # محاكاة ناتج "النموذج الأساسي" (Model A) — أقل دقة، صياغة أبسط
    model_a_outputs = [
        "Customer wants money back, product was broken.",
        "Login problem after password reset.",
        "Card expired so renewal did not work.",
        "Customer wants data before deleting account.",
        "Package was late, customer unhappy.",
    ]

    # محاكاة ناتج "النموذج المُحسَّن" (Model B) — أقرب دلاليًا وتفصيلًا للمرجع
    model_b_outputs = [
        "Customer requested a refund for a damaged item delivered last week.",
        "User is unable to log in after resetting their password two times.",
        "Subscription renewal failed due to an expired credit card.",
        "Customer wants to export their data before account closure.",
        "Delivery was delayed by five days; customer is requesting compensation.",
    ]

    print("=" * 75)
    print("القسم 1: المقاييس اللفظية والدلالية لكل عيّنة")
    print("=" * 75)

    bleu_a = [compute_bleu(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    bleu_b = [compute_bleu(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    rouge_a = [compute_rouge_l(o, r) for o, r in zip(model_a_outputs, reference_summaries)]
    rouge_b = [compute_rouge_l(o, r) for o, r in zip(model_b_outputs, reference_summaries)]
    bert_a = compute_bertscore_batch(model_a_outputs, reference_summaries)
    bert_b = compute_bertscore_batch(model_b_outputs, reference_summaries)

    print(f"{'عينة':<6}{'BLEU-A':<10}{'BLEU-B':<10}{'ROUGE-A':<10}{'ROUGE-B':<10}"
          f"{'BERT-A':<10}{'BERT-B':<10}")
    for i in range(len(reference_summaries)):
        print(f"{i+1:<6}{bleu_a[i]:<10.3f}{bleu_b[i]:<10.3f}{rouge_a[i]:<10.3f}"
              f"{rouge_b[i]:<10.3f}{bert_a[i]:<10.3f}{bert_b[i]:<10.3f}")

    print(f"\nمتوسط BLEU    : A={np.mean(bleu_a):.3f} | B={np.mean(bleu_b):.3f}")
    print(f"متوسط ROUGE-L : A={np.mean(rouge_a):.3f} | B={np.mean(rouge_b):.3f}")
    print(f"متوسط BERTScore: A={np.mean(bert_a):.3f} | B={np.mean(bert_b):.3f}")

    print("\n" + "=" * 75)
    print("القسم 2: هل الفرق حقيقي إحصائيًا؟ (Paired Bootstrap على BERTScore)")
    print("=" * 75)

    # ملاحظة منهجية: 5 عينات فقط للتوضيح؛ في الواقع نحتاج مئات العينات
    # لنتائج Bootstrap موثوقة إحصائيًا (انظر قسم الأخطاء الشائعة أدناه)
    result = paired_bootstrap_test(bert_a, bert_b, n_bootstrap=10_000)

    print(f"الفرق المُلاحَظ (B - A)   : {result['mean_diff']:.4f}")
    print(f"فاصل الثقة 95%           : [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"ذو دلالة إحصائية؟       : {'نعم ✅' if result['is_significant'] else 'لا ❌ (عينة صغيرة جدًا هنا)'}")
    print(f"Cohen's d (حجم الأثر)   : {result['cohens_d']:.3f} ({result['effect_size_label']})")

    print("\n" + "=" * 75)
    print("القسم 3: Perplexity كمقياس جوهري مستقل عن المهمة")
    print("=" * 75)

    ppl_eval = PerplexityEvaluator("gpt2")
    sample_text = (
        "The customer service team resolved the issue within 24 hours and "
        "issued a full refund after confirming the product defect through photos."
    )
    ppl = ppl_eval.compute(sample_text)
    print(f"Perplexity (GPT-2) على نص دعم العملاء: {ppl:.2f}")
    print("تذكير: PPL تقيس الطلاقة الإحصائية فقط، وليست مؤشرًا على دقّة التلخيص"
          " أو فائدته — لهذا احتجنا الأقسام 1 و2 أعلاه لتقييم المهمة الفعلية.")
