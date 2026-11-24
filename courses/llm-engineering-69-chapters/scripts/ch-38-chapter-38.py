#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — استراتيجيات التوليد (Decoding Strategies)
Large Language Model Engineering — Decoding

Book slug : llm-engineering-69-chapters
Chapter   : ch-38-chapter-38 (lesson 38)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-38-chapter-38.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — استراتيجيات التوليد (Decoding Strategies)
Large Language Model Engineering — Decoding

Book slug : llm-engineering-69-chapters
Chapter   : ch-38-chapter-38 (lesson 38)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-38-chapter-38.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — استراتيجيات التوليد (Decoding Strategies)
Large Language Model Engineering — Decoding

Book slug : llm-engineering-69-chapters
Chapter   : ch-38-chapter-38 (lesson 38)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_38.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 38: الفصل 38: استراتيجيات التوليد (Decoding Strategies)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_decoding() -> None:
    logits = [2.4, 1.7, 0.8, 0.2, -0.3]
    probabilities = softmax(logits)
    ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    top_k = ranked[:3]
    cumulative, top_p = (0.0, [])
    for index in ranked:
        top_p.append(index)
        cumulative += probabilities[index]
        if cumulative >= 0.85:
            break
    print('greedy token:', ranked[0])
    print('top-k candidates:', top_k)
    print('top-p candidates:', top_p, '| mass:', round(sum((probabilities[i] for i in top_p)), 3))
    print('temperature=0.5 top token:', max(range(len(logits)), key=lambda i: logits[i] / 0.5))

if __name__ == "__main__":
    print("Chapter 38 lab: Compare greedy, temperature, top-k, and top-p decoding")
    run_decoding()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
"""
محرك استراتيجيات التوليد (Decoding Engine) من الصفر
يعمل فوق أي نموذج Causal LM من Hugging Face
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# يمكن استبدال الاسم بـ "gpt2" أو أي نموذج عربي آخر متوافق
MODEL_NAME = "aubmindlab/aragpt2-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------
# 1) إعدادات التوليد الموحّدة
# ---------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# 2) الأدوات المساعدة: عقوبة التكرار، حجب n-grams، الفلاتر
# ---------------------------------------------------------------------
def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """يعاقب الرموز التي ظهرت سابقًا في التسلسل."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    seen_tokens = torch.unique(generated_ids)
    seen_logits = logits[seen_tokens]
    penalized = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
    logits[seen_tokens] = penalized
    return logits


def block_repeated_ngrams(
    logits: torch.Tensor, generated_ids: torch.Tensor, n: int
) -> torch.Tensor:
    """يمنع تمامًا أي رمز يُنتج تكرار n-gram ظهر مسبقًا."""
    if n <= 0 or generated_ids.size(0) < n - 1:
        return logits
    ids = generated_ids.tolist()
    ngram_to_next = {}
    for i in range(len(ids) - n + 1):
        prefix = tuple(ids[i : i + n - 1])
        next_token = ids[i + n - 1]
        ngram_to_next.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(ids[-(n - 1):]) if n > 1 else tuple()
    banned_tokens = ngram_to_next.get(current_prefix, set())
    if banned_tokens:
        logits = logits.clone()
        for tok in banned_tokens:
            logits[tok] = float("-inf")
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """يُبقي فقط أعلى k رمزًا، ويحوّل الباقي إلى -inf."""
    if not k or k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k)
    min_keep_value = top_values[-1]
    return torch.where(
        logits < min_keep_value, torch.full_like(logits, float("-inf")), logits
    )


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus Sampling: يُبقي أصغر مجموعة تجمع احتمالها >= p."""
    if not p or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # نريد إبقاء أول رمز يتجاوز العتبة p أيضًا (لذا نزيح القناع يمينًا)
    sorted_mask_to_remove = cumulative_probs > p
    sorted_mask_to_remove[..., 1:] = sorted_mask_to_remove[..., :-1].clone()
    sorted_mask_to_remove[..., 0] = False

    mask_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    mask_to_remove.scatter_(0, sorted_indices, sorted_mask_to_remove)
    return logits.masked_fill(mask_to_remove, float("-inf"))


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p Sampling: عتبة نسبية لأعلى احتمال في التوزيع."""
    if not min_p or min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return logits.masked_fill(probs < threshold, float("-inf"))


# ---------------------------------------------------------------------
# 3) حلقة التوليد الموحّدة (Greedy + Sampling)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str, config: GenerationConfig) -> str:
    if config.seed is not None:
        torch.manual_seed(config.seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0]
    past_key_values = None
    eos_id = tokenizer.eos_token_id

    for _ in range(config.max_new_tokens):
        if past_key_values is None:
            model_inputs = generated.unsqueeze(0)
        else:
            model_inputs = generated[-1:].unsqueeze(0)

        outputs = model(
            input_ids=model_inputs, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)
        logits = block_repeated_ngrams(logits, generated, config.no_repeat_ngram_size)

        if config.do_sample:
            logits = logits / max(config.temperature, 1e-5)
            logits = top_k_filter(logits, config.top_k)
            logits = top_p_filter(logits, config.top_p)
            logits = min_p_filter(logits, config.min_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=0)
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 4) Beam Search من الصفر (تعليمي — بدون تحسين الأداء الإنتاجي)
# ---------------------------------------------------------------------
@dataclass
class Beam:
    tokens: torch.Tensor
    log_prob: float
    finished: bool = False


@torch.no_grad()
def beam_search_decode(
    prompt: str,
    num_beams: int = 4,
    max_new_tokens: int = 40,
    length_penalty: float = 0.8,
) -> str:
    eos_id = tokenizer.eos_token_id
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
    beams = [Beam(tokens=input_ids, log_prob=0.0)]

    def normalized_score(beam: Beam) -> float:
        length = beam.tokens.size(0)
        return beam.log_prob / (length ** length_penalty)

    for _ in range(max_new_tokens):
        candidates = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            outputs = model(input_ids=beam.tokens.unsqueeze(0))
            log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)
            for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_tokens = torch.cat(
                    [beam.tokens, torch.tensor([tok_id], device=device)]
                )
                candidates.append(
                    Beam(
                        tokens=new_tokens,
                        log_prob=beam.log_prob + lp,
                        finished=(tok_id == eos_id),
                    )
                )

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:num_beams]

        if all(b.finished for b in beams):
            break

    best_beam = max(beams, key=normalized_score)
    return tokenizer.decode(best_beam.tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 5) التجربة المقارنة: نفس الـ Prompt، استراتيجيات مختلفة
# ---------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "في يوم من الأيام، كان هناك مهندس يحاول بناء نموذج لغوي، وفجأة"

    experiments = {
        "Greedy": GenerationConfig(do_sample=False),
        "Temperature=1.5 (عشوائية عالية)": GenerationConfig(
            do_sample=True, temperature=1.5, seed=42
        ),
        "Top-k=50": GenerationConfig(
            do_sample=True, temperature=1.0, top_k=50, seed=42
        ),
        "Top-p=0.9": GenerationConfig(
            do_sample=True, temperature=1.0, top_p=0.9, seed=42
        ),
        "متوازن (top_k=50, top_p=0.9, temp=0.8, rep_penalty=1.2)": GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            seed=42,
        ),
    }

    for name, cfg in experiments.items():
        text = generate(prompt, cfg)
        print(f"\n=== {name} ===\n{text}")

    beam_text = beam_search_decode(prompt, num_beams=5, max_new_tokens=40)
    print(f"\n=== Beam Search (B=5) ===\n{beam_text}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# مكافئ Beam Search الإنتاجي:
output_ids_beam = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=5,
    length_penalty=0.8,
    early_stopping=True,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids_beam[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
def compute_perplexity(text: str, model, tokenizer) -> float:
    """احسب Perplexity النص عبر إعادة تمريره للنموذج نفسه."""
    # تلميح: استخدم CrossEntropyLoss على logits مقابل input_ids المُزاحة
    ...

def compute_distinct_n(text: str, n: int = 2) -> float:
    """نسبة n-grams الفريدة إلى الإجمالي — مقياس تنوع كلاسيكي."""
    ...

def run_comparison(prompt: str, strategies: dict) -> None:
    """شغّل كل استراتيجية، اطبع النص + Perplexity + Distinct-2 + الزمن في جدول."""
    ...

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mirostat_sample(logits: torch.Tensor, target_surprise: float, mu: float, learning_rate: float = 0.1):
    """
    نسخة مبسطة من Mirostat v2:
    - رتّب الاحتمالات تنازليًا
    - احسب قيمة المفاجأة (surprise) = -log2(p) لكل رمز
    - اقتطع عند أول رمز تتجاوز مفاجأته mu
    - اعيّن من المجموعة المتبقية
    - حدّث mu بناءً على الفرق بين المفاجأة الفعلية والمستهدفة
    """
    ...
    # أعد (next_token, updated_mu)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 38: الفصل 38: استراتيجيات التوليد (Decoding Strategies)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_decoding() -> None:
    logits = [2.4, 1.7, 0.8, 0.2, -0.3]
    probabilities = softmax(logits)
    ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    top_k = ranked[:3]
    cumulative, top_p = (0.0, [])
    for index in ranked:
        top_p.append(index)
        cumulative += probabilities[index]
        if cumulative >= 0.85:
            break
    print('greedy token:', ranked[0])
    print('top-k candidates:', top_k)
    print('top-p candidates:', top_p, '| mass:', round(sum((probabilities[i] for i in top_p)), 3))
    print('temperature=0.5 top token:', max(range(len(logits)), key=lambda i: logits[i] / 0.5))

if __name__ == "__main__":
    print("Chapter 38 lab: Compare greedy, temperature, top-k, and top-p decoding")
    run_decoding()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
محرك استراتيجيات التوليد (Decoding Engine) من الصفر
يعمل فوق أي نموذج Causal LM من Hugging Face
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# يمكن استبدال الاسم بـ "gpt2" أو أي نموذج عربي آخر متوافق
MODEL_NAME = "aubmindlab/aragpt2-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------
# 1) إعدادات التوليد الموحّدة
# ---------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# 2) الأدوات المساعدة: عقوبة التكرار، حجب n-grams، الفلاتر
# ---------------------------------------------------------------------
def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """يعاقب الرموز التي ظهرت سابقًا في التسلسل."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    seen_tokens = torch.unique(generated_ids)
    seen_logits = logits[seen_tokens]
    penalized = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
    logits[seen_tokens] = penalized
    return logits


def block_repeated_ngrams(
    logits: torch.Tensor, generated_ids: torch.Tensor, n: int
) -> torch.Tensor:
    """يمنع تمامًا أي رمز يُنتج تكرار n-gram ظهر مسبقًا."""
    if n <= 0 or generated_ids.size(0) < n - 1:
        return logits
    ids = generated_ids.tolist()
    ngram_to_next = {}
    for i in range(len(ids) - n + 1):
        prefix = tuple(ids[i : i + n - 1])
        next_token = ids[i + n - 1]
        ngram_to_next.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(ids[-(n - 1):]) if n > 1 else tuple()
    banned_tokens = ngram_to_next.get(current_prefix, set())
    if banned_tokens:
        logits = logits.clone()
        for tok in banned_tokens:
            logits[tok] = float("-inf")
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """يُبقي فقط أعلى k رمزًا، ويحوّل الباقي إلى -inf."""
    if not k or k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k)
    min_keep_value = top_values[-1]
    return torch.where(
        logits < min_keep_value, torch.full_like(logits, float("-inf")), logits
    )


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus Sampling: يُبقي أصغر مجموعة تجمع احتمالها >= p."""
    if not p or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # نريد إبقاء أول رمز يتجاوز العتبة p أيضًا (لذا نزيح القناع يمينًا)
    sorted_mask_to_remove = cumulative_probs > p
    sorted_mask_to_remove[..., 1:] = sorted_mask_to_remove[..., :-1].clone()
    sorted_mask_to_remove[..., 0] = False

    mask_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    mask_to_remove.scatter_(0, sorted_indices, sorted_mask_to_remove)
    return logits.masked_fill(mask_to_remove, float("-inf"))


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p Sampling: عتبة نسبية لأعلى احتمال في التوزيع."""
    if not min_p or min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return logits.masked_fill(probs < threshold, float("-inf"))


# ---------------------------------------------------------------------
# 3) حلقة التوليد الموحّدة (Greedy + Sampling)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str, config: GenerationConfig) -> str:
    if config.seed is not None:
        torch.manual_seed(config.seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0]
    past_key_values = None
    eos_id = tokenizer.eos_token_id

    for _ in range(config.max_new_tokens):
        if past_key_values is None:
            model_inputs = generated.unsqueeze(0)
        else:
            model_inputs = generated[-1:].unsqueeze(0)

        outputs = model(
            input_ids=model_inputs, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)
        logits = block_repeated_ngrams(logits, generated, config.no_repeat_ngram_size)

        if config.do_sample:
            logits = logits / max(config.temperature, 1e-5)
            logits = top_k_filter(logits, config.top_k)
            logits = top_p_filter(logits, config.top_p)
            logits = min_p_filter(logits, config.min_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=0)
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 4) Beam Search من الصفر (تعليمي — بدون تحسين الأداء الإنتاجي)
# ---------------------------------------------------------------------
@dataclass
class Beam:
    tokens: torch.Tensor
    log_prob: float
    finished: bool = False


@torch.no_grad()
def beam_search_decode(
    prompt: str,
    num_beams: int = 4,
    max_new_tokens: int = 40,
    length_penalty: float = 0.8,
) -> str:
    eos_id = tokenizer.eos_token_id
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
    beams = [Beam(tokens=input_ids, log_prob=0.0)]

    def normalized_score(beam: Beam) -> float:
        length = beam.tokens.size(0)
        return beam.log_prob / (length ** length_penalty)

    for _ in range(max_new_tokens):
        candidates = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            outputs = model(input_ids=beam.tokens.unsqueeze(0))
            log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)
            for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_tokens = torch.cat(
                    [beam.tokens, torch.tensor([tok_id], device=device)]
                )
                candidates.append(
                    Beam(
                        tokens=new_tokens,
                        log_prob=beam.log_prob + lp,
                        finished=(tok_id == eos_id),
                    )
                )

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:num_beams]

        if all(b.finished for b in beams):
            break

    best_beam = max(beams, key=normalized_score)
    return tokenizer.decode(best_beam.tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 5) التجربة المقارنة: نفس الـ Prompt، استراتيجيات مختلفة
# ---------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "في يوم من الأيام، كان هناك مهندس يحاول بناء نموذج لغوي، وفجأة"

    experiments = {
        "Greedy": GenerationConfig(do_sample=False),
        "Temperature=1.5 (عشوائية عالية)": GenerationConfig(
            do_sample=True, temperature=1.5, seed=42
        ),
        "Top-k=50": GenerationConfig(
            do_sample=True, temperature=1.0, top_k=50, seed=42
        ),
        "Top-p=0.9": GenerationConfig(
            do_sample=True, temperature=1.0, top_p=0.9, seed=42
        ),
        "متوازن (top_k=50, top_p=0.9, temp=0.8, rep_penalty=1.2)": GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            seed=42,
        ),
    }

    for name, cfg in experiments.items():
        text = generate(prompt, cfg)
        print(f"\n=== {name} ===\n{text}")

    beam_text = beam_search_decode(prompt, num_beams=5, max_new_tokens=40)
    print(f"\n=== Beam Search (B=5) ===\n{beam_text}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# مكافئ Beam Search الإنتاجي:
output_ids_beam = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=5,
    length_penalty=0.8,
    early_stopping=True,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids_beam[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compute_perplexity(text: str, model, tokenizer) -> float:
    """احسب Perplexity النص عبر إعادة تمريره للنموذج نفسه."""
    # تلميح: استخدم CrossEntropyLoss على logits مقابل input_ids المُزاحة
    ...

def compute_distinct_n(text: str, n: int = 2) -> float:
    """نسبة n-grams الفريدة إلى الإجمالي — مقياس تنوع كلاسيكي."""
    ...

def run_comparison(prompt: str, strategies: dict) -> None:
    """شغّل كل استراتيجية، اطبع النص + Perplexity + Distinct-2 + الزمن في جدول."""
    ...

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mirostat_sample(logits: torch.Tensor, target_surprise: float, mu: float, learning_rate: float = 0.1):
    """
    نسخة مبسطة من Mirostat v2:
    - رتّب الاحتمالات تنازليًا
    - احسب قيمة المفاجأة (surprise) = -log2(p) لكل رمز
    - اقتطع عند أول رمز تتجاوز مفاجأته mu
    - اعيّن من المجموعة المتبقية
    - حدّث mu بناءً على الفرق بين المفاجأة الفعلية والمستهدفة
    """
    ...
    # أعد (next_token, updated_mu)

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — استراتيجيات التوليد (Decoding Strategies)
Large Language Model Engineering — Decoding

Book slug : llm-engineering-69-chapters
Chapter   : ch-38-chapter-38 (lesson 38)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_38.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 38: الفصل 38: استراتيجيات التوليد (Decoding Strategies)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_decoding() -> None:
    logits = [2.4, 1.7, 0.8, 0.2, -0.3]
    probabilities = softmax(logits)
    ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    top_k = ranked[:3]
    cumulative, top_p = (0.0, [])
    for index in ranked:
        top_p.append(index)
        cumulative += probabilities[index]
        if cumulative >= 0.85:
            break
    print('greedy token:', ranked[0])
    print('top-k candidates:', top_k)
    print('top-p candidates:', top_p, '| mass:', round(sum((probabilities[i] for i in top_p)), 3))
    print('temperature=0.5 top token:', max(range(len(logits)), key=lambda i: logits[i] / 0.5))

if __name__ == "__main__":
    print("Chapter 38 lab: Compare greedy, temperature, top-k, and top-p decoding")
    run_decoding()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
"""
محرك استراتيجيات التوليد (Decoding Engine) من الصفر
يعمل فوق أي نموذج Causal LM من Hugging Face
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# يمكن استبدال الاسم بـ "gpt2" أو أي نموذج عربي آخر متوافق
MODEL_NAME = "aubmindlab/aragpt2-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------
# 1) إعدادات التوليد الموحّدة
# ---------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# 2) الأدوات المساعدة: عقوبة التكرار، حجب n-grams، الفلاتر
# ---------------------------------------------------------------------
def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """يعاقب الرموز التي ظهرت سابقًا في التسلسل."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    seen_tokens = torch.unique(generated_ids)
    seen_logits = logits[seen_tokens]
    penalized = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
    logits[seen_tokens] = penalized
    return logits


def block_repeated_ngrams(
    logits: torch.Tensor, generated_ids: torch.Tensor, n: int
) -> torch.Tensor:
    """يمنع تمامًا أي رمز يُنتج تكرار n-gram ظهر مسبقًا."""
    if n <= 0 or generated_ids.size(0) < n - 1:
        return logits
    ids = generated_ids.tolist()
    ngram_to_next = {}
    for i in range(len(ids) - n + 1):
        prefix = tuple(ids[i : i + n - 1])
        next_token = ids[i + n - 1]
        ngram_to_next.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(ids[-(n - 1):]) if n > 1 else tuple()
    banned_tokens = ngram_to_next.get(current_prefix, set())
    if banned_tokens:
        logits = logits.clone()
        for tok in banned_tokens:
            logits[tok] = float("-inf")
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """يُبقي فقط أعلى k رمزًا، ويحوّل الباقي إلى -inf."""
    if not k or k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k)
    min_keep_value = top_values[-1]
    return torch.where(
        logits < min_keep_value, torch.full_like(logits, float("-inf")), logits
    )


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus Sampling: يُبقي أصغر مجموعة تجمع احتمالها >= p."""
    if not p or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # نريد إبقاء أول رمز يتجاوز العتبة p أيضًا (لذا نزيح القناع يمينًا)
    sorted_mask_to_remove = cumulative_probs > p
    sorted_mask_to_remove[..., 1:] = sorted_mask_to_remove[..., :-1].clone()
    sorted_mask_to_remove[..., 0] = False

    mask_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    mask_to_remove.scatter_(0, sorted_indices, sorted_mask_to_remove)
    return logits.masked_fill(mask_to_remove, float("-inf"))


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p Sampling: عتبة نسبية لأعلى احتمال في التوزيع."""
    if not min_p or min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return logits.masked_fill(probs < threshold, float("-inf"))


# ---------------------------------------------------------------------
# 3) حلقة التوليد الموحّدة (Greedy + Sampling)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str, config: GenerationConfig) -> str:
    if config.seed is not None:
        torch.manual_seed(config.seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0]
    past_key_values = None
    eos_id = tokenizer.eos_token_id

    for _ in range(config.max_new_tokens):
        if past_key_values is None:
            model_inputs = generated.unsqueeze(0)
        else:
            model_inputs = generated[-1:].unsqueeze(0)

        outputs = model(
            input_ids=model_inputs, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)
        logits = block_repeated_ngrams(logits, generated, config.no_repeat_ngram_size)

        if config.do_sample:
            logits = logits / max(config.temperature, 1e-5)
            logits = top_k_filter(logits, config.top_k)
            logits = top_p_filter(logits, config.top_p)
            logits = min_p_filter(logits, config.min_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=0)
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 4) Beam Search من الصفر (تعليمي — بدون تحسين الأداء الإنتاجي)
# ---------------------------------------------------------------------
@dataclass
class Beam:
    tokens: torch.Tensor
    log_prob: float
    finished: bool = False


@torch.no_grad()
def beam_search_decode(
    prompt: str,
    num_beams: int = 4,
    max_new_tokens: int = 40,
    length_penalty: float = 0.8,
) -> str:
    eos_id = tokenizer.eos_token_id
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
    beams = [Beam(tokens=input_ids, log_prob=0.0)]

    def normalized_score(beam: Beam) -> float:
        length = beam.tokens.size(0)
        return beam.log_prob / (length ** length_penalty)

    for _ in range(max_new_tokens):
        candidates = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            outputs = model(input_ids=beam.tokens.unsqueeze(0))
            log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)
            for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_tokens = torch.cat(
                    [beam.tokens, torch.tensor([tok_id], device=device)]
                )
                candidates.append(
                    Beam(
                        tokens=new_tokens,
                        log_prob=beam.log_prob + lp,
                        finished=(tok_id == eos_id),
                    )
                )

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:num_beams]

        if all(b.finished for b in beams):
            break

    best_beam = max(beams, key=normalized_score)
    return tokenizer.decode(best_beam.tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 5) التجربة المقارنة: نفس الـ Prompt، استراتيجيات مختلفة
# ---------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "في يوم من الأيام، كان هناك مهندس يحاول بناء نموذج لغوي، وفجأة"

    experiments = {
        "Greedy": GenerationConfig(do_sample=False),
        "Temperature=1.5 (عشوائية عالية)": GenerationConfig(
            do_sample=True, temperature=1.5, seed=42
        ),
        "Top-k=50": GenerationConfig(
            do_sample=True, temperature=1.0, top_k=50, seed=42
        ),
        "Top-p=0.9": GenerationConfig(
            do_sample=True, temperature=1.0, top_p=0.9, seed=42
        ),
        "متوازن (top_k=50, top_p=0.9, temp=0.8, rep_penalty=1.2)": GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            seed=42,
        ),
    }

    for name, cfg in experiments.items():
        text = generate(prompt, cfg)
        print(f"\n=== {name} ===\n{text}")

    beam_text = beam_search_decode(prompt, num_beams=5, max_new_tokens=40)
    print(f"\n=== Beam Search (B=5) ===\n{beam_text}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# مكافئ Beam Search الإنتاجي:
output_ids_beam = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=5,
    length_penalty=0.8,
    early_stopping=True,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids_beam[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
def compute_perplexity(text: str, model, tokenizer) -> float:
    """احسب Perplexity النص عبر إعادة تمريره للنموذج نفسه."""
    # تلميح: استخدم CrossEntropyLoss على logits مقابل input_ids المُزاحة
    ...

def compute_distinct_n(text: str, n: int = 2) -> float:
    """نسبة n-grams الفريدة إلى الإجمالي — مقياس تنوع كلاسيكي."""
    ...

def run_comparison(prompt: str, strategies: dict) -> None:
    """شغّل كل استراتيجية، اطبع النص + Perplexity + Distinct-2 + الزمن في جدول."""
    ...

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-08-multimodal-models/chapter-38.md
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mirostat_sample(logits: torch.Tensor, target_surprise: float, mu: float, learning_rate: float = 0.1):
    """
    نسخة مبسطة من Mirostat v2:
    - رتّب الاحتمالات تنازليًا
    - احسب قيمة المفاجأة (surprise) = -log2(p) لكل رمز
    - اقتطع عند أول رمز تتجاوز مفاجأته mu
    - اعيّن من المجموعة المتبقية
    - حدّث mu بناءً على الفرق بين المفاجأة الفعلية والمستهدفة
    """
    ...
    # أعد (next_token, updated_mu)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 38: الفصل 38: استراتيجيات التوليد (Decoding Strategies)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_decoding() -> None:
    logits = [2.4, 1.7, 0.8, 0.2, -0.3]
    probabilities = softmax(logits)
    ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    top_k = ranked[:3]
    cumulative, top_p = (0.0, [])
    for index in ranked:
        top_p.append(index)
        cumulative += probabilities[index]
        if cumulative >= 0.85:
            break
    print('greedy token:', ranked[0])
    print('top-k candidates:', top_k)
    print('top-p candidates:', top_p, '| mass:', round(sum((probabilities[i] for i in top_p)), 3))
    print('temperature=0.5 top token:', max(range(len(logits)), key=lambda i: logits[i] / 0.5))

if __name__ == "__main__":
    print("Chapter 38 lab: Compare greedy, temperature, top-k, and top-p decoding")
    run_decoding()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
محرك استراتيجيات التوليد (Decoding Engine) من الصفر
يعمل فوق أي نموذج Causal LM من Hugging Face
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# يمكن استبدال الاسم بـ "gpt2" أو أي نموذج عربي آخر متوافق
MODEL_NAME = "aubmindlab/aragpt2-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------
# 1) إعدادات التوليد الموحّدة
# ---------------------------------------------------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# 2) الأدوات المساعدة: عقوبة التكرار، حجب n-grams، الفلاتر
# ---------------------------------------------------------------------
def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """يعاقب الرموز التي ظهرت سابقًا في التسلسل."""
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    seen_tokens = torch.unique(generated_ids)
    seen_logits = logits[seen_tokens]
    penalized = torch.where(seen_logits > 0, seen_logits / penalty, seen_logits * penalty)
    logits[seen_tokens] = penalized
    return logits


def block_repeated_ngrams(
    logits: torch.Tensor, generated_ids: torch.Tensor, n: int
) -> torch.Tensor:
    """يمنع تمامًا أي رمز يُنتج تكرار n-gram ظهر مسبقًا."""
    if n <= 0 or generated_ids.size(0) < n - 1:
        return logits
    ids = generated_ids.tolist()
    ngram_to_next = {}
    for i in range(len(ids) - n + 1):
        prefix = tuple(ids[i : i + n - 1])
        next_token = ids[i + n - 1]
        ngram_to_next.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(ids[-(n - 1):]) if n > 1 else tuple()
    banned_tokens = ngram_to_next.get(current_prefix, set())
    if banned_tokens:
        logits = logits.clone()
        for tok in banned_tokens:
            logits[tok] = float("-inf")
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """يُبقي فقط أعلى k رمزًا، ويحوّل الباقي إلى -inf."""
    if not k or k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k)
    min_keep_value = top_values[-1]
    return torch.where(
        logits < min_keep_value, torch.full_like(logits, float("-inf")), logits
    )


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus Sampling: يُبقي أصغر مجموعة تجمع احتمالها >= p."""
    if not p or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # نريد إبقاء أول رمز يتجاوز العتبة p أيضًا (لذا نزيح القناع يمينًا)
    sorted_mask_to_remove = cumulative_probs > p
    sorted_mask_to_remove[..., 1:] = sorted_mask_to_remove[..., :-1].clone()
    sorted_mask_to_remove[..., 0] = False

    mask_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    mask_to_remove.scatter_(0, sorted_indices, sorted_mask_to_remove)
    return logits.masked_fill(mask_to_remove, float("-inf"))


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p Sampling: عتبة نسبية لأعلى احتمال في التوزيع."""
    if not min_p or min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    return logits.masked_fill(probs < threshold, float("-inf"))


# ---------------------------------------------------------------------
# 3) حلقة التوليد الموحّدة (Greedy + Sampling)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str, config: GenerationConfig) -> str:
    if config.seed is not None:
        torch.manual_seed(config.seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids[0]
    past_key_values = None
    eos_id = tokenizer.eos_token_id

    for _ in range(config.max_new_tokens):
        if past_key_values is None:
            model_inputs = generated.unsqueeze(0)
        else:
            model_inputs = generated[-1:].unsqueeze(0)

        outputs = model(
            input_ids=model_inputs, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)
        logits = block_repeated_ngrams(logits, generated, config.no_repeat_ngram_size)

        if config.do_sample:
            logits = logits / max(config.temperature, 1e-5)
            logits = top_k_filter(logits, config.top_k)
            logits = top_p_filter(logits, config.top_p)
            logits = min_p_filter(logits, config.min_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=0)
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 4) Beam Search من الصفر (تعليمي — بدون تحسين الأداء الإنتاجي)
# ---------------------------------------------------------------------
@dataclass
class Beam:
    tokens: torch.Tensor
    log_prob: float
    finished: bool = False


@torch.no_grad()
def beam_search_decode(
    prompt: str,
    num_beams: int = 4,
    max_new_tokens: int = 40,
    length_penalty: float = 0.8,
) -> str:
    eos_id = tokenizer.eos_token_id
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
    beams = [Beam(tokens=input_ids, log_prob=0.0)]

    def normalized_score(beam: Beam) -> float:
        length = beam.tokens.size(0)
        return beam.log_prob / (length ** length_penalty)

    for _ in range(max_new_tokens):
        candidates = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            outputs = model(input_ids=beam.tokens.unsqueeze(0))
            log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)
            for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_tokens = torch.cat(
                    [beam.tokens, torch.tensor([tok_id], device=device)]
                )
                candidates.append(
                    Beam(
                        tokens=new_tokens,
                        log_prob=beam.log_prob + lp,
                        finished=(tok_id == eos_id),
                    )
                )

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:num_beams]

        if all(b.finished for b in beams):
            break

    best_beam = max(beams, key=normalized_score)
    return tokenizer.decode(best_beam.tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------
# 5) التجربة المقارنة: نفس الـ Prompt، استراتيجيات مختلفة
# ---------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "في يوم من الأيام، كان هناك مهندس يحاول بناء نموذج لغوي، وفجأة"

    experiments = {
        "Greedy": GenerationConfig(do_sample=False),
        "Temperature=1.5 (عشوائية عالية)": GenerationConfig(
            do_sample=True, temperature=1.5, seed=42
        ),
        "Top-k=50": GenerationConfig(
            do_sample=True, temperature=1.0, top_k=50, seed=42
        ),
        "Top-p=0.9": GenerationConfig(
            do_sample=True, temperature=1.0, top_p=0.9, seed=42
        ),
        "متوازن (top_k=50, top_p=0.9, temp=0.8, rep_penalty=1.2)": GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            seed=42,
        ),
    }

    for name, cfg in experiments.items():
        text = generate(prompt, cfg)
        print(f"\n=== {name} ===\n{text}")

    beam_text = beam_search_decode(prompt, num_beams=5, max_new_tokens=40)
    print(f"\n=== Beam Search (B=5) ===\n{beam_text}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# مكافئ Beam Search الإنتاجي:
output_ids_beam = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=5,
    length_penalty=0.8,
    early_stopping=True,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output_ids_beam[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def compute_perplexity(text: str, model, tokenizer) -> float:
    """احسب Perplexity النص عبر إعادة تمريره للنموذج نفسه."""
    # تلميح: استخدم CrossEntropyLoss على logits مقابل input_ids المُزاحة
    ...

def compute_distinct_n(text: str, n: int = 2) -> float:
    """نسبة n-grams الفريدة إلى الإجمالي — مقياس تنوع كلاسيكي."""
    ...

def run_comparison(prompt: str, strategies: dict) -> None:
    """شغّل كل استراتيجية، اطبع النص + Perplexity + Distinct-2 + الزمن في جدول."""
    ...

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-38-chapter-38.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def mirostat_sample(logits: torch.Tensor, target_surprise: float, mu: float, learning_rate: float = 0.1):
    """
    نسخة مبسطة من Mirostat v2:
    - رتّب الاحتمالات تنازليًا
    - احسب قيمة المفاجأة (surprise) = -log2(p) لكل رمز
    - اقتطع عند أول رمز تتجاوز مفاجأته mu
    - اعيّن من المجموعة المتبقية
    - حدّث mu بناءً على الفرق بين المفاجأة الفعلية والمستهدفة
    """
    ...
    # أعد (next_token, updated_mu)
