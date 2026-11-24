#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching
Large Language Model Engineering — Quantization

Book slug : llm-engineering-69-chapters
Chapter   : ch-59-chapter-59 (lesson 59)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-59-chapter-59.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching
Large Language Model Engineering — Quantization

Book slug : llm-engineering-69-chapters
Chapter   : ch-59-chapter-59 (lesson 59)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-59-chapter-59.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching
Large Language Model Engineering — Quantization

Book slug : llm-engineering-69-chapters
Chapter   : ch-59-chapter-59 (lesson 59)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_59.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 59: الفصل 59: تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_inference() -> None:
    weights, scales, quantized = ([0.12, -0.77, 0.34, 1.0], 127, [])
    for value in weights:
        quantized.append(round(value * scales))
    restored = [value / scales for value in quantized]
    acceptance = 0.7
    draft_tokens = 8
    expected_accepted = draft_tokens * acceptance
    print('FP weights:', weights)
    print('INT8 reconstruction:', [round(x, 3) for x in restored])
    print('estimated accepted draft tokens:', round(expected_accepted, 2), 'of', draft_tokens)

if __name__ == "__main__":
    print("Chapter 59 lab: Compare quantization and speculative decoding costs")
    run_inference()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-59.md
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer

MODEL_NAME = "facebook/opt-1.3b"  # نموذج صغير لأغراض العرض؛ المبدأ ينطبق على 70B بنفس الطريقة

def load_fp16_model():
    """تحميل النموذج بالدقة الكاملة (baseline) لقياس الذاكرة والسرعة الأصلية"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    return model, tokenizer


def quantize_with_gptq(model_name: str, bits: int = 4):
    """تكميم GPTQ حقيقي باستخدام مكتبة optimum من Hugging Face"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPTQ يحتاج بيانات معايرة (calibration dataset) صغيرة لحساب Hessian التقريبية
    calibration_dataset = [
        "الذكاء الاصطناعي يُغيّر طريقة عملنا في كل القطاعات.",
        "تحسين الاستدلال أمر حيوي لتشغيل النماذج الكبيرة بكفاءة.",
        "الحوسبة السحابية توفر مرونة كبيرة للشركات الناشئة.",
    ] * 20  # يُستحسن عمليًا 128-512 عينة متنوعة من بيانات حقيقية مشابهة لتوزيع الاستخدام الفعلي

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=128,          # حجم المجموعة لمشاركة نفس scale/zero_point (دقة أعلى مع تجميع أصغر)
        desc_act=False,          # ترتيب الأعمدة حسب الأهمية (دقة أعلى لكن أبطأ قليلًا في الاستدلال)
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model, tokenizer


def measure_memory_and_speed(model, tokenizer, prompt: str, label: str, n_runs: int = 5):
    """قياس موحّد: ذاكرة GPU المستخدمة + زمن توليد 50 توكن"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # تشغيل تمهيدي (warmup) لا يُحتسب في القياس
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_second = 50 / elapsed

    print(f"\n📊 {label}")
    print(f"  ذروة ذاكرة GPU: {peak_memory_gb:.2f} GB")
    print(f"  زمن توليد 50 توكن: {elapsed:.3f} ثانية")
    print(f"  السرعة: {tokens_per_second:.1f} توكن/ثانية")

    return {"memory_gb": peak_memory_gb, "tokens_per_sec": tokens_per_second}


if __name__ == "__main__":
    prompt = "الذكاء الاصطناعي في المستقبل سوف"

    print("=== تحميل النموذج الأصلي (FP16) ===")
    fp16_model, tokenizer = load_fp16_model()
    fp16_stats = measure_memory_and_speed(fp16_model, tokenizer, prompt, "FP16 (الأصلي)")
    del fp16_model
    torch.cuda.empty_cache()

    print("\n=== تكميم النموذج بـ GPTQ (4-bit) ===")
    gptq_model, tokenizer = quantize_with_gptq(MODEL_NAME, bits=4)
    gptq_stats = measure_memory_and_speed(gptq_model, tokenizer, prompt, "GPTQ INT4")

    print(f"\n✅ تقليل الذاكرة: {fp16_stats['memory_gb'] / gptq_stats['memory_gb']:.2f}x")
    print(f"✅ تغيّر السرعة: {gptq_stats['tokens_per_sec'] / fp16_stats['tokens_per_sec']:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-59.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    """
    تطبيق تعليمي كامل لـ Speculative Decoding يوضح الآلية الرياضية خطوة بخطوة
    (في الإنتاج: استخدم دعم vLLM/TGI المدمج، لكن فهم الآلية الداخلية ضروري للمقابلات)
    """

    def __init__(self, draft_model_name: str, target_model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _get_next_token_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """يُرجع توزيع الاحتمالات الكامل (softmax) للتوكن التالي فقط"""
        logits = model(input_ids).logits[:, -1, :]
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, K: int = 4) -> dict:
        """
        الخوارزمية الكاملة: K توكن مقترحة من draft model، تحقق موازٍ من target model
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_tokens = 0
        accepted_total, proposed_total = 0, 0

        while generated_tokens < max_new_tokens:
            # ─── المرحلة 1: النموذج الصغير يقترح K توكن بتوليد تسلسلي سريع ───
            draft_input = input_ids.clone()
            draft_probs_list = []
            draft_tokens = []

            for _ in range(K):
                probs = self._get_next_token_probs(self.draft_model, draft_input)
                next_token = torch.multinomial(probs, num_samples=1)
                draft_probs_list.append(probs.gather(-1, next_token))
                draft_tokens.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)  # [1, K]
            candidate_sequence = torch.cat([input_ids, draft_tokens_tensor], dim=1)

            # ─── المرحلة 2: النموذج الكبير يتحقق من كل K توكن بنداء واحد موازٍ ───
            target_logits = self.target_model(candidate_sequence).logits
            # نحتاج احتمالات target عند كل موضع اقترح فيه draft توكنًا
            target_probs_at_positions = F.softmax(
                target_logits[:, input_ids.shape[1] - 1: -1, :], dim=-1
            )  # [1, K, vocab_size]

            # ─── المرحلة 3: قرار القبول/الرفض لكل توكن بالترتيب ───
            n_accepted = 0
            for i in range(K):
                token_id = draft_tokens[i]
                p_target = target_probs_at_positions[:, i, :].gather(-1, token_id).item()
                q_draft = draft_probs_list[i].item()

                acceptance_ratio = min(1.0, p_target / (q_draft + 1e-10))
                r = torch.rand(1).item()

                proposed_total += 1
                if r <= acceptance_ratio:
                    n_accepted += 1
                    accepted_total += 1
                else:
                    break  # نرفض هذا التوكن وكل ما بعده (اعتمدوا على توكن مرفوض)

            # إضافة التوكنات المقبولة فعليًا للتسلسل النهائي
            if n_accepted > 0:
                input_ids = torch.cat([input_ids, draft_tokens_tensor[:, :n_accepted]], dim=1)
                generated_tokens += n_accepted

            # إذا رُفض توكن، نولّد توكنًا تصحيحيًا من توزيع معدَّل (يضمن صحة التوزيع النهائي رياضيًا)
            if n_accepted < K:
                p_target_rejected = target_probs_at_positions[:, n_accepted, :]
                q_draft_rejected = self._get_next_token_probs(
                    self.draft_model, torch.cat([input_ids], dim=1)
                )
                corrected_dist = torch.clamp(p_target_rejected - q_draft_rejected, min=0)
                corrected_dist = corrected_dist / (corrected_dist.sum() + 1e-10)
                correction_token = torch.multinomial(corrected_dist, num_samples=1)
                input_ids = torch.cat([input_ids, correction_token], dim=1)
                generated_tokens += 1

        acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "text": generated_text,
            "acceptance_rate": acceptance_rate,
            "tokens_generated": generated_tokens,
        }


# ───────────── تشغيل تجريبي: نموذج صغير كـ draft، نموذج أكبر كـ target ─────────────
if __name__ == "__main__":
    decoder = SpeculativeDecoder(
        draft_model_name="facebook/opt-125m",   # نموذج صغير سريع جدًا
        target_model_name="facebook/opt-1.3b",  # نموذج أكبر وأدق
    )

    result = decoder.generate("الذكاء الاصطناعي سيُغيّر", max_new_tokens=40, K=4)
    print(f"النص المُولَّد: {result['text']}")
    print(f"نسبة قبول اقتراحات النموذج الصغير: {result['acceptance_rate']*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-59.md
# ────────────────────────────────────────────────────────────────────────
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class InferenceRequest:
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None


class StaticBatchingSimulator:
    """المحاكي التقليدي: ينتظر الكل حتى ينتهي أطول طلب في الدفعة قبل قبول دفعة جديدة"""

    def __init__(self, batch_size: int, time_per_token: float = 0.02):
        self.batch_size = batch_size
        self.time_per_token = time_per_token

    def run(self, requests: List[InferenceRequest]) -> float:
        total_time = 0.0
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            max_tokens_in_batch = max(r.max_new_tokens for r in batch)
            # الجميع ينتظر حتى ينتهي أطول طلب، حتى لو انتهى هو بعد توكن واحد فقط!
            batch_time = max_tokens_in_batch * self.time_per_token
            total_time += batch_time
            for r in batch:
                r.finish_time = total_time
        return total_time


class ContinuousBatchingSimulator:
    """
    محاكاة مبسّطة لـ Continuous Batching: في كل "خطوة" (step) نولّد توكنًا واحدًا
    لكل الطلبات النشطة، وفور انتهاء طلب، يحل محله فورًا طلب جديد من قائمة الانتظار
    """

    def __init__(self, max_batch_size: int, time_per_step: float = 0.02):
        self.max_batch_size = max_batch_size
        self.time_per_step = time_per_step

    def run(self, requests: List[InferenceRequest]) -> float:
        queue = sorted(requests, key=lambda r: r.arrival_time)
        active: List[InferenceRequest] = []
        current_time = 0.0

        while queue or active:
            # املأ الدفعة النشطة بطلبات جديدة من قائمة الانتظار حتى السقف الأقصى
            while len(active) < self.max_batch_size and queue:
                new_request = queue.pop(0)
                new_request.status = RequestStatus.RUNNING
                active.append(new_request)

            if not active:
                break

            # خطوة واحدة: كل الطلبات النشطة تتقدم بتوكن واحد في نفس "النداء" الموازي
            current_time += self.time_per_step
            still_active = []
            for r in active:
                r.generated_tokens += 1
                if r.generated_tokens >= r.max_new_tokens:
                    r.status = RequestStatus.FINISHED
                    r.finish_time = current_time
                    # ✨ فور انتهاء هذا الطلب، تتحرر "فتحة" يملؤها طلب جديد فورًا في الخطوة القادمة
                else:
                    still_active.append(r)
            active = still_active

        return current_time


def compare_batching_strategies():
    """مقارنة كمية: طلبات متنوعة الطول (بعضها قصير جدًا وبعضها طويل) - الحالة الواقعية الشائعة"""
    random.seed(42)
    requests_static = [
        InferenceRequest(request_id=i, prompt_tokens=50, max_new_tokens=random.choice([10, 20, 100, 150]))
        for i in range(20)
    ]
    requests_continuous = [
        InferenceRequest(request_id=r.request_id, prompt_tokens=r.prompt_tokens, max_new_tokens=r.max_new_tokens)
        for r in requests_static
    ]  # نفس الطلبات بالضبط لمقارنة عادلة

    static_sim = StaticBatchingSimulator(batch_size=4, time_per_token=0.02)
    continuous_sim = ContinuousBatchingSimulator(max_batch_size=4, time_per_step=0.02)

    static_total_time = static_sim.run(requests_static)
    continuous_total_time = continuous_sim.run(requests_continuous)

    print(f"⏱️  Static Batching:     {static_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"⏱️  Continuous Batching: {continuous_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"🚀 التسريع: {static_total_time / continuous_total_time:.2f}x")

    avg_wait_static = sum(r.finish_time for r in requests_static) / len(requests_static)
    avg_wait_continuous = sum(r.finish_time for r in requests_continuous) / len(requests_continuous)
    print(f"\n📊 متوسط زمن الانتظار (Static):     {avg_wait_static:.2f} ثانية")
    print(f"📊 متوسط زمن الانتظار (Continuous): {avg_wait_continuous:.2f} ثانية")


if __name__ == "__main__":
    compare_batching_strategies()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 59: الفصل 59: تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_inference() -> None:
    weights, scales, quantized = ([0.12, -0.77, 0.34, 1.0], 127, [])
    for value in weights:
        quantized.append(round(value * scales))
    restored = [value / scales for value in quantized]
    acceptance = 0.7
    draft_tokens = 8
    expected_accepted = draft_tokens * acceptance
    print('FP weights:', weights)
    print('INT8 reconstruction:', [round(x, 3) for x in restored])
    print('estimated accepted draft tokens:', round(expected_accepted, 2), 'of', draft_tokens)

if __name__ == "__main__":
    print("Chapter 59 lab: Compare quantization and speculative decoding costs")
    run_inference()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer

MODEL_NAME = "facebook/opt-1.3b"  # نموذج صغير لأغراض العرض؛ المبدأ ينطبق على 70B بنفس الطريقة

def load_fp16_model():
    """تحميل النموذج بالدقة الكاملة (baseline) لقياس الذاكرة والسرعة الأصلية"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    return model, tokenizer


def quantize_with_gptq(model_name: str, bits: int = 4):
    """تكميم GPTQ حقيقي باستخدام مكتبة optimum من Hugging Face"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPTQ يحتاج بيانات معايرة (calibration dataset) صغيرة لحساب Hessian التقريبية
    calibration_dataset = [
        "الذكاء الاصطناعي يُغيّر طريقة عملنا في كل القطاعات.",
        "تحسين الاستدلال أمر حيوي لتشغيل النماذج الكبيرة بكفاءة.",
        "الحوسبة السحابية توفر مرونة كبيرة للشركات الناشئة.",
    ] * 20  # يُستحسن عمليًا 128-512 عينة متنوعة من بيانات حقيقية مشابهة لتوزيع الاستخدام الفعلي

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=128,          # حجم المجموعة لمشاركة نفس scale/zero_point (دقة أعلى مع تجميع أصغر)
        desc_act=False,          # ترتيب الأعمدة حسب الأهمية (دقة أعلى لكن أبطأ قليلًا في الاستدلال)
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model, tokenizer


def measure_memory_and_speed(model, tokenizer, prompt: str, label: str, n_runs: int = 5):
    """قياس موحّد: ذاكرة GPU المستخدمة + زمن توليد 50 توكن"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # تشغيل تمهيدي (warmup) لا يُحتسب في القياس
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_second = 50 / elapsed

    print(f"\n📊 {label}")
    print(f"  ذروة ذاكرة GPU: {peak_memory_gb:.2f} GB")
    print(f"  زمن توليد 50 توكن: {elapsed:.3f} ثانية")
    print(f"  السرعة: {tokens_per_second:.1f} توكن/ثانية")

    return {"memory_gb": peak_memory_gb, "tokens_per_sec": tokens_per_second}


if __name__ == "__main__":
    prompt = "الذكاء الاصطناعي في المستقبل سوف"

    print("=== تحميل النموذج الأصلي (FP16) ===")
    fp16_model, tokenizer = load_fp16_model()
    fp16_stats = measure_memory_and_speed(fp16_model, tokenizer, prompt, "FP16 (الأصلي)")
    del fp16_model
    torch.cuda.empty_cache()

    print("\n=== تكميم النموذج بـ GPTQ (4-bit) ===")
    gptq_model, tokenizer = quantize_with_gptq(MODEL_NAME, bits=4)
    gptq_stats = measure_memory_and_speed(gptq_model, tokenizer, prompt, "GPTQ INT4")

    print(f"\n✅ تقليل الذاكرة: {fp16_stats['memory_gb'] / gptq_stats['memory_gb']:.2f}x")
    print(f"✅ تغيّر السرعة: {gptq_stats['tokens_per_sec'] / fp16_stats['tokens_per_sec']:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    """
    تطبيق تعليمي كامل لـ Speculative Decoding يوضح الآلية الرياضية خطوة بخطوة
    (في الإنتاج: استخدم دعم vLLM/TGI المدمج، لكن فهم الآلية الداخلية ضروري للمقابلات)
    """

    def __init__(self, draft_model_name: str, target_model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _get_next_token_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """يُرجع توزيع الاحتمالات الكامل (softmax) للتوكن التالي فقط"""
        logits = model(input_ids).logits[:, -1, :]
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, K: int = 4) -> dict:
        """
        الخوارزمية الكاملة: K توكن مقترحة من draft model، تحقق موازٍ من target model
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_tokens = 0
        accepted_total, proposed_total = 0, 0

        while generated_tokens < max_new_tokens:
            # ─── المرحلة 1: النموذج الصغير يقترح K توكن بتوليد تسلسلي سريع ───
            draft_input = input_ids.clone()
            draft_probs_list = []
            draft_tokens = []

            for _ in range(K):
                probs = self._get_next_token_probs(self.draft_model, draft_input)
                next_token = torch.multinomial(probs, num_samples=1)
                draft_probs_list.append(probs.gather(-1, next_token))
                draft_tokens.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)  # [1, K]
            candidate_sequence = torch.cat([input_ids, draft_tokens_tensor], dim=1)

            # ─── المرحلة 2: النموذج الكبير يتحقق من كل K توكن بنداء واحد موازٍ ───
            target_logits = self.target_model(candidate_sequence).logits
            # نحتاج احتمالات target عند كل موضع اقترح فيه draft توكنًا
            target_probs_at_positions = F.softmax(
                target_logits[:, input_ids.shape[1] - 1: -1, :], dim=-1
            )  # [1, K, vocab_size]

            # ─── المرحلة 3: قرار القبول/الرفض لكل توكن بالترتيب ───
            n_accepted = 0
            for i in range(K):
                token_id = draft_tokens[i]
                p_target = target_probs_at_positions[:, i, :].gather(-1, token_id).item()
                q_draft = draft_probs_list[i].item()

                acceptance_ratio = min(1.0, p_target / (q_draft + 1e-10))
                r = torch.rand(1).item()

                proposed_total += 1
                if r <= acceptance_ratio:
                    n_accepted += 1
                    accepted_total += 1
                else:
                    break  # نرفض هذا التوكن وكل ما بعده (اعتمدوا على توكن مرفوض)

            # إضافة التوكنات المقبولة فعليًا للتسلسل النهائي
            if n_accepted > 0:
                input_ids = torch.cat([input_ids, draft_tokens_tensor[:, :n_accepted]], dim=1)
                generated_tokens += n_accepted

            # إذا رُفض توكن، نولّد توكنًا تصحيحيًا من توزيع معدَّل (يضمن صحة التوزيع النهائي رياضيًا)
            if n_accepted < K:
                p_target_rejected = target_probs_at_positions[:, n_accepted, :]
                q_draft_rejected = self._get_next_token_probs(
                    self.draft_model, torch.cat([input_ids], dim=1)
                )
                corrected_dist = torch.clamp(p_target_rejected - q_draft_rejected, min=0)
                corrected_dist = corrected_dist / (corrected_dist.sum() + 1e-10)
                correction_token = torch.multinomial(corrected_dist, num_samples=1)
                input_ids = torch.cat([input_ids, correction_token], dim=1)
                generated_tokens += 1

        acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "text": generated_text,
            "acceptance_rate": acceptance_rate,
            "tokens_generated": generated_tokens,
        }


# ───────────── تشغيل تجريبي: نموذج صغير كـ draft، نموذج أكبر كـ target ─────────────
if __name__ == "__main__":
    decoder = SpeculativeDecoder(
        draft_model_name="facebook/opt-125m",   # نموذج صغير سريع جدًا
        target_model_name="facebook/opt-1.3b",  # نموذج أكبر وأدق
    )

    result = decoder.generate("الذكاء الاصطناعي سيُغيّر", max_new_tokens=40, K=4)
    print(f"النص المُولَّد: {result['text']}")
    print(f"نسبة قبول اقتراحات النموذج الصغير: {result['acceptance_rate']*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class InferenceRequest:
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None


class StaticBatchingSimulator:
    """المحاكي التقليدي: ينتظر الكل حتى ينتهي أطول طلب في الدفعة قبل قبول دفعة جديدة"""

    def __init__(self, batch_size: int, time_per_token: float = 0.02):
        self.batch_size = batch_size
        self.time_per_token = time_per_token

    def run(self, requests: List[InferenceRequest]) -> float:
        total_time = 0.0
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            max_tokens_in_batch = max(r.max_new_tokens for r in batch)
            # الجميع ينتظر حتى ينتهي أطول طلب، حتى لو انتهى هو بعد توكن واحد فقط!
            batch_time = max_tokens_in_batch * self.time_per_token
            total_time += batch_time
            for r in batch:
                r.finish_time = total_time
        return total_time


class ContinuousBatchingSimulator:
    """
    محاكاة مبسّطة لـ Continuous Batching: في كل "خطوة" (step) نولّد توكنًا واحدًا
    لكل الطلبات النشطة، وفور انتهاء طلب، يحل محله فورًا طلب جديد من قائمة الانتظار
    """

    def __init__(self, max_batch_size: int, time_per_step: float = 0.02):
        self.max_batch_size = max_batch_size
        self.time_per_step = time_per_step

    def run(self, requests: List[InferenceRequest]) -> float:
        queue = sorted(requests, key=lambda r: r.arrival_time)
        active: List[InferenceRequest] = []
        current_time = 0.0

        while queue or active:
            # املأ الدفعة النشطة بطلبات جديدة من قائمة الانتظار حتى السقف الأقصى
            while len(active) < self.max_batch_size and queue:
                new_request = queue.pop(0)
                new_request.status = RequestStatus.RUNNING
                active.append(new_request)

            if not active:
                break

            # خطوة واحدة: كل الطلبات النشطة تتقدم بتوكن واحد في نفس "النداء" الموازي
            current_time += self.time_per_step
            still_active = []
            for r in active:
                r.generated_tokens += 1
                if r.generated_tokens >= r.max_new_tokens:
                    r.status = RequestStatus.FINISHED
                    r.finish_time = current_time
                    # ✨ فور انتهاء هذا الطلب، تتحرر "فتحة" يملؤها طلب جديد فورًا في الخطوة القادمة
                else:
                    still_active.append(r)
            active = still_active

        return current_time


def compare_batching_strategies():
    """مقارنة كمية: طلبات متنوعة الطول (بعضها قصير جدًا وبعضها طويل) - الحالة الواقعية الشائعة"""
    random.seed(42)
    requests_static = [
        InferenceRequest(request_id=i, prompt_tokens=50, max_new_tokens=random.choice([10, 20, 100, 150]))
        for i in range(20)
    ]
    requests_continuous = [
        InferenceRequest(request_id=r.request_id, prompt_tokens=r.prompt_tokens, max_new_tokens=r.max_new_tokens)
        for r in requests_static
    ]  # نفس الطلبات بالضبط لمقارنة عادلة

    static_sim = StaticBatchingSimulator(batch_size=4, time_per_token=0.02)
    continuous_sim = ContinuousBatchingSimulator(max_batch_size=4, time_per_step=0.02)

    static_total_time = static_sim.run(requests_static)
    continuous_total_time = continuous_sim.run(requests_continuous)

    print(f"⏱️  Static Batching:     {static_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"⏱️  Continuous Batching: {continuous_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"🚀 التسريع: {static_total_time / continuous_total_time:.2f}x")

    avg_wait_static = sum(r.finish_time for r in requests_static) / len(requests_static)
    avg_wait_continuous = sum(r.finish_time for r in requests_continuous) / len(requests_continuous)
    print(f"\n📊 متوسط زمن الانتظار (Static):     {avg_wait_static:.2f} ثانية")
    print(f"📊 متوسط زمن الانتظار (Continuous): {avg_wait_continuous:.2f} ثانية")


if __name__ == "__main__":
    compare_batching_strategies()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching
Large Language Model Engineering — Quantization

Book slug : llm-engineering-69-chapters
Chapter   : ch-59-chapter-59 (lesson 59)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_59.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 59: الفصل 59: تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_inference() -> None:
    weights, scales, quantized = ([0.12, -0.77, 0.34, 1.0], 127, [])
    for value in weights:
        quantized.append(round(value * scales))
    restored = [value / scales for value in quantized]
    acceptance = 0.7
    draft_tokens = 8
    expected_accepted = draft_tokens * acceptance
    print('FP weights:', weights)
    print('INT8 reconstruction:', [round(x, 3) for x in restored])
    print('estimated accepted draft tokens:', round(expected_accepted, 2), 'of', draft_tokens)

if __name__ == "__main__":
    print("Chapter 59 lab: Compare quantization and speculative decoding costs")
    run_inference()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-11-deployment-llmops/chapter-59.md
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer

MODEL_NAME = "facebook/opt-1.3b"  # نموذج صغير لأغراض العرض؛ المبدأ ينطبق على 70B بنفس الطريقة

def load_fp16_model():
    """تحميل النموذج بالدقة الكاملة (baseline) لقياس الذاكرة والسرعة الأصلية"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    return model, tokenizer


def quantize_with_gptq(model_name: str, bits: int = 4):
    """تكميم GPTQ حقيقي باستخدام مكتبة optimum من Hugging Face"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPTQ يحتاج بيانات معايرة (calibration dataset) صغيرة لحساب Hessian التقريبية
    calibration_dataset = [
        "الذكاء الاصطناعي يُغيّر طريقة عملنا في كل القطاعات.",
        "تحسين الاستدلال أمر حيوي لتشغيل النماذج الكبيرة بكفاءة.",
        "الحوسبة السحابية توفر مرونة كبيرة للشركات الناشئة.",
    ] * 20  # يُستحسن عمليًا 128-512 عينة متنوعة من بيانات حقيقية مشابهة لتوزيع الاستخدام الفعلي

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=128,          # حجم المجموعة لمشاركة نفس scale/zero_point (دقة أعلى مع تجميع أصغر)
        desc_act=False,          # ترتيب الأعمدة حسب الأهمية (دقة أعلى لكن أبطأ قليلًا في الاستدلال)
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model, tokenizer


def measure_memory_and_speed(model, tokenizer, prompt: str, label: str, n_runs: int = 5):
    """قياس موحّد: ذاكرة GPU المستخدمة + زمن توليد 50 توكن"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # تشغيل تمهيدي (warmup) لا يُحتسب في القياس
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_second = 50 / elapsed

    print(f"\n📊 {label}")
    print(f"  ذروة ذاكرة GPU: {peak_memory_gb:.2f} GB")
    print(f"  زمن توليد 50 توكن: {elapsed:.3f} ثانية")
    print(f"  السرعة: {tokens_per_second:.1f} توكن/ثانية")

    return {"memory_gb": peak_memory_gb, "tokens_per_sec": tokens_per_second}


if __name__ == "__main__":
    prompt = "الذكاء الاصطناعي في المستقبل سوف"

    print("=== تحميل النموذج الأصلي (FP16) ===")
    fp16_model, tokenizer = load_fp16_model()
    fp16_stats = measure_memory_and_speed(fp16_model, tokenizer, prompt, "FP16 (الأصلي)")
    del fp16_model
    torch.cuda.empty_cache()

    print("\n=== تكميم النموذج بـ GPTQ (4-bit) ===")
    gptq_model, tokenizer = quantize_with_gptq(MODEL_NAME, bits=4)
    gptq_stats = measure_memory_and_speed(gptq_model, tokenizer, prompt, "GPTQ INT4")

    print(f"\n✅ تقليل الذاكرة: {fp16_stats['memory_gb'] / gptq_stats['memory_gb']:.2f}x")
    print(f"✅ تغيّر السرعة: {gptq_stats['tokens_per_sec'] / fp16_stats['tokens_per_sec']:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-11-deployment-llmops/chapter-59.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    """
    تطبيق تعليمي كامل لـ Speculative Decoding يوضح الآلية الرياضية خطوة بخطوة
    (في الإنتاج: استخدم دعم vLLM/TGI المدمج، لكن فهم الآلية الداخلية ضروري للمقابلات)
    """

    def __init__(self, draft_model_name: str, target_model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _get_next_token_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """يُرجع توزيع الاحتمالات الكامل (softmax) للتوكن التالي فقط"""
        logits = model(input_ids).logits[:, -1, :]
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, K: int = 4) -> dict:
        """
        الخوارزمية الكاملة: K توكن مقترحة من draft model، تحقق موازٍ من target model
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_tokens = 0
        accepted_total, proposed_total = 0, 0

        while generated_tokens < max_new_tokens:
            # ─── المرحلة 1: النموذج الصغير يقترح K توكن بتوليد تسلسلي سريع ───
            draft_input = input_ids.clone()
            draft_probs_list = []
            draft_tokens = []

            for _ in range(K):
                probs = self._get_next_token_probs(self.draft_model, draft_input)
                next_token = torch.multinomial(probs, num_samples=1)
                draft_probs_list.append(probs.gather(-1, next_token))
                draft_tokens.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)  # [1, K]
            candidate_sequence = torch.cat([input_ids, draft_tokens_tensor], dim=1)

            # ─── المرحلة 2: النموذج الكبير يتحقق من كل K توكن بنداء واحد موازٍ ───
            target_logits = self.target_model(candidate_sequence).logits
            # نحتاج احتمالات target عند كل موضع اقترح فيه draft توكنًا
            target_probs_at_positions = F.softmax(
                target_logits[:, input_ids.shape[1] - 1: -1, :], dim=-1
            )  # [1, K, vocab_size]

            # ─── المرحلة 3: قرار القبول/الرفض لكل توكن بالترتيب ───
            n_accepted = 0
            for i in range(K):
                token_id = draft_tokens[i]
                p_target = target_probs_at_positions[:, i, :].gather(-1, token_id).item()
                q_draft = draft_probs_list[i].item()

                acceptance_ratio = min(1.0, p_target / (q_draft + 1e-10))
                r = torch.rand(1).item()

                proposed_total += 1
                if r <= acceptance_ratio:
                    n_accepted += 1
                    accepted_total += 1
                else:
                    break  # نرفض هذا التوكن وكل ما بعده (اعتمدوا على توكن مرفوض)

            # إضافة التوكنات المقبولة فعليًا للتسلسل النهائي
            if n_accepted > 0:
                input_ids = torch.cat([input_ids, draft_tokens_tensor[:, :n_accepted]], dim=1)
                generated_tokens += n_accepted

            # إذا رُفض توكن، نولّد توكنًا تصحيحيًا من توزيع معدَّل (يضمن صحة التوزيع النهائي رياضيًا)
            if n_accepted < K:
                p_target_rejected = target_probs_at_positions[:, n_accepted, :]
                q_draft_rejected = self._get_next_token_probs(
                    self.draft_model, torch.cat([input_ids], dim=1)
                )
                corrected_dist = torch.clamp(p_target_rejected - q_draft_rejected, min=0)
                corrected_dist = corrected_dist / (corrected_dist.sum() + 1e-10)
                correction_token = torch.multinomial(corrected_dist, num_samples=1)
                input_ids = torch.cat([input_ids, correction_token], dim=1)
                generated_tokens += 1

        acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "text": generated_text,
            "acceptance_rate": acceptance_rate,
            "tokens_generated": generated_tokens,
        }


# ───────────── تشغيل تجريبي: نموذج صغير كـ draft، نموذج أكبر كـ target ─────────────
if __name__ == "__main__":
    decoder = SpeculativeDecoder(
        draft_model_name="facebook/opt-125m",   # نموذج صغير سريع جدًا
        target_model_name="facebook/opt-1.3b",  # نموذج أكبر وأدق
    )

    result = decoder.generate("الذكاء الاصطناعي سيُغيّر", max_new_tokens=40, K=4)
    print(f"النص المُولَّد: {result['text']}")
    print(f"نسبة قبول اقتراحات النموذج الصغير: {result['acceptance_rate']*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-11-deployment-llmops/chapter-59.md
# ────────────────────────────────────────────────────────────────────────
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class InferenceRequest:
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None


class StaticBatchingSimulator:
    """المحاكي التقليدي: ينتظر الكل حتى ينتهي أطول طلب في الدفعة قبل قبول دفعة جديدة"""

    def __init__(self, batch_size: int, time_per_token: float = 0.02):
        self.batch_size = batch_size
        self.time_per_token = time_per_token

    def run(self, requests: List[InferenceRequest]) -> float:
        total_time = 0.0
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            max_tokens_in_batch = max(r.max_new_tokens for r in batch)
            # الجميع ينتظر حتى ينتهي أطول طلب، حتى لو انتهى هو بعد توكن واحد فقط!
            batch_time = max_tokens_in_batch * self.time_per_token
            total_time += batch_time
            for r in batch:
                r.finish_time = total_time
        return total_time


class ContinuousBatchingSimulator:
    """
    محاكاة مبسّطة لـ Continuous Batching: في كل "خطوة" (step) نولّد توكنًا واحدًا
    لكل الطلبات النشطة، وفور انتهاء طلب، يحل محله فورًا طلب جديد من قائمة الانتظار
    """

    def __init__(self, max_batch_size: int, time_per_step: float = 0.02):
        self.max_batch_size = max_batch_size
        self.time_per_step = time_per_step

    def run(self, requests: List[InferenceRequest]) -> float:
        queue = sorted(requests, key=lambda r: r.arrival_time)
        active: List[InferenceRequest] = []
        current_time = 0.0

        while queue or active:
            # املأ الدفعة النشطة بطلبات جديدة من قائمة الانتظار حتى السقف الأقصى
            while len(active) < self.max_batch_size and queue:
                new_request = queue.pop(0)
                new_request.status = RequestStatus.RUNNING
                active.append(new_request)

            if not active:
                break

            # خطوة واحدة: كل الطلبات النشطة تتقدم بتوكن واحد في نفس "النداء" الموازي
            current_time += self.time_per_step
            still_active = []
            for r in active:
                r.generated_tokens += 1
                if r.generated_tokens >= r.max_new_tokens:
                    r.status = RequestStatus.FINISHED
                    r.finish_time = current_time
                    # ✨ فور انتهاء هذا الطلب، تتحرر "فتحة" يملؤها طلب جديد فورًا في الخطوة القادمة
                else:
                    still_active.append(r)
            active = still_active

        return current_time


def compare_batching_strategies():
    """مقارنة كمية: طلبات متنوعة الطول (بعضها قصير جدًا وبعضها طويل) - الحالة الواقعية الشائعة"""
    random.seed(42)
    requests_static = [
        InferenceRequest(request_id=i, prompt_tokens=50, max_new_tokens=random.choice([10, 20, 100, 150]))
        for i in range(20)
    ]
    requests_continuous = [
        InferenceRequest(request_id=r.request_id, prompt_tokens=r.prompt_tokens, max_new_tokens=r.max_new_tokens)
        for r in requests_static
    ]  # نفس الطلبات بالضبط لمقارنة عادلة

    static_sim = StaticBatchingSimulator(batch_size=4, time_per_token=0.02)
    continuous_sim = ContinuousBatchingSimulator(max_batch_size=4, time_per_step=0.02)

    static_total_time = static_sim.run(requests_static)
    continuous_total_time = continuous_sim.run(requests_continuous)

    print(f"⏱️  Static Batching:     {static_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"⏱️  Continuous Batching: {continuous_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"🚀 التسريع: {static_total_time / continuous_total_time:.2f}x")

    avg_wait_static = sum(r.finish_time for r in requests_static) / len(requests_static)
    avg_wait_continuous = sum(r.finish_time for r in requests_continuous) / len(requests_continuous)
    print(f"\n📊 متوسط زمن الانتظار (Static):     {avg_wait_static:.2f} ثانية")
    print(f"📊 متوسط زمن الانتظار (Continuous): {avg_wait_continuous:.2f} ثانية")


if __name__ == "__main__":
    compare_batching_strategies()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 59: الفصل 59: تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_inference() -> None:
    weights, scales, quantized = ([0.12, -0.77, 0.34, 1.0], 127, [])
    for value in weights:
        quantized.append(round(value * scales))
    restored = [value / scales for value in quantized]
    acceptance = 0.7
    draft_tokens = 8
    expected_accepted = draft_tokens * acceptance
    print('FP weights:', weights)
    print('INT8 reconstruction:', [round(x, 3) for x in restored])
    print('estimated accepted draft tokens:', round(expected_accepted, 2), 'of', draft_tokens)

if __name__ == "__main__":
    print("Chapter 59 lab: Compare quantization and speculative decoding costs")
    run_inference()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer

MODEL_NAME = "facebook/opt-1.3b"  # نموذج صغير لأغراض العرض؛ المبدأ ينطبق على 70B بنفس الطريقة

def load_fp16_model():
    """تحميل النموذج بالدقة الكاملة (baseline) لقياس الذاكرة والسرعة الأصلية"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    return model, tokenizer


def quantize_with_gptq(model_name: str, bits: int = 4):
    """تكميم GPTQ حقيقي باستخدام مكتبة optimum من Hugging Face"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPTQ يحتاج بيانات معايرة (calibration dataset) صغيرة لحساب Hessian التقريبية
    calibration_dataset = [
        "الذكاء الاصطناعي يُغيّر طريقة عملنا في كل القطاعات.",
        "تحسين الاستدلال أمر حيوي لتشغيل النماذج الكبيرة بكفاءة.",
        "الحوسبة السحابية توفر مرونة كبيرة للشركات الناشئة.",
    ] * 20  # يُستحسن عمليًا 128-512 عينة متنوعة من بيانات حقيقية مشابهة لتوزيع الاستخدام الفعلي

    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        group_size=128,          # حجم المجموعة لمشاركة نفس scale/zero_point (دقة أعلى مع تجميع أصغر)
        desc_act=False,          # ترتيب الأعمدة حسب الأهمية (دقة أعلى لكن أبطأ قليلًا في الاستدلال)
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model, tokenizer


def measure_memory_and_speed(model, tokenizer, prompt: str, label: str, n_runs: int = 5):
    """قياس موحّد: ذاكرة GPU المستخدمة + زمن توليد 50 توكن"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # تشغيل تمهيدي (warmup) لا يُحتسب في القياس
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens_per_second = 50 / elapsed

    print(f"\n📊 {label}")
    print(f"  ذروة ذاكرة GPU: {peak_memory_gb:.2f} GB")
    print(f"  زمن توليد 50 توكن: {elapsed:.3f} ثانية")
    print(f"  السرعة: {tokens_per_second:.1f} توكن/ثانية")

    return {"memory_gb": peak_memory_gb, "tokens_per_sec": tokens_per_second}


if __name__ == "__main__":
    prompt = "الذكاء الاصطناعي في المستقبل سوف"

    print("=== تحميل النموذج الأصلي (FP16) ===")
    fp16_model, tokenizer = load_fp16_model()
    fp16_stats = measure_memory_and_speed(fp16_model, tokenizer, prompt, "FP16 (الأصلي)")
    del fp16_model
    torch.cuda.empty_cache()

    print("\n=== تكميم النموذج بـ GPTQ (4-bit) ===")
    gptq_model, tokenizer = quantize_with_gptq(MODEL_NAME, bits=4)
    gptq_stats = measure_memory_and_speed(gptq_model, tokenizer, prompt, "GPTQ INT4")

    print(f"\n✅ تقليل الذاكرة: {fp16_stats['memory_gb'] / gptq_stats['memory_gb']:.2f}x")
    print(f"✅ تغيّر السرعة: {gptq_stats['tokens_per_sec'] / fp16_stats['tokens_per_sec']:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    """
    تطبيق تعليمي كامل لـ Speculative Decoding يوضح الآلية الرياضية خطوة بخطوة
    (في الإنتاج: استخدم دعم vLLM/TGI المدمج، لكن فهم الآلية الداخلية ضروري للمقابلات)
    """

    def __init__(self, draft_model_name: str, target_model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _get_next_token_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """يُرجع توزيع الاحتمالات الكامل (softmax) للتوكن التالي فقط"""
        logits = model(input_ids).logits[:, -1, :]
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, K: int = 4) -> dict:
        """
        الخوارزمية الكاملة: K توكن مقترحة من draft model، تحقق موازٍ من target model
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_tokens = 0
        accepted_total, proposed_total = 0, 0

        while generated_tokens < max_new_tokens:
            # ─── المرحلة 1: النموذج الصغير يقترح K توكن بتوليد تسلسلي سريع ───
            draft_input = input_ids.clone()
            draft_probs_list = []
            draft_tokens = []

            for _ in range(K):
                probs = self._get_next_token_probs(self.draft_model, draft_input)
                next_token = torch.multinomial(probs, num_samples=1)
                draft_probs_list.append(probs.gather(-1, next_token))
                draft_tokens.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)  # [1, K]
            candidate_sequence = torch.cat([input_ids, draft_tokens_tensor], dim=1)

            # ─── المرحلة 2: النموذج الكبير يتحقق من كل K توكن بنداء واحد موازٍ ───
            target_logits = self.target_model(candidate_sequence).logits
            # نحتاج احتمالات target عند كل موضع اقترح فيه draft توكنًا
            target_probs_at_positions = F.softmax(
                target_logits[:, input_ids.shape[1] - 1: -1, :], dim=-1
            )  # [1, K, vocab_size]

            # ─── المرحلة 3: قرار القبول/الرفض لكل توكن بالترتيب ───
            n_accepted = 0
            for i in range(K):
                token_id = draft_tokens[i]
                p_target = target_probs_at_positions[:, i, :].gather(-1, token_id).item()
                q_draft = draft_probs_list[i].item()

                acceptance_ratio = min(1.0, p_target / (q_draft + 1e-10))
                r = torch.rand(1).item()

                proposed_total += 1
                if r <= acceptance_ratio:
                    n_accepted += 1
                    accepted_total += 1
                else:
                    break  # نرفض هذا التوكن وكل ما بعده (اعتمدوا على توكن مرفوض)

            # إضافة التوكنات المقبولة فعليًا للتسلسل النهائي
            if n_accepted > 0:
                input_ids = torch.cat([input_ids, draft_tokens_tensor[:, :n_accepted]], dim=1)
                generated_tokens += n_accepted

            # إذا رُفض توكن، نولّد توكنًا تصحيحيًا من توزيع معدَّل (يضمن صحة التوزيع النهائي رياضيًا)
            if n_accepted < K:
                p_target_rejected = target_probs_at_positions[:, n_accepted, :]
                q_draft_rejected = self._get_next_token_probs(
                    self.draft_model, torch.cat([input_ids], dim=1)
                )
                corrected_dist = torch.clamp(p_target_rejected - q_draft_rejected, min=0)
                corrected_dist = corrected_dist / (corrected_dist.sum() + 1e-10)
                correction_token = torch.multinomial(corrected_dist, num_samples=1)
                input_ids = torch.cat([input_ids, correction_token], dim=1)
                generated_tokens += 1

        acceptance_rate = accepted_total / proposed_total if proposed_total > 0 else 0
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        return {
            "text": generated_text,
            "acceptance_rate": acceptance_rate,
            "tokens_generated": generated_tokens,
        }


# ───────────── تشغيل تجريبي: نموذج صغير كـ draft، نموذج أكبر كـ target ─────────────
if __name__ == "__main__":
    decoder = SpeculativeDecoder(
        draft_model_name="facebook/opt-125m",   # نموذج صغير سريع جدًا
        target_model_name="facebook/opt-1.3b",  # نموذج أكبر وأدق
    )

    result = decoder.generate("الذكاء الاصطناعي سيُغيّر", max_new_tokens=40, K=4)
    print(f"النص المُولَّد: {result['text']}")
    print(f"نسبة قبول اقتراحات النموذج الصغير: {result['acceptance_rate']*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-59-chapter-59.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class InferenceRequest:
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None


class StaticBatchingSimulator:
    """المحاكي التقليدي: ينتظر الكل حتى ينتهي أطول طلب في الدفعة قبل قبول دفعة جديدة"""

    def __init__(self, batch_size: int, time_per_token: float = 0.02):
        self.batch_size = batch_size
        self.time_per_token = time_per_token

    def run(self, requests: List[InferenceRequest]) -> float:
        total_time = 0.0
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            max_tokens_in_batch = max(r.max_new_tokens for r in batch)
            # الجميع ينتظر حتى ينتهي أطول طلب، حتى لو انتهى هو بعد توكن واحد فقط!
            batch_time = max_tokens_in_batch * self.time_per_token
            total_time += batch_time
            for r in batch:
                r.finish_time = total_time
        return total_time


class ContinuousBatchingSimulator:
    """
    محاكاة مبسّطة لـ Continuous Batching: في كل "خطوة" (step) نولّد توكنًا واحدًا
    لكل الطلبات النشطة، وفور انتهاء طلب، يحل محله فورًا طلب جديد من قائمة الانتظار
    """

    def __init__(self, max_batch_size: int, time_per_step: float = 0.02):
        self.max_batch_size = max_batch_size
        self.time_per_step = time_per_step

    def run(self, requests: List[InferenceRequest]) -> float:
        queue = sorted(requests, key=lambda r: r.arrival_time)
        active: List[InferenceRequest] = []
        current_time = 0.0

        while queue or active:
            # املأ الدفعة النشطة بطلبات جديدة من قائمة الانتظار حتى السقف الأقصى
            while len(active) < self.max_batch_size and queue:
                new_request = queue.pop(0)
                new_request.status = RequestStatus.RUNNING
                active.append(new_request)

            if not active:
                break

            # خطوة واحدة: كل الطلبات النشطة تتقدم بتوكن واحد في نفس "النداء" الموازي
            current_time += self.time_per_step
            still_active = []
            for r in active:
                r.generated_tokens += 1
                if r.generated_tokens >= r.max_new_tokens:
                    r.status = RequestStatus.FINISHED
                    r.finish_time = current_time
                    # ✨ فور انتهاء هذا الطلب، تتحرر "فتحة" يملؤها طلب جديد فورًا في الخطوة القادمة
                else:
                    still_active.append(r)
            active = still_active

        return current_time


def compare_batching_strategies():
    """مقارنة كمية: طلبات متنوعة الطول (بعضها قصير جدًا وبعضها طويل) - الحالة الواقعية الشائعة"""
    random.seed(42)
    requests_static = [
        InferenceRequest(request_id=i, prompt_tokens=50, max_new_tokens=random.choice([10, 20, 100, 150]))
        for i in range(20)
    ]
    requests_continuous = [
        InferenceRequest(request_id=r.request_id, prompt_tokens=r.prompt_tokens, max_new_tokens=r.max_new_tokens)
        for r in requests_static
    ]  # نفس الطلبات بالضبط لمقارنة عادلة

    static_sim = StaticBatchingSimulator(batch_size=4, time_per_token=0.02)
    continuous_sim = ContinuousBatchingSimulator(max_batch_size=4, time_per_step=0.02)

    static_total_time = static_sim.run(requests_static)
    continuous_total_time = continuous_sim.run(requests_continuous)

    print(f"⏱️  Static Batching:     {static_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"⏱️  Continuous Batching: {continuous_total_time:.2f} ثانية لمعالجة 20 طلبًا")
    print(f"🚀 التسريع: {static_total_time / continuous_total_time:.2f}x")

    avg_wait_static = sum(r.finish_time for r in requests_static) / len(requests_static)
    avg_wait_continuous = sum(r.finish_time for r in requests_continuous) / len(requests_continuous)
    print(f"\n📊 متوسط زمن الانتظار (Static):     {avg_wait_static:.2f} ثانية")
    print(f"📊 متوسط زمن الانتظار (Continuous): {avg_wait_continuous:.2f} ثانية")


if __name__ == "__main__":
    compare_batching_strategies()
