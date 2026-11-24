#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — دمج النماذج (Model Merging)
Large Language Model Engineering — Model

Book slug : llm-engineering-69-chapters
Chapter   : ch-28-chapter-28 (lesson 28)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-28-chapter-28.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — دمج النماذج (Model Merging)
Large Language Model Engineering — Model

Book slug : llm-engineering-69-chapters
Chapter   : ch-28-chapter-28 (lesson 28)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-28-chapter-28.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — دمج النماذج (Model Merging)
Large Language Model Engineering — Model

Book slug : llm-engineering-69-chapters
Chapter   : ch-28-chapter-28 (lesson 28)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 28: الفصل 28: دمج النماذج (Model Merging)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_merging() -> None:
    base, task_a, task_b = ([0.2, -0.1, 0.5], [0.4, -0.1, 0.5], [0.2, 0.1, 0.7])
    delta_a = [a - b for a, b in zip(task_a, base)]
    delta_b = [a - b for a, b in zip(task_b, base)]
    merged = [b + 0.5 * (a + c) for b, a, c in zip(base, delta_a, delta_b)]
    print('task vectors:', delta_a, delta_b)
    print('merged weights:', [round(x, 3) for x in merged])
    print('all shapes compatible:', len(base) == len(task_a) == len(task_b))

if __name__ == "__main__":
    print("Chapter 28 lab: Merge compatible model weight vectors")
    run_merging()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
import torch
from typing import List, Dict

def linear_average_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """
    دمج بسيط بالمتوسط الموزون لمجموعة من النماذج

    🟡 الأبسط والأسرع، لكن الأقل تحكمًا في حل التعارضات
    بين النماذج (كما سنرى لاحقًا مع TIES)
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    assert abs(sum(weights) - 1.0) < 1e-6, "يجب أن يكون مجموع الأوزان = 1"

    merged_state_dict = {}

    # التحقق من تطابق المفاتيح (المعمارية) بين كل النماذج
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        assert set(sd.keys()) == keys, (
            "❌ النماذج لا تملك نفس البنية - لا يمكن الدمج الخطي!"
        )

    for key in keys:
        # تجميع موزون لنفس الطبقة عبر كل النماذج
        merged_tensor = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )
        # إعادة التحويل لنفس نوع البيانات الأصلي
        merged_state_dict[key] = merged_tensor.to(state_dicts[0][key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def compute_task_vector(
    base_state_dict: Dict[str, torch.Tensor],
    finetuned_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    حساب متجه المهمة: الفرق بين النموذج المضبوط والنموذج الأساسي

    🟡 هذا هو "جوهر ما تعلمه النموذج" من الضبط الدقيق
    """
    task_vector = {}
    for key in base_state_dict:
        task_vector[key] = (
            finetuned_state_dict[key].float() - base_state_dict[key].float()
        )
    return task_vector


def apply_task_arithmetic(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    scaling_coefficients: List[float],
) -> Dict[str, torch.Tensor]:
    """
    تطبيق Task Arithmetic: دمج عدة متجهات مهام على النموذج الأساسي

    scaling_coefficients موجبة → إضافة القدرة (Task Addition)
    scaling_coefficients سالبة → حذف السلوك (Task Negation)

    💡 نصيحة: ابدأ بـ λ=1.0 لكل متجه، وقلّله إذا رأيت تدهورًا في الأداء
    """
    merged_state_dict = {}

    for key in base_state_dict:
        # ابدأ من النموذج الأساسي
        merged_value = base_state_dict[key].float().clone()

        # أضف كل متجه مهمة مع معامل التحجيم الخاص به
        for tv, lam in zip(task_vectors, scaling_coefficients):
            merged_value += lam * tv[key]

        merged_state_dict[key] = merged_value.to(base_state_dict[key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def ties_merge(
    task_vectors: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,  # احتفظ بأعلى 20% من القيم فقط
    scaling_coefficient: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    تطبيق TIES-Merging الكامل: Trim → Elect Sign → Disjoint Merge

    🟡 هذا يحل مشكلة "التشويش المتبادل" بين متجهات المهام المختلفة
    """
    merged_task_vector = {}
    keys = task_vectors[0].keys()

    for key in keys:
        # جمع نفس الطبقة من كل متجهات المهام في Tensor واحد
        # الشكل: (num_tasks, *param_shape)
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)

        # ========== الخطوة 1: TRIM ==========
        # نحسب العتبة (threshold) لكل متجه مهمة على حدة
        trimmed = stacked.clone()
        for i in range(stacked.shape[0]):
            flat_abs = stacked[i].abs().flatten()
            k = max(1, int(len(flat_abs) * trim_ratio))
            # إيجاد قيمة العتبة: أصغر قيمة بين أعلى k قيمة
            threshold = torch.topk(flat_abs, k).values.min()
            # تصفير كل القيم الأصغر من العتبة
            mask = stacked[i].abs() >= threshold
            trimmed[i] = stacked[i] * mask

        # ========== الخطوة 2: ELECT SIGN ==========
        # مجموع الإشارات عبر كل متجهات المهام (بعد التقليم)
        sign_sum = torch.sign(trimmed).sum(dim=0)
        # الإشارة الفائزة: موجبة إن كان المجموع > 0، سالبة إن كان < 0
        elected_sign = torch.sign(sign_sum)

        # ========== الخطوة 3: DISJOINT MERGE ==========
        # لكل متجه، احتفظ فقط بالقيم المتفقة مع الإشارة الفائزة
        agreement_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0))
        agreed_values = trimmed * agreement_mask

        # المتوسط فقط على القيم "غير الصفرية والمتفقة"
        # (تجنب تمييع القيم بقسمتها على عدد كل المهام دائمًا)
        count_nonzero = agreement_mask.sum(dim=0).clamp(min=1)
        merged_value = agreed_values.sum(dim=0) / count_nonzero

        merged_task_vector[key] = scaling_coefficient * merged_value

    return merged_task_vector

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify(
    task_vector: Dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    تطبيق DARE: حذف عشوائي + إعادة تحجيم

    🟡 مفاجأة بحثية: يمكن حذف حتى 90-99% من قيم متجه المهمة
    عشوائيًا والاحتفاظ بمعظم الأداء! هذا يدل على أن متجهات
    المهام "كثيفة الضوضاء" أكثر مما نعتقد

    عادة تُستخدم DARE قبل TIES أو Task Arithmetic كخطوة تمهيدية
    لتقليل التداخل (Interference) بين متجهات متعددة
    """
    torch.manual_seed(seed)
    sparsified = {}

    for key, tensor in task_vector.items():
        # قناع Bernoulli: 1 يعني "احتفظ"، 0 يعني "احذف"
        mask = torch.bernoulli(
            torch.full_like(tensor, 1 - drop_rate)
        )
        # الحذف + إعادة التحجيم للحفاظ على التوقع الرياضي للقيمة
        sparsified[key] = (tensor * mask) / (1 - drop_rate)

    return sparsified

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    t: float = 0.5,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    دمج كروي بين تنسورين (عادة يُطبّق على كل طبقة على حدة)

    🟡 يُستخدم غالبًا لدمج نموذجين فقط (وليس عدة نماذج)
    ويُفضّل عندما تكون النماذج "متشابهة جدًا" في الاتجاه
    """
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    # تطبيع المتجهات لحساب الزاوية بينها فقط
    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # حساب الزاوية بين المتجهين عبر جيب التمام (Cosine Similarity)
    dot_product = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متطابقة)
    # نستخدم Linear Interpolation لتجنب القسمة على صفر
    if sin_omega.abs() < epsilon:
        result = (1 - t) * a_flat + t * b_flat
    else:
        coef_a = torch.sin((1 - t) * omega) / sin_omega
        coef_b = torch.sin(t * omega) / sin_omega
        result = coef_a * a_flat + coef_b * b_flat

    return result.reshape(tensor_a.shape).to(tensor_a.dtype)


def slerp_merge_models(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    t: float = 0.5
) -> Dict[str, torch.Tensor]:
    """تطبيق SLERP على كل طبقة في النموذج بشكل منفصل"""
    merged = {}
    for key in state_dict_a:
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_28_model_merging.py
=============================
مشروع متكامل: دمج نموذجين مُضبوطين باستخدام 5 تقنيات مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ModelMerger:
    """
    🧰 أداة شاملة لدمج النماذج بكل التقنيات التي تعلمناها
    مصمّمة لتكون قابلة لإعادة الاستخدام عبر أي نموذجين بنفس المعمارية
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"📥 تحميل النموذج الأساسي: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_state_dict = {
            k: v.clone() for k, v in base_model.state_dict().items()
        }
        del base_model  # تحرير الذاكرة

    def load_finetuned(self, path_or_name: str) -> Dict[str, torch.Tensor]:
        """تحميل state_dict لنموذج مُضبوط"""
        model = AutoModelForCausalLM.from_pretrained(path_or_name)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        return state_dict

    def compute_task_vector(
        self, finetuned_sd: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """حساب متجه المهمة بالنسبة للنموذج الأساسي"""
        return {
            k: finetuned_sd[k].float() - self.base_state_dict[k].float()
            for k in self.base_state_dict
        }

    def build_model_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ):
        """بناء نموذج PyTorch كامل من state_dict مدموج"""
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        # التأكد من مطابقة أنواع البيانات قبل التحميل
        casted_sd = {
            k: v.to(model.state_dict()[k].dtype)
            for k, v in state_dict.items()
        }
        model.load_state_dict(casted_sd)
        return model.to(device)

    # ============================================================
    # طرق الدمج الخمسة - واجهة موحّدة لكل منها
    # ============================================================

    def merge_linear_average(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 1: متوسط خطي بسيط"""
        if weights is None:
            weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = sum(
                w * fm[key].float()
                for w, fm in zip(weights, finetuned_models)
            )
        return merged

    def merge_task_arithmetic(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        coefficients: List[float]
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 2: Task Arithmetic"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        merged = {}
        for key in self.base_state_dict:
            value = self.base_state_dict[key].float().clone()
            for tv, lam in zip(task_vectors, coefficients):
                value += lam * tv[key]
            merged[key] = value
        return merged

    def merge_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 3: TIES-Merging"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]
        merged_tv = ties_merge(task_vectors, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_dare_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        drop_rate: float = 0.9,
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 4: DARE + TIES معًا (شائع جدًا في الممارسة العملية)"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        # خطوة DARE أولًا: تخفيف كل متجه مهمة
        sparsified = [
            dare_sparsify(tv, drop_rate=drop_rate, seed=42 + i)
            for i, tv in enumerate(task_vectors)
        ]

        # ثم TIES على النتيجة المخففة
        merged_tv = ties_merge(sparsified, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_slerp(
        self,
        model_a: Dict[str, torch.Tensor],
        model_b: Dict[str, torch.Tensor],
        t: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 5: SLERP (لنموذجين فقط)"""
        return slerp_merge_models(model_a, model_b, t=t)


def evaluate_merged_model(model, tokenizer, test_prompts: List[Dict]):
    """
    🧪 دالة تقييم موحّدة: تختبر النموذج المدموج على
    مطالبات من كلا النطاقين (طبي + عربي أدبي) لقياس
    مدى نجاح الدمج في الحفاظ على القدرتين معًا
    """
    model.eval()
    results = []

    for item in test_prompts:
        prompt = item["prompt"]
        domain = item["domain"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"domain": domain, "prompt": prompt, "output": generated})

    return results


def main():
    """تشغيل التجربة الكاملة: مقارنة كل طرق الدمج"""

    BASE_MODEL = "openai-community/gpt2"

    merger = ModelMerger(BASE_MODEL)

    # 📥 تحميل النموذجين المضبوطين (من الفصل 27 مثلًا، أو أي نماذج مماثلة)
    print("\n📥 تحميل النموذج الطبي المُضبوط...")
    medical_sd = merger.load_finetuned("./medical_arabic_gpt2")  # من الفصل 27

    print("📥 تحميل نموذج افتراضي ثانٍ (أدبي عربي)...")
    # في مشروع حقيقي: نموذج مُضبوط فعليًا على نصوص أدبية
    # هنا نفترض وجوده بنفس الطريقة للتوضيح
    literary_sd = merger.load_finetuned("./literary_arabic_gpt2")

    test_prompts = [
        {"prompt": "العلاج الأمثل لارتفاع ضغط الدم", "domain": "طبي"},
        {"prompt": "في ليالي الشتاء الطويلة، كان الشاعر", "domain": "أدبي"},
    ]

    print("\n" + "=" * 70)
    print("🔬 مقارنة طرق الدمج المختلفة")
    print("=" * 70)

    # --- التجربة 1: Linear Averaging ---
    print("\n1️⃣ Linear Averaging (50/50)...")
    merged_linear = merger.merge_linear_average(
        [medical_sd, literary_sd], weights=[0.5, 0.5]
    )
    model_linear = merger.build_model_from_state_dict(merged_linear)
    results_linear = evaluate_merged_model(model_linear, merger.tokenizer, test_prompts)
    for r in results_linear:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 2: Task Arithmetic ---
    print("\n2️⃣ Task Arithmetic (λ=0.6 لكل مهمة)...")
    merged_ta = merger.merge_task_arithmetic(
        [medical_sd, literary_sd], coefficients=[0.6, 0.6]
    )
    model_ta = merger.build_model_from_state_dict(merged_ta)
    results_ta = evaluate_merged_model(model_ta, merger.tokenizer, test_prompts)
    for r in results_ta:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 3: TIES-Merging ---
    print("\n3️⃣ TIES-Merging (trim_ratio=0.2)...")
    merged_ties = merger.merge_ties(
        [medical_sd, literary_sd], trim_ratio=0.2, scaling=1.0
    )
    model_ties = merger.build_model_from_state_dict(merged_ties)
    results_ties = evaluate_merged_model(model_ties, merger.tokenizer, test_prompts)
    for r in results_ties:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 4: DARE + TIES ---
    print("\n4️⃣ DARE + TIES (drop_rate=0.9)...")
    merged_dare = merger.merge_dare_ties(
        [medical_sd, literary_sd], drop_rate=0.9, trim_ratio=0.2
    )
    model_dare = merger.build_model_from_state_dict(merged_dare)
    results_dare = evaluate_merged_model(model_dare, merger.tokenizer, test_prompts)
    for r in results_dare:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 5: SLERP ---
    print("\n5️⃣ SLERP (t=0.5)...")
    merged_slerp = merger.merge_slerp(medical_sd, literary_sd, t=0.5)
    model_slerp = merger.build_model_from_state_dict(merged_slerp)
    results_slerp = evaluate_merged_model(model_slerp, merger.tokenizer, test_prompts)
    for r in results_slerp:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    print("\n" + "=" * 70)
    print("✅ اكتملت المقارنة! راجع المخرجات أعلاه لتقييم أي طريقة")
    print("   حافظت على القدرتين (طبي + أدبي) بشكل أفضل")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
# أو برمجيًا عبر Python API
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

config = MergeConfiguration.model_validate({
    "models": [
        {"model": "meta-llama/Llama-2-7b-hf"},
        {
            "model": "./medical-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
        {
            "model": "./arabic-literary-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
    ],
    "merge_method": "ties",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "parameters": {"normalize": True},
    "dtype": "bfloat16"
})

run_merge(
    config,
    out_path="./merged_output_model",
    options=MergeOptions(cuda=True, low_cpu_memory=True)
)
print("✅ تم الدمج! النموذج جاهز في ./merged_output_model")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 28: الفصل 28: دمج النماذج (Model Merging)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_merging() -> None:
    base, task_a, task_b = ([0.2, -0.1, 0.5], [0.4, -0.1, 0.5], [0.2, 0.1, 0.7])
    delta_a = [a - b for a, b in zip(task_a, base)]
    delta_b = [a - b for a, b in zip(task_b, base)]
    merged = [b + 0.5 * (a + c) for b, a, c in zip(base, delta_a, delta_b)]
    print('task vectors:', delta_a, delta_b)
    print('merged weights:', [round(x, 3) for x in merged])
    print('all shapes compatible:', len(base) == len(task_a) == len(task_b))

if __name__ == "__main__":
    print("Chapter 28 lab: Merge compatible model weight vectors")
    run_merging()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
from typing import List, Dict

def linear_average_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """
    دمج بسيط بالمتوسط الموزون لمجموعة من النماذج

    🟡 الأبسط والأسرع، لكن الأقل تحكمًا في حل التعارضات
    بين النماذج (كما سنرى لاحقًا مع TIES)
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    assert abs(sum(weights) - 1.0) < 1e-6, "يجب أن يكون مجموع الأوزان = 1"

    merged_state_dict = {}

    # التحقق من تطابق المفاتيح (المعمارية) بين كل النماذج
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        assert set(sd.keys()) == keys, (
            "❌ النماذج لا تملك نفس البنية - لا يمكن الدمج الخطي!"
        )

    for key in keys:
        # تجميع موزون لنفس الطبقة عبر كل النماذج
        merged_tensor = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )
        # إعادة التحويل لنفس نوع البيانات الأصلي
        merged_state_dict[key] = merged_tensor.to(state_dicts[0][key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def compute_task_vector(
    base_state_dict: Dict[str, torch.Tensor],
    finetuned_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    حساب متجه المهمة: الفرق بين النموذج المضبوط والنموذج الأساسي

    🟡 هذا هو "جوهر ما تعلمه النموذج" من الضبط الدقيق
    """
    task_vector = {}
    for key in base_state_dict:
        task_vector[key] = (
            finetuned_state_dict[key].float() - base_state_dict[key].float()
        )
    return task_vector


def apply_task_arithmetic(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    scaling_coefficients: List[float],
) -> Dict[str, torch.Tensor]:
    """
    تطبيق Task Arithmetic: دمج عدة متجهات مهام على النموذج الأساسي

    scaling_coefficients موجبة → إضافة القدرة (Task Addition)
    scaling_coefficients سالبة → حذف السلوك (Task Negation)

    💡 نصيحة: ابدأ بـ λ=1.0 لكل متجه، وقلّله إذا رأيت تدهورًا في الأداء
    """
    merged_state_dict = {}

    for key in base_state_dict:
        # ابدأ من النموذج الأساسي
        merged_value = base_state_dict[key].float().clone()

        # أضف كل متجه مهمة مع معامل التحجيم الخاص به
        for tv, lam in zip(task_vectors, scaling_coefficients):
            merged_value += lam * tv[key]

        merged_state_dict[key] = merged_value.to(base_state_dict[key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def ties_merge(
    task_vectors: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,  # احتفظ بأعلى 20% من القيم فقط
    scaling_coefficient: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    تطبيق TIES-Merging الكامل: Trim → Elect Sign → Disjoint Merge

    🟡 هذا يحل مشكلة "التشويش المتبادل" بين متجهات المهام المختلفة
    """
    merged_task_vector = {}
    keys = task_vectors[0].keys()

    for key in keys:
        # جمع نفس الطبقة من كل متجهات المهام في Tensor واحد
        # الشكل: (num_tasks, *param_shape)
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)

        # ========== الخطوة 1: TRIM ==========
        # نحسب العتبة (threshold) لكل متجه مهمة على حدة
        trimmed = stacked.clone()
        for i in range(stacked.shape[0]):
            flat_abs = stacked[i].abs().flatten()
            k = max(1, int(len(flat_abs) * trim_ratio))
            # إيجاد قيمة العتبة: أصغر قيمة بين أعلى k قيمة
            threshold = torch.topk(flat_abs, k).values.min()
            # تصفير كل القيم الأصغر من العتبة
            mask = stacked[i].abs() >= threshold
            trimmed[i] = stacked[i] * mask

        # ========== الخطوة 2: ELECT SIGN ==========
        # مجموع الإشارات عبر كل متجهات المهام (بعد التقليم)
        sign_sum = torch.sign(trimmed).sum(dim=0)
        # الإشارة الفائزة: موجبة إن كان المجموع > 0، سالبة إن كان < 0
        elected_sign = torch.sign(sign_sum)

        # ========== الخطوة 3: DISJOINT MERGE ==========
        # لكل متجه، احتفظ فقط بالقيم المتفقة مع الإشارة الفائزة
        agreement_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0))
        agreed_values = trimmed * agreement_mask

        # المتوسط فقط على القيم "غير الصفرية والمتفقة"
        # (تجنب تمييع القيم بقسمتها على عدد كل المهام دائمًا)
        count_nonzero = agreement_mask.sum(dim=0).clamp(min=1)
        merged_value = agreed_values.sum(dim=0) / count_nonzero

        merged_task_vector[key] = scaling_coefficient * merged_value

    return merged_task_vector

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify(
    task_vector: Dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    تطبيق DARE: حذف عشوائي + إعادة تحجيم

    🟡 مفاجأة بحثية: يمكن حذف حتى 90-99% من قيم متجه المهمة
    عشوائيًا والاحتفاظ بمعظم الأداء! هذا يدل على أن متجهات
    المهام "كثيفة الضوضاء" أكثر مما نعتقد

    عادة تُستخدم DARE قبل TIES أو Task Arithmetic كخطوة تمهيدية
    لتقليل التداخل (Interference) بين متجهات متعددة
    """
    torch.manual_seed(seed)
    sparsified = {}

    for key, tensor in task_vector.items():
        # قناع Bernoulli: 1 يعني "احتفظ"، 0 يعني "احذف"
        mask = torch.bernoulli(
            torch.full_like(tensor, 1 - drop_rate)
        )
        # الحذف + إعادة التحجيم للحفاظ على التوقع الرياضي للقيمة
        sparsified[key] = (tensor * mask) / (1 - drop_rate)

    return sparsified

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    t: float = 0.5,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    دمج كروي بين تنسورين (عادة يُطبّق على كل طبقة على حدة)

    🟡 يُستخدم غالبًا لدمج نموذجين فقط (وليس عدة نماذج)
    ويُفضّل عندما تكون النماذج "متشابهة جدًا" في الاتجاه
    """
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    # تطبيع المتجهات لحساب الزاوية بينها فقط
    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # حساب الزاوية بين المتجهين عبر جيب التمام (Cosine Similarity)
    dot_product = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متطابقة)
    # نستخدم Linear Interpolation لتجنب القسمة على صفر
    if sin_omega.abs() < epsilon:
        result = (1 - t) * a_flat + t * b_flat
    else:
        coef_a = torch.sin((1 - t) * omega) / sin_omega
        coef_b = torch.sin(t * omega) / sin_omega
        result = coef_a * a_flat + coef_b * b_flat

    return result.reshape(tensor_a.shape).to(tensor_a.dtype)


def slerp_merge_models(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    t: float = 0.5
) -> Dict[str, torch.Tensor]:
    """تطبيق SLERP على كل طبقة في النموذج بشكل منفصل"""
    merged = {}
    for key in state_dict_a:
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
chapter_28_model_merging.py
=============================
مشروع متكامل: دمج نموذجين مُضبوطين باستخدام 5 تقنيات مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ModelMerger:
    """
    🧰 أداة شاملة لدمج النماذج بكل التقنيات التي تعلمناها
    مصمّمة لتكون قابلة لإعادة الاستخدام عبر أي نموذجين بنفس المعمارية
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"📥 تحميل النموذج الأساسي: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_state_dict = {
            k: v.clone() for k, v in base_model.state_dict().items()
        }
        del base_model  # تحرير الذاكرة

    def load_finetuned(self, path_or_name: str) -> Dict[str, torch.Tensor]:
        """تحميل state_dict لنموذج مُضبوط"""
        model = AutoModelForCausalLM.from_pretrained(path_or_name)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        return state_dict

    def compute_task_vector(
        self, finetuned_sd: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """حساب متجه المهمة بالنسبة للنموذج الأساسي"""
        return {
            k: finetuned_sd[k].float() - self.base_state_dict[k].float()
            for k in self.base_state_dict
        }

    def build_model_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ):
        """بناء نموذج PyTorch كامل من state_dict مدموج"""
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        # التأكد من مطابقة أنواع البيانات قبل التحميل
        casted_sd = {
            k: v.to(model.state_dict()[k].dtype)
            for k, v in state_dict.items()
        }
        model.load_state_dict(casted_sd)
        return model.to(device)

    # ============================================================
    # طرق الدمج الخمسة - واجهة موحّدة لكل منها
    # ============================================================

    def merge_linear_average(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 1: متوسط خطي بسيط"""
        if weights is None:
            weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = sum(
                w * fm[key].float()
                for w, fm in zip(weights, finetuned_models)
            )
        return merged

    def merge_task_arithmetic(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        coefficients: List[float]
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 2: Task Arithmetic"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        merged = {}
        for key in self.base_state_dict:
            value = self.base_state_dict[key].float().clone()
            for tv, lam in zip(task_vectors, coefficients):
                value += lam * tv[key]
            merged[key] = value
        return merged

    def merge_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 3: TIES-Merging"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]
        merged_tv = ties_merge(task_vectors, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_dare_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        drop_rate: float = 0.9,
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 4: DARE + TIES معًا (شائع جدًا في الممارسة العملية)"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        # خطوة DARE أولًا: تخفيف كل متجه مهمة
        sparsified = [
            dare_sparsify(tv, drop_rate=drop_rate, seed=42 + i)
            for i, tv in enumerate(task_vectors)
        ]

        # ثم TIES على النتيجة المخففة
        merged_tv = ties_merge(sparsified, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_slerp(
        self,
        model_a: Dict[str, torch.Tensor],
        model_b: Dict[str, torch.Tensor],
        t: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 5: SLERP (لنموذجين فقط)"""
        return slerp_merge_models(model_a, model_b, t=t)


def evaluate_merged_model(model, tokenizer, test_prompts: List[Dict]):
    """
    🧪 دالة تقييم موحّدة: تختبر النموذج المدموج على
    مطالبات من كلا النطاقين (طبي + عربي أدبي) لقياس
    مدى نجاح الدمج في الحفاظ على القدرتين معًا
    """
    model.eval()
    results = []

    for item in test_prompts:
        prompt = item["prompt"]
        domain = item["domain"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"domain": domain, "prompt": prompt, "output": generated})

    return results


def main():
    """تشغيل التجربة الكاملة: مقارنة كل طرق الدمج"""

    BASE_MODEL = "openai-community/gpt2"

    merger = ModelMerger(BASE_MODEL)

    # 📥 تحميل النموذجين المضبوطين (من الفصل 27 مثلًا، أو أي نماذج مماثلة)
    print("\n📥 تحميل النموذج الطبي المُضبوط...")
    medical_sd = merger.load_finetuned("./medical_arabic_gpt2")  # من الفصل 27

    print("📥 تحميل نموذج افتراضي ثانٍ (أدبي عربي)...")
    # في مشروع حقيقي: نموذج مُضبوط فعليًا على نصوص أدبية
    # هنا نفترض وجوده بنفس الطريقة للتوضيح
    literary_sd = merger.load_finetuned("./literary_arabic_gpt2")

    test_prompts = [
        {"prompt": "العلاج الأمثل لارتفاع ضغط الدم", "domain": "طبي"},
        {"prompt": "في ليالي الشتاء الطويلة، كان الشاعر", "domain": "أدبي"},
    ]

    print("\n" + "=" * 70)
    print("🔬 مقارنة طرق الدمج المختلفة")
    print("=" * 70)

    # --- التجربة 1: Linear Averaging ---
    print("\n1️⃣ Linear Averaging (50/50)...")
    merged_linear = merger.merge_linear_average(
        [medical_sd, literary_sd], weights=[0.5, 0.5]
    )
    model_linear = merger.build_model_from_state_dict(merged_linear)
    results_linear = evaluate_merged_model(model_linear, merger.tokenizer, test_prompts)
    for r in results_linear:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 2: Task Arithmetic ---
    print("\n2️⃣ Task Arithmetic (λ=0.6 لكل مهمة)...")
    merged_ta = merger.merge_task_arithmetic(
        [medical_sd, literary_sd], coefficients=[0.6, 0.6]
    )
    model_ta = merger.build_model_from_state_dict(merged_ta)
    results_ta = evaluate_merged_model(model_ta, merger.tokenizer, test_prompts)
    for r in results_ta:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 3: TIES-Merging ---
    print("\n3️⃣ TIES-Merging (trim_ratio=0.2)...")
    merged_ties = merger.merge_ties(
        [medical_sd, literary_sd], trim_ratio=0.2, scaling=1.0
    )
    model_ties = merger.build_model_from_state_dict(merged_ties)
    results_ties = evaluate_merged_model(model_ties, merger.tokenizer, test_prompts)
    for r in results_ties:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 4: DARE + TIES ---
    print("\n4️⃣ DARE + TIES (drop_rate=0.9)...")
    merged_dare = merger.merge_dare_ties(
        [medical_sd, literary_sd], drop_rate=0.9, trim_ratio=0.2
    )
    model_dare = merger.build_model_from_state_dict(merged_dare)
    results_dare = evaluate_merged_model(model_dare, merger.tokenizer, test_prompts)
    for r in results_dare:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 5: SLERP ---
    print("\n5️⃣ SLERP (t=0.5)...")
    merged_slerp = merger.merge_slerp(medical_sd, literary_sd, t=0.5)
    model_slerp = merger.build_model_from_state_dict(merged_slerp)
    results_slerp = evaluate_merged_model(model_slerp, merger.tokenizer, test_prompts)
    for r in results_slerp:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    print("\n" + "=" * 70)
    print("✅ اكتملت المقارنة! راجع المخرجات أعلاه لتقييم أي طريقة")
    print("   حافظت على القدرتين (طبي + أدبي) بشكل أفضل")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# أو برمجيًا عبر Python API
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

config = MergeConfiguration.model_validate({
    "models": [
        {"model": "meta-llama/Llama-2-7b-hf"},
        {
            "model": "./medical-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
        {
            "model": "./arabic-literary-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
    ],
    "merge_method": "ties",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "parameters": {"normalize": True},
    "dtype": "bfloat16"
})

run_merge(
    config,
    out_path="./merged_output_model",
    options=MergeOptions(cuda=True, low_cpu_memory=True)
)
print("✅ تم الدمج! النموذج جاهز في ./merged_output_model")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — دمج النماذج (Model Merging)
Large Language Model Engineering — Model

Book slug : llm-engineering-69-chapters
Chapter   : ch-28-chapter-28 (lesson 28)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_28.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 28: الفصل 28: دمج النماذج (Model Merging)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_merging() -> None:
    base, task_a, task_b = ([0.2, -0.1, 0.5], [0.4, -0.1, 0.5], [0.2, 0.1, 0.7])
    delta_a = [a - b for a, b in zip(task_a, base)]
    delta_b = [a - b for a, b in zip(task_b, base)]
    merged = [b + 0.5 * (a + c) for b, a, c in zip(base, delta_a, delta_b)]
    print('task vectors:', delta_a, delta_b)
    print('merged weights:', [round(x, 3) for x in merged])
    print('all shapes compatible:', len(base) == len(task_a) == len(task_b))

if __name__ == "__main__":
    print("Chapter 28 lab: Merge compatible model weight vectors")
    run_merging()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
import torch
from typing import List, Dict

def linear_average_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """
    دمج بسيط بالمتوسط الموزون لمجموعة من النماذج

    🟡 الأبسط والأسرع، لكن الأقل تحكمًا في حل التعارضات
    بين النماذج (كما سنرى لاحقًا مع TIES)
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    assert abs(sum(weights) - 1.0) < 1e-6, "يجب أن يكون مجموع الأوزان = 1"

    merged_state_dict = {}

    # التحقق من تطابق المفاتيح (المعمارية) بين كل النماذج
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        assert set(sd.keys()) == keys, (
            "❌ النماذج لا تملك نفس البنية - لا يمكن الدمج الخطي!"
        )

    for key in keys:
        # تجميع موزون لنفس الطبقة عبر كل النماذج
        merged_tensor = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )
        # إعادة التحويل لنفس نوع البيانات الأصلي
        merged_state_dict[key] = merged_tensor.to(state_dicts[0][key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def compute_task_vector(
    base_state_dict: Dict[str, torch.Tensor],
    finetuned_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    حساب متجه المهمة: الفرق بين النموذج المضبوط والنموذج الأساسي

    🟡 هذا هو "جوهر ما تعلمه النموذج" من الضبط الدقيق
    """
    task_vector = {}
    for key in base_state_dict:
        task_vector[key] = (
            finetuned_state_dict[key].float() - base_state_dict[key].float()
        )
    return task_vector


def apply_task_arithmetic(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    scaling_coefficients: List[float],
) -> Dict[str, torch.Tensor]:
    """
    تطبيق Task Arithmetic: دمج عدة متجهات مهام على النموذج الأساسي

    scaling_coefficients موجبة → إضافة القدرة (Task Addition)
    scaling_coefficients سالبة → حذف السلوك (Task Negation)

    💡 نصيحة: ابدأ بـ λ=1.0 لكل متجه، وقلّله إذا رأيت تدهورًا في الأداء
    """
    merged_state_dict = {}

    for key in base_state_dict:
        # ابدأ من النموذج الأساسي
        merged_value = base_state_dict[key].float().clone()

        # أضف كل متجه مهمة مع معامل التحجيم الخاص به
        for tv, lam in zip(task_vectors, scaling_coefficients):
            merged_value += lam * tv[key]

        merged_state_dict[key] = merged_value.to(base_state_dict[key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def ties_merge(
    task_vectors: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,  # احتفظ بأعلى 20% من القيم فقط
    scaling_coefficient: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    تطبيق TIES-Merging الكامل: Trim → Elect Sign → Disjoint Merge

    🟡 هذا يحل مشكلة "التشويش المتبادل" بين متجهات المهام المختلفة
    """
    merged_task_vector = {}
    keys = task_vectors[0].keys()

    for key in keys:
        # جمع نفس الطبقة من كل متجهات المهام في Tensor واحد
        # الشكل: (num_tasks, *param_shape)
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)

        # ========== الخطوة 1: TRIM ==========
        # نحسب العتبة (threshold) لكل متجه مهمة على حدة
        trimmed = stacked.clone()
        for i in range(stacked.shape[0]):
            flat_abs = stacked[i].abs().flatten()
            k = max(1, int(len(flat_abs) * trim_ratio))
            # إيجاد قيمة العتبة: أصغر قيمة بين أعلى k قيمة
            threshold = torch.topk(flat_abs, k).values.min()
            # تصفير كل القيم الأصغر من العتبة
            mask = stacked[i].abs() >= threshold
            trimmed[i] = stacked[i] * mask

        # ========== الخطوة 2: ELECT SIGN ==========
        # مجموع الإشارات عبر كل متجهات المهام (بعد التقليم)
        sign_sum = torch.sign(trimmed).sum(dim=0)
        # الإشارة الفائزة: موجبة إن كان المجموع > 0، سالبة إن كان < 0
        elected_sign = torch.sign(sign_sum)

        # ========== الخطوة 3: DISJOINT MERGE ==========
        # لكل متجه، احتفظ فقط بالقيم المتفقة مع الإشارة الفائزة
        agreement_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0))
        agreed_values = trimmed * agreement_mask

        # المتوسط فقط على القيم "غير الصفرية والمتفقة"
        # (تجنب تمييع القيم بقسمتها على عدد كل المهام دائمًا)
        count_nonzero = agreement_mask.sum(dim=0).clamp(min=1)
        merged_value = agreed_values.sum(dim=0) / count_nonzero

        merged_task_vector[key] = scaling_coefficient * merged_value

    return merged_task_vector

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify(
    task_vector: Dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    تطبيق DARE: حذف عشوائي + إعادة تحجيم

    🟡 مفاجأة بحثية: يمكن حذف حتى 90-99% من قيم متجه المهمة
    عشوائيًا والاحتفاظ بمعظم الأداء! هذا يدل على أن متجهات
    المهام "كثيفة الضوضاء" أكثر مما نعتقد

    عادة تُستخدم DARE قبل TIES أو Task Arithmetic كخطوة تمهيدية
    لتقليل التداخل (Interference) بين متجهات متعددة
    """
    torch.manual_seed(seed)
    sparsified = {}

    for key, tensor in task_vector.items():
        # قناع Bernoulli: 1 يعني "احتفظ"، 0 يعني "احذف"
        mask = torch.bernoulli(
            torch.full_like(tensor, 1 - drop_rate)
        )
        # الحذف + إعادة التحجيم للحفاظ على التوقع الرياضي للقيمة
        sparsified[key] = (tensor * mask) / (1 - drop_rate)

    return sparsified

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    t: float = 0.5,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    دمج كروي بين تنسورين (عادة يُطبّق على كل طبقة على حدة)

    🟡 يُستخدم غالبًا لدمج نموذجين فقط (وليس عدة نماذج)
    ويُفضّل عندما تكون النماذج "متشابهة جدًا" في الاتجاه
    """
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    # تطبيع المتجهات لحساب الزاوية بينها فقط
    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # حساب الزاوية بين المتجهين عبر جيب التمام (Cosine Similarity)
    dot_product = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متطابقة)
    # نستخدم Linear Interpolation لتجنب القسمة على صفر
    if sin_omega.abs() < epsilon:
        result = (1 - t) * a_flat + t * b_flat
    else:
        coef_a = torch.sin((1 - t) * omega) / sin_omega
        coef_b = torch.sin(t * omega) / sin_omega
        result = coef_a * a_flat + coef_b * b_flat

    return result.reshape(tensor_a.shape).to(tensor_a.dtype)


def slerp_merge_models(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    t: float = 0.5
) -> Dict[str, torch.Tensor]:
    """تطبيق SLERP على كل طبقة في النموذج بشكل منفصل"""
    merged = {}
    for key in state_dict_a:
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_28_model_merging.py
=============================
مشروع متكامل: دمج نموذجين مُضبوطين باستخدام 5 تقنيات مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ModelMerger:
    """
    🧰 أداة شاملة لدمج النماذج بكل التقنيات التي تعلمناها
    مصمّمة لتكون قابلة لإعادة الاستخدام عبر أي نموذجين بنفس المعمارية
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"📥 تحميل النموذج الأساسي: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_state_dict = {
            k: v.clone() for k, v in base_model.state_dict().items()
        }
        del base_model  # تحرير الذاكرة

    def load_finetuned(self, path_or_name: str) -> Dict[str, torch.Tensor]:
        """تحميل state_dict لنموذج مُضبوط"""
        model = AutoModelForCausalLM.from_pretrained(path_or_name)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        return state_dict

    def compute_task_vector(
        self, finetuned_sd: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """حساب متجه المهمة بالنسبة للنموذج الأساسي"""
        return {
            k: finetuned_sd[k].float() - self.base_state_dict[k].float()
            for k in self.base_state_dict
        }

    def build_model_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ):
        """بناء نموذج PyTorch كامل من state_dict مدموج"""
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        # التأكد من مطابقة أنواع البيانات قبل التحميل
        casted_sd = {
            k: v.to(model.state_dict()[k].dtype)
            for k, v in state_dict.items()
        }
        model.load_state_dict(casted_sd)
        return model.to(device)

    # ============================================================
    # طرق الدمج الخمسة - واجهة موحّدة لكل منها
    # ============================================================

    def merge_linear_average(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 1: متوسط خطي بسيط"""
        if weights is None:
            weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = sum(
                w * fm[key].float()
                for w, fm in zip(weights, finetuned_models)
            )
        return merged

    def merge_task_arithmetic(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        coefficients: List[float]
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 2: Task Arithmetic"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        merged = {}
        for key in self.base_state_dict:
            value = self.base_state_dict[key].float().clone()
            for tv, lam in zip(task_vectors, coefficients):
                value += lam * tv[key]
            merged[key] = value
        return merged

    def merge_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 3: TIES-Merging"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]
        merged_tv = ties_merge(task_vectors, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_dare_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        drop_rate: float = 0.9,
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 4: DARE + TIES معًا (شائع جدًا في الممارسة العملية)"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        # خطوة DARE أولًا: تخفيف كل متجه مهمة
        sparsified = [
            dare_sparsify(tv, drop_rate=drop_rate, seed=42 + i)
            for i, tv in enumerate(task_vectors)
        ]

        # ثم TIES على النتيجة المخففة
        merged_tv = ties_merge(sparsified, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_slerp(
        self,
        model_a: Dict[str, torch.Tensor],
        model_b: Dict[str, torch.Tensor],
        t: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 5: SLERP (لنموذجين فقط)"""
        return slerp_merge_models(model_a, model_b, t=t)


def evaluate_merged_model(model, tokenizer, test_prompts: List[Dict]):
    """
    🧪 دالة تقييم موحّدة: تختبر النموذج المدموج على
    مطالبات من كلا النطاقين (طبي + عربي أدبي) لقياس
    مدى نجاح الدمج في الحفاظ على القدرتين معًا
    """
    model.eval()
    results = []

    for item in test_prompts:
        prompt = item["prompt"]
        domain = item["domain"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"domain": domain, "prompt": prompt, "output": generated})

    return results


def main():
    """تشغيل التجربة الكاملة: مقارنة كل طرق الدمج"""

    BASE_MODEL = "openai-community/gpt2"

    merger = ModelMerger(BASE_MODEL)

    # 📥 تحميل النموذجين المضبوطين (من الفصل 27 مثلًا، أو أي نماذج مماثلة)
    print("\n📥 تحميل النموذج الطبي المُضبوط...")
    medical_sd = merger.load_finetuned("./medical_arabic_gpt2")  # من الفصل 27

    print("📥 تحميل نموذج افتراضي ثانٍ (أدبي عربي)...")
    # في مشروع حقيقي: نموذج مُضبوط فعليًا على نصوص أدبية
    # هنا نفترض وجوده بنفس الطريقة للتوضيح
    literary_sd = merger.load_finetuned("./literary_arabic_gpt2")

    test_prompts = [
        {"prompt": "العلاج الأمثل لارتفاع ضغط الدم", "domain": "طبي"},
        {"prompt": "في ليالي الشتاء الطويلة، كان الشاعر", "domain": "أدبي"},
    ]

    print("\n" + "=" * 70)
    print("🔬 مقارنة طرق الدمج المختلفة")
    print("=" * 70)

    # --- التجربة 1: Linear Averaging ---
    print("\n1️⃣ Linear Averaging (50/50)...")
    merged_linear = merger.merge_linear_average(
        [medical_sd, literary_sd], weights=[0.5, 0.5]
    )
    model_linear = merger.build_model_from_state_dict(merged_linear)
    results_linear = evaluate_merged_model(model_linear, merger.tokenizer, test_prompts)
    for r in results_linear:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 2: Task Arithmetic ---
    print("\n2️⃣ Task Arithmetic (λ=0.6 لكل مهمة)...")
    merged_ta = merger.merge_task_arithmetic(
        [medical_sd, literary_sd], coefficients=[0.6, 0.6]
    )
    model_ta = merger.build_model_from_state_dict(merged_ta)
    results_ta = evaluate_merged_model(model_ta, merger.tokenizer, test_prompts)
    for r in results_ta:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 3: TIES-Merging ---
    print("\n3️⃣ TIES-Merging (trim_ratio=0.2)...")
    merged_ties = merger.merge_ties(
        [medical_sd, literary_sd], trim_ratio=0.2, scaling=1.0
    )
    model_ties = merger.build_model_from_state_dict(merged_ties)
    results_ties = evaluate_merged_model(model_ties, merger.tokenizer, test_prompts)
    for r in results_ties:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 4: DARE + TIES ---
    print("\n4️⃣ DARE + TIES (drop_rate=0.9)...")
    merged_dare = merger.merge_dare_ties(
        [medical_sd, literary_sd], drop_rate=0.9, trim_ratio=0.2
    )
    model_dare = merger.build_model_from_state_dict(merged_dare)
    results_dare = evaluate_merged_model(model_dare, merger.tokenizer, test_prompts)
    for r in results_dare:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 5: SLERP ---
    print("\n5️⃣ SLERP (t=0.5)...")
    merged_slerp = merger.merge_slerp(medical_sd, literary_sd, t=0.5)
    model_slerp = merger.build_model_from_state_dict(merged_slerp)
    results_slerp = evaluate_merged_model(model_slerp, merger.tokenizer, test_prompts)
    for r in results_slerp:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    print("\n" + "=" * 70)
    print("✅ اكتملت المقارنة! راجع المخرجات أعلاه لتقييم أي طريقة")
    print("   حافظت على القدرتين (طبي + أدبي) بشكل أفضل")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-06-model-reuse-adaptation/chapter-28.md
# ────────────────────────────────────────────────────────────────────────
# أو برمجيًا عبر Python API
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

config = MergeConfiguration.model_validate({
    "models": [
        {"model": "meta-llama/Llama-2-7b-hf"},
        {
            "model": "./medical-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
        {
            "model": "./arabic-literary-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
    ],
    "merge_method": "ties",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "parameters": {"normalize": True},
    "dtype": "bfloat16"
})

run_merge(
    config,
    out_path="./merged_output_model",
    options=MergeOptions(cuda=True, low_cpu_memory=True)
)
print("✅ تم الدمج! النموذج جاهز في ./merged_output_model")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 28: الفصل 28: دمج النماذج (Model Merging)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_merging() -> None:
    base, task_a, task_b = ([0.2, -0.1, 0.5], [0.4, -0.1, 0.5], [0.2, 0.1, 0.7])
    delta_a = [a - b for a, b in zip(task_a, base)]
    delta_b = [a - b for a, b in zip(task_b, base)]
    merged = [b + 0.5 * (a + c) for b, a, c in zip(base, delta_a, delta_b)]
    print('task vectors:', delta_a, delta_b)
    print('merged weights:', [round(x, 3) for x in merged])
    print('all shapes compatible:', len(base) == len(task_a) == len(task_b))

if __name__ == "__main__":
    print("Chapter 28 lab: Merge compatible model weight vectors")
    run_merging()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
from typing import List, Dict

def linear_average_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """
    دمج بسيط بالمتوسط الموزون لمجموعة من النماذج

    🟡 الأبسط والأسرع، لكن الأقل تحكمًا في حل التعارضات
    بين النماذج (كما سنرى لاحقًا مع TIES)
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    assert abs(sum(weights) - 1.0) < 1e-6, "يجب أن يكون مجموع الأوزان = 1"

    merged_state_dict = {}

    # التحقق من تطابق المفاتيح (المعمارية) بين كل النماذج
    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        assert set(sd.keys()) == keys, (
            "❌ النماذج لا تملك نفس البنية - لا يمكن الدمج الخطي!"
        )

    for key in keys:
        # تجميع موزون لنفس الطبقة عبر كل النماذج
        merged_tensor = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )
        # إعادة التحويل لنفس نوع البيانات الأصلي
        merged_state_dict[key] = merged_tensor.to(state_dicts[0][key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def compute_task_vector(
    base_state_dict: Dict[str, torch.Tensor],
    finetuned_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    حساب متجه المهمة: الفرق بين النموذج المضبوط والنموذج الأساسي

    🟡 هذا هو "جوهر ما تعلمه النموذج" من الضبط الدقيق
    """
    task_vector = {}
    for key in base_state_dict:
        task_vector[key] = (
            finetuned_state_dict[key].float() - base_state_dict[key].float()
        )
    return task_vector


def apply_task_arithmetic(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: List[Dict[str, torch.Tensor]],
    scaling_coefficients: List[float],
) -> Dict[str, torch.Tensor]:
    """
    تطبيق Task Arithmetic: دمج عدة متجهات مهام على النموذج الأساسي

    scaling_coefficients موجبة → إضافة القدرة (Task Addition)
    scaling_coefficients سالبة → حذف السلوك (Task Negation)

    💡 نصيحة: ابدأ بـ λ=1.0 لكل متجه، وقلّله إذا رأيت تدهورًا في الأداء
    """
    merged_state_dict = {}

    for key in base_state_dict:
        # ابدأ من النموذج الأساسي
        merged_value = base_state_dict[key].float().clone()

        # أضف كل متجه مهمة مع معامل التحجيم الخاص به
        for tv, lam in zip(task_vectors, scaling_coefficients):
            merged_value += lam * tv[key]

        merged_state_dict[key] = merged_value.to(base_state_dict[key].dtype)

    return merged_state_dict

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def ties_merge(
    task_vectors: List[Dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,  # احتفظ بأعلى 20% من القيم فقط
    scaling_coefficient: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    تطبيق TIES-Merging الكامل: Trim → Elect Sign → Disjoint Merge

    🟡 هذا يحل مشكلة "التشويش المتبادل" بين متجهات المهام المختلفة
    """
    merged_task_vector = {}
    keys = task_vectors[0].keys()

    for key in keys:
        # جمع نفس الطبقة من كل متجهات المهام في Tensor واحد
        # الشكل: (num_tasks, *param_shape)
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)

        # ========== الخطوة 1: TRIM ==========
        # نحسب العتبة (threshold) لكل متجه مهمة على حدة
        trimmed = stacked.clone()
        for i in range(stacked.shape[0]):
            flat_abs = stacked[i].abs().flatten()
            k = max(1, int(len(flat_abs) * trim_ratio))
            # إيجاد قيمة العتبة: أصغر قيمة بين أعلى k قيمة
            threshold = torch.topk(flat_abs, k).values.min()
            # تصفير كل القيم الأصغر من العتبة
            mask = stacked[i].abs() >= threshold
            trimmed[i] = stacked[i] * mask

        # ========== الخطوة 2: ELECT SIGN ==========
        # مجموع الإشارات عبر كل متجهات المهام (بعد التقليم)
        sign_sum = torch.sign(trimmed).sum(dim=0)
        # الإشارة الفائزة: موجبة إن كان المجموع > 0، سالبة إن كان < 0
        elected_sign = torch.sign(sign_sum)

        # ========== الخطوة 3: DISJOINT MERGE ==========
        # لكل متجه، احتفظ فقط بالقيم المتفقة مع الإشارة الفائزة
        agreement_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0))
        agreed_values = trimmed * agreement_mask

        # المتوسط فقط على القيم "غير الصفرية والمتفقة"
        # (تجنب تمييع القيم بقسمتها على عدد كل المهام دائمًا)
        count_nonzero = agreement_mask.sum(dim=0).clamp(min=1)
        merged_value = agreed_values.sum(dim=0) / count_nonzero

        merged_task_vector[key] = scaling_coefficient * merged_value

    return merged_task_vector

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify(
    task_vector: Dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    تطبيق DARE: حذف عشوائي + إعادة تحجيم

    🟡 مفاجأة بحثية: يمكن حذف حتى 90-99% من قيم متجه المهمة
    عشوائيًا والاحتفاظ بمعظم الأداء! هذا يدل على أن متجهات
    المهام "كثيفة الضوضاء" أكثر مما نعتقد

    عادة تُستخدم DARE قبل TIES أو Task Arithmetic كخطوة تمهيدية
    لتقليل التداخل (Interference) بين متجهات متعددة
    """
    torch.manual_seed(seed)
    sparsified = {}

    for key, tensor in task_vector.items():
        # قناع Bernoulli: 1 يعني "احتفظ"، 0 يعني "احذف"
        mask = torch.bernoulli(
            torch.full_like(tensor, 1 - drop_rate)
        )
        # الحذف + إعادة التحجيم للحفاظ على التوقع الرياضي للقيمة
        sparsified[key] = (tensor * mask) / (1 - drop_rate)

    return sparsified

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    t: float = 0.5,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    دمج كروي بين تنسورين (عادة يُطبّق على كل طبقة على حدة)

    🟡 يُستخدم غالبًا لدمج نموذجين فقط (وليس عدة نماذج)
    ويُفضّل عندما تكون النماذج "متشابهة جدًا" في الاتجاه
    """
    a_flat = tensor_a.flatten().float()
    b_flat = tensor_b.flatten().float()

    # تطبيع المتجهات لحساب الزاوية بينها فقط
    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # حساب الزاوية بين المتجهين عبر جيب التمام (Cosine Similarity)
    dot_product = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متطابقة)
    # نستخدم Linear Interpolation لتجنب القسمة على صفر
    if sin_omega.abs() < epsilon:
        result = (1 - t) * a_flat + t * b_flat
    else:
        coef_a = torch.sin((1 - t) * omega) / sin_omega
        coef_b = torch.sin(t * omega) / sin_omega
        result = coef_a * a_flat + coef_b * b_flat

    return result.reshape(tensor_a.shape).to(tensor_a.dtype)


def slerp_merge_models(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    t: float = 0.5
) -> Dict[str, torch.Tensor]:
    """تطبيق SLERP على كل طبقة في النموذج بشكل منفصل"""
    merged = {}
    for key in state_dict_a:
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
chapter_28_model_merging.py
=============================
مشروع متكامل: دمج نموذجين مُضبوطين باستخدام 5 تقنيات مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ModelMerger:
    """
    🧰 أداة شاملة لدمج النماذج بكل التقنيات التي تعلمناها
    مصمّمة لتكون قابلة لإعادة الاستخدام عبر أي نموذجين بنفس المعمارية
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"📥 تحميل النموذج الأساسي: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_state_dict = {
            k: v.clone() for k, v in base_model.state_dict().items()
        }
        del base_model  # تحرير الذاكرة

    def load_finetuned(self, path_or_name: str) -> Dict[str, torch.Tensor]:
        """تحميل state_dict لنموذج مُضبوط"""
        model = AutoModelForCausalLM.from_pretrained(path_or_name)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        return state_dict

    def compute_task_vector(
        self, finetuned_sd: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """حساب متجه المهمة بالنسبة للنموذج الأساسي"""
        return {
            k: finetuned_sd[k].float() - self.base_state_dict[k].float()
            for k in self.base_state_dict
        }

    def build_model_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ):
        """بناء نموذج PyTorch كامل من state_dict مدموج"""
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        # التأكد من مطابقة أنواع البيانات قبل التحميل
        casted_sd = {
            k: v.to(model.state_dict()[k].dtype)
            for k, v in state_dict.items()
        }
        model.load_state_dict(casted_sd)
        return model.to(device)

    # ============================================================
    # طرق الدمج الخمسة - واجهة موحّدة لكل منها
    # ============================================================

    def merge_linear_average(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 1: متوسط خطي بسيط"""
        if weights is None:
            weights = [1.0 / len(finetuned_models)] * len(finetuned_models)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = sum(
                w * fm[key].float()
                for w, fm in zip(weights, finetuned_models)
            )
        return merged

    def merge_task_arithmetic(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        coefficients: List[float]
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 2: Task Arithmetic"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        merged = {}
        for key in self.base_state_dict:
            value = self.base_state_dict[key].float().clone()
            for tv, lam in zip(task_vectors, coefficients):
                value += lam * tv[key]
            merged[key] = value
        return merged

    def merge_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 3: TIES-Merging"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]
        merged_tv = ties_merge(task_vectors, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_dare_ties(
        self,
        finetuned_models: List[Dict[str, torch.Tensor]],
        drop_rate: float = 0.9,
        trim_ratio: float = 0.2,
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 4: DARE + TIES معًا (شائع جدًا في الممارسة العملية)"""
        task_vectors = [self.compute_task_vector(fm) for fm in finetuned_models]

        # خطوة DARE أولًا: تخفيف كل متجه مهمة
        sparsified = [
            dare_sparsify(tv, drop_rate=drop_rate, seed=42 + i)
            for i, tv in enumerate(task_vectors)
        ]

        # ثم TIES على النتيجة المخففة
        merged_tv = ties_merge(sparsified, trim_ratio, scaling)

        merged = {}
        for key in self.base_state_dict:
            merged[key] = self.base_state_dict[key].float() + merged_tv[key]
        return merged

    def merge_slerp(
        self,
        model_a: Dict[str, torch.Tensor],
        model_b: Dict[str, torch.Tensor],
        t: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """الطريقة 5: SLERP (لنموذجين فقط)"""
        return slerp_merge_models(model_a, model_b, t=t)


def evaluate_merged_model(model, tokenizer, test_prompts: List[Dict]):
    """
    🧪 دالة تقييم موحّدة: تختبر النموذج المدموج على
    مطالبات من كلا النطاقين (طبي + عربي أدبي) لقياس
    مدى نجاح الدمج في الحفاظ على القدرتين معًا
    """
    model.eval()
    results = []

    for item in test_prompts:
        prompt = item["prompt"]
        domain = item["domain"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"domain": domain, "prompt": prompt, "output": generated})

    return results


def main():
    """تشغيل التجربة الكاملة: مقارنة كل طرق الدمج"""

    BASE_MODEL = "openai-community/gpt2"

    merger = ModelMerger(BASE_MODEL)

    # 📥 تحميل النموذجين المضبوطين (من الفصل 27 مثلًا، أو أي نماذج مماثلة)
    print("\n📥 تحميل النموذج الطبي المُضبوط...")
    medical_sd = merger.load_finetuned("./medical_arabic_gpt2")  # من الفصل 27

    print("📥 تحميل نموذج افتراضي ثانٍ (أدبي عربي)...")
    # في مشروع حقيقي: نموذج مُضبوط فعليًا على نصوص أدبية
    # هنا نفترض وجوده بنفس الطريقة للتوضيح
    literary_sd = merger.load_finetuned("./literary_arabic_gpt2")

    test_prompts = [
        {"prompt": "العلاج الأمثل لارتفاع ضغط الدم", "domain": "طبي"},
        {"prompt": "في ليالي الشتاء الطويلة، كان الشاعر", "domain": "أدبي"},
    ]

    print("\n" + "=" * 70)
    print("🔬 مقارنة طرق الدمج المختلفة")
    print("=" * 70)

    # --- التجربة 1: Linear Averaging ---
    print("\n1️⃣ Linear Averaging (50/50)...")
    merged_linear = merger.merge_linear_average(
        [medical_sd, literary_sd], weights=[0.5, 0.5]
    )
    model_linear = merger.build_model_from_state_dict(merged_linear)
    results_linear = evaluate_merged_model(model_linear, merger.tokenizer, test_prompts)
    for r in results_linear:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 2: Task Arithmetic ---
    print("\n2️⃣ Task Arithmetic (λ=0.6 لكل مهمة)...")
    merged_ta = merger.merge_task_arithmetic(
        [medical_sd, literary_sd], coefficients=[0.6, 0.6]
    )
    model_ta = merger.build_model_from_state_dict(merged_ta)
    results_ta = evaluate_merged_model(model_ta, merger.tokenizer, test_prompts)
    for r in results_ta:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 3: TIES-Merging ---
    print("\n3️⃣ TIES-Merging (trim_ratio=0.2)...")
    merged_ties = merger.merge_ties(
        [medical_sd, literary_sd], trim_ratio=0.2, scaling=1.0
    )
    model_ties = merger.build_model_from_state_dict(merged_ties)
    results_ties = evaluate_merged_model(model_ties, merger.tokenizer, test_prompts)
    for r in results_ties:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 4: DARE + TIES ---
    print("\n4️⃣ DARE + TIES (drop_rate=0.9)...")
    merged_dare = merger.merge_dare_ties(
        [medical_sd, literary_sd], drop_rate=0.9, trim_ratio=0.2
    )
    model_dare = merger.build_model_from_state_dict(merged_dare)
    results_dare = evaluate_merged_model(model_dare, merger.tokenizer, test_prompts)
    for r in results_dare:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    # --- التجربة 5: SLERP ---
    print("\n5️⃣ SLERP (t=0.5)...")
    merged_slerp = merger.merge_slerp(medical_sd, literary_sd, t=0.5)
    model_slerp = merger.build_model_from_state_dict(merged_slerp)
    results_slerp = evaluate_merged_model(model_slerp, merger.tokenizer, test_prompts)
    for r in results_slerp:
        print(f"   [{r['domain']}] {r['output'][:100]}...")

    print("\n" + "=" * 70)
    print("✅ اكتملت المقارنة! راجع المخرجات أعلاه لتقييم أي طريقة")
    print("   حافظت على القدرتين (طبي + أدبي) بشكل أفضل")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-28-chapter-28.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# أو برمجيًا عبر Python API
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions

config = MergeConfiguration.model_validate({
    "models": [
        {"model": "meta-llama/Llama-2-7b-hf"},
        {
            "model": "./medical-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
        {
            "model": "./arabic-literary-llama-7b",
            "parameters": {"weight": 0.4, "density": 0.5}
        },
    ],
    "merge_method": "ties",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "parameters": {"normalize": True},
    "dtype": "bfloat16"
})

run_merge(
    config,
    out_path="./merged_output_model",
    options=MergeOptions(cuda=True, low_cpu_memory=True)
)
print("✅ تم الدمج! النموذج جاهز في ./merged_output_model")
