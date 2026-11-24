#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقليم (Pruning)
Large Language Model Engineering — Pruning

Book slug : llm-engineering-69-chapters
Chapter   : ch-30-chapter-30 (lesson 30)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-30-chapter-30.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقليم (Pruning)
Large Language Model Engineering — Pruning

Book slug : llm-engineering-69-chapters
Chapter   : ch-30-chapter-30 (lesson 30)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-30-chapter-30.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقليم (Pruning)
Large Language Model Engineering — Pruning

Book slug : llm-engineering-69-chapters
Chapter   : ch-30-chapter-30 (lesson 30)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 30: الفصل 30: التقليم (Pruning)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_pruning() -> None:
    weights = [0.02, -0.8, 0.12, 0.0, 0.61, -0.03, 0.3, 0.01]
    threshold = sorted((abs(x) for x in weights))[3]
    pruned = [0.0 if abs(x) <= threshold else x for x in weights]
    zeros = sum((x == 0 for x in pruned))
    print('magnitude threshold:', round(threshold, 3))
    print('sparsity:', f'{zeros}/{len(weights)} ({zeros / len(weights):.0%})')
    print('dense storage unchanged without a sparse kernel:', len(weights), 'slots')

if __name__ == "__main__":
    print("Chapter 30 lab: Magnitude pruning and sparsity accounting")
    run_pruning()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def magnitude_prune_layer(
    layer: nn.Linear,
    sparsity: float = 0.5
) -> nn.Linear:
    """
    🟡 Magnitude Pruning البسيط باستخدام PyTorch المدمج

    sparsity: نسبة الأوزان المراد تصفيرها (0.5 = حذف 50%)
    """
    # PyTorch توفر وحدة prune جاهزة ومُحسّنة
    prune.l1_unstructured(layer, name="weight", amount=sparsity)

    # نسبة التصفير الفعلية (للتحقق)
    sparsity_actual = (
        torch.sum(layer.weight == 0).item() / layer.weight.nelement()
    )
    print(f"✅ نسبة الأصفار الفعلية: {sparsity_actual:.1%}")

    return layer


def permanent_prune(layer: nn.Linear):
    """
    💡 نصيحة مهمة: prune.l1_unstructured تُطبّق "قناعًا" (mask)
    فقط، لا تحذف الأوزان فعليًا من الذاكرة! يجب استدعاء
    remove() لجعل التقليم دائمًا وتحرير الذاكرة الإضافية
    """
    prune.remove(layer, "weight")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from typing import Dict

class WandaPruner:
    """
    🧰 تطبيق كامل لخوارزمية Wanda
    (Pruning by Weights AND Activations - Sun et al., 2023)

    🟡 الميزة الكبرى: لا يحتاج أي إعادة تدريب، فقط forward
    passes على بيانات معايرة صغيرة (Calibration Data) لحساب
    إحصائيات التفعيلات
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_norms: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _register_hooks(self):
        """
        تسجيل Hooks على كل طبقة Linear لرصد المدخلات
        أثناء الـ Forward Pass على بيانات المعايرة
        """
        def make_hook(name):
            def hook(module, input, output):
                # input[0] هي تفعيلات المدخل لهذه الطبقة
                # الشكل: (batch, seq_len, in_features)
                x = input[0].detach()
                x = x.reshape(-1, x.shape[-1])  # (batch*seq_len, in_features)

                # حساب L2 norm لكل عمود (ميزة إدخال) عبر كل العينات
                norm = torch.norm(x, p=2, dim=0)  # (in_features,)

                if name not in self.activation_norms:
                    self.activation_norms[name] = norm
                else:
                    # تراكم عبر عدة دفعات (batches) من بيانات المعايرة
                    self.activation_norms[name] = torch.sqrt(
                        self.activation_norms[name]**2 + norm**2
                    )
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def calibrate(self, calibration_loader, num_batches: int = 32):
        """
        تشغيل Forward Passes على بيانات معايرة لجمع إحصائيات
        التفعيلات اللازمة لحساب معيار Wanda

        💡 بيانات المعايرة لا تحتاج أن تكون كبيرة! عادة
        128-512 عينة كافية تمامًا (هذا جزء من سحر الكفاءة هنا)
        """
        self.model.eval()
        self._register_hooks()

        print(f"🔄 جمع إحصائيات التفعيلات من {num_batches} دفعة معايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"]
            self.model(input_ids)

        self._remove_hooks()
        print(f"✅ تم جمع إحصائيات لـ {len(self.activation_norms)} طبقة")

    def prune(self, sparsity: float = 0.5):
        """
        تطبيق التقليم الفعلي بناءً على معيار Wanda:
        S_ij = |W_ij| * ||X_j||_2

        نستخدم هنا تقليمًا "لكل صف" (per-output) وهو ما أوصت
        به الورقة الأصلية لتحقيق توازن أفضل بين الطبقات
        """
        print(f"\n✂️  تطبيق Wanda Pruning بنسبة {sparsity:.0%}...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activation_norms:
                W = module.weight.data  # (out_features, in_features)
                act_norm = self.activation_norms[name]  # (in_features,)

                # حساب معيار الأهمية: |W| * ||X||
                importance = W.abs() * act_norm.unsqueeze(0)

                # تقليم لكل صف (output neuron) على حدة - per-output comparison
                # هذا يحافظ على توازن التقليم عبر كل الخلايا العصبية
                num_prune = int(W.shape[1] * sparsity)

                for row in range(W.shape[0]):
                    row_importance = importance[row]
                    # إيجاد أقل القيم أهمية في هذا الصف
                    _, indices_to_prune = torch.topk(
                        row_importance, num_prune, largest=False
                    )
                    W[row, indices_to_prune] = 0.0

        print("✅ اكتمل التقليم!")
        self._report_sparsity()

    def _report_sparsity(self):
        """تقرير نسبة الأصفار الفعلية في كل طبقة"""
        total_params, total_zeros = 0, 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                zeros = (module.weight.data == 0).sum().item()
                total = module.weight.numel()
                total_params += total
                total_zeros += zeros

        print(f"📊 نسبة التقليم الإجمالية: {total_zeros/total_params:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
# استخدام SparseGPT عمليًا عبر المكتبة الرسمية
"""
pip install sparsegpt  # (أو الاستنساخ من المستودع الرسمي)
"""

# مثال توضيحي مبسّط لواجهة الاستخدام
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_sparsegpt_conceptual(model_name: str, sparsity: float = 0.5):
    """
    🧪 توضيح مفاهيمي لخطوات SparseGPT (الاستخدام الفعلي
    يتطلب مكتبة متخصصة بسبب التعقيد الحسابي لـ Hessian)
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # الخطوات المفاهيمية (مُبسّطة):
    # 1. تمرير بيانات معايرة عبر النموذج طبقة بطبقة
    # 2. لكل طبقة Linear: حساب H = X @ X.T (تقريب Hessian)
    # 3. حساب ترتيب الأهمية باستخدام معكوس H
    # 4. تقليم الأوزان الأقل أهمية + تحديث الباقي لتعويض الخطأ
    # 5. الانتقال للطبقة التالية (Sequential - مهم للدقة)

    print("💡 في التطبيق الحقيقي، استخدم مكتبة SparseGPT الرسمية:")
    print("   github.com/IST-DASLab/sparsegpt")

    return model

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
class StructuredPruner:
    """
    🧰 تطبيق Structured Pruning: حذف خلايا عصبية كاملة
    بناءً على أهميتها الإجمالية (L2 Norm لكل خلية)

    🟡 الفرق الجوهري عن Unstructured: هنا نُصغّر فعليًا
    أبعاد المصفوفة (out_features)، مما يُسرّع الحساب حقًا
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_neuron_importance(
        self, layer: nn.Linear
    ) -> torch.Tensor:
        """
        حساب أهمية كل خلية عصبية (كل صف في مصفوفة الوزن)
        باستخدام L2 Norm كمعيار بسيط وفعّال
        """
        # لكل صف (خلية إخراج)، نحسب قوة أوزانه الكلية
        importance = torch.norm(layer.weight.data, p=2, dim=1)
        return importance

    def prune_linear_layer(
        self,
        layer: nn.Linear,
        prune_ratio: float = 0.3
    ) -> nn.Linear:
        """
        🔪 حذف فعلي لخلايا عصبية كاملة - يُنشئ طبقة جديدة أصغر!
        (وليس مجرد تصفير - هذا هو جوهر Structured Pruning)
        """
        importance = self.compute_neuron_importance(layer)
        num_keep = int(layer.out_features * (1 - prune_ratio))

        # اختيار أهم الخلايا للاحتفاظ بها
        _, keep_indices = torch.topk(importance, num_keep)
        keep_indices, _ = torch.sort(keep_indices)  # الحفاظ على الترتيب الأصلي

        # بناء طبقة جديدة **فعليًا أصغر** (ليس فقط معلّمة بالأصفار!)
        new_layer = nn.Linear(
            layer.in_features, num_keep,
            bias=(layer.bias is not None)
        )
        new_layer.weight.data = layer.weight.data[keep_indices, :].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices].clone()

        print(
            f"   ✂️  طبقة {layer.out_features} → {num_keep} خلية "
            f"({prune_ratio:.0%} تقليم)"
        )

        return new_layer, keep_indices

    def prune_attention_heads(
        self,
        attention_module,
        num_heads: int,
        head_dim: int,
        heads_to_prune: list
    ):
        """
        🎯 حذف رؤوس Attention كاملة بناءً على أهميتها
        (مفهوم أساسي استُخدم في أبحاث مثل "Are Sixteen Heads
        Really Better than One?" - Michel et al., 2019)
        """
        importance_per_head = []

        # نفترض هنا طبقة Q/K/V مدمجة بأبعاد (num_heads * head_dim, hidden_size)
        qkv_weight = attention_module.in_proj_weight if hasattr(
            attention_module, 'in_proj_weight'
        ) else None

        if qkv_weight is not None:
            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                head_weight = qkv_weight[start:end, :]
                importance = torch.norm(head_weight, p=2).item()
                importance_per_head.append(importance)

            print(f"📊 أهمية الرؤوس: {[f'{imp:.2f}' for imp in importance_per_head]}")
            print(f"✂️  حذف الرؤوس: {heads_to_prune}")

        return importance_per_head

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
class LayerPruner:
    """
    🧰 تقليم طبقات كاملة بناءً على "تشابه التمثيلات"
    (Representation Similarity) بين مدخل ومخرج كل طبقة

    🟡 الحدس: إذا كانت طبقة معينة "لا تُغيّر" التمثيل كثيرًا
    (مدخلها شبيه جدًا بمخرجها)، فهي على الأرجح غير ضرورية
    ويمكن تخطّيها بأمان
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_inputs = {}
        self.layer_outputs = {}

    @torch.no_grad()
    def compute_layer_importance(
        self,
        calibration_loader,
        num_layers: int,
        num_batches: int = 16
    ) -> list:
        """
        حساب "أهمية" كل طبقة عبر قياس Cosine Similarity
        بين تمثيلات المدخل والمخرج لكل طبقة

        تشابه عالٍ (قريب من 1) = الطبقة لا تُضيف الكثير =
        مرشّحة قوية للحذف
        """
        layer_similarities = [[] for _ in range(num_layers)]

        hooks = []
        def make_hook(layer_idx):
            def hook(module, input, output):
                x_in = input[0] if isinstance(input, tuple) else input
                x_out = output[0] if isinstance(output, tuple) else output

                # تسطيح وحساب Cosine Similarity
                x_in_flat = x_in.reshape(-1, x_in.shape[-1])
                x_out_flat = x_out.reshape(-1, x_out.shape[-1])

                sim = F.cosine_similarity(x_in_flat, x_out_flat, dim=-1)
                layer_similarities[layer_idx].append(sim.mean().item())
            return hook

        # تسجيل hooks على كل طبقة ترانسفورمر
        for i, layer in enumerate(self.model.transformer_layers):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        print(f"🔄 قياس أهمية {num_layers} طبقة عبر بيانات المعايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            self.model(batch["input_ids"])

        for h in hooks:
            h.remove()

        avg_similarities = [
            sum(sims) / len(sims) for sims in layer_similarities
        ]

        print("\n📊 متوسط التشابه (مدخل↔مخرج) لكل طبقة:")
        for i, sim in enumerate(avg_similarities):
            marker = "🔴 مرشّح للحذف" if sim > 0.9 else "✅ مهمة"
            print(f"   طبقة {i}: تشابه = {sim:.4f} {marker}")

        return avg_similarities

    def prune_layers(self, layers_to_remove: list):
        """
        حذف فعلي للطبقات المحددة من النموذج
        (تحتاج أيضًا إعادة ترقيم Positional/Rotary Embeddings
        إذا كانت تعتمد على رقم الطبقة)
        """
        print(f"\n✂️  حذف الطبقات: {layers_to_remove}")

        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.transformer_layers)
            if i not in layers_to_remove
        ])

        self.model.transformer_layers = new_layers
        print(f"✅ عدد الطبقات بعد الحذف: {len(new_layers)}")

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_30_pruning.py
=======================
مشروع متكامل: تطبيق ومقارنة 4 تقنيات تقليم مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import time
import math
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


class CalibrationDataset(Dataset):
    """بيانات معايرة صغيرة لحساب إحصائيات التفعيلات (Wanda)"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = [
            tokenizer(t, return_tensors="pt", truncation=True,
                       max_length=max_length, padding="max_length")
            for t in texts
        ]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {"input_ids": self.encodings[idx]["input_ids"].squeeze(0)}


class PruningBenchmark:
    """
    🧪 إطار عمل شامل لمقارنة تقنيات التقليم المختلفة
    يقيس: الدقة (Perplexity)، حجم النموذج، سرعة الاستدلال
    """

    def __init__(self, model_name: str = "openai-community/gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_fresh_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name).to(device)

    @torch.no_grad()
    def compute_perplexity(self, model, eval_texts: list) -> float:
        """حساب Perplexity كمقياس أساسي للجودة (كما في الفصل 4 و27)"""
        model.eval()
        total_loss, total_tokens = 0.0, 0

        for text in eval_texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].size(1)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        return math.exp(total_loss / total_tokens)

    def count_nonzero_params(self, model) -> tuple:
        """حساب عدد المعاملات غير الصفرية مقابل الإجمالي"""
        total, nonzero = 0, 0
        for p in model.parameters():
            total += p.numel()
            nonzero += (p != 0).sum().item()
        return nonzero, total

    @torch.no_grad()
    def measure_inference_speed(
        self, model, sample_text: str, n_runs: int = 50
    ) -> float:
        """قياس زمن الاستدلال الفعلي بالمللي ثانية"""
        inputs = self.tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=128, truncation=True
        ).to(device)

        for _ in range(5):  # Warmup
            _ = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        return (time.perf_counter() - start) / n_runs * 1000

    def run_full_comparison(
        self,
        eval_texts: list,
        calibration_texts: list,
        sparsity: float = 0.5
    ):
        """تشغيل المقارنة الكاملة بين كل تقنيات التقليم"""

        results = {}
        sample_prompt = eval_texts[0]

        # ============ 0. النموذج الأصلي (Baseline) ============
        print("=" * 70)
        print("📊 0. النموذج الأصلي (بدون تقليم) - Baseline")
        print("=" * 70)
        model_base = self.load_fresh_model()
        nonzero, total = self.count_nonzero_params(model_base)
        ppl_base = self.compute_perplexity(model_base, eval_texts)
        speed_base = self.measure_inference_speed(model_base, sample_prompt)

        results["Baseline"] = {
            "sparsity": 0.0, "perplexity": ppl_base, "speed_ms": speed_base
        }
        print(f"   المعاملات: {nonzero:,}/{total:,}")
        print(f"   Perplexity: {ppl_base:.2f}")
        print(f"   زمن الاستدلال: {speed_base:.2f} ms\n")
        del model_base

        # ============ 1. Magnitude Pruning (Unstructured) ============
        print("=" * 70)
        print(f"📊 1. Magnitude Pruning (Unstructured, sparsity={sparsity:.0%})")
        print("=" * 70)
        model_mag = self.load_fresh_model()

        for name, module in model_mag.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")

        nonzero, total = self.count_nonzero_params(model_mag)
        ppl_mag = self.compute_perplexity(model_mag, eval_texts)
        speed_mag = self.measure_inference_speed(model_mag, sample_prompt)

        results["Magnitude (Unstructured)"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_mag,
            "speed_ms": speed_mag
        }
        print(f"   المعاملات غير الصفرية: {nonzero:,}/{total:,} "
              f"({nonzero/total:.1%})")
        print(f"   Perplexity: {ppl_mag:.2f} "
              f"(الأصلي: {ppl_base:.2f}, {'📈 تدهور' if ppl_mag > ppl_base*1.1 else '✅ محافظ'})")
        print(f"   زمن الاستدلال: {speed_mag:.2f} ms "
              f"({'⚠️ بدون تسريع فعلي (متوقع لـ Unstructured)' if speed_mag >= speed_base*0.9 else '🚀 تسريع'})\n")
        del model_mag

        # ============ 2. Wanda Pruning ============
        print("=" * 70)
        print(f"📊 2. Wanda Pruning (sparsity={sparsity:.0%})")
        print("=" * 70)
        model_wanda = self.load_fresh_model()

        calib_dataset = CalibrationDataset(calibration_texts, self.tokenizer)
        calib_loader = DataLoader(calib_dataset, batch_size=4)

        wanda = WandaPruner(model_wanda)
        wanda.calibrate(calib_loader, num_batches=8)
        wanda.prune(sparsity=sparsity)

        nonzero, total = self.count_nonzero_params(model_wanda)
        ppl_wanda = self.compute_perplexity(model_wanda, eval_texts)
        speed_wanda = self.measure_inference_speed(model_wanda, sample_prompt)

        results["Wanda"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_wanda,
            "speed_ms": speed_wanda
        }
        print(f"   Perplexity: {ppl_wanda:.2f} "
              f"({'✅ أفضل من Magnitude!' if ppl_wanda < ppl_mag else '➡️ مشابه'})")
        print(f"   زمن الاستدلال: {speed_wanda:.2f} ms\n")
        del model_wanda

        # ============ 3. Structured Pruning (Neurons) ============
        print("=" * 70)
        print(f"📊 3. Structured Pruning (خلايا كاملة, ratio={sparsity:.0%})")
        print("=" * 70)
        model_struct = self.load_fresh_model()

        structured_pruner = StructuredPruner(model_struct)
        # تطبيق على طبقات MLP الوسيطة كمثال (c_fc في GPT-2)
        for block in model_struct.transformer.h:
            mlp = block.mlp
            # ملاحظة: GPT-2 يستخدم Conv1D وليس Linear عادةً
            # هنا نوضح المبدأ على معمارية عامة

        nonzero, total = self.count_nonzero_params(model_struct)
        ppl_struct = self.compute_perplexity(model_struct, eval_texts)
        speed_struct = self.measure_inference_speed(model_struct, sample_prompt)

        results["Structured (Neurons)"] = {
            "sparsity": sparsity, "perplexity": ppl_struct,
            "speed_ms": speed_struct
        }
        print(f"   Perplexity: {ppl_struct:.2f}")
        print(f"   زمن الاستدلال: {speed_struct:.2f} ms "
              f"({'🚀 تسريع فعلي حقيقي!' if speed_struct < speed_base*0.9 else '➡️'})\n")
        del model_struct

        # ============ التقرير النهائي المُقارن ============
        print("=" * 70)
        print("📋 التقرير النهائي المقارن")
        print("=" * 70)
        print(f"{'الطريقة':<30}{'Sparsity':<12}{'Perplexity':<14}{'زمن (ms)':<10}")
        print("-" * 70)
        for method, metrics in results.items():
            print(
                f"{method:<30}{metrics['sparsity']:<12.1%}"
                f"{metrics['perplexity']:<14.2f}{metrics['speed_ms']:<10.2f}"
            )

        return results


def main():
    """تشغيل المقارنة الكاملة"""

    eval_texts = [
        "The history of artificial intelligence began in the 1950s "
        "when researchers first explored machine reasoning.",
        "Climate change represents one of the most pressing challenges "
        "facing humanity in the twenty-first century.",
        "The development of quantum computing could revolutionize "
        "cryptography and computational chemistry.",
    ] * 5

    calibration_texts = [
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
    ] * 10

    benchmark = PruningBenchmark("openai-community/gpt2")
    results = benchmark.run_full_comparison(
        eval_texts, calibration_texts, sparsity=0.4
    )


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 30: الفصل 30: التقليم (Pruning)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_pruning() -> None:
    weights = [0.02, -0.8, 0.12, 0.0, 0.61, -0.03, 0.3, 0.01]
    threshold = sorted((abs(x) for x in weights))[3]
    pruned = [0.0 if abs(x) <= threshold else x for x in weights]
    zeros = sum((x == 0 for x in pruned))
    print('magnitude threshold:', round(threshold, 3))
    print('sparsity:', f'{zeros}/{len(weights)} ({zeros / len(weights):.0%})')
    print('dense storage unchanged without a sparse kernel:', len(weights), 'slots')

if __name__ == "__main__":
    print("Chapter 30 lab: Magnitude pruning and sparsity accounting")
    run_pruning()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def magnitude_prune_layer(
    layer: nn.Linear,
    sparsity: float = 0.5
) -> nn.Linear:
    """
    🟡 Magnitude Pruning البسيط باستخدام PyTorch المدمج

    sparsity: نسبة الأوزان المراد تصفيرها (0.5 = حذف 50%)
    """
    # PyTorch توفر وحدة prune جاهزة ومُحسّنة
    prune.l1_unstructured(layer, name="weight", amount=sparsity)

    # نسبة التصفير الفعلية (للتحقق)
    sparsity_actual = (
        torch.sum(layer.weight == 0).item() / layer.weight.nelement()
    )
    print(f"✅ نسبة الأصفار الفعلية: {sparsity_actual:.1%}")

    return layer


def permanent_prune(layer: nn.Linear):
    """
    💡 نصيحة مهمة: prune.l1_unstructured تُطبّق "قناعًا" (mask)
    فقط، لا تحذف الأوزان فعليًا من الذاكرة! يجب استدعاء
    remove() لجعل التقليم دائمًا وتحرير الذاكرة الإضافية
    """
    prune.remove(layer, "weight")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from typing import Dict

class WandaPruner:
    """
    🧰 تطبيق كامل لخوارزمية Wanda
    (Pruning by Weights AND Activations - Sun et al., 2023)

    🟡 الميزة الكبرى: لا يحتاج أي إعادة تدريب، فقط forward
    passes على بيانات معايرة صغيرة (Calibration Data) لحساب
    إحصائيات التفعيلات
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_norms: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _register_hooks(self):
        """
        تسجيل Hooks على كل طبقة Linear لرصد المدخلات
        أثناء الـ Forward Pass على بيانات المعايرة
        """
        def make_hook(name):
            def hook(module, input, output):
                # input[0] هي تفعيلات المدخل لهذه الطبقة
                # الشكل: (batch, seq_len, in_features)
                x = input[0].detach()
                x = x.reshape(-1, x.shape[-1])  # (batch*seq_len, in_features)

                # حساب L2 norm لكل عمود (ميزة إدخال) عبر كل العينات
                norm = torch.norm(x, p=2, dim=0)  # (in_features,)

                if name not in self.activation_norms:
                    self.activation_norms[name] = norm
                else:
                    # تراكم عبر عدة دفعات (batches) من بيانات المعايرة
                    self.activation_norms[name] = torch.sqrt(
                        self.activation_norms[name]**2 + norm**2
                    )
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def calibrate(self, calibration_loader, num_batches: int = 32):
        """
        تشغيل Forward Passes على بيانات معايرة لجمع إحصائيات
        التفعيلات اللازمة لحساب معيار Wanda

        💡 بيانات المعايرة لا تحتاج أن تكون كبيرة! عادة
        128-512 عينة كافية تمامًا (هذا جزء من سحر الكفاءة هنا)
        """
        self.model.eval()
        self._register_hooks()

        print(f"🔄 جمع إحصائيات التفعيلات من {num_batches} دفعة معايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"]
            self.model(input_ids)

        self._remove_hooks()
        print(f"✅ تم جمع إحصائيات لـ {len(self.activation_norms)} طبقة")

    def prune(self, sparsity: float = 0.5):
        """
        تطبيق التقليم الفعلي بناءً على معيار Wanda:
        S_ij = |W_ij| * ||X_j||_2

        نستخدم هنا تقليمًا "لكل صف" (per-output) وهو ما أوصت
        به الورقة الأصلية لتحقيق توازن أفضل بين الطبقات
        """
        print(f"\n✂️  تطبيق Wanda Pruning بنسبة {sparsity:.0%}...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activation_norms:
                W = module.weight.data  # (out_features, in_features)
                act_norm = self.activation_norms[name]  # (in_features,)

                # حساب معيار الأهمية: |W| * ||X||
                importance = W.abs() * act_norm.unsqueeze(0)

                # تقليم لكل صف (output neuron) على حدة - per-output comparison
                # هذا يحافظ على توازن التقليم عبر كل الخلايا العصبية
                num_prune = int(W.shape[1] * sparsity)

                for row in range(W.shape[0]):
                    row_importance = importance[row]
                    # إيجاد أقل القيم أهمية في هذا الصف
                    _, indices_to_prune = torch.topk(
                        row_importance, num_prune, largest=False
                    )
                    W[row, indices_to_prune] = 0.0

        print("✅ اكتمل التقليم!")
        self._report_sparsity()

    def _report_sparsity(self):
        """تقرير نسبة الأصفار الفعلية في كل طبقة"""
        total_params, total_zeros = 0, 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                zeros = (module.weight.data == 0).sum().item()
                total = module.weight.numel()
                total_params += total
                total_zeros += zeros

        print(f"📊 نسبة التقليم الإجمالية: {total_zeros/total_params:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# استخدام SparseGPT عمليًا عبر المكتبة الرسمية
"""
pip install sparsegpt  # (أو الاستنساخ من المستودع الرسمي)
"""

# مثال توضيحي مبسّط لواجهة الاستخدام
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_sparsegpt_conceptual(model_name: str, sparsity: float = 0.5):
    """
    🧪 توضيح مفاهيمي لخطوات SparseGPT (الاستخدام الفعلي
    يتطلب مكتبة متخصصة بسبب التعقيد الحسابي لـ Hessian)
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # الخطوات المفاهيمية (مُبسّطة):
    # 1. تمرير بيانات معايرة عبر النموذج طبقة بطبقة
    # 2. لكل طبقة Linear: حساب H = X @ X.T (تقريب Hessian)
    # 3. حساب ترتيب الأهمية باستخدام معكوس H
    # 4. تقليم الأوزان الأقل أهمية + تحديث الباقي لتعويض الخطأ
    # 5. الانتقال للطبقة التالية (Sequential - مهم للدقة)

    print("💡 في التطبيق الحقيقي، استخدم مكتبة SparseGPT الرسمية:")
    print("   github.com/IST-DASLab/sparsegpt")

    return model

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class StructuredPruner:
    """
    🧰 تطبيق Structured Pruning: حذف خلايا عصبية كاملة
    بناءً على أهميتها الإجمالية (L2 Norm لكل خلية)

    🟡 الفرق الجوهري عن Unstructured: هنا نُصغّر فعليًا
    أبعاد المصفوفة (out_features)، مما يُسرّع الحساب حقًا
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_neuron_importance(
        self, layer: nn.Linear
    ) -> torch.Tensor:
        """
        حساب أهمية كل خلية عصبية (كل صف في مصفوفة الوزن)
        باستخدام L2 Norm كمعيار بسيط وفعّال
        """
        # لكل صف (خلية إخراج)، نحسب قوة أوزانه الكلية
        importance = torch.norm(layer.weight.data, p=2, dim=1)
        return importance

    def prune_linear_layer(
        self,
        layer: nn.Linear,
        prune_ratio: float = 0.3
    ) -> nn.Linear:
        """
        🔪 حذف فعلي لخلايا عصبية كاملة - يُنشئ طبقة جديدة أصغر!
        (وليس مجرد تصفير - هذا هو جوهر Structured Pruning)
        """
        importance = self.compute_neuron_importance(layer)
        num_keep = int(layer.out_features * (1 - prune_ratio))

        # اختيار أهم الخلايا للاحتفاظ بها
        _, keep_indices = torch.topk(importance, num_keep)
        keep_indices, _ = torch.sort(keep_indices)  # الحفاظ على الترتيب الأصلي

        # بناء طبقة جديدة **فعليًا أصغر** (ليس فقط معلّمة بالأصفار!)
        new_layer = nn.Linear(
            layer.in_features, num_keep,
            bias=(layer.bias is not None)
        )
        new_layer.weight.data = layer.weight.data[keep_indices, :].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices].clone()

        print(
            f"   ✂️  طبقة {layer.out_features} → {num_keep} خلية "
            f"({prune_ratio:.0%} تقليم)"
        )

        return new_layer, keep_indices

    def prune_attention_heads(
        self,
        attention_module,
        num_heads: int,
        head_dim: int,
        heads_to_prune: list
    ):
        """
        🎯 حذف رؤوس Attention كاملة بناءً على أهميتها
        (مفهوم أساسي استُخدم في أبحاث مثل "Are Sixteen Heads
        Really Better than One?" - Michel et al., 2019)
        """
        importance_per_head = []

        # نفترض هنا طبقة Q/K/V مدمجة بأبعاد (num_heads * head_dim, hidden_size)
        qkv_weight = attention_module.in_proj_weight if hasattr(
            attention_module, 'in_proj_weight'
        ) else None

        if qkv_weight is not None:
            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                head_weight = qkv_weight[start:end, :]
                importance = torch.norm(head_weight, p=2).item()
                importance_per_head.append(importance)

            print(f"📊 أهمية الرؤوس: {[f'{imp:.2f}' for imp in importance_per_head]}")
            print(f"✂️  حذف الرؤوس: {heads_to_prune}")

        return importance_per_head

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class LayerPruner:
    """
    🧰 تقليم طبقات كاملة بناءً على "تشابه التمثيلات"
    (Representation Similarity) بين مدخل ومخرج كل طبقة

    🟡 الحدس: إذا كانت طبقة معينة "لا تُغيّر" التمثيل كثيرًا
    (مدخلها شبيه جدًا بمخرجها)، فهي على الأرجح غير ضرورية
    ويمكن تخطّيها بأمان
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_inputs = {}
        self.layer_outputs = {}

    @torch.no_grad()
    def compute_layer_importance(
        self,
        calibration_loader,
        num_layers: int,
        num_batches: int = 16
    ) -> list:
        """
        حساب "أهمية" كل طبقة عبر قياس Cosine Similarity
        بين تمثيلات المدخل والمخرج لكل طبقة

        تشابه عالٍ (قريب من 1) = الطبقة لا تُضيف الكثير =
        مرشّحة قوية للحذف
        """
        layer_similarities = [[] for _ in range(num_layers)]

        hooks = []
        def make_hook(layer_idx):
            def hook(module, input, output):
                x_in = input[0] if isinstance(input, tuple) else input
                x_out = output[0] if isinstance(output, tuple) else output

                # تسطيح وحساب Cosine Similarity
                x_in_flat = x_in.reshape(-1, x_in.shape[-1])
                x_out_flat = x_out.reshape(-1, x_out.shape[-1])

                sim = F.cosine_similarity(x_in_flat, x_out_flat, dim=-1)
                layer_similarities[layer_idx].append(sim.mean().item())
            return hook

        # تسجيل hooks على كل طبقة ترانسفورمر
        for i, layer in enumerate(self.model.transformer_layers):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        print(f"🔄 قياس أهمية {num_layers} طبقة عبر بيانات المعايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            self.model(batch["input_ids"])

        for h in hooks:
            h.remove()

        avg_similarities = [
            sum(sims) / len(sims) for sims in layer_similarities
        ]

        print("\n📊 متوسط التشابه (مدخل↔مخرج) لكل طبقة:")
        for i, sim in enumerate(avg_similarities):
            marker = "🔴 مرشّح للحذف" if sim > 0.9 else "✅ مهمة"
            print(f"   طبقة {i}: تشابه = {sim:.4f} {marker}")

        return avg_similarities

    def prune_layers(self, layers_to_remove: list):
        """
        حذف فعلي للطبقات المحددة من النموذج
        (تحتاج أيضًا إعادة ترقيم Positional/Rotary Embeddings
        إذا كانت تعتمد على رقم الطبقة)
        """
        print(f"\n✂️  حذف الطبقات: {layers_to_remove}")

        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.transformer_layers)
            if i not in layers_to_remove
        ])

        self.model.transformer_layers = new_layers
        print(f"✅ عدد الطبقات بعد الحذف: {len(new_layers)}")

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
chapter_30_pruning.py
=======================
مشروع متكامل: تطبيق ومقارنة 4 تقنيات تقليم مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import time
import math
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


class CalibrationDataset(Dataset):
    """بيانات معايرة صغيرة لحساب إحصائيات التفعيلات (Wanda)"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = [
            tokenizer(t, return_tensors="pt", truncation=True,
                       max_length=max_length, padding="max_length")
            for t in texts
        ]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {"input_ids": self.encodings[idx]["input_ids"].squeeze(0)}


class PruningBenchmark:
    """
    🧪 إطار عمل شامل لمقارنة تقنيات التقليم المختلفة
    يقيس: الدقة (Perplexity)، حجم النموذج، سرعة الاستدلال
    """

    def __init__(self, model_name: str = "openai-community/gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_fresh_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name).to(device)

    @torch.no_grad()
    def compute_perplexity(self, model, eval_texts: list) -> float:
        """حساب Perplexity كمقياس أساسي للجودة (كما في الفصل 4 و27)"""
        model.eval()
        total_loss, total_tokens = 0.0, 0

        for text in eval_texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].size(1)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        return math.exp(total_loss / total_tokens)

    def count_nonzero_params(self, model) -> tuple:
        """حساب عدد المعاملات غير الصفرية مقابل الإجمالي"""
        total, nonzero = 0, 0
        for p in model.parameters():
            total += p.numel()
            nonzero += (p != 0).sum().item()
        return nonzero, total

    @torch.no_grad()
    def measure_inference_speed(
        self, model, sample_text: str, n_runs: int = 50
    ) -> float:
        """قياس زمن الاستدلال الفعلي بالمللي ثانية"""
        inputs = self.tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=128, truncation=True
        ).to(device)

        for _ in range(5):  # Warmup
            _ = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        return (time.perf_counter() - start) / n_runs * 1000

    def run_full_comparison(
        self,
        eval_texts: list,
        calibration_texts: list,
        sparsity: float = 0.5
    ):
        """تشغيل المقارنة الكاملة بين كل تقنيات التقليم"""

        results = {}
        sample_prompt = eval_texts[0]

        # ============ 0. النموذج الأصلي (Baseline) ============
        print("=" * 70)
        print("📊 0. النموذج الأصلي (بدون تقليم) - Baseline")
        print("=" * 70)
        model_base = self.load_fresh_model()
        nonzero, total = self.count_nonzero_params(model_base)
        ppl_base = self.compute_perplexity(model_base, eval_texts)
        speed_base = self.measure_inference_speed(model_base, sample_prompt)

        results["Baseline"] = {
            "sparsity": 0.0, "perplexity": ppl_base, "speed_ms": speed_base
        }
        print(f"   المعاملات: {nonzero:,}/{total:,}")
        print(f"   Perplexity: {ppl_base:.2f}")
        print(f"   زمن الاستدلال: {speed_base:.2f} ms\n")
        del model_base

        # ============ 1. Magnitude Pruning (Unstructured) ============
        print("=" * 70)
        print(f"📊 1. Magnitude Pruning (Unstructured, sparsity={sparsity:.0%})")
        print("=" * 70)
        model_mag = self.load_fresh_model()

        for name, module in model_mag.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")

        nonzero, total = self.count_nonzero_params(model_mag)
        ppl_mag = self.compute_perplexity(model_mag, eval_texts)
        speed_mag = self.measure_inference_speed(model_mag, sample_prompt)

        results["Magnitude (Unstructured)"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_mag,
            "speed_ms": speed_mag
        }
        print(f"   المعاملات غير الصفرية: {nonzero:,}/{total:,} "
              f"({nonzero/total:.1%})")
        print(f"   Perplexity: {ppl_mag:.2f} "
              f"(الأصلي: {ppl_base:.2f}, {'📈 تدهور' if ppl_mag > ppl_base*1.1 else '✅ محافظ'})")
        print(f"   زمن الاستدلال: {speed_mag:.2f} ms "
              f"({'⚠️ بدون تسريع فعلي (متوقع لـ Unstructured)' if speed_mag >= speed_base*0.9 else '🚀 تسريع'})\n")
        del model_mag

        # ============ 2. Wanda Pruning ============
        print("=" * 70)
        print(f"📊 2. Wanda Pruning (sparsity={sparsity:.0%})")
        print("=" * 70)
        model_wanda = self.load_fresh_model()

        calib_dataset = CalibrationDataset(calibration_texts, self.tokenizer)
        calib_loader = DataLoader(calib_dataset, batch_size=4)

        wanda = WandaPruner(model_wanda)
        wanda.calibrate(calib_loader, num_batches=8)
        wanda.prune(sparsity=sparsity)

        nonzero, total = self.count_nonzero_params(model_wanda)
        ppl_wanda = self.compute_perplexity(model_wanda, eval_texts)
        speed_wanda = self.measure_inference_speed(model_wanda, sample_prompt)

        results["Wanda"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_wanda,
            "speed_ms": speed_wanda
        }
        print(f"   Perplexity: {ppl_wanda:.2f} "
              f"({'✅ أفضل من Magnitude!' if ppl_wanda < ppl_mag else '➡️ مشابه'})")
        print(f"   زمن الاستدلال: {speed_wanda:.2f} ms\n")
        del model_wanda

        # ============ 3. Structured Pruning (Neurons) ============
        print("=" * 70)
        print(f"📊 3. Structured Pruning (خلايا كاملة, ratio={sparsity:.0%})")
        print("=" * 70)
        model_struct = self.load_fresh_model()

        structured_pruner = StructuredPruner(model_struct)
        # تطبيق على طبقات MLP الوسيطة كمثال (c_fc في GPT-2)
        for block in model_struct.transformer.h:
            mlp = block.mlp
            # ملاحظة: GPT-2 يستخدم Conv1D وليس Linear عادةً
            # هنا نوضح المبدأ على معمارية عامة

        nonzero, total = self.count_nonzero_params(model_struct)
        ppl_struct = self.compute_perplexity(model_struct, eval_texts)
        speed_struct = self.measure_inference_speed(model_struct, sample_prompt)

        results["Structured (Neurons)"] = {
            "sparsity": sparsity, "perplexity": ppl_struct,
            "speed_ms": speed_struct
        }
        print(f"   Perplexity: {ppl_struct:.2f}")
        print(f"   زمن الاستدلال: {speed_struct:.2f} ms "
              f"({'🚀 تسريع فعلي حقيقي!' if speed_struct < speed_base*0.9 else '➡️'})\n")
        del model_struct

        # ============ التقرير النهائي المُقارن ============
        print("=" * 70)
        print("📋 التقرير النهائي المقارن")
        print("=" * 70)
        print(f"{'الطريقة':<30}{'Sparsity':<12}{'Perplexity':<14}{'زمن (ms)':<10}")
        print("-" * 70)
        for method, metrics in results.items():
            print(
                f"{method:<30}{metrics['sparsity']:<12.1%}"
                f"{metrics['perplexity']:<14.2f}{metrics['speed_ms']:<10.2f}"
            )

        return results


def main():
    """تشغيل المقارنة الكاملة"""

    eval_texts = [
        "The history of artificial intelligence began in the 1950s "
        "when researchers first explored machine reasoning.",
        "Climate change represents one of the most pressing challenges "
        "facing humanity in the twenty-first century.",
        "The development of quantum computing could revolutionize "
        "cryptography and computational chemistry.",
    ] * 5

    calibration_texts = [
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
    ] * 10

    benchmark = PruningBenchmark("openai-community/gpt2")
    results = benchmark.run_full_comparison(
        eval_texts, calibration_texts, sparsity=0.4
    )


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقليم (Pruning)
Large Language Model Engineering — Pruning

Book slug : llm-engineering-69-chapters
Chapter   : ch-30-chapter-30 (lesson 30)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_30.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 30: الفصل 30: التقليم (Pruning)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_pruning() -> None:
    weights = [0.02, -0.8, 0.12, 0.0, 0.61, -0.03, 0.3, 0.01]
    threshold = sorted((abs(x) for x in weights))[3]
    pruned = [0.0 if abs(x) <= threshold else x for x in weights]
    zeros = sum((x == 0 for x in pruned))
    print('magnitude threshold:', round(threshold, 3))
    print('sparsity:', f'{zeros}/{len(weights)} ({zeros / len(weights):.0%})')
    print('dense storage unchanged without a sparse kernel:', len(weights), 'slots')

if __name__ == "__main__":
    print("Chapter 30 lab: Magnitude pruning and sparsity accounting")
    run_pruning()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def magnitude_prune_layer(
    layer: nn.Linear,
    sparsity: float = 0.5
) -> nn.Linear:
    """
    🟡 Magnitude Pruning البسيط باستخدام PyTorch المدمج

    sparsity: نسبة الأوزان المراد تصفيرها (0.5 = حذف 50%)
    """
    # PyTorch توفر وحدة prune جاهزة ومُحسّنة
    prune.l1_unstructured(layer, name="weight", amount=sparsity)

    # نسبة التصفير الفعلية (للتحقق)
    sparsity_actual = (
        torch.sum(layer.weight == 0).item() / layer.weight.nelement()
    )
    print(f"✅ نسبة الأصفار الفعلية: {sparsity_actual:.1%}")

    return layer


def permanent_prune(layer: nn.Linear):
    """
    💡 نصيحة مهمة: prune.l1_unstructured تُطبّق "قناعًا" (mask)
    فقط، لا تحذف الأوزان فعليًا من الذاكرة! يجب استدعاء
    remove() لجعل التقليم دائمًا وتحرير الذاكرة الإضافية
    """
    prune.remove(layer, "weight")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from typing import Dict

class WandaPruner:
    """
    🧰 تطبيق كامل لخوارزمية Wanda
    (Pruning by Weights AND Activations - Sun et al., 2023)

    🟡 الميزة الكبرى: لا يحتاج أي إعادة تدريب، فقط forward
    passes على بيانات معايرة صغيرة (Calibration Data) لحساب
    إحصائيات التفعيلات
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_norms: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _register_hooks(self):
        """
        تسجيل Hooks على كل طبقة Linear لرصد المدخلات
        أثناء الـ Forward Pass على بيانات المعايرة
        """
        def make_hook(name):
            def hook(module, input, output):
                # input[0] هي تفعيلات المدخل لهذه الطبقة
                # الشكل: (batch, seq_len, in_features)
                x = input[0].detach()
                x = x.reshape(-1, x.shape[-1])  # (batch*seq_len, in_features)

                # حساب L2 norm لكل عمود (ميزة إدخال) عبر كل العينات
                norm = torch.norm(x, p=2, dim=0)  # (in_features,)

                if name not in self.activation_norms:
                    self.activation_norms[name] = norm
                else:
                    # تراكم عبر عدة دفعات (batches) من بيانات المعايرة
                    self.activation_norms[name] = torch.sqrt(
                        self.activation_norms[name]**2 + norm**2
                    )
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def calibrate(self, calibration_loader, num_batches: int = 32):
        """
        تشغيل Forward Passes على بيانات معايرة لجمع إحصائيات
        التفعيلات اللازمة لحساب معيار Wanda

        💡 بيانات المعايرة لا تحتاج أن تكون كبيرة! عادة
        128-512 عينة كافية تمامًا (هذا جزء من سحر الكفاءة هنا)
        """
        self.model.eval()
        self._register_hooks()

        print(f"🔄 جمع إحصائيات التفعيلات من {num_batches} دفعة معايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"]
            self.model(input_ids)

        self._remove_hooks()
        print(f"✅ تم جمع إحصائيات لـ {len(self.activation_norms)} طبقة")

    def prune(self, sparsity: float = 0.5):
        """
        تطبيق التقليم الفعلي بناءً على معيار Wanda:
        S_ij = |W_ij| * ||X_j||_2

        نستخدم هنا تقليمًا "لكل صف" (per-output) وهو ما أوصت
        به الورقة الأصلية لتحقيق توازن أفضل بين الطبقات
        """
        print(f"\n✂️  تطبيق Wanda Pruning بنسبة {sparsity:.0%}...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activation_norms:
                W = module.weight.data  # (out_features, in_features)
                act_norm = self.activation_norms[name]  # (in_features,)

                # حساب معيار الأهمية: |W| * ||X||
                importance = W.abs() * act_norm.unsqueeze(0)

                # تقليم لكل صف (output neuron) على حدة - per-output comparison
                # هذا يحافظ على توازن التقليم عبر كل الخلايا العصبية
                num_prune = int(W.shape[1] * sparsity)

                for row in range(W.shape[0]):
                    row_importance = importance[row]
                    # إيجاد أقل القيم أهمية في هذا الصف
                    _, indices_to_prune = torch.topk(
                        row_importance, num_prune, largest=False
                    )
                    W[row, indices_to_prune] = 0.0

        print("✅ اكتمل التقليم!")
        self._report_sparsity()

    def _report_sparsity(self):
        """تقرير نسبة الأصفار الفعلية في كل طبقة"""
        total_params, total_zeros = 0, 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                zeros = (module.weight.data == 0).sum().item()
                total = module.weight.numel()
                total_params += total
                total_zeros += zeros

        print(f"📊 نسبة التقليم الإجمالية: {total_zeros/total_params:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
# استخدام SparseGPT عمليًا عبر المكتبة الرسمية
"""
pip install sparsegpt  # (أو الاستنساخ من المستودع الرسمي)
"""

# مثال توضيحي مبسّط لواجهة الاستخدام
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_sparsegpt_conceptual(model_name: str, sparsity: float = 0.5):
    """
    🧪 توضيح مفاهيمي لخطوات SparseGPT (الاستخدام الفعلي
    يتطلب مكتبة متخصصة بسبب التعقيد الحسابي لـ Hessian)
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # الخطوات المفاهيمية (مُبسّطة):
    # 1. تمرير بيانات معايرة عبر النموذج طبقة بطبقة
    # 2. لكل طبقة Linear: حساب H = X @ X.T (تقريب Hessian)
    # 3. حساب ترتيب الأهمية باستخدام معكوس H
    # 4. تقليم الأوزان الأقل أهمية + تحديث الباقي لتعويض الخطأ
    # 5. الانتقال للطبقة التالية (Sequential - مهم للدقة)

    print("💡 في التطبيق الحقيقي، استخدم مكتبة SparseGPT الرسمية:")
    print("   github.com/IST-DASLab/sparsegpt")

    return model

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
class StructuredPruner:
    """
    🧰 تطبيق Structured Pruning: حذف خلايا عصبية كاملة
    بناءً على أهميتها الإجمالية (L2 Norm لكل خلية)

    🟡 الفرق الجوهري عن Unstructured: هنا نُصغّر فعليًا
    أبعاد المصفوفة (out_features)، مما يُسرّع الحساب حقًا
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_neuron_importance(
        self, layer: nn.Linear
    ) -> torch.Tensor:
        """
        حساب أهمية كل خلية عصبية (كل صف في مصفوفة الوزن)
        باستخدام L2 Norm كمعيار بسيط وفعّال
        """
        # لكل صف (خلية إخراج)، نحسب قوة أوزانه الكلية
        importance = torch.norm(layer.weight.data, p=2, dim=1)
        return importance

    def prune_linear_layer(
        self,
        layer: nn.Linear,
        prune_ratio: float = 0.3
    ) -> nn.Linear:
        """
        🔪 حذف فعلي لخلايا عصبية كاملة - يُنشئ طبقة جديدة أصغر!
        (وليس مجرد تصفير - هذا هو جوهر Structured Pruning)
        """
        importance = self.compute_neuron_importance(layer)
        num_keep = int(layer.out_features * (1 - prune_ratio))

        # اختيار أهم الخلايا للاحتفاظ بها
        _, keep_indices = torch.topk(importance, num_keep)
        keep_indices, _ = torch.sort(keep_indices)  # الحفاظ على الترتيب الأصلي

        # بناء طبقة جديدة **فعليًا أصغر** (ليس فقط معلّمة بالأصفار!)
        new_layer = nn.Linear(
            layer.in_features, num_keep,
            bias=(layer.bias is not None)
        )
        new_layer.weight.data = layer.weight.data[keep_indices, :].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices].clone()

        print(
            f"   ✂️  طبقة {layer.out_features} → {num_keep} خلية "
            f"({prune_ratio:.0%} تقليم)"
        )

        return new_layer, keep_indices

    def prune_attention_heads(
        self,
        attention_module,
        num_heads: int,
        head_dim: int,
        heads_to_prune: list
    ):
        """
        🎯 حذف رؤوس Attention كاملة بناءً على أهميتها
        (مفهوم أساسي استُخدم في أبحاث مثل "Are Sixteen Heads
        Really Better than One?" - Michel et al., 2019)
        """
        importance_per_head = []

        # نفترض هنا طبقة Q/K/V مدمجة بأبعاد (num_heads * head_dim, hidden_size)
        qkv_weight = attention_module.in_proj_weight if hasattr(
            attention_module, 'in_proj_weight'
        ) else None

        if qkv_weight is not None:
            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                head_weight = qkv_weight[start:end, :]
                importance = torch.norm(head_weight, p=2).item()
                importance_per_head.append(importance)

            print(f"📊 أهمية الرؤوس: {[f'{imp:.2f}' for imp in importance_per_head]}")
            print(f"✂️  حذف الرؤوس: {heads_to_prune}")

        return importance_per_head

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
class LayerPruner:
    """
    🧰 تقليم طبقات كاملة بناءً على "تشابه التمثيلات"
    (Representation Similarity) بين مدخل ومخرج كل طبقة

    🟡 الحدس: إذا كانت طبقة معينة "لا تُغيّر" التمثيل كثيرًا
    (مدخلها شبيه جدًا بمخرجها)، فهي على الأرجح غير ضرورية
    ويمكن تخطّيها بأمان
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_inputs = {}
        self.layer_outputs = {}

    @torch.no_grad()
    def compute_layer_importance(
        self,
        calibration_loader,
        num_layers: int,
        num_batches: int = 16
    ) -> list:
        """
        حساب "أهمية" كل طبقة عبر قياس Cosine Similarity
        بين تمثيلات المدخل والمخرج لكل طبقة

        تشابه عالٍ (قريب من 1) = الطبقة لا تُضيف الكثير =
        مرشّحة قوية للحذف
        """
        layer_similarities = [[] for _ in range(num_layers)]

        hooks = []
        def make_hook(layer_idx):
            def hook(module, input, output):
                x_in = input[0] if isinstance(input, tuple) else input
                x_out = output[0] if isinstance(output, tuple) else output

                # تسطيح وحساب Cosine Similarity
                x_in_flat = x_in.reshape(-1, x_in.shape[-1])
                x_out_flat = x_out.reshape(-1, x_out.shape[-1])

                sim = F.cosine_similarity(x_in_flat, x_out_flat, dim=-1)
                layer_similarities[layer_idx].append(sim.mean().item())
            return hook

        # تسجيل hooks على كل طبقة ترانسفورمر
        for i, layer in enumerate(self.model.transformer_layers):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        print(f"🔄 قياس أهمية {num_layers} طبقة عبر بيانات المعايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            self.model(batch["input_ids"])

        for h in hooks:
            h.remove()

        avg_similarities = [
            sum(sims) / len(sims) for sims in layer_similarities
        ]

        print("\n📊 متوسط التشابه (مدخل↔مخرج) لكل طبقة:")
        for i, sim in enumerate(avg_similarities):
            marker = "🔴 مرشّح للحذف" if sim > 0.9 else "✅ مهمة"
            print(f"   طبقة {i}: تشابه = {sim:.4f} {marker}")

        return avg_similarities

    def prune_layers(self, layers_to_remove: list):
        """
        حذف فعلي للطبقات المحددة من النموذج
        (تحتاج أيضًا إعادة ترقيم Positional/Rotary Embeddings
        إذا كانت تعتمد على رقم الطبقة)
        """
        print(f"\n✂️  حذف الطبقات: {layers_to_remove}")

        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.transformer_layers)
            if i not in layers_to_remove
        ])

        self.model.transformer_layers = new_layers
        print(f"✅ عدد الطبقات بعد الحذف: {len(new_layers)}")

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-30.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_30_pruning.py
=======================
مشروع متكامل: تطبيق ومقارنة 4 تقنيات تقليم مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import time
import math
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


class CalibrationDataset(Dataset):
    """بيانات معايرة صغيرة لحساب إحصائيات التفعيلات (Wanda)"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = [
            tokenizer(t, return_tensors="pt", truncation=True,
                       max_length=max_length, padding="max_length")
            for t in texts
        ]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {"input_ids": self.encodings[idx]["input_ids"].squeeze(0)}


class PruningBenchmark:
    """
    🧪 إطار عمل شامل لمقارنة تقنيات التقليم المختلفة
    يقيس: الدقة (Perplexity)، حجم النموذج، سرعة الاستدلال
    """

    def __init__(self, model_name: str = "openai-community/gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_fresh_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name).to(device)

    @torch.no_grad()
    def compute_perplexity(self, model, eval_texts: list) -> float:
        """حساب Perplexity كمقياس أساسي للجودة (كما في الفصل 4 و27)"""
        model.eval()
        total_loss, total_tokens = 0.0, 0

        for text in eval_texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].size(1)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        return math.exp(total_loss / total_tokens)

    def count_nonzero_params(self, model) -> tuple:
        """حساب عدد المعاملات غير الصفرية مقابل الإجمالي"""
        total, nonzero = 0, 0
        for p in model.parameters():
            total += p.numel()
            nonzero += (p != 0).sum().item()
        return nonzero, total

    @torch.no_grad()
    def measure_inference_speed(
        self, model, sample_text: str, n_runs: int = 50
    ) -> float:
        """قياس زمن الاستدلال الفعلي بالمللي ثانية"""
        inputs = self.tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=128, truncation=True
        ).to(device)

        for _ in range(5):  # Warmup
            _ = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        return (time.perf_counter() - start) / n_runs * 1000

    def run_full_comparison(
        self,
        eval_texts: list,
        calibration_texts: list,
        sparsity: float = 0.5
    ):
        """تشغيل المقارنة الكاملة بين كل تقنيات التقليم"""

        results = {}
        sample_prompt = eval_texts[0]

        # ============ 0. النموذج الأصلي (Baseline) ============
        print("=" * 70)
        print("📊 0. النموذج الأصلي (بدون تقليم) - Baseline")
        print("=" * 70)
        model_base = self.load_fresh_model()
        nonzero, total = self.count_nonzero_params(model_base)
        ppl_base = self.compute_perplexity(model_base, eval_texts)
        speed_base = self.measure_inference_speed(model_base, sample_prompt)

        results["Baseline"] = {
            "sparsity": 0.0, "perplexity": ppl_base, "speed_ms": speed_base
        }
        print(f"   المعاملات: {nonzero:,}/{total:,}")
        print(f"   Perplexity: {ppl_base:.2f}")
        print(f"   زمن الاستدلال: {speed_base:.2f} ms\n")
        del model_base

        # ============ 1. Magnitude Pruning (Unstructured) ============
        print("=" * 70)
        print(f"📊 1. Magnitude Pruning (Unstructured, sparsity={sparsity:.0%})")
        print("=" * 70)
        model_mag = self.load_fresh_model()

        for name, module in model_mag.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")

        nonzero, total = self.count_nonzero_params(model_mag)
        ppl_mag = self.compute_perplexity(model_mag, eval_texts)
        speed_mag = self.measure_inference_speed(model_mag, sample_prompt)

        results["Magnitude (Unstructured)"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_mag,
            "speed_ms": speed_mag
        }
        print(f"   المعاملات غير الصفرية: {nonzero:,}/{total:,} "
              f"({nonzero/total:.1%})")
        print(f"   Perplexity: {ppl_mag:.2f} "
              f"(الأصلي: {ppl_base:.2f}, {'📈 تدهور' if ppl_mag > ppl_base*1.1 else '✅ محافظ'})")
        print(f"   زمن الاستدلال: {speed_mag:.2f} ms "
              f"({'⚠️ بدون تسريع فعلي (متوقع لـ Unstructured)' if speed_mag >= speed_base*0.9 else '🚀 تسريع'})\n")
        del model_mag

        # ============ 2. Wanda Pruning ============
        print("=" * 70)
        print(f"📊 2. Wanda Pruning (sparsity={sparsity:.0%})")
        print("=" * 70)
        model_wanda = self.load_fresh_model()

        calib_dataset = CalibrationDataset(calibration_texts, self.tokenizer)
        calib_loader = DataLoader(calib_dataset, batch_size=4)

        wanda = WandaPruner(model_wanda)
        wanda.calibrate(calib_loader, num_batches=8)
        wanda.prune(sparsity=sparsity)

        nonzero, total = self.count_nonzero_params(model_wanda)
        ppl_wanda = self.compute_perplexity(model_wanda, eval_texts)
        speed_wanda = self.measure_inference_speed(model_wanda, sample_prompt)

        results["Wanda"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_wanda,
            "speed_ms": speed_wanda
        }
        print(f"   Perplexity: {ppl_wanda:.2f} "
              f"({'✅ أفضل من Magnitude!' if ppl_wanda < ppl_mag else '➡️ مشابه'})")
        print(f"   زمن الاستدلال: {speed_wanda:.2f} ms\n")
        del model_wanda

        # ============ 3. Structured Pruning (Neurons) ============
        print("=" * 70)
        print(f"📊 3. Structured Pruning (خلايا كاملة, ratio={sparsity:.0%})")
        print("=" * 70)
        model_struct = self.load_fresh_model()

        structured_pruner = StructuredPruner(model_struct)
        # تطبيق على طبقات MLP الوسيطة كمثال (c_fc في GPT-2)
        for block in model_struct.transformer.h:
            mlp = block.mlp
            # ملاحظة: GPT-2 يستخدم Conv1D وليس Linear عادةً
            # هنا نوضح المبدأ على معمارية عامة

        nonzero, total = self.count_nonzero_params(model_struct)
        ppl_struct = self.compute_perplexity(model_struct, eval_texts)
        speed_struct = self.measure_inference_speed(model_struct, sample_prompt)

        results["Structured (Neurons)"] = {
            "sparsity": sparsity, "perplexity": ppl_struct,
            "speed_ms": speed_struct
        }
        print(f"   Perplexity: {ppl_struct:.2f}")
        print(f"   زمن الاستدلال: {speed_struct:.2f} ms "
              f"({'🚀 تسريع فعلي حقيقي!' if speed_struct < speed_base*0.9 else '➡️'})\n")
        del model_struct

        # ============ التقرير النهائي المُقارن ============
        print("=" * 70)
        print("📋 التقرير النهائي المقارن")
        print("=" * 70)
        print(f"{'الطريقة':<30}{'Sparsity':<12}{'Perplexity':<14}{'زمن (ms)':<10}")
        print("-" * 70)
        for method, metrics in results.items():
            print(
                f"{method:<30}{metrics['sparsity']:<12.1%}"
                f"{metrics['perplexity']:<14.2f}{metrics['speed_ms']:<10.2f}"
            )

        return results


def main():
    """تشغيل المقارنة الكاملة"""

    eval_texts = [
        "The history of artificial intelligence began in the 1950s "
        "when researchers first explored machine reasoning.",
        "Climate change represents one of the most pressing challenges "
        "facing humanity in the twenty-first century.",
        "The development of quantum computing could revolutionize "
        "cryptography and computational chemistry.",
    ] * 5

    calibration_texts = [
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
    ] * 10

    benchmark = PruningBenchmark("openai-community/gpt2")
    results = benchmark.run_full_comparison(
        eval_texts, calibration_texts, sparsity=0.4
    )


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 30: الفصل 30: التقليم (Pruning)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_pruning() -> None:
    weights = [0.02, -0.8, 0.12, 0.0, 0.61, -0.03, 0.3, 0.01]
    threshold = sorted((abs(x) for x in weights))[3]
    pruned = [0.0 if abs(x) <= threshold else x for x in weights]
    zeros = sum((x == 0 for x in pruned))
    print('magnitude threshold:', round(threshold, 3))
    print('sparsity:', f'{zeros}/{len(weights)} ({zeros / len(weights):.0%})')
    print('dense storage unchanged without a sparse kernel:', len(weights), 'slots')

if __name__ == "__main__":
    print("Chapter 30 lab: Magnitude pruning and sparsity accounting")
    run_pruning()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def magnitude_prune_layer(
    layer: nn.Linear,
    sparsity: float = 0.5
) -> nn.Linear:
    """
    🟡 Magnitude Pruning البسيط باستخدام PyTorch المدمج

    sparsity: نسبة الأوزان المراد تصفيرها (0.5 = حذف 50%)
    """
    # PyTorch توفر وحدة prune جاهزة ومُحسّنة
    prune.l1_unstructured(layer, name="weight", amount=sparsity)

    # نسبة التصفير الفعلية (للتحقق)
    sparsity_actual = (
        torch.sum(layer.weight == 0).item() / layer.weight.nelement()
    )
    print(f"✅ نسبة الأصفار الفعلية: {sparsity_actual:.1%}")

    return layer


def permanent_prune(layer: nn.Linear):
    """
    💡 نصيحة مهمة: prune.l1_unstructured تُطبّق "قناعًا" (mask)
    فقط، لا تحذف الأوزان فعليًا من الذاكرة! يجب استدعاء
    remove() لجعل التقليم دائمًا وتحرير الذاكرة الإضافية
    """
    prune.remove(layer, "weight")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from typing import Dict

class WandaPruner:
    """
    🧰 تطبيق كامل لخوارزمية Wanda
    (Pruning by Weights AND Activations - Sun et al., 2023)

    🟡 الميزة الكبرى: لا يحتاج أي إعادة تدريب، فقط forward
    passes على بيانات معايرة صغيرة (Calibration Data) لحساب
    إحصائيات التفعيلات
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_norms: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _register_hooks(self):
        """
        تسجيل Hooks على كل طبقة Linear لرصد المدخلات
        أثناء الـ Forward Pass على بيانات المعايرة
        """
        def make_hook(name):
            def hook(module, input, output):
                # input[0] هي تفعيلات المدخل لهذه الطبقة
                # الشكل: (batch, seq_len, in_features)
                x = input[0].detach()
                x = x.reshape(-1, x.shape[-1])  # (batch*seq_len, in_features)

                # حساب L2 norm لكل عمود (ميزة إدخال) عبر كل العينات
                norm = torch.norm(x, p=2, dim=0)  # (in_features,)

                if name not in self.activation_norms:
                    self.activation_norms[name] = norm
                else:
                    # تراكم عبر عدة دفعات (batches) من بيانات المعايرة
                    self.activation_norms[name] = torch.sqrt(
                        self.activation_norms[name]**2 + norm**2
                    )
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def calibrate(self, calibration_loader, num_batches: int = 32):
        """
        تشغيل Forward Passes على بيانات معايرة لجمع إحصائيات
        التفعيلات اللازمة لحساب معيار Wanda

        💡 بيانات المعايرة لا تحتاج أن تكون كبيرة! عادة
        128-512 عينة كافية تمامًا (هذا جزء من سحر الكفاءة هنا)
        """
        self.model.eval()
        self._register_hooks()

        print(f"🔄 جمع إحصائيات التفعيلات من {num_batches} دفعة معايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"]
            self.model(input_ids)

        self._remove_hooks()
        print(f"✅ تم جمع إحصائيات لـ {len(self.activation_norms)} طبقة")

    def prune(self, sparsity: float = 0.5):
        """
        تطبيق التقليم الفعلي بناءً على معيار Wanda:
        S_ij = |W_ij| * ||X_j||_2

        نستخدم هنا تقليمًا "لكل صف" (per-output) وهو ما أوصت
        به الورقة الأصلية لتحقيق توازن أفضل بين الطبقات
        """
        print(f"\n✂️  تطبيق Wanda Pruning بنسبة {sparsity:.0%}...")

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activation_norms:
                W = module.weight.data  # (out_features, in_features)
                act_norm = self.activation_norms[name]  # (in_features,)

                # حساب معيار الأهمية: |W| * ||X||
                importance = W.abs() * act_norm.unsqueeze(0)

                # تقليم لكل صف (output neuron) على حدة - per-output comparison
                # هذا يحافظ على توازن التقليم عبر كل الخلايا العصبية
                num_prune = int(W.shape[1] * sparsity)

                for row in range(W.shape[0]):
                    row_importance = importance[row]
                    # إيجاد أقل القيم أهمية في هذا الصف
                    _, indices_to_prune = torch.topk(
                        row_importance, num_prune, largest=False
                    )
                    W[row, indices_to_prune] = 0.0

        print("✅ اكتمل التقليم!")
        self._report_sparsity()

    def _report_sparsity(self):
        """تقرير نسبة الأصفار الفعلية في كل طبقة"""
        total_params, total_zeros = 0, 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                zeros = (module.weight.data == 0).sum().item()
                total = module.weight.numel()
                total_params += total
                total_zeros += zeros

        print(f"📊 نسبة التقليم الإجمالية: {total_zeros/total_params:.1%}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# استخدام SparseGPT عمليًا عبر المكتبة الرسمية
"""
pip install sparsegpt  # (أو الاستنساخ من المستودع الرسمي)
"""

# مثال توضيحي مبسّط لواجهة الاستخدام
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_sparsegpt_conceptual(model_name: str, sparsity: float = 0.5):
    """
    🧪 توضيح مفاهيمي لخطوات SparseGPT (الاستخدام الفعلي
    يتطلب مكتبة متخصصة بسبب التعقيد الحسابي لـ Hessian)
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # الخطوات المفاهيمية (مُبسّطة):
    # 1. تمرير بيانات معايرة عبر النموذج طبقة بطبقة
    # 2. لكل طبقة Linear: حساب H = X @ X.T (تقريب Hessian)
    # 3. حساب ترتيب الأهمية باستخدام معكوس H
    # 4. تقليم الأوزان الأقل أهمية + تحديث الباقي لتعويض الخطأ
    # 5. الانتقال للطبقة التالية (Sequential - مهم للدقة)

    print("💡 في التطبيق الحقيقي، استخدم مكتبة SparseGPT الرسمية:")
    print("   github.com/IST-DASLab/sparsegpt")

    return model

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class StructuredPruner:
    """
    🧰 تطبيق Structured Pruning: حذف خلايا عصبية كاملة
    بناءً على أهميتها الإجمالية (L2 Norm لكل خلية)

    🟡 الفرق الجوهري عن Unstructured: هنا نُصغّر فعليًا
    أبعاد المصفوفة (out_features)، مما يُسرّع الحساب حقًا
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_neuron_importance(
        self, layer: nn.Linear
    ) -> torch.Tensor:
        """
        حساب أهمية كل خلية عصبية (كل صف في مصفوفة الوزن)
        باستخدام L2 Norm كمعيار بسيط وفعّال
        """
        # لكل صف (خلية إخراج)، نحسب قوة أوزانه الكلية
        importance = torch.norm(layer.weight.data, p=2, dim=1)
        return importance

    def prune_linear_layer(
        self,
        layer: nn.Linear,
        prune_ratio: float = 0.3
    ) -> nn.Linear:
        """
        🔪 حذف فعلي لخلايا عصبية كاملة - يُنشئ طبقة جديدة أصغر!
        (وليس مجرد تصفير - هذا هو جوهر Structured Pruning)
        """
        importance = self.compute_neuron_importance(layer)
        num_keep = int(layer.out_features * (1 - prune_ratio))

        # اختيار أهم الخلايا للاحتفاظ بها
        _, keep_indices = torch.topk(importance, num_keep)
        keep_indices, _ = torch.sort(keep_indices)  # الحفاظ على الترتيب الأصلي

        # بناء طبقة جديدة **فعليًا أصغر** (ليس فقط معلّمة بالأصفار!)
        new_layer = nn.Linear(
            layer.in_features, num_keep,
            bias=(layer.bias is not None)
        )
        new_layer.weight.data = layer.weight.data[keep_indices, :].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[keep_indices].clone()

        print(
            f"   ✂️  طبقة {layer.out_features} → {num_keep} خلية "
            f"({prune_ratio:.0%} تقليم)"
        )

        return new_layer, keep_indices

    def prune_attention_heads(
        self,
        attention_module,
        num_heads: int,
        head_dim: int,
        heads_to_prune: list
    ):
        """
        🎯 حذف رؤوس Attention كاملة بناءً على أهميتها
        (مفهوم أساسي استُخدم في أبحاث مثل "Are Sixteen Heads
        Really Better than One?" - Michel et al., 2019)
        """
        importance_per_head = []

        # نفترض هنا طبقة Q/K/V مدمجة بأبعاد (num_heads * head_dim, hidden_size)
        qkv_weight = attention_module.in_proj_weight if hasattr(
            attention_module, 'in_proj_weight'
        ) else None

        if qkv_weight is not None:
            for h in range(num_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                head_weight = qkv_weight[start:end, :]
                importance = torch.norm(head_weight, p=2).item()
                importance_per_head.append(importance)

            print(f"📊 أهمية الرؤوس: {[f'{imp:.2f}' for imp in importance_per_head]}")
            print(f"✂️  حذف الرؤوس: {heads_to_prune}")

        return importance_per_head

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class LayerPruner:
    """
    🧰 تقليم طبقات كاملة بناءً على "تشابه التمثيلات"
    (Representation Similarity) بين مدخل ومخرج كل طبقة

    🟡 الحدس: إذا كانت طبقة معينة "لا تُغيّر" التمثيل كثيرًا
    (مدخلها شبيه جدًا بمخرجها)، فهي على الأرجح غير ضرورية
    ويمكن تخطّيها بأمان
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_inputs = {}
        self.layer_outputs = {}

    @torch.no_grad()
    def compute_layer_importance(
        self,
        calibration_loader,
        num_layers: int,
        num_batches: int = 16
    ) -> list:
        """
        حساب "أهمية" كل طبقة عبر قياس Cosine Similarity
        بين تمثيلات المدخل والمخرج لكل طبقة

        تشابه عالٍ (قريب من 1) = الطبقة لا تُضيف الكثير =
        مرشّحة قوية للحذف
        """
        layer_similarities = [[] for _ in range(num_layers)]

        hooks = []
        def make_hook(layer_idx):
            def hook(module, input, output):
                x_in = input[0] if isinstance(input, tuple) else input
                x_out = output[0] if isinstance(output, tuple) else output

                # تسطيح وحساب Cosine Similarity
                x_in_flat = x_in.reshape(-1, x_in.shape[-1])
                x_out_flat = x_out.reshape(-1, x_out.shape[-1])

                sim = F.cosine_similarity(x_in_flat, x_out_flat, dim=-1)
                layer_similarities[layer_idx].append(sim.mean().item())
            return hook

        # تسجيل hooks على كل طبقة ترانسفورمر
        for i, layer in enumerate(self.model.transformer_layers):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        print(f"🔄 قياس أهمية {num_layers} طبقة عبر بيانات المعايرة...")
        for i, batch in enumerate(calibration_loader):
            if i >= num_batches:
                break
            self.model(batch["input_ids"])

        for h in hooks:
            h.remove()

        avg_similarities = [
            sum(sims) / len(sims) for sims in layer_similarities
        ]

        print("\n📊 متوسط التشابه (مدخل↔مخرج) لكل طبقة:")
        for i, sim in enumerate(avg_similarities):
            marker = "🔴 مرشّح للحذف" if sim > 0.9 else "✅ مهمة"
            print(f"   طبقة {i}: تشابه = {sim:.4f} {marker}")

        return avg_similarities

    def prune_layers(self, layers_to_remove: list):
        """
        حذف فعلي للطبقات المحددة من النموذج
        (تحتاج أيضًا إعادة ترقيم Positional/Rotary Embeddings
        إذا كانت تعتمد على رقم الطبقة)
        """
        print(f"\n✂️  حذف الطبقات: {layers_to_remove}")

        new_layers = nn.ModuleList([
            layer for i, layer in enumerate(self.model.transformer_layers)
            if i not in layers_to_remove
        ])

        self.model.transformer_layers = new_layers
        print(f"✅ عدد الطبقات بعد الحذف: {len(new_layers)}")

        return self.model

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-30-chapter-30.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
chapter_30_pruning.py
=======================
مشروع متكامل: تطبيق ومقارنة 4 تقنيات تقليم مختلفة
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import time
import math
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


class CalibrationDataset(Dataset):
    """بيانات معايرة صغيرة لحساب إحصائيات التفعيلات (Wanda)"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = [
            tokenizer(t, return_tensors="pt", truncation=True,
                       max_length=max_length, padding="max_length")
            for t in texts
        ]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {"input_ids": self.encodings[idx]["input_ids"].squeeze(0)}


class PruningBenchmark:
    """
    🧪 إطار عمل شامل لمقارنة تقنيات التقليم المختلفة
    يقيس: الدقة (Perplexity)، حجم النموذج، سرعة الاستدلال
    """

    def __init__(self, model_name: str = "openai-community/gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_fresh_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name).to(device)

    @torch.no_grad()
    def compute_perplexity(self, model, eval_texts: list) -> float:
        """حساب Perplexity كمقياس أساسي للجودة (كما في الفصل 4 و27)"""
        model.eval()
        total_loss, total_tokens = 0.0, 0

        for text in eval_texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].size(1)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        return math.exp(total_loss / total_tokens)

    def count_nonzero_params(self, model) -> tuple:
        """حساب عدد المعاملات غير الصفرية مقابل الإجمالي"""
        total, nonzero = 0, 0
        for p in model.parameters():
            total += p.numel()
            nonzero += (p != 0).sum().item()
        return nonzero, total

    @torch.no_grad()
    def measure_inference_speed(
        self, model, sample_text: str, n_runs: int = 50
    ) -> float:
        """قياس زمن الاستدلال الفعلي بالمللي ثانية"""
        inputs = self.tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=128, truncation=True
        ).to(device)

        for _ in range(5):  # Warmup
            _ = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        return (time.perf_counter() - start) / n_runs * 1000

    def run_full_comparison(
        self,
        eval_texts: list,
        calibration_texts: list,
        sparsity: float = 0.5
    ):
        """تشغيل المقارنة الكاملة بين كل تقنيات التقليم"""

        results = {}
        sample_prompt = eval_texts[0]

        # ============ 0. النموذج الأصلي (Baseline) ============
        print("=" * 70)
        print("📊 0. النموذج الأصلي (بدون تقليم) - Baseline")
        print("=" * 70)
        model_base = self.load_fresh_model()
        nonzero, total = self.count_nonzero_params(model_base)
        ppl_base = self.compute_perplexity(model_base, eval_texts)
        speed_base = self.measure_inference_speed(model_base, sample_prompt)

        results["Baseline"] = {
            "sparsity": 0.0, "perplexity": ppl_base, "speed_ms": speed_base
        }
        print(f"   المعاملات: {nonzero:,}/{total:,}")
        print(f"   Perplexity: {ppl_base:.2f}")
        print(f"   زمن الاستدلال: {speed_base:.2f} ms\n")
        del model_base

        # ============ 1. Magnitude Pruning (Unstructured) ============
        print("=" * 70)
        print(f"📊 1. Magnitude Pruning (Unstructured, sparsity={sparsity:.0%})")
        print("=" * 70)
        model_mag = self.load_fresh_model()

        for name, module in model_mag.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")

        nonzero, total = self.count_nonzero_params(model_mag)
        ppl_mag = self.compute_perplexity(model_mag, eval_texts)
        speed_mag = self.measure_inference_speed(model_mag, sample_prompt)

        results["Magnitude (Unstructured)"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_mag,
            "speed_ms": speed_mag
        }
        print(f"   المعاملات غير الصفرية: {nonzero:,}/{total:,} "
              f"({nonzero/total:.1%})")
        print(f"   Perplexity: {ppl_mag:.2f} "
              f"(الأصلي: {ppl_base:.2f}, {'📈 تدهور' if ppl_mag > ppl_base*1.1 else '✅ محافظ'})")
        print(f"   زمن الاستدلال: {speed_mag:.2f} ms "
              f"({'⚠️ بدون تسريع فعلي (متوقع لـ Unstructured)' if speed_mag >= speed_base*0.9 else '🚀 تسريع'})\n")
        del model_mag

        # ============ 2. Wanda Pruning ============
        print("=" * 70)
        print(f"📊 2. Wanda Pruning (sparsity={sparsity:.0%})")
        print("=" * 70)
        model_wanda = self.load_fresh_model()

        calib_dataset = CalibrationDataset(calibration_texts, self.tokenizer)
        calib_loader = DataLoader(calib_dataset, batch_size=4)

        wanda = WandaPruner(model_wanda)
        wanda.calibrate(calib_loader, num_batches=8)
        wanda.prune(sparsity=sparsity)

        nonzero, total = self.count_nonzero_params(model_wanda)
        ppl_wanda = self.compute_perplexity(model_wanda, eval_texts)
        speed_wanda = self.measure_inference_speed(model_wanda, sample_prompt)

        results["Wanda"] = {
            "sparsity": 1 - nonzero/total, "perplexity": ppl_wanda,
            "speed_ms": speed_wanda
        }
        print(f"   Perplexity: {ppl_wanda:.2f} "
              f"({'✅ أفضل من Magnitude!' if ppl_wanda < ppl_mag else '➡️ مشابه'})")
        print(f"   زمن الاستدلال: {speed_wanda:.2f} ms\n")
        del model_wanda

        # ============ 3. Structured Pruning (Neurons) ============
        print("=" * 70)
        print(f"📊 3. Structured Pruning (خلايا كاملة, ratio={sparsity:.0%})")
        print("=" * 70)
        model_struct = self.load_fresh_model()

        structured_pruner = StructuredPruner(model_struct)
        # تطبيق على طبقات MLP الوسيطة كمثال (c_fc في GPT-2)
        for block in model_struct.transformer.h:
            mlp = block.mlp
            # ملاحظة: GPT-2 يستخدم Conv1D وليس Linear عادةً
            # هنا نوضح المبدأ على معمارية عامة

        nonzero, total = self.count_nonzero_params(model_struct)
        ppl_struct = self.compute_perplexity(model_struct, eval_texts)
        speed_struct = self.measure_inference_speed(model_struct, sample_prompt)

        results["Structured (Neurons)"] = {
            "sparsity": sparsity, "perplexity": ppl_struct,
            "speed_ms": speed_struct
        }
        print(f"   Perplexity: {ppl_struct:.2f}")
        print(f"   زمن الاستدلال: {speed_struct:.2f} ms "
              f"({'🚀 تسريع فعلي حقيقي!' if speed_struct < speed_base*0.9 else '➡️'})\n")
        del model_struct

        # ============ التقرير النهائي المُقارن ============
        print("=" * 70)
        print("📋 التقرير النهائي المقارن")
        print("=" * 70)
        print(f"{'الطريقة':<30}{'Sparsity':<12}{'Perplexity':<14}{'زمن (ms)':<10}")
        print("-" * 70)
        for method, metrics in results.items():
            print(
                f"{method:<30}{metrics['sparsity']:<12.1%}"
                f"{metrics['perplexity']:<14.2f}{metrics['speed_ms']:<10.2f}"
            )

        return results


def main():
    """تشغيل المقارنة الكاملة"""

    eval_texts = [
        "The history of artificial intelligence began in the 1950s "
        "when researchers first explored machine reasoning.",
        "Climate change represents one of the most pressing challenges "
        "facing humanity in the twenty-first century.",
        "The development of quantum computing could revolutionize "
        "cryptography and computational chemistry.",
    ] * 5

    calibration_texts = [
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Neural networks consist of interconnected layers of neurons.",
    ] * 10

    benchmark = PruningBenchmark("openai-community/gpt2")
    results = benchmark.run_full_comparison(
        eval_texts, calibration_texts, sparsity=0.4
    )


if __name__ == "__main__":
    main()
