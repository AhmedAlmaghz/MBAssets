#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تجميع معماري (Architecture Stitching)
Large Language Model Engineering — Architecture

Book slug : llm-engineering-69-chapters
Chapter   : ch-31-chapter-31 (lesson 31)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-31-chapter-31.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تجميع معماري (Architecture Stitching)
Large Language Model Engineering — Architecture

Book slug : llm-engineering-69-chapters
Chapter   : ch-31-chapter-31 (lesson 31)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-31-chapter-31.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تجميع معماري (Architecture Stitching)
Large Language Model Engineering — Architecture

Book slug : llm-engineering-69-chapters
Chapter   : ch-31-chapter-31 (lesson 31)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 31: الفصل 31: تجميع معماري (Architecture Stitching)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_stitching() -> None:
    layers = [{'name': 'block_a', 'width': 4, 'out': 4}, {'name': 'block_b', 'width': 4, 'out': 4}]
    compatible = all((left['out'] == right['width'] for left, right in zip(layers, layers[1:])))
    stitched_depth = sum((1 for layer in layers if layer['width'] == layer['out']))
    print('adjacent dimensions compatible:', compatible)
    print('stitched blocks:', stitched_depth)
    print('final representation width:', layers[-1]['out'])

if __name__ == "__main__":
    print("Chapter 31 lab: Check shape compatibility when stitching layers")
    run_stitching()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_31_depth_upscaling.py
================================
مشروع متكامل: تطبيق Depth Upscaling (أسلوب SOLAR)
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel
)
from typing import List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)


class DepthUpscaler:
    """
    🧰 أداة شاملة لتطبيق Depth Upscaling (أسلوب SOLAR)
    على أي نموذج Transformer قائم على HuggingFace

    🟡 يدعم أي نسبة تكبير، وليس فقط حالة SOLAR المحددة (32→48)
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_transformer_layers(self, model):
        """
        استخراج قائمة طبقات Transformer من النموذج
        (يختلف اسم الخاصية حسب المعمارية: h في GPT-2،
        layers في LLaMA، إلخ)
        """
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h  # GPT-2 style
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers  # LLaMA/Mistral style
        else:
            raise ValueError("❌ معمارية غير مدعومة - تحقق من اسم خاصية الطبقات")

    def _set_transformer_layers(self, model, new_layers: nn.ModuleList):
        """إعادة تعيين قائمة الطبقات الجديدة في النموذج"""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model.transformer.h = new_layers
            model.config.n_layer = len(new_layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = new_layers
            model.config.num_hidden_layers = len(new_layers)

    def upscale(
        self,
        target_num_layers: int,
        verbose: bool = True
    ):
        """
        🏗️ العملية الكاملة لـ Depth Upscaling

        target_num_layers: العدد النهائي المطلوب من الطبقات
        (يجب أن يكون بين L و 2L حيث L هو عدد طبقات النموذج الأصلي)
        """
        print(f"📥 تحميل نسختين من النموذج الأساسي: {self.base_model_name}")
        model_a = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        model_b = AutoModelForCausalLM.from_pretrained(self.base_model_name)

        original_layers = self._get_transformer_layers(model_a)
        L = len(original_layers)

        if not (L < target_num_layers <= 2 * L):
            raise ValueError(
                f"❌ target_num_layers يجب أن يكون بين {L+1} و {2*L} "
                f"(النموذج الأصلي له {L} طبقة)"
            )

        # حساب عدد الطبقات المحذوفة من كل نسخة
        m = (2 * L - target_num_layers) // 2

        if verbose:
            print(f"\n📐 حسابات Depth Upscaling:")
            print(f"   عدد طبقات النموذج الأصلي (L): {L}")
            print(f"   عدد الطبقات المستهدف: {target_num_layers}")
            print(f"   عدد الطبقات المحذوفة من كل نسخة (m): {m}")

        layers_a = self._get_transformer_layers(model_a)
        layers_b = self._get_transformer_layers(model_b)

        # الاحتفاظ بالطبقات الأولى من A (حذف آخر m طبقة)
        kept_from_a = list(layers_a[:L - m])
        # الاحتفاظ بالطبقات الأخيرة من B (حذف أول m طبقة)
        kept_from_b = list(layers_b[m:])

        if verbose:
            print(f"\n✂️  من النسخة A: احتفظنا بالطبقات [0:{L-m}] "
                  f"({len(kept_from_a)} طبقة)")
            print(f"✂️  من النسخة B: احتفظنا بالطبقات [{m}:{L}] "
                  f"({len(kept_from_b)} طبقة)")

        # الخياطة: دمج القائمتين معًا بالترتيب
        stitched_layers = nn.ModuleList(kept_from_a + kept_from_b)

        if verbose:
            print(f"\n🧵 النموذج النهائي: {len(stitched_layers)} طبقة "
                  f"(بدلًا من {L} أصلًا)")

        # بناء النموذج النهائي باستخدام هيكل model_a مع الطبقات الجديدة
        self._set_transformer_layers(model_a, stitched_layers)

        # تحديث أي embeddings/buffers مرتبطة بعدد الطبقات
        # (مثلًا بعض النماذج تحتاج تحديث position_ids إذا كانت مرتبطة بعمق الشبكة)
        model_a.tie_weights()

        upscaled_params = sum(p.numel() for p in model_a.parameters())
        original_params = sum(p.numel() for p in
                                AutoModelForCausalLM.from_pretrained(
                                    self.base_model_name
                                ).parameters())

        if verbose:
            print(f"\n📊 عدد المعاملات:")
            print(f"   النموذج الأصلي: {original_params:,}")
            print(f"   النموذج بعد التكبير: {upscaled_params:,}")
            print(f"   نسبة الزيادة: {upscaled_params/original_params:.2f}x")

        del model_b  # تحرير الذاكرة
        return model_a.to(device)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
def sanity_check_upscaled_model(
    model, tokenizer, original_model_name: str, test_prompt: str
):
    """
    🧪 فحص سريع للتأكد أن النموذج المُكبَّر لا يزال "متماسكًا"
    قبل الاستثمار في Continued Pretraining
    """
    print("\n" + "=" * 70)
    print("🔍 فحص السلامة (Sanity Check) للنموذج المُكبَّر")
    print("=" * 70)

    # تحميل النموذج الأصلي للمقارنة
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name
    ).to(device)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"\n📝 المطالبة: '{test_prompt}'")

    with torch.no_grad():
        # توليد من النموذج الأصلي
        output_orig = original_model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)

        # توليد من النموذج المُكبَّر
        output_upscaled = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_upscaled = tokenizer.decode(
            output_upscaled[0], skip_special_tokens=True
        )

        # حساب Perplexity لكليهما على نفس النص
        ppl_orig = compute_perplexity_single(original_model, tokenizer, test_prompt)
        ppl_upscaled = compute_perplexity_single(model, tokenizer, test_prompt)

    print(f"\n🔵 النموذج الأصلي:")
    print(f"   النص: {text_orig}")
    print(f"   Perplexity: {ppl_orig:.2f}")

    print(f"\n🟢 النموذج المُكبَّر (قبل أي تدريب إضافي!):")
    print(f"   النص: {text_upscaled}")
    print(f"   Perplexity: {ppl_upscaled:.2f}")

    ratio = ppl_upscaled / ppl_orig
    print(f"\n📊 نسبة Perplexity (مُكبَّر/أصلي): {ratio:.2f}x")

    if ratio < 5.0:
        print("✅ ممتاز! النموذج المُكبَّر لا يزال 'متماسكًا' بشكل معقول")
        print("   رغم عدم أي تدريب إضافي - جاهز لـ Continued Pretraining")
    else:
        print("⚠️  تدهور كبير - قد تحتاج مراجعة نقطة الخياطة أو نسبة m")

    del original_model
    return ratio


def compute_perplexity_single(model, tokenizer, text: str) -> float:
    """حساب Perplexity لنص واحد"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة Depth Upscaling الكاملة End-to-End"""

    BASE_MODEL = "openai-community/gpt2"  # 12 طبقة أصلًا

    upscaler = DepthUpscaler(BASE_MODEL)

    # تكبير من 12 طبقة إلى 18 طبقة (نفس نسبة SOLAR تقريبًا: 32→48)
    upscaled_model = upscaler.upscale(target_num_layers=18, verbose=True)

    # فحص السلامة قبل أي تدريب إضافي
    test_prompt = "The history of artificial intelligence began"
    sanity_check_upscaled_model(
        upscaled_model, upscaler.tokenizer, BASE_MODEL, test_prompt
    )

    print("\n" + "=" * 70)
    print("📌 الخطوة التالية في مشروع حقيقي:")
    print("=" * 70)
    print("""
    الآن بعد التحقق من سلامة النموذج المُكبَّر، الخطوة التالية
    في مشروع إنتاجي حقيقي هي تطبيق Continued Pretraining
    (كما تعلمنا بالتفصيل الكامل في الفصل 27!) على كمية معتدلة
    من البيانات (عادة أقل بكثير من Pretraining الأصلي -
    فريق SOLAR استخدم حوالي 1-2% فقط من حجم بيانات
    Pretraining الكامل) لـ:

    1. 'تنعيم' خط الخياطة بين الأجزاء المختلفة
    2. السماح للطبقات المكررة بالتمايز والتخصص تدريجيًا
    3. استعادة وتحسين الأداء العام للنموذج الأعمق الجديد
    """)

    # حفظ النموذج المُكبَّر للاستخدام في Continued Pretraining
    upscaled_model.save_pretrained("./solar_style_upscaled_model")
    upscaler.tokenizer.save_pretrained("./solar_style_upscaled_model")
    print("\n💾 تم حفظ النموذج المُكبَّر، جاهز لخطوة Continued Pretraining")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
class ModelGrafter:
    """
    🧰 أداة لـ Model Grafting: دمج طبقات من نماذج مختلفة
    (وليس بالضرورة نسخًا من نفس النموذج كما في Depth Upscaling)

    🟡 الاستخدام الأشهر: أخذ الطبقات الأولى من نموذج "عام قوي"
    (معالجة لغوية أساسية ممتازة) ودمجها مع الطبقات الأخيرة
    من نموذج "متخصص" (قرارات نهائية مُحسَّنة لنطاق معين)
    """

    def __init__(self):
        pass

    def graft(
        self,
        model_front: nn.Module,     # مصدر الطبقات الأولى
        model_back: nn.Module,      # مصدر الطبقات الأخيرة
        split_point_front: int,     # كم طبقة نأخذ من model_front
        split_point_back: int,      # من أي طبقة نبدأ الأخذ من model_back
    ):
        """
        ⚠️ شرط أساسي: يجب أن يكون hidden_size متطابقًا بين
        النموذجين، وإلا فإن الخياطة مستحيلة رياضيًا بدون
        طبقة إسقاط (Projection Layer) إضافية بين الجزأين
        """
        layers_front = self._get_layers(model_front)
        layers_back = self._get_layers(model_back)

        # التحقق من التوافق البنيوي
        front_hidden = model_front.config.hidden_size if hasattr(
            model_front.config, 'hidden_size'
        ) else model_front.config.n_embd
        back_hidden = model_back.config.hidden_size if hasattr(
            model_back.config, 'hidden_size'
        ) else model_back.config.n_embd

        if front_hidden != back_hidden:
            raise ValueError(
                f"❌ أبعاد غير متوافقة: {front_hidden} != {back_hidden}\n"
                f"   يلزم إضافة طبقة Projection للتوافق - راجع صندوق التعمّق أدناه"
            )

        grafted = nn.ModuleList(
            list(layers_front[:split_point_front]) +
            list(layers_back[split_point_back:])
        )

        print(f"🧵 Grafting ناجح: "
              f"{split_point_front} طبقة من النموذج الأول + "
              f"{len(layers_back) - split_point_back} طبقة من النموذج الثاني "
              f"= {len(grafted)} طبقة إجمالًا")

        return grafted

    def _get_layers(self, model):
        if hasattr(model, "transformer"):
            return model.transformer.h
        return model.model.layers

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 31: الفصل 31: تجميع معماري (Architecture Stitching)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_stitching() -> None:
    layers = [{'name': 'block_a', 'width': 4, 'out': 4}, {'name': 'block_b', 'width': 4, 'out': 4}]
    compatible = all((left['out'] == right['width'] for left, right in zip(layers, layers[1:])))
    stitched_depth = sum((1 for layer in layers if layer['width'] == layer['out']))
    print('adjacent dimensions compatible:', compatible)
    print('stitched blocks:', stitched_depth)
    print('final representation width:', layers[-1]['out'])

if __name__ == "__main__":
    print("Chapter 31 lab: Check shape compatibility when stitching layers")
    run_stitching()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
chapter_31_depth_upscaling.py
================================
مشروع متكامل: تطبيق Depth Upscaling (أسلوب SOLAR)
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel
)
from typing import List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)


class DepthUpscaler:
    """
    🧰 أداة شاملة لتطبيق Depth Upscaling (أسلوب SOLAR)
    على أي نموذج Transformer قائم على HuggingFace

    🟡 يدعم أي نسبة تكبير، وليس فقط حالة SOLAR المحددة (32→48)
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_transformer_layers(self, model):
        """
        استخراج قائمة طبقات Transformer من النموذج
        (يختلف اسم الخاصية حسب المعمارية: h في GPT-2،
        layers في LLaMA، إلخ)
        """
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h  # GPT-2 style
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers  # LLaMA/Mistral style
        else:
            raise ValueError("❌ معمارية غير مدعومة - تحقق من اسم خاصية الطبقات")

    def _set_transformer_layers(self, model, new_layers: nn.ModuleList):
        """إعادة تعيين قائمة الطبقات الجديدة في النموذج"""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model.transformer.h = new_layers
            model.config.n_layer = len(new_layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = new_layers
            model.config.num_hidden_layers = len(new_layers)

    def upscale(
        self,
        target_num_layers: int,
        verbose: bool = True
    ):
        """
        🏗️ العملية الكاملة لـ Depth Upscaling

        target_num_layers: العدد النهائي المطلوب من الطبقات
        (يجب أن يكون بين L و 2L حيث L هو عدد طبقات النموذج الأصلي)
        """
        print(f"📥 تحميل نسختين من النموذج الأساسي: {self.base_model_name}")
        model_a = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        model_b = AutoModelForCausalLM.from_pretrained(self.base_model_name)

        original_layers = self._get_transformer_layers(model_a)
        L = len(original_layers)

        if not (L < target_num_layers <= 2 * L):
            raise ValueError(
                f"❌ target_num_layers يجب أن يكون بين {L+1} و {2*L} "
                f"(النموذج الأصلي له {L} طبقة)"
            )

        # حساب عدد الطبقات المحذوفة من كل نسخة
        m = (2 * L - target_num_layers) // 2

        if verbose:
            print(f"\n📐 حسابات Depth Upscaling:")
            print(f"   عدد طبقات النموذج الأصلي (L): {L}")
            print(f"   عدد الطبقات المستهدف: {target_num_layers}")
            print(f"   عدد الطبقات المحذوفة من كل نسخة (m): {m}")

        layers_a = self._get_transformer_layers(model_a)
        layers_b = self._get_transformer_layers(model_b)

        # الاحتفاظ بالطبقات الأولى من A (حذف آخر m طبقة)
        kept_from_a = list(layers_a[:L - m])
        # الاحتفاظ بالطبقات الأخيرة من B (حذف أول m طبقة)
        kept_from_b = list(layers_b[m:])

        if verbose:
            print(f"\n✂️  من النسخة A: احتفظنا بالطبقات [0:{L-m}] "
                  f"({len(kept_from_a)} طبقة)")
            print(f"✂️  من النسخة B: احتفظنا بالطبقات [{m}:{L}] "
                  f"({len(kept_from_b)} طبقة)")

        # الخياطة: دمج القائمتين معًا بالترتيب
        stitched_layers = nn.ModuleList(kept_from_a + kept_from_b)

        if verbose:
            print(f"\n🧵 النموذج النهائي: {len(stitched_layers)} طبقة "
                  f"(بدلًا من {L} أصلًا)")

        # بناء النموذج النهائي باستخدام هيكل model_a مع الطبقات الجديدة
        self._set_transformer_layers(model_a, stitched_layers)

        # تحديث أي embeddings/buffers مرتبطة بعدد الطبقات
        # (مثلًا بعض النماذج تحتاج تحديث position_ids إذا كانت مرتبطة بعمق الشبكة)
        model_a.tie_weights()

        upscaled_params = sum(p.numel() for p in model_a.parameters())
        original_params = sum(p.numel() for p in
                                AutoModelForCausalLM.from_pretrained(
                                    self.base_model_name
                                ).parameters())

        if verbose:
            print(f"\n📊 عدد المعاملات:")
            print(f"   النموذج الأصلي: {original_params:,}")
            print(f"   النموذج بعد التكبير: {upscaled_params:,}")
            print(f"   نسبة الزيادة: {upscaled_params/original_params:.2f}x")

        del model_b  # تحرير الذاكرة
        return model_a.to(device)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def sanity_check_upscaled_model(
    model, tokenizer, original_model_name: str, test_prompt: str
):
    """
    🧪 فحص سريع للتأكد أن النموذج المُكبَّر لا يزال "متماسكًا"
    قبل الاستثمار في Continued Pretraining
    """
    print("\n" + "=" * 70)
    print("🔍 فحص السلامة (Sanity Check) للنموذج المُكبَّر")
    print("=" * 70)

    # تحميل النموذج الأصلي للمقارنة
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name
    ).to(device)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"\n📝 المطالبة: '{test_prompt}'")

    with torch.no_grad():
        # توليد من النموذج الأصلي
        output_orig = original_model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)

        # توليد من النموذج المُكبَّر
        output_upscaled = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_upscaled = tokenizer.decode(
            output_upscaled[0], skip_special_tokens=True
        )

        # حساب Perplexity لكليهما على نفس النص
        ppl_orig = compute_perplexity_single(original_model, tokenizer, test_prompt)
        ppl_upscaled = compute_perplexity_single(model, tokenizer, test_prompt)

    print(f"\n🔵 النموذج الأصلي:")
    print(f"   النص: {text_orig}")
    print(f"   Perplexity: {ppl_orig:.2f}")

    print(f"\n🟢 النموذج المُكبَّر (قبل أي تدريب إضافي!):")
    print(f"   النص: {text_upscaled}")
    print(f"   Perplexity: {ppl_upscaled:.2f}")

    ratio = ppl_upscaled / ppl_orig
    print(f"\n📊 نسبة Perplexity (مُكبَّر/أصلي): {ratio:.2f}x")

    if ratio < 5.0:
        print("✅ ممتاز! النموذج المُكبَّر لا يزال 'متماسكًا' بشكل معقول")
        print("   رغم عدم أي تدريب إضافي - جاهز لـ Continued Pretraining")
    else:
        print("⚠️  تدهور كبير - قد تحتاج مراجعة نقطة الخياطة أو نسبة m")

    del original_model
    return ratio


def compute_perplexity_single(model, tokenizer, text: str) -> float:
    """حساب Perplexity لنص واحد"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة Depth Upscaling الكاملة End-to-End"""

    BASE_MODEL = "openai-community/gpt2"  # 12 طبقة أصلًا

    upscaler = DepthUpscaler(BASE_MODEL)

    # تكبير من 12 طبقة إلى 18 طبقة (نفس نسبة SOLAR تقريبًا: 32→48)
    upscaled_model = upscaler.upscale(target_num_layers=18, verbose=True)

    # فحص السلامة قبل أي تدريب إضافي
    test_prompt = "The history of artificial intelligence began"
    sanity_check_upscaled_model(
        upscaled_model, upscaler.tokenizer, BASE_MODEL, test_prompt
    )

    print("\n" + "=" * 70)
    print("📌 الخطوة التالية في مشروع حقيقي:")
    print("=" * 70)
    print("""
    الآن بعد التحقق من سلامة النموذج المُكبَّر، الخطوة التالية
    في مشروع إنتاجي حقيقي هي تطبيق Continued Pretraining
    (كما تعلمنا بالتفصيل الكامل في الفصل 27!) على كمية معتدلة
    من البيانات (عادة أقل بكثير من Pretraining الأصلي -
    فريق SOLAR استخدم حوالي 1-2% فقط من حجم بيانات
    Pretraining الكامل) لـ:

    1. 'تنعيم' خط الخياطة بين الأجزاء المختلفة
    2. السماح للطبقات المكررة بالتمايز والتخصص تدريجيًا
    3. استعادة وتحسين الأداء العام للنموذج الأعمق الجديد
    """)

    # حفظ النموذج المُكبَّر للاستخدام في Continued Pretraining
    upscaled_model.save_pretrained("./solar_style_upscaled_model")
    upscaler.tokenizer.save_pretrained("./solar_style_upscaled_model")
    print("\n💾 تم حفظ النموذج المُكبَّر، جاهز لخطوة Continued Pretraining")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ModelGrafter:
    """
    🧰 أداة لـ Model Grafting: دمج طبقات من نماذج مختلفة
    (وليس بالضرورة نسخًا من نفس النموذج كما في Depth Upscaling)

    🟡 الاستخدام الأشهر: أخذ الطبقات الأولى من نموذج "عام قوي"
    (معالجة لغوية أساسية ممتازة) ودمجها مع الطبقات الأخيرة
    من نموذج "متخصص" (قرارات نهائية مُحسَّنة لنطاق معين)
    """

    def __init__(self):
        pass

    def graft(
        self,
        model_front: nn.Module,     # مصدر الطبقات الأولى
        model_back: nn.Module,      # مصدر الطبقات الأخيرة
        split_point_front: int,     # كم طبقة نأخذ من model_front
        split_point_back: int,      # من أي طبقة نبدأ الأخذ من model_back
    ):
        """
        ⚠️ شرط أساسي: يجب أن يكون hidden_size متطابقًا بين
        النموذجين، وإلا فإن الخياطة مستحيلة رياضيًا بدون
        طبقة إسقاط (Projection Layer) إضافية بين الجزأين
        """
        layers_front = self._get_layers(model_front)
        layers_back = self._get_layers(model_back)

        # التحقق من التوافق البنيوي
        front_hidden = model_front.config.hidden_size if hasattr(
            model_front.config, 'hidden_size'
        ) else model_front.config.n_embd
        back_hidden = model_back.config.hidden_size if hasattr(
            model_back.config, 'hidden_size'
        ) else model_back.config.n_embd

        if front_hidden != back_hidden:
            raise ValueError(
                f"❌ أبعاد غير متوافقة: {front_hidden} != {back_hidden}\n"
                f"   يلزم إضافة طبقة Projection للتوافق - راجع صندوق التعمّق أدناه"
            )

        grafted = nn.ModuleList(
            list(layers_front[:split_point_front]) +
            list(layers_back[split_point_back:])
        )

        print(f"🧵 Grafting ناجح: "
              f"{split_point_front} طبقة من النموذج الأول + "
              f"{len(layers_back) - split_point_back} طبقة من النموذج الثاني "
              f"= {len(grafted)} طبقة إجمالًا")

        return grafted

    def _get_layers(self, model):
        if hasattr(model, "transformer"):
            return model.transformer.h
        return model.model.layers

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — تجميع معماري (Architecture Stitching)
Large Language Model Engineering — Architecture

Book slug : llm-engineering-69-chapters
Chapter   : ch-31-chapter-31 (lesson 31)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_31.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 31: الفصل 31: تجميع معماري (Architecture Stitching)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_stitching() -> None:
    layers = [{'name': 'block_a', 'width': 4, 'out': 4}, {'name': 'block_b', 'width': 4, 'out': 4}]
    compatible = all((left['out'] == right['width'] for left, right in zip(layers, layers[1:])))
    stitched_depth = sum((1 for layer in layers if layer['width'] == layer['out']))
    print('adjacent dimensions compatible:', compatible)
    print('stitched blocks:', stitched_depth)
    print('final representation width:', layers[-1]['out'])

if __name__ == "__main__":
    print("Chapter 31 lab: Check shape compatibility when stitching layers")
    run_stitching()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_31_depth_upscaling.py
================================
مشروع متكامل: تطبيق Depth Upscaling (أسلوب SOLAR)
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel
)
from typing import List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)


class DepthUpscaler:
    """
    🧰 أداة شاملة لتطبيق Depth Upscaling (أسلوب SOLAR)
    على أي نموذج Transformer قائم على HuggingFace

    🟡 يدعم أي نسبة تكبير، وليس فقط حالة SOLAR المحددة (32→48)
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_transformer_layers(self, model):
        """
        استخراج قائمة طبقات Transformer من النموذج
        (يختلف اسم الخاصية حسب المعمارية: h في GPT-2،
        layers في LLaMA، إلخ)
        """
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h  # GPT-2 style
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers  # LLaMA/Mistral style
        else:
            raise ValueError("❌ معمارية غير مدعومة - تحقق من اسم خاصية الطبقات")

    def _set_transformer_layers(self, model, new_layers: nn.ModuleList):
        """إعادة تعيين قائمة الطبقات الجديدة في النموذج"""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model.transformer.h = new_layers
            model.config.n_layer = len(new_layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = new_layers
            model.config.num_hidden_layers = len(new_layers)

    def upscale(
        self,
        target_num_layers: int,
        verbose: bool = True
    ):
        """
        🏗️ العملية الكاملة لـ Depth Upscaling

        target_num_layers: العدد النهائي المطلوب من الطبقات
        (يجب أن يكون بين L و 2L حيث L هو عدد طبقات النموذج الأصلي)
        """
        print(f"📥 تحميل نسختين من النموذج الأساسي: {self.base_model_name}")
        model_a = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        model_b = AutoModelForCausalLM.from_pretrained(self.base_model_name)

        original_layers = self._get_transformer_layers(model_a)
        L = len(original_layers)

        if not (L < target_num_layers <= 2 * L):
            raise ValueError(
                f"❌ target_num_layers يجب أن يكون بين {L+1} و {2*L} "
                f"(النموذج الأصلي له {L} طبقة)"
            )

        # حساب عدد الطبقات المحذوفة من كل نسخة
        m = (2 * L - target_num_layers) // 2

        if verbose:
            print(f"\n📐 حسابات Depth Upscaling:")
            print(f"   عدد طبقات النموذج الأصلي (L): {L}")
            print(f"   عدد الطبقات المستهدف: {target_num_layers}")
            print(f"   عدد الطبقات المحذوفة من كل نسخة (m): {m}")

        layers_a = self._get_transformer_layers(model_a)
        layers_b = self._get_transformer_layers(model_b)

        # الاحتفاظ بالطبقات الأولى من A (حذف آخر m طبقة)
        kept_from_a = list(layers_a[:L - m])
        # الاحتفاظ بالطبقات الأخيرة من B (حذف أول m طبقة)
        kept_from_b = list(layers_b[m:])

        if verbose:
            print(f"\n✂️  من النسخة A: احتفظنا بالطبقات [0:{L-m}] "
                  f"({len(kept_from_a)} طبقة)")
            print(f"✂️  من النسخة B: احتفظنا بالطبقات [{m}:{L}] "
                  f"({len(kept_from_b)} طبقة)")

        # الخياطة: دمج القائمتين معًا بالترتيب
        stitched_layers = nn.ModuleList(kept_from_a + kept_from_b)

        if verbose:
            print(f"\n🧵 النموذج النهائي: {len(stitched_layers)} طبقة "
                  f"(بدلًا من {L} أصلًا)")

        # بناء النموذج النهائي باستخدام هيكل model_a مع الطبقات الجديدة
        self._set_transformer_layers(model_a, stitched_layers)

        # تحديث أي embeddings/buffers مرتبطة بعدد الطبقات
        # (مثلًا بعض النماذج تحتاج تحديث position_ids إذا كانت مرتبطة بعمق الشبكة)
        model_a.tie_weights()

        upscaled_params = sum(p.numel() for p in model_a.parameters())
        original_params = sum(p.numel() for p in
                                AutoModelForCausalLM.from_pretrained(
                                    self.base_model_name
                                ).parameters())

        if verbose:
            print(f"\n📊 عدد المعاملات:")
            print(f"   النموذج الأصلي: {original_params:,}")
            print(f"   النموذج بعد التكبير: {upscaled_params:,}")
            print(f"   نسبة الزيادة: {upscaled_params/original_params:.2f}x")

        del model_b  # تحرير الذاكرة
        return model_a.to(device)

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
def sanity_check_upscaled_model(
    model, tokenizer, original_model_name: str, test_prompt: str
):
    """
    🧪 فحص سريع للتأكد أن النموذج المُكبَّر لا يزال "متماسكًا"
    قبل الاستثمار في Continued Pretraining
    """
    print("\n" + "=" * 70)
    print("🔍 فحص السلامة (Sanity Check) للنموذج المُكبَّر")
    print("=" * 70)

    # تحميل النموذج الأصلي للمقارنة
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name
    ).to(device)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"\n📝 المطالبة: '{test_prompt}'")

    with torch.no_grad():
        # توليد من النموذج الأصلي
        output_orig = original_model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)

        # توليد من النموذج المُكبَّر
        output_upscaled = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_upscaled = tokenizer.decode(
            output_upscaled[0], skip_special_tokens=True
        )

        # حساب Perplexity لكليهما على نفس النص
        ppl_orig = compute_perplexity_single(original_model, tokenizer, test_prompt)
        ppl_upscaled = compute_perplexity_single(model, tokenizer, test_prompt)

    print(f"\n🔵 النموذج الأصلي:")
    print(f"   النص: {text_orig}")
    print(f"   Perplexity: {ppl_orig:.2f}")

    print(f"\n🟢 النموذج المُكبَّر (قبل أي تدريب إضافي!):")
    print(f"   النص: {text_upscaled}")
    print(f"   Perplexity: {ppl_upscaled:.2f}")

    ratio = ppl_upscaled / ppl_orig
    print(f"\n📊 نسبة Perplexity (مُكبَّر/أصلي): {ratio:.2f}x")

    if ratio < 5.0:
        print("✅ ممتاز! النموذج المُكبَّر لا يزال 'متماسكًا' بشكل معقول")
        print("   رغم عدم أي تدريب إضافي - جاهز لـ Continued Pretraining")
    else:
        print("⚠️  تدهور كبير - قد تحتاج مراجعة نقطة الخياطة أو نسبة m")

    del original_model
    return ratio


def compute_perplexity_single(model, tokenizer, text: str) -> float:
    """حساب Perplexity لنص واحد"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة Depth Upscaling الكاملة End-to-End"""

    BASE_MODEL = "openai-community/gpt2"  # 12 طبقة أصلًا

    upscaler = DepthUpscaler(BASE_MODEL)

    # تكبير من 12 طبقة إلى 18 طبقة (نفس نسبة SOLAR تقريبًا: 32→48)
    upscaled_model = upscaler.upscale(target_num_layers=18, verbose=True)

    # فحص السلامة قبل أي تدريب إضافي
    test_prompt = "The history of artificial intelligence began"
    sanity_check_upscaled_model(
        upscaled_model, upscaler.tokenizer, BASE_MODEL, test_prompt
    )

    print("\n" + "=" * 70)
    print("📌 الخطوة التالية في مشروع حقيقي:")
    print("=" * 70)
    print("""
    الآن بعد التحقق من سلامة النموذج المُكبَّر، الخطوة التالية
    في مشروع إنتاجي حقيقي هي تطبيق Continued Pretraining
    (كما تعلمنا بالتفصيل الكامل في الفصل 27!) على كمية معتدلة
    من البيانات (عادة أقل بكثير من Pretraining الأصلي -
    فريق SOLAR استخدم حوالي 1-2% فقط من حجم بيانات
    Pretraining الكامل) لـ:

    1. 'تنعيم' خط الخياطة بين الأجزاء المختلفة
    2. السماح للطبقات المكررة بالتمايز والتخصص تدريجيًا
    3. استعادة وتحسين الأداء العام للنموذج الأعمق الجديد
    """)

    # حفظ النموذج المُكبَّر للاستخدام في Continued Pretraining
    upscaled_model.save_pretrained("./solar_style_upscaled_model")
    upscaler.tokenizer.save_pretrained("./solar_style_upscaled_model")
    print("\n💾 تم حفظ النموذج المُكبَّر، جاهز لخطوة Continued Pretraining")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-31.md
# ────────────────────────────────────────────────────────────────────────
class ModelGrafter:
    """
    🧰 أداة لـ Model Grafting: دمج طبقات من نماذج مختلفة
    (وليس بالضرورة نسخًا من نفس النموذج كما في Depth Upscaling)

    🟡 الاستخدام الأشهر: أخذ الطبقات الأولى من نموذج "عام قوي"
    (معالجة لغوية أساسية ممتازة) ودمجها مع الطبقات الأخيرة
    من نموذج "متخصص" (قرارات نهائية مُحسَّنة لنطاق معين)
    """

    def __init__(self):
        pass

    def graft(
        self,
        model_front: nn.Module,     # مصدر الطبقات الأولى
        model_back: nn.Module,      # مصدر الطبقات الأخيرة
        split_point_front: int,     # كم طبقة نأخذ من model_front
        split_point_back: int,      # من أي طبقة نبدأ الأخذ من model_back
    ):
        """
        ⚠️ شرط أساسي: يجب أن يكون hidden_size متطابقًا بين
        النموذجين، وإلا فإن الخياطة مستحيلة رياضيًا بدون
        طبقة إسقاط (Projection Layer) إضافية بين الجزأين
        """
        layers_front = self._get_layers(model_front)
        layers_back = self._get_layers(model_back)

        # التحقق من التوافق البنيوي
        front_hidden = model_front.config.hidden_size if hasattr(
            model_front.config, 'hidden_size'
        ) else model_front.config.n_embd
        back_hidden = model_back.config.hidden_size if hasattr(
            model_back.config, 'hidden_size'
        ) else model_back.config.n_embd

        if front_hidden != back_hidden:
            raise ValueError(
                f"❌ أبعاد غير متوافقة: {front_hidden} != {back_hidden}\n"
                f"   يلزم إضافة طبقة Projection للتوافق - راجع صندوق التعمّق أدناه"
            )

        grafted = nn.ModuleList(
            list(layers_front[:split_point_front]) +
            list(layers_back[split_point_back:])
        )

        print(f"🧵 Grafting ناجح: "
              f"{split_point_front} طبقة من النموذج الأول + "
              f"{len(layers_back) - split_point_back} طبقة من النموذج الثاني "
              f"= {len(grafted)} طبقة إجمالًا")

        return grafted

    def _get_layers(self, model):
        if hasattr(model, "transformer"):
            return model.transformer.h
        return model.model.layers

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 31: الفصل 31: تجميع معماري (Architecture Stitching)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_stitching() -> None:
    layers = [{'name': 'block_a', 'width': 4, 'out': 4}, {'name': 'block_b', 'width': 4, 'out': 4}]
    compatible = all((left['out'] == right['width'] for left, right in zip(layers, layers[1:])))
    stitched_depth = sum((1 for layer in layers if layer['width'] == layer['out']))
    print('adjacent dimensions compatible:', compatible)
    print('stitched blocks:', stitched_depth)
    print('final representation width:', layers[-1]['out'])

if __name__ == "__main__":
    print("Chapter 31 lab: Check shape compatibility when stitching layers")
    run_stitching()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
chapter_31_depth_upscaling.py
================================
مشروع متكامل: تطبيق Depth Upscaling (أسلوب SOLAR)
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import copy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel
)
from typing import List
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)


class DepthUpscaler:
    """
    🧰 أداة شاملة لتطبيق Depth Upscaling (أسلوب SOLAR)
    على أي نموذج Transformer قائم على HuggingFace

    🟡 يدعم أي نسبة تكبير، وليس فقط حالة SOLAR المحددة (32→48)
    """

    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_transformer_layers(self, model):
        """
        استخراج قائمة طبقات Transformer من النموذج
        (يختلف اسم الخاصية حسب المعمارية: h في GPT-2،
        layers في LLaMA، إلخ)
        """
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h  # GPT-2 style
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers  # LLaMA/Mistral style
        else:
            raise ValueError("❌ معمارية غير مدعومة - تحقق من اسم خاصية الطبقات")

    def _set_transformer_layers(self, model, new_layers: nn.ModuleList):
        """إعادة تعيين قائمة الطبقات الجديدة في النموذج"""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model.transformer.h = new_layers
            model.config.n_layer = len(new_layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = new_layers
            model.config.num_hidden_layers = len(new_layers)

    def upscale(
        self,
        target_num_layers: int,
        verbose: bool = True
    ):
        """
        🏗️ العملية الكاملة لـ Depth Upscaling

        target_num_layers: العدد النهائي المطلوب من الطبقات
        (يجب أن يكون بين L و 2L حيث L هو عدد طبقات النموذج الأصلي)
        """
        print(f"📥 تحميل نسختين من النموذج الأساسي: {self.base_model_name}")
        model_a = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        model_b = AutoModelForCausalLM.from_pretrained(self.base_model_name)

        original_layers = self._get_transformer_layers(model_a)
        L = len(original_layers)

        if not (L < target_num_layers <= 2 * L):
            raise ValueError(
                f"❌ target_num_layers يجب أن يكون بين {L+1} و {2*L} "
                f"(النموذج الأصلي له {L} طبقة)"
            )

        # حساب عدد الطبقات المحذوفة من كل نسخة
        m = (2 * L - target_num_layers) // 2

        if verbose:
            print(f"\n📐 حسابات Depth Upscaling:")
            print(f"   عدد طبقات النموذج الأصلي (L): {L}")
            print(f"   عدد الطبقات المستهدف: {target_num_layers}")
            print(f"   عدد الطبقات المحذوفة من كل نسخة (m): {m}")

        layers_a = self._get_transformer_layers(model_a)
        layers_b = self._get_transformer_layers(model_b)

        # الاحتفاظ بالطبقات الأولى من A (حذف آخر m طبقة)
        kept_from_a = list(layers_a[:L - m])
        # الاحتفاظ بالطبقات الأخيرة من B (حذف أول m طبقة)
        kept_from_b = list(layers_b[m:])

        if verbose:
            print(f"\n✂️  من النسخة A: احتفظنا بالطبقات [0:{L-m}] "
                  f"({len(kept_from_a)} طبقة)")
            print(f"✂️  من النسخة B: احتفظنا بالطبقات [{m}:{L}] "
                  f"({len(kept_from_b)} طبقة)")

        # الخياطة: دمج القائمتين معًا بالترتيب
        stitched_layers = nn.ModuleList(kept_from_a + kept_from_b)

        if verbose:
            print(f"\n🧵 النموذج النهائي: {len(stitched_layers)} طبقة "
                  f"(بدلًا من {L} أصلًا)")

        # بناء النموذج النهائي باستخدام هيكل model_a مع الطبقات الجديدة
        self._set_transformer_layers(model_a, stitched_layers)

        # تحديث أي embeddings/buffers مرتبطة بعدد الطبقات
        # (مثلًا بعض النماذج تحتاج تحديث position_ids إذا كانت مرتبطة بعمق الشبكة)
        model_a.tie_weights()

        upscaled_params = sum(p.numel() for p in model_a.parameters())
        original_params = sum(p.numel() for p in
                                AutoModelForCausalLM.from_pretrained(
                                    self.base_model_name
                                ).parameters())

        if verbose:
            print(f"\n📊 عدد المعاملات:")
            print(f"   النموذج الأصلي: {original_params:,}")
            print(f"   النموذج بعد التكبير: {upscaled_params:,}")
            print(f"   نسبة الزيادة: {upscaled_params/original_params:.2f}x")

        del model_b  # تحرير الذاكرة
        return model_a.to(device)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def sanity_check_upscaled_model(
    model, tokenizer, original_model_name: str, test_prompt: str
):
    """
    🧪 فحص سريع للتأكد أن النموذج المُكبَّر لا يزال "متماسكًا"
    قبل الاستثمار في Continued Pretraining
    """
    print("\n" + "=" * 70)
    print("🔍 فحص السلامة (Sanity Check) للنموذج المُكبَّر")
    print("=" * 70)

    # تحميل النموذج الأصلي للمقارنة
    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name
    ).to(device)

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    print(f"\n📝 المطالبة: '{test_prompt}'")

    with torch.no_grad():
        # توليد من النموذج الأصلي
        output_orig = original_model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)

        # توليد من النموذج المُكبَّر
        output_upscaled = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text_upscaled = tokenizer.decode(
            output_upscaled[0], skip_special_tokens=True
        )

        # حساب Perplexity لكليهما على نفس النص
        ppl_orig = compute_perplexity_single(original_model, tokenizer, test_prompt)
        ppl_upscaled = compute_perplexity_single(model, tokenizer, test_prompt)

    print(f"\n🔵 النموذج الأصلي:")
    print(f"   النص: {text_orig}")
    print(f"   Perplexity: {ppl_orig:.2f}")

    print(f"\n🟢 النموذج المُكبَّر (قبل أي تدريب إضافي!):")
    print(f"   النص: {text_upscaled}")
    print(f"   Perplexity: {ppl_upscaled:.2f}")

    ratio = ppl_upscaled / ppl_orig
    print(f"\n📊 نسبة Perplexity (مُكبَّر/أصلي): {ratio:.2f}x")

    if ratio < 5.0:
        print("✅ ممتاز! النموذج المُكبَّر لا يزال 'متماسكًا' بشكل معقول")
        print("   رغم عدم أي تدريب إضافي - جاهز لـ Continued Pretraining")
    else:
        print("⚠️  تدهور كبير - قد تحتاج مراجعة نقطة الخياطة أو نسبة m")

    del original_model
    return ratio


def compute_perplexity_single(model, tokenizer, text: str) -> float:
    """حساب Perplexity لنص واحد"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة Depth Upscaling الكاملة End-to-End"""

    BASE_MODEL = "openai-community/gpt2"  # 12 طبقة أصلًا

    upscaler = DepthUpscaler(BASE_MODEL)

    # تكبير من 12 طبقة إلى 18 طبقة (نفس نسبة SOLAR تقريبًا: 32→48)
    upscaled_model = upscaler.upscale(target_num_layers=18, verbose=True)

    # فحص السلامة قبل أي تدريب إضافي
    test_prompt = "The history of artificial intelligence began"
    sanity_check_upscaled_model(
        upscaled_model, upscaler.tokenizer, BASE_MODEL, test_prompt
    )

    print("\n" + "=" * 70)
    print("📌 الخطوة التالية في مشروع حقيقي:")
    print("=" * 70)
    print("""
    الآن بعد التحقق من سلامة النموذج المُكبَّر، الخطوة التالية
    في مشروع إنتاجي حقيقي هي تطبيق Continued Pretraining
    (كما تعلمنا بالتفصيل الكامل في الفصل 27!) على كمية معتدلة
    من البيانات (عادة أقل بكثير من Pretraining الأصلي -
    فريق SOLAR استخدم حوالي 1-2% فقط من حجم بيانات
    Pretraining الكامل) لـ:

    1. 'تنعيم' خط الخياطة بين الأجزاء المختلفة
    2. السماح للطبقات المكررة بالتمايز والتخصص تدريجيًا
    3. استعادة وتحسين الأداء العام للنموذج الأعمق الجديد
    """)

    # حفظ النموذج المُكبَّر للاستخدام في Continued Pretraining
    upscaled_model.save_pretrained("./solar_style_upscaled_model")
    upscaler.tokenizer.save_pretrained("./solar_style_upscaled_model")
    print("\n💾 تم حفظ النموذج المُكبَّر، جاهز لخطوة Continued Pretraining")


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-31-chapter-31.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class ModelGrafter:
    """
    🧰 أداة لـ Model Grafting: دمج طبقات من نماذج مختلفة
    (وليس بالضرورة نسخًا من نفس النموذج كما في Depth Upscaling)

    🟡 الاستخدام الأشهر: أخذ الطبقات الأولى من نموذج "عام قوي"
    (معالجة لغوية أساسية ممتازة) ودمجها مع الطبقات الأخيرة
    من نموذج "متخصص" (قرارات نهائية مُحسَّنة لنطاق معين)
    """

    def __init__(self):
        pass

    def graft(
        self,
        model_front: nn.Module,     # مصدر الطبقات الأولى
        model_back: nn.Module,      # مصدر الطبقات الأخيرة
        split_point_front: int,     # كم طبقة نأخذ من model_front
        split_point_back: int,      # من أي طبقة نبدأ الأخذ من model_back
    ):
        """
        ⚠️ شرط أساسي: يجب أن يكون hidden_size متطابقًا بين
        النموذجين، وإلا فإن الخياطة مستحيلة رياضيًا بدون
        طبقة إسقاط (Projection Layer) إضافية بين الجزأين
        """
        layers_front = self._get_layers(model_front)
        layers_back = self._get_layers(model_back)

        # التحقق من التوافق البنيوي
        front_hidden = model_front.config.hidden_size if hasattr(
            model_front.config, 'hidden_size'
        ) else model_front.config.n_embd
        back_hidden = model_back.config.hidden_size if hasattr(
            model_back.config, 'hidden_size'
        ) else model_back.config.n_embd

        if front_hidden != back_hidden:
            raise ValueError(
                f"❌ أبعاد غير متوافقة: {front_hidden} != {back_hidden}\n"
                f"   يلزم إضافة طبقة Projection للتوافق - راجع صندوق التعمّق أدناه"
            )

        grafted = nn.ModuleList(
            list(layers_front[:split_point_front]) +
            list(layers_back[split_point_back:])
        )

        print(f"🧵 Grafting ناجح: "
              f"{split_point_front} طبقة من النموذج الأول + "
              f"{len(layers_back) - split_point_back} طبقة من النموذج الثاني "
              f"= {len(grafted)} طبقة إجمالًا")

        return grafted

    def _get_layers(self, model):
        if hasattr(model, "transformer"):
            return model.transformer.h
        return model.model.layers
