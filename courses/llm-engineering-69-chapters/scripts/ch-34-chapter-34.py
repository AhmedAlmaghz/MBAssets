#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA
Large Language Model Engineering — PEFT

Book slug : llm-engineering-69-chapters
Chapter   : ch-34-chapter-34 (lesson 34)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-34-chapter-34.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA
Large Language Model Engineering — PEFT

Book slug : llm-engineering-69-chapters
Chapter   : ch-34-chapter-34 (lesson 34)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-34-chapter-34.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA
Large Language Model Engineering — PEFT

Book slug : llm-engineering-69-chapters
Chapter   : ch-34-chapter-34 (lesson 34)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_34.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 34: الفصل 34: الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_lora() -> None:
    base = [[0.2, 0.1], [-0.3, 0.4]]
    a, b, alpha, rank = ([[1.0], [0.5]], [[0.2, -0.1]], 2.0, 1)
    delta = [[alpha / rank * sum((b[k][j] * a[i][k] for k in range(rank))) for j in range(2)] for i in range(2)]
    adapted = [[base[i][j] + delta[i][j] for j in range(2)] for i in range(2)]
    print('low-rank update:', delta)
    print('adapted weights:', adapted)
    print('trainable adapter parameters:', len(a) * rank + rank * len(b[0]))

if __name__ == "__main__":
    print("Chapter 34 lab: Apply a low-rank LoRA weight update")
    run_lora()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-34.md
# ────────────────────────────────────────────────────────────────────────
"""
بناء LoRA من الصفر لفهم الآلية الداخلية قبل استخدام مكتبات جاهزة
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    طبقة خطية مع إضافة LoRA.
    تُغلّف طبقة nn.Linear أصلية مجمَّدة، وتضيف مسارًا موازيًا منخفض الرتبة.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر الكفاءة
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A: تهيئة عشوائية (Kaiming) — تُسقط المدخل على حيّز منخفض الرتبة
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بصفر — تضمن أن ΔW = 0 عند بداية التدريب
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # المسار الأصلي (مجمَّد تمامًا، بدون تدرّجات)
        base_output = self.base_layer(x)

        # المسار الموازي منخفض الرتبة: x -> A -> B -> scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة واحدة نهائية
        — مفيد جدًا عند النشر الإنتاجي (Inference) لتفادي أي Overhead إضافي.
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.base_layer.weight.data + delta_w
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        return merged

    def count_trainable_params(self) -> dict[str, int]:
        trainable = self.lora_A.numel() + self.lora_B.numel()
        frozen = sum(p.numel() for p in self.base_layer.parameters())
        return {"trainable": trainable, "frozen": frozen, "ratio": trainable / (trainable + frozen)}


def inject_lora(model: nn.Module, target_module_names: list[str], rank: int = 8, alpha: int = 16) -> nn.Module:
    """
    يستبدل كل طبقة nn.Linear باسمها ضمن target_module_names بنسخة LoRALinear
    — هذا بالضبط ما تفعله مكتبة peft داخليًا بشكل أكثر تعقيدًا وعمومية.
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_module_names):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
    return model


# ------------------------------------------------------------------
# تحقق سريع: تأكيد أن الإخراج متطابق عند بداية التدريب (B=0)
# ------------------------------------------------------------------

torch.manual_seed(42)
original_layer = nn.Linear(512, 512)
lora_layer = LoRALinear(original_layer, rank=8, alpha=16)

x_test = torch.randn(4, 512)
with torch.no_grad():
    original_out = original_layer(x_test)
    lora_out = lora_layer(x_test)

assert torch.allclose(original_out, lora_out, atol=1e-6), "يجب أن يتطابق الإخراج عند B=0!"
print("✅ تحقّقنا: عند بداية التدريب، LoRA لا تُغيّر سلوك النموذج إطلاقًا")

stats = lora_layer.count_trainable_params()
print(f"معاملات قابلة للتدريب: {stats['trainable']:,}")
print(f"معاملات مجمَّدة: {stats['frozen']:,}")
print(f"نسبة المعاملات القابلة للتدريب: {stats['ratio']:.4%}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-34.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع كامل: ضبط نموذج عربي بـ QLoRA لتصنيف المشاعر
المتطلبات: pip install transformers peft bitsandbytes accelerate datasets trl
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # يمكن استبداله بأي نموذج أكبر لإظهار فائدة QLoRA الحقيقية


# ------------------------------------------------------------------
# 1) إعداد التكمية لتحميل النموذج بـ 4-bit (NF4)
# ------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                    # NF4 بدل التكمية الخطية العادية
    bnb_4bit_compute_dtype=torch.bfloat16,         # دقة الحساب أثناء Dequantize المؤقت
    bnb_4bit_use_double_quant=True,                # Double Quantization لضغط عوامل القياس أيضًا
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# تحضير النموذج 4-bit للتدريب (يُثبّت LayerNorm بدقة fp32 لاستقرار التدريب
# ويُفعّل Gradient Checkpointing لتوفير ذاكرة إضافية)
model = prepare_model_for_kbit_training(model)


# ------------------------------------------------------------------
# 2) إعداد LoRA Config وحقنها في النموذج المُكمَّم
# ------------------------------------------------------------------

lora_config = LoraConfig(
    r=16,                           # الرتبة — توازن بين القدرة التعبيرية والكفاءة
    lora_alpha=32,                  # alpha = 2r، قاعدة عملية شائعة
    target_modules=[                # الطبقات المستهدفة داخل كل Transformer Block
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# المتوقع: شيء مثل "trainable params: 18,874,368 || all params: 1,562,714,624 || trainable%: 1.2077"


# ------------------------------------------------------------------
# 3) بناء بيانات تدريب بصيغة تعليمات (Instruction Format)
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """صنّف المشاعر في المراجعة التالية إلى: إيجابي، سلبي، أو محايد.

المراجعة: {review}
التصنيف:"""


raw_examples = [
    {"review": "المنتج رائع جدًا والتوصيل كان سريعًا، تجربة ممتازة!", "label": "إيجابي"},
    {"review": "جودة سيئة جدًا ولم يطابق الوصف إطلاقًا، لا أنصح به.", "label": "سلبي"},
    {"review": "المنتج عادي، لا بأس به لكن لا شيء مميز فيه.", "label": "محايد"},
    {"review": "خدمة العملاء كانت متعاونة لكن المنتج تأخر كثيرًا في الوصول.", "label": "محايد"},
    {"review": "أفضل عملية شراء قمت بها هذا العام، ممتاز بكل المقاييس.", "label": "إيجابي"},
    {"review": "تحطّم المنتج من أول استخدام، إهدار كامل للمال.", "label": "سلبي"},
    # في مشروع حقيقي: مئات أو آلاف الأمثلة الموسومة
]


def format_example(example: dict) -> dict:
    full_text = PROMPT_TEMPLATE.format(review=example["review"]) + f" {example['label']}"
    return {"text": full_text}


dataset = Dataset.from_list(raw_examples).map(format_example)


# ------------------------------------------------------------------
# 4) إعداد التدريب باستخدام SFTTrainer من مكتبة trl
# ------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./qlora-arabic-sentiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # batch فعّال = 8، رغم batch صغير فعليًا على GPU
    num_train_epochs=3,
    learning_rate=2e-4,              # LR أعلى من Full FT معتاد — LoRA يحتمل معدلات تعلّم أكبر
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="paged_adamw_8bit",        # Paged Optimizer من QLoRA — يمنع OOM spikes
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
)

trainer.train()


# ------------------------------------------------------------------
# 5) حفظ Adapter فقط (ليس النموذج كاملًا!) — ميزة عملية هائلة
# ------------------------------------------------------------------

model.save_pretrained("./qlora-arabic-sentiment-adapter")
# حجم الملف المحفوظ: عادة بضعة ميجابايتات فقط، وليس جيجابايتات كاملة النموذج!


# ------------------------------------------------------------------
# 6) الاستدلال: تحميل النموذج الأساسي + الـ Adapter معًا
# ------------------------------------------------------------------

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)
inference_model = PeftModel.from_pretrained(base_model, "./qlora-arabic-sentiment-adapter")

test_review = "الجودة فوق التوقعات لكن السعر مرتفع قليلًا مقارنة بالمنافسين."
prompt = PROMPT_TEMPLATE.format(review=test_review)
inputs = tokenizer(prompt, return_tensors="pt").to(inference_model.device)

with torch.no_grad():
    output = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False)

result = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"التصنيف المتوقع: {result.strip()}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 34: الفصل 34: الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_lora() -> None:
    base = [[0.2, 0.1], [-0.3, 0.4]]
    a, b, alpha, rank = ([[1.0], [0.5]], [[0.2, -0.1]], 2.0, 1)
    delta = [[alpha / rank * sum((b[k][j] * a[i][k] for k in range(rank))) for j in range(2)] for i in range(2)]
    adapted = [[base[i][j] + delta[i][j] for j in range(2)] for i in range(2)]
    print('low-rank update:', delta)
    print('adapted weights:', adapted)
    print('trainable adapter parameters:', len(a) * rank + rank * len(b[0]))

if __name__ == "__main__":
    print("Chapter 34 lab: Apply a low-rank LoRA weight update")
    run_lora()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
بناء LoRA من الصفر لفهم الآلية الداخلية قبل استخدام مكتبات جاهزة
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    طبقة خطية مع إضافة LoRA.
    تُغلّف طبقة nn.Linear أصلية مجمَّدة، وتضيف مسارًا موازيًا منخفض الرتبة.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر الكفاءة
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A: تهيئة عشوائية (Kaiming) — تُسقط المدخل على حيّز منخفض الرتبة
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بصفر — تضمن أن ΔW = 0 عند بداية التدريب
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # المسار الأصلي (مجمَّد تمامًا، بدون تدرّجات)
        base_output = self.base_layer(x)

        # المسار الموازي منخفض الرتبة: x -> A -> B -> scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة واحدة نهائية
        — مفيد جدًا عند النشر الإنتاجي (Inference) لتفادي أي Overhead إضافي.
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.base_layer.weight.data + delta_w
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        return merged

    def count_trainable_params(self) -> dict[str, int]:
        trainable = self.lora_A.numel() + self.lora_B.numel()
        frozen = sum(p.numel() for p in self.base_layer.parameters())
        return {"trainable": trainable, "frozen": frozen, "ratio": trainable / (trainable + frozen)}


def inject_lora(model: nn.Module, target_module_names: list[str], rank: int = 8, alpha: int = 16) -> nn.Module:
    """
    يستبدل كل طبقة nn.Linear باسمها ضمن target_module_names بنسخة LoRALinear
    — هذا بالضبط ما تفعله مكتبة peft داخليًا بشكل أكثر تعقيدًا وعمومية.
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_module_names):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
    return model


# ------------------------------------------------------------------
# تحقق سريع: تأكيد أن الإخراج متطابق عند بداية التدريب (B=0)
# ------------------------------------------------------------------

torch.manual_seed(42)
original_layer = nn.Linear(512, 512)
lora_layer = LoRALinear(original_layer, rank=8, alpha=16)

x_test = torch.randn(4, 512)
with torch.no_grad():
    original_out = original_layer(x_test)
    lora_out = lora_layer(x_test)

assert torch.allclose(original_out, lora_out, atol=1e-6), "يجب أن يتطابق الإخراج عند B=0!"
print("✅ تحقّقنا: عند بداية التدريب، LoRA لا تُغيّر سلوك النموذج إطلاقًا")

stats = lora_layer.count_trainable_params()
print(f"معاملات قابلة للتدريب: {stats['trainable']:,}")
print(f"معاملات مجمَّدة: {stats['frozen']:,}")
print(f"نسبة المعاملات القابلة للتدريب: {stats['ratio']:.4%}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
مشروع كامل: ضبط نموذج عربي بـ QLoRA لتصنيف المشاعر
المتطلبات: pip install transformers peft bitsandbytes accelerate datasets trl
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # يمكن استبداله بأي نموذج أكبر لإظهار فائدة QLoRA الحقيقية


# ------------------------------------------------------------------
# 1) إعداد التكمية لتحميل النموذج بـ 4-bit (NF4)
# ------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                    # NF4 بدل التكمية الخطية العادية
    bnb_4bit_compute_dtype=torch.bfloat16,         # دقة الحساب أثناء Dequantize المؤقت
    bnb_4bit_use_double_quant=True,                # Double Quantization لضغط عوامل القياس أيضًا
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# تحضير النموذج 4-bit للتدريب (يُثبّت LayerNorm بدقة fp32 لاستقرار التدريب
# ويُفعّل Gradient Checkpointing لتوفير ذاكرة إضافية)
model = prepare_model_for_kbit_training(model)


# ------------------------------------------------------------------
# 2) إعداد LoRA Config وحقنها في النموذج المُكمَّم
# ------------------------------------------------------------------

lora_config = LoraConfig(
    r=16,                           # الرتبة — توازن بين القدرة التعبيرية والكفاءة
    lora_alpha=32,                  # alpha = 2r، قاعدة عملية شائعة
    target_modules=[                # الطبقات المستهدفة داخل كل Transformer Block
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# المتوقع: شيء مثل "trainable params: 18,874,368 || all params: 1,562,714,624 || trainable%: 1.2077"


# ------------------------------------------------------------------
# 3) بناء بيانات تدريب بصيغة تعليمات (Instruction Format)
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """صنّف المشاعر في المراجعة التالية إلى: إيجابي، سلبي، أو محايد.

المراجعة: {review}
التصنيف:"""


raw_examples = [
    {"review": "المنتج رائع جدًا والتوصيل كان سريعًا، تجربة ممتازة!", "label": "إيجابي"},
    {"review": "جودة سيئة جدًا ولم يطابق الوصف إطلاقًا، لا أنصح به.", "label": "سلبي"},
    {"review": "المنتج عادي، لا بأس به لكن لا شيء مميز فيه.", "label": "محايد"},
    {"review": "خدمة العملاء كانت متعاونة لكن المنتج تأخر كثيرًا في الوصول.", "label": "محايد"},
    {"review": "أفضل عملية شراء قمت بها هذا العام، ممتاز بكل المقاييس.", "label": "إيجابي"},
    {"review": "تحطّم المنتج من أول استخدام، إهدار كامل للمال.", "label": "سلبي"},
    # في مشروع حقيقي: مئات أو آلاف الأمثلة الموسومة
]


def format_example(example: dict) -> dict:
    full_text = PROMPT_TEMPLATE.format(review=example["review"]) + f" {example['label']}"
    return {"text": full_text}


dataset = Dataset.from_list(raw_examples).map(format_example)


# ------------------------------------------------------------------
# 4) إعداد التدريب باستخدام SFTTrainer من مكتبة trl
# ------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./qlora-arabic-sentiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # batch فعّال = 8، رغم batch صغير فعليًا على GPU
    num_train_epochs=3,
    learning_rate=2e-4,              # LR أعلى من Full FT معتاد — LoRA يحتمل معدلات تعلّم أكبر
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="paged_adamw_8bit",        # Paged Optimizer من QLoRA — يمنع OOM spikes
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
)

trainer.train()


# ------------------------------------------------------------------
# 5) حفظ Adapter فقط (ليس النموذج كاملًا!) — ميزة عملية هائلة
# ------------------------------------------------------------------

model.save_pretrained("./qlora-arabic-sentiment-adapter")
# حجم الملف المحفوظ: عادة بضعة ميجابايتات فقط، وليس جيجابايتات كاملة النموذج!


# ------------------------------------------------------------------
# 6) الاستدلال: تحميل النموذج الأساسي + الـ Adapter معًا
# ------------------------------------------------------------------

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)
inference_model = PeftModel.from_pretrained(base_model, "./qlora-arabic-sentiment-adapter")

test_review = "الجودة فوق التوقعات لكن السعر مرتفع قليلًا مقارنة بالمنافسين."
prompt = PROMPT_TEMPLATE.format(review=test_review)
inputs = tokenizer(prompt, return_tensors="pt").to(inference_model.device)

with torch.no_grad():
    output = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False)

result = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"التصنيف المتوقع: {result.strip()}")

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA
Large Language Model Engineering — PEFT

Book slug : llm-engineering-69-chapters
Chapter   : ch-34-chapter-34 (lesson 34)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_34.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 34: الفصل 34: الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_lora() -> None:
    base = [[0.2, 0.1], [-0.3, 0.4]]
    a, b, alpha, rank = ([[1.0], [0.5]], [[0.2, -0.1]], 2.0, 1)
    delta = [[alpha / rank * sum((b[k][j] * a[i][k] for k in range(rank))) for j in range(2)] for i in range(2)]
    adapted = [[base[i][j] + delta[i][j] for j in range(2)] for i in range(2)]
    print('low-rank update:', delta)
    print('adapted weights:', adapted)
    print('trainable adapter parameters:', len(a) * rank + rank * len(b[0]))

if __name__ == "__main__":
    print("Chapter 34 lab: Apply a low-rank LoRA weight update")
    run_lora()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-07-fine-tuning-alignment/chapter-34.md
# ────────────────────────────────────────────────────────────────────────
"""
بناء LoRA من الصفر لفهم الآلية الداخلية قبل استخدام مكتبات جاهزة
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    طبقة خطية مع إضافة LoRA.
    تُغلّف طبقة nn.Linear أصلية مجمَّدة، وتضيف مسارًا موازيًا منخفض الرتبة.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر الكفاءة
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A: تهيئة عشوائية (Kaiming) — تُسقط المدخل على حيّز منخفض الرتبة
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بصفر — تضمن أن ΔW = 0 عند بداية التدريب
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # المسار الأصلي (مجمَّد تمامًا، بدون تدرّجات)
        base_output = self.base_layer(x)

        # المسار الموازي منخفض الرتبة: x -> A -> B -> scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة واحدة نهائية
        — مفيد جدًا عند النشر الإنتاجي (Inference) لتفادي أي Overhead إضافي.
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.base_layer.weight.data + delta_w
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        return merged

    def count_trainable_params(self) -> dict[str, int]:
        trainable = self.lora_A.numel() + self.lora_B.numel()
        frozen = sum(p.numel() for p in self.base_layer.parameters())
        return {"trainable": trainable, "frozen": frozen, "ratio": trainable / (trainable + frozen)}


def inject_lora(model: nn.Module, target_module_names: list[str], rank: int = 8, alpha: int = 16) -> nn.Module:
    """
    يستبدل كل طبقة nn.Linear باسمها ضمن target_module_names بنسخة LoRALinear
    — هذا بالضبط ما تفعله مكتبة peft داخليًا بشكل أكثر تعقيدًا وعمومية.
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_module_names):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
    return model


# ------------------------------------------------------------------
# تحقق سريع: تأكيد أن الإخراج متطابق عند بداية التدريب (B=0)
# ------------------------------------------------------------------

torch.manual_seed(42)
original_layer = nn.Linear(512, 512)
lora_layer = LoRALinear(original_layer, rank=8, alpha=16)

x_test = torch.randn(4, 512)
with torch.no_grad():
    original_out = original_layer(x_test)
    lora_out = lora_layer(x_test)

assert torch.allclose(original_out, lora_out, atol=1e-6), "يجب أن يتطابق الإخراج عند B=0!"
print("✅ تحقّقنا: عند بداية التدريب، LoRA لا تُغيّر سلوك النموذج إطلاقًا")

stats = lora_layer.count_trainable_params()
print(f"معاملات قابلة للتدريب: {stats['trainable']:,}")
print(f"معاملات مجمَّدة: {stats['frozen']:,}")
print(f"نسبة المعاملات القابلة للتدريب: {stats['ratio']:.4%}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-07-fine-tuning-alignment/chapter-34.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع كامل: ضبط نموذج عربي بـ QLoRA لتصنيف المشاعر
المتطلبات: pip install transformers peft bitsandbytes accelerate datasets trl
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # يمكن استبداله بأي نموذج أكبر لإظهار فائدة QLoRA الحقيقية


# ------------------------------------------------------------------
# 1) إعداد التكمية لتحميل النموذج بـ 4-bit (NF4)
# ------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                    # NF4 بدل التكمية الخطية العادية
    bnb_4bit_compute_dtype=torch.bfloat16,         # دقة الحساب أثناء Dequantize المؤقت
    bnb_4bit_use_double_quant=True,                # Double Quantization لضغط عوامل القياس أيضًا
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# تحضير النموذج 4-bit للتدريب (يُثبّت LayerNorm بدقة fp32 لاستقرار التدريب
# ويُفعّل Gradient Checkpointing لتوفير ذاكرة إضافية)
model = prepare_model_for_kbit_training(model)


# ------------------------------------------------------------------
# 2) إعداد LoRA Config وحقنها في النموذج المُكمَّم
# ------------------------------------------------------------------

lora_config = LoraConfig(
    r=16,                           # الرتبة — توازن بين القدرة التعبيرية والكفاءة
    lora_alpha=32,                  # alpha = 2r، قاعدة عملية شائعة
    target_modules=[                # الطبقات المستهدفة داخل كل Transformer Block
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# المتوقع: شيء مثل "trainable params: 18,874,368 || all params: 1,562,714,624 || trainable%: 1.2077"


# ------------------------------------------------------------------
# 3) بناء بيانات تدريب بصيغة تعليمات (Instruction Format)
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """صنّف المشاعر في المراجعة التالية إلى: إيجابي، سلبي، أو محايد.

المراجعة: {review}
التصنيف:"""


raw_examples = [
    {"review": "المنتج رائع جدًا والتوصيل كان سريعًا، تجربة ممتازة!", "label": "إيجابي"},
    {"review": "جودة سيئة جدًا ولم يطابق الوصف إطلاقًا، لا أنصح به.", "label": "سلبي"},
    {"review": "المنتج عادي، لا بأس به لكن لا شيء مميز فيه.", "label": "محايد"},
    {"review": "خدمة العملاء كانت متعاونة لكن المنتج تأخر كثيرًا في الوصول.", "label": "محايد"},
    {"review": "أفضل عملية شراء قمت بها هذا العام، ممتاز بكل المقاييس.", "label": "إيجابي"},
    {"review": "تحطّم المنتج من أول استخدام، إهدار كامل للمال.", "label": "سلبي"},
    # في مشروع حقيقي: مئات أو آلاف الأمثلة الموسومة
]


def format_example(example: dict) -> dict:
    full_text = PROMPT_TEMPLATE.format(review=example["review"]) + f" {example['label']}"
    return {"text": full_text}


dataset = Dataset.from_list(raw_examples).map(format_example)


# ------------------------------------------------------------------
# 4) إعداد التدريب باستخدام SFTTrainer من مكتبة trl
# ------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./qlora-arabic-sentiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # batch فعّال = 8، رغم batch صغير فعليًا على GPU
    num_train_epochs=3,
    learning_rate=2e-4,              # LR أعلى من Full FT معتاد — LoRA يحتمل معدلات تعلّم أكبر
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="paged_adamw_8bit",        # Paged Optimizer من QLoRA — يمنع OOM spikes
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
)

trainer.train()


# ------------------------------------------------------------------
# 5) حفظ Adapter فقط (ليس النموذج كاملًا!) — ميزة عملية هائلة
# ------------------------------------------------------------------

model.save_pretrained("./qlora-arabic-sentiment-adapter")
# حجم الملف المحفوظ: عادة بضعة ميجابايتات فقط، وليس جيجابايتات كاملة النموذج!


# ------------------------------------------------------------------
# 6) الاستدلال: تحميل النموذج الأساسي + الـ Adapter معًا
# ------------------------------------------------------------------

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)
inference_model = PeftModel.from_pretrained(base_model, "./qlora-arabic-sentiment-adapter")

test_review = "الجودة فوق التوقعات لكن السعر مرتفع قليلًا مقارنة بالمنافسين."
prompt = PROMPT_TEMPLATE.format(review=test_review)
inputs = tokenizer(prompt, return_tensors="pt").to(inference_model.device)

with torch.no_grad():
    output = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False)

result = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"التصنيف المتوقع: {result.strip()}")

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 34: الفصل 34: الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_lora() -> None:
    base = [[0.2, 0.1], [-0.3, 0.4]]
    a, b, alpha, rank = ([[1.0], [0.5]], [[0.2, -0.1]], 2.0, 1)
    delta = [[alpha / rank * sum((b[k][j] * a[i][k] for k in range(rank))) for j in range(2)] for i in range(2)]
    adapted = [[base[i][j] + delta[i][j] for j in range(2)] for i in range(2)]
    print('low-rank update:', delta)
    print('adapted weights:', adapted)
    print('trainable adapter parameters:', len(a) * rank + rank * len(b[0]))

if __name__ == "__main__":
    print("Chapter 34 lab: Apply a low-rank LoRA weight update")
    run_lora()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
بناء LoRA من الصفر لفهم الآلية الداخلية قبل استخدام مكتبات جاهزة
Python 3.13+ | PyTorch 2.x
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    طبقة خطية مع إضافة LoRA.
    تُغلّف طبقة nn.Linear أصلية مجمَّدة، وتضيف مسارًا موازيًا منخفض الرتبة.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # تجميد الطبقة الأصلية بالكامل — هذا هو جوهر الكفاءة
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # A: تهيئة عشوائية (Kaiming) — تُسقط المدخل على حيّز منخفض الرتبة
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: تهيئة بصفر — تضمن أن ΔW = 0 عند بداية التدريب
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # المسار الأصلي (مجمَّد تمامًا، بدون تدرّجات)
        base_output = self.base_layer(x)

        # المسار الموازي منخفض الرتبة: x -> A -> B -> scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """
        دمج أوزان LoRA في الطبقة الأصلية لإنتاج طبقة واحدة نهائية
        — مفيد جدًا عند النشر الإنتاجي (Inference) لتفادي أي Overhead إضافي.
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.base_layer.weight.data + delta_w
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        return merged

    def count_trainable_params(self) -> dict[str, int]:
        trainable = self.lora_A.numel() + self.lora_B.numel()
        frozen = sum(p.numel() for p in self.base_layer.parameters())
        return {"trainable": trainable, "frozen": frozen, "ratio": trainable / (trainable + frozen)}


def inject_lora(model: nn.Module, target_module_names: list[str], rank: int = 8, alpha: int = 16) -> nn.Module:
    """
    يستبدل كل طبقة nn.Linear باسمها ضمن target_module_names بنسخة LoRALinear
    — هذا بالضبط ما تفعله مكتبة peft داخليًا بشكل أكثر تعقيدًا وعمومية.
    """
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_module_names):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
    return model


# ------------------------------------------------------------------
# تحقق سريع: تأكيد أن الإخراج متطابق عند بداية التدريب (B=0)
# ------------------------------------------------------------------

torch.manual_seed(42)
original_layer = nn.Linear(512, 512)
lora_layer = LoRALinear(original_layer, rank=8, alpha=16)

x_test = torch.randn(4, 512)
with torch.no_grad():
    original_out = original_layer(x_test)
    lora_out = lora_layer(x_test)

assert torch.allclose(original_out, lora_out, atol=1e-6), "يجب أن يتطابق الإخراج عند B=0!"
print("✅ تحقّقنا: عند بداية التدريب، LoRA لا تُغيّر سلوك النموذج إطلاقًا")

stats = lora_layer.count_trainable_params()
print(f"معاملات قابلة للتدريب: {stats['trainable']:,}")
print(f"معاملات مجمَّدة: {stats['frozen']:,}")
print(f"نسبة المعاملات القابلة للتدريب: {stats['ratio']:.4%}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-34-chapter-34.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
مشروع كامل: ضبط نموذج عربي بـ QLoRA لتصنيف المشاعر
المتطلبات: pip install transformers peft bitsandbytes accelerate datasets trl
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # يمكن استبداله بأي نموذج أكبر لإظهار فائدة QLoRA الحقيقية


# ------------------------------------------------------------------
# 1) إعداد التكمية لتحميل النموذج بـ 4-bit (NF4)
# ------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                    # NF4 بدل التكمية الخطية العادية
    bnb_4bit_compute_dtype=torch.bfloat16,         # دقة الحساب أثناء Dequantize المؤقت
    bnb_4bit_use_double_quant=True,                # Double Quantization لضغط عوامل القياس أيضًا
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# تحضير النموذج 4-bit للتدريب (يُثبّت LayerNorm بدقة fp32 لاستقرار التدريب
# ويُفعّل Gradient Checkpointing لتوفير ذاكرة إضافية)
model = prepare_model_for_kbit_training(model)


# ------------------------------------------------------------------
# 2) إعداد LoRA Config وحقنها في النموذج المُكمَّم
# ------------------------------------------------------------------

lora_config = LoraConfig(
    r=16,                           # الرتبة — توازن بين القدرة التعبيرية والكفاءة
    lora_alpha=32,                  # alpha = 2r، قاعدة عملية شائعة
    target_modules=[                # الطبقات المستهدفة داخل كل Transformer Block
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# المتوقع: شيء مثل "trainable params: 18,874,368 || all params: 1,562,714,624 || trainable%: 1.2077"


# ------------------------------------------------------------------
# 3) بناء بيانات تدريب بصيغة تعليمات (Instruction Format)
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """صنّف المشاعر في المراجعة التالية إلى: إيجابي، سلبي، أو محايد.

المراجعة: {review}
التصنيف:"""


raw_examples = [
    {"review": "المنتج رائع جدًا والتوصيل كان سريعًا، تجربة ممتازة!", "label": "إيجابي"},
    {"review": "جودة سيئة جدًا ولم يطابق الوصف إطلاقًا، لا أنصح به.", "label": "سلبي"},
    {"review": "المنتج عادي، لا بأس به لكن لا شيء مميز فيه.", "label": "محايد"},
    {"review": "خدمة العملاء كانت متعاونة لكن المنتج تأخر كثيرًا في الوصول.", "label": "محايد"},
    {"review": "أفضل عملية شراء قمت بها هذا العام، ممتاز بكل المقاييس.", "label": "إيجابي"},
    {"review": "تحطّم المنتج من أول استخدام، إهدار كامل للمال.", "label": "سلبي"},
    # في مشروع حقيقي: مئات أو آلاف الأمثلة الموسومة
]


def format_example(example: dict) -> dict:
    full_text = PROMPT_TEMPLATE.format(review=example["review"]) + f" {example['label']}"
    return {"text": full_text}


dataset = Dataset.from_list(raw_examples).map(format_example)


# ------------------------------------------------------------------
# 4) إعداد التدريب باستخدام SFTTrainer من مكتبة trl
# ------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./qlora-arabic-sentiment",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # batch فعّال = 8، رغم batch صغير فعليًا على GPU
    num_train_epochs=3,
    learning_rate=2e-4,              # LR أعلى من Full FT معتاد — LoRA يحتمل معدلات تعلّم أكبر
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="paged_adamw_8bit",        # Paged Optimizer من QLoRA — يمنع OOM spikes
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
)

trainer.train()


# ------------------------------------------------------------------
# 5) حفظ Adapter فقط (ليس النموذج كاملًا!) — ميزة عملية هائلة
# ------------------------------------------------------------------

model.save_pretrained("./qlora-arabic-sentiment-adapter")
# حجم الملف المحفوظ: عادة بضعة ميجابايتات فقط، وليس جيجابايتات كاملة النموذج!


# ------------------------------------------------------------------
# 6) الاستدلال: تحميل النموذج الأساسي + الـ Adapter معًا
# ------------------------------------------------------------------

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)
inference_model = PeftModel.from_pretrained(base_model, "./qlora-arabic-sentiment-adapter")

test_review = "الجودة فوق التوقعات لكن السعر مرتفع قليلًا مقارنة بالمنافسين."
prompt = PROMPT_TEMPLATE.format(review=test_review)
inputs = tokenizer(prompt, return_tensors="pt").to(inference_model.device)

with torch.no_grad():
    output = inference_model.generate(**inputs, max_new_tokens=10, do_sample=False)

result = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"التصنيف المتوقع: {result.strip()}")
