#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أساسيات PyTorch العملية والبنية التحتية الحسابية
Large Language Model Engineering — PyTorch

Book slug : llm-engineering-69-chapters
Chapter   : ch-05-chapter-05 (lesson 5)
Code files: 23 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-05-chapter-05.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أساسيات PyTorch العملية والبنية التحتية الحسابية
Large Language Model Engineering — PyTorch

Book slug : llm-engineering-69-chapters
Chapter   : ch-05-chapter-05 (lesson 5)
Code files: 22 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-05-chapter-05.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أساسيات PyTorch العملية والبنية التحتية الحسابية
Large Language Model Engineering — PyTorch

Book slug : llm-engineering-69-chapters
Chapter   : ch-05-chapter-05 (lesson 5)
Code files: 21 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 05: الفصل الخامس: أساسيات PyTorch العملية والبنية التحتية الحسابية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_torch() -> None:
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[0.5, -0.25], [0.1, 0.3]])
    loss = (x @ weights).square().mean()
    loss.backward()
    print('torch:', torch.__version__, '| device:', x.device)
    print('input/output shapes:', tuple(x.shape), tuple((x @ weights).shape))
    print('loss / gradient norm:', round(loss.item(), 4), round(x.grad.norm().item(), 4))

if __name__ == "__main__":
    print("Chapter 05 lab: Tensor shapes, autograd, and devices")
    run_torch()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # الدقة الكاملة، 4 بايت/عنصر
x_fp16 = torch.randn(1000, 1000, dtype=torch.float16)   # نصف الدقة، 2 بايت/عنصر
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # "Brain Float"، 2 بايت/عنصر

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
dtype_precision_demo.py
توضيح عملي للفرق بين fp32, fp16, bf16 من حيث المدى والدقة.
"""
from __future__ import annotations

import torch

torch.manual_seed(0)

# --- اختبار المدى الديناميكي (Dynamic Range) ---
large_value = torch.tensor(70000.0)
print("قيمة كبيرة في fp32:", large_value.to(torch.float32).item())
print("قيمة كبيرة في fp16:", large_value.to(torch.float16).item())   # inf! فاض الحد
print("قيمة كبيرة في bf16:", large_value.to(torch.bfloat16).item())  # يبقى محدودًا بدقة

# --- اختبار الدقة (Precision) ---
precise_value = torch.tensor(1.0001234567)
print("\nقيمة دقيقة في fp32:", precise_value.to(torch.float32).item())
print("قيمة دقيقة في fp16:", precise_value.to(torch.float16).item())
print("قيمة دقيقة في bf16:", precise_value.to(torch.bfloat16).item())  # أقل دقة من fp16!

# --- حجم الذاكرة الفعلي ---
tensor_1m_params = torch.randn(1_000_000)
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    size_mb = tensor_1m_params.to(dtype).element_size() * tensor_1m_params.numel() / 1e6
    print(f"\n1M عنصر بنوع {dtype}: {size_mb:.2f} MB")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
device_management.py
نمط احترافي قياسي لإدارة الأجهزة، متوافق مع أي بيئة (CPU فقط، GPU واحد، عدة GPUs).
"""
from __future__ import annotations

import torch


def get_best_device() -> torch.device:
    """يختار أفضل جهاز متاح بترتيب الأولوية: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # لأجهزة Mac بمعالج Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


device = get_best_device()
print(f"الجهاز المُستخدم: {device}")

# --- النمط القياسي: انقل النموذج والبيانات لنفس الجهاز دائمًا ---
model = torch.nn.Linear(10, 5).to(device)
x = torch.randn(32, 10).to(device)
output = model(x)   # يعمل بشكل صحيح لأن كل شيء على نفس الجهاز

# ⚠️ خطأ كلاسيكي: نسيان نقل أحد الأطراف
# y = torch.randn(32, 10)        # هذا يبقى على CPU!
# output = model(y)              # RuntimeError: Expected all tensors to be on the same device

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
contiguous_memory_demo.py
فهم الفرق بين view() و reshape()، ولماذا contiguous() أحيانًا ضرورية.
"""
from __future__ import annotations

import torch

x = torch.randn(4, 6)
print("x متصلة في الذاكرة (contiguous)?", x.is_contiguous())  # True

x_transposed = x.t()   # النقل (Transpose) لا ينسخ البيانات، فقط يُغيّر "الخرائط" (strides)
print("x_transposed متصلة؟", x_transposed.is_contiguous())    # False!

try:
    x_transposed.view(24)   # يفشل! view تتطلب ذاكرة متصلة فعليًا
except RuntimeError as e:
    print(f"\n❌ خطأ متوقع: {e}")

# ✅ الحل الأول: استخدم reshape() التي تتعامل مع الحالتين تلقائيًا (تنسخ إن لزم الأمر)
x_reshaped = x_transposed.reshape(24)
print("\n✅ reshape() نجحت:", x_reshaped.shape)

# ✅ الحل الثاني: اجعلها متصلة صراحة ثم استخدم view()
x_contiguous = x_transposed.contiguous()
x_viewed = x_contiguous.view(24)
print("✅ contiguous() + view() نجحت:", x_viewed.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
memory_estimator.py
أداة عملية لتقدير ذاكرة GPU المطلوبة قبل بدء أي تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEstimate:
    params_gb: float
    gradients_gb: float
    optimizer_state_gb: float
    total_training_gb: float
    inference_only_gb: float


def estimate_memory(
    num_params: int,
    *,
    bytes_per_param: int = 4,       # 4=fp32, 2=fp16/bf16
    optimizer: str = "adamw",       # "adamw" يحتاج عزمين، "sgd" لا يحتاج شيء إضافي
) -> MemoryEstimate:
    """
    تقدير تقريبي (Rule of Thumb) لذاكرة GPU اللازمة.
    لا يشمل ذاكرة Activations (تعتمد على batch_size وseq_len ومعمارية النموذج).
    """
    params_bytes = num_params * bytes_per_param
    gradients_bytes = num_params * bytes_per_param  # نفس دقة الأوزان عادة

    optimizer_multiplier = {"adamw": 2, "adam": 2, "sgd": 0, "sgd_momentum": 1}
    opt_bytes = num_params * bytes_per_param * optimizer_multiplier.get(optimizer, 2)

    total_bytes = params_bytes + gradients_bytes + opt_bytes
    gb = 1024 ** 3

    return MemoryEstimate(
        params_gb=params_bytes / gb,
        gradients_gb=gradients_bytes / gb,
        optimizer_state_gb=opt_bytes / gb,
        total_training_gb=total_bytes / gb,
        inference_only_gb=params_bytes / gb,  # الاستدلال يحتاج فقط الأوزان
    )


if __name__ == "__main__":
    scenarios = [
        ("نموذج صغير (125M، fp32، AdamW)", 125_000_000, 4, "adamw"),
        ("نموذج 1B (fp32، AdamW)", 1_000_000_000, 4, "adamw"),
        ("نموذج 7B (fp32، AdamW)", 7_000_000_000, 4, "adamw"),
        ("نموذج 7B (bf16 مختلط، AdamW)", 7_000_000_000, 2, "adamw"),
        ("نموذج 7B — استدلال فقط (fp16)", 7_000_000_000, 2, "adamw"),
    ]

    for name, n_params, bytes_pp, opt in scenarios:
        est = estimate_memory(n_params, bytes_per_param=bytes_pp, optimizer=opt)
        print(f"\n{name}")
        print(f"  الأوزان            : {est.params_gb:8.2f} GB")
        print(f"  التدرجات           : {est.gradients_gb:8.2f} GB")
        print(f"  حالة Optimizer      : {est.optimizer_state_gb:8.2f} GB")
        print(f"  الإجمالي (تدريب)    : {est.total_training_gb:8.2f} GB  ⚠️ بدون Activations")
        print(f"  الإجمالي (استدلال)  : {est.inference_only_gb:8.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
mixed_precision_training.py
مقارنة عملية بين التدريب بـ fp32 الكامل وبـ Mixed Precision (bf16/fp16)،
مع قياس الذاكرة والوقت الفعليين على GPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def train_step_fp32(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_amp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    device_type: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    # autocast يُحوّل العمليات المناسبة تلقائيًا لـ bf16/fp16 داخل هذا السياق فقط
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    # GradScaler ضروري مع fp16 (ليس bf16) لمنع فيضان/تلاشي التدرجات الصغيرة جدًا
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(256, 1024, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # --- تدريب fp32 عادي ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_fp32(model, optimizer, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    fp32_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # --- إعادة تهيئة النموذج لمقارنة عادلة ---
    model2 = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_amp(model2, optimizer2, scaler, x, y, device_type)
    if device.type == "cuda":
        torch.cuda.synchronize()
    amp_time = time.perf_counter() - start
    amp_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    print(f"الجهاز: {device}")
    print(f"fp32 عادي     : {fp32_time:.3f} ثانية | ذروة الذاكرة: {fp32_peak_mem:.3f} GB")
    print(f"Mixed Precision: {amp_time:.3f} ثانية | ذروة الذاكرة: {amp_peak_mem:.3f} GB")
    if device.type == "cuda":
        print(f"\nتسريع: {fp32_time/amp_time:.2f}x | توفير ذاكرة: {(1 - amp_peak_mem/fp32_peak_mem)*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
torch_compile_demo.py
استخدام torch.compile لتسريع التدريب تلقائيًا، مع ملاحظات عملية حول التحذيرات الشائعة.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h   # اتصال تخطّي (Residual Connection) — سنتعمّق فيها لاحقًا


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBlock(dim=512).to(device)

    # السطر السحري: compile النموذج دون تغيير أي منطق آخر
    compiled_model = torch.compile(model)

    x = torch.randn(32, 128, 512, device=device)

    # أول استدعاء يكون أبطأ (يحدث "تسخين" Warm-up أثناء التصريف JIT)
    out_first = compiled_model(x)
    print("الناتج الأول (مع تكلفة التصريف):", out_first.shape)

    # الاستدعاءات اللاحقة بنفس الشكل (Shape) تكون أسرع بشكل ملحوظ
    out_second = compiled_model(x)
    print("الناتج الثاني (مُسرَّع):", out_second.shape)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
flops_estimator.py
تقدير وقت التدريب التقريبي باستخدام قاعدة 6ND، بالاعتماد على معدّل FLOPs
الفعلي لبطاقة GPU معيّنة (Model FLOPs Utilization - MFU).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTimeEstimate:
    total_flops: float
    estimated_hours: float
    estimated_days: float


# أرقام تقريبية لأداء الذروة النظري لبعض بطاقات GPU الشائعة (TFLOPS بدقة bf16)
GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    "A100_80GB": 312.0,
    "H100_80GB": 989.0,
    "RTX_4090": 165.0,
}


def estimate_training_time(
    num_params: int,
    num_tokens: int,
    gpu_name: str,
    num_gpus: int = 1,
    mfu: float = 0.4,   # Model FLOPs Utilization: الكفاءة الفعلية، عادة 30-50%
) -> TrainingTimeEstimate:
    """
    مبدأ MFU: بطاقات GPU تُعلن عن "أداء الذروة النظري"، لكن لا يمكن الوصول إليه
    فعليًا بسبب نقل البيانات، التزامن، واختناقات أخرى. 40-50% اعتبار جيد وواقعي.
    """
    total_flops = 6 * num_params * num_tokens
    peak_flops_per_gpu = GPU_PEAK_TFLOPS_BF16[gpu_name] * 1e12
    effective_flops_per_sec = peak_flops_per_gpu * num_gpus * mfu

    seconds = total_flops / effective_flops_per_sec
    hours = seconds / 3600
    days = hours / 24

    return TrainingTimeEstimate(total_flops=total_flops, estimated_hours=hours, estimated_days=days)


if __name__ == "__main__":
    # مثال: تدريب نموذج 1B بارامتر على 20B توكن (نسبة Chinchilla تقريبية: 20 توكن/بارامتر)
    scenarios = [
        ("1B على 20B توكن، 8×A100", 1_000_000_000, 20_000_000_000, "A100_80GB", 8),
        ("7B على 1.4T توكن، 64×A100", 7_000_000_000, 1_400_000_000_000, "A100_80GB", 64),
        ("7B على 1.4T توكن، 64×H100", 7_000_000_000, 1_400_000_000_000, "H100_80GB", 64),
    ]

    for name, n_params, n_tokens, gpu, n_gpus in scenarios:
        est = estimate_training_time(n_params, n_tokens, gpu, n_gpus)
        print(f"\n{name}")
        print(f"  إجمالي FLOPs المطلوبة : {est.total_flops:.3e}")
        print(f"  الوقت التقديري       : {est.estimated_hours:,.1f} ساعة (~{est.estimated_days:,.1f} يوم)")

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
pretraining_readiness_check.py
سكربت تشخيصي شامل: يفحص البيئة، يُقدّر الذاكرة والوقت، ويُحذّر من المشاكل
المحتملة قبل إطلاق تدريب فعلي مكلف.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReadinessReport:
    device_info: str
    num_gpus: int
    total_gpu_memory_gb: float
    estimated_required_memory_gb: float
    memory_sufficient: bool
    warnings: list[str]


def check_training_readiness(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    bytes_per_param: int = 2,  # bf16 الافتراضي الحديث
    safety_margin: float = 1.3,  # هامش أمان 30% لذاكرة Activations غير المحسوبة بدقة
) -> ReadinessReport:
    warnings: list[str] = []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        device_info = "لا يوجد GPU متاح — سيتم التدريب على CPU (بطيء جدًا لنماذج كبيرة)"
        total_gpu_mem = 0.0
        warnings.append("⚠️ لا يوجد GPU — مناسب فقط للتجارب الصغيرة جدًا أو الاختبار")
    else:
        device_info = torch.cuda.get_device_name(0)
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    # تقدير ذاكرة الحالة الأساسية (أوزان + تدرجات + AdamW)
    base_memory_bytes = num_params * bytes_per_param * 4  # 4x: weight+grad+m+v
    base_memory_gb = base_memory_bytes / 1e9

    # تقدير تقريبي جدًا لذاكرة Activations (قاعدة تقريبية شائعة لمحوّلات)
    activations_gb = (
        batch_size * seq_len * hidden_dim * num_layers * bytes_per_param * 2
    ) / 1e9

    estimated_total = (base_memory_gb + activations_gb) * safety_margin

    if num_gpus > 0 and estimated_total > total_gpu_mem:
        warnings.append(
            f"⚠️ الذاكرة المُقدَّرة ({estimated_total:.1f}GB) تتجاوز "
            f"الذاكرة المتاحة ({total_gpu_mem:.1f}GB)! فكّر في:"
        )
        warnings.append("   - تقليل batch_size أو seq_len")
        warnings.append("   - استخدام Gradient Checkpointing")
        warnings.append("   - استخدام Gradient Accumulation بدل دفعة كبيرة واحدة")
        warnings.append("   - التدريب الموزّع عبر عدة GPUs (ZeRO/FSDP — الفصل 23)")

    if bytes_per_param == 4:
        warnings.append("💡 تلميح: تستخدم fp32 الكاملة — جرّب bf16 لتوفير نصف الذاكرة تقريبًا")

    return ReadinessReport(
        device_info=device_info,
        num_gpus=max(num_gpus, 1),
        total_gpu_memory_gb=total_gpu_mem,
        estimated_required_memory_gb=estimated_total,
        memory_sufficient=(estimated_total <= total_gpu_mem) if num_gpus > 0 else False,
        warnings=warnings,
    )


if __name__ == "__main__":
    report = check_training_readiness(
        num_params=1_000_000_000,
        batch_size=16,
        seq_len=2048,
        hidden_dim=2048,
        num_layers=24,
        bytes_per_param=2,
    )

    print(f"الجهاز: {report.device_info}")
    print(f"عدد GPUs: {report.num_gpus}")
    print(f"الذاكرة الكلية المتاحة: {report.total_gpu_memory_gb:.1f} GB")
    print(f"الذاكرة المُقدَّرة المطلوبة: {report.estimated_required_memory_gb:.1f} GB")
    print(f"هل الذاكرة كافية؟ {'✅ نعم' if report.memory_sufficient else '❌ لا'}")
    if report.warnings:
        print("\nالتحذيرات والتوصيات:")
        for w in report.warnings:
            print(f"  {w}")

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ كلاسيكي: نسيان نقل البيانات الجديدة للجهاز في كل حلقة
model = model.to(device)
for xb, yb in loader:
    output = model(xb)   # RuntimeError: xb لا يزال على CPU!

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: انقل كل دفعة بيانات للجهاز بشكل صريح في كل مرة
model = model.to(device)
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    output = model(xb)

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تخزين loss مع رسمه الحسابي الكامل يُراكم الذاكرة بلا داعٍ
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss)   # يحتفظ بكل الرسم الحسابي لكل دفعة! تسريب ذاكرة فعلي

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم .item() لاستخراج القيمة العددية فقط، وتخلّص من الرسم
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss.item())   # رقم Python عادي، لا رسم حسابي مرتبط به

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ هدر للذاكرة: PyTorch يبني رسمًا حسابيًا كاملًا حتى أثناء الاستدلال فقط
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: عطّل تتبع التدرجات تمامًا عند عدم الحاجة لـ backward()
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [017] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عمليات CUDA غير متزامنة (Asynchronous)؛ القياس سيكون غير دقيق
start = time.perf_counter()
output = model(x)
elapsed = time.perf_counter() - start   # قد يقيس فقط وقت "إطلاق" العملية، لا تنفيذها!

# ────────────────────────────────────────────────────────────────────────
# [018] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استدعِ synchronize() قبل وبعد القياس لضمان اكتمال كل عمليات GPU فعليًا
torch.cuda.synchronize()
start = time.perf_counter()
output = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ────────────────────────────────────────────────────────────────────────
# [019] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ غير مُحسَّن: تحميل البيانات يصبح عنق الزجاجة (Bottleneck) بدل GPU
loader = DataLoader(dataset, batch_size=64)

# ────────────────────────────────────────────────────────────────────────
# [020] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: استخدم عدة عمليات لتحميل البيانات بالتوازي مع التدريب
loader = DataLoader(
    dataset, batch_size=64,
    num_workers=4,       # عمليات موازية لتحضير الدفعات التالية مسبقًا
    pin_memory=True,     # يُسرّع نقل البيانات من CPU إلى GPU
)

# ────────────────────────────────────────────────────────────────────────
# [021] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
training_dashboard.py — هيكل بداية للتمرين
أداة CLI شاملة لتقييم جاهزية تدريب نموذج معيّن.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    num_params: int
    hidden_dim: int
    num_layers: int
    seq_len: int
    batch_size: int


@dataclass(frozen=True)
class HardwareSpec:
    gpu_name: str
    num_gpus: int
    memory_per_gpu_gb: float


def build_arg_parser() -> argparse.ArgumentParser:
    """
    TODO: أنشئ argparse.ArgumentParser بالحقول التالية:
    --num-params (int, مطلوب)
    --hidden-dim (int, افتراضي 2048)
    --num-layers (int, افتراضي 24)
    --seq-len (int, افتراضي 2048)
    --batch-size (int, افتراضي 16)
    --num-tokens (int, للتدريب الكامل، مطلوب لحساب FLOPs)
    --gpu-name (str, من قائمة محددة مسبقًا)
    --num-gpus (int, افتراضي 1)
    --dtype (str، إما "fp32" أو "bf16"، افتراضي "bf16")
    """
    raise NotImplementedError


def generate_full_report(model_spec: ModelSpec, hw_spec: HardwareSpec, num_tokens: int, dtype: str) -> str:
    """
    TODO: ادمج estimate_memory() و estimate_training_time() و check_training_readiness()
    (من الأمثلة السابقة في هذا الفصل) في تقرير نصي واحد منسّق وجميل، يتضمن:
    1. ملخص المواصفات المُدخلة
    2. تقدير الذاكرة (كافية أم لا، مع تحذيرات)
    3. تقدير الوقت الإجمالي للتدريب
    4. توصيات محددة إن لزم الأمر (تقليل batch_size، تفعيل bf16، عدد GPUs إضافي مطلوب...)
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # TODO: ابنِ ModelSpec و HardwareSpec من args، واستدعِ generate_full_report، واطبع النتيجة

    # اختبر أداتك بسيناريوهات مختلفة من سطر الأوامر، مثال:
    # python training_dashboard.py --num-params 7000000000 --num-tokens 1400000000000 \
    #     --gpu-name A100_80GB --num-gpus 8 --dtype bf16

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 05: الفصل الخامس: أساسيات PyTorch العملية والبنية التحتية الحسابية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_torch() -> None:
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[0.5, -0.25], [0.1, 0.3]])
    loss = (x @ weights).square().mean()
    loss.backward()
    print('torch:', torch.__version__, '| device:', x.device)
    print('input/output shapes:', tuple(x.shape), tuple((x @ weights).shape))
    print('loss / gradient norm:', round(loss.item(), 4), round(x.grad.norm().item(), 4))

if __name__ == "__main__":
    print("Chapter 05 lab: Tensor shapes, autograd, and devices")
    run_torch()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # الدقة الكاملة، 4 بايت/عنصر
x_fp16 = torch.randn(1000, 1000, dtype=torch.float16)   # نصف الدقة، 2 بايت/عنصر
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # "Brain Float"، 2 بايت/عنصر

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
dtype_precision_demo.py
توضيح عملي للفرق بين fp32, fp16, bf16 من حيث المدى والدقة.
"""
from __future__ import annotations

import torch

torch.manual_seed(0)

# --- اختبار المدى الديناميكي (Dynamic Range) ---
large_value = torch.tensor(70000.0)
print("قيمة كبيرة في fp32:", large_value.to(torch.float32).item())
print("قيمة كبيرة في fp16:", large_value.to(torch.float16).item())   # inf! فاض الحد
print("قيمة كبيرة في bf16:", large_value.to(torch.bfloat16).item())  # يبقى محدودًا بدقة

# --- اختبار الدقة (Precision) ---
precise_value = torch.tensor(1.0001234567)
print("\nقيمة دقيقة في fp32:", precise_value.to(torch.float32).item())
print("قيمة دقيقة في fp16:", precise_value.to(torch.float16).item())
print("قيمة دقيقة في bf16:", precise_value.to(torch.bfloat16).item())  # أقل دقة من fp16!

# --- حجم الذاكرة الفعلي ---
tensor_1m_params = torch.randn(1_000_000)
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    size_mb = tensor_1m_params.to(dtype).element_size() * tensor_1m_params.numel() / 1e6
    print(f"\n1M عنصر بنوع {dtype}: {size_mb:.2f} MB")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
device_management.py
نمط احترافي قياسي لإدارة الأجهزة، متوافق مع أي بيئة (CPU فقط، GPU واحد، عدة GPUs).
"""
from __future__ import annotations

import torch


def get_best_device() -> torch.device:
    """يختار أفضل جهاز متاح بترتيب الأولوية: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # لأجهزة Mac بمعالج Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


device = get_best_device()
print(f"الجهاز المُستخدم: {device}")

# --- النمط القياسي: انقل النموذج والبيانات لنفس الجهاز دائمًا ---
model = torch.nn.Linear(10, 5).to(device)
x = torch.randn(32, 10).to(device)
output = model(x)   # يعمل بشكل صحيح لأن كل شيء على نفس الجهاز

# ⚠️ خطأ كلاسيكي: نسيان نقل أحد الأطراف
# y = torch.randn(32, 10)        # هذا يبقى على CPU!
# output = model(y)              # RuntimeError: Expected all tensors to be on the same device

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
contiguous_memory_demo.py
فهم الفرق بين view() و reshape()، ولماذا contiguous() أحيانًا ضرورية.
"""
from __future__ import annotations

import torch

x = torch.randn(4, 6)
print("x متصلة في الذاكرة (contiguous)?", x.is_contiguous())  # True

x_transposed = x.t()   # النقل (Transpose) لا ينسخ البيانات، فقط يُغيّر "الخرائط" (strides)
print("x_transposed متصلة؟", x_transposed.is_contiguous())    # False!

try:
    x_transposed.view(24)   # يفشل! view تتطلب ذاكرة متصلة فعليًا
except RuntimeError as e:
    print(f"\n❌ خطأ متوقع: {e}")

# ✅ الحل الأول: استخدم reshape() التي تتعامل مع الحالتين تلقائيًا (تنسخ إن لزم الأمر)
x_reshaped = x_transposed.reshape(24)
print("\n✅ reshape() نجحت:", x_reshaped.shape)

# ✅ الحل الثاني: اجعلها متصلة صراحة ثم استخدم view()
x_contiguous = x_transposed.contiguous()
x_viewed = x_contiguous.view(24)
print("✅ contiguous() + view() نجحت:", x_viewed.shape)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
memory_estimator.py
أداة عملية لتقدير ذاكرة GPU المطلوبة قبل بدء أي تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEstimate:
    params_gb: float
    gradients_gb: float
    optimizer_state_gb: float
    total_training_gb: float
    inference_only_gb: float


def estimate_memory(
    num_params: int,
    *,
    bytes_per_param: int = 4,       # 4=fp32, 2=fp16/bf16
    optimizer: str = "adamw",       # "adamw" يحتاج عزمين، "sgd" لا يحتاج شيء إضافي
) -> MemoryEstimate:
    """
    تقدير تقريبي (Rule of Thumb) لذاكرة GPU اللازمة.
    لا يشمل ذاكرة Activations (تعتمد على batch_size وseq_len ومعمارية النموذج).
    """
    params_bytes = num_params * bytes_per_param
    gradients_bytes = num_params * bytes_per_param  # نفس دقة الأوزان عادة

    optimizer_multiplier = {"adamw": 2, "adam": 2, "sgd": 0, "sgd_momentum": 1}
    opt_bytes = num_params * bytes_per_param * optimizer_multiplier.get(optimizer, 2)

    total_bytes = params_bytes + gradients_bytes + opt_bytes
    gb = 1024 ** 3

    return MemoryEstimate(
        params_gb=params_bytes / gb,
        gradients_gb=gradients_bytes / gb,
        optimizer_state_gb=opt_bytes / gb,
        total_training_gb=total_bytes / gb,
        inference_only_gb=params_bytes / gb,  # الاستدلال يحتاج فقط الأوزان
    )


if __name__ == "__main__":
    scenarios = [
        ("نموذج صغير (125M، fp32، AdamW)", 125_000_000, 4, "adamw"),
        ("نموذج 1B (fp32، AdamW)", 1_000_000_000, 4, "adamw"),
        ("نموذج 7B (fp32، AdamW)", 7_000_000_000, 4, "adamw"),
        ("نموذج 7B (bf16 مختلط، AdamW)", 7_000_000_000, 2, "adamw"),
        ("نموذج 7B — استدلال فقط (fp16)", 7_000_000_000, 2, "adamw"),
    ]

    for name, n_params, bytes_pp, opt in scenarios:
        est = estimate_memory(n_params, bytes_per_param=bytes_pp, optimizer=opt)
        print(f"\n{name}")
        print(f"  الأوزان            : {est.params_gb:8.2f} GB")
        print(f"  التدرجات           : {est.gradients_gb:8.2f} GB")
        print(f"  حالة Optimizer      : {est.optimizer_state_gb:8.2f} GB")
        print(f"  الإجمالي (تدريب)    : {est.total_training_gb:8.2f} GB  ⚠️ بدون Activations")
        print(f"  الإجمالي (استدلال)  : {est.inference_only_gb:8.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
mixed_precision_training.py
مقارنة عملية بين التدريب بـ fp32 الكامل وبـ Mixed Precision (bf16/fp16)،
مع قياس الذاكرة والوقت الفعليين على GPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def train_step_fp32(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_amp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    device_type: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    # autocast يُحوّل العمليات المناسبة تلقائيًا لـ bf16/fp16 داخل هذا السياق فقط
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    # GradScaler ضروري مع fp16 (ليس bf16) لمنع فيضان/تلاشي التدرجات الصغيرة جدًا
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(256, 1024, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # --- تدريب fp32 عادي ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_fp32(model, optimizer, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    fp32_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # --- إعادة تهيئة النموذج لمقارنة عادلة ---
    model2 = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_amp(model2, optimizer2, scaler, x, y, device_type)
    if device.type == "cuda":
        torch.cuda.synchronize()
    amp_time = time.perf_counter() - start
    amp_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    print(f"الجهاز: {device}")
    print(f"fp32 عادي     : {fp32_time:.3f} ثانية | ذروة الذاكرة: {fp32_peak_mem:.3f} GB")
    print(f"Mixed Precision: {amp_time:.3f} ثانية | ذروة الذاكرة: {amp_peak_mem:.3f} GB")
    if device.type == "cuda":
        print(f"\nتسريع: {fp32_time/amp_time:.2f}x | توفير ذاكرة: {(1 - amp_peak_mem/fp32_peak_mem)*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
torch_compile_demo.py
استخدام torch.compile لتسريع التدريب تلقائيًا، مع ملاحظات عملية حول التحذيرات الشائعة.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h   # اتصال تخطّي (Residual Connection) — سنتعمّق فيها لاحقًا


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBlock(dim=512).to(device)

    # السطر السحري: compile النموذج دون تغيير أي منطق آخر
    compiled_model = torch.compile(model)

    x = torch.randn(32, 128, 512, device=device)

    # أول استدعاء يكون أبطأ (يحدث "تسخين" Warm-up أثناء التصريف JIT)
    out_first = compiled_model(x)
    print("الناتج الأول (مع تكلفة التصريف):", out_first.shape)

    # الاستدعاءات اللاحقة بنفس الشكل (Shape) تكون أسرع بشكل ملحوظ
    out_second = compiled_model(x)
    print("الناتج الثاني (مُسرَّع):", out_second.shape)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
flops_estimator.py
تقدير وقت التدريب التقريبي باستخدام قاعدة 6ND، بالاعتماد على معدّل FLOPs
الفعلي لبطاقة GPU معيّنة (Model FLOPs Utilization - MFU).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTimeEstimate:
    total_flops: float
    estimated_hours: float
    estimated_days: float


# أرقام تقريبية لأداء الذروة النظري لبعض بطاقات GPU الشائعة (TFLOPS بدقة bf16)
GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    "A100_80GB": 312.0,
    "H100_80GB": 989.0,
    "RTX_4090": 165.0,
}


def estimate_training_time(
    num_params: int,
    num_tokens: int,
    gpu_name: str,
    num_gpus: int = 1,
    mfu: float = 0.4,   # Model FLOPs Utilization: الكفاءة الفعلية، عادة 30-50%
) -> TrainingTimeEstimate:
    """
    مبدأ MFU: بطاقات GPU تُعلن عن "أداء الذروة النظري"، لكن لا يمكن الوصول إليه
    فعليًا بسبب نقل البيانات، التزامن، واختناقات أخرى. 40-50% اعتبار جيد وواقعي.
    """
    total_flops = 6 * num_params * num_tokens
    peak_flops_per_gpu = GPU_PEAK_TFLOPS_BF16[gpu_name] * 1e12
    effective_flops_per_sec = peak_flops_per_gpu * num_gpus * mfu

    seconds = total_flops / effective_flops_per_sec
    hours = seconds / 3600
    days = hours / 24

    return TrainingTimeEstimate(total_flops=total_flops, estimated_hours=hours, estimated_days=days)


if __name__ == "__main__":
    # مثال: تدريب نموذج 1B بارامتر على 20B توكن (نسبة Chinchilla تقريبية: 20 توكن/بارامتر)
    scenarios = [
        ("1B على 20B توكن، 8×A100", 1_000_000_000, 20_000_000_000, "A100_80GB", 8),
        ("7B على 1.4T توكن، 64×A100", 7_000_000_000, 1_400_000_000_000, "A100_80GB", 64),
        ("7B على 1.4T توكن، 64×H100", 7_000_000_000, 1_400_000_000_000, "H100_80GB", 64),
    ]

    for name, n_params, n_tokens, gpu, n_gpus in scenarios:
        est = estimate_training_time(n_params, n_tokens, gpu, n_gpus)
        print(f"\n{name}")
        print(f"  إجمالي FLOPs المطلوبة : {est.total_flops:.3e}")
        print(f"  الوقت التقديري       : {est.estimated_hours:,.1f} ساعة (~{est.estimated_days:,.1f} يوم)")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
"""
pretraining_readiness_check.py
سكربت تشخيصي شامل: يفحص البيئة، يُقدّر الذاكرة والوقت، ويُحذّر من المشاكل
المحتملة قبل إطلاق تدريب فعلي مكلف.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReadinessReport:
    device_info: str
    num_gpus: int
    total_gpu_memory_gb: float
    estimated_required_memory_gb: float
    memory_sufficient: bool
    warnings: list[str]


def check_training_readiness(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    bytes_per_param: int = 2,  # bf16 الافتراضي الحديث
    safety_margin: float = 1.3,  # هامش أمان 30% لذاكرة Activations غير المحسوبة بدقة
) -> ReadinessReport:
    warnings: list[str] = []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        device_info = "لا يوجد GPU متاح — سيتم التدريب على CPU (بطيء جدًا لنماذج كبيرة)"
        total_gpu_mem = 0.0
        warnings.append("⚠️ لا يوجد GPU — مناسب فقط للتجارب الصغيرة جدًا أو الاختبار")
    else:
        device_info = torch.cuda.get_device_name(0)
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    # تقدير ذاكرة الحالة الأساسية (أوزان + تدرجات + AdamW)
    base_memory_bytes = num_params * bytes_per_param * 4  # 4x: weight+grad+m+v
    base_memory_gb = base_memory_bytes / 1e9

    # تقدير تقريبي جدًا لذاكرة Activations (قاعدة تقريبية شائعة لمحوّلات)
    activations_gb = (
        batch_size * seq_len * hidden_dim * num_layers * bytes_per_param * 2
    ) / 1e9

    estimated_total = (base_memory_gb + activations_gb) * safety_margin

    if num_gpus > 0 and estimated_total > total_gpu_mem:
        warnings.append(
            f"⚠️ الذاكرة المُقدَّرة ({estimated_total:.1f}GB) تتجاوز "
            f"الذاكرة المتاحة ({total_gpu_mem:.1f}GB)! فكّر في:"
        )
        warnings.append("   - تقليل batch_size أو seq_len")
        warnings.append("   - استخدام Gradient Checkpointing")
        warnings.append("   - استخدام Gradient Accumulation بدل دفعة كبيرة واحدة")
        warnings.append("   - التدريب الموزّع عبر عدة GPUs (ZeRO/FSDP — الفصل 23)")

    if bytes_per_param == 4:
        warnings.append("💡 تلميح: تستخدم fp32 الكاملة — جرّب bf16 لتوفير نصف الذاكرة تقريبًا")

    return ReadinessReport(
        device_info=device_info,
        num_gpus=max(num_gpus, 1),
        total_gpu_memory_gb=total_gpu_mem,
        estimated_required_memory_gb=estimated_total,
        memory_sufficient=(estimated_total <= total_gpu_mem) if num_gpus > 0 else False,
        warnings=warnings,
    )


if __name__ == "__main__":
    report = check_training_readiness(
        num_params=1_000_000_000,
        batch_size=16,
        seq_len=2048,
        hidden_dim=2048,
        num_layers=24,
        bytes_per_param=2,
    )

    print(f"الجهاز: {report.device_info}")
    print(f"عدد GPUs: {report.num_gpus}")
    print(f"الذاكرة الكلية المتاحة: {report.total_gpu_memory_gb:.1f} GB")
    print(f"الذاكرة المُقدَّرة المطلوبة: {report.estimated_required_memory_gb:.1f} GB")
    print(f"هل الذاكرة كافية؟ {'✅ نعم' if report.memory_sufficient else '❌ لا'}")
    if report.warnings:
        print("\nالتحذيرات والتوصيات:")
        for w in report.warnings:
            print(f"  {w}")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ كلاسيكي: نسيان نقل البيانات الجديدة للجهاز في كل حلقة
model = model.to(device)
for xb, yb in loader:
    output = model(xb)   # RuntimeError: xb لا يزال على CPU!

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: انقل كل دفعة بيانات للجهاز بشكل صريح في كل مرة
model = model.to(device)
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    output = model(xb)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تخزين loss مع رسمه الحسابي الكامل يُراكم الذاكرة بلا داعٍ
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss)   # يحتفظ بكل الرسم الحسابي لكل دفعة! تسريب ذاكرة فعلي

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم .item() لاستخراج القيمة العددية فقط، وتخلّص من الرسم
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss.item())   # رقم Python عادي، لا رسم حسابي مرتبط به

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ❌ هدر للذاكرة: PyTorch يبني رسمًا حسابيًا كاملًا حتى أثناء الاستدلال فقط
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: عطّل تتبع التدرجات تمامًا عند عدم الحاجة لـ backward()
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عمليات CUDA غير متزامنة (Asynchronous)؛ القياس سيكون غير دقيق
start = time.perf_counter()
output = model(x)
elapsed = time.perf_counter() - start   # قد يقيس فقط وقت "إطلاق" العملية، لا تنفيذها!

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استدعِ synchronize() قبل وبعد القياس لضمان اكتمال كل عمليات GPU فعليًا
torch.cuda.synchronize()
start = time.perf_counter()
output = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
# ❌ غير مُحسَّن: تحميل البيانات يصبح عنق الزجاجة (Bottleneck) بدل GPU
loader = DataLoader(dataset, batch_size=64)

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: استخدم عدة عمليات لتحميل البيانات بالتوازي مع التدريب
loader = DataLoader(
    dataset, batch_size=64,
    num_workers=4,       # عمليات موازية لتحضير الدفعات التالية مسبقًا
    pin_memory=True,     # يُسرّع نقل البيانات من CPU إلى GPU
)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
"""
training_dashboard.py — هيكل بداية للتمرين
أداة CLI شاملة لتقييم جاهزية تدريب نموذج معيّن.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    num_params: int
    hidden_dim: int
    num_layers: int
    seq_len: int
    batch_size: int


@dataclass(frozen=True)
class HardwareSpec:
    gpu_name: str
    num_gpus: int
    memory_per_gpu_gb: float


def build_arg_parser() -> argparse.ArgumentParser:
    """
    TODO: أنشئ argparse.ArgumentParser بالحقول التالية:
    --num-params (int, مطلوب)
    --hidden-dim (int, افتراضي 2048)
    --num-layers (int, افتراضي 24)
    --seq-len (int, افتراضي 2048)
    --batch-size (int, افتراضي 16)
    --num-tokens (int, للتدريب الكامل، مطلوب لحساب FLOPs)
    --gpu-name (str, من قائمة محددة مسبقًا)
    --num-gpus (int, افتراضي 1)
    --dtype (str، إما "fp32" أو "bf16"، افتراضي "bf16")
    """
    raise NotImplementedError


def generate_full_report(model_spec: ModelSpec, hw_spec: HardwareSpec, num_tokens: int, dtype: str) -> str:
    """
    TODO: ادمج estimate_memory() و estimate_training_time() و check_training_readiness()
    (من الأمثلة السابقة في هذا الفصل) في تقرير نصي واحد منسّق وجميل، يتضمن:
    1. ملخص المواصفات المُدخلة
    2. تقدير الذاكرة (كافية أم لا، مع تحذيرات)
    3. تقدير الوقت الإجمالي للتدريب
    4. توصيات محددة إن لزم الأمر (تقليل batch_size، تفعيل bf16، عدد GPUs إضافي مطلوب...)
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # TODO: ابنِ ModelSpec و HardwareSpec من args، واستدعِ generate_full_report، واطبع النتيجة

    # اختبر أداتك بسيناريوهات مختلفة من سطر الأوامر، مثال:
    # python training_dashboard.py --num-params 7000000000 --num-tokens 1400000000000 \
    #     --gpu-name A100_80GB --num-gpus 8 --dtype bf16

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — أساسيات PyTorch العملية والبنية التحتية الحسابية
Large Language Model Engineering — PyTorch

Book slug : llm-engineering-69-chapters
Chapter   : ch-05-chapter-05 (lesson 5)
Code files: 21 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_05.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 05: الفصل الخامس: أساسيات PyTorch العملية والبنية التحتية الحسابية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_torch() -> None:
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[0.5, -0.25], [0.1, 0.3]])
    loss = (x @ weights).square().mean()
    loss.backward()
    print('torch:', torch.__version__, '| device:', x.device)
    print('input/output shapes:', tuple(x.shape), tuple((x @ weights).shape))
    print('loss / gradient norm:', round(loss.item(), 4), round(x.grad.norm().item(), 4))

if __name__ == "__main__":
    print("Chapter 05 lab: Tensor shapes, autograd, and devices")
    run_torch()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # الدقة الكاملة، 4 بايت/عنصر
x_fp16 = torch.randn(1000, 1000, dtype=torch.float16)   # نصف الدقة، 2 بايت/عنصر
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # "Brain Float"، 2 بايت/عنصر

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
dtype_precision_demo.py
توضيح عملي للفرق بين fp32, fp16, bf16 من حيث المدى والدقة.
"""
from __future__ import annotations

import torch

torch.manual_seed(0)

# --- اختبار المدى الديناميكي (Dynamic Range) ---
large_value = torch.tensor(70000.0)
print("قيمة كبيرة في fp32:", large_value.to(torch.float32).item())
print("قيمة كبيرة في fp16:", large_value.to(torch.float16).item())   # inf! فاض الحد
print("قيمة كبيرة في bf16:", large_value.to(torch.bfloat16).item())  # يبقى محدودًا بدقة

# --- اختبار الدقة (Precision) ---
precise_value = torch.tensor(1.0001234567)
print("\nقيمة دقيقة في fp32:", precise_value.to(torch.float32).item())
print("قيمة دقيقة في fp16:", precise_value.to(torch.float16).item())
print("قيمة دقيقة في bf16:", precise_value.to(torch.bfloat16).item())  # أقل دقة من fp16!

# --- حجم الذاكرة الفعلي ---
tensor_1m_params = torch.randn(1_000_000)
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    size_mb = tensor_1m_params.to(dtype).element_size() * tensor_1m_params.numel() / 1e6
    print(f"\n1M عنصر بنوع {dtype}: {size_mb:.2f} MB")

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
device_management.py
نمط احترافي قياسي لإدارة الأجهزة، متوافق مع أي بيئة (CPU فقط، GPU واحد، عدة GPUs).
"""
from __future__ import annotations

import torch


def get_best_device() -> torch.device:
    """يختار أفضل جهاز متاح بترتيب الأولوية: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # لأجهزة Mac بمعالج Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


device = get_best_device()
print(f"الجهاز المُستخدم: {device}")

# --- النمط القياسي: انقل النموذج والبيانات لنفس الجهاز دائمًا ---
model = torch.nn.Linear(10, 5).to(device)
x = torch.randn(32, 10).to(device)
output = model(x)   # يعمل بشكل صحيح لأن كل شيء على نفس الجهاز

# ⚠️ خطأ كلاسيكي: نسيان نقل أحد الأطراف
# y = torch.randn(32, 10)        # هذا يبقى على CPU!
# output = model(y)              # RuntimeError: Expected all tensors to be on the same device

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
contiguous_memory_demo.py
فهم الفرق بين view() و reshape()، ولماذا contiguous() أحيانًا ضرورية.
"""
from __future__ import annotations

import torch

x = torch.randn(4, 6)
print("x متصلة في الذاكرة (contiguous)?", x.is_contiguous())  # True

x_transposed = x.t()   # النقل (Transpose) لا ينسخ البيانات، فقط يُغيّر "الخرائط" (strides)
print("x_transposed متصلة؟", x_transposed.is_contiguous())    # False!

try:
    x_transposed.view(24)   # يفشل! view تتطلب ذاكرة متصلة فعليًا
except RuntimeError as e:
    print(f"\n❌ خطأ متوقع: {e}")

# ✅ الحل الأول: استخدم reshape() التي تتعامل مع الحالتين تلقائيًا (تنسخ إن لزم الأمر)
x_reshaped = x_transposed.reshape(24)
print("\n✅ reshape() نجحت:", x_reshaped.shape)

# ✅ الحل الثاني: اجعلها متصلة صراحة ثم استخدم view()
x_contiguous = x_transposed.contiguous()
x_viewed = x_contiguous.view(24)
print("✅ contiguous() + view() نجحت:", x_viewed.shape)

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
memory_estimator.py
أداة عملية لتقدير ذاكرة GPU المطلوبة قبل بدء أي تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEstimate:
    params_gb: float
    gradients_gb: float
    optimizer_state_gb: float
    total_training_gb: float
    inference_only_gb: float


def estimate_memory(
    num_params: int,
    *,
    bytes_per_param: int = 4,       # 4=fp32, 2=fp16/bf16
    optimizer: str = "adamw",       # "adamw" يحتاج عزمين، "sgd" لا يحتاج شيء إضافي
) -> MemoryEstimate:
    """
    تقدير تقريبي (Rule of Thumb) لذاكرة GPU اللازمة.
    لا يشمل ذاكرة Activations (تعتمد على batch_size وseq_len ومعمارية النموذج).
    """
    params_bytes = num_params * bytes_per_param
    gradients_bytes = num_params * bytes_per_param  # نفس دقة الأوزان عادة

    optimizer_multiplier = {"adamw": 2, "adam": 2, "sgd": 0, "sgd_momentum": 1}
    opt_bytes = num_params * bytes_per_param * optimizer_multiplier.get(optimizer, 2)

    total_bytes = params_bytes + gradients_bytes + opt_bytes
    gb = 1024 ** 3

    return MemoryEstimate(
        params_gb=params_bytes / gb,
        gradients_gb=gradients_bytes / gb,
        optimizer_state_gb=opt_bytes / gb,
        total_training_gb=total_bytes / gb,
        inference_only_gb=params_bytes / gb,  # الاستدلال يحتاج فقط الأوزان
    )


if __name__ == "__main__":
    scenarios = [
        ("نموذج صغير (125M، fp32، AdamW)", 125_000_000, 4, "adamw"),
        ("نموذج 1B (fp32، AdamW)", 1_000_000_000, 4, "adamw"),
        ("نموذج 7B (fp32، AdamW)", 7_000_000_000, 4, "adamw"),
        ("نموذج 7B (bf16 مختلط، AdamW)", 7_000_000_000, 2, "adamw"),
        ("نموذج 7B — استدلال فقط (fp16)", 7_000_000_000, 2, "adamw"),
    ]

    for name, n_params, bytes_pp, opt in scenarios:
        est = estimate_memory(n_params, bytes_per_param=bytes_pp, optimizer=opt)
        print(f"\n{name}")
        print(f"  الأوزان            : {est.params_gb:8.2f} GB")
        print(f"  التدرجات           : {est.gradients_gb:8.2f} GB")
        print(f"  حالة Optimizer      : {est.optimizer_state_gb:8.2f} GB")
        print(f"  الإجمالي (تدريب)    : {est.total_training_gb:8.2f} GB  ⚠️ بدون Activations")
        print(f"  الإجمالي (استدلال)  : {est.inference_only_gb:8.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
mixed_precision_training.py
مقارنة عملية بين التدريب بـ fp32 الكامل وبـ Mixed Precision (bf16/fp16)،
مع قياس الذاكرة والوقت الفعليين على GPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def train_step_fp32(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_amp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    device_type: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    # autocast يُحوّل العمليات المناسبة تلقائيًا لـ bf16/fp16 داخل هذا السياق فقط
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    # GradScaler ضروري مع fp16 (ليس bf16) لمنع فيضان/تلاشي التدرجات الصغيرة جدًا
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(256, 1024, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # --- تدريب fp32 عادي ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_fp32(model, optimizer, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    fp32_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # --- إعادة تهيئة النموذج لمقارنة عادلة ---
    model2 = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_amp(model2, optimizer2, scaler, x, y, device_type)
    if device.type == "cuda":
        torch.cuda.synchronize()
    amp_time = time.perf_counter() - start
    amp_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    print(f"الجهاز: {device}")
    print(f"fp32 عادي     : {fp32_time:.3f} ثانية | ذروة الذاكرة: {fp32_peak_mem:.3f} GB")
    print(f"Mixed Precision: {amp_time:.3f} ثانية | ذروة الذاكرة: {amp_peak_mem:.3f} GB")
    if device.type == "cuda":
        print(f"\nتسريع: {fp32_time/amp_time:.2f}x | توفير ذاكرة: {(1 - amp_peak_mem/fp32_peak_mem)*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
torch_compile_demo.py
استخدام torch.compile لتسريع التدريب تلقائيًا، مع ملاحظات عملية حول التحذيرات الشائعة.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h   # اتصال تخطّي (Residual Connection) — سنتعمّق فيها لاحقًا


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBlock(dim=512).to(device)

    # السطر السحري: compile النموذج دون تغيير أي منطق آخر
    compiled_model = torch.compile(model)

    x = torch.randn(32, 128, 512, device=device)

    # أول استدعاء يكون أبطأ (يحدث "تسخين" Warm-up أثناء التصريف JIT)
    out_first = compiled_model(x)
    print("الناتج الأول (مع تكلفة التصريف):", out_first.shape)

    # الاستدعاءات اللاحقة بنفس الشكل (Shape) تكون أسرع بشكل ملحوظ
    out_second = compiled_model(x)
    print("الناتج الثاني (مُسرَّع):", out_second.shape)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
flops_estimator.py
تقدير وقت التدريب التقريبي باستخدام قاعدة 6ND، بالاعتماد على معدّل FLOPs
الفعلي لبطاقة GPU معيّنة (Model FLOPs Utilization - MFU).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTimeEstimate:
    total_flops: float
    estimated_hours: float
    estimated_days: float


# أرقام تقريبية لأداء الذروة النظري لبعض بطاقات GPU الشائعة (TFLOPS بدقة bf16)
GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    "A100_80GB": 312.0,
    "H100_80GB": 989.0,
    "RTX_4090": 165.0,
}


def estimate_training_time(
    num_params: int,
    num_tokens: int,
    gpu_name: str,
    num_gpus: int = 1,
    mfu: float = 0.4,   # Model FLOPs Utilization: الكفاءة الفعلية، عادة 30-50%
) -> TrainingTimeEstimate:
    """
    مبدأ MFU: بطاقات GPU تُعلن عن "أداء الذروة النظري"، لكن لا يمكن الوصول إليه
    فعليًا بسبب نقل البيانات، التزامن، واختناقات أخرى. 40-50% اعتبار جيد وواقعي.
    """
    total_flops = 6 * num_params * num_tokens
    peak_flops_per_gpu = GPU_PEAK_TFLOPS_BF16[gpu_name] * 1e12
    effective_flops_per_sec = peak_flops_per_gpu * num_gpus * mfu

    seconds = total_flops / effective_flops_per_sec
    hours = seconds / 3600
    days = hours / 24

    return TrainingTimeEstimate(total_flops=total_flops, estimated_hours=hours, estimated_days=days)


if __name__ == "__main__":
    # مثال: تدريب نموذج 1B بارامتر على 20B توكن (نسبة Chinchilla تقريبية: 20 توكن/بارامتر)
    scenarios = [
        ("1B على 20B توكن، 8×A100", 1_000_000_000, 20_000_000_000, "A100_80GB", 8),
        ("7B على 1.4T توكن، 64×A100", 7_000_000_000, 1_400_000_000_000, "A100_80GB", 64),
        ("7B على 1.4T توكن، 64×H100", 7_000_000_000, 1_400_000_000_000, "H100_80GB", 64),
    ]

    for name, n_params, n_tokens, gpu, n_gpus in scenarios:
        est = estimate_training_time(n_params, n_tokens, gpu, n_gpus)
        print(f"\n{name}")
        print(f"  إجمالي FLOPs المطلوبة : {est.total_flops:.3e}")
        print(f"  الوقت التقديري       : {est.estimated_hours:,.1f} ساعة (~{est.estimated_days:,.1f} يوم)")

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
pretraining_readiness_check.py
سكربت تشخيصي شامل: يفحص البيئة، يُقدّر الذاكرة والوقت، ويُحذّر من المشاكل
المحتملة قبل إطلاق تدريب فعلي مكلف.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReadinessReport:
    device_info: str
    num_gpus: int
    total_gpu_memory_gb: float
    estimated_required_memory_gb: float
    memory_sufficient: bool
    warnings: list[str]


def check_training_readiness(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    bytes_per_param: int = 2,  # bf16 الافتراضي الحديث
    safety_margin: float = 1.3,  # هامش أمان 30% لذاكرة Activations غير المحسوبة بدقة
) -> ReadinessReport:
    warnings: list[str] = []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        device_info = "لا يوجد GPU متاح — سيتم التدريب على CPU (بطيء جدًا لنماذج كبيرة)"
        total_gpu_mem = 0.0
        warnings.append("⚠️ لا يوجد GPU — مناسب فقط للتجارب الصغيرة جدًا أو الاختبار")
    else:
        device_info = torch.cuda.get_device_name(0)
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    # تقدير ذاكرة الحالة الأساسية (أوزان + تدرجات + AdamW)
    base_memory_bytes = num_params * bytes_per_param * 4  # 4x: weight+grad+m+v
    base_memory_gb = base_memory_bytes / 1e9

    # تقدير تقريبي جدًا لذاكرة Activations (قاعدة تقريبية شائعة لمحوّلات)
    activations_gb = (
        batch_size * seq_len * hidden_dim * num_layers * bytes_per_param * 2
    ) / 1e9

    estimated_total = (base_memory_gb + activations_gb) * safety_margin

    if num_gpus > 0 and estimated_total > total_gpu_mem:
        warnings.append(
            f"⚠️ الذاكرة المُقدَّرة ({estimated_total:.1f}GB) تتجاوز "
            f"الذاكرة المتاحة ({total_gpu_mem:.1f}GB)! فكّر في:"
        )
        warnings.append("   - تقليل batch_size أو seq_len")
        warnings.append("   - استخدام Gradient Checkpointing")
        warnings.append("   - استخدام Gradient Accumulation بدل دفعة كبيرة واحدة")
        warnings.append("   - التدريب الموزّع عبر عدة GPUs (ZeRO/FSDP — الفصل 23)")

    if bytes_per_param == 4:
        warnings.append("💡 تلميح: تستخدم fp32 الكاملة — جرّب bf16 لتوفير نصف الذاكرة تقريبًا")

    return ReadinessReport(
        device_info=device_info,
        num_gpus=max(num_gpus, 1),
        total_gpu_memory_gb=total_gpu_mem,
        estimated_required_memory_gb=estimated_total,
        memory_sufficient=(estimated_total <= total_gpu_mem) if num_gpus > 0 else False,
        warnings=warnings,
    )


if __name__ == "__main__":
    report = check_training_readiness(
        num_params=1_000_000_000,
        batch_size=16,
        seq_len=2048,
        hidden_dim=2048,
        num_layers=24,
        bytes_per_param=2,
    )

    print(f"الجهاز: {report.device_info}")
    print(f"عدد GPUs: {report.num_gpus}")
    print(f"الذاكرة الكلية المتاحة: {report.total_gpu_memory_gb:.1f} GB")
    print(f"الذاكرة المُقدَّرة المطلوبة: {report.estimated_required_memory_gb:.1f} GB")
    print(f"هل الذاكرة كافية؟ {'✅ نعم' if report.memory_sufficient else '❌ لا'}")
    if report.warnings:
        print("\nالتحذيرات والتوصيات:")
        for w in report.warnings:
            print(f"  {w}")

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ كلاسيكي: نسيان نقل البيانات الجديدة للجهاز في كل حلقة
model = model.to(device)
for xb, yb in loader:
    output = model(xb)   # RuntimeError: xb لا يزال على CPU!

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: انقل كل دفعة بيانات للجهاز بشكل صريح في كل مرة
model = model.to(device)
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    output = model(xb)

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تخزين loss مع رسمه الحسابي الكامل يُراكم الذاكرة بلا داعٍ
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss)   # يحتفظ بكل الرسم الحسابي لكل دفعة! تسريب ذاكرة فعلي

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم .item() لاستخراج القيمة العددية فقط، وتخلّص من الرسم
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss.item())   # رقم Python عادي، لا رسم حسابي مرتبط به

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ هدر للذاكرة: PyTorch يبني رسمًا حسابيًا كاملًا حتى أثناء الاستدلال فقط
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: عطّل تتبع التدرجات تمامًا عند عدم الحاجة لـ backward()
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [017] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عمليات CUDA غير متزامنة (Asynchronous)؛ القياس سيكون غير دقيق
start = time.perf_counter()
output = model(x)
elapsed = time.perf_counter() - start   # قد يقيس فقط وقت "إطلاق" العملية، لا تنفيذها!

# ────────────────────────────────────────────────────────────────────────
# [018] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استدعِ synchronize() قبل وبعد القياس لضمان اكتمال كل عمليات GPU فعليًا
torch.cuda.synchronize()
start = time.perf_counter()
output = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ────────────────────────────────────────────────────────────────────────
# [019] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ❌ غير مُحسَّن: تحميل البيانات يصبح عنق الزجاجة (Bottleneck) بدل GPU
loader = DataLoader(dataset, batch_size=64)

# ────────────────────────────────────────────────────────────────────────
# [020] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: استخدم عدة عمليات لتحميل البيانات بالتوازي مع التدريب
loader = DataLoader(
    dataset, batch_size=64,
    num_workers=4,       # عمليات موازية لتحضير الدفعات التالية مسبقًا
    pin_memory=True,     # يُسرّع نقل البيانات من CPU إلى GPU
)

# ────────────────────────────────────────────────────────────────────────
# [021] book/chapters/part-01-foundations/chapter-05.md
# ────────────────────────────────────────────────────────────────────────
"""
training_dashboard.py — هيكل بداية للتمرين
أداة CLI شاملة لتقييم جاهزية تدريب نموذج معيّن.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    num_params: int
    hidden_dim: int
    num_layers: int
    seq_len: int
    batch_size: int


@dataclass(frozen=True)
class HardwareSpec:
    gpu_name: str
    num_gpus: int
    memory_per_gpu_gb: float


def build_arg_parser() -> argparse.ArgumentParser:
    """
    TODO: أنشئ argparse.ArgumentParser بالحقول التالية:
    --num-params (int, مطلوب)
    --hidden-dim (int, افتراضي 2048)
    --num-layers (int, افتراضي 24)
    --seq-len (int, افتراضي 2048)
    --batch-size (int, افتراضي 16)
    --num-tokens (int, للتدريب الكامل، مطلوب لحساب FLOPs)
    --gpu-name (str, من قائمة محددة مسبقًا)
    --num-gpus (int, افتراضي 1)
    --dtype (str، إما "fp32" أو "bf16"، افتراضي "bf16")
    """
    raise NotImplementedError


def generate_full_report(model_spec: ModelSpec, hw_spec: HardwareSpec, num_tokens: int, dtype: str) -> str:
    """
    TODO: ادمج estimate_memory() و estimate_training_time() و check_training_readiness()
    (من الأمثلة السابقة في هذا الفصل) في تقرير نصي واحد منسّق وجميل، يتضمن:
    1. ملخص المواصفات المُدخلة
    2. تقدير الذاكرة (كافية أم لا، مع تحذيرات)
    3. تقدير الوقت الإجمالي للتدريب
    4. توصيات محددة إن لزم الأمر (تقليل batch_size، تفعيل bf16، عدد GPUs إضافي مطلوب...)
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # TODO: ابنِ ModelSpec و HardwareSpec من args، واستدعِ generate_full_report، واطبع النتيجة

    # اختبر أداتك بسيناريوهات مختلفة من سطر الأوامر، مثال:
    # python training_dashboard.py --num-params 7000000000 --num-tokens 1400000000000 \
    #     --gpu-name A100_80GB --num-gpus 8 --dtype bf16

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 05: الفصل الخامس: أساسيات PyTorch العملية والبنية التحتية الحسابية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_torch() -> None:
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[0.5, -0.25], [0.1, 0.3]])
    loss = (x @ weights).square().mean()
    loss.backward()
    print('torch:', torch.__version__, '| device:', x.device)
    print('input/output shapes:', tuple(x.shape), tuple((x @ weights).shape))
    print('loss / gradient norm:', round(loss.item(), 4), round(x.grad.norm().item(), 4))

if __name__ == "__main__":
    print("Chapter 05 lab: Tensor shapes, autograd, and devices")
    run_torch()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # الدقة الكاملة، 4 بايت/عنصر
x_fp16 = torch.randn(1000, 1000, dtype=torch.float16)   # نصف الدقة، 2 بايت/عنصر
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # "Brain Float"، 2 بايت/عنصر

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
dtype_precision_demo.py
توضيح عملي للفرق بين fp32, fp16, bf16 من حيث المدى والدقة.
"""
from __future__ import annotations

import torch

torch.manual_seed(0)

# --- اختبار المدى الديناميكي (Dynamic Range) ---
large_value = torch.tensor(70000.0)
print("قيمة كبيرة في fp32:", large_value.to(torch.float32).item())
print("قيمة كبيرة في fp16:", large_value.to(torch.float16).item())   # inf! فاض الحد
print("قيمة كبيرة في bf16:", large_value.to(torch.bfloat16).item())  # يبقى محدودًا بدقة

# --- اختبار الدقة (Precision) ---
precise_value = torch.tensor(1.0001234567)
print("\nقيمة دقيقة في fp32:", precise_value.to(torch.float32).item())
print("قيمة دقيقة في fp16:", precise_value.to(torch.float16).item())
print("قيمة دقيقة في bf16:", precise_value.to(torch.bfloat16).item())  # أقل دقة من fp16!

# --- حجم الذاكرة الفعلي ---
tensor_1m_params = torch.randn(1_000_000)
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    size_mb = tensor_1m_params.to(dtype).element_size() * tensor_1m_params.numel() / 1e6
    print(f"\n1M عنصر بنوع {dtype}: {size_mb:.2f} MB")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
device_management.py
نمط احترافي قياسي لإدارة الأجهزة، متوافق مع أي بيئة (CPU فقط، GPU واحد، عدة GPUs).
"""
from __future__ import annotations

import torch


def get_best_device() -> torch.device:
    """يختار أفضل جهاز متاح بترتيب الأولوية: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # لأجهزة Mac بمعالج Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


device = get_best_device()
print(f"الجهاز المُستخدم: {device}")

# --- النمط القياسي: انقل النموذج والبيانات لنفس الجهاز دائمًا ---
model = torch.nn.Linear(10, 5).to(device)
x = torch.randn(32, 10).to(device)
output = model(x)   # يعمل بشكل صحيح لأن كل شيء على نفس الجهاز

# ⚠️ خطأ كلاسيكي: نسيان نقل أحد الأطراف
# y = torch.randn(32, 10)        # هذا يبقى على CPU!
# output = model(y)              # RuntimeError: Expected all tensors to be on the same device

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
contiguous_memory_demo.py
فهم الفرق بين view() و reshape()، ولماذا contiguous() أحيانًا ضرورية.
"""
from __future__ import annotations

import torch

x = torch.randn(4, 6)
print("x متصلة في الذاكرة (contiguous)?", x.is_contiguous())  # True

x_transposed = x.t()   # النقل (Transpose) لا ينسخ البيانات، فقط يُغيّر "الخرائط" (strides)
print("x_transposed متصلة؟", x_transposed.is_contiguous())    # False!

try:
    x_transposed.view(24)   # يفشل! view تتطلب ذاكرة متصلة فعليًا
except RuntimeError as e:
    print(f"\n❌ خطأ متوقع: {e}")

# ✅ الحل الأول: استخدم reshape() التي تتعامل مع الحالتين تلقائيًا (تنسخ إن لزم الأمر)
x_reshaped = x_transposed.reshape(24)
print("\n✅ reshape() نجحت:", x_reshaped.shape)

# ✅ الحل الثاني: اجعلها متصلة صراحة ثم استخدم view()
x_contiguous = x_transposed.contiguous()
x_viewed = x_contiguous.view(24)
print("✅ contiguous() + view() نجحت:", x_viewed.shape)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
memory_estimator.py
أداة عملية لتقدير ذاكرة GPU المطلوبة قبل بدء أي تدريب.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryEstimate:
    params_gb: float
    gradients_gb: float
    optimizer_state_gb: float
    total_training_gb: float
    inference_only_gb: float


def estimate_memory(
    num_params: int,
    *,
    bytes_per_param: int = 4,       # 4=fp32, 2=fp16/bf16
    optimizer: str = "adamw",       # "adamw" يحتاج عزمين، "sgd" لا يحتاج شيء إضافي
) -> MemoryEstimate:
    """
    تقدير تقريبي (Rule of Thumb) لذاكرة GPU اللازمة.
    لا يشمل ذاكرة Activations (تعتمد على batch_size وseq_len ومعمارية النموذج).
    """
    params_bytes = num_params * bytes_per_param
    gradients_bytes = num_params * bytes_per_param  # نفس دقة الأوزان عادة

    optimizer_multiplier = {"adamw": 2, "adam": 2, "sgd": 0, "sgd_momentum": 1}
    opt_bytes = num_params * bytes_per_param * optimizer_multiplier.get(optimizer, 2)

    total_bytes = params_bytes + gradients_bytes + opt_bytes
    gb = 1024 ** 3

    return MemoryEstimate(
        params_gb=params_bytes / gb,
        gradients_gb=gradients_bytes / gb,
        optimizer_state_gb=opt_bytes / gb,
        total_training_gb=total_bytes / gb,
        inference_only_gb=params_bytes / gb,  # الاستدلال يحتاج فقط الأوزان
    )


if __name__ == "__main__":
    scenarios = [
        ("نموذج صغير (125M، fp32، AdamW)", 125_000_000, 4, "adamw"),
        ("نموذج 1B (fp32، AdamW)", 1_000_000_000, 4, "adamw"),
        ("نموذج 7B (fp32، AdamW)", 7_000_000_000, 4, "adamw"),
        ("نموذج 7B (bf16 مختلط، AdamW)", 7_000_000_000, 2, "adamw"),
        ("نموذج 7B — استدلال فقط (fp16)", 7_000_000_000, 2, "adamw"),
    ]

    for name, n_params, bytes_pp, opt in scenarios:
        est = estimate_memory(n_params, bytes_per_param=bytes_pp, optimizer=opt)
        print(f"\n{name}")
        print(f"  الأوزان            : {est.params_gb:8.2f} GB")
        print(f"  التدرجات           : {est.gradients_gb:8.2f} GB")
        print(f"  حالة Optimizer      : {est.optimizer_state_gb:8.2f} GB")
        print(f"  الإجمالي (تدريب)    : {est.total_training_gb:8.2f} GB  ⚠️ بدون Activations")
        print(f"  الإجمالي (استدلال)  : {est.inference_only_gb:8.2f} GB")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
mixed_precision_training.py
مقارنة عملية بين التدريب بـ fp32 الكامل وبـ Mixed Precision (bf16/fp16)،
مع قياس الذاكرة والوقت الفعليين على GPU.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def train_step_fp32(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_amp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    x: torch.Tensor,
    y: torch.Tensor,
    device_type: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    # autocast يُحوّل العمليات المناسبة تلقائيًا لـ bf16/fp16 داخل هذا السياق فقط
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    # GradScaler ضروري مع fp16 (ليس bf16) لمنع فيضان/تلاشي التدرجات الصغيرة جدًا
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(256, 1024, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # --- تدريب fp32 عادي ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_fp32(model, optimizer, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    fp32_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # --- إعادة تهيئة النموذج لمقارنة عادلة ---
    model2 = nn.Sequential(
        nn.Linear(1024, 2048), nn.GELU(),
        nn.Linear(2048, 2048), nn.GELU(),
        nn.Linear(2048, 10),
    ).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        train_step_amp(model2, optimizer2, scaler, x, y, device_type)
    if device.type == "cuda":
        torch.cuda.synchronize()
    amp_time = time.perf_counter() - start
    amp_peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    print(f"الجهاز: {device}")
    print(f"fp32 عادي     : {fp32_time:.3f} ثانية | ذروة الذاكرة: {fp32_peak_mem:.3f} GB")
    print(f"Mixed Precision: {amp_time:.3f} ثانية | ذروة الذاكرة: {amp_peak_mem:.3f} GB")
    if device.type == "cuda":
        print(f"\nتسريع: {fp32_time/amp_time:.2f}x | توفير ذاكرة: {(1 - amp_peak_mem/fp32_peak_mem)*100:.1f}%")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
torch_compile_demo.py
استخدام torch.compile لتسريع التدريب تلقائيًا، مع ملاحظات عملية حول التحذيرات الشائعة.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h   # اتصال تخطّي (Residual Connection) — سنتعمّق فيها لاحقًا


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBlock(dim=512).to(device)

    # السطر السحري: compile النموذج دون تغيير أي منطق آخر
    compiled_model = torch.compile(model)

    x = torch.randn(32, 128, 512, device=device)

    # أول استدعاء يكون أبطأ (يحدث "تسخين" Warm-up أثناء التصريف JIT)
    out_first = compiled_model(x)
    print("الناتج الأول (مع تكلفة التصريف):", out_first.shape)

    # الاستدعاءات اللاحقة بنفس الشكل (Shape) تكون أسرع بشكل ملحوظ
    out_second = compiled_model(x)
    print("الناتج الثاني (مُسرَّع):", out_second.shape)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
"""
flops_estimator.py
تقدير وقت التدريب التقريبي باستخدام قاعدة 6ND، بالاعتماد على معدّل FLOPs
الفعلي لبطاقة GPU معيّنة (Model FLOPs Utilization - MFU).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTimeEstimate:
    total_flops: float
    estimated_hours: float
    estimated_days: float


# أرقام تقريبية لأداء الذروة النظري لبعض بطاقات GPU الشائعة (TFLOPS بدقة bf16)
GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    "A100_80GB": 312.0,
    "H100_80GB": 989.0,
    "RTX_4090": 165.0,
}


def estimate_training_time(
    num_params: int,
    num_tokens: int,
    gpu_name: str,
    num_gpus: int = 1,
    mfu: float = 0.4,   # Model FLOPs Utilization: الكفاءة الفعلية، عادة 30-50%
) -> TrainingTimeEstimate:
    """
    مبدأ MFU: بطاقات GPU تُعلن عن "أداء الذروة النظري"، لكن لا يمكن الوصول إليه
    فعليًا بسبب نقل البيانات، التزامن، واختناقات أخرى. 40-50% اعتبار جيد وواقعي.
    """
    total_flops = 6 * num_params * num_tokens
    peak_flops_per_gpu = GPU_PEAK_TFLOPS_BF16[gpu_name] * 1e12
    effective_flops_per_sec = peak_flops_per_gpu * num_gpus * mfu

    seconds = total_flops / effective_flops_per_sec
    hours = seconds / 3600
    days = hours / 24

    return TrainingTimeEstimate(total_flops=total_flops, estimated_hours=hours, estimated_days=days)


if __name__ == "__main__":
    # مثال: تدريب نموذج 1B بارامتر على 20B توكن (نسبة Chinchilla تقريبية: 20 توكن/بارامتر)
    scenarios = [
        ("1B على 20B توكن، 8×A100", 1_000_000_000, 20_000_000_000, "A100_80GB", 8),
        ("7B على 1.4T توكن، 64×A100", 7_000_000_000, 1_400_000_000_000, "A100_80GB", 64),
        ("7B على 1.4T توكن، 64×H100", 7_000_000_000, 1_400_000_000_000, "H100_80GB", 64),
    ]

    for name, n_params, n_tokens, gpu, n_gpus in scenarios:
        est = estimate_training_time(n_params, n_tokens, gpu, n_gpus)
        print(f"\n{name}")
        print(f"  إجمالي FLOPs المطلوبة : {est.total_flops:.3e}")
        print(f"  الوقت التقديري       : {est.estimated_hours:,.1f} ساعة (~{est.estimated_days:,.1f} يوم)")

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
"""
pretraining_readiness_check.py
سكربت تشخيصي شامل: يفحص البيئة، يُقدّر الذاكرة والوقت، ويُحذّر من المشاكل
المحتملة قبل إطلاق تدريب فعلي مكلف.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReadinessReport:
    device_info: str
    num_gpus: int
    total_gpu_memory_gb: float
    estimated_required_memory_gb: float
    memory_sufficient: bool
    warnings: list[str]


def check_training_readiness(
    num_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    bytes_per_param: int = 2,  # bf16 الافتراضي الحديث
    safety_margin: float = 1.3,  # هامش أمان 30% لذاكرة Activations غير المحسوبة بدقة
) -> ReadinessReport:
    warnings: list[str] = []

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        device_info = "لا يوجد GPU متاح — سيتم التدريب على CPU (بطيء جدًا لنماذج كبيرة)"
        total_gpu_mem = 0.0
        warnings.append("⚠️ لا يوجد GPU — مناسب فقط للتجارب الصغيرة جدًا أو الاختبار")
    else:
        device_info = torch.cuda.get_device_name(0)
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    # تقدير ذاكرة الحالة الأساسية (أوزان + تدرجات + AdamW)
    base_memory_bytes = num_params * bytes_per_param * 4  # 4x: weight+grad+m+v
    base_memory_gb = base_memory_bytes / 1e9

    # تقدير تقريبي جدًا لذاكرة Activations (قاعدة تقريبية شائعة لمحوّلات)
    activations_gb = (
        batch_size * seq_len * hidden_dim * num_layers * bytes_per_param * 2
    ) / 1e9

    estimated_total = (base_memory_gb + activations_gb) * safety_margin

    if num_gpus > 0 and estimated_total > total_gpu_mem:
        warnings.append(
            f"⚠️ الذاكرة المُقدَّرة ({estimated_total:.1f}GB) تتجاوز "
            f"الذاكرة المتاحة ({total_gpu_mem:.1f}GB)! فكّر في:"
        )
        warnings.append("   - تقليل batch_size أو seq_len")
        warnings.append("   - استخدام Gradient Checkpointing")
        warnings.append("   - استخدام Gradient Accumulation بدل دفعة كبيرة واحدة")
        warnings.append("   - التدريب الموزّع عبر عدة GPUs (ZeRO/FSDP — الفصل 23)")

    if bytes_per_param == 4:
        warnings.append("💡 تلميح: تستخدم fp32 الكاملة — جرّب bf16 لتوفير نصف الذاكرة تقريبًا")

    return ReadinessReport(
        device_info=device_info,
        num_gpus=max(num_gpus, 1),
        total_gpu_memory_gb=total_gpu_mem,
        estimated_required_memory_gb=estimated_total,
        memory_sufficient=(estimated_total <= total_gpu_mem) if num_gpus > 0 else False,
        warnings=warnings,
    )


if __name__ == "__main__":
    report = check_training_readiness(
        num_params=1_000_000_000,
        batch_size=16,
        seq_len=2048,
        hidden_dim=2048,
        num_layers=24,
        bytes_per_param=2,
    )

    print(f"الجهاز: {report.device_info}")
    print(f"عدد GPUs: {report.num_gpus}")
    print(f"الذاكرة الكلية المتاحة: {report.total_gpu_memory_gb:.1f} GB")
    print(f"الذاكرة المُقدَّرة المطلوبة: {report.estimated_required_memory_gb:.1f} GB")
    print(f"هل الذاكرة كافية؟ {'✅ نعم' if report.memory_sufficient else '❌ لا'}")
    if report.warnings:
        print("\nالتحذيرات والتوصيات:")
        for w in report.warnings:
            print(f"  {w}")

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ كلاسيكي: نسيان نقل البيانات الجديدة للجهاز في كل حلقة
model = model.to(device)
for xb, yb in loader:
    output = model(xb)   # RuntimeError: xb لا يزال على CPU!

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: انقل كل دفعة بيانات للجهاز بشكل صريح في كل مرة
model = model.to(device)
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    output = model(xb)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: تخزين loss مع رسمه الحسابي الكامل يُراكم الذاكرة بلا داعٍ
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss)   # يحتفظ بكل الرسم الحسابي لكل دفعة! تسريب ذاكرة فعلي

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استخدم .item() لاستخراج القيمة العددية فقط، وتخلّص من الرسم
losses = []
for xb, yb in loader:
    loss = compute_loss(model, xb, yb)
    losses.append(loss.item())   # رقم Python عادي، لا رسم حسابي مرتبط به

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ❌ هدر للذاكرة: PyTorch يبني رسمًا حسابيًا كاملًا حتى أثناء الاستدلال فقط
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: عطّل تتبع التدرجات تمامًا عند عدم الحاجة لـ backward()
@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)

# ────────────────────────────────────────────────────────────────────────
# [019] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-18
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عمليات CUDA غير متزامنة (Asynchronous)؛ القياس سيكون غير دقيق
start = time.perf_counter()
output = model(x)
elapsed = time.perf_counter() - start   # قد يقيس فقط وقت "إطلاق" العملية، لا تنفيذها!

# ────────────────────────────────────────────────────────────────────────
# [020] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-19
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: استدعِ synchronize() قبل وبعد القياس لضمان اكتمال كل عمليات GPU فعليًا
torch.cuda.synchronize()
start = time.perf_counter()
output = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ────────────────────────────────────────────────────────────────────────
# [021] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-20
# ────────────────────────────────────────────────────────────────────────
# ❌ غير مُحسَّن: تحميل البيانات يصبح عنق الزجاجة (Bottleneck) بدل GPU
loader = DataLoader(dataset, batch_size=64)

# ────────────────────────────────────────────────────────────────────────
# [022] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-21
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: استخدم عدة عمليات لتحميل البيانات بالتوازي مع التدريب
loader = DataLoader(
    dataset, batch_size=64,
    num_workers=4,       # عمليات موازية لتحضير الدفعات التالية مسبقًا
    pin_memory=True,     # يُسرّع نقل البيانات من CPU إلى GPU
)

# ────────────────────────────────────────────────────────────────────────
# [023] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-05-chapter-05.ipynb#cell-22
# ────────────────────────────────────────────────────────────────────────
"""
training_dashboard.py — هيكل بداية للتمرين
أداة CLI شاملة لتقييم جاهزية تدريب نموذج معيّن.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelSpec:
    num_params: int
    hidden_dim: int
    num_layers: int
    seq_len: int
    batch_size: int


@dataclass(frozen=True)
class HardwareSpec:
    gpu_name: str
    num_gpus: int
    memory_per_gpu_gb: float


def build_arg_parser() -> argparse.ArgumentParser:
    """
    TODO: أنشئ argparse.ArgumentParser بالحقول التالية:
    --num-params (int, مطلوب)
    --hidden-dim (int, افتراضي 2048)
    --num-layers (int, افتراضي 24)
    --seq-len (int, افتراضي 2048)
    --batch-size (int, افتراضي 16)
    --num-tokens (int, للتدريب الكامل، مطلوب لحساب FLOPs)
    --gpu-name (str, من قائمة محددة مسبقًا)
    --num-gpus (int, افتراضي 1)
    --dtype (str، إما "fp32" أو "bf16"، افتراضي "bf16")
    """
    raise NotImplementedError


def generate_full_report(model_spec: ModelSpec, hw_spec: HardwareSpec, num_tokens: int, dtype: str) -> str:
    """
    TODO: ادمج estimate_memory() و estimate_training_time() و check_training_readiness()
    (من الأمثلة السابقة في هذا الفصل) في تقرير نصي واحد منسّق وجميل، يتضمن:
    1. ملخص المواصفات المُدخلة
    2. تقدير الذاكرة (كافية أم لا، مع تحذيرات)
    3. تقدير الوقت الإجمالي للتدريب
    4. توصيات محددة إن لزم الأمر (تقليل batch_size، تفعيل bf16، عدد GPUs إضافي مطلوب...)
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # TODO: ابنِ ModelSpec و HardwareSpec من args، واستدعِ generate_full_report، واطبع النتيجة

    # اختبر أداتك بسيناريوهات مختلفة من سطر الأوامر، مثال:
    # python training_dashboard.py --num-params 7000000000 --num-tokens 1400000000000 \
    #     --gpu-name A100_80GB --num-gpus 8 --dtype bf16
