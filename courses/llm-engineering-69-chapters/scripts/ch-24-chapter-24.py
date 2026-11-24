#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing
Large Language Model Engineering — Pretraining

Book slug : llm-engineering-69-chapters
Chapter   : ch-24-chapter-24 (lesson 24)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-24-chapter-24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing
Large Language Model Engineering — Pretraining

Book slug : llm-engineering-69-chapters
Chapter   : ch-24-chapter-24 (lesson 24)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-24-chapter-24.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing
Large Language Model Engineering — Pretraining

Book slug : llm-engineering-69-chapters
Chapter   : ch-24-chapter-24 (lesson 24)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 24: الفصل 24: حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_checkpoint() -> None:
    checkpoint = {'step': 120, 'model': {'w': [0.2, 0.6]}, 'optimizer_step': 120, 'seed': 7}
    encoded = json.dumps(checkpoint, sort_keys=True)
    restored = json.loads(encoded)
    assert restored['step'] == restored['optimizer_step']
    print('checkpoint bytes:', len(encoded.encode('utf-8')))
    print('resume step:', restored['step'], '| seed:', restored['seed'])
    print('integrity check:', restored['model']['w'] == [0.2, 0.6])

if __name__ == "__main__":
    print("Chapter 24 lab: Save, validate, and resume a training checkpoint")
    run_checkpoint()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class TrainingState:
    """حاوية موحّدة لكل ما يُشكّل "حالة التدريب الكاملة" القابلة للاستئناف.
    تجميع هذه الحقول في مكان واحد يمنع نسيان أي مكوّن ضروري عند الحفظ."""

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = global_step

    def to_dict(self) -> dict:
        # get_state_dict الحديثة في PyTorch 2.x تتعامل تلقائيًا مع تعقيدات
        # FSDP Sharding دون الحاجة لكتابة منطق تجميع يدوي معقّد
        model_sd, optim_sd = get_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )
        return {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "rng_state_cuda": torch.cuda.get_rng_state_all(),
            "rng_state_cpu": torch.get_rng_state(),
        }

    def load_from_dict(self, state: dict) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        torch.cuda.set_rng_state_all(state["rng_state_cuda"])
        torch.set_rng_state(state["rng_state_cpu"])


def save_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> None:
    """حفظ موزّع: كل جهاز يحفظ شريحته فقط، بالتوازي، إلى مجلد مشترك
    (عادة على نظام ملفات موزّع مثل Lustre أو S3-compatible storage)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = training_state.to_dict()

    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )

    if dist.get_rank() == 0:
        print(f"✅ تم حفظ نقطة تفتيش عند الخطوة {training_state.global_step} في {checkpoint_dir}")


def load_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> int:
    """استعادة موزّعة: يمكن أن تتم بعدد أجهزة مختلف تمامًا عن وقت الحفظ
    بفضل آلية Resharding التلقائية في torch.distributed.checkpoint."""
    state_dict = training_state.to_dict()  # هيكل فارغ تُملأ قيمه من الملفات

    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(str(checkpoint_dir)),
    )
    training_state.load_from_dict(state_dict)

    if dist.get_rank() == 0:
        print(f"✅ تم استئناف التدريب من الخطوة {training_state.global_step}")
    return training_state.global_step

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn


def clip_gradients_and_report(
    model: nn.Module, max_norm: float = 1.0
) -> float:
    """تقليم التدرّجات وإرجاع المعيار الفعلي قبل التقليم — هذا الرقم
    هو أحد أهم المقاييس للمراقبة، لأن قفزاته المفاجئة تُنذر بالخطر مبكرًا."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return total_norm.item()


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """جدولة شائعة جدًا في تدريب LLMs: Warmup خطي ثم تناقص جيبي-تجيبي (Cosine Decay).
    هذا هو النمط المستخدم فعليًا في GPT-3، LLaMA، وأغلب النماذج الحديثة."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # نمو خطي من قيمة قريبة من الصفر حتى 1.0
                return current_step / max(1, warmup_steps)

            # بعد الـWarmup: تناقص جيبي-تجيبي نحو min_lr_ratio
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda=lr_lambda)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import deque


class LossSpikeDetector:
    """كاشف قفزات الخسارة المفاجئة بناءً على متوسط متحرك وانحراف معياري،
    مع آلية لإصدار إشارة "توقف واستعادة" عند الحاجة."""

    def __init__(self, window_size: int = 100, spike_threshold_std: float = 6.0) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.spike_threshold_std = spike_threshold_std

    def update_and_check(self, current_loss: float) -> bool:
        """يرجع True إذا كانت الخسارة الحالية "قفزة شاذة" تستدعي تدخلاً."""
        if len(self.window) < 20:  # لا نحكم إلا بعد تجميع عيّنات كافية
            self.window.append(current_loss)
            return False

        mean_loss = sum(self.window) / len(self.window)
        variance = sum((x - mean_loss) ** 2 for x in self.window) / len(self.window)
        std_loss = variance ** 0.5

        is_spike = (
            current_loss > mean_loss + self.spike_threshold_std * std_loss
            or current_loss != current_loss  # فحص NaN (NaN != NaN دائمًا True)
        )

        if not is_spike:
            self.window.append(current_loss)  # لا نُلوّث النافذة بقيم شاذة

        return is_spike

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn


class TrainingMonitor:
    """جامع مقاييس مبسّط يمكن ربطه بـ Weights & Biases أو TensorBoard
    في بيئة إنتاجية حقيقية. هنا نُبقيه مستقلاً لتوضيح المنطق الأساسي."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
        }

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        self.history["loss"].append(loss)
        self.history["grad_norm"].append(grad_norm)
        self.history["learning_rate"].append(learning_rate)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }

        # مقاييس اختيارية أعمق: معايير التفعيلات لكل طبقة (مكلفة حسابيًا
        # نسبيًا، لذا تُحسب أحيانًا فقط كل N خطوة وليس كل خطوة)
        if model is not None:
            param_norms = {
                f"param_norm/{name}": param.detach().norm().item()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            # في بيئة حقيقية: نُرسل هذا لنظام مراقبة بدل طباعته فقط
            metrics.update(
                {k: v for k, v in list(param_norms.items())[:3]}  # عيّنة توضيحية فقط
            )

        return metrics

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlock(nn.Module):
    """كتلة Transformer تستخدم Activation Checkpointing: التفعيلات الداخلية
    للكتلة لا تُخزَّن، بل تُعاد حسابها عند الحاجة أثناء Backward."""

    def __init__(self, d_model: int, n_heads: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """المنطق الفعلي للكتلة، مفصول في دالة مستقلة ليسهل تمريره
        إلى torch.utils.checkpoint.checkpoint."""
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # use_reentrant=False هو الخيار الموصى به حديثًا في PyTorch
            # (أكثر توافقًا مع الميزات الحديثة مثل torch.compile وDDP/FSDP)
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class SelectiveCheckpointing:
    """استراتيجية حديثة أكثر دقة: بدلاً من تطبيق Checkpointing على كل
    الكتل بشكل متساوٍ، نُطبّقه فقط على العمليات "الرخيصة لإعادة الحساب
    لكن المكلفة في الذاكرة" (مثل الـMatmul)، ونتجنّبه على العمليات
    "المكلفة لإعادة حسابها" (مثل Attention نفسها في بعض التطبيقات).
    هذا هو النمط المستخدم في torch.compile + torch.utils.checkpoint الحديثة
    عبر context managers متخصصة مثل create_selective_checkpoint_contexts."""

    @staticmethod
    def estimate_memory_savings(
        num_layers: int, seq_len: int, d_model: int, batch_size: int, bytes_per_elem: int = 2
    ) -> dict[str, float]:
        """تقدير تقريبي لتوفير الذاكرة الناتج عن Full Checkpointing."""
        activation_size_per_layer = seq_len * d_model * batch_size * bytes_per_elem

        without_checkpointing = num_layers * activation_size_per_layer
        # تقريب شائع: فقط sqrt(L) من نقاط التفتيش تُحفَظ فعليًا
        with_checkpointing = (num_layers ** 0.5) * activation_size_per_layer

        return {
            "بدون_checkpointing_GB": without_checkpointing / 1e9,
            "مع_checkpointing_GB": with_checkpointing / 1e9,
            "نسبة_التوفير": 1 - (with_checkpointing / without_checkpointing),
        }


if __name__ == "__main__":
    savings = SelectiveCheckpointing.estimate_memory_savings(
        num_layers=80, seq_len=4096, d_model=8192, batch_size=4
    )
    for key, value in savings.items():
        print(f"{key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def production_pretraining_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    total_steps: int,
    checkpoint_dir: Path,
    checkpoint_every: int = 500,
    grad_clip_norm: float = 1.0,
    rank: int = 0,
) -> None:
    """حلقة تدريب تجمع: Checkpointing دوري + كشف Loss Spikes + Gradient
    Clipping + مراقبة شاملة — هيكل مطابق لما يُستخدم فعليًا في تدريب LLMs
    على نطاق واسع، مع تبسيط بعض التفاصيل الموزّعة لأغراض التوضيح."""

    monitor = TrainingMonitor()
    spike_detector = LossSpikeDetector(window_size=100, spike_threshold_std=6.0)
    global_step = 0

    # محاولة استئناف تلقائي إن وُجدت نقطة تفتيش سابقة
    latest_ckpt = checkpoint_dir / "latest"
    if latest_ckpt.exists():
        training_state = TrainingState(model, optimizer, scheduler, global_step)
        global_step = load_checkpoint(training_state, latest_ckpt)

    data_iterator = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)

        input_ids, labels = batch
        input_ids, labels = input_ids.cuda(), labels.cuda()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()

        # تقليم التدرّجات قبل الخطوة — خط الدفاع الأول ضد الانفجار
        grad_norm = clip_gradients_and_report(model, max_norm=grad_clip_norm)

        current_loss = loss.item()

        # كشف القفزات — خط الدفاع الثاني: التدخل عند الحاجة
        if spike_detector.update_and_check(current_loss):
            if rank == 0:
                print(
                    f"⚠️ تم رصد Loss Spike عند الخطوة {global_step}: "
                    f"loss={current_loss:.4f}. إيقاف الخطوة واستعادة آخر نقطة سليمة."
                )
            # تخطّي هذه الخطوة كليًا (لا نطبّق optimizer.step() على تدرّج مسمم)
            # في نظام إنتاجي حقيقي: نستعيد هنا آخر checkpoint ونتخطى هذه الدفعة بعينها
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()
        global_step += 1

        if rank == 0 and global_step % 10 == 0:
            metrics = monitor.log_step(
                loss=current_loss,
                grad_norm=grad_norm,
                learning_rate=scheduler.get_last_lr()[0],
            )
            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"GradNorm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['learning_rate']:.2e}"
            )

        if global_step % checkpoint_every == 0:
            training_state = TrainingState(model, optimizer, scheduler, global_step)
            step_ckpt_dir = checkpoint_dir / f"step_{global_step}"
            save_checkpoint(training_state, step_ckpt_dir)
            # تحديث "آخر نقطة" دون الحاجة لإعادة الحفظ الكامل (رابط رمزي مثلاً)
            # في نظام ملفات حقيقي: يُفضَّل استخدام symlink أو نسخ خفيف الوزن

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عند الاستئناف، يبدأ Data Loader من الصفر دائمًا،
# مما يعني أن النموذج سيرى نفس البيانات المبكرة مرة أخرى بشكل غير مقصود
for epoch in range(epochs):
    for batch in train_loader:
        ...

# ✅ صحيح: تتبّع صريح لعدد الدفعات المُعالَجة واستئناف منطقي
# (في الممارسة العملية: تُستخدم أدوات متخصصة مثل StatefulDataLoader
# من TorchData، أو تُحسَب العيّنات المتخطّاة يدويًا عند الاستئناف)
batches_to_skip = global_step % len(train_loader)
for i, batch in enumerate(train_loader):
    if i < batches_to_skip:
        continue
    ...

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع عند الانتقال من جهاز واحد إلى موزّع: حساب grad_norm
# محليًا فقط دون اعتبار أن DDP يُجري All-Reduce تلقائيًا بعد backward()،
# مما قد يؤدي لقص غير متسق إذا تم استدعاء clip_grad_norm_ في توقيت خاطئ
loss.backward()
# بعض الأكواد الخاطئة تستدعي clip هنا قبل اكتمال المزامنة الفعلية

# ✅ صحيح: clip_grad_norm_ يجب أن يُستدعى دائمًا بعد backward() الكامل
# (الذي يتضمن المزامنة عبر DDP/FSDP تلقائيًا) وقبل optimizer.step() مباشرة
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: FP16 بدون GradScaler يؤدي لـUnderflow/Overflow متكرر وOnan
scaler = None
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x)
loss.backward()  # قد ينتج تدرّجات صفرية أو NaN بصمت!

# ✅ أفضل: استخدم BF16 إن كان العتاد يدعمه (A100+) — لا حاجة حتى لـGradScaler
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()  # BF16 لديه نفس المدى الديناميكي لـ FP32، استقرار أعلى بكثير

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: محاكاة ومعالجة Loss Spike في حلقة تدريب حقيقية

المطلوب إكماله:
1. بناء نموذج صغير (MLP أو Transformer صغير) وبيانات تدريب اصطناعية طبيعية
2. حقن "عيّنات سامة" بشكل متعمّد كل عدد معين من الخطوات (قيم متطرفة إحصائيًا)
   يمكنك ضرب بعض عيّنات الإدخال في عامل كبير جدًا (مثلاً × 1000) لمحاكاة شذوذ البيانات
3. تشغيل حلقتين تدريب متوازيتين للمقارنة:
   أ) حلقة "ساذجة" بدون أي حماية (بدون Clipping، بدون كاشف Spikes)
   ب) حلقة "محمية" بكل الآليات المذكورة في هذا الفصل
4. رسم بياني لمنحنى الخسارة لكلتا الحلقتين على نفس الرسم البياني
5. قياس: هل انهارت الحلقة الساذجة فعلاً إلى NaN؟ هل نجت الحلقة المحمية؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def inject_poisoned_batch(
    clean_batch: torch.Tensor, poison_scale: float = 1000.0
) -> torch.Tensor:
    """TODO: أرجع نسخة من الدفعة بعد ضرب عيّنة واحدة عشوائية فيها
    بعامل poison_scale، لمحاكاة بيانات شاذة إحصائيًا من خط أنابيب حقيقي."""
    raise NotImplementedError


def run_naive_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: حلقة تدريب بلا أي حماية — يجب أن تُظهر انهيارًا فعليًا (NaN)
    عند نقاط حقن السموم. أرجع قائمة بقيم الخسارة عبر الزمن."""
    raise NotImplementedError


def run_protected_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: نفس السيناريو، لكن استخدم:
    - clip_gradients_and_report لتقليم التدرّجات
    - LossSpikeDetector لرصد القفزات وتخطي الخطوات المسمومة
    يجب أن تنجو هذه الحلقة من الانهيار رغم نفس البيانات المسمومة بالضبط."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: قارن المسارين بصريًا ورقميًا، واستخلص استنتاجك الخاص
    # حول فعالية كل آلية حماية على حدة (جرّب تعطيل Clipping فقط،
    # ثم تعطيل Spike Detection فقط، لمعرفة مساهمة كل منهما بمفرده)
    pass

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 24: الفصل 24: حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_checkpoint() -> None:
    checkpoint = {'step': 120, 'model': {'w': [0.2, 0.6]}, 'optimizer_step': 120, 'seed': 7}
    encoded = json.dumps(checkpoint, sort_keys=True)
    restored = json.loads(encoded)
    assert restored['step'] == restored['optimizer_step']
    print('checkpoint bytes:', len(encoded.encode('utf-8')))
    print('resume step:', restored['step'], '| seed:', restored['seed'])
    print('integrity check:', restored['model']['w'] == [0.2, 0.6])

if __name__ == "__main__":
    print("Chapter 24 lab: Save, validate, and resume a training checkpoint")
    run_checkpoint()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class TrainingState:
    """حاوية موحّدة لكل ما يُشكّل "حالة التدريب الكاملة" القابلة للاستئناف.
    تجميع هذه الحقول في مكان واحد يمنع نسيان أي مكوّن ضروري عند الحفظ."""

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = global_step

    def to_dict(self) -> dict:
        # get_state_dict الحديثة في PyTorch 2.x تتعامل تلقائيًا مع تعقيدات
        # FSDP Sharding دون الحاجة لكتابة منطق تجميع يدوي معقّد
        model_sd, optim_sd = get_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )
        return {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "rng_state_cuda": torch.cuda.get_rng_state_all(),
            "rng_state_cpu": torch.get_rng_state(),
        }

    def load_from_dict(self, state: dict) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        torch.cuda.set_rng_state_all(state["rng_state_cuda"])
        torch.set_rng_state(state["rng_state_cpu"])


def save_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> None:
    """حفظ موزّع: كل جهاز يحفظ شريحته فقط، بالتوازي، إلى مجلد مشترك
    (عادة على نظام ملفات موزّع مثل Lustre أو S3-compatible storage)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = training_state.to_dict()

    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )

    if dist.get_rank() == 0:
        print(f"✅ تم حفظ نقطة تفتيش عند الخطوة {training_state.global_step} في {checkpoint_dir}")


def load_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> int:
    """استعادة موزّعة: يمكن أن تتم بعدد أجهزة مختلف تمامًا عن وقت الحفظ
    بفضل آلية Resharding التلقائية في torch.distributed.checkpoint."""
    state_dict = training_state.to_dict()  # هيكل فارغ تُملأ قيمه من الملفات

    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(str(checkpoint_dir)),
    )
    training_state.load_from_dict(state_dict)

    if dist.get_rank() == 0:
        print(f"✅ تم استئناف التدريب من الخطوة {training_state.global_step}")
    return training_state.global_step

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn


def clip_gradients_and_report(
    model: nn.Module, max_norm: float = 1.0
) -> float:
    """تقليم التدرّجات وإرجاع المعيار الفعلي قبل التقليم — هذا الرقم
    هو أحد أهم المقاييس للمراقبة، لأن قفزاته المفاجئة تُنذر بالخطر مبكرًا."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return total_norm.item()


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """جدولة شائعة جدًا في تدريب LLMs: Warmup خطي ثم تناقص جيبي-تجيبي (Cosine Decay).
    هذا هو النمط المستخدم فعليًا في GPT-3، LLaMA، وأغلب النماذج الحديثة."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # نمو خطي من قيمة قريبة من الصفر حتى 1.0
                return current_step / max(1, warmup_steps)

            # بعد الـWarmup: تناقص جيبي-تجيبي نحو min_lr_ratio
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda=lr_lambda)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import deque


class LossSpikeDetector:
    """كاشف قفزات الخسارة المفاجئة بناءً على متوسط متحرك وانحراف معياري،
    مع آلية لإصدار إشارة "توقف واستعادة" عند الحاجة."""

    def __init__(self, window_size: int = 100, spike_threshold_std: float = 6.0) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.spike_threshold_std = spike_threshold_std

    def update_and_check(self, current_loss: float) -> bool:
        """يرجع True إذا كانت الخسارة الحالية "قفزة شاذة" تستدعي تدخلاً."""
        if len(self.window) < 20:  # لا نحكم إلا بعد تجميع عيّنات كافية
            self.window.append(current_loss)
            return False

        mean_loss = sum(self.window) / len(self.window)
        variance = sum((x - mean_loss) ** 2 for x in self.window) / len(self.window)
        std_loss = variance ** 0.5

        is_spike = (
            current_loss > mean_loss + self.spike_threshold_std * std_loss
            or current_loss != current_loss  # فحص NaN (NaN != NaN دائمًا True)
        )

        if not is_spike:
            self.window.append(current_loss)  # لا نُلوّث النافذة بقيم شاذة

        return is_spike

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn


class TrainingMonitor:
    """جامع مقاييس مبسّط يمكن ربطه بـ Weights & Biases أو TensorBoard
    في بيئة إنتاجية حقيقية. هنا نُبقيه مستقلاً لتوضيح المنطق الأساسي."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
        }

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        self.history["loss"].append(loss)
        self.history["grad_norm"].append(grad_norm)
        self.history["learning_rate"].append(learning_rate)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }

        # مقاييس اختيارية أعمق: معايير التفعيلات لكل طبقة (مكلفة حسابيًا
        # نسبيًا، لذا تُحسب أحيانًا فقط كل N خطوة وليس كل خطوة)
        if model is not None:
            param_norms = {
                f"param_norm/{name}": param.detach().norm().item()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            # في بيئة حقيقية: نُرسل هذا لنظام مراقبة بدل طباعته فقط
            metrics.update(
                {k: v for k, v in list(param_norms.items())[:3]}  # عيّنة توضيحية فقط
            )

        return metrics

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlock(nn.Module):
    """كتلة Transformer تستخدم Activation Checkpointing: التفعيلات الداخلية
    للكتلة لا تُخزَّن، بل تُعاد حسابها عند الحاجة أثناء Backward."""

    def __init__(self, d_model: int, n_heads: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """المنطق الفعلي للكتلة، مفصول في دالة مستقلة ليسهل تمريره
        إلى torch.utils.checkpoint.checkpoint."""
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # use_reentrant=False هو الخيار الموصى به حديثًا في PyTorch
            # (أكثر توافقًا مع الميزات الحديثة مثل torch.compile وDDP/FSDP)
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class SelectiveCheckpointing:
    """استراتيجية حديثة أكثر دقة: بدلاً من تطبيق Checkpointing على كل
    الكتل بشكل متساوٍ، نُطبّقه فقط على العمليات "الرخيصة لإعادة الحساب
    لكن المكلفة في الذاكرة" (مثل الـMatmul)، ونتجنّبه على العمليات
    "المكلفة لإعادة حسابها" (مثل Attention نفسها في بعض التطبيقات).
    هذا هو النمط المستخدم في torch.compile + torch.utils.checkpoint الحديثة
    عبر context managers متخصصة مثل create_selective_checkpoint_contexts."""

    @staticmethod
    def estimate_memory_savings(
        num_layers: int, seq_len: int, d_model: int, batch_size: int, bytes_per_elem: int = 2
    ) -> dict[str, float]:
        """تقدير تقريبي لتوفير الذاكرة الناتج عن Full Checkpointing."""
        activation_size_per_layer = seq_len * d_model * batch_size * bytes_per_elem

        without_checkpointing = num_layers * activation_size_per_layer
        # تقريب شائع: فقط sqrt(L) من نقاط التفتيش تُحفَظ فعليًا
        with_checkpointing = (num_layers ** 0.5) * activation_size_per_layer

        return {
            "بدون_checkpointing_GB": without_checkpointing / 1e9,
            "مع_checkpointing_GB": with_checkpointing / 1e9,
            "نسبة_التوفير": 1 - (with_checkpointing / without_checkpointing),
        }


if __name__ == "__main__":
    savings = SelectiveCheckpointing.estimate_memory_savings(
        num_layers=80, seq_len=4096, d_model=8192, batch_size=4
    )
    for key, value in savings.items():
        print(f"{key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def production_pretraining_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    total_steps: int,
    checkpoint_dir: Path,
    checkpoint_every: int = 500,
    grad_clip_norm: float = 1.0,
    rank: int = 0,
) -> None:
    """حلقة تدريب تجمع: Checkpointing دوري + كشف Loss Spikes + Gradient
    Clipping + مراقبة شاملة — هيكل مطابق لما يُستخدم فعليًا في تدريب LLMs
    على نطاق واسع، مع تبسيط بعض التفاصيل الموزّعة لأغراض التوضيح."""

    monitor = TrainingMonitor()
    spike_detector = LossSpikeDetector(window_size=100, spike_threshold_std=6.0)
    global_step = 0

    # محاولة استئناف تلقائي إن وُجدت نقطة تفتيش سابقة
    latest_ckpt = checkpoint_dir / "latest"
    if latest_ckpt.exists():
        training_state = TrainingState(model, optimizer, scheduler, global_step)
        global_step = load_checkpoint(training_state, latest_ckpt)

    data_iterator = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)

        input_ids, labels = batch
        input_ids, labels = input_ids.cuda(), labels.cuda()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()

        # تقليم التدرّجات قبل الخطوة — خط الدفاع الأول ضد الانفجار
        grad_norm = clip_gradients_and_report(model, max_norm=grad_clip_norm)

        current_loss = loss.item()

        # كشف القفزات — خط الدفاع الثاني: التدخل عند الحاجة
        if spike_detector.update_and_check(current_loss):
            if rank == 0:
                print(
                    f"⚠️ تم رصد Loss Spike عند الخطوة {global_step}: "
                    f"loss={current_loss:.4f}. إيقاف الخطوة واستعادة آخر نقطة سليمة."
                )
            # تخطّي هذه الخطوة كليًا (لا نطبّق optimizer.step() على تدرّج مسمم)
            # في نظام إنتاجي حقيقي: نستعيد هنا آخر checkpoint ونتخطى هذه الدفعة بعينها
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()
        global_step += 1

        if rank == 0 and global_step % 10 == 0:
            metrics = monitor.log_step(
                loss=current_loss,
                grad_norm=grad_norm,
                learning_rate=scheduler.get_last_lr()[0],
            )
            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"GradNorm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['learning_rate']:.2e}"
            )

        if global_step % checkpoint_every == 0:
            training_state = TrainingState(model, optimizer, scheduler, global_step)
            step_ckpt_dir = checkpoint_dir / f"step_{global_step}"
            save_checkpoint(training_state, step_ckpt_dir)
            # تحديث "آخر نقطة" دون الحاجة لإعادة الحفظ الكامل (رابط رمزي مثلاً)
            # في نظام ملفات حقيقي: يُفضَّل استخدام symlink أو نسخ خفيف الوزن

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عند الاستئناف، يبدأ Data Loader من الصفر دائمًا،
# مما يعني أن النموذج سيرى نفس البيانات المبكرة مرة أخرى بشكل غير مقصود
for epoch in range(epochs):
    for batch in train_loader:
        ...

# ✅ صحيح: تتبّع صريح لعدد الدفعات المُعالَجة واستئناف منطقي
# (في الممارسة العملية: تُستخدم أدوات متخصصة مثل StatefulDataLoader
# من TorchData، أو تُحسَب العيّنات المتخطّاة يدويًا عند الاستئناف)
batches_to_skip = global_step % len(train_loader)
for i, batch in enumerate(train_loader):
    if i < batches_to_skip:
        continue
    ...

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع عند الانتقال من جهاز واحد إلى موزّع: حساب grad_norm
# محليًا فقط دون اعتبار أن DDP يُجري All-Reduce تلقائيًا بعد backward()،
# مما قد يؤدي لقص غير متسق إذا تم استدعاء clip_grad_norm_ في توقيت خاطئ
loss.backward()
# بعض الأكواد الخاطئة تستدعي clip هنا قبل اكتمال المزامنة الفعلية

# ✅ صحيح: clip_grad_norm_ يجب أن يُستدعى دائمًا بعد backward() الكامل
# (الذي يتضمن المزامنة عبر DDP/FSDP تلقائيًا) وقبل optimizer.step() مباشرة
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: FP16 بدون GradScaler يؤدي لـUnderflow/Overflow متكرر وOnan
scaler = None
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x)
loss.backward()  # قد ينتج تدرّجات صفرية أو NaN بصمت!

# ✅ أفضل: استخدم BF16 إن كان العتاد يدعمه (A100+) — لا حاجة حتى لـGradScaler
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()  # BF16 لديه نفس المدى الديناميكي لـ FP32، استقرار أعلى بكثير

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: محاكاة ومعالجة Loss Spike في حلقة تدريب حقيقية

المطلوب إكماله:
1. بناء نموذج صغير (MLP أو Transformer صغير) وبيانات تدريب اصطناعية طبيعية
2. حقن "عيّنات سامة" بشكل متعمّد كل عدد معين من الخطوات (قيم متطرفة إحصائيًا)
   يمكنك ضرب بعض عيّنات الإدخال في عامل كبير جدًا (مثلاً × 1000) لمحاكاة شذوذ البيانات
3. تشغيل حلقتين تدريب متوازيتين للمقارنة:
   أ) حلقة "ساذجة" بدون أي حماية (بدون Clipping، بدون كاشف Spikes)
   ب) حلقة "محمية" بكل الآليات المذكورة في هذا الفصل
4. رسم بياني لمنحنى الخسارة لكلتا الحلقتين على نفس الرسم البياني
5. قياس: هل انهارت الحلقة الساذجة فعلاً إلى NaN؟ هل نجت الحلقة المحمية؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def inject_poisoned_batch(
    clean_batch: torch.Tensor, poison_scale: float = 1000.0
) -> torch.Tensor:
    """TODO: أرجع نسخة من الدفعة بعد ضرب عيّنة واحدة عشوائية فيها
    بعامل poison_scale، لمحاكاة بيانات شاذة إحصائيًا من خط أنابيب حقيقي."""
    raise NotImplementedError


def run_naive_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: حلقة تدريب بلا أي حماية — يجب أن تُظهر انهيارًا فعليًا (NaN)
    عند نقاط حقن السموم. أرجع قائمة بقيم الخسارة عبر الزمن."""
    raise NotImplementedError


def run_protected_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: نفس السيناريو، لكن استخدم:
    - clip_gradients_and_report لتقليم التدرّجات
    - LossSpikeDetector لرصد القفزات وتخطي الخطوات المسمومة
    يجب أن تنجو هذه الحلقة من الانهيار رغم نفس البيانات المسمومة بالضبط."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: قارن المسارين بصريًا ورقميًا، واستخلص استنتاجك الخاص
    # حول فعالية كل آلية حماية على حدة (جرّب تعطيل Clipping فقط،
    # ثم تعطيل Spike Detection فقط، لمعرفة مساهمة كل منهما بمفرده)
    pass

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing
Large Language Model Engineering — Pretraining

Book slug : llm-engineering-69-chapters
Chapter   : ch-24-chapter-24 (lesson 24)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_24.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 24: الفصل 24: حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_checkpoint() -> None:
    checkpoint = {'step': 120, 'model': {'w': [0.2, 0.6]}, 'optimizer_step': 120, 'seed': 7}
    encoded = json.dumps(checkpoint, sort_keys=True)
    restored = json.loads(encoded)
    assert restored['step'] == restored['optimizer_step']
    print('checkpoint bytes:', len(encoded.encode('utf-8')))
    print('resume step:', restored['step'], '| seed:', restored['seed'])
    print('integrity check:', restored['model']['w'] == [0.2, 0.6])

if __name__ == "__main__":
    print("Chapter 24 lab: Save, validate, and resume a training checkpoint")
    run_checkpoint()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class TrainingState:
    """حاوية موحّدة لكل ما يُشكّل "حالة التدريب الكاملة" القابلة للاستئناف.
    تجميع هذه الحقول في مكان واحد يمنع نسيان أي مكوّن ضروري عند الحفظ."""

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = global_step

    def to_dict(self) -> dict:
        # get_state_dict الحديثة في PyTorch 2.x تتعامل تلقائيًا مع تعقيدات
        # FSDP Sharding دون الحاجة لكتابة منطق تجميع يدوي معقّد
        model_sd, optim_sd = get_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )
        return {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "rng_state_cuda": torch.cuda.get_rng_state_all(),
            "rng_state_cpu": torch.get_rng_state(),
        }

    def load_from_dict(self, state: dict) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        torch.cuda.set_rng_state_all(state["rng_state_cuda"])
        torch.set_rng_state(state["rng_state_cpu"])


def save_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> None:
    """حفظ موزّع: كل جهاز يحفظ شريحته فقط، بالتوازي، إلى مجلد مشترك
    (عادة على نظام ملفات موزّع مثل Lustre أو S3-compatible storage)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = training_state.to_dict()

    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )

    if dist.get_rank() == 0:
        print(f"✅ تم حفظ نقطة تفتيش عند الخطوة {training_state.global_step} في {checkpoint_dir}")


def load_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> int:
    """استعادة موزّعة: يمكن أن تتم بعدد أجهزة مختلف تمامًا عن وقت الحفظ
    بفضل آلية Resharding التلقائية في torch.distributed.checkpoint."""
    state_dict = training_state.to_dict()  # هيكل فارغ تُملأ قيمه من الملفات

    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(str(checkpoint_dir)),
    )
    training_state.load_from_dict(state_dict)

    if dist.get_rank() == 0:
        print(f"✅ تم استئناف التدريب من الخطوة {training_state.global_step}")
    return training_state.global_step

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn


def clip_gradients_and_report(
    model: nn.Module, max_norm: float = 1.0
) -> float:
    """تقليم التدرّجات وإرجاع المعيار الفعلي قبل التقليم — هذا الرقم
    هو أحد أهم المقاييس للمراقبة، لأن قفزاته المفاجئة تُنذر بالخطر مبكرًا."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return total_norm.item()


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """جدولة شائعة جدًا في تدريب LLMs: Warmup خطي ثم تناقص جيبي-تجيبي (Cosine Decay).
    هذا هو النمط المستخدم فعليًا في GPT-3، LLaMA، وأغلب النماذج الحديثة."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # نمو خطي من قيمة قريبة من الصفر حتى 1.0
                return current_step / max(1, warmup_steps)

            # بعد الـWarmup: تناقص جيبي-تجيبي نحو min_lr_ratio
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda=lr_lambda)

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import deque


class LossSpikeDetector:
    """كاشف قفزات الخسارة المفاجئة بناءً على متوسط متحرك وانحراف معياري،
    مع آلية لإصدار إشارة "توقف واستعادة" عند الحاجة."""

    def __init__(self, window_size: int = 100, spike_threshold_std: float = 6.0) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.spike_threshold_std = spike_threshold_std

    def update_and_check(self, current_loss: float) -> bool:
        """يرجع True إذا كانت الخسارة الحالية "قفزة شاذة" تستدعي تدخلاً."""
        if len(self.window) < 20:  # لا نحكم إلا بعد تجميع عيّنات كافية
            self.window.append(current_loss)
            return False

        mean_loss = sum(self.window) / len(self.window)
        variance = sum((x - mean_loss) ** 2 for x in self.window) / len(self.window)
        std_loss = variance ** 0.5

        is_spike = (
            current_loss > mean_loss + self.spike_threshold_std * std_loss
            or current_loss != current_loss  # فحص NaN (NaN != NaN دائمًا True)
        )

        if not is_spike:
            self.window.append(current_loss)  # لا نُلوّث النافذة بقيم شاذة

        return is_spike

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn


class TrainingMonitor:
    """جامع مقاييس مبسّط يمكن ربطه بـ Weights & Biases أو TensorBoard
    في بيئة إنتاجية حقيقية. هنا نُبقيه مستقلاً لتوضيح المنطق الأساسي."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
        }

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        self.history["loss"].append(loss)
        self.history["grad_norm"].append(grad_norm)
        self.history["learning_rate"].append(learning_rate)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }

        # مقاييس اختيارية أعمق: معايير التفعيلات لكل طبقة (مكلفة حسابيًا
        # نسبيًا، لذا تُحسب أحيانًا فقط كل N خطوة وليس كل خطوة)
        if model is not None:
            param_norms = {
                f"param_norm/{name}": param.detach().norm().item()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            # في بيئة حقيقية: نُرسل هذا لنظام مراقبة بدل طباعته فقط
            metrics.update(
                {k: v for k, v in list(param_norms.items())[:3]}  # عيّنة توضيحية فقط
            )

        return metrics

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlock(nn.Module):
    """كتلة Transformer تستخدم Activation Checkpointing: التفعيلات الداخلية
    للكتلة لا تُخزَّن، بل تُعاد حسابها عند الحاجة أثناء Backward."""

    def __init__(self, d_model: int, n_heads: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """المنطق الفعلي للكتلة، مفصول في دالة مستقلة ليسهل تمريره
        إلى torch.utils.checkpoint.checkpoint."""
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # use_reentrant=False هو الخيار الموصى به حديثًا في PyTorch
            # (أكثر توافقًا مع الميزات الحديثة مثل torch.compile وDDP/FSDP)
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class SelectiveCheckpointing:
    """استراتيجية حديثة أكثر دقة: بدلاً من تطبيق Checkpointing على كل
    الكتل بشكل متساوٍ، نُطبّقه فقط على العمليات "الرخيصة لإعادة الحساب
    لكن المكلفة في الذاكرة" (مثل الـMatmul)، ونتجنّبه على العمليات
    "المكلفة لإعادة حسابها" (مثل Attention نفسها في بعض التطبيقات).
    هذا هو النمط المستخدم في torch.compile + torch.utils.checkpoint الحديثة
    عبر context managers متخصصة مثل create_selective_checkpoint_contexts."""

    @staticmethod
    def estimate_memory_savings(
        num_layers: int, seq_len: int, d_model: int, batch_size: int, bytes_per_elem: int = 2
    ) -> dict[str, float]:
        """تقدير تقريبي لتوفير الذاكرة الناتج عن Full Checkpointing."""
        activation_size_per_layer = seq_len * d_model * batch_size * bytes_per_elem

        without_checkpointing = num_layers * activation_size_per_layer
        # تقريب شائع: فقط sqrt(L) من نقاط التفتيش تُحفَظ فعليًا
        with_checkpointing = (num_layers ** 0.5) * activation_size_per_layer

        return {
            "بدون_checkpointing_GB": without_checkpointing / 1e9,
            "مع_checkpointing_GB": with_checkpointing / 1e9,
            "نسبة_التوفير": 1 - (with_checkpointing / without_checkpointing),
        }


if __name__ == "__main__":
    savings = SelectiveCheckpointing.estimate_memory_savings(
        num_layers=80, seq_len=4096, d_model=8192, batch_size=4
    )
    for key, value in savings.items():
        print(f"{key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def production_pretraining_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    total_steps: int,
    checkpoint_dir: Path,
    checkpoint_every: int = 500,
    grad_clip_norm: float = 1.0,
    rank: int = 0,
) -> None:
    """حلقة تدريب تجمع: Checkpointing دوري + كشف Loss Spikes + Gradient
    Clipping + مراقبة شاملة — هيكل مطابق لما يُستخدم فعليًا في تدريب LLMs
    على نطاق واسع، مع تبسيط بعض التفاصيل الموزّعة لأغراض التوضيح."""

    monitor = TrainingMonitor()
    spike_detector = LossSpikeDetector(window_size=100, spike_threshold_std=6.0)
    global_step = 0

    # محاولة استئناف تلقائي إن وُجدت نقطة تفتيش سابقة
    latest_ckpt = checkpoint_dir / "latest"
    if latest_ckpt.exists():
        training_state = TrainingState(model, optimizer, scheduler, global_step)
        global_step = load_checkpoint(training_state, latest_ckpt)

    data_iterator = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)

        input_ids, labels = batch
        input_ids, labels = input_ids.cuda(), labels.cuda()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()

        # تقليم التدرّجات قبل الخطوة — خط الدفاع الأول ضد الانفجار
        grad_norm = clip_gradients_and_report(model, max_norm=grad_clip_norm)

        current_loss = loss.item()

        # كشف القفزات — خط الدفاع الثاني: التدخل عند الحاجة
        if spike_detector.update_and_check(current_loss):
            if rank == 0:
                print(
                    f"⚠️ تم رصد Loss Spike عند الخطوة {global_step}: "
                    f"loss={current_loss:.4f}. إيقاف الخطوة واستعادة آخر نقطة سليمة."
                )
            # تخطّي هذه الخطوة كليًا (لا نطبّق optimizer.step() على تدرّج مسمم)
            # في نظام إنتاجي حقيقي: نستعيد هنا آخر checkpoint ونتخطى هذه الدفعة بعينها
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()
        global_step += 1

        if rank == 0 and global_step % 10 == 0:
            metrics = monitor.log_step(
                loss=current_loss,
                grad_norm=grad_norm,
                learning_rate=scheduler.get_last_lr()[0],
            )
            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"GradNorm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['learning_rate']:.2e}"
            )

        if global_step % checkpoint_every == 0:
            training_state = TrainingState(model, optimizer, scheduler, global_step)
            step_ckpt_dir = checkpoint_dir / f"step_{global_step}"
            save_checkpoint(training_state, step_ckpt_dir)
            # تحديث "آخر نقطة" دون الحاجة لإعادة الحفظ الكامل (رابط رمزي مثلاً)
            # في نظام ملفات حقيقي: يُفضَّل استخدام symlink أو نسخ خفيف الوزن

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عند الاستئناف، يبدأ Data Loader من الصفر دائمًا،
# مما يعني أن النموذج سيرى نفس البيانات المبكرة مرة أخرى بشكل غير مقصود
for epoch in range(epochs):
    for batch in train_loader:
        ...

# ✅ صحيح: تتبّع صريح لعدد الدفعات المُعالَجة واستئناف منطقي
# (في الممارسة العملية: تُستخدم أدوات متخصصة مثل StatefulDataLoader
# من TorchData، أو تُحسَب العيّنات المتخطّاة يدويًا عند الاستئناف)
batches_to_skip = global_step % len(train_loader)
for i, batch in enumerate(train_loader):
    if i < batches_to_skip:
        continue
    ...

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع عند الانتقال من جهاز واحد إلى موزّع: حساب grad_norm
# محليًا فقط دون اعتبار أن DDP يُجري All-Reduce تلقائيًا بعد backward()،
# مما قد يؤدي لقص غير متسق إذا تم استدعاء clip_grad_norm_ في توقيت خاطئ
loss.backward()
# بعض الأكواد الخاطئة تستدعي clip هنا قبل اكتمال المزامنة الفعلية

# ✅ صحيح: clip_grad_norm_ يجب أن يُستدعى دائمًا بعد backward() الكامل
# (الذي يتضمن المزامنة عبر DDP/FSDP تلقائيًا) وقبل optimizer.step() مباشرة
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: FP16 بدون GradScaler يؤدي لـUnderflow/Overflow متكرر وOnan
scaler = None
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x)
loss.backward()  # قد ينتج تدرّجات صفرية أو NaN بصمت!

# ✅ أفضل: استخدم BF16 إن كان العتاد يدعمه (A100+) — لا حاجة حتى لـGradScaler
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()  # BF16 لديه نفس المدى الديناميكي لـ FP32، استقرار أعلى بكثير

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-05-build-an-llm/chapter-24.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: محاكاة ومعالجة Loss Spike في حلقة تدريب حقيقية

المطلوب إكماله:
1. بناء نموذج صغير (MLP أو Transformer صغير) وبيانات تدريب اصطناعية طبيعية
2. حقن "عيّنات سامة" بشكل متعمّد كل عدد معين من الخطوات (قيم متطرفة إحصائيًا)
   يمكنك ضرب بعض عيّنات الإدخال في عامل كبير جدًا (مثلاً × 1000) لمحاكاة شذوذ البيانات
3. تشغيل حلقتين تدريب متوازيتين للمقارنة:
   أ) حلقة "ساذجة" بدون أي حماية (بدون Clipping، بدون كاشف Spikes)
   ب) حلقة "محمية" بكل الآليات المذكورة في هذا الفصل
4. رسم بياني لمنحنى الخسارة لكلتا الحلقتين على نفس الرسم البياني
5. قياس: هل انهارت الحلقة الساذجة فعلاً إلى NaN؟ هل نجت الحلقة المحمية؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def inject_poisoned_batch(
    clean_batch: torch.Tensor, poison_scale: float = 1000.0
) -> torch.Tensor:
    """TODO: أرجع نسخة من الدفعة بعد ضرب عيّنة واحدة عشوائية فيها
    بعامل poison_scale، لمحاكاة بيانات شاذة إحصائيًا من خط أنابيب حقيقي."""
    raise NotImplementedError


def run_naive_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: حلقة تدريب بلا أي حماية — يجب أن تُظهر انهيارًا فعليًا (NaN)
    عند نقاط حقن السموم. أرجع قائمة بقيم الخسارة عبر الزمن."""
    raise NotImplementedError


def run_protected_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: نفس السيناريو، لكن استخدم:
    - clip_gradients_and_report لتقليم التدرّجات
    - LossSpikeDetector لرصد القفزات وتخطي الخطوات المسمومة
    يجب أن تنجو هذه الحلقة من الانهيار رغم نفس البيانات المسمومة بالضبط."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: قارن المسارين بصريًا ورقميًا، واستخلص استنتاجك الخاص
    # حول فعالية كل آلية حماية على حدة (جرّب تعطيل Clipping فقط،
    # ثم تعطيل Spike Detection فقط، لمعرفة مساهمة كل منهما بمفرده)
    pass

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 24: الفصل 24: حلقة Pretraining عمليًا — إدارة نقاط الحفظ، تشخيص Loss Spikes، وGradient Checkpointing."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_checkpoint() -> None:
    checkpoint = {'step': 120, 'model': {'w': [0.2, 0.6]}, 'optimizer_step': 120, 'seed': 7}
    encoded = json.dumps(checkpoint, sort_keys=True)
    restored = json.loads(encoded)
    assert restored['step'] == restored['optimizer_step']
    print('checkpoint bytes:', len(encoded.encode('utf-8')))
    print('resume step:', restored['step'], '| seed:', restored['seed'])
    print('integrity check:', restored['model']['w'] == [0.2, 0.6])

if __name__ == "__main__":
    print("Chapter 24 lab: Save, validate, and resume a training checkpoint")
    run_checkpoint()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class TrainingState:
    """حاوية موحّدة لكل ما يُشكّل "حالة التدريب الكاملة" القابلة للاستئناف.
    تجميع هذه الحقول في مكان واحد يمنع نسيان أي مكوّن ضروري عند الحفظ."""

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        global_step: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = global_step

    def to_dict(self) -> dict:
        # get_state_dict الحديثة في PyTorch 2.x تتعامل تلقائيًا مع تعقيدات
        # FSDP Sharding دون الحاجة لكتابة منطق تجميع يدوي معقّد
        model_sd, optim_sd = get_state_dict(
            self.model,
            self.optimizer,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )
        return {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "rng_state_cuda": torch.cuda.get_rng_state_all(),
            "rng_state_cpu": torch.get_rng_state(),
        }

    def load_from_dict(self, state: dict) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        torch.cuda.set_rng_state_all(state["rng_state_cuda"])
        torch.set_rng_state(state["rng_state_cpu"])


def save_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> None:
    """حفظ موزّع: كل جهاز يحفظ شريحته فقط، بالتوازي، إلى مجلد مشترك
    (عادة على نظام ملفات موزّع مثل Lustre أو S3-compatible storage)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = training_state.to_dict()

    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp.FileSystemWriter(str(checkpoint_dir)),
    )

    if dist.get_rank() == 0:
        print(f"✅ تم حفظ نقطة تفتيش عند الخطوة {training_state.global_step} في {checkpoint_dir}")


def load_checkpoint(training_state: TrainingState, checkpoint_dir: Path) -> int:
    """استعادة موزّعة: يمكن أن تتم بعدد أجهزة مختلف تمامًا عن وقت الحفظ
    بفضل آلية Resharding التلقائية في torch.distributed.checkpoint."""
    state_dict = training_state.to_dict()  # هيكل فارغ تُملأ قيمه من الملفات

    dist_cp.load(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(str(checkpoint_dir)),
    )
    training_state.load_from_dict(state_dict)

    if dist.get_rank() == 0:
        print(f"✅ تم استئناف التدريب من الخطوة {training_state.global_step}")
    return training_state.global_step

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn


def clip_gradients_and_report(
    model: nn.Module, max_norm: float = 1.0
) -> float:
    """تقليم التدرّجات وإرجاع المعيار الفعلي قبل التقليم — هذا الرقم
    هو أحد أهم المقاييس للمراقبة، لأن قفزاته المفاجئة تُنذر بالخطر مبكرًا."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return total_norm.item()


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """جدولة شائعة جدًا في تدريب LLMs: Warmup خطي ثم تناقص جيبي-تجيبي (Cosine Decay).
    هذا هو النمط المستخدم فعليًا في GPT-3، LLaMA، وأغلب النماذج الحديثة."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # نمو خطي من قيمة قريبة من الصفر حتى 1.0
                return current_step / max(1, warmup_steps)

            # بعد الـWarmup: تناقص جيبي-تجيبي نحو min_lr_ratio
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        super().__init__(optimizer, lr_lambda=lr_lambda)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections import deque


class LossSpikeDetector:
    """كاشف قفزات الخسارة المفاجئة بناءً على متوسط متحرك وانحراف معياري،
    مع آلية لإصدار إشارة "توقف واستعادة" عند الحاجة."""

    def __init__(self, window_size: int = 100, spike_threshold_std: float = 6.0) -> None:
        self.window: deque[float] = deque(maxlen=window_size)
        self.spike_threshold_std = spike_threshold_std

    def update_and_check(self, current_loss: float) -> bool:
        """يرجع True إذا كانت الخسارة الحالية "قفزة شاذة" تستدعي تدخلاً."""
        if len(self.window) < 20:  # لا نحكم إلا بعد تجميع عيّنات كافية
            self.window.append(current_loss)
            return False

        mean_loss = sum(self.window) / len(self.window)
        variance = sum((x - mean_loss) ** 2 for x in self.window) / len(self.window)
        std_loss = variance ** 0.5

        is_spike = (
            current_loss > mean_loss + self.spike_threshold_std * std_loss
            or current_loss != current_loss  # فحص NaN (NaN != NaN دائمًا True)
        )

        if not is_spike:
            self.window.append(current_loss)  # لا نُلوّث النافذة بقيم شاذة

        return is_spike

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn


class TrainingMonitor:
    """جامع مقاييس مبسّط يمكن ربطه بـ Weights & Biases أو TensorBoard
    في بيئة إنتاجية حقيقية. هنا نُبقيه مستقلاً لتوضيح المنطق الأساسي."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
        }

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        self.history["loss"].append(loss)
        self.history["grad_norm"].append(grad_norm)
        self.history["learning_rate"].append(learning_rate)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }

        # مقاييس اختيارية أعمق: معايير التفعيلات لكل طبقة (مكلفة حسابيًا
        # نسبيًا، لذا تُحسب أحيانًا فقط كل N خطوة وليس كل خطوة)
        if model is not None:
            param_norms = {
                f"param_norm/{name}": param.detach().norm().item()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            # في بيئة حقيقية: نُرسل هذا لنظام مراقبة بدل طباعته فقط
            metrics.update(
                {k: v for k, v in list(param_norms.items())[:3]}  # عيّنة توضيحية فقط
            )

        return metrics

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlock(nn.Module):
    """كتلة Transformer تستخدم Activation Checkpointing: التفعيلات الداخلية
    للكتلة لا تُخزَّن، بل تُعاد حسابها عند الحاجة أثناء Backward."""

    def __init__(self, d_model: int, n_heads: int, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """المنطق الفعلي للكتلة، مفصول في دالة مستقلة ليسهل تمريره
        إلى torch.utils.checkpoint.checkpoint."""
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # use_reentrant=False هو الخيار الموصى به حديثًا في PyTorch
            # (أكثر توافقًا مع الميزات الحديثة مثل torch.compile وDDP/FSDP)
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class SelectiveCheckpointing:
    """استراتيجية حديثة أكثر دقة: بدلاً من تطبيق Checkpointing على كل
    الكتل بشكل متساوٍ، نُطبّقه فقط على العمليات "الرخيصة لإعادة الحساب
    لكن المكلفة في الذاكرة" (مثل الـMatmul)، ونتجنّبه على العمليات
    "المكلفة لإعادة حسابها" (مثل Attention نفسها في بعض التطبيقات).
    هذا هو النمط المستخدم في torch.compile + torch.utils.checkpoint الحديثة
    عبر context managers متخصصة مثل create_selective_checkpoint_contexts."""

    @staticmethod
    def estimate_memory_savings(
        num_layers: int, seq_len: int, d_model: int, batch_size: int, bytes_per_elem: int = 2
    ) -> dict[str, float]:
        """تقدير تقريبي لتوفير الذاكرة الناتج عن Full Checkpointing."""
        activation_size_per_layer = seq_len * d_model * batch_size * bytes_per_elem

        without_checkpointing = num_layers * activation_size_per_layer
        # تقريب شائع: فقط sqrt(L) من نقاط التفتيش تُحفَظ فعليًا
        with_checkpointing = (num_layers ** 0.5) * activation_size_per_layer

        return {
            "بدون_checkpointing_GB": without_checkpointing / 1e9,
            "مع_checkpointing_GB": with_checkpointing / 1e9,
            "نسبة_التوفير": 1 - (with_checkpointing / without_checkpointing),
        }


if __name__ == "__main__":
    savings = SelectiveCheckpointing.estimate_memory_savings(
        num_layers=80, seq_len=4096, d_model=8192, batch_size=4
    )
    for key, value in savings.items():
        print(f"{key}: {value:.2f}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def production_pretraining_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    total_steps: int,
    checkpoint_dir: Path,
    checkpoint_every: int = 500,
    grad_clip_norm: float = 1.0,
    rank: int = 0,
) -> None:
    """حلقة تدريب تجمع: Checkpointing دوري + كشف Loss Spikes + Gradient
    Clipping + مراقبة شاملة — هيكل مطابق لما يُستخدم فعليًا في تدريب LLMs
    على نطاق واسع، مع تبسيط بعض التفاصيل الموزّعة لأغراض التوضيح."""

    monitor = TrainingMonitor()
    spike_detector = LossSpikeDetector(window_size=100, spike_threshold_std=6.0)
    global_step = 0

    # محاولة استئناف تلقائي إن وُجدت نقطة تفتيش سابقة
    latest_ckpt = checkpoint_dir / "latest"
    if latest_ckpt.exists():
        training_state = TrainingState(model, optimizer, scheduler, global_step)
        global_step = load_checkpoint(training_state, latest_ckpt)

    data_iterator = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)

        input_ids, labels = batch
        input_ids, labels = input_ids.cuda(), labels.cuda()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()

        # تقليم التدرّجات قبل الخطوة — خط الدفاع الأول ضد الانفجار
        grad_norm = clip_gradients_and_report(model, max_norm=grad_clip_norm)

        current_loss = loss.item()

        # كشف القفزات — خط الدفاع الثاني: التدخل عند الحاجة
        if spike_detector.update_and_check(current_loss):
            if rank == 0:
                print(
                    f"⚠️ تم رصد Loss Spike عند الخطوة {global_step}: "
                    f"loss={current_loss:.4f}. إيقاف الخطوة واستعادة آخر نقطة سليمة."
                )
            # تخطّي هذه الخطوة كليًا (لا نطبّق optimizer.step() على تدرّج مسمم)
            # في نظام إنتاجي حقيقي: نستعيد هنا آخر checkpoint ونتخطى هذه الدفعة بعينها
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        scheduler.step()
        global_step += 1

        if rank == 0 and global_step % 10 == 0:
            metrics = monitor.log_step(
                loss=current_loss,
                grad_norm=grad_norm,
                learning_rate=scheduler.get_last_lr()[0],
            )
            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"GradNorm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['learning_rate']:.2e}"
            )

        if global_step % checkpoint_every == 0:
            training_state = TrainingState(model, optimizer, scheduler, global_step)
            step_ckpt_dir = checkpoint_dir / f"step_{global_step}"
            save_checkpoint(training_state, step_ckpt_dir)
            # تحديث "آخر نقطة" دون الحاجة لإعادة الحفظ الكامل (رابط رمزي مثلاً)
            # في نظام ملفات حقيقي: يُفضَّل استخدام symlink أو نسخ خفيف الوزن

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: عند الاستئناف، يبدأ Data Loader من الصفر دائمًا،
# مما يعني أن النموذج سيرى نفس البيانات المبكرة مرة أخرى بشكل غير مقصود
for epoch in range(epochs):
    for batch in train_loader:
        ...

# ✅ صحيح: تتبّع صريح لعدد الدفعات المُعالَجة واستئناف منطقي
# (في الممارسة العملية: تُستخدم أدوات متخصصة مثل StatefulDataLoader
# من TorchData، أو تُحسَب العيّنات المتخطّاة يدويًا عند الاستئناف)
batches_to_skip = global_step % len(train_loader)
for i, batch in enumerate(train_loader):
    if i < batches_to_skip:
        continue
    ...

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ شائع عند الانتقال من جهاز واحد إلى موزّع: حساب grad_norm
# محليًا فقط دون اعتبار أن DDP يُجري All-Reduce تلقائيًا بعد backward()،
# مما قد يؤدي لقص غير متسق إذا تم استدعاء clip_grad_norm_ في توقيت خاطئ
loss.backward()
# بعض الأكواد الخاطئة تستدعي clip هنا قبل اكتمال المزامنة الفعلية

# ✅ صحيح: clip_grad_norm_ يجب أن يُستدعى دائمًا بعد backward() الكامل
# (الذي يتضمن المزامنة عبر DDP/FSDP تلقائيًا) وقبل optimizer.step() مباشرة
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ خطير: FP16 بدون GradScaler يؤدي لـUnderflow/Overflow متكرر وOnan
scaler = None
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x)
loss.backward()  # قد ينتج تدرّجات صفرية أو NaN بصمت!

# ✅ أفضل: استخدم BF16 إن كان العتاد يدعمه (A100+) — لا حاجة حتى لـGradScaler
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(x)
loss.backward()  # BF16 لديه نفس المدى الديناميكي لـ FP32، استقرار أعلى بكثير

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-24-chapter-24.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: محاكاة ومعالجة Loss Spike في حلقة تدريب حقيقية

المطلوب إكماله:
1. بناء نموذج صغير (MLP أو Transformer صغير) وبيانات تدريب اصطناعية طبيعية
2. حقن "عيّنات سامة" بشكل متعمّد كل عدد معين من الخطوات (قيم متطرفة إحصائيًا)
   يمكنك ضرب بعض عيّنات الإدخال في عامل كبير جدًا (مثلاً × 1000) لمحاكاة شذوذ البيانات
3. تشغيل حلقتين تدريب متوازيتين للمقارنة:
   أ) حلقة "ساذجة" بدون أي حماية (بدون Clipping، بدون كاشف Spikes)
   ب) حلقة "محمية" بكل الآليات المذكورة في هذا الفصل
4. رسم بياني لمنحنى الخسارة لكلتا الحلقتين على نفس الرسم البياني
5. قياس: هل انهارت الحلقة الساذجة فعلاً إلى NaN؟ هل نجت الحلقة المحمية؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def inject_poisoned_batch(
    clean_batch: torch.Tensor, poison_scale: float = 1000.0
) -> torch.Tensor:
    """TODO: أرجع نسخة من الدفعة بعد ضرب عيّنة واحدة عشوائية فيها
    بعامل poison_scale، لمحاكاة بيانات شاذة إحصائيًا من خط أنابيب حقيقي."""
    raise NotImplementedError


def run_naive_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: حلقة تدريب بلا أي حماية — يجب أن تُظهر انهيارًا فعليًا (NaN)
    عند نقاط حقن السموم. أرجع قائمة بقيم الخسارة عبر الزمن."""
    raise NotImplementedError


def run_protected_training_loop(
    model: nn.Module, data: torch.Tensor, poison_every: int = 50, steps: int = 300
) -> list[float]:
    """TODO: نفس السيناريو، لكن استخدم:
    - clip_gradients_and_report لتقليم التدرّجات
    - LossSpikeDetector لرصد القفزات وتخطي الخطوات المسمومة
    يجب أن تنجو هذه الحلقة من الانهيار رغم نفس البيانات المسمومة بالضبط."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: قارن المسارين بصريًا ورقميًا، واستخلص استنتاجك الخاص
    # حول فعالية كل آلية حماية على حدة (جرّب تعطيل Clipping فقط،
    # ثم تعطيل Spike Detection فقط، لمعرفة مساهمة كل منهما بمفرده)
    pass
