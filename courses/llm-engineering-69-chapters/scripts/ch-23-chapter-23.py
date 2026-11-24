#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP
Large Language Model Engineering — Data

Book slug : llm-engineering-69-chapters
Chapter   : ch-23-chapter-23 (lesson 23)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-23-chapter-23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP
Large Language Model Engineering — Data

Book slug : llm-engineering-69-chapters
Chapter   : ch-23-chapter-23 (lesson 23)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-23-chapter-23.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP
Large Language Model Engineering — Data

Book slug : llm-engineering-69-chapters
Chapter   : ch-23-chapter-23 (lesson 23)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 23: الفصل 23: التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_distributed() -> None:
    batch, data_parallel, tensor_parallel, pipeline_stages = (96, 4, 2, 3)
    per_replica_batch = batch // data_parallel
    print('global batch:', batch)
    print('batch per data-parallel replica:', per_replica_batch)
    print('model partitions (tensor × pipeline):', tensor_parallel * pipeline_stages)
    print('3-D ranks:', data_parallel * tensor_parallel * pipeline_stages)

if __name__ == "__main__":
    print("Chapter 23 lab: Shard a batch across a 3-D parallel layout")
    run_distributed()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_distributed(rank: int, world_size: int) -> None:
    """تهيئة مجموعة العمليات الموزّعة باستخدام backend NCCL (الأمثل لـ GPU)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_ddp_worker(rank: int, world_size: int, epochs: int = 3) -> None:
    """دالة التدريب التي تُشغَّل بشكل مستقل على كل عملية/GPU."""
    torch.manual_seed(42)
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # نموذج توضيحي بسيط — في الواقع سيكون هذا LLM ضخمًا
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    # لفّ النموذج بـ DDP — هذا يتولى تلقائيًا:
    # 1) توزيع التدرّجات عبر All-Reduce بعد backward()
    # 2) Bucketing للتدرّجات لتقليل عدد عمليات الاتصال الشبكي
    ddp_model = DDP(model, device_ids=[rank])

    # بيانات توضيحية موزّعة بـ DistributedSampler
    # الضامن الأساسي بأن كل GPU يرى جزءًا مختلفًا من البيانات دون تداخل
    dataset = TensorDataset(
        torch.randn(10_000, 1024),
        torch.randn(10_000, 1024),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # ضروري لإعادة خلط البيانات بشكل مختلف كل حقبة
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = ddp_model(x)
            loss = loss_fn(out, y)
            loss.backward()  # هنا يحدث All-Reduce تلقائيًا خلف الكواليس
            optimizer.step()

            if rank == 0 and step % 50 == 0:
                print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_ddp_worker, args=(world_size,), nprocs=world_size, join=True
    )

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة عموديًا: كل جهاز يملك جزءًا من الأعمدة (المخرجات).
    لا يحتاج اتصال شبكي أثناء forward — فقط عند الحاجة لدمج النتيجة لاحقًا."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert out_features % world_size == 0, "يجب أن يكون out_features قابلاً للقسمة على عدد الأجهزة"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل جهاز يملك فقط شريحته من مصفوفة الأوزان
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # كل جهاز يحسب جزءه المحلي فقط — بدون أي اتصال شبكي هنا
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة أفقيًا: كل جهاز يملك جزءًا من الصفوف (المدخلات).
    يحتاج All-Reduce في نهاية forward لجمع المساهمات الجزئية من كل الأجهزة."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert in_features % world_size == 0, "يجب أن يكون in_features قابلاً للقسمة على عدد الأجهزة"
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_output = F.linear(x, self.weight)
        # All-Reduce: كل جهاز يحصل على مجموع المساهمات الجزئية من كل الأجهزة
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output


class TensorParallelMLP(nn.Module):
    """كتلة MLP موزّعة بـ Tensor Parallelism على نمط Megatron-LM:
    Column Parallel ثم Row Parallel، مع All-Reduce واحد فقط في النهاية."""

    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int) -> None:
        super().__init__()
        # التوسيع: تقسيم عمودي — بلا اتصال شبكي
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.activation = nn.GELU()
        # التضييق: تقسيم أفقي — اتصال شبكي واحد فقط هنا
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))   # محلي بالكامل، بلا اتصال
        out = self.fc2(h)                  # All-Reduce واحد هنا فقط
        return out

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PipelineStageConfig:
    stage_id: int
    num_stages: int
    num_microbatches: int


class PipelineStage(nn.Module):
    """مرحلة واحدة من الأنبوب — تحمل مجموعة فرعية من طبقات النموذج الكامل."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def split_model_into_stages(
    all_layers: list[nn.Module], num_stages: int
) -> list[PipelineStage]:
    """تقسيم قائمة طبقات متتالية إلى مراحل متوازنة تقريبًا للأنبوب."""
    n = len(all_layers)
    base_size = n // num_stages
    remainder = n % num_stages

    stages: list[PipelineStage] = []
    start = 0
    for stage_idx in range(num_stages):
        # توزيع الطبقات الإضافية على المراحل الأولى لموازنة الحمل
        size = base_size + (1 if stage_idx < remainder else 0)
        stages.append(PipelineStage(all_layers[start : start + size]))
        start += size
    return stages


def simulate_gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[str]]:
    """محاكاة جدول زمني مبسّط لـ GPipe لتوضيح متى يعمل كل جهاز
    وفي أي مرحلة زمنية (time slot)، بما يكشف بصريًا حجم الفقاعة."""
    total_slots = num_stages + num_microbatches - 1
    schedule: list[list[str]] = [["idle"] * total_slots for _ in range(num_stages)]

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            time_slot = stage + mb
            schedule[stage][time_slot] = f"F(mb={mb})"

    bubble_slots = sum(row.count("idle") for row in schedule)
    total_cells = num_stages * total_slots
    print(f"نسبة الفقاعة التقريبية: {bubble_slots / total_cells:.2%}")
    return schedule


if __name__ == "__main__":
    # مثال: 4 مراحل، 8 دفعات مصغّرة -> فقاعة أقل بكثير من 4 مراحل و 1 دفعة فقط
    schedule = simulate_gpipe_schedule(num_stages=4, num_microbatches=8)
    for stage_idx, row in enumerate(schedule):
        print(f"GPU{stage_idx}: {row}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class TransformerBlock(nn.Module):
    """كتلة Transformer توضيحية مبسّطة — في الواقع هذه ستكون
    كتلة كاملة بـ Self-Attention و FFN كما في الفصل الحادي عشر."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def wrap_model_with_fsdp(model: nn.Module, device: torch.device) -> FSDP:
    """تغليف النموذج بـ FSDP بسياسة تقسيم تلقائية على مستوى كل كتلة Transformer،
    مع تفعيل الدقة المختلطة BF16 لتسريع الحساب وتقليل حجم الاتصال الشبكي."""

    # سياسة Auto-Wrap: تُحدَّد وحدة التقسيم (Sharding Unit) عند كل TransformerBlock
    # بدلاً من النموذج كاملاً كوحدة واحدة — هذا يسمح بتداخل الاتصال مع الحساب
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,     # الأوزان أثناء الحساب بـ BF16
        reduce_dtype=torch.bfloat16,    # تخفيض التدرّجات بـ BF16 (اتصال أسرع)
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # مطابق لـ ZeRO Stage 3
        device_id=device,
        use_orig_params=True,  # يسمح بالتوافق السلس مع torch.compile ومُحسِّنات PyTorch الحديثة
    )
    return fsdp_model


def train_step_fsdp(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    loss.backward()  # Reduce-Scatter يحدث هنا تلقائيًا خلف الكواليس
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: نفس ترتيب البيانات بالضبط يتكرر كل حقبة لكل جهاز
for epoch in range(epochs):
    for x, y in loader:
        ...

# ✅ صحيح: ضروري لإعادة خلط فعّال عبر الحقبات
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for x, y in loader:
        ...

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: هذا يجعل "Global Batch Size" يتغير تلقائيًا مع عدد الأجهزة!
batch_size = 32  # نفس القيمة بغض النظر عن world_size
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
# مع 8 أجهزة: Global Batch = 32 × 8 = 256 (قد يكون هذا غير مقصود!)

# ✅ صحيح: حدّد Global Batch بوضوح، واشتق منه الحجم المحلي
global_batch_size = 512
local_batch_size = global_batch_size // world_size
loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: قد يسبب تعارضات غير متوقعة أو فشل صامت في تحسينات الـ compiler
fsdp_model = FSDP(model, use_orig_params=False)
compiled_model = torch.compile(fsdp_model)

# ✅ صحيح: use_orig_params=True ضروري للتوافق السلس الحديث
fsdp_model = FSDP(model, use_orig_params=True)
compiled_model = torch.compile(fsdp_model)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: قياس ومقارنة استهلاك الذاكرة بين DDP وFSDP
على نموذج Transformer صغير قابل للتوسيع.

المطلوب إكماله:
1. بناء نموذج Transformer بحجم قابل للضبط (استخدم عدة إعدادات: صغير/متوسط/كبير)
2. تشغيل تدريب قصير (عدة خطوات) بكل من DDP وFSDP
3. تسجيل torch.cuda.max_memory_allocated() لكل إعداد
4. رسم بياني يقارن استهلاك الذاكرة مقابل حجم النموذج للطريقتين
5. تحقق تجريبيًا: عند أي حجم نموذج يصبح FSDP ضروريًا (لا يعمل DDP إطلاقًا)؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scalable_transformer(
    vocab_size: int = 32_000,
    d_model: int = 1024,
    n_layers: int = 12,
    n_heads: int = 16,
) -> nn.Module:
    """TODO: استخدم MiniTransformerLM المعرّف سابقًا في هذا الفصل،
    لكن اجعل الأبعاد قابلة للضبط عبر المعاملات لتجربة أحجام مختلفة."""
    raise NotImplementedError


def measure_peak_memory_ddp(model_config: dict, world_size: int) -> float:
    """TODO: شغّل خطوة تدريب واحدة باستخدام DDP وأرجع ذروة الذاكرة بالجيجابايت.
    استخدم torch.multiprocessing.spawn كما في مثال train_ddp_worker.
    تذكّر: torch.cuda.reset_peak_memory_stats() قبل البدء،
    و torch.cuda.max_memory_allocated() بعد الانتهاء."""
    raise NotImplementedError


def measure_peak_memory_fsdp(model_config: dict, world_size: int) -> float:
    """TODO: نفس الفكرة لكن باستخدام wrap_model_with_fsdp المعرّفة سابقًا."""
    raise NotImplementedError


def run_comparison_experiment() -> None:
    """TODO: جرّب 3-4 أحجام نموذج مختلفة (مثلاً: 50M، 200M، 500M، 1B معامل)
    وسجّل ذروة الذاكرة لكل من DDP وFSDP، ثم اطبع جدول مقارنة نصي.
    سؤال تأملي: هل الفرق بين DDP وFSDP يتسع أم يضيق كلما كبر النموذج؟ لماذا؟"""
    raise NotImplementedError


if __name__ == "__main__":
    run_comparison_experiment()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 23: الفصل 23: التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_distributed() -> None:
    batch, data_parallel, tensor_parallel, pipeline_stages = (96, 4, 2, 3)
    per_replica_batch = batch // data_parallel
    print('global batch:', batch)
    print('batch per data-parallel replica:', per_replica_batch)
    print('model partitions (tensor × pipeline):', tensor_parallel * pipeline_stages)
    print('3-D ranks:', data_parallel * tensor_parallel * pipeline_stages)

if __name__ == "__main__":
    print("Chapter 23 lab: Shard a batch across a 3-D parallel layout")
    run_distributed()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_distributed(rank: int, world_size: int) -> None:
    """تهيئة مجموعة العمليات الموزّعة باستخدام backend NCCL (الأمثل لـ GPU)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_ddp_worker(rank: int, world_size: int, epochs: int = 3) -> None:
    """دالة التدريب التي تُشغَّل بشكل مستقل على كل عملية/GPU."""
    torch.manual_seed(42)
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # نموذج توضيحي بسيط — في الواقع سيكون هذا LLM ضخمًا
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    # لفّ النموذج بـ DDP — هذا يتولى تلقائيًا:
    # 1) توزيع التدرّجات عبر All-Reduce بعد backward()
    # 2) Bucketing للتدرّجات لتقليل عدد عمليات الاتصال الشبكي
    ddp_model = DDP(model, device_ids=[rank])

    # بيانات توضيحية موزّعة بـ DistributedSampler
    # الضامن الأساسي بأن كل GPU يرى جزءًا مختلفًا من البيانات دون تداخل
    dataset = TensorDataset(
        torch.randn(10_000, 1024),
        torch.randn(10_000, 1024),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # ضروري لإعادة خلط البيانات بشكل مختلف كل حقبة
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = ddp_model(x)
            loss = loss_fn(out, y)
            loss.backward()  # هنا يحدث All-Reduce تلقائيًا خلف الكواليس
            optimizer.step()

            if rank == 0 and step % 50 == 0:
                print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_ddp_worker, args=(world_size,), nprocs=world_size, join=True
    )

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة عموديًا: كل جهاز يملك جزءًا من الأعمدة (المخرجات).
    لا يحتاج اتصال شبكي أثناء forward — فقط عند الحاجة لدمج النتيجة لاحقًا."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert out_features % world_size == 0, "يجب أن يكون out_features قابلاً للقسمة على عدد الأجهزة"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل جهاز يملك فقط شريحته من مصفوفة الأوزان
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # كل جهاز يحسب جزءه المحلي فقط — بدون أي اتصال شبكي هنا
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة أفقيًا: كل جهاز يملك جزءًا من الصفوف (المدخلات).
    يحتاج All-Reduce في نهاية forward لجمع المساهمات الجزئية من كل الأجهزة."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert in_features % world_size == 0, "يجب أن يكون in_features قابلاً للقسمة على عدد الأجهزة"
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_output = F.linear(x, self.weight)
        # All-Reduce: كل جهاز يحصل على مجموع المساهمات الجزئية من كل الأجهزة
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output


class TensorParallelMLP(nn.Module):
    """كتلة MLP موزّعة بـ Tensor Parallelism على نمط Megatron-LM:
    Column Parallel ثم Row Parallel، مع All-Reduce واحد فقط في النهاية."""

    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int) -> None:
        super().__init__()
        # التوسيع: تقسيم عمودي — بلا اتصال شبكي
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.activation = nn.GELU()
        # التضييق: تقسيم أفقي — اتصال شبكي واحد فقط هنا
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))   # محلي بالكامل، بلا اتصال
        out = self.fc2(h)                  # All-Reduce واحد هنا فقط
        return out

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PipelineStageConfig:
    stage_id: int
    num_stages: int
    num_microbatches: int


class PipelineStage(nn.Module):
    """مرحلة واحدة من الأنبوب — تحمل مجموعة فرعية من طبقات النموذج الكامل."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def split_model_into_stages(
    all_layers: list[nn.Module], num_stages: int
) -> list[PipelineStage]:
    """تقسيم قائمة طبقات متتالية إلى مراحل متوازنة تقريبًا للأنبوب."""
    n = len(all_layers)
    base_size = n // num_stages
    remainder = n % num_stages

    stages: list[PipelineStage] = []
    start = 0
    for stage_idx in range(num_stages):
        # توزيع الطبقات الإضافية على المراحل الأولى لموازنة الحمل
        size = base_size + (1 if stage_idx < remainder else 0)
        stages.append(PipelineStage(all_layers[start : start + size]))
        start += size
    return stages


def simulate_gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[str]]:
    """محاكاة جدول زمني مبسّط لـ GPipe لتوضيح متى يعمل كل جهاز
    وفي أي مرحلة زمنية (time slot)، بما يكشف بصريًا حجم الفقاعة."""
    total_slots = num_stages + num_microbatches - 1
    schedule: list[list[str]] = [["idle"] * total_slots for _ in range(num_stages)]

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            time_slot = stage + mb
            schedule[stage][time_slot] = f"F(mb={mb})"

    bubble_slots = sum(row.count("idle") for row in schedule)
    total_cells = num_stages * total_slots
    print(f"نسبة الفقاعة التقريبية: {bubble_slots / total_cells:.2%}")
    return schedule


if __name__ == "__main__":
    # مثال: 4 مراحل، 8 دفعات مصغّرة -> فقاعة أقل بكثير من 4 مراحل و 1 دفعة فقط
    schedule = simulate_gpipe_schedule(num_stages=4, num_microbatches=8)
    for stage_idx, row in enumerate(schedule):
        print(f"GPU{stage_idx}: {row}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class TransformerBlock(nn.Module):
    """كتلة Transformer توضيحية مبسّطة — في الواقع هذه ستكون
    كتلة كاملة بـ Self-Attention و FFN كما في الفصل الحادي عشر."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def wrap_model_with_fsdp(model: nn.Module, device: torch.device) -> FSDP:
    """تغليف النموذج بـ FSDP بسياسة تقسيم تلقائية على مستوى كل كتلة Transformer،
    مع تفعيل الدقة المختلطة BF16 لتسريع الحساب وتقليل حجم الاتصال الشبكي."""

    # سياسة Auto-Wrap: تُحدَّد وحدة التقسيم (Sharding Unit) عند كل TransformerBlock
    # بدلاً من النموذج كاملاً كوحدة واحدة — هذا يسمح بتداخل الاتصال مع الحساب
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,     # الأوزان أثناء الحساب بـ BF16
        reduce_dtype=torch.bfloat16,    # تخفيض التدرّجات بـ BF16 (اتصال أسرع)
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # مطابق لـ ZeRO Stage 3
        device_id=device,
        use_orig_params=True,  # يسمح بالتوافق السلس مع torch.compile ومُحسِّنات PyTorch الحديثة
    )
    return fsdp_model


def train_step_fsdp(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    loss.backward()  # Reduce-Scatter يحدث هنا تلقائيًا خلف الكواليس
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: نفس ترتيب البيانات بالضبط يتكرر كل حقبة لكل جهاز
for epoch in range(epochs):
    for x, y in loader:
        ...

# ✅ صحيح: ضروري لإعادة خلط فعّال عبر الحقبات
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for x, y in loader:
        ...

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: هذا يجعل "Global Batch Size" يتغير تلقائيًا مع عدد الأجهزة!
batch_size = 32  # نفس القيمة بغض النظر عن world_size
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
# مع 8 أجهزة: Global Batch = 32 × 8 = 256 (قد يكون هذا غير مقصود!)

# ✅ صحيح: حدّد Global Batch بوضوح، واشتق منه الحجم المحلي
global_batch_size = 512
local_batch_size = global_batch_size // world_size
loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: قد يسبب تعارضات غير متوقعة أو فشل صامت في تحسينات الـ compiler
fsdp_model = FSDP(model, use_orig_params=False)
compiled_model = torch.compile(fsdp_model)

# ✅ صحيح: use_orig_params=True ضروري للتوافق السلس الحديث
fsdp_model = FSDP(model, use_orig_params=True)
compiled_model = torch.compile(fsdp_model)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: قياس ومقارنة استهلاك الذاكرة بين DDP وFSDP
على نموذج Transformer صغير قابل للتوسيع.

المطلوب إكماله:
1. بناء نموذج Transformer بحجم قابل للضبط (استخدم عدة إعدادات: صغير/متوسط/كبير)
2. تشغيل تدريب قصير (عدة خطوات) بكل من DDP وFSDP
3. تسجيل torch.cuda.max_memory_allocated() لكل إعداد
4. رسم بياني يقارن استهلاك الذاكرة مقابل حجم النموذج للطريقتين
5. تحقق تجريبيًا: عند أي حجم نموذج يصبح FSDP ضروريًا (لا يعمل DDP إطلاقًا)؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scalable_transformer(
    vocab_size: int = 32_000,
    d_model: int = 1024,
    n_layers: int = 12,
    n_heads: int = 16,
) -> nn.Module:
    """TODO: استخدم MiniTransformerLM المعرّف سابقًا في هذا الفصل،
    لكن اجعل الأبعاد قابلة للضبط عبر المعاملات لتجربة أحجام مختلفة."""
    raise NotImplementedError


def measure_peak_memory_ddp(model_config: dict, world_size: int) -> float:
    """TODO: شغّل خطوة تدريب واحدة باستخدام DDP وأرجع ذروة الذاكرة بالجيجابايت.
    استخدم torch.multiprocessing.spawn كما في مثال train_ddp_worker.
    تذكّر: torch.cuda.reset_peak_memory_stats() قبل البدء،
    و torch.cuda.max_memory_allocated() بعد الانتهاء."""
    raise NotImplementedError


def measure_peak_memory_fsdp(model_config: dict, world_size: int) -> float:
    """TODO: نفس الفكرة لكن باستخدام wrap_model_with_fsdp المعرّفة سابقًا."""
    raise NotImplementedError


def run_comparison_experiment() -> None:
    """TODO: جرّب 3-4 أحجام نموذج مختلفة (مثلاً: 50M، 200M، 500M، 1B معامل)
    وسجّل ذروة الذاكرة لكل من DDP وFSDP، ثم اطبع جدول مقارنة نصي.
    سؤال تأملي: هل الفرق بين DDP وFSDP يتسع أم يضيق كلما كبر النموذج؟ لماذا؟"""
    raise NotImplementedError


if __name__ == "__main__":
    run_comparison_experiment()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP
Large Language Model Engineering — Data

Book slug : llm-engineering-69-chapters
Chapter   : ch-23-chapter-23 (lesson 23)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_23.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 23: الفصل 23: التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_distributed() -> None:
    batch, data_parallel, tensor_parallel, pipeline_stages = (96, 4, 2, 3)
    per_replica_batch = batch // data_parallel
    print('global batch:', batch)
    print('batch per data-parallel replica:', per_replica_batch)
    print('model partitions (tensor × pipeline):', tensor_parallel * pipeline_stages)
    print('3-D ranks:', data_parallel * tensor_parallel * pipeline_stages)

if __name__ == "__main__":
    print("Chapter 23 lab: Shard a batch across a 3-D parallel layout")
    run_distributed()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_distributed(rank: int, world_size: int) -> None:
    """تهيئة مجموعة العمليات الموزّعة باستخدام backend NCCL (الأمثل لـ GPU)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_ddp_worker(rank: int, world_size: int, epochs: int = 3) -> None:
    """دالة التدريب التي تُشغَّل بشكل مستقل على كل عملية/GPU."""
    torch.manual_seed(42)
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # نموذج توضيحي بسيط — في الواقع سيكون هذا LLM ضخمًا
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    # لفّ النموذج بـ DDP — هذا يتولى تلقائيًا:
    # 1) توزيع التدرّجات عبر All-Reduce بعد backward()
    # 2) Bucketing للتدرّجات لتقليل عدد عمليات الاتصال الشبكي
    ddp_model = DDP(model, device_ids=[rank])

    # بيانات توضيحية موزّعة بـ DistributedSampler
    # الضامن الأساسي بأن كل GPU يرى جزءًا مختلفًا من البيانات دون تداخل
    dataset = TensorDataset(
        torch.randn(10_000, 1024),
        torch.randn(10_000, 1024),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # ضروري لإعادة خلط البيانات بشكل مختلف كل حقبة
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = ddp_model(x)
            loss = loss_fn(out, y)
            loss.backward()  # هنا يحدث All-Reduce تلقائيًا خلف الكواليس
            optimizer.step()

            if rank == 0 and step % 50 == 0:
                print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_ddp_worker, args=(world_size,), nprocs=world_size, join=True
    )

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة عموديًا: كل جهاز يملك جزءًا من الأعمدة (المخرجات).
    لا يحتاج اتصال شبكي أثناء forward — فقط عند الحاجة لدمج النتيجة لاحقًا."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert out_features % world_size == 0, "يجب أن يكون out_features قابلاً للقسمة على عدد الأجهزة"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل جهاز يملك فقط شريحته من مصفوفة الأوزان
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # كل جهاز يحسب جزءه المحلي فقط — بدون أي اتصال شبكي هنا
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة أفقيًا: كل جهاز يملك جزءًا من الصفوف (المدخلات).
    يحتاج All-Reduce في نهاية forward لجمع المساهمات الجزئية من كل الأجهزة."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert in_features % world_size == 0, "يجب أن يكون in_features قابلاً للقسمة على عدد الأجهزة"
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_output = F.linear(x, self.weight)
        # All-Reduce: كل جهاز يحصل على مجموع المساهمات الجزئية من كل الأجهزة
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output


class TensorParallelMLP(nn.Module):
    """كتلة MLP موزّعة بـ Tensor Parallelism على نمط Megatron-LM:
    Column Parallel ثم Row Parallel، مع All-Reduce واحد فقط في النهاية."""

    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int) -> None:
        super().__init__()
        # التوسيع: تقسيم عمودي — بلا اتصال شبكي
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.activation = nn.GELU()
        # التضييق: تقسيم أفقي — اتصال شبكي واحد فقط هنا
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))   # محلي بالكامل، بلا اتصال
        out = self.fc2(h)                  # All-Reduce واحد هنا فقط
        return out

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PipelineStageConfig:
    stage_id: int
    num_stages: int
    num_microbatches: int


class PipelineStage(nn.Module):
    """مرحلة واحدة من الأنبوب — تحمل مجموعة فرعية من طبقات النموذج الكامل."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def split_model_into_stages(
    all_layers: list[nn.Module], num_stages: int
) -> list[PipelineStage]:
    """تقسيم قائمة طبقات متتالية إلى مراحل متوازنة تقريبًا للأنبوب."""
    n = len(all_layers)
    base_size = n // num_stages
    remainder = n % num_stages

    stages: list[PipelineStage] = []
    start = 0
    for stage_idx in range(num_stages):
        # توزيع الطبقات الإضافية على المراحل الأولى لموازنة الحمل
        size = base_size + (1 if stage_idx < remainder else 0)
        stages.append(PipelineStage(all_layers[start : start + size]))
        start += size
    return stages


def simulate_gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[str]]:
    """محاكاة جدول زمني مبسّط لـ GPipe لتوضيح متى يعمل كل جهاز
    وفي أي مرحلة زمنية (time slot)، بما يكشف بصريًا حجم الفقاعة."""
    total_slots = num_stages + num_microbatches - 1
    schedule: list[list[str]] = [["idle"] * total_slots for _ in range(num_stages)]

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            time_slot = stage + mb
            schedule[stage][time_slot] = f"F(mb={mb})"

    bubble_slots = sum(row.count("idle") for row in schedule)
    total_cells = num_stages * total_slots
    print(f"نسبة الفقاعة التقريبية: {bubble_slots / total_cells:.2%}")
    return schedule


if __name__ == "__main__":
    # مثال: 4 مراحل، 8 دفعات مصغّرة -> فقاعة أقل بكثير من 4 مراحل و 1 دفعة فقط
    schedule = simulate_gpipe_schedule(num_stages=4, num_microbatches=8)
    for stage_idx, row in enumerate(schedule):
        print(f"GPU{stage_idx}: {row}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class TransformerBlock(nn.Module):
    """كتلة Transformer توضيحية مبسّطة — في الواقع هذه ستكون
    كتلة كاملة بـ Self-Attention و FFN كما في الفصل الحادي عشر."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def wrap_model_with_fsdp(model: nn.Module, device: torch.device) -> FSDP:
    """تغليف النموذج بـ FSDP بسياسة تقسيم تلقائية على مستوى كل كتلة Transformer،
    مع تفعيل الدقة المختلطة BF16 لتسريع الحساب وتقليل حجم الاتصال الشبكي."""

    # سياسة Auto-Wrap: تُحدَّد وحدة التقسيم (Sharding Unit) عند كل TransformerBlock
    # بدلاً من النموذج كاملاً كوحدة واحدة — هذا يسمح بتداخل الاتصال مع الحساب
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,     # الأوزان أثناء الحساب بـ BF16
        reduce_dtype=torch.bfloat16,    # تخفيض التدرّجات بـ BF16 (اتصال أسرع)
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # مطابق لـ ZeRO Stage 3
        device_id=device,
        use_orig_params=True,  # يسمح بالتوافق السلس مع torch.compile ومُحسِّنات PyTorch الحديثة
    )
    return fsdp_model


def train_step_fsdp(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    loss.backward()  # Reduce-Scatter يحدث هنا تلقائيًا خلف الكواليس
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: نفس ترتيب البيانات بالضبط يتكرر كل حقبة لكل جهاز
for epoch in range(epochs):
    for x, y in loader:
        ...

# ✅ صحيح: ضروري لإعادة خلط فعّال عبر الحقبات
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for x, y in loader:
        ...

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: هذا يجعل "Global Batch Size" يتغير تلقائيًا مع عدد الأجهزة!
batch_size = 32  # نفس القيمة بغض النظر عن world_size
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
# مع 8 أجهزة: Global Batch = 32 × 8 = 256 (قد يكون هذا غير مقصود!)

# ✅ صحيح: حدّد Global Batch بوضوح، واشتق منه الحجم المحلي
global_batch_size = 512
local_batch_size = global_batch_size // world_size
loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: قد يسبب تعارضات غير متوقعة أو فشل صامت في تحسينات الـ compiler
fsdp_model = FSDP(model, use_orig_params=False)
compiled_model = torch.compile(fsdp_model)

# ✅ صحيح: use_orig_params=True ضروري للتوافق السلس الحديث
fsdp_model = FSDP(model, use_orig_params=True)
compiled_model = torch.compile(fsdp_model)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-23.md
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: قياس ومقارنة استهلاك الذاكرة بين DDP وFSDP
على نموذج Transformer صغير قابل للتوسيع.

المطلوب إكماله:
1. بناء نموذج Transformer بحجم قابل للضبط (استخدم عدة إعدادات: صغير/متوسط/كبير)
2. تشغيل تدريب قصير (عدة خطوات) بكل من DDP وFSDP
3. تسجيل torch.cuda.max_memory_allocated() لكل إعداد
4. رسم بياني يقارن استهلاك الذاكرة مقابل حجم النموذج للطريقتين
5. تحقق تجريبيًا: عند أي حجم نموذج يصبح FSDP ضروريًا (لا يعمل DDP إطلاقًا)؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scalable_transformer(
    vocab_size: int = 32_000,
    d_model: int = 1024,
    n_layers: int = 12,
    n_heads: int = 16,
) -> nn.Module:
    """TODO: استخدم MiniTransformerLM المعرّف سابقًا في هذا الفصل،
    لكن اجعل الأبعاد قابلة للضبط عبر المعاملات لتجربة أحجام مختلفة."""
    raise NotImplementedError


def measure_peak_memory_ddp(model_config: dict, world_size: int) -> float:
    """TODO: شغّل خطوة تدريب واحدة باستخدام DDP وأرجع ذروة الذاكرة بالجيجابايت.
    استخدم torch.multiprocessing.spawn كما في مثال train_ddp_worker.
    تذكّر: torch.cuda.reset_peak_memory_stats() قبل البدء،
    و torch.cuda.max_memory_allocated() بعد الانتهاء."""
    raise NotImplementedError


def measure_peak_memory_fsdp(model_config: dict, world_size: int) -> float:
    """TODO: نفس الفكرة لكن باستخدام wrap_model_with_fsdp المعرّفة سابقًا."""
    raise NotImplementedError


def run_comparison_experiment() -> None:
    """TODO: جرّب 3-4 أحجام نموذج مختلفة (مثلاً: 50M، 200M، 500M، 1B معامل)
    وسجّل ذروة الذاكرة لكل من DDP وFSDP، ثم اطبع جدول مقارنة نصي.
    سؤال تأملي: هل الفرق بين DDP وFSDP يتسع أم يضيق كلما كبر النموذج؟ لماذا؟"""
    raise NotImplementedError


if __name__ == "__main__":
    run_comparison_experiment()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 23: الفصل 23: التدريب الموزّع — Data/Tensor/Pipeline Parallelism، 3D Parallelism، ZeRO وFSDP."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_distributed() -> None:
    batch, data_parallel, tensor_parallel, pipeline_stages = (96, 4, 2, 3)
    per_replica_batch = batch // data_parallel
    print('global batch:', batch)
    print('batch per data-parallel replica:', per_replica_batch)
    print('model partitions (tensor × pipeline):', tensor_parallel * pipeline_stages)
    print('3-D ranks:', data_parallel * tensor_parallel * pipeline_stages)

if __name__ == "__main__":
    print("Chapter 23 lab: Shard a batch across a 3-D parallel layout")
    run_distributed()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def setup_distributed(rank: int, world_size: int) -> None:
    """تهيئة مجموعة العمليات الموزّعة باستخدام backend NCCL (الأمثل لـ GPU)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def train_ddp_worker(rank: int, world_size: int, epochs: int = 3) -> None:
    """دالة التدريب التي تُشغَّل بشكل مستقل على كل عملية/GPU."""
    torch.manual_seed(42)
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # نموذج توضيحي بسيط — في الواقع سيكون هذا LLM ضخمًا
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)

    # لفّ النموذج بـ DDP — هذا يتولى تلقائيًا:
    # 1) توزيع التدرّجات عبر All-Reduce بعد backward()
    # 2) Bucketing للتدرّجات لتقليل عدد عمليات الاتصال الشبكي
    ddp_model = DDP(model, device_ids=[rank])

    # بيانات توضيحية موزّعة بـ DistributedSampler
    # الضامن الأساسي بأن كل GPU يرى جزءًا مختلفًا من البيانات دون تداخل
    dataset = TensorDataset(
        torch.randn(10_000, 1024),
        torch.randn(10_000, 1024),
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # ضروري لإعادة خلط البيانات بشكل مختلف كل حقبة
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = ddp_model(x)
            loss = loss_fn(out, y)
            loss.backward()  # هنا يحدث All-Reduce تلقائيًا خلف الكواليس
            optimizer.step()

            if rank == 0 and step % 50 == 0:
                print(f"[Epoch {epoch}] Step {step} | Loss: {loss.item():.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_ddp_worker, args=(world_size,), nprocs=world_size, join=True
    )

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة عموديًا: كل جهاز يملك جزءًا من الأعمدة (المخرجات).
    لا يحتاج اتصال شبكي أثناء forward — فقط عند الحاجة لدمج النتيجة لاحقًا."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert out_features % world_size == 0, "يجب أن يكون out_features قابلاً للقسمة على عدد الأجهزة"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل جهاز يملك فقط شريحته من مصفوفة الأوزان
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # كل جهاز يحسب جزءه المحلي فقط — بدون أي اتصال شبكي هنا
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """طبقة خطية مُقسّمة أفقيًا: كل جهاز يملك جزءًا من الصفوف (المدخلات).
    يحتاج All-Reduce في نهاية forward لجمع المساهمات الجزئية من كل الأجهزة."""

    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int) -> None:
        super().__init__()
        assert in_features % world_size == 0, "يجب أن يكون in_features قابلاً للقسمة على عدد الأجهزة"
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_output = F.linear(x, self.weight)
        # All-Reduce: كل جهاز يحصل على مجموع المساهمات الجزئية من كل الأجهزة
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        return local_output


class TensorParallelMLP(nn.Module):
    """كتلة MLP موزّعة بـ Tensor Parallelism على نمط Megatron-LM:
    Column Parallel ثم Row Parallel، مع All-Reduce واحد فقط في النهاية."""

    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int) -> None:
        super().__init__()
        # التوسيع: تقسيم عمودي — بلا اتصال شبكي
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.activation = nn.GELU()
        # التضييق: تقسيم أفقي — اتصال شبكي واحد فقط هنا
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))   # محلي بالكامل، بلا اتصال
        out = self.fc2(h)                  # All-Reduce واحد هنا فقط
        return out

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PipelineStageConfig:
    stage_id: int
    num_stages: int
    num_microbatches: int


class PipelineStage(nn.Module):
    """مرحلة واحدة من الأنبوب — تحمل مجموعة فرعية من طبقات النموذج الكامل."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def split_model_into_stages(
    all_layers: list[nn.Module], num_stages: int
) -> list[PipelineStage]:
    """تقسيم قائمة طبقات متتالية إلى مراحل متوازنة تقريبًا للأنبوب."""
    n = len(all_layers)
    base_size = n // num_stages
    remainder = n % num_stages

    stages: list[PipelineStage] = []
    start = 0
    for stage_idx in range(num_stages):
        # توزيع الطبقات الإضافية على المراحل الأولى لموازنة الحمل
        size = base_size + (1 if stage_idx < remainder else 0)
        stages.append(PipelineStage(all_layers[start : start + size]))
        start += size
    return stages


def simulate_gpipe_schedule(num_stages: int, num_microbatches: int) -> list[list[str]]:
    """محاكاة جدول زمني مبسّط لـ GPipe لتوضيح متى يعمل كل جهاز
    وفي أي مرحلة زمنية (time slot)، بما يكشف بصريًا حجم الفقاعة."""
    total_slots = num_stages + num_microbatches - 1
    schedule: list[list[str]] = [["idle"] * total_slots for _ in range(num_stages)]

    for mb in range(num_microbatches):
        for stage in range(num_stages):
            time_slot = stage + mb
            schedule[stage][time_slot] = f"F(mb={mb})"

    bubble_slots = sum(row.count("idle") for row in schedule)
    total_cells = num_stages * total_slots
    print(f"نسبة الفقاعة التقريبية: {bubble_slots / total_cells:.2%}")
    return schedule


if __name__ == "__main__":
    # مثال: 4 مراحل، 8 دفعات مصغّرة -> فقاعة أقل بكثير من 4 مراحل و 1 دفعة فقط
    schedule = simulate_gpipe_schedule(num_stages=4, num_microbatches=8)
    for stage_idx, row in enumerate(schedule):
        print(f"GPU{stage_idx}: {row}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class TransformerBlock(nn.Module):
    """كتلة Transformer توضيحية مبسّطة — في الواقع هذه ستكون
    كتلة كاملة بـ Self-Attention و FFN كما في الفصل الحادي عشر."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def wrap_model_with_fsdp(model: nn.Module, device: torch.device) -> FSDP:
    """تغليف النموذج بـ FSDP بسياسة تقسيم تلقائية على مستوى كل كتلة Transformer،
    مع تفعيل الدقة المختلطة BF16 لتسريع الحساب وتقليل حجم الاتصال الشبكي."""

    # سياسة Auto-Wrap: تُحدَّد وحدة التقسيم (Sharding Unit) عند كل TransformerBlock
    # بدلاً من النموذج كاملاً كوحدة واحدة — هذا يسمح بتداخل الاتصال مع الحساب
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,     # الأوزان أثناء الحساب بـ BF16
        reduce_dtype=torch.bfloat16,    # تخفيض التدرّجات بـ BF16 (اتصال أسرع)
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # مطابق لـ ZeRO Stage 3
        device_id=device,
        use_orig_params=True,  # يسمح بالتوافق السلس مع torch.compile ومُحسِّنات PyTorch الحديثة
    )
    return fsdp_model


def train_step_fsdp(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )
    loss.backward()  # Reduce-Scatter يحدث هنا تلقائيًا خلف الكواليس
    optimizer.step()
    return loss.item()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: نفس ترتيب البيانات بالضبط يتكرر كل حقبة لكل جهاز
for epoch in range(epochs):
    for x, y in loader:
        ...

# ✅ صحيح: ضروري لإعادة خلط فعّال عبر الحقبات
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for x, y in loader:
        ...

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: هذا يجعل "Global Batch Size" يتغير تلقائيًا مع عدد الأجهزة!
batch_size = 32  # نفس القيمة بغض النظر عن world_size
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
# مع 8 أجهزة: Global Batch = 32 × 8 = 256 (قد يكون هذا غير مقصود!)

# ✅ صحيح: حدّد Global Batch بوضوح، واشتق منه الحجم المحلي
global_batch_size = 512
local_batch_size = global_batch_size // world_size
loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: قد يسبب تعارضات غير متوقعة أو فشل صامت في تحسينات الـ compiler
fsdp_model = FSDP(model, use_orig_params=False)
compiled_model = torch.compile(fsdp_model)

# ✅ صحيح: use_orig_params=True ضروري للتوافق السلس الحديث
fsdp_model = FSDP(model, use_orig_params=True)
compiled_model = torch.compile(fsdp_model)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-23-chapter-23.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
"""
مشروع تطبيقي: قياس ومقارنة استهلاك الذاكرة بين DDP وFSDP
على نموذج Transformer صغير قابل للتوسيع.

المطلوب إكماله:
1. بناء نموذج Transformer بحجم قابل للضبط (استخدم عدة إعدادات: صغير/متوسط/كبير)
2. تشغيل تدريب قصير (عدة خطوات) بكل من DDP وFSDP
3. تسجيل torch.cuda.max_memory_allocated() لكل إعداد
4. رسم بياني يقارن استهلاك الذاكرة مقابل حجم النموذج للطريقتين
5. تحقق تجريبيًا: عند أي حجم نموذج يصبح FSDP ضروريًا (لا يعمل DDP إطلاقًا)؟
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_scalable_transformer(
    vocab_size: int = 32_000,
    d_model: int = 1024,
    n_layers: int = 12,
    n_heads: int = 16,
) -> nn.Module:
    """TODO: استخدم MiniTransformerLM المعرّف سابقًا في هذا الفصل،
    لكن اجعل الأبعاد قابلة للضبط عبر المعاملات لتجربة أحجام مختلفة."""
    raise NotImplementedError


def measure_peak_memory_ddp(model_config: dict, world_size: int) -> float:
    """TODO: شغّل خطوة تدريب واحدة باستخدام DDP وأرجع ذروة الذاكرة بالجيجابايت.
    استخدم torch.multiprocessing.spawn كما في مثال train_ddp_worker.
    تذكّر: torch.cuda.reset_peak_memory_stats() قبل البدء،
    و torch.cuda.max_memory_allocated() بعد الانتهاء."""
    raise NotImplementedError


def measure_peak_memory_fsdp(model_config: dict, world_size: int) -> float:
    """TODO: نفس الفكرة لكن باستخدام wrap_model_with_fsdp المعرّفة سابقًا."""
    raise NotImplementedError


def run_comparison_experiment() -> None:
    """TODO: جرّب 3-4 أحجام نموذج مختلفة (مثلاً: 50M، 200M، 500M، 1B معامل)
    وسجّل ذروة الذاكرة لكل من DDP وFSDP، ثم اطبع جدول مقارنة نصي.
    سؤال تأملي: هل الفرق بين DDP وFSDP يتسع أم يضيق كلما كبر النموذج؟ لماذا؟"""
    raise NotImplementedError


if __name__ == "__main__":
    run_comparison_experiment()
