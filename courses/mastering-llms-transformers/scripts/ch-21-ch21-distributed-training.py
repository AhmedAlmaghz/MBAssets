#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التوازي الموزّع للتدريب
Mastering Transformers and Large Language Models — Distributed Training

Book slug : mastering-llms-transformers
Chapter   : ch-21-ch21-distributed-training (lesson 21)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-21-ch21-distributed-training.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التوازي الموزّع للتدريب
Mastering Transformers and Large Language Models — Distributed Training

Book slug : mastering-llms-transformers
Chapter   : ch-21-ch21-distributed-training (lesson 21)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-21-ch21-distributed-training.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التوازي الموزّع للتدريب
Mastering Transformers and Large Language Models — Distributed Training

Book slug : mastering-llms-transformers
Chapter   : ch-21-ch21-distributed-training (lesson 21)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_data_parallel(rank: int, world_size: int):
    """
    إعداد بيئة PyTorch Distributed للتدريب بتوازي البيانات.
    'rank' هو رقم تعريف العملية الحالية (0, 1, 2, ...)،
    'world_size' هو إجمالي عدد العمليات (عادة = عدد GPUs).
    """
    dist.init_process_group(
        backend="nccl",  # NCCL هو backend المُحسَّن للاتصال بين GPUs من NVIDIA
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def train_step_with_data_parallelism(model: DDP, optimizer,
                                       micro_batch_input, micro_batch_labels):
    """
    خطوة تدريب واحدة بتوازي البيانات. الأهم هنا أن استدعاء loss.backward()
    يُفعِّل تلقائيًا (ضمنيًا عبر DDP hooks) عملية All-Reduce لتدرجات كل
    المعاملات عبر كل الـ GPUs المشاركة، بحيث يحصل كل GPU على متوسط
    التدرجات من كل النسخ قبل تحديث الأوزان محليًا.
    """
    optimizer.zero_grad()

    logits = model(micro_batch_input)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), micro_batch_labels.view(-1)
    )

    # هذا الاستدعاء يُشغِّل All-Reduce تلقائيًا بفضل DDP
    loss.backward()

    # كل GPU يُحدِّث أوزانه محليًا، لكن بنفس التدرج المُتوسَّط بالضبط
    # مما يضمن تطابق كل النسخ بعد كل خطوة
    optimizer.step()

    return loss.item()

def demonstrate_allreduce_communication_cost(num_params: float,
                                               num_gpus: int,
                                               bandwidth_gbps: float = 400):
    """
    تقدير زمن الاتصال اللازم لعملية All-Reduce لتدرجات نموذج بحجم
    num_params معامل، عبر num_gpus وحدة GPU متصلة بسرعة bandwidth_gbps
    (NVLink الحديث يوفر ~400-900 GB/s بين GPUs داخل نفس العقدة).

    نستخدم تقريب خوارزمية Ring-AllReduce، التي تنقل بيانات بحجم
    يقارب 2×(num_gpus-1)/num_gpus من حجم التدرجات الكلي عبر الشبكة،
    بغض النظر عمليًا عن عدد الـ GPUs (خاصية مهمة لقابلية التوسع).
    """
    gradient_size_bytes = num_params * 4  # افتراض FP32 للتدرجات
    gradient_size_gb = gradient_size_bytes / 1e9

    # خوارزمية Ring All-Reduce: الحجم المنقول تقريبًا ثابت تقريبيًا
    # بغض النظر عن num_gpus (لا يتوسع خطيًا سلبًا مع عدد العقد)
    data_transferred_gb = 2 * gradient_size_gb * (num_gpus - 1) / num_gpus

    communication_time_sec = data_transferred_gb / bandwidth_gbps

    print(f"حجم النموذج: {num_params/1e9:.1f} مليار معامل")
    print(f"حجم التدرجات (FP32): {gradient_size_gb:.2f} GB")
    print(f"عدد GPUs: {num_gpus}")
    print(f"زمن All-Reduce التقديري: {communication_time_sec*1000:.2f} ميلي ثانية")

demonstrate_allreduce_communication_cost(num_params=7e9, num_gpus=8)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة عموديًا عبر عدة GPUs (Megatron-LM style).
    كل GPU يملك عمودًا فرعيًا من مصفوفة الوزن الكاملة، ويحسب شريحته
    من الناتج بشكل مستقل تمامًا دون أي اتصال شبكي أثناء الحساب نفسه.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0, \
            "عدد المخرجات يجب أن يقبل القسمة على عدد GPUs"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل GPU يُنشئ فقط شريحته الخاصة من مصفوفة الوزن الكاملة
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_partition) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x هنا كاملة (مُكرَّرة/مبثوثة على كل GPU)، والناتج شريحة فقط
        return x @ self.weight  # [batch, seq, out_features_per_partition]

class RowParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة أفقيًا. المدخل هنا هو شريحة (ناتج من طبقة
    ColumnParallel سابقة)، والناتج يتطلب عملية All-Reduce لجمع
    المساهمات من كل الـ GPUs قبل الحصول على الناتج الكامل الصحيح.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int, process_group=None):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group

        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_partition, out_features) * 0.02
        )

    def forward(self, x_partition: torch.Tensor) -> torch.Tensor:
        partial_output = x_partition @ self.weight

        # All-Reduce: جمع المساهمات الجزئية من كل الـ GPUs
        # هذه هي نقطة الاتصال الشبكي الوحيدة في مسار TP لهذه الطبقة
        if dist.is_initialized():
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM,
                             group=self.process_group)
        return partial_output

class TensorParallelFFN(nn.Module):
    """
    طبقة Feed-Forward كاملة مُقسَّمة بتوازي المُوتِرات بنمط Megatron-LM:
    - الطبقة الأولى: Column-Parallel (بدون اتصال شبكي)
    - التفعيل: يُطبَّق محليًا على كل شريحة
    - الطبقة الثانية: Row-Parallel (اتصال All-Reduce واحد فقط في النهاية)

    النتيجة: اتصال شبكي واحد فقط لكل طبقة FFN كاملة، بدلاً من
    اتصال بعد كل عملية ضرب مصفوفات منفردة.
    """
    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(x))  # بدون اتصال شبكي
        output = self.fc2(hidden)               # اتصال All-Reduce واحد هنا
        return output

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PipelineStage(nn.Module):
    """
    مرحلة واحدة من Pipeline Parallelism: تحتوي على مجموعة فرعية
    من طبقات النموذج الكلي، وتعمل كجزء من خط أنابيب متعدد المراحل.
    """
    def __init__(self, layers: nn.ModuleList, stage_id: int,
                 is_first_stage: bool, is_last_stage: bool):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def simulate_pipeline_schedule(num_stages: int, num_microbatches: int):
    """
    محاكاة جدول زمني 1F1B (One-Forward-One-Backward) لتوضيح كيف
    تتداخل عمليات Forward وBackward عبر المراحل لتقليل فقاعات التوقف،
    مقارنة بالنهج الساذج (معالجة كل الـ Forward ثم كل الـ Backward).
    """
    print(f"محاكاة Pipeline بـ {num_stages} مراحل و{num_microbatches} micro-batches")
    print("=" * 70)

    # الجدول الزمني الساذج (GPipe الأساسي بدون تداخل):
    # فقاعة = (num_stages - 1) خطوة زمنية ضائعة في البداية والنهاية
    naive_total_time = num_microbatches + 2 * (num_stages - 1)
    useful_time = num_microbatches
    naive_efficiency = useful_time / naive_total_time

    print(f"\nالجدول الساذج (Fill-Drain):")
    print(f"   الزمن الكلي: {naive_total_time} وحدة زمنية")
    print(f"   الزمن المُفيد: {useful_time} وحدة")
    print(f"   الكفاءة: {naive_efficiency*100:.1f}%")

    # بزيادة عدد الـ micro-batches، تتحسن الكفاءة لأن الفقاعة الثابتة
    # (2×(num_stages-1)) تصبح نسبة أصغر من الزمن الكلي
    print(f"\nتأثير زيادة عدد micro-batches على الكفاءة (مع ثبات num_stages={num_stages}):")
    for mb in [num_microbatches, num_microbatches*2, num_microbatches*4, num_microbatches*8]:
        total = mb + 2*(num_stages-1)
        eff = mb / total
        print(f"   {mb} micro-batches → كفاءة {eff*100:.1f}%")

simulate_pipeline_schedule(num_stages=4, num_microbatches=8)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_zero_memory_savings(num_params: float, num_gpus: int,
                                    optimizer_type: str = "adamw",
                                    precision_bytes: int = 2):
    """
    حساب توفير الذاكرة لكل مرحلة من ZeRO مقارنة بـ Data Parallelism
    التقليدي (بدون أي تقسيم)، بافتراض تدريب بدقة مختلطة (Mixed Precision).

    الافتراضات القياسية لـ Mixed Precision Training:
    - الأوزان: نسخة FP16/BF16 (2 بايت) + نسخة FP32 أصلية (4 بايت) = 6 بايت/معامل
    - التدرجات: FP16/BF16 (2 بايت/معامل)
    - حالات AdamW: متوسطين (m, v) بدقة FP32 = 8 بايت/معامل
    """
    # الذاكرة الإجمالية المطلوبة لكل "نوع" من البيانات (بالبايت لكل معامل)
    weights_memory_per_param = 2 + 4  # BF16 عمل + FP32 أصلية (master weights)
    gradients_memory_per_param = 2    # BF16
    optimizer_memory_per_param = 8    # AdamW: m + v بدقة FP32 (4+4)

    total_per_gpu_no_zero = (weights_memory_per_param +
                               gradients_memory_per_param +
                               optimizer_memory_per_param) * num_params

    scenarios = {
        "Data Parallelism (بدون ZeRO) - كل GPU يُكرِّر كل شيء": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param,
        },
        "ZeRO-1 (تقسيم Optimizer States فقط)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-2 (+ تقسيم Gradients)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-3 (+ تقسيم Parameters نفسها)": {
            "weights": weights_memory_per_param / num_gpus,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
    }

    print(f"نموذج بحجم {num_params/1e9:.0f} مليار معامل، موزَّع على {num_gpus} GPU")
    print("=" * 75)
    for name, breakdown in scenarios.items():
        total_bytes_per_param = sum(breakdown.values())
        total_gb = (total_bytes_per_param * num_params) / 1e9
        print(f"\n{name}:")
        print(f"   الذاكرة المطلوبة لكل GPU: {total_gb:.1f} GB")

calculate_zero_memory_savings(num_params=70e9, num_gpus=64)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp_model(model: nn.Module, transformer_layer_class) -> FSDP:
    """
    تغليف نموذج Transformer بـ FSDP لتفعيل تقسيم الأوزان/التدرجات/
    حالات المُحسِّن بنمط ZeRO-3 تلقائيًا عبر كل الـ GPUs المشاركة.
    """
    # سياسة التغليف التلقائي: نُغلِّف كل طبقة Transformer بشكل منفصل
    # (بدلاً من تغليف النموذج كاملاً كوحدة واحدة)، مما يسمح بتداخل
    # الاتصال الشبكي (All-Gather/Reduce-Scatter) مع الحوسبة بكفاءة أعلى
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )

    # إعدادات الدقة المختلطة: BF16 للحوسبة والاتصال، FP32 للتجميع النهائي
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # هذا يعني ZeRO-3 الكامل
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # تداخل الاتصال مع الحوسبة
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,  # تحكم في استهلاك الذاكرة أثناء All-Gather
    )

    return fsdp_model

def compare_sharding_strategies():
    """
    مقارنة استراتيجيات التقسيم المتاحة في FSDP، كل منها يناظر
    مرحلة مختلفة من ZeRO بالضبط:
    """
    strategies = {
        "FULL_SHARD": "يناظر ZeRO-3 بالكامل: تقسيم Params+Gradients+Optimizer States",
        "SHARD_GRAD_OP": "يناظر ZeRO-2: تقسيم Gradients+Optimizer States فقط، الأوزان مُكرَّرة",
        "NO_SHARD": "يناظر DDP التقليدي: لا تقسيم إطلاقًا (كل شيء مُكرَّر)",
        "HYBRID_SHARD": "FULL_SHARD داخل كل عقدة + تكرار بين العقد (توازن اتصال/ذاكرة)",
    }
    for strategy, description in strategies.items():
        print(f"{strategy}: {description}")

compare_sharding_strategies()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def calculate_3d_parallelism_configuration(total_gpus: int,
                                             tensor_parallel_size: int,
                                             pipeline_parallel_size: int) -> dict:
    """
    حساب حجم Data Parallelism الناتج ضمنيًا من اختيار أحجام TP وPP،
    بما أن: total_gpus = TP_size × PP_size × DP_size
    """
    assert total_gpus % (tensor_parallel_size * pipeline_parallel_size) == 0, \
        "إجمالي الـ GPUs يجب أن يقبل القسمة على حاصل ضرب TP×PP"

    data_parallel_size = total_gpus // (tensor_parallel_size * pipeline_parallel_size)

    return {
        "total_gpus": total_gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "effective_batch_multiplier": data_parallel_size,
    }

config = calculate_3d_parallelism_configuration(
    total_gpus=1024, tensor_parallel_size=8, pipeline_parallel_size=16
)
for k, v in config.items():
    print(f"{k}: {v}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def solve_memory_feasibility_example():
    num_params = 13e9
    num_gpus = 32
    gpu_memory_gb = 80

    bytes_per_param_total = 6 + 2 + 8  # 16 بايت/معامل

    memory_no_zero_gb = (num_params * bytes_per_param_total) / 1e9
    memory_zero3_gb = memory_no_zero_gb / num_gpus

    print(f"الذاكرة بدون ZeRO (لكل GPU): {memory_no_zero_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_no_zero_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"الذاكرة مع ZeRO-3 (لكل GPU): {memory_zero3_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_zero3_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"\nالمساحة المتبقية لكل GPU بعد ZeRO-3: "
          f"{gpu_memory_gb - memory_zero3_gb:.1f} GB (للتنشيطات وBatch أكبر)")

solve_memory_feasibility_example()

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_data_parallel(rank: int, world_size: int):
    """
    إعداد بيئة PyTorch Distributed للتدريب بتوازي البيانات.
    'rank' هو رقم تعريف العملية الحالية (0, 1, 2, ...)،
    'world_size' هو إجمالي عدد العمليات (عادة = عدد GPUs).
    """
    dist.init_process_group(
        backend="nccl",  # NCCL هو backend المُحسَّن للاتصال بين GPUs من NVIDIA
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def train_step_with_data_parallelism(model: DDP, optimizer,
                                       micro_batch_input, micro_batch_labels):
    """
    خطوة تدريب واحدة بتوازي البيانات. الأهم هنا أن استدعاء loss.backward()
    يُفعِّل تلقائيًا (ضمنيًا عبر DDP hooks) عملية All-Reduce لتدرجات كل
    المعاملات عبر كل الـ GPUs المشاركة، بحيث يحصل كل GPU على متوسط
    التدرجات من كل النسخ قبل تحديث الأوزان محليًا.
    """
    optimizer.zero_grad()

    logits = model(micro_batch_input)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), micro_batch_labels.view(-1)
    )

    # هذا الاستدعاء يُشغِّل All-Reduce تلقائيًا بفضل DDP
    loss.backward()

    # كل GPU يُحدِّث أوزانه محليًا، لكن بنفس التدرج المُتوسَّط بالضبط
    # مما يضمن تطابق كل النسخ بعد كل خطوة
    optimizer.step()

    return loss.item()

def demonstrate_allreduce_communication_cost(num_params: float,
                                               num_gpus: int,
                                               bandwidth_gbps: float = 400):
    """
    تقدير زمن الاتصال اللازم لعملية All-Reduce لتدرجات نموذج بحجم
    num_params معامل، عبر num_gpus وحدة GPU متصلة بسرعة bandwidth_gbps
    (NVLink الحديث يوفر ~400-900 GB/s بين GPUs داخل نفس العقدة).

    نستخدم تقريب خوارزمية Ring-AllReduce، التي تنقل بيانات بحجم
    يقارب 2×(num_gpus-1)/num_gpus من حجم التدرجات الكلي عبر الشبكة،
    بغض النظر عمليًا عن عدد الـ GPUs (خاصية مهمة لقابلية التوسع).
    """
    gradient_size_bytes = num_params * 4  # افتراض FP32 للتدرجات
    gradient_size_gb = gradient_size_bytes / 1e9

    # خوارزمية Ring All-Reduce: الحجم المنقول تقريبًا ثابت تقريبيًا
    # بغض النظر عن num_gpus (لا يتوسع خطيًا سلبًا مع عدد العقد)
    data_transferred_gb = 2 * gradient_size_gb * (num_gpus - 1) / num_gpus

    communication_time_sec = data_transferred_gb / bandwidth_gbps

    print(f"حجم النموذج: {num_params/1e9:.1f} مليار معامل")
    print(f"حجم التدرجات (FP32): {gradient_size_gb:.2f} GB")
    print(f"عدد GPUs: {num_gpus}")
    print(f"زمن All-Reduce التقديري: {communication_time_sec*1000:.2f} ميلي ثانية")

demonstrate_allreduce_communication_cost(num_params=7e9, num_gpus=8)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة عموديًا عبر عدة GPUs (Megatron-LM style).
    كل GPU يملك عمودًا فرعيًا من مصفوفة الوزن الكاملة، ويحسب شريحته
    من الناتج بشكل مستقل تمامًا دون أي اتصال شبكي أثناء الحساب نفسه.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0, \
            "عدد المخرجات يجب أن يقبل القسمة على عدد GPUs"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل GPU يُنشئ فقط شريحته الخاصة من مصفوفة الوزن الكاملة
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_partition) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x هنا كاملة (مُكرَّرة/مبثوثة على كل GPU)، والناتج شريحة فقط
        return x @ self.weight  # [batch, seq, out_features_per_partition]

class RowParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة أفقيًا. المدخل هنا هو شريحة (ناتج من طبقة
    ColumnParallel سابقة)، والناتج يتطلب عملية All-Reduce لجمع
    المساهمات من كل الـ GPUs قبل الحصول على الناتج الكامل الصحيح.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int, process_group=None):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group

        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_partition, out_features) * 0.02
        )

    def forward(self, x_partition: torch.Tensor) -> torch.Tensor:
        partial_output = x_partition @ self.weight

        # All-Reduce: جمع المساهمات الجزئية من كل الـ GPUs
        # هذه هي نقطة الاتصال الشبكي الوحيدة في مسار TP لهذه الطبقة
        if dist.is_initialized():
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM,
                             group=self.process_group)
        return partial_output

class TensorParallelFFN(nn.Module):
    """
    طبقة Feed-Forward كاملة مُقسَّمة بتوازي المُوتِرات بنمط Megatron-LM:
    - الطبقة الأولى: Column-Parallel (بدون اتصال شبكي)
    - التفعيل: يُطبَّق محليًا على كل شريحة
    - الطبقة الثانية: Row-Parallel (اتصال All-Reduce واحد فقط في النهاية)

    النتيجة: اتصال شبكي واحد فقط لكل طبقة FFN كاملة، بدلاً من
    اتصال بعد كل عملية ضرب مصفوفات منفردة.
    """
    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(x))  # بدون اتصال شبكي
        output = self.fc2(hidden)               # اتصال All-Reduce واحد هنا
        return output

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PipelineStage(nn.Module):
    """
    مرحلة واحدة من Pipeline Parallelism: تحتوي على مجموعة فرعية
    من طبقات النموذج الكلي، وتعمل كجزء من خط أنابيب متعدد المراحل.
    """
    def __init__(self, layers: nn.ModuleList, stage_id: int,
                 is_first_stage: bool, is_last_stage: bool):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def simulate_pipeline_schedule(num_stages: int, num_microbatches: int):
    """
    محاكاة جدول زمني 1F1B (One-Forward-One-Backward) لتوضيح كيف
    تتداخل عمليات Forward وBackward عبر المراحل لتقليل فقاعات التوقف،
    مقارنة بالنهج الساذج (معالجة كل الـ Forward ثم كل الـ Backward).
    """
    print(f"محاكاة Pipeline بـ {num_stages} مراحل و{num_microbatches} micro-batches")
    print("=" * 70)

    # الجدول الزمني الساذج (GPipe الأساسي بدون تداخل):
    # فقاعة = (num_stages - 1) خطوة زمنية ضائعة في البداية والنهاية
    naive_total_time = num_microbatches + 2 * (num_stages - 1)
    useful_time = num_microbatches
    naive_efficiency = useful_time / naive_total_time

    print(f"\nالجدول الساذج (Fill-Drain):")
    print(f"   الزمن الكلي: {naive_total_time} وحدة زمنية")
    print(f"   الزمن المُفيد: {useful_time} وحدة")
    print(f"   الكفاءة: {naive_efficiency*100:.1f}%")

    # بزيادة عدد الـ micro-batches، تتحسن الكفاءة لأن الفقاعة الثابتة
    # (2×(num_stages-1)) تصبح نسبة أصغر من الزمن الكلي
    print(f"\nتأثير زيادة عدد micro-batches على الكفاءة (مع ثبات num_stages={num_stages}):")
    for mb in [num_microbatches, num_microbatches*2, num_microbatches*4, num_microbatches*8]:
        total = mb + 2*(num_stages-1)
        eff = mb / total
        print(f"   {mb} micro-batches → كفاءة {eff*100:.1f}%")

simulate_pipeline_schedule(num_stages=4, num_microbatches=8)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_zero_memory_savings(num_params: float, num_gpus: int,
                                    optimizer_type: str = "adamw",
                                    precision_bytes: int = 2):
    """
    حساب توفير الذاكرة لكل مرحلة من ZeRO مقارنة بـ Data Parallelism
    التقليدي (بدون أي تقسيم)، بافتراض تدريب بدقة مختلطة (Mixed Precision).

    الافتراضات القياسية لـ Mixed Precision Training:
    - الأوزان: نسخة FP16/BF16 (2 بايت) + نسخة FP32 أصلية (4 بايت) = 6 بايت/معامل
    - التدرجات: FP16/BF16 (2 بايت/معامل)
    - حالات AdamW: متوسطين (m, v) بدقة FP32 = 8 بايت/معامل
    """
    # الذاكرة الإجمالية المطلوبة لكل "نوع" من البيانات (بالبايت لكل معامل)
    weights_memory_per_param = 2 + 4  # BF16 عمل + FP32 أصلية (master weights)
    gradients_memory_per_param = 2    # BF16
    optimizer_memory_per_param = 8    # AdamW: m + v بدقة FP32 (4+4)

    total_per_gpu_no_zero = (weights_memory_per_param +
                               gradients_memory_per_param +
                               optimizer_memory_per_param) * num_params

    scenarios = {
        "Data Parallelism (بدون ZeRO) - كل GPU يُكرِّر كل شيء": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param,
        },
        "ZeRO-1 (تقسيم Optimizer States فقط)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-2 (+ تقسيم Gradients)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-3 (+ تقسيم Parameters نفسها)": {
            "weights": weights_memory_per_param / num_gpus,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
    }

    print(f"نموذج بحجم {num_params/1e9:.0f} مليار معامل، موزَّع على {num_gpus} GPU")
    print("=" * 75)
    for name, breakdown in scenarios.items():
        total_bytes_per_param = sum(breakdown.values())
        total_gb = (total_bytes_per_param * num_params) / 1e9
        print(f"\n{name}:")
        print(f"   الذاكرة المطلوبة لكل GPU: {total_gb:.1f} GB")

calculate_zero_memory_savings(num_params=70e9, num_gpus=64)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp_model(model: nn.Module, transformer_layer_class) -> FSDP:
    """
    تغليف نموذج Transformer بـ FSDP لتفعيل تقسيم الأوزان/التدرجات/
    حالات المُحسِّن بنمط ZeRO-3 تلقائيًا عبر كل الـ GPUs المشاركة.
    """
    # سياسة التغليف التلقائي: نُغلِّف كل طبقة Transformer بشكل منفصل
    # (بدلاً من تغليف النموذج كاملاً كوحدة واحدة)، مما يسمح بتداخل
    # الاتصال الشبكي (All-Gather/Reduce-Scatter) مع الحوسبة بكفاءة أعلى
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )

    # إعدادات الدقة المختلطة: BF16 للحوسبة والاتصال، FP32 للتجميع النهائي
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # هذا يعني ZeRO-3 الكامل
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # تداخل الاتصال مع الحوسبة
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,  # تحكم في استهلاك الذاكرة أثناء All-Gather
    )

    return fsdp_model

def compare_sharding_strategies():
    """
    مقارنة استراتيجيات التقسيم المتاحة في FSDP، كل منها يناظر
    مرحلة مختلفة من ZeRO بالضبط:
    """
    strategies = {
        "FULL_SHARD": "يناظر ZeRO-3 بالكامل: تقسيم Params+Gradients+Optimizer States",
        "SHARD_GRAD_OP": "يناظر ZeRO-2: تقسيم Gradients+Optimizer States فقط، الأوزان مُكرَّرة",
        "NO_SHARD": "يناظر DDP التقليدي: لا تقسيم إطلاقًا (كل شيء مُكرَّر)",
        "HYBRID_SHARD": "FULL_SHARD داخل كل عقدة + تكرار بين العقد (توازن اتصال/ذاكرة)",
    }
    for strategy, description in strategies.items():
        print(f"{strategy}: {description}")

compare_sharding_strategies()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def calculate_3d_parallelism_configuration(total_gpus: int,
                                             tensor_parallel_size: int,
                                             pipeline_parallel_size: int) -> dict:
    """
    حساب حجم Data Parallelism الناتج ضمنيًا من اختيار أحجام TP وPP،
    بما أن: total_gpus = TP_size × PP_size × DP_size
    """
    assert total_gpus % (tensor_parallel_size * pipeline_parallel_size) == 0, \
        "إجمالي الـ GPUs يجب أن يقبل القسمة على حاصل ضرب TP×PP"

    data_parallel_size = total_gpus // (tensor_parallel_size * pipeline_parallel_size)

    return {
        "total_gpus": total_gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "effective_batch_multiplier": data_parallel_size,
    }

config = calculate_3d_parallelism_configuration(
    total_gpus=1024, tensor_parallel_size=8, pipeline_parallel_size=16
)
for k, v in config.items():
    print(f"{k}: {v}")

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def solve_memory_feasibility_example():
    num_params = 13e9
    num_gpus = 32
    gpu_memory_gb = 80

    bytes_per_param_total = 6 + 2 + 8  # 16 بايت/معامل

    memory_no_zero_gb = (num_params * bytes_per_param_total) / 1e9
    memory_zero3_gb = memory_no_zero_gb / num_gpus

    print(f"الذاكرة بدون ZeRO (لكل GPU): {memory_no_zero_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_no_zero_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"الذاكرة مع ZeRO-3 (لكل GPU): {memory_zero3_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_zero3_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"\nالمساحة المتبقية لكل GPU بعد ZeRO-3: "
          f"{gpu_memory_gb - memory_zero3_gb:.1f} GB (للتنشيطات وBatch أكبر)")

solve_memory_feasibility_example()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التوازي الموزّع للتدريب
Mastering Transformers and Large Language Models — Distributed Training

Book slug : mastering-llms-transformers
Chapter   : ch-21-ch21-distributed-training (lesson 21)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [002] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_data_parallel(rank: int, world_size: int):
    """
    إعداد بيئة PyTorch Distributed للتدريب بتوازي البيانات.
    'rank' هو رقم تعريف العملية الحالية (0, 1, 2, ...)،
    'world_size' هو إجمالي عدد العمليات (عادة = عدد GPUs).
    """
    dist.init_process_group(
        backend="nccl",  # NCCL هو backend المُحسَّن للاتصال بين GPUs من NVIDIA
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def train_step_with_data_parallelism(model: DDP, optimizer,
                                       micro_batch_input, micro_batch_labels):
    """
    خطوة تدريب واحدة بتوازي البيانات. الأهم هنا أن استدعاء loss.backward()
    يُفعِّل تلقائيًا (ضمنيًا عبر DDP hooks) عملية All-Reduce لتدرجات كل
    المعاملات عبر كل الـ GPUs المشاركة، بحيث يحصل كل GPU على متوسط
    التدرجات من كل النسخ قبل تحديث الأوزان محليًا.
    """
    optimizer.zero_grad()

    logits = model(micro_batch_input)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), micro_batch_labels.view(-1)
    )

    # هذا الاستدعاء يُشغِّل All-Reduce تلقائيًا بفضل DDP
    loss.backward()

    # كل GPU يُحدِّث أوزانه محليًا، لكن بنفس التدرج المُتوسَّط بالضبط
    # مما يضمن تطابق كل النسخ بعد كل خطوة
    optimizer.step()

    return loss.item()

def demonstrate_allreduce_communication_cost(num_params: float,
                                               num_gpus: int,
                                               bandwidth_gbps: float = 400):
    """
    تقدير زمن الاتصال اللازم لعملية All-Reduce لتدرجات نموذج بحجم
    num_params معامل، عبر num_gpus وحدة GPU متصلة بسرعة bandwidth_gbps
    (NVLink الحديث يوفر ~400-900 GB/s بين GPUs داخل نفس العقدة).

    نستخدم تقريب خوارزمية Ring-AllReduce، التي تنقل بيانات بحجم
    يقارب 2×(num_gpus-1)/num_gpus من حجم التدرجات الكلي عبر الشبكة،
    بغض النظر عمليًا عن عدد الـ GPUs (خاصية مهمة لقابلية التوسع).
    """
    gradient_size_bytes = num_params * 4  # افتراض FP32 للتدرجات
    gradient_size_gb = gradient_size_bytes / 1e9

    # خوارزمية Ring All-Reduce: الحجم المنقول تقريبًا ثابت تقريبيًا
    # بغض النظر عن num_gpus (لا يتوسع خطيًا سلبًا مع عدد العقد)
    data_transferred_gb = 2 * gradient_size_gb * (num_gpus - 1) / num_gpus

    communication_time_sec = data_transferred_gb / bandwidth_gbps

    print(f"حجم النموذج: {num_params/1e9:.1f} مليار معامل")
    print(f"حجم التدرجات (FP32): {gradient_size_gb:.2f} GB")
    print(f"عدد GPUs: {num_gpus}")
    print(f"زمن All-Reduce التقديري: {communication_time_sec*1000:.2f} ميلي ثانية")

demonstrate_allreduce_communication_cost(num_params=7e9, num_gpus=8)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة عموديًا عبر عدة GPUs (Megatron-LM style).
    كل GPU يملك عمودًا فرعيًا من مصفوفة الوزن الكاملة، ويحسب شريحته
    من الناتج بشكل مستقل تمامًا دون أي اتصال شبكي أثناء الحساب نفسه.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0, \
            "عدد المخرجات يجب أن يقبل القسمة على عدد GPUs"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل GPU يُنشئ فقط شريحته الخاصة من مصفوفة الوزن الكاملة
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_partition) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x هنا كاملة (مُكرَّرة/مبثوثة على كل GPU)، والناتج شريحة فقط
        return x @ self.weight  # [batch, seq, out_features_per_partition]

class RowParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة أفقيًا. المدخل هنا هو شريحة (ناتج من طبقة
    ColumnParallel سابقة)، والناتج يتطلب عملية All-Reduce لجمع
    المساهمات من كل الـ GPUs قبل الحصول على الناتج الكامل الصحيح.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int, process_group=None):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group

        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_partition, out_features) * 0.02
        )

    def forward(self, x_partition: torch.Tensor) -> torch.Tensor:
        partial_output = x_partition @ self.weight

        # All-Reduce: جمع المساهمات الجزئية من كل الـ GPUs
        # هذه هي نقطة الاتصال الشبكي الوحيدة في مسار TP لهذه الطبقة
        if dist.is_initialized():
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM,
                             group=self.process_group)
        return partial_output

class TensorParallelFFN(nn.Module):
    """
    طبقة Feed-Forward كاملة مُقسَّمة بتوازي المُوتِرات بنمط Megatron-LM:
    - الطبقة الأولى: Column-Parallel (بدون اتصال شبكي)
    - التفعيل: يُطبَّق محليًا على كل شريحة
    - الطبقة الثانية: Row-Parallel (اتصال All-Reduce واحد فقط في النهاية)

    النتيجة: اتصال شبكي واحد فقط لكل طبقة FFN كاملة، بدلاً من
    اتصال بعد كل عملية ضرب مصفوفات منفردة.
    """
    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(x))  # بدون اتصال شبكي
        output = self.fc2(hidden)               # اتصال All-Reduce واحد هنا
        return output

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class PipelineStage(nn.Module):
    """
    مرحلة واحدة من Pipeline Parallelism: تحتوي على مجموعة فرعية
    من طبقات النموذج الكلي، وتعمل كجزء من خط أنابيب متعدد المراحل.
    """
    def __init__(self, layers: nn.ModuleList, stage_id: int,
                 is_first_stage: bool, is_last_stage: bool):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def simulate_pipeline_schedule(num_stages: int, num_microbatches: int):
    """
    محاكاة جدول زمني 1F1B (One-Forward-One-Backward) لتوضيح كيف
    تتداخل عمليات Forward وBackward عبر المراحل لتقليل فقاعات التوقف،
    مقارنة بالنهج الساذج (معالجة كل الـ Forward ثم كل الـ Backward).
    """
    print(f"محاكاة Pipeline بـ {num_stages} مراحل و{num_microbatches} micro-batches")
    print("=" * 70)

    # الجدول الزمني الساذج (GPipe الأساسي بدون تداخل):
    # فقاعة = (num_stages - 1) خطوة زمنية ضائعة في البداية والنهاية
    naive_total_time = num_microbatches + 2 * (num_stages - 1)
    useful_time = num_microbatches
    naive_efficiency = useful_time / naive_total_time

    print(f"\nالجدول الساذج (Fill-Drain):")
    print(f"   الزمن الكلي: {naive_total_time} وحدة زمنية")
    print(f"   الزمن المُفيد: {useful_time} وحدة")
    print(f"   الكفاءة: {naive_efficiency*100:.1f}%")

    # بزيادة عدد الـ micro-batches، تتحسن الكفاءة لأن الفقاعة الثابتة
    # (2×(num_stages-1)) تصبح نسبة أصغر من الزمن الكلي
    print(f"\nتأثير زيادة عدد micro-batches على الكفاءة (مع ثبات num_stages={num_stages}):")
    for mb in [num_microbatches, num_microbatches*2, num_microbatches*4, num_microbatches*8]:
        total = mb + 2*(num_stages-1)
        eff = mb / total
        print(f"   {mb} micro-batches → كفاءة {eff*100:.1f}%")

simulate_pipeline_schedule(num_stages=4, num_microbatches=8)

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def calculate_zero_memory_savings(num_params: float, num_gpus: int,
                                    optimizer_type: str = "adamw",
                                    precision_bytes: int = 2):
    """
    حساب توفير الذاكرة لكل مرحلة من ZeRO مقارنة بـ Data Parallelism
    التقليدي (بدون أي تقسيم)، بافتراض تدريب بدقة مختلطة (Mixed Precision).

    الافتراضات القياسية لـ Mixed Precision Training:
    - الأوزان: نسخة FP16/BF16 (2 بايت) + نسخة FP32 أصلية (4 بايت) = 6 بايت/معامل
    - التدرجات: FP16/BF16 (2 بايت/معامل)
    - حالات AdamW: متوسطين (m, v) بدقة FP32 = 8 بايت/معامل
    """
    # الذاكرة الإجمالية المطلوبة لكل "نوع" من البيانات (بالبايت لكل معامل)
    weights_memory_per_param = 2 + 4  # BF16 عمل + FP32 أصلية (master weights)
    gradients_memory_per_param = 2    # BF16
    optimizer_memory_per_param = 8    # AdamW: m + v بدقة FP32 (4+4)

    total_per_gpu_no_zero = (weights_memory_per_param +
                               gradients_memory_per_param +
                               optimizer_memory_per_param) * num_params

    scenarios = {
        "Data Parallelism (بدون ZeRO) - كل GPU يُكرِّر كل شيء": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param,
        },
        "ZeRO-1 (تقسيم Optimizer States فقط)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-2 (+ تقسيم Gradients)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-3 (+ تقسيم Parameters نفسها)": {
            "weights": weights_memory_per_param / num_gpus,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
    }

    print(f"نموذج بحجم {num_params/1e9:.0f} مليار معامل، موزَّع على {num_gpus} GPU")
    print("=" * 75)
    for name, breakdown in scenarios.items():
        total_bytes_per_param = sum(breakdown.values())
        total_gb = (total_bytes_per_param * num_params) / 1e9
        print(f"\n{name}:")
        print(f"   الذاكرة المطلوبة لكل GPU: {total_gb:.1f} GB")

calculate_zero_memory_savings(num_params=70e9, num_gpus=64)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp_model(model: nn.Module, transformer_layer_class) -> FSDP:
    """
    تغليف نموذج Transformer بـ FSDP لتفعيل تقسيم الأوزان/التدرجات/
    حالات المُحسِّن بنمط ZeRO-3 تلقائيًا عبر كل الـ GPUs المشاركة.
    """
    # سياسة التغليف التلقائي: نُغلِّف كل طبقة Transformer بشكل منفصل
    # (بدلاً من تغليف النموذج كاملاً كوحدة واحدة)، مما يسمح بتداخل
    # الاتصال الشبكي (All-Gather/Reduce-Scatter) مع الحوسبة بكفاءة أعلى
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )

    # إعدادات الدقة المختلطة: BF16 للحوسبة والاتصال، FP32 للتجميع النهائي
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # هذا يعني ZeRO-3 الكامل
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # تداخل الاتصال مع الحوسبة
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,  # تحكم في استهلاك الذاكرة أثناء All-Gather
    )

    return fsdp_model

def compare_sharding_strategies():
    """
    مقارنة استراتيجيات التقسيم المتاحة في FSDP، كل منها يناظر
    مرحلة مختلفة من ZeRO بالضبط:
    """
    strategies = {
        "FULL_SHARD": "يناظر ZeRO-3 بالكامل: تقسيم Params+Gradients+Optimizer States",
        "SHARD_GRAD_OP": "يناظر ZeRO-2: تقسيم Gradients+Optimizer States فقط، الأوزان مُكرَّرة",
        "NO_SHARD": "يناظر DDP التقليدي: لا تقسيم إطلاقًا (كل شيء مُكرَّر)",
        "HYBRID_SHARD": "FULL_SHARD داخل كل عقدة + تكرار بين العقد (توازن اتصال/ذاكرة)",
    }
    for strategy, description in strategies.items():
        print(f"{strategy}: {description}")

compare_sharding_strategies()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def calculate_3d_parallelism_configuration(total_gpus: int,
                                             tensor_parallel_size: int,
                                             pipeline_parallel_size: int) -> dict:
    """
    حساب حجم Data Parallelism الناتج ضمنيًا من اختيار أحجام TP وPP،
    بما أن: total_gpus = TP_size × PP_size × DP_size
    """
    assert total_gpus % (tensor_parallel_size * pipeline_parallel_size) == 0, \
        "إجمالي الـ GPUs يجب أن يقبل القسمة على حاصل ضرب TP×PP"

    data_parallel_size = total_gpus // (tensor_parallel_size * pipeline_parallel_size)

    return {
        "total_gpus": total_gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "effective_batch_multiplier": data_parallel_size,
    }

config = calculate_3d_parallelism_configuration(
    total_gpus=1024, tensor_parallel_size=8, pipeline_parallel_size=16
)
for k, v in config.items():
    print(f"{k}: {v}")

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def solve_memory_feasibility_example():
    num_params = 13e9
    num_gpus = 32
    gpu_memory_gb = 80

    bytes_per_param_total = 6 + 2 + 8  # 16 بايت/معامل

    memory_no_zero_gb = (num_params * bytes_per_param_total) / 1e9
    memory_zero3_gb = memory_no_zero_gb / num_gpus

    print(f"الذاكرة بدون ZeRO (لكل GPU): {memory_no_zero_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_no_zero_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"الذاكرة مع ZeRO-3 (لكل GPU): {memory_zero3_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_zero3_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"\nالمساحة المتبقية لكل GPU بعد ZeRO-3: "
          f"{gpu_memory_gb - memory_zero3_gb:.1f} GB (للتنشيطات وBatch أكبر)")

solve_memory_feasibility_example()

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-05-building-llms/ch21-distributed-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
# === إعداد البيئة ===
# يفحص هذا القسم الحزم المطلوبة ويحاول تثبيت الناقص منها تلقائيًا.
# شغّل هذه الخلية أولًا، ثم بقية خلايا الدفتر.
import importlib.util
import subprocess
import sys

print("Python:", sys.version.split()[0])

# (اسم وحدة الاستيراد، اسم حزمة pip)
REQUIRED = [
    ("torch", "torch")
]
AUTO_INSTALL = {"PIL", "bs4", "datasets", "faiss", "fasttext", "librosa", "networkx", "opentelemetry", "peft", "prometheus_client", "pydantic", "rank_bm25", "requests", "sentence_transformers", "sentencepiece", "soundfile", "tokenizers", "trafilatura", "transformers", "wandb", "yaml"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if not REQUIRED:
    print("لا توجد مكتبات خارجية مطلوبة؛ يكفي Python القياسي.")
else:
    missing = [(module, package) for module, package in REQUIRED if not has_module(module)]
    auto_specs = []
    manual_specs = []
    for module, package in missing:
        specs = package.split()
        if module in AUTO_INSTALL:
            auto_specs.extend(specs)
        else:
            manual_specs.extend(specs)
    auto_specs = list(dict.fromkeys(auto_specs))
    manual_specs = list(dict.fromkeys(manual_specs))

    if auto_specs:
        print("⏳ تثبيت تلقائي:", " ".join(auto_specs))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *auto_specs])
            print("✅ تم التثبيت.")
        except Exception as exc:
            print("⚠️ تعذّر التثبيت التلقائي:", exc)
            print("شغّل يدويًا: %pip install", " ".join(auto_specs))
    if manual_specs:
        print("⚠️ حزم تحتاج تثبيتًا يدويًا:", " ".join(manual_specs))
        print("شغّل: %pip install", " ".join(manual_specs))
    if not auto_specs and not manual_specs:
        print("✅ كل الحزم المطلوبة متاحة.")

    print("\nالحالة النهائية:")
    for module, package in REQUIRED:
        print(("✅" if has_module(module) else "⚠️"), package)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_data_parallel(rank: int, world_size: int):
    """
    إعداد بيئة PyTorch Distributed للتدريب بتوازي البيانات.
    'rank' هو رقم تعريف العملية الحالية (0, 1, 2, ...)،
    'world_size' هو إجمالي عدد العمليات (عادة = عدد GPUs).
    """
    dist.init_process_group(
        backend="nccl",  # NCCL هو backend المُحسَّن للاتصال بين GPUs من NVIDIA
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def train_step_with_data_parallelism(model: DDP, optimizer,
                                       micro_batch_input, micro_batch_labels):
    """
    خطوة تدريب واحدة بتوازي البيانات. الأهم هنا أن استدعاء loss.backward()
    يُفعِّل تلقائيًا (ضمنيًا عبر DDP hooks) عملية All-Reduce لتدرجات كل
    المعاملات عبر كل الـ GPUs المشاركة، بحيث يحصل كل GPU على متوسط
    التدرجات من كل النسخ قبل تحديث الأوزان محليًا.
    """
    optimizer.zero_grad()

    logits = model(micro_batch_input)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), micro_batch_labels.view(-1)
    )

    # هذا الاستدعاء يُشغِّل All-Reduce تلقائيًا بفضل DDP
    loss.backward()

    # كل GPU يُحدِّث أوزانه محليًا، لكن بنفس التدرج المُتوسَّط بالضبط
    # مما يضمن تطابق كل النسخ بعد كل خطوة
    optimizer.step()

    return loss.item()

def demonstrate_allreduce_communication_cost(num_params: float,
                                               num_gpus: int,
                                               bandwidth_gbps: float = 400):
    """
    تقدير زمن الاتصال اللازم لعملية All-Reduce لتدرجات نموذج بحجم
    num_params معامل، عبر num_gpus وحدة GPU متصلة بسرعة bandwidth_gbps
    (NVLink الحديث يوفر ~400-900 GB/s بين GPUs داخل نفس العقدة).

    نستخدم تقريب خوارزمية Ring-AllReduce، التي تنقل بيانات بحجم
    يقارب 2×(num_gpus-1)/num_gpus من حجم التدرجات الكلي عبر الشبكة،
    بغض النظر عمليًا عن عدد الـ GPUs (خاصية مهمة لقابلية التوسع).
    """
    gradient_size_bytes = num_params * 4  # افتراض FP32 للتدرجات
    gradient_size_gb = gradient_size_bytes / 1e9

    # خوارزمية Ring All-Reduce: الحجم المنقول تقريبًا ثابت تقريبيًا
    # بغض النظر عن num_gpus (لا يتوسع خطيًا سلبًا مع عدد العقد)
    data_transferred_gb = 2 * gradient_size_gb * (num_gpus - 1) / num_gpus

    communication_time_sec = data_transferred_gb / bandwidth_gbps

    print(f"حجم النموذج: {num_params/1e9:.1f} مليار معامل")
    print(f"حجم التدرجات (FP32): {gradient_size_gb:.2f} GB")
    print(f"عدد GPUs: {num_gpus}")
    print(f"زمن All-Reduce التقديري: {communication_time_sec*1000:.2f} ميلي ثانية")

demonstrate_allreduce_communication_cost(num_params=7e9, num_gpus=8)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة عموديًا عبر عدة GPUs (Megatron-LM style).
    كل GPU يملك عمودًا فرعيًا من مصفوفة الوزن الكاملة، ويحسب شريحته
    من الناتج بشكل مستقل تمامًا دون أي اتصال شبكي أثناء الحساب نفسه.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0, \
            "عدد المخرجات يجب أن يقبل القسمة على عدد GPUs"
        self.out_features_per_partition = out_features // world_size
        self.rank = rank
        self.world_size = world_size

        # كل GPU يُنشئ فقط شريحته الخاصة من مصفوفة الوزن الكاملة
        self.weight = nn.Parameter(
            torch.randn(in_features, self.out_features_per_partition) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x هنا كاملة (مُكرَّرة/مبثوثة على كل GPU)، والناتج شريحة فقط
        return x @ self.weight  # [batch, seq, out_features_per_partition]

class RowParallelLinear(nn.Module):
    """
    طبقة خطية مُقسَّمة أفقيًا. المدخل هنا هو شريحة (ناتج من طبقة
    ColumnParallel سابقة)، والناتج يتطلب عملية All-Reduce لجمع
    المساهمات من كل الـ GPUs قبل الحصول على الناتج الكامل الصحيح.
    """
    def __init__(self, in_features: int, out_features: int,
                 world_size: int, rank: int, process_group=None):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group

        self.weight = nn.Parameter(
            torch.randn(self.in_features_per_partition, out_features) * 0.02
        )

    def forward(self, x_partition: torch.Tensor) -> torch.Tensor:
        partial_output = x_partition @ self.weight

        # All-Reduce: جمع المساهمات الجزئية من كل الـ GPUs
        # هذه هي نقطة الاتصال الشبكي الوحيدة في مسار TP لهذه الطبقة
        if dist.is_initialized():
            dist.all_reduce(partial_output, op=dist.ReduceOp.SUM,
                             group=self.process_group)
        return partial_output

class TensorParallelFFN(nn.Module):
    """
    طبقة Feed-Forward كاملة مُقسَّمة بتوازي المُوتِرات بنمط Megatron-LM:
    - الطبقة الأولى: Column-Parallel (بدون اتصال شبكي)
    - التفعيل: يُطبَّق محليًا على كل شريحة
    - الطبقة الثانية: Row-Parallel (اتصال All-Reduce واحد فقط في النهاية)

    النتيجة: اتصال شبكي واحد فقط لكل طبقة FFN كاملة، بدلاً من
    اتصال بعد كل عملية ضرب مصفوفات منفردة.
    """
    def __init__(self, d_model: int, d_ff: int, world_size: int, rank: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.fc2 = RowParallelLinear(d_ff, d_model, world_size, rank)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(x))  # بدون اتصال شبكي
        output = self.fc2(hidden)               # اتصال All-Reduce واحد هنا
        return output

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class PipelineStage(nn.Module):
    """
    مرحلة واحدة من Pipeline Parallelism: تحتوي على مجموعة فرعية
    من طبقات النموذج الكلي، وتعمل كجزء من خط أنابيب متعدد المراحل.
    """
    def __init__(self, layers: nn.ModuleList, stage_id: int,
                 is_first_stage: bool, is_last_stage: bool):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def simulate_pipeline_schedule(num_stages: int, num_microbatches: int):
    """
    محاكاة جدول زمني 1F1B (One-Forward-One-Backward) لتوضيح كيف
    تتداخل عمليات Forward وBackward عبر المراحل لتقليل فقاعات التوقف،
    مقارنة بالنهج الساذج (معالجة كل الـ Forward ثم كل الـ Backward).
    """
    print(f"محاكاة Pipeline بـ {num_stages} مراحل و{num_microbatches} micro-batches")
    print("=" * 70)

    # الجدول الزمني الساذج (GPipe الأساسي بدون تداخل):
    # فقاعة = (num_stages - 1) خطوة زمنية ضائعة في البداية والنهاية
    naive_total_time = num_microbatches + 2 * (num_stages - 1)
    useful_time = num_microbatches
    naive_efficiency = useful_time / naive_total_time

    print(f"\nالجدول الساذج (Fill-Drain):")
    print(f"   الزمن الكلي: {naive_total_time} وحدة زمنية")
    print(f"   الزمن المُفيد: {useful_time} وحدة")
    print(f"   الكفاءة: {naive_efficiency*100:.1f}%")

    # بزيادة عدد الـ micro-batches، تتحسن الكفاءة لأن الفقاعة الثابتة
    # (2×(num_stages-1)) تصبح نسبة أصغر من الزمن الكلي
    print(f"\nتأثير زيادة عدد micro-batches على الكفاءة (مع ثبات num_stages={num_stages}):")
    for mb in [num_microbatches, num_microbatches*2, num_microbatches*4, num_microbatches*8]:
        total = mb + 2*(num_stages-1)
        eff = mb / total
        print(f"   {mb} micro-batches → كفاءة {eff*100:.1f}%")

simulate_pipeline_schedule(num_stages=4, num_microbatches=8)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def calculate_zero_memory_savings(num_params: float, num_gpus: int,
                                    optimizer_type: str = "adamw",
                                    precision_bytes: int = 2):
    """
    حساب توفير الذاكرة لكل مرحلة من ZeRO مقارنة بـ Data Parallelism
    التقليدي (بدون أي تقسيم)، بافتراض تدريب بدقة مختلطة (Mixed Precision).

    الافتراضات القياسية لـ Mixed Precision Training:
    - الأوزان: نسخة FP16/BF16 (2 بايت) + نسخة FP32 أصلية (4 بايت) = 6 بايت/معامل
    - التدرجات: FP16/BF16 (2 بايت/معامل)
    - حالات AdamW: متوسطين (m, v) بدقة FP32 = 8 بايت/معامل
    """
    # الذاكرة الإجمالية المطلوبة لكل "نوع" من البيانات (بالبايت لكل معامل)
    weights_memory_per_param = 2 + 4  # BF16 عمل + FP32 أصلية (master weights)
    gradients_memory_per_param = 2    # BF16
    optimizer_memory_per_param = 8    # AdamW: m + v بدقة FP32 (4+4)

    total_per_gpu_no_zero = (weights_memory_per_param +
                               gradients_memory_per_param +
                               optimizer_memory_per_param) * num_params

    scenarios = {
        "Data Parallelism (بدون ZeRO) - كل GPU يُكرِّر كل شيء": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param,
        },
        "ZeRO-1 (تقسيم Optimizer States فقط)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-2 (+ تقسيم Gradients)": {
            "weights": weights_memory_per_param,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
        "ZeRO-3 (+ تقسيم Parameters نفسها)": {
            "weights": weights_memory_per_param / num_gpus,
            "gradients": gradients_memory_per_param / num_gpus,
            "optimizer": optimizer_memory_per_param / num_gpus,
        },
    }

    print(f"نموذج بحجم {num_params/1e9:.0f} مليار معامل، موزَّع على {num_gpus} GPU")
    print("=" * 75)
    for name, breakdown in scenarios.items():
        total_bytes_per_param = sum(breakdown.values())
        total_gb = (total_bytes_per_param * num_params) / 1e9
        print(f"\n{name}:")
        print(f"   الذاكرة المطلوبة لكل GPU: {total_gb:.1f} GB")

calculate_zero_memory_savings(num_params=70e9, num_gpus=64)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

def setup_fsdp_model(model: nn.Module, transformer_layer_class) -> FSDP:
    """
    تغليف نموذج Transformer بـ FSDP لتفعيل تقسيم الأوزان/التدرجات/
    حالات المُحسِّن بنمط ZeRO-3 تلقائيًا عبر كل الـ GPUs المشاركة.
    """
    # سياسة التغليف التلقائي: نُغلِّف كل طبقة Transformer بشكل منفصل
    # (بدلاً من تغليف النموذج كاملاً كوحدة واحدة)، مما يسمح بتداخل
    # الاتصال الشبكي (All-Gather/Reduce-Scatter) مع الحوسبة بكفاءة أعلى
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )

    # إعدادات الدقة المختلطة: BF16 للحوسبة والاتصال، FP32 للتجميع النهائي
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # هذا يعني ZeRO-3 الكامل
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # تداخل الاتصال مع الحوسبة
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,  # تحكم في استهلاك الذاكرة أثناء All-Gather
    )

    return fsdp_model

def compare_sharding_strategies():
    """
    مقارنة استراتيجيات التقسيم المتاحة في FSDP، كل منها يناظر
    مرحلة مختلفة من ZeRO بالضبط:
    """
    strategies = {
        "FULL_SHARD": "يناظر ZeRO-3 بالكامل: تقسيم Params+Gradients+Optimizer States",
        "SHARD_GRAD_OP": "يناظر ZeRO-2: تقسيم Gradients+Optimizer States فقط، الأوزان مُكرَّرة",
        "NO_SHARD": "يناظر DDP التقليدي: لا تقسيم إطلاقًا (كل شيء مُكرَّر)",
        "HYBRID_SHARD": "FULL_SHARD داخل كل عقدة + تكرار بين العقد (توازن اتصال/ذاكرة)",
    }
    for strategy, description in strategies.items():
        print(f"{strategy}: {description}")

compare_sharding_strategies()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def calculate_3d_parallelism_configuration(total_gpus: int,
                                             tensor_parallel_size: int,
                                             pipeline_parallel_size: int) -> dict:
    """
    حساب حجم Data Parallelism الناتج ضمنيًا من اختيار أحجام TP وPP،
    بما أن: total_gpus = TP_size × PP_size × DP_size
    """
    assert total_gpus % (tensor_parallel_size * pipeline_parallel_size) == 0, \
        "إجمالي الـ GPUs يجب أن يقبل القسمة على حاصل ضرب TP×PP"

    data_parallel_size = total_gpus // (tensor_parallel_size * pipeline_parallel_size)

    return {
        "total_gpus": total_gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "data_parallel_size": data_parallel_size,
        "effective_batch_multiplier": data_parallel_size,
    }

config = calculate_3d_parallelism_configuration(
    total_gpus=1024, tensor_parallel_size=8, pipeline_parallel_size=16
)
for k, v in config.items():
    print(f"{k}: {v}")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def solve_memory_feasibility_example():
    num_params = 13e9
    num_gpus = 32
    gpu_memory_gb = 80

    bytes_per_param_total = 6 + 2 + 8  # 16 بايت/معامل

    memory_no_zero_gb = (num_params * bytes_per_param_total) / 1e9
    memory_zero3_gb = memory_no_zero_gb / num_gpus

    print(f"الذاكرة بدون ZeRO (لكل GPU): {memory_no_zero_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_no_zero_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"الذاكرة مع ZeRO-3 (لكل GPU): {memory_zero3_gb:.1f} GB "
          f"({'❌ مستحيل' if memory_zero3_gb > gpu_memory_gb else '✅ ممكن'} "
          f"على GPU بـ {gpu_memory_gb}GB)")
    print(f"\nالمساحة المتبقية لكل GPU بعد ZeRO-3: "
          f"{gpu_memory_gb - memory_zero3_gb:.1f} GB (للتنشيطات وBatch أكبر)")

solve_memory_feasibility_example()

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-llms-transformers/notebooks/ch-21-ch21-distributed-training.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
