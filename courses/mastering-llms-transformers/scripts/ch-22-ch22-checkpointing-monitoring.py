#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حفظ نقاط التفتيش ومراقبة التدريب
Mastering Transformers and Large Language Models — Checkpointing Monitoring

Book slug : mastering-llms-transformers
Chapter   : ch-22-ch22-checkpointing-monitoring (lesson 22)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-22-ch22-checkpointing-monitoring.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حفظ نقاط التفتيش ومراقبة التدريب
Mastering Transformers and Large Language Models — Checkpointing Monitoring

Book slug : mastering-llms-transformers
Chapter   : ch-22-ch22-checkpointing-monitoring (lesson 22)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-22-ch22-checkpointing-monitoring.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حفظ نقاط التفتيش ومراقبة التدريب
Mastering Transformers and Large Language Models — Checkpointing Monitoring

Book slug : mastering-llms-transformers
Chapter   : ch-22-ch22-checkpointing-monitoring (lesson 22)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-1
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
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("wandb", "wandb")
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
# [002] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import os
import json
import random
import numpy as np
from datetime import datetime

class CheckpointManager:
    """
    مدير Checkpoints كامل للتدريب طويل الأمد، يضمن إمكانية استئناف
    التدريب من أي نقطة توقف بشكل مطابق تمامًا (Deterministic Resume)
    لما كان سيحدث لو لم يتوقف التدريب إطلاقًا.
    """
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         lr_scheduler,
                         step: int,
                         epoch: int,
                         loss: float,
                         extra_metadata: dict = None):
        """
        حفظ Checkpoint كامل يحتوي كل ما يلزم لاستئناف التدريب
        بشكل مطابق تمامًا من حيث الحالة الحسابية.
        """
        checkpoint = {
            # 1. أوزان النموذج - العنصر الأوضح لكن غير الكافي وحده
            "model_state_dict": model.state_dict(),

            # 2. حالات المُحسِّن - حرجة جدًا! بدونها، Adam/AdamW يفقد
            # متوسطاته المتحركة (m, v) المتراكمة ويبدأ التحسين "من الصفر"
            # فعليًا حتى لو كانت الأوزان محفوظة، مما يسبب اضطرابًا كبيرًا
            # في منحنى الخسارة عند الاستئناف
            "optimizer_state_dict": optimizer.state_dict(),

            # 3. حالة جدول معدل التعلم (مهم لمعرفة أين نحن في Warmup/Decay)
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,

            # 4. عداد الخطوات/العصور - لاستئناف جدول LR وDataLoader بشكل صحيح
            "step": step,
            "epoch": epoch,

            # 5. قيمة الخسارة عند هذه النقطة - للمقارنة والتحقق من السلامة
            "loss": loss,

            # 6. حالة مولدات الأرقام العشوائية - لضمان قابلية إعادة الإنتاج
            # الكاملة (نفس تسلسل Dropout/أي عشوائية أخرى عند الاستئناف)
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },

            # 7. بيانات وصفية إضافية (تاريخ الحفظ، إصدار الكود، إعدادات التدريب)
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                **(extra_metadata or {}),
            },
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{step}.pt"
        )
        # الحفظ بشكل ذري (atomic): نكتب لملف مؤقت أولاً ثم نُعيد تسميته،
        # لتفادي ملف checkpoint تالف إذا انقطع التيار أثناء الكتابة
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, checkpoint_path)

        print(f"✅ تم حفظ Checkpoint عند الخطوة {step} في {checkpoint_path}")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """الاحتفاظ فقط بآخر N نسخة لتوفير مساحة التخزين (checkpoints ضخمة جدًا)"""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ], key=lambda x: int(x.split("_")[-1].replace(".pt", "")))

        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))
            print(f"🗑️  حُذف Checkpoint قديم: {oldest}")

    def load_checkpoint(self, checkpoint_path: str, model, optimizer, lr_scheduler):
        """استعادة كاملة لحالة التدريب من Checkpoint محفوظ"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler and checkpoint["lr_scheduler_state_dict"]:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # استعادة حالة العشوائية لضمان الاستمرارية المطابقة
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])

        print(f"✅ تم استعادة Checkpoint من الخطوة {checkpoint['step']}، "
              f"الخسارة عند الحفظ: {checkpoint['loss']:.4f}")

        return checkpoint["step"], checkpoint["epoch"]

    def find_latest_checkpoint(self) -> str:
        """إيجاد أحدث Checkpoint متاح تلقائيًا (لاستئناف تلقائي بعد انقطاع)"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ]
        if not checkpoints:
            return None
        latest = max(checkpoints,
                      key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        return os.path.join(self.checkpoint_dir, latest)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_fsdp_checkpoint_consolidated(fsdp_model: FSDP, save_path: str, rank: int):
    """
    حفظ Checkpoint مُجمَّع (Consolidated) من نموذج FSDP: يُجمِّع كل الشرائح
    المُوزَّعة في state_dict واحد كامل على rank=0 فقط، مناسب للتوزيع
    اللاحق (نشر النموذج، أو تحميله لاحقًا على عدد GPUs مختلف).

    العيب: يتطلب ذاكرة كافية على rank=0 لاستيعاب النموذج الكامل لحظيًا.
    """
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,  # نقل الحالة المُجمَّعة لذاكرة CPU لتفادي OOM على GPU
        rank0_only=True,       # فقط rank=0 يحتفظ بالنسخة الكاملة النهائية
    )

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        full_state_dict = fsdp_model.state_dict()

    if rank == 0:
        torch.save(full_state_dict, save_path)
        print(f"✅ تم حفظ Checkpoint مُجمَّع كامل في {save_path}")

def save_fsdp_checkpoint_sharded(fsdp_model: FSDP, save_dir: str):
    """
    حفظ Checkpoint مُجزَّأ (Sharded): كل GPU يحفظ شريحته الخاصة فقط
    في ملف منفصل. أسرع بكثير وأقل استهلاكًا للذاكرة من النهج المُجمَّع،
    وهو الخيار المُفضَّل للـ checkpoints الدورية أثناء التدريب (الاستئناف
    يكون غالبًا على نفس عدد GPUs بالضبط).
    """
    from torch.distributed.fsdp import ShardedStateDictConfig

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True)
    ):
        sharded_state_dict = fsdp_model.state_dict()
        rank = dist.get_rank()
        torch.save(sharded_state_dict,
                   os.path.join(save_dir, f"shard_rank_{rank}.pt"))

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from collections import deque

class TrainingMonitor:
    """
    نظام مراقبة شامل للتدريب يتتبع المقاييس الحيوية ويكتشف المشاكل
    مبكرًا قبل أن تُهدر ساعات/أيام من الحوسبة على تدريب فاسد.
    """
    def __init__(self, window_size: int = 100,
                 loss_spike_threshold: float = 3.0,
                 grad_norm_explosion_threshold: float = 100.0):
        self.window_size = window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_explosion_threshold = grad_norm_explosion_threshold

        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.alerts_log = []

    def compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """
        حساب القاعدة الإقليدية الكلية (L2 norm) لكل تدرجات النموذج،
        وهو مقياس أساسي لاستقرار التدريب. قيم مرتفعة جدًا فجأة تشير
        لعدم استقرار رقمي ("Gradient Explosion").
        """
        total_norm_squared = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_squared += param_norm.item() ** 2
        return total_norm_squared ** 0.5

    def log_step(self, step: int, loss: float, model: torch.nn.Module,
                 learning_rate: float, step_duration: float) -> dict:
        """
        تسجيل مقاييس خطوة تدريب واحدة، مع كشف تلقائي عن الحالات الشاذة.
        يُعيد قاموسًا بأي تنبيهات اكتُشفت (فارغ إذا كل شيء طبيعي).
        """
        grad_norm = self.compute_gradient_norm(model)
        alerts = {}

        # 1. كشف Loss Spike: الخسارة الحالية أعلى بكثير من المتوسط المتحرك الأخير
        if len(self.loss_history) >= 10:
            recent_mean = sum(list(self.loss_history)[-10:]) / 10
            recent_std = (sum((x - recent_mean)**2
                          for x in list(self.loss_history)[-10:]) / 10) ** 0.5
            if recent_std > 0 and loss > recent_mean + self.loss_spike_threshold * recent_std:
                alerts["loss_spike"] = (
                    f"⚠️ Loss Spike عند الخطوة {step}: "
                    f"{loss:.4f} (المتوسط الأخير: {recent_mean:.4f} ± {recent_std:.4f})"
                )

        # 2. كشف Gradient Explosion
        if grad_norm > self.grad_norm_explosion_threshold:
            alerts["grad_explosion"] = (
                f"⚠️ Gradient Norm مرتفع جدًا عند الخطوة {step}: {grad_norm:.2f}"
            )

        # 3. كشف NaN/Inf (الأخطر على الإطلاق - يُفسد التدريب فورًا)
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            alerts["nan_inf_loss"] = (
                f"🔴 خطأ حرج: الخسارة NaN/Inf عند الخطوة {step}! "
                f"يجب إيقاف التدريب فورًا والاستعادة من آخر checkpoint سليم"
            )

        # تحديث السجلات
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.step_times.append(step_duration)

        if alerts:
            self.alerts_log.extend(alerts.values())
            for msg in alerts.values():
                print(msg)

        return {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "step_duration_sec": step_duration,
            "tokens_per_sec": None,  # يُحسَب خارجيًا حسب batch_size×seq_len
            "alerts": alerts,
        }

    def get_training_speed_stats(self) -> dict:
        """إحصائيات سرعة التدريب لتقدير الوقت المتبقي والتكلفة"""
        if not self.step_times:
            return {}
        avg_step_time = sum(self.step_times) / len(self.step_times)
        return {
            "avg_step_time_sec": avg_step_time,
            "steps_per_hour": 3600 / avg_step_time if avg_step_time > 0 else 0,
        }

def estimate_remaining_training_time(monitor: TrainingMonitor,
                                       current_step: int,
                                       total_steps: int) -> str:
    """تقدير الوقت المتبقي للتدريب بناء على متوسط سرعة الخطوات الأخيرة"""
    stats = monitor.get_training_speed_stats()
    if not stats:
        return "غير متاح بعد (بيانات غير كافية)"

    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps * stats["avg_step_time_sec"]
    remaining_hours = remaining_seconds / 3600
    remaining_days = remaining_hours / 24

    return (f"الوقت المتبقي التقديري: {remaining_hours:.1f} ساعة "
            f"({remaining_days:.1f} يوم) عند السرعة الحالية "
            f"({stats['steps_per_hour']:.0f} خطوة/ساعة)")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import wandb

def setup_wandb_tracking(project_name: str, run_config: dict):
    """
    إعداد تتبع تجريبي كامل بـ Weights & Biases، الأداة المعيارية
    صناعيًا لتتبع تجارب تدريب LLMs (تستخدمها معظم المختبرات البحثية
    الكبرى: OpenAI, Meta AI, DeepMind وغيرها).
    """
    wandb.init(
        project=project_name,
        config=run_config,
        # متابعة استهلاك GPU/CPU/الذاكرة تلقائيًا
        settings=wandb.Settings(_disable_stats=False),
    )

def log_training_step_to_wandb(step: int, metrics: dict):
    """تسجيل مقاييس خطوة واحدة في لوحة wandb التفاعلية"""
    wandb.log({
        "train/loss": metrics["loss"],
        "train/grad_norm": metrics["grad_norm"],
        "train/learning_rate": metrics["learning_rate"],
        "train/step_duration_sec": metrics["step_duration_sec"],
        "system/gpu_memory_allocated_gb":
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "system/gpu_memory_reserved_gb":
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
    }, step=step)

def log_evaluation_metrics_to_wandb(step: int, eval_metrics: dict):
    """تسجيل مقاييس التقييم الدورية (كل N خطوة على مجموعة تحقق محجوزة)"""
    wandb.log({
        "eval/loss": eval_metrics["eval_loss"],
        "eval/perplexity": eval_metrics["perplexity"],
    }, step=step)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def estimate_training_cost(num_params: float, num_tokens: float,
                             gpu_type: str = "H100",
                             mfu: float = 0.4) -> dict:
    """
    تقدير تكلفة تدريب نموذج بالدولار، بناء على:
    - ميزانية الحوسبة المطلوبة C ≈ 6ND
    - معدل الأداء الفعلي للـ GPU (مع اعتبار MFU - نسبة الاستغلال الفعلي)
    - التكلفة التقريبية لساعة GPU واحدة (أسعار سوقية تقريبية، تتغير بمرور الوقت)
    """
    gpu_specs = {
        "H100": {"peak_flops_bf16": 989e12, "cost_per_hour_usd": 2.5},  # أسعار Cloud تقريبية
        "A100": {"peak_flops_bf16": 312e12, "cost_per_hour_usd": 1.5},
        "V100": {"peak_flops_bf16": 125e12, "cost_per_hour_usd": 0.9},
    }

    specs = gpu_specs[gpu_type]
    total_flops_required = 6 * num_params * num_tokens

    # الأداء الفعلي = الأداء النظري الأقصى × نسبة الاستغلال الفعلي (MFU)
    # MFU النموذجي في تدريب LLM حقيقي يتراوح 30%-55% عادة (نادرًا يتجاوز 60%)
    effective_flops_per_sec = specs["peak_flops_bf16"] * mfu

    total_gpu_seconds = total_flops_required / effective_flops_per_sec
    total_gpu_hours = total_gpu_seconds / 3600
    total_cost_usd = total_gpu_hours * specs["cost_per_hour_usd"]

    return {
        "total_flops": total_flops_required,
        "gpu_type": gpu_type,
        "mfu_assumed": mfu,
        "total_gpu_hours": total_gpu_hours,
        "estimated_cost_usd": total_cost_usd,
    }

def compare_training_cost_scenarios():
    """مقارنة تكلفة تدريب عدة أحجام نماذج واقعية على H100"""
    scenarios = [
        {"name": "نموذج صغير توضيحي", "N": 125e6, "D": 2.5e9},
        {"name": "نموذج متوسط (مشروع هذا الجزء)", "N": 150e6, "D": 3e9},
        {"name": "LLaMA-2 7B (تقريبي)", "N": 7e9, "D": 2e12},
        {"name": "LLaMA-2 70B (تقريبي)", "N": 70e9, "D": 2e12},
    ]

    print(f"{'النموذج':<35}{'GPU-hours':<15}{'التكلفة التقديرية (USD)':<25}")
    print("-" * 75)
    for s in scenarios:
        result = estimate_training_cost(s["N"], s["D"], gpu_type="H100", mfu=0.4)
        print(f"{s['name']:<35}{result['total_gpu_hours']:<15,.0f}"
              f"${result['estimated_cost_usd']:<24,.0f}")

compare_training_cost_scenarios()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def simulate_loss_spike_detection_and_recovery():
    """محاكاة سيناريو واقعي لاكتشاف Loss Spike والتعافي منه"""
    monitor = TrainingMonitor(loss_spike_threshold=3.0)
    checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")

    # محاكاة تاريخ خسارة طبيعي يتذبذب حول 2.3 بانحراف معياري صغير
    import random
    random.seed(42)
    simulated_losses = [2.3 + random.gauss(0, 0.05) for _ in range(20)]
    simulated_losses.append(8.7)  # ← القفزة المفاجئة

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)
            self.fc.weight.grad = torch.randn(10, 10) * 50  # تدرج كبير محاكاة للانفجار

    dummy_model = DummyModel()

    for step, loss in enumerate(simulated_losses):
        result = monitor.log_step(
            step=step, loss=loss, model=dummy_model,
            learning_rate=3e-4, step_duration=1.2
        )
        if result["alerts"]:
            print(f"\n🚨 عند الخطوة {step}: اكتُشفت مشكلة!")
            print("الإجراء المُوصى به:")
            print("   1. إيقاف التدريب الحالي فورًا")
            print("   2. العودة لآخر checkpoint سليم (قبل بداية القفزة)")
            print("   3. فحص batch البيانات التي سبقت القفزة (غالبًا السبب)")
            print("   4. إن تكررت المشكلة: تقليل معدل التعلم أو تشديد Gradient Clipping")
            break

simulate_loss_spike_detection_and_recovery()

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-1
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
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("wandb", "wandb")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import os
import json
import random
import numpy as np
from datetime import datetime

class CheckpointManager:
    """
    مدير Checkpoints كامل للتدريب طويل الأمد، يضمن إمكانية استئناف
    التدريب من أي نقطة توقف بشكل مطابق تمامًا (Deterministic Resume)
    لما كان سيحدث لو لم يتوقف التدريب إطلاقًا.
    """
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         lr_scheduler,
                         step: int,
                         epoch: int,
                         loss: float,
                         extra_metadata: dict = None):
        """
        حفظ Checkpoint كامل يحتوي كل ما يلزم لاستئناف التدريب
        بشكل مطابق تمامًا من حيث الحالة الحسابية.
        """
        checkpoint = {
            # 1. أوزان النموذج - العنصر الأوضح لكن غير الكافي وحده
            "model_state_dict": model.state_dict(),

            # 2. حالات المُحسِّن - حرجة جدًا! بدونها، Adam/AdamW يفقد
            # متوسطاته المتحركة (m, v) المتراكمة ويبدأ التحسين "من الصفر"
            # فعليًا حتى لو كانت الأوزان محفوظة، مما يسبب اضطرابًا كبيرًا
            # في منحنى الخسارة عند الاستئناف
            "optimizer_state_dict": optimizer.state_dict(),

            # 3. حالة جدول معدل التعلم (مهم لمعرفة أين نحن في Warmup/Decay)
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,

            # 4. عداد الخطوات/العصور - لاستئناف جدول LR وDataLoader بشكل صحيح
            "step": step,
            "epoch": epoch,

            # 5. قيمة الخسارة عند هذه النقطة - للمقارنة والتحقق من السلامة
            "loss": loss,

            # 6. حالة مولدات الأرقام العشوائية - لضمان قابلية إعادة الإنتاج
            # الكاملة (نفس تسلسل Dropout/أي عشوائية أخرى عند الاستئناف)
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },

            # 7. بيانات وصفية إضافية (تاريخ الحفظ، إصدار الكود، إعدادات التدريب)
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                **(extra_metadata or {}),
            },
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{step}.pt"
        )
        # الحفظ بشكل ذري (atomic): نكتب لملف مؤقت أولاً ثم نُعيد تسميته،
        # لتفادي ملف checkpoint تالف إذا انقطع التيار أثناء الكتابة
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, checkpoint_path)

        print(f"✅ تم حفظ Checkpoint عند الخطوة {step} في {checkpoint_path}")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """الاحتفاظ فقط بآخر N نسخة لتوفير مساحة التخزين (checkpoints ضخمة جدًا)"""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ], key=lambda x: int(x.split("_")[-1].replace(".pt", "")))

        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))
            print(f"🗑️  حُذف Checkpoint قديم: {oldest}")

    def load_checkpoint(self, checkpoint_path: str, model, optimizer, lr_scheduler):
        """استعادة كاملة لحالة التدريب من Checkpoint محفوظ"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler and checkpoint["lr_scheduler_state_dict"]:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # استعادة حالة العشوائية لضمان الاستمرارية المطابقة
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])

        print(f"✅ تم استعادة Checkpoint من الخطوة {checkpoint['step']}، "
              f"الخسارة عند الحفظ: {checkpoint['loss']:.4f}")

        return checkpoint["step"], checkpoint["epoch"]

    def find_latest_checkpoint(self) -> str:
        """إيجاد أحدث Checkpoint متاح تلقائيًا (لاستئناف تلقائي بعد انقطاع)"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ]
        if not checkpoints:
            return None
        latest = max(checkpoints,
                      key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        return os.path.join(self.checkpoint_dir, latest)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_fsdp_checkpoint_consolidated(fsdp_model: FSDP, save_path: str, rank: int):
    """
    حفظ Checkpoint مُجمَّع (Consolidated) من نموذج FSDP: يُجمِّع كل الشرائح
    المُوزَّعة في state_dict واحد كامل على rank=0 فقط، مناسب للتوزيع
    اللاحق (نشر النموذج، أو تحميله لاحقًا على عدد GPUs مختلف).

    العيب: يتطلب ذاكرة كافية على rank=0 لاستيعاب النموذج الكامل لحظيًا.
    """
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,  # نقل الحالة المُجمَّعة لذاكرة CPU لتفادي OOM على GPU
        rank0_only=True,       # فقط rank=0 يحتفظ بالنسخة الكاملة النهائية
    )

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        full_state_dict = fsdp_model.state_dict()

    if rank == 0:
        torch.save(full_state_dict, save_path)
        print(f"✅ تم حفظ Checkpoint مُجمَّع كامل في {save_path}")

def save_fsdp_checkpoint_sharded(fsdp_model: FSDP, save_dir: str):
    """
    حفظ Checkpoint مُجزَّأ (Sharded): كل GPU يحفظ شريحته الخاصة فقط
    في ملف منفصل. أسرع بكثير وأقل استهلاكًا للذاكرة من النهج المُجمَّع،
    وهو الخيار المُفضَّل للـ checkpoints الدورية أثناء التدريب (الاستئناف
    يكون غالبًا على نفس عدد GPUs بالضبط).
    """
    from torch.distributed.fsdp import ShardedStateDictConfig

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True)
    ):
        sharded_state_dict = fsdp_model.state_dict()
        rank = dist.get_rank()
        torch.save(sharded_state_dict,
                   os.path.join(save_dir, f"shard_rank_{rank}.pt"))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from collections import deque

class TrainingMonitor:
    """
    نظام مراقبة شامل للتدريب يتتبع المقاييس الحيوية ويكتشف المشاكل
    مبكرًا قبل أن تُهدر ساعات/أيام من الحوسبة على تدريب فاسد.
    """
    def __init__(self, window_size: int = 100,
                 loss_spike_threshold: float = 3.0,
                 grad_norm_explosion_threshold: float = 100.0):
        self.window_size = window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_explosion_threshold = grad_norm_explosion_threshold

        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.alerts_log = []

    def compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """
        حساب القاعدة الإقليدية الكلية (L2 norm) لكل تدرجات النموذج،
        وهو مقياس أساسي لاستقرار التدريب. قيم مرتفعة جدًا فجأة تشير
        لعدم استقرار رقمي ("Gradient Explosion").
        """
        total_norm_squared = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_squared += param_norm.item() ** 2
        return total_norm_squared ** 0.5

    def log_step(self, step: int, loss: float, model: torch.nn.Module,
                 learning_rate: float, step_duration: float) -> dict:
        """
        تسجيل مقاييس خطوة تدريب واحدة، مع كشف تلقائي عن الحالات الشاذة.
        يُعيد قاموسًا بأي تنبيهات اكتُشفت (فارغ إذا كل شيء طبيعي).
        """
        grad_norm = self.compute_gradient_norm(model)
        alerts = {}

        # 1. كشف Loss Spike: الخسارة الحالية أعلى بكثير من المتوسط المتحرك الأخير
        if len(self.loss_history) >= 10:
            recent_mean = sum(list(self.loss_history)[-10:]) / 10
            recent_std = (sum((x - recent_mean)**2
                          for x in list(self.loss_history)[-10:]) / 10) ** 0.5
            if recent_std > 0 and loss > recent_mean + self.loss_spike_threshold * recent_std:
                alerts["loss_spike"] = (
                    f"⚠️ Loss Spike عند الخطوة {step}: "
                    f"{loss:.4f} (المتوسط الأخير: {recent_mean:.4f} ± {recent_std:.4f})"
                )

        # 2. كشف Gradient Explosion
        if grad_norm > self.grad_norm_explosion_threshold:
            alerts["grad_explosion"] = (
                f"⚠️ Gradient Norm مرتفع جدًا عند الخطوة {step}: {grad_norm:.2f}"
            )

        # 3. كشف NaN/Inf (الأخطر على الإطلاق - يُفسد التدريب فورًا)
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            alerts["nan_inf_loss"] = (
                f"🔴 خطأ حرج: الخسارة NaN/Inf عند الخطوة {step}! "
                f"يجب إيقاف التدريب فورًا والاستعادة من آخر checkpoint سليم"
            )

        # تحديث السجلات
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.step_times.append(step_duration)

        if alerts:
            self.alerts_log.extend(alerts.values())
            for msg in alerts.values():
                print(msg)

        return {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "step_duration_sec": step_duration,
            "tokens_per_sec": None,  # يُحسَب خارجيًا حسب batch_size×seq_len
            "alerts": alerts,
        }

    def get_training_speed_stats(self) -> dict:
        """إحصائيات سرعة التدريب لتقدير الوقت المتبقي والتكلفة"""
        if not self.step_times:
            return {}
        avg_step_time = sum(self.step_times) / len(self.step_times)
        return {
            "avg_step_time_sec": avg_step_time,
            "steps_per_hour": 3600 / avg_step_time if avg_step_time > 0 else 0,
        }

def estimate_remaining_training_time(monitor: TrainingMonitor,
                                       current_step: int,
                                       total_steps: int) -> str:
    """تقدير الوقت المتبقي للتدريب بناء على متوسط سرعة الخطوات الأخيرة"""
    stats = monitor.get_training_speed_stats()
    if not stats:
        return "غير متاح بعد (بيانات غير كافية)"

    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps * stats["avg_step_time_sec"]
    remaining_hours = remaining_seconds / 3600
    remaining_days = remaining_hours / 24

    return (f"الوقت المتبقي التقديري: {remaining_hours:.1f} ساعة "
            f"({remaining_days:.1f} يوم) عند السرعة الحالية "
            f"({stats['steps_per_hour']:.0f} خطوة/ساعة)")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import wandb

def setup_wandb_tracking(project_name: str, run_config: dict):
    """
    إعداد تتبع تجريبي كامل بـ Weights & Biases، الأداة المعيارية
    صناعيًا لتتبع تجارب تدريب LLMs (تستخدمها معظم المختبرات البحثية
    الكبرى: OpenAI, Meta AI, DeepMind وغيرها).
    """
    wandb.init(
        project=project_name,
        config=run_config,
        # متابعة استهلاك GPU/CPU/الذاكرة تلقائيًا
        settings=wandb.Settings(_disable_stats=False),
    )

def log_training_step_to_wandb(step: int, metrics: dict):
    """تسجيل مقاييس خطوة واحدة في لوحة wandb التفاعلية"""
    wandb.log({
        "train/loss": metrics["loss"],
        "train/grad_norm": metrics["grad_norm"],
        "train/learning_rate": metrics["learning_rate"],
        "train/step_duration_sec": metrics["step_duration_sec"],
        "system/gpu_memory_allocated_gb":
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "system/gpu_memory_reserved_gb":
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
    }, step=step)

def log_evaluation_metrics_to_wandb(step: int, eval_metrics: dict):
    """تسجيل مقاييس التقييم الدورية (كل N خطوة على مجموعة تحقق محجوزة)"""
    wandb.log({
        "eval/loss": eval_metrics["eval_loss"],
        "eval/perplexity": eval_metrics["perplexity"],
    }, step=step)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def estimate_training_cost(num_params: float, num_tokens: float,
                             gpu_type: str = "H100",
                             mfu: float = 0.4) -> dict:
    """
    تقدير تكلفة تدريب نموذج بالدولار، بناء على:
    - ميزانية الحوسبة المطلوبة C ≈ 6ND
    - معدل الأداء الفعلي للـ GPU (مع اعتبار MFU - نسبة الاستغلال الفعلي)
    - التكلفة التقريبية لساعة GPU واحدة (أسعار سوقية تقريبية، تتغير بمرور الوقت)
    """
    gpu_specs = {
        "H100": {"peak_flops_bf16": 989e12, "cost_per_hour_usd": 2.5},  # أسعار Cloud تقريبية
        "A100": {"peak_flops_bf16": 312e12, "cost_per_hour_usd": 1.5},
        "V100": {"peak_flops_bf16": 125e12, "cost_per_hour_usd": 0.9},
    }

    specs = gpu_specs[gpu_type]
    total_flops_required = 6 * num_params * num_tokens

    # الأداء الفعلي = الأداء النظري الأقصى × نسبة الاستغلال الفعلي (MFU)
    # MFU النموذجي في تدريب LLM حقيقي يتراوح 30%-55% عادة (نادرًا يتجاوز 60%)
    effective_flops_per_sec = specs["peak_flops_bf16"] * mfu

    total_gpu_seconds = total_flops_required / effective_flops_per_sec
    total_gpu_hours = total_gpu_seconds / 3600
    total_cost_usd = total_gpu_hours * specs["cost_per_hour_usd"]

    return {
        "total_flops": total_flops_required,
        "gpu_type": gpu_type,
        "mfu_assumed": mfu,
        "total_gpu_hours": total_gpu_hours,
        "estimated_cost_usd": total_cost_usd,
    }

def compare_training_cost_scenarios():
    """مقارنة تكلفة تدريب عدة أحجام نماذج واقعية على H100"""
    scenarios = [
        {"name": "نموذج صغير توضيحي", "N": 125e6, "D": 2.5e9},
        {"name": "نموذج متوسط (مشروع هذا الجزء)", "N": 150e6, "D": 3e9},
        {"name": "LLaMA-2 7B (تقريبي)", "N": 7e9, "D": 2e12},
        {"name": "LLaMA-2 70B (تقريبي)", "N": 70e9, "D": 2e12},
    ]

    print(f"{'النموذج':<35}{'GPU-hours':<15}{'التكلفة التقديرية (USD)':<25}")
    print("-" * 75)
    for s in scenarios:
        result = estimate_training_cost(s["N"], s["D"], gpu_type="H100", mfu=0.4)
        print(f"{s['name']:<35}{result['total_gpu_hours']:<15,.0f}"
              f"${result['estimated_cost_usd']:<24,.0f}")

compare_training_cost_scenarios()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def simulate_loss_spike_detection_and_recovery():
    """محاكاة سيناريو واقعي لاكتشاف Loss Spike والتعافي منه"""
    monitor = TrainingMonitor(loss_spike_threshold=3.0)
    checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")

    # محاكاة تاريخ خسارة طبيعي يتذبذب حول 2.3 بانحراف معياري صغير
    import random
    random.seed(42)
    simulated_losses = [2.3 + random.gauss(0, 0.05) for _ in range(20)]
    simulated_losses.append(8.7)  # ← القفزة المفاجئة

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)
            self.fc.weight.grad = torch.randn(10, 10) * 50  # تدرج كبير محاكاة للانفجار

    dummy_model = DummyModel()

    for step, loss in enumerate(simulated_losses):
        result = monitor.log_step(
            step=step, loss=loss, model=dummy_model,
            learning_rate=3e-4, step_duration=1.2
        )
        if result["alerts"]:
            print(f"\n🚨 عند الخطوة {step}: اكتُشفت مشكلة!")
            print("الإجراء المُوصى به:")
            print("   1. إيقاف التدريب الحالي فورًا")
            print("   2. العودة لآخر checkpoint سليم (قبل بداية القفزة)")
            print("   3. فحص batch البيانات التي سبقت القفزة (غالبًا السبب)")
            print("   4. إن تكررت المشكلة: تقليل معدل التعلم أو تشديد Gradient Clipping")
            break

simulate_loss_spike_detection_and_recovery()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — حفظ نقاط التفتيش ومراقبة التدريب
Mastering Transformers and Large Language Models — Checkpointing Monitoring

Book slug : mastering-llms-transformers
Chapter   : ch-22-ch22-checkpointing-monitoring (lesson 22)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-1
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
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("wandb", "wandb")
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
# [002] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import os
import json
import random
import numpy as np
from datetime import datetime

class CheckpointManager:
    """
    مدير Checkpoints كامل للتدريب طويل الأمد، يضمن إمكانية استئناف
    التدريب من أي نقطة توقف بشكل مطابق تمامًا (Deterministic Resume)
    لما كان سيحدث لو لم يتوقف التدريب إطلاقًا.
    """
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         lr_scheduler,
                         step: int,
                         epoch: int,
                         loss: float,
                         extra_metadata: dict = None):
        """
        حفظ Checkpoint كامل يحتوي كل ما يلزم لاستئناف التدريب
        بشكل مطابق تمامًا من حيث الحالة الحسابية.
        """
        checkpoint = {
            # 1. أوزان النموذج - العنصر الأوضح لكن غير الكافي وحده
            "model_state_dict": model.state_dict(),

            # 2. حالات المُحسِّن - حرجة جدًا! بدونها، Adam/AdamW يفقد
            # متوسطاته المتحركة (m, v) المتراكمة ويبدأ التحسين "من الصفر"
            # فعليًا حتى لو كانت الأوزان محفوظة، مما يسبب اضطرابًا كبيرًا
            # في منحنى الخسارة عند الاستئناف
            "optimizer_state_dict": optimizer.state_dict(),

            # 3. حالة جدول معدل التعلم (مهم لمعرفة أين نحن في Warmup/Decay)
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,

            # 4. عداد الخطوات/العصور - لاستئناف جدول LR وDataLoader بشكل صحيح
            "step": step,
            "epoch": epoch,

            # 5. قيمة الخسارة عند هذه النقطة - للمقارنة والتحقق من السلامة
            "loss": loss,

            # 6. حالة مولدات الأرقام العشوائية - لضمان قابلية إعادة الإنتاج
            # الكاملة (نفس تسلسل Dropout/أي عشوائية أخرى عند الاستئناف)
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },

            # 7. بيانات وصفية إضافية (تاريخ الحفظ، إصدار الكود، إعدادات التدريب)
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                **(extra_metadata or {}),
            },
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{step}.pt"
        )
        # الحفظ بشكل ذري (atomic): نكتب لملف مؤقت أولاً ثم نُعيد تسميته،
        # لتفادي ملف checkpoint تالف إذا انقطع التيار أثناء الكتابة
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, checkpoint_path)

        print(f"✅ تم حفظ Checkpoint عند الخطوة {step} في {checkpoint_path}")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """الاحتفاظ فقط بآخر N نسخة لتوفير مساحة التخزين (checkpoints ضخمة جدًا)"""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ], key=lambda x: int(x.split("_")[-1].replace(".pt", "")))

        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))
            print(f"🗑️  حُذف Checkpoint قديم: {oldest}")

    def load_checkpoint(self, checkpoint_path: str, model, optimizer, lr_scheduler):
        """استعادة كاملة لحالة التدريب من Checkpoint محفوظ"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler and checkpoint["lr_scheduler_state_dict"]:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # استعادة حالة العشوائية لضمان الاستمرارية المطابقة
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])

        print(f"✅ تم استعادة Checkpoint من الخطوة {checkpoint['step']}، "
              f"الخسارة عند الحفظ: {checkpoint['loss']:.4f}")

        return checkpoint["step"], checkpoint["epoch"]

    def find_latest_checkpoint(self) -> str:
        """إيجاد أحدث Checkpoint متاح تلقائيًا (لاستئناف تلقائي بعد انقطاع)"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ]
        if not checkpoints:
            return None
        latest = max(checkpoints,
                      key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        return os.path.join(self.checkpoint_dir, latest)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_fsdp_checkpoint_consolidated(fsdp_model: FSDP, save_path: str, rank: int):
    """
    حفظ Checkpoint مُجمَّع (Consolidated) من نموذج FSDP: يُجمِّع كل الشرائح
    المُوزَّعة في state_dict واحد كامل على rank=0 فقط، مناسب للتوزيع
    اللاحق (نشر النموذج، أو تحميله لاحقًا على عدد GPUs مختلف).

    العيب: يتطلب ذاكرة كافية على rank=0 لاستيعاب النموذج الكامل لحظيًا.
    """
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,  # نقل الحالة المُجمَّعة لذاكرة CPU لتفادي OOM على GPU
        rank0_only=True,       # فقط rank=0 يحتفظ بالنسخة الكاملة النهائية
    )

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        full_state_dict = fsdp_model.state_dict()

    if rank == 0:
        torch.save(full_state_dict, save_path)
        print(f"✅ تم حفظ Checkpoint مُجمَّع كامل في {save_path}")

def save_fsdp_checkpoint_sharded(fsdp_model: FSDP, save_dir: str):
    """
    حفظ Checkpoint مُجزَّأ (Sharded): كل GPU يحفظ شريحته الخاصة فقط
    في ملف منفصل. أسرع بكثير وأقل استهلاكًا للذاكرة من النهج المُجمَّع،
    وهو الخيار المُفضَّل للـ checkpoints الدورية أثناء التدريب (الاستئناف
    يكون غالبًا على نفس عدد GPUs بالضبط).
    """
    from torch.distributed.fsdp import ShardedStateDictConfig

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True)
    ):
        sharded_state_dict = fsdp_model.state_dict()
        rank = dist.get_rank()
        torch.save(sharded_state_dict,
                   os.path.join(save_dir, f"shard_rank_{rank}.pt"))

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from collections import deque

class TrainingMonitor:
    """
    نظام مراقبة شامل للتدريب يتتبع المقاييس الحيوية ويكتشف المشاكل
    مبكرًا قبل أن تُهدر ساعات/أيام من الحوسبة على تدريب فاسد.
    """
    def __init__(self, window_size: int = 100,
                 loss_spike_threshold: float = 3.0,
                 grad_norm_explosion_threshold: float = 100.0):
        self.window_size = window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_explosion_threshold = grad_norm_explosion_threshold

        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.alerts_log = []

    def compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """
        حساب القاعدة الإقليدية الكلية (L2 norm) لكل تدرجات النموذج،
        وهو مقياس أساسي لاستقرار التدريب. قيم مرتفعة جدًا فجأة تشير
        لعدم استقرار رقمي ("Gradient Explosion").
        """
        total_norm_squared = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_squared += param_norm.item() ** 2
        return total_norm_squared ** 0.5

    def log_step(self, step: int, loss: float, model: torch.nn.Module,
                 learning_rate: float, step_duration: float) -> dict:
        """
        تسجيل مقاييس خطوة تدريب واحدة، مع كشف تلقائي عن الحالات الشاذة.
        يُعيد قاموسًا بأي تنبيهات اكتُشفت (فارغ إذا كل شيء طبيعي).
        """
        grad_norm = self.compute_gradient_norm(model)
        alerts = {}

        # 1. كشف Loss Spike: الخسارة الحالية أعلى بكثير من المتوسط المتحرك الأخير
        if len(self.loss_history) >= 10:
            recent_mean = sum(list(self.loss_history)[-10:]) / 10
            recent_std = (sum((x - recent_mean)**2
                          for x in list(self.loss_history)[-10:]) / 10) ** 0.5
            if recent_std > 0 and loss > recent_mean + self.loss_spike_threshold * recent_std:
                alerts["loss_spike"] = (
                    f"⚠️ Loss Spike عند الخطوة {step}: "
                    f"{loss:.4f} (المتوسط الأخير: {recent_mean:.4f} ± {recent_std:.4f})"
                )

        # 2. كشف Gradient Explosion
        if grad_norm > self.grad_norm_explosion_threshold:
            alerts["grad_explosion"] = (
                f"⚠️ Gradient Norm مرتفع جدًا عند الخطوة {step}: {grad_norm:.2f}"
            )

        # 3. كشف NaN/Inf (الأخطر على الإطلاق - يُفسد التدريب فورًا)
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            alerts["nan_inf_loss"] = (
                f"🔴 خطأ حرج: الخسارة NaN/Inf عند الخطوة {step}! "
                f"يجب إيقاف التدريب فورًا والاستعادة من آخر checkpoint سليم"
            )

        # تحديث السجلات
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.step_times.append(step_duration)

        if alerts:
            self.alerts_log.extend(alerts.values())
            for msg in alerts.values():
                print(msg)

        return {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "step_duration_sec": step_duration,
            "tokens_per_sec": None,  # يُحسَب خارجيًا حسب batch_size×seq_len
            "alerts": alerts,
        }

    def get_training_speed_stats(self) -> dict:
        """إحصائيات سرعة التدريب لتقدير الوقت المتبقي والتكلفة"""
        if not self.step_times:
            return {}
        avg_step_time = sum(self.step_times) / len(self.step_times)
        return {
            "avg_step_time_sec": avg_step_time,
            "steps_per_hour": 3600 / avg_step_time if avg_step_time > 0 else 0,
        }

def estimate_remaining_training_time(monitor: TrainingMonitor,
                                       current_step: int,
                                       total_steps: int) -> str:
    """تقدير الوقت المتبقي للتدريب بناء على متوسط سرعة الخطوات الأخيرة"""
    stats = monitor.get_training_speed_stats()
    if not stats:
        return "غير متاح بعد (بيانات غير كافية)"

    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps * stats["avg_step_time_sec"]
    remaining_hours = remaining_seconds / 3600
    remaining_days = remaining_hours / 24

    return (f"الوقت المتبقي التقديري: {remaining_hours:.1f} ساعة "
            f"({remaining_days:.1f} يوم) عند السرعة الحالية "
            f"({stats['steps_per_hour']:.0f} خطوة/ساعة)")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import wandb

def setup_wandb_tracking(project_name: str, run_config: dict):
    """
    إعداد تتبع تجريبي كامل بـ Weights & Biases، الأداة المعيارية
    صناعيًا لتتبع تجارب تدريب LLMs (تستخدمها معظم المختبرات البحثية
    الكبرى: OpenAI, Meta AI, DeepMind وغيرها).
    """
    wandb.init(
        project=project_name,
        config=run_config,
        # متابعة استهلاك GPU/CPU/الذاكرة تلقائيًا
        settings=wandb.Settings(_disable_stats=False),
    )

def log_training_step_to_wandb(step: int, metrics: dict):
    """تسجيل مقاييس خطوة واحدة في لوحة wandb التفاعلية"""
    wandb.log({
        "train/loss": metrics["loss"],
        "train/grad_norm": metrics["grad_norm"],
        "train/learning_rate": metrics["learning_rate"],
        "train/step_duration_sec": metrics["step_duration_sec"],
        "system/gpu_memory_allocated_gb":
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "system/gpu_memory_reserved_gb":
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
    }, step=step)

def log_evaluation_metrics_to_wandb(step: int, eval_metrics: dict):
    """تسجيل مقاييس التقييم الدورية (كل N خطوة على مجموعة تحقق محجوزة)"""
    wandb.log({
        "eval/loss": eval_metrics["eval_loss"],
        "eval/perplexity": eval_metrics["perplexity"],
    }, step=step)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def estimate_training_cost(num_params: float, num_tokens: float,
                             gpu_type: str = "H100",
                             mfu: float = 0.4) -> dict:
    """
    تقدير تكلفة تدريب نموذج بالدولار، بناء على:
    - ميزانية الحوسبة المطلوبة C ≈ 6ND
    - معدل الأداء الفعلي للـ GPU (مع اعتبار MFU - نسبة الاستغلال الفعلي)
    - التكلفة التقريبية لساعة GPU واحدة (أسعار سوقية تقريبية، تتغير بمرور الوقت)
    """
    gpu_specs = {
        "H100": {"peak_flops_bf16": 989e12, "cost_per_hour_usd": 2.5},  # أسعار Cloud تقريبية
        "A100": {"peak_flops_bf16": 312e12, "cost_per_hour_usd": 1.5},
        "V100": {"peak_flops_bf16": 125e12, "cost_per_hour_usd": 0.9},
    }

    specs = gpu_specs[gpu_type]
    total_flops_required = 6 * num_params * num_tokens

    # الأداء الفعلي = الأداء النظري الأقصى × نسبة الاستغلال الفعلي (MFU)
    # MFU النموذجي في تدريب LLM حقيقي يتراوح 30%-55% عادة (نادرًا يتجاوز 60%)
    effective_flops_per_sec = specs["peak_flops_bf16"] * mfu

    total_gpu_seconds = total_flops_required / effective_flops_per_sec
    total_gpu_hours = total_gpu_seconds / 3600
    total_cost_usd = total_gpu_hours * specs["cost_per_hour_usd"]

    return {
        "total_flops": total_flops_required,
        "gpu_type": gpu_type,
        "mfu_assumed": mfu,
        "total_gpu_hours": total_gpu_hours,
        "estimated_cost_usd": total_cost_usd,
    }

def compare_training_cost_scenarios():
    """مقارنة تكلفة تدريب عدة أحجام نماذج واقعية على H100"""
    scenarios = [
        {"name": "نموذج صغير توضيحي", "N": 125e6, "D": 2.5e9},
        {"name": "نموذج متوسط (مشروع هذا الجزء)", "N": 150e6, "D": 3e9},
        {"name": "LLaMA-2 7B (تقريبي)", "N": 7e9, "D": 2e12},
        {"name": "LLaMA-2 70B (تقريبي)", "N": 70e9, "D": 2e12},
    ]

    print(f"{'النموذج':<35}{'GPU-hours':<15}{'التكلفة التقديرية (USD)':<25}")
    print("-" * 75)
    for s in scenarios:
        result = estimate_training_cost(s["N"], s["D"], gpu_type="H100", mfu=0.4)
        print(f"{s['name']:<35}{result['total_gpu_hours']:<15,.0f}"
              f"${result['estimated_cost_usd']:<24,.0f}")

compare_training_cost_scenarios()

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def simulate_loss_spike_detection_and_recovery():
    """محاكاة سيناريو واقعي لاكتشاف Loss Spike والتعافي منه"""
    monitor = TrainingMonitor(loss_spike_threshold=3.0)
    checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")

    # محاكاة تاريخ خسارة طبيعي يتذبذب حول 2.3 بانحراف معياري صغير
    import random
    random.seed(42)
    simulated_losses = [2.3 + random.gauss(0, 0.05) for _ in range(20)]
    simulated_losses.append(8.7)  # ← القفزة المفاجئة

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)
            self.fc.weight.grad = torch.randn(10, 10) * 50  # تدرج كبير محاكاة للانفجار

    dummy_model = DummyModel()

    for step, loss in enumerate(simulated_losses):
        result = monitor.log_step(
            step=step, loss=loss, model=dummy_model,
            learning_rate=3e-4, step_duration=1.2
        )
        if result["alerts"]:
            print(f"\n🚨 عند الخطوة {step}: اكتُشفت مشكلة!")
            print("الإجراء المُوصى به:")
            print("   1. إيقاف التدريب الحالي فورًا")
            print("   2. العودة لآخر checkpoint سليم (قبل بداية القفزة)")
            print("   3. فحص batch البيانات التي سبقت القفزة (غالبًا السبب)")
            print("   4. إن تكررت المشكلة: تقليل معدل التعلم أو تشديد Gradient Clipping")
            break

simulate_loss_spike_detection_and_recovery()

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-05-building-llms/ch22-checkpointing-monitoring.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-2
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
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("wandb", "wandb")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import os
import json
import random
import numpy as np
from datetime import datetime

class CheckpointManager:
    """
    مدير Checkpoints كامل للتدريب طويل الأمد، يضمن إمكانية استئناف
    التدريب من أي نقطة توقف بشكل مطابق تمامًا (Deterministic Resume)
    لما كان سيحدث لو لم يتوقف التدريب إطلاقًا.
    """
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self,
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         lr_scheduler,
                         step: int,
                         epoch: int,
                         loss: float,
                         extra_metadata: dict = None):
        """
        حفظ Checkpoint كامل يحتوي كل ما يلزم لاستئناف التدريب
        بشكل مطابق تمامًا من حيث الحالة الحسابية.
        """
        checkpoint = {
            # 1. أوزان النموذج - العنصر الأوضح لكن غير الكافي وحده
            "model_state_dict": model.state_dict(),

            # 2. حالات المُحسِّن - حرجة جدًا! بدونها، Adam/AdamW يفقد
            # متوسطاته المتحركة (m, v) المتراكمة ويبدأ التحسين "من الصفر"
            # فعليًا حتى لو كانت الأوزان محفوظة، مما يسبب اضطرابًا كبيرًا
            # في منحنى الخسارة عند الاستئناف
            "optimizer_state_dict": optimizer.state_dict(),

            # 3. حالة جدول معدل التعلم (مهم لمعرفة أين نحن في Warmup/Decay)
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,

            # 4. عداد الخطوات/العصور - لاستئناف جدول LR وDataLoader بشكل صحيح
            "step": step,
            "epoch": epoch,

            # 5. قيمة الخسارة عند هذه النقطة - للمقارنة والتحقق من السلامة
            "loss": loss,

            # 6. حالة مولدات الأرقام العشوائية - لضمان قابلية إعادة الإنتاج
            # الكاملة (نفس تسلسل Dropout/أي عشوائية أخرى عند الاستئناف)
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },

            # 7. بيانات وصفية إضافية (تاريخ الحفظ، إصدار الكود، إعدادات التدريب)
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                **(extra_metadata or {}),
            },
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_step_{step}.pt"
        )
        # الحفظ بشكل ذري (atomic): نكتب لملف مؤقت أولاً ثم نُعيد تسميته،
        # لتفادي ملف checkpoint تالف إذا انقطع التيار أثناء الكتابة
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, checkpoint_path)

        print(f"✅ تم حفظ Checkpoint عند الخطوة {step} في {checkpoint_path}")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """الاحتفاظ فقط بآخر N نسخة لتوفير مساحة التخزين (checkpoints ضخمة جدًا)"""
        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ], key=lambda x: int(x.split("_")[-1].replace(".pt", "")))

        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest))
            print(f"🗑️  حُذف Checkpoint قديم: {oldest}")

    def load_checkpoint(self, checkpoint_path: str, model, optimizer, lr_scheduler):
        """استعادة كاملة لحالة التدريب من Checkpoint محفوظ"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler and checkpoint["lr_scheduler_state_dict"]:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # استعادة حالة العشوائية لضمان الاستمرارية المطابقة
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state["torch"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])

        print(f"✅ تم استعادة Checkpoint من الخطوة {checkpoint['step']}، "
              f"الخسارة عند الحفظ: {checkpoint['loss']:.4f}")

        return checkpoint["step"], checkpoint["epoch"]

    def find_latest_checkpoint(self) -> str:
        """إيجاد أحدث Checkpoint متاح تلقائيًا (لاستئناف تلقائي بعد انقطاع)"""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_step_")
        ]
        if not checkpoints:
            return None
        latest = max(checkpoints,
                      key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        return os.path.join(self.checkpoint_dir, latest)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_fsdp_checkpoint_consolidated(fsdp_model: FSDP, save_path: str, rank: int):
    """
    حفظ Checkpoint مُجمَّع (Consolidated) من نموذج FSDP: يُجمِّع كل الشرائح
    المُوزَّعة في state_dict واحد كامل على rank=0 فقط، مناسب للتوزيع
    اللاحق (نشر النموذج، أو تحميله لاحقًا على عدد GPUs مختلف).

    العيب: يتطلب ذاكرة كافية على rank=0 لاستيعاب النموذج الكامل لحظيًا.
    """
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,  # نقل الحالة المُجمَّعة لذاكرة CPU لتفادي OOM على GPU
        rank0_only=True,       # فقط rank=0 يحتفظ بالنسخة الكاملة النهائية
    )

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        full_state_dict = fsdp_model.state_dict()

    if rank == 0:
        torch.save(full_state_dict, save_path)
        print(f"✅ تم حفظ Checkpoint مُجمَّع كامل في {save_path}")

def save_fsdp_checkpoint_sharded(fsdp_model: FSDP, save_dir: str):
    """
    حفظ Checkpoint مُجزَّأ (Sharded): كل GPU يحفظ شريحته الخاصة فقط
    في ملف منفصل. أسرع بكثير وأقل استهلاكًا للذاكرة من النهج المُجمَّع،
    وهو الخيار المُفضَّل للـ checkpoints الدورية أثناء التدريب (الاستئناف
    يكون غالبًا على نفس عدد GPUs بالضبط).
    """
    from torch.distributed.fsdp import ShardedStateDictConfig

    with FSDP.state_dict_type(
        fsdp_model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True)
    ):
        sharded_state_dict = fsdp_model.state_dict()
        rank = dist.get_rank()
        torch.save(sharded_state_dict,
                   os.path.join(save_dir, f"shard_rank_{rank}.pt"))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import time
from collections import deque

class TrainingMonitor:
    """
    نظام مراقبة شامل للتدريب يتتبع المقاييس الحيوية ويكتشف المشاكل
    مبكرًا قبل أن تُهدر ساعات/أيام من الحوسبة على تدريب فاسد.
    """
    def __init__(self, window_size: int = 100,
                 loss_spike_threshold: float = 3.0,
                 grad_norm_explosion_threshold: float = 100.0):
        self.window_size = window_size
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_explosion_threshold = grad_norm_explosion_threshold

        self.loss_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        self.alerts_log = []

    def compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """
        حساب القاعدة الإقليدية الكلية (L2 norm) لكل تدرجات النموذج،
        وهو مقياس أساسي لاستقرار التدريب. قيم مرتفعة جدًا فجأة تشير
        لعدم استقرار رقمي ("Gradient Explosion").
        """
        total_norm_squared = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_squared += param_norm.item() ** 2
        return total_norm_squared ** 0.5

    def log_step(self, step: int, loss: float, model: torch.nn.Module,
                 learning_rate: float, step_duration: float) -> dict:
        """
        تسجيل مقاييس خطوة تدريب واحدة، مع كشف تلقائي عن الحالات الشاذة.
        يُعيد قاموسًا بأي تنبيهات اكتُشفت (فارغ إذا كل شيء طبيعي).
        """
        grad_norm = self.compute_gradient_norm(model)
        alerts = {}

        # 1. كشف Loss Spike: الخسارة الحالية أعلى بكثير من المتوسط المتحرك الأخير
        if len(self.loss_history) >= 10:
            recent_mean = sum(list(self.loss_history)[-10:]) / 10
            recent_std = (sum((x - recent_mean)**2
                          for x in list(self.loss_history)[-10:]) / 10) ** 0.5
            if recent_std > 0 and loss > recent_mean + self.loss_spike_threshold * recent_std:
                alerts["loss_spike"] = (
                    f"⚠️ Loss Spike عند الخطوة {step}: "
                    f"{loss:.4f} (المتوسط الأخير: {recent_mean:.4f} ± {recent_std:.4f})"
                )

        # 2. كشف Gradient Explosion
        if grad_norm > self.grad_norm_explosion_threshold:
            alerts["grad_explosion"] = (
                f"⚠️ Gradient Norm مرتفع جدًا عند الخطوة {step}: {grad_norm:.2f}"
            )

        # 3. كشف NaN/Inf (الأخطر على الإطلاق - يُفسد التدريب فورًا)
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            alerts["nan_inf_loss"] = (
                f"🔴 خطأ حرج: الخسارة NaN/Inf عند الخطوة {step}! "
                f"يجب إيقاف التدريب فورًا والاستعادة من آخر checkpoint سليم"
            )

        # تحديث السجلات
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.step_times.append(step_duration)

        if alerts:
            self.alerts_log.extend(alerts.values())
            for msg in alerts.values():
                print(msg)

        return {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "step_duration_sec": step_duration,
            "tokens_per_sec": None,  # يُحسَب خارجيًا حسب batch_size×seq_len
            "alerts": alerts,
        }

    def get_training_speed_stats(self) -> dict:
        """إحصائيات سرعة التدريب لتقدير الوقت المتبقي والتكلفة"""
        if not self.step_times:
            return {}
        avg_step_time = sum(self.step_times) / len(self.step_times)
        return {
            "avg_step_time_sec": avg_step_time,
            "steps_per_hour": 3600 / avg_step_time if avg_step_time > 0 else 0,
        }

def estimate_remaining_training_time(monitor: TrainingMonitor,
                                       current_step: int,
                                       total_steps: int) -> str:
    """تقدير الوقت المتبقي للتدريب بناء على متوسط سرعة الخطوات الأخيرة"""
    stats = monitor.get_training_speed_stats()
    if not stats:
        return "غير متاح بعد (بيانات غير كافية)"

    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps * stats["avg_step_time_sec"]
    remaining_hours = remaining_seconds / 3600
    remaining_days = remaining_hours / 24

    return (f"الوقت المتبقي التقديري: {remaining_hours:.1f} ساعة "
            f"({remaining_days:.1f} يوم) عند السرعة الحالية "
            f"({stats['steps_per_hour']:.0f} خطوة/ساعة)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import wandb

def setup_wandb_tracking(project_name: str, run_config: dict):
    """
    إعداد تتبع تجريبي كامل بـ Weights & Biases، الأداة المعيارية
    صناعيًا لتتبع تجارب تدريب LLMs (تستخدمها معظم المختبرات البحثية
    الكبرى: OpenAI, Meta AI, DeepMind وغيرها).
    """
    wandb.init(
        project=project_name,
        config=run_config,
        # متابعة استهلاك GPU/CPU/الذاكرة تلقائيًا
        settings=wandb.Settings(_disable_stats=False),
    )

def log_training_step_to_wandb(step: int, metrics: dict):
    """تسجيل مقاييس خطوة واحدة في لوحة wandb التفاعلية"""
    wandb.log({
        "train/loss": metrics["loss"],
        "train/grad_norm": metrics["grad_norm"],
        "train/learning_rate": metrics["learning_rate"],
        "train/step_duration_sec": metrics["step_duration_sec"],
        "system/gpu_memory_allocated_gb":
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "system/gpu_memory_reserved_gb":
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
    }, step=step)

def log_evaluation_metrics_to_wandb(step: int, eval_metrics: dict):
    """تسجيل مقاييس التقييم الدورية (كل N خطوة على مجموعة تحقق محجوزة)"""
    wandb.log({
        "eval/loss": eval_metrics["eval_loss"],
        "eval/perplexity": eval_metrics["perplexity"],
    }, step=step)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def estimate_training_cost(num_params: float, num_tokens: float,
                             gpu_type: str = "H100",
                             mfu: float = 0.4) -> dict:
    """
    تقدير تكلفة تدريب نموذج بالدولار، بناء على:
    - ميزانية الحوسبة المطلوبة C ≈ 6ND
    - معدل الأداء الفعلي للـ GPU (مع اعتبار MFU - نسبة الاستغلال الفعلي)
    - التكلفة التقريبية لساعة GPU واحدة (أسعار سوقية تقريبية، تتغير بمرور الوقت)
    """
    gpu_specs = {
        "H100": {"peak_flops_bf16": 989e12, "cost_per_hour_usd": 2.5},  # أسعار Cloud تقريبية
        "A100": {"peak_flops_bf16": 312e12, "cost_per_hour_usd": 1.5},
        "V100": {"peak_flops_bf16": 125e12, "cost_per_hour_usd": 0.9},
    }

    specs = gpu_specs[gpu_type]
    total_flops_required = 6 * num_params * num_tokens

    # الأداء الفعلي = الأداء النظري الأقصى × نسبة الاستغلال الفعلي (MFU)
    # MFU النموذجي في تدريب LLM حقيقي يتراوح 30%-55% عادة (نادرًا يتجاوز 60%)
    effective_flops_per_sec = specs["peak_flops_bf16"] * mfu

    total_gpu_seconds = total_flops_required / effective_flops_per_sec
    total_gpu_hours = total_gpu_seconds / 3600
    total_cost_usd = total_gpu_hours * specs["cost_per_hour_usd"]

    return {
        "total_flops": total_flops_required,
        "gpu_type": gpu_type,
        "mfu_assumed": mfu,
        "total_gpu_hours": total_gpu_hours,
        "estimated_cost_usd": total_cost_usd,
    }

def compare_training_cost_scenarios():
    """مقارنة تكلفة تدريب عدة أحجام نماذج واقعية على H100"""
    scenarios = [
        {"name": "نموذج صغير توضيحي", "N": 125e6, "D": 2.5e9},
        {"name": "نموذج متوسط (مشروع هذا الجزء)", "N": 150e6, "D": 3e9},
        {"name": "LLaMA-2 7B (تقريبي)", "N": 7e9, "D": 2e12},
        {"name": "LLaMA-2 70B (تقريبي)", "N": 70e9, "D": 2e12},
    ]

    print(f"{'النموذج':<35}{'GPU-hours':<15}{'التكلفة التقديرية (USD)':<25}")
    print("-" * 75)
    for s in scenarios:
        result = estimate_training_cost(s["N"], s["D"], gpu_type="H100", mfu=0.4)
        print(f"{s['name']:<35}{result['total_gpu_hours']:<15,.0f}"
              f"${result['estimated_cost_usd']:<24,.0f}")

compare_training_cost_scenarios()

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def simulate_loss_spike_detection_and_recovery():
    """محاكاة سيناريو واقعي لاكتشاف Loss Spike والتعافي منه"""
    monitor = TrainingMonitor(loss_spike_threshold=3.0)
    checkpoint_manager = CheckpointManager(checkpoint_dir="./checkpoints")

    # محاكاة تاريخ خسارة طبيعي يتذبذب حول 2.3 بانحراف معياري صغير
    import random
    random.seed(42)
    simulated_losses = [2.3 + random.gauss(0, 0.05) for _ in range(20)]
    simulated_losses.append(8.7)  # ← القفزة المفاجئة

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)
            self.fc.weight.grad = torch.randn(10, 10) * 50  # تدرج كبير محاكاة للانفجار

    dummy_model = DummyModel()

    for step, loss in enumerate(simulated_losses):
        result = monitor.log_step(
            step=step, loss=loss, model=dummy_model,
            learning_rate=3e-4, step_duration=1.2
        )
        if result["alerts"]:
            print(f"\n🚨 عند الخطوة {step}: اكتُشفت مشكلة!")
            print("الإجراء المُوصى به:")
            print("   1. إيقاف التدريب الحالي فورًا")
            print("   2. العودة لآخر checkpoint سليم (قبل بداية القفزة)")
            print("   3. فحص batch البيانات التي سبقت القفزة (غالبًا السبب)")
            print("   4. إن تكررت المشكلة: تقليل معدل التعلم أو تشديد Gradient Clipping")
            break

simulate_loss_spike_detection_and_recovery()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-22-ch22-checkpointing-monitoring.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
