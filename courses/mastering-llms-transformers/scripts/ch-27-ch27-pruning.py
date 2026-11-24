#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقليم وضغط النماذج: Pruning
Mastering Transformers and Large Language Models — Pruning

Book slug : mastering-llms-transformers
Chapter   : ch-27-ch27-pruning (lesson 27)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-27-ch27-pruning.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقليم وضغط النماذج: Pruning
Mastering Transformers and Large Language Models — Pruning

Book slug : mastering-llms-transformers
Chapter   : ch-27-ch27-pruning (lesson 27)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-27-ch27-pruning.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقليم وضغط النماذج: Pruning
Mastering Transformers and Large Language Models — Pruning

Book slug : mastering-llms-transformers
Chapter   : ch-27-ch27-pruning (lesson 27)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-1
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
    ("peft", "peft"),
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
# [002] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

def magnitude_pruning_unstructured(model: nn.Module, sparsity_ratio: float = 0.5):
    """
    أبسط طريقة تقليم: إزالة الأوزان ذات أصغر قيمة مطلقة (Magnitude)
    بافتراض أنها الأقل مساهمة في المخرج النهائي. فعّالة كخط أساس
    لكنها "عمياء" عن السياق الفعلي لاستخدام الوزن أثناء الاستدلال.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            flat_weight = weight.abs().flatten()
            k = int(len(flat_weight) * sparsity_ratio)
            if k == 0:
                continue
            threshold = torch.topk(flat_weight, k, largest=False).values.max()
            mask = weight.abs() > threshold
            module.weight.data = weight * mask

            print(f"{name}: نسبة التقليم الفعلية = "
                  f"{(mask == 0).float().mean().item():.2%}")
    return model

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SparseGPTPruner:
    """
    تطبيق مُبسَّط لمبدأ SparseGPT: بدل التقليم "الأعمى" بناءً على
    قيمة الوزن فقط، نحسب تقليمًا يُقلِّل خطأ إعادة بناء مخرجات
    الطبقة (باستخدام معلومات إحصائية عن توزيع المُدخلات الفعلية -
    مصفوفة Hessian التقريبية H = X^T X)، مع تحديث الأوزان المتبقية
    للتعويض عن إزالة الأوزان المُقلَّمة (مبدأ OBS - Optimal Brain Surgeon).
    """
    def __init__(self, layer: nn.Linear, sparsity_ratio: float = 0.5,
                 damp_percent: float = 0.01):
        self.layer = layer
        self.sparsity_ratio = sparsity_ratio
        self.damp_percent = damp_percent  # لتفادي عدم استقرار معكوس المصفوفة
        self.hessian_accumulator = torch.zeros(
            layer.in_features, layer.in_features
        )
        self.n_samples = 0

    def accumulate_hessian_statistics(self, layer_input: torch.Tensor):
        """
        تجميع إحصائيات H = X^T X عبر دفعات متعددة من بيانات معايرة
        (Calibration Data - عادة بضع مئات من العيّنات التمثيلية).
        """
        # layer_input: [batch * seq_len, in_features]
        batch_samples = layer_input.shape[0]
        self.hessian_accumulator *= self.n_samples / (self.n_samples + batch_samples)
        self.n_samples += batch_samples
        scaled_input = layer_input * (2.0 / self.n_samples) ** 0.5
        self.hessian_accumulator += scaled_input.T @ scaled_input

    def prune_with_weight_update(self) -> nn.Linear:
        """
        تطبيق التقليم مع تحديث تعويضي للأوزان المتبقية (مبدأ OBS
        المُبسَّط): لكل وزن نُزيله، نُحدِّث الأوزان الأخرى في نفس
        الصف لتعويض الخطأ الناتج قدر الإمكان، باستخدام معكوس
        مصفوفة Hessian.
        """
        H = self.hessian_accumulator.clone()
        damp = self.damp_percent * torch.mean(torch.diag(H))
        H += torch.eye(H.shape[0]) * damp  # تثبيت رقمي (damping)

        H_inv = torch.linalg.inv(H)
        weight = self.layer.weight.data.clone()  # [out_features, in_features]

        # درجة أهمية كل وزن = (قيمة الوزن)² / (قيمة H_inv القطرية المقابلة)
        # - هذا هو معيار OBS الكلاسيكي لتقدير "ثمن" إزالة كل وزن
        diag_H_inv = torch.diag(H_inv)
        importance_scores = (weight ** 2) / diag_H_inv.unsqueeze(0)

        n_to_prune = int(weight.shape[1] * self.sparsity_ratio)
        for row in range(weight.shape[0]):
            row_importance = importance_scores[row]
            prune_indices = torch.topk(
                row_importance, n_to_prune, largest=False
            ).indices

            for idx in prune_indices:
                # تحديث تعويضي لباقي الأوزان في نفس الصف (مبدأ OBS)
                delta = (weight[row, idx] / diag_H_inv[idx]) * H_inv[idx, :]
                weight[row, :] -= delta
                weight[row, idx] = 0.0  # التأكد من التصفير التام

        self.layer.weight.data = weight
        return self.layer

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def wanda_pruning(layer: nn.Linear, calibration_activations: torch.Tensor,
                   sparsity_ratio: float = 0.5) -> nn.Linear:
    """
    Wanda (Sun et al., 2023): معيار تقليم بسيط جدًا لكنه فعّال بشكل
    مُدهش: درجة أهمية الوزن W_ij = |W_ij| × ||X_j||₂
    (قيمة الوزن المطلقة × معيار L2 لتنشيطات العمود المقابل عبر
    عيّنات المعايرة). لا حاجة لحساب Hessian أو تحديث تعويضي معقّد -
    فقط ترتيب وتصفير مباشر، مع نتائج منافسة جدًا لـ SparseGPT الأكثر تعقيدًا.
    """
    weight = layer.weight.data  # [out_features, in_features]

    # معيار L2 لكل عمود (ميزة إدخال) عبر عيّنات المعايرة
    activation_norms = calibration_activations.norm(p=2, dim=0)  # [in_features]

    # درجة الأهمية = |الوزن| × معيار تنشيط عموده
    importance_scores = weight.abs() * activation_norms.unsqueeze(0)

    n_to_prune_per_row = int(weight.shape[1] * sparsity_ratio)
    for row in range(weight.shape[0]):
        prune_indices = torch.topk(
            importance_scores[row], n_to_prune_per_row, largest=False
        ).indices
        weight[row, prune_indices] = 0.0

    layer.weight.data = weight
    return layer

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class StructuredLLMPruner:
    """
    تطبيق مفاهيمي لمبدأ LLM-Pruner (Ma et al., 2023): تحديد "مجموعات
    مترابطة" من الأوزان (Coupled Structures - مثل رأس انتباه كامل أو
    قناة FFN كاملة بكل ما يتصل بها من أوزان دخل وخرج)، تقدير أهميتها
    الجماعية عبر معلومات التدرج (Gradient-based Importance)، ثم إزالة
    المجموعات الأقل أهمية بالكامل - يليها إعادة ضبط سريع (LoRA)
    لاستعادة الأداء المفقود.
    """
    def __init__(self, model, importance_calibration_data):
        self.model = model
        self.calibration_data = importance_calibration_data

    def compute_head_importance(self, layer_idx: int) -> torch.Tensor:
        """
        تقدير أهمية كل رأس انتباه عبر حساسية الخسارة لإزالته
        (Taylor Expansion تقريبي: |gradient × weight|، مجمّعة على
        مستوى الرأس الكامل بدل الوزن الفردي).
        """
        self.model.zero_grad()
        for batch in self.calibration_data:
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()

        attention_layer = self.model.get_attention_layer(layer_idx)
        n_heads = attention_layer.n_heads
        head_dim = attention_layer.head_dim

        head_importance_scores = []
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = start + head_dim
            # الأهمية = مجموع |وزن × تدرجه| لكل أوزان هذا الرأس تحديدًا
            weight_slice = attention_layer.weight.data[start:end, :]
            grad_slice = attention_layer.weight.grad[start:end, :]
            importance = (weight_slice * grad_slice).abs().sum()
            head_importance_scores.append(importance.item())

        return torch.tensor(head_importance_scores)

    def prune_least_important_heads(self, layer_idx: int, n_heads_to_remove: int):
        """إزالة الرؤوس الأقل أهمية بالكامل (تقليم بنيوي حقيقي)."""
        importance = self.compute_head_importance(layer_idx)
        heads_to_remove = torch.topk(
            importance, n_heads_to_remove, largest=False
        ).indices.tolist()

        attention_layer = self.model.get_attention_layer(layer_idx)
        attention_layer.remove_heads(heads_to_remove)  # دالة مفاهيمية
        return self.model

    def post_pruning_recovery(self, recovery_dataset, lora_rank: int = 16):
        """
        خطوة ضرورية دائمًا بعد التقليم البنيوي: إعادة ضبط سريع
        وخفيف التكلفة (عبر LoRA - الجزء السابع) لاستعادة أكبر قدر
        ممكن من الأداء المفقود جراء إزالة المكوّنات البنيوية.
        """
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
        )
        recoverable_model = get_peft_model(self.model, lora_config)
        # تدريب LoRA قصير (عادة 1-2% فقط من حجم بيانات CPT الأصلية كافٍ)
        return recoverable_model

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch

W = torch.tensor([
    [0.10,  0.90, -0.20, 0.05],
    [0.70, -0.30,  0.40, 0.02],
    [-0.60, 0.20,  0.15, 0.80],
])
act_norms = torch.tensor([2.0, 0.5, 1.5, 0.1])

# --- Magnitude: نحتفظ بأكبر قيمتين مطلقًا في كل صف ---
def magnitude_prune(W, keep=2):
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = W[r].abs().topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

# --- Wanda: |W| × ||X|| ---
def wanda_prune(W, act_norms, keep=2):
    importance = W.abs() * act_norms.unsqueeze(0)
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = importance[r].topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

print("Magnitude:\n", magnitude_prune(W))
print("Wanda:\n", wanda_prune(W, act_norms))

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-1
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
    ("peft", "peft"),
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

def magnitude_pruning_unstructured(model: nn.Module, sparsity_ratio: float = 0.5):
    """
    أبسط طريقة تقليم: إزالة الأوزان ذات أصغر قيمة مطلقة (Magnitude)
    بافتراض أنها الأقل مساهمة في المخرج النهائي. فعّالة كخط أساس
    لكنها "عمياء" عن السياق الفعلي لاستخدام الوزن أثناء الاستدلال.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            flat_weight = weight.abs().flatten()
            k = int(len(flat_weight) * sparsity_ratio)
            if k == 0:
                continue
            threshold = torch.topk(flat_weight, k, largest=False).values.max()
            mask = weight.abs() > threshold
            module.weight.data = weight * mask

            print(f"{name}: نسبة التقليم الفعلية = "
                  f"{(mask == 0).float().mean().item():.2%}")
    return model

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SparseGPTPruner:
    """
    تطبيق مُبسَّط لمبدأ SparseGPT: بدل التقليم "الأعمى" بناءً على
    قيمة الوزن فقط، نحسب تقليمًا يُقلِّل خطأ إعادة بناء مخرجات
    الطبقة (باستخدام معلومات إحصائية عن توزيع المُدخلات الفعلية -
    مصفوفة Hessian التقريبية H = X^T X)، مع تحديث الأوزان المتبقية
    للتعويض عن إزالة الأوزان المُقلَّمة (مبدأ OBS - Optimal Brain Surgeon).
    """
    def __init__(self, layer: nn.Linear, sparsity_ratio: float = 0.5,
                 damp_percent: float = 0.01):
        self.layer = layer
        self.sparsity_ratio = sparsity_ratio
        self.damp_percent = damp_percent  # لتفادي عدم استقرار معكوس المصفوفة
        self.hessian_accumulator = torch.zeros(
            layer.in_features, layer.in_features
        )
        self.n_samples = 0

    def accumulate_hessian_statistics(self, layer_input: torch.Tensor):
        """
        تجميع إحصائيات H = X^T X عبر دفعات متعددة من بيانات معايرة
        (Calibration Data - عادة بضع مئات من العيّنات التمثيلية).
        """
        # layer_input: [batch * seq_len, in_features]
        batch_samples = layer_input.shape[0]
        self.hessian_accumulator *= self.n_samples / (self.n_samples + batch_samples)
        self.n_samples += batch_samples
        scaled_input = layer_input * (2.0 / self.n_samples) ** 0.5
        self.hessian_accumulator += scaled_input.T @ scaled_input

    def prune_with_weight_update(self) -> nn.Linear:
        """
        تطبيق التقليم مع تحديث تعويضي للأوزان المتبقية (مبدأ OBS
        المُبسَّط): لكل وزن نُزيله، نُحدِّث الأوزان الأخرى في نفس
        الصف لتعويض الخطأ الناتج قدر الإمكان، باستخدام معكوس
        مصفوفة Hessian.
        """
        H = self.hessian_accumulator.clone()
        damp = self.damp_percent * torch.mean(torch.diag(H))
        H += torch.eye(H.shape[0]) * damp  # تثبيت رقمي (damping)

        H_inv = torch.linalg.inv(H)
        weight = self.layer.weight.data.clone()  # [out_features, in_features]

        # درجة أهمية كل وزن = (قيمة الوزن)² / (قيمة H_inv القطرية المقابلة)
        # - هذا هو معيار OBS الكلاسيكي لتقدير "ثمن" إزالة كل وزن
        diag_H_inv = torch.diag(H_inv)
        importance_scores = (weight ** 2) / diag_H_inv.unsqueeze(0)

        n_to_prune = int(weight.shape[1] * self.sparsity_ratio)
        for row in range(weight.shape[0]):
            row_importance = importance_scores[row]
            prune_indices = torch.topk(
                row_importance, n_to_prune, largest=False
            ).indices

            for idx in prune_indices:
                # تحديث تعويضي لباقي الأوزان في نفس الصف (مبدأ OBS)
                delta = (weight[row, idx] / diag_H_inv[idx]) * H_inv[idx, :]
                weight[row, :] -= delta
                weight[row, idx] = 0.0  # التأكد من التصفير التام

        self.layer.weight.data = weight
        return self.layer

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def wanda_pruning(layer: nn.Linear, calibration_activations: torch.Tensor,
                   sparsity_ratio: float = 0.5) -> nn.Linear:
    """
    Wanda (Sun et al., 2023): معيار تقليم بسيط جدًا لكنه فعّال بشكل
    مُدهش: درجة أهمية الوزن W_ij = |W_ij| × ||X_j||₂
    (قيمة الوزن المطلقة × معيار L2 لتنشيطات العمود المقابل عبر
    عيّنات المعايرة). لا حاجة لحساب Hessian أو تحديث تعويضي معقّد -
    فقط ترتيب وتصفير مباشر، مع نتائج منافسة جدًا لـ SparseGPT الأكثر تعقيدًا.
    """
    weight = layer.weight.data  # [out_features, in_features]

    # معيار L2 لكل عمود (ميزة إدخال) عبر عيّنات المعايرة
    activation_norms = calibration_activations.norm(p=2, dim=0)  # [in_features]

    # درجة الأهمية = |الوزن| × معيار تنشيط عموده
    importance_scores = weight.abs() * activation_norms.unsqueeze(0)

    n_to_prune_per_row = int(weight.shape[1] * sparsity_ratio)
    for row in range(weight.shape[0]):
        prune_indices = torch.topk(
            importance_scores[row], n_to_prune_per_row, largest=False
        ).indices
        weight[row, prune_indices] = 0.0

    layer.weight.data = weight
    return layer

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class StructuredLLMPruner:
    """
    تطبيق مفاهيمي لمبدأ LLM-Pruner (Ma et al., 2023): تحديد "مجموعات
    مترابطة" من الأوزان (Coupled Structures - مثل رأس انتباه كامل أو
    قناة FFN كاملة بكل ما يتصل بها من أوزان دخل وخرج)، تقدير أهميتها
    الجماعية عبر معلومات التدرج (Gradient-based Importance)، ثم إزالة
    المجموعات الأقل أهمية بالكامل - يليها إعادة ضبط سريع (LoRA)
    لاستعادة الأداء المفقود.
    """
    def __init__(self, model, importance_calibration_data):
        self.model = model
        self.calibration_data = importance_calibration_data

    def compute_head_importance(self, layer_idx: int) -> torch.Tensor:
        """
        تقدير أهمية كل رأس انتباه عبر حساسية الخسارة لإزالته
        (Taylor Expansion تقريبي: |gradient × weight|، مجمّعة على
        مستوى الرأس الكامل بدل الوزن الفردي).
        """
        self.model.zero_grad()
        for batch in self.calibration_data:
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()

        attention_layer = self.model.get_attention_layer(layer_idx)
        n_heads = attention_layer.n_heads
        head_dim = attention_layer.head_dim

        head_importance_scores = []
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = start + head_dim
            # الأهمية = مجموع |وزن × تدرجه| لكل أوزان هذا الرأس تحديدًا
            weight_slice = attention_layer.weight.data[start:end, :]
            grad_slice = attention_layer.weight.grad[start:end, :]
            importance = (weight_slice * grad_slice).abs().sum()
            head_importance_scores.append(importance.item())

        return torch.tensor(head_importance_scores)

    def prune_least_important_heads(self, layer_idx: int, n_heads_to_remove: int):
        """إزالة الرؤوس الأقل أهمية بالكامل (تقليم بنيوي حقيقي)."""
        importance = self.compute_head_importance(layer_idx)
        heads_to_remove = torch.topk(
            importance, n_heads_to_remove, largest=False
        ).indices.tolist()

        attention_layer = self.model.get_attention_layer(layer_idx)
        attention_layer.remove_heads(heads_to_remove)  # دالة مفاهيمية
        return self.model

    def post_pruning_recovery(self, recovery_dataset, lora_rank: int = 16):
        """
        خطوة ضرورية دائمًا بعد التقليم البنيوي: إعادة ضبط سريع
        وخفيف التكلفة (عبر LoRA - الجزء السابع) لاستعادة أكبر قدر
        ممكن من الأداء المفقود جراء إزالة المكوّنات البنيوية.
        """
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
        )
        recoverable_model = get_peft_model(self.model, lora_config)
        # تدريب LoRA قصير (عادة 1-2% فقط من حجم بيانات CPT الأصلية كافٍ)
        return recoverable_model

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch

W = torch.tensor([
    [0.10,  0.90, -0.20, 0.05],
    [0.70, -0.30,  0.40, 0.02],
    [-0.60, 0.20,  0.15, 0.80],
])
act_norms = torch.tensor([2.0, 0.5, 1.5, 0.1])

# --- Magnitude: نحتفظ بأكبر قيمتين مطلقًا في كل صف ---
def magnitude_prune(W, keep=2):
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = W[r].abs().topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

# --- Wanda: |W| × ||X|| ---
def wanda_prune(W, act_norms, keep=2):
    importance = W.abs() * act_norms.unsqueeze(0)
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = importance[r].topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

print("Magnitude:\n", magnitude_prune(W))
print("Wanda:\n", wanda_prune(W, act_norms))

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — التقليم وضغط النماذج: Pruning
Mastering Transformers and Large Language Models — Pruning

Book slug : mastering-llms-transformers
Chapter   : ch-27-ch27-pruning (lesson 27)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-1
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
    ("peft", "peft"),
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
# [002] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

def magnitude_pruning_unstructured(model: nn.Module, sparsity_ratio: float = 0.5):
    """
    أبسط طريقة تقليم: إزالة الأوزان ذات أصغر قيمة مطلقة (Magnitude)
    بافتراض أنها الأقل مساهمة في المخرج النهائي. فعّالة كخط أساس
    لكنها "عمياء" عن السياق الفعلي لاستخدام الوزن أثناء الاستدلال.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            flat_weight = weight.abs().flatten()
            k = int(len(flat_weight) * sparsity_ratio)
            if k == 0:
                continue
            threshold = torch.topk(flat_weight, k, largest=False).values.max()
            mask = weight.abs() > threshold
            module.weight.data = weight * mask

            print(f"{name}: نسبة التقليم الفعلية = "
                  f"{(mask == 0).float().mean().item():.2%}")
    return model

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class SparseGPTPruner:
    """
    تطبيق مُبسَّط لمبدأ SparseGPT: بدل التقليم "الأعمى" بناءً على
    قيمة الوزن فقط، نحسب تقليمًا يُقلِّل خطأ إعادة بناء مخرجات
    الطبقة (باستخدام معلومات إحصائية عن توزيع المُدخلات الفعلية -
    مصفوفة Hessian التقريبية H = X^T X)، مع تحديث الأوزان المتبقية
    للتعويض عن إزالة الأوزان المُقلَّمة (مبدأ OBS - Optimal Brain Surgeon).
    """
    def __init__(self, layer: nn.Linear, sparsity_ratio: float = 0.5,
                 damp_percent: float = 0.01):
        self.layer = layer
        self.sparsity_ratio = sparsity_ratio
        self.damp_percent = damp_percent  # لتفادي عدم استقرار معكوس المصفوفة
        self.hessian_accumulator = torch.zeros(
            layer.in_features, layer.in_features
        )
        self.n_samples = 0

    def accumulate_hessian_statistics(self, layer_input: torch.Tensor):
        """
        تجميع إحصائيات H = X^T X عبر دفعات متعددة من بيانات معايرة
        (Calibration Data - عادة بضع مئات من العيّنات التمثيلية).
        """
        # layer_input: [batch * seq_len, in_features]
        batch_samples = layer_input.shape[0]
        self.hessian_accumulator *= self.n_samples / (self.n_samples + batch_samples)
        self.n_samples += batch_samples
        scaled_input = layer_input * (2.0 / self.n_samples) ** 0.5
        self.hessian_accumulator += scaled_input.T @ scaled_input

    def prune_with_weight_update(self) -> nn.Linear:
        """
        تطبيق التقليم مع تحديث تعويضي للأوزان المتبقية (مبدأ OBS
        المُبسَّط): لكل وزن نُزيله، نُحدِّث الأوزان الأخرى في نفس
        الصف لتعويض الخطأ الناتج قدر الإمكان، باستخدام معكوس
        مصفوفة Hessian.
        """
        H = self.hessian_accumulator.clone()
        damp = self.damp_percent * torch.mean(torch.diag(H))
        H += torch.eye(H.shape[0]) * damp  # تثبيت رقمي (damping)

        H_inv = torch.linalg.inv(H)
        weight = self.layer.weight.data.clone()  # [out_features, in_features]

        # درجة أهمية كل وزن = (قيمة الوزن)² / (قيمة H_inv القطرية المقابلة)
        # - هذا هو معيار OBS الكلاسيكي لتقدير "ثمن" إزالة كل وزن
        diag_H_inv = torch.diag(H_inv)
        importance_scores = (weight ** 2) / diag_H_inv.unsqueeze(0)

        n_to_prune = int(weight.shape[1] * self.sparsity_ratio)
        for row in range(weight.shape[0]):
            row_importance = importance_scores[row]
            prune_indices = torch.topk(
                row_importance, n_to_prune, largest=False
            ).indices

            for idx in prune_indices:
                # تحديث تعويضي لباقي الأوزان في نفس الصف (مبدأ OBS)
                delta = (weight[row, idx] / diag_H_inv[idx]) * H_inv[idx, :]
                weight[row, :] -= delta
                weight[row, idx] = 0.0  # التأكد من التصفير التام

        self.layer.weight.data = weight
        return self.layer

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def wanda_pruning(layer: nn.Linear, calibration_activations: torch.Tensor,
                   sparsity_ratio: float = 0.5) -> nn.Linear:
    """
    Wanda (Sun et al., 2023): معيار تقليم بسيط جدًا لكنه فعّال بشكل
    مُدهش: درجة أهمية الوزن W_ij = |W_ij| × ||X_j||₂
    (قيمة الوزن المطلقة × معيار L2 لتنشيطات العمود المقابل عبر
    عيّنات المعايرة). لا حاجة لحساب Hessian أو تحديث تعويضي معقّد -
    فقط ترتيب وتصفير مباشر، مع نتائج منافسة جدًا لـ SparseGPT الأكثر تعقيدًا.
    """
    weight = layer.weight.data  # [out_features, in_features]

    # معيار L2 لكل عمود (ميزة إدخال) عبر عيّنات المعايرة
    activation_norms = calibration_activations.norm(p=2, dim=0)  # [in_features]

    # درجة الأهمية = |الوزن| × معيار تنشيط عموده
    importance_scores = weight.abs() * activation_norms.unsqueeze(0)

    n_to_prune_per_row = int(weight.shape[1] * sparsity_ratio)
    for row in range(weight.shape[0]):
        prune_indices = torch.topk(
            importance_scores[row], n_to_prune_per_row, largest=False
        ).indices
        weight[row, prune_indices] = 0.0

    layer.weight.data = weight
    return layer

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class StructuredLLMPruner:
    """
    تطبيق مفاهيمي لمبدأ LLM-Pruner (Ma et al., 2023): تحديد "مجموعات
    مترابطة" من الأوزان (Coupled Structures - مثل رأس انتباه كامل أو
    قناة FFN كاملة بكل ما يتصل بها من أوزان دخل وخرج)، تقدير أهميتها
    الجماعية عبر معلومات التدرج (Gradient-based Importance)، ثم إزالة
    المجموعات الأقل أهمية بالكامل - يليها إعادة ضبط سريع (LoRA)
    لاستعادة الأداء المفقود.
    """
    def __init__(self, model, importance_calibration_data):
        self.model = model
        self.calibration_data = importance_calibration_data

    def compute_head_importance(self, layer_idx: int) -> torch.Tensor:
        """
        تقدير أهمية كل رأس انتباه عبر حساسية الخسارة لإزالته
        (Taylor Expansion تقريبي: |gradient × weight|، مجمّعة على
        مستوى الرأس الكامل بدل الوزن الفردي).
        """
        self.model.zero_grad()
        for batch in self.calibration_data:
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()

        attention_layer = self.model.get_attention_layer(layer_idx)
        n_heads = attention_layer.n_heads
        head_dim = attention_layer.head_dim

        head_importance_scores = []
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = start + head_dim
            # الأهمية = مجموع |وزن × تدرجه| لكل أوزان هذا الرأس تحديدًا
            weight_slice = attention_layer.weight.data[start:end, :]
            grad_slice = attention_layer.weight.grad[start:end, :]
            importance = (weight_slice * grad_slice).abs().sum()
            head_importance_scores.append(importance.item())

        return torch.tensor(head_importance_scores)

    def prune_least_important_heads(self, layer_idx: int, n_heads_to_remove: int):
        """إزالة الرؤوس الأقل أهمية بالكامل (تقليم بنيوي حقيقي)."""
        importance = self.compute_head_importance(layer_idx)
        heads_to_remove = torch.topk(
            importance, n_heads_to_remove, largest=False
        ).indices.tolist()

        attention_layer = self.model.get_attention_layer(layer_idx)
        attention_layer.remove_heads(heads_to_remove)  # دالة مفاهيمية
        return self.model

    def post_pruning_recovery(self, recovery_dataset, lora_rank: int = 16):
        """
        خطوة ضرورية دائمًا بعد التقليم البنيوي: إعادة ضبط سريع
        وخفيف التكلفة (عبر LoRA - الجزء السابع) لاستعادة أكبر قدر
        ممكن من الأداء المفقود جراء إزالة المكوّنات البنيوية.
        """
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
        )
        recoverable_model = get_peft_model(self.model, lora_config)
        # تدريب LoRA قصير (عادة 1-2% فقط من حجم بيانات CPT الأصلية كافٍ)
        return recoverable_model

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
import torch

W = torch.tensor([
    [0.10,  0.90, -0.20, 0.05],
    [0.70, -0.30,  0.40, 0.02],
    [-0.60, 0.20,  0.15, 0.80],
])
act_norms = torch.tensor([2.0, 0.5, 1.5, 0.1])

# --- Magnitude: نحتفظ بأكبر قيمتين مطلقًا في كل صف ---
def magnitude_prune(W, keep=2):
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = W[r].abs().topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

# --- Wanda: |W| × ||X|| ---
def wanda_prune(W, act_norms, keep=2):
    importance = W.abs() * act_norms.unsqueeze(0)
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = importance[r].topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

print("Magnitude:\n", magnitude_prune(W))
print("Wanda:\n", wanda_prune(W, act_norms))

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-06-model-reuse/ch27-pruning.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-2
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
    ("peft", "peft"),
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

def magnitude_pruning_unstructured(model: nn.Module, sparsity_ratio: float = 0.5):
    """
    أبسط طريقة تقليم: إزالة الأوزان ذات أصغر قيمة مطلقة (Magnitude)
    بافتراض أنها الأقل مساهمة في المخرج النهائي. فعّالة كخط أساس
    لكنها "عمياء" عن السياق الفعلي لاستخدام الوزن أثناء الاستدلال.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            flat_weight = weight.abs().flatten()
            k = int(len(flat_weight) * sparsity_ratio)
            if k == 0:
                continue
            threshold = torch.topk(flat_weight, k, largest=False).values.max()
            mask = weight.abs() > threshold
            module.weight.data = weight * mask

            print(f"{name}: نسبة التقليم الفعلية = "
                  f"{(mask == 0).float().mean().item():.2%}")
    return model

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class SparseGPTPruner:
    """
    تطبيق مُبسَّط لمبدأ SparseGPT: بدل التقليم "الأعمى" بناءً على
    قيمة الوزن فقط، نحسب تقليمًا يُقلِّل خطأ إعادة بناء مخرجات
    الطبقة (باستخدام معلومات إحصائية عن توزيع المُدخلات الفعلية -
    مصفوفة Hessian التقريبية H = X^T X)، مع تحديث الأوزان المتبقية
    للتعويض عن إزالة الأوزان المُقلَّمة (مبدأ OBS - Optimal Brain Surgeon).
    """
    def __init__(self, layer: nn.Linear, sparsity_ratio: float = 0.5,
                 damp_percent: float = 0.01):
        self.layer = layer
        self.sparsity_ratio = sparsity_ratio
        self.damp_percent = damp_percent  # لتفادي عدم استقرار معكوس المصفوفة
        self.hessian_accumulator = torch.zeros(
            layer.in_features, layer.in_features
        )
        self.n_samples = 0

    def accumulate_hessian_statistics(self, layer_input: torch.Tensor):
        """
        تجميع إحصائيات H = X^T X عبر دفعات متعددة من بيانات معايرة
        (Calibration Data - عادة بضع مئات من العيّنات التمثيلية).
        """
        # layer_input: [batch * seq_len, in_features]
        batch_samples = layer_input.shape[0]
        self.hessian_accumulator *= self.n_samples / (self.n_samples + batch_samples)
        self.n_samples += batch_samples
        scaled_input = layer_input * (2.0 / self.n_samples) ** 0.5
        self.hessian_accumulator += scaled_input.T @ scaled_input

    def prune_with_weight_update(self) -> nn.Linear:
        """
        تطبيق التقليم مع تحديث تعويضي للأوزان المتبقية (مبدأ OBS
        المُبسَّط): لكل وزن نُزيله، نُحدِّث الأوزان الأخرى في نفس
        الصف لتعويض الخطأ الناتج قدر الإمكان، باستخدام معكوس
        مصفوفة Hessian.
        """
        H = self.hessian_accumulator.clone()
        damp = self.damp_percent * torch.mean(torch.diag(H))
        H += torch.eye(H.shape[0]) * damp  # تثبيت رقمي (damping)

        H_inv = torch.linalg.inv(H)
        weight = self.layer.weight.data.clone()  # [out_features, in_features]

        # درجة أهمية كل وزن = (قيمة الوزن)² / (قيمة H_inv القطرية المقابلة)
        # - هذا هو معيار OBS الكلاسيكي لتقدير "ثمن" إزالة كل وزن
        diag_H_inv = torch.diag(H_inv)
        importance_scores = (weight ** 2) / diag_H_inv.unsqueeze(0)

        n_to_prune = int(weight.shape[1] * self.sparsity_ratio)
        for row in range(weight.shape[0]):
            row_importance = importance_scores[row]
            prune_indices = torch.topk(
                row_importance, n_to_prune, largest=False
            ).indices

            for idx in prune_indices:
                # تحديث تعويضي لباقي الأوزان في نفس الصف (مبدأ OBS)
                delta = (weight[row, idx] / diag_H_inv[idx]) * H_inv[idx, :]
                weight[row, :] -= delta
                weight[row, idx] = 0.0  # التأكد من التصفير التام

        self.layer.weight.data = weight
        return self.layer

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def wanda_pruning(layer: nn.Linear, calibration_activations: torch.Tensor,
                   sparsity_ratio: float = 0.5) -> nn.Linear:
    """
    Wanda (Sun et al., 2023): معيار تقليم بسيط جدًا لكنه فعّال بشكل
    مُدهش: درجة أهمية الوزن W_ij = |W_ij| × ||X_j||₂
    (قيمة الوزن المطلقة × معيار L2 لتنشيطات العمود المقابل عبر
    عيّنات المعايرة). لا حاجة لحساب Hessian أو تحديث تعويضي معقّد -
    فقط ترتيب وتصفير مباشر، مع نتائج منافسة جدًا لـ SparseGPT الأكثر تعقيدًا.
    """
    weight = layer.weight.data  # [out_features, in_features]

    # معيار L2 لكل عمود (ميزة إدخال) عبر عيّنات المعايرة
    activation_norms = calibration_activations.norm(p=2, dim=0)  # [in_features]

    # درجة الأهمية = |الوزن| × معيار تنشيط عموده
    importance_scores = weight.abs() * activation_norms.unsqueeze(0)

    n_to_prune_per_row = int(weight.shape[1] * sparsity_ratio)
    for row in range(weight.shape[0]):
        prune_indices = torch.topk(
            importance_scores[row], n_to_prune_per_row, largest=False
        ).indices
        weight[row, prune_indices] = 0.0

    layer.weight.data = weight
    return layer

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class StructuredLLMPruner:
    """
    تطبيق مفاهيمي لمبدأ LLM-Pruner (Ma et al., 2023): تحديد "مجموعات
    مترابطة" من الأوزان (Coupled Structures - مثل رأس انتباه كامل أو
    قناة FFN كاملة بكل ما يتصل بها من أوزان دخل وخرج)، تقدير أهميتها
    الجماعية عبر معلومات التدرج (Gradient-based Importance)، ثم إزالة
    المجموعات الأقل أهمية بالكامل - يليها إعادة ضبط سريع (LoRA)
    لاستعادة الأداء المفقود.
    """
    def __init__(self, model, importance_calibration_data):
        self.model = model
        self.calibration_data = importance_calibration_data

    def compute_head_importance(self, layer_idx: int) -> torch.Tensor:
        """
        تقدير أهمية كل رأس انتباه عبر حساسية الخسارة لإزالته
        (Taylor Expansion تقريبي: |gradient × weight|، مجمّعة على
        مستوى الرأس الكامل بدل الوزن الفردي).
        """
        self.model.zero_grad()
        for batch in self.calibration_data:
            outputs = self.model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()

        attention_layer = self.model.get_attention_layer(layer_idx)
        n_heads = attention_layer.n_heads
        head_dim = attention_layer.head_dim

        head_importance_scores = []
        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = start + head_dim
            # الأهمية = مجموع |وزن × تدرجه| لكل أوزان هذا الرأس تحديدًا
            weight_slice = attention_layer.weight.data[start:end, :]
            grad_slice = attention_layer.weight.grad[start:end, :]
            importance = (weight_slice * grad_slice).abs().sum()
            head_importance_scores.append(importance.item())

        return torch.tensor(head_importance_scores)

    def prune_least_important_heads(self, layer_idx: int, n_heads_to_remove: int):
        """إزالة الرؤوس الأقل أهمية بالكامل (تقليم بنيوي حقيقي)."""
        importance = self.compute_head_importance(layer_idx)
        heads_to_remove = torch.topk(
            importance, n_heads_to_remove, largest=False
        ).indices.tolist()

        attention_layer = self.model.get_attention_layer(layer_idx)
        attention_layer.remove_heads(heads_to_remove)  # دالة مفاهيمية
        return self.model

    def post_pruning_recovery(self, recovery_dataset, lora_rank: int = 16):
        """
        خطوة ضرورية دائمًا بعد التقليم البنيوي: إعادة ضبط سريع
        وخفيف التكلفة (عبر LoRA - الجزء السابع) لاستعادة أكبر قدر
        ممكن من الأداء المفقود جراء إزالة المكوّنات البنيوية.
        """
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
        )
        recoverable_model = get_peft_model(self.model, lora_config)
        # تدريب LoRA قصير (عادة 1-2% فقط من حجم بيانات CPT الأصلية كافٍ)
        return recoverable_model

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
import torch

W = torch.tensor([
    [0.10,  0.90, -0.20, 0.05],
    [0.70, -0.30,  0.40, 0.02],
    [-0.60, 0.20,  0.15, 0.80],
])
act_norms = torch.tensor([2.0, 0.5, 1.5, 0.1])

# --- Magnitude: نحتفظ بأكبر قيمتين مطلقًا في كل صف ---
def magnitude_prune(W, keep=2):
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = W[r].abs().topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

# --- Wanda: |W| × ||X|| ---
def wanda_prune(W, act_norms, keep=2):
    importance = W.abs() * act_norms.unsqueeze(0)
    out = torch.zeros_like(W)
    for r in range(W.shape[0]):
        idx = importance[r].topk(keep).indices
        out[r, idx] = W[r, idx]
    return out

print("Magnitude:\n", magnitude_prune(W))
print("Wanda:\n", wanda_prune(W, act_norms))

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-27-ch27-pruning.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
