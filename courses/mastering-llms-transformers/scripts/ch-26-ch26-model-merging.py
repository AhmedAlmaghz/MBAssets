#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — دمج النماذج: Model Merging
Mastering Transformers and Large Language Models — Model Merging

Book slug : mastering-llms-transformers
Chapter   : ch-26-ch26-model-merging (lesson 26)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-26-ch26-model-merging.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — دمج النماذج: Model Merging
Mastering Transformers and Large Language Models — Model Merging

Book slug : mastering-llms-transformers
Chapter   : ch-26-ch26-model-merging (lesson 26)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-26-ch26-model-merging.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — دمج النماذج: Model Merging
Mastering Transformers and Large Language Models — Model Merging

Book slug : mastering-llms-transformers
Chapter   : ch-26-ch26-model-merging (lesson 26)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-1
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
# [002] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import copy

def uniform_model_soup(model_checkpoints: list) -> dict:
    """
    Model Soup بأبسط صوره (Wortsman et al., 2022): متوسط حسابي بسيط
    لأوزان عدة نماذج بنفس المعمارية تمامًا - عادة نسخ من نفس النموذج
    مُدرَّبة بـ hyperparameters مختلفة قليلاً (بذور عشوائية، LR، إلخ).
    النتيجة غالبًا أفضل من أي نموذج فردي، وأحيانًا أفضل من أفضل نموذج
    في المجموعة - بدون أي تكلفة تدريب إضافية!
    """
    averaged_state_dict = copy.deepcopy(model_checkpoints[0])
    n_models = len(model_checkpoints)

    for key in averaged_state_dict.keys():
        # جمع نفس الوزن عبر كل النماذج ثم القسمة على العدد
        stacked = torch.stack([ckpt[key].float() for ckpt in model_checkpoints])
        averaged_state_dict[key] = stacked.mean(dim=0).to(model_checkpoints[0][key].dtype)

    return averaged_state_dict

def greedy_model_soup(model_checkpoints: list, validation_fn, model_template) -> dict:
    """
    Greedy Soup (أكثر فعالية من Uniform Soup): ترتيب النماذج حسب أدائها
    الفردي على مجموعة تحقق، ثم إضافتها واحدًا تلو الآخر للمزيج - مع
    الاحتفاظ فقط بالإضافات التي تُحسِّن (أو لا تُضعِف) أداء المزيج الحالي.
    """
    # الخطوة 1: تقييم كل نموذج منفردًا وترتيبها تنازليًا
    scored_models = []
    for ckpt in model_checkpoints:
        model_template.load_state_dict(ckpt)
        score = validation_fn(model_template)
        scored_models.append((ckpt, score))
    scored_models.sort(key=lambda x: x[1], reverse=True)

    # الخطوة 2: البدء بأفضل نموذج، ثم إضافة الباقي بالتتابع
    current_soup = copy.deepcopy(scored_models[0][0])
    model_template.load_state_dict(current_soup)
    current_best_score = validation_fn(model_template)
    n_ingredients = 1

    for ckpt, _ in scored_models[1:]:
        # تجربة إضافة هذا النموذج للمزيج الحالي (متوسط مُرجَّح بعدد المكوّنات)
        candidate_soup = {}
        for key in current_soup.keys():
            candidate_soup[key] = (
                current_soup[key].float() * n_ingredients + ckpt[key].float()
            ) / (n_ingredients + 1)
            candidate_soup[key] = candidate_soup[key].to(current_soup[key].dtype)

        model_template.load_state_dict(candidate_soup)
        candidate_score = validation_fn(model_template)

        if candidate_score >= current_best_score:
            current_soup = candidate_soup
            current_best_score = candidate_score
            n_ingredients += 1
            print(f"✅ تمت إضافة نموذج - الدرجة الجديدة: {candidate_score:.4f}")
        else:
            print(f"❌ تم تجاهل نموذج - لم يُحسِّن الأداء ({candidate_score:.4f})")

    return current_soup

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class TaskArithmetic:
    """
    تطبيق Task Vectors (Ilharco et al., 2022): استخراج ودمج "متجهات
    المهام" للسماح بدمج مرن للقدرات، أو حتى إزالة قدرات غير مرغوبة
    (مثل: إزالة ميل للسمية عبر "طرح" متجه مهمة سلبي).
    """
    def __init__(self, base_model_state_dict: dict):
        self.base_state_dict = base_model_state_dict

    def extract_task_vector(self, finetuned_state_dict: dict) -> dict:
        """استخراج متجه المهمة = الفرق بين النموذج المُضبَّط والأساسي."""
        task_vector = {}
        for key in self.base_state_dict.keys():
            task_vector[key] = (
                finetuned_state_dict[key].float() - self.base_state_dict[key].float()
            )
        return task_vector

    def apply_task_vectors(self, task_vectors_with_weights: list) -> dict:
        """
        دمج عدة متجهات مهام بأوزان (λ) مختلفة فوق النموذج الأساسي.
        task_vectors_with_weights: قائمة من (task_vector_dict, lambda_weight)
        """
        merged_state_dict = {}
        for key in self.base_state_dict.keys():
            combined_delta = sum(
                tv[key] * weight for tv, weight in task_vectors_with_weights
            )
            merged_state_dict[key] = (
                self.base_state_dict[key].float() + combined_delta
            ).to(self.base_state_dict[key].dtype)
        return merged_state_dict

    def negate_task_vector(self, task_vector: dict, strength: float = 1.0) -> dict:
        """
        إزالة مهارة/سلوك: طرح متجه المهمة بدل جمعه. مفيد لإزالة
        سلوكيات غير مرغوبة (مثلًا: متجه مهمة مُستخرَج من نموذج
        مُدرَّب عمدًا على استجابات سامة، ثم طرحه لتقليل السمية).
        """
        return self.apply_task_vectors([(task_vector, -strength)])

# --- مثال استخدام: دمج قدرة برمجية وقدرة رياضية في نموذج عام ---
arithmetic = TaskArithmetic(base_model_state_dict=base_llama_weights)

code_task_vector = arithmetic.extract_task_vector(code_finetuned_weights)
math_task_vector = arithmetic.extract_task_vector(math_finetuned_weights)

# دمج القدرتين معًا بأوزان متساوية (λ=0.5 لكل منهما شائع كقيمة ابتدائية)
merged_weights = arithmetic.apply_task_vectors([
    (code_task_vector, 0.5),
    (math_task_vector, 0.5),
])

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TIESMerging:
    """
    تطبيق كامل لخوارزمية TIES-Merging (Yadav et al., 2023):
    حل مشكلة تداخل الإشارات بين عدة Task Vectors عبر ثلاث خطوات
    منهجية: التقليم (Trim)، انتخاب الإشارة (Elect Sign)، والدمج
    الانتقائي (Disjoint Merge).
    """
    def __init__(self, base_state_dict: dict, trim_density: float = 0.2):
        self.base_state_dict = base_state_dict
        self.trim_density = trim_density  # نسبة القيم المُحتفَظ بها (الأعلى قيمة)

    def trim_task_vector(self, task_vector: dict) -> dict:
        """
        الخطوة 1 - Trim: الاحتفاظ فقط بأعلى k% من القيم المطلقة لكل
        متجه مهمة (القيم الأكبر تأثيرًا)، وتصفير الباقي. هذا يُزيل
        "الضجيج" الصغير الذي يُسبِّب معظم التداخل غير المفيد.
        """
        trimmed = {}
        for key, tensor in task_vector.items():
            flat = tensor.flatten().abs()
            k = int(len(flat) * self.trim_density)
            if k == 0:
                trimmed[key] = torch.zeros_like(tensor)
                continue
            threshold = torch.topk(flat, k).values.min()
            mask = tensor.abs() >= threshold
            trimmed[key] = tensor * mask
        return trimmed

    def elect_sign(self, trimmed_task_vectors: list) -> dict:
        """
        الخطوة 2 - Elect Sign: لكل وزن، تحديد "الإشارة الغالبة" عبر
        كل متجهات المهام (بناءً على مجموع القيم الموجبة مقابل السالبة
        المُرجَّح بالحجم)، ثم الاحتفاظ لاحقًا فقط بالتحديثات المتوافقة
        مع هذه الإشارة الغالبة.
        """
        elected_signs = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign_sum = stacked.sum(dim=0)
            elected_signs[key] = torch.sign(sign_sum)
        return elected_signs

    def disjoint_merge(self, trimmed_task_vectors: list,
                        elected_signs: dict) -> dict:
        """
        الخطوة 3 - Disjoint Merge: لكل وزن، نأخذ متوسط القيم فقط من
        متجهات المهام التي "تتفق" إشارتها مع الإشارة الغالبة المُنتخَبة،
        متجاهلين تمامًا القيم المتعارضة (بدل إلغائها بالجمع كما في
        Task Arithmetic البسيط).
        """
        merged_delta = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign = elected_signs[key]

            # قناع القيم المتوافقة مع الإشارة الغالبة فقط
            agreement_mask = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
            agreement_count = agreement_mask.sum(dim=0).clamp(min=1)

            masked_values = stacked * agreement_mask
            merged_delta[key] = masked_values.sum(dim=0) / agreement_count

        return merged_delta

    def merge(self, task_vectors: list, scaling_factor: float = 1.0) -> dict:
        """الخط الكامل: Trim → Elect Sign → Disjoint Merge → إضافة للأساس."""
        trimmed_vectors = [self.trim_task_vector(tv) for tv in task_vectors]
        elected_signs = self.elect_sign(trimmed_vectors)
        merged_delta = self.disjoint_merge(trimmed_vectors, elected_signs)

        final_state_dict = {}
        for key in self.base_state_dict.keys():
            final_state_dict[key] = (
                self.base_state_dict[key].float()
                + scaling_factor * merged_delta[key]
            ).to(self.base_state_dict[key].dtype)
        return final_state_dict

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify_task_vector(task_vector: dict, drop_rate: float = 0.9) -> dict:
    """
    DARE (Yu et al., 2023): إسقاط عشوائي (Random Drop) لنسبة كبيرة جدًا
    من قيم متجه المهمة (حتى 90-99%!)، مع إعادة قياس (Rescale) القيم
    المتبقية للتعويض عن الكتلة المفقودة. النتيجة المُدهشة: الأداء
    يبقى شبه ثابت حتى مع drop_rate=0.9، مما يُظهر أن معظم التحديثات
    في Task Vector "زائدة عن الحاجة" (Redundant) ويمكن حذفها بأمان،
    وهذا يُقلِّل تداخل الإشارات بشكل كبير عند الدمج اللاحق.
    """
    sparsified = {}
    for key, tensor in task_vector.items():
        mask = torch.rand_like(tensor) > drop_rate  # الاحتفاظ بنسبة (1-drop_rate)
        rescale_factor = 1.0 / (1.0 - drop_rate)  # إعادة القياس للتعويض
        sparsified[key] = tensor * mask * rescale_factor
    return sparsified

# DARE غالبًا تُستخدم كخطوة تمهيدية قبل TIES أو Task Arithmetic
# (DARE-TIES: تطبيق DARE أولاً لتخفيف الكثافة، ثم TIES للدمج النهائي)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(weight_a: torch.Tensor, weight_b: torch.Tensor,
                 t: float = 0.5, epsilon: float = 1e-8) -> torch.Tensor:
    """
    الاستيفاء الخطي الكروي (SLERP) بين وزنين - أكثر دقة هندسيًا من
    المتوسط الخطي البسيط lerp = (1-t)*a + t*b عند دمج نموذجين فقط،
    لأنه يحافظ على "القاعدة" (norm) الهندسية للمتجهات بدل تقليصها
    بشكل مصطنع كما يحدث أحيانًا مع المتوسط الخطي المباشر.
    """
    a_flat = weight_a.flatten().float()
    b_flat = weight_b.flatten().float()

    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # زاوية التشابه بين المتجهين
    dot_product = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega) + epsilon

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متماثلة)، استخدم lerp عادي كبديل آمن
    if sin_omega.abs() < 1e-6:
        result_flat = (1 - t) * a_flat + t * b_flat
    else:
        coeff_a = torch.sin((1 - t) * omega) / sin_omega
        coeff_b = torch.sin(t * omega) / sin_omega
        result_flat = coeff_a * a_flat + coeff_b * b_flat

    return result_flat.reshape(weight_a.shape).to(weight_a.dtype)

def slerp_merge_full_models(state_dict_a: dict, state_dict_b: dict,
                             t: float = 0.5) -> dict:
    """تطبيق SLERP على كل أوزان النموذج طبقة بطبقة."""
    merged = {}
    for key in state_dict_a.keys():
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def frankenmerge_layers(model_a_layers: list, model_b_layers: list,
                         layer_recipe: list) -> list:
    """
    Frankenmerging: بناء نموذج جديد بتجميع طبقات كاملة من نماذج مختلفة
    (مشابهة البنية) حسب "وصفة" مُحدَّدة يدويًا أو تجريبيًا. هذا النهج
    استخدمته نماذج مثل Goliath-120B (دمج طبقات من نسختين من Llama 2 70B)
    لإنتاج نموذج أكبر وأحيانًا أقوى دون أي تدريب إضافي على الإطلاق.

    layer_recipe: قائمة من tuples (المصدر, رقم_الطبقة)
    مثال: [("a", 0), ("a", 1), ("b", 0), ("a", 2), ("b", 1), ...]
    """
    new_model_layers = []
    for source, layer_idx in layer_recipe:
        if source == "a":
            new_model_layers.append(copy.deepcopy(model_a_layers[layer_idx]))
        elif source == "b":
            new_model_layers.append(copy.deepcopy(model_b_layers[layer_idx]))
    return new_model_layers

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch

# أوزان أساس افتراضية (3 قيم فقط للتوضيح)
theta_base = torch.tensor([1.0, -0.5, 0.2])

# متجه مهمة A (برمجة) ومتجه مهمة B (رياضيات)
tau_a = torch.tensor([0.4, 0.3, -0.1])
tau_b = torch.tensor([0.2, -0.4, 0.3])

# 1) Task Arithmetic البسيط: جمع مباشر
tau_sum = tau_a + tau_b
print("tau_sum       =", tau_sum.tolist())        # [0.6, -0.1, 0.2]
print("theta_sum     =", (theta_base + tau_sum).tolist())

# 2) TIES مبسّط: تقليم القيم الصغيرة ثم انتخاب الإشارة الغالبة
def trim(tau, keep=2):
    threshold = tau.abs().topk(keep).values.min()
    return torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))

ta, tb = trim(tau_a), trim(tau_b)
print("trimmed A     =", ta.tolist())             # [0.4, 0.3, 0.0]
print("trimmed B     =", tb.tolist())             # [0.0, -0.4, 0.3]

# الإشارة الغالبة لكل موضع: مجموع القيم الموقّعة
elected = torch.sign(ta + tb)
print("elected sign  =", elected.tolist())        # [1, -1, 1]

# الدمج المنفصل: متوسط القيم المتوافقة فقط مع الإشارة الغالبة
mask_a = (torch.sign(ta) == elected) & (ta != 0)
mask_b = (torch.sign(tb) == elected) & (tb != 0)
tau_ties = torch.where(mask_a, ta, torch.zeros_like(ta)) + \
           torch.where(mask_b, tb, torch.zeros_like(tb))
print("tau_ties      =", tau_ties.tolist())       # [0.4, -0.4, 0.3]
print("theta_ties    =", (theta_base + tau_ties).tolist())

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import copy

def uniform_model_soup(model_checkpoints: list) -> dict:
    """
    Model Soup بأبسط صوره (Wortsman et al., 2022): متوسط حسابي بسيط
    لأوزان عدة نماذج بنفس المعمارية تمامًا - عادة نسخ من نفس النموذج
    مُدرَّبة بـ hyperparameters مختلفة قليلاً (بذور عشوائية، LR، إلخ).
    النتيجة غالبًا أفضل من أي نموذج فردي، وأحيانًا أفضل من أفضل نموذج
    في المجموعة - بدون أي تكلفة تدريب إضافية!
    """
    averaged_state_dict = copy.deepcopy(model_checkpoints[0])
    n_models = len(model_checkpoints)

    for key in averaged_state_dict.keys():
        # جمع نفس الوزن عبر كل النماذج ثم القسمة على العدد
        stacked = torch.stack([ckpt[key].float() for ckpt in model_checkpoints])
        averaged_state_dict[key] = stacked.mean(dim=0).to(model_checkpoints[0][key].dtype)

    return averaged_state_dict

def greedy_model_soup(model_checkpoints: list, validation_fn, model_template) -> dict:
    """
    Greedy Soup (أكثر فعالية من Uniform Soup): ترتيب النماذج حسب أدائها
    الفردي على مجموعة تحقق، ثم إضافتها واحدًا تلو الآخر للمزيج - مع
    الاحتفاظ فقط بالإضافات التي تُحسِّن (أو لا تُضعِف) أداء المزيج الحالي.
    """
    # الخطوة 1: تقييم كل نموذج منفردًا وترتيبها تنازليًا
    scored_models = []
    for ckpt in model_checkpoints:
        model_template.load_state_dict(ckpt)
        score = validation_fn(model_template)
        scored_models.append((ckpt, score))
    scored_models.sort(key=lambda x: x[1], reverse=True)

    # الخطوة 2: البدء بأفضل نموذج، ثم إضافة الباقي بالتتابع
    current_soup = copy.deepcopy(scored_models[0][0])
    model_template.load_state_dict(current_soup)
    current_best_score = validation_fn(model_template)
    n_ingredients = 1

    for ckpt, _ in scored_models[1:]:
        # تجربة إضافة هذا النموذج للمزيج الحالي (متوسط مُرجَّح بعدد المكوّنات)
        candidate_soup = {}
        for key in current_soup.keys():
            candidate_soup[key] = (
                current_soup[key].float() * n_ingredients + ckpt[key].float()
            ) / (n_ingredients + 1)
            candidate_soup[key] = candidate_soup[key].to(current_soup[key].dtype)

        model_template.load_state_dict(candidate_soup)
        candidate_score = validation_fn(model_template)

        if candidate_score >= current_best_score:
            current_soup = candidate_soup
            current_best_score = candidate_score
            n_ingredients += 1
            print(f"✅ تمت إضافة نموذج - الدرجة الجديدة: {candidate_score:.4f}")
        else:
            print(f"❌ تم تجاهل نموذج - لم يُحسِّن الأداء ({candidate_score:.4f})")

    return current_soup

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class TaskArithmetic:
    """
    تطبيق Task Vectors (Ilharco et al., 2022): استخراج ودمج "متجهات
    المهام" للسماح بدمج مرن للقدرات، أو حتى إزالة قدرات غير مرغوبة
    (مثل: إزالة ميل للسمية عبر "طرح" متجه مهمة سلبي).
    """
    def __init__(self, base_model_state_dict: dict):
        self.base_state_dict = base_model_state_dict

    def extract_task_vector(self, finetuned_state_dict: dict) -> dict:
        """استخراج متجه المهمة = الفرق بين النموذج المُضبَّط والأساسي."""
        task_vector = {}
        for key in self.base_state_dict.keys():
            task_vector[key] = (
                finetuned_state_dict[key].float() - self.base_state_dict[key].float()
            )
        return task_vector

    def apply_task_vectors(self, task_vectors_with_weights: list) -> dict:
        """
        دمج عدة متجهات مهام بأوزان (λ) مختلفة فوق النموذج الأساسي.
        task_vectors_with_weights: قائمة من (task_vector_dict, lambda_weight)
        """
        merged_state_dict = {}
        for key in self.base_state_dict.keys():
            combined_delta = sum(
                tv[key] * weight for tv, weight in task_vectors_with_weights
            )
            merged_state_dict[key] = (
                self.base_state_dict[key].float() + combined_delta
            ).to(self.base_state_dict[key].dtype)
        return merged_state_dict

    def negate_task_vector(self, task_vector: dict, strength: float = 1.0) -> dict:
        """
        إزالة مهارة/سلوك: طرح متجه المهمة بدل جمعه. مفيد لإزالة
        سلوكيات غير مرغوبة (مثلًا: متجه مهمة مُستخرَج من نموذج
        مُدرَّب عمدًا على استجابات سامة، ثم طرحه لتقليل السمية).
        """
        return self.apply_task_vectors([(task_vector, -strength)])

# --- مثال استخدام: دمج قدرة برمجية وقدرة رياضية في نموذج عام ---
arithmetic = TaskArithmetic(base_model_state_dict=base_llama_weights)

code_task_vector = arithmetic.extract_task_vector(code_finetuned_weights)
math_task_vector = arithmetic.extract_task_vector(math_finetuned_weights)

# دمج القدرتين معًا بأوزان متساوية (λ=0.5 لكل منهما شائع كقيمة ابتدائية)
merged_weights = arithmetic.apply_task_vectors([
    (code_task_vector, 0.5),
    (math_task_vector, 0.5),
])

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TIESMerging:
    """
    تطبيق كامل لخوارزمية TIES-Merging (Yadav et al., 2023):
    حل مشكلة تداخل الإشارات بين عدة Task Vectors عبر ثلاث خطوات
    منهجية: التقليم (Trim)، انتخاب الإشارة (Elect Sign)، والدمج
    الانتقائي (Disjoint Merge).
    """
    def __init__(self, base_state_dict: dict, trim_density: float = 0.2):
        self.base_state_dict = base_state_dict
        self.trim_density = trim_density  # نسبة القيم المُحتفَظ بها (الأعلى قيمة)

    def trim_task_vector(self, task_vector: dict) -> dict:
        """
        الخطوة 1 - Trim: الاحتفاظ فقط بأعلى k% من القيم المطلقة لكل
        متجه مهمة (القيم الأكبر تأثيرًا)، وتصفير الباقي. هذا يُزيل
        "الضجيج" الصغير الذي يُسبِّب معظم التداخل غير المفيد.
        """
        trimmed = {}
        for key, tensor in task_vector.items():
            flat = tensor.flatten().abs()
            k = int(len(flat) * self.trim_density)
            if k == 0:
                trimmed[key] = torch.zeros_like(tensor)
                continue
            threshold = torch.topk(flat, k).values.min()
            mask = tensor.abs() >= threshold
            trimmed[key] = tensor * mask
        return trimmed

    def elect_sign(self, trimmed_task_vectors: list) -> dict:
        """
        الخطوة 2 - Elect Sign: لكل وزن، تحديد "الإشارة الغالبة" عبر
        كل متجهات المهام (بناءً على مجموع القيم الموجبة مقابل السالبة
        المُرجَّح بالحجم)، ثم الاحتفاظ لاحقًا فقط بالتحديثات المتوافقة
        مع هذه الإشارة الغالبة.
        """
        elected_signs = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign_sum = stacked.sum(dim=0)
            elected_signs[key] = torch.sign(sign_sum)
        return elected_signs

    def disjoint_merge(self, trimmed_task_vectors: list,
                        elected_signs: dict) -> dict:
        """
        الخطوة 3 - Disjoint Merge: لكل وزن، نأخذ متوسط القيم فقط من
        متجهات المهام التي "تتفق" إشارتها مع الإشارة الغالبة المُنتخَبة،
        متجاهلين تمامًا القيم المتعارضة (بدل إلغائها بالجمع كما في
        Task Arithmetic البسيط).
        """
        merged_delta = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign = elected_signs[key]

            # قناع القيم المتوافقة مع الإشارة الغالبة فقط
            agreement_mask = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
            agreement_count = agreement_mask.sum(dim=0).clamp(min=1)

            masked_values = stacked * agreement_mask
            merged_delta[key] = masked_values.sum(dim=0) / agreement_count

        return merged_delta

    def merge(self, task_vectors: list, scaling_factor: float = 1.0) -> dict:
        """الخط الكامل: Trim → Elect Sign → Disjoint Merge → إضافة للأساس."""
        trimmed_vectors = [self.trim_task_vector(tv) for tv in task_vectors]
        elected_signs = self.elect_sign(trimmed_vectors)
        merged_delta = self.disjoint_merge(trimmed_vectors, elected_signs)

        final_state_dict = {}
        for key in self.base_state_dict.keys():
            final_state_dict[key] = (
                self.base_state_dict[key].float()
                + scaling_factor * merged_delta[key]
            ).to(self.base_state_dict[key].dtype)
        return final_state_dict

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify_task_vector(task_vector: dict, drop_rate: float = 0.9) -> dict:
    """
    DARE (Yu et al., 2023): إسقاط عشوائي (Random Drop) لنسبة كبيرة جدًا
    من قيم متجه المهمة (حتى 90-99%!)، مع إعادة قياس (Rescale) القيم
    المتبقية للتعويض عن الكتلة المفقودة. النتيجة المُدهشة: الأداء
    يبقى شبه ثابت حتى مع drop_rate=0.9، مما يُظهر أن معظم التحديثات
    في Task Vector "زائدة عن الحاجة" (Redundant) ويمكن حذفها بأمان،
    وهذا يُقلِّل تداخل الإشارات بشكل كبير عند الدمج اللاحق.
    """
    sparsified = {}
    for key, tensor in task_vector.items():
        mask = torch.rand_like(tensor) > drop_rate  # الاحتفاظ بنسبة (1-drop_rate)
        rescale_factor = 1.0 / (1.0 - drop_rate)  # إعادة القياس للتعويض
        sparsified[key] = tensor * mask * rescale_factor
    return sparsified

# DARE غالبًا تُستخدم كخطوة تمهيدية قبل TIES أو Task Arithmetic
# (DARE-TIES: تطبيق DARE أولاً لتخفيف الكثافة، ثم TIES للدمج النهائي)

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(weight_a: torch.Tensor, weight_b: torch.Tensor,
                 t: float = 0.5, epsilon: float = 1e-8) -> torch.Tensor:
    """
    الاستيفاء الخطي الكروي (SLERP) بين وزنين - أكثر دقة هندسيًا من
    المتوسط الخطي البسيط lerp = (1-t)*a + t*b عند دمج نموذجين فقط،
    لأنه يحافظ على "القاعدة" (norm) الهندسية للمتجهات بدل تقليصها
    بشكل مصطنع كما يحدث أحيانًا مع المتوسط الخطي المباشر.
    """
    a_flat = weight_a.flatten().float()
    b_flat = weight_b.flatten().float()

    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # زاوية التشابه بين المتجهين
    dot_product = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega) + epsilon

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متماثلة)، استخدم lerp عادي كبديل آمن
    if sin_omega.abs() < 1e-6:
        result_flat = (1 - t) * a_flat + t * b_flat
    else:
        coeff_a = torch.sin((1 - t) * omega) / sin_omega
        coeff_b = torch.sin(t * omega) / sin_omega
        result_flat = coeff_a * a_flat + coeff_b * b_flat

    return result_flat.reshape(weight_a.shape).to(weight_a.dtype)

def slerp_merge_full_models(state_dict_a: dict, state_dict_b: dict,
                             t: float = 0.5) -> dict:
    """تطبيق SLERP على كل أوزان النموذج طبقة بطبقة."""
    merged = {}
    for key in state_dict_a.keys():
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def frankenmerge_layers(model_a_layers: list, model_b_layers: list,
                         layer_recipe: list) -> list:
    """
    Frankenmerging: بناء نموذج جديد بتجميع طبقات كاملة من نماذج مختلفة
    (مشابهة البنية) حسب "وصفة" مُحدَّدة يدويًا أو تجريبيًا. هذا النهج
    استخدمته نماذج مثل Goliath-120B (دمج طبقات من نسختين من Llama 2 70B)
    لإنتاج نموذج أكبر وأحيانًا أقوى دون أي تدريب إضافي على الإطلاق.

    layer_recipe: قائمة من tuples (المصدر, رقم_الطبقة)
    مثال: [("a", 0), ("a", 1), ("b", 0), ("a", 2), ("b", 1), ...]
    """
    new_model_layers = []
    for source, layer_idx in layer_recipe:
        if source == "a":
            new_model_layers.append(copy.deepcopy(model_a_layers[layer_idx]))
        elif source == "b":
            new_model_layers.append(copy.deepcopy(model_b_layers[layer_idx]))
    return new_model_layers

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch

# أوزان أساس افتراضية (3 قيم فقط للتوضيح)
theta_base = torch.tensor([1.0, -0.5, 0.2])

# متجه مهمة A (برمجة) ومتجه مهمة B (رياضيات)
tau_a = torch.tensor([0.4, 0.3, -0.1])
tau_b = torch.tensor([0.2, -0.4, 0.3])

# 1) Task Arithmetic البسيط: جمع مباشر
tau_sum = tau_a + tau_b
print("tau_sum       =", tau_sum.tolist())        # [0.6, -0.1, 0.2]
print("theta_sum     =", (theta_base + tau_sum).tolist())

# 2) TIES مبسّط: تقليم القيم الصغيرة ثم انتخاب الإشارة الغالبة
def trim(tau, keep=2):
    threshold = tau.abs().topk(keep).values.min()
    return torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))

ta, tb = trim(tau_a), trim(tau_b)
print("trimmed A     =", ta.tolist())             # [0.4, 0.3, 0.0]
print("trimmed B     =", tb.tolist())             # [0.0, -0.4, 0.3]

# الإشارة الغالبة لكل موضع: مجموع القيم الموقّعة
elected = torch.sign(ta + tb)
print("elected sign  =", elected.tolist())        # [1, -1, 1]

# الدمج المنفصل: متوسط القيم المتوافقة فقط مع الإشارة الغالبة
mask_a = (torch.sign(ta) == elected) & (ta != 0)
mask_b = (torch.sign(tb) == elected) & (tb != 0)
tau_ties = torch.where(mask_a, ta, torch.zeros_like(ta)) + \
           torch.where(mask_b, tb, torch.zeros_like(tb))
print("tau_ties      =", tau_ties.tolist())       # [0.4, -0.4, 0.3]
print("theta_ties    =", (theta_base + tau_ties).tolist())

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — دمج النماذج: Model Merging
Mastering Transformers and Large Language Models — Model Merging

Book slug : mastering-llms-transformers
Chapter   : ch-26-ch26-model-merging (lesson 26)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-1
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
# [002] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import copy

def uniform_model_soup(model_checkpoints: list) -> dict:
    """
    Model Soup بأبسط صوره (Wortsman et al., 2022): متوسط حسابي بسيط
    لأوزان عدة نماذج بنفس المعمارية تمامًا - عادة نسخ من نفس النموذج
    مُدرَّبة بـ hyperparameters مختلفة قليلاً (بذور عشوائية، LR، إلخ).
    النتيجة غالبًا أفضل من أي نموذج فردي، وأحيانًا أفضل من أفضل نموذج
    في المجموعة - بدون أي تكلفة تدريب إضافية!
    """
    averaged_state_dict = copy.deepcopy(model_checkpoints[0])
    n_models = len(model_checkpoints)

    for key in averaged_state_dict.keys():
        # جمع نفس الوزن عبر كل النماذج ثم القسمة على العدد
        stacked = torch.stack([ckpt[key].float() for ckpt in model_checkpoints])
        averaged_state_dict[key] = stacked.mean(dim=0).to(model_checkpoints[0][key].dtype)

    return averaged_state_dict

def greedy_model_soup(model_checkpoints: list, validation_fn, model_template) -> dict:
    """
    Greedy Soup (أكثر فعالية من Uniform Soup): ترتيب النماذج حسب أدائها
    الفردي على مجموعة تحقق، ثم إضافتها واحدًا تلو الآخر للمزيج - مع
    الاحتفاظ فقط بالإضافات التي تُحسِّن (أو لا تُضعِف) أداء المزيج الحالي.
    """
    # الخطوة 1: تقييم كل نموذج منفردًا وترتيبها تنازليًا
    scored_models = []
    for ckpt in model_checkpoints:
        model_template.load_state_dict(ckpt)
        score = validation_fn(model_template)
        scored_models.append((ckpt, score))
    scored_models.sort(key=lambda x: x[1], reverse=True)

    # الخطوة 2: البدء بأفضل نموذج، ثم إضافة الباقي بالتتابع
    current_soup = copy.deepcopy(scored_models[0][0])
    model_template.load_state_dict(current_soup)
    current_best_score = validation_fn(model_template)
    n_ingredients = 1

    for ckpt, _ in scored_models[1:]:
        # تجربة إضافة هذا النموذج للمزيج الحالي (متوسط مُرجَّح بعدد المكوّنات)
        candidate_soup = {}
        for key in current_soup.keys():
            candidate_soup[key] = (
                current_soup[key].float() * n_ingredients + ckpt[key].float()
            ) / (n_ingredients + 1)
            candidate_soup[key] = candidate_soup[key].to(current_soup[key].dtype)

        model_template.load_state_dict(candidate_soup)
        candidate_score = validation_fn(model_template)

        if candidate_score >= current_best_score:
            current_soup = candidate_soup
            current_best_score = candidate_score
            n_ingredients += 1
            print(f"✅ تمت إضافة نموذج - الدرجة الجديدة: {candidate_score:.4f}")
        else:
            print(f"❌ تم تجاهل نموذج - لم يُحسِّن الأداء ({candidate_score:.4f})")

    return current_soup

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class TaskArithmetic:
    """
    تطبيق Task Vectors (Ilharco et al., 2022): استخراج ودمج "متجهات
    المهام" للسماح بدمج مرن للقدرات، أو حتى إزالة قدرات غير مرغوبة
    (مثل: إزالة ميل للسمية عبر "طرح" متجه مهمة سلبي).
    """
    def __init__(self, base_model_state_dict: dict):
        self.base_state_dict = base_model_state_dict

    def extract_task_vector(self, finetuned_state_dict: dict) -> dict:
        """استخراج متجه المهمة = الفرق بين النموذج المُضبَّط والأساسي."""
        task_vector = {}
        for key in self.base_state_dict.keys():
            task_vector[key] = (
                finetuned_state_dict[key].float() - self.base_state_dict[key].float()
            )
        return task_vector

    def apply_task_vectors(self, task_vectors_with_weights: list) -> dict:
        """
        دمج عدة متجهات مهام بأوزان (λ) مختلفة فوق النموذج الأساسي.
        task_vectors_with_weights: قائمة من (task_vector_dict, lambda_weight)
        """
        merged_state_dict = {}
        for key in self.base_state_dict.keys():
            combined_delta = sum(
                tv[key] * weight for tv, weight in task_vectors_with_weights
            )
            merged_state_dict[key] = (
                self.base_state_dict[key].float() + combined_delta
            ).to(self.base_state_dict[key].dtype)
        return merged_state_dict

    def negate_task_vector(self, task_vector: dict, strength: float = 1.0) -> dict:
        """
        إزالة مهارة/سلوك: طرح متجه المهمة بدل جمعه. مفيد لإزالة
        سلوكيات غير مرغوبة (مثلًا: متجه مهمة مُستخرَج من نموذج
        مُدرَّب عمدًا على استجابات سامة، ثم طرحه لتقليل السمية).
        """
        return self.apply_task_vectors([(task_vector, -strength)])

# --- مثال استخدام: دمج قدرة برمجية وقدرة رياضية في نموذج عام ---
arithmetic = TaskArithmetic(base_model_state_dict=base_llama_weights)

code_task_vector = arithmetic.extract_task_vector(code_finetuned_weights)
math_task_vector = arithmetic.extract_task_vector(math_finetuned_weights)

# دمج القدرتين معًا بأوزان متساوية (λ=0.5 لكل منهما شائع كقيمة ابتدائية)
merged_weights = arithmetic.apply_task_vectors([
    (code_task_vector, 0.5),
    (math_task_vector, 0.5),
])

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TIESMerging:
    """
    تطبيق كامل لخوارزمية TIES-Merging (Yadav et al., 2023):
    حل مشكلة تداخل الإشارات بين عدة Task Vectors عبر ثلاث خطوات
    منهجية: التقليم (Trim)، انتخاب الإشارة (Elect Sign)، والدمج
    الانتقائي (Disjoint Merge).
    """
    def __init__(self, base_state_dict: dict, trim_density: float = 0.2):
        self.base_state_dict = base_state_dict
        self.trim_density = trim_density  # نسبة القيم المُحتفَظ بها (الأعلى قيمة)

    def trim_task_vector(self, task_vector: dict) -> dict:
        """
        الخطوة 1 - Trim: الاحتفاظ فقط بأعلى k% من القيم المطلقة لكل
        متجه مهمة (القيم الأكبر تأثيرًا)، وتصفير الباقي. هذا يُزيل
        "الضجيج" الصغير الذي يُسبِّب معظم التداخل غير المفيد.
        """
        trimmed = {}
        for key, tensor in task_vector.items():
            flat = tensor.flatten().abs()
            k = int(len(flat) * self.trim_density)
            if k == 0:
                trimmed[key] = torch.zeros_like(tensor)
                continue
            threshold = torch.topk(flat, k).values.min()
            mask = tensor.abs() >= threshold
            trimmed[key] = tensor * mask
        return trimmed

    def elect_sign(self, trimmed_task_vectors: list) -> dict:
        """
        الخطوة 2 - Elect Sign: لكل وزن، تحديد "الإشارة الغالبة" عبر
        كل متجهات المهام (بناءً على مجموع القيم الموجبة مقابل السالبة
        المُرجَّح بالحجم)، ثم الاحتفاظ لاحقًا فقط بالتحديثات المتوافقة
        مع هذه الإشارة الغالبة.
        """
        elected_signs = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign_sum = stacked.sum(dim=0)
            elected_signs[key] = torch.sign(sign_sum)
        return elected_signs

    def disjoint_merge(self, trimmed_task_vectors: list,
                        elected_signs: dict) -> dict:
        """
        الخطوة 3 - Disjoint Merge: لكل وزن، نأخذ متوسط القيم فقط من
        متجهات المهام التي "تتفق" إشارتها مع الإشارة الغالبة المُنتخَبة،
        متجاهلين تمامًا القيم المتعارضة (بدل إلغائها بالجمع كما في
        Task Arithmetic البسيط).
        """
        merged_delta = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign = elected_signs[key]

            # قناع القيم المتوافقة مع الإشارة الغالبة فقط
            agreement_mask = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
            agreement_count = agreement_mask.sum(dim=0).clamp(min=1)

            masked_values = stacked * agreement_mask
            merged_delta[key] = masked_values.sum(dim=0) / agreement_count

        return merged_delta

    def merge(self, task_vectors: list, scaling_factor: float = 1.0) -> dict:
        """الخط الكامل: Trim → Elect Sign → Disjoint Merge → إضافة للأساس."""
        trimmed_vectors = [self.trim_task_vector(tv) for tv in task_vectors]
        elected_signs = self.elect_sign(trimmed_vectors)
        merged_delta = self.disjoint_merge(trimmed_vectors, elected_signs)

        final_state_dict = {}
        for key in self.base_state_dict.keys():
            final_state_dict[key] = (
                self.base_state_dict[key].float()
                + scaling_factor * merged_delta[key]
            ).to(self.base_state_dict[key].dtype)
        return final_state_dict

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify_task_vector(task_vector: dict, drop_rate: float = 0.9) -> dict:
    """
    DARE (Yu et al., 2023): إسقاط عشوائي (Random Drop) لنسبة كبيرة جدًا
    من قيم متجه المهمة (حتى 90-99%!)، مع إعادة قياس (Rescale) القيم
    المتبقية للتعويض عن الكتلة المفقودة. النتيجة المُدهشة: الأداء
    يبقى شبه ثابت حتى مع drop_rate=0.9، مما يُظهر أن معظم التحديثات
    في Task Vector "زائدة عن الحاجة" (Redundant) ويمكن حذفها بأمان،
    وهذا يُقلِّل تداخل الإشارات بشكل كبير عند الدمج اللاحق.
    """
    sparsified = {}
    for key, tensor in task_vector.items():
        mask = torch.rand_like(tensor) > drop_rate  # الاحتفاظ بنسبة (1-drop_rate)
        rescale_factor = 1.0 / (1.0 - drop_rate)  # إعادة القياس للتعويض
        sparsified[key] = tensor * mask * rescale_factor
    return sparsified

# DARE غالبًا تُستخدم كخطوة تمهيدية قبل TIES أو Task Arithmetic
# (DARE-TIES: تطبيق DARE أولاً لتخفيف الكثافة، ثم TIES للدمج النهائي)

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(weight_a: torch.Tensor, weight_b: torch.Tensor,
                 t: float = 0.5, epsilon: float = 1e-8) -> torch.Tensor:
    """
    الاستيفاء الخطي الكروي (SLERP) بين وزنين - أكثر دقة هندسيًا من
    المتوسط الخطي البسيط lerp = (1-t)*a + t*b عند دمج نموذجين فقط،
    لأنه يحافظ على "القاعدة" (norm) الهندسية للمتجهات بدل تقليصها
    بشكل مصطنع كما يحدث أحيانًا مع المتوسط الخطي المباشر.
    """
    a_flat = weight_a.flatten().float()
    b_flat = weight_b.flatten().float()

    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # زاوية التشابه بين المتجهين
    dot_product = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega) + epsilon

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متماثلة)، استخدم lerp عادي كبديل آمن
    if sin_omega.abs() < 1e-6:
        result_flat = (1 - t) * a_flat + t * b_flat
    else:
        coeff_a = torch.sin((1 - t) * omega) / sin_omega
        coeff_b = torch.sin(t * omega) / sin_omega
        result_flat = coeff_a * a_flat + coeff_b * b_flat

    return result_flat.reshape(weight_a.shape).to(weight_a.dtype)

def slerp_merge_full_models(state_dict_a: dict, state_dict_b: dict,
                             t: float = 0.5) -> dict:
    """تطبيق SLERP على كل أوزان النموذج طبقة بطبقة."""
    merged = {}
    for key in state_dict_a.keys():
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [007] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def frankenmerge_layers(model_a_layers: list, model_b_layers: list,
                         layer_recipe: list) -> list:
    """
    Frankenmerging: بناء نموذج جديد بتجميع طبقات كاملة من نماذج مختلفة
    (مشابهة البنية) حسب "وصفة" مُحدَّدة يدويًا أو تجريبيًا. هذا النهج
    استخدمته نماذج مثل Goliath-120B (دمج طبقات من نسختين من Llama 2 70B)
    لإنتاج نموذج أكبر وأحيانًا أقوى دون أي تدريب إضافي على الإطلاق.

    layer_recipe: قائمة من tuples (المصدر, رقم_الطبقة)
    مثال: [("a", 0), ("a", 1), ("b", 0), ("a", 2), ("b", 1), ...]
    """
    new_model_layers = []
    for source, layer_idx in layer_recipe:
        if source == "a":
            new_model_layers.append(copy.deepcopy(model_a_layers[layer_idx]))
        elif source == "b":
            new_model_layers.append(copy.deepcopy(model_b_layers[layer_idx]))
    return new_model_layers

# ────────────────────────────────────────────────────────────────────────
# [008] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
import torch

# أوزان أساس افتراضية (3 قيم فقط للتوضيح)
theta_base = torch.tensor([1.0, -0.5, 0.2])

# متجه مهمة A (برمجة) ومتجه مهمة B (رياضيات)
tau_a = torch.tensor([0.4, 0.3, -0.1])
tau_b = torch.tensor([0.2, -0.4, 0.3])

# 1) Task Arithmetic البسيط: جمع مباشر
tau_sum = tau_a + tau_b
print("tau_sum       =", tau_sum.tolist())        # [0.6, -0.1, 0.2]
print("theta_sum     =", (theta_base + tau_sum).tolist())

# 2) TIES مبسّط: تقليم القيم الصغيرة ثم انتخاب الإشارة الغالبة
def trim(tau, keep=2):
    threshold = tau.abs().topk(keep).values.min()
    return torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))

ta, tb = trim(tau_a), trim(tau_b)
print("trimmed A     =", ta.tolist())             # [0.4, 0.3, 0.0]
print("trimmed B     =", tb.tolist())             # [0.0, -0.4, 0.3]

# الإشارة الغالبة لكل موضع: مجموع القيم الموقّعة
elected = torch.sign(ta + tb)
print("elected sign  =", elected.tolist())        # [1, -1, 1]

# الدمج المنفصل: متوسط القيم المتوافقة فقط مع الإشارة الغالبة
mask_a = (torch.sign(ta) == elected) & (ta != 0)
mask_b = (torch.sign(tb) == elected) & (tb != 0)
tau_ties = torch.where(mask_a, ta, torch.zeros_like(ta)) + \
           torch.where(mask_b, tb, torch.zeros_like(tb))
print("tau_ties      =", tau_ties.tolist())       # [0.4, -0.4, 0.3]
print("theta_ties    =", (theta_base + tau_ties).tolist())

# ────────────────────────────────────────────────────────────────────────
# [009] content/notebooks/part-06-model-reuse/ch26-model-merging.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import copy

def uniform_model_soup(model_checkpoints: list) -> dict:
    """
    Model Soup بأبسط صوره (Wortsman et al., 2022): متوسط حسابي بسيط
    لأوزان عدة نماذج بنفس المعمارية تمامًا - عادة نسخ من نفس النموذج
    مُدرَّبة بـ hyperparameters مختلفة قليلاً (بذور عشوائية، LR، إلخ).
    النتيجة غالبًا أفضل من أي نموذج فردي، وأحيانًا أفضل من أفضل نموذج
    في المجموعة - بدون أي تكلفة تدريب إضافية!
    """
    averaged_state_dict = copy.deepcopy(model_checkpoints[0])
    n_models = len(model_checkpoints)

    for key in averaged_state_dict.keys():
        # جمع نفس الوزن عبر كل النماذج ثم القسمة على العدد
        stacked = torch.stack([ckpt[key].float() for ckpt in model_checkpoints])
        averaged_state_dict[key] = stacked.mean(dim=0).to(model_checkpoints[0][key].dtype)

    return averaged_state_dict

def greedy_model_soup(model_checkpoints: list, validation_fn, model_template) -> dict:
    """
    Greedy Soup (أكثر فعالية من Uniform Soup): ترتيب النماذج حسب أدائها
    الفردي على مجموعة تحقق، ثم إضافتها واحدًا تلو الآخر للمزيج - مع
    الاحتفاظ فقط بالإضافات التي تُحسِّن (أو لا تُضعِف) أداء المزيج الحالي.
    """
    # الخطوة 1: تقييم كل نموذج منفردًا وترتيبها تنازليًا
    scored_models = []
    for ckpt in model_checkpoints:
        model_template.load_state_dict(ckpt)
        score = validation_fn(model_template)
        scored_models.append((ckpt, score))
    scored_models.sort(key=lambda x: x[1], reverse=True)

    # الخطوة 2: البدء بأفضل نموذج، ثم إضافة الباقي بالتتابع
    current_soup = copy.deepcopy(scored_models[0][0])
    model_template.load_state_dict(current_soup)
    current_best_score = validation_fn(model_template)
    n_ingredients = 1

    for ckpt, _ in scored_models[1:]:
        # تجربة إضافة هذا النموذج للمزيج الحالي (متوسط مُرجَّح بعدد المكوّنات)
        candidate_soup = {}
        for key in current_soup.keys():
            candidate_soup[key] = (
                current_soup[key].float() * n_ingredients + ckpt[key].float()
            ) / (n_ingredients + 1)
            candidate_soup[key] = candidate_soup[key].to(current_soup[key].dtype)

        model_template.load_state_dict(candidate_soup)
        candidate_score = validation_fn(model_template)

        if candidate_score >= current_best_score:
            current_soup = candidate_soup
            current_best_score = candidate_score
            n_ingredients += 1
            print(f"✅ تمت إضافة نموذج - الدرجة الجديدة: {candidate_score:.4f}")
        else:
            print(f"❌ تم تجاهل نموذج - لم يُحسِّن الأداء ({candidate_score:.4f})")

    return current_soup

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class TaskArithmetic:
    """
    تطبيق Task Vectors (Ilharco et al., 2022): استخراج ودمج "متجهات
    المهام" للسماح بدمج مرن للقدرات، أو حتى إزالة قدرات غير مرغوبة
    (مثل: إزالة ميل للسمية عبر "طرح" متجه مهمة سلبي).
    """
    def __init__(self, base_model_state_dict: dict):
        self.base_state_dict = base_model_state_dict

    def extract_task_vector(self, finetuned_state_dict: dict) -> dict:
        """استخراج متجه المهمة = الفرق بين النموذج المُضبَّط والأساسي."""
        task_vector = {}
        for key in self.base_state_dict.keys():
            task_vector[key] = (
                finetuned_state_dict[key].float() - self.base_state_dict[key].float()
            )
        return task_vector

    def apply_task_vectors(self, task_vectors_with_weights: list) -> dict:
        """
        دمج عدة متجهات مهام بأوزان (λ) مختلفة فوق النموذج الأساسي.
        task_vectors_with_weights: قائمة من (task_vector_dict, lambda_weight)
        """
        merged_state_dict = {}
        for key in self.base_state_dict.keys():
            combined_delta = sum(
                tv[key] * weight for tv, weight in task_vectors_with_weights
            )
            merged_state_dict[key] = (
                self.base_state_dict[key].float() + combined_delta
            ).to(self.base_state_dict[key].dtype)
        return merged_state_dict

    def negate_task_vector(self, task_vector: dict, strength: float = 1.0) -> dict:
        """
        إزالة مهارة/سلوك: طرح متجه المهمة بدل جمعه. مفيد لإزالة
        سلوكيات غير مرغوبة (مثلًا: متجه مهمة مُستخرَج من نموذج
        مُدرَّب عمدًا على استجابات سامة، ثم طرحه لتقليل السمية).
        """
        return self.apply_task_vectors([(task_vector, -strength)])

# --- مثال استخدام: دمج قدرة برمجية وقدرة رياضية في نموذج عام ---
arithmetic = TaskArithmetic(base_model_state_dict=base_llama_weights)

code_task_vector = arithmetic.extract_task_vector(code_finetuned_weights)
math_task_vector = arithmetic.extract_task_vector(math_finetuned_weights)

# دمج القدرتين معًا بأوزان متساوية (λ=0.5 لكل منهما شائع كقيمة ابتدائية)
merged_weights = arithmetic.apply_task_vectors([
    (code_task_vector, 0.5),
    (math_task_vector, 0.5),
])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class TIESMerging:
    """
    تطبيق كامل لخوارزمية TIES-Merging (Yadav et al., 2023):
    حل مشكلة تداخل الإشارات بين عدة Task Vectors عبر ثلاث خطوات
    منهجية: التقليم (Trim)، انتخاب الإشارة (Elect Sign)، والدمج
    الانتقائي (Disjoint Merge).
    """
    def __init__(self, base_state_dict: dict, trim_density: float = 0.2):
        self.base_state_dict = base_state_dict
        self.trim_density = trim_density  # نسبة القيم المُحتفَظ بها (الأعلى قيمة)

    def trim_task_vector(self, task_vector: dict) -> dict:
        """
        الخطوة 1 - Trim: الاحتفاظ فقط بأعلى k% من القيم المطلقة لكل
        متجه مهمة (القيم الأكبر تأثيرًا)، وتصفير الباقي. هذا يُزيل
        "الضجيج" الصغير الذي يُسبِّب معظم التداخل غير المفيد.
        """
        trimmed = {}
        for key, tensor in task_vector.items():
            flat = tensor.flatten().abs()
            k = int(len(flat) * self.trim_density)
            if k == 0:
                trimmed[key] = torch.zeros_like(tensor)
                continue
            threshold = torch.topk(flat, k).values.min()
            mask = tensor.abs() >= threshold
            trimmed[key] = tensor * mask
        return trimmed

    def elect_sign(self, trimmed_task_vectors: list) -> dict:
        """
        الخطوة 2 - Elect Sign: لكل وزن، تحديد "الإشارة الغالبة" عبر
        كل متجهات المهام (بناءً على مجموع القيم الموجبة مقابل السالبة
        المُرجَّح بالحجم)، ثم الاحتفاظ لاحقًا فقط بالتحديثات المتوافقة
        مع هذه الإشارة الغالبة.
        """
        elected_signs = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign_sum = stacked.sum(dim=0)
            elected_signs[key] = torch.sign(sign_sum)
        return elected_signs

    def disjoint_merge(self, trimmed_task_vectors: list,
                        elected_signs: dict) -> dict:
        """
        الخطوة 3 - Disjoint Merge: لكل وزن، نأخذ متوسط القيم فقط من
        متجهات المهام التي "تتفق" إشارتها مع الإشارة الغالبة المُنتخَبة،
        متجاهلين تمامًا القيم المتعارضة (بدل إلغائها بالجمع كما في
        Task Arithmetic البسيط).
        """
        merged_delta = {}
        for key in self.base_state_dict.keys():
            stacked = torch.stack([tv[key] for tv in trimmed_task_vectors])
            sign = elected_signs[key]

            # قناع القيم المتوافقة مع الإشارة الغالبة فقط
            agreement_mask = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
            agreement_count = agreement_mask.sum(dim=0).clamp(min=1)

            masked_values = stacked * agreement_mask
            merged_delta[key] = masked_values.sum(dim=0) / agreement_count

        return merged_delta

    def merge(self, task_vectors: list, scaling_factor: float = 1.0) -> dict:
        """الخط الكامل: Trim → Elect Sign → Disjoint Merge → إضافة للأساس."""
        trimmed_vectors = [self.trim_task_vector(tv) for tv in task_vectors]
        elected_signs = self.elect_sign(trimmed_vectors)
        merged_delta = self.disjoint_merge(trimmed_vectors, elected_signs)

        final_state_dict = {}
        for key in self.base_state_dict.keys():
            final_state_dict[key] = (
                self.base_state_dict[key].float()
                + scaling_factor * merged_delta[key]
            ).to(self.base_state_dict[key].dtype)
        return final_state_dict

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
def dare_sparsify_task_vector(task_vector: dict, drop_rate: float = 0.9) -> dict:
    """
    DARE (Yu et al., 2023): إسقاط عشوائي (Random Drop) لنسبة كبيرة جدًا
    من قيم متجه المهمة (حتى 90-99%!)، مع إعادة قياس (Rescale) القيم
    المتبقية للتعويض عن الكتلة المفقودة. النتيجة المُدهشة: الأداء
    يبقى شبه ثابت حتى مع drop_rate=0.9، مما يُظهر أن معظم التحديثات
    في Task Vector "زائدة عن الحاجة" (Redundant) ويمكن حذفها بأمان،
    وهذا يُقلِّل تداخل الإشارات بشكل كبير عند الدمج اللاحق.
    """
    sparsified = {}
    for key, tensor in task_vector.items():
        mask = torch.rand_like(tensor) > drop_rate  # الاحتفاظ بنسبة (1-drop_rate)
        rescale_factor = 1.0 / (1.0 - drop_rate)  # إعادة القياس للتعويض
        sparsified[key] = tensor * mask * rescale_factor
    return sparsified

# DARE غالبًا تُستخدم كخطوة تمهيدية قبل TIES أو Task Arithmetic
# (DARE-TIES: تطبيق DARE أولاً لتخفيف الكثافة، ثم TIES للدمج النهائي)

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
def slerp_merge(weight_a: torch.Tensor, weight_b: torch.Tensor,
                 t: float = 0.5, epsilon: float = 1e-8) -> torch.Tensor:
    """
    الاستيفاء الخطي الكروي (SLERP) بين وزنين - أكثر دقة هندسيًا من
    المتوسط الخطي البسيط lerp = (1-t)*a + t*b عند دمج نموذجين فقط،
    لأنه يحافظ على "القاعدة" (norm) الهندسية للمتجهات بدل تقليصها
    بشكل مصطنع كما يحدث أحيانًا مع المتوسط الخطي المباشر.
    """
    a_flat = weight_a.flatten().float()
    b_flat = weight_b.flatten().float()

    a_norm = a_flat / (a_flat.norm() + epsilon)
    b_norm = b_flat / (b_flat.norm() + epsilon)

    # زاوية التشابه بين المتجهين
    dot_product = torch.clamp((a_norm * b_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega) + epsilon

    # إذا كانت الزاوية صغيرة جدًا (متجهات شبه متماثلة)، استخدم lerp عادي كبديل آمن
    if sin_omega.abs() < 1e-6:
        result_flat = (1 - t) * a_flat + t * b_flat
    else:
        coeff_a = torch.sin((1 - t) * omega) / sin_omega
        coeff_b = torch.sin(t * omega) / sin_omega
        result_flat = coeff_a * a_flat + coeff_b * b_flat

    return result_flat.reshape(weight_a.shape).to(weight_a.dtype)

def slerp_merge_full_models(state_dict_a: dict, state_dict_b: dict,
                             t: float = 0.5) -> dict:
    """تطبيق SLERP على كل أوزان النموذج طبقة بطبقة."""
    merged = {}
    for key in state_dict_a.keys():
        merged[key] = slerp_merge(state_dict_a[key], state_dict_b[key], t=t)
    return merged

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
def frankenmerge_layers(model_a_layers: list, model_b_layers: list,
                         layer_recipe: list) -> list:
    """
    Frankenmerging: بناء نموذج جديد بتجميع طبقات كاملة من نماذج مختلفة
    (مشابهة البنية) حسب "وصفة" مُحدَّدة يدويًا أو تجريبيًا. هذا النهج
    استخدمته نماذج مثل Goliath-120B (دمج طبقات من نسختين من Llama 2 70B)
    لإنتاج نموذج أكبر وأحيانًا أقوى دون أي تدريب إضافي على الإطلاق.

    layer_recipe: قائمة من tuples (المصدر, رقم_الطبقة)
    مثال: [("a", 0), ("a", 1), ("b", 0), ("a", 2), ("b", 1), ...]
    """
    new_model_layers = []
    for source, layer_idx in layer_recipe:
        if source == "a":
            new_model_layers.append(copy.deepcopy(model_a_layers[layer_idx]))
        elif source == "b":
            new_model_layers.append(copy.deepcopy(model_b_layers[layer_idx]))
    return new_model_layers

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
import torch

# أوزان أساس افتراضية (3 قيم فقط للتوضيح)
theta_base = torch.tensor([1.0, -0.5, 0.2])

# متجه مهمة A (برمجة) ومتجه مهمة B (رياضيات)
tau_a = torch.tensor([0.4, 0.3, -0.1])
tau_b = torch.tensor([0.2, -0.4, 0.3])

# 1) Task Arithmetic البسيط: جمع مباشر
tau_sum = tau_a + tau_b
print("tau_sum       =", tau_sum.tolist())        # [0.6, -0.1, 0.2]
print("theta_sum     =", (theta_base + tau_sum).tolist())

# 2) TIES مبسّط: تقليم القيم الصغيرة ثم انتخاب الإشارة الغالبة
def trim(tau, keep=2):
    threshold = tau.abs().topk(keep).values.min()
    return torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))

ta, tb = trim(tau_a), trim(tau_b)
print("trimmed A     =", ta.tolist())             # [0.4, 0.3, 0.0]
print("trimmed B     =", tb.tolist())             # [0.0, -0.4, 0.3]

# الإشارة الغالبة لكل موضع: مجموع القيم الموقّعة
elected = torch.sign(ta + tb)
print("elected sign  =", elected.tolist())        # [1, -1, 1]

# الدمج المنفصل: متوسط القيم المتوافقة فقط مع الإشارة الغالبة
mask_a = (torch.sign(ta) == elected) & (ta != 0)
mask_b = (torch.sign(tb) == elected) & (tb != 0)
tau_ties = torch.where(mask_a, ta, torch.zeros_like(ta)) + \
           torch.where(mask_b, tb, torch.zeros_like(tb))
print("tau_ties      =", tau_ties.tolist())       # [0.4, -0.4, 0.3]
print("theta_ties    =", (theta_base + tau_ties).tolist())

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/mastering-llms-transformers/notebooks/ch-26-ch26-model-merging.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
