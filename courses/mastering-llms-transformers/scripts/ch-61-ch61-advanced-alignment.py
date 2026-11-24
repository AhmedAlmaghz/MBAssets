#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المحاذاة المتقدمة وسلامة الأنظمة الفائقة
Mastering Transformers and Large Language Models — Advanced Alignment

Book slug : mastering-llms-transformers
Chapter   : ch-61-ch61-advanced-alignment (lesson 61)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-61-ch61-advanced-alignment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المحاذاة المتقدمة وسلامة الأنظمة الفائقة
Mastering Transformers and Large Language Models — Advanced Alignment

Book slug : mastering-llms-transformers
Chapter   : ch-61-ch61-advanced-alignment (lesson 61)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-61-ch61-advanced-alignment.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المحاذاة المتقدمة وسلامة الأنظمة الفائقة
Mastering Transformers and Large Language Models — Advanced Alignment

Book slug : mastering-llms-transformers
Chapter   : ch-61-ch61-advanced-alignment (lesson 61)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-1
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
# [002] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class WeakToStrongExperiment:
    """
    محاكاة لتجربة Weak-to-Strong Generalization من OpenAI.
    الهدف: قياس مدى قدرة نموذج قوي على "تجاوز" أخطاء مُعلِّمه الضعيف
    بدلًا من مجرد تقليد أخطائه.
    """
    def __init__(self, weak_model, strong_pretrained_model, ground_truth_dataset):
        self.weak_model = weak_model
        self.strong_model = strong_pretrained_model
        self.ground_truth = ground_truth_dataset

    def generate_weak_labels(self, unlabeled_data):
        """النموذج الضعيف يُسمِّي البيانات (قد تحتوي أخطاء)."""
        return [self.weak_model.predict(x) for x in unlabeled_data]

    def finetune_strong_on_weak_labels(self, unlabeled_data, weak_labels):
        """تدريب النموذج القوي على تسميات ضعيفة الجودة فقط."""
        # النموذج القوي يحتفظ بمعرفته المُسبقة (Pretraining) الغنية
        # وقد "يُصحح" بعض أخطاء المُعلِّم الضعيف اعتمادًا على فهمه الأعمق
        finetuned_strong = self.strong_model.finetune(
            unlabeled_data, weak_labels
        )
        return finetuned_strong

    def compute_performance_gap_recovered(self, finetuned_strong):
        """
        المقياس الأساسي: Performance Gap Recovered (PGR)
        PGR = (أداء Strong-w2s - أداء Weak) / (أداء Strong-ceiling - أداء Weak)
        PGR = 1.0 يعني تجاوز كامل لحدود المُعلِّم الضعيف (نتيجة مثالية)
        PGR = 0.0 يعني تقليد كامل لأخطاء المُعلِّم (نتيجة سيئة)
        """
        weak_perf = self._evaluate(self.weak_model)
        strong_w2s_perf = self._evaluate(finetuned_strong)
        strong_ceiling_perf = self._evaluate_with_ground_truth(self.strong_model)

        pgr = (strong_w2s_perf - weak_perf) / (strong_ceiling_perf - weak_perf + 1e-8)
        return pgr

    def _evaluate(self, model):
        correct = sum(
            1 for x, y_true in self.ground_truth
            if model.predict(x) == y_true
        )
        return correct / len(self.ground_truth)

    def _evaluate_with_ground_truth(self, model):
        # تدريب نفس النموذج القوي لكن على تسميات صحيحة 100%، كحد أعلى مرجعي
        return self._evaluate(model)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
CONSTITUTIONAL_PRINCIPLES = [
    {
        "name": "تجنب الضرر",
        "critique_prompt": (
            "راجع الاستجابة التالية وحدد ما إذا كانت تحتوي معلومات "
            "قد تُستخدم لإيذاء الآخرين أو كسر القانون. اشرح بالتفصيل."
        ),
        "revision_prompt": (
            "بناءً على النقد أعلاه، أعد صياغة الاستجابة لتكون مفيدة "
            "قدر الإمكان مع إزالة أي محتوى ضار."
        ),
    },
    {
        "name": "احترام الخصوصية",
        "critique_prompt": (
            "هل تحتوي هذه الاستجابة معلومات شخصية حساسة عن أفراد حقيقيين "
            "دون مبرر واضح؟"
        ),
        "revision_prompt": "أعد الصياغة مع حماية خصوصية أي أفراد مذكورين.",
    },
    {
        "name": "الصدق والتواضع المعرفي",
        "critique_prompt": (
            "هل تُعبِّر هذه الاستجابة عن ثقة مفرطة في معلومات قد تكون "
            "غير مؤكدة أو قديمة؟"
        ),
        "revision_prompt": (
            "أعد الصياغة مع إضافة تحفظات مناسبة حول درجة اليقين."
        ),
    },
]

class ConstitutionalAIPipeline:
    """تطبيق مبسّط لخط أنابيب Constitutional AI (مرحلة SL-CAI)."""

    def __init__(self, base_model, principles=CONSTITUTIONAL_PRINCIPLES):
        self.model = base_model
        self.principles = principles

    def self_critique_and_revise(self, prompt: str, initial_response: str) -> str:
        """تطبيق دورة نقد-تعديل ذاتية عبر كل المبادئ الدستورية بالتتابع."""
        current_response = initial_response

        for principle in self.principles:
            critique = self.model.generate(
                f"السؤال الأصلي: {prompt}\n"
                f"الاستجابة: {current_response}\n\n"
                f"{principle['critique_prompt']}"
            )

            # إن لم يجد النقد مشكلة جوهرية، انتقل للمبدأ التالي دون تعديل
            if self._critique_indicates_no_issue(critique):
                continue

            current_response = self.model.generate(
                f"الاستجابة: {current_response}\n"
                f"النقد: {critique}\n\n"
                f"{principle['revision_prompt']}"
            )

        return current_response

    def _critique_indicates_no_issue(self, critique_text: str) -> bool:
        no_issue_markers = ["لا توجد مشكلة", "الاستجابة آمنة", "لا حاجة للتعديل"]
        return any(marker in critique_text for marker in no_issue_markers)

    def build_sft_dataset(self, prompts: list) -> list:
        """بناء مجموعة بيانات SFT من أزواج (prompt, استجابة مُنقَّحة دستوريًا)."""
        dataset = []
        for prompt in prompts:
            initial = self.model.generate(prompt)
            revised = self.self_critique_and_revise(prompt, initial)
            dataset.append({"prompt": prompt, "response": revised})
        return dataset

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) لتفكيك التنشيطات المتشابكة (Polysemantic)
    إلى "ميزات" أحادية المعنى (Monosemantic Features) — المنهجية
    المستخدمة في ورقة Anthropic "Towards Monosemanticity" و"Scaling
    Monosemanticity" (اكتشاف ميزات Claude 3 Sonnet).
    """
    def __init__(self, activation_dim: int, dictionary_size: int,
                 sparsity_coefficient: float = 1e-3):
        super().__init__()
        # dictionary_size عادة أكبر بكثير من activation_dim (overcomplete)
        # مثال: activation_dim=4096, dictionary_size=16384 أو حتى مليون+
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=True)
        self.sparsity_coefficient = sparsity_coefficient

        # تهيئة الـ decoder weights لتكون متعامدة قدر الإمكان (ممارسة شائعة)
        nn.init.orthogonal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def forward(self, activations: torch.Tensor):
        """
        activations: شكلها [batch, activation_dim] — تنشيطات طبقة معينة
        من النموذج الأصلي (غالبًا residual stream أو MLP output)
        """
        # الترميز مع ReLU لفرض الإيجابية والندرة (sparsity)
        features = torch.relu(self.encoder(activations))  # [batch, dictionary_size]

        # إعادة البناء
        reconstruction = self.decoder(features)  # [batch, activation_dim]

        return reconstruction, features

    def compute_loss(self, activations: torch.Tensor):
        """
        دالة الخسارة المركّبة: خطأ إعادة البناء + عقوبة الندرة (L1)
        """
        reconstruction, features = self.forward(activations)

        # خسارة إعادة البناء (MSE)
        reconstruction_loss = torch.nn.functional.mse_loss(
            reconstruction, activations
        )

        # عقوبة الندرة L1: تُجبر معظم الميزات على أن تكون صفرًا
        # لكل عيّنة — هذا ما يُنتج "ميزات أحادية المعنى" بدل تنشيط كثيف
        sparsity_loss = features.abs().sum(dim=-1).mean()

        total_loss = reconstruction_loss + self.sparsity_coefficient * sparsity_loss

        # نسبة الندرة الفعلية (للمراقبة أثناء التدريب)
        with torch.no_grad():
            l0_sparsity = (features > 0).float().sum(dim=-1).mean()

        return total_loss, {
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "avg_active_features": l0_sparsity.item(),  # الهدف: رقم صغير جدًا
        }

def interpret_feature(sae: SparseAutoencoder, feature_index: int,
                       text_dataset: list, model, layer_name: str,
                       top_k: int = 20) -> list:
    """
    تفسير ميزة محددة: إيجاد أعلى K نصوص تُنشِّط هذه الميزة بأعلى قيمة،
    لفهم "ماذا تمثّل" هذه الميزة دلاليًا (مثال: قد نكتشف أن الميزة #4821
    تُنشَّط حصريًا عند ذكر "خيانة الثقة" في أي سياق - عربي أو مجازي).
    """
    activations_cache = []
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda module, inp, out: activations_cache.append(out.detach())
    )

    feature_activations = []
    for text in text_dataset:
        activations_cache.clear()
        _ = model(text)
        layer_activation = activations_cache[0]
        _, features = sae.forward(layer_activation)
        max_activation = features[:, feature_index].max().item()
        feature_activations.append((text, max_activation))

    hook.remove()
    feature_activations.sort(key=lambda x: x[1], reverse=True)
    return feature_activations[:top_k]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class WeakToStrongExperiment:
    """
    محاكاة لتجربة Weak-to-Strong Generalization من OpenAI.
    الهدف: قياس مدى قدرة نموذج قوي على "تجاوز" أخطاء مُعلِّمه الضعيف
    بدلًا من مجرد تقليد أخطائه.
    """
    def __init__(self, weak_model, strong_pretrained_model, ground_truth_dataset):
        self.weak_model = weak_model
        self.strong_model = strong_pretrained_model
        self.ground_truth = ground_truth_dataset

    def generate_weak_labels(self, unlabeled_data):
        """النموذج الضعيف يُسمِّي البيانات (قد تحتوي أخطاء)."""
        return [self.weak_model.predict(x) for x in unlabeled_data]

    def finetune_strong_on_weak_labels(self, unlabeled_data, weak_labels):
        """تدريب النموذج القوي على تسميات ضعيفة الجودة فقط."""
        # النموذج القوي يحتفظ بمعرفته المُسبقة (Pretraining) الغنية
        # وقد "يُصحح" بعض أخطاء المُعلِّم الضعيف اعتمادًا على فهمه الأعمق
        finetuned_strong = self.strong_model.finetune(
            unlabeled_data, weak_labels
        )
        return finetuned_strong

    def compute_performance_gap_recovered(self, finetuned_strong):
        """
        المقياس الأساسي: Performance Gap Recovered (PGR)
        PGR = (أداء Strong-w2s - أداء Weak) / (أداء Strong-ceiling - أداء Weak)
        PGR = 1.0 يعني تجاوز كامل لحدود المُعلِّم الضعيف (نتيجة مثالية)
        PGR = 0.0 يعني تقليد كامل لأخطاء المُعلِّم (نتيجة سيئة)
        """
        weak_perf = self._evaluate(self.weak_model)
        strong_w2s_perf = self._evaluate(finetuned_strong)
        strong_ceiling_perf = self._evaluate_with_ground_truth(self.strong_model)

        pgr = (strong_w2s_perf - weak_perf) / (strong_ceiling_perf - weak_perf + 1e-8)
        return pgr

    def _evaluate(self, model):
        correct = sum(
            1 for x, y_true in self.ground_truth
            if model.predict(x) == y_true
        )
        return correct / len(self.ground_truth)

    def _evaluate_with_ground_truth(self, model):
        # تدريب نفس النموذج القوي لكن على تسميات صحيحة 100%، كحد أعلى مرجعي
        return self._evaluate(model)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
CONSTITUTIONAL_PRINCIPLES = [
    {
        "name": "تجنب الضرر",
        "critique_prompt": (
            "راجع الاستجابة التالية وحدد ما إذا كانت تحتوي معلومات "
            "قد تُستخدم لإيذاء الآخرين أو كسر القانون. اشرح بالتفصيل."
        ),
        "revision_prompt": (
            "بناءً على النقد أعلاه، أعد صياغة الاستجابة لتكون مفيدة "
            "قدر الإمكان مع إزالة أي محتوى ضار."
        ),
    },
    {
        "name": "احترام الخصوصية",
        "critique_prompt": (
            "هل تحتوي هذه الاستجابة معلومات شخصية حساسة عن أفراد حقيقيين "
            "دون مبرر واضح؟"
        ),
        "revision_prompt": "أعد الصياغة مع حماية خصوصية أي أفراد مذكورين.",
    },
    {
        "name": "الصدق والتواضع المعرفي",
        "critique_prompt": (
            "هل تُعبِّر هذه الاستجابة عن ثقة مفرطة في معلومات قد تكون "
            "غير مؤكدة أو قديمة؟"
        ),
        "revision_prompt": (
            "أعد الصياغة مع إضافة تحفظات مناسبة حول درجة اليقين."
        ),
    },
]

class ConstitutionalAIPipeline:
    """تطبيق مبسّط لخط أنابيب Constitutional AI (مرحلة SL-CAI)."""

    def __init__(self, base_model, principles=CONSTITUTIONAL_PRINCIPLES):
        self.model = base_model
        self.principles = principles

    def self_critique_and_revise(self, prompt: str, initial_response: str) -> str:
        """تطبيق دورة نقد-تعديل ذاتية عبر كل المبادئ الدستورية بالتتابع."""
        current_response = initial_response

        for principle in self.principles:
            critique = self.model.generate(
                f"السؤال الأصلي: {prompt}\n"
                f"الاستجابة: {current_response}\n\n"
                f"{principle['critique_prompt']}"
            )

            # إن لم يجد النقد مشكلة جوهرية، انتقل للمبدأ التالي دون تعديل
            if self._critique_indicates_no_issue(critique):
                continue

            current_response = self.model.generate(
                f"الاستجابة: {current_response}\n"
                f"النقد: {critique}\n\n"
                f"{principle['revision_prompt']}"
            )

        return current_response

    def _critique_indicates_no_issue(self, critique_text: str) -> bool:
        no_issue_markers = ["لا توجد مشكلة", "الاستجابة آمنة", "لا حاجة للتعديل"]
        return any(marker in critique_text for marker in no_issue_markers)

    def build_sft_dataset(self, prompts: list) -> list:
        """بناء مجموعة بيانات SFT من أزواج (prompt, استجابة مُنقَّحة دستوريًا)."""
        dataset = []
        for prompt in prompts:
            initial = self.model.generate(prompt)
            revised = self.self_critique_and_revise(prompt, initial)
            dataset.append({"prompt": prompt, "response": revised})
        return dataset

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) لتفكيك التنشيطات المتشابكة (Polysemantic)
    إلى "ميزات" أحادية المعنى (Monosemantic Features) — المنهجية
    المستخدمة في ورقة Anthropic "Towards Monosemanticity" و"Scaling
    Monosemanticity" (اكتشاف ميزات Claude 3 Sonnet).
    """
    def __init__(self, activation_dim: int, dictionary_size: int,
                 sparsity_coefficient: float = 1e-3):
        super().__init__()
        # dictionary_size عادة أكبر بكثير من activation_dim (overcomplete)
        # مثال: activation_dim=4096, dictionary_size=16384 أو حتى مليون+
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=True)
        self.sparsity_coefficient = sparsity_coefficient

        # تهيئة الـ decoder weights لتكون متعامدة قدر الإمكان (ممارسة شائعة)
        nn.init.orthogonal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def forward(self, activations: torch.Tensor):
        """
        activations: شكلها [batch, activation_dim] — تنشيطات طبقة معينة
        من النموذج الأصلي (غالبًا residual stream أو MLP output)
        """
        # الترميز مع ReLU لفرض الإيجابية والندرة (sparsity)
        features = torch.relu(self.encoder(activations))  # [batch, dictionary_size]

        # إعادة البناء
        reconstruction = self.decoder(features)  # [batch, activation_dim]

        return reconstruction, features

    def compute_loss(self, activations: torch.Tensor):
        """
        دالة الخسارة المركّبة: خطأ إعادة البناء + عقوبة الندرة (L1)
        """
        reconstruction, features = self.forward(activations)

        # خسارة إعادة البناء (MSE)
        reconstruction_loss = torch.nn.functional.mse_loss(
            reconstruction, activations
        )

        # عقوبة الندرة L1: تُجبر معظم الميزات على أن تكون صفرًا
        # لكل عيّنة — هذا ما يُنتج "ميزات أحادية المعنى" بدل تنشيط كثيف
        sparsity_loss = features.abs().sum(dim=-1).mean()

        total_loss = reconstruction_loss + self.sparsity_coefficient * sparsity_loss

        # نسبة الندرة الفعلية (للمراقبة أثناء التدريب)
        with torch.no_grad():
            l0_sparsity = (features > 0).float().sum(dim=-1).mean()

        return total_loss, {
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "avg_active_features": l0_sparsity.item(),  # الهدف: رقم صغير جدًا
        }

def interpret_feature(sae: SparseAutoencoder, feature_index: int,
                       text_dataset: list, model, layer_name: str,
                       top_k: int = 20) -> list:
    """
    تفسير ميزة محددة: إيجاد أعلى K نصوص تُنشِّط هذه الميزة بأعلى قيمة،
    لفهم "ماذا تمثّل" هذه الميزة دلاليًا (مثال: قد نكتشف أن الميزة #4821
    تُنشَّط حصريًا عند ذكر "خيانة الثقة" في أي سياق - عربي أو مجازي).
    """
    activations_cache = []
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda module, inp, out: activations_cache.append(out.detach())
    )

    feature_activations = []
    for text in text_dataset:
        activations_cache.clear()
        _ = model(text)
        layer_activation = activations_cache[0]
        _, features = sae.forward(layer_activation)
        max_activation = features[:, feature_index].max().item()
        feature_activations.append((text, max_activation))

    hook.remove()
    feature_activations.sort(key=lambda x: x[1], reverse=True)
    return feature_activations[:top_k]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المحاذاة المتقدمة وسلامة الأنظمة الفائقة
Mastering Transformers and Large Language Models — Advanced Alignment

Book slug : mastering-llms-transformers
Chapter   : ch-61-ch61-advanced-alignment (lesson 61)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-1
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
# [002] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class WeakToStrongExperiment:
    """
    محاكاة لتجربة Weak-to-Strong Generalization من OpenAI.
    الهدف: قياس مدى قدرة نموذج قوي على "تجاوز" أخطاء مُعلِّمه الضعيف
    بدلًا من مجرد تقليد أخطائه.
    """
    def __init__(self, weak_model, strong_pretrained_model, ground_truth_dataset):
        self.weak_model = weak_model
        self.strong_model = strong_pretrained_model
        self.ground_truth = ground_truth_dataset

    def generate_weak_labels(self, unlabeled_data):
        """النموذج الضعيف يُسمِّي البيانات (قد تحتوي أخطاء)."""
        return [self.weak_model.predict(x) for x in unlabeled_data]

    def finetune_strong_on_weak_labels(self, unlabeled_data, weak_labels):
        """تدريب النموذج القوي على تسميات ضعيفة الجودة فقط."""
        # النموذج القوي يحتفظ بمعرفته المُسبقة (Pretraining) الغنية
        # وقد "يُصحح" بعض أخطاء المُعلِّم الضعيف اعتمادًا على فهمه الأعمق
        finetuned_strong = self.strong_model.finetune(
            unlabeled_data, weak_labels
        )
        return finetuned_strong

    def compute_performance_gap_recovered(self, finetuned_strong):
        """
        المقياس الأساسي: Performance Gap Recovered (PGR)
        PGR = (أداء Strong-w2s - أداء Weak) / (أداء Strong-ceiling - أداء Weak)
        PGR = 1.0 يعني تجاوز كامل لحدود المُعلِّم الضعيف (نتيجة مثالية)
        PGR = 0.0 يعني تقليد كامل لأخطاء المُعلِّم (نتيجة سيئة)
        """
        weak_perf = self._evaluate(self.weak_model)
        strong_w2s_perf = self._evaluate(finetuned_strong)
        strong_ceiling_perf = self._evaluate_with_ground_truth(self.strong_model)

        pgr = (strong_w2s_perf - weak_perf) / (strong_ceiling_perf - weak_perf + 1e-8)
        return pgr

    def _evaluate(self, model):
        correct = sum(
            1 for x, y_true in self.ground_truth
            if model.predict(x) == y_true
        )
        return correct / len(self.ground_truth)

    def _evaluate_with_ground_truth(self, model):
        # تدريب نفس النموذج القوي لكن على تسميات صحيحة 100%، كحد أعلى مرجعي
        return self._evaluate(model)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
CONSTITUTIONAL_PRINCIPLES = [
    {
        "name": "تجنب الضرر",
        "critique_prompt": (
            "راجع الاستجابة التالية وحدد ما إذا كانت تحتوي معلومات "
            "قد تُستخدم لإيذاء الآخرين أو كسر القانون. اشرح بالتفصيل."
        ),
        "revision_prompt": (
            "بناءً على النقد أعلاه، أعد صياغة الاستجابة لتكون مفيدة "
            "قدر الإمكان مع إزالة أي محتوى ضار."
        ),
    },
    {
        "name": "احترام الخصوصية",
        "critique_prompt": (
            "هل تحتوي هذه الاستجابة معلومات شخصية حساسة عن أفراد حقيقيين "
            "دون مبرر واضح؟"
        ),
        "revision_prompt": "أعد الصياغة مع حماية خصوصية أي أفراد مذكورين.",
    },
    {
        "name": "الصدق والتواضع المعرفي",
        "critique_prompt": (
            "هل تُعبِّر هذه الاستجابة عن ثقة مفرطة في معلومات قد تكون "
            "غير مؤكدة أو قديمة؟"
        ),
        "revision_prompt": (
            "أعد الصياغة مع إضافة تحفظات مناسبة حول درجة اليقين."
        ),
    },
]

class ConstitutionalAIPipeline:
    """تطبيق مبسّط لخط أنابيب Constitutional AI (مرحلة SL-CAI)."""

    def __init__(self, base_model, principles=CONSTITUTIONAL_PRINCIPLES):
        self.model = base_model
        self.principles = principles

    def self_critique_and_revise(self, prompt: str, initial_response: str) -> str:
        """تطبيق دورة نقد-تعديل ذاتية عبر كل المبادئ الدستورية بالتتابع."""
        current_response = initial_response

        for principle in self.principles:
            critique = self.model.generate(
                f"السؤال الأصلي: {prompt}\n"
                f"الاستجابة: {current_response}\n\n"
                f"{principle['critique_prompt']}"
            )

            # إن لم يجد النقد مشكلة جوهرية، انتقل للمبدأ التالي دون تعديل
            if self._critique_indicates_no_issue(critique):
                continue

            current_response = self.model.generate(
                f"الاستجابة: {current_response}\n"
                f"النقد: {critique}\n\n"
                f"{principle['revision_prompt']}"
            )

        return current_response

    def _critique_indicates_no_issue(self, critique_text: str) -> bool:
        no_issue_markers = ["لا توجد مشكلة", "الاستجابة آمنة", "لا حاجة للتعديل"]
        return any(marker in critique_text for marker in no_issue_markers)

    def build_sft_dataset(self, prompts: list) -> list:
        """بناء مجموعة بيانات SFT من أزواج (prompt, استجابة مُنقَّحة دستوريًا)."""
        dataset = []
        for prompt in prompts:
            initial = self.model.generate(prompt)
            revised = self.self_critique_and_revise(prompt, initial)
            dataset.append({"prompt": prompt, "response": revised})
        return dataset

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) لتفكيك التنشيطات المتشابكة (Polysemantic)
    إلى "ميزات" أحادية المعنى (Monosemantic Features) — المنهجية
    المستخدمة في ورقة Anthropic "Towards Monosemanticity" و"Scaling
    Monosemanticity" (اكتشاف ميزات Claude 3 Sonnet).
    """
    def __init__(self, activation_dim: int, dictionary_size: int,
                 sparsity_coefficient: float = 1e-3):
        super().__init__()
        # dictionary_size عادة أكبر بكثير من activation_dim (overcomplete)
        # مثال: activation_dim=4096, dictionary_size=16384 أو حتى مليون+
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=True)
        self.sparsity_coefficient = sparsity_coefficient

        # تهيئة الـ decoder weights لتكون متعامدة قدر الإمكان (ممارسة شائعة)
        nn.init.orthogonal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def forward(self, activations: torch.Tensor):
        """
        activations: شكلها [batch, activation_dim] — تنشيطات طبقة معينة
        من النموذج الأصلي (غالبًا residual stream أو MLP output)
        """
        # الترميز مع ReLU لفرض الإيجابية والندرة (sparsity)
        features = torch.relu(self.encoder(activations))  # [batch, dictionary_size]

        # إعادة البناء
        reconstruction = self.decoder(features)  # [batch, activation_dim]

        return reconstruction, features

    def compute_loss(self, activations: torch.Tensor):
        """
        دالة الخسارة المركّبة: خطأ إعادة البناء + عقوبة الندرة (L1)
        """
        reconstruction, features = self.forward(activations)

        # خسارة إعادة البناء (MSE)
        reconstruction_loss = torch.nn.functional.mse_loss(
            reconstruction, activations
        )

        # عقوبة الندرة L1: تُجبر معظم الميزات على أن تكون صفرًا
        # لكل عيّنة — هذا ما يُنتج "ميزات أحادية المعنى" بدل تنشيط كثيف
        sparsity_loss = features.abs().sum(dim=-1).mean()

        total_loss = reconstruction_loss + self.sparsity_coefficient * sparsity_loss

        # نسبة الندرة الفعلية (للمراقبة أثناء التدريب)
        with torch.no_grad():
            l0_sparsity = (features > 0).float().sum(dim=-1).mean()

        return total_loss, {
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "avg_active_features": l0_sparsity.item(),  # الهدف: رقم صغير جدًا
        }

def interpret_feature(sae: SparseAutoencoder, feature_index: int,
                       text_dataset: list, model, layer_name: str,
                       top_k: int = 20) -> list:
    """
    تفسير ميزة محددة: إيجاد أعلى K نصوص تُنشِّط هذه الميزة بأعلى قيمة،
    لفهم "ماذا تمثّل" هذه الميزة دلاليًا (مثال: قد نكتشف أن الميزة #4821
    تُنشَّط حصريًا عند ذكر "خيانة الثقة" في أي سياق - عربي أو مجازي).
    """
    activations_cache = []
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda module, inp, out: activations_cache.append(out.detach())
    )

    feature_activations = []
    for text in text_dataset:
        activations_cache.clear()
        _ = model(text)
        layer_activation = activations_cache[0]
        _, features = sae.forward(layer_activation)
        max_activation = features[:, feature_index].max().item()
        feature_activations.append((text, max_activation))

    hook.remove()
    feature_activations.sort(key=lambda x: x[1], reverse=True)
    return feature_activations[:top_k]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-12-future/ch61-advanced-alignment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class WeakToStrongExperiment:
    """
    محاكاة لتجربة Weak-to-Strong Generalization من OpenAI.
    الهدف: قياس مدى قدرة نموذج قوي على "تجاوز" أخطاء مُعلِّمه الضعيف
    بدلًا من مجرد تقليد أخطائه.
    """
    def __init__(self, weak_model, strong_pretrained_model, ground_truth_dataset):
        self.weak_model = weak_model
        self.strong_model = strong_pretrained_model
        self.ground_truth = ground_truth_dataset

    def generate_weak_labels(self, unlabeled_data):
        """النموذج الضعيف يُسمِّي البيانات (قد تحتوي أخطاء)."""
        return [self.weak_model.predict(x) for x in unlabeled_data]

    def finetune_strong_on_weak_labels(self, unlabeled_data, weak_labels):
        """تدريب النموذج القوي على تسميات ضعيفة الجودة فقط."""
        # النموذج القوي يحتفظ بمعرفته المُسبقة (Pretraining) الغنية
        # وقد "يُصحح" بعض أخطاء المُعلِّم الضعيف اعتمادًا على فهمه الأعمق
        finetuned_strong = self.strong_model.finetune(
            unlabeled_data, weak_labels
        )
        return finetuned_strong

    def compute_performance_gap_recovered(self, finetuned_strong):
        """
        المقياس الأساسي: Performance Gap Recovered (PGR)
        PGR = (أداء Strong-w2s - أداء Weak) / (أداء Strong-ceiling - أداء Weak)
        PGR = 1.0 يعني تجاوز كامل لحدود المُعلِّم الضعيف (نتيجة مثالية)
        PGR = 0.0 يعني تقليد كامل لأخطاء المُعلِّم (نتيجة سيئة)
        """
        weak_perf = self._evaluate(self.weak_model)
        strong_w2s_perf = self._evaluate(finetuned_strong)
        strong_ceiling_perf = self._evaluate_with_ground_truth(self.strong_model)

        pgr = (strong_w2s_perf - weak_perf) / (strong_ceiling_perf - weak_perf + 1e-8)
        return pgr

    def _evaluate(self, model):
        correct = sum(
            1 for x, y_true in self.ground_truth
            if model.predict(x) == y_true
        )
        return correct / len(self.ground_truth)

    def _evaluate_with_ground_truth(self, model):
        # تدريب نفس النموذج القوي لكن على تسميات صحيحة 100%، كحد أعلى مرجعي
        return self._evaluate(model)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
CONSTITUTIONAL_PRINCIPLES = [
    {
        "name": "تجنب الضرر",
        "critique_prompt": (
            "راجع الاستجابة التالية وحدد ما إذا كانت تحتوي معلومات "
            "قد تُستخدم لإيذاء الآخرين أو كسر القانون. اشرح بالتفصيل."
        ),
        "revision_prompt": (
            "بناءً على النقد أعلاه، أعد صياغة الاستجابة لتكون مفيدة "
            "قدر الإمكان مع إزالة أي محتوى ضار."
        ),
    },
    {
        "name": "احترام الخصوصية",
        "critique_prompt": (
            "هل تحتوي هذه الاستجابة معلومات شخصية حساسة عن أفراد حقيقيين "
            "دون مبرر واضح؟"
        ),
        "revision_prompt": "أعد الصياغة مع حماية خصوصية أي أفراد مذكورين.",
    },
    {
        "name": "الصدق والتواضع المعرفي",
        "critique_prompt": (
            "هل تُعبِّر هذه الاستجابة عن ثقة مفرطة في معلومات قد تكون "
            "غير مؤكدة أو قديمة؟"
        ),
        "revision_prompt": (
            "أعد الصياغة مع إضافة تحفظات مناسبة حول درجة اليقين."
        ),
    },
]

class ConstitutionalAIPipeline:
    """تطبيق مبسّط لخط أنابيب Constitutional AI (مرحلة SL-CAI)."""

    def __init__(self, base_model, principles=CONSTITUTIONAL_PRINCIPLES):
        self.model = base_model
        self.principles = principles

    def self_critique_and_revise(self, prompt: str, initial_response: str) -> str:
        """تطبيق دورة نقد-تعديل ذاتية عبر كل المبادئ الدستورية بالتتابع."""
        current_response = initial_response

        for principle in self.principles:
            critique = self.model.generate(
                f"السؤال الأصلي: {prompt}\n"
                f"الاستجابة: {current_response}\n\n"
                f"{principle['critique_prompt']}"
            )

            # إن لم يجد النقد مشكلة جوهرية، انتقل للمبدأ التالي دون تعديل
            if self._critique_indicates_no_issue(critique):
                continue

            current_response = self.model.generate(
                f"الاستجابة: {current_response}\n"
                f"النقد: {critique}\n\n"
                f"{principle['revision_prompt']}"
            )

        return current_response

    def _critique_indicates_no_issue(self, critique_text: str) -> bool:
        no_issue_markers = ["لا توجد مشكلة", "الاستجابة آمنة", "لا حاجة للتعديل"]
        return any(marker in critique_text for marker in no_issue_markers)

    def build_sft_dataset(self, prompts: list) -> list:
        """بناء مجموعة بيانات SFT من أزواج (prompt, استجابة مُنقَّحة دستوريًا)."""
        dataset = []
        for prompt in prompts:
            initial = self.model.generate(prompt)
            revised = self.self_critique_and_revise(prompt, initial)
            dataset.append({"prompt": prompt, "response": revised})
        return dataset

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) لتفكيك التنشيطات المتشابكة (Polysemantic)
    إلى "ميزات" أحادية المعنى (Monosemantic Features) — المنهجية
    المستخدمة في ورقة Anthropic "Towards Monosemanticity" و"Scaling
    Monosemanticity" (اكتشاف ميزات Claude 3 Sonnet).
    """
    def __init__(self, activation_dim: int, dictionary_size: int,
                 sparsity_coefficient: float = 1e-3):
        super().__init__()
        # dictionary_size عادة أكبر بكثير من activation_dim (overcomplete)
        # مثال: activation_dim=4096, dictionary_size=16384 أو حتى مليون+
        self.encoder = nn.Linear(activation_dim, dictionary_size, bias=True)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=True)
        self.sparsity_coefficient = sparsity_coefficient

        # تهيئة الـ decoder weights لتكون متعامدة قدر الإمكان (ممارسة شائعة)
        nn.init.orthogonal_(self.decoder.weight)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def forward(self, activations: torch.Tensor):
        """
        activations: شكلها [batch, activation_dim] — تنشيطات طبقة معينة
        من النموذج الأصلي (غالبًا residual stream أو MLP output)
        """
        # الترميز مع ReLU لفرض الإيجابية والندرة (sparsity)
        features = torch.relu(self.encoder(activations))  # [batch, dictionary_size]

        # إعادة البناء
        reconstruction = self.decoder(features)  # [batch, activation_dim]

        return reconstruction, features

    def compute_loss(self, activations: torch.Tensor):
        """
        دالة الخسارة المركّبة: خطأ إعادة البناء + عقوبة الندرة (L1)
        """
        reconstruction, features = self.forward(activations)

        # خسارة إعادة البناء (MSE)
        reconstruction_loss = torch.nn.functional.mse_loss(
            reconstruction, activations
        )

        # عقوبة الندرة L1: تُجبر معظم الميزات على أن تكون صفرًا
        # لكل عيّنة — هذا ما يُنتج "ميزات أحادية المعنى" بدل تنشيط كثيف
        sparsity_loss = features.abs().sum(dim=-1).mean()

        total_loss = reconstruction_loss + self.sparsity_coefficient * sparsity_loss

        # نسبة الندرة الفعلية (للمراقبة أثناء التدريب)
        with torch.no_grad():
            l0_sparsity = (features > 0).float().sum(dim=-1).mean()

        return total_loss, {
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "avg_active_features": l0_sparsity.item(),  # الهدف: رقم صغير جدًا
        }

def interpret_feature(sae: SparseAutoencoder, feature_index: int,
                       text_dataset: list, model, layer_name: str,
                       top_k: int = 20) -> list:
    """
    تفسير ميزة محددة: إيجاد أعلى K نصوص تُنشِّط هذه الميزة بأعلى قيمة،
    لفهم "ماذا تمثّل" هذه الميزة دلاليًا (مثال: قد نكتشف أن الميزة #4821
    تُنشَّط حصريًا عند ذكر "خيانة الثقة" في أي سياق - عربي أو مجازي).
    """
    activations_cache = []
    hook = model.get_layer(layer_name).register_forward_hook(
        lambda module, inp, out: activations_cache.append(out.detach())
    )

    feature_activations = []
    for text in text_dataset:
        activations_cache.clear()
        _ = model(text)
        layer_activation = activations_cache[0]
        _, features = sae.forward(layer_activation)
        max_activation = features[:, feature_index].max().item()
        feature_activations.append((text, max_activation))

    hook.remove()
    feature_activations.sort(key=lambda x: x[1], reverse=True)
    return feature_activations[:top_k]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-61-ch61-advanced-alignment.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
