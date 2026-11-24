#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقطير المعرفة: Knowledge Distillation
Mastering Transformers and Large Language Models — Knowledge Distillation

Book slug : mastering-llms-transformers
Chapter   : ch-25-ch25-knowledge-distillation (lesson 25)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-25-ch25-knowledge-distillation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقطير المعرفة: Knowledge Distillation
Mastering Transformers and Large Language Models — Knowledge Distillation

Book slug : mastering-llms-transformers
Chapter   : ch-25-ch25-knowledge-distillation (lesson 25)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-25-ch25-knowledge-distillation.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقطير المعرفة: Knowledge Distillation
Mastering Transformers and Large Language Models — Knowledge Distillation

Book slug : mastering-llms-transformers
Chapter   : ch-25-ch25-knowledge-distillation (lesson 25)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    التطبيق الكامل لخسارة التقطير المعرفي الكلاسيكية (Hinton et al., 2015)
    مع دمج Hard Label Loss وSoft Label Loss، قابلة للتطبيق مباشرة على
    نماذج اللغة الكبيرة (Response-Based Distillation على مستوى الرمز).
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # وزن Hard Loss؛ (1-alpha) لـ Soft Loss

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> dict:
        """
        student_logits, teacher_logits: [batch, seq_len, vocab_size]
        true_labels: [batch, seq_len] (رموز الهدف الحقيقية - Hard Labels)
        """
        vocab_size = student_logits.size(-1)

        # --- 1. Hard Label Loss (كما في SFT عادي) ---
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            true_labels.view(-1),
            ignore_index=-100,  # لتجاهل رموز الحشو (padding)
            reduction="mean",
        )

        # --- 2. Soft Label Loss (جوهر التقطير - KL Divergence) ---
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        # يجب فصل تدرج المُعلِّم - لا نُحدِّث أوزانه إطلاقًا
        teacher_probs = teacher_probs.detach()

        # KL Divergence لكل رمز، مع تجاهل رموز الحشو عبر attention_mask
        soft_loss_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)  # [batch, seq_len]

        if attention_mask is not None:
            soft_loss = (soft_loss_per_token * attention_mask).sum() / attention_mask.sum()
        else:
            soft_loss = soft_loss_per_token.mean()

        # الضرب بـ T² ضروري لموازنة مقياس التدرجات (كما أثبت Hinton et al.)
        # لأن تدرجات soft loss تتناسب عكسيًا مع T²
        soft_loss = soft_loss * (self.temperature ** 2)

        # --- 3. الدمج ---
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
        }

class ResponseBasedDistillationTrainer:
    """خط أنابيب تدريب كامل للتقطير القائم على الاستجابة لنماذج اللغة."""
    def __init__(self, teacher_model, student_model, tokenizer,
                 temperature: float = 4.0, alpha: float = 0.3):
        self.teacher = teacher_model.eval()  # المُعلِّم دائمًا في وضع التقييم
        self.student = student_model
        self.tokenizer = tokenizer
        self.loss_fn = KnowledgeDistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=2e-5)

        # تجميد كامل لأوزان المُعلِّم - توفير ذاكرة وضمان عدم التحديث
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits

        student_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_logits = student_outputs.logits

        losses = self.loss_fn(
            student_logits, teacher_logits, labels, attention_mask
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return losses

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class FeatureBasedDistillationLoss(nn.Module):
    """
    تقطير قائم على الميزات (بأسلوب TinyBERT): مطابقة Hidden States
    بين طبقات مُحدَّدة في المُعلِّم والتلميذ، مع إسقاط خطي (Linear
    Projection) للتعويض عن اختلاف الأبعاد بين النموذجين.
    """
    def __init__(self, teacher_hidden_dim: int, student_hidden_dim: int,
                 layer_mapping: dict):
        """
        layer_mapping: قاموس يُحدِّد أي طبقة من التلميذ تُقارَن بأي طبقة
        من المُعلِّم. مثال: {0: 0, 1: 4, 2: 8, 3: 12} يعني طبقة التلميذ 0
        تُقارَن بطبقة المُعلِّم 0، وطبقة التلميذ 1 تُقارَن بطبقة المُعلِّم 4
        (تخطي طبقات - شائع عند ضغط نموذج 12 طبقة إلى 4 طبقات مثلًا)
        """
        super().__init__()
        self.layer_mapping = layer_mapping
        # إسقاط خطي لمحاذاة أبعاد التلميذ (أصغر عادة) مع أبعاد المُعلِّم
        self.projections = nn.ModuleDict({
            str(s_layer): nn.Linear(student_hidden_dim, teacher_hidden_dim)
            for s_layer in layer_mapping.keys()
        })

    def forward(self, student_hidden_states: list, teacher_hidden_states: list):
        """
        student_hidden_states, teacher_hidden_states: قوائم من تنسورات
        [batch, seq_len, hidden_dim] لكل طبقة من طبقات كل نموذج.
        """
        total_feature_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_hidden = student_hidden_states[student_layer_idx]
            teacher_hidden = teacher_hidden_states[teacher_layer_idx].detach()

            # إسقاط تمثيل التلميذ لنفس بُعد المُعلِّم
            projected_student = self.projections[str(student_layer_idx)](
                student_hidden
            )

            # خسارة MSE بين التمثيلين المُحاذَيين
            layer_loss = F.mse_loss(projected_student, teacher_hidden)
            total_feature_loss += layer_loss

        return total_feature_loss / len(self.layer_mapping)

class AttentionBasedDistillationLoss(nn.Module):
    """
    تقطير خرائط الانتباه نفسها (Attention Map Distillation - أسلوب
    TinyBERT المتقدم): إجبار التلميذ على تعلّم "أين ينظر" المُعلِّم،
    وليس فقط ماذا ينتج - هذا ينقل معرفة بنيوية أعمق عن العلاقات
    اللغوية بين الرموز.
    """
    def __init__(self, layer_mapping: dict):
        super().__init__()
        self.layer_mapping = layer_mapping

    def forward(self, student_attentions: list, teacher_attentions: list):
        """
        student_attentions, teacher_attentions: قوائم تنسورات
        [batch, n_heads, seq_len, seq_len] لكل طبقة.
        """
        total_attn_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_attn = student_attentions[student_layer_idx]
            teacher_attn = teacher_attentions[teacher_layer_idx].detach()

            # قد يختلف عدد الرؤوس بين النموذجين - متوسط عبر الرؤوس كحل مبسّط
            student_attn_avg = student_attn.mean(dim=1)
            teacher_attn_avg = teacher_attn.mean(dim=1)

            # MSE بين خرائط الانتباه (بعد قص القيم الصغيرة جدًا لتفادي عدم استقرار log)
            attn_loss = F.mse_loss(student_attn_avg, teacher_attn_avg)
            total_attn_loss += attn_loss

        return total_attn_loss / len(self.layer_mapping)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ReasoningDistillationPipeline:
    """
    خط أنابيب لتقطير قدرة الاستدلال من نموذج استدلال كبير (مثل
    DeepSeek-R1) إلى نموذج أصغر، بأسلوب SFT على سلاسل تفكير مُولَّدة
    من المُعلِّم (وليس عبر RL مباشرة على التلميذ - أرخص بكثير).
    """
    def __init__(self, teacher_reasoning_model, student_model,
                 verification_fn):
        self.teacher = teacher_reasoning_model
        self.student = student_model
        self.verify = verification_fn  # دالة تحقق قابلة للبرمجة (الفصل 60)

    def generate_verified_reasoning_dataset(self, problems: list,
                                             n_samples_per_problem: int = 4) -> list:
        """
        توليد سلاسل تفكير من المُعلِّم لكل مسألة، والاحتفاظ فقط بالسلاسل
        التي تُفضي لإجابة صحيحة (Rejection Sampling) - هذا يضمن أن
        التلميذ يتعلّم فقط من أنماط تفكير "ناجحة" فعليًا، لا عشوائية.
        """
        verified_dataset = []
        for problem in problems:
            correct_reasoning_found = False
            for _ in range(n_samples_per_problem):
                reasoning_trace = self.teacher.generate(
                    problem["question"], temperature=0.7, max_tokens=4096
                )
                is_correct = self.verify(reasoning_trace, problem["ground_truth"])
                if is_correct:
                    verified_dataset.append({
                        "question": problem["question"],
                        "reasoning_trace": reasoning_trace,
                    })
                    correct_reasoning_found = True
                    break  # نكتفي بأول سلسلة صحيحة لكل مسألة (اقتصاد في الحوسبة)

            if not correct_reasoning_found:
                # تجاهل المسائل التي فشل المُعلِّم نفسه في حلها بشكل صحيح
                continue

        return verified_dataset

    def distill_via_sft(self, verified_dataset: list, num_epochs: int = 3):
        """
        التقطير هنا يتم عبر SFT عادي (الجزء 7) على سلاسل التفكير
        المُولَّدة والمُتحقَّق منها - وليس عبر مطابقة logits كما في
        القسم 3.1. هذا النهج أثبت فعالية مذهلة في تقارير DeepSeek-R1:
        نموذج Qwen 7B مُقطَّر بهذه الطريقة تفوّق على بعض النماذج
        الأكبر بكثير المُدرَّبة بـ RL مباشرة عليها.
        """
        for epoch in range(num_epochs):
            for example in verified_dataset:
                full_text = (
                    f"السؤال: {example['question']}\n"
                    f"{example['reasoning_trace']}"
                )
                # تدريب SFT قياسي - راجع الجزء السابع للتفاصيل الكاملة
                self._sft_step(full_text)

    def _sft_step(self, text: str):
        raise NotImplementedError("يُطبَّق عبر حلقة SFT قياسية - راجع الجزء السابع")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
teacher = GPT2LMHeadModel.from_pretrained("gpt2-large")   # 774M معامل
student = GPT2LMHeadModel.from_pretrained("gpt2")         # 124M معامل

trainer = ResponseBasedDistillationTrainer(
    teacher_model=teacher, student_model=student, tokenizer=tokenizer,
    temperature=4.0, alpha=0.3,
)

# محاكاة خطوة تدريب واحدة

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    التطبيق الكامل لخسارة التقطير المعرفي الكلاسيكية (Hinton et al., 2015)
    مع دمج Hard Label Loss وSoft Label Loss، قابلة للتطبيق مباشرة على
    نماذج اللغة الكبيرة (Response-Based Distillation على مستوى الرمز).
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # وزن Hard Loss؛ (1-alpha) لـ Soft Loss

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> dict:
        """
        student_logits, teacher_logits: [batch, seq_len, vocab_size]
        true_labels: [batch, seq_len] (رموز الهدف الحقيقية - Hard Labels)
        """
        vocab_size = student_logits.size(-1)

        # --- 1. Hard Label Loss (كما في SFT عادي) ---
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            true_labels.view(-1),
            ignore_index=-100,  # لتجاهل رموز الحشو (padding)
            reduction="mean",
        )

        # --- 2. Soft Label Loss (جوهر التقطير - KL Divergence) ---
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        # يجب فصل تدرج المُعلِّم - لا نُحدِّث أوزانه إطلاقًا
        teacher_probs = teacher_probs.detach()

        # KL Divergence لكل رمز، مع تجاهل رموز الحشو عبر attention_mask
        soft_loss_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)  # [batch, seq_len]

        if attention_mask is not None:
            soft_loss = (soft_loss_per_token * attention_mask).sum() / attention_mask.sum()
        else:
            soft_loss = soft_loss_per_token.mean()

        # الضرب بـ T² ضروري لموازنة مقياس التدرجات (كما أثبت Hinton et al.)
        # لأن تدرجات soft loss تتناسب عكسيًا مع T²
        soft_loss = soft_loss * (self.temperature ** 2)

        # --- 3. الدمج ---
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
        }

class ResponseBasedDistillationTrainer:
    """خط أنابيب تدريب كامل للتقطير القائم على الاستجابة لنماذج اللغة."""
    def __init__(self, teacher_model, student_model, tokenizer,
                 temperature: float = 4.0, alpha: float = 0.3):
        self.teacher = teacher_model.eval()  # المُعلِّم دائمًا في وضع التقييم
        self.student = student_model
        self.tokenizer = tokenizer
        self.loss_fn = KnowledgeDistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=2e-5)

        # تجميد كامل لأوزان المُعلِّم - توفير ذاكرة وضمان عدم التحديث
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits

        student_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_logits = student_outputs.logits

        losses = self.loss_fn(
            student_logits, teacher_logits, labels, attention_mask
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return losses

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class FeatureBasedDistillationLoss(nn.Module):
    """
    تقطير قائم على الميزات (بأسلوب TinyBERT): مطابقة Hidden States
    بين طبقات مُحدَّدة في المُعلِّم والتلميذ، مع إسقاط خطي (Linear
    Projection) للتعويض عن اختلاف الأبعاد بين النموذجين.
    """
    def __init__(self, teacher_hidden_dim: int, student_hidden_dim: int,
                 layer_mapping: dict):
        """
        layer_mapping: قاموس يُحدِّد أي طبقة من التلميذ تُقارَن بأي طبقة
        من المُعلِّم. مثال: {0: 0, 1: 4, 2: 8, 3: 12} يعني طبقة التلميذ 0
        تُقارَن بطبقة المُعلِّم 0، وطبقة التلميذ 1 تُقارَن بطبقة المُعلِّم 4
        (تخطي طبقات - شائع عند ضغط نموذج 12 طبقة إلى 4 طبقات مثلًا)
        """
        super().__init__()
        self.layer_mapping = layer_mapping
        # إسقاط خطي لمحاذاة أبعاد التلميذ (أصغر عادة) مع أبعاد المُعلِّم
        self.projections = nn.ModuleDict({
            str(s_layer): nn.Linear(student_hidden_dim, teacher_hidden_dim)
            for s_layer in layer_mapping.keys()
        })

    def forward(self, student_hidden_states: list, teacher_hidden_states: list):
        """
        student_hidden_states, teacher_hidden_states: قوائم من تنسورات
        [batch, seq_len, hidden_dim] لكل طبقة من طبقات كل نموذج.
        """
        total_feature_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_hidden = student_hidden_states[student_layer_idx]
            teacher_hidden = teacher_hidden_states[teacher_layer_idx].detach()

            # إسقاط تمثيل التلميذ لنفس بُعد المُعلِّم
            projected_student = self.projections[str(student_layer_idx)](
                student_hidden
            )

            # خسارة MSE بين التمثيلين المُحاذَيين
            layer_loss = F.mse_loss(projected_student, teacher_hidden)
            total_feature_loss += layer_loss

        return total_feature_loss / len(self.layer_mapping)

class AttentionBasedDistillationLoss(nn.Module):
    """
    تقطير خرائط الانتباه نفسها (Attention Map Distillation - أسلوب
    TinyBERT المتقدم): إجبار التلميذ على تعلّم "أين ينظر" المُعلِّم،
    وليس فقط ماذا ينتج - هذا ينقل معرفة بنيوية أعمق عن العلاقات
    اللغوية بين الرموز.
    """
    def __init__(self, layer_mapping: dict):
        super().__init__()
        self.layer_mapping = layer_mapping

    def forward(self, student_attentions: list, teacher_attentions: list):
        """
        student_attentions, teacher_attentions: قوائم تنسورات
        [batch, n_heads, seq_len, seq_len] لكل طبقة.
        """
        total_attn_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_attn = student_attentions[student_layer_idx]
            teacher_attn = teacher_attentions[teacher_layer_idx].detach()

            # قد يختلف عدد الرؤوس بين النموذجين - متوسط عبر الرؤوس كحل مبسّط
            student_attn_avg = student_attn.mean(dim=1)
            teacher_attn_avg = teacher_attn.mean(dim=1)

            # MSE بين خرائط الانتباه (بعد قص القيم الصغيرة جدًا لتفادي عدم استقرار log)
            attn_loss = F.mse_loss(student_attn_avg, teacher_attn_avg)
            total_attn_loss += attn_loss

        return total_attn_loss / len(self.layer_mapping)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ReasoningDistillationPipeline:
    """
    خط أنابيب لتقطير قدرة الاستدلال من نموذج استدلال كبير (مثل
    DeepSeek-R1) إلى نموذج أصغر، بأسلوب SFT على سلاسل تفكير مُولَّدة
    من المُعلِّم (وليس عبر RL مباشرة على التلميذ - أرخص بكثير).
    """
    def __init__(self, teacher_reasoning_model, student_model,
                 verification_fn):
        self.teacher = teacher_reasoning_model
        self.student = student_model
        self.verify = verification_fn  # دالة تحقق قابلة للبرمجة (الفصل 60)

    def generate_verified_reasoning_dataset(self, problems: list,
                                             n_samples_per_problem: int = 4) -> list:
        """
        توليد سلاسل تفكير من المُعلِّم لكل مسألة، والاحتفاظ فقط بالسلاسل
        التي تُفضي لإجابة صحيحة (Rejection Sampling) - هذا يضمن أن
        التلميذ يتعلّم فقط من أنماط تفكير "ناجحة" فعليًا، لا عشوائية.
        """
        verified_dataset = []
        for problem in problems:
            correct_reasoning_found = False
            for _ in range(n_samples_per_problem):
                reasoning_trace = self.teacher.generate(
                    problem["question"], temperature=0.7, max_tokens=4096
                )
                is_correct = self.verify(reasoning_trace, problem["ground_truth"])
                if is_correct:
                    verified_dataset.append({
                        "question": problem["question"],
                        "reasoning_trace": reasoning_trace,
                    })
                    correct_reasoning_found = True
                    break  # نكتفي بأول سلسلة صحيحة لكل مسألة (اقتصاد في الحوسبة)

            if not correct_reasoning_found:
                # تجاهل المسائل التي فشل المُعلِّم نفسه في حلها بشكل صحيح
                continue

        return verified_dataset

    def distill_via_sft(self, verified_dataset: list, num_epochs: int = 3):
        """
        التقطير هنا يتم عبر SFT عادي (الجزء 7) على سلاسل التفكير
        المُولَّدة والمُتحقَّق منها - وليس عبر مطابقة logits كما في
        القسم 3.1. هذا النهج أثبت فعالية مذهلة في تقارير DeepSeek-R1:
        نموذج Qwen 7B مُقطَّر بهذه الطريقة تفوّق على بعض النماذج
        الأكبر بكثير المُدرَّبة بـ RL مباشرة عليها.
        """
        for epoch in range(num_epochs):
            for example in verified_dataset:
                full_text = (
                    f"السؤال: {example['question']}\n"
                    f"{example['reasoning_trace']}"
                )
                # تدريب SFT قياسي - راجع الجزء السابع للتفاصيل الكاملة
                self._sft_step(full_text)

    def _sft_step(self, text: str):
        raise NotImplementedError("يُطبَّق عبر حلقة SFT قياسية - راجع الجزء السابع")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
teacher = GPT2LMHeadModel.from_pretrained("gpt2-large")   # 774M معامل
student = GPT2LMHeadModel.from_pretrained("gpt2")         # 124M معامل

trainer = ResponseBasedDistillationTrainer(
    teacher_model=teacher, student_model=student, tokenizer=tokenizer,
    temperature=4.0, alpha=0.3,
)

# محاكاة خطوة تدريب واحدة

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — تقطير المعرفة: Knowledge Distillation
Mastering Transformers and Large Language Models — Knowledge Distillation

Book slug : mastering-llms-transformers
Chapter   : ch-25-ch25-knowledge-distillation (lesson 25)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-1
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [002] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    التطبيق الكامل لخسارة التقطير المعرفي الكلاسيكية (Hinton et al., 2015)
    مع دمج Hard Label Loss وSoft Label Loss، قابلة للتطبيق مباشرة على
    نماذج اللغة الكبيرة (Response-Based Distillation على مستوى الرمز).
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # وزن Hard Loss؛ (1-alpha) لـ Soft Loss

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> dict:
        """
        student_logits, teacher_logits: [batch, seq_len, vocab_size]
        true_labels: [batch, seq_len] (رموز الهدف الحقيقية - Hard Labels)
        """
        vocab_size = student_logits.size(-1)

        # --- 1. Hard Label Loss (كما في SFT عادي) ---
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            true_labels.view(-1),
            ignore_index=-100,  # لتجاهل رموز الحشو (padding)
            reduction="mean",
        )

        # --- 2. Soft Label Loss (جوهر التقطير - KL Divergence) ---
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        # يجب فصل تدرج المُعلِّم - لا نُحدِّث أوزانه إطلاقًا
        teacher_probs = teacher_probs.detach()

        # KL Divergence لكل رمز، مع تجاهل رموز الحشو عبر attention_mask
        soft_loss_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)  # [batch, seq_len]

        if attention_mask is not None:
            soft_loss = (soft_loss_per_token * attention_mask).sum() / attention_mask.sum()
        else:
            soft_loss = soft_loss_per_token.mean()

        # الضرب بـ T² ضروري لموازنة مقياس التدرجات (كما أثبت Hinton et al.)
        # لأن تدرجات soft loss تتناسب عكسيًا مع T²
        soft_loss = soft_loss * (self.temperature ** 2)

        # --- 3. الدمج ---
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
        }

class ResponseBasedDistillationTrainer:
    """خط أنابيب تدريب كامل للتقطير القائم على الاستجابة لنماذج اللغة."""
    def __init__(self, teacher_model, student_model, tokenizer,
                 temperature: float = 4.0, alpha: float = 0.3):
        self.teacher = teacher_model.eval()  # المُعلِّم دائمًا في وضع التقييم
        self.student = student_model
        self.tokenizer = tokenizer
        self.loss_fn = KnowledgeDistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=2e-5)

        # تجميد كامل لأوزان المُعلِّم - توفير ذاكرة وضمان عدم التحديث
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits

        student_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_logits = student_outputs.logits

        losses = self.loss_fn(
            student_logits, teacher_logits, labels, attention_mask
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return losses

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class FeatureBasedDistillationLoss(nn.Module):
    """
    تقطير قائم على الميزات (بأسلوب TinyBERT): مطابقة Hidden States
    بين طبقات مُحدَّدة في المُعلِّم والتلميذ، مع إسقاط خطي (Linear
    Projection) للتعويض عن اختلاف الأبعاد بين النموذجين.
    """
    def __init__(self, teacher_hidden_dim: int, student_hidden_dim: int,
                 layer_mapping: dict):
        """
        layer_mapping: قاموس يُحدِّد أي طبقة من التلميذ تُقارَن بأي طبقة
        من المُعلِّم. مثال: {0: 0, 1: 4, 2: 8, 3: 12} يعني طبقة التلميذ 0
        تُقارَن بطبقة المُعلِّم 0، وطبقة التلميذ 1 تُقارَن بطبقة المُعلِّم 4
        (تخطي طبقات - شائع عند ضغط نموذج 12 طبقة إلى 4 طبقات مثلًا)
        """
        super().__init__()
        self.layer_mapping = layer_mapping
        # إسقاط خطي لمحاذاة أبعاد التلميذ (أصغر عادة) مع أبعاد المُعلِّم
        self.projections = nn.ModuleDict({
            str(s_layer): nn.Linear(student_hidden_dim, teacher_hidden_dim)
            for s_layer in layer_mapping.keys()
        })

    def forward(self, student_hidden_states: list, teacher_hidden_states: list):
        """
        student_hidden_states, teacher_hidden_states: قوائم من تنسورات
        [batch, seq_len, hidden_dim] لكل طبقة من طبقات كل نموذج.
        """
        total_feature_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_hidden = student_hidden_states[student_layer_idx]
            teacher_hidden = teacher_hidden_states[teacher_layer_idx].detach()

            # إسقاط تمثيل التلميذ لنفس بُعد المُعلِّم
            projected_student = self.projections[str(student_layer_idx)](
                student_hidden
            )

            # خسارة MSE بين التمثيلين المُحاذَيين
            layer_loss = F.mse_loss(projected_student, teacher_hidden)
            total_feature_loss += layer_loss

        return total_feature_loss / len(self.layer_mapping)

class AttentionBasedDistillationLoss(nn.Module):
    """
    تقطير خرائط الانتباه نفسها (Attention Map Distillation - أسلوب
    TinyBERT المتقدم): إجبار التلميذ على تعلّم "أين ينظر" المُعلِّم،
    وليس فقط ماذا ينتج - هذا ينقل معرفة بنيوية أعمق عن العلاقات
    اللغوية بين الرموز.
    """
    def __init__(self, layer_mapping: dict):
        super().__init__()
        self.layer_mapping = layer_mapping

    def forward(self, student_attentions: list, teacher_attentions: list):
        """
        student_attentions, teacher_attentions: قوائم تنسورات
        [batch, n_heads, seq_len, seq_len] لكل طبقة.
        """
        total_attn_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_attn = student_attentions[student_layer_idx]
            teacher_attn = teacher_attentions[teacher_layer_idx].detach()

            # قد يختلف عدد الرؤوس بين النموذجين - متوسط عبر الرؤوس كحل مبسّط
            student_attn_avg = student_attn.mean(dim=1)
            teacher_attn_avg = teacher_attn.mean(dim=1)

            # MSE بين خرائط الانتباه (بعد قص القيم الصغيرة جدًا لتفادي عدم استقرار log)
            attn_loss = F.mse_loss(student_attn_avg, teacher_attn_avg)
            total_attn_loss += attn_loss

        return total_attn_loss / len(self.layer_mapping)

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ReasoningDistillationPipeline:
    """
    خط أنابيب لتقطير قدرة الاستدلال من نموذج استدلال كبير (مثل
    DeepSeek-R1) إلى نموذج أصغر، بأسلوب SFT على سلاسل تفكير مُولَّدة
    من المُعلِّم (وليس عبر RL مباشرة على التلميذ - أرخص بكثير).
    """
    def __init__(self, teacher_reasoning_model, student_model,
                 verification_fn):
        self.teacher = teacher_reasoning_model
        self.student = student_model
        self.verify = verification_fn  # دالة تحقق قابلة للبرمجة (الفصل 60)

    def generate_verified_reasoning_dataset(self, problems: list,
                                             n_samples_per_problem: int = 4) -> list:
        """
        توليد سلاسل تفكير من المُعلِّم لكل مسألة، والاحتفاظ فقط بالسلاسل
        التي تُفضي لإجابة صحيحة (Rejection Sampling) - هذا يضمن أن
        التلميذ يتعلّم فقط من أنماط تفكير "ناجحة" فعليًا، لا عشوائية.
        """
        verified_dataset = []
        for problem in problems:
            correct_reasoning_found = False
            for _ in range(n_samples_per_problem):
                reasoning_trace = self.teacher.generate(
                    problem["question"], temperature=0.7, max_tokens=4096
                )
                is_correct = self.verify(reasoning_trace, problem["ground_truth"])
                if is_correct:
                    verified_dataset.append({
                        "question": problem["question"],
                        "reasoning_trace": reasoning_trace,
                    })
                    correct_reasoning_found = True
                    break  # نكتفي بأول سلسلة صحيحة لكل مسألة (اقتصاد في الحوسبة)

            if not correct_reasoning_found:
                # تجاهل المسائل التي فشل المُعلِّم نفسه في حلها بشكل صحيح
                continue

        return verified_dataset

    def distill_via_sft(self, verified_dataset: list, num_epochs: int = 3):
        """
        التقطير هنا يتم عبر SFT عادي (الجزء 7) على سلاسل التفكير
        المُولَّدة والمُتحقَّق منها - وليس عبر مطابقة logits كما في
        القسم 3.1. هذا النهج أثبت فعالية مذهلة في تقارير DeepSeek-R1:
        نموذج Qwen 7B مُقطَّر بهذه الطريقة تفوّق على بعض النماذج
        الأكبر بكثير المُدرَّبة بـ RL مباشرة عليها.
        """
        for epoch in range(num_epochs):
            for example in verified_dataset:
                full_text = (
                    f"السؤال: {example['question']}\n"
                    f"{example['reasoning_trace']}"
                )
                # تدريب SFT قياسي - راجع الجزء السابع للتفاصيل الكاملة
                self._sft_step(full_text)

    def _sft_step(self, text: str):
        raise NotImplementedError("يُطبَّق عبر حلقة SFT قياسية - راجع الجزء السابع")

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
teacher = GPT2LMHeadModel.from_pretrained("gpt2-large")   # 774M معامل
student = GPT2LMHeadModel.from_pretrained("gpt2")         # 124M معامل

trainer = ResponseBasedDistillationTrainer(
    teacher_model=teacher, student_model=student, tokenizer=tokenizer,
    temperature=4.0, alpha=0.3,
)

# محاكاة خطوة تدريب واحدة

# ────────────────────────────────────────────────────────────────────────
# [006] content/notebooks/part-06-model-reuse/ch25-knowledge-distillation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-2
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
    ("torch", "torch"),
    ("transformers", "transformers")
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    التطبيق الكامل لخسارة التقطير المعرفي الكلاسيكية (Hinton et al., 2015)
    مع دمج Hard Label Loss وSoft Label Loss، قابلة للتطبيق مباشرة على
    نماذج اللغة الكبيرة (Response-Based Distillation على مستوى الرمز).
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # وزن Hard Loss؛ (1-alpha) لـ Soft Loss

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> dict:
        """
        student_logits, teacher_logits: [batch, seq_len, vocab_size]
        true_labels: [batch, seq_len] (رموز الهدف الحقيقية - Hard Labels)
        """
        vocab_size = student_logits.size(-1)

        # --- 1. Hard Label Loss (كما في SFT عادي) ---
        hard_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            true_labels.view(-1),
            ignore_index=-100,  # لتجاهل رموز الحشو (padding)
            reduction="mean",
        )

        # --- 2. Soft Label Loss (جوهر التقطير - KL Divergence) ---
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1
        )
        # يجب فصل تدرج المُعلِّم - لا نُحدِّث أوزانه إطلاقًا
        teacher_probs = teacher_probs.detach()

        # KL Divergence لكل رمز، مع تجاهل رموز الحشو عبر attention_mask
        soft_loss_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)  # [batch, seq_len]

        if attention_mask is not None:
            soft_loss = (soft_loss_per_token * attention_mask).sum() / attention_mask.sum()
        else:
            soft_loss = soft_loss_per_token.mean()

        # الضرب بـ T² ضروري لموازنة مقياس التدرجات (كما أثبت Hinton et al.)
        # لأن تدرجات soft loss تتناسب عكسيًا مع T²
        soft_loss = soft_loss * (self.temperature ** 2)

        # --- 3. الدمج ---
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss.item(),
            "soft_loss": soft_loss.item(),
        }

class ResponseBasedDistillationTrainer:
    """خط أنابيب تدريب كامل للتقطير القائم على الاستجابة لنماذج اللغة."""
    def __init__(self, teacher_model, student_model, tokenizer,
                 temperature: float = 4.0, alpha: float = 0.3):
        self.teacher = teacher_model.eval()  # المُعلِّم دائمًا في وضع التقييم
        self.student = student_model
        self.tokenizer = tokenizer
        self.loss_fn = KnowledgeDistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=2e-5)

        # تجميد كامل لأوزان المُعلِّم - توفير ذاكرة وضمان عدم التحديث
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits

        student_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_logits = student_outputs.logits

        losses = self.loss_fn(
            student_logits, teacher_logits, labels, attention_mask
        )

        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return losses

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class FeatureBasedDistillationLoss(nn.Module):
    """
    تقطير قائم على الميزات (بأسلوب TinyBERT): مطابقة Hidden States
    بين طبقات مُحدَّدة في المُعلِّم والتلميذ، مع إسقاط خطي (Linear
    Projection) للتعويض عن اختلاف الأبعاد بين النموذجين.
    """
    def __init__(self, teacher_hidden_dim: int, student_hidden_dim: int,
                 layer_mapping: dict):
        """
        layer_mapping: قاموس يُحدِّد أي طبقة من التلميذ تُقارَن بأي طبقة
        من المُعلِّم. مثال: {0: 0, 1: 4, 2: 8, 3: 12} يعني طبقة التلميذ 0
        تُقارَن بطبقة المُعلِّم 0، وطبقة التلميذ 1 تُقارَن بطبقة المُعلِّم 4
        (تخطي طبقات - شائع عند ضغط نموذج 12 طبقة إلى 4 طبقات مثلًا)
        """
        super().__init__()
        self.layer_mapping = layer_mapping
        # إسقاط خطي لمحاذاة أبعاد التلميذ (أصغر عادة) مع أبعاد المُعلِّم
        self.projections = nn.ModuleDict({
            str(s_layer): nn.Linear(student_hidden_dim, teacher_hidden_dim)
            for s_layer in layer_mapping.keys()
        })

    def forward(self, student_hidden_states: list, teacher_hidden_states: list):
        """
        student_hidden_states, teacher_hidden_states: قوائم من تنسورات
        [batch, seq_len, hidden_dim] لكل طبقة من طبقات كل نموذج.
        """
        total_feature_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_hidden = student_hidden_states[student_layer_idx]
            teacher_hidden = teacher_hidden_states[teacher_layer_idx].detach()

            # إسقاط تمثيل التلميذ لنفس بُعد المُعلِّم
            projected_student = self.projections[str(student_layer_idx)](
                student_hidden
            )

            # خسارة MSE بين التمثيلين المُحاذَيين
            layer_loss = F.mse_loss(projected_student, teacher_hidden)
            total_feature_loss += layer_loss

        return total_feature_loss / len(self.layer_mapping)

class AttentionBasedDistillationLoss(nn.Module):
    """
    تقطير خرائط الانتباه نفسها (Attention Map Distillation - أسلوب
    TinyBERT المتقدم): إجبار التلميذ على تعلّم "أين ينظر" المُعلِّم،
    وليس فقط ماذا ينتج - هذا ينقل معرفة بنيوية أعمق عن العلاقات
    اللغوية بين الرموز.
    """
    def __init__(self, layer_mapping: dict):
        super().__init__()
        self.layer_mapping = layer_mapping

    def forward(self, student_attentions: list, teacher_attentions: list):
        """
        student_attentions, teacher_attentions: قوائم تنسورات
        [batch, n_heads, seq_len, seq_len] لكل طبقة.
        """
        total_attn_loss = 0.0
        for student_layer_idx, teacher_layer_idx in self.layer_mapping.items():
            student_attn = student_attentions[student_layer_idx]
            teacher_attn = teacher_attentions[teacher_layer_idx].detach()

            # قد يختلف عدد الرؤوس بين النموذجين - متوسط عبر الرؤوس كحل مبسّط
            student_attn_avg = student_attn.mean(dim=1)
            teacher_attn_avg = teacher_attn.mean(dim=1)

            # MSE بين خرائط الانتباه (بعد قص القيم الصغيرة جدًا لتفادي عدم استقرار log)
            attn_loss = F.mse_loss(student_attn_avg, teacher_attn_avg)
            total_attn_loss += attn_loss

        return total_attn_loss / len(self.layer_mapping)

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class ReasoningDistillationPipeline:
    """
    خط أنابيب لتقطير قدرة الاستدلال من نموذج استدلال كبير (مثل
    DeepSeek-R1) إلى نموذج أصغر، بأسلوب SFT على سلاسل تفكير مُولَّدة
    من المُعلِّم (وليس عبر RL مباشرة على التلميذ - أرخص بكثير).
    """
    def __init__(self, teacher_reasoning_model, student_model,
                 verification_fn):
        self.teacher = teacher_reasoning_model
        self.student = student_model
        self.verify = verification_fn  # دالة تحقق قابلة للبرمجة (الفصل 60)

    def generate_verified_reasoning_dataset(self, problems: list,
                                             n_samples_per_problem: int = 4) -> list:
        """
        توليد سلاسل تفكير من المُعلِّم لكل مسألة، والاحتفاظ فقط بالسلاسل
        التي تُفضي لإجابة صحيحة (Rejection Sampling) - هذا يضمن أن
        التلميذ يتعلّم فقط من أنماط تفكير "ناجحة" فعليًا، لا عشوائية.
        """
        verified_dataset = []
        for problem in problems:
            correct_reasoning_found = False
            for _ in range(n_samples_per_problem):
                reasoning_trace = self.teacher.generate(
                    problem["question"], temperature=0.7, max_tokens=4096
                )
                is_correct = self.verify(reasoning_trace, problem["ground_truth"])
                if is_correct:
                    verified_dataset.append({
                        "question": problem["question"],
                        "reasoning_trace": reasoning_trace,
                    })
                    correct_reasoning_found = True
                    break  # نكتفي بأول سلسلة صحيحة لكل مسألة (اقتصاد في الحوسبة)

            if not correct_reasoning_found:
                # تجاهل المسائل التي فشل المُعلِّم نفسه في حلها بشكل صحيح
                continue

        return verified_dataset

    def distill_via_sft(self, verified_dataset: list, num_epochs: int = 3):
        """
        التقطير هنا يتم عبر SFT عادي (الجزء 7) على سلاسل التفكير
        المُولَّدة والمُتحقَّق منها - وليس عبر مطابقة logits كما في
        القسم 3.1. هذا النهج أثبت فعالية مذهلة في تقارير DeepSeek-R1:
        نموذج Qwen 7B مُقطَّر بهذه الطريقة تفوّق على بعض النماذج
        الأكبر بكثير المُدرَّبة بـ RL مباشرة عليها.
        """
        for epoch in range(num_epochs):
            for example in verified_dataset:
                full_text = (
                    f"السؤال: {example['question']}\n"
                    f"{example['reasoning_trace']}"
                )
                # تدريب SFT قياسي - راجع الجزء السابع للتفاصيل الكاملة
                self._sft_step(full_text)

    def _sft_step(self, text: str):
        raise NotImplementedError("يُطبَّق عبر حلقة SFT قياسية - راجع الجزء السابع")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
teacher = GPT2LMHeadModel.from_pretrained("gpt2-large")   # 774M معامل
student = GPT2LMHeadModel.from_pretrained("gpt2")         # 124M معامل

trainer = ResponseBasedDistillationTrainer(
    teacher_model=teacher, student_model=student, tokenizer=tokenizer,
    temperature=4.0, alpha=0.3,
)

# محاكاة خطوة تدريب واحدة

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/mastering-llms-transformers/notebooks/ch-25-ch25-knowledge-distillation.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
