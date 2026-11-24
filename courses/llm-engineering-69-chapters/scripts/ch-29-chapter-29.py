#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقطير المعرفي (Knowledge Distillation)
Large Language Model Engineering — Knowledge

Book slug : llm-engineering-69-chapters
Chapter   : ch-29-chapter-29 (lesson 29)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-29-chapter-29.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقطير المعرفي (Knowledge Distillation)
Large Language Model Engineering — Knowledge

Book slug : llm-engineering-69-chapters
Chapter   : ch-29-chapter-29 (lesson 29)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-29-chapter-29.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقطير المعرفي (Knowledge Distillation)
Large Language Model Engineering — Knowledge

Book slug : llm-engineering-69-chapters
Chapter   : ch-29-chapter-29 (lesson 29)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 29: الفصل 29: التقطير المعرفي (Knowledge Distillation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_distillation() -> None:
    teacher_logits, student_logits, temperature = ([3.2, 1.1, 0.2], [2.5, 1.4, 0.4], 2.0)
    teacher = softmax([x / temperature for x in teacher_logits])
    student = softmax([x / temperature for x in student_logits])
    kl = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    print('teacher probabilities:', [round(x, 3) for x in teacher])
    print('student probabilities:', [round(x, 3) for x in student])
    print('soft-target KL divergence:', round(kl, 5))

if __name__ == "__main__":
    print("Chapter 29 lab: Distill a teacher distribution into a student")
    run_distillation()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def demonstrate_temperature_effect():
    """
    🎯 توضيح مرئي لتأثير Temperature على توزيع الاحتمالات
    """
    logits = torch.tensor([4.0, 1.0, 0.8, -2.0])  # Logits خام من النموذج
    labels = ["باريس", "ليون", "مرسيليا", "لندن"]

    print(f"{'Temperature':<15}{'باريس':<10}{'ليون':<10}{'مرسيليا':<10}{'لندن':<10}")
    print("-" * 55)

    for T in [1.0, 2.0, 4.0, 10.0]:
        probs = F.softmax(logits / T, dim=0)
        print(f"T = {T:<11}" + "".join(f"{p.item():<10.3f}" for p in probs))

demonstrate_temperature_effect()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    🟡 دالة الخسارة الكاملة للتقطير المعرفي
    (الصياغة الكلاسيكية من ورقة Hinton et al. 2015)

    Parameters:
    -----------
    student_logits: الـ logits الخام من النموذج الطالب (قبل Softmax)
    teacher_logits: الـ logits الخام من النموذج المعلّم (قبل Softmax)
    labels: الإجابات الصحيحة (Hard Labels)
    temperature: درجة الحرارة لتليين التوزيعات (عادة 2-10)
    alpha: وزن الخسارة الصلبة (1-alpha هو وزن الخسارة الناعمة)
    """

    # ===== الجزء 1: Hard Loss - التعلّم من الإجابات الصحيحة =====
    hard_loss = F.cross_entropy(student_logits, labels)

    # ===== الجزء 2: Soft Loss - التعلّم من "طريقة تفكير" المعلّم =====
    # تطبيق Temperature على كلا النموذجين
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL-Divergence: يقيس "المسافة" بين توزيعين احتماليين
    # reduction='batchmean' هو المعيار القياسي الصحيح لـ KLDivLoss
    soft_loss = F.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)  # تصحيح T² كما شرحنا في صندوق التعمّق

    # ===== الدمج النهائي =====
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss, hard_loss.item(), soft_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class FeatureDistillationLoss(nn.Module):
    """
    🟡 Feature-based Distillation: نقل المعرفة من الطبقات الوسيطة

    هذا يتطلب "مطابقة" أبعاد الطبقات بين المعلّم والطالب
    (عادة الطالب له أبعاد أصغر، فنحتاج طبقة إسقاط - Projection)
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # طبقة إسقاط خطية لمطابقة أبعاد الطالب مع المعلّم
        # (ضرورية لأن الطالب عادة أصغر: مثلًا 384 بدلًا من 768)
        self.projection = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_hidden: torch.Tensor,  # (batch, seq_len, student_dim)
        teacher_hidden: torch.Tensor,  # (batch, seq_len, teacher_dim)
    ) -> torch.Tensor:
        # إسقاط تمثيلات الطالب لنفس بُعد المعلّم
        projected_student = self.projection(student_hidden)

        # MSE Loss بين التمثيلات المُسقطة وتمثيلات المعلّم
        # (بديل شائع: Cosine Similarity Loss)
        feature_loss = F.mse_loss(projected_student, teacher_hidden)

        return feature_loss

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_29_knowledge_distillation.py
======================================
مشروع متكامل: تقطير معرفي من BERT-base إلى نموذج Transformer مصغّر
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass
import numpy as np
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


@dataclass
class DistillationConfig:
    """تكوين شامل لتجربة التقطير"""
    teacher_model_name: str = "bert-base-uncased"
    num_labels: int = 2  # تصنيف ثنائي (إيجابي/سلبي)

    # معمارية الطالب (أصغر بكثير من المعلّم)
    student_hidden_size: int = 256      # مقابل 768 في BERT-base
    student_num_layers: int = 4          # مقابل 12 في BERT-base
    student_num_heads: int = 4           # مقابل 12 في BERT-base
    student_intermediate_size: int = 1024 # مقابل 3072 في BERT-base
    max_seq_length: int = 128
    vocab_size: int = 30522  # نفس قاموس BERT للتبسيط

    # معاملات التقطير
    temperature: float = 4.0
    alpha: float = 0.3           # وزن Hard Loss
    beta: float = 0.3            # وزن Feature Loss
    # (1 - alpha - beta) هو وزن Soft Loss تلقائيًا

    # إعدادات التدريب
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1


config = DistillationConfig()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class StudentTransformerEncoder(nn.Module):
    """
    🏗️ نموذج Transformer Encoder مصغّر من الصفر
    (يُعيد استخدام مفاهيم الفصل 11: بنية Transformer الكاملة)

    هذا سيكون "الطالب" - أصغر بكثير من BERT-base:
    - BERT-base: 110M معامل، 12 طبقة، hidden_size=768
    - طالبنا: ~5-10M معامل، 4 طبقات، hidden_size=256
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

        # التضمينات (Embeddings) - نفس القاموس لكن بُعد أصغر
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.student_hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_length, config.student_hidden_size
        )
        self.embedding_norm = nn.LayerNorm(config.student_hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        # طبقات Transformer Encoder (مكدّسة - Stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.student_hidden_size,
            nhead=config.student_num_heads,
            dim_feedforward=config.student_intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN - أكثر استقرارًا (كما ناقشنا في الفصل 11)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.student_num_layers
        )

        # رأس التصنيف
        self.classifier = nn.Linear(
            config.student_hidden_size, config.num_labels
        )

        self._init_weights()

    def _init_weights(self):
        """تهيئة الأوزان بطريقة Xavier - ممارسة قياسية للـ Transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False
    ):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = (
            self.token_embeddings(input_ids) +
            self.position_embeddings(positions)
        )
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))

        # تحويل attention_mask لصيغة PyTorch المطلوبة
        # (True تعني "تجاهل" في TransformerEncoder - عكس الاتفاقية الشائعة!)
        padding_mask = (attention_mask == 0)

        hidden_states = self.encoder(
            embeddings, src_key_padding_mask=padding_mask
        )

        # استخدام تمثيل أول توكن ([CLS]-style) للتصنيف
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)

        if output_hidden_states:
            return logits, hidden_states
        return logits

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    """
    Dataset يُحضّر البيانات مع تشغيل النموذج المعلّم مسبقًا
    لحساب الـ logits الخاصة به (لتسريع التدريب - لا نُعيد حساب
    المعلّم في كل Forward Pass من حلقة التدريب)

    💡 نصيحة أداء: في مشاريع حقيقية بحجم كبير، يُفضّل حساب
    logits المعلّم مرة واحدة وتخزينها على القرص (Caching)
    بدلًا من إعادة حسابها في كل Epoch
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        teacher_model,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

        print("🔄 حساب تنبؤات المعلّم مسبقًا (Teacher Forward Pass)...")
        teacher_model.eval()
        self.teacher_logits = []
        self.teacher_hidden_states = []

        with torch.no_grad():
            for i in range(0, len(texts), 32):  # معالجة دفعية لتسريع الحساب
                batch_texts = texts[i:i+32]
                inputs = tokenizer(
                    batch_texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt"
                ).to(device)

                outputs = teacher_model(
                    **inputs, output_hidden_states=True
                )
                self.teacher_logits.append(outputs.logits.cpu())
                # نأخذ تمثيلات آخر طبقة مخفية فقط للتبسيط
                self.teacher_hidden_states.append(
                    outputs.hidden_states[-1].cpu()
                )

        self.teacher_logits = torch.cat(self.teacher_logits, dim=0)
        self.teacher_hidden_states = torch.cat(self.teacher_hidden_states, dim=0)
        print(f"✅ تم حساب logits لـ {len(texts)} عينة")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "teacher_logits": self.teacher_logits[idx],
            "teacher_hidden": self.teacher_hidden_states[idx]
        }

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class DistillationTrainer:
    """
    🎓 مُدرّب متكامل يُطبّق التقطير المعرفي الكامل:
    Response-based (Soft Labels) + Feature-based (Hidden States)
    """

    def __init__(self, config: DistillationConfig):
        self.config = config

        print(f"📥 تحميل المعلّم: {config.teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name
        )
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.teacher_model_name, num_labels=config.num_labels
        ).to(device)

        # تجميد كل معاملات المعلّم - لن يُدرّب أبدًا!
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"📊 عدد معاملات المعلّم: {teacher_params:,}")

        print("🏗️  بناء الطالب من الصفر...")
        self.student_model = StudentTransformerEncoder(config).to(device)
        student_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"📊 عدد معاملات الطالب: {student_params:,}")
        print(f"📉 نسبة الضغط: {teacher_params / student_params:.1f}x أصغر!\n")

        # طبقة إسقاط لمطابقة أبعاد Feature Distillation
        self.feature_projector = nn.Linear(
            config.student_hidden_size,
            self.teacher_model.config.hidden_size
        ).to(device)

    def compute_combined_loss(self, batch):
        """
        🟡 حساب دالة الخسارة المركّبة الكاملة:
        Hard Loss + Soft Loss (Response) + Feature Loss
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)
        teacher_hidden = batch["teacher_hidden"].to(device)

        student_logits, student_hidden = self.student_model(
            input_ids, attention_mask, output_hidden_states=True
        )

        # === 1. Hard Loss ===
        hard_loss = F.cross_entropy(student_logits, labels)

        # === 2. Soft Loss (Response-based Distillation) ===
        T = self.config.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (T ** 2)

        # === 3. Feature Loss (Feature-based Distillation) ===
        projected_student_hidden = self.feature_projector(student_hidden)
        # مطابقة طول التسلسل (الطالب والمعلّم لهما نفس max_length هنا)
        feature_loss = F.mse_loss(projected_student_hidden, teacher_hidden)

        # === الدمج النهائي بالأوزان المحددة في config ===
        alpha, beta = self.config.alpha, self.config.beta
        gamma = 1 - alpha - beta  # وزن soft loss

        total_loss = (
            alpha * hard_loss +
            gamma * soft_loss +
            beta * feature_loss
        )

        return total_loss, {
            "hard": hard_loss.item(),
            "soft": soft_loss.item(),
            "feature": feature_loss.item()
        }

    def train(self, train_dataset, eval_dataset=None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        params = list(self.student_model.parameters()) + \
                 list(self.feature_projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )

        print("🚀 بدء عملية التقطير المعرفي...\n")
        self.student_model.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = {"total": [], "hard": [], "soft": [], "feature": []}

            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_components = self.compute_combined_loss(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses["total"].append(loss.item())
                for k, v in loss_components.items():
                    epoch_losses[k].append(v)

            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(
                f"📘 Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Total={avg['total']:.4f} | Hard={avg['hard']:.4f} | "
                f"Soft={avg['soft']:.4f} | Feature={avg['feature']:.4f}"
            )

            if eval_dataset is not None:
                accuracy = self.evaluate(eval_dataset)
                print(f"   ✅ دقة التقييم: {accuracy:.2%}")

        print("\n🎉 اكتمل التقطير المعرفي بنجاح!")

    @torch.no_grad()
    def evaluate(self, dataset):
        """تقييم دقة الطالب مقابل الـ labels الحقيقية"""
        self.student_model.eval()
        loader = DataLoader(dataset, batch_size=32)
        correct, total = 0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.student_model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        self.student_model.train()
        return correct / total

    @torch.no_grad()
    def compare_teacher_student_speed(self, sample_text: str, n_runs: int = 100):
        """
        🏁 مقارنة عملية: سرعة الاستدلال بين المعلّم والطالب
        """
        import time

        inputs_teacher = self.teacher_tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=self.config.max_seq_length, truncation=True
        ).to(device)

        # تسخين (Warmup) لتجنب قياس وقت تحميل الذاكرة الأولي
        for _ in range(5):
            _ = self.teacher_model(**inputs_teacher)
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )

        # قياس زمن المعلّم
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.teacher_model(**inputs_teacher)
        teacher_time = (time.perf_counter() - start) / n_runs

        # قياس زمن الطالب
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )
        student_time = (time.perf_counter() - start) / n_runs

        print(f"\n⚡ مقارنة السرعة (متوسط {n_runs} تشغيلة):")
        print(f"   المعلّم: {teacher_time*1000:.2f} ms")
        print(f"   الطالب:  {student_time*1000:.2f} ms")
        print(f"   🚀 تسريع: {teacher_time/student_time:.1f}x أسرع!")

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة التقطير المعرفي الكاملة"""

    # بيانات تجريبية لتصنيف المشاعر (في الواقع: IMDB أو SST-2)
    train_texts = [
        "This movie was absolutely fantastic, loved every minute!",
        "Terrible waste of time, one of the worst films I've seen.",
        "A masterpiece of modern cinema, truly inspiring.",
        "Boring and predictable, I wouldn't recommend it.",
        "The acting was superb and the plot kept me engaged.",
        "Disappointing sequel that fails to capture the original's magic.",
    ] * 50  # تكرار للمحاكاة فقط

    train_labels = [1, 0, 1, 0, 1, 0] * 50

    eval_texts = train_texts[:40]
    eval_labels = train_labels[:40]

    trainer = DistillationTrainer(config)

    # بناء Dataset (يحسب logits المعلّم مسبقًا)
    train_dataset = DistillationDataset(
        train_texts, train_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )
    eval_dataset = DistillationDataset(
        eval_texts, eval_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )

    # التدريب
    trainer.train(train_dataset, eval_dataset)

    # مقارنة السرعة بين المعلّم والطالب
    trainer.compare_teacher_student_speed(
        "This film exceeded all my expectations!"
    )


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 29: الفصل 29: التقطير المعرفي (Knowledge Distillation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_distillation() -> None:
    teacher_logits, student_logits, temperature = ([3.2, 1.1, 0.2], [2.5, 1.4, 0.4], 2.0)
    teacher = softmax([x / temperature for x in teacher_logits])
    student = softmax([x / temperature for x in student_logits])
    kl = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    print('teacher probabilities:', [round(x, 3) for x in teacher])
    print('student probabilities:', [round(x, 3) for x in student])
    print('soft-target KL divergence:', round(kl, 5))

if __name__ == "__main__":
    print("Chapter 29 lab: Distill a teacher distribution into a student")
    run_distillation()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def demonstrate_temperature_effect():
    """
    🎯 توضيح مرئي لتأثير Temperature على توزيع الاحتمالات
    """
    logits = torch.tensor([4.0, 1.0, 0.8, -2.0])  # Logits خام من النموذج
    labels = ["باريس", "ليون", "مرسيليا", "لندن"]

    print(f"{'Temperature':<15}{'باريس':<10}{'ليون':<10}{'مرسيليا':<10}{'لندن':<10}")
    print("-" * 55)

    for T in [1.0, 2.0, 4.0, 10.0]:
        probs = F.softmax(logits / T, dim=0)
        print(f"T = {T:<11}" + "".join(f"{p.item():<10.3f}" for p in probs))

demonstrate_temperature_effect()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    🟡 دالة الخسارة الكاملة للتقطير المعرفي
    (الصياغة الكلاسيكية من ورقة Hinton et al. 2015)

    Parameters:
    -----------
    student_logits: الـ logits الخام من النموذج الطالب (قبل Softmax)
    teacher_logits: الـ logits الخام من النموذج المعلّم (قبل Softmax)
    labels: الإجابات الصحيحة (Hard Labels)
    temperature: درجة الحرارة لتليين التوزيعات (عادة 2-10)
    alpha: وزن الخسارة الصلبة (1-alpha هو وزن الخسارة الناعمة)
    """

    # ===== الجزء 1: Hard Loss - التعلّم من الإجابات الصحيحة =====
    hard_loss = F.cross_entropy(student_logits, labels)

    # ===== الجزء 2: Soft Loss - التعلّم من "طريقة تفكير" المعلّم =====
    # تطبيق Temperature على كلا النموذجين
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL-Divergence: يقيس "المسافة" بين توزيعين احتماليين
    # reduction='batchmean' هو المعيار القياسي الصحيح لـ KLDivLoss
    soft_loss = F.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)  # تصحيح T² كما شرحنا في صندوق التعمّق

    # ===== الدمج النهائي =====
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss, hard_loss.item(), soft_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class FeatureDistillationLoss(nn.Module):
    """
    🟡 Feature-based Distillation: نقل المعرفة من الطبقات الوسيطة

    هذا يتطلب "مطابقة" أبعاد الطبقات بين المعلّم والطالب
    (عادة الطالب له أبعاد أصغر، فنحتاج طبقة إسقاط - Projection)
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # طبقة إسقاط خطية لمطابقة أبعاد الطالب مع المعلّم
        # (ضرورية لأن الطالب عادة أصغر: مثلًا 384 بدلًا من 768)
        self.projection = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_hidden: torch.Tensor,  # (batch, seq_len, student_dim)
        teacher_hidden: torch.Tensor,  # (batch, seq_len, teacher_dim)
    ) -> torch.Tensor:
        # إسقاط تمثيلات الطالب لنفس بُعد المعلّم
        projected_student = self.projection(student_hidden)

        # MSE Loss بين التمثيلات المُسقطة وتمثيلات المعلّم
        # (بديل شائع: Cosine Similarity Loss)
        feature_loss = F.mse_loss(projected_student, teacher_hidden)

        return feature_loss

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
chapter_29_knowledge_distillation.py
======================================
مشروع متكامل: تقطير معرفي من BERT-base إلى نموذج Transformer مصغّر
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass
import numpy as np
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


@dataclass
class DistillationConfig:
    """تكوين شامل لتجربة التقطير"""
    teacher_model_name: str = "bert-base-uncased"
    num_labels: int = 2  # تصنيف ثنائي (إيجابي/سلبي)

    # معمارية الطالب (أصغر بكثير من المعلّم)
    student_hidden_size: int = 256      # مقابل 768 في BERT-base
    student_num_layers: int = 4          # مقابل 12 في BERT-base
    student_num_heads: int = 4           # مقابل 12 في BERT-base
    student_intermediate_size: int = 1024 # مقابل 3072 في BERT-base
    max_seq_length: int = 128
    vocab_size: int = 30522  # نفس قاموس BERT للتبسيط

    # معاملات التقطير
    temperature: float = 4.0
    alpha: float = 0.3           # وزن Hard Loss
    beta: float = 0.3            # وزن Feature Loss
    # (1 - alpha - beta) هو وزن Soft Loss تلقائيًا

    # إعدادات التدريب
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1


config = DistillationConfig()

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
class StudentTransformerEncoder(nn.Module):
    """
    🏗️ نموذج Transformer Encoder مصغّر من الصفر
    (يُعيد استخدام مفاهيم الفصل 11: بنية Transformer الكاملة)

    هذا سيكون "الطالب" - أصغر بكثير من BERT-base:
    - BERT-base: 110M معامل، 12 طبقة، hidden_size=768
    - طالبنا: ~5-10M معامل، 4 طبقات، hidden_size=256
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

        # التضمينات (Embeddings) - نفس القاموس لكن بُعد أصغر
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.student_hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_length, config.student_hidden_size
        )
        self.embedding_norm = nn.LayerNorm(config.student_hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        # طبقات Transformer Encoder (مكدّسة - Stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.student_hidden_size,
            nhead=config.student_num_heads,
            dim_feedforward=config.student_intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN - أكثر استقرارًا (كما ناقشنا في الفصل 11)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.student_num_layers
        )

        # رأس التصنيف
        self.classifier = nn.Linear(
            config.student_hidden_size, config.num_labels
        )

        self._init_weights()

    def _init_weights(self):
        """تهيئة الأوزان بطريقة Xavier - ممارسة قياسية للـ Transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False
    ):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = (
            self.token_embeddings(input_ids) +
            self.position_embeddings(positions)
        )
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))

        # تحويل attention_mask لصيغة PyTorch المطلوبة
        # (True تعني "تجاهل" في TransformerEncoder - عكس الاتفاقية الشائعة!)
        padding_mask = (attention_mask == 0)

        hidden_states = self.encoder(
            embeddings, src_key_padding_mask=padding_mask
        )

        # استخدام تمثيل أول توكن ([CLS]-style) للتصنيف
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)

        if output_hidden_states:
            return logits, hidden_states
        return logits

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    """
    Dataset يُحضّر البيانات مع تشغيل النموذج المعلّم مسبقًا
    لحساب الـ logits الخاصة به (لتسريع التدريب - لا نُعيد حساب
    المعلّم في كل Forward Pass من حلقة التدريب)

    💡 نصيحة أداء: في مشاريع حقيقية بحجم كبير، يُفضّل حساب
    logits المعلّم مرة واحدة وتخزينها على القرص (Caching)
    بدلًا من إعادة حسابها في كل Epoch
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        teacher_model,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

        print("🔄 حساب تنبؤات المعلّم مسبقًا (Teacher Forward Pass)...")
        teacher_model.eval()
        self.teacher_logits = []
        self.teacher_hidden_states = []

        with torch.no_grad():
            for i in range(0, len(texts), 32):  # معالجة دفعية لتسريع الحساب
                batch_texts = texts[i:i+32]
                inputs = tokenizer(
                    batch_texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt"
                ).to(device)

                outputs = teacher_model(
                    **inputs, output_hidden_states=True
                )
                self.teacher_logits.append(outputs.logits.cpu())
                # نأخذ تمثيلات آخر طبقة مخفية فقط للتبسيط
                self.teacher_hidden_states.append(
                    outputs.hidden_states[-1].cpu()
                )

        self.teacher_logits = torch.cat(self.teacher_logits, dim=0)
        self.teacher_hidden_states = torch.cat(self.teacher_hidden_states, dim=0)
        print(f"✅ تم حساب logits لـ {len(texts)} عينة")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "teacher_logits": self.teacher_logits[idx],
            "teacher_hidden": self.teacher_hidden_states[idx]
        }

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class DistillationTrainer:
    """
    🎓 مُدرّب متكامل يُطبّق التقطير المعرفي الكامل:
    Response-based (Soft Labels) + Feature-based (Hidden States)
    """

    def __init__(self, config: DistillationConfig):
        self.config = config

        print(f"📥 تحميل المعلّم: {config.teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name
        )
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.teacher_model_name, num_labels=config.num_labels
        ).to(device)

        # تجميد كل معاملات المعلّم - لن يُدرّب أبدًا!
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"📊 عدد معاملات المعلّم: {teacher_params:,}")

        print("🏗️  بناء الطالب من الصفر...")
        self.student_model = StudentTransformerEncoder(config).to(device)
        student_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"📊 عدد معاملات الطالب: {student_params:,}")
        print(f"📉 نسبة الضغط: {teacher_params / student_params:.1f}x أصغر!\n")

        # طبقة إسقاط لمطابقة أبعاد Feature Distillation
        self.feature_projector = nn.Linear(
            config.student_hidden_size,
            self.teacher_model.config.hidden_size
        ).to(device)

    def compute_combined_loss(self, batch):
        """
        🟡 حساب دالة الخسارة المركّبة الكاملة:
        Hard Loss + Soft Loss (Response) + Feature Loss
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)
        teacher_hidden = batch["teacher_hidden"].to(device)

        student_logits, student_hidden = self.student_model(
            input_ids, attention_mask, output_hidden_states=True
        )

        # === 1. Hard Loss ===
        hard_loss = F.cross_entropy(student_logits, labels)

        # === 2. Soft Loss (Response-based Distillation) ===
        T = self.config.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (T ** 2)

        # === 3. Feature Loss (Feature-based Distillation) ===
        projected_student_hidden = self.feature_projector(student_hidden)
        # مطابقة طول التسلسل (الطالب والمعلّم لهما نفس max_length هنا)
        feature_loss = F.mse_loss(projected_student_hidden, teacher_hidden)

        # === الدمج النهائي بالأوزان المحددة في config ===
        alpha, beta = self.config.alpha, self.config.beta
        gamma = 1 - alpha - beta  # وزن soft loss

        total_loss = (
            alpha * hard_loss +
            gamma * soft_loss +
            beta * feature_loss
        )

        return total_loss, {
            "hard": hard_loss.item(),
            "soft": soft_loss.item(),
            "feature": feature_loss.item()
        }

    def train(self, train_dataset, eval_dataset=None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        params = list(self.student_model.parameters()) + \
                 list(self.feature_projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )

        print("🚀 بدء عملية التقطير المعرفي...\n")
        self.student_model.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = {"total": [], "hard": [], "soft": [], "feature": []}

            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_components = self.compute_combined_loss(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses["total"].append(loss.item())
                for k, v in loss_components.items():
                    epoch_losses[k].append(v)

            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(
                f"📘 Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Total={avg['total']:.4f} | Hard={avg['hard']:.4f} | "
                f"Soft={avg['soft']:.4f} | Feature={avg['feature']:.4f}"
            )

            if eval_dataset is not None:
                accuracy = self.evaluate(eval_dataset)
                print(f"   ✅ دقة التقييم: {accuracy:.2%}")

        print("\n🎉 اكتمل التقطير المعرفي بنجاح!")

    @torch.no_grad()
    def evaluate(self, dataset):
        """تقييم دقة الطالب مقابل الـ labels الحقيقية"""
        self.student_model.eval()
        loader = DataLoader(dataset, batch_size=32)
        correct, total = 0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.student_model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        self.student_model.train()
        return correct / total

    @torch.no_grad()
    def compare_teacher_student_speed(self, sample_text: str, n_runs: int = 100):
        """
        🏁 مقارنة عملية: سرعة الاستدلال بين المعلّم والطالب
        """
        import time

        inputs_teacher = self.teacher_tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=self.config.max_seq_length, truncation=True
        ).to(device)

        # تسخين (Warmup) لتجنب قياس وقت تحميل الذاكرة الأولي
        for _ in range(5):
            _ = self.teacher_model(**inputs_teacher)
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )

        # قياس زمن المعلّم
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.teacher_model(**inputs_teacher)
        teacher_time = (time.perf_counter() - start) / n_runs

        # قياس زمن الطالب
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )
        student_time = (time.perf_counter() - start) / n_runs

        print(f"\n⚡ مقارنة السرعة (متوسط {n_runs} تشغيلة):")
        print(f"   المعلّم: {teacher_time*1000:.2f} ms")
        print(f"   الطالب:  {student_time*1000:.2f} ms")
        print(f"   🚀 تسريع: {teacher_time/student_time:.1f}x أسرع!")

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة التقطير المعرفي الكاملة"""

    # بيانات تجريبية لتصنيف المشاعر (في الواقع: IMDB أو SST-2)
    train_texts = [
        "This movie was absolutely fantastic, loved every minute!",
        "Terrible waste of time, one of the worst films I've seen.",
        "A masterpiece of modern cinema, truly inspiring.",
        "Boring and predictable, I wouldn't recommend it.",
        "The acting was superb and the plot kept me engaged.",
        "Disappointing sequel that fails to capture the original's magic.",
    ] * 50  # تكرار للمحاكاة فقط

    train_labels = [1, 0, 1, 0, 1, 0] * 50

    eval_texts = train_texts[:40]
    eval_labels = train_labels[:40]

    trainer = DistillationTrainer(config)

    # بناء Dataset (يحسب logits المعلّم مسبقًا)
    train_dataset = DistillationDataset(
        train_texts, train_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )
    eval_dataset = DistillationDataset(
        eval_texts, eval_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )

    # التدريب
    trainer.train(train_dataset, eval_dataset)

    # مقارنة السرعة بين المعلّم والطالب
    trainer.compare_teacher_student_speed(
        "This film exceeded all my expectations!"
    )


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقطير المعرفي (Knowledge Distillation)
Large Language Model Engineering — Knowledge

Book slug : llm-engineering-69-chapters
Chapter   : ch-29-chapter-29 (lesson 29)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_29.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 29: الفصل 29: التقطير المعرفي (Knowledge Distillation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_distillation() -> None:
    teacher_logits, student_logits, temperature = ([3.2, 1.1, 0.2], [2.5, 1.4, 0.4], 2.0)
    teacher = softmax([x / temperature for x in teacher_logits])
    student = softmax([x / temperature for x in student_logits])
    kl = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    print('teacher probabilities:', [round(x, 3) for x in teacher])
    print('student probabilities:', [round(x, 3) for x in student])
    print('soft-target KL divergence:', round(kl, 5))

if __name__ == "__main__":
    print("Chapter 29 lab: Distill a teacher distribution into a student")
    run_distillation()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def demonstrate_temperature_effect():
    """
    🎯 توضيح مرئي لتأثير Temperature على توزيع الاحتمالات
    """
    logits = torch.tensor([4.0, 1.0, 0.8, -2.0])  # Logits خام من النموذج
    labels = ["باريس", "ليون", "مرسيليا", "لندن"]

    print(f"{'Temperature':<15}{'باريس':<10}{'ليون':<10}{'مرسيليا':<10}{'لندن':<10}")
    print("-" * 55)

    for T in [1.0, 2.0, 4.0, 10.0]:
        probs = F.softmax(logits / T, dim=0)
        print(f"T = {T:<11}" + "".join(f"{p.item():<10.3f}" for p in probs))

demonstrate_temperature_effect()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    🟡 دالة الخسارة الكاملة للتقطير المعرفي
    (الصياغة الكلاسيكية من ورقة Hinton et al. 2015)

    Parameters:
    -----------
    student_logits: الـ logits الخام من النموذج الطالب (قبل Softmax)
    teacher_logits: الـ logits الخام من النموذج المعلّم (قبل Softmax)
    labels: الإجابات الصحيحة (Hard Labels)
    temperature: درجة الحرارة لتليين التوزيعات (عادة 2-10)
    alpha: وزن الخسارة الصلبة (1-alpha هو وزن الخسارة الناعمة)
    """

    # ===== الجزء 1: Hard Loss - التعلّم من الإجابات الصحيحة =====
    hard_loss = F.cross_entropy(student_logits, labels)

    # ===== الجزء 2: Soft Loss - التعلّم من "طريقة تفكير" المعلّم =====
    # تطبيق Temperature على كلا النموذجين
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL-Divergence: يقيس "المسافة" بين توزيعين احتماليين
    # reduction='batchmean' هو المعيار القياسي الصحيح لـ KLDivLoss
    soft_loss = F.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)  # تصحيح T² كما شرحنا في صندوق التعمّق

    # ===== الدمج النهائي =====
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss, hard_loss.item(), soft_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class FeatureDistillationLoss(nn.Module):
    """
    🟡 Feature-based Distillation: نقل المعرفة من الطبقات الوسيطة

    هذا يتطلب "مطابقة" أبعاد الطبقات بين المعلّم والطالب
    (عادة الطالب له أبعاد أصغر، فنحتاج طبقة إسقاط - Projection)
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # طبقة إسقاط خطية لمطابقة أبعاد الطالب مع المعلّم
        # (ضرورية لأن الطالب عادة أصغر: مثلًا 384 بدلًا من 768)
        self.projection = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_hidden: torch.Tensor,  # (batch, seq_len, student_dim)
        teacher_hidden: torch.Tensor,  # (batch, seq_len, teacher_dim)
    ) -> torch.Tensor:
        # إسقاط تمثيلات الطالب لنفس بُعد المعلّم
        projected_student = self.projection(student_hidden)

        # MSE Loss بين التمثيلات المُسقطة وتمثيلات المعلّم
        # (بديل شائع: Cosine Similarity Loss)
        feature_loss = F.mse_loss(projected_student, teacher_hidden)

        return feature_loss

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
"""
chapter_29_knowledge_distillation.py
======================================
مشروع متكامل: تقطير معرفي من BERT-base إلى نموذج Transformer مصغّر
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass
import numpy as np
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


@dataclass
class DistillationConfig:
    """تكوين شامل لتجربة التقطير"""
    teacher_model_name: str = "bert-base-uncased"
    num_labels: int = 2  # تصنيف ثنائي (إيجابي/سلبي)

    # معمارية الطالب (أصغر بكثير من المعلّم)
    student_hidden_size: int = 256      # مقابل 768 في BERT-base
    student_num_layers: int = 4          # مقابل 12 في BERT-base
    student_num_heads: int = 4           # مقابل 12 في BERT-base
    student_intermediate_size: int = 1024 # مقابل 3072 في BERT-base
    max_seq_length: int = 128
    vocab_size: int = 30522  # نفس قاموس BERT للتبسيط

    # معاملات التقطير
    temperature: float = 4.0
    alpha: float = 0.3           # وزن Hard Loss
    beta: float = 0.3            # وزن Feature Loss
    # (1 - alpha - beta) هو وزن Soft Loss تلقائيًا

    # إعدادات التدريب
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1


config = DistillationConfig()

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class StudentTransformerEncoder(nn.Module):
    """
    🏗️ نموذج Transformer Encoder مصغّر من الصفر
    (يُعيد استخدام مفاهيم الفصل 11: بنية Transformer الكاملة)

    هذا سيكون "الطالب" - أصغر بكثير من BERT-base:
    - BERT-base: 110M معامل، 12 طبقة، hidden_size=768
    - طالبنا: ~5-10M معامل، 4 طبقات، hidden_size=256
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

        # التضمينات (Embeddings) - نفس القاموس لكن بُعد أصغر
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.student_hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_length, config.student_hidden_size
        )
        self.embedding_norm = nn.LayerNorm(config.student_hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        # طبقات Transformer Encoder (مكدّسة - Stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.student_hidden_size,
            nhead=config.student_num_heads,
            dim_feedforward=config.student_intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN - أكثر استقرارًا (كما ناقشنا في الفصل 11)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.student_num_layers
        )

        # رأس التصنيف
        self.classifier = nn.Linear(
            config.student_hidden_size, config.num_labels
        )

        self._init_weights()

    def _init_weights(self):
        """تهيئة الأوزان بطريقة Xavier - ممارسة قياسية للـ Transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False
    ):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = (
            self.token_embeddings(input_ids) +
            self.position_embeddings(positions)
        )
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))

        # تحويل attention_mask لصيغة PyTorch المطلوبة
        # (True تعني "تجاهل" في TransformerEncoder - عكس الاتفاقية الشائعة!)
        padding_mask = (attention_mask == 0)

        hidden_states = self.encoder(
            embeddings, src_key_padding_mask=padding_mask
        )

        # استخدام تمثيل أول توكن ([CLS]-style) للتصنيف
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)

        if output_hidden_states:
            return logits, hidden_states
        return logits

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    """
    Dataset يُحضّر البيانات مع تشغيل النموذج المعلّم مسبقًا
    لحساب الـ logits الخاصة به (لتسريع التدريب - لا نُعيد حساب
    المعلّم في كل Forward Pass من حلقة التدريب)

    💡 نصيحة أداء: في مشاريع حقيقية بحجم كبير، يُفضّل حساب
    logits المعلّم مرة واحدة وتخزينها على القرص (Caching)
    بدلًا من إعادة حسابها في كل Epoch
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        teacher_model,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

        print("🔄 حساب تنبؤات المعلّم مسبقًا (Teacher Forward Pass)...")
        teacher_model.eval()
        self.teacher_logits = []
        self.teacher_hidden_states = []

        with torch.no_grad():
            for i in range(0, len(texts), 32):  # معالجة دفعية لتسريع الحساب
                batch_texts = texts[i:i+32]
                inputs = tokenizer(
                    batch_texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt"
                ).to(device)

                outputs = teacher_model(
                    **inputs, output_hidden_states=True
                )
                self.teacher_logits.append(outputs.logits.cpu())
                # نأخذ تمثيلات آخر طبقة مخفية فقط للتبسيط
                self.teacher_hidden_states.append(
                    outputs.hidden_states[-1].cpu()
                )

        self.teacher_logits = torch.cat(self.teacher_logits, dim=0)
        self.teacher_hidden_states = torch.cat(self.teacher_hidden_states, dim=0)
        print(f"✅ تم حساب logits لـ {len(texts)} عينة")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "teacher_logits": self.teacher_logits[idx],
            "teacher_hidden": self.teacher_hidden_states[idx]
        }

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
class DistillationTrainer:
    """
    🎓 مُدرّب متكامل يُطبّق التقطير المعرفي الكامل:
    Response-based (Soft Labels) + Feature-based (Hidden States)
    """

    def __init__(self, config: DistillationConfig):
        self.config = config

        print(f"📥 تحميل المعلّم: {config.teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name
        )
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.teacher_model_name, num_labels=config.num_labels
        ).to(device)

        # تجميد كل معاملات المعلّم - لن يُدرّب أبدًا!
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"📊 عدد معاملات المعلّم: {teacher_params:,}")

        print("🏗️  بناء الطالب من الصفر...")
        self.student_model = StudentTransformerEncoder(config).to(device)
        student_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"📊 عدد معاملات الطالب: {student_params:,}")
        print(f"📉 نسبة الضغط: {teacher_params / student_params:.1f}x أصغر!\n")

        # طبقة إسقاط لمطابقة أبعاد Feature Distillation
        self.feature_projector = nn.Linear(
            config.student_hidden_size,
            self.teacher_model.config.hidden_size
        ).to(device)

    def compute_combined_loss(self, batch):
        """
        🟡 حساب دالة الخسارة المركّبة الكاملة:
        Hard Loss + Soft Loss (Response) + Feature Loss
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)
        teacher_hidden = batch["teacher_hidden"].to(device)

        student_logits, student_hidden = self.student_model(
            input_ids, attention_mask, output_hidden_states=True
        )

        # === 1. Hard Loss ===
        hard_loss = F.cross_entropy(student_logits, labels)

        # === 2. Soft Loss (Response-based Distillation) ===
        T = self.config.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (T ** 2)

        # === 3. Feature Loss (Feature-based Distillation) ===
        projected_student_hidden = self.feature_projector(student_hidden)
        # مطابقة طول التسلسل (الطالب والمعلّم لهما نفس max_length هنا)
        feature_loss = F.mse_loss(projected_student_hidden, teacher_hidden)

        # === الدمج النهائي بالأوزان المحددة في config ===
        alpha, beta = self.config.alpha, self.config.beta
        gamma = 1 - alpha - beta  # وزن soft loss

        total_loss = (
            alpha * hard_loss +
            gamma * soft_loss +
            beta * feature_loss
        )

        return total_loss, {
            "hard": hard_loss.item(),
            "soft": soft_loss.item(),
            "feature": feature_loss.item()
        }

    def train(self, train_dataset, eval_dataset=None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        params = list(self.student_model.parameters()) + \
                 list(self.feature_projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )

        print("🚀 بدء عملية التقطير المعرفي...\n")
        self.student_model.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = {"total": [], "hard": [], "soft": [], "feature": []}

            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_components = self.compute_combined_loss(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses["total"].append(loss.item())
                for k, v in loss_components.items():
                    epoch_losses[k].append(v)

            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(
                f"📘 Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Total={avg['total']:.4f} | Hard={avg['hard']:.4f} | "
                f"Soft={avg['soft']:.4f} | Feature={avg['feature']:.4f}"
            )

            if eval_dataset is not None:
                accuracy = self.evaluate(eval_dataset)
                print(f"   ✅ دقة التقييم: {accuracy:.2%}")

        print("\n🎉 اكتمل التقطير المعرفي بنجاح!")

    @torch.no_grad()
    def evaluate(self, dataset):
        """تقييم دقة الطالب مقابل الـ labels الحقيقية"""
        self.student_model.eval()
        loader = DataLoader(dataset, batch_size=32)
        correct, total = 0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.student_model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        self.student_model.train()
        return correct / total

    @torch.no_grad()
    def compare_teacher_student_speed(self, sample_text: str, n_runs: int = 100):
        """
        🏁 مقارنة عملية: سرعة الاستدلال بين المعلّم والطالب
        """
        import time

        inputs_teacher = self.teacher_tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=self.config.max_seq_length, truncation=True
        ).to(device)

        # تسخين (Warmup) لتجنب قياس وقت تحميل الذاكرة الأولي
        for _ in range(5):
            _ = self.teacher_model(**inputs_teacher)
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )

        # قياس زمن المعلّم
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.teacher_model(**inputs_teacher)
        teacher_time = (time.perf_counter() - start) / n_runs

        # قياس زمن الطالب
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )
        student_time = (time.perf_counter() - start) / n_runs

        print(f"\n⚡ مقارنة السرعة (متوسط {n_runs} تشغيلة):")
        print(f"   المعلّم: {teacher_time*1000:.2f} ms")
        print(f"   الطالب:  {student_time*1000:.2f} ms")
        print(f"   🚀 تسريع: {teacher_time/student_time:.1f}x أسرع!")

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-06-model-reuse-adaptation/chapter-29.md
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة التقطير المعرفي الكاملة"""

    # بيانات تجريبية لتصنيف المشاعر (في الواقع: IMDB أو SST-2)
    train_texts = [
        "This movie was absolutely fantastic, loved every minute!",
        "Terrible waste of time, one of the worst films I've seen.",
        "A masterpiece of modern cinema, truly inspiring.",
        "Boring and predictable, I wouldn't recommend it.",
        "The acting was superb and the plot kept me engaged.",
        "Disappointing sequel that fails to capture the original's magic.",
    ] * 50  # تكرار للمحاكاة فقط

    train_labels = [1, 0, 1, 0, 1, 0] * 50

    eval_texts = train_texts[:40]
    eval_labels = train_labels[:40]

    trainer = DistillationTrainer(config)

    # بناء Dataset (يحسب logits المعلّم مسبقًا)
    train_dataset = DistillationDataset(
        train_texts, train_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )
    eval_dataset = DistillationDataset(
        eval_texts, eval_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )

    # التدريب
    trainer.train(train_dataset, eval_dataset)

    # مقارنة السرعة بين المعلّم والطالب
    trainer.compare_teacher_student_speed(
        "This film exceeded all my expectations!"
    )


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 29: الفصل 29: التقطير المعرفي (Knowledge Distillation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_distillation() -> None:
    teacher_logits, student_logits, temperature = ([3.2, 1.1, 0.2], [2.5, 1.4, 0.4], 2.0)
    teacher = softmax([x / temperature for x in teacher_logits])
    student = softmax([x / temperature for x in student_logits])
    kl = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    print('teacher probabilities:', [round(x, 3) for x in teacher])
    print('student probabilities:', [round(x, 3) for x in student])
    print('soft-target KL divergence:', round(kl, 5))

if __name__ == "__main__":
    print("Chapter 29 lab: Distill a teacher distribution into a student")
    run_distillation()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def demonstrate_temperature_effect():
    """
    🎯 توضيح مرئي لتأثير Temperature على توزيع الاحتمالات
    """
    logits = torch.tensor([4.0, 1.0, 0.8, -2.0])  # Logits خام من النموذج
    labels = ["باريس", "ليون", "مرسيليا", "لندن"]

    print(f"{'Temperature':<15}{'باريس':<10}{'ليون':<10}{'مرسيليا':<10}{'لندن':<10}")
    print("-" * 55)

    for T in [1.0, 2.0, 4.0, 10.0]:
        probs = F.softmax(logits / T, dim=0)
        print(f"T = {T:<11}" + "".join(f"{p.item():<10.3f}" for p in probs))

demonstrate_temperature_effect()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.3
) -> torch.Tensor:
    """
    🟡 دالة الخسارة الكاملة للتقطير المعرفي
    (الصياغة الكلاسيكية من ورقة Hinton et al. 2015)

    Parameters:
    -----------
    student_logits: الـ logits الخام من النموذج الطالب (قبل Softmax)
    teacher_logits: الـ logits الخام من النموذج المعلّم (قبل Softmax)
    labels: الإجابات الصحيحة (Hard Labels)
    temperature: درجة الحرارة لتليين التوزيعات (عادة 2-10)
    alpha: وزن الخسارة الصلبة (1-alpha هو وزن الخسارة الناعمة)
    """

    # ===== الجزء 1: Hard Loss - التعلّم من الإجابات الصحيحة =====
    hard_loss = F.cross_entropy(student_logits, labels)

    # ===== الجزء 2: Soft Loss - التعلّم من "طريقة تفكير" المعلّم =====
    # تطبيق Temperature على كلا النموذجين
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL-Divergence: يقيس "المسافة" بين توزيعين احتماليين
    # reduction='batchmean' هو المعيار القياسي الصحيح لـ KLDivLoss
    soft_loss = F.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)  # تصحيح T² كما شرحنا في صندوق التعمّق

    # ===== الدمج النهائي =====
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss, hard_loss.item(), soft_loss.item()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class FeatureDistillationLoss(nn.Module):
    """
    🟡 Feature-based Distillation: نقل المعرفة من الطبقات الوسيطة

    هذا يتطلب "مطابقة" أبعاد الطبقات بين المعلّم والطالب
    (عادة الطالب له أبعاد أصغر، فنحتاج طبقة إسقاط - Projection)
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # طبقة إسقاط خطية لمطابقة أبعاد الطالب مع المعلّم
        # (ضرورية لأن الطالب عادة أصغر: مثلًا 384 بدلًا من 768)
        self.projection = nn.Linear(student_dim, teacher_dim)

    def forward(
        self,
        student_hidden: torch.Tensor,  # (batch, seq_len, student_dim)
        teacher_hidden: torch.Tensor,  # (batch, seq_len, teacher_dim)
    ) -> torch.Tensor:
        # إسقاط تمثيلات الطالب لنفس بُعد المعلّم
        projected_student = self.projection(student_hidden)

        # MSE Loss بين التمثيلات المُسقطة وتمثيلات المعلّم
        # (بديل شائع: Cosine Similarity Loss)
        feature_loss = F.mse_loss(projected_student, teacher_hidden)

        return feature_loss

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
chapter_29_knowledge_distillation.py
======================================
مشروع متكامل: تقطير معرفي من BERT-base إلى نموذج Transformer مصغّر
المتطلبات: PyTorch 2.5+, Transformers 4.46+, Python 3.13+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from dataclasses import dataclass
import numpy as np
import math

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)


@dataclass
class DistillationConfig:
    """تكوين شامل لتجربة التقطير"""
    teacher_model_name: str = "bert-base-uncased"
    num_labels: int = 2  # تصنيف ثنائي (إيجابي/سلبي)

    # معمارية الطالب (أصغر بكثير من المعلّم)
    student_hidden_size: int = 256      # مقابل 768 في BERT-base
    student_num_layers: int = 4          # مقابل 12 في BERT-base
    student_num_heads: int = 4           # مقابل 12 في BERT-base
    student_intermediate_size: int = 1024 # مقابل 3072 في BERT-base
    max_seq_length: int = 128
    vocab_size: int = 30522  # نفس قاموس BERT للتبسيط

    # معاملات التقطير
    temperature: float = 4.0
    alpha: float = 0.3           # وزن Hard Loss
    beta: float = 0.3            # وزن Feature Loss
    # (1 - alpha - beta) هو وزن Soft Loss تلقائيًا

    # إعدادات التدريب
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1


config = DistillationConfig()

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
class StudentTransformerEncoder(nn.Module):
    """
    🏗️ نموذج Transformer Encoder مصغّر من الصفر
    (يُعيد استخدام مفاهيم الفصل 11: بنية Transformer الكاملة)

    هذا سيكون "الطالب" - أصغر بكثير من BERT-base:
    - BERT-base: 110M معامل، 12 طبقة، hidden_size=768
    - طالبنا: ~5-10M معامل، 4 طبقات، hidden_size=256
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

        # التضمينات (Embeddings) - نفس القاموس لكن بُعد أصغر
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.student_hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_length, config.student_hidden_size
        )
        self.embedding_norm = nn.LayerNorm(config.student_hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)

        # طبقات Transformer Encoder (مكدّسة - Stacked)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.student_hidden_size,
            nhead=config.student_num_heads,
            dim_feedforward=config.student_intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN - أكثر استقرارًا (كما ناقشنا في الفصل 11)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.student_num_layers
        )

        # رأس التصنيف
        self.classifier = nn.Linear(
            config.student_hidden_size, config.num_labels
        )

        self._init_weights()

    def _init_weights(self):
        """تهيئة الأوزان بطريقة Xavier - ممارسة قياسية للـ Transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False
    ):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = (
            self.token_embeddings(input_ids) +
            self.position_embeddings(positions)
        )
        embeddings = self.embedding_dropout(self.embedding_norm(embeddings))

        # تحويل attention_mask لصيغة PyTorch المطلوبة
        # (True تعني "تجاهل" في TransformerEncoder - عكس الاتفاقية الشائعة!)
        padding_mask = (attention_mask == 0)

        hidden_states = self.encoder(
            embeddings, src_key_padding_mask=padding_mask
        )

        # استخدام تمثيل أول توكن ([CLS]-style) للتصنيف
        pooled_output = hidden_states[:, 0, :]
        logits = self.classifier(pooled_output)

        if output_hidden_states:
            return logits, hidden_states
        return logits

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    """
    Dataset يُحضّر البيانات مع تشغيل النموذج المعلّم مسبقًا
    لحساب الـ logits الخاصة به (لتسريع التدريب - لا نُعيد حساب
    المعلّم في كل Forward Pass من حلقة التدريب)

    💡 نصيحة أداء: في مشاريع حقيقية بحجم كبير، يُفضّل حساب
    logits المعلّم مرة واحدة وتخزينها على القرص (Caching)
    بدلًا من إعادة حسابها في كل Epoch
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        teacher_model,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

        print("🔄 حساب تنبؤات المعلّم مسبقًا (Teacher Forward Pass)...")
        teacher_model.eval()
        self.teacher_logits = []
        self.teacher_hidden_states = []

        with torch.no_grad():
            for i in range(0, len(texts), 32):  # معالجة دفعية لتسريع الحساب
                batch_texts = texts[i:i+32]
                inputs = tokenizer(
                    batch_texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt"
                ).to(device)

                outputs = teacher_model(
                    **inputs, output_hidden_states=True
                )
                self.teacher_logits.append(outputs.logits.cpu())
                # نأخذ تمثيلات آخر طبقة مخفية فقط للتبسيط
                self.teacher_hidden_states.append(
                    outputs.hidden_states[-1].cpu()
                )

        self.teacher_logits = torch.cat(self.teacher_logits, dim=0)
        self.teacher_hidden_states = torch.cat(self.teacher_hidden_states, dim=0)
        print(f"✅ تم حساب logits لـ {len(texts)} عينة")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "teacher_logits": self.teacher_logits[idx],
            "teacher_hidden": self.teacher_hidden_states[idx]
        }

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
class DistillationTrainer:
    """
    🎓 مُدرّب متكامل يُطبّق التقطير المعرفي الكامل:
    Response-based (Soft Labels) + Feature-based (Hidden States)
    """

    def __init__(self, config: DistillationConfig):
        self.config = config

        print(f"📥 تحميل المعلّم: {config.teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name
        )
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            config.teacher_model_name, num_labels=config.num_labels
        ).to(device)

        # تجميد كل معاملات المعلّم - لن يُدرّب أبدًا!
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"📊 عدد معاملات المعلّم: {teacher_params:,}")

        print("🏗️  بناء الطالب من الصفر...")
        self.student_model = StudentTransformerEncoder(config).to(device)
        student_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"📊 عدد معاملات الطالب: {student_params:,}")
        print(f"📉 نسبة الضغط: {teacher_params / student_params:.1f}x أصغر!\n")

        # طبقة إسقاط لمطابقة أبعاد Feature Distillation
        self.feature_projector = nn.Linear(
            config.student_hidden_size,
            self.teacher_model.config.hidden_size
        ).to(device)

    def compute_combined_loss(self, batch):
        """
        🟡 حساب دالة الخسارة المركّبة الكاملة:
        Hard Loss + Soft Loss (Response) + Feature Loss
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)
        teacher_hidden = batch["teacher_hidden"].to(device)

        student_logits, student_hidden = self.student_model(
            input_ids, attention_mask, output_hidden_states=True
        )

        # === 1. Hard Loss ===
        hard_loss = F.cross_entropy(student_logits, labels)

        # === 2. Soft Loss (Response-based Distillation) ===
        T = self.config.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (T ** 2)

        # === 3. Feature Loss (Feature-based Distillation) ===
        projected_student_hidden = self.feature_projector(student_hidden)
        # مطابقة طول التسلسل (الطالب والمعلّم لهما نفس max_length هنا)
        feature_loss = F.mse_loss(projected_student_hidden, teacher_hidden)

        # === الدمج النهائي بالأوزان المحددة في config ===
        alpha, beta = self.config.alpha, self.config.beta
        gamma = 1 - alpha - beta  # وزن soft loss

        total_loss = (
            alpha * hard_loss +
            gamma * soft_loss +
            beta * feature_loss
        )

        return total_loss, {
            "hard": hard_loss.item(),
            "soft": soft_loss.item(),
            "feature": feature_loss.item()
        }

    def train(self, train_dataset, eval_dataset=None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        params = list(self.student_model.parameters()) + \
                 list(self.feature_projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )

        print("🚀 بدء عملية التقطير المعرفي...\n")
        self.student_model.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = {"total": [], "hard": [], "soft": [], "feature": []}

            for batch in train_loader:
                optimizer.zero_grad()
                loss, loss_components = self.compute_combined_loss(batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses["total"].append(loss.item())
                for k, v in loss_components.items():
                    epoch_losses[k].append(v)

            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(
                f"📘 Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Total={avg['total']:.4f} | Hard={avg['hard']:.4f} | "
                f"Soft={avg['soft']:.4f} | Feature={avg['feature']:.4f}"
            )

            if eval_dataset is not None:
                accuracy = self.evaluate(eval_dataset)
                print(f"   ✅ دقة التقييم: {accuracy:.2%}")

        print("\n🎉 اكتمل التقطير المعرفي بنجاح!")

    @torch.no_grad()
    def evaluate(self, dataset):
        """تقييم دقة الطالب مقابل الـ labels الحقيقية"""
        self.student_model.eval()
        loader = DataLoader(dataset, batch_size=32)
        correct, total = 0, 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.student_model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        self.student_model.train()
        return correct / total

    @torch.no_grad()
    def compare_teacher_student_speed(self, sample_text: str, n_runs: int = 100):
        """
        🏁 مقارنة عملية: سرعة الاستدلال بين المعلّم والطالب
        """
        import time

        inputs_teacher = self.teacher_tokenizer(
            sample_text, return_tensors="pt", padding="max_length",
            max_length=self.config.max_seq_length, truncation=True
        ).to(device)

        # تسخين (Warmup) لتجنب قياس وقت تحميل الذاكرة الأولي
        for _ in range(5):
            _ = self.teacher_model(**inputs_teacher)
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )

        # قياس زمن المعلّم
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.teacher_model(**inputs_teacher)
        teacher_time = (time.perf_counter() - start) / n_runs

        # قياس زمن الطالب
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.student_model(
                inputs_teacher["input_ids"], inputs_teacher["attention_mask"]
            )
        student_time = (time.perf_counter() - start) / n_runs

        print(f"\n⚡ مقارنة السرعة (متوسط {n_runs} تشغيلة):")
        print(f"   المعلّم: {teacher_time*1000:.2f} ms")
        print(f"   الطالب:  {student_time*1000:.2f} ms")
        print(f"   🚀 تسريع: {teacher_time/student_time:.1f}x أسرع!")

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-29-chapter-29.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
def main():
    """تشغيل تجربة التقطير المعرفي الكاملة"""

    # بيانات تجريبية لتصنيف المشاعر (في الواقع: IMDB أو SST-2)
    train_texts = [
        "This movie was absolutely fantastic, loved every minute!",
        "Terrible waste of time, one of the worst films I've seen.",
        "A masterpiece of modern cinema, truly inspiring.",
        "Boring and predictable, I wouldn't recommend it.",
        "The acting was superb and the plot kept me engaged.",
        "Disappointing sequel that fails to capture the original's magic.",
    ] * 50  # تكرار للمحاكاة فقط

    train_labels = [1, 0, 1, 0, 1, 0] * 50

    eval_texts = train_texts[:40]
    eval_labels = train_labels[:40]

    trainer = DistillationTrainer(config)

    # بناء Dataset (يحسب logits المعلّم مسبقًا)
    train_dataset = DistillationDataset(
        train_texts, train_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )
    eval_dataset = DistillationDataset(
        eval_texts, eval_labels,
        trainer.teacher_tokenizer, trainer.teacher_model,
        max_length=config.max_seq_length
    )

    # التدريب
    trainer.train(train_dataset, eval_dataset)

    # مقارنة السرعة بين المعلّم والطالب
    trainer.compare_teacher_student_speed(
        "This film exceeded all my expectations!"
    )


if __name__ == "__main__":
    main()
