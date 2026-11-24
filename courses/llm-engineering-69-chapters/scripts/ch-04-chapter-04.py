#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم والتعميم — Overfitting وRegularization وPerplexity
Large Language Model Engineering — Overfitting

Book slug : llm-engineering-69-chapters
Chapter   : ch-04-chapter-04 (lesson 4)
Code files: 18 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-04-chapter-04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم والتعميم — Overfitting وRegularization وPerplexity
Large Language Model Engineering — Overfitting

Book slug : llm-engineering-69-chapters
Chapter   : ch-04-chapter-04 (lesson 4)
Code files: 17 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-04-chapter-04.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم والتعميم — Overfitting وRegularization وPerplexity
Large Language Model Engineering — Overfitting

Book slug : llm-engineering-69-chapters
Chapter   : ch-04-chapter-04 (lesson 4)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 04: الفصل الرابع: التقييم والتعميم — Overfitting وRegularization وPerplexity."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_validation() -> None:
    train = [0.8, 0.7, 0.9, 0.6]
    validation = [0.4, 0.5, 0.3]
    tokens = [0.2, 0.3, 0.25]
    perplexity = math.exp(-sum((math.log(p) for p in tokens)) / len(tokens))
    print('train mean loss:', round(statistics.mean(train), 3))
    print('validation mean loss:', round(statistics.mean(validation), 3))
    print('generalization gap:', round(statistics.mean(validation) - statistics.mean(train), 3))
    print('toy-token perplexity:', round(perplexity, 3))

if __name__ == "__main__":
    print("Chapter 04 lab: Generalization and perplexity")
    run_validation()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
dropout_and_weight_decay.py
تطبيق Inverted Dropout يدويًا من الصفر مقارنة بـ nn.Dropout الرسمي،
وتوضيح تجريبي لأثر Weight Decay على حجم الأوزان عبر الزمن.
Python 3.13+ | PyTorch
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ManualDropout(nn.Module):
    """تطبيق Inverted Dropout يدويًا؛ سلوكه مطابق لـ nn.Dropout الرسمي."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p يجب أن تقع في المجال [0, 1)، وردت القيمة: {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob


def compare_dropout_implementations() -> None:
    torch.manual_seed(0)
    x = torch.ones(3, 10)

    manual = ManualDropout(p=0.4)
    official = nn.Dropout(p=0.4)

    manual.train(); official.train()
    print("=== أثناء training (قيم متوقَّعة: صفر أو 1/(1-p)=1.667) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])

    manual.eval(); official.eval()
    print("\n=== أثناء eval (يجب أن تكون مطابقة تمامًا لـ x) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])


def demonstrate_weight_decay_effect(num_steps: int = 200) -> None:
    """يُظهر تجريبيًا كيف يُقلّص weight_decay حجم الأوزان مع ثبات كل شيء آخر."""
    torch.manual_seed(1)

    model_a = nn.Linear(20, 20)
    model_b = nn.Linear(20, 20)
    model_b.load_state_dict(model_a.state_dict())  # ابدأ من نفس الأوزان تمامًا

    opt_a = torch.optim.AdamW(model_a.parameters(), lr=0.05, weight_decay=0.0)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.05, weight_decay=0.3)

    dummy_x = torch.randn(16, 20)
    initial_norm = model_a.weight.norm().item()

    for _ in range(num_steps):
        for model, opt in [(model_a, opt_a), (model_b, opt_b)]:
            opt.zero_grad(set_to_none=True)
            # خسارة تعسفية بسيطة فقط لإظهار أثر الـ decay على حجم الأوزان
            loss = model(dummy_x).pow(2).mean()
            loss.backward()
            opt.step()

    print(f"\nمعيار الأوزان الابتدائي       : {initial_norm:.4f}")
    print(f"بدون weight_decay (النهائي)   : {model_a.weight.norm().item():.4f}")
    print(f"مع weight_decay=0.3 (النهائي) : {model_b.weight.norm().item():.4f}  ← أصغر بوضوح")


if __name__ == "__main__":
    compare_dropout_implementations()
    demonstrate_weight_decay_effect()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
early_stopping.py
تطبيق Early Stopping احترافي مع حفظ واسترجاع أفضل حالة للنموذج تلقائيًا.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float = field(default=float("inf"), init=False)
    counter: int = field(default=0, init=False)
    best_state_dict: dict | None = field(default=None, init=False)
    should_stop: bool = field(default=False, init=False)
    best_epoch: int = field(default=-1, init=False)

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print(f"✅ تمت استعادة أفضل نسخة من epoch={self.best_epoch} "
                  f"(val_loss={self.best_loss:.4f})")


if __name__ == "__main__":
    # محاكاة سريعة لتوضيح السلوك دون تدريب فعلي
    simulated_val_losses = [2.1, 1.8, 1.5, 1.4, 1.42, 1.45, 1.5, 1.6]
    stopper = EarlyStopping(patience=3, min_delta=1e-3)
    dummy_model = nn.Linear(4, 4)

    for epoch, vloss in enumerate(simulated_val_losses):
        stopper.step(vloss, dummy_model, epoch)
        print(f"epoch={epoch} val_loss={vloss} counter={stopper.counter} "
              f"should_stop={stopper.should_stop}")
        if stopper.should_stop:
            print(f"\n⏹️ توقف عند epoch={epoch}، لكن أفضل نتيجة كانت في epoch={stopper.best_epoch}")
            break

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
perplexity_metrics.py
حساب Perplexity بدقة من Cross-Entropy Loss، مع معالجة صحيحة لرموز الحشو (Padding)،
ومقارنة Perplexity على جمل "متوقَّعة" مقابل جمل "عشوائية".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def perplexity_from_loss(cross_entropy_nats: float) -> float:
    """العلاقة الجوهرية: PPL = e^(loss)، بشرط أن تكون loss بوحدة nats (ln الطبيعي)."""
    return math.exp(cross_entropy_nats)


@torch.no_grad()
def corpus_perplexity(
    model: nn.Module,
    token_ids: torch.Tensor,     # (batch, seq_len)
    pad_token_id: int = 0,
) -> float:
    """
    حساب Perplexity على كامل مجموعة بيانات، مع تجاهل توكنز الحشو تمامًا
    عبر ignore_index — خطوة حرجة: تجاهلها يُحرّف النتيجة بشكل كبير إحصائيًا
    إذا كانت نسبة الحشو عالية (شائع جدًا مع تسلسلات متغيّرة الطول).
    """
    model.eval()
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    logits = model(inputs)  # (batch, seq_len-1, vocab_size)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
    )
    return perplexity_from_loss(loss.item())


if __name__ == "__main__":
    vocab_size, seq_len, batch_size = 500, 15, 4

    class TinyLM(nn.Module):
        def __init__(self, vocab_size: int, confident: bool) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 16)
            self.proj = nn.Linear(16, vocab_size)
            self.confident = confident  # لمحاكاة نموذج "واثق" اصطناعيًا للتوضيح فقط

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.proj(self.embed(x))
            if self.confident:
                logits = logits * 10.0  # تضخيم الفروقات = ثقة أعلى بمن توقعه النموذج
            return logits

    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))

    random_model = TinyLM(vocab_size, confident=False)
    confident_model = TinyLM(vocab_size, confident=True)
    # نسخ نفس الأوزان الابتدائية لضمان مقارنة عادلة لتأثير "الثقة" فقط
    confident_model.load_state_dict(
        {k: v.clone() for k, v in random_model.state_dict().items()}
    )

    ppl_random = corpus_perplexity(random_model, tokens)
    ppl_confident = corpus_perplexity(confident_model, tokens)

    print(f"Perplexity (نموذج عشوائي التنبؤ): {ppl_random:.2f}")
    print(f"Perplexity (نفس الأوزان لكن logits مُضخَّمة): {ppl_confident:.2f}")
    print(f"\n(ملاحظة تعليمية: تضخيم logits عشوائيًا دون تدريب حقيقي "
          f"لا يعني بالضرورة تحسنًا — فقط يُغيّر حدة التوزيع، "
          f"وقد يُسيء للـ PPL إن كانت الثقة في غير محلّها!)")

    # تذكير مباشر بالعلاقة الأساسية
    example_loss = 2.3
    print(f"\nمثال مباشر: Cross-Entropy Loss = {example_loss} nats")
    print(f"=> Perplexity = exp({example_loss}) = {perplexity_from_loss(example_loss):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_end_to_end.py
تجربة كاملة: توليد Overfitting متعمّد، تشخيصه عبر Perplexity،
ثم علاجه بـ Dropout + Weight Decay + Early Stopping.
نموذج لغة بسيط على مستوى الحرف (character-level) لتسريع التجربة وتوضيحها.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass(frozen=True)
class CharLMConfig:
    vocab_size: int
    embed_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.0


class CharLM(nn.Module):
    """نموذج لغة بسيط: Embedding -> GRU -> Linear، لغرض توضيحي تعليمي."""

    def __init__(self, cfg: CharLMConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop_in = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.drop_out = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_in(self.embed(x))
        out, _ = self.gru(h)
        out = self.drop_out(out)
        return self.proj(out)


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def batchify(data: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
    n_chunks = (len(data) - 1) // (seq_len * batch_size)
    usable = n_chunks * seq_len * batch_size
    return data[:usable].view(batch_size, -1)


@torch.no_grad()
def eval_perplexity(model: nn.Module, data: torch.Tensor, seq_len: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, data.size(1) - seq_len - 1, seq_len):
        xb = data[:, i : i + seq_len]
        yb = data[:, i + 1 : i + seq_len + 1]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += loss.item() * yb.numel()
        total_tokens += yb.numel()
    return math.exp(total_loss / total_tokens)


def run_training(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    num_epochs: int,
    weight_decay: float = 0.0,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    train_hist, val_hist = [], []

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, train_data.size(1) - seq_len - 1, seq_len):
            xb = train_data[:, i : i + seq_len]
            yb = train_data[:, i + 1 : i + seq_len + 1]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_ppl = eval_perplexity(model, train_data, seq_len)
        val_ppl = eval_perplexity(model, val_data, seq_len)
        train_hist.append(train_ppl)
        val_hist.append(val_ppl)
        gap_marker = "⚠️" if (val_ppl - train_ppl) > train_ppl else ""
        print(f"Epoch {epoch+1:02d} | Train PPL={train_ppl:7.2f} | Val PPL={val_ppl:7.2f} {gap_marker}")

    return train_hist, val_hist


if __name__ == "__main__":
    text = (
        "في مدينة قديمة جدا كان هناك حكيم يجلس كل يوم تحت شجرة كبيرة "
        "يستمع الى اسئلة الناس ويجيب عليها بحكمة عميقة وبساطة شديدة "
    ) * 25
    split = int(0.8 * len(text))
    train_text, val_text = text[:split], text[split:]

    stoi, itos = build_vocab(text)
    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    seq_len, batch_size = 32, 8
    train_batches = batchify(train_ids, seq_len, batch_size)
    val_batches = batchify(val_ids, seq_len, batch_size)

    print("=== تجربة 1: بدون Regularization (Overfitting متوقَّع بقوة) ===")
    model_plain = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.0))
    train_hist_plain, val_hist_plain = run_training(
        model_plain, train_batches, val_batches, seq_len, num_epochs=20
    )

    print("\n=== تجربة 2: مع Dropout=0.3 + Weight Decay=0.1 ===")
    model_reg = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.3))
    train_hist_reg, val_hist_reg = run_training(
        model_reg, train_batches, val_batches, seq_len,
        num_epochs=20, weight_decay=0.1,
    )

    print("\n=== الخلاصة المقارنة ===")
    gap_plain = val_hist_plain[-1] - train_hist_plain[-1]
    gap_reg = val_hist_reg[-1] - train_hist_reg[-1]
    print(f"بدون Regularization: فجوة PPL النهائية = {gap_plain:.2f}")
    print(f"مع Regularization  : فجوة PPL النهائية = {gap_reg:.2f}")
    print(f"\n(الهدف ليس أقل Train PPL، بل أصغر فجوة مع أفضل Val PPL ممكن)")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التقييم يحدث بينما Dropout/BatchNorm لا يزالان في وضع "تدريب"
def evaluate(model, loader):
    for xb, yb in loader:
        preds = model(xb)   # Dropout يُطفئ عصبونات هنا أيضًا بالخطأ، نتيجة غير موثوقة!

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
    model.train()  # أعده لوضع train قبل متابعة حلقة التدريب

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: حساب mean/std على كل البيانات، بما فيها val/test، قبل التقسيم
mean, std = all_data.mean(), all_data.std()
normalized = (all_data - mean) / std
train, val, test = split(normalized)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم أولًا، ثم احسب الإحصائيات من Train فقط
train, val, test = split(all_data)
mean, std = train.mean(), train.std()
train = (train - mean) / std
val = (val - mean) / std      # يستخدم mean/std من train حصرًا
test = (test - mean) / std

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منهجي خطير جدًا
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    test_acc = evaluate(model, test_loader)    # تسريب! قرار مبني على test
    if test_acc > best_acc:
        best_lr = lr

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: Validation لاتخاذ القرار، Test فقط للتقرير النهائي مرة واحدة
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    val_acc = evaluate(model, val_loader)
    if val_acc > best_acc:
        best_lr = lr

final_model = train(lr=best_lr)
final_test_acc = evaluate(final_model, test_loader)  # مرة واحدة، في النهاية فقط

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ غير كافٍ ومُضلِّل لنماذج اللغة
accuracy = (predictions == targets).float().mean()

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: Perplexity كمقياس أساسي، مع مقاييس تكميلية حسب المهمة لاحقًا
ppl = corpus_perplexity(model, val_tokens, pad_token_id=PAD_ID)

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خفي وشائع: افتراض أن كل PPL محسوبة بنفس الطريقة
# بعض المكتبات تستخدم log2 بدل ln دون توضيح كافٍ في التوثيق!
ppl_from_source_a = 2 ** loss_in_bits   # bits
ppl_from_source_b = math.exp(loss_in_nats)  # nats
# مقارنة هذين الرقمين مباشرة = خطأ منهجي!

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: وحّد القاعدة اللوغاريتمية أولًا قبل أي مقارنة
# nats -> bits: bits_loss = nats_loss / ln(2)
# bits -> nats: nats_loss = bits_loss * ln(2)
nats_loss_from_b = loss_in_bits * math.log(2)
ppl_unified = math.exp(nats_loss_from_b)   # الآن قابلة للمقارنة مع ppl_from_source_b أعلاه

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_detector.py — هيكل بداية للتمرين
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DiagnosisResult(Enum):
    HEALTHY = auto()
    UNDERFITTING = auto()
    OVERFITTING = auto()
    DIVERGING = auto()


@dataclass(frozen=True)
class DiagnosisReport:
    result: DiagnosisResult
    final_train_loss: float
    final_val_loss: float
    gap: float
    recommendation: str


def diagnose_training(
    train_losses: list[float],
    val_losses: list[float],
    *,
    gap_ratio_threshold: float = 0.3,   # نسبة الفجوة إلى train_loss لاعتبارها Overfitting
    high_loss_threshold: float = 1.0,
) -> DiagnosisReport:
    """
    TODO: اكتب منطق التشخيص بالخطوات التالية:

    1. gap = val_losses[-1] - train_losses[-1]
    2. gap_ratio = gap / max(train_losses[-1], 1e-8)
    3. تحقق من DIVERGING أولًا: هل train_losses[-1] أكبر من train_losses في منتصف
       التدريب تقريبًا (مثلًا train_losses[len(train_losses)//2])؟
    4. تحقق من UNDERFITTING: train_losses[-1] مرتفع (> high_loss_threshold)
       و gap_ratio صغيرة نسبيًا (النموذج سيء على الطرفين بالتساوي تقريبًا)
    5. تحقق من OVERFITTING: gap_ratio > gap_ratio_threshold
    6. إن لم تنطبق أي حالة أعلاه: HEALTHY

    لكل حالة، أنشئ توصية نصية دقيقة وعملية (انظر الأمثلة في نص الفصل أعلاه
    كمرجع لصياغة التوصيات).
    """
    raise NotImplementedError


def plot_diagnosis(
    train_losses: list[float], val_losses: list[float], report: DiagnosisReport
) -> None:
    """
    TODO: استخدم matplotlib لرسم المنحنيين معًا، مع:
    - عنوان يتضمن نتيجة التشخيص (report.result.name)
    - تظليل خلفية بلون يعكس الحالة (أخضر فاتح=HEALTHY، أحمر فاتح=OVERFITTING،
      برتقالي فاتح=UNDERFITTING، أحمر غامق=DIVERGING)
    - نص التوصية كعنوان فرعي (subtitle) أسفل العنوان الرئيسي
    """
    raise NotImplementedError


if __name__ == "__main__":
    scenarios = {
        "healthy":      ([2.5, 1.8, 1.2, 0.9, 0.7], [2.6, 1.9, 1.3, 1.0, 0.75]),
        "overfitting":  ([2.5, 1.5, 0.8, 0.3, 0.1], [2.6, 1.8, 1.5, 1.7, 2.0]),
        "underfitting": ([2.5, 2.3, 2.2, 2.15, 2.1], [2.6, 2.4, 2.3, 2.25, 2.2]),
        "diverging":    ([2.5, 3.0, 4.5, 6.0, 9.0], [2.6, 3.1, 4.6, 6.2, 9.5]),
    }

    for name, (train_l, val_l) in scenarios.items():
        report = diagnose_training(train_l, val_l)
        status = "✅" if report.result == DiagnosisResult.HEALTHY else "⚠️"
        print(f"{status} [{name}] → {report.result.name}: {report.recommendation}")

    # تحدٍّ إضافي: جرّب تصميم سيناريو "حدّي" يقع بين Healthy وOverfitting،
    # واختبر حساسية عتبة gap_ratio_threshold — هل 0.3 عتبة مناسبة دائمًا؟
    # متى قد تحتاج لتعديلها حسب طبيعة المهمة (تصنيف بسيط vs نموذج لغة ضخم)؟

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 04: الفصل الرابع: التقييم والتعميم — Overfitting وRegularization وPerplexity."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_validation() -> None:
    train = [0.8, 0.7, 0.9, 0.6]
    validation = [0.4, 0.5, 0.3]
    tokens = [0.2, 0.3, 0.25]
    perplexity = math.exp(-sum((math.log(p) for p in tokens)) / len(tokens))
    print('train mean loss:', round(statistics.mean(train), 3))
    print('validation mean loss:', round(statistics.mean(validation), 3))
    print('generalization gap:', round(statistics.mean(validation) - statistics.mean(train), 3))
    print('toy-token perplexity:', round(perplexity, 3))

if __name__ == "__main__":
    print("Chapter 04 lab: Generalization and perplexity")
    run_validation()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
dropout_and_weight_decay.py
تطبيق Inverted Dropout يدويًا من الصفر مقارنة بـ nn.Dropout الرسمي،
وتوضيح تجريبي لأثر Weight Decay على حجم الأوزان عبر الزمن.
Python 3.13+ | PyTorch
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ManualDropout(nn.Module):
    """تطبيق Inverted Dropout يدويًا؛ سلوكه مطابق لـ nn.Dropout الرسمي."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p يجب أن تقع في المجال [0, 1)، وردت القيمة: {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob


def compare_dropout_implementations() -> None:
    torch.manual_seed(0)
    x = torch.ones(3, 10)

    manual = ManualDropout(p=0.4)
    official = nn.Dropout(p=0.4)

    manual.train(); official.train()
    print("=== أثناء training (قيم متوقَّعة: صفر أو 1/(1-p)=1.667) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])

    manual.eval(); official.eval()
    print("\n=== أثناء eval (يجب أن تكون مطابقة تمامًا لـ x) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])


def demonstrate_weight_decay_effect(num_steps: int = 200) -> None:
    """يُظهر تجريبيًا كيف يُقلّص weight_decay حجم الأوزان مع ثبات كل شيء آخر."""
    torch.manual_seed(1)

    model_a = nn.Linear(20, 20)
    model_b = nn.Linear(20, 20)
    model_b.load_state_dict(model_a.state_dict())  # ابدأ من نفس الأوزان تمامًا

    opt_a = torch.optim.AdamW(model_a.parameters(), lr=0.05, weight_decay=0.0)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.05, weight_decay=0.3)

    dummy_x = torch.randn(16, 20)
    initial_norm = model_a.weight.norm().item()

    for _ in range(num_steps):
        for model, opt in [(model_a, opt_a), (model_b, opt_b)]:
            opt.zero_grad(set_to_none=True)
            # خسارة تعسفية بسيطة فقط لإظهار أثر الـ decay على حجم الأوزان
            loss = model(dummy_x).pow(2).mean()
            loss.backward()
            opt.step()

    print(f"\nمعيار الأوزان الابتدائي       : {initial_norm:.4f}")
    print(f"بدون weight_decay (النهائي)   : {model_a.weight.norm().item():.4f}")
    print(f"مع weight_decay=0.3 (النهائي) : {model_b.weight.norm().item():.4f}  ← أصغر بوضوح")


if __name__ == "__main__":
    compare_dropout_implementations()
    demonstrate_weight_decay_effect()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
early_stopping.py
تطبيق Early Stopping احترافي مع حفظ واسترجاع أفضل حالة للنموذج تلقائيًا.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float = field(default=float("inf"), init=False)
    counter: int = field(default=0, init=False)
    best_state_dict: dict | None = field(default=None, init=False)
    should_stop: bool = field(default=False, init=False)
    best_epoch: int = field(default=-1, init=False)

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print(f"✅ تمت استعادة أفضل نسخة من epoch={self.best_epoch} "
                  f"(val_loss={self.best_loss:.4f})")


if __name__ == "__main__":
    # محاكاة سريعة لتوضيح السلوك دون تدريب فعلي
    simulated_val_losses = [2.1, 1.8, 1.5, 1.4, 1.42, 1.45, 1.5, 1.6]
    stopper = EarlyStopping(patience=3, min_delta=1e-3)
    dummy_model = nn.Linear(4, 4)

    for epoch, vloss in enumerate(simulated_val_losses):
        stopper.step(vloss, dummy_model, epoch)
        print(f"epoch={epoch} val_loss={vloss} counter={stopper.counter} "
              f"should_stop={stopper.should_stop}")
        if stopper.should_stop:
            print(f"\n⏹️ توقف عند epoch={epoch}، لكن أفضل نتيجة كانت في epoch={stopper.best_epoch}")
            break

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
perplexity_metrics.py
حساب Perplexity بدقة من Cross-Entropy Loss، مع معالجة صحيحة لرموز الحشو (Padding)،
ومقارنة Perplexity على جمل "متوقَّعة" مقابل جمل "عشوائية".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def perplexity_from_loss(cross_entropy_nats: float) -> float:
    """العلاقة الجوهرية: PPL = e^(loss)، بشرط أن تكون loss بوحدة nats (ln الطبيعي)."""
    return math.exp(cross_entropy_nats)


@torch.no_grad()
def corpus_perplexity(
    model: nn.Module,
    token_ids: torch.Tensor,     # (batch, seq_len)
    pad_token_id: int = 0,
) -> float:
    """
    حساب Perplexity على كامل مجموعة بيانات، مع تجاهل توكنز الحشو تمامًا
    عبر ignore_index — خطوة حرجة: تجاهلها يُحرّف النتيجة بشكل كبير إحصائيًا
    إذا كانت نسبة الحشو عالية (شائع جدًا مع تسلسلات متغيّرة الطول).
    """
    model.eval()
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    logits = model(inputs)  # (batch, seq_len-1, vocab_size)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
    )
    return perplexity_from_loss(loss.item())


if __name__ == "__main__":
    vocab_size, seq_len, batch_size = 500, 15, 4

    class TinyLM(nn.Module):
        def __init__(self, vocab_size: int, confident: bool) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 16)
            self.proj = nn.Linear(16, vocab_size)
            self.confident = confident  # لمحاكاة نموذج "واثق" اصطناعيًا للتوضيح فقط

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.proj(self.embed(x))
            if self.confident:
                logits = logits * 10.0  # تضخيم الفروقات = ثقة أعلى بمن توقعه النموذج
            return logits

    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))

    random_model = TinyLM(vocab_size, confident=False)
    confident_model = TinyLM(vocab_size, confident=True)
    # نسخ نفس الأوزان الابتدائية لضمان مقارنة عادلة لتأثير "الثقة" فقط
    confident_model.load_state_dict(
        {k: v.clone() for k, v in random_model.state_dict().items()}
    )

    ppl_random = corpus_perplexity(random_model, tokens)
    ppl_confident = corpus_perplexity(confident_model, tokens)

    print(f"Perplexity (نموذج عشوائي التنبؤ): {ppl_random:.2f}")
    print(f"Perplexity (نفس الأوزان لكن logits مُضخَّمة): {ppl_confident:.2f}")
    print(f"\n(ملاحظة تعليمية: تضخيم logits عشوائيًا دون تدريب حقيقي "
          f"لا يعني بالضرورة تحسنًا — فقط يُغيّر حدة التوزيع، "
          f"وقد يُسيء للـ PPL إن كانت الثقة في غير محلّها!)")

    # تذكير مباشر بالعلاقة الأساسية
    example_loss = 2.3
    print(f"\nمثال مباشر: Cross-Entropy Loss = {example_loss} nats")
    print(f"=> Perplexity = exp({example_loss}) = {perplexity_from_loss(example_loss):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_end_to_end.py
تجربة كاملة: توليد Overfitting متعمّد، تشخيصه عبر Perplexity،
ثم علاجه بـ Dropout + Weight Decay + Early Stopping.
نموذج لغة بسيط على مستوى الحرف (character-level) لتسريع التجربة وتوضيحها.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass(frozen=True)
class CharLMConfig:
    vocab_size: int
    embed_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.0


class CharLM(nn.Module):
    """نموذج لغة بسيط: Embedding -> GRU -> Linear، لغرض توضيحي تعليمي."""

    def __init__(self, cfg: CharLMConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop_in = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.drop_out = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_in(self.embed(x))
        out, _ = self.gru(h)
        out = self.drop_out(out)
        return self.proj(out)


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def batchify(data: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
    n_chunks = (len(data) - 1) // (seq_len * batch_size)
    usable = n_chunks * seq_len * batch_size
    return data[:usable].view(batch_size, -1)


@torch.no_grad()
def eval_perplexity(model: nn.Module, data: torch.Tensor, seq_len: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, data.size(1) - seq_len - 1, seq_len):
        xb = data[:, i : i + seq_len]
        yb = data[:, i + 1 : i + seq_len + 1]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += loss.item() * yb.numel()
        total_tokens += yb.numel()
    return math.exp(total_loss / total_tokens)


def run_training(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    num_epochs: int,
    weight_decay: float = 0.0,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    train_hist, val_hist = [], []

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, train_data.size(1) - seq_len - 1, seq_len):
            xb = train_data[:, i : i + seq_len]
            yb = train_data[:, i + 1 : i + seq_len + 1]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_ppl = eval_perplexity(model, train_data, seq_len)
        val_ppl = eval_perplexity(model, val_data, seq_len)
        train_hist.append(train_ppl)
        val_hist.append(val_ppl)
        gap_marker = "⚠️" if (val_ppl - train_ppl) > train_ppl else ""
        print(f"Epoch {epoch+1:02d} | Train PPL={train_ppl:7.2f} | Val PPL={val_ppl:7.2f} {gap_marker}")

    return train_hist, val_hist


if __name__ == "__main__":
    text = (
        "في مدينة قديمة جدا كان هناك حكيم يجلس كل يوم تحت شجرة كبيرة "
        "يستمع الى اسئلة الناس ويجيب عليها بحكمة عميقة وبساطة شديدة "
    ) * 25
    split = int(0.8 * len(text))
    train_text, val_text = text[:split], text[split:]

    stoi, itos = build_vocab(text)
    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    seq_len, batch_size = 32, 8
    train_batches = batchify(train_ids, seq_len, batch_size)
    val_batches = batchify(val_ids, seq_len, batch_size)

    print("=== تجربة 1: بدون Regularization (Overfitting متوقَّع بقوة) ===")
    model_plain = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.0))
    train_hist_plain, val_hist_plain = run_training(
        model_plain, train_batches, val_batches, seq_len, num_epochs=20
    )

    print("\n=== تجربة 2: مع Dropout=0.3 + Weight Decay=0.1 ===")
    model_reg = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.3))
    train_hist_reg, val_hist_reg = run_training(
        model_reg, train_batches, val_batches, seq_len,
        num_epochs=20, weight_decay=0.1,
    )

    print("\n=== الخلاصة المقارنة ===")
    gap_plain = val_hist_plain[-1] - train_hist_plain[-1]
    gap_reg = val_hist_reg[-1] - train_hist_reg[-1]
    print(f"بدون Regularization: فجوة PPL النهائية = {gap_plain:.2f}")
    print(f"مع Regularization  : فجوة PPL النهائية = {gap_reg:.2f}")
    print(f"\n(الهدف ليس أقل Train PPL، بل أصغر فجوة مع أفضل Val PPL ممكن)")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التقييم يحدث بينما Dropout/BatchNorm لا يزالان في وضع "تدريب"
def evaluate(model, loader):
    for xb, yb in loader:
        preds = model(xb)   # Dropout يُطفئ عصبونات هنا أيضًا بالخطأ، نتيجة غير موثوقة!

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
    model.train()  # أعده لوضع train قبل متابعة حلقة التدريب

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: حساب mean/std على كل البيانات، بما فيها val/test، قبل التقسيم
mean, std = all_data.mean(), all_data.std()
normalized = (all_data - mean) / std
train, val, test = split(normalized)

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم أولًا، ثم احسب الإحصائيات من Train فقط
train, val, test = split(all_data)
mean, std = train.mean(), train.std()
train = (train - mean) / std
val = (val - mean) / std      # يستخدم mean/std من train حصرًا
test = (test - mean) / std

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منهجي خطير جدًا
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    test_acc = evaluate(model, test_loader)    # تسريب! قرار مبني على test
    if test_acc > best_acc:
        best_lr = lr

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: Validation لاتخاذ القرار، Test فقط للتقرير النهائي مرة واحدة
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    val_acc = evaluate(model, val_loader)
    if val_acc > best_acc:
        best_lr = lr

final_model = train(lr=best_lr)
final_test_acc = evaluate(final_model, test_loader)  # مرة واحدة، في النهاية فقط

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ غير كافٍ ومُضلِّل لنماذج اللغة
accuracy = (predictions == targets).float().mean()

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: Perplexity كمقياس أساسي، مع مقاييس تكميلية حسب المهمة لاحقًا
ppl = corpus_perplexity(model, val_tokens, pad_token_id=PAD_ID)

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خفي وشائع: افتراض أن كل PPL محسوبة بنفس الطريقة
# بعض المكتبات تستخدم log2 بدل ln دون توضيح كافٍ في التوثيق!
ppl_from_source_a = 2 ** loss_in_bits   # bits
ppl_from_source_b = math.exp(loss_in_nats)  # nats
# مقارنة هذين الرقمين مباشرة = خطأ منهجي!

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: وحّد القاعدة اللوغاريتمية أولًا قبل أي مقارنة
# nats -> bits: bits_loss = nats_loss / ln(2)
# bits -> nats: nats_loss = bits_loss * ln(2)
nats_loss_from_b = loss_in_bits * math.log(2)
ppl_unified = math.exp(nats_loss_from_b)   # الآن قابلة للمقارنة مع ppl_from_source_b أعلاه

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_detector.py — هيكل بداية للتمرين
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DiagnosisResult(Enum):
    HEALTHY = auto()
    UNDERFITTING = auto()
    OVERFITTING = auto()
    DIVERGING = auto()


@dataclass(frozen=True)
class DiagnosisReport:
    result: DiagnosisResult
    final_train_loss: float
    final_val_loss: float
    gap: float
    recommendation: str


def diagnose_training(
    train_losses: list[float],
    val_losses: list[float],
    *,
    gap_ratio_threshold: float = 0.3,   # نسبة الفجوة إلى train_loss لاعتبارها Overfitting
    high_loss_threshold: float = 1.0,
) -> DiagnosisReport:
    """
    TODO: اكتب منطق التشخيص بالخطوات التالية:

    1. gap = val_losses[-1] - train_losses[-1]
    2. gap_ratio = gap / max(train_losses[-1], 1e-8)
    3. تحقق من DIVERGING أولًا: هل train_losses[-1] أكبر من train_losses في منتصف
       التدريب تقريبًا (مثلًا train_losses[len(train_losses)//2])؟
    4. تحقق من UNDERFITTING: train_losses[-1] مرتفع (> high_loss_threshold)
       و gap_ratio صغيرة نسبيًا (النموذج سيء على الطرفين بالتساوي تقريبًا)
    5. تحقق من OVERFITTING: gap_ratio > gap_ratio_threshold
    6. إن لم تنطبق أي حالة أعلاه: HEALTHY

    لكل حالة، أنشئ توصية نصية دقيقة وعملية (انظر الأمثلة في نص الفصل أعلاه
    كمرجع لصياغة التوصيات).
    """
    raise NotImplementedError


def plot_diagnosis(
    train_losses: list[float], val_losses: list[float], report: DiagnosisReport
) -> None:
    """
    TODO: استخدم matplotlib لرسم المنحنيين معًا، مع:
    - عنوان يتضمن نتيجة التشخيص (report.result.name)
    - تظليل خلفية بلون يعكس الحالة (أخضر فاتح=HEALTHY، أحمر فاتح=OVERFITTING،
      برتقالي فاتح=UNDERFITTING، أحمر غامق=DIVERGING)
    - نص التوصية كعنوان فرعي (subtitle) أسفل العنوان الرئيسي
    """
    raise NotImplementedError


if __name__ == "__main__":
    scenarios = {
        "healthy":      ([2.5, 1.8, 1.2, 0.9, 0.7], [2.6, 1.9, 1.3, 1.0, 0.75]),
        "overfitting":  ([2.5, 1.5, 0.8, 0.3, 0.1], [2.6, 1.8, 1.5, 1.7, 2.0]),
        "underfitting": ([2.5, 2.3, 2.2, 2.15, 2.1], [2.6, 2.4, 2.3, 2.25, 2.2]),
        "diverging":    ([2.5, 3.0, 4.5, 6.0, 9.0], [2.6, 3.1, 4.6, 6.2, 9.5]),
    }

    for name, (train_l, val_l) in scenarios.items():
        report = diagnose_training(train_l, val_l)
        status = "✅" if report.result == DiagnosisResult.HEALTHY else "⚠️"
        print(f"{status} [{name}] → {report.result.name}: {report.recommendation}")

    # تحدٍّ إضافي: جرّب تصميم سيناريو "حدّي" يقع بين Healthy وOverfitting،
    # واختبر حساسية عتبة gap_ratio_threshold — هل 0.3 عتبة مناسبة دائمًا؟
    # متى قد تحتاج لتعديلها حسب طبيعة المهمة (تصنيف بسيط vs نموذج لغة ضخم)؟

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — التقييم والتعميم — Overfitting وRegularization وPerplexity
Large Language Model Engineering — Overfitting

Book slug : llm-engineering-69-chapters
Chapter   : ch-04-chapter-04 (lesson 4)
Code files: 16 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_04.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 04: الفصل الرابع: التقييم والتعميم — Overfitting وRegularization وPerplexity."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_validation() -> None:
    train = [0.8, 0.7, 0.9, 0.6]
    validation = [0.4, 0.5, 0.3]
    tokens = [0.2, 0.3, 0.25]
    perplexity = math.exp(-sum((math.log(p) for p in tokens)) / len(tokens))
    print('train mean loss:', round(statistics.mean(train), 3))
    print('validation mean loss:', round(statistics.mean(validation), 3))
    print('generalization gap:', round(statistics.mean(validation) - statistics.mean(train), 3))
    print('toy-token perplexity:', round(perplexity, 3))

if __name__ == "__main__":
    print("Chapter 04 lab: Generalization and perplexity")
    run_validation()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
dropout_and_weight_decay.py
تطبيق Inverted Dropout يدويًا من الصفر مقارنة بـ nn.Dropout الرسمي،
وتوضيح تجريبي لأثر Weight Decay على حجم الأوزان عبر الزمن.
Python 3.13+ | PyTorch
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ManualDropout(nn.Module):
    """تطبيق Inverted Dropout يدويًا؛ سلوكه مطابق لـ nn.Dropout الرسمي."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p يجب أن تقع في المجال [0, 1)، وردت القيمة: {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob


def compare_dropout_implementations() -> None:
    torch.manual_seed(0)
    x = torch.ones(3, 10)

    manual = ManualDropout(p=0.4)
    official = nn.Dropout(p=0.4)

    manual.train(); official.train()
    print("=== أثناء training (قيم متوقَّعة: صفر أو 1/(1-p)=1.667) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])

    manual.eval(); official.eval()
    print("\n=== أثناء eval (يجب أن تكون مطابقة تمامًا لـ x) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])


def demonstrate_weight_decay_effect(num_steps: int = 200) -> None:
    """يُظهر تجريبيًا كيف يُقلّص weight_decay حجم الأوزان مع ثبات كل شيء آخر."""
    torch.manual_seed(1)

    model_a = nn.Linear(20, 20)
    model_b = nn.Linear(20, 20)
    model_b.load_state_dict(model_a.state_dict())  # ابدأ من نفس الأوزان تمامًا

    opt_a = torch.optim.AdamW(model_a.parameters(), lr=0.05, weight_decay=0.0)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.05, weight_decay=0.3)

    dummy_x = torch.randn(16, 20)
    initial_norm = model_a.weight.norm().item()

    for _ in range(num_steps):
        for model, opt in [(model_a, opt_a), (model_b, opt_b)]:
            opt.zero_grad(set_to_none=True)
            # خسارة تعسفية بسيطة فقط لإظهار أثر الـ decay على حجم الأوزان
            loss = model(dummy_x).pow(2).mean()
            loss.backward()
            opt.step()

    print(f"\nمعيار الأوزان الابتدائي       : {initial_norm:.4f}")
    print(f"بدون weight_decay (النهائي)   : {model_a.weight.norm().item():.4f}")
    print(f"مع weight_decay=0.3 (النهائي) : {model_b.weight.norm().item():.4f}  ← أصغر بوضوح")


if __name__ == "__main__":
    compare_dropout_implementations()
    demonstrate_weight_decay_effect()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
early_stopping.py
تطبيق Early Stopping احترافي مع حفظ واسترجاع أفضل حالة للنموذج تلقائيًا.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float = field(default=float("inf"), init=False)
    counter: int = field(default=0, init=False)
    best_state_dict: dict | None = field(default=None, init=False)
    should_stop: bool = field(default=False, init=False)
    best_epoch: int = field(default=-1, init=False)

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print(f"✅ تمت استعادة أفضل نسخة من epoch={self.best_epoch} "
                  f"(val_loss={self.best_loss:.4f})")


if __name__ == "__main__":
    # محاكاة سريعة لتوضيح السلوك دون تدريب فعلي
    simulated_val_losses = [2.1, 1.8, 1.5, 1.4, 1.42, 1.45, 1.5, 1.6]
    stopper = EarlyStopping(patience=3, min_delta=1e-3)
    dummy_model = nn.Linear(4, 4)

    for epoch, vloss in enumerate(simulated_val_losses):
        stopper.step(vloss, dummy_model, epoch)
        print(f"epoch={epoch} val_loss={vloss} counter={stopper.counter} "
              f"should_stop={stopper.should_stop}")
        if stopper.should_stop:
            print(f"\n⏹️ توقف عند epoch={epoch}، لكن أفضل نتيجة كانت في epoch={stopper.best_epoch}")
            break

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
perplexity_metrics.py
حساب Perplexity بدقة من Cross-Entropy Loss، مع معالجة صحيحة لرموز الحشو (Padding)،
ومقارنة Perplexity على جمل "متوقَّعة" مقابل جمل "عشوائية".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def perplexity_from_loss(cross_entropy_nats: float) -> float:
    """العلاقة الجوهرية: PPL = e^(loss)، بشرط أن تكون loss بوحدة nats (ln الطبيعي)."""
    return math.exp(cross_entropy_nats)


@torch.no_grad()
def corpus_perplexity(
    model: nn.Module,
    token_ids: torch.Tensor,     # (batch, seq_len)
    pad_token_id: int = 0,
) -> float:
    """
    حساب Perplexity على كامل مجموعة بيانات، مع تجاهل توكنز الحشو تمامًا
    عبر ignore_index — خطوة حرجة: تجاهلها يُحرّف النتيجة بشكل كبير إحصائيًا
    إذا كانت نسبة الحشو عالية (شائع جدًا مع تسلسلات متغيّرة الطول).
    """
    model.eval()
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    logits = model(inputs)  # (batch, seq_len-1, vocab_size)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
    )
    return perplexity_from_loss(loss.item())


if __name__ == "__main__":
    vocab_size, seq_len, batch_size = 500, 15, 4

    class TinyLM(nn.Module):
        def __init__(self, vocab_size: int, confident: bool) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 16)
            self.proj = nn.Linear(16, vocab_size)
            self.confident = confident  # لمحاكاة نموذج "واثق" اصطناعيًا للتوضيح فقط

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.proj(self.embed(x))
            if self.confident:
                logits = logits * 10.0  # تضخيم الفروقات = ثقة أعلى بمن توقعه النموذج
            return logits

    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))

    random_model = TinyLM(vocab_size, confident=False)
    confident_model = TinyLM(vocab_size, confident=True)
    # نسخ نفس الأوزان الابتدائية لضمان مقارنة عادلة لتأثير "الثقة" فقط
    confident_model.load_state_dict(
        {k: v.clone() for k, v in random_model.state_dict().items()}
    )

    ppl_random = corpus_perplexity(random_model, tokens)
    ppl_confident = corpus_perplexity(confident_model, tokens)

    print(f"Perplexity (نموذج عشوائي التنبؤ): {ppl_random:.2f}")
    print(f"Perplexity (نفس الأوزان لكن logits مُضخَّمة): {ppl_confident:.2f}")
    print(f"\n(ملاحظة تعليمية: تضخيم logits عشوائيًا دون تدريب حقيقي "
          f"لا يعني بالضرورة تحسنًا — فقط يُغيّر حدة التوزيع، "
          f"وقد يُسيء للـ PPL إن كانت الثقة في غير محلّها!)")

    # تذكير مباشر بالعلاقة الأساسية
    example_loss = 2.3
    print(f"\nمثال مباشر: Cross-Entropy Loss = {example_loss} nats")
    print(f"=> Perplexity = exp({example_loss}) = {perplexity_from_loss(example_loss):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_end_to_end.py
تجربة كاملة: توليد Overfitting متعمّد، تشخيصه عبر Perplexity،
ثم علاجه بـ Dropout + Weight Decay + Early Stopping.
نموذج لغة بسيط على مستوى الحرف (character-level) لتسريع التجربة وتوضيحها.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass(frozen=True)
class CharLMConfig:
    vocab_size: int
    embed_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.0


class CharLM(nn.Module):
    """نموذج لغة بسيط: Embedding -> GRU -> Linear، لغرض توضيحي تعليمي."""

    def __init__(self, cfg: CharLMConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop_in = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.drop_out = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_in(self.embed(x))
        out, _ = self.gru(h)
        out = self.drop_out(out)
        return self.proj(out)


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def batchify(data: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
    n_chunks = (len(data) - 1) // (seq_len * batch_size)
    usable = n_chunks * seq_len * batch_size
    return data[:usable].view(batch_size, -1)


@torch.no_grad()
def eval_perplexity(model: nn.Module, data: torch.Tensor, seq_len: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, data.size(1) - seq_len - 1, seq_len):
        xb = data[:, i : i + seq_len]
        yb = data[:, i + 1 : i + seq_len + 1]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += loss.item() * yb.numel()
        total_tokens += yb.numel()
    return math.exp(total_loss / total_tokens)


def run_training(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    num_epochs: int,
    weight_decay: float = 0.0,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    train_hist, val_hist = [], []

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, train_data.size(1) - seq_len - 1, seq_len):
            xb = train_data[:, i : i + seq_len]
            yb = train_data[:, i + 1 : i + seq_len + 1]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_ppl = eval_perplexity(model, train_data, seq_len)
        val_ppl = eval_perplexity(model, val_data, seq_len)
        train_hist.append(train_ppl)
        val_hist.append(val_ppl)
        gap_marker = "⚠️" if (val_ppl - train_ppl) > train_ppl else ""
        print(f"Epoch {epoch+1:02d} | Train PPL={train_ppl:7.2f} | Val PPL={val_ppl:7.2f} {gap_marker}")

    return train_hist, val_hist


if __name__ == "__main__":
    text = (
        "في مدينة قديمة جدا كان هناك حكيم يجلس كل يوم تحت شجرة كبيرة "
        "يستمع الى اسئلة الناس ويجيب عليها بحكمة عميقة وبساطة شديدة "
    ) * 25
    split = int(0.8 * len(text))
    train_text, val_text = text[:split], text[split:]

    stoi, itos = build_vocab(text)
    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    seq_len, batch_size = 32, 8
    train_batches = batchify(train_ids, seq_len, batch_size)
    val_batches = batchify(val_ids, seq_len, batch_size)

    print("=== تجربة 1: بدون Regularization (Overfitting متوقَّع بقوة) ===")
    model_plain = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.0))
    train_hist_plain, val_hist_plain = run_training(
        model_plain, train_batches, val_batches, seq_len, num_epochs=20
    )

    print("\n=== تجربة 2: مع Dropout=0.3 + Weight Decay=0.1 ===")
    model_reg = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.3))
    train_hist_reg, val_hist_reg = run_training(
        model_reg, train_batches, val_batches, seq_len,
        num_epochs=20, weight_decay=0.1,
    )

    print("\n=== الخلاصة المقارنة ===")
    gap_plain = val_hist_plain[-1] - train_hist_plain[-1]
    gap_reg = val_hist_reg[-1] - train_hist_reg[-1]
    print(f"بدون Regularization: فجوة PPL النهائية = {gap_plain:.2f}")
    print(f"مع Regularization  : فجوة PPL النهائية = {gap_reg:.2f}")
    print(f"\n(الهدف ليس أقل Train PPL، بل أصغر فجوة مع أفضل Val PPL ممكن)")

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التقييم يحدث بينما Dropout/BatchNorm لا يزالان في وضع "تدريب"
def evaluate(model, loader):
    for xb, yb in loader:
        preds = model(xb)   # Dropout يُطفئ عصبونات هنا أيضًا بالخطأ، نتيجة غير موثوقة!

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
    model.train()  # أعده لوضع train قبل متابعة حلقة التدريب

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: حساب mean/std على كل البيانات، بما فيها val/test، قبل التقسيم
mean, std = all_data.mean(), all_data.std()
normalized = (all_data - mean) / std
train, val, test = split(normalized)

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم أولًا، ثم احسب الإحصائيات من Train فقط
train, val, test = split(all_data)
mean, std = train.mean(), train.std()
train = (train - mean) / std
val = (val - mean) / std      # يستخدم mean/std من train حصرًا
test = (test - mean) / std

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منهجي خطير جدًا
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    test_acc = evaluate(model, test_loader)    # تسريب! قرار مبني على test
    if test_acc > best_acc:
        best_lr = lr

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: Validation لاتخاذ القرار، Test فقط للتقرير النهائي مرة واحدة
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    val_acc = evaluate(model, val_loader)
    if val_acc > best_acc:
        best_lr = lr

final_model = train(lr=best_lr)
final_test_acc = evaluate(final_model, test_loader)  # مرة واحدة، في النهاية فقط

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ غير كافٍ ومُضلِّل لنماذج اللغة
accuracy = (predictions == targets).float().mean()

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: Perplexity كمقياس أساسي، مع مقاييس تكميلية حسب المهمة لاحقًا
ppl = corpus_perplexity(model, val_tokens, pad_token_id=PAD_ID)

# ────────────────────────────────────────────────────────────────────────
# [014] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خفي وشائع: افتراض أن كل PPL محسوبة بنفس الطريقة
# بعض المكتبات تستخدم log2 بدل ln دون توضيح كافٍ في التوثيق!
ppl_from_source_a = 2 ** loss_in_bits   # bits
ppl_from_source_b = math.exp(loss_in_nats)  # nats
# مقارنة هذين الرقمين مباشرة = خطأ منهجي!

# ────────────────────────────────────────────────────────────────────────
# [015] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: وحّد القاعدة اللوغاريتمية أولًا قبل أي مقارنة
# nats -> bits: bits_loss = nats_loss / ln(2)
# bits -> nats: nats_loss = bits_loss * ln(2)
nats_loss_from_b = loss_in_bits * math.log(2)
ppl_unified = math.exp(nats_loss_from_b)   # الآن قابلة للمقارنة مع ppl_from_source_b أعلاه

# ────────────────────────────────────────────────────────────────────────
# [016] book/chapters/part-01-foundations/chapter-04.md
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_detector.py — هيكل بداية للتمرين
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DiagnosisResult(Enum):
    HEALTHY = auto()
    UNDERFITTING = auto()
    OVERFITTING = auto()
    DIVERGING = auto()


@dataclass(frozen=True)
class DiagnosisReport:
    result: DiagnosisResult
    final_train_loss: float
    final_val_loss: float
    gap: float
    recommendation: str


def diagnose_training(
    train_losses: list[float],
    val_losses: list[float],
    *,
    gap_ratio_threshold: float = 0.3,   # نسبة الفجوة إلى train_loss لاعتبارها Overfitting
    high_loss_threshold: float = 1.0,
) -> DiagnosisReport:
    """
    TODO: اكتب منطق التشخيص بالخطوات التالية:

    1. gap = val_losses[-1] - train_losses[-1]
    2. gap_ratio = gap / max(train_losses[-1], 1e-8)
    3. تحقق من DIVERGING أولًا: هل train_losses[-1] أكبر من train_losses في منتصف
       التدريب تقريبًا (مثلًا train_losses[len(train_losses)//2])؟
    4. تحقق من UNDERFITTING: train_losses[-1] مرتفع (> high_loss_threshold)
       و gap_ratio صغيرة نسبيًا (النموذج سيء على الطرفين بالتساوي تقريبًا)
    5. تحقق من OVERFITTING: gap_ratio > gap_ratio_threshold
    6. إن لم تنطبق أي حالة أعلاه: HEALTHY

    لكل حالة، أنشئ توصية نصية دقيقة وعملية (انظر الأمثلة في نص الفصل أعلاه
    كمرجع لصياغة التوصيات).
    """
    raise NotImplementedError


def plot_diagnosis(
    train_losses: list[float], val_losses: list[float], report: DiagnosisReport
) -> None:
    """
    TODO: استخدم matplotlib لرسم المنحنيين معًا، مع:
    - عنوان يتضمن نتيجة التشخيص (report.result.name)
    - تظليل خلفية بلون يعكس الحالة (أخضر فاتح=HEALTHY، أحمر فاتح=OVERFITTING،
      برتقالي فاتح=UNDERFITTING، أحمر غامق=DIVERGING)
    - نص التوصية كعنوان فرعي (subtitle) أسفل العنوان الرئيسي
    """
    raise NotImplementedError


if __name__ == "__main__":
    scenarios = {
        "healthy":      ([2.5, 1.8, 1.2, 0.9, 0.7], [2.6, 1.9, 1.3, 1.0, 0.75]),
        "overfitting":  ([2.5, 1.5, 0.8, 0.3, 0.1], [2.6, 1.8, 1.5, 1.7, 2.0]),
        "underfitting": ([2.5, 2.3, 2.2, 2.15, 2.1], [2.6, 2.4, 2.3, 2.25, 2.2]),
        "diverging":    ([2.5, 3.0, 4.5, 6.0, 9.0], [2.6, 3.1, 4.6, 6.2, 9.5]),
    }

    for name, (train_l, val_l) in scenarios.items():
        report = diagnose_training(train_l, val_l)
        status = "✅" if report.result == DiagnosisResult.HEALTHY else "⚠️"
        print(f"{status} [{name}] → {report.result.name}: {report.recommendation}")

    # تحدٍّ إضافي: جرّب تصميم سيناريو "حدّي" يقع بين Healthy وOverfitting،
    # واختبر حساسية عتبة gap_ratio_threshold — هل 0.3 عتبة مناسبة دائمًا؟
    # متى قد تحتاج لتعديلها حسب طبيعة المهمة (تصنيف بسيط vs نموذج لغة ضخم)؟

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 04: الفصل الرابع: التقييم والتعميم — Overfitting وRegularization وPerplexity."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_validation() -> None:
    train = [0.8, 0.7, 0.9, 0.6]
    validation = [0.4, 0.5, 0.3]
    tokens = [0.2, 0.3, 0.25]
    perplexity = math.exp(-sum((math.log(p) for p in tokens)) / len(tokens))
    print('train mean loss:', round(statistics.mean(train), 3))
    print('validation mean loss:', round(statistics.mean(validation), 3))
    print('generalization gap:', round(statistics.mean(validation) - statistics.mean(train), 3))
    print('toy-token perplexity:', round(perplexity, 3))

if __name__ == "__main__":
    print("Chapter 04 lab: Generalization and perplexity")
    run_validation()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
dropout_and_weight_decay.py
تطبيق Inverted Dropout يدويًا من الصفر مقارنة بـ nn.Dropout الرسمي،
وتوضيح تجريبي لأثر Weight Decay على حجم الأوزان عبر الزمن.
Python 3.13+ | PyTorch
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ManualDropout(nn.Module):
    """تطبيق Inverted Dropout يدويًا؛ سلوكه مطابق لـ nn.Dropout الرسمي."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p يجب أن تقع في المجال [0, 1)، وردت القيمة: {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob


def compare_dropout_implementations() -> None:
    torch.manual_seed(0)
    x = torch.ones(3, 10)

    manual = ManualDropout(p=0.4)
    official = nn.Dropout(p=0.4)

    manual.train(); official.train()
    print("=== أثناء training (قيم متوقَّعة: صفر أو 1/(1-p)=1.667) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])

    manual.eval(); official.eval()
    print("\n=== أثناء eval (يجب أن تكون مطابقة تمامًا لـ x) ===")
    print("يدوي :", manual(x)[0])
    print("رسمي :", official(x)[0])


def demonstrate_weight_decay_effect(num_steps: int = 200) -> None:
    """يُظهر تجريبيًا كيف يُقلّص weight_decay حجم الأوزان مع ثبات كل شيء آخر."""
    torch.manual_seed(1)

    model_a = nn.Linear(20, 20)
    model_b = nn.Linear(20, 20)
    model_b.load_state_dict(model_a.state_dict())  # ابدأ من نفس الأوزان تمامًا

    opt_a = torch.optim.AdamW(model_a.parameters(), lr=0.05, weight_decay=0.0)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.05, weight_decay=0.3)

    dummy_x = torch.randn(16, 20)
    initial_norm = model_a.weight.norm().item()

    for _ in range(num_steps):
        for model, opt in [(model_a, opt_a), (model_b, opt_b)]:
            opt.zero_grad(set_to_none=True)
            # خسارة تعسفية بسيطة فقط لإظهار أثر الـ decay على حجم الأوزان
            loss = model(dummy_x).pow(2).mean()
            loss.backward()
            opt.step()

    print(f"\nمعيار الأوزان الابتدائي       : {initial_norm:.4f}")
    print(f"بدون weight_decay (النهائي)   : {model_a.weight.norm().item():.4f}")
    print(f"مع weight_decay=0.3 (النهائي) : {model_b.weight.norm().item():.4f}  ← أصغر بوضوح")


if __name__ == "__main__":
    compare_dropout_implementations()
    demonstrate_weight_decay_effect()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
early_stopping.py
تطبيق Early Stopping احترافي مع حفظ واسترجاع أفضل حالة للنموذج تلقائيًا.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float = field(default=float("inf"), init=False)
    counter: int = field(default=0, init=False)
    best_state_dict: dict | None = field(default=None, init=False)
    should_stop: bool = field(default=False, init=False)
    best_epoch: int = field(default=-1, init=False)

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print(f"✅ تمت استعادة أفضل نسخة من epoch={self.best_epoch} "
                  f"(val_loss={self.best_loss:.4f})")


if __name__ == "__main__":
    # محاكاة سريعة لتوضيح السلوك دون تدريب فعلي
    simulated_val_losses = [2.1, 1.8, 1.5, 1.4, 1.42, 1.45, 1.5, 1.6]
    stopper = EarlyStopping(patience=3, min_delta=1e-3)
    dummy_model = nn.Linear(4, 4)

    for epoch, vloss in enumerate(simulated_val_losses):
        stopper.step(vloss, dummy_model, epoch)
        print(f"epoch={epoch} val_loss={vloss} counter={stopper.counter} "
              f"should_stop={stopper.should_stop}")
        if stopper.should_stop:
            print(f"\n⏹️ توقف عند epoch={epoch}، لكن أفضل نتيجة كانت في epoch={stopper.best_epoch}")
            break

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
"""
perplexity_metrics.py
حساب Perplexity بدقة من Cross-Entropy Loss، مع معالجة صحيحة لرموز الحشو (Padding)،
ومقارنة Perplexity على جمل "متوقَّعة" مقابل جمل "عشوائية".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def perplexity_from_loss(cross_entropy_nats: float) -> float:
    """العلاقة الجوهرية: PPL = e^(loss)، بشرط أن تكون loss بوحدة nats (ln الطبيعي)."""
    return math.exp(cross_entropy_nats)


@torch.no_grad()
def corpus_perplexity(
    model: nn.Module,
    token_ids: torch.Tensor,     # (batch, seq_len)
    pad_token_id: int = 0,
) -> float:
    """
    حساب Perplexity على كامل مجموعة بيانات، مع تجاهل توكنز الحشو تمامًا
    عبر ignore_index — خطوة حرجة: تجاهلها يُحرّف النتيجة بشكل كبير إحصائيًا
    إذا كانت نسبة الحشو عالية (شائع جدًا مع تسلسلات متغيّرة الطول).
    """
    model.eval()
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    logits = model(inputs)  # (batch, seq_len-1, vocab_size)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
    )
    return perplexity_from_loss(loss.item())


if __name__ == "__main__":
    vocab_size, seq_len, batch_size = 500, 15, 4

    class TinyLM(nn.Module):
        def __init__(self, vocab_size: int, confident: bool) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, 16)
            self.proj = nn.Linear(16, vocab_size)
            self.confident = confident  # لمحاكاة نموذج "واثق" اصطناعيًا للتوضيح فقط

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.proj(self.embed(x))
            if self.confident:
                logits = logits * 10.0  # تضخيم الفروقات = ثقة أعلى بمن توقعه النموذج
            return logits

    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))

    random_model = TinyLM(vocab_size, confident=False)
    confident_model = TinyLM(vocab_size, confident=True)
    # نسخ نفس الأوزان الابتدائية لضمان مقارنة عادلة لتأثير "الثقة" فقط
    confident_model.load_state_dict(
        {k: v.clone() for k, v in random_model.state_dict().items()}
    )

    ppl_random = corpus_perplexity(random_model, tokens)
    ppl_confident = corpus_perplexity(confident_model, tokens)

    print(f"Perplexity (نموذج عشوائي التنبؤ): {ppl_random:.2f}")
    print(f"Perplexity (نفس الأوزان لكن logits مُضخَّمة): {ppl_confident:.2f}")
    print(f"\n(ملاحظة تعليمية: تضخيم logits عشوائيًا دون تدريب حقيقي "
          f"لا يعني بالضرورة تحسنًا — فقط يُغيّر حدة التوزيع، "
          f"وقد يُسيء للـ PPL إن كانت الثقة في غير محلّها!)")

    # تذكير مباشر بالعلاقة الأساسية
    example_loss = 2.3
    print(f"\nمثال مباشر: Cross-Entropy Loss = {example_loss} nats")
    print(f"=> Perplexity = exp({example_loss}) = {perplexity_from_loss(example_loss):.3f}")

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_end_to_end.py
تجربة كاملة: توليد Overfitting متعمّد، تشخيصه عبر Perplexity،
ثم علاجه بـ Dropout + Weight Decay + Early Stopping.
نموذج لغة بسيط على مستوى الحرف (character-level) لتسريع التجربة وتوضيحها.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


@dataclass(frozen=True)
class CharLMConfig:
    vocab_size: int
    embed_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.0


class CharLM(nn.Module):
    """نموذج لغة بسيط: Embedding -> GRU -> Linear، لغرض توضيحي تعليمي."""

    def __init__(self, cfg: CharLMConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop_in = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.drop_out = nn.Dropout(cfg.dropout)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_in(self.embed(x))
        out, _ = self.gru(h)
        out = self.drop_out(out)
        return self.proj(out)


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def batchify(data: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
    n_chunks = (len(data) - 1) // (seq_len * batch_size)
    usable = n_chunks * seq_len * batch_size
    return data[:usable].view(batch_size, -1)


@torch.no_grad()
def eval_perplexity(model: nn.Module, data: torch.Tensor, seq_len: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, data.size(1) - seq_len - 1, seq_len):
        xb = data[:, i : i + seq_len]
        yb = data[:, i + 1 : i + seq_len + 1]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += loss.item() * yb.numel()
        total_tokens += yb.numel()
    return math.exp(total_loss / total_tokens)


def run_training(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    num_epochs: int,
    weight_decay: float = 0.0,
) -> tuple[list[float], list[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    train_hist, val_hist = [], []

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, train_data.size(1) - seq_len - 1, seq_len):
            xb = train_data[:, i : i + seq_len]
            yb = train_data[:, i + 1 : i + seq_len + 1]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_ppl = eval_perplexity(model, train_data, seq_len)
        val_ppl = eval_perplexity(model, val_data, seq_len)
        train_hist.append(train_ppl)
        val_hist.append(val_ppl)
        gap_marker = "⚠️" if (val_ppl - train_ppl) > train_ppl else ""
        print(f"Epoch {epoch+1:02d} | Train PPL={train_ppl:7.2f} | Val PPL={val_ppl:7.2f} {gap_marker}")

    return train_hist, val_hist


if __name__ == "__main__":
    text = (
        "في مدينة قديمة جدا كان هناك حكيم يجلس كل يوم تحت شجرة كبيرة "
        "يستمع الى اسئلة الناس ويجيب عليها بحكمة عميقة وبساطة شديدة "
    ) * 25
    split = int(0.8 * len(text))
    train_text, val_text = text[:split], text[split:]

    stoi, itos = build_vocab(text)
    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    seq_len, batch_size = 32, 8
    train_batches = batchify(train_ids, seq_len, batch_size)
    val_batches = batchify(val_ids, seq_len, batch_size)

    print("=== تجربة 1: بدون Regularization (Overfitting متوقَّع بقوة) ===")
    model_plain = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.0))
    train_hist_plain, val_hist_plain = run_training(
        model_plain, train_batches, val_batches, seq_len, num_epochs=20
    )

    print("\n=== تجربة 2: مع Dropout=0.3 + Weight Decay=0.1 ===")
    model_reg = CharLM(CharLMConfig(vocab_size=len(stoi), dropout=0.3))
    train_hist_reg, val_hist_reg = run_training(
        model_reg, train_batches, val_batches, seq_len,
        num_epochs=20, weight_decay=0.1,
    )

    print("\n=== الخلاصة المقارنة ===")
    gap_plain = val_hist_plain[-1] - train_hist_plain[-1]
    gap_reg = val_hist_reg[-1] - train_hist_reg[-1]
    print(f"بدون Regularization: فجوة PPL النهائية = {gap_plain:.2f}")
    print(f"مع Regularization  : فجوة PPL النهائية = {gap_reg:.2f}")
    print(f"\n(الهدف ليس أقل Train PPL، بل أصغر فجوة مع أفضل Val PPL ممكن)")

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التقييم يحدث بينما Dropout/BatchNorm لا يزالان في وضع "تدريب"
def evaluate(model, loader):
    for xb, yb in loader:
        preds = model(xb)   # Dropout يُطفئ عصبونات هنا أيضًا بالخطأ، نتيجة غير موثوقة!

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح
def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
    model.train()  # أعده لوضع train قبل متابعة حلقة التدريب

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: حساب mean/std على كل البيانات، بما فيها val/test، قبل التقسيم
mean, std = all_data.mean(), all_data.std()
normalized = (all_data - mean) / std
train, val, test = split(normalized)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: قسّم أولًا، ثم احسب الإحصائيات من Train فقط
train, val, test = split(all_data)
mean, std = train.mean(), train.std()
train = (train - mean) / std
val = (val - mean) / std      # يستخدم mean/std من train حصرًا
test = (test - mean) / std

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ منهجي خطير جدًا
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    test_acc = evaluate(model, test_loader)    # تسريب! قرار مبني على test
    if test_acc > best_acc:
        best_lr = lr

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: Validation لاتخاذ القرار، Test فقط للتقرير النهائي مرة واحدة
for lr in [1e-3, 3e-4, 1e-4]:
    model = train(lr=lr)
    val_acc = evaluate(model, val_loader)
    if val_acc > best_acc:
        best_lr = lr

final_model = train(lr=best_lr)
final_test_acc = evaluate(final_model, test_loader)  # مرة واحدة، في النهاية فقط

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ❌ غير كافٍ ومُضلِّل لنماذج اللغة
accuracy = (predictions == targets).float().mean()

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
# ✅ أفضل: Perplexity كمقياس أساسي، مع مقاييس تكميلية حسب المهمة لاحقًا
ppl = corpus_perplexity(model, val_tokens, pad_token_id=PAD_ID)

# ────────────────────────────────────────────────────────────────────────
# [016] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-15
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ خفي وشائع: افتراض أن كل PPL محسوبة بنفس الطريقة
# بعض المكتبات تستخدم log2 بدل ln دون توضيح كافٍ في التوثيق!
ppl_from_source_a = 2 ** loss_in_bits   # bits
ppl_from_source_b = math.exp(loss_in_nats)  # nats
# مقارنة هذين الرقمين مباشرة = خطأ منهجي!

# ────────────────────────────────────────────────────────────────────────
# [017] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-16
# ────────────────────────────────────────────────────────────────────────
# ✅ صحيح: وحّد القاعدة اللوغاريتمية أولًا قبل أي مقارنة
# nats -> bits: bits_loss = nats_loss / ln(2)
# bits -> nats: nats_loss = bits_loss * ln(2)
nats_loss_from_b = loss_in_bits * math.log(2)
ppl_unified = math.exp(nats_loss_from_b)   # الآن قابلة للمقارنة مع ppl_from_source_b أعلاه

# ────────────────────────────────────────────────────────────────────────
# [018] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-04-chapter-04.ipynb#cell-17
# ────────────────────────────────────────────────────────────────────────
"""
overfitting_detector.py — هيكل بداية للتمرين
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DiagnosisResult(Enum):
    HEALTHY = auto()
    UNDERFITTING = auto()
    OVERFITTING = auto()
    DIVERGING = auto()


@dataclass(frozen=True)
class DiagnosisReport:
    result: DiagnosisResult
    final_train_loss: float
    final_val_loss: float
    gap: float
    recommendation: str


def diagnose_training(
    train_losses: list[float],
    val_losses: list[float],
    *,
    gap_ratio_threshold: float = 0.3,   # نسبة الفجوة إلى train_loss لاعتبارها Overfitting
    high_loss_threshold: float = 1.0,
) -> DiagnosisReport:
    """
    TODO: اكتب منطق التشخيص بالخطوات التالية:

    1. gap = val_losses[-1] - train_losses[-1]
    2. gap_ratio = gap / max(train_losses[-1], 1e-8)
    3. تحقق من DIVERGING أولًا: هل train_losses[-1] أكبر من train_losses في منتصف
       التدريب تقريبًا (مثلًا train_losses[len(train_losses)//2])؟
    4. تحقق من UNDERFITTING: train_losses[-1] مرتفع (> high_loss_threshold)
       و gap_ratio صغيرة نسبيًا (النموذج سيء على الطرفين بالتساوي تقريبًا)
    5. تحقق من OVERFITTING: gap_ratio > gap_ratio_threshold
    6. إن لم تنطبق أي حالة أعلاه: HEALTHY

    لكل حالة، أنشئ توصية نصية دقيقة وعملية (انظر الأمثلة في نص الفصل أعلاه
    كمرجع لصياغة التوصيات).
    """
    raise NotImplementedError


def plot_diagnosis(
    train_losses: list[float], val_losses: list[float], report: DiagnosisReport
) -> None:
    """
    TODO: استخدم matplotlib لرسم المنحنيين معًا، مع:
    - عنوان يتضمن نتيجة التشخيص (report.result.name)
    - تظليل خلفية بلون يعكس الحالة (أخضر فاتح=HEALTHY، أحمر فاتح=OVERFITTING،
      برتقالي فاتح=UNDERFITTING، أحمر غامق=DIVERGING)
    - نص التوصية كعنوان فرعي (subtitle) أسفل العنوان الرئيسي
    """
    raise NotImplementedError


if __name__ == "__main__":
    scenarios = {
        "healthy":      ([2.5, 1.8, 1.2, 0.9, 0.7], [2.6, 1.9, 1.3, 1.0, 0.75]),
        "overfitting":  ([2.5, 1.5, 0.8, 0.3, 0.1], [2.6, 1.8, 1.5, 1.7, 2.0]),
        "underfitting": ([2.5, 2.3, 2.2, 2.15, 2.1], [2.6, 2.4, 2.3, 2.25, 2.2]),
        "diverging":    ([2.5, 3.0, 4.5, 6.0, 9.0], [2.6, 3.1, 4.6, 6.2, 9.5]),
    }

    for name, (train_l, val_l) in scenarios.items():
        report = diagnose_training(train_l, val_l)
        status = "✅" if report.result == DiagnosisResult.HEALTHY else "⚠️"
        print(f"{status} [{name}] → {report.result.name}: {report.recommendation}")

    # تحدٍّ إضافي: جرّب تصميم سيناريو "حدّي" يقع بين Healthy وOverfitting،
    # واختبر حساسية عتبة gap_ratio_threshold — هل 0.3 عتبة مناسبة دائمًا؟
    # متى قد تحتاج لتعديلها حسب طبيعة المهمة (تصنيف بسيط vs نموذج لغة ضخم)؟
