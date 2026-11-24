#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر
Large Language Model Engineering — Mini-LLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-26-chapter-26 (lesson 26)
Code files: 15 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-26-chapter-26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر
Large Language Model Engineering — Mini-LLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-26-chapter-26 (lesson 26)
Code files: 14 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-26-chapter-26.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر
Large Language Model Engineering — Mini-LLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-26-chapter-26 (lesson 26)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 26: الفصل 26: مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bigram() -> None:
    text = 'to be or not to be that is the question to be'
    pairs = Counter(zip(text.split(), text.split()[1:]))
    next_words = {word: sorted(((count, nxt) for (prev, nxt), count in pairs.items() if prev == word), reverse=True) for word in set(text.split())}
    print("learned next-word counts for 'to':", next_words['to'])
    print('most likely continuation:', next_words['to'][0][1])
    print('model parameters:', len(pairs), 'observed bigrams')

if __name__ == "__main__":
    print("Chapter 26 lab: Train a tiny next-character language model")
    run_bigram()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniLLMConfig:
    """تهيئة كاملة لتجربة Mini-LLM — كل الأبعاد مُصمَّمة لتناسب
    التدريب على جهاز واحد في وقت معقول، مع الحفاظ على نفس المبادئ
    المعمارية المستخدمة في النماذج الإنتاجية الضخمة."""

    # إعدادات المفردات والتوكنايزر
    vocab_size: int = 8000          # صغير نسبيًا، مناسب لمجموعة بيانات محدودة
    max_seq_len: int = 256          # طول سياق متواضع يكفي للتجربة

    # إعدادات المعمارية (نسب متوازنة على نمط LLaMA لكن مُصغَّرة)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4              # GQA: نصف عدد رؤوس الاستعلام (مرجع: الفصل 15)
    d_ff: int = 683                  # ≈ (2/3) × 4 × d_model، مُقرَّب (مرجع: الفصل 18)
    rope_theta: float = 10000.0

    # إعدادات التدريب
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    total_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    def estimate_param_count(self) -> int:
        """تقدير تقريبي لعدد المعاملات الإجمالي، تطبيقًا مباشرًا
        لصيغ الفصل الرابع عشر (دون احتساب embeddings المربوطة)."""
        embed_params = self.vocab_size * self.d_model
        per_layer_attn = 2 * self.d_model * self.d_model + 2 * (
            self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        )
        per_layer_ffn = 3 * self.d_model * self.d_ff  # SwiGLU: 3 مصفوفات
        per_layer_norm = 2 * self.d_model             # RMSNorm × 2
        per_layer_total = per_layer_attn + per_layer_ffn + per_layer_norm
        total = embed_params + self.n_layers * per_layer_total
        return total


if __name__ == "__main__":
    config = MiniLLMConfig()
    n_params = config.estimate_param_count()
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params / 1e6:.1f}M)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from pathlib import Path


def clean_text_simple(raw_text: str) -> str:
    """تنظيف مُبسَّط (نسخة مُصغَّرة من مبادئ الفصل 20): إزالة أسطر فارغة
    متكررة، توحيد المسافات، إزالة أحرف تحكم غريبة."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    return text.strip()


def split_train_val(text: str, val_ratio: float = 0.02) -> tuple[str, str]:
    """تقسيم بسيط حسب الموضع — في مشروع حقيقي يُفضَّل التقسيم حسب
    الوثيقة الكاملة لتجنب تسرب معلومات بين المجموعتين (الفصل 4)."""
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


def prepare_corpus(raw_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = raw_path.read_text(encoding="utf-8")
    cleaned = clean_text_simple(raw_text)
    train_text, val_text = split_train_val(cleaned)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    print(f"✅ بيانات التدريب: {len(train_text):,} حرف")
    print(f"✅ بيانات التحقق: {len(val_text):,} حرف")
    return train_path, val_path

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def train_mini_tokenizer(
    train_file: Path, vocab_size: int, output_path: Path
) -> Tokenizer:
    """تدريب BPE بسيط باستخدام مكتبة tokenizers من Hugging Face،
    تطبيقًا مباشرًا لمنهجية الفصل الحادي والعشرين لكن بحجم مفردات أصغر."""

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<|im_start|>", "<|im_end|>",  # رموز Chat Template (الفصل 25)
        ],
        min_frequency=2,
    )

    tokenizer.train(files=[str(train_file)], trainer=trainer)
    tokenizer.save(str(output_path))
    print(f"✅ تم تدريب Tokenizer بمفردات حجمها {tokenizer.get_vocab_size():,}")
    return tokenizer

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """تطبيق مباشر لـ RMSNorm من الفصل الثامن عشر."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """حساب مسبق لجيب وجيب التمام اللازمين لتدوير RoPE (الفصل الخامس عشر)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق التدوير الموضعي على استعلامات/مفاتيح الانتباه."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """تطبيق مُصغَّر لـ GQA (الفصل الخامس عشر): عدد رؤوس المفاتيح/القيم
    أقل من رؤوس الاستعلام، مع تكرار (repeat_interleave) لمطابقة الأبعاد."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # تكرار K/V لمطابقة عدد رؤوس الاستعلام (جوهر GQA)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # استخدام الانتباه المُحسَّن المدمج في PyTorch (يُطبّق منطق
        # مشابه لـ FlashAttention داخليًا — الفصل الثالث عشر)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn_out)


class SwiGLUFeedForward(nn.Module):
    """تطبيق مباشر لـ SwiGLU من الفصل الثامن عشر."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniTransformerBlock(nn.Module):
    """كتلة Transformer كاملة بنمط Pre-Norm الحديث (الفصل 18)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(
            config.d_model, config.n_heads, config.n_kv_heads, config.max_seq_len
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """النموذج الكامل: Embedding + مكدّس كتل Transformer + رأس اللغة،
    مع ربط الأوزان (Weight Tying) بين Embedding وLM Head (الفصل 18/21)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: مشاركة نفس المصفوفة بين embedding وlm_head
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """توليد نص عبر أخذ عينات (Sampling) من توزيع Top-K بدرجة حرارة
        معينة — تطبيق مباشر لمفاهيم استراتيجيات التوليد (الفصل 38 لاحقًا)."""
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature

            top_values, top_indices = torch.topk(next_logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(-1, sampled_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


class TextDataset:
    """مجموعة بيانات بسيطة: تحميل نص مُرمَّز كاملاً في الذاكرة
    (مناسب لحجم مشروعنا المُصغَّر)، وأخذ نوافذ عشوائية منه."""

    def __init__(self, text_path: Path, tokenizer: Tokenizer, seq_len: int) -> None:
        text = text_path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        rng = np.random.default_rng()
        starts = rng.integers(0, max_start, size=batch_size)

        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: MiniLLMConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_mini_llm(
    config: MiniLLMConfig,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    checkpoint_dir: Path,
    device: torch.device,
) -> MiniLLM:
    torch.manual_seed(42)

    model = MiniLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(optimizer, config)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(config.total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_dataset.get_batch(config.batch_size, device)
            _, loss = model(x, labels=y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:5d}/{config.total_steps} | "
                f"Loss: {accumulated_loss:.4f} | "
                f"GradNorm: {grad_norm:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        if step % 500 == 0 and step > 0:
            val_loss = evaluate_perplexity(model, val_dataset, device, n_batches=20)
            print(f"  📊 Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": config, "step": step},
                    checkpoint_dir / "best_model.pt",
                )
                print(f"  ✅ نقطة تفتيش جديدة أفضل محفوظة عند الخطوة {step}")

    return model


@torch.no_grad()
def evaluate_perplexity(
    model: MiniLLM, dataset: TextDataset, device: torch.device, n_batches: int = 20
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch(model.config.batch_size, device)
        _, loss = model(x, labels=y)
        total_loss += loss.item()
    return total_loss / n_batches

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from tokenizers import Tokenizer


def build_toy_instruction_data() -> list[tuple[str, str]]:
    """مجموعة تعليمات بسيطة جدًا توضيحيًا — في مشروع حقيقي أكبر،
    استخدم منهجية Self-Instruct الكاملة من الفصل 25."""
    return [
        ("ما هو ناتج 2 زائد 2؟", "ناتج 2 زائد 2 هو 4."),
        ("اذكر لون السماء.", "لون السماء أزرق في النهار الصافي."),
        ("ما عاصمة مصر؟", "عاصمة مصر هي القاهرة."),
        # ... يُفترض توسيع هذه القائمة لعشرات أو مئات الأمثلة فعليًا
    ]


def format_sft_example(instruction: str, response: str) -> str:
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )


def sft_finetune_mini_llm(
    model: MiniLLM,
    tokenizer: Tokenizer,
    sft_data: list[tuple[str, str]],
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
) -> MiniLLM:
    """SFT مُصغَّر: معدل تعلّم أقل بعشرة أضعاف من Pretraining،
    حقبات قليلة جدًا — تطبيق مباشر لمبادئ الفصل الخامس والعشرين."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    assistant_marker = tokenizer.encode("assistant\n").ids

    for epoch in range(epochs):
        total_loss = 0.0
        for instruction, response in sft_data:
            full_text = format_sft_example(instruction, response)
            ids = tokenizer.encode(full_text).ids
            input_ids = torch.tensor([ids], device=device)

            # بناء قناع خسارة مُبسَّط: نُقنّع كل شيء قبل أول ظهور لعلامة
            # "assistant\n" — تبسيط توضيحي لمنطق الفصل 25 الكامل
            labels = input_ids.clone()
            marker_pos = _find_subsequence(ids, assistant_marker)
            if marker_pos is not None:
                mask_end = marker_pos + len(assistant_marker)
                labels[0, :mask_end] = -100

            _, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        print(f"SFT Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(sft_data):.4f}")

    return model


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    n, m = len(sequence), len(subsequence)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            return i
    return None

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ الجهاز المستخدم: {device}")

    workspace = Path("./mini_llm_workspace")
    workspace.mkdir(exist_ok=True)

    # المرحلة 1: تحضير البيانات
    raw_corpus_path = Path("raw_corpus.txt")  # ضع هنا ملف نصي (عدة ميجابايتات)
    train_path, val_path = prepare_corpus(raw_corpus_path, workspace / "data")

    # المرحلة 2: تدريب Tokenizer
    config = MiniLLMConfig()
    tokenizer = train_mini_tokenizer(
        train_path, config.vocab_size, workspace / "tokenizer.json"
    )

    # المرحلة 3+4: بناء النموذج والتدريب
    train_dataset = TextDataset(train_path, tokenizer, config.max_seq_len)
    val_dataset = TextDataset(val_path, tokenizer, config.max_seq_len)

    model = pretrain_mini_llm(
        config, train_dataset, val_dataset, workspace / "checkpoints", device
    )

    # المرحلة 5: تقييم نوعي قبل SFT
    print("\n=== توليد نموذجي قبل SFT ===")
    prompt_ids = torch.tensor(
        [tokenizer.encode("في يوم من الأيام").ids], device=device
    )
    generated = model.generate(prompt_ids, max_new_tokens=50)
    print(tokenizer.decode(generated[0].tolist()))

    # المرحلة 6: SFT
    sft_data = build_toy_instruction_data()
    model = sft_finetune_mini_llm(model, tokenizer, sft_data, device)

    # المرحلة 7: تقييم نوعي بعد SFT
    print("\n=== الاستجابة بعد SFT ===")
    test_prompt = "<|im_start|>user\nما عاصمة مصر؟<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt).ids], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=30, temperature=0.3)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مفردات كبيرة جدًا (50000) على نص صغير (بضع ميجابايتات فقط)
# يؤدي إلى رموز نادرة جدًا لا تتعلّم تمثيلات جيدة لها إطلاقًا
config = MiniLLMConfig(vocab_size=50_000)

# ✅ صحيح: حجم مفردات متناسب مع حجم البيانات (قاعدة تقريبية:
# حوالي رمز واحد لكل 1000-2000 حرف في بيانات التدريب)
config = MiniLLMConfig(vocab_size=8_000)

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التوليد في وضع train() يُفعّل Dropout عشوائيًا،
# مما يُنتج تذبذبًا غير مرغوب فيه في جودة التوليد
generated = model.generate(prompt_ids, max_new_tokens=50)

# ✅ صحيح: الدالة generate المعرّفة أعلاه تستدعي self.eval() داخليًا،
# لكن تأكد دائمًا من هذا التفصيل في أي كود توليد تكتبه بنفسك
model.eval()
with torch.no_grad():
    generated = model.generate(prompt_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاعتماد على تجربة واحدة فقط بـ seed=42 للحكم على جودة
# المعمارية أو الإعدادات — قد تكون النتيجة مجرد "حظ" إحصائي
torch.manual_seed(42)
model = pretrain_mini_llm(...)

# ✅ أفضل: شغّل التجربة بـ 2-3 بذور مختلفة على الأقل عند مقارنة
# إعدادات مختلفة (معدل تعلّم، حجم دفعة) للتأكد من أن الفروقات حقيقية
for seed in [42, 123, 777]:
    torch.manual_seed(seed)
    model = pretrain_mini_llm(...)

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الإعجاب بانخفاض Perplexity على بيانات التدريب فقط
train_ppl = evaluate_perplexity(model, train_dataset, device)
print(f"النموذج ممتاز! PPL = {train_ppl}")  # قد يكون هذا مجرد حفظ (Overfitting)!

# ✅ صحيح: الحكم دائمًا بناءً على مجموعة Validation المنفصلة تمامًا
val_ppl = evaluate_perplexity(model, val_dataset, device)
print(f"PPL الحقيقي (Validation): {val_ppl}")

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
"""
تحدي موسّع: ثلاث تجارب تحليلية على Mini-LLM

التجربة الأولى - أثر حجم النموذج:
درّب 3 نسخ من Mini-LLM بأحجام مختلفة (d_model=128، 256، 512 مع تعديل
باقي الأبعاد تناسبيًا) على نفس البيانات وعدد الخطوات، وارسم منحنيات
Validation Loss الثلاثة على نفس الرسم البياني. هل يتبع السلوك المُلاحظ
نمط Scaling Laws الذي درسناه في الفصل 22 (ولو بشكل تقريبي جدًا على
هذا النطاق الصغير)؟

التجربة الثانية - أثر RoPE مقابل Learned Positional Embeddings:
ابنِ نسخة بديلة من MiniLLM تستخدم nn.Embedding عادي للمواضع بدلاً
من RoPE، درّب كلا النسختين، وقارن: هل تتفوق إحداهما في التعميم على
أطوال سياق أطول من تلك المستخدمة في التدريب؟ (اختبر على تسلسلات
أطول من max_seq_len الأصلي وراقب استقرار Perplexity)

التجربة الثالثة - أثر جودة بيانات SFT:
درّب نسختين من SFT: واحدة بـ10 أمثلة فقط، وأخرى بـ100 مثال (يمكنك
توليدها يدويًا أو عبر تنويع قوالب بسيطة). قارن سلوك التوقف الصحيح
(<|im_end|>) ودقة الإجابات بين النسختين على مجموعة اختبار من 20 سؤالاً
لم تظهر في أي من مجموعتي SFT.
"""

from __future__ import annotations


def experiment_model_scaling() -> None:
    """TODO: نفّذ التجربة الأولى وارسم النتائج باستخدام matplotlib."""
    raise NotImplementedError


def experiment_positional_encoding() -> None:
    """TODO: نفّذ التجربة الثانية، مع بناء نسخة MiniLLM بـ Learned
    Positional Embeddings كبديل لـ RoPE للمقارنة."""
    raise NotImplementedError


def experiment_sft_data_quality() -> None:
    """TODO: نفّذ التجربة الثالثة مع تقرير كمي (نسبة الإجابات الصحيحة،
    نسبة التوقف الصحيح) وليس فقط ملاحظات نوعية."""
    raise NotImplementedError


if __name__ == "__main__":
    experiment_model_scaling()
    experiment_positional_encoding()
    experiment_sft_data_quality()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 26: الفصل 26: مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bigram() -> None:
    text = 'to be or not to be that is the question to be'
    pairs = Counter(zip(text.split(), text.split()[1:]))
    next_words = {word: sorted(((count, nxt) for (prev, nxt), count in pairs.items() if prev == word), reverse=True) for word in set(text.split())}
    print("learned next-word counts for 'to':", next_words['to'])
    print('most likely continuation:', next_words['to'][0][1])
    print('model parameters:', len(pairs), 'observed bigrams')

if __name__ == "__main__":
    print("Chapter 26 lab: Train a tiny next-character language model")
    run_bigram()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniLLMConfig:
    """تهيئة كاملة لتجربة Mini-LLM — كل الأبعاد مُصمَّمة لتناسب
    التدريب على جهاز واحد في وقت معقول، مع الحفاظ على نفس المبادئ
    المعمارية المستخدمة في النماذج الإنتاجية الضخمة."""

    # إعدادات المفردات والتوكنايزر
    vocab_size: int = 8000          # صغير نسبيًا، مناسب لمجموعة بيانات محدودة
    max_seq_len: int = 256          # طول سياق متواضع يكفي للتجربة

    # إعدادات المعمارية (نسب متوازنة على نمط LLaMA لكن مُصغَّرة)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4              # GQA: نصف عدد رؤوس الاستعلام (مرجع: الفصل 15)
    d_ff: int = 683                  # ≈ (2/3) × 4 × d_model، مُقرَّب (مرجع: الفصل 18)
    rope_theta: float = 10000.0

    # إعدادات التدريب
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    total_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    def estimate_param_count(self) -> int:
        """تقدير تقريبي لعدد المعاملات الإجمالي، تطبيقًا مباشرًا
        لصيغ الفصل الرابع عشر (دون احتساب embeddings المربوطة)."""
        embed_params = self.vocab_size * self.d_model
        per_layer_attn = 2 * self.d_model * self.d_model + 2 * (
            self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        )
        per_layer_ffn = 3 * self.d_model * self.d_ff  # SwiGLU: 3 مصفوفات
        per_layer_norm = 2 * self.d_model             # RMSNorm × 2
        per_layer_total = per_layer_attn + per_layer_ffn + per_layer_norm
        total = embed_params + self.n_layers * per_layer_total
        return total


if __name__ == "__main__":
    config = MiniLLMConfig()
    n_params = config.estimate_param_count()
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params / 1e6:.1f}M)")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from pathlib import Path


def clean_text_simple(raw_text: str) -> str:
    """تنظيف مُبسَّط (نسخة مُصغَّرة من مبادئ الفصل 20): إزالة أسطر فارغة
    متكررة، توحيد المسافات، إزالة أحرف تحكم غريبة."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    return text.strip()


def split_train_val(text: str, val_ratio: float = 0.02) -> tuple[str, str]:
    """تقسيم بسيط حسب الموضع — في مشروع حقيقي يُفضَّل التقسيم حسب
    الوثيقة الكاملة لتجنب تسرب معلومات بين المجموعتين (الفصل 4)."""
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


def prepare_corpus(raw_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = raw_path.read_text(encoding="utf-8")
    cleaned = clean_text_simple(raw_text)
    train_text, val_text = split_train_val(cleaned)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    print(f"✅ بيانات التدريب: {len(train_text):,} حرف")
    print(f"✅ بيانات التحقق: {len(val_text):,} حرف")
    return train_path, val_path

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def train_mini_tokenizer(
    train_file: Path, vocab_size: int, output_path: Path
) -> Tokenizer:
    """تدريب BPE بسيط باستخدام مكتبة tokenizers من Hugging Face،
    تطبيقًا مباشرًا لمنهجية الفصل الحادي والعشرين لكن بحجم مفردات أصغر."""

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<|im_start|>", "<|im_end|>",  # رموز Chat Template (الفصل 25)
        ],
        min_frequency=2,
    )

    tokenizer.train(files=[str(train_file)], trainer=trainer)
    tokenizer.save(str(output_path))
    print(f"✅ تم تدريب Tokenizer بمفردات حجمها {tokenizer.get_vocab_size():,}")
    return tokenizer

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """تطبيق مباشر لـ RMSNorm من الفصل الثامن عشر."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """حساب مسبق لجيب وجيب التمام اللازمين لتدوير RoPE (الفصل الخامس عشر)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق التدوير الموضعي على استعلامات/مفاتيح الانتباه."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """تطبيق مُصغَّر لـ GQA (الفصل الخامس عشر): عدد رؤوس المفاتيح/القيم
    أقل من رؤوس الاستعلام، مع تكرار (repeat_interleave) لمطابقة الأبعاد."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # تكرار K/V لمطابقة عدد رؤوس الاستعلام (جوهر GQA)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # استخدام الانتباه المُحسَّن المدمج في PyTorch (يُطبّق منطق
        # مشابه لـ FlashAttention داخليًا — الفصل الثالث عشر)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn_out)


class SwiGLUFeedForward(nn.Module):
    """تطبيق مباشر لـ SwiGLU من الفصل الثامن عشر."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniTransformerBlock(nn.Module):
    """كتلة Transformer كاملة بنمط Pre-Norm الحديث (الفصل 18)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(
            config.d_model, config.n_heads, config.n_kv_heads, config.max_seq_len
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """النموذج الكامل: Embedding + مكدّس كتل Transformer + رأس اللغة،
    مع ربط الأوزان (Weight Tying) بين Embedding وLM Head (الفصل 18/21)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: مشاركة نفس المصفوفة بين embedding وlm_head
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """توليد نص عبر أخذ عينات (Sampling) من توزيع Top-K بدرجة حرارة
        معينة — تطبيق مباشر لمفاهيم استراتيجيات التوليد (الفصل 38 لاحقًا)."""
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature

            top_values, top_indices = torch.topk(next_logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(-1, sampled_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


class TextDataset:
    """مجموعة بيانات بسيطة: تحميل نص مُرمَّز كاملاً في الذاكرة
    (مناسب لحجم مشروعنا المُصغَّر)، وأخذ نوافذ عشوائية منه."""

    def __init__(self, text_path: Path, tokenizer: Tokenizer, seq_len: int) -> None:
        text = text_path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        rng = np.random.default_rng()
        starts = rng.integers(0, max_start, size=batch_size)

        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: MiniLLMConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_mini_llm(
    config: MiniLLMConfig,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    checkpoint_dir: Path,
    device: torch.device,
) -> MiniLLM:
    torch.manual_seed(42)

    model = MiniLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(optimizer, config)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(config.total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_dataset.get_batch(config.batch_size, device)
            _, loss = model(x, labels=y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:5d}/{config.total_steps} | "
                f"Loss: {accumulated_loss:.4f} | "
                f"GradNorm: {grad_norm:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        if step % 500 == 0 and step > 0:
            val_loss = evaluate_perplexity(model, val_dataset, device, n_batches=20)
            print(f"  📊 Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": config, "step": step},
                    checkpoint_dir / "best_model.pt",
                )
                print(f"  ✅ نقطة تفتيش جديدة أفضل محفوظة عند الخطوة {step}")

    return model


@torch.no_grad()
def evaluate_perplexity(
    model: MiniLLM, dataset: TextDataset, device: torch.device, n_batches: int = 20
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch(model.config.batch_size, device)
        _, loss = model(x, labels=y)
        total_loss += loss.item()
    return total_loss / n_batches

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from tokenizers import Tokenizer


def build_toy_instruction_data() -> list[tuple[str, str]]:
    """مجموعة تعليمات بسيطة جدًا توضيحيًا — في مشروع حقيقي أكبر،
    استخدم منهجية Self-Instruct الكاملة من الفصل 25."""
    return [
        ("ما هو ناتج 2 زائد 2؟", "ناتج 2 زائد 2 هو 4."),
        ("اذكر لون السماء.", "لون السماء أزرق في النهار الصافي."),
        ("ما عاصمة مصر؟", "عاصمة مصر هي القاهرة."),
        # ... يُفترض توسيع هذه القائمة لعشرات أو مئات الأمثلة فعليًا
    ]


def format_sft_example(instruction: str, response: str) -> str:
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )


def sft_finetune_mini_llm(
    model: MiniLLM,
    tokenizer: Tokenizer,
    sft_data: list[tuple[str, str]],
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
) -> MiniLLM:
    """SFT مُصغَّر: معدل تعلّم أقل بعشرة أضعاف من Pretraining،
    حقبات قليلة جدًا — تطبيق مباشر لمبادئ الفصل الخامس والعشرين."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    assistant_marker = tokenizer.encode("assistant\n").ids

    for epoch in range(epochs):
        total_loss = 0.0
        for instruction, response in sft_data:
            full_text = format_sft_example(instruction, response)
            ids = tokenizer.encode(full_text).ids
            input_ids = torch.tensor([ids], device=device)

            # بناء قناع خسارة مُبسَّط: نُقنّع كل شيء قبل أول ظهور لعلامة
            # "assistant\n" — تبسيط توضيحي لمنطق الفصل 25 الكامل
            labels = input_ids.clone()
            marker_pos = _find_subsequence(ids, assistant_marker)
            if marker_pos is not None:
                mask_end = marker_pos + len(assistant_marker)
                labels[0, :mask_end] = -100

            _, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        print(f"SFT Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(sft_data):.4f}")

    return model


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    n, m = len(sequence), len(subsequence)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            return i
    return None

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ الجهاز المستخدم: {device}")

    workspace = Path("./mini_llm_workspace")
    workspace.mkdir(exist_ok=True)

    # المرحلة 1: تحضير البيانات
    raw_corpus_path = Path("raw_corpus.txt")  # ضع هنا ملف نصي (عدة ميجابايتات)
    train_path, val_path = prepare_corpus(raw_corpus_path, workspace / "data")

    # المرحلة 2: تدريب Tokenizer
    config = MiniLLMConfig()
    tokenizer = train_mini_tokenizer(
        train_path, config.vocab_size, workspace / "tokenizer.json"
    )

    # المرحلة 3+4: بناء النموذج والتدريب
    train_dataset = TextDataset(train_path, tokenizer, config.max_seq_len)
    val_dataset = TextDataset(val_path, tokenizer, config.max_seq_len)

    model = pretrain_mini_llm(
        config, train_dataset, val_dataset, workspace / "checkpoints", device
    )

    # المرحلة 5: تقييم نوعي قبل SFT
    print("\n=== توليد نموذجي قبل SFT ===")
    prompt_ids = torch.tensor(
        [tokenizer.encode("في يوم من الأيام").ids], device=device
    )
    generated = model.generate(prompt_ids, max_new_tokens=50)
    print(tokenizer.decode(generated[0].tolist()))

    # المرحلة 6: SFT
    sft_data = build_toy_instruction_data()
    model = sft_finetune_mini_llm(model, tokenizer, sft_data, device)

    # المرحلة 7: تقييم نوعي بعد SFT
    print("\n=== الاستجابة بعد SFT ===")
    test_prompt = "<|im_start|>user\nما عاصمة مصر؟<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt).ids], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=30, temperature=0.3)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مفردات كبيرة جدًا (50000) على نص صغير (بضع ميجابايتات فقط)
# يؤدي إلى رموز نادرة جدًا لا تتعلّم تمثيلات جيدة لها إطلاقًا
config = MiniLLMConfig(vocab_size=50_000)

# ✅ صحيح: حجم مفردات متناسب مع حجم البيانات (قاعدة تقريبية:
# حوالي رمز واحد لكل 1000-2000 حرف في بيانات التدريب)
config = MiniLLMConfig(vocab_size=8_000)

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التوليد في وضع train() يُفعّل Dropout عشوائيًا،
# مما يُنتج تذبذبًا غير مرغوب فيه في جودة التوليد
generated = model.generate(prompt_ids, max_new_tokens=50)

# ✅ صحيح: الدالة generate المعرّفة أعلاه تستدعي self.eval() داخليًا،
# لكن تأكد دائمًا من هذا التفصيل في أي كود توليد تكتبه بنفسك
model.eval()
with torch.no_grad():
    generated = model.generate(prompt_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاعتماد على تجربة واحدة فقط بـ seed=42 للحكم على جودة
# المعمارية أو الإعدادات — قد تكون النتيجة مجرد "حظ" إحصائي
torch.manual_seed(42)
model = pretrain_mini_llm(...)

# ✅ أفضل: شغّل التجربة بـ 2-3 بذور مختلفة على الأقل عند مقارنة
# إعدادات مختلفة (معدل تعلّم، حجم دفعة) للتأكد من أن الفروقات حقيقية
for seed in [42, 123, 777]:
    torch.manual_seed(seed)
    model = pretrain_mini_llm(...)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الإعجاب بانخفاض Perplexity على بيانات التدريب فقط
train_ppl = evaluate_perplexity(model, train_dataset, device)
print(f"النموذج ممتاز! PPL = {train_ppl}")  # قد يكون هذا مجرد حفظ (Overfitting)!

# ✅ صحيح: الحكم دائمًا بناءً على مجموعة Validation المنفصلة تمامًا
val_ppl = evaluate_perplexity(model, val_dataset, device)
print(f"PPL الحقيقي (Validation): {val_ppl}")

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
"""
تحدي موسّع: ثلاث تجارب تحليلية على Mini-LLM

التجربة الأولى - أثر حجم النموذج:
درّب 3 نسخ من Mini-LLM بأحجام مختلفة (d_model=128، 256، 512 مع تعديل
باقي الأبعاد تناسبيًا) على نفس البيانات وعدد الخطوات، وارسم منحنيات
Validation Loss الثلاثة على نفس الرسم البياني. هل يتبع السلوك المُلاحظ
نمط Scaling Laws الذي درسناه في الفصل 22 (ولو بشكل تقريبي جدًا على
هذا النطاق الصغير)؟

التجربة الثانية - أثر RoPE مقابل Learned Positional Embeddings:
ابنِ نسخة بديلة من MiniLLM تستخدم nn.Embedding عادي للمواضع بدلاً
من RoPE، درّب كلا النسختين، وقارن: هل تتفوق إحداهما في التعميم على
أطوال سياق أطول من تلك المستخدمة في التدريب؟ (اختبر على تسلسلات
أطول من max_seq_len الأصلي وراقب استقرار Perplexity)

التجربة الثالثة - أثر جودة بيانات SFT:
درّب نسختين من SFT: واحدة بـ10 أمثلة فقط، وأخرى بـ100 مثال (يمكنك
توليدها يدويًا أو عبر تنويع قوالب بسيطة). قارن سلوك التوقف الصحيح
(<|im_end|>) ودقة الإجابات بين النسختين على مجموعة اختبار من 20 سؤالاً
لم تظهر في أي من مجموعتي SFT.
"""

from __future__ import annotations


def experiment_model_scaling() -> None:
    """TODO: نفّذ التجربة الأولى وارسم النتائج باستخدام matplotlib."""
    raise NotImplementedError


def experiment_positional_encoding() -> None:
    """TODO: نفّذ التجربة الثانية، مع بناء نسخة MiniLLM بـ Learned
    Positional Embeddings كبديل لـ RoPE للمقارنة."""
    raise NotImplementedError


def experiment_sft_data_quality() -> None:
    """TODO: نفّذ التجربة الثالثة مع تقرير كمي (نسبة الإجابات الصحيحة،
    نسبة التوقف الصحيح) وليس فقط ملاحظات نوعية."""
    raise NotImplementedError


if __name__ == "__main__":
    experiment_model_scaling()
    experiment_positional_encoding()
    experiment_sft_data_quality()

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر
Large Language Model Engineering — Mini-LLM

Book slug : llm-engineering-69-chapters
Chapter   : ch-26-chapter-26 (lesson 26)
Code files: 13 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_26.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 26: الفصل 26: مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bigram() -> None:
    text = 'to be or not to be that is the question to be'
    pairs = Counter(zip(text.split(), text.split()[1:]))
    next_words = {word: sorted(((count, nxt) for (prev, nxt), count in pairs.items() if prev == word), reverse=True) for word in set(text.split())}
    print("learned next-word counts for 'to':", next_words['to'])
    print('most likely continuation:', next_words['to'][0][1])
    print('model parameters:', len(pairs), 'observed bigrams')

if __name__ == "__main__":
    print("Chapter 26 lab: Train a tiny next-character language model")
    run_bigram()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniLLMConfig:
    """تهيئة كاملة لتجربة Mini-LLM — كل الأبعاد مُصمَّمة لتناسب
    التدريب على جهاز واحد في وقت معقول، مع الحفاظ على نفس المبادئ
    المعمارية المستخدمة في النماذج الإنتاجية الضخمة."""

    # إعدادات المفردات والتوكنايزر
    vocab_size: int = 8000          # صغير نسبيًا، مناسب لمجموعة بيانات محدودة
    max_seq_len: int = 256          # طول سياق متواضع يكفي للتجربة

    # إعدادات المعمارية (نسب متوازنة على نمط LLaMA لكن مُصغَّرة)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4              # GQA: نصف عدد رؤوس الاستعلام (مرجع: الفصل 15)
    d_ff: int = 683                  # ≈ (2/3) × 4 × d_model، مُقرَّب (مرجع: الفصل 18)
    rope_theta: float = 10000.0

    # إعدادات التدريب
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    total_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    def estimate_param_count(self) -> int:
        """تقدير تقريبي لعدد المعاملات الإجمالي، تطبيقًا مباشرًا
        لصيغ الفصل الرابع عشر (دون احتساب embeddings المربوطة)."""
        embed_params = self.vocab_size * self.d_model
        per_layer_attn = 2 * self.d_model * self.d_model + 2 * (
            self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        )
        per_layer_ffn = 3 * self.d_model * self.d_ff  # SwiGLU: 3 مصفوفات
        per_layer_norm = 2 * self.d_model             # RMSNorm × 2
        per_layer_total = per_layer_attn + per_layer_ffn + per_layer_norm
        total = embed_params + self.n_layers * per_layer_total
        return total


if __name__ == "__main__":
    config = MiniLLMConfig()
    n_params = config.estimate_param_count()
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params / 1e6:.1f}M)")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from pathlib import Path


def clean_text_simple(raw_text: str) -> str:
    """تنظيف مُبسَّط (نسخة مُصغَّرة من مبادئ الفصل 20): إزالة أسطر فارغة
    متكررة، توحيد المسافات، إزالة أحرف تحكم غريبة."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    return text.strip()


def split_train_val(text: str, val_ratio: float = 0.02) -> tuple[str, str]:
    """تقسيم بسيط حسب الموضع — في مشروع حقيقي يُفضَّل التقسيم حسب
    الوثيقة الكاملة لتجنب تسرب معلومات بين المجموعتين (الفصل 4)."""
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


def prepare_corpus(raw_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = raw_path.read_text(encoding="utf-8")
    cleaned = clean_text_simple(raw_text)
    train_text, val_text = split_train_val(cleaned)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    print(f"✅ بيانات التدريب: {len(train_text):,} حرف")
    print(f"✅ بيانات التحقق: {len(val_text):,} حرف")
    return train_path, val_path

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def train_mini_tokenizer(
    train_file: Path, vocab_size: int, output_path: Path
) -> Tokenizer:
    """تدريب BPE بسيط باستخدام مكتبة tokenizers من Hugging Face،
    تطبيقًا مباشرًا لمنهجية الفصل الحادي والعشرين لكن بحجم مفردات أصغر."""

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<|im_start|>", "<|im_end|>",  # رموز Chat Template (الفصل 25)
        ],
        min_frequency=2,
    )

    tokenizer.train(files=[str(train_file)], trainer=trainer)
    tokenizer.save(str(output_path))
    print(f"✅ تم تدريب Tokenizer بمفردات حجمها {tokenizer.get_vocab_size():,}")
    return tokenizer

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """تطبيق مباشر لـ RMSNorm من الفصل الثامن عشر."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """حساب مسبق لجيب وجيب التمام اللازمين لتدوير RoPE (الفصل الخامس عشر)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق التدوير الموضعي على استعلامات/مفاتيح الانتباه."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """تطبيق مُصغَّر لـ GQA (الفصل الخامس عشر): عدد رؤوس المفاتيح/القيم
    أقل من رؤوس الاستعلام، مع تكرار (repeat_interleave) لمطابقة الأبعاد."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # تكرار K/V لمطابقة عدد رؤوس الاستعلام (جوهر GQA)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # استخدام الانتباه المُحسَّن المدمج في PyTorch (يُطبّق منطق
        # مشابه لـ FlashAttention داخليًا — الفصل الثالث عشر)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn_out)


class SwiGLUFeedForward(nn.Module):
    """تطبيق مباشر لـ SwiGLU من الفصل الثامن عشر."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniTransformerBlock(nn.Module):
    """كتلة Transformer كاملة بنمط Pre-Norm الحديث (الفصل 18)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(
            config.d_model, config.n_heads, config.n_kv_heads, config.max_seq_len
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """النموذج الكامل: Embedding + مكدّس كتل Transformer + رأس اللغة،
    مع ربط الأوزان (Weight Tying) بين Embedding وLM Head (الفصل 18/21)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: مشاركة نفس المصفوفة بين embedding وlm_head
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """توليد نص عبر أخذ عينات (Sampling) من توزيع Top-K بدرجة حرارة
        معينة — تطبيق مباشر لمفاهيم استراتيجيات التوليد (الفصل 38 لاحقًا)."""
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature

            top_values, top_indices = torch.topk(next_logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(-1, sampled_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


class TextDataset:
    """مجموعة بيانات بسيطة: تحميل نص مُرمَّز كاملاً في الذاكرة
    (مناسب لحجم مشروعنا المُصغَّر)، وأخذ نوافذ عشوائية منه."""

    def __init__(self, text_path: Path, tokenizer: Tokenizer, seq_len: int) -> None:
        text = text_path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        rng = np.random.default_rng()
        starts = rng.integers(0, max_start, size=batch_size)

        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: MiniLLMConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_mini_llm(
    config: MiniLLMConfig,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    checkpoint_dir: Path,
    device: torch.device,
) -> MiniLLM:
    torch.manual_seed(42)

    model = MiniLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(optimizer, config)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(config.total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_dataset.get_batch(config.batch_size, device)
            _, loss = model(x, labels=y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:5d}/{config.total_steps} | "
                f"Loss: {accumulated_loss:.4f} | "
                f"GradNorm: {grad_norm:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        if step % 500 == 0 and step > 0:
            val_loss = evaluate_perplexity(model, val_dataset, device, n_batches=20)
            print(f"  📊 Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": config, "step": step},
                    checkpoint_dir / "best_model.pt",
                )
                print(f"  ✅ نقطة تفتيش جديدة أفضل محفوظة عند الخطوة {step}")

    return model


@torch.no_grad()
def evaluate_perplexity(
    model: MiniLLM, dataset: TextDataset, device: torch.device, n_batches: int = 20
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch(model.config.batch_size, device)
        _, loss = model(x, labels=y)
        total_loss += loss.item()
    return total_loss / n_batches

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from tokenizers import Tokenizer


def build_toy_instruction_data() -> list[tuple[str, str]]:
    """مجموعة تعليمات بسيطة جدًا توضيحيًا — في مشروع حقيقي أكبر،
    استخدم منهجية Self-Instruct الكاملة من الفصل 25."""
    return [
        ("ما هو ناتج 2 زائد 2؟", "ناتج 2 زائد 2 هو 4."),
        ("اذكر لون السماء.", "لون السماء أزرق في النهار الصافي."),
        ("ما عاصمة مصر؟", "عاصمة مصر هي القاهرة."),
        # ... يُفترض توسيع هذه القائمة لعشرات أو مئات الأمثلة فعليًا
    ]


def format_sft_example(instruction: str, response: str) -> str:
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )


def sft_finetune_mini_llm(
    model: MiniLLM,
    tokenizer: Tokenizer,
    sft_data: list[tuple[str, str]],
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
) -> MiniLLM:
    """SFT مُصغَّر: معدل تعلّم أقل بعشرة أضعاف من Pretraining،
    حقبات قليلة جدًا — تطبيق مباشر لمبادئ الفصل الخامس والعشرين."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    assistant_marker = tokenizer.encode("assistant\n").ids

    for epoch in range(epochs):
        total_loss = 0.0
        for instruction, response in sft_data:
            full_text = format_sft_example(instruction, response)
            ids = tokenizer.encode(full_text).ids
            input_ids = torch.tensor([ids], device=device)

            # بناء قناع خسارة مُبسَّط: نُقنّع كل شيء قبل أول ظهور لعلامة
            # "assistant\n" — تبسيط توضيحي لمنطق الفصل 25 الكامل
            labels = input_ids.clone()
            marker_pos = _find_subsequence(ids, assistant_marker)
            if marker_pos is not None:
                mask_end = marker_pos + len(assistant_marker)
                labels[0, :mask_end] = -100

            _, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        print(f"SFT Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(sft_data):.4f}")

    return model


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    n, m = len(sequence), len(subsequence)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            return i
    return None

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ الجهاز المستخدم: {device}")

    workspace = Path("./mini_llm_workspace")
    workspace.mkdir(exist_ok=True)

    # المرحلة 1: تحضير البيانات
    raw_corpus_path = Path("raw_corpus.txt")  # ضع هنا ملف نصي (عدة ميجابايتات)
    train_path, val_path = prepare_corpus(raw_corpus_path, workspace / "data")

    # المرحلة 2: تدريب Tokenizer
    config = MiniLLMConfig()
    tokenizer = train_mini_tokenizer(
        train_path, config.vocab_size, workspace / "tokenizer.json"
    )

    # المرحلة 3+4: بناء النموذج والتدريب
    train_dataset = TextDataset(train_path, tokenizer, config.max_seq_len)
    val_dataset = TextDataset(val_path, tokenizer, config.max_seq_len)

    model = pretrain_mini_llm(
        config, train_dataset, val_dataset, workspace / "checkpoints", device
    )

    # المرحلة 5: تقييم نوعي قبل SFT
    print("\n=== توليد نموذجي قبل SFT ===")
    prompt_ids = torch.tensor(
        [tokenizer.encode("في يوم من الأيام").ids], device=device
    )
    generated = model.generate(prompt_ids, max_new_tokens=50)
    print(tokenizer.decode(generated[0].tolist()))

    # المرحلة 6: SFT
    sft_data = build_toy_instruction_data()
    model = sft_finetune_mini_llm(model, tokenizer, sft_data, device)

    # المرحلة 7: تقييم نوعي بعد SFT
    print("\n=== الاستجابة بعد SFT ===")
    test_prompt = "<|im_start|>user\nما عاصمة مصر؟<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt).ids], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=30, temperature=0.3)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مفردات كبيرة جدًا (50000) على نص صغير (بضع ميجابايتات فقط)
# يؤدي إلى رموز نادرة جدًا لا تتعلّم تمثيلات جيدة لها إطلاقًا
config = MiniLLMConfig(vocab_size=50_000)

# ✅ صحيح: حجم مفردات متناسب مع حجم البيانات (قاعدة تقريبية:
# حوالي رمز واحد لكل 1000-2000 حرف في بيانات التدريب)
config = MiniLLMConfig(vocab_size=8_000)

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التوليد في وضع train() يُفعّل Dropout عشوائيًا،
# مما يُنتج تذبذبًا غير مرغوب فيه في جودة التوليد
generated = model.generate(prompt_ids, max_new_tokens=50)

# ✅ صحيح: الدالة generate المعرّفة أعلاه تستدعي self.eval() داخليًا،
# لكن تأكد دائمًا من هذا التفصيل في أي كود توليد تكتبه بنفسك
model.eval()
with torch.no_grad():
    generated = model.generate(prompt_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [011] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاعتماد على تجربة واحدة فقط بـ seed=42 للحكم على جودة
# المعمارية أو الإعدادات — قد تكون النتيجة مجرد "حظ" إحصائي
torch.manual_seed(42)
model = pretrain_mini_llm(...)

# ✅ أفضل: شغّل التجربة بـ 2-3 بذور مختلفة على الأقل عند مقارنة
# إعدادات مختلفة (معدل تعلّم، حجم دفعة) للتأكد من أن الفروقات حقيقية
for seed in [42, 123, 777]:
    torch.manual_seed(seed)
    model = pretrain_mini_llm(...)

# ────────────────────────────────────────────────────────────────────────
# [012] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الإعجاب بانخفاض Perplexity على بيانات التدريب فقط
train_ppl = evaluate_perplexity(model, train_dataset, device)
print(f"النموذج ممتاز! PPL = {train_ppl}")  # قد يكون هذا مجرد حفظ (Overfitting)!

# ✅ صحيح: الحكم دائمًا بناءً على مجموعة Validation المنفصلة تمامًا
val_ppl = evaluate_perplexity(model, val_dataset, device)
print(f"PPL الحقيقي (Validation): {val_ppl}")

# ────────────────────────────────────────────────────────────────────────
# [013] book/chapters/part-05-build-an-llm/chapter-26.md
# ────────────────────────────────────────────────────────────────────────
"""
تحدي موسّع: ثلاث تجارب تحليلية على Mini-LLM

التجربة الأولى - أثر حجم النموذج:
درّب 3 نسخ من Mini-LLM بأحجام مختلفة (d_model=128، 256، 512 مع تعديل
باقي الأبعاد تناسبيًا) على نفس البيانات وعدد الخطوات، وارسم منحنيات
Validation Loss الثلاثة على نفس الرسم البياني. هل يتبع السلوك المُلاحظ
نمط Scaling Laws الذي درسناه في الفصل 22 (ولو بشكل تقريبي جدًا على
هذا النطاق الصغير)؟

التجربة الثانية - أثر RoPE مقابل Learned Positional Embeddings:
ابنِ نسخة بديلة من MiniLLM تستخدم nn.Embedding عادي للمواضع بدلاً
من RoPE، درّب كلا النسختين، وقارن: هل تتفوق إحداهما في التعميم على
أطوال سياق أطول من تلك المستخدمة في التدريب؟ (اختبر على تسلسلات
أطول من max_seq_len الأصلي وراقب استقرار Perplexity)

التجربة الثالثة - أثر جودة بيانات SFT:
درّب نسختين من SFT: واحدة بـ10 أمثلة فقط، وأخرى بـ100 مثال (يمكنك
توليدها يدويًا أو عبر تنويع قوالب بسيطة). قارن سلوك التوقف الصحيح
(<|im_end|>) ودقة الإجابات بين النسختين على مجموعة اختبار من 20 سؤالاً
لم تظهر في أي من مجموعتي SFT.
"""

from __future__ import annotations


def experiment_model_scaling() -> None:
    """TODO: نفّذ التجربة الأولى وارسم النتائج باستخدام matplotlib."""
    raise NotImplementedError


def experiment_positional_encoding() -> None:
    """TODO: نفّذ التجربة الثانية، مع بناء نسخة MiniLLM بـ Learned
    Positional Embeddings كبديل لـ RoPE للمقارنة."""
    raise NotImplementedError


def experiment_sft_data_quality() -> None:
    """TODO: نفّذ التجربة الثالثة مع تقرير كمي (نسبة الإجابات الصحيحة،
    نسبة التوقف الصحيح) وليس فقط ملاحظات نوعية."""
    raise NotImplementedError


if __name__ == "__main__":
    experiment_model_scaling()
    experiment_positional_encoding()
    experiment_sft_data_quality()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 26: الفصل 26: مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bigram() -> None:
    text = 'to be or not to be that is the question to be'
    pairs = Counter(zip(text.split(), text.split()[1:]))
    next_words = {word: sorted(((count, nxt) for (prev, nxt), count in pairs.items() if prev == word), reverse=True) for word in set(text.split())}
    print("learned next-word counts for 'to':", next_words['to'])
    print('most likely continuation:', next_words['to'][0][1])
    print('model parameters:', len(pairs), 'observed bigrams')

if __name__ == "__main__":
    print("Chapter 26 lab: Train a tiny next-character language model")
    run_bigram()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniLLMConfig:
    """تهيئة كاملة لتجربة Mini-LLM — كل الأبعاد مُصمَّمة لتناسب
    التدريب على جهاز واحد في وقت معقول، مع الحفاظ على نفس المبادئ
    المعمارية المستخدمة في النماذج الإنتاجية الضخمة."""

    # إعدادات المفردات والتوكنايزر
    vocab_size: int = 8000          # صغير نسبيًا، مناسب لمجموعة بيانات محدودة
    max_seq_len: int = 256          # طول سياق متواضع يكفي للتجربة

    # إعدادات المعمارية (نسب متوازنة على نمط LLaMA لكن مُصغَّرة)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4              # GQA: نصف عدد رؤوس الاستعلام (مرجع: الفصل 15)
    d_ff: int = 683                  # ≈ (2/3) × 4 × d_model، مُقرَّب (مرجع: الفصل 18)
    rope_theta: float = 10000.0

    # إعدادات التدريب
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    total_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    def estimate_param_count(self) -> int:
        """تقدير تقريبي لعدد المعاملات الإجمالي، تطبيقًا مباشرًا
        لصيغ الفصل الرابع عشر (دون احتساب embeddings المربوطة)."""
        embed_params = self.vocab_size * self.d_model
        per_layer_attn = 2 * self.d_model * self.d_model + 2 * (
            self.d_model * (self.d_model * self.n_kv_heads // self.n_heads)
        )
        per_layer_ffn = 3 * self.d_model * self.d_ff  # SwiGLU: 3 مصفوفات
        per_layer_norm = 2 * self.d_model             # RMSNorm × 2
        per_layer_total = per_layer_attn + per_layer_ffn + per_layer_norm
        total = embed_params + self.n_layers * per_layer_total
        return total


if __name__ == "__main__":
    config = MiniLLMConfig()
    n_params = config.estimate_param_count()
    print(f"عدد المعاملات التقريبي: {n_params:,} (≈ {n_params / 1e6:.1f}M)")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from pathlib import Path


def clean_text_simple(raw_text: str) -> str:
    """تنظيف مُبسَّط (نسخة مُصغَّرة من مبادئ الفصل 20): إزالة أسطر فارغة
    متكررة، توحيد المسافات، إزالة أحرف تحكم غريبة."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\S\n]+\n", "\n", text)
    return text.strip()


def split_train_val(text: str, val_ratio: float = 0.02) -> tuple[str, str]:
    """تقسيم بسيط حسب الموضع — في مشروع حقيقي يُفضَّل التقسيم حسب
    الوثيقة الكاملة لتجنب تسرب معلومات بين المجموعتين (الفصل 4)."""
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


def prepare_corpus(raw_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = raw_path.read_text(encoding="utf-8")
    cleaned = clean_text_simple(raw_text)
    train_text, val_text = split_train_val(cleaned)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    print(f"✅ بيانات التدريب: {len(train_text):,} حرف")
    print(f"✅ بيانات التحقق: {len(val_text):,} حرف")
    return train_path, val_path

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def train_mini_tokenizer(
    train_file: Path, vocab_size: int, output_path: Path
) -> Tokenizer:
    """تدريب BPE بسيط باستخدام مكتبة tokenizers من Hugging Face،
    تطبيقًا مباشرًا لمنهجية الفصل الحادي والعشرين لكن بحجم مفردات أصغر."""

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<|im_start|>", "<|im_end|>",  # رموز Chat Template (الفصل 25)
        ],
        min_frequency=2,
    )

    tokenizer.train(files=[str(train_file)], trainer=trainer)
    tokenizer.save(str(output_path))
    print(f"✅ تم تدريب Tokenizer بمفردات حجمها {tokenizer.get_vocab_size():,}")
    return tokenizer

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """تطبيق مباشر لـ RMSNorm من الفصل الثامن عشر."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """حساب مسبق لجيب وجيب التمام اللازمين لتدوير RoPE (الفصل الخامس عشر)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """تطبيق التدوير الموضعي على استعلامات/مفاتيح الانتباه."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2), :].unsqueeze(0).unsqueeze(0)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class GroupedQueryAttention(nn.Module):
    """تطبيق مُصغَّر لـ GQA (الفصل الخامس عشر): عدد رؤوس المفاتيح/القيم
    أقل من رؤوس الاستعلام، مع تكرار (repeat_interleave) لمطابقة الأبعاد."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int) -> None:
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # تكرار K/V لمطابقة عدد رؤوس الاستعلام (جوهر GQA)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # استخدام الانتباه المُحسَّن المدمج في PyTorch (يُطبّق منطق
        # مشابه لـ FlashAttention داخليًا — الفصل الثالث عشر)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn_out)


class SwiGLUFeedForward(nn.Module):
    """تطبيق مباشر لـ SwiGLU من الفصل الثامن عشر."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MiniTransformerBlock(nn.Module):
    """كتلة Transformer كاملة بنمط Pre-Norm الحديث (الفصل 18)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(
            config.d_model, config.n_heads, config.n_kv_heads, config.max_seq_len
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """النموذج الكامل: Embedding + مكدّس كتل Transformer + رأس اللغة،
    مع ربط الأوزان (Weight Tying) بين Embedding وLM Head (الفصل 18/21)."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [MiniTransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying: مشاركة نفس المصفوفة بين embedding وlm_head
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """توليد نص عبر أخذ عينات (Sampling) من توزيع Top-K بدرجة حرارة
        معينة — تطبيق مباشر لمفاهيم استراتيجيات التوليد (الفصل 38 لاحقًا)."""
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature

            top_values, top_indices = torch.topk(next_logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(-1, sampled_idx)

            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


class TextDataset:
    """مجموعة بيانات بسيطة: تحميل نص مُرمَّز كاملاً في الذاكرة
    (مناسب لحجم مشروعنا المُصغَّر)، وأخذ نوافذ عشوائية منه."""

    def __init__(self, text_path: Path, tokenizer: Tokenizer, seq_len: int) -> None:
        text = text_path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        rng = np.random.default_rng()
        starts = rng.integers(0, max_start, size=batch_size)

        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: MiniLLMConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_mini_llm(
    config: MiniLLMConfig,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    checkpoint_dir: Path,
    device: torch.device,
) -> MiniLLM:
    torch.manual_seed(42)

    model = MiniLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_scheduler(optimizer, config)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(config.total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_dataset.get_batch(config.batch_size, device)
            _, loss = model(x, labels=y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:5d}/{config.total_steps} | "
                f"Loss: {accumulated_loss:.4f} | "
                f"GradNorm: {grad_norm:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        if step % 500 == 0 and step > 0:
            val_loss = evaluate_perplexity(model, val_dataset, device, n_batches=20)
            print(f"  📊 Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state": model.state_dict(), "config": config, "step": step},
                    checkpoint_dir / "best_model.pt",
                )
                print(f"  ✅ نقطة تفتيش جديدة أفضل محفوظة عند الخطوة {step}")

    return model


@torch.no_grad()
def evaluate_perplexity(
    model: MiniLLM, dataset: TextDataset, device: torch.device, n_batches: int = 20
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        x, y = dataset.get_batch(model.config.batch_size, device)
        _, loss = model(x, labels=y)
        total_loss += loss.item()
    return total_loss / n_batches

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
from tokenizers import Tokenizer


def build_toy_instruction_data() -> list[tuple[str, str]]:
    """مجموعة تعليمات بسيطة جدًا توضيحيًا — في مشروع حقيقي أكبر،
    استخدم منهجية Self-Instruct الكاملة من الفصل 25."""
    return [
        ("ما هو ناتج 2 زائد 2؟", "ناتج 2 زائد 2 هو 4."),
        ("اذكر لون السماء.", "لون السماء أزرق في النهار الصافي."),
        ("ما عاصمة مصر؟", "عاصمة مصر هي القاهرة."),
        # ... يُفترض توسيع هذه القائمة لعشرات أو مئات الأمثلة فعليًا
    ]


def format_sft_example(instruction: str, response: str) -> str:
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
    )


def sft_finetune_mini_llm(
    model: MiniLLM,
    tokenizer: Tokenizer,
    sft_data: list[tuple[str, str]],
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
) -> MiniLLM:
    """SFT مُصغَّر: معدل تعلّم أقل بعشرة أضعاف من Pretraining،
    حقبات قليلة جدًا — تطبيق مباشر لمبادئ الفصل الخامس والعشرين."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    assistant_marker = tokenizer.encode("assistant\n").ids

    for epoch in range(epochs):
        total_loss = 0.0
        for instruction, response in sft_data:
            full_text = format_sft_example(instruction, response)
            ids = tokenizer.encode(full_text).ids
            input_ids = torch.tensor([ids], device=device)

            # بناء قناع خسارة مُبسَّط: نُقنّع كل شيء قبل أول ظهور لعلامة
            # "assistant\n" — تبسيط توضيحي لمنطق الفصل 25 الكامل
            labels = input_ids.clone()
            marker_pos = _find_subsequence(ids, assistant_marker)
            if marker_pos is not None:
                mask_end = marker_pos + len(assistant_marker)
                labels[0, :mask_end] = -100

            _, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        print(f"SFT Epoch {epoch + 1}/{epochs} | Avg Loss: {total_loss / len(sft_data):.4f}")

    return model


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int | None:
    n, m = len(sequence), len(subsequence)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            return i
    return None

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ الجهاز المستخدم: {device}")

    workspace = Path("./mini_llm_workspace")
    workspace.mkdir(exist_ok=True)

    # المرحلة 1: تحضير البيانات
    raw_corpus_path = Path("raw_corpus.txt")  # ضع هنا ملف نصي (عدة ميجابايتات)
    train_path, val_path = prepare_corpus(raw_corpus_path, workspace / "data")

    # المرحلة 2: تدريب Tokenizer
    config = MiniLLMConfig()
    tokenizer = train_mini_tokenizer(
        train_path, config.vocab_size, workspace / "tokenizer.json"
    )

    # المرحلة 3+4: بناء النموذج والتدريب
    train_dataset = TextDataset(train_path, tokenizer, config.max_seq_len)
    val_dataset = TextDataset(val_path, tokenizer, config.max_seq_len)

    model = pretrain_mini_llm(
        config, train_dataset, val_dataset, workspace / "checkpoints", device
    )

    # المرحلة 5: تقييم نوعي قبل SFT
    print("\n=== توليد نموذجي قبل SFT ===")
    prompt_ids = torch.tensor(
        [tokenizer.encode("في يوم من الأيام").ids], device=device
    )
    generated = model.generate(prompt_ids, max_new_tokens=50)
    print(tokenizer.decode(generated[0].tolist()))

    # المرحلة 6: SFT
    sft_data = build_toy_instruction_data()
    model = sft_finetune_mini_llm(model, tokenizer, sft_data, device)

    # المرحلة 7: تقييم نوعي بعد SFT
    print("\n=== الاستجابة بعد SFT ===")
    test_prompt = "<|im_start|>user\nما عاصمة مصر؟<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = torch.tensor([tokenizer.encode(test_prompt).ids], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=30, temperature=0.3)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: مفردات كبيرة جدًا (50000) على نص صغير (بضع ميجابايتات فقط)
# يؤدي إلى رموز نادرة جدًا لا تتعلّم تمثيلات جيدة لها إطلاقًا
config = MiniLLMConfig(vocab_size=50_000)

# ✅ صحيح: حجم مفردات متناسب مع حجم البيانات (قاعدة تقريبية:
# حوالي رمز واحد لكل 1000-2000 حرف في بيانات التدريب)
config = MiniLLMConfig(vocab_size=8_000)

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: التوليد في وضع train() يُفعّل Dropout عشوائيًا،
# مما يُنتج تذبذبًا غير مرغوب فيه في جودة التوليد
generated = model.generate(prompt_ids, max_new_tokens=50)

# ✅ صحيح: الدالة generate المعرّفة أعلاه تستدعي self.eval() داخليًا،
# لكن تأكد دائمًا من هذا التفصيل في أي كود توليد تكتبه بنفسك
model.eval()
with torch.no_grad():
    generated = model.generate(prompt_ids, max_new_tokens=50)

# ────────────────────────────────────────────────────────────────────────
# [013] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-12
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الاعتماد على تجربة واحدة فقط بـ seed=42 للحكم على جودة
# المعمارية أو الإعدادات — قد تكون النتيجة مجرد "حظ" إحصائي
torch.manual_seed(42)
model = pretrain_mini_llm(...)

# ✅ أفضل: شغّل التجربة بـ 2-3 بذور مختلفة على الأقل عند مقارنة
# إعدادات مختلفة (معدل تعلّم، حجم دفعة) للتأكد من أن الفروقات حقيقية
for seed in [42, 123, 777]:
    torch.manual_seed(seed)
    model = pretrain_mini_llm(...)

# ────────────────────────────────────────────────────────────────────────
# [014] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-13
# ────────────────────────────────────────────────────────────────────────
# ❌ خطأ: الإعجاب بانخفاض Perplexity على بيانات التدريب فقط
train_ppl = evaluate_perplexity(model, train_dataset, device)
print(f"النموذج ممتاز! PPL = {train_ppl}")  # قد يكون هذا مجرد حفظ (Overfitting)!

# ✅ صحيح: الحكم دائمًا بناءً على مجموعة Validation المنفصلة تمامًا
val_ppl = evaluate_perplexity(model, val_dataset, device)
print(f"PPL الحقيقي (Validation): {val_ppl}")

# ────────────────────────────────────────────────────────────────────────
# [015] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-26-chapter-26.ipynb#cell-14
# ────────────────────────────────────────────────────────────────────────
"""
تحدي موسّع: ثلاث تجارب تحليلية على Mini-LLM

التجربة الأولى - أثر حجم النموذج:
درّب 3 نسخ من Mini-LLM بأحجام مختلفة (d_model=128، 256، 512 مع تعديل
باقي الأبعاد تناسبيًا) على نفس البيانات وعدد الخطوات، وارسم منحنيات
Validation Loss الثلاثة على نفس الرسم البياني. هل يتبع السلوك المُلاحظ
نمط Scaling Laws الذي درسناه في الفصل 22 (ولو بشكل تقريبي جدًا على
هذا النطاق الصغير)؟

التجربة الثانية - أثر RoPE مقابل Learned Positional Embeddings:
ابنِ نسخة بديلة من MiniLLM تستخدم nn.Embedding عادي للمواضع بدلاً
من RoPE، درّب كلا النسختين، وقارن: هل تتفوق إحداهما في التعميم على
أطوال سياق أطول من تلك المستخدمة في التدريب؟ (اختبر على تسلسلات
أطول من max_seq_len الأصلي وراقب استقرار Perplexity)

التجربة الثالثة - أثر جودة بيانات SFT:
درّب نسختين من SFT: واحدة بـ10 أمثلة فقط، وأخرى بـ100 مثال (يمكنك
توليدها يدويًا أو عبر تنويع قوالب بسيطة). قارن سلوك التوقف الصحيح
(<|im_end|>) ودقة الإجابات بين النسختين على مجموعة اختبار من 20 سؤالاً
لم تظهر في أي من مجموعتي SFT.
"""

from __future__ import annotations


def experiment_model_scaling() -> None:
    """TODO: نفّذ التجربة الأولى وارسم النتائج باستخدام matplotlib."""
    raise NotImplementedError


def experiment_positional_encoding() -> None:
    """TODO: نفّذ التجربة الثانية، مع بناء نسخة MiniLLM بـ Learned
    Positional Embeddings كبديل لـ RoPE للمقارنة."""
    raise NotImplementedError


def experiment_sft_data_quality() -> None:
    """TODO: نفّذ التجربة الثالثة مع تقرير كمي (نسبة الإجابات الصحيحة،
    نسبة التوقف الصحيح) وليس فقط ملاحظات نوعية."""
    raise NotImplementedError


if __name__ == "__main__":
    experiment_model_scaling()
    experiment_positional_encoding()
    experiment_sft_data_quality()
