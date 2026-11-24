#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الدافع وراء Self-Attention
Large Language Model Engineering — Self-Attention

Book slug : llm-engineering-69-chapters
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 9 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-09-chapter-09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الدافع وراء Self-Attention
Large Language Model Engineering — Self-Attention

Book slug : llm-engineering-69-chapters
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 8 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-09-chapter-09.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الدافع وراء Self-Attention
Large Language Model Engineering — Self-Attention

Book slug : llm-engineering-69-chapters
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 09: الفصل التاسع: الدافع وراء Self-Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 09 lab: Alignment attention over encoder states")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
"""
seq2seq_bahdanau_attention.py
نموذج Seq2Seq مع Bahdanau (Additive) Attention من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


# =====================================================================
# 1) المُرمِّز (Encoder): LSTM ثنائي الاتجاه، يُحتفَظ بكل الحالات المخفية
# =====================================================================
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        # دمج الاتجاهين إلى hidden_size واحد كي يتوافق مع فك المُرمِّز أحادي الاتجاه
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        src: (batch, src_len)
        يُعيد:
          encoder_outputs: (batch, src_len, hidden_size*2) -- كل الحالات! هذا هو المفتاح
          (h_0_dec, c_0_dec): الحالة الابتدائية لفك المُرمِّز
        """
        embedded = self.embedding(src)                       # (batch, src_len, emb_dim)
        outputs, (h_n, c_n) = self.lstm(embedded)             # outputs: (batch, src_len, 2*hidden)

        # h_n شكلها (2, batch, hidden) لأن ثنائي الاتجاه (forward + backward)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)        # (batch, 2*hidden)
        c_combined = torch.cat([c_n[0], c_n[1]], dim=1)

        h_0_dec = torch.tanh(self.fc_hidden(h_combined)).unsqueeze(0)  # (1, batch, hidden)
        c_0_dec = torch.tanh(self.fc_cell(c_combined)).unsqueeze(0)

        return outputs, (h_0_dec, c_0_dec)


# =====================================================================
# 2) آلية Bahdanau Attention (Additive Scoring)
# =====================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_size: int, encoder_output_size: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.W_a = nn.Linear(decoder_hidden_size, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_output_size, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        decoder_hidden:  (batch, decoder_hidden_size)         -- s_{t-1}، استعلام اللحظة الحالية
        encoder_outputs: (batch, src_len, encoder_output_size) -- h_1...h_{T_x}
        يُعيد:
          context: (batch, encoder_output_size)   -- c_t، متجه السياق المُحسوب ديناميكيًا
          attn_weights: (batch, src_len)           -- α_{t,i}، أوزان الانتباه (للتصوّر والتفسير!)
        """
        src_len = encoder_outputs.size(1)

        # وسّع الحالة المخفية لتتوافق مع كل خطوة زمنية في المُرمِّز
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_hidden_size)

        # e_{t,i} = v_a^T tanh(W_a s_{t-1} + U_a h_i)
        energy = torch.tanh(
            self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs)
        )                                                    # (batch, src_len, attn_dim)
        scores = self.v_a(energy).squeeze(2)                  # (batch, src_len)

        attn_weights = F.softmax(scores, dim=1)               # α_{t,i} -- يجمع 1 عبر src_len
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # (batch, 1, src_len) @ (batch, src_len, enc_size) = (batch, 1, enc_size) → (batch, enc_size)

        return context, attn_weights


# =====================================================================
# 3) فك المُرمِّز (Decoder) الذي يستخدم Attention في كل خطوة توليد
# =====================================================================
class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_output_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size, encoder_output_size)
        self.lstm = nn.LSTM(
            embedding_dim + encoder_output_size, hidden_size, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size + encoder_output_size + embedding_dim, vocab_size)

    def forward_step(
        self,
        input_token: Tensor,                       # (batch, 1) -- كلمة واحدة فقط
        state: tuple[Tensor, Tensor],               # (h_{t-1}, c_{t-1})
        encoder_outputs: Tensor,                    # (batch, src_len, encoder_output_size)
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        h_prev, c_prev = state
        embedded = self.embedding(input_token)       # (batch, 1, emb_dim)

        # نحسب الانتباه باستخدام آخر حالة مخفية لفك المُرمِّز كاستعلام
        context, attn_weights = self.attention(h_prev.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)               # (batch, 1, encoder_output_size)

        lstm_input = torch.cat([embedded, context], dim=2)   # دمج المدخل الحالي مع السياق الديناميكي
        output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))

        combined = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        logits = self.fc_out(combined)                # (batch, vocab_size)

        return logits, (h_t, c_t), attn_weights


# =====================================================================
# 4) اختبار سريع للأشكال (Sanity Check)
# =====================================================================
def sanity_check() -> None:
    batch, src_len, src_vocab, tgt_vocab = 4, 12, 3000, 3500
    embedding_dim, hidden_size = 128, 256

    encoder = EncoderLSTM(src_vocab, embedding_dim, hidden_size)
    decoder = DecoderWithAttention(tgt_vocab, embedding_dim, hidden_size, hidden_size * 2)

    src = torch.randint(1, src_vocab, (batch, src_len))
    encoder_outputs, (h0, c0) = encoder(src)
    print(f"encoder_outputs: {encoder_outputs.shape}")   # (4, 12, 512)

    first_token = torch.randint(1, tgt_vocab, (batch, 1))
    logits, (h1, c1), attn_weights = decoder.forward_step(first_token, (h0, c0), encoder_outputs)
    print(f"logits: {logits.shape}")           # (4, 3500)
    print(f"attn_weights: {attn_weights.shape}")  # (4, 12) -- توزيع احتمالي على كلمات المصدر!
    print(f"مجموع أوزان الانتباه (يجب أن يساوي 1 تقريبًا): {attn_weights[0].sum().item():.4f}")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# حالات مُرمِّز مُبسَّطة جدًا (3 كلمات، أبعاد=4 للتوضيح فقط)
h_encoder = torch.tensor([
    [1.0, 0.2, 0.1, 0.0],   # h1: "القطة"
    [0.1, 1.0, 0.3, 0.2],   # h2: "تأكل"
    [0.0, 0.1, 1.0, 0.1],   # h3: "السمك"
])

# استعلام فك المُرمِّز عند توليد "cat" -- قريب جدًا من h1 عمدًا للتوضيح
s_query_step1 = torch.tensor([0.9, 0.3, 0.05, 0.1])

scores = h_encoder @ s_query_step1  # ضرب نقطي بسيط (أسلوب Luong)
attn_weights = F.softmax(scores, dim=0)

print("الدرجات الخام:", scores)
print("أوزان الانتباه بعد Softmax:", attn_weights)
print("مجموع الأوزان (يجب = 1):", attn_weights.sum().item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
scores = h_encoder @ query  # ❌ يشمل درجات لرموز حشو وهمية!
attn_weights = F.softmax(scores, dim=0)  # سيُوزّع احتمالًا حقيقيًا على الحشو

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
scores = scores.masked_fill(padding_mask == 0, float('-inf'))
attn_weights = F.softmax(scores, dim=0)  # ✅ الحشو يحصل على احتمال ≈ 0

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
# فكر خاطئ: "استخدمت Attention، إذن أصبح نموذجي سريعًا ومتوازيًا تمامًا"

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
"""
translation_with_attention_visualization.py
مشروع تطبيقي: ترجمة آلية صغيرة مع تصوير خرائط الانتباه.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def translate_and_visualize(
    sentence: str,
    encoder: "EncoderLSTM",
    decoder: "DecoderWithAttention",
    src_vocab: dict[str, int],
    tgt_vocab: dict[int, str],
    max_length: int = 20,
) -> tuple[str, np.ndarray]:
    """
    يترجم جملة واحدة بأسلوب Greedy Decoding،
    ويُعيد الترجمة النصية + مصفوفة أوزان الانتباه الكاملة (target_len, source_len)
    """
    # TODO 1: رمّز الجملة المصدر وحوّلها لـ Tensor
    # TODO 2: مرّرها عبر الـ Encoder للحصول على encoder_outputs والحالة الابتدائية
    # TODO 3: حلقة توليد Greedy: ابدأ برمز <BOS>، استدعِ decoder.forward_step() تكراريًا
    #         حتى الوصول لرمز <EOS> أو max_length، واحفظ attn_weights في كل خطوة
    raise NotImplementedError


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path: str = "attention_heatmap.png",
) -> None:
    """يرسم خريطة حرارية: محور أفقي=كلمات المصدر، محور رأسي=كلمات الهدف المُولَّدة"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap="viridis")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens)

    plt.colorbar(im)
    plt.xlabel("الجملة المصدر")
    plt.ylabel("الجملة المُترجَمة")
    plt.title("خريطة أوزان الانتباه (Attention Heatmap)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"تم حفظ الخريطة في: {output_path}")

# تحدٍّ إضافي: قارن خريطة الانتباه لجملة بترتيب كلمات متشابه بين
# العربية والإنجليزية، مقابل جملة بترتيب مختلف جوهريًا (مثل جملة تحتوي
# صفة متأخرة في إحدى اللغتين ومتقدمة في الأخرى)، وناقش الفروقات.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 09: الفصل التاسع: الدافع وراء Self-Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 09 lab: Alignment attention over encoder states")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
seq2seq_bahdanau_attention.py
نموذج Seq2Seq مع Bahdanau (Additive) Attention من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


# =====================================================================
# 1) المُرمِّز (Encoder): LSTM ثنائي الاتجاه، يُحتفَظ بكل الحالات المخفية
# =====================================================================
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        # دمج الاتجاهين إلى hidden_size واحد كي يتوافق مع فك المُرمِّز أحادي الاتجاه
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        src: (batch, src_len)
        يُعيد:
          encoder_outputs: (batch, src_len, hidden_size*2) -- كل الحالات! هذا هو المفتاح
          (h_0_dec, c_0_dec): الحالة الابتدائية لفك المُرمِّز
        """
        embedded = self.embedding(src)                       # (batch, src_len, emb_dim)
        outputs, (h_n, c_n) = self.lstm(embedded)             # outputs: (batch, src_len, 2*hidden)

        # h_n شكلها (2, batch, hidden) لأن ثنائي الاتجاه (forward + backward)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)        # (batch, 2*hidden)
        c_combined = torch.cat([c_n[0], c_n[1]], dim=1)

        h_0_dec = torch.tanh(self.fc_hidden(h_combined)).unsqueeze(0)  # (1, batch, hidden)
        c_0_dec = torch.tanh(self.fc_cell(c_combined)).unsqueeze(0)

        return outputs, (h_0_dec, c_0_dec)


# =====================================================================
# 2) آلية Bahdanau Attention (Additive Scoring)
# =====================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_size: int, encoder_output_size: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.W_a = nn.Linear(decoder_hidden_size, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_output_size, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        decoder_hidden:  (batch, decoder_hidden_size)         -- s_{t-1}، استعلام اللحظة الحالية
        encoder_outputs: (batch, src_len, encoder_output_size) -- h_1...h_{T_x}
        يُعيد:
          context: (batch, encoder_output_size)   -- c_t، متجه السياق المُحسوب ديناميكيًا
          attn_weights: (batch, src_len)           -- α_{t,i}، أوزان الانتباه (للتصوّر والتفسير!)
        """
        src_len = encoder_outputs.size(1)

        # وسّع الحالة المخفية لتتوافق مع كل خطوة زمنية في المُرمِّز
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_hidden_size)

        # e_{t,i} = v_a^T tanh(W_a s_{t-1} + U_a h_i)
        energy = torch.tanh(
            self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs)
        )                                                    # (batch, src_len, attn_dim)
        scores = self.v_a(energy).squeeze(2)                  # (batch, src_len)

        attn_weights = F.softmax(scores, dim=1)               # α_{t,i} -- يجمع 1 عبر src_len
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # (batch, 1, src_len) @ (batch, src_len, enc_size) = (batch, 1, enc_size) → (batch, enc_size)

        return context, attn_weights


# =====================================================================
# 3) فك المُرمِّز (Decoder) الذي يستخدم Attention في كل خطوة توليد
# =====================================================================
class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_output_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size, encoder_output_size)
        self.lstm = nn.LSTM(
            embedding_dim + encoder_output_size, hidden_size, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size + encoder_output_size + embedding_dim, vocab_size)

    def forward_step(
        self,
        input_token: Tensor,                       # (batch, 1) -- كلمة واحدة فقط
        state: tuple[Tensor, Tensor],               # (h_{t-1}, c_{t-1})
        encoder_outputs: Tensor,                    # (batch, src_len, encoder_output_size)
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        h_prev, c_prev = state
        embedded = self.embedding(input_token)       # (batch, 1, emb_dim)

        # نحسب الانتباه باستخدام آخر حالة مخفية لفك المُرمِّز كاستعلام
        context, attn_weights = self.attention(h_prev.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)               # (batch, 1, encoder_output_size)

        lstm_input = torch.cat([embedded, context], dim=2)   # دمج المدخل الحالي مع السياق الديناميكي
        output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))

        combined = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        logits = self.fc_out(combined)                # (batch, vocab_size)

        return logits, (h_t, c_t), attn_weights


# =====================================================================
# 4) اختبار سريع للأشكال (Sanity Check)
# =====================================================================
def sanity_check() -> None:
    batch, src_len, src_vocab, tgt_vocab = 4, 12, 3000, 3500
    embedding_dim, hidden_size = 128, 256

    encoder = EncoderLSTM(src_vocab, embedding_dim, hidden_size)
    decoder = DecoderWithAttention(tgt_vocab, embedding_dim, hidden_size, hidden_size * 2)

    src = torch.randint(1, src_vocab, (batch, src_len))
    encoder_outputs, (h0, c0) = encoder(src)
    print(f"encoder_outputs: {encoder_outputs.shape}")   # (4, 12, 512)

    first_token = torch.randint(1, tgt_vocab, (batch, 1))
    logits, (h1, c1), attn_weights = decoder.forward_step(first_token, (h0, c0), encoder_outputs)
    print(f"logits: {logits.shape}")           # (4, 3500)
    print(f"attn_weights: {attn_weights.shape}")  # (4, 12) -- توزيع احتمالي على كلمات المصدر!
    print(f"مجموع أوزان الانتباه (يجب أن يساوي 1 تقريبًا): {attn_weights[0].sum().item():.4f}")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# حالات مُرمِّز مُبسَّطة جدًا (3 كلمات، أبعاد=4 للتوضيح فقط)
h_encoder = torch.tensor([
    [1.0, 0.2, 0.1, 0.0],   # h1: "القطة"
    [0.1, 1.0, 0.3, 0.2],   # h2: "تأكل"
    [0.0, 0.1, 1.0, 0.1],   # h3: "السمك"
])

# استعلام فك المُرمِّز عند توليد "cat" -- قريب جدًا من h1 عمدًا للتوضيح
s_query_step1 = torch.tensor([0.9, 0.3, 0.05, 0.1])

scores = h_encoder @ s_query_step1  # ضرب نقطي بسيط (أسلوب Luong)
attn_weights = F.softmax(scores, dim=0)

print("الدرجات الخام:", scores)
print("أوزان الانتباه بعد Softmax:", attn_weights)
print("مجموع الأوزان (يجب = 1):", attn_weights.sum().item())

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
scores = h_encoder @ query  # ❌ يشمل درجات لرموز حشو وهمية!
attn_weights = F.softmax(scores, dim=0)  # سيُوزّع احتمالًا حقيقيًا على الحشو

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
scores = scores.masked_fill(padding_mask == 0, float('-inf'))
attn_weights = F.softmax(scores, dim=0)  # ✅ الحشو يحصل على احتمال ≈ 0

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# فكر خاطئ: "استخدمت Attention، إذن أصبح نموذجي سريعًا ومتوازيًا تمامًا"

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
"""
translation_with_attention_visualization.py
مشروع تطبيقي: ترجمة آلية صغيرة مع تصوير خرائط الانتباه.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def translate_and_visualize(
    sentence: str,
    encoder: "EncoderLSTM",
    decoder: "DecoderWithAttention",
    src_vocab: dict[str, int],
    tgt_vocab: dict[int, str],
    max_length: int = 20,
) -> tuple[str, np.ndarray]:
    """
    يترجم جملة واحدة بأسلوب Greedy Decoding،
    ويُعيد الترجمة النصية + مصفوفة أوزان الانتباه الكاملة (target_len, source_len)
    """
    # TODO 1: رمّز الجملة المصدر وحوّلها لـ Tensor
    # TODO 2: مرّرها عبر الـ Encoder للحصول على encoder_outputs والحالة الابتدائية
    # TODO 3: حلقة توليد Greedy: ابدأ برمز <BOS>، استدعِ decoder.forward_step() تكراريًا
    #         حتى الوصول لرمز <EOS> أو max_length، واحفظ attn_weights في كل خطوة
    raise NotImplementedError


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path: str = "attention_heatmap.png",
) -> None:
    """يرسم خريطة حرارية: محور أفقي=كلمات المصدر، محور رأسي=كلمات الهدف المُولَّدة"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap="viridis")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens)

    plt.colorbar(im)
    plt.xlabel("الجملة المصدر")
    plt.ylabel("الجملة المُترجَمة")
    plt.title("خريطة أوزان الانتباه (Attention Heatmap)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"تم حفظ الخريطة في: {output_path}")

# تحدٍّ إضافي: قارن خريطة الانتباه لجملة بترتيب كلمات متشابه بين
# العربية والإنجليزية، مقابل جملة بترتيب مختلف جوهريًا (مثل جملة تحتوي
# صفة متأخرة في إحدى اللغتين ومتقدمة في الأخرى)، وناقش الفروقات.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — الدافع وراء Self-Attention
Large Language Model Engineering — Self-Attention

Book slug : llm-engineering-69-chapters
Chapter   : ch-09-chapter-09 (lesson 9)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_09.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 09: الفصل التاسع: الدافع وراء Self-Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 09 lab: Alignment attention over encoder states")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
"""
seq2seq_bahdanau_attention.py
نموذج Seq2Seq مع Bahdanau (Additive) Attention من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


# =====================================================================
# 1) المُرمِّز (Encoder): LSTM ثنائي الاتجاه، يُحتفَظ بكل الحالات المخفية
# =====================================================================
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        # دمج الاتجاهين إلى hidden_size واحد كي يتوافق مع فك المُرمِّز أحادي الاتجاه
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        src: (batch, src_len)
        يُعيد:
          encoder_outputs: (batch, src_len, hidden_size*2) -- كل الحالات! هذا هو المفتاح
          (h_0_dec, c_0_dec): الحالة الابتدائية لفك المُرمِّز
        """
        embedded = self.embedding(src)                       # (batch, src_len, emb_dim)
        outputs, (h_n, c_n) = self.lstm(embedded)             # outputs: (batch, src_len, 2*hidden)

        # h_n شكلها (2, batch, hidden) لأن ثنائي الاتجاه (forward + backward)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)        # (batch, 2*hidden)
        c_combined = torch.cat([c_n[0], c_n[1]], dim=1)

        h_0_dec = torch.tanh(self.fc_hidden(h_combined)).unsqueeze(0)  # (1, batch, hidden)
        c_0_dec = torch.tanh(self.fc_cell(c_combined)).unsqueeze(0)

        return outputs, (h_0_dec, c_0_dec)


# =====================================================================
# 2) آلية Bahdanau Attention (Additive Scoring)
# =====================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_size: int, encoder_output_size: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.W_a = nn.Linear(decoder_hidden_size, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_output_size, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        decoder_hidden:  (batch, decoder_hidden_size)         -- s_{t-1}، استعلام اللحظة الحالية
        encoder_outputs: (batch, src_len, encoder_output_size) -- h_1...h_{T_x}
        يُعيد:
          context: (batch, encoder_output_size)   -- c_t، متجه السياق المُحسوب ديناميكيًا
          attn_weights: (batch, src_len)           -- α_{t,i}، أوزان الانتباه (للتصوّر والتفسير!)
        """
        src_len = encoder_outputs.size(1)

        # وسّع الحالة المخفية لتتوافق مع كل خطوة زمنية في المُرمِّز
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_hidden_size)

        # e_{t,i} = v_a^T tanh(W_a s_{t-1} + U_a h_i)
        energy = torch.tanh(
            self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs)
        )                                                    # (batch, src_len, attn_dim)
        scores = self.v_a(energy).squeeze(2)                  # (batch, src_len)

        attn_weights = F.softmax(scores, dim=1)               # α_{t,i} -- يجمع 1 عبر src_len
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # (batch, 1, src_len) @ (batch, src_len, enc_size) = (batch, 1, enc_size) → (batch, enc_size)

        return context, attn_weights


# =====================================================================
# 3) فك المُرمِّز (Decoder) الذي يستخدم Attention في كل خطوة توليد
# =====================================================================
class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_output_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size, encoder_output_size)
        self.lstm = nn.LSTM(
            embedding_dim + encoder_output_size, hidden_size, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size + encoder_output_size + embedding_dim, vocab_size)

    def forward_step(
        self,
        input_token: Tensor,                       # (batch, 1) -- كلمة واحدة فقط
        state: tuple[Tensor, Tensor],               # (h_{t-1}, c_{t-1})
        encoder_outputs: Tensor,                    # (batch, src_len, encoder_output_size)
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        h_prev, c_prev = state
        embedded = self.embedding(input_token)       # (batch, 1, emb_dim)

        # نحسب الانتباه باستخدام آخر حالة مخفية لفك المُرمِّز كاستعلام
        context, attn_weights = self.attention(h_prev.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)               # (batch, 1, encoder_output_size)

        lstm_input = torch.cat([embedded, context], dim=2)   # دمج المدخل الحالي مع السياق الديناميكي
        output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))

        combined = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        logits = self.fc_out(combined)                # (batch, vocab_size)

        return logits, (h_t, c_t), attn_weights


# =====================================================================
# 4) اختبار سريع للأشكال (Sanity Check)
# =====================================================================
def sanity_check() -> None:
    batch, src_len, src_vocab, tgt_vocab = 4, 12, 3000, 3500
    embedding_dim, hidden_size = 128, 256

    encoder = EncoderLSTM(src_vocab, embedding_dim, hidden_size)
    decoder = DecoderWithAttention(tgt_vocab, embedding_dim, hidden_size, hidden_size * 2)

    src = torch.randint(1, src_vocab, (batch, src_len))
    encoder_outputs, (h0, c0) = encoder(src)
    print(f"encoder_outputs: {encoder_outputs.shape}")   # (4, 12, 512)

    first_token = torch.randint(1, tgt_vocab, (batch, 1))
    logits, (h1, c1), attn_weights = decoder.forward_step(first_token, (h0, c0), encoder_outputs)
    print(f"logits: {logits.shape}")           # (4, 3500)
    print(f"attn_weights: {attn_weights.shape}")  # (4, 12) -- توزيع احتمالي على كلمات المصدر!
    print(f"مجموع أوزان الانتباه (يجب أن يساوي 1 تقريبًا): {attn_weights[0].sum().item():.4f}")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# حالات مُرمِّز مُبسَّطة جدًا (3 كلمات، أبعاد=4 للتوضيح فقط)
h_encoder = torch.tensor([
    [1.0, 0.2, 0.1, 0.0],   # h1: "القطة"
    [0.1, 1.0, 0.3, 0.2],   # h2: "تأكل"
    [0.0, 0.1, 1.0, 0.1],   # h3: "السمك"
])

# استعلام فك المُرمِّز عند توليد "cat" -- قريب جدًا من h1 عمدًا للتوضيح
s_query_step1 = torch.tensor([0.9, 0.3, 0.05, 0.1])

scores = h_encoder @ s_query_step1  # ضرب نقطي بسيط (أسلوب Luong)
attn_weights = F.softmax(scores, dim=0)

print("الدرجات الخام:", scores)
print("أوزان الانتباه بعد Softmax:", attn_weights)
print("مجموع الأوزان (يجب = 1):", attn_weights.sum().item())

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
scores = h_encoder @ query  # ❌ يشمل درجات لرموز حشو وهمية!
attn_weights = F.softmax(scores, dim=0)  # سيُوزّع احتمالًا حقيقيًا على الحشو

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
scores = scores.masked_fill(padding_mask == 0, float('-inf'))
attn_weights = F.softmax(scores, dim=0)  # ✅ الحشو يحصل على احتمال ≈ 0

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
# فكر خاطئ: "استخدمت Attention، إذن أصبح نموذجي سريعًا ومتوازيًا تمامًا"

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-03-transformers-core/chapter-09.md
# ────────────────────────────────────────────────────────────────────────
"""
translation_with_attention_visualization.py
مشروع تطبيقي: ترجمة آلية صغيرة مع تصوير خرائط الانتباه.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def translate_and_visualize(
    sentence: str,
    encoder: "EncoderLSTM",
    decoder: "DecoderWithAttention",
    src_vocab: dict[str, int],
    tgt_vocab: dict[int, str],
    max_length: int = 20,
) -> tuple[str, np.ndarray]:
    """
    يترجم جملة واحدة بأسلوب Greedy Decoding،
    ويُعيد الترجمة النصية + مصفوفة أوزان الانتباه الكاملة (target_len, source_len)
    """
    # TODO 1: رمّز الجملة المصدر وحوّلها لـ Tensor
    # TODO 2: مرّرها عبر الـ Encoder للحصول على encoder_outputs والحالة الابتدائية
    # TODO 3: حلقة توليد Greedy: ابدأ برمز <BOS>، استدعِ decoder.forward_step() تكراريًا
    #         حتى الوصول لرمز <EOS> أو max_length، واحفظ attn_weights في كل خطوة
    raise NotImplementedError


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path: str = "attention_heatmap.png",
) -> None:
    """يرسم خريطة حرارية: محور أفقي=كلمات المصدر، محور رأسي=كلمات الهدف المُولَّدة"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap="viridis")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens)

    plt.colorbar(im)
    plt.xlabel("الجملة المصدر")
    plt.ylabel("الجملة المُترجَمة")
    plt.title("خريطة أوزان الانتباه (Attention Heatmap)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"تم حفظ الخريطة في: {output_path}")

# تحدٍّ إضافي: قارن خريطة الانتباه لجملة بترتيب كلمات متشابه بين
# العربية والإنجليزية، مقابل جملة بترتيب مختلف جوهريًا (مثل جملة تحتوي
# صفة متأخرة في إحدى اللغتين ومتقدمة في الأخرى)، وناقش الفروقات.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 09: الفصل التاسع: الدافع وراء Self-Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 09 lab: Alignment attention over encoder states")
    run_alignment()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
seq2seq_bahdanau_attention.py
نموذج Seq2Seq مع Bahdanau (Additive) Attention من الصفر.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


# =====================================================================
# 1) المُرمِّز (Encoder): LSTM ثنائي الاتجاه، يُحتفَظ بكل الحالات المخفية
# =====================================================================
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )
        # دمج الاتجاهين إلى hidden_size واحد كي يتوافق مع فك المُرمِّز أحادي الاتجاه
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        src: (batch, src_len)
        يُعيد:
          encoder_outputs: (batch, src_len, hidden_size*2) -- كل الحالات! هذا هو المفتاح
          (h_0_dec, c_0_dec): الحالة الابتدائية لفك المُرمِّز
        """
        embedded = self.embedding(src)                       # (batch, src_len, emb_dim)
        outputs, (h_n, c_n) = self.lstm(embedded)             # outputs: (batch, src_len, 2*hidden)

        # h_n شكلها (2, batch, hidden) لأن ثنائي الاتجاه (forward + backward)
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1)        # (batch, 2*hidden)
        c_combined = torch.cat([c_n[0], c_n[1]], dim=1)

        h_0_dec = torch.tanh(self.fc_hidden(h_combined)).unsqueeze(0)  # (1, batch, hidden)
        c_0_dec = torch.tanh(self.fc_cell(c_combined)).unsqueeze(0)

        return outputs, (h_0_dec, c_0_dec)


# =====================================================================
# 2) آلية Bahdanau Attention (Additive Scoring)
# =====================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_size: int, encoder_output_size: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.W_a = nn.Linear(decoder_hidden_size, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_output_size, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        decoder_hidden:  (batch, decoder_hidden_size)         -- s_{t-1}، استعلام اللحظة الحالية
        encoder_outputs: (batch, src_len, encoder_output_size) -- h_1...h_{T_x}
        يُعيد:
          context: (batch, encoder_output_size)   -- c_t، متجه السياق المُحسوب ديناميكيًا
          attn_weights: (batch, src_len)           -- α_{t,i}، أوزان الانتباه (للتصوّر والتفسير!)
        """
        src_len = encoder_outputs.size(1)

        # وسّع الحالة المخفية لتتوافق مع كل خطوة زمنية في المُرمِّز
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_hidden_size)

        # e_{t,i} = v_a^T tanh(W_a s_{t-1} + U_a h_i)
        energy = torch.tanh(
            self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs)
        )                                                    # (batch, src_len, attn_dim)
        scores = self.v_a(energy).squeeze(2)                  # (batch, src_len)

        attn_weights = F.softmax(scores, dim=1)               # α_{t,i} -- يجمع 1 عبر src_len
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # (batch, 1, src_len) @ (batch, src_len, enc_size) = (batch, 1, enc_size) → (batch, enc_size)

        return context, attn_weights


# =====================================================================
# 3) فك المُرمِّز (Decoder) الذي يستخدم Attention في كل خطوة توليد
# =====================================================================
class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        encoder_output_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size, encoder_output_size)
        self.lstm = nn.LSTM(
            embedding_dim + encoder_output_size, hidden_size, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size + encoder_output_size + embedding_dim, vocab_size)

    def forward_step(
        self,
        input_token: Tensor,                       # (batch, 1) -- كلمة واحدة فقط
        state: tuple[Tensor, Tensor],               # (h_{t-1}, c_{t-1})
        encoder_outputs: Tensor,                    # (batch, src_len, encoder_output_size)
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        h_prev, c_prev = state
        embedded = self.embedding(input_token)       # (batch, 1, emb_dim)

        # نحسب الانتباه باستخدام آخر حالة مخفية لفك المُرمِّز كاستعلام
        context, attn_weights = self.attention(h_prev.squeeze(0), encoder_outputs)
        context = context.unsqueeze(1)               # (batch, 1, encoder_output_size)

        lstm_input = torch.cat([embedded, context], dim=2)   # دمج المدخل الحالي مع السياق الديناميكي
        output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))

        combined = torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1)
        logits = self.fc_out(combined)                # (batch, vocab_size)

        return logits, (h_t, c_t), attn_weights


# =====================================================================
# 4) اختبار سريع للأشكال (Sanity Check)
# =====================================================================
def sanity_check() -> None:
    batch, src_len, src_vocab, tgt_vocab = 4, 12, 3000, 3500
    embedding_dim, hidden_size = 128, 256

    encoder = EncoderLSTM(src_vocab, embedding_dim, hidden_size)
    decoder = DecoderWithAttention(tgt_vocab, embedding_dim, hidden_size, hidden_size * 2)

    src = torch.randint(1, src_vocab, (batch, src_len))
    encoder_outputs, (h0, c0) = encoder(src)
    print(f"encoder_outputs: {encoder_outputs.shape}")   # (4, 12, 512)

    first_token = torch.randint(1, tgt_vocab, (batch, 1))
    logits, (h1, c1), attn_weights = decoder.forward_step(first_token, (h0, c0), encoder_outputs)
    print(f"logits: {logits.shape}")           # (4, 3500)
    print(f"attn_weights: {attn_weights.shape}")  # (4, 12) -- توزيع احتمالي على كلمات المصدر!
    print(f"مجموع أوزان الانتباه (يجب أن يساوي 1 تقريبًا): {attn_weights[0].sum().item():.4f}")


if __name__ == "__main__":
    sanity_check()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# حالات مُرمِّز مُبسَّطة جدًا (3 كلمات، أبعاد=4 للتوضيح فقط)
h_encoder = torch.tensor([
    [1.0, 0.2, 0.1, 0.0],   # h1: "القطة"
    [0.1, 1.0, 0.3, 0.2],   # h2: "تأكل"
    [0.0, 0.1, 1.0, 0.1],   # h3: "السمك"
])

# استعلام فك المُرمِّز عند توليد "cat" -- قريب جدًا من h1 عمدًا للتوضيح
s_query_step1 = torch.tensor([0.9, 0.3, 0.05, 0.1])

scores = h_encoder @ s_query_step1  # ضرب نقطي بسيط (أسلوب Luong)
attn_weights = F.softmax(scores, dim=0)

print("الدرجات الخام:", scores)
print("أوزان الانتباه بعد Softmax:", attn_weights)
print("مجموع الأوزان (يجب = 1):", attn_weights.sum().item())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
scores = h_encoder @ query  # ❌ يشمل درجات لرموز حشو وهمية!
attn_weights = F.softmax(scores, dim=0)  # سيُوزّع احتمالًا حقيقيًا على الحشو

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
scores = scores.masked_fill(padding_mask == 0, float('-inf'))
attn_weights = F.softmax(scores, dim=0)  # ✅ الحشو يحصل على احتمال ≈ 0

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
# فكر خاطئ: "استخدمت Attention، إذن أصبح نموذجي سريعًا ومتوازيًا تمامًا"

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-09-chapter-09.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
"""
translation_with_attention_visualization.py
مشروع تطبيقي: ترجمة آلية صغيرة مع تصوير خرائط الانتباه.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def translate_and_visualize(
    sentence: str,
    encoder: "EncoderLSTM",
    decoder: "DecoderWithAttention",
    src_vocab: dict[str, int],
    tgt_vocab: dict[int, str],
    max_length: int = 20,
) -> tuple[str, np.ndarray]:
    """
    يترجم جملة واحدة بأسلوب Greedy Decoding،
    ويُعيد الترجمة النصية + مصفوفة أوزان الانتباه الكاملة (target_len, source_len)
    """
    # TODO 1: رمّز الجملة المصدر وحوّلها لـ Tensor
    # TODO 2: مرّرها عبر الـ Encoder للحصول على encoder_outputs والحالة الابتدائية
    # TODO 3: حلقة توليد Greedy: ابدأ برمز <BOS>، استدعِ decoder.forward_step() تكراريًا
    #         حتى الوصول لرمز <EOS> أو max_length، واحفظ attn_weights في كل خطوة
    raise NotImplementedError


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path: str = "attention_heatmap.png",
) -> None:
    """يرسم خريطة حرارية: محور أفقي=كلمات المصدر، محور رأسي=كلمات الهدف المُولَّدة"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap="viridis")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens)

    plt.colorbar(im)
    plt.xlabel("الجملة المصدر")
    plt.ylabel("الجملة المُترجَمة")
    plt.title("خريطة أوزان الانتباه (Attention Heatmap)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"تم حفظ الخريطة في: {output_path}")

# تحدٍّ إضافي: قارن خريطة الانتباه لجملة بترتيب كلمات متشابه بين
# العربية والإنجليزية، مقابل جملة بترتيب مختلف جوهريًا (مثل جملة تحتوي
# صفة متأخرة في إحدى اللغتين ومتقدمة في الأخرى)، وناقش الفروقات.
