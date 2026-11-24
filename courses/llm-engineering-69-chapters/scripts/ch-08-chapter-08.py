#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من الكلمة إلى الجملة — RNN وLSTM وGRU
Large Language Model Engineering — RNN

Book slug : llm-engineering-69-chapters
Chapter   : ch-08-chapter-08 (lesson 8)
Code files: 12 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-08-chapter-08.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من الكلمة إلى الجملة — RNN وLSTM وGRU
Large Language Model Engineering — RNN

Book slug : llm-engineering-69-chapters
Chapter   : ch-08-chapter-08 (lesson 8)
Code files: 11 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-08-chapter-08.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من الكلمة إلى الجملة — RNN وLSTM وGRU
Large Language Model Engineering — RNN

Book slug : llm-engineering-69-chapters
Chapter   : ch-08-chapter-08 (lesson 8)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 08: الفصل الثامن: من الكلمة إلى الجملة — RNN وLSTM وGRU."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_recurrent() -> None:
    inputs = [0.2, 0.8, -0.1, 0.5]
    state, gate = (0.0, 0.65)
    states = []
    for value in inputs:
        candidate = math.tanh(0.7 * state + 0.8 * value)
        state = gate * state + (1.0 - gate) * candidate
        states.append(round(state, 4))
    print('input sequence:', inputs)
    print('gated hidden states:', states)

if __name__ == "__main__":
    print("Chapter 08 lab: A recurrent state with a learned gate")
    run_recurrent()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
"""
rnn_lstm_from_scratch.py
بناء خلايا RNN وLSTM من الصفر لفهم الآلية الداخلية بعمق،
ثم مقارنتها بالوحدات المدمجة في PyTorch.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) خلية RNN بسيطة (Vanilla RNN) مبنية يدويًا
# =====================================================================
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        يُعيد: h_t (batch, hidden_size)
        """
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


class VanillaRNN(nn.Module):
    """يُطبّق الخلية عبر كامل التسلسل الزمني (Unrolling يدوي)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaRNNCell(input_size, hidden_size)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        x: (batch, seq_len, input_size)
        يُعيد: (كل الحالات المخفية (batch, seq_len, hidden_size), آخر حالة مخفية)
        """
        batch_size, seq_len, _ = x.shape
        h_t = h0 if h0 is not None else torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t


# =====================================================================
# 2) خلية LSTM مبنية يدويًا بكل البوابات الأربع
# =====================================================================
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # نُجمّع كل البوابات في مصفوفة واحدة كبيرة لكفاءة حسابية أعلى
        # (بدلًا من 4 طبقات Linear منفصلة -- حيلة شائعة في التطبيقات الفعلية)
        self.gates = nn.Linear(combined_size, 4 * hidden_size)

    def forward(
        self, x_t: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        x_t: (batch, input_size)
        state: (h_prev, C_prev), كل منهما (batch, hidden_size)
        يُعيد: (h_t, C_t)
        """
        h_prev, c_prev = state
        combined = torch.cat([x_t, h_prev], dim=1)

        gates = self.gates(combined)  # (batch, 4*hidden_size)
        f_gate, i_gate, g_candidate, o_gate = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_gate)        # بوابة النسيان
        i_t = torch.sigmoid(i_gate)        # بوابة الإدخال
        c_tilde = torch.tanh(g_candidate)  # المرشح الجديد
        o_t = torch.sigmoid(o_gate)        # بوابة الإخراج

        c_t = f_t * c_prev + i_t * c_tilde     # تحديث حالة الخلية (المسار الخطي الحاسم!)
        h_t = o_t * torch.tanh(c_t)             # الحالة المخفية الجديدة

        return h_t, c_t


class LSTMFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCellFromScratch(input_size, hidden_size)

    def forward(
        self, x: Tensor, init_state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        if init_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_state

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t, c_t)


# =====================================================================
# 3) التحقق العددي: هل تطبيقنا اليدوي يُطابق سلوك PyTorch المدمج منطقيًا؟
# =====================================================================
def sanity_check_shapes() -> None:
    batch, seq_len, input_size, hidden_size = 4, 10, 16, 32

    x = torch.randn(batch, seq_len, input_size)

    my_rnn = VanillaRNN(input_size, hidden_size)
    outputs, h_last = my_rnn(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    assert h_last.shape == (batch, hidden_size)
    print(f"✅ VanillaRNN من الصفر: outputs={outputs.shape}, h_last={h_last.shape}")

    my_lstm = LSTMFromScratch(input_size, hidden_size)
    outputs, (h_last, c_last) = my_lstm(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    print(f"✅ LSTM من الصفر: outputs={outputs.shape}, h_last={h_last.shape}, c_last={c_last.shape}")

    # الوحدة المدمجة الرسمية في PyTorch -- للاستخدام الفعلي في الإنتاج
    official_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    outputs_official, (h_n, c_n) = official_lstm(x)
    assert outputs_official.shape == (batch, seq_len, hidden_size)
    print(f"✅ nn.LSTM الرسمية: outputs={outputs_official.shape}")


if __name__ == "__main__":
    sanity_check_shapes()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
"""
lstm_sentiment_classifier.py
مصنّف مشاعر كامل باستخدام Bidirectional Multi-layer LSTM.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(42)


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,       # مثال: سلبي / محايد / إيجابي
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # يقرأ الجملة من اليمين لليسار ومن اليسار لليمين معًا
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout بين الطبقات فقط
        )

        # ثنائي الاتجاه ⟹ حجم الإخراج = hidden_size * 2 (دمج الاتجاهين)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """
        input_ids: (batch, seq_len) -- مُرمّزة ومُحشوّة (padded)
        lengths:   (batch,) -- الطول الحقيقي لكل جملة قبل الحشو
        يُعيد: logits (batch, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # حيلة حاسمة للأداء والصحة: تجاهل خطوات الحشو (padding) في حساب LSTM
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.lstm(packed)
        # لا نحتاج فك الحزمة هنا لأننا سنستخدم h_n مباشرة

        # h_n شكلها: (num_layers * num_directions, batch, hidden_size)
        # نأخذ آخر طبقة فقط، لكلا الاتجاهين، وندمجهما
        forward_last = h_n[-2, :, :]   # آخر حالة من الاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة من الاتجاه الخلفي
        final_hidden = torch.cat([forward_last, backward_last], dim=1)

        return self.classifier(final_hidden)


def demo_training_step() -> None:
    """مثال توضيحي لخطوة تدريب واحدة كاملة."""
    vocab_size, batch_size, max_seq_len = 5000, 8, 20

    model = BiLSTMSentimentClassifier(vocab_size=vocab_size, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # بيانات وهمية للتوضيح: جمل بأطوال مختلفة ومُحشوّة بـ 0
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_len))
    lengths = torch.randint(5, max_seq_len + 1, (batch_size,))
    labels = torch.randint(0, 3, (batch_size,))

    optimizer.zero_grad()
    logits = model(input_ids, lengths)
    loss = criterion(logits, labels)
    loss.backward()

    # حيلة لا غنى عنها مع RNN/LSTM: Gradient Clipping لمنع الانفجار
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()
    print(f"Loss: {loss.item():.4f} | logits shape: {logits.shape}")


if __name__ == "__main__":
    demo_training_step()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
import torch

h0 = torch.zeros(1, 2)
c0 = torch.zeros(1, 2)

f1 = torch.sigmoid(torch.tensor([[2.0, -1.0]]))
i1 = torch.sigmoid(torch.tensor([[0.5, 1.5]]))
c_tilde1 = torch.tanh(torch.tensor([[0.8, -0.3]]))
o1 = torch.sigmoid(torch.tensor([[1.0, 1.0]]))

c1 = f1 * c0 + i1 * c_tilde1
h1 = o1 * torch.tanh(c1)

print(f"f1={f1}")           # tensor([[0.8808, 0.2689]])
print(f"i1={i1}")           # tensor([[0.6225, 0.8176]])
print(f"c_tilde1={c_tilde1}") # tensor([[0.6640, -0.2913]])
print(f"C1={c1}")           # tensor([[0.4133, -0.2382]])
print(f"h1={h1}")           # tensor([[0.2855, -0.1717]])

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256)  # ❌ الافتراضي: (seq_len, batch, features)!
x = torch.randn(32, 50, 128)  # تتوقع (batch, seq_len, features) لكن هذا ليس الافتراضي
output, _ = lstm(x)  # سيعمل بدون خطأ لكن بترتيب أبعاد خاطئ تمامًا! خطأ صامت خطير

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
classifier = nn.Linear(256, 10)  # ❌ خطأ في الأبعاد! الإخراج الفعلي حجمه 512 وليس 256

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
classifier = nn.Linear(256 * 2, 10)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
outputs, (h_n, c_n) = lstm(x)
# إذا احتجت الانتباه (Attention) على كل كلمة، استخدام h_n فقط خطأ شائع
result = classifier(h_n[-1])  # ❌ إذا كنت تحتاج معلومات من كل الخطوات الزمنية، فقدتها هنا

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
"""
char_level_lstm_generator.py
نموذج لغوي على مستوى الحرف لتوليد نص عربي باستخدام LSTM.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


class CharLSTM(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 256, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.embedding(x)
        output, new_state = self.lstm(embedded, state)
        logits = self.fc(output)
        return logits, new_state


# TODO 1: حمّل نصًا عربيًا طويلًا (رواية، مجموعة مقالات) وابنِ قاموس حرف↔رقم
def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    raise NotImplementedError


# TODO 2: قسّم النص إلى تسلسلات تدريب بطول ثابت (مثلًا 100 حرف) بأسلوب Sliding Window
def create_training_sequences(
    encoded_text: list[int], seq_length: int = 100
) -> list[tuple[list[int], list[int]]]:
    """كل عيّنة: (تسلسل مدخل، نفس التسلسل مُزاح بخطوة واحدة = الهدف)"""
    raise NotImplementedError


# TODO 3: اكتب حلقة تدريب كاملة (استلهم من full_training_loop.py في الفصل 3)


# TODO 4: اكتب دالة التوليد بأسلوب Autoregressive مع Temperature Sampling
@torch.no_grad()
def generate_text(
    model: CharLSTM,
    start_text: str,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
    length: int = 300,
    temperature: float = 0.8,
) -> str:
    """
    يبدأ من start_text، ويُولّد حرفًا تلو الآخر، معيدًا تغذية الحرف المُولَّد
    كمدخل للخطوة التالية (Autoregressive)، مع التحكم بالعشوائية عبر temperature.
    """
    raise NotImplementedError

# تحدٍّ إضافي: قارن جودة النص المُولَّد بين temperature=0.3 (محافظ)
# و temperature=1.2 (جريء وعشوائي أكثر)، وناقش المقايضة بينهما.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 08: الفصل الثامن: من الكلمة إلى الجملة — RNN وLSTM وGRU."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_recurrent() -> None:
    inputs = [0.2, 0.8, -0.1, 0.5]
    state, gate = (0.0, 0.65)
    states = []
    for value in inputs:
        candidate = math.tanh(0.7 * state + 0.8 * value)
        state = gate * state + (1.0 - gate) * candidate
        states.append(round(state, 4))
    print('input sequence:', inputs)
    print('gated hidden states:', states)

if __name__ == "__main__":
    print("Chapter 08 lab: A recurrent state with a learned gate")
    run_recurrent()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
rnn_lstm_from_scratch.py
بناء خلايا RNN وLSTM من الصفر لفهم الآلية الداخلية بعمق،
ثم مقارنتها بالوحدات المدمجة في PyTorch.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) خلية RNN بسيطة (Vanilla RNN) مبنية يدويًا
# =====================================================================
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        يُعيد: h_t (batch, hidden_size)
        """
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


class VanillaRNN(nn.Module):
    """يُطبّق الخلية عبر كامل التسلسل الزمني (Unrolling يدوي)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaRNNCell(input_size, hidden_size)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        x: (batch, seq_len, input_size)
        يُعيد: (كل الحالات المخفية (batch, seq_len, hidden_size), آخر حالة مخفية)
        """
        batch_size, seq_len, _ = x.shape
        h_t = h0 if h0 is not None else torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t


# =====================================================================
# 2) خلية LSTM مبنية يدويًا بكل البوابات الأربع
# =====================================================================
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # نُجمّع كل البوابات في مصفوفة واحدة كبيرة لكفاءة حسابية أعلى
        # (بدلًا من 4 طبقات Linear منفصلة -- حيلة شائعة في التطبيقات الفعلية)
        self.gates = nn.Linear(combined_size, 4 * hidden_size)

    def forward(
        self, x_t: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        x_t: (batch, input_size)
        state: (h_prev, C_prev), كل منهما (batch, hidden_size)
        يُعيد: (h_t, C_t)
        """
        h_prev, c_prev = state
        combined = torch.cat([x_t, h_prev], dim=1)

        gates = self.gates(combined)  # (batch, 4*hidden_size)
        f_gate, i_gate, g_candidate, o_gate = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_gate)        # بوابة النسيان
        i_t = torch.sigmoid(i_gate)        # بوابة الإدخال
        c_tilde = torch.tanh(g_candidate)  # المرشح الجديد
        o_t = torch.sigmoid(o_gate)        # بوابة الإخراج

        c_t = f_t * c_prev + i_t * c_tilde     # تحديث حالة الخلية (المسار الخطي الحاسم!)
        h_t = o_t * torch.tanh(c_t)             # الحالة المخفية الجديدة

        return h_t, c_t


class LSTMFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCellFromScratch(input_size, hidden_size)

    def forward(
        self, x: Tensor, init_state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        if init_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_state

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t, c_t)


# =====================================================================
# 3) التحقق العددي: هل تطبيقنا اليدوي يُطابق سلوك PyTorch المدمج منطقيًا؟
# =====================================================================
def sanity_check_shapes() -> None:
    batch, seq_len, input_size, hidden_size = 4, 10, 16, 32

    x = torch.randn(batch, seq_len, input_size)

    my_rnn = VanillaRNN(input_size, hidden_size)
    outputs, h_last = my_rnn(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    assert h_last.shape == (batch, hidden_size)
    print(f"✅ VanillaRNN من الصفر: outputs={outputs.shape}, h_last={h_last.shape}")

    my_lstm = LSTMFromScratch(input_size, hidden_size)
    outputs, (h_last, c_last) = my_lstm(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    print(f"✅ LSTM من الصفر: outputs={outputs.shape}, h_last={h_last.shape}, c_last={c_last.shape}")

    # الوحدة المدمجة الرسمية في PyTorch -- للاستخدام الفعلي في الإنتاج
    official_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    outputs_official, (h_n, c_n) = official_lstm(x)
    assert outputs_official.shape == (batch, seq_len, hidden_size)
    print(f"✅ nn.LSTM الرسمية: outputs={outputs_official.shape}")


if __name__ == "__main__":
    sanity_check_shapes()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
lstm_sentiment_classifier.py
مصنّف مشاعر كامل باستخدام Bidirectional Multi-layer LSTM.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(42)


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,       # مثال: سلبي / محايد / إيجابي
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # يقرأ الجملة من اليمين لليسار ومن اليسار لليمين معًا
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout بين الطبقات فقط
        )

        # ثنائي الاتجاه ⟹ حجم الإخراج = hidden_size * 2 (دمج الاتجاهين)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """
        input_ids: (batch, seq_len) -- مُرمّزة ومُحشوّة (padded)
        lengths:   (batch,) -- الطول الحقيقي لكل جملة قبل الحشو
        يُعيد: logits (batch, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # حيلة حاسمة للأداء والصحة: تجاهل خطوات الحشو (padding) في حساب LSTM
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.lstm(packed)
        # لا نحتاج فك الحزمة هنا لأننا سنستخدم h_n مباشرة

        # h_n شكلها: (num_layers * num_directions, batch, hidden_size)
        # نأخذ آخر طبقة فقط، لكلا الاتجاهين، وندمجهما
        forward_last = h_n[-2, :, :]   # آخر حالة من الاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة من الاتجاه الخلفي
        final_hidden = torch.cat([forward_last, backward_last], dim=1)

        return self.classifier(final_hidden)


def demo_training_step() -> None:
    """مثال توضيحي لخطوة تدريب واحدة كاملة."""
    vocab_size, batch_size, max_seq_len = 5000, 8, 20

    model = BiLSTMSentimentClassifier(vocab_size=vocab_size, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # بيانات وهمية للتوضيح: جمل بأطوال مختلفة ومُحشوّة بـ 0
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_len))
    lengths = torch.randint(5, max_seq_len + 1, (batch_size,))
    labels = torch.randint(0, 3, (batch_size,))

    optimizer.zero_grad()
    logits = model(input_ids, lengths)
    loss = criterion(logits, labels)
    loss.backward()

    # حيلة لا غنى عنها مع RNN/LSTM: Gradient Clipping لمنع الانفجار
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()
    print(f"Loss: {loss.item():.4f} | logits shape: {logits.shape}")


if __name__ == "__main__":
    demo_training_step()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch

h0 = torch.zeros(1, 2)
c0 = torch.zeros(1, 2)

f1 = torch.sigmoid(torch.tensor([[2.0, -1.0]]))
i1 = torch.sigmoid(torch.tensor([[0.5, 1.5]]))
c_tilde1 = torch.tanh(torch.tensor([[0.8, -0.3]]))
o1 = torch.sigmoid(torch.tensor([[1.0, 1.0]]))

c1 = f1 * c0 + i1 * c_tilde1
h1 = o1 * torch.tanh(c1)

print(f"f1={f1}")           # tensor([[0.8808, 0.2689]])
print(f"i1={i1}")           # tensor([[0.6225, 0.8176]])
print(f"c_tilde1={c_tilde1}") # tensor([[0.6640, -0.2913]])
print(f"C1={c1}")           # tensor([[0.4133, -0.2382]])
print(f"h1={h1}")           # tensor([[0.2855, -0.1717]])

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256)  # ❌ الافتراضي: (seq_len, batch, features)!
x = torch.randn(32, 50, 128)  # تتوقع (batch, seq_len, features) لكن هذا ليس الافتراضي
output, _ = lstm(x)  # سيعمل بدون خطأ لكن بترتيب أبعاد خاطئ تمامًا! خطأ صامت خطير

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
classifier = nn.Linear(256, 10)  # ❌ خطأ في الأبعاد! الإخراج الفعلي حجمه 512 وليس 256

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
classifier = nn.Linear(256 * 2, 10)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
outputs, (h_n, c_n) = lstm(x)
# إذا احتجت الانتباه (Attention) على كل كلمة، استخدام h_n فقط خطأ شائع
result = classifier(h_n[-1])  # ❌ إذا كنت تحتاج معلومات من كل الخطوات الزمنية، فقدتها هنا

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
"""
char_level_lstm_generator.py
نموذج لغوي على مستوى الحرف لتوليد نص عربي باستخدام LSTM.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


class CharLSTM(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 256, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.embedding(x)
        output, new_state = self.lstm(embedded, state)
        logits = self.fc(output)
        return logits, new_state


# TODO 1: حمّل نصًا عربيًا طويلًا (رواية، مجموعة مقالات) وابنِ قاموس حرف↔رقم
def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    raise NotImplementedError


# TODO 2: قسّم النص إلى تسلسلات تدريب بطول ثابت (مثلًا 100 حرف) بأسلوب Sliding Window
def create_training_sequences(
    encoded_text: list[int], seq_length: int = 100
) -> list[tuple[list[int], list[int]]]:
    """كل عيّنة: (تسلسل مدخل، نفس التسلسل مُزاح بخطوة واحدة = الهدف)"""
    raise NotImplementedError


# TODO 3: اكتب حلقة تدريب كاملة (استلهم من full_training_loop.py في الفصل 3)


# TODO 4: اكتب دالة التوليد بأسلوب Autoregressive مع Temperature Sampling
@torch.no_grad()
def generate_text(
    model: CharLSTM,
    start_text: str,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
    length: int = 300,
    temperature: float = 0.8,
) -> str:
    """
    يبدأ من start_text، ويُولّد حرفًا تلو الآخر، معيدًا تغذية الحرف المُولَّد
    كمدخل للخطوة التالية (Autoregressive)، مع التحكم بالعشوائية عبر temperature.
    """
    raise NotImplementedError

# تحدٍّ إضافي: قارن جودة النص المُولَّد بين temperature=0.3 (محافظ)
# و temperature=1.2 (جريء وعشوائي أكثر)، وناقش المقايضة بينهما.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — من الكلمة إلى الجملة — RNN وLSTM وGRU
Large Language Model Engineering — RNN

Book slug : llm-engineering-69-chapters
Chapter   : ch-08-chapter-08 (lesson 8)
Code files: 10 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_08.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 08: الفصل الثامن: من الكلمة إلى الجملة — RNN وLSTM وGRU."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_recurrent() -> None:
    inputs = [0.2, 0.8, -0.1, 0.5]
    state, gate = (0.0, 0.65)
    states = []
    for value in inputs:
        candidate = math.tanh(0.7 * state + 0.8 * value)
        state = gate * state + (1.0 - gate) * candidate
        states.append(round(state, 4))
    print('input sequence:', inputs)
    print('gated hidden states:', states)

if __name__ == "__main__":
    print("Chapter 08 lab: A recurrent state with a learned gate")
    run_recurrent()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
"""
rnn_lstm_from_scratch.py
بناء خلايا RNN وLSTM من الصفر لفهم الآلية الداخلية بعمق،
ثم مقارنتها بالوحدات المدمجة في PyTorch.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) خلية RNN بسيطة (Vanilla RNN) مبنية يدويًا
# =====================================================================
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        يُعيد: h_t (batch, hidden_size)
        """
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


class VanillaRNN(nn.Module):
    """يُطبّق الخلية عبر كامل التسلسل الزمني (Unrolling يدوي)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaRNNCell(input_size, hidden_size)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        x: (batch, seq_len, input_size)
        يُعيد: (كل الحالات المخفية (batch, seq_len, hidden_size), آخر حالة مخفية)
        """
        batch_size, seq_len, _ = x.shape
        h_t = h0 if h0 is not None else torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t


# =====================================================================
# 2) خلية LSTM مبنية يدويًا بكل البوابات الأربع
# =====================================================================
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # نُجمّع كل البوابات في مصفوفة واحدة كبيرة لكفاءة حسابية أعلى
        # (بدلًا من 4 طبقات Linear منفصلة -- حيلة شائعة في التطبيقات الفعلية)
        self.gates = nn.Linear(combined_size, 4 * hidden_size)

    def forward(
        self, x_t: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        x_t: (batch, input_size)
        state: (h_prev, C_prev), كل منهما (batch, hidden_size)
        يُعيد: (h_t, C_t)
        """
        h_prev, c_prev = state
        combined = torch.cat([x_t, h_prev], dim=1)

        gates = self.gates(combined)  # (batch, 4*hidden_size)
        f_gate, i_gate, g_candidate, o_gate = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_gate)        # بوابة النسيان
        i_t = torch.sigmoid(i_gate)        # بوابة الإدخال
        c_tilde = torch.tanh(g_candidate)  # المرشح الجديد
        o_t = torch.sigmoid(o_gate)        # بوابة الإخراج

        c_t = f_t * c_prev + i_t * c_tilde     # تحديث حالة الخلية (المسار الخطي الحاسم!)
        h_t = o_t * torch.tanh(c_t)             # الحالة المخفية الجديدة

        return h_t, c_t


class LSTMFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCellFromScratch(input_size, hidden_size)

    def forward(
        self, x: Tensor, init_state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        if init_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_state

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t, c_t)


# =====================================================================
# 3) التحقق العددي: هل تطبيقنا اليدوي يُطابق سلوك PyTorch المدمج منطقيًا؟
# =====================================================================
def sanity_check_shapes() -> None:
    batch, seq_len, input_size, hidden_size = 4, 10, 16, 32

    x = torch.randn(batch, seq_len, input_size)

    my_rnn = VanillaRNN(input_size, hidden_size)
    outputs, h_last = my_rnn(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    assert h_last.shape == (batch, hidden_size)
    print(f"✅ VanillaRNN من الصفر: outputs={outputs.shape}, h_last={h_last.shape}")

    my_lstm = LSTMFromScratch(input_size, hidden_size)
    outputs, (h_last, c_last) = my_lstm(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    print(f"✅ LSTM من الصفر: outputs={outputs.shape}, h_last={h_last.shape}, c_last={c_last.shape}")

    # الوحدة المدمجة الرسمية في PyTorch -- للاستخدام الفعلي في الإنتاج
    official_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    outputs_official, (h_n, c_n) = official_lstm(x)
    assert outputs_official.shape == (batch, seq_len, hidden_size)
    print(f"✅ nn.LSTM الرسمية: outputs={outputs_official.shape}")


if __name__ == "__main__":
    sanity_check_shapes()

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
"""
lstm_sentiment_classifier.py
مصنّف مشاعر كامل باستخدام Bidirectional Multi-layer LSTM.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(42)


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,       # مثال: سلبي / محايد / إيجابي
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # يقرأ الجملة من اليمين لليسار ومن اليسار لليمين معًا
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout بين الطبقات فقط
        )

        # ثنائي الاتجاه ⟹ حجم الإخراج = hidden_size * 2 (دمج الاتجاهين)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """
        input_ids: (batch, seq_len) -- مُرمّزة ومُحشوّة (padded)
        lengths:   (batch,) -- الطول الحقيقي لكل جملة قبل الحشو
        يُعيد: logits (batch, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # حيلة حاسمة للأداء والصحة: تجاهل خطوات الحشو (padding) في حساب LSTM
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.lstm(packed)
        # لا نحتاج فك الحزمة هنا لأننا سنستخدم h_n مباشرة

        # h_n شكلها: (num_layers * num_directions, batch, hidden_size)
        # نأخذ آخر طبقة فقط، لكلا الاتجاهين، وندمجهما
        forward_last = h_n[-2, :, :]   # آخر حالة من الاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة من الاتجاه الخلفي
        final_hidden = torch.cat([forward_last, backward_last], dim=1)

        return self.classifier(final_hidden)


def demo_training_step() -> None:
    """مثال توضيحي لخطوة تدريب واحدة كاملة."""
    vocab_size, batch_size, max_seq_len = 5000, 8, 20

    model = BiLSTMSentimentClassifier(vocab_size=vocab_size, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # بيانات وهمية للتوضيح: جمل بأطوال مختلفة ومُحشوّة بـ 0
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_len))
    lengths = torch.randint(5, max_seq_len + 1, (batch_size,))
    labels = torch.randint(0, 3, (batch_size,))

    optimizer.zero_grad()
    logits = model(input_ids, lengths)
    loss = criterion(logits, labels)
    loss.backward()

    # حيلة لا غنى عنها مع RNN/LSTM: Gradient Clipping لمنع الانفجار
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()
    print(f"Loss: {loss.item():.4f} | logits shape: {logits.shape}")


if __name__ == "__main__":
    demo_training_step()

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
import torch

h0 = torch.zeros(1, 2)
c0 = torch.zeros(1, 2)

f1 = torch.sigmoid(torch.tensor([[2.0, -1.0]]))
i1 = torch.sigmoid(torch.tensor([[0.5, 1.5]]))
c_tilde1 = torch.tanh(torch.tensor([[0.8, -0.3]]))
o1 = torch.sigmoid(torch.tensor([[1.0, 1.0]]))

c1 = f1 * c0 + i1 * c_tilde1
h1 = o1 * torch.tanh(c1)

print(f"f1={f1}")           # tensor([[0.8808, 0.2689]])
print(f"i1={i1}")           # tensor([[0.6225, 0.8176]])
print(f"c_tilde1={c_tilde1}") # tensor([[0.6640, -0.2913]])
print(f"C1={c1}")           # tensor([[0.4133, -0.2382]])
print(f"h1={h1}")           # tensor([[0.2855, -0.1717]])

# ────────────────────────────────────────────────────────────────────────
# [005] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256)  # ❌ الافتراضي: (seq_len, batch, features)!
x = torch.randn(32, 50, 128)  # تتوقع (batch, seq_len, features) لكن هذا ليس الافتراضي
output, _ = lstm(x)  # سيعمل بدون خطأ لكن بترتيب أبعاد خاطئ تمامًا! خطأ صامت خطير

# ────────────────────────────────────────────────────────────────────────
# [006] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [007] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
classifier = nn.Linear(256, 10)  # ❌ خطأ في الأبعاد! الإخراج الفعلي حجمه 512 وليس 256

# ────────────────────────────────────────────────────────────────────────
# [008] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
classifier = nn.Linear(256 * 2, 10)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [009] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
outputs, (h_n, c_n) = lstm(x)
# إذا احتجت الانتباه (Attention) على كل كلمة، استخدام h_n فقط خطأ شائع
result = classifier(h_n[-1])  # ❌ إذا كنت تحتاج معلومات من كل الخطوات الزمنية، فقدتها هنا

# ────────────────────────────────────────────────────────────────────────
# [010] book/chapters/part-02-embeddings/chapter-08.md
# ────────────────────────────────────────────────────────────────────────
"""
char_level_lstm_generator.py
نموذج لغوي على مستوى الحرف لتوليد نص عربي باستخدام LSTM.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


class CharLSTM(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 256, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.embedding(x)
        output, new_state = self.lstm(embedded, state)
        logits = self.fc(output)
        return logits, new_state


# TODO 1: حمّل نصًا عربيًا طويلًا (رواية، مجموعة مقالات) وابنِ قاموس حرف↔رقم
def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    raise NotImplementedError


# TODO 2: قسّم النص إلى تسلسلات تدريب بطول ثابت (مثلًا 100 حرف) بأسلوب Sliding Window
def create_training_sequences(
    encoded_text: list[int], seq_length: int = 100
) -> list[tuple[list[int], list[int]]]:
    """كل عيّنة: (تسلسل مدخل، نفس التسلسل مُزاح بخطوة واحدة = الهدف)"""
    raise NotImplementedError


# TODO 3: اكتب حلقة تدريب كاملة (استلهم من full_training_loop.py في الفصل 3)


# TODO 4: اكتب دالة التوليد بأسلوب Autoregressive مع Temperature Sampling
@torch.no_grad()
def generate_text(
    model: CharLSTM,
    start_text: str,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
    length: int = 300,
    temperature: float = 0.8,
) -> str:
    """
    يبدأ من start_text، ويُولّد حرفًا تلو الآخر، معيدًا تغذية الحرف المُولَّد
    كمدخل للخطوة التالية (Autoregressive)، مع التحكم بالعشوائية عبر temperature.
    """
    raise NotImplementedError

# تحدٍّ إضافي: قارن جودة النص المُولَّد بين temperature=0.3 (محافظ)
# و temperature=1.2 (جريء وعشوائي أكثر)، وناقش المقايضة بينهما.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 08: الفصل الثامن: من الكلمة إلى الجملة — RNN وLSTM وGRU."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_recurrent() -> None:
    inputs = [0.2, 0.8, -0.1, 0.5]
    state, gate = (0.0, 0.65)
    states = []
    for value in inputs:
        candidate = math.tanh(0.7 * state + 0.8 * value)
        state = gate * state + (1.0 - gate) * candidate
        states.append(round(state, 4))
    print('input sequence:', inputs)
    print('gated hidden states:', states)

if __name__ == "__main__":
    print("Chapter 08 lab: A recurrent state with a learned gate")
    run_recurrent()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
rnn_lstm_from_scratch.py
بناء خلايا RNN وLSTM من الصفر لفهم الآلية الداخلية بعمق،
ثم مقارنتها بالوحدات المدمجة في PyTorch.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

torch.manual_seed(42)


# =====================================================================
# 1) خلية RNN بسيطة (Vanilla RNN) مبنية يدويًا
# =====================================================================
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        x_t:    (batch, input_size)
        h_prev: (batch, hidden_size)
        يُعيد: h_t (batch, hidden_size)
        """
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


class VanillaRNN(nn.Module):
    """يُطبّق الخلية عبر كامل التسلسل الزمني (Unrolling يدوي)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = VanillaRNNCell(input_size, hidden_size)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        x: (batch, seq_len, input_size)
        يُعيد: (كل الحالات المخفية (batch, seq_len, hidden_size), آخر حالة مخفية)
        """
        batch_size, seq_len, _ = x.shape
        h_t = h0 if h0 is not None else torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t = self.cell(x[:, t, :], h_t)
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), h_t


# =====================================================================
# 2) خلية LSTM مبنية يدويًا بكل البوابات الأربع
# =====================================================================
class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # نُجمّع كل البوابات في مصفوفة واحدة كبيرة لكفاءة حسابية أعلى
        # (بدلًا من 4 طبقات Linear منفصلة -- حيلة شائعة في التطبيقات الفعلية)
        self.gates = nn.Linear(combined_size, 4 * hidden_size)

    def forward(
        self, x_t: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """
        x_t: (batch, input_size)
        state: (h_prev, C_prev), كل منهما (batch, hidden_size)
        يُعيد: (h_t, C_t)
        """
        h_prev, c_prev = state
        combined = torch.cat([x_t, h_prev], dim=1)

        gates = self.gates(combined)  # (batch, 4*hidden_size)
        f_gate, i_gate, g_candidate, o_gate = gates.chunk(4, dim=1)

        f_t = torch.sigmoid(f_gate)        # بوابة النسيان
        i_t = torch.sigmoid(i_gate)        # بوابة الإدخال
        c_tilde = torch.tanh(g_candidate)  # المرشح الجديد
        o_t = torch.sigmoid(o_gate)        # بوابة الإخراج

        c_t = f_t * c_prev + i_t * c_tilde     # تحديث حالة الخلية (المسار الخطي الحاسم!)
        h_t = o_t * torch.tanh(c_t)             # الحالة المخفية الجديدة

        return h_t, c_t


class LSTMFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCellFromScratch(input_size, hidden_size)

    def forward(
        self, x: Tensor, init_state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        if init_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = init_state

        outputs: list[Tensor] = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(h_t)

        return torch.stack(outputs, dim=1), (h_t, c_t)


# =====================================================================
# 3) التحقق العددي: هل تطبيقنا اليدوي يُطابق سلوك PyTorch المدمج منطقيًا؟
# =====================================================================
def sanity_check_shapes() -> None:
    batch, seq_len, input_size, hidden_size = 4, 10, 16, 32

    x = torch.randn(batch, seq_len, input_size)

    my_rnn = VanillaRNN(input_size, hidden_size)
    outputs, h_last = my_rnn(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    assert h_last.shape == (batch, hidden_size)
    print(f"✅ VanillaRNN من الصفر: outputs={outputs.shape}, h_last={h_last.shape}")

    my_lstm = LSTMFromScratch(input_size, hidden_size)
    outputs, (h_last, c_last) = my_lstm(x)
    assert outputs.shape == (batch, seq_len, hidden_size)
    print(f"✅ LSTM من الصفر: outputs={outputs.shape}, h_last={h_last.shape}, c_last={c_last.shape}")

    # الوحدة المدمجة الرسمية في PyTorch -- للاستخدام الفعلي في الإنتاج
    official_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    outputs_official, (h_n, c_n) = official_lstm(x)
    assert outputs_official.shape == (batch, seq_len, hidden_size)
    print(f"✅ nn.LSTM الرسمية: outputs={outputs_official.shape}")


if __name__ == "__main__":
    sanity_check_shapes()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
lstm_sentiment_classifier.py
مصنّف مشاعر كامل باستخدام Bidirectional Multi-layer LSTM.
Python 3.13+ | PyTorch 2.x
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(42)


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,       # مثال: سلبي / محايد / إيجابي
        pad_idx: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,     # يقرأ الجملة من اليمين لليسار ومن اليسار لليمين معًا
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout بين الطبقات فقط
        )

        # ثنائي الاتجاه ⟹ حجم الإخراج = hidden_size * 2 (دمج الاتجاهين)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """
        input_ids: (batch, seq_len) -- مُرمّزة ومُحشوّة (padded)
        lengths:   (batch,) -- الطول الحقيقي لكل جملة قبل الحشو
        يُعيد: logits (batch, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # حيلة حاسمة للأداء والصحة: تجاهل خطوات الحشو (padding) في حساب LSTM
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.lstm(packed)
        # لا نحتاج فك الحزمة هنا لأننا سنستخدم h_n مباشرة

        # h_n شكلها: (num_layers * num_directions, batch, hidden_size)
        # نأخذ آخر طبقة فقط، لكلا الاتجاهين، وندمجهما
        forward_last = h_n[-2, :, :]   # آخر حالة من الاتجاه الأمامي
        backward_last = h_n[-1, :, :]  # آخر حالة من الاتجاه الخلفي
        final_hidden = torch.cat([forward_last, backward_last], dim=1)

        return self.classifier(final_hidden)


def demo_training_step() -> None:
    """مثال توضيحي لخطوة تدريب واحدة كاملة."""
    vocab_size, batch_size, max_seq_len = 5000, 8, 20

    model = BiLSTMSentimentClassifier(vocab_size=vocab_size, num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # بيانات وهمية للتوضيح: جمل بأطوال مختلفة ومُحشوّة بـ 0
    input_ids = torch.randint(1, vocab_size, (batch_size, max_seq_len))
    lengths = torch.randint(5, max_seq_len + 1, (batch_size,))
    labels = torch.randint(0, 3, (batch_size,))

    optimizer.zero_grad()
    logits = model(input_ids, lengths)
    loss = criterion(logits, labels)
    loss.backward()

    # حيلة لا غنى عنها مع RNN/LSTM: Gradient Clipping لمنع الانفجار
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    optimizer.step()
    print(f"Loss: {loss.item():.4f} | logits shape: {logits.shape}")


if __name__ == "__main__":
    demo_training_step()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch

h0 = torch.zeros(1, 2)
c0 = torch.zeros(1, 2)

f1 = torch.sigmoid(torch.tensor([[2.0, -1.0]]))
i1 = torch.sigmoid(torch.tensor([[0.5, 1.5]]))
c_tilde1 = torch.tanh(torch.tensor([[0.8, -0.3]]))
o1 = torch.sigmoid(torch.tensor([[1.0, 1.0]]))

c1 = f1 * c0 + i1 * c_tilde1
h1 = o1 * torch.tanh(c1)

print(f"f1={f1}")           # tensor([[0.8808, 0.2689]])
print(f"i1={i1}")           # tensor([[0.6225, 0.8176]])
print(f"c_tilde1={c_tilde1}") # tensor([[0.6640, -0.2913]])
print(f"C1={c1}")           # tensor([[0.4133, -0.2382]])
print(f"h1={h1}")           # tensor([[0.2855, -0.1717]])

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256)  # ❌ الافتراضي: (seq_len, batch, features)!
x = torch.randn(32, 50, 128)  # تتوقع (batch, seq_len, features) لكن هذا ليس الافتراضي
output, _ = lstm(x)  # سيعمل بدون خطأ لكن بترتيب أبعاد خاطئ تمامًا! خطأ صامت خطير

# ────────────────────────────────────────────────────────────────────────
# [008] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-7
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [009] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-8
# ────────────────────────────────────────────────────────────────────────
lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
classifier = nn.Linear(256, 10)  # ❌ خطأ في الأبعاد! الإخراج الفعلي حجمه 512 وليس 256

# ────────────────────────────────────────────────────────────────────────
# [010] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-9
# ────────────────────────────────────────────────────────────────────────
classifier = nn.Linear(256 * 2, 10)  # ✅

# ────────────────────────────────────────────────────────────────────────
# [011] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-10
# ────────────────────────────────────────────────────────────────────────
outputs, (h_n, c_n) = lstm(x)
# إذا احتجت الانتباه (Attention) على كل كلمة، استخدام h_n فقط خطأ شائع
result = classifier(h_n[-1])  # ❌ إذا كنت تحتاج معلومات من كل الخطوات الزمنية، فقدتها هنا

# ────────────────────────────────────────────────────────────────────────
# [012] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-08-chapter-08.ipynb#cell-11
# ────────────────────────────────────────────────────────────────────────
"""
char_level_lstm_generator.py
نموذج لغوي على مستوى الحرف لتوليد نص عربي باستخدام LSTM.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.manual_seed(42)


class CharLSTM(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 256, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        embedded = self.embedding(x)
        output, new_state = self.lstm(embedded, state)
        logits = self.fc(output)
        return logits, new_state


# TODO 1: حمّل نصًا عربيًا طويلًا (رواية، مجموعة مقالات) وابنِ قاموس حرف↔رقم
def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    raise NotImplementedError


# TODO 2: قسّم النص إلى تسلسلات تدريب بطول ثابت (مثلًا 100 حرف) بأسلوب Sliding Window
def create_training_sequences(
    encoded_text: list[int], seq_length: int = 100
) -> list[tuple[list[int], list[int]]]:
    """كل عيّنة: (تسلسل مدخل، نفس التسلسل مُزاح بخطوة واحدة = الهدف)"""
    raise NotImplementedError


# TODO 3: اكتب حلقة تدريب كاملة (استلهم من full_training_loop.py في الفصل 3)


# TODO 4: اكتب دالة التوليد بأسلوب Autoregressive مع Temperature Sampling
@torch.no_grad()
def generate_text(
    model: CharLSTM,
    start_text: str,
    char2idx: dict[str, int],
    idx2char: dict[int, str],
    length: int = 300,
    temperature: float = 0.8,
) -> str:
    """
    يبدأ من start_text، ويُولّد حرفًا تلو الآخر، معيدًا تغذية الحرف المُولَّد
    كمدخل للخطوة التالية (Autoregressive)، مع التحكم بالعشوائية عبر temperature.
    """
    raise NotImplementedError

# تحدٍّ إضافي: قارن جودة النص المُولَّد بين temperature=0.3 (محافظ)
# و temperature=1.2 (جريء وعشوائي أكثر)، وناقش المقايضة بينهما.
