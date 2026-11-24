#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — من التكرار إلى الانتباه: آلية Attention التاريخية
Mastering Transformers and Large Language Models — Attention History

Book slug : mastering-llms-transformers
Chapter   : ch-09-ch09-attention-history (lesson 9)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-09-ch09-attention-history.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — من التكرار إلى الانتباه: آلية Attention التاريخية
Mastering Transformers and Large Language Models — Attention History

Book slug : mastering-llms-transformers
Chapter   : ch-09-ch09-attention-history (lesson 9)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-09-ch09-attention-history.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — من التكرار إلى الانتباه: آلية Attention التاريخية
Mastering Transformers and Large Language Models — Attention History

Book slug : mastering-llms-transformers
Chapter   : ch-09-ch09-attention-history (lesson 9)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch09-attention-history.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch09-attention-history.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Bahdanau (Additive) Attention — من الصفر بـ PyTorch
# ============================================================
class BahdanauAttention(nn.Module):
    """
    آلية انتباه Bahdanau الإضافية (Additive/Concat Attention).
    تُستخدم عادة مع decoder RNN حيث تُحسب قبل خطوة RNN (input feeding).
    """
    def __init__(self, decoder_dim: int, encoder_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden:  (batch, decoder_dim)       حالة المفكك في t-1
        encoder_outputs: (batch, src_len, encoder_dim)  كل حالات المرمز
        mask: (batch, src_len)  1 للمواضع الحقيقية، 0 للـ padding
        """
        batch, src_len, _ = encoder_outputs.shape

        # وسّع حالة المفكك لتتوافق مع طول التسلسل المصدر
        dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_dim)

        # e_{t,i} = v^T tanh(W_a h_dec + U_a h_enc)
        energy = torch.tanh(self.W_a(dec_expanded) + self.U_a(encoder_outputs))
        scores = self.v_a(energy).squeeze(-1)          # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)        # α_{t,i}

        # c_t = Σ α_{t,i} h_i^enc
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                    # (batch, encoder_dim)

        return context, attn_weights

# ============================================================
# 2. Luong (Multiplicative) Attention — بثلاث صيغ score
# ============================================================
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, method: str = "general"):
        super().__init__()
        assert method in ("dot", "general", "concat")
        self.method = method
        if method == "general":
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == "concat":
            self.W_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # h_dec^T h_enc
            return torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == "general":
            transformed = self.W_a(encoder_outputs)
            return torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        else:  # concat
            src_len = encoder_outputs.size(1)
            dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.W_a(torch.cat([dec_expanded, encoder_outputs], dim=2)))
            return self.v_a(energy).squeeze(-1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        scores = self.score(decoder_hidden, encoder_outputs)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

# ============================================================
# 3. Seq2Seq كامل مع Attention (تطبيق للترجمة العربية-الإنجليزية)
# ============================================================
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # دمج الاتجاهين الأمامي والخلفي لحالة البداية للمفكك
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[0], hidden[1]], dim=1)))
        return outputs, hidden  # outputs: (batch, src_len, 2*hidden_dim)

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.rnn = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        embedded = self.embedding(input_token).unsqueeze(1)   # (batch, 1, embed_dim)

        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(1)
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat([output, context, embedded.squeeze(1)], dim=1))
        return prediction, hidden, attn_weights

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch09-attention-history.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Bahdanau (Additive) Attention — من الصفر بـ PyTorch
# ============================================================
class BahdanauAttention(nn.Module):
    """
    آلية انتباه Bahdanau الإضافية (Additive/Concat Attention).
    تُستخدم عادة مع decoder RNN حيث تُحسب قبل خطوة RNN (input feeding).
    """
    def __init__(self, decoder_dim: int, encoder_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden:  (batch, decoder_dim)       حالة المفكك في t-1
        encoder_outputs: (batch, src_len, encoder_dim)  كل حالات المرمز
        mask: (batch, src_len)  1 للمواضع الحقيقية، 0 للـ padding
        """
        batch, src_len, _ = encoder_outputs.shape

        # وسّع حالة المفكك لتتوافق مع طول التسلسل المصدر
        dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_dim)

        # e_{t,i} = v^T tanh(W_a h_dec + U_a h_enc)
        energy = torch.tanh(self.W_a(dec_expanded) + self.U_a(encoder_outputs))
        scores = self.v_a(energy).squeeze(-1)          # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)        # α_{t,i}

        # c_t = Σ α_{t,i} h_i^enc
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                    # (batch, encoder_dim)

        return context, attn_weights

# ============================================================
# 2. Luong (Multiplicative) Attention — بثلاث صيغ score
# ============================================================
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, method: str = "general"):
        super().__init__()
        assert method in ("dot", "general", "concat")
        self.method = method
        if method == "general":
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == "concat":
            self.W_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # h_dec^T h_enc
            return torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == "general":
            transformed = self.W_a(encoder_outputs)
            return torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        else:  # concat
            src_len = encoder_outputs.size(1)
            dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.W_a(torch.cat([dec_expanded, encoder_outputs], dim=2)))
            return self.v_a(energy).squeeze(-1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        scores = self.score(decoder_hidden, encoder_outputs)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

# ============================================================
# 3. Seq2Seq كامل مع Attention (تطبيق للترجمة العربية-الإنجليزية)
# ============================================================
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # دمج الاتجاهين الأمامي والخلفي لحالة البداية للمفكك
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[0], hidden[1]], dim=1)))
        return outputs, hidden  # outputs: (batch, src_len, 2*hidden_dim)

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.rnn = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        embedded = self.embedding(input_token).unsqueeze(1)   # (batch, 1, embed_dim)

        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(1)
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat([output, context, embedded.squeeze(1)], dim=1))
        return prediction, hidden, attn_weights

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — من التكرار إلى الانتباه: آلية Attention التاريخية
Mastering Transformers and Large Language Models — Attention History

Book slug : mastering-llms-transformers
Chapter   : ch-09-ch09-attention-history (lesson 9)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch09-attention-history.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch09-attention-history.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Bahdanau (Additive) Attention — من الصفر بـ PyTorch
# ============================================================
class BahdanauAttention(nn.Module):
    """
    آلية انتباه Bahdanau الإضافية (Additive/Concat Attention).
    تُستخدم عادة مع decoder RNN حيث تُحسب قبل خطوة RNN (input feeding).
    """
    def __init__(self, decoder_dim: int, encoder_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden:  (batch, decoder_dim)       حالة المفكك في t-1
        encoder_outputs: (batch, src_len, encoder_dim)  كل حالات المرمز
        mask: (batch, src_len)  1 للمواضع الحقيقية، 0 للـ padding
        """
        batch, src_len, _ = encoder_outputs.shape

        # وسّع حالة المفكك لتتوافق مع طول التسلسل المصدر
        dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_dim)

        # e_{t,i} = v^T tanh(W_a h_dec + U_a h_enc)
        energy = torch.tanh(self.W_a(dec_expanded) + self.U_a(encoder_outputs))
        scores = self.v_a(energy).squeeze(-1)          # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)        # α_{t,i}

        # c_t = Σ α_{t,i} h_i^enc
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                    # (batch, encoder_dim)

        return context, attn_weights

# ============================================================
# 2. Luong (Multiplicative) Attention — بثلاث صيغ score
# ============================================================
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, method: str = "general"):
        super().__init__()
        assert method in ("dot", "general", "concat")
        self.method = method
        if method == "general":
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == "concat":
            self.W_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # h_dec^T h_enc
            return torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == "general":
            transformed = self.W_a(encoder_outputs)
            return torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        else:  # concat
            src_len = encoder_outputs.size(1)
            dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.W_a(torch.cat([dec_expanded, encoder_outputs], dim=2)))
            return self.v_a(energy).squeeze(-1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        scores = self.score(decoder_hidden, encoder_outputs)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

# ============================================================
# 3. Seq2Seq كامل مع Attention (تطبيق للترجمة العربية-الإنجليزية)
# ============================================================
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # دمج الاتجاهين الأمامي والخلفي لحالة البداية للمفكك
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[0], hidden[1]], dim=1)))
        return outputs, hidden  # outputs: (batch, src_len, 2*hidden_dim)

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.rnn = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        embedded = self.embedding(input_token).unsqueeze(1)   # (batch, 1, embed_dim)

        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(1)
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat([output, context, embedded.squeeze(1)], dim=1))
        return prediction, hidden, attn_weights

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch09-attention-history.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Bahdanau (Additive) Attention — من الصفر بـ PyTorch
# ============================================================
class BahdanauAttention(nn.Module):
    """
    آلية انتباه Bahdanau الإضافية (Additive/Concat Attention).
    تُستخدم عادة مع decoder RNN حيث تُحسب قبل خطوة RNN (input feeding).
    """
    def __init__(self, decoder_dim: int, encoder_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_a = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden:  (batch, decoder_dim)       حالة المفكك في t-1
        encoder_outputs: (batch, src_len, encoder_dim)  كل حالات المرمز
        mask: (batch, src_len)  1 للمواضع الحقيقية، 0 للـ padding
        """
        batch, src_len, _ = encoder_outputs.shape

        # وسّع حالة المفكك لتتوافق مع طول التسلسل المصدر
        dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # (batch, src_len, decoder_dim)

        # e_{t,i} = v^T tanh(W_a h_dec + U_a h_enc)
        energy = torch.tanh(self.W_a(dec_expanded) + self.U_a(encoder_outputs))
        scores = self.v_a(energy).squeeze(-1)          # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)        # α_{t,i}

        # c_t = Σ α_{t,i} h_i^enc
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)                    # (batch, encoder_dim)

        return context, attn_weights

# ============================================================
# 2. Luong (Multiplicative) Attention — بثلاث صيغ score
# ============================================================
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim: int, method: str = "general"):
        super().__init__()
        assert method in ("dot", "general", "concat")
        self.method = method
        if method == "general":
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == "concat":
            self.W_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # h_dec^T h_enc
            return torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == "general":
            transformed = self.W_a(encoder_outputs)
            return torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        else:  # concat
            src_len = encoder_outputs.size(1)
            dec_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.W_a(torch.cat([dec_expanded, encoder_outputs], dim=2)))
            return self.v_a(energy).squeeze(-1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        scores = self.score(decoder_hidden, encoder_outputs)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

# ============================================================
# 3. Seq2Seq كامل مع Attention (تطبيق للترجمة العربية-الإنجليزية)
# ============================================================
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # دمج الاتجاهين الأمامي والخلفي لحالة البداية للمفكك
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[0], hidden[1]], dim=1)))
        return outputs, hidden  # outputs: (batch, src_len, 2*hidden_dim)

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.rnn = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        embedded = self.embedding(input_token).unsqueeze(1)   # (batch, 1, embed_dim)

        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = output.squeeze(1)
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat([output, context, embedded.squeeze(1)], dim=1))
        return prediction, hidden, attn_weights

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-09-ch09-attention-history.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
