#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الانتباه الذاتي متعدد الرؤوس
Mastering Transformers and Large Language Models — Self Attention

Book slug : mastering-llms-transformers
Chapter   : ch-10-ch10-self-attention (lesson 10)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-10-ch10-self-attention.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الانتباه الذاتي متعدد الرؤوس
Mastering Transformers and Large Language Models — Self Attention

Book slug : mastering-llms-transformers
Chapter   : ch-10-ch10-self-attention (lesson 10)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-10-ch10-self-attention.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الانتباه الذاتي متعدد الرؤوس
Mastering Transformers and Large Language Models — Self Attention

Book slug : mastering-llms-transformers
Chapter   : ch-10-ch10-self-attention (lesson 10)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch10-self-attention.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch10-self-attention.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Scaled Dot-Product Attention من الصفر بـ NumPy
#    (لفهم الآلية الرياضية حرفيًا دون أي تجريد)
# ============================================================
def softmax_numpy(x, axis=-1):
    """Softmax مستقرة عدديًا."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    Q: (n, d_k)   K: (n, d_k)   V: (n, d_v)
    mask: (n, n) اختياري، صفر حيث نمنع الانتباه

    العائد: (output, attn_weights)
    """
    d_k = Q.shape[-1]

    # QK^T  → (n, n)
    scores = Q @ K.T

    # التحجيم: القسمة على جذر d_k
    scores = scores / np.sqrt(d_k)

    # الإخفاء (إن وُجد)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # softmax على كل صف (لكل استعلام، توزيع احتمالي على كل المفاتيح)
    attn_weights = softmax_numpy(scores, axis=-1)

    # التجميع الموزون للقيم
    output = attn_weights @ V

    return output, attn_weights

# ============================================================
# 2. Multi-Head Attention كامل — PyTorch (جاهز للإنتاج)
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    تطبيق كامل لـ Multi-Head Attention كما في ورقة
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يُقسَم على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # إسقاطات خطية موحدة لكل الرؤوس دفعة واحدة (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len_q, seq_len_k) أو قابل للبث (broadcastable)
        """
        Q = self.split_heads(self.W_q(query))   # (batch, h, seq_len, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, h, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)      # (batch, h, seq_len_q, d_k)
        context = self.combine_heads(context)         # (batch, seq_len_q, d_model)

        output = self.W_o(context)
        return output, attn_weights

# ============================================================
# 3. أقنعة جاهزة (Masks)
# ============================================================
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """قناع سببي (ذاتي الانحدار) يمنع النظر للمستقبل."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

def create_padding_mask(token_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """قناع يخفي رموز الحشو."""
    mask = (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch10-self-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Scaled Dot-Product Attention من الصفر بـ NumPy
#    (لفهم الآلية الرياضية حرفيًا دون أي تجريد)
# ============================================================
def softmax_numpy(x, axis=-1):
    """Softmax مستقرة عدديًا."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    Q: (n, d_k)   K: (n, d_k)   V: (n, d_v)
    mask: (n, n) اختياري، صفر حيث نمنع الانتباه

    العائد: (output, attn_weights)
    """
    d_k = Q.shape[-1]

    # QK^T  → (n, n)
    scores = Q @ K.T

    # التحجيم: القسمة على جذر d_k
    scores = scores / np.sqrt(d_k)

    # الإخفاء (إن وُجد)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # softmax على كل صف (لكل استعلام، توزيع احتمالي على كل المفاتيح)
    attn_weights = softmax_numpy(scores, axis=-1)

    # التجميع الموزون للقيم
    output = attn_weights @ V

    return output, attn_weights

# ============================================================
# 2. Multi-Head Attention كامل — PyTorch (جاهز للإنتاج)
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    تطبيق كامل لـ Multi-Head Attention كما في ورقة
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يُقسَم على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # إسقاطات خطية موحدة لكل الرؤوس دفعة واحدة (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len_q, seq_len_k) أو قابل للبث (broadcastable)
        """
        Q = self.split_heads(self.W_q(query))   # (batch, h, seq_len, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, h, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)      # (batch, h, seq_len_q, d_k)
        context = self.combine_heads(context)         # (batch, seq_len_q, d_model)

        output = self.W_o(context)
        return output, attn_weights

# ============================================================
# 3. أقنعة جاهزة (Masks)
# ============================================================
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """قناع سببي (ذاتي الانحدار) يمنع النظر للمستقبل."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

def create_padding_mask(token_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """قناع يخفي رموز الحشو."""
    mask = (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الانتباه الذاتي متعدد الرؤوس
Mastering Transformers and Large Language Models — Self Attention

Book slug : mastering-llms-transformers
Chapter   : ch-10-ch10-self-attention (lesson 10)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch10-self-attention.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch10-self-attention.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Scaled Dot-Product Attention من الصفر بـ NumPy
#    (لفهم الآلية الرياضية حرفيًا دون أي تجريد)
# ============================================================
def softmax_numpy(x, axis=-1):
    """Softmax مستقرة عدديًا."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    Q: (n, d_k)   K: (n, d_k)   V: (n, d_v)
    mask: (n, n) اختياري، صفر حيث نمنع الانتباه

    العائد: (output, attn_weights)
    """
    d_k = Q.shape[-1]

    # QK^T  → (n, n)
    scores = Q @ K.T

    # التحجيم: القسمة على جذر d_k
    scores = scores / np.sqrt(d_k)

    # الإخفاء (إن وُجد)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # softmax على كل صف (لكل استعلام، توزيع احتمالي على كل المفاتيح)
    attn_weights = softmax_numpy(scores, axis=-1)

    # التجميع الموزون للقيم
    output = attn_weights @ V

    return output, attn_weights

# ============================================================
# 2. Multi-Head Attention كامل — PyTorch (جاهز للإنتاج)
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    تطبيق كامل لـ Multi-Head Attention كما في ورقة
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يُقسَم على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # إسقاطات خطية موحدة لكل الرؤوس دفعة واحدة (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len_q, seq_len_k) أو قابل للبث (broadcastable)
        """
        Q = self.split_heads(self.W_q(query))   # (batch, h, seq_len, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, h, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)      # (batch, h, seq_len_q, d_k)
        context = self.combine_heads(context)         # (batch, seq_len_q, d_model)

        output = self.W_o(context)
        return output, attn_weights

# ============================================================
# 3. أقنعة جاهزة (Masks)
# ============================================================
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """قناع سببي (ذاتي الانحدار) يمنع النظر للمستقبل."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

def create_padding_mask(token_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """قناع يخفي رموز الحشو."""
    mask = (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch10-self-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Scaled Dot-Product Attention من الصفر بـ NumPy
#    (لفهم الآلية الرياضية حرفيًا دون أي تجريد)
# ============================================================
def softmax_numpy(x, axis=-1):
    """Softmax مستقرة عدديًا."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention_numpy(Q, K, V, mask=None):
    """
    Q: (n, d_k)   K: (n, d_k)   V: (n, d_v)
    mask: (n, n) اختياري، صفر حيث نمنع الانتباه

    العائد: (output, attn_weights)
    """
    d_k = Q.shape[-1]

    # QK^T  → (n, n)
    scores = Q @ K.T

    # التحجيم: القسمة على جذر d_k
    scores = scores / np.sqrt(d_k)

    # الإخفاء (إن وُجد)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # softmax على كل صف (لكل استعلام، توزيع احتمالي على كل المفاتيح)
    attn_weights = softmax_numpy(scores, axis=-1)

    # التجميع الموزون للقيم
    output = attn_weights @ V

    return output, attn_weights

# ============================================================
# 2. Multi-Head Attention كامل — PyTorch (جاهز للإنتاج)
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    تطبيق كامل لـ Multi-Head Attention كما في ورقة
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model يجب أن يُقسَم على num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # إسقاطات خطية موحدة لكل الرؤوس دفعة واحدة (أكفأ حسابيًا من رؤوس منفصلة)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k) """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """ (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model) """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len_q, seq_len_k) أو قابل للبث (broadcastable)
        """
        Q = self.split_heads(self.W_q(query))   # (batch, h, seq_len, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, h, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)      # (batch, h, seq_len_q, d_k)
        context = self.combine_heads(context)         # (batch, seq_len_q, d_model)

        output = self.W_o(context)
        return output, attn_weights

# ============================================================
# 3. أقنعة جاهزة (Masks)
# ============================================================
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """قناع سببي (ذاتي الانحدار) يمنع النظر للمستقبل."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

def create_padding_mask(token_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """قناع يخفي رموز الحشو."""
    mask = (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-10-ch10-self-attention.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
