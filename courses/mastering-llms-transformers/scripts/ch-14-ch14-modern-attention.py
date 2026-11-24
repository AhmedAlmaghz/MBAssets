#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي المتقدم وآليات الانتباه الفعّالة
Mastering Transformers and Large Language Models — Modern Attention

Book slug : mastering-llms-transformers
Chapter   : ch-14-ch14-modern-attention (lesson 14)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-14-ch14-modern-attention.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي المتقدم وآليات الانتباه الفعّالة
Mastering Transformers and Large Language Models — Modern Attention

Book slug : mastering-llms-transformers
Chapter   : ch-14-ch14-modern-attention (lesson 14)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-14-ch14-modern-attention.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي المتقدم وآليات الانتباه الفعّالة
Mastering Transformers and Large Language Models — Modern Attention

Book slug : mastering-llms-transformers
Chapter   : ch-14-ch14-modern-attention (lesson 14)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch14-modern-attention.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch14-modern-attention.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. RoPE (Rotary Positional Embedding) — التطبيق الكامل
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # ترددات مختلفة لكل زوج من الأبعاد: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        # نُكرّر كل تردد مرتين لتغطية الزوج الكامل من الأبعاد
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    """تدوير نصفي: [x1, x2] → [-x2, x1]، خطوة أساسية في تطبيق RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    تطبيق الدوران على Q وK.
    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # للبث على batch وnum_heads
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated

class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-Head Attention مع RoPE بدلاً من الترميز الموضعي الإضافي."""
    def __init__(self, d_model, num_heads, max_seq_len=8192, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # تطبيق RoPE على Q وK فقط (ليس V)
        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 2. ALiBi (Attention with Linear Biases)
# ============================================================
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    حساب الميول (slopes) لكل رأس، كمتتالية هندسية كما في الورقة الأصلية.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        # معالجة الحالات غير القوى الصحيحة لـ 2 (تفصيل من الورقة الأصلية)
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
        return torch.tensor(slopes + extra)

class MultiHeadAttentionWithALiBi(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        self.dropout = nn.Dropout(dropout)

    def _get_alibi_bias(self, seq_len, device):
        # مصفوفة المسافات بين كل زوج مواضع: (seq_len, seq_len)
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # نريد فقط المسافات غير الموجبة (causal) لتعكس البعد للخلف
        relative_positions = -relative_positions.abs().float()

        # لكل رأس: ميل خاص × مصفوفة المسافات
        alibi_bias = self.slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)
        return alibi_bias  # (num_heads, seq_len, seq_len)

    def forward(self, x, causal_mask=None):
        batch_size, seq_len, _ = x.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # إضافة تحيز ALiBi (بدلًا من أي ترميز موضعي على Q,K)
        alibi_bias = self._get_alibi_bias(seq_len, x.device).unsqueeze(0)
        scores = scores + alibi_bias

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 3. Grouped-Query Attention (GQA) — يشمل MQA كحالة خاصة
# ============================================================
class GroupedQueryAttention(nn.Module):
    """
    GQA: تقسيم رؤوس Query إلى مجموعات، كل مجموعة تتشارك K,V.
    num_kv_groups=1  → مكافئ MQA
    num_kv_groups=num_heads → مكافئ MHA القياسي
    """
    def __init__(self, d_model, num_heads, num_kv_groups, max_seq_len=8192, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads يجب أن يُقسَم على num_kv_groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_groups  # عدد رؤوس Q لكل مجموعة K,V

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # ملاحظة: K,V بأبعاد أصغر بكثير — فقط num_kv_groups بدلًا من num_heads
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # تكرار K,V لتُطابق عدد رؤوس Q (كل مجموعة K,V تُستخدَم لعدة رؤوس Q)
        K = K.repeat_interleave(self.group_size, dim=1)  # (batch, num_heads, seq_len, d_k)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

def compute_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_len,
                            batch_size=1, bytes_per_element=2):
    """حساب حجم KV Cache بالجيجابايت لمقارنة MHA/GQA/MQA."""
    # العامل 2: K و V منفصلان
    total_bytes = (2 * num_layers * num_kv_heads * head_dim *
                   seq_len * batch_size * bytes_per_element)
    return total_bytes / (1024 ** 3)  # تحويل لجيجابايت

# ============================================================
# 4. Flash Attention — استخدام التطبيق الرسمي المُحسَّن (PyTorch 2.0+)
# ============================================================
def flash_attention_demo(Q, K, V, is_causal=True):
    """
    PyTorch 2.0+ يوفر scaled_dot_product_attention المُحسَّنة
    التي تستخدم Flash Attention تلقائيًا عند توفر العتاد المناسب.
    """
    # هذه الدالة تختار تلقائيًا أسرع تطبيق متاح (Flash/Memory-Efficient/Math)
    output = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=is_causal, dropout_p=0.0
    )
    return output

# مقارنة عملية: التنفيذ اليدوي مقابل Flash Attention المدمج
def compare_standard_vs_flash(seq_len=4096, d_model=512, num_heads=8, device='cuda'):
    import time

    batch_size = 2
    d_k = d_model // num_heads
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)

    # التنفيذ القياسي (يدوي، يخزّن مصفوفة n×n كاملة)
    def standard_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_standard = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_flash = flash_attention_demo(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    flash_time = time.time() - start
    flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Standard: {standard_time*1000:.2f}ms, {standard_memory:.2f}MB")
    print(f"Flash:    {flash_time*1000:.2f}ms, {flash_memory:.2f}MB")
    print(f"تسريع: {standard_time/flash_time:.2f}x | توفير ذاكرة: {standard_memory/flash_memory:.2f}x")

# ============================================================
# 5. مفهوم Ring Attention المبسّط (توزيع عبر عدة أجهزة)
# ============================================================
def ring_attention_concept_demo(num_devices: int, total_seq_len: int, d_model: int):
    """
    توضيح مفاهيمي مبسّط لفكرة Ring Attention (التنفيذ الفعلي
    يتطلب أدوات توزيع مثل torch.distributed وNCCL):

    1. يُقسَّم التسلسل الطويل إلى num_devices أجزاء متساوية
    2. كل جهاز يحتفظ بجزئه من Q محليًا طوال الوقت
    3. أجزاء K,V تدور بين الأجهزة في "حلقة" (ring) عبر num_devices خطوة
    4. في كل خطوة دوران، يحسب كل جهاز انتباه جزئي بين Q المحلي
       وK,V الحالي القادم من الجهاز المجاور، ويُراكم النتيجة
    5. بعد اكتمال الدورة الكاملة، كل جهاز يملك نتيجة انتباه كاملة
       ومصححة لجزئه من Q مقابل كل أجزاء K,V من كل الأجهزة
    """
    chunk_size = total_seq_len // num_devices
    print(f"عدد الأجهزة: {num_devices}")
    print(f"طول كل جزء محلي: {chunk_size} رمز")
    print(f"الذاكرة المطلوبة لكل جهاز لتخزين Q,K,V المحلية فقط: "
          f"~{3 * chunk_size * d_model * 2 / (1024**2):.2f} MB (بدلاً من التسلسل الكامل)")
    print("كل جهاز يحسب تدريجيًا، عبر دورات تبادل، انتباهًا كاملًا دون "
          "الحاجة لتخزين التسلسل الكامل في ذاكرة جهاز واحد")

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch14-modern-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. RoPE (Rotary Positional Embedding) — التطبيق الكامل
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # ترددات مختلفة لكل زوج من الأبعاد: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        # نُكرّر كل تردد مرتين لتغطية الزوج الكامل من الأبعاد
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    """تدوير نصفي: [x1, x2] → [-x2, x1]، خطوة أساسية في تطبيق RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    تطبيق الدوران على Q وK.
    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # للبث على batch وnum_heads
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated

class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-Head Attention مع RoPE بدلاً من الترميز الموضعي الإضافي."""
    def __init__(self, d_model, num_heads, max_seq_len=8192, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # تطبيق RoPE على Q وK فقط (ليس V)
        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 2. ALiBi (Attention with Linear Biases)
# ============================================================
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    حساب الميول (slopes) لكل رأس، كمتتالية هندسية كما في الورقة الأصلية.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        # معالجة الحالات غير القوى الصحيحة لـ 2 (تفصيل من الورقة الأصلية)
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
        return torch.tensor(slopes + extra)

class MultiHeadAttentionWithALiBi(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        self.dropout = nn.Dropout(dropout)

    def _get_alibi_bias(self, seq_len, device):
        # مصفوفة المسافات بين كل زوج مواضع: (seq_len, seq_len)
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # نريد فقط المسافات غير الموجبة (causal) لتعكس البعد للخلف
        relative_positions = -relative_positions.abs().float()

        # لكل رأس: ميل خاص × مصفوفة المسافات
        alibi_bias = self.slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)
        return alibi_bias  # (num_heads, seq_len, seq_len)

    def forward(self, x, causal_mask=None):
        batch_size, seq_len, _ = x.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # إضافة تحيز ALiBi (بدلًا من أي ترميز موضعي على Q,K)
        alibi_bias = self._get_alibi_bias(seq_len, x.device).unsqueeze(0)
        scores = scores + alibi_bias

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 3. Grouped-Query Attention (GQA) — يشمل MQA كحالة خاصة
# ============================================================
class GroupedQueryAttention(nn.Module):
    """
    GQA: تقسيم رؤوس Query إلى مجموعات، كل مجموعة تتشارك K,V.
    num_kv_groups=1  → مكافئ MQA
    num_kv_groups=num_heads → مكافئ MHA القياسي
    """
    def __init__(self, d_model, num_heads, num_kv_groups, max_seq_len=8192, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads يجب أن يُقسَم على num_kv_groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_groups  # عدد رؤوس Q لكل مجموعة K,V

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # ملاحظة: K,V بأبعاد أصغر بكثير — فقط num_kv_groups بدلًا من num_heads
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # تكرار K,V لتُطابق عدد رؤوس Q (كل مجموعة K,V تُستخدَم لعدة رؤوس Q)
        K = K.repeat_interleave(self.group_size, dim=1)  # (batch, num_heads, seq_len, d_k)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

def compute_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_len,
                            batch_size=1, bytes_per_element=2):
    """حساب حجم KV Cache بالجيجابايت لمقارنة MHA/GQA/MQA."""
    # العامل 2: K و V منفصلان
    total_bytes = (2 * num_layers * num_kv_heads * head_dim *
                   seq_len * batch_size * bytes_per_element)
    return total_bytes / (1024 ** 3)  # تحويل لجيجابايت

# ============================================================
# 4. Flash Attention — استخدام التطبيق الرسمي المُحسَّن (PyTorch 2.0+)
# ============================================================
def flash_attention_demo(Q, K, V, is_causal=True):
    """
    PyTorch 2.0+ يوفر scaled_dot_product_attention المُحسَّنة
    التي تستخدم Flash Attention تلقائيًا عند توفر العتاد المناسب.
    """
    # هذه الدالة تختار تلقائيًا أسرع تطبيق متاح (Flash/Memory-Efficient/Math)
    output = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=is_causal, dropout_p=0.0
    )
    return output

# مقارنة عملية: التنفيذ اليدوي مقابل Flash Attention المدمج
def compare_standard_vs_flash(seq_len=4096, d_model=512, num_heads=8, device='cuda'):
    import time

    batch_size = 2
    d_k = d_model // num_heads
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)

    # التنفيذ القياسي (يدوي، يخزّن مصفوفة n×n كاملة)
    def standard_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_standard = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_flash = flash_attention_demo(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    flash_time = time.time() - start
    flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Standard: {standard_time*1000:.2f}ms, {standard_memory:.2f}MB")
    print(f"Flash:    {flash_time*1000:.2f}ms, {flash_memory:.2f}MB")
    print(f"تسريع: {standard_time/flash_time:.2f}x | توفير ذاكرة: {standard_memory/flash_memory:.2f}x")

# ============================================================
# 5. مفهوم Ring Attention المبسّط (توزيع عبر عدة أجهزة)
# ============================================================
def ring_attention_concept_demo(num_devices: int, total_seq_len: int, d_model: int):
    """
    توضيح مفاهيمي مبسّط لفكرة Ring Attention (التنفيذ الفعلي
    يتطلب أدوات توزيع مثل torch.distributed وNCCL):

    1. يُقسَّم التسلسل الطويل إلى num_devices أجزاء متساوية
    2. كل جهاز يحتفظ بجزئه من Q محليًا طوال الوقت
    3. أجزاء K,V تدور بين الأجهزة في "حلقة" (ring) عبر num_devices خطوة
    4. في كل خطوة دوران، يحسب كل جهاز انتباه جزئي بين Q المحلي
       وK,V الحالي القادم من الجهاز المجاور، ويُراكم النتيجة
    5. بعد اكتمال الدورة الكاملة، كل جهاز يملك نتيجة انتباه كاملة
       ومصححة لجزئه من Q مقابل كل أجزاء K,V من كل الأجهزة
    """
    chunk_size = total_seq_len // num_devices
    print(f"عدد الأجهزة: {num_devices}")
    print(f"طول كل جزء محلي: {chunk_size} رمز")
    print(f"الذاكرة المطلوبة لكل جهاز لتخزين Q,K,V المحلية فقط: "
          f"~{3 * chunk_size * d_model * 2 / (1024**2):.2f} MB (بدلاً من التسلسل الكامل)")
    print("كل جهاز يحسب تدريجيًا، عبر دورات تبادل، انتباهًا كاملًا دون "
          "الحاجة لتخزين التسلسل الكامل في ذاكرة جهاز واحد")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي المتقدم وآليات الانتباه الفعّالة
Mastering Transformers and Large Language Models — Modern Attention

Book slug : mastering-llms-transformers
Chapter   : ch-14-ch14-modern-attention (lesson 14)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch14-modern-attention.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch14-modern-attention.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. RoPE (Rotary Positional Embedding) — التطبيق الكامل
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # ترددات مختلفة لكل زوج من الأبعاد: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        # نُكرّر كل تردد مرتين لتغطية الزوج الكامل من الأبعاد
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    """تدوير نصفي: [x1, x2] → [-x2, x1]، خطوة أساسية في تطبيق RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    تطبيق الدوران على Q وK.
    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # للبث على batch وnum_heads
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated

class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-Head Attention مع RoPE بدلاً من الترميز الموضعي الإضافي."""
    def __init__(self, d_model, num_heads, max_seq_len=8192, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # تطبيق RoPE على Q وK فقط (ليس V)
        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 2. ALiBi (Attention with Linear Biases)
# ============================================================
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    حساب الميول (slopes) لكل رأس، كمتتالية هندسية كما في الورقة الأصلية.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        # معالجة الحالات غير القوى الصحيحة لـ 2 (تفصيل من الورقة الأصلية)
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
        return torch.tensor(slopes + extra)

class MultiHeadAttentionWithALiBi(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        self.dropout = nn.Dropout(dropout)

    def _get_alibi_bias(self, seq_len, device):
        # مصفوفة المسافات بين كل زوج مواضع: (seq_len, seq_len)
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # نريد فقط المسافات غير الموجبة (causal) لتعكس البعد للخلف
        relative_positions = -relative_positions.abs().float()

        # لكل رأس: ميل خاص × مصفوفة المسافات
        alibi_bias = self.slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)
        return alibi_bias  # (num_heads, seq_len, seq_len)

    def forward(self, x, causal_mask=None):
        batch_size, seq_len, _ = x.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # إضافة تحيز ALiBi (بدلًا من أي ترميز موضعي على Q,K)
        alibi_bias = self._get_alibi_bias(seq_len, x.device).unsqueeze(0)
        scores = scores + alibi_bias

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 3. Grouped-Query Attention (GQA) — يشمل MQA كحالة خاصة
# ============================================================
class GroupedQueryAttention(nn.Module):
    """
    GQA: تقسيم رؤوس Query إلى مجموعات، كل مجموعة تتشارك K,V.
    num_kv_groups=1  → مكافئ MQA
    num_kv_groups=num_heads → مكافئ MHA القياسي
    """
    def __init__(self, d_model, num_heads, num_kv_groups, max_seq_len=8192, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads يجب أن يُقسَم على num_kv_groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_groups  # عدد رؤوس Q لكل مجموعة K,V

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # ملاحظة: K,V بأبعاد أصغر بكثير — فقط num_kv_groups بدلًا من num_heads
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # تكرار K,V لتُطابق عدد رؤوس Q (كل مجموعة K,V تُستخدَم لعدة رؤوس Q)
        K = K.repeat_interleave(self.group_size, dim=1)  # (batch, num_heads, seq_len, d_k)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

def compute_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_len,
                            batch_size=1, bytes_per_element=2):
    """حساب حجم KV Cache بالجيجابايت لمقارنة MHA/GQA/MQA."""
    # العامل 2: K و V منفصلان
    total_bytes = (2 * num_layers * num_kv_heads * head_dim *
                   seq_len * batch_size * bytes_per_element)
    return total_bytes / (1024 ** 3)  # تحويل لجيجابايت

# ============================================================
# 4. Flash Attention — استخدام التطبيق الرسمي المُحسَّن (PyTorch 2.0+)
# ============================================================
def flash_attention_demo(Q, K, V, is_causal=True):
    """
    PyTorch 2.0+ يوفر scaled_dot_product_attention المُحسَّنة
    التي تستخدم Flash Attention تلقائيًا عند توفر العتاد المناسب.
    """
    # هذه الدالة تختار تلقائيًا أسرع تطبيق متاح (Flash/Memory-Efficient/Math)
    output = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=is_causal, dropout_p=0.0
    )
    return output

# مقارنة عملية: التنفيذ اليدوي مقابل Flash Attention المدمج
def compare_standard_vs_flash(seq_len=4096, d_model=512, num_heads=8, device='cuda'):
    import time

    batch_size = 2
    d_k = d_model // num_heads
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)

    # التنفيذ القياسي (يدوي، يخزّن مصفوفة n×n كاملة)
    def standard_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_standard = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_flash = flash_attention_demo(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    flash_time = time.time() - start
    flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Standard: {standard_time*1000:.2f}ms, {standard_memory:.2f}MB")
    print(f"Flash:    {flash_time*1000:.2f}ms, {flash_memory:.2f}MB")
    print(f"تسريع: {standard_time/flash_time:.2f}x | توفير ذاكرة: {standard_memory/flash_memory:.2f}x")

# ============================================================
# 5. مفهوم Ring Attention المبسّط (توزيع عبر عدة أجهزة)
# ============================================================
def ring_attention_concept_demo(num_devices: int, total_seq_len: int, d_model: int):
    """
    توضيح مفاهيمي مبسّط لفكرة Ring Attention (التنفيذ الفعلي
    يتطلب أدوات توزيع مثل torch.distributed وNCCL):

    1. يُقسَّم التسلسل الطويل إلى num_devices أجزاء متساوية
    2. كل جهاز يحتفظ بجزئه من Q محليًا طوال الوقت
    3. أجزاء K,V تدور بين الأجهزة في "حلقة" (ring) عبر num_devices خطوة
    4. في كل خطوة دوران، يحسب كل جهاز انتباه جزئي بين Q المحلي
       وK,V الحالي القادم من الجهاز المجاور، ويُراكم النتيجة
    5. بعد اكتمال الدورة الكاملة، كل جهاز يملك نتيجة انتباه كاملة
       ومصححة لجزئه من Q مقابل كل أجزاء K,V من كل الأجهزة
    """
    chunk_size = total_seq_len // num_devices
    print(f"عدد الأجهزة: {num_devices}")
    print(f"طول كل جزء محلي: {chunk_size} رمز")
    print(f"الذاكرة المطلوبة لكل جهاز لتخزين Q,K,V المحلية فقط: "
          f"~{3 * chunk_size * d_model * 2 / (1024**2):.2f} MB (بدلاً من التسلسل الكامل)")
    print("كل جهاز يحسب تدريجيًا، عبر دورات تبادل، انتباهًا كاملًا دون "
          "الحاجة لتخزين التسلسل الكامل في ذاكرة جهاز واحد")

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch14-modern-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. RoPE (Rotary Positional Embedding) — التطبيق الكامل
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # ترددات مختلفة لكل زوج من الأبعاد: θ_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        # نُكرّر كل تردد مرتين لتغطية الزوج الكامل من الأبعاد
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    """تدوير نصفي: [x1, x2] → [-x2, x1]، خطوة أساسية في تطبيق RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    تطبيق الدوران على Q وK.
    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # للبث على batch وnum_heads
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated

class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-Head Attention مع RoPE بدلاً من الترميز الموضعي الإضافي."""
    def __init__(self, d_model, num_heads, max_seq_len=8192, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # تطبيق RoPE على Q وK فقط (ليس V)
        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 2. ALiBi (Attention with Linear Biases)
# ============================================================
def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    حساب الميول (slopes) لكل رأس، كمتتالية هندسية كما في الورقة الأصلية.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(num_heads))
    else:
        # معالجة الحالات غير القوى الصحيحة لـ 2 (تفصيل من الورقة الأصلية)
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
        return torch.tensor(slopes + extra)

class MultiHeadAttentionWithALiBi(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        self.dropout = nn.Dropout(dropout)

    def _get_alibi_bias(self, seq_len, device):
        # مصفوفة المسافات بين كل زوج مواضع: (seq_len, seq_len)
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # نريد فقط المسافات غير الموجبة (causal) لتعكس البعد للخلف
        relative_positions = -relative_positions.abs().float()

        # لكل رأس: ميل خاص × مصفوفة المسافات
        alibi_bias = self.slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)
        return alibi_bias  # (num_heads, seq_len, seq_len)

    def forward(self, x, causal_mask=None):
        batch_size, seq_len, _ = x.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # إضافة تحيز ALiBi (بدلًا من أي ترميز موضعي على Q,K)
        alibi_bias = self._get_alibi_bias(seq_len, x.device).unsqueeze(0)
        scores = scores + alibi_bias

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)

# ============================================================
# 3. Grouped-Query Attention (GQA) — يشمل MQA كحالة خاصة
# ============================================================
class GroupedQueryAttention(nn.Module):
    """
    GQA: تقسيم رؤوس Query إلى مجموعات، كل مجموعة تتشارك K,V.
    num_kv_groups=1  → مكافئ MQA
    num_kv_groups=num_heads → مكافئ MHA القياسي
    """
    def __init__(self, d_model, num_heads, num_kv_groups, max_seq_len=8192, dropout=0.1):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads يجب أن يُقسَم على num_kv_groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_groups  # عدد رؤوس Q لكل مجموعة K,V

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # ملاحظة: K,V بأبعاد أصغر بكثير — فقط num_kv_groups بدلًا من num_heads
        self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_groups, self.d_k).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # تكرار K,V لتُطابق عدد رؤوس Q (كل مجموعة K,V تُستخدَم لعدة رؤوس Q)
        K = K.repeat_interleave(self.group_size, dim=1)  # (batch, num_heads, seq_len, d_k)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

def compute_kv_cache_size(num_layers, num_kv_heads, head_dim, seq_len,
                            batch_size=1, bytes_per_element=2):
    """حساب حجم KV Cache بالجيجابايت لمقارنة MHA/GQA/MQA."""
    # العامل 2: K و V منفصلان
    total_bytes = (2 * num_layers * num_kv_heads * head_dim *
                   seq_len * batch_size * bytes_per_element)
    return total_bytes / (1024 ** 3)  # تحويل لجيجابايت

# ============================================================
# 4. Flash Attention — استخدام التطبيق الرسمي المُحسَّن (PyTorch 2.0+)
# ============================================================
def flash_attention_demo(Q, K, V, is_causal=True):
    """
    PyTorch 2.0+ يوفر scaled_dot_product_attention المُحسَّنة
    التي تستخدم Flash Attention تلقائيًا عند توفر العتاد المناسب.
    """
    # هذه الدالة تختار تلقائيًا أسرع تطبيق متاح (Flash/Memory-Efficient/Math)
    output = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=is_causal, dropout_p=0.0
    )
    return output

# مقارنة عملية: التنفيذ اليدوي مقابل Flash Attention المدمج
def compare_standard_vs_flash(seq_len=4096, d_model=512, num_heads=8, device='cuda'):
    import time

    batch_size = 2
    d_k = d_model // num_heads
    Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device, dtype=torch.float16)

    # التنفيذ القياسي (يدوي، يخزّن مصفوفة n×n كاملة)
    def standard_attention(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_standard = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    standard_time = time.time() - start
    standard_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    out_flash = flash_attention_demo(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    flash_time = time.time() - start
    flash_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Standard: {standard_time*1000:.2f}ms, {standard_memory:.2f}MB")
    print(f"Flash:    {flash_time*1000:.2f}ms, {flash_memory:.2f}MB")
    print(f"تسريع: {standard_time/flash_time:.2f}x | توفير ذاكرة: {standard_memory/flash_memory:.2f}x")

# ============================================================
# 5. مفهوم Ring Attention المبسّط (توزيع عبر عدة أجهزة)
# ============================================================
def ring_attention_concept_demo(num_devices: int, total_seq_len: int, d_model: int):
    """
    توضيح مفاهيمي مبسّط لفكرة Ring Attention (التنفيذ الفعلي
    يتطلب أدوات توزيع مثل torch.distributed وNCCL):

    1. يُقسَّم التسلسل الطويل إلى num_devices أجزاء متساوية
    2. كل جهاز يحتفظ بجزئه من Q محليًا طوال الوقت
    3. أجزاء K,V تدور بين الأجهزة في "حلقة" (ring) عبر num_devices خطوة
    4. في كل خطوة دوران، يحسب كل جهاز انتباه جزئي بين Q المحلي
       وK,V الحالي القادم من الجهاز المجاور، ويُراكم النتيجة
    5. بعد اكتمال الدورة الكاملة، كل جهاز يملك نتيجة انتباه كاملة
       ومصححة لجزئه من Q مقابل كل أجزاء K,V من كل الأجهزة
    """
    chunk_size = total_seq_len // num_devices
    print(f"عدد الأجهزة: {num_devices}")
    print(f"طول كل جزء محلي: {chunk_size} رمز")
    print(f"الذاكرة المطلوبة لكل جهاز لتخزين Q,K,V المحلية فقط: "
          f"~{3 * chunk_size * d_model * 2 / (1024**2):.2f} MB (بدلاً من التسلسل الكامل)")
    print("كل جهاز يحسب تدريجيًا، عبر دورات تبادل، انتباهًا كاملًا دون "
          "الحاجة لتخزين التسلسل الكامل في ذاكرة جهاز واحد")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-14-ch14-modern-attention.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
