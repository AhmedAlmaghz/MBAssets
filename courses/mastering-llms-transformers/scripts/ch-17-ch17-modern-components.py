#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المكوّنات الحديثة: RMSNorm وSwiGLU
Mastering Transformers and Large Language Models — Modern Components

Book slug : mastering-llms-transformers
Chapter   : ch-17-ch17-modern-components (lesson 17)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-17-ch17-modern-components.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المكوّنات الحديثة: RMSNorm وSwiGLU
Mastering Transformers and Large Language Models — Modern Components

Book slug : mastering-llms-transformers
Chapter   : ch-17-ch17-modern-components (lesson 17)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-17-ch17-modern-components.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المكوّنات الحديثة: RMSNorm وSwiGLU
Mastering Transformers and Large Language Models — Modern Components

Book slug : mastering-llms-transformers
Chapter   : ch-17-ch17-modern-components (lesson 17)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch17-modern-components.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch17-modern-components.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. RMSNorm من الصفر
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

# ============================================================
# 2. SwiGLU FeedForward Network من الصفر
# ============================================================
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            # الصيغة القياسية في LLaMA: 2/3 من 4×d_model، مقرَّبة لأقرب مضاعف لـ 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 256 * ((d_ff + 255) // 256)

        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # بوابة SiLU
        self.W3 = nn.Linear(d_model, d_ff, bias=False)   # المحتوى
        self.W2 = nn.Linear(d_ff, d_model, bias=False)   # الإسقاط النهائي
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W1(x))       # SiLU(xW1)
        content = self.W3(x)            # xW3
        hidden = gate * content         # الضرب العنصري (البوابة)
        return self.dropout(self.W2(hidden))

# ============================================================
# 3. ModernTransformerBlock — المعمارية المرجعية لنماذج 2023+
#    (يدمج كل مكونات الجزء الرابع: RoPE + GQA + RMSNorm + SwiGLU)
# ============================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer حديثة بأسلوب LLaMA/Mistral/Qwen:
    - RMSNorm بدلاً من LayerNorm
    - GQA مع RoPE بدلاً من MHA التقليدي
    - SwiGLU بدلاً من ReLU/GELU FFN
    - Pre-Norm (كما في الفصل 11)
    """
    def __init__(self, d_model, num_heads, num_kv_groups, d_ff=None,
                 max_seq_len=8192, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(   # من الفصل 14
            d_model, num_heads, num_kv_groups, max_seq_len, dropout
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ModernLanguageModel(nn.Module):
    """نموذج لغوي كامل بالمعمارية الحديثة المرجعية (أسلوب LLaMA)."""
    def __init__(self, vocab_size, d_model=4096, num_layers=32,
                 num_heads=32, num_kv_groups=8, d_ff=None, max_seq_len=8192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, num_kv_groups, d_ff, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)  # من الفصل 10

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مقارنة عملية لسرعة RMSNorm مقابل LayerNorm
# ============================================================
def compare_norm_speed(d_model=4096, batch_size=32, seq_len=2048,
                         num_iterations=1000, device='cuda'):
    import time

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    layer_norm = nn.LayerNorm(d_model).to(device)
    rms_norm = RMSNorm(d_model).to(device)

    # تسخين (warmup) لضمان قياس دقيق
    for _ in range(10):
        _ = layer_norm(x)
        _ = rms_norm(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    layer_norm_time = time.time() - start

    start = time.time()
    for _ in range(num_iterations):
        _ = rms_norm(x)
    torch.cuda.synchronize()
    rms_norm_time = time.time() - start

    print(f"LayerNorm: {layer_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"RMSNorm:   {rms_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"نسبة التسريع: {layer_norm_time/rms_norm_time:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch17-modern-components.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. RMSNorm من الصفر
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

# ============================================================
# 2. SwiGLU FeedForward Network من الصفر
# ============================================================
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            # الصيغة القياسية في LLaMA: 2/3 من 4×d_model، مقرَّبة لأقرب مضاعف لـ 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 256 * ((d_ff + 255) // 256)

        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # بوابة SiLU
        self.W3 = nn.Linear(d_model, d_ff, bias=False)   # المحتوى
        self.W2 = nn.Linear(d_ff, d_model, bias=False)   # الإسقاط النهائي
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W1(x))       # SiLU(xW1)
        content = self.W3(x)            # xW3
        hidden = gate * content         # الضرب العنصري (البوابة)
        return self.dropout(self.W2(hidden))

# ============================================================
# 3. ModernTransformerBlock — المعمارية المرجعية لنماذج 2023+
#    (يدمج كل مكونات الجزء الرابع: RoPE + GQA + RMSNorm + SwiGLU)
# ============================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer حديثة بأسلوب LLaMA/Mistral/Qwen:
    - RMSNorm بدلاً من LayerNorm
    - GQA مع RoPE بدلاً من MHA التقليدي
    - SwiGLU بدلاً من ReLU/GELU FFN
    - Pre-Norm (كما في الفصل 11)
    """
    def __init__(self, d_model, num_heads, num_kv_groups, d_ff=None,
                 max_seq_len=8192, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(   # من الفصل 14
            d_model, num_heads, num_kv_groups, max_seq_len, dropout
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ModernLanguageModel(nn.Module):
    """نموذج لغوي كامل بالمعمارية الحديثة المرجعية (أسلوب LLaMA)."""
    def __init__(self, vocab_size, d_model=4096, num_layers=32,
                 num_heads=32, num_kv_groups=8, d_ff=None, max_seq_len=8192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, num_kv_groups, d_ff, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)  # من الفصل 10

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مقارنة عملية لسرعة RMSNorm مقابل LayerNorm
# ============================================================
def compare_norm_speed(d_model=4096, batch_size=32, seq_len=2048,
                         num_iterations=1000, device='cuda'):
    import time

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    layer_norm = nn.LayerNorm(d_model).to(device)
    rms_norm = RMSNorm(d_model).to(device)

    # تسخين (warmup) لضمان قياس دقيق
    for _ in range(10):
        _ = layer_norm(x)
        _ = rms_norm(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    layer_norm_time = time.time() - start

    start = time.time()
    for _ in range(num_iterations):
        _ = rms_norm(x)
    torch.cuda.synchronize()
    rms_norm_time = time.time() - start

    print(f"LayerNorm: {layer_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"RMSNorm:   {rms_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"نسبة التسريع: {layer_norm_time/rms_norm_time:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — المكوّنات الحديثة: RMSNorm وSwiGLU
Mastering Transformers and Large Language Models — Modern Components

Book slug : mastering-llms-transformers
Chapter   : ch-17-ch17-modern-components (lesson 17)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-04-modern-architectures/ch17-modern-components.ipynb#cell-1
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
# [002] content/notebooks/part-04-modern-architectures/ch17-modern-components.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. RMSNorm من الصفر
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

# ============================================================
# 2. SwiGLU FeedForward Network من الصفر
# ============================================================
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            # الصيغة القياسية في LLaMA: 2/3 من 4×d_model، مقرَّبة لأقرب مضاعف لـ 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 256 * ((d_ff + 255) // 256)

        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # بوابة SiLU
        self.W3 = nn.Linear(d_model, d_ff, bias=False)   # المحتوى
        self.W2 = nn.Linear(d_ff, d_model, bias=False)   # الإسقاط النهائي
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W1(x))       # SiLU(xW1)
        content = self.W3(x)            # xW3
        hidden = gate * content         # الضرب العنصري (البوابة)
        return self.dropout(self.W2(hidden))

# ============================================================
# 3. ModernTransformerBlock — المعمارية المرجعية لنماذج 2023+
#    (يدمج كل مكونات الجزء الرابع: RoPE + GQA + RMSNorm + SwiGLU)
# ============================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer حديثة بأسلوب LLaMA/Mistral/Qwen:
    - RMSNorm بدلاً من LayerNorm
    - GQA مع RoPE بدلاً من MHA التقليدي
    - SwiGLU بدلاً من ReLU/GELU FFN
    - Pre-Norm (كما في الفصل 11)
    """
    def __init__(self, d_model, num_heads, num_kv_groups, d_ff=None,
                 max_seq_len=8192, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(   # من الفصل 14
            d_model, num_heads, num_kv_groups, max_seq_len, dropout
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ModernLanguageModel(nn.Module):
    """نموذج لغوي كامل بالمعمارية الحديثة المرجعية (أسلوب LLaMA)."""
    def __init__(self, vocab_size, d_model=4096, num_layers=32,
                 num_heads=32, num_kv_groups=8, d_ff=None, max_seq_len=8192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, num_kv_groups, d_ff, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)  # من الفصل 10

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مقارنة عملية لسرعة RMSNorm مقابل LayerNorm
# ============================================================
def compare_norm_speed(d_model=4096, batch_size=32, seq_len=2048,
                         num_iterations=1000, device='cuda'):
    import time

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    layer_norm = nn.LayerNorm(d_model).to(device)
    rms_norm = RMSNorm(d_model).to(device)

    # تسخين (warmup) لضمان قياس دقيق
    for _ in range(10):
        _ = layer_norm(x)
        _ = rms_norm(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    layer_norm_time = time.time() - start

    start = time.time()
    for _ in range(num_iterations):
        _ = rms_norm(x)
    torch.cuda.synchronize()
    rms_norm_time = time.time() - start

    print(f"LayerNorm: {layer_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"RMSNorm:   {rms_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"نسبة التسريع: {layer_norm_time/rms_norm_time:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-04-modern-architectures/ch17-modern-components.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. RMSNorm من الصفر
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt(mean(x²) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

# ============================================================
# 2. SwiGLU FeedForward Network من الصفر
# ============================================================
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            # الصيغة القياسية في LLaMA: 2/3 من 4×d_model، مقرَّبة لأقرب مضاعف لـ 256
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = 256 * ((d_ff + 255) // 256)

        self.W1 = nn.Linear(d_model, d_ff, bias=False)   # بوابة SiLU
        self.W3 = nn.Linear(d_model, d_ff, bias=False)   # المحتوى
        self.W2 = nn.Linear(d_ff, d_model, bias=False)   # الإسقاط النهائي
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W1(x))       # SiLU(xW1)
        content = self.W3(x)            # xW3
        hidden = gate * content         # الضرب العنصري (البوابة)
        return self.dropout(self.W2(hidden))

# ============================================================
# 3. ModernTransformerBlock — المعمارية المرجعية لنماذج 2023+
#    (يدمج كل مكونات الجزء الرابع: RoPE + GQA + RMSNorm + SwiGLU)
# ============================================================
class ModernTransformerBlock(nn.Module):
    """
    كتلة Transformer حديثة بأسلوب LLaMA/Mistral/Qwen:
    - RMSNorm بدلاً من LayerNorm
    - GQA مع RoPE بدلاً من MHA التقليدي
    - SwiGLU بدلاً من ReLU/GELU FFN
    - Pre-Norm (كما في الفصل 11)
    """
    def __init__(self, d_model, num_heads, num_kv_groups, d_ff=None,
                 max_seq_len=8192, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(   # من الفصل 14
            d_model, num_heads, num_kv_groups, max_seq_len, dropout
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ModernLanguageModel(nn.Module):
    """نموذج لغوي كامل بالمعمارية الحديثة المرجعية (أسلوب LLaMA)."""
    def __init__(self, vocab_size, d_model=4096, num_layers=32,
                 num_heads=32, num_kv_groups=8, d_ff=None, max_seq_len=8192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, num_heads, num_kv_groups, d_ff, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)  # من الفصل 10

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================
# 4. مقارنة عملية لسرعة RMSNorm مقابل LayerNorm
# ============================================================
def compare_norm_speed(d_model=4096, batch_size=32, seq_len=2048,
                         num_iterations=1000, device='cuda'):
    import time

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    layer_norm = nn.LayerNorm(d_model).to(device)
    rms_norm = RMSNorm(d_model).to(device)

    # تسخين (warmup) لضمان قياس دقيق
    for _ in range(10):
        _ = layer_norm(x)
        _ = rms_norm(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    layer_norm_time = time.time() - start

    start = time.time()
    for _ in range(num_iterations):
        _ = rms_norm(x)
    torch.cuda.synchronize()
    rms_norm_time = time.time() - start

    print(f"LayerNorm: {layer_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"RMSNorm:   {rms_norm_time*1000:.2f}ms لـ {num_iterations} تكرار")
    print(f"نسبة التسريع: {layer_norm_time/rms_norm_time:.2f}x")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-17-ch17-modern-components.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
