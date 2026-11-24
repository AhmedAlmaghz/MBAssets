#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي والتطبيع وبنية المحوّل الكاملة
Mastering Transformers and Large Language Models — Transformer Architecture

Book slug : mastering-llms-transformers
Chapter   : ch-11-ch11-transformer-architecture (lesson 11)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-11-ch11-transformer-architecture.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي والتطبيع وبنية المحوّل الكاملة
Mastering Transformers and Large Language Models — Transformer Architecture

Book slug : mastering-llms-transformers
Chapter   : ch-11-ch11-transformer-architecture (lesson 11)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-11-ch11-transformer-architecture.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي والتطبيع وبنية المحوّل الكاملة
Mastering Transformers and Large Language Models — Transformer Architecture

Book slug : mastering-llms-transformers
Chapter   : ch-11-ch11-transformer-architecture (lesson 11)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch11-transformer-architecture.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch11-transformer-architecture.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Positional Encoding الجيبي الكامل
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model) محسوبة بصيغة أسية لأسباب استقرار عددي
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية

        pe = pe.unsqueeze(0)   # (1, max_len, d_model) لتسهيل البث على الدفعة
        self.register_buffer('pe', pe)   # ليست معاملًا قابلًا للتعلم، لكنها تُحفظ مع النموذج

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (batch, seq_len, d_model) """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ============================================================
# 2. Layer Normalization من الصفر (لفهم الآلية حرفيًا)
# ============================================================
class LayerNormFromScratch(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

# ============================================================
# 3. Position-wise Feed-Forward Network
# ============================================================
class PositionWiseFeedForward(nn.Module):
    """
    شبكة تغذية أمامية تُطبَّق بشكل مستقل على كل موضع (position)
    بنفس الأوزان، عادة بتوسيع البعد 4 أضعاف ثم إعادته.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ============================================================
# 4. طبقة Encoder كاملة (Pre-LN، النمط الحديث الأكثر استقرارًا)
# ============================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        from_scratch_mha = MultiHeadAttention  # من الفصل العاشر
        self.self_attn = from_scratch_mha(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Pre-LN Self-Attention + Residual ---
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout1(attn_output)

        # --- Pre-LN FeedForward + Residual ---
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout2(ff_output)

        return x, attn_weights

# ============================================================
# 5. طبقة Decoder كاملة (Masked Self-Attn + Cross-Attn + FFN)
# ============================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.norm3 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Masked Self-Attention (سببي)
        normed_x = self.norm1(x)
        self_attn_out, _ = self.masked_self_attn(normed_x, normed_x, normed_x, self_mask)
        x = x + self.dropout1(self_attn_out)

        # Cross-Attention: Q من الـ Decoder، K,V من الـ Encoder
        normed_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            normed_x, encoder_output, encoder_output, cross_mask
        )
        x = x + self.dropout2(cross_attn_out)

        # FeedForward
        normed_x = self.norm3(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout3(ff_out)

        return x, cross_attn_weights

# ============================================================
# 6. معمارية Transformer الكاملة (Encoder-Decoder)
# ============================================================
class FullTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNormFromScratch(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)  # تحجيم تقليدي من الورقة الأصلية
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask)
        return self.final_norm(x)

    def decode(self, tgt, encoder_output, tgt_mask, cross_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, encoder_output, tgt_mask, cross_mask)
        return self.final_norm(x), cross_attn

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output, cross_attn = self.decode(tgt, encoder_output, tgt_mask, cross_mask)
        logits = self.output_projection(decoder_output)
        return logits, cross_attn

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch11-transformer-architecture.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Positional Encoding الجيبي الكامل
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model) محسوبة بصيغة أسية لأسباب استقرار عددي
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية

        pe = pe.unsqueeze(0)   # (1, max_len, d_model) لتسهيل البث على الدفعة
        self.register_buffer('pe', pe)   # ليست معاملًا قابلًا للتعلم، لكنها تُحفظ مع النموذج

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (batch, seq_len, d_model) """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ============================================================
# 2. Layer Normalization من الصفر (لفهم الآلية حرفيًا)
# ============================================================
class LayerNormFromScratch(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

# ============================================================
# 3. Position-wise Feed-Forward Network
# ============================================================
class PositionWiseFeedForward(nn.Module):
    """
    شبكة تغذية أمامية تُطبَّق بشكل مستقل على كل موضع (position)
    بنفس الأوزان، عادة بتوسيع البعد 4 أضعاف ثم إعادته.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ============================================================
# 4. طبقة Encoder كاملة (Pre-LN، النمط الحديث الأكثر استقرارًا)
# ============================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        from_scratch_mha = MultiHeadAttention  # من الفصل العاشر
        self.self_attn = from_scratch_mha(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Pre-LN Self-Attention + Residual ---
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout1(attn_output)

        # --- Pre-LN FeedForward + Residual ---
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout2(ff_output)

        return x, attn_weights

# ============================================================
# 5. طبقة Decoder كاملة (Masked Self-Attn + Cross-Attn + FFN)
# ============================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.norm3 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Masked Self-Attention (سببي)
        normed_x = self.norm1(x)
        self_attn_out, _ = self.masked_self_attn(normed_x, normed_x, normed_x, self_mask)
        x = x + self.dropout1(self_attn_out)

        # Cross-Attention: Q من الـ Decoder، K,V من الـ Encoder
        normed_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            normed_x, encoder_output, encoder_output, cross_mask
        )
        x = x + self.dropout2(cross_attn_out)

        # FeedForward
        normed_x = self.norm3(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout3(ff_out)

        return x, cross_attn_weights

# ============================================================
# 6. معمارية Transformer الكاملة (Encoder-Decoder)
# ============================================================
class FullTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNormFromScratch(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)  # تحجيم تقليدي من الورقة الأصلية
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask)
        return self.final_norm(x)

    def decode(self, tgt, encoder_output, tgt_mask, cross_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, encoder_output, tgt_mask, cross_mask)
        return self.final_norm(x), cross_attn

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output, cross_attn = self.decode(tgt, encoder_output, tgt_mask, cross_mask)
        logits = self.output_projection(decoder_output)
        return logits, cross_attn

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الترميز الموضعي والتطبيع وبنية المحوّل الكاملة
Mastering Transformers and Large Language Models — Transformer Architecture

Book slug : mastering-llms-transformers
Chapter   : ch-11-ch11-transformer-architecture (lesson 11)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch11-transformer-architecture.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch11-transformer-architecture.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Positional Encoding الجيبي الكامل
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model) محسوبة بصيغة أسية لأسباب استقرار عددي
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية

        pe = pe.unsqueeze(0)   # (1, max_len, d_model) لتسهيل البث على الدفعة
        self.register_buffer('pe', pe)   # ليست معاملًا قابلًا للتعلم، لكنها تُحفظ مع النموذج

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (batch, seq_len, d_model) """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ============================================================
# 2. Layer Normalization من الصفر (لفهم الآلية حرفيًا)
# ============================================================
class LayerNormFromScratch(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

# ============================================================
# 3. Position-wise Feed-Forward Network
# ============================================================
class PositionWiseFeedForward(nn.Module):
    """
    شبكة تغذية أمامية تُطبَّق بشكل مستقل على كل موضع (position)
    بنفس الأوزان، عادة بتوسيع البعد 4 أضعاف ثم إعادته.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ============================================================
# 4. طبقة Encoder كاملة (Pre-LN، النمط الحديث الأكثر استقرارًا)
# ============================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        from_scratch_mha = MultiHeadAttention  # من الفصل العاشر
        self.self_attn = from_scratch_mha(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Pre-LN Self-Attention + Residual ---
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout1(attn_output)

        # --- Pre-LN FeedForward + Residual ---
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout2(ff_output)

        return x, attn_weights

# ============================================================
# 5. طبقة Decoder كاملة (Masked Self-Attn + Cross-Attn + FFN)
# ============================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.norm3 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Masked Self-Attention (سببي)
        normed_x = self.norm1(x)
        self_attn_out, _ = self.masked_self_attn(normed_x, normed_x, normed_x, self_mask)
        x = x + self.dropout1(self_attn_out)

        # Cross-Attention: Q من الـ Decoder، K,V من الـ Encoder
        normed_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            normed_x, encoder_output, encoder_output, cross_mask
        )
        x = x + self.dropout2(cross_attn_out)

        # FeedForward
        normed_x = self.norm3(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout3(ff_out)

        return x, cross_attn_weights

# ============================================================
# 6. معمارية Transformer الكاملة (Encoder-Decoder)
# ============================================================
class FullTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNormFromScratch(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)  # تحجيم تقليدي من الورقة الأصلية
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask)
        return self.final_norm(x)

    def decode(self, tgt, encoder_output, tgt_mask, cross_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, encoder_output, tgt_mask, cross_mask)
        return self.final_norm(x), cross_attn

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output, cross_attn = self.decode(tgt, encoder_output, tgt_mask, cross_mask)
        logits = self.output_projection(decoder_output)
        return logits, cross_attn

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch11-transformer-architecture.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. Positional Encoding الجيبي الكامل
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/d_model) محسوبة بصيغة أسية لأسباب استقرار عددي
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # الأبعاد الزوجية
        pe[:, 1::2] = torch.cos(position * div_term)   # الأبعاد الفردية

        pe = pe.unsqueeze(0)   # (1, max_len, d_model) لتسهيل البث على الدفعة
        self.register_buffer('pe', pe)   # ليست معاملًا قابلًا للتعلم، لكنها تُحفظ مع النموذج

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (batch, seq_len, d_model) """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# ============================================================
# 2. Layer Normalization من الصفر (لفهم الآلية حرفيًا)
# ============================================================
class LayerNormFromScratch(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

# ============================================================
# 3. Position-wise Feed-Forward Network
# ============================================================
class PositionWiseFeedForward(nn.Module):
    """
    شبكة تغذية أمامية تُطبَّق بشكل مستقل على كل موضع (position)
    بنفس الأوزان، عادة بتوسيع البعد 4 أضعاف ثم إعادته.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

# ============================================================
# 4. طبقة Encoder كاملة (Pre-LN، النمط الحديث الأكثر استقرارًا)
# ============================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        from_scratch_mha = MultiHeadAttention  # من الفصل العاشر
        self.self_attn = from_scratch_mha(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # --- Pre-LN Self-Attention + Residual ---
        normed_x = self.norm1(x)
        attn_output, attn_weights = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout1(attn_output)

        # --- Pre-LN FeedForward + Residual ---
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout2(ff_output)

        return x, attn_weights

# ============================================================
# 5. طبقة Decoder كاملة (Masked Self-Attn + Cross-Attn + FFN)
# ============================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormFromScratch(d_model)
        self.norm2 = LayerNormFromScratch(d_model)
        self.norm3 = LayerNormFromScratch(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Masked Self-Attention (سببي)
        normed_x = self.norm1(x)
        self_attn_out, _ = self.masked_self_attn(normed_x, normed_x, normed_x, self_mask)
        x = x + self.dropout1(self_attn_out)

        # Cross-Attention: Q من الـ Decoder، K,V من الـ Encoder
        normed_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            normed_x, encoder_output, encoder_output, cross_mask
        )
        x = x + self.dropout2(cross_attn_out)

        # FeedForward
        normed_x = self.norm3(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout3(ff_out)

        return x, cross_attn_weights

# ============================================================
# 6. معمارية Transformer الكاملة (Encoder-Decoder)
# ============================================================
class FullTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNormFromScratch(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)  # تحجيم تقليدي من الورقة الأصلية
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x, _ = layer(x, src_mask)
        return self.final_norm(x)

    def decode(self, tgt, encoder_output, tgt_mask, cross_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, encoder_output, tgt_mask, cross_mask)
        return self.final_norm(x), cross_attn

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output, cross_attn = self.decode(tgt, encoder_output, tgt_mask, cross_mask)
        logits = self.output_projection(decoder_output)
        return logits, cross_attn

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-11-ch11-transformer-architecture.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
