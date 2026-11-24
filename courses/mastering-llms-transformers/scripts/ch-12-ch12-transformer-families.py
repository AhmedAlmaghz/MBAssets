#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — عائلات المحوّلات: BERT وGPT وT5
Mastering Transformers and Large Language Models — Transformer Families

Book slug : mastering-llms-transformers
Chapter   : ch-12-ch12-transformer-families (lesson 12)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-12-ch12-transformer-families.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — عائلات المحوّلات: BERT وGPT وT5
Mastering Transformers and Large Language Models — Transformer Families

Book slug : mastering-llms-transformers
Chapter   : ch-12-ch12-transformer-families (lesson 12)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-12-ch12-transformer-families.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — عائلات المحوّلات: BERT وGPT وT5
Mastering Transformers and Large Language Models — Transformer Families

Book slug : mastering-llms-transformers
Chapter   : ch-12-ch12-transformer-families (lesson 12)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch12-transformer-families.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch12-transformer-families.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. MiniBERT — نموذج Encoder-only مصغّر كامل
# ============================================================
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, d_model)   # لجملة A أو B (NSP)
        self.position_embedding = nn.Embedding(max_len, d_model)  # Learned PE
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # رؤوس التدريب المسبق
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, segment_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = (self.token_embedding(input_ids)
             + self.segment_embedding(segment_ids)
             + self.position_embedding(positions))
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x, _ = layer(x, attention_mask)

        # رمز [CLS] هو أول موضع، يُستخدم لـ NSP/التصنيف
        cls_representation = x[:, 0, :]

        mlm_logits = self.mlm_head(x)              # (batch, seq_len, vocab_size)
        nsp_logits = self.nsp_head(cls_representation)  # (batch, 2)

        return mlm_logits, nsp_logits

def mask_tokens_for_mlm(input_ids, vocab_size, mask_token_id,
                         special_token_ids, mlm_probability=0.15):
    """
    تطبيق استراتيجية إخفاء BERT: 80% [MASK], 10% عشوائي, 10% بلا تغيير.
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # لا نُخفي الرموز الخاصة (CLS, SEP, PAD)
    special_mask = torch.isin(input_ids, torch.tensor(special_token_ids))
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # تجاهل الخسارة على الرموز غير المُخفاة

    # 80%: استبدال بـ [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10%: استبدال برمز عشوائي
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                       & masked_indices & ~indices_replaced)
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% المتبقية: تبقى كما هي (لا حاجة لفعل شيء)
    return input_ids, labels

# ============================================================
# 2. MiniGPT — نموذج Decoder-only مصغّر كامل (بالتوليد)
# ============================================================
class GPTBlock(nn.Module):
    """طبقة Decoder مبسّطة بدون Cross-Attention (لا يوجد Encoder)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask):
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, causal_mask)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x, attn_weights

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ربط أوزان التضمين بطبقة الإخراج (weight tying) — تقنية قياسية لتقليل المعاملات
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)

        for block in self.blocks:
            x, _ = block(x, causal_mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        توليد نص ذاتي الانحدار، رمزًا تلو الآخر.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # اقتصاص السياق إذا تجاوز الطول الأقصى
            input_cropped = input_ids[:, -self.max_len:]

            logits = self.forward(input_cropped)
            # نأخذ فقط logits آخر موضع (التنبؤ بالكلمة التالية)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ============================================================
# 3. مفهوم T5 المبسّط — Span Corruption + Text-to-Text
# ============================================================
def create_span_corruption_example(tokens: list, noise_density: float = 0.15,
                                     mean_span_length: float = 3.0,
                                     sentinel_start_id: int = 32000):
    """
    تنفيذ مبسّط لاستراتيجية T5 Span Corruption:
    يختار نطاقات عشوائية متجاورة ويستبدلها برمز sentinel واحد.
    """
    num_tokens = len(tokens)
    num_to_mask = max(1, int(round(num_tokens * noise_density)))

    masked_input = []
    target = []
    sentinel_id = sentinel_start_id

    i = 0
    masked_count = 0
    while i < num_tokens:
        if masked_count < num_to_mask and np.random.random() < (noise_density / mean_span_length):
            span_length = min(
                np.random.poisson(mean_span_length) + 1,
                num_tokens - i
            )
            masked_input.append(sentinel_id)
            target.append(sentinel_id)
            target.extend(tokens[i:i + span_length])
            sentinel_id += 1
            masked_count += span_length
            i += span_length
        else:
            masked_input.append(tokens[i])
            i += 1

    target.append(sentinel_id)  # رمز إنهاء الهدف
    return masked_input, target

def format_as_text_to_text(task_name: str, input_text: str) -> str:
    """تنسيق المدخل بصيغة T5 الموحدة (بادئة المهمة + النص)."""
    task_prefixes = {
        "translation_ar_en": "translate Arabic to English: ",
        "summarization": "summarize: ",
        "sentiment": "sst2 sentence: ",
        "cola": "cola sentence: ",
    }
    prefix = task_prefixes.get(task_name, "")
    return prefix + input_text

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch12-transformer-families.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. MiniBERT — نموذج Encoder-only مصغّر كامل
# ============================================================
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, d_model)   # لجملة A أو B (NSP)
        self.position_embedding = nn.Embedding(max_len, d_model)  # Learned PE
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # رؤوس التدريب المسبق
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, segment_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = (self.token_embedding(input_ids)
             + self.segment_embedding(segment_ids)
             + self.position_embedding(positions))
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x, _ = layer(x, attention_mask)

        # رمز [CLS] هو أول موضع، يُستخدم لـ NSP/التصنيف
        cls_representation = x[:, 0, :]

        mlm_logits = self.mlm_head(x)              # (batch, seq_len, vocab_size)
        nsp_logits = self.nsp_head(cls_representation)  # (batch, 2)

        return mlm_logits, nsp_logits

def mask_tokens_for_mlm(input_ids, vocab_size, mask_token_id,
                         special_token_ids, mlm_probability=0.15):
    """
    تطبيق استراتيجية إخفاء BERT: 80% [MASK], 10% عشوائي, 10% بلا تغيير.
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # لا نُخفي الرموز الخاصة (CLS, SEP, PAD)
    special_mask = torch.isin(input_ids, torch.tensor(special_token_ids))
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # تجاهل الخسارة على الرموز غير المُخفاة

    # 80%: استبدال بـ [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10%: استبدال برمز عشوائي
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                       & masked_indices & ~indices_replaced)
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% المتبقية: تبقى كما هي (لا حاجة لفعل شيء)
    return input_ids, labels

# ============================================================
# 2. MiniGPT — نموذج Decoder-only مصغّر كامل (بالتوليد)
# ============================================================
class GPTBlock(nn.Module):
    """طبقة Decoder مبسّطة بدون Cross-Attention (لا يوجد Encoder)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask):
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, causal_mask)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x, attn_weights

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ربط أوزان التضمين بطبقة الإخراج (weight tying) — تقنية قياسية لتقليل المعاملات
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)

        for block in self.blocks:
            x, _ = block(x, causal_mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        توليد نص ذاتي الانحدار، رمزًا تلو الآخر.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # اقتصاص السياق إذا تجاوز الطول الأقصى
            input_cropped = input_ids[:, -self.max_len:]

            logits = self.forward(input_cropped)
            # نأخذ فقط logits آخر موضع (التنبؤ بالكلمة التالية)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ============================================================
# 3. مفهوم T5 المبسّط — Span Corruption + Text-to-Text
# ============================================================
def create_span_corruption_example(tokens: list, noise_density: float = 0.15,
                                     mean_span_length: float = 3.0,
                                     sentinel_start_id: int = 32000):
    """
    تنفيذ مبسّط لاستراتيجية T5 Span Corruption:
    يختار نطاقات عشوائية متجاورة ويستبدلها برمز sentinel واحد.
    """
    num_tokens = len(tokens)
    num_to_mask = max(1, int(round(num_tokens * noise_density)))

    masked_input = []
    target = []
    sentinel_id = sentinel_start_id

    i = 0
    masked_count = 0
    while i < num_tokens:
        if masked_count < num_to_mask and np.random.random() < (noise_density / mean_span_length):
            span_length = min(
                np.random.poisson(mean_span_length) + 1,
                num_tokens - i
            )
            masked_input.append(sentinel_id)
            target.append(sentinel_id)
            target.extend(tokens[i:i + span_length])
            sentinel_id += 1
            masked_count += span_length
            i += span_length
        else:
            masked_input.append(tokens[i])
            i += 1

    target.append(sentinel_id)  # رمز إنهاء الهدف
    return masked_input, target

def format_as_text_to_text(task_name: str, input_text: str) -> str:
    """تنسيق المدخل بصيغة T5 الموحدة (بادئة المهمة + النص)."""
    task_prefixes = {
        "translation_ar_en": "translate Arabic to English: ",
        "summarization": "summarize: ",
        "sentiment": "sst2 sentence: ",
        "cola": "cola sentence: ",
    }
    prefix = task_prefixes.get(task_name, "")
    return prefix + input_text

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — عائلات المحوّلات: BERT وGPT وT5
Mastering Transformers and Large Language Models — Transformer Families

Book slug : mastering-llms-transformers
Chapter   : ch-12-ch12-transformer-families (lesson 12)
Code files: 3 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-03-transformers/ch12-transformer-families.ipynb#cell-1
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
# [002] content/notebooks/part-03-transformers/ch12-transformer-families.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. MiniBERT — نموذج Encoder-only مصغّر كامل
# ============================================================
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, d_model)   # لجملة A أو B (NSP)
        self.position_embedding = nn.Embedding(max_len, d_model)  # Learned PE
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # رؤوس التدريب المسبق
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, segment_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = (self.token_embedding(input_ids)
             + self.segment_embedding(segment_ids)
             + self.position_embedding(positions))
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x, _ = layer(x, attention_mask)

        # رمز [CLS] هو أول موضع، يُستخدم لـ NSP/التصنيف
        cls_representation = x[:, 0, :]

        mlm_logits = self.mlm_head(x)              # (batch, seq_len, vocab_size)
        nsp_logits = self.nsp_head(cls_representation)  # (batch, 2)

        return mlm_logits, nsp_logits

def mask_tokens_for_mlm(input_ids, vocab_size, mask_token_id,
                         special_token_ids, mlm_probability=0.15):
    """
    تطبيق استراتيجية إخفاء BERT: 80% [MASK], 10% عشوائي, 10% بلا تغيير.
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # لا نُخفي الرموز الخاصة (CLS, SEP, PAD)
    special_mask = torch.isin(input_ids, torch.tensor(special_token_ids))
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # تجاهل الخسارة على الرموز غير المُخفاة

    # 80%: استبدال بـ [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10%: استبدال برمز عشوائي
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                       & masked_indices & ~indices_replaced)
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% المتبقية: تبقى كما هي (لا حاجة لفعل شيء)
    return input_ids, labels

# ============================================================
# 2. MiniGPT — نموذج Decoder-only مصغّر كامل (بالتوليد)
# ============================================================
class GPTBlock(nn.Module):
    """طبقة Decoder مبسّطة بدون Cross-Attention (لا يوجد Encoder)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask):
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, causal_mask)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x, attn_weights

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ربط أوزان التضمين بطبقة الإخراج (weight tying) — تقنية قياسية لتقليل المعاملات
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)

        for block in self.blocks:
            x, _ = block(x, causal_mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        توليد نص ذاتي الانحدار، رمزًا تلو الآخر.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # اقتصاص السياق إذا تجاوز الطول الأقصى
            input_cropped = input_ids[:, -self.max_len:]

            logits = self.forward(input_cropped)
            # نأخذ فقط logits آخر موضع (التنبؤ بالكلمة التالية)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ============================================================
# 3. مفهوم T5 المبسّط — Span Corruption + Text-to-Text
# ============================================================
def create_span_corruption_example(tokens: list, noise_density: float = 0.15,
                                     mean_span_length: float = 3.0,
                                     sentinel_start_id: int = 32000):
    """
    تنفيذ مبسّط لاستراتيجية T5 Span Corruption:
    يختار نطاقات عشوائية متجاورة ويستبدلها برمز sentinel واحد.
    """
    num_tokens = len(tokens)
    num_to_mask = max(1, int(round(num_tokens * noise_density)))

    masked_input = []
    target = []
    sentinel_id = sentinel_start_id

    i = 0
    masked_count = 0
    while i < num_tokens:
        if masked_count < num_to_mask and np.random.random() < (noise_density / mean_span_length):
            span_length = min(
                np.random.poisson(mean_span_length) + 1,
                num_tokens - i
            )
            masked_input.append(sentinel_id)
            target.append(sentinel_id)
            target.extend(tokens[i:i + span_length])
            sentinel_id += 1
            masked_count += span_length
            i += span_length
        else:
            masked_input.append(tokens[i])
            i += 1

    target.append(sentinel_id)  # رمز إنهاء الهدف
    return masked_input, target

def format_as_text_to_text(task_name: str, input_text: str) -> str:
    """تنسيق المدخل بصيغة T5 الموحدة (بادئة المهمة + النص)."""
    task_prefixes = {
        "translation_ar_en": "translate Arabic to English: ",
        "summarization": "summarize: ",
        "sentiment": "sst2 sentence: ",
        "cola": "cola sentence: ",
    }
    prefix = task_prefixes.get(task_name, "")
    return prefix + input_text

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-03-transformers/ch12-transformer-families.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import math

# ============================================================
# 1. MiniBERT — نموذج Encoder-only مصغّر كامل
# ============================================================
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, d_model)   # لجملة A أو B (NSP)
        self.position_embedding = nn.Embedding(max_len, d_model)  # Learned PE
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # رؤوس التدريب المسبق
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, segment_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = (self.token_embedding(input_ids)
             + self.segment_embedding(segment_ids)
             + self.position_embedding(positions))
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x, _ = layer(x, attention_mask)

        # رمز [CLS] هو أول موضع، يُستخدم لـ NSP/التصنيف
        cls_representation = x[:, 0, :]

        mlm_logits = self.mlm_head(x)              # (batch, seq_len, vocab_size)
        nsp_logits = self.nsp_head(cls_representation)  # (batch, 2)

        return mlm_logits, nsp_logits

def mask_tokens_for_mlm(input_ids, vocab_size, mask_token_id,
                         special_token_ids, mlm_probability=0.15):
    """
    تطبيق استراتيجية إخفاء BERT: 80% [MASK], 10% عشوائي, 10% بلا تغيير.
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # لا نُخفي الرموز الخاصة (CLS, SEP, PAD)
    special_mask = torch.isin(input_ids, torch.tensor(special_token_ids))
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # تجاهل الخسارة على الرموز غير المُخفاة

    # 80%: استبدال بـ [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id

    # 10%: استبدال برمز عشوائي
    indices_random = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                       & masked_indices & ~indices_replaced)
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_tokens[indices_random]

    # 10% المتبقية: تبقى كما هي (لا حاجة لفعل شيء)
    return input_ids, labels

# ============================================================
# 2. MiniGPT — نموذج Decoder-only مصغّر كامل (بالتوليد)
# ============================================================
class GPTBlock(nn.Module):
    """طبقة Decoder مبسّطة بدون Cross-Attention (لا يوجد Encoder)."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask):
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, causal_mask)
        x = x + attn_out

        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x, attn_weights

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ربط أوزان التضمين بطبقة الإخراج (weight tying) — تقنية قياسية لتقليل المعاملات
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = create_causal_mask(seq_len).to(input_ids.device)

        for block in self.blocks:
            x, _ = block(x, causal_mask)

        x = self.final_ln(x)
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        توليد نص ذاتي الانحدار، رمزًا تلو الآخر.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # اقتصاص السياق إذا تجاوز الطول الأقصى
            input_cropped = input_ids[:, -self.max_len:]

            logits = self.forward(input_cropped)
            # نأخذ فقط logits آخر موضع (التنبؤ بالكلمة التالية)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ============================================================
# 3. مفهوم T5 المبسّط — Span Corruption + Text-to-Text
# ============================================================
def create_span_corruption_example(tokens: list, noise_density: float = 0.15,
                                     mean_span_length: float = 3.0,
                                     sentinel_start_id: int = 32000):
    """
    تنفيذ مبسّط لاستراتيجية T5 Span Corruption:
    يختار نطاقات عشوائية متجاورة ويستبدلها برمز sentinel واحد.
    """
    num_tokens = len(tokens)
    num_to_mask = max(1, int(round(num_tokens * noise_density)))

    masked_input = []
    target = []
    sentinel_id = sentinel_start_id

    i = 0
    masked_count = 0
    while i < num_tokens:
        if masked_count < num_to_mask and np.random.random() < (noise_density / mean_span_length):
            span_length = min(
                np.random.poisson(mean_span_length) + 1,
                num_tokens - i
            )
            masked_input.append(sentinel_id)
            target.append(sentinel_id)
            target.extend(tokens[i:i + span_length])
            sentinel_id += 1
            masked_count += span_length
            i += span_length
        else:
            masked_input.append(tokens[i])
            i += 1

    target.append(sentinel_id)  # رمز إنهاء الهدف
    return masked_input, target

def format_as_text_to_text(task_name: str, input_text: str) -> str:
    """تنسيق المدخل بصيغة T5 الموحدة (بادئة المهمة + النص)."""
    task_prefixes = {
        "translation_ar_en": "translate Arabic to English: ",
        "summarization": "summarize: ",
        "sentiment": "sst2 sentence: ",
        "cola": "cola sentence: ",
    }
    prefix = task_prefixes.get(task_name, "")
    return prefix + input_text

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-12-ch12-transformer-families.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
