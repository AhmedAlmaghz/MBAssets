#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الصوت والكلام متعدد الوسائط
Mastering Transformers and Large Language Models — Speech Audio

Book slug : mastering-llms-transformers
Chapter   : ch-39-ch39-speech-audio (lesson 39)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-39-ch39-speech-audio.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الصوت والكلام متعدد الوسائط
Mastering Transformers and Large Language Models — Speech Audio

Book slug : mastering-llms-transformers
Chapter   : ch-39-ch39-speech-audio (lesson 39)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-39-ch39-speech-audio.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الصوت والكلام متعدد الوسائط
Mastering Transformers and Large Language Models — Speech Audio

Book slug : mastering-llms-transformers
Chapter   : ch-39-ch39-speech-audio (lesson 39)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-1
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
# [002] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class WhisperStyleEncoder(nn.Module):
    """
    مُرمِّز صوتي: يأخذ Mel-Spectrogram ويُخرج تمثيلات سياقية
    يستخدم Conv1D أولاً لتقليل الطول الزمني، ثم Transformer Encoder عادي
    """
    def __init__(self, n_mels=80, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1500, d_model))  # أقصى طول زمني

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, mel_spectrogram):
        # mel_spectrogram: [batch, n_mels, time]
        x = torch.nn.functional.gelu(self.conv1(mel_spectrogram))
        x = torch.nn.functional.gelu(self.conv2(x))          # تقليل الطول الزمني 2x
        x = x.permute(0, 2, 1)                                # [batch, time/2, d_model]
        x = x + self.pos_embed[:x.shape[1]]
        return self.encoder(x)

class WhisperStyleDecoder(nn.Module):
    """مُفكِّك نصّي عادي (مثل GPT) لكن مع Cross-Attention على مخرجات المُرمِّز الصوتي"""
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, target_ids, encoder_output, causal_mask):
        x = self.token_embed(target_ids)
        out = self.decoder(tgt=x, memory=encoder_output, tgt_mask=causal_mask)
        return self.output_proj(out)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: يستخدم عدة codebooks متتالية، كل واحد يُرمِّز "البقايا" (residual)
    من الـ codebook السابق — يسمح بدقة عالية بعدد رموز معقول
    """
    def __init__(self, num_quantizers=8, codebook_size=1024, dim=256):
        super().__init__()
        self.quantizers = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)
        ])

    def forward(self, z):  # z: [batch, time, dim] — مخرج من مُرمِّز صوتي عصبي (CNN)
        residual = z
        quantized_out = torch.zeros_like(z)
        all_indices = []

        for codebook in self.quantizers:
            # إيجاد أقرب متجه في الـ codebook لكل موضع زمني
            distances = torch.cdist(residual, codebook.weight.unsqueeze(0).expand(z.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)              # [batch, time]
            quantized = codebook(indices)                    # [batch, time, dim]

            quantized_out = quantized_out + quantized
            residual = residual - quantized                  # الباقي يُرمَّز بالـ codebook التالي
            all_indices.append(indices)

        return quantized_out, torch.stack(all_indices, dim=-1)  # [batch, time, num_quantizers]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class WhisperStyleEncoder(nn.Module):
    """
    مُرمِّز صوتي: يأخذ Mel-Spectrogram ويُخرج تمثيلات سياقية
    يستخدم Conv1D أولاً لتقليل الطول الزمني، ثم Transformer Encoder عادي
    """
    def __init__(self, n_mels=80, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1500, d_model))  # أقصى طول زمني

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, mel_spectrogram):
        # mel_spectrogram: [batch, n_mels, time]
        x = torch.nn.functional.gelu(self.conv1(mel_spectrogram))
        x = torch.nn.functional.gelu(self.conv2(x))          # تقليل الطول الزمني 2x
        x = x.permute(0, 2, 1)                                # [batch, time/2, d_model]
        x = x + self.pos_embed[:x.shape[1]]
        return self.encoder(x)

class WhisperStyleDecoder(nn.Module):
    """مُفكِّك نصّي عادي (مثل GPT) لكن مع Cross-Attention على مخرجات المُرمِّز الصوتي"""
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, target_ids, encoder_output, causal_mask):
        x = self.token_embed(target_ids)
        out = self.decoder(tgt=x, memory=encoder_output, tgt_mask=causal_mask)
        return self.output_proj(out)

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: يستخدم عدة codebooks متتالية، كل واحد يُرمِّز "البقايا" (residual)
    من الـ codebook السابق — يسمح بدقة عالية بعدد رموز معقول
    """
    def __init__(self, num_quantizers=8, codebook_size=1024, dim=256):
        super().__init__()
        self.quantizers = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)
        ])

    def forward(self, z):  # z: [batch, time, dim] — مخرج من مُرمِّز صوتي عصبي (CNN)
        residual = z
        quantized_out = torch.zeros_like(z)
        all_indices = []

        for codebook in self.quantizers:
            # إيجاد أقرب متجه في الـ codebook لكل موضع زمني
            distances = torch.cdist(residual, codebook.weight.unsqueeze(0).expand(z.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)              # [batch, time]
            quantized = codebook(indices)                    # [batch, time, dim]

            quantized_out = quantized_out + quantized
            residual = residual - quantized                  # الباقي يُرمَّز بالـ codebook التالي
            all_indices.append(indices)

        return quantized_out, torch.stack(all_indices, dim=-1)  # [batch, time, num_quantizers]

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — الصوت والكلام متعدد الوسائط
Mastering Transformers and Large Language Models — Speech Audio

Book slug : mastering-llms-transformers
Chapter   : ch-39-ch39-speech-audio (lesson 39)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-1
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
# [002] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class WhisperStyleEncoder(nn.Module):
    """
    مُرمِّز صوتي: يأخذ Mel-Spectrogram ويُخرج تمثيلات سياقية
    يستخدم Conv1D أولاً لتقليل الطول الزمني، ثم Transformer Encoder عادي
    """
    def __init__(self, n_mels=80, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1500, d_model))  # أقصى طول زمني

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, mel_spectrogram):
        # mel_spectrogram: [batch, n_mels, time]
        x = torch.nn.functional.gelu(self.conv1(mel_spectrogram))
        x = torch.nn.functional.gelu(self.conv2(x))          # تقليل الطول الزمني 2x
        x = x.permute(0, 2, 1)                                # [batch, time/2, d_model]
        x = x + self.pos_embed[:x.shape[1]]
        return self.encoder(x)

class WhisperStyleDecoder(nn.Module):
    """مُفكِّك نصّي عادي (مثل GPT) لكن مع Cross-Attention على مخرجات المُرمِّز الصوتي"""
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, target_ids, encoder_output, causal_mask):
        x = self.token_embed(target_ids)
        out = self.decoder(tgt=x, memory=encoder_output, tgt_mask=causal_mask)
        return self.output_proj(out)

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: يستخدم عدة codebooks متتالية، كل واحد يُرمِّز "البقايا" (residual)
    من الـ codebook السابق — يسمح بدقة عالية بعدد رموز معقول
    """
    def __init__(self, num_quantizers=8, codebook_size=1024, dim=256):
        super().__init__()
        self.quantizers = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)
        ])

    def forward(self, z):  # z: [batch, time, dim] — مخرج من مُرمِّز صوتي عصبي (CNN)
        residual = z
        quantized_out = torch.zeros_like(z)
        all_indices = []

        for codebook in self.quantizers:
            # إيجاد أقرب متجه في الـ codebook لكل موضع زمني
            distances = torch.cdist(residual, codebook.weight.unsqueeze(0).expand(z.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)              # [batch, time]
            quantized = codebook(indices)                    # [batch, time, dim]

            quantized_out = quantized_out + quantized
            residual = residual - quantized                  # الباقي يُرمَّز بالـ codebook التالي
            all_indices.append(indices)

        return quantized_out, torch.stack(all_indices, dim=-1)  # [batch, time, num_quantizers]

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch39-speech-audio.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class WhisperStyleEncoder(nn.Module):
    """
    مُرمِّز صوتي: يأخذ Mel-Spectrogram ويُخرج تمثيلات سياقية
    يستخدم Conv1D أولاً لتقليل الطول الزمني، ثم Transformer Encoder عادي
    """
    def __init__(self, n_mels=80, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1500, d_model))  # أقصى طول زمني

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, mel_spectrogram):
        # mel_spectrogram: [batch, n_mels, time]
        x = torch.nn.functional.gelu(self.conv1(mel_spectrogram))
        x = torch.nn.functional.gelu(self.conv2(x))          # تقليل الطول الزمني 2x
        x = x.permute(0, 2, 1)                                # [batch, time/2, d_model]
        x = x + self.pos_embed[:x.shape[1]]
        return self.encoder(x)

class WhisperStyleDecoder(nn.Module):
    """مُفكِّك نصّي عادي (مثل GPT) لكن مع Cross-Attention على مخرجات المُرمِّز الصوتي"""
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, target_ids, encoder_output, causal_mask):
        x = self.token_embed(target_ids)
        out = self.decoder(tgt=x, memory=encoder_output, tgt_mask=causal_mask)
        return self.output_proj(out)

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class ResidualVectorQuantizer(nn.Module):
    """
    RVQ: يستخدم عدة codebooks متتالية، كل واحد يُرمِّز "البقايا" (residual)
    من الـ codebook السابق — يسمح بدقة عالية بعدد رموز معقول
    """
    def __init__(self, num_quantizers=8, codebook_size=1024, dim=256):
        super().__init__()
        self.quantizers = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_quantizers)
        ])

    def forward(self, z):  # z: [batch, time, dim] — مخرج من مُرمِّز صوتي عصبي (CNN)
        residual = z
        quantized_out = torch.zeros_like(z)
        all_indices = []

        for codebook in self.quantizers:
            # إيجاد أقرب متجه في الـ codebook لكل موضع زمني
            distances = torch.cdist(residual, codebook.weight.unsqueeze(0).expand(z.shape[0], -1, -1))
            indices = distances.argmin(dim=-1)              # [batch, time]
            quantized = codebook(indices)                    # [batch, time, dim]

            quantized_out = quantized_out + quantized
            residual = residual - quantized                  # الباقي يُرمَّز بالـ codebook التالي
            all_indices.append(indices)

        return quantized_out, torch.stack(all_indices, dim=-1)  # [batch, time, num_quantizers]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-39-ch39-speech-audio.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
