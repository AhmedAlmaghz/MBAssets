#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — النماذج الموحّدة: Any-to-Any
Mastering Transformers and Large Language Models — Unified Multimodal

Book slug : mastering-llms-transformers
Chapter   : ch-40-ch40-unified-multimodal (lesson 40)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-40-ch40-unified-multimodal.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — النماذج الموحّدة: Any-to-Any
Mastering Transformers and Large Language Models — Unified Multimodal

Book slug : mastering-llms-transformers
Chapter   : ch-40-ch40-unified-multimodal (lesson 40)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-40-ch40-unified-multimodal.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — النماذج الموحّدة: Any-to-Any
Mastering Transformers and Large Language Models — Unified Multimodal

Book slug : mastering-llms-transformers
Chapter   : ch-40-ch40-unified-multimodal (lesson 40)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-1
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
# [002] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class UnifiedMultimodalVocabulary:
    """
    توضيح مفاهيمي: كيف نُركّب vocabulary موحّد من عدة وسائط
    (نهج مشابه لما استخدمته Chameleon من Meta)
    """
    def __init__(self, text_vocab_size=32000,
                 image_codebook_size=8192,    # من VQ-GAN
                 audio_codebook_size=1024):   # من RVQ (الفصل 39)
        self.text_range = (0, text_vocab_size)
        self.image_range = (text_vocab_size, text_vocab_size + image_codebook_size)
        self.audio_range = (
            text_vocab_size + image_codebook_size,
            text_vocab_size + image_codebook_size + audio_codebook_size
        )
        self.total_vocab_size = self.audio_range[1]

        # رموز خاصة لتحديد بداية/نهاية كل وسيط ضمن التسلسل
        self.special_tokens = {
            "<img_start>": self.total_vocab_size,
            "<img_end>": self.total_vocab_size + 1,
            "<audio_start>": self.total_vocab_size + 2,
            "<audio_end>": self.total_vocab_size + 3,
        }
        self.total_vocab_size += len(self.special_tokens)

    def encode_interleaved_sequence(self, text_tokens, image_tokens=None, audio_tokens=None):
        """
        بناء تسلسل متداخل (interleaved) من عدة وسائط
        مثال: "صف هذه الصورة: <img_start>...<img_end> ثم استمع: <audio_start>...<audio_end>"
        """
        sequence = text_tokens[:]
        if image_tokens is not None:
            # إزاحة رموز الصورة لتقع ضمن نطاقها المخصص في الـ vocabulary الموحّد
            shifted_image_tokens = [t + self.image_range[0] for t in image_tokens]
            sequence += [self.special_tokens["<img_start>"]] + shifted_image_tokens + \
                        [self.special_tokens["<img_end>"]]
        if audio_tokens is not None:
            shifted_audio_tokens = [t + self.audio_range[0] for t in audio_tokens]
            sequence += [self.special_tokens["<audio_start>"]] + shifted_audio_tokens + \
                        [self.special_tokens["<audio_end>"]]
        return sequence

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class UnifiedAnyToAnyTransformer(nn.Module):
    """
    نموذج موحّد مفاهيميًا — نفس MiniGPT من الفصل 12
    لكن بمفردات (vocabulary) موسَّعة لتشمل رموز الصورة والصوت
    """
    def __init__(self, unified_vocab_size, d_model=4096, n_layers=32, n_heads=32, max_seq_len=8192):
        super().__init__()
        self.token_embedding = nn.Embedding(unified_vocab_size, d_model)
        # استخدام ModernTransformerBlock من الفصل 17 (RoPE + GQA + RMSNorm + SwiGLU)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, unified_vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # Weight Tying (الفصل 12)

    def forward(self, unified_token_ids):
        x = self.token_embedding(unified_token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)

    def generate_any_modality(self, prompt_ids, vocab_ranges, max_new_tokens=512):
        """
        التوليد يحدث تلقائيًا بأي وسيط حسب أي نطاق رموز يختاره النموذج!
        إذا بدأ بتوليد رمز ضمن image_range، فك ترميزه عبر VQ-GAN decoder
        إذا بدأ بتوليد رمز ضمن text_range، فك ترميزه عبر tokenizer النصي
        """
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in [vocab_ranges["eos_token_id"]]:
                break
        return generated

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def sample_video_frames(video_tensor, num_frames=16, strategy="uniform"):
    """
    video_tensor: [total_frames, C, H, W]
    """
    total_frames = video_tensor.shape[0]
    if strategy == "uniform":
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
    elif strategy == "keyframe":
        # اختيار الإطارات ذات أكبر تغيّر بصري (مبسّط)
        diffs = torch.abs(video_tensor[1:] - video_tensor[:-1]).mean(dim=(1, 2, 3))
        indices = torch.topk(diffs, num_frames).indices.sort().values
    return video_tensor[indices]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class UnifiedMultimodalVocabulary:
    """
    توضيح مفاهيمي: كيف نُركّب vocabulary موحّد من عدة وسائط
    (نهج مشابه لما استخدمته Chameleon من Meta)
    """
    def __init__(self, text_vocab_size=32000,
                 image_codebook_size=8192,    # من VQ-GAN
                 audio_codebook_size=1024):   # من RVQ (الفصل 39)
        self.text_range = (0, text_vocab_size)
        self.image_range = (text_vocab_size, text_vocab_size + image_codebook_size)
        self.audio_range = (
            text_vocab_size + image_codebook_size,
            text_vocab_size + image_codebook_size + audio_codebook_size
        )
        self.total_vocab_size = self.audio_range[1]

        # رموز خاصة لتحديد بداية/نهاية كل وسيط ضمن التسلسل
        self.special_tokens = {
            "<img_start>": self.total_vocab_size,
            "<img_end>": self.total_vocab_size + 1,
            "<audio_start>": self.total_vocab_size + 2,
            "<audio_end>": self.total_vocab_size + 3,
        }
        self.total_vocab_size += len(self.special_tokens)

    def encode_interleaved_sequence(self, text_tokens, image_tokens=None, audio_tokens=None):
        """
        بناء تسلسل متداخل (interleaved) من عدة وسائط
        مثال: "صف هذه الصورة: <img_start>...<img_end> ثم استمع: <audio_start>...<audio_end>"
        """
        sequence = text_tokens[:]
        if image_tokens is not None:
            # إزاحة رموز الصورة لتقع ضمن نطاقها المخصص في الـ vocabulary الموحّد
            shifted_image_tokens = [t + self.image_range[0] for t in image_tokens]
            sequence += [self.special_tokens["<img_start>"]] + shifted_image_tokens + \
                        [self.special_tokens["<img_end>"]]
        if audio_tokens is not None:
            shifted_audio_tokens = [t + self.audio_range[0] for t in audio_tokens]
            sequence += [self.special_tokens["<audio_start>"]] + shifted_audio_tokens + \
                        [self.special_tokens["<audio_end>"]]
        return sequence

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class UnifiedAnyToAnyTransformer(nn.Module):
    """
    نموذج موحّد مفاهيميًا — نفس MiniGPT من الفصل 12
    لكن بمفردات (vocabulary) موسَّعة لتشمل رموز الصورة والصوت
    """
    def __init__(self, unified_vocab_size, d_model=4096, n_layers=32, n_heads=32, max_seq_len=8192):
        super().__init__()
        self.token_embedding = nn.Embedding(unified_vocab_size, d_model)
        # استخدام ModernTransformerBlock من الفصل 17 (RoPE + GQA + RMSNorm + SwiGLU)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, unified_vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # Weight Tying (الفصل 12)

    def forward(self, unified_token_ids):
        x = self.token_embedding(unified_token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)

    def generate_any_modality(self, prompt_ids, vocab_ranges, max_new_tokens=512):
        """
        التوليد يحدث تلقائيًا بأي وسيط حسب أي نطاق رموز يختاره النموذج!
        إذا بدأ بتوليد رمز ضمن image_range، فك ترميزه عبر VQ-GAN decoder
        إذا بدأ بتوليد رمز ضمن text_range، فك ترميزه عبر tokenizer النصي
        """
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in [vocab_ranges["eos_token_id"]]:
                break
        return generated

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def sample_video_frames(video_tensor, num_frames=16, strategy="uniform"):
    """
    video_tensor: [total_frames, C, H, W]
    """
    total_frames = video_tensor.shape[0]
    if strategy == "uniform":
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
    elif strategy == "keyframe":
        # اختيار الإطارات ذات أكبر تغيّر بصري (مبسّط)
        diffs = torch.abs(video_tensor[1:] - video_tensor[:-1]).mean(dim=(1, 2, 3))
        indices = torch.topk(diffs, num_frames).indices.sort().values
    return video_tensor[indices]

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — النماذج الموحّدة: Any-to-Any
Mastering Transformers and Large Language Models — Unified Multimodal

Book slug : mastering-llms-transformers
Chapter   : ch-40-ch40-unified-multimodal (lesson 40)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-1
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
# [002] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
class UnifiedMultimodalVocabulary:
    """
    توضيح مفاهيمي: كيف نُركّب vocabulary موحّد من عدة وسائط
    (نهج مشابه لما استخدمته Chameleon من Meta)
    """
    def __init__(self, text_vocab_size=32000,
                 image_codebook_size=8192,    # من VQ-GAN
                 audio_codebook_size=1024):   # من RVQ (الفصل 39)
        self.text_range = (0, text_vocab_size)
        self.image_range = (text_vocab_size, text_vocab_size + image_codebook_size)
        self.audio_range = (
            text_vocab_size + image_codebook_size,
            text_vocab_size + image_codebook_size + audio_codebook_size
        )
        self.total_vocab_size = self.audio_range[1]

        # رموز خاصة لتحديد بداية/نهاية كل وسيط ضمن التسلسل
        self.special_tokens = {
            "<img_start>": self.total_vocab_size,
            "<img_end>": self.total_vocab_size + 1,
            "<audio_start>": self.total_vocab_size + 2,
            "<audio_end>": self.total_vocab_size + 3,
        }
        self.total_vocab_size += len(self.special_tokens)

    def encode_interleaved_sequence(self, text_tokens, image_tokens=None, audio_tokens=None):
        """
        بناء تسلسل متداخل (interleaved) من عدة وسائط
        مثال: "صف هذه الصورة: <img_start>...<img_end> ثم استمع: <audio_start>...<audio_end>"
        """
        sequence = text_tokens[:]
        if image_tokens is not None:
            # إزاحة رموز الصورة لتقع ضمن نطاقها المخصص في الـ vocabulary الموحّد
            shifted_image_tokens = [t + self.image_range[0] for t in image_tokens]
            sequence += [self.special_tokens["<img_start>"]] + shifted_image_tokens + \
                        [self.special_tokens["<img_end>"]]
        if audio_tokens is not None:
            shifted_audio_tokens = [t + self.audio_range[0] for t in audio_tokens]
            sequence += [self.special_tokens["<audio_start>"]] + shifted_audio_tokens + \
                        [self.special_tokens["<audio_end>"]]
        return sequence

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class UnifiedAnyToAnyTransformer(nn.Module):
    """
    نموذج موحّد مفاهيميًا — نفس MiniGPT من الفصل 12
    لكن بمفردات (vocabulary) موسَّعة لتشمل رموز الصورة والصوت
    """
    def __init__(self, unified_vocab_size, d_model=4096, n_layers=32, n_heads=32, max_seq_len=8192):
        super().__init__()
        self.token_embedding = nn.Embedding(unified_vocab_size, d_model)
        # استخدام ModernTransformerBlock من الفصل 17 (RoPE + GQA + RMSNorm + SwiGLU)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, unified_vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # Weight Tying (الفصل 12)

    def forward(self, unified_token_ids):
        x = self.token_embedding(unified_token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)

    def generate_any_modality(self, prompt_ids, vocab_ranges, max_new_tokens=512):
        """
        التوليد يحدث تلقائيًا بأي وسيط حسب أي نطاق رموز يختاره النموذج!
        إذا بدأ بتوليد رمز ضمن image_range، فك ترميزه عبر VQ-GAN decoder
        إذا بدأ بتوليد رمز ضمن text_range، فك ترميزه عبر tokenizer النصي
        """
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in [vocab_ranges["eos_token_id"]]:
                break
        return generated

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
def sample_video_frames(video_tensor, num_frames=16, strategy="uniform"):
    """
    video_tensor: [total_frames, C, H, W]
    """
    total_frames = video_tensor.shape[0]
    if strategy == "uniform":
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
    elif strategy == "keyframe":
        # اختيار الإطارات ذات أكبر تغيّر بصري (مبسّط)
        diffs = torch.abs(video_tensor[1:] - video_tensor[:-1]).mean(dim=(1, 2, 3))
        indices = torch.topk(diffs, num_frames).indices.sort().values
    return video_tensor[indices]

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-08-multimodal/ch40-unified-multimodal.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
class UnifiedMultimodalVocabulary:
    """
    توضيح مفاهيمي: كيف نُركّب vocabulary موحّد من عدة وسائط
    (نهج مشابه لما استخدمته Chameleon من Meta)
    """
    def __init__(self, text_vocab_size=32000,
                 image_codebook_size=8192,    # من VQ-GAN
                 audio_codebook_size=1024):   # من RVQ (الفصل 39)
        self.text_range = (0, text_vocab_size)
        self.image_range = (text_vocab_size, text_vocab_size + image_codebook_size)
        self.audio_range = (
            text_vocab_size + image_codebook_size,
            text_vocab_size + image_codebook_size + audio_codebook_size
        )
        self.total_vocab_size = self.audio_range[1]

        # رموز خاصة لتحديد بداية/نهاية كل وسيط ضمن التسلسل
        self.special_tokens = {
            "<img_start>": self.total_vocab_size,
            "<img_end>": self.total_vocab_size + 1,
            "<audio_start>": self.total_vocab_size + 2,
            "<audio_end>": self.total_vocab_size + 3,
        }
        self.total_vocab_size += len(self.special_tokens)

    def encode_interleaved_sequence(self, text_tokens, image_tokens=None, audio_tokens=None):
        """
        بناء تسلسل متداخل (interleaved) من عدة وسائط
        مثال: "صف هذه الصورة: <img_start>...<img_end> ثم استمع: <audio_start>...<audio_end>"
        """
        sequence = text_tokens[:]
        if image_tokens is not None:
            # إزاحة رموز الصورة لتقع ضمن نطاقها المخصص في الـ vocabulary الموحّد
            shifted_image_tokens = [t + self.image_range[0] for t in image_tokens]
            sequence += [self.special_tokens["<img_start>"]] + shifted_image_tokens + \
                        [self.special_tokens["<img_end>"]]
        if audio_tokens is not None:
            shifted_audio_tokens = [t + self.audio_range[0] for t in audio_tokens]
            sequence += [self.special_tokens["<audio_start>"]] + shifted_audio_tokens + \
                        [self.special_tokens["<audio_end>"]]
        return sequence

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class UnifiedAnyToAnyTransformer(nn.Module):
    """
    نموذج موحّد مفاهيميًا — نفس MiniGPT من الفصل 12
    لكن بمفردات (vocabulary) موسَّعة لتشمل رموز الصورة والصوت
    """
    def __init__(self, unified_vocab_size, d_model=4096, n_layers=32, n_heads=32, max_seq_len=8192):
        super().__init__()
        self.token_embedding = nn.Embedding(unified_vocab_size, d_model)
        # استخدام ModernTransformerBlock من الفصل 17 (RoPE + GQA + RMSNorm + SwiGLU)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, unified_vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # Weight Tying (الفصل 12)

    def forward(self, unified_token_ids):
        x = self.token_embedding(unified_token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_head(x)

    def generate_any_modality(self, prompt_ids, vocab_ranges, max_new_tokens=512):
        """
        التوليد يحدث تلقائيًا بأي وسيط حسب أي نطاق رموز يختاره النموذج!
        إذا بدأ بتوليد رمز ضمن image_range، فك ترميزه عبر VQ-GAN decoder
        إذا بدأ بتوليد رمز ضمن text_range، فك ترميزه عبر tokenizer النصي
        """
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() in [vocab_ranges["eos_token_id"]]:
                break
        return generated

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
def sample_video_frames(video_tensor, num_frames=16, strategy="uniform"):
    """
    video_tensor: [total_frames, C, H, W]
    """
    total_frames = video_tensor.shape[0]
    if strategy == "uniform":
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
    elif strategy == "keyframe":
        # اختيار الإطارات ذات أكبر تغيّر بصري (مبسّط)
        diffs = torch.abs(video_tensor[1:] - video_tensor[:-1]).mean(dim=(1, 2, 3))
        indices = torch.topk(diffs, num_frames).indices.sort().values
    return video_tensor[indices]

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-40-ch40-unified-multimodal.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
