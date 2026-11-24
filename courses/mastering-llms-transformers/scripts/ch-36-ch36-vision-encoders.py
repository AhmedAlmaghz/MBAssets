#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مُشفِّرات الرؤية: ViT وCLIP
Mastering Transformers and Large Language Models — Vision Encoders

Book slug : mastering-llms-transformers
Chapter   : ch-36-ch36-vision-encoders (lesson 36)
Code files: 7 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-36-ch36-vision-encoders.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مُشفِّرات الرؤية: ViT وCLIP
Mastering Transformers and Large Language Models — Vision Encoders

Book slug : mastering-llms-transformers
Chapter   : ch-36-ch36-vision-encoders (lesson 36)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/mastering-llms-transformers/code/ch-36-ch36-vision-encoders.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مُشفِّرات الرؤية: ViT وCLIP
Mastering Transformers and Large Language Models — Vision Encoders

Book slug : mastering-llms-transformers
Chapter   : ch-36-ch36-vision-encoders (lesson 36)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-1
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
# [002] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — التطبيق الكامل
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """تحويل الصورة إلى تسلسل من تضمينات الرقع (Patches)"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # طبقة Conv2D بخطوة (stride) مساوية لحجم الرقعة = تقسيم + إسقاط خطي في خطوة واحدة
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.projection(x)          # [batch, embed_dim, H/patch, W/patch]
        x = x.flatten(2)                # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)           # [batch, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """ViT كامل: Patch Embedding + [CLS] + Positional Embedding + Transformer Encoder"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.0,
                 num_classes=1000, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # توكن [CLS] قابل للتعلّم، يُمثّل تلخيصًا لكامل الصورة (كما في BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # تضمين موضعي قابل للتعلّم لكل رقعة + توكن [CLS]
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # Pre-LN كما في المعماريات الحديثة
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_patch_features=False):
        batch_size = x.shape[0]
        patches = self.patch_embed(x)                        # [batch, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)      # [batch, num_patches+1, embed_dim]
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        if return_patch_features:
            # مهم جدًا لاحقًا عند ربط ViT بـ LLM: نحتاج تمثيلات كل الرقع وليس فقط [CLS]
            return encoded[:, 1:, :]   # [batch, num_patches, embed_dim]

        cls_output = encoded[:, 0]     # للتصنيف: نستخدم [CLS] فقط
        logits = self.head(cls_output)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
CLIP: التعلّم التبايني بين الصور والنصوص — تطبيق كامل
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512,
                 vision_feature_dim=768, text_feature_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder   # مثلًا ViT بدون رأس تصنيف
        self.text_encoder = text_encoder       # مثلًا Transformer Encoder نصي

        # إسقاط خطي لتوحيد أبعاد التمثيلين في فضاء مشترك واحد
        self.vision_projection = nn.Linear(vision_feature_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_feature_dim, embed_dim, bias=False)

        # درجة حرارة قابلة للتعلّم (تُهيَّأ لوغاريتميًا لضمان الاستقرار)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images):
        features = self.vision_encoder(images, return_patch_features=False)
        projected = self.vision_projection(features)
        return F.normalize(projected, dim=-1)   # تطبيع على الكرة الواحدية

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids, attention_mask)  # تمثيل جملة مُجمَّع
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encode_image(images)       # [batch, embed_dim]
        text_embeds = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        # مصفوفة تشابه بين كل صورة وكل نص في الدفعة [batch, batch]
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

def clip_contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    # القطر الرئيسي هو الأزواج الصحيحة (صورة i مع نصها الصحيح i)

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_image + loss_text) / 2

def clip_zero_shot_classify(clip_model, image, class_names: list, tokenizer, device="cuda"):
    """تصنيف صورة بدون أي تدريب إضافي (Zero-shot) باستخدام CLIP"""
    prompts = [f"صورة لـ {name}" for name in class_names]
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embed = clip_model.encode_image(image.unsqueeze(0).to(device))
        text_embeds = clip_model.encode_text(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        similarities = (image_embed @ text_embeds.T).squeeze(0)
        probs = F.softmax(similarities * clip_model.logit_scale.exp(), dim=0)

    best_idx = probs.argmax().item()
    return class_names[best_idx], probs[best_idx].item()

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def patch_count(image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0, "يجب أن يقبل الحجم القسمة على حجم الرقعة"
    return (image_size // patch_size) ** 2

for size in (224, 336, 448):
    n = patch_count(size, 16)
    print(f"{size}×{size} / 16 = {n} رقعة (+1 لـ[CLS] = {n + 1})")

# خسارة CLIP التباينية على دفعة من 2
def clip_loss(image_emb, text_emb, temperature=0.07):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(image_emb))
    loss_i = F.cross_entropy(logits, labels)      # صورة → نص
    loss_t = F.cross_entropy(logits.T, labels)    # نص → صورة
    return (loss_i + loss_t) / 2

image_emb = torch.tensor([[1.0, 0.2, 0.0], [0.1, 1.0, 0.1]])
text_emb = torch.tensor([[0.9, 0.3, 0.1], [0.2, 0.8, 0.2]])
print("CLIP loss:", clip_loss(image_emb, text_emb).item())

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-1
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
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — التطبيق الكامل
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """تحويل الصورة إلى تسلسل من تضمينات الرقع (Patches)"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # طبقة Conv2D بخطوة (stride) مساوية لحجم الرقعة = تقسيم + إسقاط خطي في خطوة واحدة
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.projection(x)          # [batch, embed_dim, H/patch, W/patch]
        x = x.flatten(2)                # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)           # [batch, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """ViT كامل: Patch Embedding + [CLS] + Positional Embedding + Transformer Encoder"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.0,
                 num_classes=1000, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # توكن [CLS] قابل للتعلّم، يُمثّل تلخيصًا لكامل الصورة (كما في BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # تضمين موضعي قابل للتعلّم لكل رقعة + توكن [CLS]
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # Pre-LN كما في المعماريات الحديثة
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_patch_features=False):
        batch_size = x.shape[0]
        patches = self.patch_embed(x)                        # [batch, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)      # [batch, num_patches+1, embed_dim]
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        if return_patch_features:
            # مهم جدًا لاحقًا عند ربط ViT بـ LLM: نحتاج تمثيلات كل الرقع وليس فقط [CLS]
            return encoded[:, 1:, :]   # [batch, num_patches, embed_dim]

        cls_output = encoded[:, 0]     # للتصنيف: نستخدم [CLS] فقط
        logits = self.head(cls_output)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
CLIP: التعلّم التبايني بين الصور والنصوص — تطبيق كامل
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512,
                 vision_feature_dim=768, text_feature_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder   # مثلًا ViT بدون رأس تصنيف
        self.text_encoder = text_encoder       # مثلًا Transformer Encoder نصي

        # إسقاط خطي لتوحيد أبعاد التمثيلين في فضاء مشترك واحد
        self.vision_projection = nn.Linear(vision_feature_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_feature_dim, embed_dim, bias=False)

        # درجة حرارة قابلة للتعلّم (تُهيَّأ لوغاريتميًا لضمان الاستقرار)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images):
        features = self.vision_encoder(images, return_patch_features=False)
        projected = self.vision_projection(features)
        return F.normalize(projected, dim=-1)   # تطبيع على الكرة الواحدية

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids, attention_mask)  # تمثيل جملة مُجمَّع
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encode_image(images)       # [batch, embed_dim]
        text_embeds = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        # مصفوفة تشابه بين كل صورة وكل نص في الدفعة [batch, batch]
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

def clip_contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    # القطر الرئيسي هو الأزواج الصحيحة (صورة i مع نصها الصحيح i)

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_image + loss_text) / 2

def clip_zero_shot_classify(clip_model, image, class_names: list, tokenizer, device="cuda"):
    """تصنيف صورة بدون أي تدريب إضافي (Zero-shot) باستخدام CLIP"""
    prompts = [f"صورة لـ {name}" for name in class_names]
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embed = clip_model.encode_image(image.unsqueeze(0).to(device))
        text_embeds = clip_model.encode_text(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        similarities = (image_embed @ text_embeds.T).squeeze(0)
        probs = F.softmax(similarities * clip_model.logit_scale.exp(), dim=0)

    best_idx = probs.argmax().item()
    return class_names[best_idx], probs[best_idx].item()

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def patch_count(image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0, "يجب أن يقبل الحجم القسمة على حجم الرقعة"
    return (image_size // patch_size) ** 2

for size in (224, 336, 448):
    n = patch_count(size, 16)
    print(f"{size}×{size} / 16 = {n} رقعة (+1 لـ[CLS] = {n + 1})")

# خسارة CLIP التباينية على دفعة من 2
def clip_loss(image_emb, text_emb, temperature=0.07):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(image_emb))
    loss_i = F.cross_entropy(logits, labels)      # صورة → نص
    loss_t = F.cross_entropy(logits.T, labels)    # نص → صورة
    return (loss_i + loss_t) / 2

image_emb = torch.tensor([[1.0, 0.2, 0.0], [0.1, 1.0, 0.1]])
text_emb = torch.tensor([[0.9, 0.3, 0.1], [0.2, 0.8, 0.2]])
print("CLIP loss:", clip_loss(image_emb, text_emb).item())

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إتقان المحوّلات والنماذج اللغوية الكبيرة — مُشفِّرات الرؤية: ViT وCLIP
Mastering Transformers and Large Language Models — Vision Encoders

Book slug : mastering-llms-transformers
Chapter   : ch-36-ch36-vision-encoders (lesson 36)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-1
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
# [002] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — التطبيق الكامل
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """تحويل الصورة إلى تسلسل من تضمينات الرقع (Patches)"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # طبقة Conv2D بخطوة (stride) مساوية لحجم الرقعة = تقسيم + إسقاط خطي في خطوة واحدة
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.projection(x)          # [batch, embed_dim, H/patch, W/patch]
        x = x.flatten(2)                # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)           # [batch, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """ViT كامل: Patch Embedding + [CLS] + Positional Embedding + Transformer Encoder"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.0,
                 num_classes=1000, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # توكن [CLS] قابل للتعلّم، يُمثّل تلخيصًا لكامل الصورة (كما في BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # تضمين موضعي قابل للتعلّم لكل رقعة + توكن [CLS]
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # Pre-LN كما في المعماريات الحديثة
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_patch_features=False):
        batch_size = x.shape[0]
        patches = self.patch_embed(x)                        # [batch, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)      # [batch, num_patches+1, embed_dim]
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        if return_patch_features:
            # مهم جدًا لاحقًا عند ربط ViT بـ LLM: نحتاج تمثيلات كل الرقع وليس فقط [CLS]
            return encoded[:, 1:, :]   # [batch, num_patches, embed_dim]

        cls_output = encoded[:, 0]     # للتصنيف: نستخدم [CLS] فقط
        logits = self.head(cls_output)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [003] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
CLIP: التعلّم التبايني بين الصور والنصوص — تطبيق كامل
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512,
                 vision_feature_dim=768, text_feature_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder   # مثلًا ViT بدون رأس تصنيف
        self.text_encoder = text_encoder       # مثلًا Transformer Encoder نصي

        # إسقاط خطي لتوحيد أبعاد التمثيلين في فضاء مشترك واحد
        self.vision_projection = nn.Linear(vision_feature_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_feature_dim, embed_dim, bias=False)

        # درجة حرارة قابلة للتعلّم (تُهيَّأ لوغاريتميًا لضمان الاستقرار)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images):
        features = self.vision_encoder(images, return_patch_features=False)
        projected = self.vision_projection(features)
        return F.normalize(projected, dim=-1)   # تطبيع على الكرة الواحدية

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids, attention_mask)  # تمثيل جملة مُجمَّع
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encode_image(images)       # [batch, embed_dim]
        text_embeds = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        # مصفوفة تشابه بين كل صورة وكل نص في الدفعة [batch, batch]
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

def clip_contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    # القطر الرئيسي هو الأزواج الصحيحة (صورة i مع نصها الصحيح i)

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_image + loss_text) / 2

def clip_zero_shot_classify(clip_model, image, class_names: list, tokenizer, device="cuda"):
    """تصنيف صورة بدون أي تدريب إضافي (Zero-shot) باستخدام CLIP"""
    prompts = [f"صورة لـ {name}" for name in class_names]
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embed = clip_model.encode_image(image.unsqueeze(0).to(device))
        text_embeds = clip_model.encode_text(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        similarities = (image_embed @ text_embeds.T).squeeze(0)
        probs = F.softmax(similarities * clip_model.logit_scale.exp(), dim=0)

    best_idx = probs.argmax().item()
    return class_names[best_idx], probs[best_idx].item()

# ────────────────────────────────────────────────────────────────────────
# [004] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def patch_count(image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0, "يجب أن يقبل الحجم القسمة على حجم الرقعة"
    return (image_size // patch_size) ** 2

for size in (224, 336, 448):
    n = patch_count(size, 16)
    print(f"{size}×{size} / 16 = {n} رقعة (+1 لـ[CLS] = {n + 1})")

# خسارة CLIP التباينية على دفعة من 2
def clip_loss(image_emb, text_emb, temperature=0.07):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(image_emb))
    loss_i = F.cross_entropy(logits, labels)      # صورة → نص
    loss_t = F.cross_entropy(logits.T, labels)    # نص → صورة
    return (loss_i + loss_t) / 2

image_emb = torch.tensor([[1.0, 0.2, 0.0], [0.1, 1.0, 0.1]])
text_emb = torch.tensor([[0.9, 0.3, 0.1], [0.2, 0.8, 0.2]])
print("CLIP loss:", clip_loss(image_emb, text_emb).item())

# ────────────────────────────────────────────────────────────────────────
# [005] content/notebooks/part-08-multimodal/ch36-vision-encoders.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-2
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
# [004] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
Vision Transformer (ViT) من الصفر — التطبيق الكامل
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """تحويل الصورة إلى تسلسل من تضمينات الرقع (Patches)"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # طبقة Conv2D بخطوة (stride) مساوية لحجم الرقعة = تقسيم + إسقاط خطي في خطوة واحدة
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.projection(x)          # [batch, embed_dim, H/patch, W/patch]
        x = x.flatten(2)                # [batch, embed_dim, num_patches]
        x = x.transpose(1, 2)           # [batch, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """ViT كامل: Patch Embedding + [CLS] + Positional Embedding + Transformer Encoder"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.0,
                 num_classes=1000, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # توكن [CLS] قابل للتعلّم، يُمثّل تلخيصًا لكامل الصورة (كما في BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # تضمين موضعي قابل للتعلّم لكل رقعة + توكن [CLS]
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,  # Pre-LN كما في المعماريات الحديثة
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_patch_features=False):
        batch_size = x.shape[0]
        patches = self.patch_embed(x)                        # [batch, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)      # [batch, num_patches+1, embed_dim]
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        if return_patch_features:
            # مهم جدًا لاحقًا عند ربط ViT بـ LLM: نحتاج تمثيلات كل الرقع وليس فقط [CLS]
            return encoded[:, 1:, :]   # [batch, num_patches, embed_dim]

        cls_output = encoded[:, 0]     # للتصنيف: نستخدم [CLS] فقط
        logits = self.head(cls_output)
        return logits

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
"""
CLIP: التعلّم التبايني بين الصور والنصوص — تطبيق كامل
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512,
                 vision_feature_dim=768, text_feature_dim=768):
        super().__init__()
        self.vision_encoder = vision_encoder   # مثلًا ViT بدون رأس تصنيف
        self.text_encoder = text_encoder       # مثلًا Transformer Encoder نصي

        # إسقاط خطي لتوحيد أبعاد التمثيلين في فضاء مشترك واحد
        self.vision_projection = nn.Linear(vision_feature_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_feature_dim, embed_dim, bias=False)

        # درجة حرارة قابلة للتعلّم (تُهيَّأ لوغاريتميًا لضمان الاستقرار)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images):
        features = self.vision_encoder(images, return_patch_features=False)
        projected = self.vision_projection(features)
        return F.normalize(projected, dim=-1)   # تطبيع على الكرة الواحدية

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids, attention_mask)  # تمثيل جملة مُجمَّع
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1)

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encode_image(images)       # [batch, embed_dim]
        text_embeds = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        # مصفوفة تشابه بين كل صورة وكل نص في الدفعة [batch, batch]
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

def clip_contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    # القطر الرئيسي هو الأزواج الصحيحة (صورة i مع نصها الصحيح i)

    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_image + loss_text) / 2

def clip_zero_shot_classify(clip_model, image, class_names: list, tokenizer, device="cuda"):
    """تصنيف صورة بدون أي تدريب إضافي (Zero-shot) باستخدام CLIP"""
    prompts = [f"صورة لـ {name}" for name in class_names]
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        image_embed = clip_model.encode_image(image.unsqueeze(0).to(device))
        text_embeds = clip_model.encode_text(
            text_inputs["input_ids"], text_inputs["attention_mask"]
        )
        similarities = (image_embed @ text_embeds.T).squeeze(0)
        probs = F.softmax(similarities * clip_model.logit_scale.exp(), dim=0)

    best_idx = probs.argmax().item()
    return class_names[best_idx], probs[best_idx].item()

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

def patch_count(image_size: int, patch_size: int) -> int:
    assert image_size % patch_size == 0, "يجب أن يقبل الحجم القسمة على حجم الرقعة"
    return (image_size // patch_size) ** 2

for size in (224, 336, 448):
    n = patch_count(size, 16)
    print(f"{size}×{size} / 16 = {n} رقعة (+1 لـ[CLS] = {n + 1})")

# خسارة CLIP التباينية على دفعة من 2
def clip_loss(image_emb, text_emb, temperature=0.07):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(image_emb))
    loss_i = F.cross_entropy(logits, labels)      # صورة → نص
    loss_t = F.cross_entropy(logits.T, labels)    # نص → صورة
    return (loss_i + loss_t) / 2

image_emb = torch.tensor([[1.0, 0.2, 0.0], [0.1, 1.0, 0.1]])
text_emb = torch.tensor([[0.9, 0.3, 0.1], [0.2, 0.8, 0.2]])
print("CLIP loss:", clip_loss(image_emb, text_emb).item())

# ────────────────────────────────────────────────────────────────────────
# [007] maghzbook-download/mastering-llms-transformers/notebooks/ch-36-ch36-vision-encoders.ipynb#cell-6
# ────────────────────────────────────────────────────────────────────────
# 🧪 مساحة تجربة
# اكتب هنا حلّ التمرين أو أي تجربة إضافية.
