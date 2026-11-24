#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2
Large Language Model Engineering — VLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-41-chapter-41 (lesson 41)
Code files: 6 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-41-chapter-41.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2
Large Language Model Engineering — VLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-41-chapter-41 (lesson 41)
Code files: 5 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] maghzbook-download/llm-engineering-69-chapters/code/ch-41-chapter-41.py
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2
Large Language Model Engineering — VLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-41-chapter-41 (lesson 41)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_41.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 41: الفصل 41: نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vlm() -> None:
    visual_width, language_width, tokens = (4, 3, 2)
    projector = [[0.1 * (row + col + 1) for col in range(language_width)] for row in range(visual_width)]
    visual_tokens = [[1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 0.5]]
    projected = [[sum((vector[i] * projector[i][j] for i in range(visual_width))) for j in range(language_width)] for vector in visual_tokens]
    print('visual feature shape:', (tokens, visual_width))
    print('projected language-token shape:', (tokens, language_width))
    print('projected tokens:', [[round(x, 2) for x in row] for row in projected])

if __name__ == "__main__":
    print("Chapter 41 lab: Project visual features into language-token space")
    run_vlm()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-41.md
# ────────────────────────────────────────────────────────────────────────
"""
VLM مصغّر بأسلوب LLaVA — ربط Vision Encoder بنموذج لغوي عبر Connector بسيط
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مشفّر الرؤية (نفس فكرة ViT من الفصل 39، لكن مبسّط)
# ---------------------------------------------------------------------
@dataclass
class VisionConfig:
    image_size: int = 28
    patch_size: int = 4
    num_channels: int = 1
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class SimpleViT(nn.Module):
    """مشفّر رؤية مصغّر — يُنتج تمثيلًا لكل رقعة (بدون [CLS] هنا، نحتاج كل الرقع)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches + self.position_embedding
        encoded = self.encoder(patches)
        return self.norm(encoded)  # [B, N, D_vision] — نُعيد كل الرقع، لا رمز CLS وحده


# ---------------------------------------------------------------------
# 2) Connector (بأسلوب LLaVA): MLP يُسقط كل رقعة لبُعد الـ LLM
# ---------------------------------------------------------------------
class VisionLanguageConnector(nn.Module):
    """
    يحوّل تمثيلات ViT [B, N, D_vision] إلى "رموز بصرية"
    [B, N, D_llm] قابلة للحقن مباشرة في تسلسل مدخلات الـ LLM.
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_features)


# ---------------------------------------------------------------------
# 3) نموذج لغوي بسيط (Decoder-only) — "عمودنا الفقري اللغوي"
# ---------------------------------------------------------------------
@dataclass
class LLMConfig:
    vocab_size: int
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    max_seq_len: int = 64


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleLLM(nn.Module):
    """
    نموذج لغوي بسيط يقبل مدخلات Embeddings مباشرة
    (بدلًا من فقط معرّفات رموز) — هذا أساسي لدمج الرموز البصرية!
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.embed_dim)
        )
        self.blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(text_ids)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        يقبل input_embeds جاهزة (قد تكون دمجًا من رموز بصرية + نصية)
        بدلًا من معرّفات رموز خام — هذا ما يسمح بحقن الرؤية.
        """
        seq_len = input_embeds.size(1)
        x = input_embeds + self.position_embedding[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------
# 4) نموذج VLM الكامل — يجمع الثلاثة معًا
# ---------------------------------------------------------------------
class MiniLLaVA(nn.Module):
    def __init__(
        self, vision_config: VisionConfig, llm_config: LLMConfig, freeze_vision: bool = True
    ):
        super().__init__()
        self.vision_encoder = SimpleViT(vision_config)
        self.connector = VisionLanguageConnector(
            vision_dim=vision_config.embed_dim, llm_dim=llm_config.embed_dim
        )
        self.llm = SimpleLLM(llm_config)

        # في التدريب الحقيقي: مشفّر الرؤية مُدرَّب مسبقًا ويُجمَّد في المرحلة الأولى
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        # 1. استخراج تمثيلات الرقع من الصورة
        vision_features = self.vision_encoder(images)        # [B, N_patches, D_vision]

        # 2. تحويلها إلى "رموز بصرية" بنفس بُعد الـ LLM
        visual_tokens = self.connector(vision_features)      # [B, N_patches, D_llm]

        # 3. تضمين النص عبر جدول Embedding القياسي للـ LLM
        text_embeds = self.llm.embed_text(text_ids)           # [B, T, D_llm]

        # 4. الدمج: الرموز البصرية أولًا، ثم النصية — بالضبط كما في LLaVA
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # 5. تمرير التسلسل المدموج عبر الـ LLM كأنه نص عادي بالكامل!
        logits = self.llm(combined_embeds)

        # نهتم فقط بتنبؤات الجزء النصي (بعد الرموز البصرية)
        num_visual_tokens = visual_tokens.size(1)
        text_logits = logits[:, num_visual_tokens:, :]
        return text_logits


# ---------------------------------------------------------------------
# 5) بيانات تدريبية: Image Captioning على Fashion-MNIST
# ---------------------------------------------------------------------
CLASS_NAMES = [
    "تيشيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء", "حقيبة", "كاحل",
]
VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted(set("".join(CLASS_NAMES) + " هذهصورة"))
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
MAX_CAPTION_LEN = 10


def tokenize_caption(caption: str) -> list[int]:
    tokens = ["<bos>"] + list(caption) + ["<eos>"]
    ids = [TOKEN_TO_ID.get(ch, 0) for ch in tokens]
    ids += [TOKEN_TO_ID["<pad>"]] * (MAX_CAPTION_LEN - len(ids))
    return ids[:MAX_CAPTION_LEN]


class CaptioningDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"هذه صورة {CLASS_NAMES[label]}"
        caption_ids = torch.tensor(tokenize_caption(caption), dtype=torch.long)
        # مدخل الـ LLM: كل الرموز عدا الأخير / الهدف: كل الرموز عدا الأول (shift)
        return image, caption_ids[:-1], caption_ids[1:]


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_vlm(model: MiniLLaVA, train_loader: DataLoader, num_epochs: int = 15):
    model.to(device)
    # فقط البارامترات القابلة للتدريب (Connector + LLM؛ ViT مُجمَّد)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)
    pad_id = TOKEN_TO_ID["<pad>"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, input_ids, target_ids in train_loader:
            images = images.to(device)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=pad_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: توليد وصف لصورة جديدة (Autoregressive Generation)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_caption(model: MiniLLaVA, image: torch.Tensor, max_length: int = 10) -> str:
    model.eval()
    image = image.unsqueeze(0).to(device)

    vision_features = model.vision_encoder(image)
    visual_tokens = model.connector(vision_features)

    generated_ids = [TOKEN_TO_ID["<bos>"]]
    for _ in range(max_length):
        current_ids = torch.tensor([generated_ids], device=device)
        text_embeds = model.llm.embed_text(current_ids)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        logits = model.llm(combined)

        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax().item()
        generated_ids.append(next_token)

        if next_token == TOKEN_TO_ID["<eos>"]:
            break

    id_to_token = {v: k for k, v in TOKEN_TO_ID.items()}
    tokens = [id_to_token[i] for i in generated_ids[1:-1]]  # إزالة bos/eos
    return "".join(tokens)


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])
    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataset = CaptioningDataset(train_base)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    vision_config = VisionConfig(image_size=28, patch_size=4, embed_dim=128)
    llm_config = LLMConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=128,
        max_seq_len=vision_config.num_patches + MAX_CAPTION_LEN,
    )

    model = MiniLLaVA(vision_config, llm_config, freeze_vision=False)
    # ملاحظة: في هذا المثال التعليمي الصغير لم نجمّد ViT لصغر حجم البيانات،
    # لكن في الإنتاج الحقيقي يُجمَّد ViT دائمًا في المرحلة الأولى (محاذاة الميزات).

    train_vlm(model, train_loader, num_epochs=15)

    # اختبار على صور جديدة
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    for i in range(5):
        image, label = test_base[i]
        caption = generate_caption(model, image)
        print(f"الفئة الحقيقية: {CLASS_NAMES[label]} | الوصف المُولَّد: {caption}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-41.md
# ────────────────────────────────────────────────────────────────────────
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("diagram.png")
prompt = "USER: <image>\nما الخطأ في هذا الرسم البياني؟\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output_ids[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-41.md
# ────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    """
    بدل تعليق ثابت، كل مثال يحتوي: صورة + سؤال + إجابة.
    مثال: (صورة فستان، "ما لون هذه القطعة؟", "لا يمكن تحديد اللون من صورة رمادية")
    أو أسئلة بسيطة مبنية على الفئة: "ما نوع هذه القطعة؟" -> اسم الفئة.
    """
    def __getitem__(self, idx):
        # صمّم تنسيق تسلسل يجمع: [صورة] + [رموز السؤال] + [رموز الإجابة]
        # تلميح: استخدم فاصلًا خاصًا بين السؤال والإجابة، واحسب الخسارة
        # فقط على جزء الإجابة (أخفِ Loss عن رموز السؤال بـ ignore_index)
        ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 41: الفصل 41: نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vlm() -> None:
    visual_width, language_width, tokens = (4, 3, 2)
    projector = [[0.1 * (row + col + 1) for col in range(language_width)] for row in range(visual_width)]
    visual_tokens = [[1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 0.5]]
    projected = [[sum((vector[i] * projector[i][j] for i in range(visual_width))) for j in range(language_width)] for vector in visual_tokens]
    print('visual feature shape:', (tokens, visual_width))
    print('projected language-token shape:', (tokens, language_width))
    print('projected tokens:', [[round(x, 2) for x in row] for row in projected])

if __name__ == "__main__":
    print("Chapter 41 lab: Project visual features into language-token space")
    run_vlm()

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""
VLM مصغّر بأسلوب LLaVA — ربط Vision Encoder بنموذج لغوي عبر Connector بسيط
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مشفّر الرؤية (نفس فكرة ViT من الفصل 39، لكن مبسّط)
# ---------------------------------------------------------------------
@dataclass
class VisionConfig:
    image_size: int = 28
    patch_size: int = 4
    num_channels: int = 1
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class SimpleViT(nn.Module):
    """مشفّر رؤية مصغّر — يُنتج تمثيلًا لكل رقعة (بدون [CLS] هنا، نحتاج كل الرقع)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches + self.position_embedding
        encoded = self.encoder(patches)
        return self.norm(encoded)  # [B, N, D_vision] — نُعيد كل الرقع، لا رمز CLS وحده


# ---------------------------------------------------------------------
# 2) Connector (بأسلوب LLaVA): MLP يُسقط كل رقعة لبُعد الـ LLM
# ---------------------------------------------------------------------
class VisionLanguageConnector(nn.Module):
    """
    يحوّل تمثيلات ViT [B, N, D_vision] إلى "رموز بصرية"
    [B, N, D_llm] قابلة للحقن مباشرة في تسلسل مدخلات الـ LLM.
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_features)


# ---------------------------------------------------------------------
# 3) نموذج لغوي بسيط (Decoder-only) — "عمودنا الفقري اللغوي"
# ---------------------------------------------------------------------
@dataclass
class LLMConfig:
    vocab_size: int
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    max_seq_len: int = 64


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleLLM(nn.Module):
    """
    نموذج لغوي بسيط يقبل مدخلات Embeddings مباشرة
    (بدلًا من فقط معرّفات رموز) — هذا أساسي لدمج الرموز البصرية!
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.embed_dim)
        )
        self.blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(text_ids)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        يقبل input_embeds جاهزة (قد تكون دمجًا من رموز بصرية + نصية)
        بدلًا من معرّفات رموز خام — هذا ما يسمح بحقن الرؤية.
        """
        seq_len = input_embeds.size(1)
        x = input_embeds + self.position_embedding[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------
# 4) نموذج VLM الكامل — يجمع الثلاثة معًا
# ---------------------------------------------------------------------
class MiniLLaVA(nn.Module):
    def __init__(
        self, vision_config: VisionConfig, llm_config: LLMConfig, freeze_vision: bool = True
    ):
        super().__init__()
        self.vision_encoder = SimpleViT(vision_config)
        self.connector = VisionLanguageConnector(
            vision_dim=vision_config.embed_dim, llm_dim=llm_config.embed_dim
        )
        self.llm = SimpleLLM(llm_config)

        # في التدريب الحقيقي: مشفّر الرؤية مُدرَّب مسبقًا ويُجمَّد في المرحلة الأولى
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        # 1. استخراج تمثيلات الرقع من الصورة
        vision_features = self.vision_encoder(images)        # [B, N_patches, D_vision]

        # 2. تحويلها إلى "رموز بصرية" بنفس بُعد الـ LLM
        visual_tokens = self.connector(vision_features)      # [B, N_patches, D_llm]

        # 3. تضمين النص عبر جدول Embedding القياسي للـ LLM
        text_embeds = self.llm.embed_text(text_ids)           # [B, T, D_llm]

        # 4. الدمج: الرموز البصرية أولًا، ثم النصية — بالضبط كما في LLaVA
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # 5. تمرير التسلسل المدموج عبر الـ LLM كأنه نص عادي بالكامل!
        logits = self.llm(combined_embeds)

        # نهتم فقط بتنبؤات الجزء النصي (بعد الرموز البصرية)
        num_visual_tokens = visual_tokens.size(1)
        text_logits = logits[:, num_visual_tokens:, :]
        return text_logits


# ---------------------------------------------------------------------
# 5) بيانات تدريبية: Image Captioning على Fashion-MNIST
# ---------------------------------------------------------------------
CLASS_NAMES = [
    "تيشيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء", "حقيبة", "كاحل",
]
VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted(set("".join(CLASS_NAMES) + " هذهصورة"))
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
MAX_CAPTION_LEN = 10


def tokenize_caption(caption: str) -> list[int]:
    tokens = ["<bos>"] + list(caption) + ["<eos>"]
    ids = [TOKEN_TO_ID.get(ch, 0) for ch in tokens]
    ids += [TOKEN_TO_ID["<pad>"]] * (MAX_CAPTION_LEN - len(ids))
    return ids[:MAX_CAPTION_LEN]


class CaptioningDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"هذه صورة {CLASS_NAMES[label]}"
        caption_ids = torch.tensor(tokenize_caption(caption), dtype=torch.long)
        # مدخل الـ LLM: كل الرموز عدا الأخير / الهدف: كل الرموز عدا الأول (shift)
        return image, caption_ids[:-1], caption_ids[1:]


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_vlm(model: MiniLLaVA, train_loader: DataLoader, num_epochs: int = 15):
    model.to(device)
    # فقط البارامترات القابلة للتدريب (Connector + LLM؛ ViT مُجمَّد)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)
    pad_id = TOKEN_TO_ID["<pad>"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, input_ids, target_ids in train_loader:
            images = images.to(device)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=pad_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: توليد وصف لصورة جديدة (Autoregressive Generation)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_caption(model: MiniLLaVA, image: torch.Tensor, max_length: int = 10) -> str:
    model.eval()
    image = image.unsqueeze(0).to(device)

    vision_features = model.vision_encoder(image)
    visual_tokens = model.connector(vision_features)

    generated_ids = [TOKEN_TO_ID["<bos>"]]
    for _ in range(max_length):
        current_ids = torch.tensor([generated_ids], device=device)
        text_embeds = model.llm.embed_text(current_ids)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        logits = model.llm(combined)

        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax().item()
        generated_ids.append(next_token)

        if next_token == TOKEN_TO_ID["<eos>"]:
            break

    id_to_token = {v: k for k, v in TOKEN_TO_ID.items()}
    tokens = [id_to_token[i] for i in generated_ids[1:-1]]  # إزالة bos/eos
    return "".join(tokens)


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])
    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataset = CaptioningDataset(train_base)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    vision_config = VisionConfig(image_size=28, patch_size=4, embed_dim=128)
    llm_config = LLMConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=128,
        max_seq_len=vision_config.num_patches + MAX_CAPTION_LEN,
    )

    model = MiniLLaVA(vision_config, llm_config, freeze_vision=False)
    # ملاحظة: في هذا المثال التعليمي الصغير لم نجمّد ViT لصغر حجم البيانات،
    # لكن في الإنتاج الحقيقي يُجمَّد ViT دائمًا في المرحلة الأولى (محاذاة الميزات).

    train_vlm(model, train_loader, num_epochs=15)

    # اختبار على صور جديدة
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    for i in range(5):
        image, label = test_base[i]
        caption = generate_caption(model, image)
        print(f"الفئة الحقيقية: {CLASS_NAMES[label]} | الوصف المُولَّد: {caption}")

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("diagram.png")
prompt = "USER: <image>\nما الخطأ في هذا الرسم البياني؟\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output_ids[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    """
    بدل تعليق ثابت، كل مثال يحتوي: صورة + سؤال + إجابة.
    مثال: (صورة فستان، "ما لون هذه القطعة؟", "لا يمكن تحديد اللون من صورة رمادية")
    أو أسئلة بسيطة مبنية على الفئة: "ما نوع هذه القطعة؟" -> اسم الفئة.
    """
    def __getitem__(self, idx):
        # صمّم تنسيق تسلسل يجمع: [صورة] + [رموز السؤال] + [رموز الإجابة]
        # تلميح: استخدم فاصلًا خاصًا بين السؤال والإجابة، واحسب الخسارة
        # فقط على جزء الإجابة (أخفِ Loss عن رموز السؤال بـ ignore_index)
        ...

# ────────────────────────────────────────────────────────────────────────
# [002] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
كتاب هندسة النماذج اللغوية الكبيرة — نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2
Large Language Model Engineering — VLMs

Book slug : llm-engineering-69-chapters
Chapter   : ch-41-chapter-41 (lesson 41)
Code files: 4 consolidated fragment(s)

All code fragments from this chapter are collected in this single file,
in the same order as the matching Jupyter notebook.
"""

# ────────────────────────────────────────────────────────────────────────
# [001] book_code/notebooks/chapters/chapter_41.ipynb#cell-1
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 41: الفصل 41: نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vlm() -> None:
    visual_width, language_width, tokens = (4, 3, 2)
    projector = [[0.1 * (row + col + 1) for col in range(language_width)] for row in range(visual_width)]
    visual_tokens = [[1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 0.5]]
    projected = [[sum((vector[i] * projector[i][j] for i in range(visual_width))) for j in range(language_width)] for vector in visual_tokens]
    print('visual feature shape:', (tokens, visual_width))
    print('projected language-token shape:', (tokens, language_width))
    print('projected tokens:', [[round(x, 2) for x in row] for row in projected])

if __name__ == "__main__":
    print("Chapter 41 lab: Project visual features into language-token space")
    run_vlm()

# ────────────────────────────────────────────────────────────────────────
# [002] book/chapters/part-08-multimodal-models/chapter-41.md
# ────────────────────────────────────────────────────────────────────────
"""
VLM مصغّر بأسلوب LLaVA — ربط Vision Encoder بنموذج لغوي عبر Connector بسيط
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مشفّر الرؤية (نفس فكرة ViT من الفصل 39، لكن مبسّط)
# ---------------------------------------------------------------------
@dataclass
class VisionConfig:
    image_size: int = 28
    patch_size: int = 4
    num_channels: int = 1
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class SimpleViT(nn.Module):
    """مشفّر رؤية مصغّر — يُنتج تمثيلًا لكل رقعة (بدون [CLS] هنا، نحتاج كل الرقع)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches + self.position_embedding
        encoded = self.encoder(patches)
        return self.norm(encoded)  # [B, N, D_vision] — نُعيد كل الرقع، لا رمز CLS وحده


# ---------------------------------------------------------------------
# 2) Connector (بأسلوب LLaVA): MLP يُسقط كل رقعة لبُعد الـ LLM
# ---------------------------------------------------------------------
class VisionLanguageConnector(nn.Module):
    """
    يحوّل تمثيلات ViT [B, N, D_vision] إلى "رموز بصرية"
    [B, N, D_llm] قابلة للحقن مباشرة في تسلسل مدخلات الـ LLM.
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_features)


# ---------------------------------------------------------------------
# 3) نموذج لغوي بسيط (Decoder-only) — "عمودنا الفقري اللغوي"
# ---------------------------------------------------------------------
@dataclass
class LLMConfig:
    vocab_size: int
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    max_seq_len: int = 64


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleLLM(nn.Module):
    """
    نموذج لغوي بسيط يقبل مدخلات Embeddings مباشرة
    (بدلًا من فقط معرّفات رموز) — هذا أساسي لدمج الرموز البصرية!
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.embed_dim)
        )
        self.blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(text_ids)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        يقبل input_embeds جاهزة (قد تكون دمجًا من رموز بصرية + نصية)
        بدلًا من معرّفات رموز خام — هذا ما يسمح بحقن الرؤية.
        """
        seq_len = input_embeds.size(1)
        x = input_embeds + self.position_embedding[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------
# 4) نموذج VLM الكامل — يجمع الثلاثة معًا
# ---------------------------------------------------------------------
class MiniLLaVA(nn.Module):
    def __init__(
        self, vision_config: VisionConfig, llm_config: LLMConfig, freeze_vision: bool = True
    ):
        super().__init__()
        self.vision_encoder = SimpleViT(vision_config)
        self.connector = VisionLanguageConnector(
            vision_dim=vision_config.embed_dim, llm_dim=llm_config.embed_dim
        )
        self.llm = SimpleLLM(llm_config)

        # في التدريب الحقيقي: مشفّر الرؤية مُدرَّب مسبقًا ويُجمَّد في المرحلة الأولى
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        # 1. استخراج تمثيلات الرقع من الصورة
        vision_features = self.vision_encoder(images)        # [B, N_patches, D_vision]

        # 2. تحويلها إلى "رموز بصرية" بنفس بُعد الـ LLM
        visual_tokens = self.connector(vision_features)      # [B, N_patches, D_llm]

        # 3. تضمين النص عبر جدول Embedding القياسي للـ LLM
        text_embeds = self.llm.embed_text(text_ids)           # [B, T, D_llm]

        # 4. الدمج: الرموز البصرية أولًا، ثم النصية — بالضبط كما في LLaVA
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # 5. تمرير التسلسل المدموج عبر الـ LLM كأنه نص عادي بالكامل!
        logits = self.llm(combined_embeds)

        # نهتم فقط بتنبؤات الجزء النصي (بعد الرموز البصرية)
        num_visual_tokens = visual_tokens.size(1)
        text_logits = logits[:, num_visual_tokens:, :]
        return text_logits


# ---------------------------------------------------------------------
# 5) بيانات تدريبية: Image Captioning على Fashion-MNIST
# ---------------------------------------------------------------------
CLASS_NAMES = [
    "تيشيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء", "حقيبة", "كاحل",
]
VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted(set("".join(CLASS_NAMES) + " هذهصورة"))
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
MAX_CAPTION_LEN = 10


def tokenize_caption(caption: str) -> list[int]:
    tokens = ["<bos>"] + list(caption) + ["<eos>"]
    ids = [TOKEN_TO_ID.get(ch, 0) for ch in tokens]
    ids += [TOKEN_TO_ID["<pad>"]] * (MAX_CAPTION_LEN - len(ids))
    return ids[:MAX_CAPTION_LEN]


class CaptioningDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"هذه صورة {CLASS_NAMES[label]}"
        caption_ids = torch.tensor(tokenize_caption(caption), dtype=torch.long)
        # مدخل الـ LLM: كل الرموز عدا الأخير / الهدف: كل الرموز عدا الأول (shift)
        return image, caption_ids[:-1], caption_ids[1:]


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_vlm(model: MiniLLaVA, train_loader: DataLoader, num_epochs: int = 15):
    model.to(device)
    # فقط البارامترات القابلة للتدريب (Connector + LLM؛ ViT مُجمَّد)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)
    pad_id = TOKEN_TO_ID["<pad>"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, input_ids, target_ids in train_loader:
            images = images.to(device)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=pad_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: توليد وصف لصورة جديدة (Autoregressive Generation)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_caption(model: MiniLLaVA, image: torch.Tensor, max_length: int = 10) -> str:
    model.eval()
    image = image.unsqueeze(0).to(device)

    vision_features = model.vision_encoder(image)
    visual_tokens = model.connector(vision_features)

    generated_ids = [TOKEN_TO_ID["<bos>"]]
    for _ in range(max_length):
        current_ids = torch.tensor([generated_ids], device=device)
        text_embeds = model.llm.embed_text(current_ids)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        logits = model.llm(combined)

        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax().item()
        generated_ids.append(next_token)

        if next_token == TOKEN_TO_ID["<eos>"]:
            break

    id_to_token = {v: k for k, v in TOKEN_TO_ID.items()}
    tokens = [id_to_token[i] for i in generated_ids[1:-1]]  # إزالة bos/eos
    return "".join(tokens)


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])
    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataset = CaptioningDataset(train_base)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    vision_config = VisionConfig(image_size=28, patch_size=4, embed_dim=128)
    llm_config = LLMConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=128,
        max_seq_len=vision_config.num_patches + MAX_CAPTION_LEN,
    )

    model = MiniLLaVA(vision_config, llm_config, freeze_vision=False)
    # ملاحظة: في هذا المثال التعليمي الصغير لم نجمّد ViT لصغر حجم البيانات،
    # لكن في الإنتاج الحقيقي يُجمَّد ViT دائمًا في المرحلة الأولى (محاذاة الميزات).

    train_vlm(model, train_loader, num_epochs=15)

    # اختبار على صور جديدة
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    for i in range(5):
        image, label = test_base[i]
        caption = generate_caption(model, image)
        print(f"الفئة الحقيقية: {CLASS_NAMES[label]} | الوصف المُولَّد: {caption}")

# ────────────────────────────────────────────────────────────────────────
# [003] book/chapters/part-08-multimodal-models/chapter-41.md
# ────────────────────────────────────────────────────────────────────────
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("diagram.png")
prompt = "USER: <image>\nما الخطأ في هذا الرسم البياني؟\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output_ids[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [004] book/chapters/part-08-multimodal-models/chapter-41.md
# ────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    """
    بدل تعليق ثابت، كل مثال يحتوي: صورة + سؤال + إجابة.
    مثال: (صورة فستان، "ما لون هذه القطعة؟", "لا يمكن تحديد اللون من صورة رمادية")
    أو أسئلة بسيطة مبنية على الفئة: "ما نوع هذه القطعة؟" -> اسم الفئة.
    """
    def __getitem__(self, idx):
        # صمّم تنسيق تسلسل يجمع: [صورة] + [رموز السؤال] + [رموز الإجابة]
        # تلميح: استخدم فاصلًا خاصًا بين السؤال والإجابة، واحسب الخسارة
        # فقط على جزء الإجابة (أخفِ Loss عن رموز السؤال بـ ignore_index)
        ...

# ────────────────────────────────────────────────────────────────────────
# [003] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-2
# ────────────────────────────────────────────────────────────────────────
"""Standalone mini-lab for chapter 41: الفصل 41: نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vlm() -> None:
    visual_width, language_width, tokens = (4, 3, 2)
    projector = [[0.1 * (row + col + 1) for col in range(language_width)] for row in range(visual_width)]
    visual_tokens = [[1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 0.5]]
    projected = [[sum((vector[i] * projector[i][j] for i in range(visual_width))) for j in range(language_width)] for vector in visual_tokens]
    print('visual feature shape:', (tokens, visual_width))
    print('projected language-token shape:', (tokens, language_width))
    print('projected tokens:', [[round(x, 2) for x in row] for row in projected])

if __name__ == "__main__":
    print("Chapter 41 lab: Project visual features into language-token space")
    run_vlm()

# ────────────────────────────────────────────────────────────────────────
# [004] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-3
# ────────────────────────────────────────────────────────────────────────
"""
VLM مصغّر بأسلوب LLaVA — ربط Vision Encoder بنموذج لغوي عبر Connector بسيط
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1) مشفّر الرؤية (نفس فكرة ViT من الفصل 39، لكن مبسّط)
# ---------------------------------------------------------------------
@dataclass
class VisionConfig:
    image_size: int = 28
    patch_size: int = 4
    num_channels: int = 1
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class SimpleViT(nn.Module):
    """مشفّر رؤية مصغّر — يُنتج تمثيلًا لكل رقعة (بدون [CLS] هنا، نحتاج كل الرقع)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim, nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches + self.position_embedding
        encoded = self.encoder(patches)
        return self.norm(encoded)  # [B, N, D_vision] — نُعيد كل الرقع، لا رمز CLS وحده


# ---------------------------------------------------------------------
# 2) Connector (بأسلوب LLaVA): MLP يُسقط كل رقعة لبُعد الـ LLM
# ---------------------------------------------------------------------
class VisionLanguageConnector(nn.Module):
    """
    يحوّل تمثيلات ViT [B, N, D_vision] إلى "رموز بصرية"
    [B, N, D_llm] قابلة للحقن مباشرة في تسلسل مدخلات الـ LLM.
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_features)


# ---------------------------------------------------------------------
# 3) نموذج لغوي بسيط (Decoder-only) — "عمودنا الفقري اللغوي"
# ---------------------------------------------------------------------
@dataclass
class LLMConfig:
    vocab_size: int
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    max_seq_len: int = 64


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, attn_mask=causal_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleLLM(nn.Module):
    """
    نموذج لغوي بسيط يقبل مدخلات Embeddings مباشرة
    (بدلًا من فقط معرّفات رموز) — هذا أساسي لدمج الرموز البصرية!
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.embed_dim)
        )
        self.blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(text_ids)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        يقبل input_embeds جاهزة (قد تكون دمجًا من رموز بصرية + نصية)
        بدلًا من معرّفات رموز خام — هذا ما يسمح بحقن الرؤية.
        """
        seq_len = input_embeds.size(1)
        x = input_embeds + self.position_embedding[:, :seq_len, :]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------
# 4) نموذج VLM الكامل — يجمع الثلاثة معًا
# ---------------------------------------------------------------------
class MiniLLaVA(nn.Module):
    def __init__(
        self, vision_config: VisionConfig, llm_config: LLMConfig, freeze_vision: bool = True
    ):
        super().__init__()
        self.vision_encoder = SimpleViT(vision_config)
        self.connector = VisionLanguageConnector(
            vision_dim=vision_config.embed_dim, llm_dim=llm_config.embed_dim
        )
        self.llm = SimpleLLM(llm_config)

        # في التدريب الحقيقي: مشفّر الرؤية مُدرَّب مسبقًا ويُجمَّد في المرحلة الأولى
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        # 1. استخراج تمثيلات الرقع من الصورة
        vision_features = self.vision_encoder(images)        # [B, N_patches, D_vision]

        # 2. تحويلها إلى "رموز بصرية" بنفس بُعد الـ LLM
        visual_tokens = self.connector(vision_features)      # [B, N_patches, D_llm]

        # 3. تضمين النص عبر جدول Embedding القياسي للـ LLM
        text_embeds = self.llm.embed_text(text_ids)           # [B, T, D_llm]

        # 4. الدمج: الرموز البصرية أولًا، ثم النصية — بالضبط كما في LLaVA
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # 5. تمرير التسلسل المدموج عبر الـ LLM كأنه نص عادي بالكامل!
        logits = self.llm(combined_embeds)

        # نهتم فقط بتنبؤات الجزء النصي (بعد الرموز البصرية)
        num_visual_tokens = visual_tokens.size(1)
        text_logits = logits[:, num_visual_tokens:, :]
        return text_logits


# ---------------------------------------------------------------------
# 5) بيانات تدريبية: Image Captioning على Fashion-MNIST
# ---------------------------------------------------------------------
CLASS_NAMES = [
    "تيشيرت", "بنطال", "بلوفر", "فستان", "معطف",
    "صندل", "قميص", "حذاء", "حقيبة", "كاحل",
]
VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted(set("".join(CLASS_NAMES) + " هذهصورة"))
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
MAX_CAPTION_LEN = 10


def tokenize_caption(caption: str) -> list[int]:
    tokens = ["<bos>"] + list(caption) + ["<eos>"]
    ids = [TOKEN_TO_ID.get(ch, 0) for ch in tokens]
    ids += [TOKEN_TO_ID["<pad>"]] * (MAX_CAPTION_LEN - len(ids))
    return ids[:MAX_CAPTION_LEN]


class CaptioningDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        caption = f"هذه صورة {CLASS_NAMES[label]}"
        caption_ids = torch.tensor(tokenize_caption(caption), dtype=torch.long)
        # مدخل الـ LLM: كل الرموز عدا الأخير / الهدف: كل الرموز عدا الأول (shift)
        return image, caption_ids[:-1], caption_ids[1:]


# ---------------------------------------------------------------------
# 6) حلقة التدريب
# ---------------------------------------------------------------------
def train_vlm(model: MiniLLaVA, train_loader: DataLoader, num_epochs: int = 15):
    model.to(device)
    # فقط البارامترات القابلة للتدريب (Connector + LLM؛ ViT مُجمَّد)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)
    pad_id = TOKEN_TO_ID["<pad>"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, input_ids, target_ids in train_loader:
            images = images.to(device)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=pad_id,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {running_loss/len(train_loader):.4f}")


# ---------------------------------------------------------------------
# 7) الاستدلال: توليد وصف لصورة جديدة (Autoregressive Generation)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_caption(model: MiniLLaVA, image: torch.Tensor, max_length: int = 10) -> str:
    model.eval()
    image = image.unsqueeze(0).to(device)

    vision_features = model.vision_encoder(image)
    visual_tokens = model.connector(vision_features)

    generated_ids = [TOKEN_TO_ID["<bos>"]]
    for _ in range(max_length):
        current_ids = torch.tensor([generated_ids], device=device)
        text_embeds = model.llm.embed_text(current_ids)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        logits = model.llm(combined)

        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax().item()
        generated_ids.append(next_token)

        if next_token == TOKEN_TO_ID["<eos>"]:
            break

    id_to_token = {v: k for k, v in TOKEN_TO_ID.items()}
    tokens = [id_to_token[i] for i in generated_ids[1:-1]]  # إزالة bos/eos
    return "".join(tokens)


# ---------------------------------------------------------------------
# 8) التشغيل الكامل
# ---------------------------------------------------------------------
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])
    train_base = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataset = CaptioningDataset(train_base)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    vision_config = VisionConfig(image_size=28, patch_size=4, embed_dim=128)
    llm_config = LLMConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=128,
        max_seq_len=vision_config.num_patches + MAX_CAPTION_LEN,
    )

    model = MiniLLaVA(vision_config, llm_config, freeze_vision=False)
    # ملاحظة: في هذا المثال التعليمي الصغير لم نجمّد ViT لصغر حجم البيانات،
    # لكن في الإنتاج الحقيقي يُجمَّد ViT دائمًا في المرحلة الأولى (محاذاة الميزات).

    train_vlm(model, train_loader, num_epochs=15)

    # اختبار على صور جديدة
    test_base = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    for i in range(5):
        image, label = test_base[i]
        caption = generate_caption(model, image)
        print(f"الفئة الحقيقية: {CLASS_NAMES[label]} | الوصف المُولَّد: {caption}")

# ────────────────────────────────────────────────────────────────────────
# [005] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-4
# ────────────────────────────────────────────────────────────────────────
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("diagram.png")
prompt = "USER: <image>\nما الخطأ في هذا الرسم البياني؟\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output_ids[0], skip_special_tokens=True))

# ────────────────────────────────────────────────────────────────────────
# [006] maghzbook-download/llm-engineering-69-chapters/notebooks/ch-41-chapter-41.ipynb#cell-5
# ────────────────────────────────────────────────────────────────────────
class VQADataset(Dataset):
    """
    بدل تعليق ثابت، كل مثال يحتوي: صورة + سؤال + إجابة.
    مثال: (صورة فستان، "ما لون هذه القطعة؟", "لا يمكن تحديد اللون من صورة رمادية")
    أو أسئلة بسيطة مبنية على الفئة: "ما نوع هذه القطعة؟" -> اسم الفئة.
    """
    def __getitem__(self, idx):
        # صمّم تنسيق تسلسل يجمع: [صورة] + [رموز السؤال] + [رموز الإجابة]
        # تلميح: استخدم فاصلًا خاصًا بين السؤال والإجابة، واحسب الخسارة
        # فقط على جزء الإجابة (أخفِ Loss عن رموز السؤال بـ ignore_index)
        ...
